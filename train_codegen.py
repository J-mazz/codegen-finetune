# train_codegen.py

import os
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from torch.utils.data import Dataset as TorchDataset # Alias to avoid confusion

class CustomTextDataset(TorchDataset):
    def __init__(self, tokenized_samples):
        self.samples = tokenized_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Create instances of this custom dataset
full_custom_dataset = CustomTextDataset(raw_tokenized_samples)
# --- Config ---
MODEL_NAME = "Salesforce/codegen-350M-mono"
DATA_PATH = "./final_dataset.txt"
OUTPUT_DIR = "./codegen_finetuned"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_EPOCHS = 2
VAL_SPLIT = 0.1  # 10% validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# (In train_codegen.py)
# ...
# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    # Still important to define what token to use for padding,
    # even if we manually apply it later.
    tokenizer.pad_token = tokenizer.eos_token
print(f"Using pad_token: '{tokenizer.pad_token}' with ID: {tokenizer.pad_token_id}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# --- Load and prepare dataset manually ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if not lines:
    raise ValueError(f"No data found in {DATA_PATH}")

# Tokenize each text individually, but don't pad yet, just truncate
# We'll handle padding at the batch level.
# This gives you raw token_ids for each example.
# The output of tokenizer() without padding is a list of token IDs.
raw_tokenized_samples = []
for text_sample in tqdm(lines, desc="Tokenizing samples"):
    tokenized_output = tokenizer(
        text_sample,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        # IMPORTANT: No padding here, or padding=False
        padding=False
    )
    raw_tokenized_samples.append({
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"] # Will also be unpadded
    })

# Now, raw_tokenized_samples is a list of dicts,
# e.g., [{'input_ids': [10, 20, 30], 'attention_mask': [1,1,1]}, ...]
# where each list of input_ids can be of a different length.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# --- Load and tokenize dataset ---

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if not lines:
    raise ValueError(f"No data found in {DATA_PATH}")

dataset = Dataset.from_dict({"text": lines})

def custom_collate_fn(batch_samples, pad_token_id, max_len=None):
    """
    Manually pads a batch of tokenized samples.
    batch_samples: A list of dicts, e.g., [{'input_ids': [...], 'attention_mask': [...]}, ...]
    pad_token_id: The ID to use for padding.
    max_len: If None, pads to the longest sequence in the batch.
             If set, pads all sequences to this max_len.
    """
    input_ids_list = [sample['input_ids'] for sample in batch_samples]

    if max_len is None:
        # Pad to the longest sequence in the current batch
        current_max_len = max(len(ids) for ids in input_ids_list)
    else:
        # Pad to the global MAX_SEQ_LENGTH (or other desired fixed length)
        current_max_len = max_len

    padded_input_ids = []
    padded_attention_masks = []

    for input_ids in input_ids_list:
        num_padding_tokens = current_max_len - len(input_ids)
        
        padded_ids = input_ids + [pad_token_id] * num_padding_tokens
        # Attention mask: 1 for real tokens, 0 for padding tokens
        attention_mask = [1] * len(input_ids) + [0] * num_padding_tokens
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long)
    }

# --- Train/val split (using your CustomTextDataset) ---
val_size = int(VAL_SPLIT * len(full_custom_dataset))
train_size = len(full_custom_dataset) - val_size
train_custom_dataset, val_custom_dataset = random_split(full_custom_dataset, [train_size, val_size])

# Use the custom_collate_fn in your DataLoader
# You need to pass the pad_token_id to it. functools.partial can help.
from functools import partial

collate_fn_with_padding = partial(custom_collate_fn,
                                  pad_token_id=tokenizer.pad_token_id,
                                  max_len=MAX_SEQ_LENGTH) # Or None to pad per-batch

train_loader = DataLoader(train_custom_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_padding)
val_loader = DataLoader(val_custom_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_padding)

# --- Train/val split ---
val_size = int(VAL_SPLIT * len(tokenized_dataset))
train_size = len(tokenized_dataset) - val_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Optimizer, Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int((WARMUP_EPOCHS / EPOCHS) * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# --- Helper: Calculate accuracy ---
def compute_accuracy(logits, labels, ignore_index=-100):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != ignore_index
    correct = (preds == labels) & mask
    acc = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0
    return acc

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    train_acc = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute accuracy on the fly
        batch_acc = compute_accuracy(logits, labels)
        train_loss += loss.item()
        train_acc += batch_acc

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}"
        })

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    print(f"\n[Train] Epoch {epoch}: Loss={avg_train_loss:.4f}, Accuracy={avg_train_acc:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Show sample softmax/probabilities for a random batch
            if torch.rand(1).item() < 0.01:
                probs = torch.softmax(logits[0], dim=-1)
                print(f"\nSample softmax probabilities for batch: {probs[0, :10].cpu().detach().numpy()}")

            batch_acc = compute_accuracy(logits, labels)
            val_loss += loss.item()
            val_acc += batch_acc

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    print(f"[Val]   Epoch {epoch}: Loss={avg_val_loss:.4f}, Accuracy={avg_val_acc:.4f}")
    print("-" * 40)

    # Save checkpoint each epoch
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch}")

print("✅ Training complete. Best model may be in final epoch folder.")
