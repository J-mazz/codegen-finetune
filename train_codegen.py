import os
import torch
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

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
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# --- Load and tokenize dataset ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if not lines:
    raise ValueError(f"No data found in {DATA_PATH}")

dataset = Dataset.from_dict({"text": lines})

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

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

            # For UI: print sample softmax/probabilities for a random batch
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

    # Save checkpoint each epoch (optional)
    model.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch}")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/epoch_{epoch}")

print("✅ Training complete. Best model may be in final epoch folder.")
# train_codegen.py
