# train_codegen.py
import os
import torch
from torch.utils.data import DataLoader, random_split
# Remove AdamW from this line
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW # Add this line
from tqdm.auto import tqdm
# ... rest of your imports
# --- Config ---
MODEL_NAME = "Salesforce/codegen-350M-mono"
DATA_PATH = "./final_dataset.txt" # From your uploaded script
OUTPUT_DIR = "./codegen_finetuned" # From your uploaded script
MAX_SEQ_LENGTH = 512 # From your uploaded script
BATCH_SIZE = 1 # From your uploaded script
EPOCHS = 5 # From your uploaded script
LEARNING_RATE = 5e-5 # From your uploaded script
WARMUP_EPOCHS = 2 # From your uploaded script
VAL_SPLIT = 0.1  # From your uploaded script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define Custom Dataset Class ---
class CustomTextDataset(TorchDataset):
    def __init__(self, tokenized_samples):
        self.samples = tokenized_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --- Define Custom Collate Function ---
def custom_collate_fn(batch_samples, pad_token_id, max_len=None):
    input_ids_list = [sample['input_ids'] for sample in batch_samples]

    if max_len is None:
        current_max_len = max(len(ids) for ids in input_ids_list)
    else:
        current_max_len = max_len

    padded_input_ids = []
    padded_attention_masks = []

    for input_ids in input_ids_list:
        num_padding_tokens = current_max_len - len(input_ids)
        
        padded_ids = input_ids + [pad_token_id] * num_padding_tokens
        attention_mask = [1] * len(input_ids) + [0] * num_padding_tokens
        
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long)
    }

# --- Load Tokenizer and Model (Consolidated) ---
print("🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Using pad_token: '{tokenizer.pad_token}' with ID: {tokenizer.pad_token_id}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# --- Load Data and Tokenize Manually (No Padding Here) ---
print(f"📄 Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if not lines:
    raise ValueError(f"No data found in {DATA_PATH}")

print(f"Tokenizing {len(lines)} samples manually (truncation to {MAX_SEQ_LENGTH}, no padding yet)...")
raw_tokenized_samples = []
for text_sample in tqdm(lines, desc="Tokenizing samples"):
    tokenized_output = tokenizer(
        text_sample,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False # No padding at this stage
    )
    raw_tokenized_samples.append({
        "input_ids": tokenized_output["input_ids"],
        "attention_mask": tokenized_output["attention_mask"]
    })

# --- Create Custom Datasets and DataLoaders ---
print("Creating custom datasets and DataLoaders...")
full_custom_dataset = CustomTextDataset(raw_tokenized_samples)

val_size = int(VAL_SPLIT * len(full_custom_dataset))
train_size = len(full_custom_dataset) - val_size
train_custom_dataset, val_custom_dataset = random_split(full_custom_dataset, [train_size, val_size])

# Prepare the collate function with specific arguments
collate_fn_with_padding = partial(custom_collate_fn,
                                  pad_token_id=tokenizer.pad_token_id,
                                  max_len=MAX_SEQ_LENGTH) # Pad to MAX_SEQ_LENGTH

train_loader = DataLoader(train_custom_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_padding)
val_loader = DataLoader(val_custom_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_padding)
print(f"Train DataLoader: {len(train_loader)} batches. Validation DataLoader: {len(val_loader)} batches.")


# --- Optimizer, Scheduler (This part remains the same) ---
print("Setting up optimizer and scheduler...")
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) # From your uploaded script
# Adjust total_steps if BATCH_SIZE or number of samples changed significantly with the new loader.
# It should now be based on the length of the new `train_loader`.
total_steps = len(train_loader) * EPOCHS
warmup_steps = int((WARMUP_EPOCHS / EPOCHS) * total_steps) # From your uploaded script
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps) # From your uploaded script

# --- Helper: Calculate accuracy (This part remains the same) ---
# ... (compute_accuracy function from your script) ...

# --- Training loop (This part remains the same and should now work with the new DataLoaders) ---
# ... (Your existing training and validation loop) ...

# print("✅ Training complete. Best model may be in final epoch folder.") # From your uploaded script
