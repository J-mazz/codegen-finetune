# train_codegen.py (updated for compatibility and modern Hugging Face usage)

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# --- Configuration ---
MODEL_NAME = "Salesforce/codegen-350M-mono"
DATA_PATH = "./final_dataset.txt"
OUTPUT_DIR = "./codegen_finetuned"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 1

# --- Load tokenizer and model ---
print("🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- Load dataset with streaming ---
def load_txt_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return {"text": lines}

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )

print("📄 Loading dataset from final_dataset.txt")
raw_data = load_txt_dataset(DATA_PATH)
from datasets import Dataset
train_dataset = Dataset.from_dict(raw_data)
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- Training args ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    evaluation_strategy="no"
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# --- Train ---
print("🚀 Starting training...")
trainer.train()

# --- Save model ---
print(f"💾 Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete.")
