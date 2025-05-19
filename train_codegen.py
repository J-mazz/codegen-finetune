# train_codegen.py

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TextDataset

# --- Configuration ---
MODEL_NAME = "Salesforce/codegen-350M-mono"  # You can swap to 2B or 6B depending on compute
DATA_PATH = "./final_dataset.txt"
OUTPUT_DIR = "./codegen_finetuned"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3

# --- Load Tokenizer and Model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# --- Prepare Dataset ---
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=MAX_SEQ_LENGTH
    )

print("Loading dataset...")
train_dataset = load_dataset(DATA_PATH)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    fp16=True if torch.cuda.is_available() else False,
    evaluation_strategy="no"
)

# --- Trainer Setup ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# --- Train ---
print("Starting training...")
trainer.train()

# --- Save Final Model ---
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete.")
