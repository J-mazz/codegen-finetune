# stage1_preprocess.py

import os
import pandas as pd
import re
from datasets import load_dataset
from hashlib import sha256

# Output config
PROCESSED_CSV_DIR = "./processed_csv"
os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)

# Example datasets to load
DATASETS = [
    {"hf_name": "mbpp", "split": "train", "output": "mbpp.csv"},
    {"hf_name": "codeparrot/github-code", "split": "train", "output": "codeparrot.csv"},
    {"hf_name": "open-r1/verifiable-coding-problems-python", "split": "train", "output": "verifiable_problems.csv"},
    {"hf_name": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "output": "alpaca_18k.csv"},
    {"hf_name": "MatrixStudio/Codeforces-Python-Submissions", "split": "train", "output": "codeforces_submissions.csv"},
    {"hf_name": "jtatman/python-code-dataset-500k", "split": "train", "output": "code_500k.csv", "max_samples": 25000}
]

def clean_code(text):
    text = text.strip()
    return re.sub(r'[\r\t]+', ' ', text)

def synthesize_prompt(example):
    doc = example.get("docstring") or example.get("prompt") or example.get("instruction") or "# Task: Write a function"
    return f"# Task: {doc.strip()}"

def extract_completion(example):
    return clean_code(example.get("code") or example.get("solution") or example.get("completion") or example.get("content") or "")

for config in DATASETS:
    try:
        print(f"Processing {config['hf_name']}...")
        ds = load_dataset(config["hf_name"], split=config["split"], use_auth_token=True)
    except Exception as e:
        print(f"❌ Error loading {config['hf_name']}: {e}")
        continue

    rows = []
    for i, ex in enumerate(ds):
        if "max_samples" in config and i >= config["max_samples"]:
            break
        prompt = synthesize_prompt(ex)
        completion = extract_completion(ex)
        if len(completion) < 50:
            continue
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "dataset": config["hf_name"],
            "language": ex.get("language") or "python"
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_CSV_DIR, config["output"]), index=False)

print("Stage 1 complete. CSVs saved.")

