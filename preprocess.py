# preprocess.py

import os
import pandas as pd
import re
from datasets import load_dataset
from hashlib import sha256

# Output config
PROCESSED_CSV_DIR = "./processed_csv"
os.makedirs(PROCESSED_CSV_DIR, exist_ok=True)

DEFAULT_MAX_SAMPLES = 5000

DATASETS = [
    {"hf_name": "mbpp", "split": "train", "output": "mbpp.csv"},
    {"hf_name": "open-r1/verifiable-coding-problems-python", "split": "train", "output": "verifiable_problems.csv"},
    {"hf_name": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "output": "alpaca_18k.csv"},
    {"hf_name": "MatrixStudio/Codeforces-Python-Submissions", "split": "train", "output": "codeforces_submissions.csv"},
    {"hf_name": "jtatman/python-code-dataset-500k", "split": "train", "output": "code_500k.csv", "max_samples": 25000, "use_auth": True}
]

def clean_code(text):
    text = text.strip() if isinstance(text, str) else ""
    return re.sub(r'[\r\t]+', ' ', text)

def synthesize_prompt(example):
    doc = example.get("docstring") or example.get("prompt") or example.get("instruction") or "# Task: Write a function"
    return f"# Task: {doc.strip()}"

def extract_completion(example):
    return clean_code(
        example.get("code")
        or example.get("solution")
        or example.get("completion")
        or example.get("content")
        or ""
    )

for config in DATASETS:
    try:
        print(f"\n➡️  Processing {config['hf_name']}...")
        # Dynamically build load_dataset args to avoid unexpected keyword errors
        ds_args = {
            "path": config["hf_name"],
            "split": config["split"]
        }
        if config.get("trust_remote_code", False):
            ds_args["trust_remote_code"] = True
        if config.get("use_auth", False):
            ds_args["token"] = True  # Use new HF Datasets 'token' argument

        ds = load_dataset(**ds_args)
    except Exception as e:
        print(f"❌ Error loading {config['hf_name']}: {e}")
        continue

    rows = []
    limit = config.get("max_samples", DEFAULT_MAX_SAMPLES)
    n_total = 0

    for i, ex in enumerate(ds):
        if i >= limit:
            break
        prompt = synthesize_prompt(ex)
        completion = extract_completion(ex)
        if len(completion) < 50 or len(completion) > 2000:
            continue
        rows.append({
            "prompt": prompt,
            "completion": completion,
            "dataset": config["hf_name"],
            "language": ex.get("language") or "python"
        })
        n_total += 1
        if n_total % 1000 == 0:
            print(f"  Processed {n_total} valid examples...")

    if rows:
        df = pd.DataFrame(rows)
        out_path = os.path.join(PROCESSED_CSV_DIR, config["output"])
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(rows)} examples to {out_path}\n")
    else:
        print(f"⚠️  No valid examples extracted for {config['hf_name']}.\n")

print("🎉 All datasets processed (even if some failed).")
