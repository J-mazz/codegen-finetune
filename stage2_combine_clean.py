# stage2_combine_clean.py

import os
import pandas as pd
from hashlib import sha256

# Input/output paths
PROCESSED_CSV_DIR = "./processed_csv"
FINAL_OUTPUT_PATH = "./final_dataset.txt"

# Load and combine all CSVs
all_rows = []
seen_hashes = set()

for fname in os.listdir(PROCESSED_CSV_DIR):
    if not fname.endswith(".csv"): continue
    print(f"Loading {fname}...")
    df = pd.read_csv(os.path.join(PROCESSED_CSV_DIR, fname))
    for _, row in df.iterrows():
        prompt = str(row["prompt"]).strip()
        completion = str(row["completion"]).strip()
        if not prompt or not completion: continue

        content = f"<|prompt|>\n{prompt}\n<|completion|>\n{completion}\n<|endofexample|>"
        hashval = sha256((prompt + completion).encode()).hexdigest()
        if hashval not in seen_hashes:
            seen_hashes.add(hashval)
            all_rows.append(content)

print(f"Writing {len(all_rows)} unique examples to {FINAL_OUTPUT_PATH}...")
with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(all_rows))

print("Stage 2 complete.")
