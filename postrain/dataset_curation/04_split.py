"""
Reads data/processed/all_examples.jsonl, performs a stratified train/val/test split
using scikit-learn, updates the `split` field, and writes the results out.
"""
import os
import json
import sys
from collections import Counter
from sklearn.model_selection import train_test_split

# Add src to path to import schema if needed
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "all_examples.jsonl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "all_examples_split.jsonl")

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

def get_stratify_key(labels_dict):
    """Create a string representation of active labels for stratification."""
    active = sorted([k for k, v in labels_dict.items() if v == 1])
    return "|".join(active) if active else "none"

def main():
    if not os.path.exists(IN_PATH):
        print(f"Error: {IN_PATH} not found. Run 03_clean_and_dedup.py first.")
        return

    records = []
    stratify_keys = []
    
    with open(IN_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            stratify_keys.append(get_stratify_key(rec["labels"]))

    # Handle rare classes (scikit-learn requires at least 2 instances of a class to stratify)
    counts = Counter(stratify_keys)
    safe_stratify = [k if counts[k] >= 2 else "rare_combination" for k in stratify_keys]

    # Split into Train (80%) and Temp (20%)
    train_recs, temp_recs, _, temp_strat = train_test_split(
        records, safe_stratify, train_size=TRAIN_FRAC, stratify=safe_stratify, random_state=42
    )

    # Re-evaluate rare classes in the temp split before splitting into Val/Test
    temp_counts = Counter(temp_strat)
    safe_temp_strat = [k if temp_counts[k] >= 2 else "rare_combination" for k in temp_strat]

    # Split Temp into Val (10%) and Test (10%) - i.e., 50% of the 20% temp pool
    val_recs, test_recs = train_test_split(
        temp_recs, train_size=0.5, stratify=safe_temp_strat, random_state=42
    )

    # Assign split tags
    for r in train_recs: r["split"] = "train"
    for r in val_recs: r["split"] = "val"
    for r in test_recs: r["split"] = "test"

    all_split_records = train_recs + val_recs + test_recs

    with open(OUT_PATH, "w") as out_f:
        for rec in all_split_records:
            out_f.write(json.dumps(rec) + "\n")

    print(f"Total examples: {len(all_split_records)}")
    print(f"  Train: {len(train_recs)}")
    print(f"  Val:   {len(val_recs)}")
    print(f"  Test:  {len(test_recs)}")
    print(f"Output saved to {OUT_PATH}")

if __name__ == "__main__":
    main()