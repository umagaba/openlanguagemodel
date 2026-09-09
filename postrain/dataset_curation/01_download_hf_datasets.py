"""
Pulls every manifest entry that has an `hf_id` and a decision of
"use" or "caution" from the Hugging Face Hub, and writes it to
data/raw/<source_id>/ as parquet, untouched.

Sources without an hf_id (Safecity, ConvAbuse depending on release, UCI,
CUAD variants, Contract Ambiguity, IEEE Cyberbullying) need a manual
download step — see README.md "Manual sources" section. Drop the raw
file(s) into data/raw/<source_id>/ in the same layout and this script
will skip them without complaint.
"""
import os
import yaml
from datasets import load_dataset

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "manifest.yaml")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    with open(MANIFEST_PATH) as f:
        manifest = yaml.safe_load(f)

    for entry in manifest:
        decision = entry.get("decision")
        hf_id = entry.get("hf_id")
        if decision not in ("use", "caution") or not hf_id:
            continue

        out_dir = os.path.join(RAW_DIR, entry["id"])
        if os.path.exists(out_dir):
            print(f"[skip] {entry['id']} already downloaded")
            continue

        print(f"[download] {entry['id']} <- {hf_id}")
        try:
            ds = load_dataset(hf_id)
        except Exception as e:
            print(f"  !! failed: {e}. Log this and download manually if it's gated/renamed.")
            continue

        os.makedirs(out_dir, exist_ok=True)
        for split_name, split_ds in ds.items():
            split_ds.to_parquet(os.path.join(out_dir, f"{split_name}.parquet"))
        print(f"  -> saved {sum(len(s) for s in ds.values())} rows to {out_dir}")


if __name__ == "__main__":
    main()
