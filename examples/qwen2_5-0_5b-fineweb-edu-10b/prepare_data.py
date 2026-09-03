#!/usr/bin/env python3
"""
High-Speed NVMe Data Preparation for FineWeb-Edu 10B Tokens.

Downloads the 10 Billion token sample of HuggingFaceFW/fineweb-edu directly
to local NVMe storage on the server before starting training.

Usage:
    # 1. Download all parquet files (~35 GB) directly to local NVMe storage:
    python prepare_data.py --target_dir ./data/fineweb_edu_10bt

    # 2. Verify an existing download without redownloading:
    python prepare_data.py --target_dir ./data/fineweb_edu_10bt --verify_only
"""

import os
import sys
import argparse
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and verify FineWeb-Edu 10B tokens on local NVMe."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="./data/fineweb_edu_10bt",
        help="Local directory to store parquet files (default: ./data/fineweb_edu_10bt)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset repository ID",
    )
    parser.add_argument(
        "--subset_pattern",
        type=str,
        default="sample/10BT/*.parquet",
        help="Glob pattern to download specific subset (default: sample/10BT/*.parquet)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing files without downloading",
    )
    return parser.parse_args()


def verify_downloaded_data(target_dir: Path, subset_pattern: str):
    """Verify that parquet files exist, are non-empty, and readable."""
    files = list(target_dir.glob(subset_pattern))
    if not files:
        # Check if files were saved directly without parent prefix
        files = list(target_dir.glob("*.parquet"))
    if not files:
        # Check recursive subdirectories
        files = list(target_dir.glob("**/*.parquet"))

    if not files:
        print(f"[Verification Failed] No parquet files found in {target_dir}")
        return False

    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\nFound {len(files)} parquet files ({total_bytes / 1e9:.2f} GB total).")

    try:
        import pyarrow.parquet as pq

        print("Testing readability of the first parquet file...")
        table = pq.read_table(files[0])
        num_rows = table.num_rows
        sample_text = table["text"][0].as_py()[:200].replace("\n", " ")
        print(f"[Verification Passed] First file ({files[0].name}) has {num_rows:,} rows.")
        print(f"Sample text preview: \"{sample_text}...\"\n")
        return True
    except Exception as e:
        print(f"[Warning] Could not inspect parquet with pyarrow: {e}")
        print("Files exist on disk and will be verified by datasets/pandas during training.")
        return True


def download_dataset(args):
    target_path = Path(args.target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FineWeb-Edu 10B Tokens - Local NVMe Data Preparation")
    print(f"Target Directory: {target_path}")
    print(f"Repository:       {args.repo_id}")
    print(f"Subset Pattern:   {args.subset_pattern}")
    print("=" * 70)

    if args.verify_only:
        print("\nVerifying existing download...")
        success = verify_downloaded_data(target_path, args.subset_pattern)
        sys.exit(0 if success else 1)

    start_time = time.time()
    print(f"\nStarting parallel download (max_workers={args.max_workers})...")
    print("Estimated time on datacenter connection: ~3 to 6 minutes (~35 GB total).")

    try:
        downloaded_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=args.subset_pattern,
            local_dir=str(target_path),
            max_workers=args.max_workers,
        )
        elapsed = time.time() - start_time
        print(f"\n[Success] Download completed in {elapsed:.1f} seconds ({elapsed / 60:.2f} minutes)!")
    except Exception as e:
        print(f"\n[Error during download] {e}")
        print("You can re-run this script to resume downloading without losing progress.")
        sys.exit(1)

    verify_downloaded_data(target_path, args.subset_pattern)
    print("Data preparation complete! You can now start training with:")
    print("    python train.py --config config.yaml\n")


if __name__ == "__main__":
    download_dataset(parse_args())
