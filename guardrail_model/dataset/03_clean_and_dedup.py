"""
Layered filtering, cheapest first, applied ACROSS all normalized sources
combined (dedup must be global, not per-source, per the strategy doc):

  1. basic clean (strip HTML/URLs/mentions, collapse repeats)
  2. length + symbol-ratio + boilerplate + language heuristics
  3. PII scrub (regex pass; run presidio separately for a stronger pass on
     sexual_harassment / mental_health_risk rows before final release)
  4. exact-duplicate removal (hash)
  5. near-duplicate removal (MinHash-LSH)

Writes data/processed/all_examples.jsonl
"""
import os
import sys
import json
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.text_clean import basic_clean, length_ok, symbol_ratio_ok, is_boilerplate, is_target_language
from src.pii import scrub
from datasketch import MinHash, MinHashLSH

NORM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "normalized")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "all_examples.jsonl")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

NUM_PERM = 128
LSH_THRESHOLD = 0.85  # texts >=85% shingle-similar are treated as near-duplicates


def shingles(text, k=5):
    tokens = text.split()
    return {" ".join(tokens[i:i + k]) for i in range(max(len(tokens) - k + 1, 1))}


def to_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    for sh in shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def main():
    seen_hashes = set()
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    kept = []
    dropped_counts = {"length": 0, "symbol_ratio": 0, "boilerplate": 0, "language": 0, "exact_dup": 0, "near_dup": 0}

    n_seen = 0
    with open(OUT_PATH, "w") as out_f:
        for path in glob.glob(os.path.join(NORM_DIR, "*.jsonl")):
            with open(path) as f:
                for line in f:
                    n_seen += 1
                    rec = json.loads(line)
                    text = basic_clean(rec["text"])
                    is_sensitive = rec["labels"].get("sexual_harassment") or rec["labels"].get("mental_health_risk")
                    if is_sensitive:
                        text = scrub(text)

                    if not length_ok(text):
                        dropped_counts["length"] += 1
                        continue
                    if not symbol_ratio_ok(text):
                        dropped_counts["symbol_ratio"] += 1
                        continue
                    if is_boilerplate(text):
                        dropped_counts["boilerplate"] += 1
                        continue
                    if not is_target_language(text):
                        dropped_counts["language"] += 1
                        continue

                    h = rec["text_hash"] = __import__("hashlib").sha256(
                        " ".join(text.lower().split()).encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        dropped_counts["exact_dup"] += 1
                        continue

                    mh = to_minhash(text)
                    near_dupe_keys = lsh.query(mh)
                    if near_dupe_keys:
                        dropped_counts["near_dup"] += 1
                        continue

                    lsh.insert(h, mh)
                    seen_hashes.add(h)
                    rec["text"] = text
                    out_f.write(json.dumps(rec) + "\n")
                    kept.append(rec)

    print(f"Read {n_seen} examples, kept {len(kept)}")
    print("Dropped breakdown:", dropped_counts)


if __name__ == "__main__":
    main()