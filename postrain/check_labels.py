import json
from collections import Counter

counts = Counter()
total = 0

with open("data/processed/sft/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        for l in rec.get("completion", "").splitlines():
            if l.startswith("Categories:"):
                cat = l.replace("Categories:", "").strip()
                counts[cat] += 1
                total += 1

print("\n" + "=" * 60)
print("Actual Label Distribution in train.jsonl:")
print("=" * 60)
for cat, cnt in counts.most_common():
    pct = (cnt / total * 100) if total > 0 else 0
    print(f"  {cat:35s}: {cnt:6d} ({pct:5.1f}%)")
print("=" * 60)
print(f"  Total examples: {total:,}")
print("=" * 60 + "\n")
