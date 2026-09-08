import json
import os

IN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "all_examples_split.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sft")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS_LIST = [
    "hate_speech",
    "sexual_harassment",
    "explicit_material",
    "mental_health_risk",
    "legal_ambiguity",
    "racial_profiling",
]

def format_record(rec):
    active_labels = [k for k in LABELS_LIST if rec["labels"].get(k, 0) == 1]
    
    if active_labels:
        status = "FLAGGED"
        detected_str = ", ".join(active_labels)
    else:
        status = "SAFE"
        detected_str = "none"

    prompt = (
        "You are an AI guardrail monitor. Analyze the following model output text for safety violations "
        "(hate_speech, sexual_harassment, explicit_material, mental_health_risk, legal_ambiguity, racial_profiling).\n\n"
        f"Text to monitor: \"{rec['text']}\"\n\n"
        "Output format:\nStatus: [SAFE/FLAGGED]\nCategories: [...]\nOffending line: [...]"
    )
    
    response = (
        f"Status: {status}\n"
        f"Categories: {detected_str}\n"
        f"Offending line: \"{rec['text']}\""
    )
    
    return {
        "prompt": prompt,
        "completion": response
    }

def main():
    if not os.path.exists(IN_PATH):
        print(f"Error: {IN_PATH} not found. Run 04_split.py first.")
        return

    for split in ["train", "val", "test"]:
        out_path = os.path.join(OUT_DIR, f"{split}.jsonl")
        count = 0
        with open(IN_PATH, "r") as f_in, open(out_path, "w") as f_out:
            for line in f_in:
                rec = json.loads(line)
                if rec.get("split") == split:
                    formatted = format_record(rec)
                    f_out.write(json.dumps(formatted) + "\n")
                    count += 1
        print(f"Wrote {count} examples to {out_path}")

if __name__ == "__main__":
    main()