"""
Converts each raw source into data/normalized/<source_id>.jsonl using the
UnifiedExample schema. One adapter function per source, because every
dataset ships different column names/label schemes -- this is intentional
per the strategy doc's "filter/normalize per source-type separately" rule.

Add a new adapter here whenever you wire in another manual source
(Safecity, ConvAbuse, UCI, IEEE Cyberbullying, Contract Ambiguity).
"""
import os
import sys
import json
import glob
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.schema import UnifiedExample, LABELS

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NORM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "normalized")
os.makedirs(NORM_DIR, exist_ok=True)


def empty_labels():
    return {l: 0 for l in LABELS}


def load_parquet_source(source_id):
    files = glob.glob(os.path.join(RAW_DIR, source_id, "*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------- adapters

def normalize_hatexplain(df):
    out = []
    for i, row in df.iterrows():
        text = " ".join(row["post_tokens"]) if isinstance(row.get("post_tokens"), (list,)) else row.get("text", "")
        # hatexplain stores a list of annotator label ids; take majority vote label name if present
        label_field = row.get("annotators", {}).get("label") if isinstance(row.get("annotators"), dict) else None
        majority = max(set(label_field), key=label_field.count) if label_field else None
        labels = empty_labels()
        if majority in (0, 1):   # 0=hatespeech, 1=offensive in most hatexplain releases -- VERIFY against your pull
            labels["hate_speech"] = 1
        else:
            labels["none"] = 1
        out.append(UnifiedExample(text=text, labels=labels, source_id="hatexplain",
                                   source_license="MIT", origin_id=str(i)))
    return out


def normalize_measuring_hate_speech(df):
    out = []
    for i, row in df.iterrows():
        labels = empty_labels()
        
        # Use safe dictionary get to prevent KeyErrors if columns are missing
        hs_score = row.get("hate_speech_score", 0)
        if hs_score is not None and not pd.isna(hs_score) and hs_score >= 0.5:
            labels["hate_speech"] = 1
            
        ia_score = row.get("identity_attack", 0)
        if ia_score is not None and not pd.isna(ia_score) and ia_score >= 0.5:
            labels["racial_profiling"] = 1
            
        if not any(labels[l] for l in LABELS if l != "none"):
            labels["none"] = 1
            
        text_val = row.get("text", "")
        text_str = str(text_val) if text_val is not None and not pd.isna(text_val) else ""
        
        out.append(UnifiedExample(text=text_str, labels=labels,
                                   source_id="measuring_hate_speech", source_license="CC-BY-4.0",
                                   origin_id=str(i)))
    return out


def normalize_civil_comments(df):
    out = []
    for i, row in df.iterrows():
        labels = empty_labels()
        if row.get("identity_attack", 0) >= 0.5:
            labels["hate_speech"] = 1
        if row.get("sexual_explicit", 0) >= 0.5:
            labels["sexual_harassment"] = 1
        if row.get("sexual_explicit", 0) >= 0.7:
            labels["explicit_material"] = 1
        if not any(labels[l] for l in LABELS if l != "none"):
            labels["none"] = 1
        out.append(UnifiedExample(text=str(row.get("text", "")), labels=labels,
                                   source_id="jigsaw_civil_comments", source_license="CC0",
                                   origin_id=str(i)))
    return out


def normalize_openai_moderation(df):
    """Expects a flat file with columns: text, category (see manual-source note in README)."""
    value_map = {
        "hate": "hate_speech", "hate/threatening": "hate_speech",
        "harassment": "sexual_harassment", "harassment/threatening": "sexual_harassment",
        "sexual": "explicit_material", "sexual/minors": "explicit_material",
        "self-harm": "mental_health_risk", "self-harm/intent": "mental_health_risk",
        "self-harm/instructions": "mental_health_risk",
    }
    out = []
    for i, row in df.iterrows():
        labels = empty_labels()
        mapped = value_map.get(row.get("category"))
        if mapped:
            labels[mapped] = 1
        else:
            labels["none"] = 1
        out.append(UnifiedExample(text=str(row.get("text", "")), labels=labels,
                                   source_id="openai_moderation_dataset", source_license="MIT",
                                   origin_id=str(i)))
    return out


def normalize_contract_ambiguity(df):
    """Expects columns: clause_text, ambiguous (0/1). See manual-source note in README."""
    out = []
    for i, row in df.iterrows():
        labels = empty_labels()
        labels["legal_ambiguity"] = int(bool(row.get("ambiguous", 0)))
        labels["none"] = int(not labels["legal_ambiguity"])
        out.append(UnifiedExample(text=str(row.get("clause_text", "")), labels=labels,
                                   source_id="contract_ambiguity_singhal2024",
                                   source_license="contact-authors", origin_id=str(i)))
    return out


def normalize_ieee_cyberbullying(df):
    """Expects columns: tweet_text, cyberbullying_type. See manual-source note in README."""
    value_map = {
        "ethnicity": "racial_profiling", "religion": "racial_profiling",
        "age": "hate_speech", "gender": "hate_speech", "other_cyberbullying": "hate_speech",
    }
    out = []
    for i, row in df.iterrows():
        labels = empty_labels()
        mapped = value_map.get(row.get("cyberbullying_type"))
        if mapped:
            labels[mapped] = 1
        else:
            labels["none"] = 1
        out.append(UnifiedExample(text=str(row.get("tweet_text", "")), labels=labels,
                                   source_id="ieee_cyberbullying_types",
                                   source_license="verify-ieee-dataport-terms", origin_id=str(i)))
    return out


ADAPTERS = {
    "hatexplain": normalize_hatexplain,
    "measuring_hate_speech": normalize_measuring_hate_speech,
    "jigsaw_civil_comments": normalize_civil_comments,
    "openai_moderation_dataset": normalize_openai_moderation,
    "contract_ambiguity_singhal2024": normalize_contract_ambiguity,
    "ieee_cyberbullying_types": normalize_ieee_cyberbullying,
    # Add: dynahate, convabuse, safecity, uci_user_profiling_abusive_language, cuad
    # once you've inspected their actual column names -- follow the same pattern.
}


def main():
    for source_id, adapter in ADAPTERS.items():
        df = load_parquet_source(source_id)
        if df is None:
            print(f"[skip] no raw data found for {source_id} (download it first, or place manual file)")
            continue
        examples = adapter(df)
        out_path = os.path.join(NORM_DIR, f"{source_id}.jsonl")
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_json()) + "\n")
        print(f"[ok] {source_id}: {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
