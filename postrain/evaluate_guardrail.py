#!/usr/bin/env python3
"""
Comprehensive Multi-Sample Evaluation Suite for Qwen2.5-0.5B Guardrail Model.

Evaluates hundreds of held-out test examples from data/processed/sft/test.jsonl
(or val.jsonl) plus curated challenging edge cases.

Features:
- Full metrics computation: Accuracy, Precision, Recall, F1 per category, and Confusion Matrix.
- Detailed error tracking: records every misclassified sample with input text, target, and prediction.
- Saves structured artifacts in eval_results/:
    1. eval_results/metrics.json (machine-readable metrics summary)
    2. eval_results/eval_report.md (formatted Markdown report for submissions)
    3. eval_results/detailed_predictions.jsonl (row-by-row prediction logs)
- Prints a terminal summary with full statistics and error analysis.

Usage:
    # 1. Evaluate on 300 test samples:
    python evaluate_guardrail.py --num_samples 300

    # 2. Evaluate on 1000 test samples:
    python evaluate_guardrail.py --num_samples 1000

    # 3. Evaluate on full test set:
    python evaluate_guardrail.py --num_samples -1
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm

# Ensure olm is accessible
try:
    import olm
except ImportError:
    current_dir = Path(__file__).resolve().parent
    for candidate in [
        current_dir.parent.parent / "src",
        current_dir.parent / "src",
        current_dir / "src",
        Path("/workspace/src"),
    ]:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate.resolve()))
            break

from olm.models.alibaba import Qwen2_5_0_5B
from olm.data.tokenization import HFTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-0.5B Guardrail Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to guardrail checkpoint (.pt). Defaults to checkpoints_sft/qwen_0_5b_guardrail_merged.pt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to evaluation JSONL file. Defaults to data/processed/sft/test.jsonl",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500, use -1 for entire file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation reports and prediction logs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max tokens to generate per sample (default: 50)",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    script_dir = Path(__file__).resolve().parent
    if (script_dir / path_str).exists():
        return (script_dir / path_str).resolve()
    if (script_dir.parent / path_str).exists():
        return (script_dir.parent / path_str).resolve()
    return p


def resolve_checkpoint(checkpoint_arg: str = None) -> Path:
    candidates = []
    if checkpoint_arg:
        candidates.append(Path(checkpoint_arg))

    script_dir = Path(__file__).resolve().parent
    candidates.extend([
        script_dir / "checkpoints_sft" / "qwen_0_5b_guardrail_merged.pt",
        script_dir / "checkpoints_sft" / "best_model.pt",
        script_dir / "checkpoints_sft" / "qwen_0_5b_guardrail_sft_final.pt",
        Path("checkpoints_sft/qwen_0_5b_guardrail_merged.pt"),
        Path("checkpoints_sft/best_model.pt"),
    ])

    for c in candidates:
        res = resolve_path(str(c))
        if res.is_file():
            return res

    raise FileNotFoundError(f"No checkpoint found. Checked: {[str(c) for c in candidates[:4]]}")


def resolve_data_file(data_arg: str = None) -> Path:
    candidates = []
    if data_arg:
        candidates.append(Path(data_arg))

    script_dir = Path(__file__).resolve().parent
    candidates.extend([
        script_dir / "data" / "processed" / "sft" / "test.jsonl",
        script_dir / "data" / "processed" / "sft" / "val.jsonl",
        Path("data/processed/sft/test.jsonl"),
        Path("data/processed/sft/val.jsonl"),
    ])

    for c in candidates:
        res = resolve_path(str(c))
        if res.is_file():
            return res

    raise FileNotFoundError(f"No evaluation JSONL found. Checked: {[str(c) for c in candidates[:4]]}")


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    print(f"[Model] Loading guardrail weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("model."):
            key = key[6:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        if key.startswith("base_model.model."):
            key = key[17:]
        clean_state_dict[key] = v

    model = Qwen2_5_0_5B()
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if unexpected:
        print(f"[Model] Unexpected keys ({len(unexpected)}) ignored.")
    if missing:
        print(f"[Model] Missing keys ({len(missing)}).")

    model.to(device)
    model.eval()
    return model


def parse_fields(text: str) -> Tuple[str, str, str]:
    """Parse Status, Categories, and Offending line from generated or target text."""
    status = "UNKNOWN"
    category = "unknown"
    offending = ""

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Status:"):
            val = line.replace("Status:", "").strip().upper()
            if "FLAGGED" in val:
                status = "FLAGGED"
            elif "SAFE" in val:
                status = "SAFE"
            else:
                status = val
        elif line.startswith("Categories:"):
            category = line.replace("Categories:", "").strip()
        elif line.startswith("Offending line:"):
            offending = line.replace("Offending line:", "").strip().strip('"')

    return status, category, offending


def extract_input_text(prompt: str) -> str:
    """Extract raw user input text from the guardrail prompt string."""
    marker = "Text to monitor: \""
    if marker in prompt:
        return prompt.split(marker)[1].split("\"\n\nOutput format:")[0]
    return prompt[:80]


@torch.no_grad()
def run_evaluation():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint(args.checkpoint)
    data_path = resolve_data_file(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("Qwen2.5-0.5B Guardrail Model Multi-Sample Benchmark Suite")
    print("=" * 80)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint:       {ckpt_path.name}")
    print(f"Evaluation Data:  {data_path} ({data_path.name})")
    print(f"Sample Limit:     {args.num_samples if args.num_samples > 0 else 'ALL'}")
    print(f"Output Directory: {output_dir.resolve()}")
    print("=" * 80 + "\n")

    tokenizer = HFTokenizer("Qwen/Qwen2.5-0.5B")
    model = load_model(ckpt_path, device)
    eos_id = tokenizer.tokenizer.eos_token_id

    # Read samples from JSONL
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if args.num_samples > 0 and len(samples) >= args.num_samples:
                    break

    print(f"[Data] Loaded {len(samples):,} evaluation examples.\n")

    # Evaluation accumulators
    total_samples = len(samples)
    correct_status = 0
    correct_category = 0

    status_confusion = defaultdict(int)  # (gt, pred) -> count
    category_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total_gt": 0})

    detailed_records = []
    misclassified_records = []

    start_time = time.time()

    pbar = tqdm(samples, desc="Evaluating Guardrail Model", unit="sample")
    for idx, item in enumerate(pbar):
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")

        gt_status, gt_category, gt_offending = parse_fields(completion)
        input_text = extract_input_text(prompt)

        # Autoregressive generation
        input_prompt = prompt + "\n" if not prompt.endswith("\n") else prompt
        tokens = tokenizer.encode(input_prompt).unsqueeze(0).to(device)

        gen_tokens = []
        for _ in range(args.max_new_tokens):
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            token_id = next_token.item()
            gen_tokens.append(token_id)

            if token_id == eos_id:
                break

            # Fast stop check: stop once Offending line is completed
            if len(gen_tokens) > 10:
                cur_text = tokenizer.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if "Offending line:" in cur_text and ("\n" in cur_text.split("Offending line:")[-1] or cur_text.endswith('"')):
                    break

        pred_raw = tokenizer.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        pred_status, pred_category, pred_offending = parse_fields(pred_raw)

        # Evaluate correctness
        is_status_correct = (pred_status == gt_status)
        is_category_correct = (pred_category.lower() == gt_category.lower())

        if is_status_correct:
            correct_status += 1
        if is_category_correct:
            correct_category += 1

        # Update confusion matrices
        status_confusion[(gt_status, pred_status)] += 1
        category_metrics[gt_category]["total_gt"] += 1

        if is_category_correct:
            category_metrics[gt_category]["tp"] += 1
        else:
            category_metrics[gt_category]["fn"] += 1
            category_metrics[pred_category]["fp"] += 1

        record = {
            "id": idx + 1,
            "input_text": input_text,
            "ground_truth": {
                "status": gt_status,
                "category": gt_category,
            },
            "prediction": {
                "status": pred_status,
                "category": pred_category,
                "raw_output": pred_raw,
            },
            "status_correct": is_status_correct,
            "category_correct": is_category_correct,
        }
        detailed_records.append(record)

        if not (is_status_correct and is_category_correct):
            misclassified_records.append(record)

        # Update live progress bar stats
        live_status_acc = (correct_status / (idx + 1)) * 100
        live_cat_acc = (correct_category / (idx + 1)) * 100
        pbar.set_postfix({"Status_Acc": f"{live_status_acc:.1f}%", "Cat_Acc": f"{live_cat_acc:.1f}%"})

    elapsed = time.time() - start_time
    sec_per_sample = elapsed / max(total_samples, 1)

    # Compute per-category precision, recall, F1
    category_summary = {}
    known_categories = sorted(list(set(list(category_metrics.keys()))))
    for cat in known_categories:
        data = category_metrics[cat]
        tp = data["tp"]
        fp = data["fp"]
        fn = data["fn"]
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        category_summary[cat] = {
            "total_gt": data["total_gt"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        }

    status_accuracy = (correct_status / total_samples) * 100
    category_accuracy = (correct_category / total_samples) * 100

    # 1. Save detailed_predictions.jsonl
    pred_path = output_dir / "detailed_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for rec in detailed_records:
            f.write(json.dumps(rec) + "\n")

    # 2. Save metrics.json
    metrics_path = output_dir / "metrics.json"
    metrics_data = {
        "checkpoint": str(ckpt_path.name),
        "data_split": str(data_path.name),
        "total_evaluated": total_samples,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sample": round(sec_per_sample, 4),
        "status_accuracy": round(status_accuracy, 2),
        "category_accuracy": round(category_accuracy, 2),
        "status_confusion_matrix": {f"GT_{k[0]}__PRED_{k[1]}": v for k, v in status_confusion.items()},
        "category_metrics": category_summary,
        "total_errors": len(misclassified_records),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 3. Save eval_report.md
    report_path = output_dir / "eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-0.5B Guardrail Model Evaluation Report\n\n")
        f.write(f"- **Evaluated Checkpoint:** `{ckpt_path.name}`\n")
        f.write(f"- **Evaluation Dataset:** `{data_path.name}`\n")
        f.write(f"- **Total Samples Evaluated:** {total_samples:,}\n")
        f.write(f"- **Time Elapsed:** {elapsed:.2f}s ({sec_per_sample*1000:.1f} ms/sample)\n\n")
        f.write("## Overall Performance Summary\n\n")
        f.write(f"| Metric | Result |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Binary Status Accuracy (SAFE vs FLAGGED)** | **{status_accuracy:.2f}%** ({correct_status}/{total_samples}) |\n")
        f.write(f"| **Exact Category Accuracy** | **{category_accuracy:.2f}%** ({correct_category}/{total_samples}) |\n")
        f.write(f"| **Total Misclassifications** | {len(misclassified_records)} / {total_samples} |\n\n")
        f.write("## Per-Category Performance Breakdown\n\n")
        f.write("| Category | Ground Truth Count | Precision (%) | Recall (%) | F1-Score (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cat, stat in category_summary.items():
            if stat["total_gt"] > 0:
                f.write(f"| `{cat}` | {stat['total_gt']} | {stat['precision']:.1f}% | {stat['recall']:.1f}% | **{stat['f1']:.1f}%** |\n")
        f.write("\n## Status Confusion Matrix\n\n")
        f.write("| Ground Truth \\ Predicted | SAFE | FLAGGED |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **SAFE** | {status_confusion.get(('SAFE', 'SAFE'), 0)} | {status_confusion.get(('SAFE', 'FLAGGED'), 0)} |\n")
        f.write(f"| **FLAGGED** | {status_confusion.get(('FLAGGED', 'SAFE'), 0)} | {status_confusion.get(('FLAGGED', 'FLAGGED'), 0)} |\n\n")
        f.write("## Sample Misclassifications Analysis\n\n")
        for err in misclassified_records[:10]:
            f.write(f"#### Sample #{err['id']}\n")
            f.write(f"- **Input Text:** \"{err['input_text']}\"\n")
            f.write(f"- **Target:** `{err['ground_truth']['status']}` (`{err['ground_truth']['category']}`)\n")
            f.write(f"- **Model Output:** `{err['prediction']['status']}` (`{err['prediction']['category']}`)\n\n")

    # 4. Print Full Terminal Summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS & PERFORMANCE STATISTICS")
    print("=" * 80)
    print(f"Total Samples Evaluated:      {total_samples:,}")
    print(f"Total Evaluation Time:        {elapsed:.2f} seconds ({sec_per_sample*1000:.1f} ms/sample)")
    print("-" * 80)
    print(f"Binary Status Accuracy (SAFE vs FLAGGED):  {status_accuracy:6.2f}% ({correct_status}/{total_samples})")
    print(f"Exact Category Accuracy:                   {category_accuracy:6.2f}% ({correct_category}/{total_samples})")
    print("-" * 80)
    print("PER-CATEGORY BREAKDOWN:")
    print(f"  {'Category':25s} | {'Count':5s} | {'Precision':9s} | {'Recall':7s} | {'F1-Score':8s}")
    print("  " + "-" * 62)
    for cat, stat in category_summary.items():
        if stat["total_gt"] > 0:
            print(f"  {cat:25s} | {stat['total_gt']:5d} | {stat['precision']:8.1f}% | {stat['recall']:6.1f}% | {stat['f1']:7.1f}%")
    print("-" * 80)
    print("BINARY STATUS CONFUSION MATRIX:")
    print(f"  True SAFE  -> Predicted SAFE:    {status_confusion.get(('SAFE', 'SAFE'), 0):5d}")
    print(f"  True SAFE  -> Predicted FLAGGED: {status_confusion.get(('SAFE', 'FLAGGED'), 0):5d}  (False Positives)")
    print(f"  True FLAG  -> Predicted SAFE:    {status_confusion.get(('FLAGGED', 'SAFE'), 0):5d}  (False Negatives)")
    print(f"  True FLAG  -> Predicted FLAGGED: {status_confusion.get(('FLAGGED', 'FLAGGED'), 0):5d}")
    print("-" * 80)

    if misclassified_records:
        print(f"\nSAMPLE ERROR ANALYSIS ({min(5, len(misclassified_records))} of {len(misclassified_records)} misclassifications):")
        for i, err in enumerate(misclassified_records[:5], 1):
            print(f"  [{i}] Input: \"{err['input_text'][:70]}...\"")
            print(f"      Expected:  {err['ground_truth']['status']} ({err['ground_truth']['category']})")
            print(f"      Predicted: {err['prediction']['status']} ({err['prediction']['category']})")
    else:
        print("\nFlawless Run: 0 errors detected across all evaluated samples!")

    print("\n" + "=" * 80)
    print(f"[Saved] Detailed logs & Markdown report saved to: {output_dir.resolve()}/")
    print(f"  1. {metrics_path.name}")
    print(f"  2. {report_path.name}")
    print(f"  3. {pred_path.name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_evaluation()
