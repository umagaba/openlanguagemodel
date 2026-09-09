#!/usr/bin/env python3
"""
Production SFT Post-Training Script for Qwen2.5-0.5B Guardrail Model.
Built using the olm library and optimized for 1x NVIDIA H100 SXM5 GPU.

Features:
- Loads SFT prompt-completion JSONL data with explicit progress tracking.
- Masked prompt loss and padding token masking (-100) for clean next-token classification.
- Automatically loads pretrained base weights from pretraining checkpoints.
- Applies LoRA fine-tuning targeting attention and MLP linear projections.
- Multi-callback pipeline: checkpointing, metrics logging, throughput tracking, validation.
- Saves PEFT adapter, PyTorch state dict, and merged weights for zero-overhead inference.
"""

import os
import sys
import glob
import signal
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

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
from olm.data.datasets import DataLoader
from olm.train.trainer import (
    Trainer,
    CheckpointCallback,
    MetricsLoggerCallback,
    ThroughputCallback,
    ValidationCallback,
)
from olm.train.optim import AdamW
from olm.train.schedulers import WarmupCosineScheduler


def resolve_path(path_str: str) -> Path:
    """Resolve a path relative to CWD, script directory, or repo root."""
    p = Path(path_str)
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    if (script_dir / path_str).exists():
        return (script_dir / path_str).resolve()

    repo_root = script_dir.parent
    if (repo_root / path_str).exists():
        return (repo_root / path_str).resolve()

    return p


class SFTTextDataset:
    """
    Dataset loader for SFT prompt-completion JSONL files.
    Formats examples as causal LM inputs with prompt-loss and padding masking:
      - x: [token_0, token_1, ..., token_{N-1}]
      - y: [label_1, label_2, ..., label_N]
    Prompt tokens and padding tokens in y are set to -100 so CrossEntropyLoss
    computes loss strictly on the assistant's guardrail response tokens.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: HFTokenizer,
        context_length: int = 1024,
        mask_prompt: bool = True,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.mask_prompt = mask_prompt
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        resolved_path = resolve_path(jsonl_path)
        if not resolved_path.is_file():
            print(f"[Warning] SFT dataset not found at '{jsonl_path}' (resolved: '{resolved_path}')")
            return

        pad_id = (
            self.tokenizer.tokenizer.pad_token_id
            if self.tokenizer.tokenizer.pad_token_id is not None
            else self.tokenizer.tokenizer.eos_token_id
        )
        if pad_id is None:
            pad_id = 0

        eos_id = self.tokenizer.tokenizer.eos_token_id

        with open(resolved_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in tqdm(lines, desc=f"Loading SFT Data ({resolved_path.name})"):
            try:
                rec = json.loads(line)
            except Exception:
                continue

            prompt = rec.get("prompt", "")
            completion = rec.get("completion", "")

            # Format prompt and completion strings
            prompt_text = f"{prompt}\n" if prompt else ""
            completion_text = f"{completion}"

            # Encode tokens cleanly whether tokenizer returns Tensor or list
            if prompt_text:
                prompt_enc = self.tokenizer.encode(prompt_text)
                prompt_ids = prompt_enc.tolist() if torch.is_tensor(prompt_enc) else list(prompt_enc)
            else:
                prompt_ids = []

            comp_enc = self.tokenizer.encode(completion_text)
            completion_ids = comp_enc.tolist() if torch.is_tensor(comp_enc) else list(comp_enc)
            if eos_id is not None:
                completion_ids.append(eos_id)

            full_tokens = prompt_ids + completion_ids

            if self.mask_prompt:
                full_labels = [-100] * len(prompt_ids) + list(completion_ids)
            else:
                full_labels = list(full_tokens)

            # Target sequence length is context_length + 1 to yield
            # x = tokens[:-1] and y = labels[1:] both of shape [context_length]
            target_len = self.context_length + 1
            if len(full_tokens) > target_len:
                full_tokens = full_tokens[:target_len]
                full_labels = full_labels[:target_len]
            elif len(full_tokens) < target_len:
                pad_len = target_len - len(full_tokens)
                full_tokens = full_tokens + [pad_id] * pad_len
                full_labels = full_labels + [-100] * pad_len  # Always mask padding

            x = torch.tensor(full_tokens[:-1], dtype=torch.long)
            y = torch.tensor(full_labels[1:], dtype=torch.long)
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


def load_base_weights(model: torch.nn.Module, base_model_path: Optional[str]) -> bool:
    """Load pretrained weights into the base Qwen2.5-0.5B model."""
    candidate_paths = []
    if base_model_path:
        candidate_paths.append(Path(base_model_path))

    # Standard pretraining output candidate locations
    repo_root = Path(__file__).resolve().parent.parent
    pretrain_ckpt_dir = repo_root / "examples" / "qwen2_5-0_5b-fineweb-edu-10b" / "checkpoints"

    candidate_paths.extend([
        pretrain_ckpt_dir / "qwen_0_5b_final_epoch2.pt",
        pretrain_ckpt_dir / "qwen_0_5b_final.pt",
        pretrain_ckpt_dir / "best_model.pt",
    ])

    # Also search for the latest step_*.pt checkpoint
    if pretrain_ckpt_dir.is_dir():
        step_ckpts = sorted(
            pretrain_ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
            reverse=True,
        )
        candidate_paths.extend(step_ckpts)

    found_path = None
    for cand in candidate_paths:
        resolved = resolve_path(str(cand))
        if resolved.is_file():
            found_path = resolved
            break

    if not found_path:
        print("[Warning] No pretrained base model checkpoint found.")
        print(f"          Checked candidates: {[str(c) for c in candidate_paths[:4]]}")
        print("          Proceeding with randomly initialized base weights.")
        return False

    print(f"[Model] Loading pretrained base weights from: {found_path}")
    checkpoint = torch.load(found_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print(f"[Warning] Unexpected checkpoint type {type(checkpoint)} at {found_path}")
        return False

    # Strip prefixes if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]
        clean_state_dict[clean_key] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"[Model] Pretrained weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"        Unexpected keys sample: {unexpected[:4]}")
    if missing:
        print(f"        Missing keys sample: {missing[:4]}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-Train Qwen2.5-0.5B Guardrail Model on SFT Data using olm"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_sft.yaml",
        help="Path to SFT YAML config file",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Path to pretrained base checkpoint (.pt)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to SFT checkpoint to resume (.pt)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per-device batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    resolved = resolve_path(config_path)
    if resolved.is_file():
        print(f"[Config] Loaded configuration from: {resolved}")
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    print(f"[Config] Config file not found at '{config_path}'. Using production defaults.")
    return {
        "base_model": {
            "path": "examples/qwen2_5-0_5b-fineweb-edu-10b/checkpoints/qwen_0_5b_final_epoch2.pt",
        },
        "training": {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_steps": 2000,
            "epochs": 3,
            "lr": 2e-5,
            "device": "cuda",
            "use_amp": True,
            "grad_clip_norm": 1.0,
        },
        "optimizer": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
        "scheduler": {
            "warmup_steps": 100,
            "min_lr": 1e-6,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "data": {
            "train_path": "data/processed/sft/train.jsonl",
            "val_path": "data/processed/sft/val.jsonl",
            "context_length": 1024,
            "tokenizer_name": "Qwen/Qwen2.5-0.5B",
            "mask_prompt": True,
            "num_workers": 4,
        },
        "checkpoint": {
            "checkpoint_dir": "./checkpoints_sft",
            "save_every": 250,
            "keep_last_n": 3,
            "save_best": True,
        },
        "logging": {
            "log_every": 10,
            "eval_every": 250,
        },
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    train_config = config.get("training", {})
    opt_config = config.get("optimizer", {})
    sched_config = config.get("scheduler", {})
    lora_config = config.get("lora", {})
    data_config = config.get("data", {})
    ckpt_config = config.get("checkpoint", {})
    log_config = config.get("logging", {})

    # Apply CLI overrides
    if args.batch_size:
        train_config["batch_size"] = args.batch_size
    if args.learning_rate:
        opt_config["lr"] = args.learning_rate
        train_config["lr"] = args.learning_rate
    if args.max_steps:
        train_config["max_steps"] = args.max_steps
    if args.epochs:
        train_config["epochs"] = args.epochs

    checkpoint_dir = ckpt_config.get("checkpoint_dir", "./checkpoints_sft")
    log_dir = log_config.get("log_dir", "logs_sft")
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    device_str = train_config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Enable TF32 for Hopper (H100) architecture hardware acceleration
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("Qwen2.5-0.5B Guardrail SFT Post-Training with olm Library (H100 Optimized)")
    print("=" * 80)
    print(f"Device:            {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    batch_size = train_config.get("batch_size", 16)
    grad_accum_steps = train_config.get("gradient_accumulation_steps", 2)
    context_length = data_config.get("context_length", 1024)
    mask_prompt = data_config.get("mask_prompt", True)

    # 1. Initialize Tokenizer using olm HFTokenizer
    tokenizer_name = data_config.get("tokenizer_name", "Qwen/Qwen2.5-0.5B")
    print(f"\n[1/5] Loading Tokenizer ('{tokenizer_name}')...")
    tokenizer = HFTokenizer(tokenizer_name)
    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    # 2. Load Datasets with Progress Tracking
    print("\n[2/5] Loading SFT Datasets with Progress Tracking...")
    train_path = data_config.get("train_path", "data/processed/sft/train.jsonl")
    val_path = data_config.get("val_path", "data/processed/sft/val.jsonl")

    train_dataset = SFTTextDataset(train_path, tokenizer, context_length, mask_prompt=mask_prompt)
    if len(train_dataset) == 0:
        print(f"\n[Error] No training examples found in '{train_path}'!")
        print("Please run the data curation pipeline first:")
        print("  cd postrain/dataset_curation")
        print("  python 01_download_hf_datasets.py")
        print("  python 02_normalize.py")
        print("  python 03_clean_and_dedup.py")
        print("  python 04_split.py")
        print("  python 05_format_sft.py")
        sys.exit(1)

    print(f"Loaded {len(train_dataset):,} train examples (context_length={context_length}, mask_prompt={mask_prompt}).")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    val_dataset = SFTTextDataset(val_path, tokenizer, context_length, mask_prompt=mask_prompt)
    val_loader = None
    if len(val_dataset) > 0:
        print(f"Loaded {len(val_dataset):,} validation examples.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(data_config.get("num_workers", 4), 2),
            pin_memory=(device.type == "cuda"),
        )
    else:
        print("[Info] No validation set found or validation set is empty. Proceeding without validation callback.")

    # 3. Instantiate Base Model, Load Pretrained Weights & Wrap with LoRA
    print("\n[3/5] Initializing Qwen2.5-0.5B Base Model & Loading Weights...")
    model = Qwen2_5_0_5B()

    base_model_arg = args.base_model or config.get("base_model", {}).get("path")
    load_base_weights(model, base_model_arg)
    model.to(device)

    target_modules = lora_config.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "out_proj", "up_proj", "down_proj"],
    )
    print(f"[LoRA] Wrapping model with LoRA (r={lora_config.get('r', 16)}, alpha={lora_config.get('lora_alpha', 32)})...")
    print(f"[LoRA] Target modules: {target_modules}")

    peft_config = LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("lora_alpha", 32)),
        target_modules=target_modules,
        lora_dropout=float(lora_config.get("lora_dropout", 0.05)),
        bias=lora_config.get("bias", "none"),
        task_type=None,  # None for custom olm nn.Module
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Optimizer & Scheduler Configuration
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": float(opt_config.get("weight_decay", 0.01))},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    learning_rate = float(opt_config.get("lr", 2e-5))
    optimizer = AdamW(
        optim_groups,
        lr=learning_rate,
        betas=tuple(opt_config.get("betas", [0.9, 0.95])),
        eps=float(opt_config.get("eps", 1e-8)),
        fused=(device.type == "cuda"),
    )

    max_steps = int(train_config.get("max_steps", 2000))
    epochs = int(train_config.get("epochs", 3))

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(sched_config.get("warmup_steps", 100)),
        total_steps=max_steps,
        min_lr=float(sched_config.get("min_lr", 1e-6)),
    )

    # 5. Callbacks & olm Trainer Setup
    checkpoint_cb = CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        save_every=int(ckpt_config.get("save_every", 250)),
        keep_last_n=int(ckpt_config.get("keep_last_n", 3)),
        save_best=bool(ckpt_config.get("save_best", True)) and (val_loader is not None),
    )
    metrics_cb = MetricsLoggerCallback(
        log_dir=log_dir,
        log_every=int(log_config.get("log_every", 10)),
    )
    throughput_cb = ThroughputCallback(
        log_every=int(log_config.get("log_every", 10)),
        context_length=context_length,
        batch_size=batch_size * grad_accum_steps,
    )

    callbacks = [checkpoint_cb, metrics_cb, throughput_cb]
    if val_loader is not None:
        eval_every = int(log_config.get("eval_every", 250))
        val_cb = ValidationCallback(
            val_dataloader=val_loader,
            eval_every=eval_every,
            device=str(device),
            use_amp=train_config.get("use_amp", True),
        )
        callbacks.append(val_cb)

    print("\n[4/5] Configuring olm Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        device=str(device),
        context_length=context_length,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=float(train_config.get("grad_clip_norm", 1.0)),
        use_amp=bool(train_config.get("use_amp", True)),
        scheduler=scheduler,
        callbacks=callbacks,
    )

    # Resumption handling
    if args.resume:
        resume_path = resolve_path(args.resume)
        if resume_path.is_file():
            print(f"[Checkpoint] Resuming SFT state from: {resume_path}")
            checkpoint_data = torch.load(resume_path, map_location="cpu")
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint_data and trainer.scheduler:
                trainer.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            if "step" in checkpoint_data:
                trainer.global_step = checkpoint_data["step"]
            print(f"[Checkpoint] Resumed successfully at global step {trainer.global_step}.")
        else:
            print(f"[Warning] Resume checkpoint not found at: {resume_path}")

    # Graceful exit handler
    def handle_interrupt(sig, frame):
        print(f"\n[Interrupted] Saving emergency checkpoint at step {trainer.global_step}...")
        emergency_path = os.path.join(checkpoint_dir, f"emergency_step_{trainer.global_step}.pt")
        save_dict = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if trainer.scaler:
            save_dict["scaler_state_dict"] = trainer.scaler.state_dict()
        if trainer.scheduler:
            save_dict["scheduler_state_dict"] = trainer.scheduler.state_dict()
        torch.save(save_dict, emergency_path)
        print(f"[Interrupted] Saved emergency checkpoint to: {emergency_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    print("\n" + "=" * 80)
    print("Starting SFT Post-Training Loop on H100 GPU...")
    print(f"Total Epochs:      {epochs}")
    print(f"Max Steps:         {max_steps}")
    print(f"Batch Size:        {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"Learning Rate:     {learning_rate:.2e}")
    print("=" * 80 + "\n")

    start_time = time.time()
    try:
        trainer.train(
            epochs=epochs,
            max_steps=max_steps,
            log_interval=int(log_config.get("log_every", 10)),
        )
    except KeyboardInterrupt:
        handle_interrupt(signal.SIGINT, None)

    elapsed_hours = (time.time() - start_time) / 3600
    print("\n" + "=" * 80)
    print(f"SFT Training Completed in {elapsed_hours:.2f} hours!")

    # 1. Save standard PEFT adapter
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    try:
        model.save_pretrained(adapter_dir)
        print(f"[Save] PEFT adapter saved to: {adapter_dir}")
    except Exception as e:
        print(f"[Warning] Could not save PEFT adapter via save_pretrained: {e}")

    # 2. Save full state dict for olm / checkpoint loading
    final_path = os.path.join(checkpoint_dir, "qwen_0_5b_guardrail_sft_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[Save] Final model state dict saved to: {final_path}")

    # 3. Save merged standalone model weights
    try:
        if hasattr(model, "merge_and_unload"):
            merged_model = model.merge_and_unload()
            merged_path = os.path.join(checkpoint_dir, "qwen_0_5b_guardrail_merged.pt")
            torch.save(merged_model.state_dict(), merged_path)
            print(f"[Save] Merged standalone model weights saved to: {merged_path}")
    except Exception as e:
        print(f"[Note] merge_and_unload skipped: {e}")

    print("=" * 80)


if __name__ == "__main__":
    main()
