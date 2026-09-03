#!/usr/bin/env python3
"""
Production Pretraining Script for Qwen2.5-0.5B on FineWeb-Edu (10B Tokens).

Optimized for 1x NVIDIA H100 SXM5 GPU on the Pramana SLM++ Brev Server.
Supports local NVMe parquet dataset loading, BF16 mixed precision, gradient
accumulation, periodic checkpointing, and safe resumption.

Usage:
    # 1. Standard training from config:
    python -u train.py --config config.yaml

    # 2. Resume from a checkpoint:
    python -u train.py --config config.yaml --resume checkpoints/step_5000.pt

    # 3. Background execution in tmux:
    tmux new -s pretrain
    python -u train.py --config config.yaml
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
from typing import Dict, Any, Optional

import torch

# Ensure olm is accessible (pre-installed in server environment, with repo fallback)
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
from olm.data.datasets import DataLoader, HuggingFaceTextDataset, FineWebEduDataset
from olm.train.trainer import (
    Trainer,
    CheckpointCallback,
    MetricsLoggerCallback,
    ThroughputCallback,
)
from olm.train.optim import AdamW
from olm.train.schedulers import WarmupCosineScheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen2.5-0.5B on FineWeb-Edu 10B Tokens"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) to resume training from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory for local parquet files",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override maximum training steps",
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
        help="Override optimizer learning rate",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_weights(model: torch.nn.Module, std: float = 0.02):
    """
    Applies standard transformer Gaussian initialization (std=0.02, zero biases).
    Ensures initial loss starts cleanly at theoretical uniform entropy ~11.9.
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() > 1:
            torch.nn.init.normal_(p, mean=0.0, std=std)
        elif "bias" in name and p is not None:
            torch.nn.init.zeros_(p)


def build_dataset(data_config: Dict[str, Any], tokenizer: HFTokenizer, skip_batches: int = 0):
    """
    Builds the dataset, prioritizing high-speed local NVMe parquet files if present,
    otherwise falling back to streaming over the network.
    """
    data_dir = data_config.get("data_dir", "./data/fineweb_edu_10bt/sample/10BT")
    context_length = data_config.get("context_length", 1024)

    # Search for local parquet files and sort deterministically
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        # Check subdirectories
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))

    if parquet_files:
        print(f"[Data Loader] Found {len(parquet_files)} local parquet files in {data_dir}.")
        print("[Data Loader] Reading directly from high-speed NVMe storage (zero network latency).")
        dataset = HuggingFaceTextDataset(
            dataset_name="parquet",
            split="train",
            context_length=context_length,
            text_fn=lambda ex: ex["text"],
            tokenizer=tokenizer,
            dataset_kwargs={"data_files": {"train": parquet_files}},
            streaming=True,
            skip_batches=skip_batches,
            shuffle=True,
            seed=42,
        )
    else:
        print(f"[Data Loader] No local parquet files found at '{data_dir}'.")
        print("[Data Loader] Falling back to direct streaming from HuggingFace Hub ('sample-10BT')...")
        print("[Tip] To ensure maximum H100 speed and 0 network drops, run 'python prepare_data.py' first!")
        dataset = FineWebEduDataset(
            tokenizer=tokenizer,
            subset=data_config.get("subset", "sample-10BT"),
            split="train",
            context_length=context_length,
            streaming=True,
            skip_batches=skip_batches,
            shuffle=True,
            seed=42,
        )

    return dataset


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["optimizer"]["lr"] = args.learning_rate

    train_config = config["training"]
    opt_config = config["optimizer"]
    sched_config = config.get("scheduler", {})
    data_config = config["data"]
    ckpt_config = config["checkpoint"]
    log_config = config["logging"]

    # Setup directories
    Path("logs").mkdir(exist_ok=True)
    Path(ckpt_config.get("checkpoint_dir", "./checkpoints")).mkdir(exist_ok=True)

    # Hardware & Device detection
    device_str = train_config.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Enable TF32 for Hopper/Ampere GPU acceleration
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("Qwen2.5-0.5B Pretraining on FineWeb-Edu 10B Tokens")
    print("=" * 80)
    print(f"Device:            {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM Available:    {total_vram:.2f} GB")
    print(f"Context Length:    {data_config.get('context_length', 1024)}")
    print(f"Batch Size:        {train_config.get('batch_size', 32)} (per device)")
    print(f"Grad Accum Steps:  {train_config.get('gradient_accumulation_steps', 4)}")
    eff_batch_tokens = (
        train_config.get("batch_size", 32)
        * train_config.get("gradient_accumulation_steps", 4)
        * data_config.get("context_length", 1024)
    )
    print(f"Effective Batch:   {eff_batch_tokens:,} tokens per optimizer update")
    print(f"Target Max Steps:  {train_config.get('max_steps', 76293):,} steps (~10B tokens)")
    print("=" * 80)

    # Resumption handling
    skip_batches = 0
    resume_step = 0
    if args.resume:
        print(f"\n[Checkpoint] Loading resumption checkpoint: {args.resume}")
        checkpoint_data = torch.load(args.resume, map_location="cpu")
        resume_step = checkpoint_data.get("step", 0)
        num_workers = data_config.get("num_workers", 4)
        eff_samples_per_step = train_config.get("batch_size", 32) * train_config.get("gradient_accumulation_steps", 4)
        total_samples_to_skip = resume_step * eff_samples_per_step
        # Sharded across workers so skip_batches is computed per worker
        skip_batches = total_samples_to_skip // max(1, num_workers)
        print(
            f"[Checkpoint] Resuming from step {resume_step:,} "
            f"({total_samples_to_skip:,} total samples, skipping ~{skip_batches:,} per worker)"
        )
        del checkpoint_data

    # 1. Initialize Tokenizer & Data Loader
    tokenizer_name = data_config.get("tokenizer_name", "Qwen/Qwen2.5-0.5B")
    print(f"\n[1/4] Loading Tokenizer ('{tokenizer_name}')...")
    tokenizer = HFTokenizer(tokenizer_name)
    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
    print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size:,}")

    print("\n[2/4] Initializing Dataset and DataLoader...")
    dataset = build_dataset(data_config, tokenizer, skip_batches=skip_batches)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.get("batch_size", 32),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True) if device.type == "cuda" else False,
    )

    # 2. Instantiate Model
    print("\n[3/4] Initializing Qwen2.5-0.5B Architecture...")
    model = Qwen2_5_0_5B()

    # Apply standard Gaussian initialization if starting fresh
    if not args.resume:
        init_weights(model, std=0.02)
        print("Applied standard transformer weight initialization (std=0.02).")

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

    # Resume model weights if requested
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("_orig_mod."):
                clean_k = clean_k[len("_orig_mod."):]
            if clean_k.startswith("module."):
                clean_k = clean_k[len("module."):]
            cleaned_state_dict[clean_k] = v
        model.load_state_dict(cleaned_state_dict)
        print("[Checkpoint] Model state loaded successfully.")
        del checkpoint_data

    # 3. Optimizer Configuration (with parameter group decay separation)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": opt_config.get("weight_decay", 0.1)},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    use_fused = (device.type == "cuda")
    optimizer = AdamW(
        optim_groups,
        lr=opt_config.get("lr", 3e-4),
        betas=tuple(opt_config.get("betas", [0.9, 0.95])),
        eps=float(opt_config.get("eps", 1e-8)),
        fused=use_fused,
    )
    print(f"Configured AdamW (lr={opt_config.get('lr', 3e-4)}, betas={opt_config.get('betas', [0.9, 0.95])}, fused={use_fused})")

    # 4. Callbacks
    checkpoint_dir = ckpt_config.get("checkpoint_dir", "./checkpoints")
    checkpoint_cb = CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        save_every=ckpt_config.get("save_every", 2500),
        keep_last_n=ckpt_config.get("keep_last_n", 3),
        save_best=ckpt_config.get("save_best", True),
    )
    metrics_cb = MetricsLoggerCallback(
        log_dir="logs",
        log_every=log_config.get("log_every", 20),
    )
    effective_batch = (
        train_config.get("batch_size", 32)
        * train_config.get("gradient_accumulation_steps", 4)
    )
    throughput_cb = ThroughputCallback(
        log_every=log_config.get("log_every", 20),
        context_length=data_config.get("context_length", 1024),
        batch_size=effective_batch,
    )

    grad_clip_norm = None
    if config.get("grad_clip", {}).get("enabled", True):
        grad_clip_norm = float(config.get("grad_clip", {}).get("max_norm", 1.0))

    # 5. Learning Rate Scheduler (Warmup + Cosine Decay)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=int(sched_config.get("warmup_steps", 1500)),
        total_steps=int(train_config.get("max_steps", 76293)),
        min_lr=float(sched_config.get("min_lr", 3.0e-5)),
        last_epoch=resume_step - 1 if args.resume else -1,
    )

    # 6. Trainer
    print("\n[4/4] Configuring Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=str(device),
        context_length=data_config.get("context_length", 1024),
        grad_accum_steps=train_config.get("gradient_accumulation_steps", 4),
        grad_clip_norm=grad_clip_norm,
        use_amp=train_config.get("use_amp", True),
        scheduler=scheduler,
        callbacks=[checkpoint_cb, metrics_cb, throughput_cb],
    )

    # Resume optimizer, scaler, and scheduler if requested
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location="cpu")
        if "optimizer_state_dict" in checkpoint_data:
            try:
                trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                print("[Checkpoint] Optimizer state loaded successfully.")
            except Exception as e:
                print(f"[Warning] Could not restore optimizer state: {e}")
        if "scaler_state_dict" in checkpoint_data and trainer.scaler:
            try:
                trainer.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
                print("[Checkpoint] Scaler state loaded successfully.")
            except Exception as e:
                print(f"[Warning] Could not restore scaler state: {e}")
        trainer.global_step = resume_step
        if "scheduler_state_dict" in checkpoint_data and trainer.scheduler:
            try:
                trainer.scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
                print("[Checkpoint] Scheduler state loaded successfully.")
            except Exception as e:
                print(f"[Warning] Could not restore scheduler state: {e}")
        else:
            print(f"[Checkpoint] Scheduler advanced to step {resume_step:,} (current LR: {trainer.scheduler.get_lr()[0]:.2e}).")
        del checkpoint_data

    # Graceful exit handler
    def handle_interrupt(sig, frame):
        print(f"\n[Interrupted] Received signal {sig}. Saving emergency checkpoint...")
        emergency_path = os.path.join(checkpoint_dir, f"emergency_step_{trainer.global_step}.pt")
        save_data = {
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        }
        if trainer.scaler:
            save_data["scaler_state_dict"] = trainer.scaler.state_dict()
        if trainer.scheduler:
            save_data["scheduler_state_dict"] = trainer.scheduler.state_dict()
        torch.save(save_data, emergency_path)
        print(f"[Interrupted] Saved emergency checkpoint to {emergency_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # Start Pretraining
    print("\n" + "=" * 80)
    print("Starting Pretraining Loop...")
    print("=" * 80 + "\n")
    start_time = time.time()

    try:
        trainer.train(
            epochs=1,
            max_steps=train_config.get("max_steps", 76293),
            log_interval=log_config.get("log_every", 20),
        )
    except KeyboardInterrupt:
        handle_interrupt(signal.SIGINT, None)

    elapsed_hours = (time.time() - start_time) / 3600
    print("\n" + "=" * 80)
    print(f"Pretraining Completed in {elapsed_hours:.2f} hours!")
    final_model_path = os.path.join(checkpoint_dir, "qwen_0_5b_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model weights saved to: {final_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

