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
import math
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
from olm.train.schedulers.base import SchedulerBase


class CooldownCosineScheduler(SchedulerBase):
    """
    Smooth cosine cooldown from start_step to target_step down to min_lr.
    Guarantees:
      - At start_step, LR exactly equals the start_lr (0 discontinuity/shock).
      - At target_step, LR smoothly touches min_lr.
    """

    def __init__(
        self,
        optimizer,
        start_step: int,
        target_step: int,
        min_lr: float = 3.0e-5,
        last_epoch: int = -1,
    ):
        self.start_step = start_step
        self.target_step = target_step
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.start_step:
            return self.base_lrs
        if self.last_epoch >= self.target_step:
            return [self.min_lr for _ in self.base_lrs]

        progress = (self.last_epoch - self.start_step) / max(1, self.target_step - self.start_step)
        progress = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * factor
            for base_lr in self.base_lrs
        ]


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (default: from config, e.g. 5)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Override checkpoint save interval in steps (default: 1000)",
    )
    parser.add_argument(
        "--reset_dataloader",
        action="store_true",
        help="Restart dataset stream from sample 0 (essential when resuming for Epoch 2)",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="Override starting step count (e.g. 18878 when resuming for Epoch 2)",
    )
    parser.add_argument(
        "--final_name",
        type=str,
        default=None,
        help="Filename for the final saved model weights (e.g. qwen_0_5b_final_epoch2.pt)",
    )
    parser.add_argument(
        "--cooldown",
        action="store_true",
        help="Enable cooldown phase: smoothly decays LR from current level to min_lr by target_step",
    )
    parser.add_argument(
        "--cooldown_target_step",
        type=int,
        default=37756,
        help="Target step for cooldown to reach min_lr (default: 37756, end of Epoch 2)",
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
    elif args.cooldown and args.cooldown_target_step:
        config["training"]["max_steps"] = args.cooldown_target_step
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["optimizer"]["lr"] = args.learning_rate
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.save_every:
        config["checkpoint"]["save_every"] = args.save_every
    if args.final_name:
        config["checkpoint"]["final_name"] = args.final_name

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
    batch_size = train_config.get("batch_size", 8)
    grad_accum_steps = train_config.get("gradient_accumulation_steps", 16)
    context_length = data_config.get("context_length", 1024)
    eff_batch_tokens = batch_size * grad_accum_steps * context_length
    target_epochs = train_config.get("epochs", 5)

    print(f"Context Length:    {context_length}")
    print(f"Batch Size:        {batch_size} (per device)")
    print(f"Grad Accum Steps:  {grad_accum_steps}")
    print(f"Effective Batch:   {eff_batch_tokens:,} tokens per optimizer update")
    print(f"Target Max Steps:  {train_config.get('max_steps', 76293):,} steps")
    print(f"Target Epochs:     {target_epochs}")
    if args.cooldown:
        print(f"Cooldown Mode:     ENABLED (Decaying smoothly to step {args.cooldown_target_step:,})")
    print("=" * 80)

    # Resumption handling
    skip_batches = 0
    resume_step = 0
    checkpoint_data = None
    checkpoint_lr = None

    if args.resume:
        print(f"\n[Checkpoint] Loading resumption checkpoint: {args.resume}")
        checkpoint_data = torch.load(args.resume, map_location="cpu")
        if args.initial_step is not None:
            resume_step = args.initial_step
        elif isinstance(checkpoint_data, dict) and "step" in checkpoint_data:
            resume_step = checkpoint_data["step"]
        elif "final" in str(args.resume).lower():
            # If resuming from qwen_0_5b_final.pt (epoch 1 completed at step 18878)
            resume_step = 18878
            print(f"[Checkpoint] Resuming from final checkpoint; starting step initialized to {resume_step:,} (Epoch 2).")
        else:
            resume_step = 0

        # Extract active learning rate from checkpoint
        if isinstance(checkpoint_data, dict):
            if "optimizer_state_dict" in checkpoint_data:
                try:
                    checkpoint_lr = checkpoint_data["optimizer_state_dict"]["param_groups"][0]["lr"]
                except Exception:
                    pass
            if checkpoint_lr is None and "scheduler_state_dict" in checkpoint_data:
                try:
                    checkpoint_lr = checkpoint_data["scheduler_state_dict"]["_last_lr"][0]
                except Exception:
                    pass

        # Fallback estimation of LR if needed
        if checkpoint_lr is None and args.cooldown:
            decay_steps = max(1, 76293 - 1500)
            progress = min(max((resume_step - 1500) / decay_steps, 0.0), 1.0)
            checkpoint_lr = 3.0e-5 + (opt_config.get("lr", 3.0e-4) - 3.0e-5) * 0.5 * (1.0 + math.cos(math.pi * progress))

        num_workers = data_config.get("num_workers", 4)
        eff_samples_per_step = batch_size * grad_accum_steps

        # When starting a new epoch, reset dataloader to sample 0
        if args.reset_dataloader or "final" in str(args.resume).lower():
            skip_batches = 0
            print(f"[Checkpoint] Resetting dataloader: stream starting from sample 0 (Epoch 2).")
        else:
            # Resuming mid-epoch: calculate steps completed in the current epoch (each epoch is ~18,878 steps)
            epoch_steps = 18878
            steps_in_current_epoch = resume_step % epoch_steps
            total_samples_to_skip = steps_in_current_epoch * eff_samples_per_step
            skip_batches = total_samples_to_skip // max(1, num_workers)
            print(
                f"[Checkpoint] Resuming from step {resume_step:,} "
                f"({steps_in_current_epoch:,} steps / {total_samples_to_skip:,} samples into current epoch, skipping ~{skip_batches:,} per worker)"
            )

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
    if checkpoint_data is not None:
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

    # 3. Optimizer Configuration (with parameter group decay separation)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": opt_config.get("weight_decay", 0.1)},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    use_fused = (device.type == "cuda")
    optimizer_lr = checkpoint_lr if (args.cooldown and checkpoint_lr is not None) else opt_config.get("lr", 3e-4)
    optimizer = AdamW(
        optim_groups,
        lr=optimizer_lr,
        betas=tuple(opt_config.get("betas", [0.9, 0.95])),
        eps=float(opt_config.get("eps", 1e-8)),
        fused=use_fused,
    )
    print(f"Configured AdamW (lr={optimizer_lr:.2e}, betas={opt_config.get('betas', [0.9, 0.95])}, fused={use_fused})")

    # 4. Callbacks
    checkpoint_dir = ckpt_config.get("checkpoint_dir", "./checkpoints")
    save_every = ckpt_config.get("save_every", 1000)
    keep_last_n = ckpt_config.get("keep_last_n", 5)
    checkpoint_cb = CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        save_every=save_every,
        keep_last_n=keep_last_n,
        save_best=ckpt_config.get("save_best", False),
    )
    metrics_cb = MetricsLoggerCallback(
        log_dir="logs",
        log_every=log_config.get("log_every", 20),
    )
    effective_batch = batch_size * grad_accum_steps
    throughput_cb = ThroughputCallback(
        log_every=log_config.get("log_every", 20),
        context_length=context_length,
        batch_size=effective_batch,
    )

    grad_clip_norm = None
    if config.get("grad_clip", {}).get("enabled", True):
        grad_clip_norm = float(config.get("grad_clip", {}).get("max_norm", 1.0))

    # Ensure param_groups have initial_lr populated for PyTorch scheduler resumption
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    # 5. Learning Rate Scheduler (Warmup + Cosine Decay OR Smooth Cooldown)
    if args.cooldown:
        scheduler = CooldownCosineScheduler(
            optimizer,
            start_step=resume_step,
            target_step=args.cooldown_target_step,
            min_lr=float(sched_config.get("min_lr", 3.0e-5)),
            last_epoch=resume_step - 1 if args.resume else -1,
        )
        print(
            f"[Scheduler] Configured CooldownCosineScheduler: "
            f"step {resume_step:,} ({optimizer_lr:.2e}) -> step {args.cooldown_target_step:,} ({float(sched_config.get('min_lr', 3.0e-5)):.2e})"
        )
    else:
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
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=grad_clip_norm,
        use_amp=train_config.get("use_amp", True),
        scheduler=scheduler,
        callbacks=[checkpoint_cb, metrics_cb, throughput_cb],
    )

    # Resume optimizer, scaler, and scheduler if requested
    if checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data:
            try:
                trainer.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                print("[Checkpoint] Optimizer state loaded successfully.")
                if args.cooldown:
                    for group in trainer.optimizer.param_groups:
                        group["lr"] = optimizer_lr
                        group["initial_lr"] = optimizer_lr
            except Exception as e:
                print(f"[Warning] Could not restore optimizer state: {e}")
        if "scaler_state_dict" in checkpoint_data and trainer.scaler:
            try:
                trainer.scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
                print("[Checkpoint] Scaler state loaded successfully.")
            except Exception as e:
                print(f"[Warning] Could not restore scaler state: {e}")
        trainer.global_step = resume_step
        if args.cooldown:
            current_lr = trainer.scheduler.get_lr()[0]
            print(
                f"[Cooldown] Active: scheduler advancing smoothly from step {resume_step:,} (LR: {current_lr:.2e}) "
                f"to step {args.cooldown_target_step:,} (min LR: {float(sched_config.get('min_lr', 3.0e-5)):.2e})."
            )
        elif "scheduler_state_dict" in checkpoint_data and trainer.scheduler:
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

    epochs = train_config.get("epochs", 5)
    try:
        trainer.train(
            epochs=epochs,
            max_steps=train_config.get("max_steps", 76293),
            log_interval=log_config.get("log_every", 20),
        )
    except KeyboardInterrupt:
        handle_interrupt(signal.SIGINT, None)

    elapsed_hours = (time.time() - start_time) / 3600
    print("\n" + "=" * 80)
    print(f"Pretraining Completed in {elapsed_hours:.2f} hours!")
    final_filename = ckpt_config.get(
        "final_name",
        "qwen_0_5b_final_epoch2.pt" if args.resume else "qwen_0_5b_final.pt"
    )
    final_model_path = os.path.join(checkpoint_dir, final_filename)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model weights saved to: {final_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

