#!/usr/bin/env python3
"""
Interactive Inference & Checkpoint Verification for Qwen2.5-0.5B.

Loads a saved training checkpoint and performs autoregressive text generation
to verify the model learned coherent language structure before downloading.

Usage:
    # 1. Test with default prompt:
    python inference.py --checkpoint checkpoints/best_model.pt

    # 2. Test with custom prompt:
    python inference.py --checkpoint checkpoints/step_5000.pt --prompt "Machine learning models must be trained with safety in mind because"

    # 3. Interactive prompt mode:
    python inference.py --checkpoint checkpoints/best_model.pt --interactive
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

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
    parser = argparse.ArgumentParser(description="Run inference on Qwen2.5-0.5B checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Artificial intelligence safety is critical because",
        help="Input text prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling threshold (default: 40)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Repetition penalty factor (default: 1.15)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive REPL prompt mode",
    )
    return parser.parse_args()


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: HFTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cuda",
) -> str:
    """Autoregressive text generation with top-k, top-p, and repetition penalty sampling."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    if isinstance(tokens, torch.Tensor):
        input_ids = tokens.unsqueeze(0).to(device, dtype=torch.long)
    else:
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Forward pass through model
        logits = model(input_ids)
        # Take logits from the final token position: [1, vocab_size]
        next_token_logits = logits[:, -1, :].clone()

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop if EOS token
        if next_token.item() == tokenizer.tokenizer.eos_token_id:
            break

    generated_tokens = input_ids[0].tolist()
    return tokenizer.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_model(checkpoint_path: str, device: str):
    """Initializes Qwen2.5-0.5B and loads checkpoint weights."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("Available checkpoints in ./checkpoints/:")
        if os.path.exists("./checkpoints"):
            for f in os.listdir("./checkpoints"):
                if f.endswith(".pt"):
                    print(f"  - checkpoints/{f}")
        sys.exit(1)

    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = Qwen2_5_0_5B()
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[len("_orig_mod."):]
        if clean_k.startswith("module."):
            clean_k = clean_k[len("module."):]
        cleaned_state_dict[clean_k] = v

    model.load_state_dict(cleaned_state_dict)
    step = checkpoint.get("step", "unknown") if isinstance(checkpoint, dict) else "unknown"
    print(f"Successfully loaded checkpoint from training step: {step}")

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    print("Loading Qwen2.5 tokenizer...")
    tokenizer = HFTokenizer("Qwen/Qwen2.5-0.5B")

    # Load Model
    model = load_model(args.checkpoint, device)

    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive Mode: Enter your prompt (or 'quit' / 'exit'):")
        print("=" * 60)
        while True:
            try:
                user_prompt = input("\nPrompt >> ")
                if user_prompt.strip().lower() in ["quit", "exit"]:
                    break
                if not user_prompt.strip():
                    continue

                output = generate(
                    model,
                    tokenizer,
                    user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=device,
                )
                print(f"\nResponse:\n{output}")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print("\n" + "=" * 60)
        print(f"Prompt: \"{args.prompt}\"")
        print("=" * 60)
        output = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(f"\nGenerated Output:\n{output}\n")


if __name__ == "__main__":
    main()

