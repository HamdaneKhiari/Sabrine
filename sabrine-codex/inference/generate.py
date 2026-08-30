"""
generate.py
Charge un checkpoint entraîné de Sabrina: Codex et génère du code
à partir d'un début de texte donné.

Usage :
    python -m inference.generate --prompt "def fibonacci(n):"
    python -m inference.generate --prompt "class Stack:" --max_tokens 100 --temperature 0.8
"""

import argparse
import torch
from tokenizers import Tokenizer

from model.config import SabrinaConfig
from model.architecture import SabrinaCodex


def load_model(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = SabrinaCodex(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"[Sabrina: Codex] Checkpoint chargé (itération {checkpoint['iter']})")
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="def ")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/sabrina_codex_final.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/sabrina_tokenizer.json")
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cpu"

    model, config = load_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    prompt_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n--- Prompt ---\n{args.prompt}")
    print(f"\n--- Génération ({args.max_tokens} tokens, temp={args.temperature}) ---\n")

    output_idx = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_text = tokenizer.decode(output_idx[0].tolist())
    print(generated_text)


if __name__ == "__main__":
    main()