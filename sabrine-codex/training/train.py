"""
train.py
Boucle d'entraînement pour Sabrina: Codex.

Usage :
    python training/train.py

Prérequis :
    - Un tokenizer déjà entraîné (tokenizer/sabrina_tokenizer.json)
    - Du corpus dans data/raw/
"""

import os
import time
import torch

from model.config import config
from model.architecture import SabrinaCodex
from training.dataloader import CodeDataset


def estimate_loss(model, dataset, eval_iters, batch_size, device):
    """Moyenne la loss sur plusieurs batchs, en train et en val, pour un signal plus stable."""
    model.eval()
    results = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = dataset.get_batch(split, batch_size, device)
                _, loss = model(x, targets=y)
                losses[k] = loss.item()
            results[split] = losses.mean().item()
    model.train()
    return results


def main():
    torch.manual_seed(config.seed)

    print("[Sabrina: Codex] Chargement des données...")
    dataset = CodeDataset(
        data_dir="data/processed",
        tokenizer_path="tokenizer/sabrina_tokenizer.json",
        block_size=config.block_size,
    )

    print("[Sabrina: Codex] Initialisation du modèle...")
    model = SabrinaCodex(config).to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    os.makedirs("model/checkpoints", exist_ok=True)
    os.makedirs("training/logs", exist_ok=True)
    log_path = "training/logs/train_log.csv"
    with open(log_path, "w") as f:
        f.write("iter,train_loss,val_loss,elapsed_s\n")

    start_time = time.time()

    print(f"[Sabrina: Codex] Début de l'entraînement — {config.max_iters} itérations sur {config.device}")

    for it in range(config.max_iters):
        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config.eval_iters, config.batch_size, config.device)
            elapsed = time.time() - start_time
            print(
                f"[iter {it:5d}] train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s"
            )
            with open(log_path, "a") as f:
                f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f},{elapsed:.1f}\n")

            # Sauvegarde un checkpoint à chaque évaluation
            torch.save(
                {"model_state": model.state_dict(), "config": config, "iter": it},
                f"model/checkpoints/sabrina_codex_iter{it}.pt",
            )

        x, y = dataset.get_batch("train", config.batch_size, config.device)
        logits, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    # Sauvegarde finale
    torch.save(
        {"model_state": model.state_dict(), "config": config, "iter": config.max_iters},
        "model/checkpoints/sabrina_codex_final.pt",
    )
    print("[Sabrina: Codex] Entraînement terminé. Checkpoint final sauvegardé.")


if __name__ == "__main__":
    main()