"""
train.py
Boucle d'entraînement pour Sabrina: Codex.

Usage :
    python -m training.train
    python -m training.train --resume model/checkpoints/sabrina_codex_final.pt --target_iters 88000

Prérequis :
    - Un tokenizer déjà entraîné (tokenizer/sabrina_tokenizer.json)
    - Du corpus dans data/processed/
"""

import os
import time
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Chemin d'un checkpoint depuis lequel reprendre l'entraînement (modèle + optimizer + itération)."
    )
    parser.add_argument(
        "--target_iters", type=int, default=None,
        help="Itération totale à atteindre (utile surtout avec --resume). Sans --resume, config.max_iters est utilisé."
    )
    args = parser.parse_args()

    torch.manual_seed(config.seed)

    # Bascule automatiquement sur GPU si disponible (Colab, machine avec CUDA...),
    # reste sur CPU sinon — pas besoin d'éditer la config à la main selon la machine.
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Sabrina: Codex] Device utilisé : {config.device}")

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

    start_iter = 0

    if args.resume:
        print(f"[Sabrina: Codex] Reprise depuis le checkpoint : {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print("[Sabrina: Codex] État de l'optimizer restauré (momentum Adam préservé).")
        else:
            print(
                "[Sabrina: Codex] Attention : ce checkpoint n'a pas d'état d'optimizer sauvegardé "
                "(ancien format) — l'optimizer redémarre à zéro, un léger à-coup sur la loss est possible "
                "dans les toutes premières itérations après la reprise."
            )

        start_iter = checkpoint["iter"]
        print(f"[Sabrina: Codex] Reprise à partir de l'itération {start_iter}")

    max_iters = args.target_iters if args.target_iters is not None else config.max_iters

    if start_iter >= max_iters:
        raise ValueError(
            f"target_iters ({max_iters}) doit être supérieur à l'itération de reprise ({start_iter})."
        )

    os.makedirs("model/checkpoints", exist_ok=True)
    os.makedirs("training/logs", exist_ok=True)
    log_path = "training/logs/train_log.csv"

    # En reprise, on continue le même fichier de log plutôt que de l'écraser
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("iter,train_loss,val_loss,elapsed_s\n")

    start_time = time.time()

    print(
        f"[Sabrina: Codex] Entraînement de l'itération {start_iter} à {max_iters} "
        f"sur {config.device}"
    )

    for it in range(start_iter, max_iters):
        if it % config.eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, dataset, config.eval_iters, config.batch_size, config.device)
            elapsed = time.time() - start_time
            print(
                f"[iter {it:5d}] train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s"
            )
            with open(log_path, "a") as f:
                f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f},{elapsed:.1f}\n")

            # Sauvegarde un checkpoint à chaque évaluation (modèle + optimizer, pour permettre une reprise propre)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": config,
                    "iter": it,
                },
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
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "iter": max_iters,
        },
        "model/checkpoints/sabrina_codex_final.pt",
    )
    print("[Sabrina: Codex] Entraînement terminé. Checkpoint final sauvegardé.")


if __name__ == "__main__":
    main()