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
from torch.cuda.amp import autocast, GradScaler

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


def get_lr(it: int, warmup_iters: int, max_iters: int, learning_rate: float, min_lr: float) -> float:
    """Warmup linéaire puis décroissance cosinus jusqu'à min_lr.

    Un LR fixe pendant tout l'entraînement tend à donner une loss qui oscille
    au lieu de descendre proprement (surtout en fin de run, quand le LR
    devrait diminuer pour affiner) — le warmup évite aussi une instabilité
    dans les toutes premières itérations où les gradients sont bruités.
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    if it >= max_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / max(max_iters - warmup_iters, 1)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(decay_ratio * 3.141592653589793)).item())
    return min_lr + coeff * (learning_rate - min_lr)


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

    # Mixed precision : poids fp32 gardés en maître, mais forward/backward en fp16
    # sur GPU — divise à peu près par 2 la mémoire des activations, essentiel à 319M
    # (à 15M ce n'était pas nécessaire, on n'a jamais eu besoin de le faire avant).
    # `enabled=False` sur CPU : autocast/scaler deviennent des no-op automatiquement.
    use_amp = config.device == "cuda"
    scaler = GradScaler(enabled=use_amp)

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
    best_val_loss = float("inf")

    print(
        f"[Sabrina: Codex] Entraînement de l'itération {start_iter} à {max_iters} "
        f"sur {config.device}"
    )

    for it in range(start_iter, max_iters):
        if it % config.eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, dataset, config.eval_iters, config.batch_size, config.device)
            elapsed = time.time() - start_time
            current_lr = get_lr(it, config.warmup_iters, max_iters, config.learning_rate, config.min_lr)
            print(
                f"[iter {it:5d}] train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | lr {current_lr:.2e} | {elapsed:.1f}s"
            )
            with open(log_path, "a") as f:
                f.write(f"{it},{losses['train']:.4f},{losses['val']:.4f},{elapsed:.1f}\n")

            # Un seul checkpoint "latest", écrasé à chaque éval : à 15M un fichier par
            # itération tenait sur le disque (~240 Mo), à 319M chacun pèse plusieurs Go
            # et saturait Colab en quelques évals. On garde aussi "best" séparément,
            # uniquement quand la val loss s'améliore vraiment.
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "iter": it,
            }
            torch.save(checkpoint, "model/checkpoints/sabrina_codex_latest.pt")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(checkpoint, "model/checkpoints/sabrina_codex_best.pt")

        x, y = dataset.get_batch("train", config.batch_size, config.device)

        with autocast(enabled=use_amp):
            logits, loss = model(x, targets=y)

        lr = get_lr(it, config.warmup_iters, max_iters, config.learning_rate, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # nécessaire avant clip_grad_norm_, sinon la norme est calculée sur les gradients mis à l'échelle
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

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