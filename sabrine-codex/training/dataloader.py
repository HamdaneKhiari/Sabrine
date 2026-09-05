"""
dataloader.py
Prépare les données tokenizées pour l'entraînement : charge le corpus,
le tokenize avec le tokenizer entraîné, et fournit des batchs aléatoires.

Tokenize une seule fois et met en cache sur disque (data/cache/train.bin,
data/cache/val.bin) au format memmap — les lancements suivants chargent le
cache directement (quasi instantané) au lieu de retokenizer tous les
fichiers en Python à chaque fois. Indispensable dès que le corpus dépasse
quelques dizaines de milliers de fichiers (retokenizer à chaque run devient
invivable en temps, et charger le résultat entier en RAM ne passera plus à
l'échelle sur un corpus de plusieurs milliards de tokens — le memmap laisse
l'OS ne paginer que ce qui est réellement lu).

Si tu changes les données sources (nouvelle collecte, nouveau tokenizer),
supprime le dossier data/cache/ pour forcer une retokenisation.
"""

import os
import glob
import numpy as np
import torch
from tokenizers import Tokenizer


def collect_files(data_dir: str) -> list:
    extensions = ["*.py", "*.js", "*.ts", "*.sh", "*.java", "*.cpp", "*.c", "*.go", "*.rs", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {data_dir}/")

    return files


def load_tokenizer(path: str = "tokenizer/sabrina_tokenizer.json") -> Tokenizer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer introuvable à {path}. "
            "Lance d'abord tokenizer/train_tokenizer.py."
        )
    return Tokenizer.from_file(path)


class CodeDataset:
    """Prépare un cache de tokens sur disque (memmap), et sait en tirer des batchs aléatoires."""

    def __init__(self, data_dir: str, tokenizer_path: str, block_size: int,
                 val_split: float = 0.1, cache_dir: str = "data/cache"):
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, "train.bin")
        val_cache = os.path.join(cache_dir, "val.bin")

        if os.path.exists(train_cache) and os.path.exists(val_cache):
            print(f"[Sabrina: Codex] Cache trouvé dans {cache_dir}/ — chargement direct, pas de retokenisation.")
        else:
            files = collect_files(data_dir)
            tokenizer = load_tokenizer(tokenizer_path)

            # Token de séparateur entre fichiers, encodé une seule fois
            separator_ids = tokenizer.encode("\n<|endofcode|>\n").ids

            print(
                f"[Sabrina: Codex] Aucun cache trouvé — tokenization de {len(files)} fichiers "
                f"(une seule fois ; les prochains lancements réutiliseront {cache_dir}/)..."
            )

            all_ids = []
            report_every = max(len(files) // 20, 1)  # ~20 messages de progression au total

            for i, f in enumerate(files):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                except OSError:
                    continue

                all_ids.extend(tokenizer.encode(content).ids)
                all_ids.extend(separator_ids)

                if (i + 1) % report_every == 0 or (i + 1) == len(files):
                    print(f"[Sabrina: Codex]   {i + 1}/{len(files)} fichiers tokenizés...")

            # uint16 : vocab_size=16000 tient largement sur 16 bits (max 65535) — divise
            # par 4 la place sur disque/RAM comparé à un tensor int64 par défaut.
            data = np.array(all_ids, dtype=np.uint16)
            print(f"[Sabrina: Codex] Corpus tokenizé : {len(data)} tokens")

            split_idx = int(len(data) * (1 - val_split))
            data[:split_idx].tofile(train_cache)
            data[split_idx:].tofile(val_cache)
            print(f"[Sabrina: Codex] Cache écrit dans {cache_dir}/ (train.bin, val.bin).")

        # memmap dans les deux cas (cache déjà présent ou tout juste écrit) : l'OS ne
        # charge en RAM que les portions réellement lues par get_batch, pas tout le corpus.
        self.train_data = np.memmap(train_cache, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(val_cache, dtype=np.uint16, mode="r")
        self.block_size = block_size

        min_required = block_size + 2  # marge de sécurité pour l'échantillonnage aléatoire
        if len(self.train_data) <= min_required:
            raise ValueError(
                f"Corpus d'entraînement trop petit ({len(self.train_data)} tokens) "
                f"pour block_size={block_size}. Ajoute plus de fichiers dans data/raw/, "
                f"ou réduis block_size dans model/config.py."
            )
        if len(self.val_data) <= min_required:
            raise ValueError(
                f"Corpus de validation trop petit ({len(self.val_data)} tokens) "
                f"pour block_size={block_size}. Ajoute plus de fichiers dans data/raw/, "
                f"réduis block_size, ou réduis val_split."
            )

    def get_batch(self, split: str, batch_size: int, device: str = "cpu"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))

        # .astype(np.int64) sur chaque petite tranche seulement (pas tout le corpus) —
        # torch veut du int64 pour les indices d'embedding, uint16 ne suffit pas côté torch.
        x = torch.stack([torch.from_numpy(data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + self.block_size + 1].astype(np.int64)) for i in ix])

        return x.to(device), y.to(device)