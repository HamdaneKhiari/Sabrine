"""
dataloader.py
Prépare les données tokenizées pour l'entraînement : charge le corpus,
le tokenize avec le tokenizer entraîné, et fournit des batchs aléatoires.

Tokenize une seule fois et met en cache sur disque (data/cache/train.bin,
data/cache/val.bin) au format memmap — les lancements suivants chargent le
cache directement (quasi instantané) au lieu de retokenizer tous les
fichiers en Python à chaque fois.

La tokenisation se fait par paquets de fichiers (CHUNK_FILES), écrits sur
disque au fur et à mesure, plutôt que d'accumuler tout le corpus dans une
seule liste Python avant d'écrire. Sur un gros corpus (des centaines de
milliers de fichiers), garder tout en RAM sous forme de liste d'entiers
Python peut représenter plusieurs dizaines de Go rien qu'en overhead
d'objets — au-delà de la RAM d'une instance Colab standard, ça ne plante
pas proprement, ça se met à swapper et ressemble à un blocage. Le
streaming par paquets borne la mémoire utilisée, peu importe la taille
du corpus.

Si tu changes les données sources (nouvelle collecte, nouveau tokenizer),
supprime le dossier data/cache/ pour forcer une retokenisation.
"""

import os
import glob
import numpy as np
import torch
from tokenizers import Tokenizer

CHUNK_FILES = 5000  # nombre de fichiers tokenizés avant chaque écriture sur disque


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


def _tokenize_files_to_disk(files: list, tokenizer: Tokenizer, separator_ids: list,
                             out_path: str, label: str) -> int:
    """Tokenize une liste de fichiers par paquets de CHUNK_FILES, en ajoutant chaque
    paquet directement au fichier binaire sur disque (mode append). Retourne le
    nombre total de tokens écrits.
    """
    total_tokens = 0
    report_every = max(len(files) // 20, 1)

    with open(out_path, "wb") as out:
        chunk_ids = []
        for i, f in enumerate(files):
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue

            chunk_ids.extend(tokenizer.encode(content).ids)
            chunk_ids.extend(separator_ids)

            # Paquet plein (ou dernier fichier) : on écrit et on vide la mémoire du paquet
            if len(chunk_ids) >= 0 and ((i + 1) % CHUNK_FILES == 0 or (i + 1) == len(files)):
                if chunk_ids:
                    arr = np.array(chunk_ids, dtype=np.uint16)
                    arr.tofile(out)
                    total_tokens += len(arr)
                    chunk_ids = []

            if (i + 1) % report_every == 0 or (i + 1) == len(files):
                print(f"[Sabrina: Codex]   [{label}] {i + 1}/{len(files)} fichiers tokenizés...")

    return total_tokens


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

            # Split au niveau fichier (pas au niveau token) : permet d'écrire train et val
            # dans deux fichiers séparés au fur et à mesure, sans connaître le nombre total
            # de tokens à l'avance. L'ordre des fichiers vient d'un dataset déjà mélangé,
            # donc un split par index de fichier est équivalent à un split par position de
            # token pour nos besoins (pas de dépendance à l'ordre entre fichiers).
            split_idx = int(len(files) * (1 - val_split))
            train_files, val_files = files[:split_idx], files[split_idx:]

            print(
                f"[Sabrina: Codex] Aucun cache trouvé — tokenization de {len(files)} fichiers "
                f"par paquets de {CHUNK_FILES} (une seule fois ; les prochains lancements "
                f"réutiliseront {cache_dir}/)..."
            )

            n_train = _tokenize_files_to_disk(train_files, tokenizer, separator_ids, train_cache, "train")
            n_val = _tokenize_files_to_disk(val_files, tokenizer, separator_ids, val_cache, "val")

            print(f"[Sabrina: Codex] Corpus tokenizé : {n_train + n_val} tokens (train={n_train}, val={n_val})")
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