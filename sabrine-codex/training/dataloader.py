"""
dataloader.py
Prépare les données tokenizées pour l'entraînement : charge le corpus,
le tokenize avec le tokenizer entraîné, et fournit des batchs aléatoires.
"""

import os
import glob
import torch
from tokenizers import Tokenizer


def load_corpus_text(data_dir: str = "data/raw") -> str:
    """Concatène tout le texte des fichiers de code trouvés dans data_dir."""
    extensions = ["*.py", "*.js", "*.ts", "*.sh", "*.java", "*.cpp", "*.c", "*.go", "*.rs", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé dans {data_dir}/")

    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read())

    # On sépare chaque fichier par un marqueur pour éviter que le modèle
    # n'apprenne des transitions artificielles entre deux fichiers différents
    return "\n<|endofcode|>\n".join(texts)


def load_tokenizer(path: str = "tokenizer/sabrina_tokenizer.json") -> Tokenizer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer introuvable à {path}. "
            "Lance d'abord tokenizer/train_tokenizer.py."
        )
    return Tokenizer.from_file(path)


class CodeDataset:
    """Prépare un tenseur unique de tokens, et sait en tirer des batchs aléatoires."""

    def __init__(self, data_dir: str, tokenizer_path: str, block_size: int, val_split: float = 0.1):
        text = load_corpus_text(data_dir)
        tokenizer = load_tokenizer(tokenizer_path)

        ids = tokenizer.encode(text).ids
        data = torch.tensor(ids, dtype=torch.long)

        print(f"[Sabrina: Codex] Corpus tokenizé : {len(data)} tokens")

        split_idx = int(len(data) * (1 - val_split))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
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

        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)
