"""
config.py
Hyperparamètres du modèle Sabrina: Codex.

Départ volontairement minuscule : l'objectif de cette première phase est de
valider tout le pipeline (données -> tokenizer -> modèle -> training -> génération),
pas d'avoir un modèle déjà performant. On scale une fois que ça marche de bout en bout.
"""

from dataclasses import dataclass


@dataclass
class SabrinaConfig:
    # --- Vocabulaire ---
    vocab_size: int = 32000       # doit correspondre au tokenizer entraîné

    # --- Architecture (volontairement petit pour un run CPU) ---
    n_layer: int = 4              # nombre de blocs Transformer
    n_head: int = 4               # nombre de têtes d'attention
    n_embd: int = 128             # dimension des embeddings (d_model)
    block_size: int = 256         # taille max du contexte (en tokens)
    dropout: float = 0.1

    # --- Entraînement ---
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # --- Divers ---
    device: str = "cpu"           # passera à "cuda" automatiquement plus tard si GPU dispo
    seed: int = 1337


# Instance par défaut, importable directement ailleurs :
# from model.config import config
config = SabrinaConfig()
