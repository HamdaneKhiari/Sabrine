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
    vocab_size: int = 16000       # doit correspondre au tokenizer entraîné

    # --- Architecture (rééquilibrée : moins de vocab, plus de profondeur Transformer,
    # avec weight tying — objectif ~15M avec ~34% seulement en embedding, contre 91% avant) ---
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    block_size: int = 256         # taille max du contexte (en tokens) — inchangé pour comparaison isolée
    dropout: float = 0.1

    # --- Entraînement ---
    batch_size: int = 64          # relevé de 8 : un batch plus large réduit le bruit du gradient
                                   # et donne une courbe de loss plus stable (T4 encaisse largement
                                   # cette taille pour un modèle de 15M)
    learning_rate: float = 3e-4       # taux d'apprentissage maximal, atteint après le warmup
    warmup_iters: int = 1000          # nombre d'itérations de warmup linéaire (LR croît de 0 à learning_rate)
    min_lr: float = 3e-5              # taux d'apprentissage plancher en fin de decay (learning_rate / 10)
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