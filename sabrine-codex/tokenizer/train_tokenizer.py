"""
train_tokenizer.py
Entraîne un tokenizer BPE dédié au code (Python/Bash/JS...) pour Sabrina: Codex.

Usage :
    python tokenizer/train_tokenizer.py

Prérequis :
    - Des fichiers de code dans data/raw/ (n'importe quelle extension : .py, .js, .sh...)
    - pip install tokenizers
"""

import os
import glob
from tokenizers import ByteLevelBPETokenizer

# --- Configuration ---
DATA_DIR = "data/processed"
OUTPUT_DIR = "tokenizer"
VOCAB_SIZE = 32000
MIN_FREQUENCY = 2

# Tokens spéciaux utiles pour un modèle de code
SPECIAL_TOKENS = [
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<mask>",
    "<|code|>",      # marqueur de début de bloc de code
    "<|endofcode|>", # marqueur de fin de bloc de code
]


def collect_files(data_dir: str) -> list[str]:
    """Récupère tous les fichiers texte/code dans data_dir, récursivement."""
    extensions = ["*.py", "*.js", "*.ts", "*.sh", "*.java", "*.cpp", "*.c", "*.go", "*.rs", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return files


def main():
    files = collect_files(DATA_DIR)

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {DATA_DIR}/. "
            "Ajoute des fichiers de code (.py, .js, .sh, etc.) avant de relancer."
        )

    print(f"[Sabrina: Codex] {len(files)} fichiers trouvés pour l'entraînement du tokenizer.")

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save_model(OUTPUT_DIR, "sabrina_tokenizer")

    # Sauvegarde aussi au format unifié (plus simple à recharger ensuite)
    tokenizer.save(os.path.join(OUTPUT_DIR, "sabrina_tokenizer.json"))

    print(f"[Sabrina: Codex] Tokenizer entraîné et sauvegardé dans {OUTPUT_DIR}/")
    print(f"[Sabrina: Codex] Taille du vocabulaire : {tokenizer.get_vocab_size()}")

    # --- Petit test rapide ---
    sample_code = "def hello_world(name: str) -> None:\n    print(f'Hello, {name}!')"
    encoded = tokenizer.encode(sample_code)
    print("\n--- Test d'encodage ---")
    print("Texte :", sample_code)
    print("Tokens :", encoded.tokens)
    print("Nombre de tokens :", len(encoded.tokens))


if __name__ == "__main__":
    main()