"""
sample.py
Définit CodeSample : l'unité de données qui circule dans tout le pipeline
(sources -> filtres -> manifest -> sortie finale).

Toute source, tout filtre, manipule des CodeSample — jamais du texte brut
directement. Ça garantit que la traçabilité (licence, provenance, hash)
reste attachée au contenu à chaque étape, jusqu'au bout du pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class CodeSample:
    # --- Contenu ---
    content: str                    # le code brut lui-même
    language: str                   # ex: "python", "javascript", "bash"

    # --- Provenance (obligatoire pour la traçabilité) ---
    source_name: str                # ex: "huggingface:the-stack-smol", "github:user/repo"
    source_path: str                # ex: chemin du fichier dans le repo, ou clé du dataset
    license: str                    # ex: "mit", "apache-2.0", "bsd-3-clause"

    # --- Métadonnées optionnelles ---
    repo_url: Optional[str] = None
    file_path: Optional[str] = None
    retrieved_at: Optional[str] = None   # date ISO de récupération, remplie par le pipeline

    # --- Champs calculés (remplis automatiquement, pas à la création) ---
    content_hash: str = field(default="", init=False)

    def __post_init__(self):
        # Hash du contenu normalisé (utilisé pour le dédoublonnage exact)
        normalized = self.content.strip()
        self.content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1

    def to_manifest_entry(self) -> dict:
        """Représentation destinée à être écrite dans le manifest (une ligne = un CodeSample)."""
        return {
            "content_hash": self.content_hash,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "license": self.license,
            "language": self.language,
            "repo_url": self.repo_url,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "retrieved_at": self.retrieved_at,
        }
