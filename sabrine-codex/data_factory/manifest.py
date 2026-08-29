"""
manifest.py
Trace chaque CodeSample retenu par le pipeline : provenance, licence, hash.

Écriture au fil de l'eau (append, une ligne JSON par échantillon) —
jamais d'accumulation en mémoire avant d'écrire, pour rester cohérent
avec le principe de streaming de tout le pipeline. Format JSONL : un
objet JSON par ligne, facile à lire en streaming aussi (pas besoin de
charger tout le fichier pour le parser).
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from data_factory.sample import CodeSample


class Manifest:
    def __init__(self, path: str = "data/manifest.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._seen_hashes: Optional[set] = None  # chargé paresseusement, seulement si besoin

    def record(self, sample: CodeSample) -> None:
        """Ajoute une entrée au manifest pour ce sample. Appelé un par un, jamais en batch."""
        if sample.retrieved_at is None:
            sample.retrieved_at = datetime.now(timezone.utc).isoformat()

        entry = sample.to_manifest_entry()

        # Mode "a" (append) : on ouvre/ferme à chaque appel plutôt que de garder
        # le fichier ouvert — un peu plus lent, mais plus sûr si le script
        # s'interrompt en cours de route (rien n'est perdu dans un buffer).
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Garde le cache en mémoire synchronisé, pour que already_seen()/count()
        # reflètent immédiatement ce qu'on vient d'écrire, sans relire le fichier.
        if self._seen_hashes is not None:
            self._seen_hashes.add(sample.content_hash)

    def load_seen_hashes(self) -> set:
        """
        Charge les hash déjà présents dans le manifest, pour permettre de
        reprendre une collecte interrompue sans dupliquer le travail déjà fait.
        Ne charge que les hash (pas le contenu), donc reste léger même si
        le manifest contient déjà des centaines de milliers d'entrées.
        """
        if self._seen_hashes is not None:
            return self._seen_hashes

        hashes = set()
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    hashes.add(entry["content_hash"])

        self._seen_hashes = hashes
        return hashes

    def already_seen(self, sample: CodeSample) -> bool:
        return sample.content_hash in self.load_seen_hashes()

    def count(self) -> int:
        return len(self.load_seen_hashes())
