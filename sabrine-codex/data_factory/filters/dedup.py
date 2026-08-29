"""
filters/dedup.py
Rejette les samples déjà vus — soit dans un run précédent (via le manifest),
soit plus tôt dans le run actuel (deux sources différentes qui ramènent
le même fichier, par exemple).

Dédoublonnage EXACT (hash du contenu) pour l'instant — suffisant à l'échelle
où tu es. Le jour où le volume grossit et où des quasi-doublons (même code,
un commentaire différent) posent problème, ce filtre peut être remplacé par
une version MinHash sans toucher au reste du pipeline (même contrat Filter).
"""

from typing import Optional
from data_factory.filters.base import Filter, FilterResult
from data_factory.sample import CodeSample
from data_factory.manifest import Manifest


class DedupFilter(Filter):
    def __init__(self, manifest: Optional[Manifest] = None):
        super().__init__(name="dedup_filter")
        # Seed avec les hash déjà présents dans le manifest (runs précédents),
        # puis on ajoute au fil de l'eau les hash vus dans CE run — pour
        # attraper aussi les doublons entre deux sources différentes dans
        # le même passage, avant même qu'ils soient écrits au manifest.
        self._seen: set = set(manifest.load_seen_hashes()) if manifest else set()

    def check(self, sample: CodeSample) -> FilterResult:
        if sample.content_hash in self._seen:
            return FilterResult(passed=False, reason=f"doublon exact (hash={sample.content_hash[:12]}...)")

        self._seen.add(sample.content_hash)
        return FilterResult(passed=True)
