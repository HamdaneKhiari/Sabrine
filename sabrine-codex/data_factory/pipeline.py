"""
pipeline.py
Orchestrateur : enchaîne une source -> une chaîne de filtres -> le manifest.

Reste fidèle au principe de streaming de tout le pipeline : traite un
CodeSample à la fois, jamais de liste complète en mémoire. Peut tourner
sur une source de quelques fichiers comme sur une source de plusieurs Go
sans changer de comportement.
"""

from dataclasses import dataclass, field
from typing import List

from data_factory.sources.base import DataSource
from data_factory.filters.base import Filter, run_filter_chain
from data_factory.manifest import Manifest


@dataclass
class PipelineStats:
    seen: int = 0
    accepted: int = 0
    rejected: int = 0
    rejections_by_reason: dict = field(default_factory=dict)

    def record_rejection(self, reason: str):
        # On garde seulement le nom du filtre (avant les ":"), pour agréger
        # proprement même si le détail du message varie d'un sample à l'autre
        key = reason.split("]")[0].strip("[") if "]" in reason else reason
        self.rejections_by_reason[key] = self.rejections_by_reason.get(key, 0) + 1

    def summary(self) -> str:
        lines = [
            f"Total vus       : {self.seen}",
            f"Acceptés        : {self.accepted}",
            f"Rejetés         : {self.rejected}",
        ]
        if self.rejections_by_reason:
            lines.append("Détail des rejets :")
            for reason, count in sorted(self.rejections_by_reason.items(), key=lambda x: -x[1]):
                lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


class Pipeline:
    def __init__(self, sources: List[DataSource], filters: List[Filter], manifest: Manifest):
        self.sources = sources
        self.filters = filters
        self.manifest = manifest

    def run(self, on_accept=None) -> PipelineStats:
        """
        Fait tourner le pipeline sur toutes les sources, dans l'ordre.

        on_accept : callback optionnel appelé pour chaque sample accepté,
        avec le CodeSample en argument — utile pour l'écrire quelque part
        (fichier, dataset) sans que le pipeline ait à savoir où.
        """
        stats = PipelineStats()

        for source in self.sources:
            print(f"[Pipeline] Traitement de la source : {source}")

            for sample in source.fetch():
                stats.seen += 1

                result = run_filter_chain(sample, self.filters)

                if result.passed:
                    stats.accepted += 1
                    self.manifest.record(sample)
                    if on_accept is not None:
                        on_accept(sample)
                else:
                    stats.rejected += 1
                    stats.record_rejection(result.reason)

        return stats
