"""
filters/quality_filter.py
Rejette les samples de mauvaise qualité pour l'entraînement :
- trop petits (snippets inutiles) ou trop gros (fichiers générés/vendorisés)
- lignes trop longues en moyenne (souvent du code minifié)
- une seule ligne géante (signature classique de minification)
"""

from data_factory.filters.base import Filter, FilterResult
from data_factory.sample import CodeSample


class QualityFilter(Filter):
    def __init__(
        self,
        min_size_bytes: int = 50,
        max_size_bytes: int = 1_000_000,      # 1 Mo : au-delà, souvent généré/vendorisé
        max_avg_line_length: int = 200,        # au-delà, souvent minifié
        max_single_line_ratio: float = 0.5,    # si une ligne = plus de 50% du fichier, suspect
    ):
        super().__init__(name="quality_filter")
        self.min_size_bytes = min_size_bytes
        self.max_size_bytes = max_size_bytes
        self.max_avg_line_length = max_avg_line_length
        self.max_single_line_ratio = max_single_line_ratio

    def check(self, sample: CodeSample) -> FilterResult:
        size = sample.size_bytes

        if size < self.min_size_bytes:
            return FilterResult(passed=False, reason=f"trop petit ({size} octets)")

        if size > self.max_size_bytes:
            return FilterResult(passed=False, reason=f"trop gros ({size} octets, probablement généré/vendorisé)")

        lines = sample.content.split("\n")
        non_empty_lines = [l for l in lines if l.strip()]

        if not non_empty_lines:
            return FilterResult(passed=False, reason="fichier vide (que des lignes blanches)")

        avg_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
        if avg_line_length > self.max_avg_line_length:
            return FilterResult(
                passed=False,
                reason=f"longueur de ligne moyenne trop élevée ({avg_line_length:.0f} caractères, probablement minifié)",
            )

        longest_line = max(len(l) for l in lines)
        if longest_line / max(size, 1) > self.max_single_line_ratio:
            return FilterResult(
                passed=False,
                reason="une seule ligne représente une part disproportionnée du fichier (probablement minifié)",
            )

        return FilterResult(passed=True)
