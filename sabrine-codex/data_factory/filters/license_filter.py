"""
filters/license_filter.py
Ne garde que les CodeSample dont la licence est permissive
(MIT, Apache-2.0, BSD, CC0...). Rejette tout le reste, y compris
les licences inconnues/vides — mieux vaut rejeter par prudence
qu'inclure une licence non vérifiée.
"""

from data_factory.filters.base import Filter, FilterResult
from data_factory.sample import CodeSample


# Licences considérées comme permissives (comparaison insensible à la casse,
# tirets/points normalisés). Liste volontairement explicite plutôt qu'une
# heuristique floue — mieux vaut l'étendre à la main qu'accepter par erreur
# une licence copyleft.
PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "apache2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "cc0",
    "unlicense",
    "isc",
}


class LicenseFilter(Filter):
    def __init__(self, allowed: set[str] = None):
        super().__init__(name="license_filter")
        self.allowed = allowed or PERMISSIVE_LICENSES

    def _normalize(self, license_str: str) -> str:
        return license_str.strip().lower().replace(" ", "-")

    def check(self, sample: CodeSample) -> FilterResult:
        if not sample.license:
            return FilterResult(passed=False, reason="licence absente ou vide")

        # Un CodeSample peut porter PLUSIEURS licences (dépôt dual/multi-licencié),
        # jointes par "|" en amont (voir HuggingFaceSource). On exige que TOUTES
        # soient permissives — une seule licence copyleft dans la liste suffit à
        # rejeter le fichier, par prudence juridique.
        raw_licenses = sample.license.split("|")

        for raw in raw_licenses:
            normalized = self._normalize(raw)
            if normalized not in self.allowed:
                return FilterResult(
                    passed=False,
                    reason=f"licence non permissive: {raw!r} (parmi {raw_licenses})",
                )

        return FilterResult(passed=True)