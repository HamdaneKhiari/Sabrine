"""
filters/base.py
Contrat abstrait pour les filtres du pipeline.

Un filtre prend un CodeSample et décide : le garder ou le rejeter.
Les filtres sont chaînables — le pipeline les enchaîne dans l'ordre
donné, et un CodeSample rejeté par l'un n'est jamais vu par le suivant.

Chaque filtre doit pouvoir expliquer POURQUOI il a rejeté un sample
(via reason), utile pour le débogage et pour affiner les seuils plus tard.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from data_factory.sample import CodeSample


@dataclass
class FilterResult:
    passed: bool
    reason: Optional[str] = None  # rempli seulement si passed=False, ex: "licence non permissive: gpl-3.0"


class Filter(ABC):
    """
    Contrat commun à tous les filtres (licence, qualité, dédoublonnage...).

    IMPORTANT : check() ne doit JAMAIS modifier le CodeSample — un filtre
    décide seulement s'il passe ou non, il ne transforme pas le contenu.
    (La transformation de contenu, si un jour tu en as besoin — nettoyage,
    normalisation — serait une étape à part, pas un filtre.)
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check(self, sample: CodeSample) -> FilterResult:
        """Décide si ce sample passe ce filtre. Ne doit avoir aucun effet de bord sur sample."""
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"


def run_filter_chain(sample: CodeSample, filters: list[Filter]) -> FilterResult:
    """
    Applique une liste de filtres dans l'ordre, s'arrête au premier échec
    (court-circuit — inutile de calculer les filtres suivants si un
    sample est déjà rejeté).
    """
    for f in filters:
        result = f.check(sample)
        if not result.passed:
            return FilterResult(passed=False, reason=f"[{f.name}] {result.reason}")
    return FilterResult(passed=True)
