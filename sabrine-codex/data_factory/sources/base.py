"""
sources/base.py
Interface abstraite que toute source de données DOIT implémenter.

Le contrat est volontairement minimal : une seule méthode, fetch(),
qui doit être un générateur (yield) pour ne jamais charger tout un
dataset en mémoire d'un coup. Ajouter une nouvelle source = créer une
nouvelle classe qui hérite de DataSource et implémente fetch().
"""

from abc import ABC, abstractmethod
from typing import Iterator
from data_factory.sample import CodeSample


class DataSource(ABC):
    """
    Contrat commun à toutes les sources (Hugging Face, GitHub, fichiers locaux...).

    IMPORTANT : fetch() doit toujours être un générateur (utiliser `yield`,
    jamais `return une_liste`). C'est ce qui garantit le streaming de bout
    en bout, même sur des sources qui font plusieurs Go.
    """

    def __init__(self, name: str):
        self.name = name  # identifiant de la source, utilisé dans CodeSample.source_name

    @abstractmethod
    def fetch(self) -> Iterator[CodeSample]:
        """
        Doit yield des CodeSample un par un.

        Exemple d'implémentation correcte :
            def fetch(self):
                for item in self._iter_something():
                    yield CodeSample(...)

        Exemple INCORRECT (charge tout en RAM) :
            def fetch(self):
                return [CodeSample(...) for item in self._load_everything()]
        """
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"
