"""
sources/local_source.py
Source qui lit des fichiers de code déjà présents sur le disque
(ex: data/raw/, ou n'importe quel dossier).

Utile pour :
- tester le pipeline sur du contenu réel avant de brancher une vraie
  source externe (Hugging Face, GitHub...)
- intégrer un corpus que tu as déjà récupéré autrement (téléchargement
  manuel, export d'un dataset...)

La licence est déclarée une fois pour tout le dossier (via `license`),
puisqu'une source locale n'a pas de métadonnée de licence par fichier —
c'est à toi de t'assurer que tout ce que tu mets dans ce dossier
correspond bien à la licence déclarée.
"""

import os
import glob
from typing import Iterator, Optional

from data_factory.sources.base import DataSource
from data_factory.sample import CodeSample


EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".sh": "bash",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
}


class LocalSource(DataSource):
    def __init__(self, name: str, directory: str, license: str, extensions: Optional[list] = None):
        super().__init__(name)
        self.directory = directory
        self.license = license
        self.extensions = extensions or list(EXTENSION_TO_LANGUAGE.keys())

    def fetch(self) -> Iterator[CodeSample]:
        for ext in self.extensions:
            pattern = os.path.join(self.directory, "**", f"*{ext}")
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    continue  # fichier illisible, on passe simplement au suivant

                language = EXTENSION_TO_LANGUAGE.get(ext, "unknown")
                relative_path = os.path.relpath(filepath, self.directory)

                yield CodeSample(
                    content=content,
                    language=language,
                    source_name=self.name,
                    source_path=relative_path,
                    license=self.license,
                    file_path=filepath,
                )
