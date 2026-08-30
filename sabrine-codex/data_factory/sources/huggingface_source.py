"""
sources/huggingface_source.py
Source qui streame un dataset Hugging Face (ex: bigcode/the-stack-smol),
sans jamais le télécharger entièrement en local.

Utilise `datasets.load_dataset(..., streaming=True)` : chaque exemple est
récupéré à la demande, ce qui reste cohérent avec le principe de streaming
de tout le pipeline (defini dans sources/base.py).

Nécessite : pip install datasets
"""

from typing import Iterator, Optional

from data_factory.sources.base import DataSource
from data_factory.sample import CodeSample


class HuggingFaceSource(DataSource):
    def __init__(
        self,
        name: str,
        dataset_id: str,
        content_field: str = "content",
        license_field: Optional[str] = "license",
        language: str = "python",
        split: str = "train",
        config: Optional[str] = None,
        data_dir: Optional[str] = None,
        max_total_bytes: Optional[int] = None,
        default_license: str = "unknown",
        token: Optional[str] = None,
    ):
        """
        dataset_id       : identifiant du dataset sur Hugging Face (ex: "bigcode/the-stack-smol")
        content_field    : nom du champ contenant le code dans le dataset
        license_field    : nom du champ contenant la licence (None si le dataset n'en fournit pas)
        language         : langage déclaré pour tous les CodeSample de cette source
        config           : nom de configuration du dataset, si le dataset en utilise
                            (paramètre `name` de load_dataset — différent de data_dir, voir plus bas)
        data_dir         : sous-dossier de données à charger, pour les datasets organisés ainsi
                            (ex: "data/python" pour the-stack-smol — PAS le même mécanisme que `config`)
        max_total_bytes  : arrête le stream une fois ce volume de contenu atteint
                            (approximatif — vérifié après coup, pas avant, pour rester simple)
        default_license  : licence à utiliser si license_field est absent/vide sur un exemple
                            ("unknown" par défaut : sera rejeté par LicenseFilter, par prudence)
        token            : token d'accès Hugging Face, nécessaire pour les datasets "gated"
                            (ex: bigcode/the-stack-smol). Si None, utilise le login déjà en
                            cache localement (huggingface-cli login) s'il existe.
        """
        super().__init__(name)
        self.dataset_id = dataset_id
        self.content_field = content_field
        self.license_field = license_field
        self.language = language
        self.split = split
        self.config = config
        self.data_dir = data_dir
        self.max_total_bytes = max_total_bytes
        self.default_license = default_license
        self.token = token

    def fetch(self) -> Iterator[CodeSample]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "La librairie 'datasets' est requise pour HuggingFaceSource. "
                "Installe-la avec : pip install datasets"
            )

        ds = load_dataset(
            self.dataset_id,
            self.config,
            data_dir=self.data_dir,
            split=self.split,
            streaming=True,
            token=self.token,
        )

        total_bytes = 0

        for i, row in enumerate(ds):
            content = row.get(self.content_field)
            if not content:
                continue

            license_value = self.default_license
            if self.license_field and row.get(self.license_field):
                raw_license = row[self.license_field]
                # Certains datasets exposent une LISTE de licences par fichier
                # (ex: le champ "licenses" de the-stack-smol) plutôt qu'une
                # chaîne unique. On prend la première si c'est une liste ;
                # rejette-la comme licence vide serait trop strict, mais
                # LicenseFilter refusera quand même si elle n'est pas permissive.
                if isinstance(raw_license, (list, tuple)):
                    license_value = raw_license[0] if raw_license else self.default_license
                else:
                    license_value = raw_license

            sample = CodeSample(
                content=content,
                language=self.language,
                source_name=self.name,
                source_path=row.get("path") or row.get("hexsha") or f"row_{i}",
                license=license_value,
                repo_url=row.get("repository_name") or row.get("repo_name"),
            )

            total_bytes += sample.size_bytes
            yield sample

            if self.max_total_bytes is not None and total_bytes >= self.max_total_bytes:
                print(
                    f"[{self.name}] Limite de volume atteinte "
                    f"({total_bytes / 1e6:.1f} Mo) — arrêt du stream."
                )
                break