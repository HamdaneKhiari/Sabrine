"""
run_pipeline.py
Fait tourner le pipeline complet sur the-stack-smol (Python), et sauvegarde
le contenu accepté sur disque dans data/processed/ — pas juste le manifest
(qui ne trace que la métadonnée, pas le contenu lui-même).
"""

import os
import hashlib

from data_factory.sources.huggingface_source import HuggingFaceSource
from data_factory.filters.license_filter import LicenseFilter
from data_factory.filters.quality_filter import QualityFilter
from data_factory.filters.dedup import DedupFilter
from data_factory.manifest import Manifest
from data_factory.pipeline import Pipeline

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_sample_to_disk(sample):
    """Écrit le contenu d'un CodeSample accepté sur disque, nommé par son hash
    (garantit un nom de fichier unique et stable, sans collision possible)."""
    filename = f"{sample.content_hash[:16]}.py"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(sample.content)


def main():
    manifest = Manifest(path="data/manifest.jsonl")

    source = HuggingFaceSource(
        name="the_stack_python",
        dataset_id="bigcode/the-stack",
        data_dir="data/python",
        content_field="content",
        license_field="licenses",
        language="python",
        max_total_bytes=80_000_000,
    )

    pipeline = Pipeline(
        sources=[source],
        filters=[LicenseFilter(), QualityFilter(), DedupFilter(manifest=manifest)],
        manifest=manifest,
    )

    stats = pipeline.run(on_accept=save_sample_to_disk)

    print(stats.summary())
    print(f"\nContenu sauvegardé dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()