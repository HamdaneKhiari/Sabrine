from data_factory.sources.local_source import LocalSource
from data_factory.filters.license_filter import LicenseFilter
from data_factory.filters.quality_filter import QualityFilter
from data_factory.filters.dedup import DedupFilter
from data_factory.manifest import Manifest
from data_factory.pipeline import Pipeline

manifest = Manifest(path="data/manifest.jsonl")

pipeline = Pipeline(
    sources=[LocalSource(name="local_test_corpus", directory="data/raw", license="mit")],
    filters=[LicenseFilter(), QualityFilter(), DedupFilter(manifest=manifest)],
    manifest=manifest,
)

stats = pipeline.run()
print(stats.summary())