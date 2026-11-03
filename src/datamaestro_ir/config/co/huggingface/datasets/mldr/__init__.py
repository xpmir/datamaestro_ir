"""MLDR (Multi Long Doc Retrieval), as used by MTEB's ``MultiLongDocRetrieval``.

For each language ``<lang>`` and split ``<split>`` (``dev`` and ``test``;
``train`` is not registered), registers:

- ``co.huggingface.datasets.mldr.<lang>.<split>.documents``: the corpus of the
  split, as a document store built from its parquet shards
- ``co.huggingface.datasets.mldr.<lang>.<split>.queries``: the queries
- ``co.huggingface.datasets.mldr.<lang>.<split>.qrels``: relevance judgments
- ``co.huggingface.datasets.mldr.<lang>.<split>``: the ad hoc task
  (:class:`~datamaestro_ir.data.Adhoc`)

Files are taken from `mteb/MultiLongDocRetrieval <https://huggingface.co/datasets/mteb/MultiLongDocRetrieval>`_
at the revision pinned by MTEB. Original dataset: https://huggingface.co/datasets/Shitao/MLDR
"""

from datamaestro.download.single import FileDownloader

from datamaestro_ir.data.beir import (
    BeirParquetAssessments,
    BeirParquetTopics,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.definitions import DatasetFactory

URL = "https://huggingface.co/datasets/mteb/MultiLongDocRetrieval"
REVISION = "837028901907a7d419b4ab906f28e011ce1cc824"
BASE_URL = f"{URL}/resolve/{REVISION}"

LANGUAGES = ["ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "th", "zh"]

SPLITS = ["dev", "test"]

# Number of corpus parquet shards per language (1 when absent)
CORPUS_SHARDS = {"en": 6, "ru": 2, "th": 2, "zh": 7}


def _corpus_shards(lang: str, split: str) -> dict[str, FileDownloader]:
    count = CORPUS_SHARDS.get(lang, 1)
    return {
        f"CORPUS_{i}": FileDownloader(
            f"corpus-{i}.parquet",
            f"{BASE_URL}/{lang}-corpus/{split}-{i:05d}-of-{count:05d}.parquet",
        )
        for i in range(count)
    }


def _register_all():
    datasets = DatasetFactory(
        __name__, prefix="co.huggingface.datasets.mldr", class_prefix="Mldr"
    )
    for lang in LANGUAGES:
        for split in SPLITS:
            documents = datasets.docstore(
                f"{lang}.{split}.documents",
                _corpus_shards(lang, split),
                iter_factory=beir_parquet_docstore_iter,
                url=URL,
                languages=[lang],
            )
            queries = datasets.file(
                f"{lang}.{split}.queries",
                BeirParquetTopics,
                url=URL,
                QUERIES=FileDownloader(
                    "queries.parquet",
                    f"{BASE_URL}/{lang}-queries/{split}-00000-of-00001.parquet",
                ),
            )
            qrels = datasets.file(
                f"{lang}.{split}.qrels",
                BeirParquetAssessments,
                url=URL,
                QRELS=FileDownloader(
                    "qrels.parquet",
                    f"{BASE_URL}/{lang}-qrels/{split}-00000-of-00001.parquet",
                ),
            )
            datasets.adhoc(f"{lang}.{split}", documents, queries, qrels, url=URL)


_register_all()
