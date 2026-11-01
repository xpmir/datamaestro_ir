"""LIMIT benchmark datasets (https://huggingface.co/datasets/orionweller/LIMIT).

Registers:
- ``co.huggingface.limit`` (full 50,000 document benchmark)
- ``co.huggingface.limit_small`` / ``co.huggingface.limit-small`` (46 document benchmark)

Each exposes:
- ``.documents``: BeirDocumentStore built from corpus.parquet
- ``.queries``: BeirParquetTopics from queries.parquet
- ``.qrels``: BeirParquetAssessments from default/test/0000.parquet
- Full Adhoc benchmark combining docs, queries, and qrels
"""

from pathlib import Path
from typing import Iterator
from datamaestro.download.single import FileDownloader
from datamaestro_ir.data.base import AdhocAssessedTopic, SimpleAdhocAssessment
from datamaestro_ir.data.beir import BeirParquetAssessments, BeirParquetTopics
from datamaestro_ir.definitions import DatasetFactory


def limit_parquet_docstore_iter(path: Path) -> Iterator[tuple[dict[str, str], bytes]]:
    """Iterates LIMIT corpus Parquet rows, replacing spaces in doc IDs with underscores for TREC compliance."""
    import pandas as pd

    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        title = row.get("title", "")
        text = row["text"]
        content = (title + "\0" + text).encode("utf-8")
        doc_id = str(row["_id"]).replace(" ", "_")
        yield {"id": doc_id}, content


class LimitParquetAssessments(BeirParquetAssessments):
    """LIMIT qrels from Parquet, replacing spaces in document IDs with underscores for TREC compliance."""

    def iter(self) -> Iterator[AdhocAssessedTopic]:
        import pandas as pd
        from collections import defaultdict

        df = pd.read_parquet(self.path)
        assessments = defaultdict(list)
        for _, row in df.iterrows():
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"]).replace(" ", "_")
            score = int(row.get("score", 1.0))
            assessments[qid].append(SimpleAdhocAssessment(doc_id=doc_id, rel=score))

        for qid, docs in assessments.items():
            yield AdhocAssessedTopic(topic_id=qid, assessments=docs)


LIMIT_VARIANTS = {
    "limit": {"repo": "orionweller/LIMIT", "doc_count": 50000},
    "limit_small": {"repo": "orionweller/LIMIT-small", "doc_count": 46},
    "limit-small": {"repo": "orionweller/LIMIT-small", "doc_count": 46},
}


def register_limit_subsets():
    datasets = DatasetFactory(__name__, prefix="co.huggingface", class_prefix="Limit")
    for name, meta in LIMIT_VARIANTS.items():
        repo_url = f"https://huggingface.co/datasets/{meta['repo']}"
        base_url = f"{repo_url}/resolve/refs%2Fconvert%2Fparquet"

        docs = datasets.docstore(
            f"{name}.documents",
            FileDownloader("corpus.parquet", f"{base_url}/corpus/corpus/0000.parquet"),
            iter_factory=limit_parquet_docstore_iter,
            doc_count=meta["doc_count"],
        )
        queries = datasets.file(
            f"{name}.queries",
            BeirParquetTopics,
            QUERIES=FileDownloader(
                "queries.parquet", f"{base_url}/queries/queries/0000.parquet"
            ),
        )
        qrels = datasets.file(
            f"{name}.qrels",
            LimitParquetAssessments,
            QRELS=FileDownloader(
                "qrels.parquet", f"{base_url}/default/test/0000.parquet"
            ),
        )
        datasets.adhoc(name, docs, queries, qrels, url=repo_url)


register_limit_subsets()
