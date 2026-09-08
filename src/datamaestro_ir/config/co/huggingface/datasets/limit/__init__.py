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

from datamaestro.definitions import dataset, Dataset
from datamaestro.download.single import FileDownloader
from datamaestro.download import reference
from datamaestro_ir.data import Adhoc
from datamaestro_ir.data.beir import (
    BeirDocumentStore,
    BeirParquetTopics,
    BeirParquetAssessments,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.download.docstore import docstore_builder

LIMIT_VARIANTS = {
    "limit": {
        "repo": "orionweller/LIMIT",
        "doc_count": 50000,
        "class_prefix": "Limit",
    },
    "limit_small": {
        "repo": "orionweller/LIMIT-small",
        "doc_count": 46,
        "class_prefix": "LimitSmall",
    },
    "limit-small": {
        "repo": "orionweller/LIMIT-small",
        "doc_count": 46,
        "class_prefix": "LimitSmallHyphen",
    },
}


def register_limit_subsets():
    for name, meta in LIMIT_VARIANTS.items():
        base_url = f"https://huggingface.co/datasets/{meta['repo']}/resolve/refs%2Fconvert%2Fparquet"
        prefix = meta["class_prefix"]

        # 1. Documents
        def make_docs_class(
            n=name,
            p=prefix,
            url=f"{base_url}/corpus/corpus/0000.parquet",
            count=meta["doc_count"],
        ):
            class Docs(Dataset):
                CORPUS = FileDownloader("corpus.parquet", url)
                STORE = docstore_builder(
                    source=CORPUS,
                    iter_factory=beir_parquet_docstore_iter,
                    keys=["id"],
                    doc_count=count,
                )

                def config(self) -> BeirDocumentStore:
                    return BeirDocumentStore.C(
                        id=self.__dataset__.id, path=self.STORE.path
                    )

            Docs.__name__ = f"{p}_Documents"
            Docs.__module__ = __name__
            return dataset(id=f"co.huggingface.{n}.documents")(Docs)

        DocsClass = make_docs_class()
        globals()[DocsClass.__name__] = DocsClass

        # 2. Queries
        def make_queries_class(
            n=name, p=prefix, url=f"{base_url}/queries/queries/0000.parquet"
        ):
            class Queries(Dataset):
                QUERIES = FileDownloader("queries.parquet", url)

                def config(self) -> BeirParquetTopics:
                    return BeirParquetTopics.C(
                        id=self.__dataset__.id, path=self.QUERIES.path
                    )

            Queries.__name__ = f"{p}_Queries"
            Queries.__module__ = __name__
            return dataset(id=f"co.huggingface.{n}.queries")(Queries)

        QueriesClass = make_queries_class()
        globals()[QueriesClass.__name__] = QueriesClass

        # 3. Qrels
        def make_qrels_class(
            n=name, p=prefix, url=f"{base_url}/default/test/0000.parquet"
        ):
            class Qrels(Dataset):
                QRELS = FileDownloader("qrels.parquet", url)

                def config(self) -> BeirParquetAssessments:
                    return BeirParquetAssessments.C(
                        id=self.__dataset__.id, path=self.QRELS.path
                    )

            Qrels.__name__ = f"{p}_Qrels"
            Qrels.__module__ = __name__
            return dataset(id=f"co.huggingface.{n}.qrels")(Qrels)

        QrelsClass = make_qrels_class()
        globals()[QrelsClass.__name__] = QrelsClass

        # 4. Full Adhoc Benchmark
        def make_full_class(
            n=name,
            p=prefix,
            repo=meta["repo"],
            d=DocsClass,
            q=QueriesClass,
            r=QrelsClass,
        ):
            class Full(Dataset):
                DOCS = reference(d)
                QUERIES = reference(q)
                QRELS = reference(r)

                def config(self) -> Adhoc:
                    return Adhoc.C(
                        id=self.__dataset__.id,
                        documents=self.DOCS.config(),
                        topics=self.QUERIES.config(),
                        assessments=self.QRELS.config(),
                    )

            Full.__name__ = f"{p}_Benchmark"
            Full.__module__ = __name__
            return dataset(
                id=f"co.huggingface.{n}",
                url=f"https://huggingface.co/datasets/{repo}",
            )(Full)

        FullClass = make_full_class()
        globals()[FullClass.__name__] = FullClass


register_limit_subsets()
