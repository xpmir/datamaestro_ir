"""LightOn BEIR Decontaminated benchmark datasets.

Registers all 14 decontaminated BEIR benchmarks from https://huggingface.co/datasets/lightonai
under `ai.lighton.beir_decontaminated.<dataset_name>`.
"""

from datamaestro.definitions import Dataset, dataset
from datamaestro.download import reference
from datamaestro.download.single import FileDownloader

from datamaestro_ir.data import Adhoc
from datamaestro_ir.data.beir import (
    BeirDocumentStore,
    BeirParquetAssessments,
    BeirParquetTopics,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.download.docstore import docstore_builder

LIGHTON_BEIR_DECONTAMINATED = {
    "arguana": {
        "repo": "arguana-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
    "climate_fever": {
        "repo": "climate-fever-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
    "dbpedia_entity": {
        "repo": "dbpedia-entity-decontaminated",
        "splits": {"dev": "qrels_validation.parquet", "test": "qrels_test.parquet"},
    },
    "fever": {
        "repo": "fever-decontaminated",
        "splits": {
            "train": "qrels_train.parquet",
            "dev": "qrels_validation.parquet",
            "test": "qrels_test.parquet",
        },
    },
    "fiqa": {
        "repo": "fiqa-decontaminated",
        "splits": {
            "train": "qrels_train.parquet",
            "dev": "qrels_validation.parquet",
            "test": "qrels_test.parquet",
        },
    },
    "hotpotqa": {
        "repo": "hotpotqa-decontaminated",
        "splits": {
            "train": "qrels_train.parquet",
            "dev": "qrels_validation.parquet",
            "test": "qrels_test.parquet",
        },
    },
    "msmarco": {
        "repo": "msmarco-decontaminated",
        "splits": {
            "train": "qrels_train.parquet",
            "dev": "qrels_validation.parquet",
            "test": "qrels_test.parquet",
        },
    },
    "nfcorpus": {
        "repo": "nfcorpus-decontaminated",
        "splits": {
            "train": "qrels_train.parquet",
            "dev": "qrels_validation.parquet",
            "test": "qrels_test.parquet",
        },
    },
    "nq": {
        "repo": "nq-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
    "quora": {
        "repo": "quora-decontaminated",
        "splits": {"dev": "qrels_validation.parquet", "test": "qrels_test.parquet"},
    },
    "scidocs": {
        "repo": "scidocs-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
    "scifact": {
        "repo": "scifact-decontaminated",
        "splits": {"train": "qrels_train.parquet", "test": "qrels_test.parquet"},
    },
    "trec_covid": {
        "repo": "trec-covid-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
    "webis_touche2020": {
        "repo": "webis-touche2020-decontaminated",
        "splits": {"test": "qrels_test.parquet"},
    },
}


def register_beir_decontaminated_subsets():
    for name, info in LIGHTON_BEIR_DECONTAMINATED.items():
        base_url = (
            f"https://huggingface.co/datasets/lightonai/{info['repo']}/resolve/main"
        )

        # 1. Documents
        def make_docs_class(n=name, url=f"{base_url}/corpus.parquet"):
            class Docs(Dataset):
                CORPUS = FileDownloader("corpus.parquet", url)
                STORE = docstore_builder(
                    source=CORPUS,
                    iter_factory=beir_parquet_docstore_iter,
                    keys=["id"],
                )

                def config(self) -> BeirDocumentStore:
                    return BeirDocumentStore.C(
                        id=self.__dataset__.id, path=self.STORE.path
                    )

            Docs.__name__ = f"LightOnDecontaminated_{n}_Docs"
            Docs.__module__ = __name__
            return dataset(id=f"ai.lighton.beir_decontaminated.{n}.documents")(Docs)

        DocsClass = make_docs_class()
        globals()[DocsClass.__name__] = DocsClass

        # 2. Queries
        def make_queries_class(n=name, url=f"{base_url}/queries.parquet"):
            class Queries(Dataset):
                QUERIES = FileDownloader("queries.parquet", url)

                def config(self) -> BeirParquetTopics:
                    return BeirParquetTopics.C(
                        id=self.__dataset__.id, path=self.QUERIES.path
                    )

            Queries.__name__ = f"LightOnDecontaminated_{n}_Queries"
            Queries.__module__ = __name__
            return dataset(id=f"ai.lighton.beir_decontaminated.{n}.queries")(Queries)

        QueriesClass = make_queries_class()
        globals()[QueriesClass.__name__] = QueriesClass

        # 3. Qrels per split
        qrel_classes = {}
        for split_name, qrel_file in info["splits"].items():

            def make_qrels_class(
                n=name, s=split_name, f=qrel_file, url=f"{base_url}/{qrel_file}"
            ):
                class Qrels(Dataset):
                    QRELS = FileDownloader(f"qrels_{s}.parquet", url)

                    def config(self) -> BeirParquetAssessments:
                        return BeirParquetAssessments.C(
                            id=self.__dataset__.id, path=self.QRELS.path
                        )

                Qrels.__name__ = f"LightOnDecontaminated_{n}_Qrels_{s}"
                Qrels.__module__ = __name__
                return dataset(id=f"ai.lighton.beir_decontaminated.{n}.qrels.{s}")(
                    Qrels
                )

            QrelsClass = make_qrels_class()
            qrel_classes[split_name] = QrelsClass
            globals()[QrelsClass.__name__] = QrelsClass

            # 4. Adhoc benchmark per split
            def make_adhoc_split(
                n=name, s=split_name, d=DocsClass, q=QueriesClass, r=QrelsClass
            ):
                class SplitAdhoc(Dataset):
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

                SplitAdhoc.__name__ = f"LightOnDecontaminated_{n}_{s}"
                SplitAdhoc.__module__ = __name__
                return dataset(id=f"ai.lighton.beir_decontaminated.{n}.{s}")(
                    SplitAdhoc
                )

            SplitClass = make_adhoc_split()
            globals()[SplitClass.__name__] = SplitClass

        # 5. Default Adhoc dataset (points to test split)
        test_qrels = qrel_classes["test"]

        def make_default_adhoc(
            n=name,
            repo=info["repo"],
            d=DocsClass,
            q=QueriesClass,
            r=test_qrels,
        ):
            class DefaultAdhoc(Dataset):
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

            DefaultAdhoc.__name__ = f"LightOnDecontaminated_{n}"
            DefaultAdhoc.__module__ = __name__
            return dataset(
                id=f"ai.lighton.beir_decontaminated.{n}",
                url=f"https://huggingface.co/datasets/lightonai/{repo}",
            )(DefaultAdhoc)

        DefaultClass = make_default_adhoc()
        globals()[DefaultClass.__name__] = DefaultClass


register_beir_decontaminated_subsets()
