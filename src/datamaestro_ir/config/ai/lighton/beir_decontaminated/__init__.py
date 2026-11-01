"""LightOn BEIR Decontaminated benchmark datasets.

Registers all 14 decontaminated BEIR benchmarks from https://huggingface.co/datasets/lightonai
under `ai.lighton.beir_decontaminated.<dataset_name>`.
"""

from datamaestro.download.single import FileDownloader

from datamaestro_ir.data.beir import (
    BeirParquetAssessments,
    BeirParquetTopics,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.definitions import DatasetFactory

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
    datasets = DatasetFactory(
        __name__,
        prefix="ai.lighton.beir_decontaminated",
        class_prefix="LightOnDecontaminated",
    )
    for name, info in LIGHTON_BEIR_DECONTAMINATED.items():
        repo_url = f"https://huggingface.co/datasets/lightonai/{info['repo']}"
        base_url = f"{repo_url}/resolve/main"

        docs = datasets.docstore(
            f"{name}.documents",
            FileDownloader("corpus.parquet", f"{base_url}/corpus.parquet"),
            iter_factory=beir_parquet_docstore_iter,
        )
        queries = datasets.file(
            f"{name}.queries",
            BeirParquetTopics,
            QUERIES=FileDownloader("queries.parquet", f"{base_url}/queries.parquet"),
        )

        # Qrels and ad hoc task for each split
        qrels = {}
        for split, qrels_file in info["splits"].items():
            qrels[split] = datasets.file(
                f"{name}.qrels.{split}",
                BeirParquetAssessments,
                QRELS=FileDownloader(
                    f"qrels_{split}.parquet", f"{base_url}/{qrels_file}"
                ),
            )
            datasets.adhoc(f"{name}.{split}", docs, queries, qrels[split])

        # Default ad hoc task (test split)
        datasets.adhoc(name, docs, queries, qrels["test"], url=repo_url)


register_beir_decontaminated_subsets()
