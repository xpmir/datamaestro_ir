from datamaestro.download.single import FileDownloader

from datamaestro_ir.data.beir import (
    BeirParquetAssessments,
    BeirParquetTopics,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.definitions import DatasetFactory

NANO_BEIR_DATA = {
    "arguana": "NanoArguAna",
    "climate-fever": "NanoClimateFEVER",
    "dbpedia-entity": "NanoDBPedia",
    "fever": "NanoFEVER",
    "fiqa": "NanoFiQA2018",
    "hotpotqa": "NanoHotpotQA",
    "msmarco": "NanoMSMARCO",
    "nfcorpus": "NanoNFCorpus",
    "nq": "NanoNQ",
    "quora": "NanoQuoraRetrieval",
    "scidocs": "NanoSCIDOCS",
    "scifact": "NanoSciFact",
    "webis-touche2020": "NanoTouche2020",
}

# Translations of NanoBEIR (lightonai/nanobeir-multilingual), same dataset names
NANO_BEIR_MULTILINGUAL_URL = "https://huggingface.co/datasets/lightonai/nanobeir-multilingual"
NANO_BEIR_LANGUAGES = ["ar", "de", "en", "es", "fr", "it", "no", "pt", "sv"]


def register_nano_beir_translations(datasets, name, hf_name, qrels):
    """Registers the translations of one NanoBEIR dataset

    The qrels are the English ones, shared by all languages since the
    translations keep the same relevance judgments.
    """
    base_url = f"{NANO_BEIR_MULTILINGUAL_URL}/resolve/main"
    for lang in NANO_BEIR_LANGUAGES:
        docs = datasets.docstore(
            f"{name}.{lang}.documents",
            FileDownloader("corpus.parquet", f"{base_url}/{hf_name}_corpus_{lang}.parquet"),
            iter_factory=beir_parquet_docstore_iter,
            languages=[lang],
            url=NANO_BEIR_MULTILINGUAL_URL,
        )
        queries = datasets.file(
            f"{name}.{lang}.queries",
            BeirParquetTopics,
            QUERIES=FileDownloader("queries.parquet", f"{base_url}/{hf_name}_queries_{lang}.parquet"),
            url=NANO_BEIR_MULTILINGUAL_URL,
        )
        datasets.adhoc(f"{name}.{lang}", docs, queries, qrels, url=NANO_BEIR_MULTILINGUAL_URL)


def register_nano_beir_subsets():
    datasets = DatasetFactory(
        __name__, prefix="co.huggingface.nano-beir", class_prefix="NanoBeir"
    )
    for name, hf_name in NANO_BEIR_DATA.items():
        base_url = (
            f"https://huggingface.co/datasets/zeta-alpha-ai/{hf_name}/resolve/main"
        )
        parquet = "train-00000-of-00001.parquet"

        docs = datasets.docstore(
            f"{name}.documents",
            FileDownloader("corpus.parquet", f"{base_url}/corpus/{parquet}"),
            iter_factory=beir_parquet_docstore_iter,
        )
        queries = datasets.file(
            f"{name}.queries",
            BeirParquetTopics,
            QUERIES=FileDownloader("queries.parquet", f"{base_url}/queries/{parquet}"),
        )
        qrels = datasets.file(
            f"{name}.qrels",
            BeirParquetAssessments,
            QRELS=FileDownloader("qrels.parquet", f"{base_url}/qrels/{parquet}"),
        )
        datasets.adhoc(name, docs, queries, qrels)
        register_nano_beir_translations(datasets, name, hf_name, qrels)


register_nano_beir_subsets()
