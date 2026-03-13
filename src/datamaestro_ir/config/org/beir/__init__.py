"""BEIR (Benchmarking IR) benchmark datasets.

Provides native datamaestro dataset definitions for ~15 BEIR datasets
plus 12 CQADupStack sub-datasets, using CompressedDocumentStore for
efficient document storage.

Each dataset downloads its ZIP once (transient). The docstore is built
from the corpus, and queries/qrels are copied out before cleanup.

See: https://github.com/beir-cellar/beir
"""

import json
from pathlib import Path

from datamaestro.definitions import Dataset, datatasks, dataset
from datamaestro.download import FileResource, FilesCopy, reference
from datamaestro.download.archive import ZipDownloader
from datamaestro_ir.data import Adhoc, FilteredTopics
from datamaestro_ir.data.beir import BeirAssessments, BeirDocumentStore, BeirTopics
from datamaestro_ir.download.docstore import docstore_builder

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
CQADUPSTACK_URL = f"{BEIR_URL}/cqadupstack.zip"


# --- Helpers ---


def _read_beir_corpus(source_path: Path):
    """Reads corpus.jsonl from an extracted BEIR ZIP directory."""
    corpus_file = source_path / "corpus.jsonl"
    with open(corpus_file, "rt") as fp:
        for line in fp:
            data = json.loads(line)
            doc_id = data["_id"]
            title = data.get("title", "")
            text = data.get("text", "")
            content = (title + "\0" + text).encode("utf-8")
            yield {"id": doc_id}, content


class beir_judged_qids(FileResource):
    """Extracts query IDs from a BEIR qrels TSV file."""

    def __init__(self, data_resource, split):
        super().__init__(f"judged_qids_{split}.txt")
        self._data_resource = data_resource
        self._split = split
        self._dependencies.append(data_resource)

    def _download(self, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        qrels_path = self._data_resource.path / "qrels" / f"{self._split}.tsv"
        qids = set()
        with open(qrels_path, "rt") as fp:
            next(fp)  # skip header
            for line in fp:
                parts = line.strip().split("\t")
                if parts:
                    qids.add(parts[0])
        with open(destination, "wt") as fp:
            for qid in sorted(qids):
                fp.write(f"{qid}\n")


def _single_split_files(data):
    return FilesCopy(data, {
        "queries.jsonl": "queries.jsonl",
        "test.tsv": "qrels/test.tsv",
    })


def _multi_split_files(data, splits):
    """Copy queries + all split qrels from a single transient ZIP."""
    files = {"queries.jsonl": "queries.jsonl"}
    for split in splits:
        files[f"{split}.tsv"] = f"qrels/{split}.tsv"
    return FilesCopy(data, files)


# ============================================================================
# Single-split datasets (test only)
# ============================================================================


# --- TREC-COVID ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class TrecCovid(Dataset):
    DATA = ZipDownloader(
        "data", f"{BEIR_URL}/trec-covid.zip", transient=True
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- NQ (Natural Questions) ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class Nq(Dataset):
    DATA = ZipDownloader("data", f"{BEIR_URL}/nq.zip", transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- ArguAna ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class Arguana(Dataset):
    DATA = ZipDownloader(
        "data", f"{BEIR_URL}/arguana.zip", transient=True
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- Webis-Touche-2020 ---


TOUCHE2020_URL = "https://macavaney.us/beir-webis-touche2020-v1.zip"


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class WebisTouche2020(Dataset):
    DATA = ZipDownloader("data", TOUCHE2020_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- Webis-Touche-2020 v2 ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class WebisTouche2020V2(Dataset):
    DATA = ZipDownloader(
        "data", f"{BEIR_URL}/webis-touche2020.zip", transient=True
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- Climate-FEVER ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class ClimateFever(Dataset):
    DATA = ZipDownloader(
        "data", f"{BEIR_URL}/climate-fever.zip", transient=True
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# --- SCIDOCS ---


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class Scidocs(Dataset):
    DATA = ZipDownloader(
        "data", f"{BEIR_URL}/scidocs.zip", transient=True
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


# ============================================================================
# Multi-split datasets
#
# The Collection downloads the ZIP once (transient), builds the docstore,
# copies out queries + ALL qrels, and extracts judged qids for each split.
# Split datasets just reference the Collection and use its paths.
# ============================================================================


# --- NFCorpus (train/dev/test) ---

_NFCORPUS_URL = f"{BEIR_URL}/nfcorpus.zip"
_NFCORPUS_SPLITS = ["train", "dev", "test"]


@dataset()
class NfcorpusCollection(Dataset):
    DATA = ZipDownloader("data", _NFCORPUS_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _NFCORPUS_SPLITS)
    JUDGED_QIDS_TRAIN = beir_judged_qids(DATA, "train")
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class NfcorpusTrain(Dataset):
    COLLECTION = reference(NfcorpusCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=NfcorpusCollection.data_path / "queries.jsonl")],
                qids_path=NfcorpusCollection.data_path / "judged_qids_train.txt",
            ),
            assessments=BeirAssessments.C(path=NfcorpusCollection.data_path / "train.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class NfcorpusDev(Dataset):
    COLLECTION = reference(NfcorpusCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=NfcorpusCollection.data_path / "queries.jsonl")],
                qids_path=NfcorpusCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=NfcorpusCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class NfcorpusTest(Dataset):
    COLLECTION = reference(NfcorpusCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=NfcorpusCollection.data_path / "queries.jsonl")],
                qids_path=NfcorpusCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=NfcorpusCollection.data_path / "test.tsv"),
        )


# --- HotpotQA (train/dev/test) ---

_HOTPOTQA_URL = f"{BEIR_URL}/hotpotqa.zip"
_HOTPOTQA_SPLITS = ["train", "dev", "test"]


@dataset()
class HotpotqaCollection(Dataset):
    DATA = ZipDownloader("data", _HOTPOTQA_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _HOTPOTQA_SPLITS)
    JUDGED_QIDS_TRAIN = beir_judged_qids(DATA, "train")
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class HotpotqaTrain(Dataset):
    COLLECTION = reference(HotpotqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=HotpotqaCollection.data_path / "queries.jsonl")],
                qids_path=HotpotqaCollection.data_path / "judged_qids_train.txt",
            ),
            assessments=BeirAssessments.C(path=HotpotqaCollection.data_path / "train.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class HotpotqaDev(Dataset):
    COLLECTION = reference(HotpotqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=HotpotqaCollection.data_path / "queries.jsonl")],
                qids_path=HotpotqaCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=HotpotqaCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class HotpotqaTest(Dataset):
    COLLECTION = reference(HotpotqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=HotpotqaCollection.data_path / "queries.jsonl")],
                qids_path=HotpotqaCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=HotpotqaCollection.data_path / "test.tsv"),
        )


# --- FiQA (train/dev/test) ---

_FIQA_URL = f"{BEIR_URL}/fiqa.zip"
_FIQA_SPLITS = ["train", "dev", "test"]


@dataset()
class FiqaCollection(Dataset):
    DATA = ZipDownloader("data", _FIQA_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _FIQA_SPLITS)
    JUDGED_QIDS_TRAIN = beir_judged_qids(DATA, "train")
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FiqaTrain(Dataset):
    COLLECTION = reference(FiqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FiqaCollection.data_path / "queries.jsonl")],
                qids_path=FiqaCollection.data_path / "judged_qids_train.txt",
            ),
            assessments=BeirAssessments.C(path=FiqaCollection.data_path / "train.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FiqaDev(Dataset):
    COLLECTION = reference(FiqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FiqaCollection.data_path / "queries.jsonl")],
                qids_path=FiqaCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=FiqaCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FiqaTest(Dataset):
    COLLECTION = reference(FiqaCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FiqaCollection.data_path / "queries.jsonl")],
                qids_path=FiqaCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=FiqaCollection.data_path / "test.tsv"),
        )


# --- Quora (dev/test) ---

_QUORA_URL = f"{BEIR_URL}/quora.zip"
_QUORA_SPLITS = ["dev", "test"]


@dataset()
class QuoraCollection(Dataset):
    DATA = ZipDownloader("data", _QUORA_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _QUORA_SPLITS)
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class QuoraDev(Dataset):
    COLLECTION = reference(QuoraCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=QuoraCollection.data_path / "queries.jsonl")],
                qids_path=QuoraCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=QuoraCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class QuoraTest(Dataset):
    COLLECTION = reference(QuoraCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=QuoraCollection.data_path / "queries.jsonl")],
                qids_path=QuoraCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=QuoraCollection.data_path / "test.tsv"),
        )


# --- DBpedia-Entity (dev/test) ---

_DBPEDIA_URL = f"{BEIR_URL}/dbpedia-entity.zip"
_DBPEDIA_SPLITS = ["dev", "test"]


@dataset()
class DbpediaEntityCollection(Dataset):
    DATA = ZipDownloader("data", _DBPEDIA_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _DBPEDIA_SPLITS)
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class DbpediaEntityDev(Dataset):
    COLLECTION = reference(DbpediaEntityCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=DbpediaEntityCollection.data_path / "queries.jsonl")],
                qids_path=DbpediaEntityCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=DbpediaEntityCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class DbpediaEntityTest(Dataset):
    COLLECTION = reference(DbpediaEntityCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=DbpediaEntityCollection.data_path / "queries.jsonl")],
                qids_path=DbpediaEntityCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=DbpediaEntityCollection.data_path / "test.tsv"),
        )


# --- FEVER (train/dev/test) ---

_FEVER_URL = f"{BEIR_URL}/fever.zip"
_FEVER_SPLITS = ["train", "dev", "test"]


@dataset()
class FeverCollection(Dataset):
    DATA = ZipDownloader("data", _FEVER_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _FEVER_SPLITS)
    JUDGED_QIDS_TRAIN = beir_judged_qids(DATA, "train")
    JUDGED_QIDS_DEV = beir_judged_qids(DATA, "dev")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FeverTrain(Dataset):
    COLLECTION = reference(FeverCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FeverCollection.data_path / "queries.jsonl")],
                qids_path=FeverCollection.data_path / "judged_qids_train.txt",
            ),
            assessments=BeirAssessments.C(path=FeverCollection.data_path / "train.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FeverDev(Dataset):
    COLLECTION = reference(FeverCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FeverCollection.data_path / "queries.jsonl")],
                qids_path=FeverCollection.data_path / "judged_qids_dev.txt",
            ),
            assessments=BeirAssessments.C(path=FeverCollection.data_path / "dev.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class FeverTest(Dataset):
    COLLECTION = reference(FeverCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=FeverCollection.data_path / "queries.jsonl")],
                qids_path=FeverCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=FeverCollection.data_path / "test.tsv"),
        )


# --- SciFact (train/test) ---

_SCIFACT_URL = f"{BEIR_URL}/scifact.zip"
_SCIFACT_SPLITS = ["train", "test"]


@dataset()
class ScifactCollection(Dataset):
    DATA = ZipDownloader("data", _SCIFACT_URL, transient=True)
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _multi_split_files(DATA, _SCIFACT_SPLITS)
    JUDGED_QIDS_TRAIN = beir_judged_qids(DATA, "train")
    JUDGED_QIDS_TEST = beir_judged_qids(DATA, "test")

    def config(self) -> BeirDocumentStore:
        return BeirDocumentStore.C(path=self.store.path)


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class ScifactTrain(Dataset):
    COLLECTION = reference(ScifactCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=ScifactCollection.data_path / "queries.jsonl")],
                qids_path=ScifactCollection.data_path / "judged_qids_train.txt",
            ),
            assessments=BeirAssessments.C(path=ScifactCollection.data_path / "train.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class ScifactTest(Dataset):
    COLLECTION = reference(ScifactCollection)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[BeirTopics.C(path=ScifactCollection.data_path / "queries.jsonl")],
                qids_path=ScifactCollection.data_path / "judged_qids_test.txt",
            ),
            assessments=BeirAssessments.C(path=ScifactCollection.data_path / "test.tsv"),
        )


# ============================================================================
# CQADupStack (12 sub-datasets, test only)
# ============================================================================


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackAndroid(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/android/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackEnglish(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/english/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackGaming(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/gaming/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackGis(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/gis/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackMathematica(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/mathematica/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackPhysics(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/physics/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackProgrammers(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/programmers/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackStats(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/stats/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackTex(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/tex/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackUnix(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/unix/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackWebmasters(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/webmasters/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )


@datatasks("information retrieval")
@dataset(url="https://github.com/beir-cellar/beir")
class CqadupstackWordpress(Dataset):
    DATA = ZipDownloader(
        "data", CQADUPSTACK_URL,
        subpath="cqadupstack/wordpress/", transient=True,
    )
    store = docstore_builder(DATA, iter_factory=_read_beir_corpus, keys=["id"])
    files = _single_split_files(DATA)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=BeirDocumentStore.C(path=self.store.path),
            topics=BeirTopics.C(path=self.files.path / "queries.jsonl"),
            assessments=BeirAssessments.C(path=self.files.path / "test.tsv"),
        )
