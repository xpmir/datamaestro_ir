"""LoTTE (Long-Tail Topic-stratified Evaluation) benchmark datasets.

Provides native datamaestro dataset definitions for 6 domains x 2 splits
x 2 query types = 24 IR tasks. A single 3.6GB tar.gz is downloaded once;
per-domain docstores are built, and query/qrel files are copied out.

See: https://github.com/stanford-futuredata/ColBERT
"""

from pathlib import Path

from datamaestro.definitions import Dataset, datatasks, dataset
from datamaestro.download import FilesCopy, reference
from datamaestro.download.archive import TarDownloader
from datamaestro_ir.data import Adhoc
from datamaestro_ir.data.lotte import (
    LotteAssessments,
    LotteDocumentStore,
    LotteTopics,
)
from datamaestro_ir.download.docstore import docstore_builder

LOTTE_URL = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz"

DOMAINS = ["lifestyle", "recreation", "science", "technology", "writing", "pooled"]
SPLITS = ["dev", "test"]
QUERY_TYPES = ["search", "forum"]


# --- Helpers ---


def _corpus_reader(domain, split):
    """Returns a function that reads collection.tsv for a domain/split."""

    def reader(source_path: Path):
        tsv_path = source_path / domain / split / "collection.tsv"
        with open(tsv_path, "rt") as fp:
            for line in fp:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    yield {"id": doc_id}, text.encode("utf-8")

    return reader


def _build_files_map():
    """Build the FilesCopy mapping for all query and qrel files."""
    files = {}
    for domain in DOMAINS:
        for split in SPLITS:
            for qtype in QUERY_TYPES:
                files[f"{domain}.{split}.{qtype}.queries.tsv"] = (
                    f"{domain}/{split}/questions.{qtype}.tsv"
                )
                files[f"{domain}.{split}.{qtype}.qrels.jsonl"] = (
                    f"{domain}/{split}/qas.{qtype}.jsonl"
                )
    return files


# --- Main data resource ---


@dataset()
class LotteData(Dataset):
    DATA = TarDownloader("data", LOTTE_URL, transient=True)

    # 12 docstores (one per domain/split)
    store_lifestyle_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("lifestyle", "dev"), keys=["id"]
    )
    store_lifestyle_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("lifestyle", "test"), keys=["id"]
    )
    store_recreation_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("recreation", "dev"), keys=["id"]
    )
    store_recreation_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("recreation", "test"), keys=["id"]
    )
    store_science_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("science", "dev"), keys=["id"]
    )
    store_science_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("science", "test"), keys=["id"]
    )
    store_technology_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("technology", "dev"), keys=["id"]
    )
    store_technology_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("technology", "test"), keys=["id"]
    )
    store_writing_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("writing", "dev"), keys=["id"]
    )
    store_writing_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("writing", "test"), keys=["id"]
    )
    store_pooled_dev = docstore_builder(
        DATA, iter_factory=_corpus_reader("pooled", "dev"), keys=["id"]
    )
    store_pooled_test = docstore_builder(
        DATA, iter_factory=_corpus_reader("pooled", "test"), keys=["id"]
    )

    files = FilesCopy(DATA, _build_files_map())

    def config(self) -> LotteDocumentStore:
        # This is a resource-only dataset; per-domain stores are
        # accessed via the per-task datasets below. We return one
        # store as a default.
        return LotteDocumentStore.C(
            path=self.store_pooled_dev.path,
        )


# --- Per-task base class and datasets ---


class LotteDataset(Dataset):
    """Base class for LoTTE per-task datasets."""

    DOMAIN: str
    SPLIT: str
    QTYPE: str

    LOTTE = reference(varname="lotte", reference=LotteData)

    def config(self) -> Adhoc:
        col = LotteData.__dataset__
        domain = self.DOMAIN
        split = self.SPLIT
        qtype = self.QTYPE
        return Adhoc.C(
            documents=LotteDocumentStore.C(
                path=col.datapath / f"store_{domain}_{split}",
            ),
            topics=LotteTopics.C(
                path=col.datapath / f"{domain}.{split}.{qtype}.queries.tsv",
            ),
            assessments=LotteAssessments.C(
                path=col.datapath / f"{domain}.{split}.{qtype}.qrels.jsonl",
            ),
        )


# --- Lifestyle ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class LifestyleDevSearch(LotteDataset):
    DOMAIN = "lifestyle"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class LifestyleDevForum(LotteDataset):
    DOMAIN = "lifestyle"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class LifestyleTestSearch(LotteDataset):
    DOMAIN = "lifestyle"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class LifestyleTestForum(LotteDataset):
    DOMAIN = "lifestyle"
    SPLIT = "test"
    QTYPE = "forum"


# --- Recreation ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class RecreationDevSearch(LotteDataset):
    DOMAIN = "recreation"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class RecreationDevForum(LotteDataset):
    DOMAIN = "recreation"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class RecreationTestSearch(LotteDataset):
    DOMAIN = "recreation"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class RecreationTestForum(LotteDataset):
    DOMAIN = "recreation"
    SPLIT = "test"
    QTYPE = "forum"


# --- Science ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class ScienceDevSearch(LotteDataset):
    DOMAIN = "science"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class ScienceDevForum(LotteDataset):
    DOMAIN = "science"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class ScienceTestSearch(LotteDataset):
    DOMAIN = "science"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class ScienceTestForum(LotteDataset):
    DOMAIN = "science"
    SPLIT = "test"
    QTYPE = "forum"


# --- Technology ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class TechnologyDevSearch(LotteDataset):
    DOMAIN = "technology"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class TechnologyDevForum(LotteDataset):
    DOMAIN = "technology"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class TechnologyTestSearch(LotteDataset):
    DOMAIN = "technology"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class TechnologyTestForum(LotteDataset):
    DOMAIN = "technology"
    SPLIT = "test"
    QTYPE = "forum"


# --- Writing ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class WritingDevSearch(LotteDataset):
    DOMAIN = "writing"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class WritingDevForum(LotteDataset):
    DOMAIN = "writing"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class WritingTestSearch(LotteDataset):
    DOMAIN = "writing"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class WritingTestForum(LotteDataset):
    DOMAIN = "writing"
    SPLIT = "test"
    QTYPE = "forum"


# --- Pooled ---


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class PooledDevSearch(LotteDataset):
    DOMAIN = "pooled"
    SPLIT = "dev"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class PooledDevForum(LotteDataset):
    DOMAIN = "pooled"
    SPLIT = "dev"
    QTYPE = "forum"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class PooledTestSearch(LotteDataset):
    DOMAIN = "pooled"
    SPLIT = "test"
    QTYPE = "search"


@datatasks("information retrieval")
@dataset(Adhoc, url=LOTTE_URL)
class PooledTestForum(LotteDataset):
    DOMAIN = "pooled"
    SPLIT = "test"
    QTYPE = "forum"
