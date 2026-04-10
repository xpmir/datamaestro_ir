"""TREC Adhoc datasets and tasks

See [https://trec.nist.gov/data/test_coll.html](https://trec.nist.gov/data/test_coll.html)
"""

import logging
from pathlib import Path

from datamaestro.download import FolderResource, reference
from datamaestro.download.single import FileDownloader, ConcatDownloader
from datamaestro.download.links import links
from datamaestro.stream import TransformList
from datamaestro.stream.compress import Gunzip
from datamaestro.stream.lines import Replace, Filter
from datamaestro.definitions import Dataset, dataset

from datamaestro_ir.data.trec import (
    TipsterCollection,
    TrecTopics,
    TrecAdhocAssessments,
)
from datamaestro_ir.data import Adhoc, DocumentStore
from datamaestro_ir.data.stores import TipsterDocumentStore

from . import tipster
from datamaestro_ir.config.edu.upenn.ldc.aquaint import Aquaint

logger = logging.getLogger(__name__)


class _TipsterDocstoreBuilder(FolderResource):
    """Builds a TipsterDocumentStore from a referenced document collection."""

    def __init__(self, docs_ref):
        super().__init__()
        self._docs_ref = docs_ref
        self._dependencies.append(docs_ref)

    def _download(self, destination: Path) -> None:
        import json

        import impact_index

        from datamaestro_ir.interfaces.trec import iter_tipster_collection

        docs_config = self._docs_ref.config()

        from tqdm import tqdm

        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Building TIPSTER docstore in %s", destination)
        builder = impact_index.DocumentStoreBuilder(str(destination), 4096, 3)
        for doc in tqdm(
            iter_tipster_collection(docs_config.path, docs_config.patterns),
            desc="Building docstore",
        ):
            text_item = doc["text_item"]
            content = json.dumps(
                {"title": text_item.title, "body": text_item.body}
            ).encode("utf-8")
            builder.add({"id": doc["id"]}, content)
        builder.build()


def with_docstore(docs_cls):
    """Create a .store variant of a document collection Dataset.

    Returns a new Dataset whose config() yields a TipsterDocumentStore.
    The original document files are referenced transiently (cleaned up
    after the store is built).

    Usage::

        Trec7DocumentsStore = with_docstore(Trec7Documents)
    """
    base_id = docs_cls.__dataset__.id

    @dataset(id=f"{base_id}.store")
    class _Store(Dataset):
        DOCUMENTS = reference(docs_cls)
        DOCUMENTS.transient = True
        store = _TipsterDocstoreBuilder(DOCUMENTS)

        def config(self) -> TipsterDocumentStore:
            return TipsterDocumentStore.C(path=self.store.path)

    _Store.__qualname__ = f"{docs_cls.__name__}Store"
    _Store.__name__ = f"{docs_cls.__name__}Store"
    return _Store


def with_store(adhoc_cls, docs_store_cls):
    """Create a .withstore variant of an adhoc dataset.

    The variant uses a pre-built TipsterDocumentStore instead of
    raw document files.

    Usage::

        Robust2004WithStore = with_store(Robust2004, Trec7DocumentsStore)
    """
    base_id = adhoc_cls.__dataset__.id

    @dataset(id=f"{base_id}.withstore")
    class _WithStore(Dataset):
        DOCUMENTS = reference(docs_store_cls)
        BASE = reference(adhoc_cls)

        def config(self) -> Adhoc:
            base_config = self.BASE.config()
            return Adhoc.C(
                documents=self.DOCUMENTS.config(),
                topics=base_config.topics,
                assessments=base_config.assessments,
            )

    _WithStore.__qualname__ = f"{adhoc_cls.__name__}WithStore"
    _WithStore.__name__ = f"{adhoc_cls.__name__}WithStore"
    return _WithStore


# --- TREC 1 (1992)


@dataset(id=".1.documents")
class Trec1Documents(Dataset):
    """TREC-1 to TREC-3 documents (TIPSTER volumes 1 and 2)"""

    DOCUMENTS = links(
        "documents",
        ap88=tipster.Ap88,
        ap89=tipster.Ap89,
        fr88=tipster.Fr88,
        fr89=tipster.Fr89,
        wsj87=tipster.Wsj87,
        wsj88=tipster.Wsj88,
        wsj89=tipster.Wsj89,
        wsj90=tipster.Wsj90,
        wsj91=tipster.Wsj91,
        wsj92=tipster.Wsj92,
        ziff1=tipster.Ziff1,
        ziff2=tipster.Ziff2,
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/AP*",
                "*/documents/FR*",
                "*/documents/WSJ*",
                "*/documents/ZF*",
            ],
        )


@dataset(id=".1.topics", url="")
class Trec1Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml",
        "http://trec.nist.gov/data/topics_eng/topics.51-100.gz",
        transforms=TransformList(Gunzip(), Replace(r"Number:(\s+)0", r"Number: \1")),
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["desc"])


@dataset(id=".1.assessments")
class Trec1Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        "http://trec.nist.gov/data/qrels_eng/qrels.51-100.disk1.disk2.parts1-5.tar.gz",
        transforms=TransformList(Gunzip(), Replace(r"Number:(\s+)0", r"Number: \1")),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".1")
class Trec1(Dataset):
    "Ad-hoc task of TREC 1 (1992)"

    DOCUMENTS = reference(Trec1Documents)
    TOPICS = reference(Trec1Topics)
    ASSESSMENTS = reference(Trec1Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 2 (1993)


@dataset(id=".2.topics")
class Trec2Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml",
        "http://trec.nist.gov/data/topics_eng/topics.101-150.gz",
        transforms=TransformList(Gunzip(), Replace(r"Number:(\s+)0", r"Number: \1")),
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".2.assessments")
class Trec2Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        "http://trec.nist.gov/data/qrels_eng/qrels.101-150.disk1.disk2.parts1-5.tar.gz",
        transforms=TransformList(Gunzip(), Replace(r"Number:(\s+)0", r"Number: \1")),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".2")
class Trec2(Dataset):
    "Ad-hoc task of TREC 2 (1993)"

    DOCUMENTS = reference(Trec1Documents)
    TOPICS = reference(Trec2Topics)
    ASSESSMENTS = reference(Trec2Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 3 (1994)


@dataset(id=".3.topics")
class Trec3Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.151-200.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".3.assessments")
class Trec3Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        "http://trec.nist.gov/data/qrels_eng/qrels.151-200.201-250.disks1-3.all.tar.gz",
        transforms=TransformList(Gunzip(), Filter(r"^(1\d\d|200)\s")),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".3")
class Trec3(Dataset):
    "Ad-hoc task of TREC 3 (1994)"

    DOCUMENTS = reference(Trec1Documents)
    TOPICS = reference(Trec3Topics)
    ASSESSMENTS = reference(Trec3Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 4 (1995)


@dataset(id=".4.documents")
class Trec4Documents(Dataset):
    """TREC-4 documents"""

    DOCUMENTS = links(
        "documents",
        ap88=tipster.Ap88,
        ap89=tipster.Ap89,
        ap90=tipster.Ap90,
        fr88=tipster.Fr88,
        sjm1=tipster.Sjm1,
        wsj90=tipster.Wsj90,
        wsj91=tipster.Wsj91,
        wsj92=tipster.Wsj92,
        ziff2=tipster.Ziff2,
        ziff3=tipster.Ziff3,
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/AP*",
                "*/documents/FR*",
                "*/documents/SJM*",
                "*/documents/WSJ*",
                "*/documents/ZF*",
            ],
        )


@dataset(id=".4.topics")
class Trec4Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.201-250.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".4.assessments")
class Trec4Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        "http://trec.nist.gov/data/qrels_eng/qrels.201-250.disk2.disk3.parts1-5.tar.gz",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".4")
class Trec4(Dataset):
    "Ad-hoc task of TREC 4 (1995)"

    DOCUMENTS = reference(Trec4Documents)
    TOPICS = reference(Trec4Topics)
    ASSESSMENTS = reference(Trec4Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 5 (1995)


@dataset(id=".5.documents")
class Trec5Documents(Dataset):
    """TREC-5 documents"""

    DOCUMENTS = links(
        "documents",
        ap88=tipster.Ap88,
        cr1=tipster.Cr1,
        fr88=tipster.Fr88,
        fr94=tipster.Fr94,
        ft1=tipster.Ft1,
        wsj90=tipster.Wsj90,
        wsj91=tipster.Wsj91,
        wsj9=tipster.Wsj92,
        ziff2=tipster.Ziff2,
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/AP*",
                "*/documents/*/CR*",
                "*/documents/FR*",
                "*/documents/*/FR*",
                "*/documents/*/FT*",
                "*/documents/WSJ*",
                "*/documents/ZF*",
            ],
        )


@dataset(id=".5.topics")
class Trec5Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.251-300.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".5.qrels")
class Trec5Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        url="http://trec.nist.gov/data/qrels_eng/qrels.251-300.parts1-5.tar.gz",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".5")
class Trec5(Dataset):
    "Ad-hoc task of TREC 5 (1996)"

    DOCUMENTS = reference(Trec5Documents)
    TOPICS = reference(Trec5Topics)
    ASSESSMENTS = reference(Trec5Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# -- TREC 6 (1997)


@dataset(id=".6.documents")
class Trec6Documents(Dataset):
    """TREC-5 documents"""

    DOCUMENTS = links(
        "documents",
        cr1=tipster.Cr1,
        fbis1=tipster.Fbis1,
        fr94=tipster.Fr94,
        ft1=tipster.Ft1,
        la8990=tipster.La8990,
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/*/CR*",
                "*/documents/FB*",
                "*/documents/*/FR*",
                "*/documents/*/FT*",
                "*/documents/LA*",
            ],
        )


@dataset(id=".6.topics")
class Trec6Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.301-350.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".6.qrels")
class Trec6Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        url="http://trec.nist.gov/data/qrels_eng/qrels.trec6.adhoc.parts1-5.tar.gz",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".6")
class Trec6(Dataset):
    "Ad-hoc task of TREC 6 (1997)"

    DOCUMENTS = reference(Trec6Documents)
    TOPICS = reference(Trec6Topics)
    ASSESSMENTS = reference(Trec6Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 7 (1998)


@dataset(id=".7.documents")
class Trec7Documents(Dataset):
    """TREC-7 documents"""

    DOCUMENTS = links(
        "documents",
        fbis1=tipster.Fbis1,
        fr94=tipster.Fr94,
        ft1=tipster.Ft1,
        la8990=tipster.La8990,
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/FB*",
                "*/documents/*/FR*",
                "*/documents/*/FT*",
                "*/documents/LA*",
            ],
        )


@dataset(id=".7.topics")
class Trec7Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.351-400.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".7.qrels")
class Trec7Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        url="http://trec.nist.gov/data/qrels_eng/qrels.trec7.adhoc.parts1-5.tar.gz",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".7")
class Trec7(Dataset):
    "Ad-hoc task of TREC 3 (1994)"

    DOCUMENTS = reference(Trec7Documents)
    TOPICS = reference(Trec7Topics)
    ASSESSMENTS = reference(Trec7Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC 8 (1999)


@dataset(id=".8.topics")
class Trec8Topics(Dataset):
    FILE = FileDownloader(
        "topics.sgml", "http://trec.nist.gov/data/topics_eng/topics.401-450.gz"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".8.qrels")
class Trec8Assessments(Dataset):
    FILE = ConcatDownloader(
        "assessments.qrels",
        url="https://trec.nist.gov/data/qrels_eng/qrels.trec8.adhoc.parts1-5.tar.gz",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".8")
class Trec8(Dataset):
    "Ad-hoc task of TREC 8 (1999)"

    DOCUMENTS = reference(Trec7Documents)
    TOPICS = reference(Trec8Topics)
    ASSESSMENTS = reference(Trec8Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC Robust (2004)


@dataset(id=".robust.2004.topics")
class Robust2004Topics(Dataset):
    FILE = FileDownloader("topics", "http://trec.nist.gov/data/robust/04.testset.gz")

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".robust.2004.qrels")
class Robust2004Assessments(Dataset):
    FILE = FileDownloader(
        "assessments.qrels", "http://trec.nist.gov/data/robust/qrels.robust2004.txt"
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".robust.2004")
class Robust2004(Dataset):
    "Ad-hoc task of TREC Robust (2004)"

    DOCUMENTS = reference(Trec7Documents)
    TOPICS = reference(Robust2004Topics)
    ASSESSMENTS = reference(Robust2004Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- TREC Robust (2005)


@dataset(id=".robust.2005.topics")
class Robust2005Topics(Dataset):
    FILE = FileDownloader(
        "topics", "http://trec.nist.gov/data/robust/05/05.50.topics.txt"
    )

    def config(self) -> TrecTopics:
        return TrecTopics.C(path=self.FILE.path, parts=["title", "desc"])


@dataset(id=".robust.2005.qrels")
class Robust2005Assessments(Dataset):
    FILE = FileDownloader(
        "assessments.qrels",
        url="http://trec.nist.gov/data/robust/05/TREC2005.qrels.txt",
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.FILE.path)


@dataset(id=".robust.2005")
class Robust2005(Dataset):
    "Ad-hoc task of TREC Robust (2005)"

    DOCUMENTS = reference(Aquaint)
    TOPICS = reference(Robust2005Topics)
    ASSESSMENTS = reference(Robust2005Assessments)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.DOCUMENTS.config(),
            topics=self.TOPICS.config(),
            assessments=self.ASSESSMENTS.config(),
        )


# --- Document stores ---

Trec1DocumentsStore = with_docstore(Trec1Documents)
Trec4DocumentsStore = with_docstore(Trec4Documents)
Trec5DocumentsStore = with_docstore(Trec5Documents)
Trec6DocumentsStore = with_docstore(Trec6Documents)
Trec7DocumentsStore = with_docstore(Trec7Documents)
AquaintStore = with_docstore(Aquaint)

# --- WithStore variants ---

Trec1WithStore = with_store(Trec1, Trec1DocumentsStore)
Trec2WithStore = with_store(Trec2, Trec1DocumentsStore)
Trec3WithStore = with_store(Trec3, Trec1DocumentsStore)
Trec4WithStore = with_store(Trec4, Trec4DocumentsStore)
Trec5WithStore = with_store(Trec5, Trec5DocumentsStore)
Trec6WithStore = with_store(Trec6, Trec6DocumentsStore)
Trec7WithStore = with_store(Trec7, Trec7DocumentsStore)
Trec8WithStore = with_store(Trec8, Trec7DocumentsStore)
Robust2004WithStore = with_store(Robust2004, Trec7DocumentsStore)
Robust2005WithStore = with_store(Robust2005, AquaintStore)
