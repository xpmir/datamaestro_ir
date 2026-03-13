"""Generic data types for information retrieval"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
import logging
from pathlib import Path
from attrs import define
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import random
from experimaestro import Config, field
from datamaestro.definitions import datatasks, Param, Meta
from datamaestro.data import Base
from datamaestro_ir.utils.files import auto_open
from datamaestro_ir.utils.iter import BatchIterator
from .base import (  # noqa: F401
    # Record types
    TextItem,
    IDRecord,
    TextRecord,
    IDTextRecord,
    SimpleTextItem,
    ScoredDocument,
    # Other things
    AdhocAssessment,
    AdhocAssessedTopic,
)

#: A adhoc run dictionary (query id -> doc id -> score)
AdhocRunDict = dict[str, dict[str, float]]


class Documents(Base):
    """A set of documents with identifiers

    See `IR Datasets <https://ir-datasets.com/index.html>`_ for the list of query classes
    """

    count: Meta[Optional[int]]
    """Number of documents"""

    def iter(self) -> Iterator[IDTextRecord]:
        """Returns an iterator over documents"""
        return self.iter_documents()

    def iter_documents(self) -> Iterator[IDTextRecord]:
        return self.iter()

    def iter_documents_from(self, start=0) -> Iterator[IDTextRecord]:
        """Iterate over a range of documents

        Can be specialized in a subclass for faster access

        :param start: The starting document, defaults to 0
        :return: An iterator
        """
        iter = self.iter()
        if start > 0:
            logging.info("skipping %d documents", start + 1)
            for _ in range(start + 1):
                next(iter)

        return iter

    def iter_ids(self) -> Iterator[str]:
        """Iterates over document ids

        By default, use iter_documents, which is not really efficient.
        """
        for doc in self.iter():
            yield doc["id"]

    @property
    def documentcount(self):
        """Returns the number of terms in the index"""
        if self.count is not None:
            return self.count

        raise NotImplementedError(f"For class {self.__class__}")


class FileAccess(Enum):
    """Defines how to access files (e.g. for document stores)"""

    FILE = 0
    """Direct file access"""

    MMAP = 1
    """Use mmap"""

    MEMORY = 2
    """Use memory"""


class DocumentStore(Documents):
    """A document store

    A document store can
    - match external/internal ID
    - return the document content
    - return the number of documents
    """

    file_access: Meta[FileAccess] = field(default=FileAccess.MMAP, ignore_default=True)
    """How to access the file collection (might not have any impact, depends on
    the docstore)"""

    def docid_internal2external(self, docid: int):
        """Converts an internal collection ID (integer) to an external ID"""
        raise NotImplementedError(f"For class {self.__class__}")

    def document_int(self, internal_docid: int) -> IDTextRecord:
        """Returns a document given its internal ID"""
        docid = self.docid_internal2external(internal_docid)
        return self.document_ext(docid)

    def document_ext(self, docid: str) -> IDTextRecord:
        """Returns a document given its external ID"""
        raise NotImplementedError(f"document() in {self.__class__}")

    def documents_ext(self, docids: List[str]) -> List[IDTextRecord]:
        """Returns documents given their external ID

        By default, just look using `document_ext`, but some store might
        optimize batch retrieval
        """
        return [self.document_ext(docid) for docid in docids]

    def iter_sample(
        self, randint: Optional[Callable[[int], int]]
    ) -> Iterator[IDTextRecord]:
        """Sample documents from the dataset"""
        length = self.documentcount
        randint = randint or (lambda max: random.randint(0, max - 1))
        while True:
            yield self.document_int(randint(length))


class CompressedDocumentStore(DocumentStore, ABC):
    """A document store backed by impact-index's compressed document store"""

    path: Meta[Path]
    """Path to the impact-index store directory"""

    _CONTENT_ACCESS = {
        FileAccess.MEMORY: "memory",
        FileAccess.MMAP: "mmap",
        FileAccess.FILE: "disk",
    }

    @cached_property
    def _store(self):
        import impact_index

        content_access = self._CONTENT_ACCESS[self.file_access]
        return impact_index.DocumentStore.load(str(self.path), content_access)

    @abstractmethod
    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        """Convert stored keys/content into an IDTextRecord

        :param internal_id: The 0-based document number in the store
        :param keys: Key-value metadata stored with the document
        :param content: Binary content of the document
        """
        ...

    @property
    def documentcount(self):
        if self.count is not None:
            return self.count
        return self._store.num_documents()

    def docid_internal2external(self, docid: int):
        docs = self._store.get_by_number([docid])
        return docs[0].keys[self.lookup_key]

    def document_int(self, internal_docid: int) -> IDTextRecord:
        docs = self._store.get_by_number([internal_docid])
        d = docs[0]
        return self.converter(internal_docid, d.keys, d.content)

    def document_ext(self, docid: str) -> IDTextRecord:
        docs = self._store.get_by_key(self.lookup_key, [docid])
        if docs[0] is None:
            raise KeyError(f"Document {docid} not found")
        d = docs[0]
        return self.converter(d.internal_id, d.keys, d.content)

    def documents_ext(self, docids: List[str]) -> List[IDTextRecord]:
        docs = self._store.get_by_key(self.lookup_key, docids)
        return [
            self.converter(d.internal_id, d.keys, d.content)
            if d is not None
            else None
            for d in docs
        ]

    def iter(self) -> Iterator[IDTextRecord]:
        return self.iter_documents_from(0)

    def iter_documents_from(self, start=0) -> Iterator[IDTextRecord]:
        total = self._store.num_documents()
        chunk_size = 4096
        pos = start
        while pos < total:
            end = min(pos + chunk_size, total)
            docs = self._store.get_by_number(list(range(pos, end)))
            for i, d in enumerate(docs, start=pos):
                yield self.converter(i, d.keys, d.content)
            pos = end


class PrefixedDocumentStore(DocumentStore):
    """Combines multiple DocumentStores with ID prefixes.

    Each document ID is expected to start with one of the given prefixes,
    which determines which underlying store to query.
    """

    sources: Param[List[DocumentStore]]
    prefixes: Meta[List[str]]

    def document_ext(self, docid: str):
        for prefix, source in zip(self.prefixes, self.sources):
            if docid.startswith(prefix):
                doc = source.document_ext(docid[len(prefix) :])
                return {**doc, "id": docid}
        raise KeyError(f"No matching prefix for {docid}")

    def documents_ext(self, docids: List[str]) -> List:
        # Group by prefix for batch retrieval
        from collections import defaultdict

        groups: Dict[int, List] = defaultdict(list)
        index_map: Dict[int, List] = defaultdict(list)
        for i, docid in enumerate(docids):
            for j, prefix in enumerate(self.prefixes):
                if docid.startswith(prefix):
                    groups[j].append(docid[len(prefix) :])
                    index_map[j].append(i)
                    break
            else:
                raise KeyError(f"No matching prefix for {docid}")

        results = [None] * len(docids)
        for j, stripped_ids in groups.items():
            prefix = self.prefixes[j]
            docs = self.sources[j].documents_ext(stripped_ids)
            for idx, doc in zip(index_map[j], docs):
                if doc is not None:
                    results[idx] = {**doc, "id": docids[idx]}
        return results

    def iter(self):
        for prefix, source in zip(self.prefixes, self.sources):
            for doc in source.iter():
                yield {**doc, "id": f"{prefix}{doc['id']}"}

    @property
    def documentcount(self):
        if self.count is not None:
            return self.count
        return sum(s.documentcount for s in self.sources)


class AdhocIndex(DocumentStore):
    """An index can be used to retrieve documents based on terms"""

    @property
    def termcount(self):
        """Returns the number of terms in the index"""
        raise NotImplementedError(f"For class {self.__class__}")

    def term_df(self, term: str):
        """Returns the document frequency"""
        raise NotImplementedError(f"For class {self.__class__}")


class Topics(Base, ABC):
    """A set of topics with associated IDs"""

    @abstractmethod
    def iter(self) -> Iterator[IDTextRecord]:
        """Returns an iterator over topics"""
        ...

    def __iter__(self):
        return self.iter()

    def count(self) -> Optional[int]:
        """Returns the number of topics if known"""
        return None


AdhocTopics = Topics


class FilteredTopics(Topics):
    """Merges multiple Topics sources, keeping only query IDs listed in a file"""

    topics: Param[List[Topics]]
    qids_path: Meta[Path]
    """Path to a file with one query ID per line"""

    @cached_property
    def _qids(self) -> set:
        with open(self.qids_path, "rt") as fp:
            return {line.strip() for line in fp if line.strip()}

    def iter(self) -> Iterator[IDTextRecord]:
        for source in self.topics:
            for record in source.iter():
                if record["id"] in self._qids:
                    yield record


class TopicsStore(Topics):
    """Adhoc topics store"""

    @abstractmethod
    def topic_int(self, internal_topic_id: int) -> IDTextRecord:
        """Returns a document given its internal ID"""

    @abstractmethod
    def topic_ext(self, external_topic_id: int) -> IDTextRecord:
        """Returns a document given its external ID"""


class AdhocAssessments(Base, ABC):
    """Ad-hoc assessments (qrels)"""

    def iter(self) -> Iterator[AdhocAssessedTopic]:
        """Returns an iterator over assessments"""
        raise NotImplementedError(f"For class {self.__class__}")


class AdhocRun(Base):
    """IR adhoc run"""

    @abstractmethod
    def get_dict(self) -> "AdhocRunDict":
        """Get the run as a dictionary query ID -> doc ID -> score"""
        ...


class AdhocResults(Base):
    def get_results(self) -> Dict[str, float]:
        """Returns the aggregated results

        :return: Returns a dictionary where each metric (keys) is associated
            with a value
        """
        raise NotImplementedError(f"For class {self.__class__}")


@datatasks("information retrieval")
class Adhoc(Base):
    """An Adhoc IR collection with documents, topics and their assessments"""

    documents: Param[Documents]
    """The set of documents"""

    topics: Param[Topics]
    """The set of topics"""

    assessments: Param[AdhocAssessments]
    """The set of assessments (for each topic)"""


class RerankAdhoc(Adhoc):
    """Re-ranking ad-hoc task based on an existing run"""

    run: Param[AdhocRun]
    """The run to re-rank"""


class Measure(Config):
    """An Information Retrieval measure"""

    pass


#: A single record in a triplet (may have only id or only text_item)
TripletRecord = Union[IDRecord, TextRecord, IDTextRecord]

#: A training triplet: (topic, positive doc, negative doc)
Triplets = Tuple[TripletRecord, TripletRecord, TripletRecord]


class TrainingTriplets(Base, ABC):
    """Triplet for training IR systems: query / query ID, positive document,
    negative document"""

    def iter(self) -> Iterator[Triplets]:
        """Returns an iterator over (topic, document 1, document) triplets"""
        raise NotImplementedError(f"For class {self.__class__}")

    def batch_iter(self, size: int) -> Iterator[List[Triplets]]:
        """Returns an iterator over batches of triplets

        The default implementation just concatenates triplets using `iter`, but
        some classes might use more efficient ways to provide batches of data
        """
        return BatchIterator(self.iter(), size)

    def count(self):
        """Returns the number of triplets or None"""
        return None


class TrainingTripletsLines(TrainingTriplets):
    """Training triplets with one line per triple (query texts)"""

    sep: Meta[str]
    path: Param[Path]

    doc_ids: Meta[bool]
    """True if we have documents IDs"""

    topic_ids: Meta[bool]
    """True if we have query IDs"""

    def iter(self) -> Iterator[Triplets]:
        with auto_open(self.path, "rt") as fp:
            for line in fp:
                q, pos, neg = line.strip().split(self.sep)
                yield self._topic(q), self._doc(pos), self._doc(neg)

    @cached_property
    def _doc(self):
        if self.doc_ids:
            return lambda doc: {"id": doc}
        else:
            return lambda doc: {"text_item": SimpleTextItem(doc)}

    @cached_property
    def _topic(self):
        if self.topic_ids:
            return lambda q: {"id": q}
        else:
            return lambda q: {"text_item": SimpleTextItem(q)}


@define(kw_only=True)
class PairwiseSample(ABC):
    """A a query with positive and negative samples"""

    topics: List[IDTextRecord]
    """The topic(s)"""

    positives: List[IDTextRecord]
    """Relevant documents"""

    negatives: Dict[str, List[IDTextRecord]]
    """Non relevant documents, organized in a dictionary where keys
    are the algorithm used to retrieve the negatives"""


class PairwiseSampleDataset(Base, ABC):
    """Datasets where each record is a query with positive and negative samples"""

    @abstractmethod
    def iter(self) -> Iterator[PairwiseSample]: ...
