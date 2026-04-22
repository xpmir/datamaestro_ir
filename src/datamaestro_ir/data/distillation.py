"""Distillation dataset types for IR"""

import logging
from dataclasses import dataclass
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
)

from experimaestro import Meta, Param
from datamaestro.data import Base, File
from datamaestro_ir.data import AdhocAssessments
from datamaestro_ir.data.base import (
    IDRecord,
    TextRecord,
    SimpleTextItem,
    ScoredDocument,
)

DocT = TypeVar("DocT")
DocT2 = TypeVar("DocT2")
QueryT = TypeVar("QueryT")
QueryT2 = TypeVar("QueryT2")


@dataclass
class PairwiseDistillationSample(Generic[DocT, QueryT]):
    query: QueryT
    """The query"""

    documents: Tuple[DocT, DocT]
    """Positive/negative document with teacher scores"""

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(
        self, qs: "List[QueryT2]"
    ) -> "PairwiseDistillationSample[DocT, QueryT2]":
        return PairwiseDistillationSample(qs[0], self.documents)

    def get_documents(self) -> List[DocT]:
        return list(self.documents)

    def with_documents(
        self, ds: "List[DocT2]"
    ) -> "PairwiseDistillationSample[DocT2, QueryT]":
        return PairwiseDistillationSample(self.query, tuple(ds))


class PairwiseDistillationSamples(Base, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()


class PairwiseDistillationSamplesTSV(PairwiseDistillationSamples, File):
    """A TSV file (Score 1, Score 2, Query, Document 1, Document 2)"""

    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def _parse_line(self, line: str) -> PairwiseDistillationSample:
        """Parse a single TSV line into a PairwiseDistillationSample."""
        import csv
        import io

        reader = csv.reader(io.StringIO(line), delimiter="\t")
        row = next(reader)

        if self.with_queryid:
            query = IDRecord(id=row[2])
        else:
            query = TextRecord(text_item=SimpleTextItem(row[2]))

        if self.with_docid:
            documents = (
                ScoredDocument(IDRecord(id=row[3]), float(row[0])),
                ScoredDocument(IDRecord(id=row[4]), float(row[1])),
            )
        else:
            documents = (
                ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[3])), float(row[0])
                ),
                ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[4])), float(row[1])
                ),
            )

        return PairwiseDistillationSample(query, documents)


@dataclass
class ListwiseDistillationSample(Generic[DocT, QueryT]):
    query: QueryT
    """The query"""

    documents: List[DocT]
    """List of documents with their ranking position"""

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(
        self, qs: "List[QueryT2]"
    ) -> "ListwiseDistillationSample[DocT, QueryT2]":
        return ListwiseDistillationSample(qs[0], self.documents)

    def get_documents(self) -> List[DocT]:
        return self.documents

    def with_documents(
        self, ds: "List[DocT2]"
    ) -> "ListwiseDistillationSample[DocT2, QueryT]":
        return ListwiseDistillationSample(self.query, list(ds))


class ListwiseDistillationSamples(Base, Iterable[ListwiseDistillationSample]):
    """Listwise distillation file"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()


class ListwiseDistillationSamplesTSV(ListwiseDistillationSamples, File):
    """A TSV file ("query_id", "q0", "doc_id", "rank", "score", "system")"""

    top_k: Meta[int]
    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    @staticmethod
    def _parse_trec_line(line: str) -> tuple:
        """Parse a TREC-format line, return (query_key, row_fields)."""
        parts = line.split("\t") if "\t" in line else line.split()
        return parts[0], parts

    def _build_group(self, query_key: str, rows: list) -> ListwiseDistillationSample:
        """Build a ListwiseDistillationSample from grouped TREC lines."""
        if self.with_queryid:
            query_record = IDRecord(id=query_key)
        else:
            query_record = TextRecord(text_item=SimpleTextItem(query_key))

        documents = []
        for row in rows:
            if self.with_docid:
                doc = ScoredDocument(IDRecord(id=row[2]), float(row[4]))
            else:
                doc = ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[2])), float(row[4])
                )
            documents.append(doc)

        return ListwiseDistillationSample(query_record, documents)


class ListwiseDistillationSamplesTSVWithAnnotations(ListwiseDistillationSamplesTSV):
    qrels: Param[AdhocAssessments]

    def __post_init__(self):
        self.qrels_dict = {}
        logging.info("Loading qrels into memory...")
        for qrel in self.qrels.iter():
            self.qrels_dict[qrel.topic_id] = [
                assess.doc_id for assess in qrel.assessments if assess.rel > 0
            ]


@dataclass
class PointwiseDistillationSample(Generic[DocT, QueryT]):
    """A (query, document, teacher-score) triple.

    The document carries the teacher's similarity / relevance score as a
    :class:`ScoredDocument`; this is the pointwise analogue of
    :class:`PairwiseDistillationSample` (which pairs two docs per query).
    """

    query: QueryT
    """The query"""

    document: DocT
    """The document (typically a ``ScoredDocument``) with its teacher score"""

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(
        self, qs: "List[QueryT2]"
    ) -> "PointwiseDistillationSample[DocT, QueryT2]":
        return PointwiseDistillationSample(qs[0], self.document)

    def get_documents(self) -> List[DocT]:
        return [self.document]

    def with_documents(
        self, ds: "List[DocT2]"
    ) -> "PointwiseDistillationSample[DocT2, QueryT]":
        return PointwiseDistillationSample(self.query, ds[0])


class PointwiseDistillationSamples(Base, Iterable[PointwiseDistillationSample]):
    """Iterable of pointwise distillation samples."""

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        raise NotImplementedError()


class ConcatPointwiseDistillationSamples(PointwiseDistillationSamples):
    """Concatenate several :class:`PointwiseDistillationSamples` sources
    in sequence (SQL UNION ALL semantics — no deduplication)."""

    sources: Param[List[PointwiseDistillationSamples]]
    """Sources to iterate in order."""

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        for source in self.sources:
            yield from source
