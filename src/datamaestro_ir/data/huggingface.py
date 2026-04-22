from typing import Iterator
from experimaestro import Meta, field
from datamaestro.data.huggingface import HuggingFaceDataset
from . import PairwiseSample, PairwiseSampleDataset
from .base import ScoredDocument, SimpleTextItem
from .distillation import (
    PointwiseDistillationSample,
    PointwiseDistillationSamples,
)


class HuggingFacePairwiseSampleDataset(HuggingFaceDataset, PairwiseSampleDataset):
    """Triplet for training IR systems: query / query ID, positive document, negative document

    Attributes:

        ids: True if the triplet is made of IDs, False otherwise
    """

    ids: Meta[bool] = field(default=True, ignore_default=True)

    query_id: Meta[str] = field(default="qid", ignore_default=True)
    """The name of the field containing the query ID"""

    pos_id: Meta[str] = field(default="pos", ignore_default=True)
    """The name of the field containing the positive samples"""

    neg_id: Meta[str] = field(default="neg", ignore_default=True)
    """The name of the field containing the negative samples"""

    def iter(self) -> Iterator[PairwiseSample]:
        for element in self.data:
            yield PairwiseSample(
                element[self.query_id], element[self.pos_id], element[self.neg_id]
            )


class HuggingFacePointwiseDistillationSamples(
    HuggingFaceDataset, PointwiseDistillationSamples
):
    """(query, document, teacher-score) samples from a HuggingFace dataset.

    Schema-agnostic: override the field-name ``Meta`` attributes when the
    source dataset uses different column names.
    """

    query_field: Meta[str] = field(default="query", ignore_default=True)
    """Name of the column holding the query text."""

    document_field: Meta[str] = field(default="document", ignore_default=True)
    """Name of the column holding the document text."""

    score_field: Meta[str] = field(default="similarity", ignore_default=True)
    """Name of the column holding the teacher score."""

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        for row in self.data:
            yield self._build_sample(row)

    def _build_sample(self, row) -> PointwiseDistillationSample:
        return PointwiseDistillationSample(
            query={"text_item": SimpleTextItem(row[self.query_field])},
            document=ScoredDocument(
                {"text_item": SimpleTextItem(row[self.document_field])},
                float(row[self.score_field]),
            ),
        )
