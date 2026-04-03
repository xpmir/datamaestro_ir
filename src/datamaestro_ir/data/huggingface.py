from typing import Iterator
from experimaestro import Meta, field
from datamaestro.data.huggingface import HuggingFaceDataset
from . import PairwiseSample, PairwiseSampleDataset


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
