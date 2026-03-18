from abc import ABC, abstractmethod
from typing import List, TypedDict
from attrs import define
from typing_extensions import ReadOnly


class TextItem(ABC):
    @property
    @abstractmethod
    def text(self) -> str:
        """Returns the text"""


@define
class SimpleTextItem(TextItem):
    """A topic/document with a text record"""

    text: str


@define
class AdhocAssessment:
    doc_id: str


@define
class SimpleAdhocAssessment(AdhocAssessment):
    rel: float
    """Relevance (> 0 if relevant)"""


@define
class AdhocAssessedTopic:
    topic_id: str
    """The topic ID"""

    assessments: List[AdhocAssessment]
    """List of assessments for this topic"""


class IDRecord(TypedDict):
    """A record with just an ID"""

    id: ReadOnly[str]


class TextRecord(TypedDict):
    """A record with just a text item"""

    text_item: ReadOnly[TextItem]


class IDTextRecord(IDRecord, TextRecord):
    """A record with an ID and a text item"""

    pass


@define
class ScoredDocument:
    """A data structure that associates a score with a document, allowing to sort documents by score (e.g., for nDCG)"""

    document: dict
    """The document (IDRecord, TextRecord, or IDTextRecord)"""

    score: float
    """The associated score"""

    def __repr__(self):
        return f"ScoredDocument(document=({self.document}), score={self.score})"

    # enables to sort documents by score (e.g., for nDCG)
    def __lt__(self, other):
        return self.score < other.score
