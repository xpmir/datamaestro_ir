from typing import Iterator, List, Optional
from attr import define
import json
from datamaestro.data import File

from datamaestro_text.data.ir.base import (
    SimpleTextItem,
)


from .base import (
    ConversationTree,
    EntryType,
    SimpleDecontextualizedItem,
    SingleConversationTree,
    ConversationEntry,
)
from . import ConversationDataset


@define(kw_only=True)
class QReCCDatasetEntry:
    """A query with past history"""

    conversation_no: int
    """Conversation ID"""

    turn_no: int
    """The turn in the conversation"""

    conversation_source: str
    """Conversation source"""

    question: str
    """The last issued query"""

    rewrite: str
    """Manually rewritten query"""

    context: List[str]
    """The list of queries asked by the user"""

    answer: str
    """The answer"""

    answer_url: str
    """The URL containing the answer"""


class QReCCDataset(ConversationDataset, File):
    def entries(self) -> Iterator[QReCCDatasetEntry]:
        """Iterates over re-written query with their context"""
        with self.path.open("rt") as fp:
            data = json.load(fp)

        data = [
            QReCCDatasetEntry(**{key.lower(): value for key, value in entry.items()})
            for entry in data
        ]
        return iter(data)

    def __iter__(self) -> Iterator[ConversationTree]:
        history: List[ConversationEntry] = []
        current_id: Optional[str] = None

        for entry in self.entries():
            # Creates a new conversation if needed
            if entry.conversation_no != current_id:
                if current_id is not None:
                    history.reverse()
                    yield SingleConversationTree(current_id, history)

                current_id = entry.conversation_no
                history = []

            # Add to current
            history.append(
                {
                    "id": f"{entry.conversation_no}#{entry.turn_no}",
                    "text_item": SimpleTextItem(entry.question),
                    "answer_url": entry.answer_url,
                    "decontextualized": SimpleDecontextualizedItem(entry.rewrite),
                    "entry_type": EntryType.USER_QUERY,
                }
            )

            history.append(
                {
                    "answer": entry.answer,
                    "entry_type": EntryType.SYSTEM_ANSWER,
                }
            )

        # Yields the last one
        history.reverse()
        yield SingleConversationTree(current_id, history)
