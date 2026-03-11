from typing import Iterator, List
from attr import define
import json
from datamaestro.data import File
from datamaestro_ir.data.conversation.base import (
    ConversationDataset,
    ConversationEntry,
    ConversationTree,
    SingleConversationTree,
    SimpleDecontextualizedItem,
    EntryType,
)
from datamaestro_ir.data.base import SimpleTextItem


@define(kw_only=True)
class CanardConversation:
    """A query with past history"""

    history: List[str]
    """The list of queries asked by the user"""

    query: str
    """The last issued query"""

    rewrite: str
    """Manually rewritten query"""

    dialogue_id: str
    """Conversation identifier"""

    query_no: int
    """Question number"""


class CanardDataset(ConversationDataset, File):
    """A dataset in the CANARD JSON format

    The CANARD dataset is composed of
    """

    def entries(self) -> Iterator[CanardConversation]:
        """Iterates over re-written query with their context"""
        with self.path.open("rt") as fp:
            data = json.load(fp)

        for entry in data:
            yield CanardConversation(
                history=entry["History"],
                query=entry["Question"],
                rewrite=entry["Rewrite"],
                dialogue_id=entry["QuAC_dialog_id"],
                query_no=entry["Question_no"],
            )

    def __iter__(self) -> Iterator[ConversationTree]:
        history: list[ConversationEntry] = []
        current_id = None

        for entry in self.entries():
            # Check if current conversation, otherwise we are OK
            if current_id != entry.dialogue_id:
                if current_id is not None:
                    history.reverse()
                    yield SingleConversationTree(current_id, history)
                    history = []

                current_id = entry.dialogue_id

            if not history:
                # First round
                # The two first items are the wikipedia title and section,
                # we interpret them as two user queries
                assert len(entry.history) == 2
                history.extend(
                    {
                        "text_item": SimpleTextItem(text),
                        "entry_type": EntryType.USER_QUERY,
                    }
                    for text in entry.history
                )
            else:
                # The utterance before the last is the last user query
                assert entry.history[-2] == history[-1]["text_item"].text, (
                    f"{entry.dialogue_id} {entry.history} / {history[-4:-1]}"
                )

                # The last utterance is the system side
                history.append(
                    {
                        "text_item": SimpleTextItem(entry.history[-1]),
                        "entry_type": EntryType.SYSTEM_ANSWER,
                    }
                )

            assert len(entry.history) == len(history)

            # Add to current
            history.append(
                {
                    "id": f"{entry.dialogue_id}-{entry.query_no}",
                    "text_item": SimpleTextItem(entry.query),
                    "decontextualized": SimpleDecontextualizedItem(entry.rewrite),
                    "entry_type": EntryType.USER_QUERY,
                }
            )

        if current_id:
            yield SingleConversationTree(current_id, history)
