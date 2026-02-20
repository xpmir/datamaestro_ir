from abc import ABC, abstractmethod
from enum import Enum
from datamaestro_text.data.ir.base import TextItem
from experimaestro import Param
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict
from attr import define
from datamaestro.data import Base
from datamaestro_text.data.ir import Topics
from datamaestro_text.utils.iter import FactoryIterable, LazyList
from typing_extensions import ReadOnly

# ---- Basic types


class EntryType(Enum):
    """Type of record"""

    USER_QUERY = 0
    SYSTEM_ANSWER = 1
    CLARIFYING_QUESTION = 2


class DecontextualizedItem:
    """A topic record with decontextualized versions of the topic"""

    @abstractmethod
    def get_decontextualized_query(self, mode=None) -> str:
        """Returns the decontextualized query"""
        ...


@define
class SimpleDecontextualizedItem(DecontextualizedItem):
    """A topic record with one decontextualized version of the topic"""

    decontextualized_query: str

    def get_decontextualized_query(self, mode=None) -> str:
        """Returns the decontextualized query"""
        assert mode is None

        return self.decontextualized_query


@define
class DecontextualizedDictItem(DecontextualizedItem):
    """A conversation entry providing decontextualized version of the user query"""

    default_decontextualized_key: str

    decontextualized_queries: Dict[str, str]

    def get_decontextualized_query(self, mode=None):
        return self.decontextualized_queries[mode or self.default_decontextualized_key]


@define
class RetrievedEntry:
    """List of system-retrieved documents and their relevance"""

    documents: List[str]
    """List of retrieved documents"""

    relevant_documents: Optional[Dict[int, Tuple[Optional[int], Optional[int]]]] = None
    """List of relevance status (optional), with start/stop position"""


class ConversationEntry(TypedDict, total=False):
    id: ReadOnly[str]
    text_item: ReadOnly[TextItem]
    entry_type: ReadOnly[EntryType]
    decontextualized: ReadOnly[DecontextualizedItem]
    history: ReadOnly["ConversationHistory"]
    answer: ReadOnly[str]
    answer_document_id: ReadOnly[str]
    answer_url: ReadOnly[str]
    retrieved: ReadOnly[RetrievedEntry]


#: The conversation
ConversationHistory = Sequence[ConversationEntry]


# ---- Abstract conversation representation


class ConversationNode:
    @abstractmethod
    def entry(self) -> ConversationEntry:
        """The current conversation entry"""
        ...

    @abstractmethod
    def history(self) -> ConversationHistory:
        """Preceding conversation entries, from most recent to more ancient"""
        ...

    @abstractmethod
    def parent(self) -> Optional["ConversationNode"]: ...

    @abstractmethod
    def children(self) -> List["ConversationNode"]: ...


class ConversationTree(ABC):
    """Represents a conversation tree"""

    @abstractmethod
    def root(self) -> ConversationNode: ...

    @abstractmethod
    def __iter__(self) -> Iterator[ConversationNode]:
        """Iterates over conversation nodes"""
        ...


# ---- A conversation tree


class SingleConversationTree(ConversationTree, ABC):
    """Simple conversations, based on a sequence of entries"""

    id: str
    history: List[ConversationEntry]

    def __init__(self, id: Optional[str], history: List[ConversationEntry]):
        """Create a simple conversation

        :param history: The entries, in **reverse** order (i.e. more ancient first)
        """
        self.history = history or []
        self.id = id

    def add(self, entry: ConversationEntry):
        self.history.insert(0, entry)

    def __iter__(self) -> Iterator[ConversationNode]:
        """Iterates over the conversation (starting with the beginning)"""
        for ix in reversed(range(len(self.history))):
            yield SingleConversationTreeNode(self, ix)

    def root(self):
        return SingleConversationTreeNode(self, len(self.history) - 1)


@define
class SingleConversationTreeNode(ConversationNode):
    tree: SingleConversationTree
    index: int

    @property
    def entry(self) -> ConversationEntry:
        return self.tree.history[self.index]

    @entry.setter
    def entry(self, record: ConversationEntry):
        try:
            self.tree.history[self.index] = record
        except Exception as e:
            print(e)
            raise

    def history(self) -> Sequence[ConversationEntry]:
        return self.tree.history[self.index + 1 :]

    def parent(self) -> Optional[ConversationNode]:
        return (
            SingleConversationTreeNode(self.tree, self.index + 1)
            if self.index < len(self.tree.history) - 1
            else None
        )

    def children(self) -> List[ConversationNode]:
        return (
            [SingleConversationTreeNode(self.tree, self.index - 1)]
            if self.index > 0
            else []
        )


class ConversationTreeNode(ConversationNode, ConversationTree):
    """A conversation tree node"""

    entry: ConversationEntry
    _parent: Optional["ConversationTreeNode"]
    _children: List["ConversationTreeNode"]

    def __init__(self, entry):
        self.entry = entry
        self._parent = None
        self._children = []

    def add(self, node: "ConversationTreeNode") -> "ConversationTreeNode":
        self._children.append(node)
        node._parent = self
        return node

    def conversation(self, skip_self: bool) -> ConversationHistory:
        def iterator():
            current = self.parent() if skip_self else self
            while current is not None:
                yield current.entry
                current = current.parent()

        return LazyList(FactoryIterable(iterator))

    def __iter__(self) -> Iterator["ConversationTreeNode"]:
        """Iterates over all conversation tree nodes (pre-order)"""
        yield self.entry
        for child in self._children:
            yield from child

    def parent(self) -> Optional[ConversationNode]:
        return self._parent

    def children(self) -> List[ConversationNode]:
        return self._children

    def root(self):
        return self


class ConversationDataset(Base, ABC):
    """A dataset made of conversations"""

    @abstractmethod
    def __iter__(self) -> Iterator[ConversationTree]:
        """Return an iterator over conversations"""
        ...


class ConversationUserTopics(Topics):
    """Extract user topics from conversations"""

    conversations: Param[ConversationDataset]

    def iter(self) -> Iterator[ConversationEntry]:
        """Returns an iterator over topics"""
        # Extracts topics from conversations, Each user query is a topic (can perform retrieval on it)
        # TODO: merge with xpmir.learning.DatasetConversationBase -> same logic

        records: List[ConversationEntry] = []
        for conversation in self.conversations.__iter__():
            nodes = [
                node
                for node in conversation
                if node.entry["entry_type"] == EntryType.USER_QUERY
            ]
            for node in nodes:
                records.append({**node.entry, "history": node.history()})
        return iter(records)
