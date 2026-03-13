"""TREC Conversational Assistance Track (CaST) conversation parser.

Parses CaST topic JSON files (2019-2022) into conversation trees.
"""

import json
import logging
from typing import Iterator, List, Optional

from experimaestro import Param
from datamaestro.data import File

from datamaestro_ir.data.base import SimpleTextItem

from .base import (
    ConversationDataset,
    ConversationEntry,
    ConversationTree,
    ConversationTreeNode,
    DecontextualizedDictItem,
    EntryType,
    SingleConversationTree,
)

logger = logging.getLogger(__name__)


class CastConversations(ConversationDataset, File):
    """A dataset containing TREC CaST conversations (2019-2022).

    Parses the official CaST topic JSON files and produces conversation trees
    compatible with the ConversationUserTopics extractor.

    JSON format: ``[{"number": N, "title": "...", "turn": [{"number": N,
    "raw_utterance": "...", ...}]}]``
    """

    year: Param[int]
    """CaST year (2019, 2020, 2021, or 2022)"""

    def __iter__(self) -> Iterator[ConversationTree]:
        with self.path.open("rt") as fp:
            topics = json.load(fp)

        for topic in topics:
            topic_number = topic["number"]

            if self.year == 2022:
                yield from self._parse_tree(topic)
            else:
                yield self._parse_linear(topic)

    def _parse_linear(self, topic: dict) -> ConversationTree:
        """Parse a linear conversation (2019-2021)."""
        topic_number = topic["number"]
        history: List[ConversationEntry] = []

        for turn in topic["turn"]:
            turn_number = turn["number"]
            query_id = f"{topic_number}_{turn_number}"

            entry: ConversationEntry = {
                "id": query_id,
                "text_item": SimpleTextItem(turn["raw_utterance"]),
                "entry_type": EntryType.USER_QUERY,
            }

            # Decontextualized queries (2020-2021)
            decontext = self._get_decontextualized(turn)
            if decontext is not None:
                entry["decontextualized"] = decontext

            # System answer document ID
            answer_doc_id = self._get_answer_doc_id(turn)
            if answer_doc_id is not None:
                entry["answer_document_id"] = answer_doc_id

            history.append(entry)

            # Add system answer entry if available
            answer = self._get_answer_text(turn)
            if answer is not None:
                history.append(
                    {
                        "answer": answer,
                        "entry_type": EntryType.SYSTEM_ANSWER,
                    }
                )

        history.reverse()
        return SingleConversationTree(str(topic_number), history)

    def _parse_tree(self, topic: dict) -> Iterator[ConversationTree]:
        """Parse a tree-structured conversation (2022).

        The 2022 format has alternating User/System turns forming a tree via
        parent_id links. User turns have ``utterance`` and optionally
        ``manual_rewritten_utterance``. System turns have ``response``
        and ``provenance``.
        """
        topic_number = topic["number"]

        # Build nodes indexed by turn number
        nodes = {}
        for turn in topic["turn"]:
            turn_number = turn["number"]
            query_id = f"{topic_number}_{turn_number}"
            participant = turn.get("participant", "User")

            if participant == "System":
                entry: ConversationEntry = {
                    "id": query_id,
                    "entry_type": EntryType.SYSTEM_ANSWER,
                }
                if response := turn.get("response"):
                    entry["answer"] = response
            else:
                entry = {
                    "id": query_id,
                    "text_item": SimpleTextItem(turn.get("utterance", "")),
                    "entry_type": EntryType.USER_QUERY,
                }
                decontext = self._get_decontextualized(turn)
                if decontext is not None:
                    entry["decontextualized"] = decontext

            nodes[turn_number] = ConversationTreeNode(entry)
            nodes[turn_number]._turn_parent_id = turn.get("parent")

        # Link parent-child relationships
        roots = []
        for turn_number, node in nodes.items():
            parent_id = node._turn_parent_id
            del node._turn_parent_id
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].add(node)
            else:
                roots.append(node)

        yield from roots

    def _get_decontextualized(self, turn: dict) -> Optional[DecontextualizedDictItem]:
        """Extract decontextualized queries from a turn."""
        queries = {}

        if self.year == 2020:
            if v := turn.get("manual_rewritten_utterance"):
                queries["manual"] = v
            if v := turn.get("automatic_rewritten_utterance"):
                queries["auto"] = v
        elif self.year == 2021:
            if v := turn.get("manual_rewritten_utterance"):
                queries["manual"] = v
            if v := turn.get("automatic_rewritten_utterance"):
                queries["auto"] = v
        elif self.year == 2022:
            if v := turn.get("manual_rewritten_utterance"):
                queries["manual"] = v

        if not queries:
            return None
        default_key = "manual" if "manual" in queries else next(iter(queries))
        return DecontextualizedDictItem(
            default_decontextualized_key=default_key,
            decontextualized_queries=queries,
        )

    def _get_answer_doc_id(self, turn: dict) -> Optional[str]:
        """Extract the canonical answer document ID."""
        if self.year == 2020:
            return turn.get("manual_canonical_result_id")
        elif self.year == 2021:
            return turn.get("canonical_result_id")
        return None

    def _get_answer_text(self, turn: dict) -> Optional[str]:
        """Extract the system answer text."""
        if self.year == 2022:
            return turn.get("response")
        return None
