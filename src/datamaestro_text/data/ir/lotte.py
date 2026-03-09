"""Data types for LoTTE (Long-Tail Topic-stratified Evaluation) datasets."""

import json
from pathlib import Path
from typing import Iterator, List

from datamaestro.definitions import Meta, Param

from datamaestro_text.data.ir import (
    AdhocAssessments,
    CompressedDocumentStore,
    Topics,
)
from datamaestro_text.data.ir.base import (
    AdhocAssessedTopic,
    IDTextRecord,
    SimpleAdhocAssessment,
    SimpleTextItem,
)


class LotteDocumentStore(CompressedDocumentStore):
    """Document store for LoTTE datasets.

    Content bytes encode text as UTF-8. The only key is "id" (the document ID).
    """

    lookup_key: Param[str] = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        return {
            "id": keys["id"],
            "text_item": SimpleTextItem(text=content.decode("utf-8")),
        }

    def docid_internal2external(self, docid: int):
        docs = self._store.get_by_number([docid])
        return docs[0].keys["id"]

    def document_ext(self, docid: str) -> IDTextRecord:
        docs = self._store.get_by_key("id", [docid])
        if docs[0] is None:
            raise KeyError(f"Document {docid} not found")
        d = docs[0]
        return self.converter(d.internal_id, d.keys, d.content)

    def documents_ext(self, docids: List[str]) -> List[IDTextRecord]:
        docs = self._store.get_by_key("id", docids)
        return [
            self.converter(d.internal_id, d.keys, d.content) if d is not None else None
            for d in docs
        ]


class LotteAssessments(AdhocAssessments):
    """LoTTE qrels: JSONL with qid and answer_pids fields."""

    path: Meta[Path]

    def iter(self) -> Iterator[AdhocAssessedTopic]:
        with open(self.path, "rt") as fp:
            for line in fp:
                data = json.loads(line)
                qid = str(data["qid"])
                assessments = [
                    SimpleAdhocAssessment(doc_id=str(pid), rel=1)
                    for pid in data["answer_pids"]
                ]
                if assessments:
                    yield AdhocAssessedTopic(topic_id=qid, assessments=assessments)


class LotteTopics(Topics):
    """LoTTE queries: TSV with query_id and text fields."""

    path: Meta[Path]

    def iter(self) -> Iterator[IDTextRecord]:
        with open(self.path, "rt") as fp:
            for line in fp:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    yield {
                        "id": parts[0],
                        "text_item": SimpleTextItem(text=parts[1]),
                    }
