"""Data types for BEIR benchmark datasets."""

import json
from pathlib import Path
from typing import Iterator, List

from datamaestro.definitions import Param, Meta
from experimaestro import field

from datamaestro_ir.data import (
    AdhocAssessments,
    CompressedDocumentStore,
    Topics,
)
from datamaestro_ir.data.base import (
    AdhocAssessedTopic,
    IDTextRecord,
    SimpleAdhocAssessment,
    SimpleTextItem,
)
from datamaestro_ir.data.formats import TitleDocument


class BeirDocumentStore(CompressedDocumentStore):
    """Document store for BEIR datasets.

    Content bytes encode title and text as: title_bytes + b"\\0" + text_bytes.
    The only key is "id" (the external document ID).
    """

    lookup_key: Param[str] = field(default="id", ignore_default=True)

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        text = content.decode("utf-8")
        sep = text.index("\0")
        title = text[:sep]
        body = text[sep + 1 :]
        return {
            "id": keys["id"],
            "text_item": TitleDocument(title=title, body=body),
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


class BeirAssessments(AdhocAssessments):
    """BEIR qrels: 3-column TSV with header (query-id, corpus-id, score)."""

    path: Meta[Path]

    def iter(self) -> Iterator[AdhocAssessedTopic]:
        from collections import defaultdict

        assessments = defaultdict(list)
        with open(self.path, "rt") as fp:
            next(fp)  # skip header
            for line in fp:
                parts = line.strip().split("\t")
                qid, doc_id, score = parts[0], parts[1], int(parts[2])
                assessments[qid].append(SimpleAdhocAssessment(doc_id=doc_id, rel=score))

        for qid, docs in assessments.items():
            yield AdhocAssessedTopic(topic_id=qid, assessments=docs)


class BeirTopics(Topics):
    """BEIR queries: JSONL with _id and text fields."""

    path: Meta[Path]

    def iter(self) -> Iterator[IDTextRecord]:
        with open(self.path, "rt") as fp:
            for line in fp:
                data = json.loads(line)
                yield {
                    "id": data["_id"],
                    "text_item": SimpleTextItem(text=data["text"]),
                }


class BeirParquetTopics(Topics):
    """BEIR queries from Parquet."""

    path: Meta[Path]

    def iter(self) -> Iterator[IDTextRecord]:
        import pandas as pd

        df = pd.read_parquet(self.path)
        for _, row in df.iterrows():
            yield {
                "id": str(row["_id"]),
                "text_item": SimpleTextItem(text=row["text"]),
            }


class BeirParquetAssessments(AdhocAssessments):
    """BEIR qrels from Parquet."""

    path: Meta[Path]

    def iter(self) -> Iterator[AdhocAssessedTopic]:
        import pandas as pd
        from collections import defaultdict

        df = pd.read_parquet(self.path)
        assessments = defaultdict(list)
        for _, row in df.iterrows():
            qid, doc_id, score = (
                str(row["query-id"]),
                str(row["corpus-id"]),
                int(row.get("score", 1.0)),
            )
            assessments[qid].append(SimpleAdhocAssessment(doc_id=doc_id, rel=score))

        for qid, docs in assessments.items():
            yield AdhocAssessedTopic(topic_id=qid, assessments=docs)


def beir_parquet_docstore_iter(path: Path) -> Iterator[tuple[dict[str, str], bytes]]:
    import pandas as pd

    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        # Title and text separated by \0
        title = row.get("title", "")
        text = row["text"]
        content = (title + "\0" + text).encode("utf-8")
        yield {"id": str(row["_id"])}, content
