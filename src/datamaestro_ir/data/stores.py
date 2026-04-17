import gzip
import json
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set
from experimaestro import field, Param, Meta
from datamaestro_ir.data.base import (
    IDTextRecord,
    SimpleTextItem,
)
from datamaestro_ir.data import CompressedDocumentStore, DocumentStore
from datamaestro_ir.data.formats import (
    DocumentWithTitle,
    MsMarcoDocument,
    MsMarcoV2Passage,
    TitleUrlDocument,
    WapoDocument,
)


class MsMarcoPassagesStore(CompressedDocumentStore):
    """Document store for MS MARCO passages where internal ID = external ID"""

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        return {
            "id": str(internal_id),
            "text_item": SimpleTextItem(text=content.decode("utf-8")),
        }

    def docid_internal2external(self, docid: int):
        return str(docid)

    def document_ext(self, docid: str) -> IDTextRecord:
        return self.document_int(int(docid))

    def documents_ext(self, docids: List[str]) -> List[IDTextRecord]:
        nums = [int(d) for d in docids]
        docs = self._store.get_by_number(nums)
        return [self.converter(n, d.keys, d.content) for n, d in zip(nums, docs)]


# --- CAR v2.0 paragraphs ---


class CarParagraphStore(CompressedDocumentStore):
    """Document store for TREC CAR v2.0 paragraphs.

    Each document is a simple text paragraph identified by its paragraph ID.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        return {
            "id": keys["id"],
            "text_item": SimpleTextItem(text=content.decode("utf-8")),
        }


# --- WAPO ---


class WapoDocumentStore(CompressedDocumentStore):
    """Document store for Washington Post (WAPO) v2/v4 full documents.

    Stores full WAPO documents with all metadata fields.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        data = json.loads(content)
        return {
            "id": keys["id"],
            "text_item": WapoDocument(
                url=data["url"],
                title=data["title"],
                author=data["author"],
                published_date=data["published_date"],
                kicker=data["kicker"],
                body=data["body"],
                body_paras_html=tuple(data["body_paras_html"]),
                body_media=tuple(),  # Media not stored in compressed form
            ),
        }


class WapoPassageStore(CompressedDocumentStore):
    """Document store for WAPO paragraph-level passages (CaST v0).

    Each WAPO document is split into paragraphs. Document IDs follow the
    format ``{doc_id}-{paragraph_index}`` (1-indexed) matching the official
    CaST tools script.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        return {
            "id": keys["id"],
            "text_item": SimpleTextItem(text=content.decode("utf-8")),
        }


# --- KILT ---


class KiltDocumentStore(CompressedDocumentStore):
    """Document store for KILT (Knowledge Intensive Language Tasks) knowledge source.

    Stores KILT documents with title, URL, and body text.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        data = json.loads(content)
        return {
            "id": keys["id"],
            "text_item": TitleUrlDocument(
                body=data["body"],
                title=data["title"],
                url=data["url"],
            ),
        }


# --- MS MARCO Documents ---


class MsMarcoDocumentStore(CompressedDocumentStore):
    """Document store for MS MARCO document collection (v1).

    Each document has URL, title, and body fields.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        data = json.loads(content)
        return {
            "id": keys["id"],
            "text_item": MsMarcoDocument(
                url=data["url"],
                title=data["title"],
                body=data["body"],
            ),
        }


class MsMarcoPassageV2Store(CompressedDocumentStore):
    """Document store for MS MARCO passage collection v2.

    Each passage has its text, parent document id, and character spans
    within that document.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        import re

        data = json.loads(content)
        raw_spans = data["spans"]
        if isinstance(raw_spans, str):
            spans = tuple(
                (int(a), int(b)) for a, b in re.findall(r"\((\d+),(\d+)\)", raw_spans)
            )
        else:
            spans = tuple(tuple(s) for s in raw_spans)
        return {
            "id": keys["id"],
            "text_item": MsMarcoV2Passage(
                passage=data["passage"],
                msmarco_document_id=data["docid"],
                spans=spans,
            ),
        }


class MsMarcoDocumentV2Store(CompressedDocumentStore):
    """Document store for MS MARCO document collection v2.

    Each document has URL, title, headings, and body fields.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        data = json.loads(content)
        return {
            "id": keys["id"],
            "text_item": MsMarcoDocument(
                url=data["url"],
                title=data["title"],
                body=data["body"],
            ),
        }


# --- TIPSTER ---


class TipsterDocumentStore(CompressedDocumentStore):
    """Document store for TIPSTER/AQUAINT document collections.

    Each document is stored as JSON with title and body fields,
    matching the structured output of the TIPSTER SGML parser.
    """

    lookup_key = "id"

    def converter(
        self, internal_id: int, keys: dict[str, str], content: bytes
    ) -> IDTextRecord:
        data = json.loads(content)
        return {
            "id": keys["id"],
            "text_item": DocumentWithTitle(
                title=data["title"],
                body=data["body"],
            ),
        }


# --- CaST Segmented Passages ---


class CastSegmentedPassageStore(DocumentStore):
    """Document store for CaST segmented passages (v2/v3).

    Reads a base document store and an offset file to create passage-level
    documents. Each passage is defined by character ranges applied to the
    base document text.

    Offset file format (gzipped JSONL)::

        {"id":"MARCO_00_1454834","ranges":[[[0,917]],[[918,2082]]],"md5":"..."}

    Passage IDs follow the format ``{doc_id}-{passage_index}`` (1-indexed).
    """

    base_store: Param[DocumentStore]
    """The base document store containing full documents"""

    offsets_path: Meta[Path]
    """Path to the gzipped JSONL offset file"""

    dupes_path: Meta[Optional[Path]] = field(default=None, ignore_default=True)
    """Path to the duplicates file (one doc ID per line to exclude)"""

    @cached_property
    def _dupes(self) -> Set[str]:
        if self.dupes_path is None:
            return set()
        dupes = set()
        with open(self.dupes_path, "rt") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    dupes.add(line)
        return dupes

    @cached_property
    def _offsets(self) -> Dict[str, list]:
        """Load offset file into a dict: doc_id -> list of range lists."""
        offsets = {}
        with gzip.open(self.offsets_path, "rt") as fp:
            for line in fp:
                data = json.loads(line)
                offsets[data["id"]] = data["ranges"]
        return offsets

    def document_ext(self, docid: str) -> IDTextRecord:
        # Parse passage ID: {base_doc_id}-{passage_index}
        base_id, psg_idx_str = docid.rsplit("-", 1)
        psg_idx = int(psg_idx_str) - 1  # 1-indexed to 0-indexed

        base_doc = self.base_store.document_ext(base_id)
        text_item = base_doc["text_item"]
        body = text_item.text
        title = getattr(text_item, "title", "")
        url = getattr(text_item, "url", "")

        ranges = self._offsets[base_id][psg_idx]
        passage_text = " ".join(body[start:end] for start, end in ranges)

        return {
            "id": docid,
            "text_item": TitleUrlDocument(body=passage_text, title=title, url=url),
        }

    def iter(self) -> Iterator[IDTextRecord]:
        for base_doc in self.base_store.iter():
            base_id = base_doc["id"]

            if base_id in self._dupes:
                continue

            if base_id not in self._offsets:
                continue

            text_item = base_doc["text_item"]
            body = text_item.text
            title = getattr(text_item, "title", "")
            url = getattr(text_item, "url", "")

            for psg_idx, ranges in enumerate(self._offsets[base_id]):
                passage_text = " ".join(body[start:end] for start, end in ranges)
                yield {
                    "id": f"{base_id}-{psg_idx + 1}",
                    "text_item": TitleUrlDocument(
                        body=passage_text, title=title, url=url
                    ),
                }
