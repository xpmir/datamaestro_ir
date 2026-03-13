import bz2
import gzip
import json
from functools import cached_property
from hashlib import md5, sha256
import logging
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set
from datamaestro_ir.utils.files import TQDMFileReader
from experimaestro import Constant, Param, Meta
from datamaestro_ir.data.base import (
    IDTextRecord,
    SimpleTextItem,
)
from datamaestro_ir.data import CompressedDocumentStore, DocumentStore
from datamaestro_ir.datasets.irds.data import LZ4DocumentStore
from datamaestro_ir.data.formats import (
    MsMarcoDocument,
    OrConvQADocument,
    TitleUrlDocument,
    WapoDocument,
)
from tqdm import tqdm


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
        return [
            self.converter(n, d.keys, d.content) for n, d in zip(nums, docs)
        ]


class OrConvQADocumentStore(LZ4DocumentStore):
    class NAMED_TUPLE(NamedTuple):
        id: str
        title: str
        body: str
        aid: str
        bid: int

    lookup_field: Constant[str] = "id"
    fields: Constant[List[str]] = list(NAMED_TUPLE._fields)
    index_fields: Constant[List[str]] = ["id"]

    data_cls = NAMED_TUPLE

    def converter(self, data: NAMED_TUPLE) -> IDTextRecord:
        fields = data._asdict()
        del fields["id"]
        return {"id": data.id, "text_item": OrConvQADocument(**fields)}


class IKatClueWeb22DocumentStore(LZ4DocumentStore):
    @staticmethod
    def generator(path: Path, checksums_file: Path, passages_hashes: Path):
        """Returns an iterator over iKAT 2022-25 documents

        :param path: The folder containing the files
        """

        def __iter__():
            errors = False

            assert checksums_file.is_file(), f"{checksums_file} does not exist"
            assert passages_hashes.is_file(), f"{passages_hashes} does not exist"

            # Get the list of files
            with checksums_file.open("rt") as fp:
                files = []
                for line in fp:
                    checksum, filename = line.strip().split()
                    files.append((checksum, filename))
                    if not (path / filename).is_file():
                        logging.error("File %s does not exist", path / filename)
                        errors = True

            assert not errors, "Errors detected, stopping"

            # Check the SHA256 sums
            match checksums_file.suffix:
                case ".sha256sums":
                    hasher_factory = sha256
                case _:
                    raise NotImplementedError(
                        f"Cannot handle {checksums_file.suffix} checksum files"
                    )

            for checksum, filename in tqdm(files):
                logging.info("Checking %s", filename)
                hasher = hasher_factory()
                with (path / filename).open("rb") as fp:
                    while data := fp.read(2**20):
                        hasher.update(data)

                file_checksum = hasher.hexdigest()
                assert file_checksum == checksum, (
                    f"Expected {checksum}, got {file_checksum} for {filename}"
                )

            # Get the MD5 hashes of all the passages
            logging.info("Reading the hashes of all passages")
            with TQDMFileReader(passages_hashes, "rt", bz2.open) as fp:
                passage_checksums = {}
                for line in tqdm(fp):
                    doc_id, passage_no, checksum = line.strip().split()
                    passage_checksums[f"{doc_id}:{passage_no}"] = checksum  # noqa: E231

            # Read the files
            logging.info("Starting to read the files")
            for _, filename in tqdm(files):
                with TQDMFileReader(path / filename, "rt", bz2.open) as jsonl_fp:
                    for line in jsonl_fp:
                        data = json.loads(line)
                        expected = passage_checksums[data["id"]]
                        computed = md5(data["contents"].encode("utf-8")).hexdigest()
                        assert expected == computed, (
                            f"Expected {expected}, "
                            f"got {computed} for passage {data['id']} in {filename}"
                        )
                        yield IKatClueWeb22DocumentStore.Document(**data)

        return __iter__

    class Document(NamedTuple):
        id: str
        contents: str
        url: str

    data_cls = Document
    lookup_field: Constant[str] = "id"
    index_fields: Constant[List[str]] = ["id"]

    def converter(self, data):
        return {
            "id": data.id,
            "text_item": SimpleTextItem(data.contents),
            "url": data.url,
        }


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

    Stores full WAPO documents with all metadata fields matching the irds
    WapoDoc handler output.
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

    Each document has URL, title, and body fields matching the irds handler.
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

    dupes_path: Meta[Optional[Path]] = None
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
