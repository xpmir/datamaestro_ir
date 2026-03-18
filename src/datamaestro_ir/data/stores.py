from typing import List
from datamaestro_ir.data.base import (
    IDTextRecord,
    SimpleTextItem,
)
from datamaestro_ir.data import CompressedDocumentStore


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


