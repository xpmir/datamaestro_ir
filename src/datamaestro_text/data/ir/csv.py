from pathlib import Path

from experimaestro import Param, Meta
import datamaestro_text.data.ir as ir
from datamaestro_text.data.ir.base import SimpleTextItem
from datamaestro_text.interfaces.plaintext import read_tsv


class AdhocRunWithText(ir.AdhocRun):
    "(qid, doc.id, query, passage)"

    path: Meta[Path]
    separator: Meta[str] = "\t"


class Topics(ir.Topics):
    "Pairs of query id - query using a separator"

    path: Meta[Path]
    separator: Meta[str] = "\t"

    def iter(self):
        return (
            {"id": qid, "text_item": SimpleTextItem(title)}
            for qid, title in read_tsv(self.path)
        )


class Documents(ir.Documents):
    "One line per document, format pid<SEP>text"

    path: Param[Path]
    separator: Meta[str] = "\t"

    pass
