from pathlib import Path

from experimaestro import Param, Meta, field
import datamaestro_ir.data as ir
from datamaestro_ir.data.base import SimpleTextItem
from datamaestro_ir.interfaces.plaintext import read_tsv


class AdhocRunWithText(ir.AdhocRun):
    "(qid, doc.id, query, passage)"

    path: Meta[Path]
    separator: Meta[str] = field(default="\t", ignore_default=True)


class Topics(ir.Topics):
    "Pairs of query id - query using a separator"

    path: Meta[Path]
    separator: Meta[str] = field(default="\t", ignore_default=True)

    def iter(self):
        return (
            {"id": qid, "text_item": SimpleTextItem(title)}
            for qid, title in read_tsv(self.path)
        )


class Documents(ir.Documents):
    "One line per document, format pid<SEP>text"

    path: Param[Path]
    separator: Meta[str] = field(default="\t", ignore_default=True)

    pass
