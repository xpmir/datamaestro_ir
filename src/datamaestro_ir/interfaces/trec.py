from collections import deque
from pathlib import Path
from typing import Iterator, List, Optional
import re
from datamaestro_ir.data import AdhocRunDict
from datamaestro_ir.data.base import (
    AdhocAssessedTopic,
    IDTextRecord,
    SimpleAdhocAssessment,
)
from datamaestro_ir.data.formats import DocumentWithTitle
from datamaestro_ir.data.formats import TrecTopic

# --- Runs


def parse_run(path: Path) -> AdhocRunDict:
    results = {}
    with path.open("rt") as f:
        for line in f:
            query_id, _q0, doc_id, _rank, score, _model_id = re.split(
                r"\s+", line.strip()
            )
            results.setdefault(query_id, {})[doc_id] = score

    return results


def write_run_dict(run: AdhocRunDict, run_path: Path):
    """Write run dict"""
    with run_path.open("wt") as f:
        for query_id, scored_documents in run.items():
            scored_documents = list(
                [(doc_id, score) for doc_id, score in scored_documents.items()]
            )
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            for ix, (doc_id, score) in enumerate(scored_documents):
                f.write(f"{query_id} Q0 {doc_id} {ix + 1} {score} run\n")


# --- Assessments


def parse_qrels(path: Path) -> Iterator[AdhocAssessedTopic]:
    with path.open("rt") as fp:
        _qid = None
        assessments = []

        for line in fp:
            qid, _, docno, rel = re.split(r"\s+", line.strip())
            if qid != _qid:
                if _qid is not None:
                    yield AdhocAssessedTopic(_qid, assessments)
                _qid = qid
                assessments = []
            assessments.append(SimpleAdhocAssessment(docno, int(rel)))

        yield AdhocAssessedTopic(_qid, assessments)


# ---- TOPICS


def cleanup(s: Optional[str]) -> str:
    return s.replace("\t", " ").strip() if s is not None else ""


def parse_query_format(file, xml_prefix=None) -> Iterator[IDTextRecord]:
    """Parse TREC XML query format"""
    if xml_prefix is None:
        xml_prefix = ""

    if hasattr(file, "read"):
        num, title, desc, narr, reading = None, None, None, None, None
        for line in file:
            if line.startswith("**"):
                # translation comment in older formats (e.g., TREC 3 Spanish track)
                continue
            elif line.startswith("</top>"):
                if num:
                    yield {
                        "id": num,
                        "text_item": TrecTopic(
                            cleanup(title), cleanup(desc), cleanup(narr)
                        ),
                    }
                num, title, desc, narr, reading = None, None, None, None, None
            elif line.startswith("<num>"):
                num = line[len("<num>") :].replace("Number:", "").strip()
                reading = None
            elif line.startswith(f"<{xml_prefix}title>"):
                title = line[len(f"<{xml_prefix}title>") :].strip()
                if title == "":
                    reading = "title"
                else:
                    reading = None
            elif line.startswith(f"<{xml_prefix}desc>"):
                desc = ""
                reading = "desc"
            elif line.startswith(f"<{xml_prefix}narr>"):
                narr = ""
                reading = "narr"
            elif reading == "desc":
                desc += line.strip() + " "
            elif reading == "narr":
                narr += line.strip() + " "
            elif reading == "title":
                title += line.strip() + " "
    else:
        with open(file, "rt") as f:
            yield from parse_query_format(f)


# --- TIPSTER Documents

# Title/headline tags (output first, matching ir_datasets field order)
_TITLE_TAGS = {"headline", "title", "h3", "h4"}

# Content tags from Anserini/ir_datasets
_CONTENT_TAGS_STR = "TEXT HEADLINE TITLE HL HEAD TTL DD DATE LP LEADPARA"
_BODY_TAGS = {c.lower() for c in _CONTENT_TAGS_STR.split()} - _TITLE_TAGS

# Field definitions for the SAX extractor (matches ir_datasets disks45)
_FIELD_DEFS = [
    {"docno"},
    _TITLE_TAGS,
    _BODY_TAGS,
]


class _SaxExtractor:
    """SAX-style target for lxml HTMLParser that extracts fields from
    TIPSTER SGML documents.  Matches ir_datasets' SaxExtractor behaviour."""

    IGNORE_TAGS = {"noscript", "meta", "input", "script", "style"}

    def __init__(self):
        self.field_values: list[list[str]] = [[] for _ in _FIELD_DEFS]
        self.field_stacks = [deque() if f is not None else None for f in _FIELD_DEFS]
        self.ignore_tag_stack: deque = deque()

    def _join_text(self, text_parts: list[str]) -> str:
        res = "".join(text_parts)
        res = res.replace("\r\n", "\n").replace("\r", "\n")
        res = res.replace("\t", " ")
        res = re.sub(r"\n +", "\n", res)
        res = re.sub(r" +\n", "\n", res)
        res = re.sub(r"\n{2,}", "\n", res)
        res = re.sub(r" {2,}", " ", res)
        return res.strip()

    def get_values(self) -> tuple[str, str, str]:
        return tuple(self._join_text(v) for v in self.field_values)

    def data(self, data):
        if not self.ignore_tag_stack:
            any_match = False
            for vals, stack in zip(self.field_values, self.field_stacks):
                if (stack is None and not any_match) or stack:
                    vals.append(data)
                    any_match = True

    def start(self, tag, attrs):
        tag = tag.lower()
        for tags, stack in zip(_FIELD_DEFS, self.field_stacks):
            if tags is not None and tag in tags:
                stack.append(tag)
        if tag in self.IGNORE_TAGS:
            self.ignore_tag_stack.append(tag)

    def end(self, tag):
        tag = tag.lower()
        for stack in self.field_stacks:
            if stack and stack[-1] == tag:
                stack.pop()
        if self.ignore_tag_stack and self.ignore_tag_stack[-1] == tag:
            self.ignore_tag_stack.pop()

    def close(self):
        pass

    def comment(self, data):
        pass

    def doctype(self, *args):
        pass

    def pi(self, *args):
        pass


def parse_tipster_file(path: Path) -> Iterator[IDTextRecord]:
    """Parse a single TIPSTER SGML file, yielding (id, text) records.

    Uses lxml's HTMLParser (same as ir_datasets) to handle HTML entities
    and produce text matching ir_datasets' default_text() output.
    Title/headline content is placed before body content.

    Files may be plain text or gzip-compressed.
    """
    from lxml.html import etree
    from datamaestro_ir.utils.files import auto_open

    with auto_open(path, "rb") as fp:
        content = fp.read()

    for raw_doc in content.split(b"\n</DOC>"):
        raw_doc = raw_doc.strip()
        if not raw_doc:
            continue
        raw_doc += b"\n</DOC>"

        sax = _SaxExtractor()
        parser = etree.HTMLParser(target=sax)
        parser.feed(raw_doc.decode("utf-8", errors="replace"))
        parser.close()

        doc_id, title, body = sax.get_values()
        if not doc_id:
            continue

        yield {
            "id": doc_id,
            "text_item": DocumentWithTitle(title=title, body=body),
        }


def iter_tipster_collection(
    base_path: Path, patterns: List[str]
) -> Iterator[IDTextRecord]:
    """Iterate over all documents in a TIPSTER collection.

    Files are collected from all glob patterns and sorted by path
    for deterministic ordering. Uses glob.glob (not Path.glob) so
    that recursive ** patterns follow symlinks.
    """
    from glob import glob as fnglob

    files: set[Path] = set()
    for pattern in patterns:
        glob_path = str(base_path / pattern)
        for p in fnglob(glob_path, recursive=True):
            p = Path(p)
            if p.is_file():
                files.add(p)

    for path in sorted(files):
        yield from parse_tipster_file(path)
