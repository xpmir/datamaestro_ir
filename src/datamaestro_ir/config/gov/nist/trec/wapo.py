"""Washington Post (WAPO) document collections.

The Washington Post provides document collections used in several TREC tracks.
These collections require a data use agreement with NIST and must be provided
locally via ``DatafolderPath``.

See `<https://trec.nist.gov/data/wapost/>`_ for more details.
"""

import itertools
import json
import logging
import re
from pathlib import Path

from datamaestro.context import DatafolderPath
from datamaestro.definitions import Dataset, dataset
from datamaestro.download.links import linkfolder
from datamaestro_ir.data.stores import KiltDocumentStore, WapoDocumentStore, WapoPassageStore
from datamaestro_ir.download.docstore import docstore_builder

logger = logging.getLogger(__name__)

CLEANR = re.compile(r"<.*?>")


def _wapo_raw_iter(source: Path):
    """Iterate over raw WAPO JSON lines documents.

    Handles both .jl and .jl within tar archives.
    """
    for jl_path in sorted(source.glob("**/*.jl")) + sorted(
        source.glob("**/*.jsonl")
    ):
        with open(jl_path, "rt") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)


# --- WAPO v2 ---


WAPO_V2_COUNT = 608_180


@dataset(
    url="https://trec.nist.gov/data/wapost/",
)
class WapoV2Documents(Dataset):
    """Washington Post v2 document collection.

    Contains ~608K news articles from the Washington Post. Requires a NIST
    data use agreement. Point ``DatafolderPath`` to the directory containing
    the WAPO v2 JSON lines file.
    """

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.wapo.v2", ".")],
    )

    @staticmethod
    def _reader(source: Path):
        """Read WAPO v2 documents and store as JSON."""
        for data in _wapo_raw_iter(source):
            doc_id = str(data["id"])
            title = data.get("title", "") or ""
            author = data.get("author", "") or ""
            published_date = data.get("published_date", 0) or 0

            url = data.get("article_url", "") or ""
            if url and "www.washingtonpost.com" not in url:
                url = "https://www.washingtonpost.com" + url

            kicker = ""
            body = ""
            body_paras_html = []
            if data.get("contents"):
                for item in data["contents"]:
                    if item is None:
                        continue
                    if item.get("type") == "kicker":
                        kicker = item.get("content", "")
                    elif item.get("subtype") == "paragraph":
                        content = item.get("content", "")
                        if content:
                            body_paras_html.append(content)
                            body += " " + re.sub(CLEANR, "", content)

            body = body.replace("\n", " ").strip()

            record = {
                "url": url,
                "title": title,
                "author": author,
                "published_date": published_date,
                "kicker": kicker,
                "body": body,
                "body_paras_html": body_paras_html,
            }
            yield {"id": doc_id}, json.dumps(record).encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
        doc_count=WAPO_V2_COUNT,
    )

    def config(self) -> WapoDocumentStore:
        return WapoDocumentStore.C(path=self.store.path, count=WAPO_V2_COUNT)


@dataset(
    url="https://trec.nist.gov/data/wapost/",
)
class WapoV2Passages(Dataset):
    """Washington Post v2 paragraph-level passages for CaST v0.

    Each WAPO document is split into paragraphs following the official CaST
    tools script. Paragraph IDs follow the format ``{doc_id}-{index}``
    (1-indexed). Empty paragraphs are skipped.
    """

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.wapo.v2", ".")],
    )

    @staticmethod
    def _reader(source: Path):
        """Read WAPO v2 documents and split into paragraphs."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 and lxml are required for WAPO passage extraction. "
                "Install with: pip install beautifulsoup4 lxml"
            )

        for data in _wapo_raw_iter(source):
            doc_id = str(data["id"])
            pid = itertools.count(1)

            if not data.get("contents"):
                continue

            for item in data["contents"]:
                if (
                    item is not None
                    and item.get("subtype") == "paragraph"
                    and item.get("content", "") != ""
                ):
                    text = item["content"]
                    if item.get("mime") == "text/html":
                        text = BeautifulSoup(
                            f"<OUTER>{text}</OUTER>", "lxml-xml"
                        ).get_text()
                    passage_id = f"{doc_id}-{next(pid)}"
                    yield {"id": passage_id}, text.encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
    )

    def config(self) -> WapoPassageStore:
        return WapoPassageStore.C(path=self.store.path)


# --- WAPO v4 ---


@dataset(
    url="https://trec.nist.gov/data/wapost/",
)
class WapoV4Documents(Dataset):
    """Washington Post v4 document collection.

    Contains news articles from the Washington Post (v4 release). Used in
    CaST 2021 and 2022. Requires a NIST data use agreement.
    """

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.wapo.v4", ".")],
    )

    @staticmethod
    def _reader(source: Path):
        """Read WAPO v4 documents with CaST-style processing.

        Produces one document per article with body text from paragraphs,
        HTML stripped. Matches the CaST WapoV4Docs handler behavior.
        """
        seen = set()
        for data in _wapo_raw_iter(source):
            doc_id = str(data["id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)

            title = data.get("title", "No Title") or "No Title"
            url = data.get("article_url", "") or ""
            if url and "www.washingtonpost.com" not in url:
                url = "https://www.washingtonpost.com" + url

            body = ""
            if data.get("contents"):
                for item in data["contents"]:
                    if (
                        item is not None
                        and item.get("subtype") == "paragraph"
                    ):
                        body += " " + item.get("content", "")
            body = re.sub(CLEANR, "", body).replace("\n", " ").strip()

            if not body:
                continue

            record = {"title": title, "url": url, "body": body}
            yield {"id": doc_id}, json.dumps(record).encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
    )

    def config(self) -> KiltDocumentStore:
        return KiltDocumentStore.C(path=self.store.path)
