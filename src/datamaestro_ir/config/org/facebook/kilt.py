"""KILT (Knowledge Intensive Language Tasks) knowledge source.

The KILT knowledge source is a snapshot of Wikipedia used in the KILT
benchmark and as a document collection in TREC CaST 2021-2022.

See `<https://github.com/facebookresearch/KILT>`_ for more details.
"""

import json
from pathlib import Path

from datamaestro.definitions import Dataset, dataset
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck
from hashlib import md5

from datamaestro_ir.data.stores import KiltDocumentStore
from datamaestro_ir.download.docstore import docstore_builder


@dataset(
    url="https://github.com/facebookresearch/KILT",
    size="35GB",
)
class Documents(Dataset):
    """KILT knowledge source (Wikipedia snapshot).

    Contains the full KILT knowledge source derived from a Wikipedia dump.
    Each document has a Wikipedia ID, title, URL, and body text.
    Used as a base collection in TREC CaST 2021-2022.
    """

    DOCUMENTS = FileDownloader(
        "kilt_knowledgesource.json",
        url="http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json",
        checker=HashCheck("76e8b27e03be0bd5dead79f071e6f1a3", md5),
    )

    @staticmethod
    def _reader(source: Path):
        """Read KILT JSONL documents.

        Each line is a JSON object with fields: wikipedia_id, wikipedia_title,
        text, anchors, categories, wikidata_id, history.

        For CaST, documents are stored with title, URL, and joined body text.
        """
        with open(source, "rt") as fp:
            for line in fp:
                if not line.strip():
                    continue
                data = json.loads(line)
                doc_id = data["wikipedia_id"]
                title = data.get("wikipedia_title", "")
                body = " ".join(data.get("text", [])).replace("\n", " ").strip()
                url = ""
                if hist := data.get("history"):
                    url = hist.get("url", "")

                record = {"title": title, "url": url, "body": body}
                yield {"id": doc_id}, json.dumps(record).encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
    )

    def config(self) -> KiltDocumentStore:
        return KiltDocumentStore.C(path=self.store.path)
