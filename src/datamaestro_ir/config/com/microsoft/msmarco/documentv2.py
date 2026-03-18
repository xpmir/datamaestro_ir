"""MS MARCO Document Ranking collection v2.

The MS MARCO v2 document collection contains ~12M web documents distributed
as multiple compressed JSON files in a tar archive. Used as a base collection
in TREC CaST 2022.

See `<https://microsoft.github.io/msmarco/TREC-Deep-Learning.html>`_ for more details.
"""

import gzip
import json
from hashlib import md5
from pathlib import Path

from datamaestro.annotations.agreement import useragreement
from datamaestro.definitions import Dataset, dataset
from datamaestro.download.archive import TarDownloader
from datamaestro.utils import HashCheck
from datamaestro_ir.data.stores import MsMarcoDocumentV2Store
from datamaestro_ir.download.docstore import docstore_builder

lua = useragreement(
    """Will begin downloading MS-MARCO v2 document dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx""",
    id="net.windows.msmarco",
)

DOC_COUNT = 11_959_635


@lua
@dataset(
    url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
    size="32GB",
)
class Documents(Dataset):
    """MS MARCO document collection v2.

    Contains ~12M web documents in multiple gzipped JSON files. Each document
    has an ID (``msmarco_doc_XX_NNNNN``), URL, title, headings, and body.
    Used as a base collection in TREC CaST 2022 (with MARCO_ prefix).

    Note: For CaST, the ``msmarco_doc_`` prefix is stripped from document IDs.
    """

    DOCUMENTS = TarDownloader(
        "documents",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_doc.tar",
        checker=HashCheck("398e3e7535bf0e3e0f tried25a8e52c8c0a", md5),
        transient=True,
    )

    @staticmethod
    def _reader(source: Path):
        """Read MS MARCO v2 documents from multiple GZ files."""
        for gz_path in sorted(source.rglob("*.gz")):
            with gzip.open(gz_path, "rt") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    doc_id = data["docid"]
                    body = data.get("body", "").replace("\n", " ").strip()
                    record = {
                        "url": data.get("url", ""),
                        "title": data.get("title", ""),
                        "body": body,
                    }
                    yield {"id": doc_id}, json.dumps(record).encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
        doc_count=DOC_COUNT,
    )

    def config(self) -> MsMarcoDocumentV2Store:
        return MsMarcoDocumentV2Store.C(path=self.store.path, count=DOC_COUNT)
