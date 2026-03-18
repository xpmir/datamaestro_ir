"""MS MARCO Document Ranking collection (v1).

The MS MARCO document ranking dataset contains ~3.2M web documents for
document retrieval tasks. Used as a base collection in TREC CaST 2021.

See `<https://microsoft.github.io/msmarco/>`_ for more details.
"""

import json
from hashlib import md5
from pathlib import Path

from datamaestro.annotations.agreement import useragreement
from datamaestro.definitions import Dataset, dataset
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck
from datamaestro_ir.data.stores import MsMarcoDocumentStore
from datamaestro_ir.download.docstore import docstore_builder

lua = useragreement(
    """Will begin downloading MS-MARCO document dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx""",
    id="net.windows.msmarco",
)

DOC_COUNT = 3_213_835


@lua
@dataset(
    url="https://microsoft.github.io/msmarco/",
    size="3.2GB",
)
class Documents(Dataset):
    """MS MARCO document collection v1.

    Contains ~3.2M web documents in TREC text format. Each document has an
    ID, URL, title, and body. Used as a base collection in TREC CaST 2021
    (with MARCO_ prefix).
    """

    DOCUMENTS = FileDownloader(
        "msmarco-docs.trec.gz",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.trec.gz",
        checker=HashCheck("d4863e4f342982b51b9a8fc668b2d0c0", md5),
    )

    @staticmethod
    def _reader(source: Path):
        """Read MS MARCO documents in TREC format.

        Format: ``<DOC>\\n<DOCNO>id</DOCNO>\\nurl\\ntitle\\n<BODY>\\nbody\\n</BODY>\\n</DOC>``
        """
        import gzip

        with gzip.open(source, "rt", errors="replace") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                line = line.strip()
                if line == "<DOC>":
                    docno_line = fp.readline().strip()
                    doc_id = docno_line.replace("<DOCNO>", "").replace(
                        "</DOCNO>", ""
                    ).strip()
                    url = fp.readline().strip()
                    title = fp.readline().strip()

                    # Read body until </BODY>
                    body_lines = []
                    for body_line in fp:
                        if body_line.strip() in ("</BODY>", "</DOC>"):
                            break
                        body_lines.append(body_line.rstrip("\n"))

                    body = "\n".join(body_lines).replace("\n", " ").strip()
                    record = {"url": url, "title": title, "body": body}
                    yield {"id": doc_id}, json.dumps(record).encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
        doc_count=DOC_COUNT,
    )

    def config(self) -> MsMarcoDocumentStore:
        return MsMarcoDocumentStore.C(path=self.store.path, count=DOC_COUNT)
