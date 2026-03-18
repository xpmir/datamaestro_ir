"""TREC Complex Answer Retrieval (CAR) v2.0 paragraph corpus.

The CAR paragraph corpus contains ~29.8M paragraphs extracted from Wikipedia,
used as a document collection in several TREC tracks including CaST.

See `<http://trec-car.cs.unh.edu/datareleases/>`_ for more details.
"""

from hashlib import md5
from pathlib import Path

from datamaestro.definitions import Dataset, dataset
from datamaestro.download.archive import TarDownloader
from datamaestro.utils import HashCheck
from datamaestro_ir.data.stores import CarParagraphStore
from datamaestro_ir.download.docstore import docstore_builder


DOC_COUNT = 29_794_697


@dataset(
    url="http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz",
    size="7.1GB",
)
class Documents(Dataset):
    """TREC CAR v2.0 paragraph corpus.

    Contains ~29.8M paragraphs from Wikipedia. Each paragraph is a simple
    text document identified by a unique paragraph ID.

    Requires the ``trec-car-tools`` library (``pip install trec-car-tools``).
    """

    DOCUMENTS = TarDownloader(
        "documents",
        url="http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz",
        checker=HashCheck("a404e9e1bffd8b2e91bc02b2c9993cd8", md5),
        transient=True,
    )

    @staticmethod
    def _reader(source: Path):
        """Read CAR v2.0 CBOR paragraphs."""
        from trec_car.read_data import iter_paragraphs

        cbor_path = source / "paragraphCorpus" / "dedup.articles-paragraphs.cbor"
        with open(cbor_path, "rb") as fp:
            for para in iter_paragraphs(fp):
                text = para.get_text()
                yield {"id": para.para_id}, text.encode("utf-8")

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=["id"],
        doc_count=DOC_COUNT,
    )

    def config(self) -> CarParagraphStore:
        return CarParagraphStore.C(path=self.store.path, count=DOC_COUNT)
