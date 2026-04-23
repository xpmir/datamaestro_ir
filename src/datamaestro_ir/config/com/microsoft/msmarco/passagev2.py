"""MS MARCO Passage Ranking collection v2.

The MS MARCO v2 passage collection contains ~138M passages derived
from the MS MARCO v2 documents. The source is a ~21GB tar of gzipped
JSON-Lines files and is streamed directly into the document store
rather than persisted to disk.

See `<https://microsoft.github.io/msmarco/TREC-Deep-Learning.html>`_
for more details.
"""

import json
from typing import Callable, IO, Iterator

from datamaestro.annotations.agreement import useragreement
from datamaestro.definitions import Dataset, dataset
from datamaestro.download import reference
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck
from hashlib import md5

from datamaestro_ir.data import Adhoc, RerankAdhoc
from datamaestro_ir.data.csv import Topics
from datamaestro_ir.data.stores import MsMarcoPassageV2Store
from datamaestro_ir.data.trec import TrecAdhocAssessments, TrecAdhocRun
from datamaestro_ir.download.docstore import streaming_docstore_builder
from datamaestro_ir.utils.streaming import iter_tar_gz_jsonl


lua = useragreement(
    """Will begin downloading MS-MARCO v2 passage dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx""",
    id="net.windows.msmarco",
)

DOC_COUNT = 138_364_198

# --- Document collection


def _iter_v2_passage_tar(
    stream: IO[bytes],
    mark: Callable[[int], None],
) -> Iterator[tuple[dict[str, str], bytes]]:
    """Parse a streaming MS MARCO v2 passage tarball.

    The archive holds 70 gzipped JSON-Lines shards; each line has
    ``pid``, ``passage``, ``spans`` and ``docid``.
    """
    for data in iter_tar_gz_jsonl(stream, mark):
        content = json.dumps(
            {
                "passage": data["passage"],
                "docid": data["docid"],
                "spans": data["spans"],
            }
        ).encode("utf-8")
        yield {"id": data["pid"]}, content


@lua
@dataset(
    url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
    size="21GB",
)
class Passages(Dataset):
    """MS MARCO passage collection v2.

    Contains ~138M passages derived from the MS MARCO v2 document
    collection. Each passage has an id (``msmarco_passage_BB_PPP``),
    text, parent document id, and character spans within the parent
    document.

    The tar archive is streamed from the source URL directly into the
    impact-index DocumentStore, with periodic checkpoints so an
    interrupted build can resume without re-processing already ingested
    passages.
    """

    store = streaming_docstore_builder(
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar",
        stream_factory=_iter_v2_passage_tar,
        keys=["id"],
        headers={"X-Ms-Version": "2019-12-12"},
        checkpoint_frequency=5_000_000,
        doc_count=DOC_COUNT,
    )

    def config(self) -> MsMarcoPassageV2Store:
        return MsMarcoPassageV2Store.C(path=self.store.path, count=DOC_COUNT)


# --- Train


@lua
@dataset()
class TrainQueries(Dataset):
    """MS MARCO v2 passage train queries"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv",
        checker=HashCheck("1835f44e6792c51aa98eed722a8dcc11", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class TrainQrels(Dataset):
    """MS MARCO v2 passage train qrels"""

    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv",
        checker=HashCheck("a2e37e9a9c7ca13d6e38be0512a52017", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class TrainRun(Dataset):
    """MS MARCO v2 passage train top-100 run (TREC format)"""

    RUN = FileDownloader(
        "top100.txt",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_top100.txt.gz",
        checker=HashCheck("7cd731ed984fccb2396f11a284cea800", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class Train(Dataset):
    """MS MARCO v2 passage train dataset"""

    COLLECTION = reference(Passages)
    TOPICS = reference(TrainQueries)
    QRELS = reference(TrainQrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class TrainWithrun(Dataset):
    """MS MARCO v2 passage train dataset with top-100 run"""

    TRAIN = reference(Train)
    RUN = reference(TrainRun)

    def config(self) -> RerankAdhoc:
        train = self.TRAIN.config()
        return RerankAdhoc.C(**train.__arguments__(), run=self.RUN.config())


# --- Dev 1


@lua
@dataset()
class Dev1Queries(Dataset):
    """MS MARCO v2 passage dev1 queries"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv",
        checker=HashCheck("0fa4c6d64a653142ade9fc61d7484239", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Dev1Qrels(Dataset):
    """MS MARCO v2 passage dev1 qrels"""

    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv",
        checker=HashCheck("10f9263260d206d8fb8f13864aea123a", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class Dev1Run(Dataset):
    """MS MARCO v2 passage dev1 top-100 run (TREC format)"""

    RUN = FileDownloader(
        "top100.txt",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_top100.txt.gz",
        checker=HashCheck("fee817a3ee273be8623379e5d3108c0b", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class Dev1(Dataset):
    """MS MARCO v2 passage dev1 dataset"""

    COLLECTION = reference(Passages)
    TOPICS = reference(Dev1Queries)
    QRELS = reference(Dev1Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class Dev1Withrun(Dataset):
    """MS MARCO v2 passage dev1 dataset with top-100 run"""

    DEV1 = reference(Dev1)
    RUN = reference(Dev1Run)

    def config(self) -> RerankAdhoc:
        dev1 = self.DEV1.config()
        return RerankAdhoc.C(**dev1.__arguments__(), run=self.RUN.config())


# --- Dev 2


@lua
@dataset()
class Dev2Queries(Dataset):
    """MS MARCO v2 passage dev2 queries"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv",
        checker=HashCheck("565b84dfa7ccd2f4251fa2debea5947a", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Dev2Qrels(Dataset):
    """MS MARCO v2 passage dev2 qrels"""

    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv",
        checker=HashCheck("8ed8577fa459d34b59cf69b4daa2baeb", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class Dev2Run(Dataset):
    """MS MARCO v2 passage dev2 top-100 run (TREC format)"""

    RUN = FileDownloader(
        "top100.txt",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_top100.txt.gz",
        checker=HashCheck("da532bf26169a3a2074fae774471cc9f", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class Dev2(Dataset):
    """MS MARCO v2 passage dev2 dataset"""

    COLLECTION = reference(Passages)
    TOPICS = reference(Dev2Queries)
    QRELS = reference(Dev2Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html")
class Dev2Withrun(Dataset):
    """MS MARCO v2 passage dev2 dataset with top-100 run"""

    DEV2 = reference(Dev2)
    RUN = reference(Dev2Run)

    def config(self) -> RerankAdhoc:
        dev2 = self.DEV2.config()
        return RerankAdhoc.C(**dev2.__arguments__(), run=self.RUN.config())


# --- TREC DL 2021


@lua
@dataset()
class Trec2021Queries(Dataset):
    """TREC Deep Learning 2021 passage queries (v2)"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv",
        checker=HashCheck("46d863434dda18300f5af33ee29c4b28", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Trec2021Qrels(Dataset):
    """TREC Deep Learning 2021 passage qrels (v2)"""

    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt",
        checker=HashCheck("c5b76ec95b589732edc9040302e22a2b", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class Trec2021Run(Dataset):
    """TREC Deep Learning 2021 passage top-100 run (v2)"""

    RUN = FileDownloader(
        "top100.trec",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_passage_top100.txt.gz",
        checker=HashCheck("e2be2d307da26d1a3f76eb95507672a3", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021.html")
class Trec2021(Dataset):
    """TREC Deep Learning 2021 (passage v2)"""

    COLLECTION = reference(Passages)
    TOPICS = reference(Trec2021Queries)
    QRELS = reference(Trec2021Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021.html")
class Trec2021Withrun(Dataset):
    """TREC Deep Learning 2021 (passage v2) with top-100 run"""

    TREC = reference(Trec2021)
    RUN = reference(Trec2021Run)

    def config(self) -> RerankAdhoc:
        trec = self.TREC.config()
        return RerankAdhoc.C(**trec.__arguments__(), run=self.RUN.config())


# --- TREC DL 2022


@lua
@dataset()
class Trec2022Queries(Dataset):
    """TREC Deep Learning 2022 passage queries (v2)"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv",
        checker=HashCheck("f1bfd53d80e81e58207ce557fd2211a0", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Trec2022Qrels(Dataset):
    """TREC Deep Learning 2022 passage qrels (v2, with duplicates)"""

    QRELS = FileDownloader(
        "qrels.withDupes.txt",
        url="https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt",
        checker=HashCheck("b36484d6cfd039664a570a4bf04f0eeb", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class Trec2022Run(Dataset):
    """TREC Deep Learning 2022 passage top-100 run (v2)"""

    RUN = FileDownloader(
        "top100.txt",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_passage_top100.txt.gz",
        checker=HashCheck("36004dfad64826167aeecddff1d490a6", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022.html")
class Trec2022(Dataset):
    """TREC Deep Learning 2022 (passage v2)"""

    COLLECTION = reference(Passages)
    TOPICS = reference(Trec2022Queries)
    QRELS = reference(Trec2022Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2022.html")
class Trec2022Withrun(Dataset):
    """TREC Deep Learning 2022 (passage v2) with top-100 run"""

    TREC = reference(Trec2022)
    RUN = reference(Trec2022Run)

    def config(self) -> RerankAdhoc:
        trec = self.TREC.config()
        return RerankAdhoc.C(**trec.__arguments__(), run=self.RUN.config())


# --- TREC DL 2023


@lua
@dataset()
class Trec2023Queries(Dataset):
    """TREC Deep Learning 2023 passage queries (v2)"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv",
        checker=HashCheck("7df9e17b47cc9aa5d1c9fd5b313e273c", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Trec2023Qrels(Dataset):
    """TREC Deep Learning 2023 passage qrels (v2, with duplicates)"""

    QRELS = FileDownloader(
        "qrels.withDupes.txt",
        url="https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt",
        checker=HashCheck("3a742d51ae65da2ece9c09b304b9e358", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@dataset()
class Trec2023Run(Dataset):
    """TREC Deep Learning 2023 passage top-100 run (v2)"""

    RUN = FileDownloader(
        "top100.txt",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_passage_top100.txt.gz",
        checker=HashCheck("c339ed75e1556cacb387899f34cadad1", md5),
    )

    def config(self) -> TrecAdhocRun:
        return TrecAdhocRun.C(path=self.RUN.path)


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2023.html")
class Trec2023(Dataset):
    """TREC Deep Learning 2023 (passage v2)"""

    COLLECTION = reference(Passages)
    TOPICS = reference(Trec2023Queries)
    QRELS = reference(Trec2023Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2023.html")
class Trec2023Withrun(Dataset):
    """TREC Deep Learning 2023 (passage v2) with top-100 run"""

    TREC = reference(Trec2023)
    RUN = reference(Trec2023Run)

    def config(self) -> RerankAdhoc:
        trec = self.TREC.config()
        return RerankAdhoc.C(**trec.__arguments__(), run=self.RUN.config())
