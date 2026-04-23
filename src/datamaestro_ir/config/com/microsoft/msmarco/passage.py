"""MS MARCO Passage Ranking collection.

A large-scale dataset focused on machine reading comprehension, question
answering, and passage ranking. The passage reranking task provides a query
and the top-1000 BM25 passages; a system is expected to rerank the most
relevant passage as high as possible. Not all 1000 passages are judged;
evaluation uses MRR.

**Publication**: Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. *MS MARCO: A Human
Generated MAchine Reading COmprehension Dataset.* In CoCo@NIPS.

See [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking)
for more details.
"""

import logging
import re
from pathlib import Path

from datamaestro.annotations.agreement import useragreement
from datamaestro.download.single import FileDownloader
from datamaestro.download import FileResource, FilesCopy, reference
from datamaestro.definitions import Dataset, datatasks, datatags, dataset
from datamaestro.download.archive import TarDownloader
from datamaestro_ir.data import RerankAdhoc, Adhoc, TrainingTripletsLines
from datamaestro_ir.data.csv import (
    Topics,
    AdhocRunWithText,
)
from datamaestro_ir.data.trec import TrecAdhocAssessments
from datamaestro_ir.data.stores import MsMarcoPassagesStore
from datamaestro_ir.download.docstore import docstore_builder
from datamaestro.utils import HashCheck
from hashlib import md5


class qids_file(FileResource):
    """Resource that writes query IDs to a file (one per line)."""

    def __init__(self, varname, qids):
        super().__init__(f"{varname}.txt", varname=varname)
        self._qids = qids

    def _download(self, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "wt") as fp:
            for qid in sorted(self._qids):
                fp.write(f"{qid}\n")


class judged_qids(FileResource):
    """Resource that extracts judged query IDs from a qrels reference.

    Produces a text file with one query ID per line.
    """

    def __init__(self, qrels_ref):
        super().__init__("judged_qids.txt")
        self.qrels_ref = qrels_ref
        self._dependencies.append(qrels_ref)

    def _download(self, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        assessments = self.qrels_ref.prepare()
        qids = {topic.topic_id for topic in assessments.iter()}
        with open(destination, "wt") as fp:
            for qid in sorted(qids):
                fp.write(f"{qid}\n")


# User agreement
lua = useragreement(
    """Will begin downloading MS-MARCO dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx""",
    id="net.windows.msmarco",
)

# --- Document collection


@lua
@dataset(size="2.9GB")
class Documents(Dataset):
    """MS-Marco passage collection and small query/qrel files.

    Downloads collectionandqueries.tar.gz once, builds the document store,
    and extracts query/qrel files for dev-small and eval-small splits.

    Format is TSV (pid \\t content)"""

    DOC_COUNT = 8_841_823

    DOCUMENTS = TarDownloader(
        "documents",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
        checker=HashCheck("31644046b18952c1386cd4564ba2ae69", md5),
        transient=True,
    )

    @staticmethod
    def _reader(source: Path):
        """Reads MS MARCO passages TSV (pid, content).

        Fixes the encoding issues in the MS MARCO passage collection where
        some characters are latin-1 encoded within otherwise UTF-8 text.
        Approach from ir-datasets (FixEncoding).
        """
        # Regexes to find suspicious byte sequences (latin-1 within UTF-8)
        sus = "[\x80-\xff]"
        has_sus = re.compile(sus)
        regexes = [
            re.compile(f"(...{sus}|..{sus}.|.{sus}..|{sus}...)"),
            re.compile(f"(..{sus}|.{sus}.|{sus}..)"),
            re.compile(f"(.{sus}|{sus}.)"),
        ]

        num_fixed = 0
        with open(source / "collection.tsv", "rb") as fp:
            for line in fp:
                line = line.decode("utf-8")
                _pid, content = line.rstrip("\n").split("\t", 1)
                # Fast path: skip encoding fix for lines without suspicious chars
                if has_sus.search(content):
                    for regex in regexes:
                        pos = 0
                        while pos < len(content):
                            match = regex.search(content, pos=pos)
                            if not match:
                                break
                            try:
                                fixed = match.group().encode("latin1").decode("utf8")
                                if len(fixed) == 1:
                                    content = (
                                        content[: match.start()]
                                        + fixed
                                        + content[match.end() :]
                                    )
                                    num_fixed += 1
                            except UnicodeError:
                                pass
                            pos = match.start() + 1
                yield {}, content.encode("utf-8")
        logging.info("Fixed encoding in %d passages", num_fixed)

    store = docstore_builder(
        DOCUMENTS,
        iter_factory=_reader,
        keys=[],
        doc_count=DOC_COUNT,
    )

    # These files are not used directly by the document store,
    # but they are needed by dev small queries and qrels
    files = FilesCopy(
        DOCUMENTS,
        {
            "queries.dev.small.tsv": "queries.dev.small.tsv",
            "qrels.dev.small.tsv": "qrels.dev.small.tsv",
            "queries.eval.small.tsv": "queries.eval.small.tsv",
        },
    )

    def config(self) -> MsMarcoPassagesStore:
        return MsMarcoPassagesStore.C(path=self.store.path, count=self.DOC_COUNT)


# --- Train


@lua
@dataset(size="2.5GB")
class TrainRun(Dataset):
    """

    TSV format: qid, pid, query, passage
    """

    RUN = TarDownloader(
        "run",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.train.tar.gz",
        checker=HashCheck("d99fdbd5b2ea84af8aa23194a3263052", md5),
    )

    def config(self) -> AdhocRunWithText:
        return AdhocRunWithText.C(path=self.RUN.path / "top1000.train.tsv")


@lua
@dataset()
class TrainQueries(Dataset):
    QUERIES = TarDownloader(
        "queries",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        files=["queries.train.tsv"],
        checker=HashCheck("c177b2795d5f2dcc524cf00fcd973be1", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path / "queries.train.tsv")


@lua
@dataset(size="10.1MB")
class TrainQrels(Dataset):
    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        checker=HashCheck("733fb9fe12d93e497f7289409316eccf", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class Train(Dataset):
    """MS-Marco train dataset"""

    COLLECTION = reference(Documents)
    TOPICS = reference(TrainQueries)
    QRELS = reference(TrainQrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class TrainWithrun(Dataset):
    """MSMarco train dataset, including the top-1000 to documents to re-rank"""

    TRAIN = reference(Train)
    RUN = reference(TrainRun)

    def config(self) -> RerankAdhoc:
        train = self.TRAIN.config()
        return RerankAdhoc.C(**train.__arguments__(), run=self.RUN.config())


# Training triplets


@dataset(
    url="https://github.com/microsoft/MSMARCO-Passage-Ranking",
    size="5.7GB",
)
class TrainTriplesID(Dataset):
    """Full training triples (query, positive passage, negative passage) with IDs"""

    TRIPLES = FileDownloader(
        "triples.tsv",
        size=1_841_693_309,
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz",
        checker=HashCheck("4e58f45f82f3fe99e3239ecffd8ed371", md5),
    )

    def config(self) -> TrainingTripletsLines:
        return TrainingTripletsLines.C(
            path=self.TRIPLES.path, doc_ids=True, topic_ids=True
        )


@dataset(
    url="https://github.com/microsoft/MSMARCO-Passage-Ranking",
    size="27.1GB",
)
class TrainTriplesSmallText(Dataset):
    """Small training triples (query, positive passage, negative passage) with text"""

    TRIPLES = FileDownloader(
        "triples.tsv",
        size=7_930_881_353,
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
        checker=HashCheck("c13bf99ff23ca691105ad12eab837f84", md5),
    )

    def config(self) -> TrainingTripletsLines:
        return TrainingTripletsLines.C(
            path=self.TRIPLES.path, doc_ids=False, topic_ids=False
        )


# ---
# --- Development set
# ---


@lua
@dataset()
class DevQueries(Dataset):
    QUERIES = TarDownloader(
        "queries",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
        files=["queries.dev.tsv"],
        checker=HashCheck("c177b2795d5f2dcc524cf00fcd973be1", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path / "queries.dev.tsv")


@lua
@dataset()
class DevRun(Dataset):
    RUN = TarDownloader(
        "run",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
        checker=HashCheck("8c140662bdf123a98fbfe3bb174c5831", md5),
    )

    def config(self) -> AdhocRunWithText:
        return AdhocRunWithText.C(path=self.RUN.path / "top1000.eval.tsv")


@lua
@dataset()
class DevQrels(Dataset):
    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        checker=HashCheck("9157ccaeaa8227f91722ba5770787b16", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class Dev(Dataset):
    """MS-Marco dev dataset"""

    COLLECTION = reference(Documents)
    TOPICS = reference(DevQueries)
    QRELS = reference(DevQrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class DevWithrun(Dataset):
    """MSMarco dev dataset, including the top-1000 to documents to re-rank"""

    DEV = reference(Dev)
    RUN = reference(DevRun)

    def config(self) -> RerankAdhoc:
        dev = self.DEV.config()
        return RerankAdhoc.C(**dev.__arguments__(), run=self.RUN.config())


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class DevJudged(Dataset):
    """MS-Marco dev dataset, restricted to judged queries"""

    COLLECTION = reference(Documents)
    TOPICS = reference(DevQueries)
    QRELS = reference(DevQrels)
    JUDGED_QIDS = judged_qids(QRELS)

    def config(self) -> Adhoc:
        from datamaestro_ir.data import FilteredTopics

        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[self.TOPICS.config()],
                qids_path=self.JUDGED_QIDS.path,
            ),
            assessments=self.QRELS.config(),
        )


@lua
@dataset()
class EvalWithrun(Dataset):
    RUN = TarDownloader(
        "run",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.eval.tar.gz",
        checker=HashCheck("73778cd99f6e0632d12d0b5731b20a02", md5),
    )

    def config(self) -> AdhocRunWithText:
        return AdhocRunWithText.C(path=self.RUN.path / "top1000.eval.tsv")


# ---
# --- Relevant Passages
# --- https://github.com/microsoft/MSMARCO-Passage-Ranking#relevant-passages
# ---


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class DevSmall(Dataset):
    """MS-Marco dev small dataset"""

    COLLECTION = reference(Documents)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=Topics.C(
                path=Documents.data_path / "files" / "queries.dev.small.tsv"
            ),
            assessments=TrecAdhocAssessments.C(
                path=Documents.data_path / "files" / "qrels.dev.small.tsv"
            ),
        )


@lua
@dataset(url="https://github.com/microsoft/MSMARCO-Passage-Ranking")
class EvalQueriesSmall(Dataset):
    """MS-Marco eval small queries"""

    COLLECTION = reference(Documents)

    def config(self) -> Topics:
        return Topics.C(path=Documents.data_path / "files" / "queries.eval.small.tsv")


# ---
# --- TREC 2019
# ---


@lua
@dataset()
class Trec2019Queries(Dataset):
    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        checker=HashCheck("756e60d714cee28d3b552289d6272f1d", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@dataset()
class Trec2019Run(Dataset):
    RUN = FileDownloader(
        "run.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
        checker=HashCheck("ec9e012746aa9763c7ff10b3336a3ce1", md5),
    )

    def config(self) -> AdhocRunWithText:
        return AdhocRunWithText.C(path=self.RUN.path / "top1000.eval.tsv")


@lua
@dataset()
class Trec2019Qrels(Dataset):
    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        checker=HashCheck("2f4be390198da108f6845c822e5ada14", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html")
class Trec2019(Dataset):
    "TREC Deep Learning (2019)"

    COLLECTION = reference(Documents)
    TOPICS = reference(Trec2019Queries)
    QRELS = reference(Trec2019Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html")
class Trec2019Withrun(Dataset):
    """TREC Deep Learning (2019), including the top-1000 to documents to re-rank"""

    TREC2019 = reference(Trec2019)
    RUN = reference(Trec2019Run)

    def config(self) -> RerankAdhoc:
        trec2019 = self.TREC2019.config()
        return RerankAdhoc.C(**trec2019.__arguments__(), run=self.RUN.config())


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html")
class Trec2019Judged(Dataset):
    """TREC Deep Learning (2019), restricted to judged queries"""

    COLLECTION = reference(Documents)
    TOPICS = reference(Trec2019Queries)
    QRELS = reference(Trec2019Qrels)
    JUDGED_QIDS = judged_qids(QRELS)

    def config(self) -> Adhoc:
        from datamaestro_ir.data import FilteredTopics

        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[self.TOPICS.config()],
                qids_path=self.JUDGED_QIDS.path,
            ),
            assessments=self.QRELS.config(),
        )


# ---
# --- TREC 2020
# ---


@lua
@dataset(size="12K")
class Trec2020Queries(Dataset):
    """TREC Deep Learning 2019 (topics)

    Topics of the TREC 2019 MS-Marco Deep Learning track"""

    QUERIES = FileDownloader(
        "queries.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
        checker=HashCheck("52cf21f50b4842233d35933cbc26b179", md5),
    )

    def config(self) -> Topics:
        return Topics.C(path=self.QUERIES.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@datatags("reranking")
@dataset(
    url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html",
)
class Trec2020Run(Dataset):
    """TREC Deep Learning (2020)

    Set of query/passages for the passage re-ranking task re-rank (TREC 2020)"""

    RUN = FileDownloader(
        "run.tsv",
        url="https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz",
        checker=HashCheck("aa6fbc51d66bd1dc745964c0e140a727", md5),
    )

    def config(self) -> AdhocRunWithText:
        return AdhocRunWithText.C(path=self.RUN.path / "top1000.eval.tsv")


@lua
@dataset()
class Trec2020Qrels(Dataset):
    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://trec.nist.gov/data/deep/2020qrels-pass.txt",
        checker=HashCheck("0355ccee7509ac0463e8278186cdd8d1", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html")
class Trec2020(Dataset):
    "TREC Deep Learning (2020)"

    COLLECTION = reference(Documents)
    TOPICS = reference(Trec2020Queries)
    QRELS = reference(Trec2020Qrels)

    def config(self) -> Adhoc:
        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=self.TOPICS.config(),
            assessments=self.QRELS.config(),
        )


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html")
class Trec2020Withrun(Dataset):
    """TREC Deep Learning (2020), including the top-1000 to documents to re-rank"""

    TREC2020 = reference(Trec2020)
    RUN = reference(Trec2020Run)

    def config(self) -> RerankAdhoc:
        trec2020 = self.TREC2020.config()
        return RerankAdhoc.C(**trec2020.__arguments__(), run=self.RUN.config())


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html")
class Trec2020Judged(Dataset):
    """TREC Deep Learning (2020), restricted to judged queries"""

    COLLECTION = reference(Documents)
    TOPICS = reference(Trec2020Queries)
    QRELS = reference(Trec2020Qrels)
    JUDGED_QIDS = judged_qids(QRELS)

    def config(self) -> Adhoc:
        from datamaestro_ir.data import FilteredTopics

        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[self.TOPICS.config()],
                qids_path=self.JUDGED_QIDS.path,
            ),
            assessments=self.QRELS.config(),
        )


# ---
# --- TREC DL Hard
# --- https://github.com/grill-lab/DL-Hard
# ---


@lua
@dataset(url="https://github.com/grill-lab/DL-Hard")
class TrecDlHardQrels(Dataset):
    """TREC DL-Hard qrels (passage)"""

    QRELS = FileDownloader(
        "qrels.tsv",
        url="https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/dl_hard-passage.qrels",
        checker=HashCheck("8583c2cbad56eeacb449586fe1d2a471", md5),
    )

    def config(self) -> TrecAdhocAssessments:
        return TrecAdhocAssessments.C(path=self.QRELS.path)


@lua
@datatasks("information retrieval", "passage retrieval")
@dataset(url="https://github.com/grill-lab/DL-Hard")
class TrecDlHard(Dataset):
    """A more challenging subset of TREC DL 2019 and 2020 passage queries

    See: Mackie et al., "How Deep is your Learning: the DL-HARD Annotated Deep
    Learning Dataset", SIGIR 2021.
    """

    COLLECTION = reference(Documents)
    TOPICS_2019 = reference(Trec2019Queries)
    TOPICS_2020 = reference(Trec2020Queries)
    QRELS = reference(TrecDlHardQrels)

    # From https://github.com/grill-lab/DL-Hard/blob/main/dataset/folds.json
    HARD_QIDS = qids_file(
        "hard_qids",
        [
            "915593",
            "451602",
            "966413",
            "1056204",
            "182539",
            "655914",
            "67316",
            "883915",
            "1049519",
            "174463",
            "794429",
            "588587",
            "1114646",
            "537817",
            "1065636",
            "144862",
            "443396",
            "332593",
            "1103812",
            "19335",
            "177604",
            "1108939",
            "264403",
            "86606",
            "1133485",
            "1117817",
            "705609",
            "315637",
            "673670",
            "1105792",
            "801118",
            "507445",
            "87452",
            "88495",
            "554515",
            "166046",
            "730539",
            "1108100",
            "1109707",
            "1056416",
            "190044",
            "527433",
            "489204",
            "877809",
            "1106007",
            "47923",
            "1136769",
            "1112341",
            "1103153",
            "273695",
        ],
    )

    def config(self) -> Adhoc:
        from datamaestro_ir.data import FilteredTopics

        return Adhoc.C(
            documents=self.COLLECTION.config(),
            topics=FilteredTopics.C(
                topics=[
                    self.TOPICS_2019.config(),
                    self.TOPICS_2020.config(),
                ],
                qids_path=self.HARD_QIDS.path,
            ),
            assessments=self.QRELS.config(),
        )
