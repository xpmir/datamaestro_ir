"""TREC Conversational Assistance Track (CaST) 2019-2022.

CaST evaluates conversational information seeking systems where users engage
in multi-turn conversations to satisfy complex information needs.

Each year uses different document collections:

- **2019 train (v0):** MS MARCO passages + CAR v2.0 + WAPO v2 paragraphs
- **2019 eval / 2020 (v1):** MS MARCO passages + CAR v2.0
- **2021 (v2):** MS MARCO documents v1 + WAPO v4 + KILT (segmented)
- **2022 (v3):** MS MARCO documents v2 + WAPO v4 + KILT (segmented)

See `<https://www.treccast.ai/>`_ for more details.
"""

from hashlib import md5

from datamaestro.definitions import Dataset, datatags, datatasks, dataset
from datamaestro.download import reference
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck
from datamaestro_ir.data import Adhoc, PrefixedDocumentStore
from datamaestro_ir.data.conversation.base import ConversationUserTopics
from datamaestro_ir.data.conversation.cast import CastConversations
from datamaestro_ir.data.stores import CastSegmentedPassageStore
from datamaestro_ir.data.trec import TrecAdhocAssessments

# Base collection imports
from datamaestro_ir.config.com.microsoft.msmarco.passage import (
    Documents as MsMarcoPassages,
)
from datamaestro_ir.config.gov.nist.trec.car import Documents as CarDocuments
from datamaestro_ir.config.gov.nist.trec.wapo import (
    WapoV2Passages,
    WapoV4Documents,
)
from datamaestro_ir.config.org.facebook.kilt import Documents as KiltDocuments
from datamaestro_ir.config.com.microsoft.msmarco.document import (
    Documents as MsMarcoDocumentsV1,
)
from datamaestro_ir.config.com.microsoft.msmarco.documentv2 import (
    Documents as MsMarcoDocumentsV2,
)


# ============================================================================
# Document collection classes
# ============================================================================


@dataset()
class CastV0Documents(Dataset):
    """CaST v0 document collection (2019 train).

    Combines WAPO v2 paragraphs (``WAPO_`` prefix), MS MARCO passages
    (``MARCO_`` prefix), and CAR v2.0 paragraphs (``CAR_`` prefix).
    """

    WAPO = reference(WapoV2Passages)
    MSMARCO = reference(MsMarcoPassages)
    CAR = reference(CarDocuments)

    def config(self) -> PrefixedDocumentStore:
        return PrefixedDocumentStore.C(
            sources=[
                self.WAPO.config(),
                self.MSMARCO.config(),
                self.CAR.config(),
            ],
            prefixes=["WAPO_", "MARCO_", "CAR_"],
        )


@dataset()
class CastV1Documents(Dataset):
    """CaST v1 document collection (2019 eval, 2020).

    Combines MS MARCO passages (``MARCO_`` prefix) and CAR v2.0 paragraphs
    (``CAR_`` prefix).
    """

    MSMARCO = reference(MsMarcoPassages)
    CAR = reference(CarDocuments)

    def config(self) -> PrefixedDocumentStore:
        return PrefixedDocumentStore.C(
            sources=[
                self.MSMARCO.config(),
                self.CAR.config(),
            ],
            prefixes=["MARCO_", "CAR_"],
        )


@dataset()
class CastV2Documents(Dataset):
    """CaST v2 document collection (2021).

    Combines segmented passages from MS MARCO documents v1 (``MARCO_``),
    WAPO v4 (``WAPO_``), and KILT (``KILT_``). Each base document is split
    into passages using pre-computed character offsets.
    """

    MSMARCO = reference(MsMarcoDocumentsV1)
    WAPO = reference(WapoV4Documents)
    KILT = reference(KiltDocuments)

    MSMARCO_OFFSETS = FileDownloader(
        "MARCO_v1.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2021/MARCO_v1.chunks.jsonl.gz",
        checker=HashCheck("b76c8d1e3b260764d573ce618a15525f", md5),
    )
    WAPO_OFFSETS = FileDownloader(
        "WaPo-v2.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2021/WaPo-v2.chunks.jsonl.gz",
        checker=HashCheck("900c56039b4a3edd642983c4a1e13796", md5),
    )
    KILT_OFFSETS = FileDownloader(
        "KILT-nodupes.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2021/KILT-nodupes.chunks.jsonl.gz",
        checker=HashCheck("7bd9c844ea7d8ecc7a1236ef5c7d7722", md5),
    )
    MSMARCO_DUPES = FileDownloader(
        "marco_duplicates.txt",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2021/duplicate_lists/marco_duplicates.txt",
        checker=HashCheck("549f721aec777b18f6538ddeabf6a8f3", md5),
    )
    WAPO_DUPES = FileDownloader(
        "wapo-near-duplicates",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2021/duplicate_lists/wapo-near-duplicates",
        checker=HashCheck("23bacc7e03af656dc590fd6a5476bc83", md5),
    )

    def config(self) -> PrefixedDocumentStore:
        return PrefixedDocumentStore.C(
            sources=[
                CastSegmentedPassageStore.C(
                    base_store=self.MSMARCO.config(),
                    offsets_path=self.MSMARCO_OFFSETS.path,
                    dupes_path=self.MSMARCO_DUPES.path,
                ),
                CastSegmentedPassageStore.C(
                    base_store=self.WAPO.config(),
                    offsets_path=self.WAPO_OFFSETS.path,
                    dupes_path=self.WAPO_DUPES.path,
                ),
                CastSegmentedPassageStore.C(
                    base_store=self.KILT.config(),
                    offsets_path=self.KILT_OFFSETS.path,
                ),
            ],
            prefixes=["MARCO_", "WAPO_", "KILT_"],
        )


@dataset()
class CastV3Documents(Dataset):
    """CaST v3 document collection (2022).

    Combines segmented passages from MS MARCO documents v2 (``MARCO_``),
    WAPO v4 (``WAPO_``), and KILT (``KILT_``). Uses v3-specific segmentation
    offsets and a shared duplicate list.
    """

    MSMARCO = reference(MsMarcoDocumentsV2)
    WAPO = reference(WapoV4Documents)
    KILT = reference(KiltDocuments)

    MSMARCO_OFFSETS = FileDownloader(
        "MARCO_v2.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2022/MARCO_v2.chunks.jsonl.gz",
        checker=HashCheck("2f0bbe152b4645bff744892c5b53471f", md5),
    )
    WAPO_OFFSETS = FileDownloader(
        "WaPo.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2022/WaPo.chunks.jsonl.gz",
        checker=HashCheck("1dbed58d09aeaa6046c234d056ea703e", md5),
    )
    KILT_OFFSETS = FileDownloader(
        "KILT.chunks.jsonl.gz",
        url="https://huggingface.co/datasets/daltonj/treccastweb/resolve/main/2022/KILT.chunks.jsonl.gz",
        checker=HashCheck("290cc1a172b7fc29b5ce211a87ced098", md5),
    )
    ALL_DUPES = FileDownloader(
        "all_duplicates.txt",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/duplicate_files/all_duplicates.txt",
        checker=HashCheck("2d6cbb4a2e8733423434cfe561635c21", md5),
    )

    def config(self) -> PrefixedDocumentStore:
        dupes_path = self.ALL_DUPES.path
        return PrefixedDocumentStore.C(
            sources=[
                CastSegmentedPassageStore.C(
                    base_store=self.MSMARCO.config(),
                    offsets_path=self.MSMARCO_OFFSETS.path,
                    dupes_path=dupes_path,
                ),
                CastSegmentedPassageStore.C(
                    base_store=self.WAPO.config(),
                    offsets_path=self.WAPO_OFFSETS.path,
                    dupes_path=dupes_path,
                ),
                CastSegmentedPassageStore.C(
                    base_store=self.KILT.config(),
                    offsets_path=self.KILT_OFFSETS.path,
                    dupes_path=dupes_path,
                ),
            ],
            prefixes=["MARCO_", "WAPO_", "KILT_"],
        )


# ============================================================================
# Task datasets
# ============================================================================


@datatags("conversation", "context", "query")
@datatasks("conversational search", "passage retrieval")
@dataset(
    id=".2019.train",
    url="https://www.treccast.ai/",
)
class Train2019(Dataset):
    """TREC CaST 2019 training set.

    Multi-turn conversational search training data with 30 topics. Uses the
    v0 document collection (WAPO v2 paragraphs + MS MARCO passages + CAR).
    """

    DOCUMENTS = reference(CastV0Documents)
    TOPICS = FileDownloader(
        "train_topics_v1.0.json",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/training/train_topics_v1.0.json",
        checker=HashCheck("20170e151de3d0e3e43e4e8e505bf696", md5),
    )
    QRELS = FileDownloader(
        "train_topics_mod.qrel",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/training/train_topics_mod.qrel",
        checker=HashCheck("84af6506eab4429c70784b69a62f15a9", md5),
    )

    def config(self) -> Adhoc:
        return Adhoc.C(
            topics=ConversationUserTopics.C(
                conversations=CastConversations.C(
                    path=self.TOPICS.path, year=2019
                )
            ),
            assessments=TrecAdhocAssessments.C(path=self.QRELS.path),
            documents=self.DOCUMENTS.config(),
        )


@datatags("conversation", "context", "query")
@datatasks("conversational search", "passage retrieval")
@dataset(
    id=".2019",
    url="https://www.treccast.ai/",
)
class Test2019(Dataset):
    """TREC CaST 2019 evaluation set.

    Multi-turn conversational search evaluation with 50 topics and 479 turns.
    Uses the v1 document collection (MS MARCO passages + CAR).
    """

    DOCUMENTS = reference(CastV1Documents)
    TOPICS = FileDownloader(
        "evaluation_topics_v1.0.json",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json",
        checker=HashCheck("36228218544de8dd81c09fdedc3f0663", md5),
    )
    QRELS = FileDownloader(
        "2019qrels.txt",
        url="https://trec.nist.gov/data/cast/2019qrels.txt",
        checker=HashCheck("aab2ca0fba4f2e4aac2a0a1bc7b1fb23", md5),
    )

    def config(self) -> Adhoc:
        return Adhoc.C(
            topics=ConversationUserTopics.C(
                conversations=CastConversations.C(
                    path=self.TOPICS.path, year=2019
                )
            ),
            assessments=TrecAdhocAssessments.C(path=self.QRELS.path),
            documents=self.DOCUMENTS.config(),
        )


@datatags("conversation", "context", "query")
@datatasks("conversational search", "passage retrieval")
@dataset(
    id=".2020",
    url="https://www.treccast.ai/",
)
class Test2020(Dataset):
    """TREC CaST 2020 evaluation set.

    Multi-turn conversational search with manual and automatic
    decontextualized utterances. Uses the v1 document collection
    (MS MARCO passages + CAR).
    """

    DOCUMENTS = reference(CastV1Documents)
    TOPICS = FileDownloader(
        "2020_manual_evaluation_topics_v1.0.json",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
        checker=HashCheck("98ae7aaa26cd5c8e3e27041d4a52e4ab", md5),
    )
    QRELS = FileDownloader(
        "2020qrels.txt",
        url="https://trec.nist.gov/data/cast/2020qrels.txt",
        checker=HashCheck("de6ae6a3c7f5a7b5b21a05e8e6db3f8a", md5),
    )

    def config(self) -> Adhoc:
        return Adhoc.C(
            topics=ConversationUserTopics.C(
                conversations=CastConversations.C(
                    path=self.TOPICS.path, year=2020
                )
            ),
            assessments=TrecAdhocAssessments.C(path=self.QRELS.path),
            documents=self.DOCUMENTS.config(),
        )


@datatags("conversation", "context", "query")
@datatasks("conversational search", "passage retrieval")
@dataset(
    id=".2021",
    url="https://www.treccast.ai/",
)
class Test2021(Dataset):
    """TREC CaST 2021 evaluation set.

    Multi-turn conversational search with segmented document collections.
    Uses the v2 document collection (MS MARCO docs v1 + WAPO v4 + KILT,
    segmented into passages).
    """

    DOCUMENTS = reference(CastV2Documents)
    TOPICS = FileDownloader(
        "2021_manual_evaluation_topics_v1.0.json",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2021/2021_manual_evaluation_topics_v1.0.json",
        checker=HashCheck("eafbffb3e6c1aabaabbff839d37e2116", md5),
    )
    QRELS = FileDownloader(
        "trec-cast-qrels-docs.2021.qrel",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2021/trec-cast-qrels-docs.2021.qrel",
        checker=HashCheck("3393ef56a1a2b0b68b4e0936c89a5854", md5),
    )

    def config(self) -> Adhoc:
        return Adhoc.C(
            topics=ConversationUserTopics.C(
                conversations=CastConversations.C(
                    path=self.TOPICS.path, year=2021
                )
            ),
            assessments=TrecAdhocAssessments.C(path=self.QRELS.path),
            documents=self.DOCUMENTS.config(),
        )


@datatags("conversation", "context", "query")
@datatasks("conversational search", "passage retrieval")
@dataset(
    id=".2022",
    url="https://www.treccast.ai/",
)
class Test2022(Dataset):
    """TREC CaST 2022 evaluation set.

    Multi-turn conversational search with tree-structured conversations.
    Uses the v3 document collection (MS MARCO docs v2 + WAPO v4 + KILT,
    segmented into passages).
    """

    DOCUMENTS = reference(CastV3Documents)
    TOPICS = FileDownloader(
        "2022_evaluation_topics_tree_v1.0.json",
        url="https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/2022_evaluation_topics_tree_v1.0.json",
        checker=HashCheck("a2990d49f36af95f5a95a27a0aa04a0a", md5),
    )
    QRELS = FileDownloader(
        "2022-qrels.txt",
        url="https://trec.nist.gov/data/cast/2022-qrels.txt",
        checker=HashCheck("1d1f3b2c8d5f06d2f6df3bfad0e0d5a2", md5),
    )

    def config(self) -> Adhoc:
        return Adhoc.C(
            topics=ConversationUserTopics.C(
                conversations=CastConversations.C(
                    path=self.TOPICS.path, year=2022
                )
            ),
            assessments=TrecAdhocAssessments.C(path=self.QRELS.path),
            documents=self.DOCUMENTS.config(),
        )
