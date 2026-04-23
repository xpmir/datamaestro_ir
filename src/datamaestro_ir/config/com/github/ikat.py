# See documentation on https://datamaestro.readthedocs.io

from datamaestro.definitions import Dataset, datatasks, dataset
from datamaestro_ir.data.conversation.base import ConversationUserTopics

from datamaestro.utils import HashCheck
from datamaestro.download.single import FileDownloader
from datamaestro_ir.data.conversation.ikat import IkatConversations


@datatasks("conversational search", "query rewriting")
@dataset(
    id=".2025",
    url="https://github.com/irlabamsterdam/iKAT/tree/main/2025",
)
class Test2025(Dataset):
    """Question-in-context rewriting

    iKAT is a test dataset for question-in-context rewriting that consists of
    questions each given in a dialog context together with a context-independent
    rewriting of the question.
    """

    TOPICS = FileDownloader(
        "topics.json",
        "https://raw.githubusercontent.com/irlabamsterdam/iKAT/refs/heads/main/2025/data/2025_test_topics.json",
        checker=HashCheck("16f8444a8d0a8dfe0090f478f185a63c"),
    )

    def config(self) -> ConversationUserTopics:
        return ConversationUserTopics.C(
            conversations=IkatConversations.C(path=self.TOPICS.path)
        )


@datatasks("conversational search", "query rewriting")
@dataset(
    id=".2024",
    url="https://github.com/irlabamsterdam/iKAT/tree/main/2024",
)
class Test2024(Dataset):
    """iKAT 2024 dataset"""

    QRELS = FileDownloader(
        "qrels",
        "https://trec.nist.gov/data/ikat/2024-qrels.txt",
        checker=HashCheck("57f958903ed1c12bbac207f62800814f"),
    )
    TOPICS = FileDownloader(
        "topics.json",
        "https://raw.githubusercontent.com/irlabamsterdam/iKAT/refs/heads/main/2024/data/2024_test_topics.json",
        checker=HashCheck("ad45bc6e7add2081d69ea60a0a4d1203"),
    )

    def config(self) -> ConversationUserTopics:
        return ConversationUserTopics.C(
            conversations=IkatConversations.C(path=self.TOPICS.path)
        )


@datatasks("conversational search", "query rewriting")
@dataset(
    id=".2023",
    url="https://github.com/irlabamsterdam/iKAT/tree/main/2023",
)
class Test2023(Dataset):
    """iKAT 2023 dataset"""

    QRELS = FileDownloader(
        "qrels",
        "https://trec.nist.gov/data/ikat/2023-qrels.all-turns.txt",
        checker=HashCheck("79dc121bab25b2245e52a53263e5ad1f"),
    )
    TOPICS = FileDownloader(
        "topics.json",
        "https://raw.githubusercontent.com/irlabamsterdam/iKAT/refs/heads/main/2023/data/2023_test_topics.json",
        checker=HashCheck("684fa0197cdec8c3cfb6a2e586ab83f6"),
    )

    def config(self) -> ConversationUserTopics:
        return ConversationUserTopics.C(
            conversations=IkatConversations.C(path=self.TOPICS.path)
        )
