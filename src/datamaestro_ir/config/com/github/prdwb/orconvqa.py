# See documentation on https://datamaestro.readthedocs.io

from datamaestro.definitions import Dataset, datatasks, dataset
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck


from datamaestro_ir.data.conversation.orconvqa import OrConvQADataset
from datamaestro.data.ml import Supervised


@datatasks("query rewriting")
@dataset(
    url="https://github.com/prdwb/orconvqa-release",
)
class Preprocessed(Dataset):
    """Open-Retrieval Conversational Question Answering datasets

    OrConvQA is an aggregation of three existing datasets:

    1. the QuAC dataset that offers information-seeking conversations,
    1. the CANARD dataset that consists of context-independent rewrites of QuAC questions, and
    3. the Wikipedia corpus that serves as the knowledge source of answering questions.

    Each dataset is an instance of :class:`datamaestro_ir.data.conversation.OrConvQADataset`
    """

    TRAIN = FileDownloader(
        "train.jsonl",
        "https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt",
        checker=HashCheck("7513a9ef12d8b7a4471166dc4fef77b7"),
    )
    DEV = FileDownloader(
        "dev.jsonl",
        "https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt",
        checker=HashCheck("7765658995cc9ffd5eb39a400d814b20"),
    )
    TEST = FileDownloader(
        "test.jsonl",
        "https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt",
        checker=HashCheck("0cf3a755f06297b9c02e7db45f8dc8be"),
    )

    def config(self) -> Supervised:
        return Supervised.C(
            train=OrConvQADataset.C(path=self.TRAIN.path),
            validation=OrConvQADataset.C(path=self.DEV.path),
            test=OrConvQADataset.C(path=self.TEST.path),
        )
