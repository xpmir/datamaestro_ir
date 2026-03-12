# See documentation on https://datamaestro.readthedocs.io
from hashlib import md5

from datamaestro.definitions import Dataset, dataset
from datamaestro.download.single import FileDownloader
from datamaestro.utils import HashCheck
from datamaestro_ir.data.distillation import PairwiseDistillationSamplesTSV

ZENODO_BASE = (
    "https://zenodo.org/record/4068216/files/"
)


@dataset(url="https://github.com/sebastian-hofstaetter/neural-ranking-kd")
class MsmarcoEnsembleTeacher(Dataset):
    """Training files without the text content instead using the ids from MSMARCO

    The teacher files (using the data from "Train Triples Small" with ~40
    million triples) with the format pos_score neg_score query_id pos_passage_id
    neg_passage_id (with tab separation)
    """

    SCORES = FileDownloader(
        "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv",
        url=f"{ZENODO_BASE}"
        "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
        checker=HashCheck("4d99696386f96a7f1631076bcc53ac3c", md5),
    )

    def config(self) -> PairwiseDistillationSamplesTSV:
        return PairwiseDistillationSamplesTSV.C(
            path=self.SCORES.path,
            with_docid=True,
            with_queryid=True,
        )


@dataset(url="https://github.com/sebastian-hofstaetter/neural-ranking-kd")
class MsmarcoBertTeacher(Dataset):
    """Training files without the text content instead using the ids from MSMARCO

    The teacher files (using the data from "Train Triples Small" with ~40
    million triples) with the format pos_score neg_score query_id pos_passage_id
    neg_passage_id (with tab separation)
    """

    SCORES = FileDownloader(
        "bertbase_cat_msmarcopassage_train_scores_ids.tsv",
        url=f"{ZENODO_BASE}"
        "bertbase_cat_msmarcopassage_train_scores_ids.tsv?download=1",
        checker=HashCheck("a2575af08a19b47c2041e67c9efcd917", md5),
    )

    def config(self) -> PairwiseDistillationSamplesTSV:
        return PairwiseDistillationSamplesTSV.C(
            path=self.SCORES.path,
            with_docid=True,
            with_queryid=True,
        )
