# See documentation on https://datamaestro.readthedocs.io

from datamaestro.definitions import dataset
from datamaestro_ir.data.huggingface import HuggingFacePairwiseSampleDataset
from datamaestro.download.huggingface import hf_download


@hf_download(
    "dataset",
    "sentence-transformers/msmarco-hard-negatives",
    data_files="msmarco-hard-negatives-msmarco.jsonl.gz",
    split="train",
)
@dataset(
    HuggingFacePairwiseSampleDataset,
    url="https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives",
)
def ensemble(dataset):
    """Hard negatives mined from a set of models"""
    return dataset
