"""MS MARCO hard negatives with teacher scores (OpenSearch).

For each MS MARCO passage training query, the top-100 passages retrieved by
``opensearch-neural-sparse-encoding-doc-v1``, with re-ranking scores from an
ensemble of bi-encoder and cross-encoder teachers.

Queries and documents are MS MARCO passage IDs (see
``com.microsoft.msmarco.passage``).

**Publication**: Zhichao Geng, Dongyu Ru, and Yang Yang. 2024. *Towards
Competitive Search Relevance For Inference-Free Learned Sparse Retrievers.*
arXiv:2411.04403.
"""

from datamaestro.definitions import Dataset, dataset
from datamaestro_ir.data.huggingface import HuggingFaceListwiseDistillationSamples

REPO_ID = "opensearch-project/msmarco-hard-negatives-llm-scores"
URL = f"https://huggingface.co/datasets/{REPO_ID}"
REVISION = "6b7b3ce4796e9c4f9832e5ec2877e3f1bf42bf10"


@dataset(id="", url=URL)
class MsMarcoHardNegativesLLMScores(Dataset):
    """MS MARCO top-100 hard negatives with teacher scores (502,939 queries)"""

    def config(self) -> HuggingFaceListwiseDistillationSamples:
        return HuggingFaceListwiseDistillationSamples.C(
            repo_id=REPO_ID, split="train", revision=REVISION
        )
