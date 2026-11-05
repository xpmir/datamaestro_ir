"""LightOn embeddings-fine-tuning-filtered-en dataset family.

Registers the HuggingFace dataset
``https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-en``
as a parameterized variant family::

    datamaestro prepare 'ai.lighton.embeddings_fine_tuning[name=fiqa_en,streaming=false]'
"""

from __future__ import annotations

from typing import Optional

from datamaestro.definitions import Dataset, dataset
from datamaestro.variants import AxesVariants, Axis
from datamaestro_ir.data.lighton import EmbeddingsFineTuningSamples

REPO_ID = "lightonai/embeddings-fine-tuning-filtered-en"
URL = f"https://huggingface.co/datasets/{REPO_ID}"

# Core 7 English splits used in the DenseOn / LateOn fine-tuning recipe
# (https://huggingface.co/blog/lightonai/denseon-lateon).
CONFIGS = [
    "fever_en",
    "fiqa_en",
    "hotpotqa_en",
    "msmarco_en",
    "nq_en",
    "squadv2_en",
    "trivia_en",
]

# Full list of splits available in the repository (including mLateOn additions).
ALL_CONFIGS = [
    *CONFIGS,
    "miracl_en",
    "mldr_en",
]


class EmbeddingsFineTuningVariants(AxesVariants):
    """Variant space for ``lightonai/embeddings-fine-tuning-filtered-en``."""

    name = Axis(ALL_CONFIGS)
    """Split name (e.g. ``fiqa_en``, ``msmarco_en``)."""

    revision = Axis(type=Optional[str], default=None, in_id=False)
    """HuggingFace repository commit revision SHA."""

    streaming = Axis([False, True], default=False, type=bool, in_id=False)
    """Streaming mode flag."""


@dataset(id="", url=URL, variants=EmbeddingsFineTuningVariants)
class EmbeddingsFineTuning(Dataset):
    """LightOn ``embeddings-fine-tuning-filtered-en`` contrastive fine-tuning dataset."""

    def config(self, **kw) -> EmbeddingsFineTuningSamples:
        import logging

        if kw.get("streaming", False):
            logging.warning(
                "Using streaming mode, will not download the dataset. "
                "Use streaming=false (e.g. ai.lighton.embeddings_fine_tuning[...,streaming=false]) to download."
            )
        return EmbeddingsFineTuningSamples.C(
            repo_id=REPO_ID,
            **kw,
        )

