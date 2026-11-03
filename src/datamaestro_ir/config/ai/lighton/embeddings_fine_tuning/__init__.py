"""LightOn embeddings-fine-tuning-filtered dataset family.

Registers the HuggingFace datasets
``https://huggingface.co/datasets/lightonai/embeddings-fine-tuning-filtered-<lang>``
as a parameterized variant family::

    datamaestro prepare 'ai.lighton.embeddings_fine_tuning[name=fiqa_en,streaming=false]'
    datamaestro prepare 'ai.lighton.embeddings_fine_tuning[name=fiqa_fr,streaming=false]'

The ``name`` axis is ``<source>_<lang>``; the repository is derived from the
``<lang>`` suffix.
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

# Full list of English splits available in the repository (including mLateOn additions).
ALL_CONFIGS = [
    *CONFIGS,
    "miracl_en",
    "mldr_en",
]

_CORE_SOURCES = [config.removesuffix("_en") for config in CONFIGS]

# Sources available in each language repository (not every language has every source)
SOURCES_BY_LANG = {
    "en": [*_CORE_SOURCES, "miracl", "mldr"],
    "ar": [*_CORE_SOURCES, "miracl", "mldr"],
    "de": [*_CORE_SOURCES, "mldr"],
    "es": [*_CORE_SOURCES, "miracl", "mldr"],
    "fr": [*_CORE_SOURCES, "miracl", "mldr"],
    "it": [*_CORE_SOURCES, "mldr"],
    "no": _CORE_SOURCES,
    "pt": [*_CORE_SOURCES, "mldr"],
    "sv": _CORE_SOURCES,
}

# Every ``<source>_<lang>`` split across all language repositories
LANGUAGE_CONFIGS = [
    f"{source}_{lang}" for lang, sources in SOURCES_BY_LANG.items() for source in sources
]


def repo_id_for(name: Optional[str]) -> str:
    """Repository of a split, from its ``_<lang>`` suffix (English if unknown)."""
    if not name:
        return REPO_ID
    lang = name.rsplit("_", 1)[-1]
    return f"lightonai/embeddings-fine-tuning-filtered-{lang}"


class EmbeddingsFineTuningVariants(AxesVariants):
    """Variant space for ``lightonai/embeddings-fine-tuning-filtered-<lang>``."""

    name = Axis(LANGUAGE_CONFIGS)
    """Split name (e.g. ``fiqa_en``, ``msmarco_fr``)."""

    revision = Axis(type=Optional[str], default=None, in_id=False)
    """HuggingFace repository commit revision SHA."""

    streaming = Axis([False, True], default=False, type=bool, in_id=False)
    """Streaming mode flag."""


@dataset(id="", url=URL, variants=EmbeddingsFineTuningVariants)
class EmbeddingsFineTuning(Dataset):
    """LightOn ``embeddings-fine-tuning-filtered-<lang>`` contrastive fine-tuning dataset."""

    def config(self, **kw) -> EmbeddingsFineTuningSamples:
        import logging

        if kw.get("streaming", False):
            logging.warning(
                "Using streaming mode, will not download the dataset. "
                "Use streaming=false (e.g. ai.lighton.embeddings_fine_tuning[...,streaming=false]) to download."
            )
        return EmbeddingsFineTuningSamples.C(
            repo_id=repo_id_for(kw.get("name")),
            **kw,
        )
