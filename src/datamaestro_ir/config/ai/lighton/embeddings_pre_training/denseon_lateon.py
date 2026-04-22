"""DenseON-LateON mGTE-style training recipe for
``lightonai/embeddings-pre-training``.

UNIONs three groups of configs with different per-group filter rules:

1. Standard sources (all configs except ``fw-edu``, ``wikipedia_hlp_cm``,
   and ``wikipedia_hlp_dl``): keep rows with ``drop=False`` and
   ``duplicate IS NULL`` and ``similarity >= 3.0``.
2. ``fw-edu`` (FineWeb-Edu, no rule-based filter or dedup applied
   upstream): keep only the top ~35% by similarity. No drop/duplicate
   filter — the config itself isn't pre-filtered.
3. ``wikipedia_hlp_cm`` and ``wikipedia_hlp_dl``: included as-is, no
   filter.

Id derived from the package path:
``ai.lighton.embeddings_pre_training.denseon_lateon``.

Note on config names: the reference recipe refers to ``fw_edu``,
``hlp_wikipedia_cm`` and ``hlp_wikipedia_dl``; the actual HuggingFace
config names are ``fw-edu``, ``wikipedia_hlp_cm`` and ``wikipedia_hlp_dl``
(verified against the HF datasets API at dataset-card time).
"""

from __future__ import annotations

from datamaestro.definitions import dataset, Dataset, datatags, datatasks

from datamaestro_ir.data.distillation import ConcatPointwiseDistillationSamples
from datamaestro_ir.data.lighton import EmbeddingsPreTrainingSamples

from . import CONFIGS, REPO_ID, URL


# Config names with special handling in the recipe.
_FW_EDU = "fw-edu"
_HLP_WIKI = {"wikipedia_hlp_cm", "wikipedia_hlp_dl"}

# Standard sources: everything not in the special sets above.
_STANDARD = [c for c in CONFIGS if c != _FW_EDU and c not in _HLP_WIKI]


@datatags(
    "information retrieval",
    "distillation",
    "pre-training",
    "mgte",
    "recipe",
)
@datatasks("learning to rank")
@dataset(id="", url=URL)
class DenseonLateon(Dataset):
    """DenseON-LateON mGTE-style pre-training recipe built by UNIONing
    three groups of ``lightonai/embeddings-pre-training`` configs with
    group-specific filters."""

    def config(self) -> ConcatPointwiseDistillationSamples:
        sources = []

        # (1) Standard sources — drop/duplicate filter + similarity >= 3.
        sources.extend(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=cfg,
                split="train",
                streaming=True,
                filter_drop=True,
                filter_duplicate=True,
                min_similarity=3.0,
            )
            for cfg in _STANDARD
        )

        # (2) FineWeb-Edu — top 35% by similarity, no drop/duplicate filter.
        sources.append(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=_FW_EDU,
                split="train",
                streaming=True,
                filter_drop=False,
                filter_duplicate=False,
                top_percentile=0.35,
            )
        )

        # (3) Wikipedia HLP splits — included as-is.
        sources.extend(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=cfg,
                split="train",
                streaming=True,
                filter_drop=False,
                filter_duplicate=False,
            )
            for cfg in sorted(_HLP_WIKI)
        )

        return ConcatPointwiseDistillationSamples.C(sources=sources)
