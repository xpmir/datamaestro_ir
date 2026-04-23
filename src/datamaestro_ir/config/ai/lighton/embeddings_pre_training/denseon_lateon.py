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

Variant axes:

- ``seed``: ``null`` (default) concatenates the three groups in order
  (``ConcatPointwise``). A non-null integer switches to
  :class:`~datamaestro_ir.data.distillation.RandomInterleavePointwiseDistillationSamples`
  — uniformly picking a source at each step — and propagates the seed to
  each HuggingFace source's ``.shuffle(seed=…)`` for in-source
  randomisation.
- ``download``: ``false`` (default) streams from the Hub;
  ``true`` downloads each source config to the local HF cache
  (``streaming=False``). Use with care — the full dataset is ~2TB.

Id derived from the package path:
``ai.lighton.embeddings_pre_training.denseon_lateon``.

Note on config names: the reference recipe refers to ``fw_edu``,
``hlp_wikipedia_cm`` and ``hlp_wikipedia_dl``; the actual HuggingFace
config names are ``fw-edu``, ``wikipedia_hlp_cm`` and ``wikipedia_hlp_dl``
(verified against the HF datasets API at dataset-card time).
"""

from __future__ import annotations

from typing import Optional

from datamaestro.definitions import dataset, Dataset, datatags, datatasks
from datamaestro.variants import Axis, AxesVariants

from datamaestro_ir.data.distillation import (
    ConcatPointwiseDistillationSamples,
    PointwiseDistillationSamples,
    RandomInterleavePointwiseDistillationSamples,
)
from datamaestro_ir.data.lighton import EmbeddingsPreTrainingSamples

from . import CONFIGS, REPO_ID, URL


# Config names with special handling in the recipe.
_FW_EDU = "fw-edu"
_HLP_WIKI = {"wikipedia_hlp_cm", "wikipedia_hlp_dl"}

# Standard sources: everything not in the special sets above.
_STANDARD = [c for c in CONFIGS if c != _FW_EDU and c not in _HLP_WIKI]


class DenseonLateonVariants(AxesVariants):
    # ``seed`` changes what the dataset yields (which Concat vs
    # Interleave class, plus in-source shuffle order), so it stays in
    # the id; ``elide_default=True`` drops it when left at ``None`` so
    # the pre-variants id ``…denseon_lateon`` is preserved.
    seed = Axis(type=Optional[int], default=None, elide_default=True)
    # ``download`` only toggles streaming on each source — same data,
    # different loading mode — so it's excluded from the id entirely
    # (``in_id=False``). It still reaches ``config()`` via the resolved
    # kwargs and participates in the per-variant cache key.
    download = Axis([False, True], default=False, type=bool, in_id=False)


@datatags(
    "information retrieval",
    "distillation",
    "pre-training",
    "mgte",
    "recipe",
)
@datatasks("learning to rank")
@dataset(id="", url=URL, variants=DenseonLateonVariants)
class DenseonLateon(Dataset):
    """DenseON-LateON mGTE-style pre-training recipe built by UNIONing
    three groups of ``lightonai/embeddings-pre-training`` configs with
    group-specific filters."""

    def config(
        self,
        seed: Optional[int] = None,
        download: bool = False,
    ) -> PointwiseDistillationSamples:
        streaming = not download
        shuffle_seed = seed  # None ⇒ no per-source shuffle.

        sources = []

        # (1) Standard sources — drop/duplicate filter + similarity >= 3.
        sources.extend(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=cfg,
                split="train",
                streaming=streaming,
                filter_drop=True,
                filter_duplicate=True,
                min_similarity=3.0,
                shuffle_seed=shuffle_seed,
            )
            for cfg in _STANDARD
        )

        # (2) FineWeb-Edu — top 35% by similarity, no drop/duplicate filter.
        sources.append(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=_FW_EDU,
                split="train",
                streaming=streaming,
                filter_drop=False,
                filter_duplicate=False,
                top_percentile=0.35,
                shuffle_seed=shuffle_seed,
            )
        )

        # (3) Wikipedia HLP splits — included as-is.
        sources.extend(
            EmbeddingsPreTrainingSamples.C(
                repo_id=REPO_ID,
                name=cfg,
                split="train",
                streaming=streaming,
                filter_drop=False,
                filter_duplicate=False,
                shuffle_seed=shuffle_seed,
            )
            for cfg in sorted(_HLP_WIKI)
        )

        if seed is None:
            return ConcatPointwiseDistillationSamples.C(sources=sources)
        return RandomInterleavePointwiseDistillationSamples.C(
            sources=sources,
            seed=seed,
        )
