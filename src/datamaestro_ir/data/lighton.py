"""Dataset-specific subclasses for LightOn datasets."""

from __future__ import annotations

import logging
import random
from typing import Iterator, Optional

from experimaestro import Meta, field

from .distillation import PointwiseDistillationSample
from .huggingface import HuggingFacePointwiseDistillationSamples


logger = logging.getLogger(__name__)


class EmbeddingsPreTrainingSamples(HuggingFacePointwiseDistillationSamples):
    """``lightonai/embeddings-pre-training`` pointwise samples with the
    dataset-specific recipe knobs.

    The HuggingFace dataset exposes a ``drop`` bool and a ``duplicate``
    (nullable int) column; the recommended pre-training subset is
    ``drop=False AND duplicate IS NULL``. For the FineWeb-Edu-style
    "top-K%" recipe, set :attr:`top_percentile` to a float in ``(0, 1]``.
    """

    drop_field: Meta[str] = field(default="drop", ignore_default=True)
    """Name of the drop-flag column."""

    duplicate_field: Meta[str] = field(default="duplicate", ignore_default=True)
    """Name of the duplicate-index column."""

    filter_drop: Meta[bool] = field(default=True, ignore_default=True)
    """When True, skip rows where ``drop_field`` is True."""

    filter_duplicate: Meta[bool] = field(default=True, ignore_default=True)
    """When True, skip rows where ``duplicate_field`` is not None."""

    min_similarity: Meta[Optional[float]] = field(default=None, ignore_default=True)
    """Minimum similarity (inclusive). Rows below are skipped."""

    top_percentile: Meta[Optional[float]] = field(default=None, ignore_default=True)
    """If set (e.g. ``0.35``), keep only the top fraction of rows by
    similarity. The threshold is estimated from a reservoir sample (see
    :attr:`percentile_sample_size`) so this works in streaming mode
    without a full first pass — reproduction of the upstream recipe is
    therefore accurate up to sampling error."""

    percentile_sample_size: Meta[int] = field(default=1_000_000, ignore_default=True)
    """Reservoir-sample size used to estimate the ``top_percentile``
    threshold. Larger = more faithful to the exact quantile at the cost
    of a longer warmup."""

    percentile_sample_seed: Meta[int] = field(default=0, ignore_default=True)
    """Seed for the reservoir sampler so threshold estimation is
    deterministic."""

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        threshold = self._resolve_percentile_threshold()
        for row in self.data:
            if self.filter_drop and self._get(row, self.drop_field):
                continue
            if (
                self.filter_duplicate
                and self._get(row, self.duplicate_field) is not None
            ):
                continue
            sim = float(row[self.score_field])
            if self.min_similarity is not None and sim < self.min_similarity:
                continue
            if threshold is not None and sim < threshold:
                continue
            yield self._build_sample(row)

    @staticmethod
    def _get(row, key):
        """Row accessor that tolerates missing keys (HF streaming rows
        sometimes omit optional columns)."""
        try:
            return row[key]
        except (KeyError, IndexError):
            return None

    def _resolve_percentile_threshold(self) -> Optional[float]:
        if self.top_percentile is None:
            return None
        if not (0.0 < self.top_percentile <= 1.0):
            raise ValueError(
                f"top_percentile must be in (0, 1]; got {self.top_percentile!r}"
            )

        logger.info(
            "Estimating top-%.3f similarity threshold from a reservoir "
            "sample of up to %d rows (seed=%d)",
            self.top_percentile,
            self.percentile_sample_size,
            self.percentile_sample_seed,
        )
        rng = random.Random(self.percentile_sample_seed)
        sample: list[float] = []
        k = self.percentile_sample_size
        for idx, row in enumerate(self.data):
            # Respect the drop/duplicate filters when estimating; otherwise
            # the threshold would be biased by rows we won't emit.
            if self.filter_drop and self._get(row, self.drop_field):
                continue
            if (
                self.filter_duplicate
                and self._get(row, self.duplicate_field) is not None
            ):
                continue
            sim = float(row[self.score_field])
            if len(sample) < k:
                sample.append(sim)
            else:
                j = rng.randint(0, idx)
                if j < k:
                    sample[j] = sim

        if not sample:
            logger.warning(
                "No rows available to estimate top_percentile threshold; "
                "returning +inf (nothing will be yielded)."
            )
            return float("inf")

        sample.sort(reverse=True)
        cutoff_index = max(1, int(len(sample) * self.top_percentile))
        threshold = sample[cutoff_index - 1]
        logger.info(
            "top-%.3f threshold estimated at similarity >= %.6f (from %d sampled rows)",
            self.top_percentile,
            threshold,
            len(sample),
        )
        return threshold
