"""Dataset-specific subclasses for LightOn datasets."""

from __future__ import annotations

from functools import cached_property
import logging
import random
from typing import Iterator, Optional

from experimaestro import Meta, Param, field

from datamaestro.data.huggingface import HuggingFaceDataset
from datamaestro.definitions import datatags
from datamaestro.download.huggingface import hf_download_and_prepare
from datamaestro.requirements import require
from .base import ScoredDocument, SimpleTextItem, TextRecord
from .distillation import (
    ListwiseDistillationSample,
    ListwiseDistillationSamples,
    PointwiseDistillationSample,
)
from .huggingface import HuggingFacePointwiseDistillationSamples


logger = logging.getLogger(__name__)


@datatags("pre-training")
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

    shuffle_seed: Param[Optional[int]] = field(default=None, ignore_default=True)
    """If set, pass through to the HuggingFace dataset's ``.shuffle(seed=…)``
    so iteration order is randomised per source. Meant to pair with a
    cross-source random interleave (e.g.
    :class:`~datamaestro_ir.data.distillation.RandomInterleavePointwiseDistillationSamples`)
    for fully shuffled training streams. ``Param`` — different seeds
    produce different sample orderings, so this affects identity."""

    shuffle_buffer_size: Param[int] = field(default=10_000, ignore_default=True)
    """Buffer size for streaming-mode shuffle (HF approximates a full
    permutation with a rolling buffer of this size). ``Param`` because
    the approximation quality — and thus the exact stream of rows — is
    a function of the buffer size."""

    def _iter_data(self):
        ds = self.data
        if self.shuffle_seed is not None:
            if self.streaming:
                ds = ds.shuffle(
                    seed=self.shuffle_seed,
                    buffer_size=self.shuffle_buffer_size,
                )
            else:
                ds = ds.shuffle(seed=self.shuffle_seed)
        return iter(ds)

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        threshold = self._resolve_percentile_threshold()
        for row in self._iter_data():
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


@datatags("fine-tuning")
class EmbeddingsFineTuningSamples(HuggingFaceDataset, ListwiseDistillationSamples):
    """``lightonai/embeddings-fine-tuning-filtered-en`` contrastive fine-tuning samples.

    Each split (e.g. ``fiqa_en``, ``msmarco_en``) is stored across three relational
    HuggingFace configs (``queries``, ``documents``, ``scores``). When materialized
    via :attr:`data`, the three tables are joined into a unified Arrow dataset with
    columns:
      - ``query`` (str)
      - ``positive`` (str)
      - ``negative_0`` .. ``negative_{num_negatives - 1}`` (str)
      - ``teacher_scores`` (list[float])
    """

    num_negatives: Param[int] = field(default=10, ignore_default=True)
    """Number of mined hard negatives per query to retain from the scores table."""

    def __validate__(self):
        super().__validate__()
        if self.num_negatives <= 0:
            raise ValueError(
                f"[{self.__class__.__name__}] num_negatives must be > 0, got {self.num_negatives}"
            )

    def download(self):
        """Materialise the ``queries``, ``documents``, and ``scores`` tables for split ``self.name``."""
        super(HuggingFaceDataset, self).download()
        require(self.__class__, what=f"{self.__class__.__name__} ({self.source})")

        if self.streaming or self.local_path is not None:
            return

        if not self.name:
            raise ValueError(
                f"[{self.__class__.__name__}] 'name' (split name, e.g. 'fiqa_en') must be set to download."
            )

        for table in ("queries", "documents", "scores"):
            hf_download_and_prepare(
                self.source,
                table,
                split=self.name,
                revision=self.revision,
            )

    def _load_table(self, table: str):
        if self.streaming:
            from datasets import load_dataset

            return load_dataset(
                self.source,
                table,
                data_files=f"{table}/{self.name}-*",
                split="train",
                revision=self.revision,
                streaming=True,
                verification_mode="no_checks",
            )

        builder = hf_download_and_prepare(
            self.source,
            table,
            split=self.name,
            revision=self.revision,
        )
        return builder.as_dataset(split=self.name)

    @cached_property
    def data(self):
        require(self.__class__, what=f"{self.__class__.__name__} ({self.source})")

        if self.local_path is not None:
            from datasets import load_from_disk

            try:
                return load_from_disk(str(self.local_path))
            except Exception:
                from datasets import load_dataset

                return load_dataset(str(self.local_path))

        if not self.name:
            raise ValueError(
                f"[{self.__class__.__name__}] 'name' (split name, e.g. 'fiqa_en') must be set when local_path is None."
            )

        logger.info(
            "[EmbeddingsFineTuningSamples] Loading and joining tables (queries, documents, scores) for split '%s' from '%s'...",
            self.name,
            self.source,
        )
        queries_ds = self._load_table("queries")
        documents_ds = self._load_table("documents")
        scores_ds = self._load_table("scores")

        queries_map = dict(zip(queries_ds["query_id"], queries_ds["query"]))
        documents_map = dict(zip(documents_ds["document_id"], documents_ds["document"]))
        num_negatives = self.num_negatives
        split_name = self.name

        def _join_batch(batch: dict) -> dict:
            q_texts = []
            pos_texts = []
            t_scores = []
            neg_cols = [[] for _ in range(num_negatives)]

            has_rerank = "rerank_scores" in batch
            scores_col = batch["rerank_scores"] if has_rerank else batch.get("scores")

            for idx, (qid, doc_ids) in enumerate(
                zip(batch["query_id"], batch["document_ids"])
            ):
                if len(doc_ids) < num_negatives + 1:
                    raise ValueError(
                        f"[EmbeddingsFineTuningSamples] Split '{split_name}' query_id={qid} has only "
                        f"{len(doc_ids)} document_ids, expected at least {num_negatives + 1}."
                    )
                if qid not in queries_map:
                    raise KeyError(
                        f"[EmbeddingsFineTuningSamples] Split '{split_name}' query_id={qid} missing from queries table."
                    )
                q_texts.append(queries_map[qid])

                pos_id = doc_ids[0]
                if pos_id not in documents_map:
                    raise KeyError(
                        f"[EmbeddingsFineTuningSamples] Split '{split_name}' positive document_id={pos_id} missing from documents table."
                    )
                pos_texts.append(documents_map[pos_id])

                for neg_idx in range(num_negatives):
                    neg_id = doc_ids[1 + neg_idx]
                    if neg_id not in documents_map:
                        raise KeyError(
                            f"[EmbeddingsFineTuningSamples] Split '{split_name}' negative document_id={neg_id} missing from documents table."
                        )
                    neg_cols[neg_idx].append(documents_map[neg_id])

                if scores_col is not None and scores_col[idx] is not None:
                    t_scores.append(
                        [float(s) for s in scores_col[idx][: num_negatives + 1]]
                    )
                else:
                    t_scores.append([1.0] + [0.0] * num_negatives)

            result = {
                "query": q_texts,
                "positive": pos_texts,
                "teacher_scores": t_scores,
            }
            for neg_idx in range(num_negatives):
                result[f"negative_{neg_idx}"] = neg_cols[neg_idx]
            return result

        joined_ds = scores_ds.map(
            _join_batch,
            batched=True,
            batch_size=10_000,
            remove_columns=scores_ds.column_names,
            desc=f"Joining LightOn fine-tuning split ({self.name})",
        )
        logger.info(
            "[EmbeddingsFineTuningSamples] Joined split '%s': %d rows (%d negatives/query).",
            self.name,
            len(joined_ds),
            num_negatives,
        )
        return joined_ds

    def _build_sample(self, row: dict) -> ListwiseDistillationSample:
        docs = [
            ScoredDocument(TextRecord(text_item=SimpleTextItem(row["positive"])), 1.0)
        ]
        for neg_idx in range(self.num_negatives):
            docs.append(
                ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[f"negative_{neg_idx}"])),
                    0.0,
                )
            )
        return ListwiseDistillationSample(
            query=TextRecord(text_item=SimpleTextItem(row["query"])),
            documents=docs,
        )

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        for row in self.data:
            yield self._build_sample(row)

