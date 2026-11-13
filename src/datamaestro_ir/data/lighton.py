"""Dataset-specific subclasses for LightOn datasets."""

from __future__ import annotations

from functools import cached_property
import logging
import random
from pathlib import Path
from typing import Iterator, Optional

from experimaestro import Meta, Param, field

from datamaestro.data.huggingface import HuggingFaceDataset
from datamaestro.definitions import datatags
from datamaestro.download.huggingface import hf_download_and_prepare, hf_is_prepared
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


class EmbeddingsFineTuningBase(HuggingFaceDataset):
    """Shared loading of a ``lightonai/embeddings-fine-tuning-filtered-<lang>`` split.

    Each split (e.g. ``fiqa_en``, ``msmarco_en``) is stored across three relational
    HuggingFace configs (``queries``, ``documents``, ``scores``); :meth:`_join_tables`
    resolves the ids of the ``scores`` table against the two others.
    """

    num_negatives: Param[int] = field(default=10, ignore_default=True)
    """Number of mined hard negatives per query to retain from the scores table."""

    def __validate__(self):
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
            try:
                hf_download_and_prepare(
                    self.source,
                    table,
                    split=self.name,
                    revision=self.revision,
                )
            except Exception as exc:
                from datamaestro.definitions import DownloadError

                raise DownloadError(
                    f"Could not download HuggingFace dataset {self.source} "
                    f"(table={table!r}, split={self.name!r})"
                ) from exc

    def is_prepared(self) -> bool:
        """Whether all three tables (``queries``, ``documents``, ``scores``) for ``self.name`` are cached."""
        if not super(HuggingFaceDataset, self).is_prepared():
            return False
        if self.streaming:
            return True
        if self.local_path is not None:
            return Path(self.local_path).exists()
        if not self.name:
            return False
        return all(
            hf_is_prepared(self.source, table, split=self.name)
            for table in ("queries", "documents", "scores")
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

    def _load_local(self):
        """The dataset saved at :attr:`local_path` (e.g. by a preprocessing task)."""
        from datasets import load_from_disk

        try:
            return load_from_disk(str(self.local_path))
        except Exception:
            from datasets import load_dataset

            return load_dataset(str(self.local_path))

    def _join_tables(self):
        """Loads the three tables of split :attr:`name` and resolves their ids.

        Returns ``(query_col, doc_indices, documents_table, scores_table,
        fingerprints)``: the query text of each ``scores`` row, the indices into
        ``documents_table`` of the first ``num_negatives + 1`` documents of each
        row, *position-major* (all positives, then all first negatives, ...), and
        the fingerprints of the three source datasets.
        """
        if not self.name:
            raise ValueError(
                f"[{self.__class__.__name__}] 'name' (split name, e.g. 'fiqa_en') must be set when local_path is None."
            )

        logger.info(
            "[%s] Loading and joining tables (queries, documents, scores) for split '%s' from '%s'...",
            self.__class__.__name__,
            self.name,
            self.source,
        )
        queries_ds = self._load_table("queries")
        documents_ds = self._load_table("documents")
        scores_ds = self._load_table("scores")

        import pyarrow as pa
        import pyarrow.compute as pc

        queries_table: pa.Table = queries_ds.data.table
        documents_table: pa.Table = documents_ds.data.table
        scores_table: pa.Table = scores_ds.data.table
        num_negatives = self.num_negatives
        split_name = self.name
        num_rows = len(scores_table)

        if num_rows > 0:
            doc_list_lens = pc.list_value_length(scores_table["document_ids"])
            min_len = pc.min(doc_list_lens).as_py()
            if min_len is None or min_len < num_negatives + 1:
                raise ValueError(
                    f"[{self.__class__.__name__}] Split '{split_name}' has a row with only "
                    f"{min_len} document_ids, expected at least {num_negatives + 1}."
                )

        q_indices = pc.index_in(
            scores_table["query_id"], value_set=queries_table["query_id"]
        )
        if q_indices.null_count > 0:
            raise KeyError(
                f"[{self.__class__.__name__}] Split '{split_name}' has "
                f"{q_indices.null_count} query_id(s) missing from queries table."
            )
        query_col = queries_table["query"].take(q_indices)

        # Extract positive (index 0) and negatives (indices 1..num_negatives) and
        # resolve all document_ids against documents_table in a single hash-table pass.
        per_col_doc_ids = [
            pc.list_element(scores_table["document_ids"], i)
            for i in range(num_negatives + 1)
        ]
        all_doc_ids = pa.chunked_array(
            [chunk for col in per_col_doc_ids for chunk in col.chunks]
        )
        all_doc_indices = pc.index_in(
            all_doc_ids, value_set=documents_table["document_id"]
        )
        if all_doc_indices.null_count > 0:
            raise KeyError(
                f"[{self.__class__.__name__}] Split '{split_name}' has "
                f"{all_doc_indices.null_count} document_id(s) missing from documents table."
            )
        fingerprints = (
            scores_ds._fingerprint,
            queries_ds._fingerprint,
            documents_ds._fingerprint,
        )
        return query_col, all_doc_indices, documents_table, scores_table, fingerprints


@datatags("fine-tuning")
class EmbeddingsFineTuningSamples(
    EmbeddingsFineTuningBase, ListwiseDistillationSamples
):
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

    @cached_property
    def data(self):
        require(self.__class__, what=f"{self.__class__.__name__} ({self.source})")

        if self.local_path is not None:
            return self._load_local()

        import pyarrow as pa
        import pyarrow.compute as pc
        from datasets import Dataset
        from datasets.fingerprint import Hasher
        from datasets.table import InMemoryTable

        query_col, all_doc_indices, documents_table, scores_table, fingerprints = (
            self._join_tables()
        )
        num_negatives = self.num_negatives
        num_rows = len(scores_table)

        positive_col = documents_table["document"].take(
            all_doc_indices.slice(0, num_rows)
        )
        negative_cols = {
            f"negative_{neg_idx}": documents_table["document"].take(
                all_doc_indices.slice((neg_idx + 1) * num_rows, num_rows)
            )
            for neg_idx in range(num_negatives)
        }

        score_col_name = (
            "rerank_scores"
            if "rerank_scores" in scores_table.column_names
            else ("scores" if "scores" in scores_table.column_names else None)
        )
        if score_col_name is not None and scores_table[score_col_name].null_count == 0:
            teacher_scores_col = pc.list_slice(
                scores_table[score_col_name], 0, num_negatives + 1
            )
        else:
            default_row = [1.0] + [0.0] * num_negatives
            if score_col_name is None:
                teacher_scores_col = pa.array(
                    [default_row] * num_rows, type=pa.list_(pa.float64())
                )
            else:
                raw_scores = scores_table[score_col_name].to_pylist()
                teacher_scores_col = pa.array(
                    [
                        [float(s) for s in row[: num_negatives + 1]]
                        if row is not None
                        else default_row
                        for row in raw_scores
                    ],
                    type=pa.list_(pa.float64()),
                )

        joined_table = pa.table(
            {
                "query": query_col,
                "positive": positive_col,
                "teacher_scores": teacher_scores_col,
                **negative_cols,
            }
        )
        fingerprint = Hasher.hash((*fingerprints, num_negatives))
        joined_ds = Dataset(InMemoryTable(joined_table), fingerprint=fingerprint)
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


@datatags("fine-tuning", "pointwise")
class EmbeddingsFineTuningPointwiseSamples(
    EmbeddingsFineTuningBase, HuggingFacePointwiseDistillationSamples
):
    """``lightonai/embeddings-fine-tuning-filtered-<lang>`` as pointwise
    (query, document, teacher score) distillation samples.

    Each ``scores`` row (a query, its positive and ``num_negatives`` mined hard
    negatives) gives ``num_negatives + 1`` rows, with columns:
      - ``query`` (str)
      - ``document`` (str)
      - ``score`` (float): the cross-encoder teacher score ``rerank_scores[i]``
      - ``position`` (int): ``i``, 0 for the positive

    Only ``rerank_scores`` is used as teacher: the split must have that column,
    with no missing value (there is no fallback to the bi-encoder ``scores`` or to
    binary labels). Rows are position-major (all positives, then all first
    negatives, ...): shuffle before training.
    """

    score_field: Meta[str] = field(default="score", ignore_default=True, overrides=True)
    """Name of the column holding the teacher score."""

    @cached_property
    def data(self):
        require(self.__class__, what=f"{self.__class__.__name__} ({self.source})")

        if self.local_path is not None:
            return self._load_local()
        if self.streaming:
            raise ValueError(
                f"[{self.__class__.__name__}] the pointwise view joins whole tables: use streaming=False"
            )

        import pyarrow as pa
        import pyarrow.compute as pc
        from datasets import Dataset
        from datasets.fingerprint import Hasher
        from datasets.table import InMemoryTable

        query_col, all_doc_indices, documents_table, scores_table, fingerprints = (
            self._join_tables()
        )
        import numpy as np

        n_docs = self.num_negatives + 1
        num_rows = len(scores_table)

        if "rerank_scores" not in scores_table.column_names:
            raise KeyError(
                f"[{self.__class__.__name__}] Split '{self.name}' has no 'rerank_scores' column"
            )
        rerank = scores_table["rerank_scores"]
        if rerank.null_count:
            raise ValueError(
                f"[{self.__class__.__name__}] Split '{self.name}' has {rerank.null_count} rows with missing rerank_scores"
            )
        if num_rows and pc.min(pc.list_value_length(rerank)).as_py() < n_docs:
            raise ValueError(
                f"[{self.__class__.__name__}] Split '{self.name}' has a row with fewer than "
                f"{n_docs} rerank_scores"
            )
        per_doc_scores = (
            [pc.list_element(rerank, i) for i in range(n_docs)] if num_rows else []
        )
        missing = sum(col.null_count for col in per_doc_scores)
        if missing:
            raise ValueError(
                f"[{self.__class__.__name__}] Split '{self.name}' has {missing} missing rerank_scores"
            )

        table = pa.table(
            {
                "query": pa.chunked_array(
                    [chunk for _ in range(n_docs) for chunk in query_col.chunks],
                    type=query_col.type,
                ),
                "document": documents_table["document"].take(all_doc_indices),
                "score": pa.chunked_array(
                    [chunk for col in per_doc_scores for chunk in col.chunks]
                ).cast(pa.float32())
                if num_rows
                else pa.array([], type=pa.float32()),
                "position": pa.array(
                    np.repeat(np.arange(n_docs, dtype=np.int8), num_rows)
                ),
            }
        )
        fingerprint = Hasher.hash((*fingerprints, self.num_negatives, "pointwise"))
        ds = Dataset(InMemoryTable(table), fingerprint=fingerprint)
        logger.info(
            "[%s] Split '%s': %d queries -> %d (query, document) rows.",
            self.__class__.__name__,
            self.name,
            num_rows,
            len(ds),
        )
        return ds
