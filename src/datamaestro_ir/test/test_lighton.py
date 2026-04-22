"""Tests for the LightOn embeddings-pre-training pipeline.

Covers:
- :class:`EmbeddingsPreTrainingSamples` filter semantics (drop /
  duplicate / min_similarity / top_percentile) against a synthetic row
  stream.
- :class:`ConcatPointwiseDistillationSamples` preserves source order
  (UNION ALL).
- The registered family
  (``ai.lighton.embeddings_pre_training``) exposes the declared
  variant axes and auto-derives the id from the package path.
- The ``denseon_lateon`` recipe composes the expected per-config filter
  rules (standard / fw-edu / wikipedia_hlp).
"""

from __future__ import annotations

from typing import Iterator, List

import pytest

from datamaestro_ir.data.base import SimpleTextItem
from datamaestro_ir.data.distillation import (
    ConcatPointwiseDistillationSamples,
    PointwiseDistillationSample,
    PointwiseDistillationSamples,
)
from datamaestro_ir.data.lighton import EmbeddingsPreTrainingSamples


# ---- Fake HuggingFace data ------------------------------------------------


def _rows(*rows: dict) -> List[dict]:
    return list(rows)


class _FakeSamples(EmbeddingsPreTrainingSamples):
    """Subclass that short-circuits ``self.data`` to a synthetic row list."""

    # Normally ``self.data`` would load_dataset. We monkey-patch it on
    # a per-test basis by assigning to the instance attribute (works
    # because experimaestro configs are attrs-like).
    pass


def _samples(rows, **kwargs):
    """Instantiate EmbeddingsPreTrainingSamples with a fake row list."""
    kwargs.setdefault("repo_id", "fake/fake")
    kwargs.setdefault("name", "cfg")
    kwargs.setdefault("split", "train")
    cfg = EmbeddingsPreTrainingSamples.C(**kwargs)
    # Attach a synthetic data iterable so __iter__ reads from it.
    cfg.__dict__["data"] = list(rows)
    return cfg


# ---- Filter semantics ----------------------------------------------------


class TestEmbeddingsPreTrainingFilters:
    def test_drop_filter(self):
        rows = [
            {
                "query": "q1",
                "document": "d1",
                "similarity": 5.0,
                "drop": False,
                "duplicate": None,
            },
            {
                "query": "q2",
                "document": "d2",
                "similarity": 5.0,
                "drop": True,
                "duplicate": None,
            },
        ]
        out = list(_samples(rows))
        assert len(out) == 1
        assert out[0].query["text_item"].text == "q1"

    def test_drop_filter_disabled(self):
        rows = [
            {
                "query": "q1",
                "document": "d1",
                "similarity": 5.0,
                "drop": True,
                "duplicate": None,
            },
        ]
        out = list(_samples(rows, filter_drop=False))
        assert len(out) == 1

    def test_duplicate_filter(self):
        rows = [
            {
                "query": "q1",
                "document": "d1",
                "similarity": 5.0,
                "drop": False,
                "duplicate": None,
            },
            {
                "query": "q2",
                "document": "d2",
                "similarity": 5.0,
                "drop": False,
                "duplicate": 0,
            },
        ]
        out = list(_samples(rows))
        assert len(out) == 1
        assert out[0].query["text_item"].text == "q1"

    def test_duplicate_filter_disabled(self):
        rows = [
            {
                "query": "q1",
                "document": "d1",
                "similarity": 5.0,
                "drop": False,
                "duplicate": 5,
            },
        ]
        out = list(_samples(rows, filter_duplicate=False))
        assert len(out) == 1

    def test_min_similarity(self):
        rows = [
            {
                "query": f"q{i}",
                "document": f"d{i}",
                "similarity": float(i),
                "drop": False,
                "duplicate": None,
            }
            for i in [1.0, 2.0, 3.0, 4.0]
        ]
        out = list(_samples(rows, min_similarity=3.0))
        texts = [s.query["text_item"].text for s in out]
        assert texts == ["q3.0", "q4.0"]

    def test_top_percentile_keeps_top_fraction(self):
        # 100 rows with ascending similarity; top 10% keeps the top 10.
        rows = [
            {
                "query": f"q{i}",
                "document": f"d{i}",
                "similarity": float(i),
                "drop": False,
                "duplicate": None,
            }
            for i in range(100)
        ]
        out = list(
            _samples(
                rows,
                top_percentile=0.1,
                percentile_sample_size=500,
                percentile_sample_seed=0,
            )
        )
        # ~10 rows, all with similarity >= 90.
        assert 8 <= len(out) <= 12
        for sample in out:
            assert sample.document.score >= 90.0

    def test_top_percentile_invalid_raises(self):
        rows = [
            {
                "query": "q",
                "document": "d",
                "similarity": 1.0,
                "drop": False,
                "duplicate": None,
            }
        ]
        with pytest.raises(ValueError, match="top_percentile"):
            list(_samples(rows, top_percentile=2.0))

    def test_scored_document_carries_similarity(self):
        rows = [
            {
                "query": "q",
                "document": "d",
                "similarity": 7.5,
                "drop": False,
                "duplicate": None,
            },
        ]
        out = list(_samples(rows))
        assert isinstance(out[0], PointwiseDistillationSample)
        assert out[0].document.score == 7.5
        assert out[0].document.document["text_item"].text == "d"

    def test_missing_optional_keys_tolerated(self):
        """Streaming rows sometimes omit the drop/duplicate columns;
        ``_get`` returns None so filters behave as 'no-op'."""
        rows = [
            {"query": "q", "document": "d", "similarity": 1.0},
        ]
        out = list(_samples(rows))
        assert len(out) == 1


# ---- ConcatPointwiseDistillationSamples ---------------------------------


class _Fixed(PointwiseDistillationSamples):
    """Iterable source returning fixed samples (test helper)."""

    # Can't use Param here because we need to hold raw samples;
    # stash on the instance via __dict__ after construction.

    def __iter__(self) -> Iterator[PointwiseDistillationSample]:
        return iter(self.__dict__["_samples"])


def _fixed(samples):
    inst = _Fixed.C()
    inst.__dict__["_samples"] = list(samples)
    return inst


def _mk(text):
    return PointwiseDistillationSample(
        query={"text_item": SimpleTextItem(f"q-{text}")},
        document=SimpleTextItem(text),
    )


class TestConcatPointwise:
    def test_preserves_order(self):
        a = _fixed([_mk("a1"), _mk("a2")])
        b = _fixed([_mk("b1")])
        c = _fixed([_mk("c1"), _mk("c2")])

        combined = ConcatPointwiseDistillationSamples.C(sources=[a, b, c])
        out = [s.document.text for s in combined]
        assert out == ["a1", "a2", "b1", "c1", "c2"]

    def test_empty_sources(self):
        combined = ConcatPointwiseDistillationSamples.C(sources=[])
        assert list(combined) == []


# ---- Family registration -------------------------------------------------


class TestFamilyRegistration:
    def test_family_id_from_package(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training import (
            EmbeddingsPreTraining,
        )

        assert (
            EmbeddingsPreTraining.__dataset__.id == "ai.lighton.embeddings_pre_training"
        )

    def test_family_exposes_expected_axes(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training import (
            EmbeddingsPreTraining,
        )

        axes = set(EmbeddingsPreTraining.__dataset__.variants.axes)
        assert axes == {
            "name",
            "streaming",
            "filter_drop",
            "filter_duplicate",
            "min_similarity",
            "top_percentile",
        }

    def test_family_registers_73_configs(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training import (
            EmbeddingsPreTraining,
        )

        name_axis = EmbeddingsPreTraining.__dataset__.variants.axes["name"]
        assert len(name_axis.domain) == 73

    def test_family_prepare_for_specific_variant(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training import (
            EmbeddingsPreTraining,
        )

        config = EmbeddingsPreTraining.__dataset__.prepare(
            variant_kwargs={"name": "agnews"}
        )
        # Params forwarded to EmbeddingsPreTrainingSamples
        assert config.name == "agnews"
        # Axis defaults filled in
        assert config.streaming is True
        assert config.filter_drop is True
        assert config.filter_duplicate is True
        assert config.min_similarity is None

    def test_family_rejects_unknown_config(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training import (
            EmbeddingsPreTraining,
        )

        with pytest.raises(ValueError, match="not in axis domain"):
            EmbeddingsPreTraining.__dataset__.prepare(
                variant_kwargs={"name": "does-not-exist"}
            )


# ---- Recipe -------------------------------------------------------------


class TestDenseonLateonRecipe:
    def test_recipe_id(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training.denseon_lateon import (
            DenseonLateon,
        )

        assert (
            DenseonLateon.__dataset__.id
            == "ai.lighton.embeddings_pre_training.denseon_lateon"
        )

    def test_recipe_has_73_sources(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training.denseon_lateon import (
            DenseonLateon,
        )

        config = DenseonLateon.__dataset__.prepare()
        assert len(config.sources) == 73

    def test_standard_sources_have_min_similarity_3(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training.denseon_lateon import (
            DenseonLateon,
        )

        config = DenseonLateon.__dataset__.prepare()
        agnews = next(s for s in config.sources if s.name == "agnews")
        assert agnews.min_similarity == 3.0
        assert agnews.filter_drop is True
        assert agnews.filter_duplicate is True

    def test_fw_edu_uses_top_percentile(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training.denseon_lateon import (
            DenseonLateon,
        )

        config = DenseonLateon.__dataset__.prepare()
        fw = next(s for s in config.sources if s.name == "fw-edu")
        assert fw.top_percentile == 0.35
        assert fw.min_similarity is None
        assert fw.filter_drop is False
        assert fw.filter_duplicate is False

    def test_wikipedia_hlp_is_unfiltered(self):
        from datamaestro_ir.config.ai.lighton.embeddings_pre_training.denseon_lateon import (
            DenseonLateon,
        )

        config = DenseonLateon.__dataset__.prepare()
        for cfg in ("wikipedia_hlp_cm", "wikipedia_hlp_dl"):
            src = next(s for s in config.sources if s.name == cfg)
            assert src.filter_drop is False
            assert src.filter_duplicate is False
            assert src.min_similarity is None
            assert src.top_percentile is None
