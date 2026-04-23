"""LightOn embeddings-pre-training dataset family.

Registers the HuggingFace dataset
``https://huggingface.co/datasets/lightonai/embeddings-pre-training`` as a
single variant family. Callers select a specific variant via a
query-style selector on the dataset id::

    datamaestro prepare 'ai.lighton.embeddings_pre_training[name=agnews]'

All 73 HF configs are exposed through the ``name`` axis; additional axes
cover loading mode and the dataset's recipe knobs (``drop`` filter,
``duplicate`` filter, ``similarity`` floor, and top-percentile subset).
See :class:`datamaestro.variants.AxesVariants` for the query syntax.

The recipe from the mGTE technical report lives in a sibling module
(``denseon_lateon``) because it UNIONs multiple configs with
config-specific filter rules and is therefore structurally a separate
dataset.
"""

from __future__ import annotations

from typing import Optional

from datamaestro.definitions import dataset, Dataset, datatags, datatasks
from datamaestro.variants import AxesVariants, Axis

from datamaestro_ir.data.lighton import EmbeddingsPreTrainingSamples

REPO_ID = "lightonai/embeddings-pre-training"
URL = f"https://huggingface.co/datasets/{REPO_ID}"

# All 73 configs of lightonai/embeddings-pre-training, resolved once from
# ``https://huggingface.co/api/datasets/lightonai/embeddings-pre-training``.
CONFIGS = [
    "agnews",
    "altlex",
    "amazon_qa",
    "amazon_reviews",
    "arxiv_title_abstract",
    "beir_dbpedia",
    "biorxiv_title_abstract",
    "cc_news_en",
    "cc_news_fr",
    "cnn_dailymail",
    "codesearchnet",
    "eli5",
    "gooaq_qa",
    "hermes",
    "medrxiv_title_abstract",
    "msmarco",
    "mtp",
    "nllb_eng_fra",
    "npr",
    "paq",
    "quora",
    "reddit",
    "reddit_body_comment",
    "s2orc_abstract_citation",
    "s2orc_citation_titles",
    "s2orc_title_abstract",
    "sentence_compression",
    "simplewiki",
    "stackexchange_body_body",
    "stackexchange_duplicate_questions",
    "stackexchange_qa",
    "stackexchange_title_body",
    "stackoverflow_title_body",
    "webfaq_eng",
    "webfaq_fra",
    "wikihow",
    "wikianswers",
    "wikipedia_en",
    "wikipedia_en_mgte",
    "wikipedia_hlp_dl",
    "wikipedia_hlp_cm",
    "wikipedia_fr",
    "wikipedia_it",
    "wikipedia_es",
    "wikipedia_de",
    "wikipedia_ar",
    "wikipedia-pt",
    "wikipedia-sv",
    "wikipedia-no",
    "yahoo_answer",
    "yahoo_qa",
    "yahoo_question_body",
    "fw-edu",
    "fw2-arb_Arab",
    "fw2-ces_Latn",
    "fw2-cmn_Hani",
    "fw2-dan_Latn",
    "fw2-deu_Latn",
    "fw2-ell_Grek",
    "fw2-fas_Arab",
    "fw2-fra_Latn",
    "fw2-hun_Latn",
    "fw2-ind_Latn",
    "fw2-ita_Latn",
    "fw2-jpn_Jpan",
    "fw2-nld_Latn",
    "fw2-pol_Latn",
    "fw2-por_Latn",
    "fw2-rus_Cyrl",
    "fw2-spa_Latn",
    "fw2-swe_Latn",
    "fw2-tur_Latn",
    "fw2-vie_Latn",
]


class EmbeddingsVariants(AxesVariants):
    """Variant space for ``lightonai/embeddings-pre-training``."""

    name = Axis(CONFIGS)
    """HuggingFace config name — selects one of the 73 source corpora."""

    streaming = Axis([False, True], default=True, type=bool, in_id=False)
    """Pure loading-mode flag (same data either way); excluded from the
    formatted selector and dataset id. The underlying
    ``HuggingFaceDataset.streaming`` field is also ``Meta``, so
    experimaestro's identity hash already ignores it."""

    filter_drop = Axis([True, False], default=True, type=bool)
    """When ``True``, skip rows with ``drop=True`` (the upstream dataset's
    recommended pre-training subset)."""

    filter_duplicate = Axis([True, False], default=True, type=bool)
    """When ``True``, skip rows whose ``duplicate`` column is not null."""

    min_similarity = Axis(type=Optional[float], default=None)
    """Minimum teacher similarity (inclusive). Rows below are skipped."""

    top_percentile = Axis(type=Optional[float], default=None)
    """Keep only the top fraction of rows by similarity (e.g. ``0.35``
    for the FineWeb-Edu top-35% recipe). Threshold is estimated from a
    reservoir sample, so it works in streaming mode."""


# Family — id derived from the package path (``id=""`` drops the class
# name so the registered id is exactly ``ai.lighton.embeddings_pre_training``).
@datatags("information retrieval", "distillation", "pre-training")
@datatasks("learning to rank")
@dataset(id="", url=URL, variants=EmbeddingsVariants)
class EmbeddingsPreTraining(Dataset):
    """LightOn ``embeddings-pre-training`` teacher-scored pretraining pairs.

    Teacher-scored ``(query, document, similarity)`` pairs across 73
    source corpora. Variant axes: ``name`` (config), ``streaming``,
    ``filter_drop``, ``filter_duplicate``, ``min_similarity``,
    ``top_percentile``. See :class:`EmbeddingsPreTrainingSamples` for
    filter semantics.
    """

    def config(self, **kw) -> EmbeddingsPreTrainingSamples:
        return EmbeddingsPreTrainingSamples.C(
            repo_id=REPO_ID,
            split="train",
            **kw,
        )
