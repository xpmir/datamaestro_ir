"""Ettin Reranker v1 training dataset.

Registers the HuggingFace dataset
``https://huggingface.co/datasets/cross-encoder/ettin-reranker-v1-data``
as a parameterized variant family.
"""

from __future__ import annotations

from typing import Optional

from datamaestro.definitions import dataset, Dataset
from datamaestro.variants import AxesVariants, Axis
from datamaestro_ir.data.huggingface import HuggingFacePointwiseDistillationSamples

REPO_ID = "cross-encoder/ettin-reranker-v1-data"
URL = f"https://huggingface.co/datasets/{REPO_ID}"

# The 39 configurations of cross-encoder/ettin-reranker-v1-data
CONFIGS = [
    "agnews", "altlex", "amazon_qa", "amazon_reviews", "arxiv_title_abstract",
    "beir_dbpedia", "biorxiv_title_abstract", "cc_news_en", "cnn_dailymail",
    "fw_edu", "gooaq_qa", "medrxiv_title_abstract", "msmarco", "mtp", "npr",
    "paq", "quora", "reddit", "reddit_body_comment", "rerank_fever",
    "rerank_fiqa", "rerank_hotpotqa", "rerank_msmarco", "rerank_nq",
    "rerank_squadv2", "rerank_trivia", "s2orc_abstract_citation",
    "s2orc_citation_titles", "s2orc_title_abstract", "stackexchange_body_body",
    "stackexchange_duplicate_questions", "stackexchange_qa",
    "stackexchange_title_body", "stackoverflow_title_body", "wikianswers",
    "wikihow", "yahoo_answer", "yahoo_qa", "yahoo_question_body"
]


class EttinVariants(AxesVariants):
    """Variant space for ``cross-encoder/ettin-reranker-v1-data``."""

    name = Axis(CONFIGS)
    """HuggingFace config name."""

    streaming = Axis([False, True], default=True, type=bool, in_id=False)
    """Streaming mode flag."""


@dataset(id="", url=URL, variants=EttinVariants)
class EttinRerankerV1Data(Dataset):
    """Ettin Reranker v1 teacher-scored pointwise triples."""

    def config(self, **kw) -> HuggingFacePointwiseDistillationSamples:
        import logging
        if kw.get("streaming", True):
            logging.warning(
                "Using streaming mode, will not download the dataset. "
                "Use streaming=false (e.g. cross_encoder.ettin_reranker_v1_data[...,streaming=false]) to download."
            )
        return HuggingFacePointwiseDistillationSamples.C(
            repo_id=REPO_ID,
            split="train",
            score_field="label",  # Map similarity score to "label" column
            **kw,
        )
