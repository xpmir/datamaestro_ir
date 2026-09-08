"""Tests for the LIMIT dataset integration in datamaestro-ir."""

import pytest
from datamaestro import prepare_dataset
from datamaestro.context import find_dataset
from datamaestro_ir.data import Adhoc
from datamaestro_ir.data.beir import (
    BeirDocumentStore,
    BeirParquetTopics,
    BeirParquetAssessments,
)


@pytest.mark.parametrize(
    "dataset_id",
    [
        "co.huggingface.limit",
        "co.huggingface.limit_small",
        "co.huggingface.limit-small",
        "co.huggingface.limit.documents",
        "co.huggingface.limit.queries",
        "co.huggingface.limit.qrels",
        "co.huggingface.limit_small.documents",
        "co.huggingface.limit_small.queries",
        "co.huggingface.limit_small.qrels",
    ],
)
def test_limit_registration_discovery(dataset_id: str):
    """Test that datamaestro can discover and resolve all LIMIT dataset IDs."""
    wrapper = find_dataset(dataset_id)
    assert wrapper is not None
    assert dataset_id in wrapper.aliases or wrapper.id == dataset_id


def test_limit_small_end_to_end():
    """Test downloading, building docstore, and querying LIMIT-small."""
    ds = prepare_dataset("co.huggingface.limit_small", download=True)
    assert isinstance(ds, Adhoc)
    assert ds.id == "co.huggingface.limit_small"

    # 1. Documents
    assert isinstance(ds.documents, BeirDocumentStore)
    assert ds.documents.documentcount == 46

    # Test random access by external ID
    doc = ds.documents.document_ext("Geneva Durben")
    assert doc is not None
    assert doc["id"] == "Geneva Durben"
    assert "Geneva Durben likes" in doc["text_item"].body

    # 2. Queries / Topics
    assert isinstance(ds.topics, BeirParquetTopics)
    topics = list(ds.topics.iter())
    assert len(topics) == 1000
    assert topics[0]["id"] == "query_0"
    assert "Who likes" in topics[0]["text_item"].text

    # 3. Qrels / Assessments
    assert isinstance(ds.assessments, BeirParquetAssessments)
    assessments = list(ds.assessments.iter())
    assert len(assessments) == 1000
    assert assessments[0].topic_id == "query_0"
    assert len(assessments[0].assessments) == 2
    doc_ids = {a.doc_id for a in assessments[0].assessments}
    assert "Geneva Durben" in doc_ids
    assert "Dorathea Bastress" in doc_ids
