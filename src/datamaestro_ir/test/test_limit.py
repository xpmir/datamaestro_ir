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

    # Test random access by sanitized external ID
    doc = ds.documents.document_ext("Geneva_Durben")
    assert doc is not None
    assert doc["id"] == "Geneva_Durben"
    assert "Geneva Durben likes" in doc["text_item"].body

    # Verify no doc ID contains spaces
    for d in ds.documents.iter_documents():
        assert " " not in d["id"], f"Doc ID contains space: {d['id']}"

    # 2. Queries / Topics
    assert isinstance(ds.topics, BeirParquetTopics)
    topics = list(ds.topics.iter())
    assert len(topics) == 1000
    assert topics[0]["id"] == "query_0"
    assert "Who likes" in topics[0]["text_item"].text

    # 3. Qrels / Assessments
    assessments = list(ds.assessments.iter())
    assert len(assessments) == 1000
    assert assessments[0].topic_id == "query_0"
    assert len(assessments[0].assessments) == 2
    doc_ids = {a.doc_id for a in assessments[0].assessments}
    assert "Geneva_Durben" in doc_ids
    assert "Dorathea_Bastress" in doc_ids

    # Verify no assessed doc ID contains spaces
    for assessed in assessments:
        for a in assessed.assessments:
            assert " " not in a.doc_id, f"Assessed doc ID contains space: {a.doc_id}"


def test_limit_trec_run_compatibility():
    """Verify LIMIT document IDs produce standard 6-column TREC runs that parse cleanly."""
    import tempfile
    from pathlib import Path
    from datamaestro_ir.interfaces.trec import parse_run, write_run_dict

    ds = prepare_dataset("co.huggingface.limit_small", download=True)
    doc = next(ds.documents.iter_documents())
    run = {"query_0": {doc["id"]: 2.5}}

    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        run_path = Path(f.name)
        write_run_dict(run, run_path)

        with run_path.open() as fp:
            line = fp.readline()
        tokens = line.strip().split()
        assert len(tokens) == 6, f"Expected 6 tokens in TREC run line, got {len(tokens)}: {line}"
        assert tokens[2] == doc["id"]
        assert parse_run(run_path) == {"query_0": {doc["id"]: "2.5"}}

