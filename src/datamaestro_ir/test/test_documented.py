from pathlib import Path
import pytest
from datamaestro.test.checks import DatamaestroAnalyzer


def test_documented():
    """Test if every configuration is documented"""
    pytest.importorskip("docutils")
    pytest.importorskip("sphobjinv")
    pytest.importorskip("sphinx")

    doc_path = Path(__file__).parents[3] / "docs" / "source" / "index.rst"
    analyzer = DatamaestroAnalyzer(
        doc_path, set(["datamaestro_ir"]), set(["datamaestro_ir.test"])
    )

    analyzer.analyze()
    analyzer.report()
    analyzer.assert_valid_documentation()
