"""Documents' languages metadata"""

from pathlib import Path


from datamaestro_ir.data.beir import BeirDocumentStore


def test_languages_is_metadata():
    """Setting the languages does not change the configuration identifier"""
    plain = BeirDocumentStore.C(id="test", path=Path("/tmp/store"))
    with_languages = BeirDocumentStore.C(
        id="test", path=Path("/tmp/store"), languages=["fr"]
    )
    assert with_languages.languages == ["fr"]
    assert plain.languages is None
    assert plain.__identifier__().all == with_languages.__identifier__().all
