import pytest
from backend.language_systems import LanguageSystem
from backend.language_types import InvalidLanguageCode, InvalidLanguageMapping

@pytest.fixture
def ls():
    return LanguageSystem()

def test_standardize_code(ls):
    assert ls.standardize_code("tag") == "Tagálog"
    assert ls.standardize_code("") == "-"
    assert ls.standardize_code("INVALID") == "INVALID"

def test_invalid_language_code(ls):
    with pytest.raises(InvalidLanguageCode) as exc_info:
        ls.get_family_tree("invalid_language")
    assert "Invalid language code: invalid_language" in str(exc_info.value)

def test_family_tree(ls):
    tree = ls.get_family_tree("Tagálog")
    assert "Austronesian" in tree
    assert "Philippine" in tree

def test_writing_systems(ls):
    systems = ls.get_writing_systems("Tagálog")
    assert any(s["script"] == "Baybayin" for s in systems)
    assert any(s["script"] == "Latin" for s in systems)

def test_regions(ls):
    regions = ls.get_regions("Tagálog")
    assert "Luzon (Metro Manila)" in regions 