import pytest
from backend.language_systems import LanguageSystem
from backend.language_types import InvalidLanguageCode, InvalidLanguageMapping

def test_standardize_code():
    assert LanguageSystem.TAGALOG.value == "tagalog"
    assert LanguageSystem.get_primary() == "filipino"
    assert LanguageSystem.is_valid("tagalog") == True
    assert LanguageSystem.is_valid("invalid") == False

def test_invalid_language_code():
    with pytest.raises(InvalidLanguageCode) as exc_info:
        LanguageSystem.get_related("invalid_language")
    assert "Invalid language code: invalid_language" in str(exc_info.value)

def test_family_tree():
    related = LanguageSystem.get_related("tagalog")
    assert "filipino" in related
    assert len(related) > 0

def test_writing_systems():
    script = LanguageSystem.get_script("tagalog")
    assert script == "latin"
    script = LanguageSystem.get_script("baybayin")
    assert script == "baybayin"

def test_regions():
    all_languages = LanguageSystem.get_all()
    assert "tagalog" in all_languages
    assert "filipino" in all_languages 