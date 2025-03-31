from enum import Enum
from typing import Dict, List, Optional

class LanguageSystem(Enum):
    """Enumeration of supported language systems."""
    FILIPINO = "filipino"
    TAGALOG = "tagalog"
    CEBUANO = "cebuano"
    HILIGAYNON = "hiligaynon"
    BICOL = "bicol"
    PAMPANGO = "pampango"
    PANGASINAN = "pangasinan"
    WARAY = "waray"
    KAPAMPANGAN = "kapampangan"
    BAYBAYIN = "baybayin"
    
    @classmethod
    def get_all(cls) -> List[str]:
        """Get all language system names."""
        return [member.value for member in cls]
    
    @classmethod
    def get_primary(cls) -> str:
        """Get the primary language system (Filipino)."""
        return cls.FILIPINO.value
    
    @classmethod
    def is_valid(cls, language: str) -> bool:
        """Check if a language system is valid."""
        return language in cls.get_all()
    
    @classmethod
    def get_related(cls, language: str) -> List[str]:
        """Get related language systems for a given language."""
        # Define language relationships
        relationships: Dict[str, List[str]] = {
            cls.FILIPINO.value: [cls.TAGALOG.value],
            cls.TAGALOG.value: [cls.FILIPINO.value],
            cls.CEBUANO.value: [],
            cls.HILIGAYNON.value: [],
            cls.BICOL.value: [],
            cls.PAMPANGO.value: [cls.KAPAMPANGAN.value],
            cls.KAPAMPANGAN.value: [cls.PAMPANGO.value],
            cls.PANGASINAN.value: [],
            cls.WARAY.value: [],
            cls.BAYBAYIN.value: [cls.TAGALOG.value, cls.FILIPINO.value]
        }
        return relationships.get(language, [])
    
    @classmethod
    def get_script(cls, language: str) -> Optional[str]:
        """Get the script system for a language."""
        scripts: Dict[str, str] = {
            cls.BAYBAYIN.value: "baybayin",
            cls.FILIPINO.value: "latin",
            cls.TAGALOG.value: "latin",
            cls.CEBUANO.value: "latin",
            cls.HILIGAYNON.value: "latin",
            cls.BICOL.value: "latin",
            cls.PAMPANGO.value: "latin",
            cls.KAPAMPANGAN.value: "latin",
            cls.PANGASINAN.value: "latin",
            cls.WARAY.value: "latin"
        }
        return scripts.get(language) 