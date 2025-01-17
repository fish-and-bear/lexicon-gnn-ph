from typing import TypedDict, Literal, Dict, List, Optional, Union
from dataclasses import dataclass

# Type definitions
class WritingSystemDetails(TypedDict):
    Languages: List[str]
    Period: str
    Status: Literal['Primary', 'Historical', 'Limited Use', 'Historical/Revival', 'Historical/Limited Use', 'Limited Use']

class WritingSystemInfo(TypedDict):
    script: str
    category: str
    period: str
    status: str

class RegionInfo(TypedDict):
    name: str
    subregions: Dict[str, List[str]]

# Custom exceptions
class LanguageSystemError(Exception):
    """Base exception for language system errors."""
    pass

class InvalidLanguageCode(LanguageSystemError):
    """Raised when an invalid language code is used."""
    pass

class InvalidLanguageMapping(LanguageSystemError):
    """Raised when language mappings are inconsistent."""
    pass

@dataclass
class LanguageMetadata:
    code: str
    name: str
    family: List[str]
    regions: List[str]
    writing_systems: List[WritingSystemInfo] 