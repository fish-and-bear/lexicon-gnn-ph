from typing import TypedDict, Literal, Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

class DictionarySource(Enum):
    KAIKKI = "kaikki"
    KWF = "kwf"
    ROOT_WORDS = "root_words" 
    TAGALOG_WORDS = "tagalog_words"

class SourceInfo(TypedDict):
    source: DictionarySource
    reliability: float
    timestamp: str

class Definition(TypedDict):
    sense: str
    domain: Optional[str]
    usage_notes: Optional[List[str]]
    examples: Optional[List[str]]
    sources: List[SourceInfo]  # Track all sources that provide this definition
    context: Optional[str]
    register: Optional[str]

class WordForm(TypedDict):
    form: str
    tags: List[str]
    pronunciation: Optional[str]
    script: Optional[str]
    sources: List[SourceInfo]  # Track sources for each form

class WordMetadata(TypedDict):
    etymologies: List[Dict[str, Any]]  # Keep all etymological information
    source_languages: List[str]  # All source languages
    pronunciations: Dict[str, List[Dict[str, Any]]]  # Structured pronunciation data
    scripts: Dict[str, List[Dict[str, Any]]]  # All script variations
    regional_usage: List[Dict[str, Any]]  # Regional information
    registers: List[str]  # All register information
    domains: List[str]  # All domain classifications
    grammatical_info: Dict[str, Any]  # Grammatical details
    usage_notes: List[Dict[str, Any]]  # Combined usage notes
    source_specific: Dict[DictionarySource, Any]  # Source-specific metadata

@dataclass
class ConsolidatedEntry:
    word: str
    pos: List[Dict[str, Any]]  # Enhanced POS with source info
    forms: List[WordForm]
    definitions: List[Definition]
    metadata: WordMetadata
    related_words: Dict[str, List[Dict[str, Any]]]  # Enhanced relations
    source_entries: Dict[DictionarySource, Any]
    processing_info: Dict[str, Any]  # Track processing decisions

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

class POSMapping(TypedDict):
    english: str  # English POS term
    filipino: str  # Filipino POS term
    abbreviations: List[str]  # Common abbreviations
    variants: List[str]  # Variant forms

POS_MAPPINGS = {
    'noun': POSMapping(
        english='noun',
        filipino='pangngalan',
        abbreviations=['n', 'png', 'noun'],
        variants=['pangn', 'pangng', 'pn']
    ),
    'verb': POSMapping(
        english='verb',
        filipino='pandiwa',
        abbreviations=['v', 'vrb', 'verb'],
        variants=['pnd', 'pd']
    ),
    'adjective': POSMapping(
        english='adjective',
        filipino='pang-uri',
        abbreviations=['adj', 'pang'],
        variants=['pu', 'p-uri']
    ),
    'adverb': POSMapping(
        english='adverb',
        filipino='pang-abay',
        abbreviations=['adv', 'advb'],
        variants=['pa', 'p-abay']
    ),
    'pronoun': POSMapping(
        english='pronoun',
        filipino='panghalip',
        abbreviations=['pron', 'prn'],
        variants=['ph', 'phl']
    ),
    'determiner': POSMapping(
        english='determiner',
        filipino='pananda',
        abbreviations=['det', 'dtr'],
        variants=['pnd', 'pnda']
    ),
    # Add other POS mappings...
} 

class SourceStandardization:
    """Standardized names for dictionary sources."""
    
    FILE_TO_DISPLAY = {
        'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
        'kaikki.jsonl': 'kaikki.org (Tagalog)', 
        'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
        'root_words_with_associated_words_cleaned.json': 'tagalog.com',
        'tagalog-words.json': 'diksiyonaryo.ph'
    }

    SOURCE_TO_ENUM = {
        'kaikki.jsonl': DictionarySource.KAIKKI,
        'kwf_dictionary.json': DictionarySource.KWF,
        'root_words_with_associated_words_cleaned.json': DictionarySource.ROOT_WORDS,
        'tagalog-words.json': DictionarySource.TAGALOG_WORDS
    }

    @staticmethod
    def get_display_name(filename: str) -> str:
        """Get standardized display name for source."""
        return SourceStandardization.FILE_TO_DISPLAY.get(filename, filename)

    @staticmethod
    def get_source_enum(filename: str) -> DictionarySource:
        """Get DictionarySource enum for filename."""
        return SourceStandardization.SOURCE_TO_ENUM.get(filename)