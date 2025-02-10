from typing import TypedDict, Literal, Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from backend.source_standardization import DictionarySource, SourceStandardization

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
        abbreviations=['n', 'n.', 'png', 'noun', 'png.'],
        variants=['pangn', 'pangng', 'pn', 'pangngalan']
    ),
    'verb': POSMapping(
        english='verb',
        filipino='pandiwa',
        abbreviations=['v', 'v.', 'vrb', 'verb', 'pd', 'pd.'],
        variants=['pnd', 'pandiwa']
    ),
    'adjective': POSMapping(
        english='adjective',
        filipino='pang-uri',
        abbreviations=['adj', 'adj.', 'pnr', 'pnr.', 'pang'],
        variants=['pu', 'p-uri', 'pang-uri']
    ),
    'adverb': POSMapping(
        english='adverb',
        filipino='pang-abay',
        abbreviations=['adv', 'adv.', 'advb', 'pa', 'pa.'],
        variants=['pa', 'p-abay', 'pang-abay']
    ),
    'pronoun': POSMapping(
        english='pronoun',
        filipino='panghalip',
        abbreviations=['pron', 'pron.', 'prn', 'ph', 'ph.'],
        variants=['ph', 'phl', 'panghalip']
    ),
    'interjection': POSMapping(
        english='interjection',
        filipino='padamdam',
        abbreviations=['interj', 'interj.', 'pdm', 'pdm.'],
        variants=['padamdam']
    ),
    'preposition': POSMapping(
        english='preposition',
        filipino='pang-ukol',
        abbreviations=['prep', 'prep.', 'pk', 'pk.'],
        variants=['pang-ukol']
    ),
    'affix': POSMapping(
        english='affix',
        filipino='panlapi',
        abbreviations=['aff', 'aff.', 'pl', 'pl.'],
        variants=['panlapi']
    ),
    'article': POSMapping(
        english='article',
        filipino='pantukoy',
        abbreviations=['art', 'art.', 'pt', 'pt.'],
        variants=['pantukoy']
    ),
    'conjunction': POSMapping(
        english='conjunction',
        filipino='pangatnig',
        abbreviations=['conj', 'conj.', 'ptg', 'ptg.'],
        variants=['pangatnig']
    ),
    'idiom': POSMapping(
        english='idiom',
        filipino='idyoma',
        abbreviations=['id', 'id.', 'idy', 'idy.'],
        variants=['idyoma']
    ),
    'colloquial': POSMapping(
        english='colloquial',
        filipino='kolokyal',
        abbreviations=['col', 'col.', 'kolok', 'kolok.'],
        variants=['kolokyal']
    ),
    'synonym': POSMapping(
        english='synonym',
        filipino='singkahulugan',
        abbreviations=['syn', 'syn.', 'sk', 'sk.'],
        variants=['singkahulugan']
    ),
    'antonym': POSMapping(
        english='antonym',
        filipino='di-kasingkahulugan',
        abbreviations=['ant', 'ant.', 'dks', 'dks.'],
        variants=['di-kasingkahulugan']
    ),
    'english': POSMapping(
        english='english',
        filipino='ingles',
        abbreviations=['eng', 'eng.', 'ing', 'ing.'],
        variants=['ingles']
    ),
    'spanish': POSMapping(
        english='spanish',
        filipino='espanyol',
        abbreviations=['spa', 'spa.', 'esp', 'esp.'],
        variants=['espanyol']
    ),
    'texting': POSMapping(
        english='texting',
        filipino='texting',
        abbreviations=['tx', 'tx.'],
        variants=['texting']
    ),
    'variant': POSMapping(
        english='variant',
        filipino='varyant',
        abbreviations=['var', 'var.'],
        variants=['varyant']
    )
}

def standardize_pos(pos_string: str) -> str:
    """
    Standardize a part of speech string to its full Filipino term.
    Case-insensitive and handles various abbreviations and variants.
    
    Args:
        pos_string: The POS string to standardize
        
    Returns:
        The standardized Filipino POS term
    """
    if not pos_string:
        return ""
        
    # Convert to lowercase for case-insensitive matching
    pos_lower = pos_string.lower().strip(' .')
    
    # Check each mapping
    for mapping in POS_MAPPINGS.values():
        # Check full English term
        if pos_lower == mapping['english'].lower():
            return mapping['filipino']
            
        # Check full Filipino term
        if pos_lower == mapping['filipino'].lower():
            return mapping['filipino']
            
        # Check abbreviations
        if pos_lower in [abbr.lower().strip('.') for abbr in mapping['abbreviations']]:
            return mapping['filipino']
            
        # Check variants
        if pos_lower in [var.lower() for var in mapping['variants']]:
            return mapping['filipino']
            
    # If no match found, return original
    return pos_string