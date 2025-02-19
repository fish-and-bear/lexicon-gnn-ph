"""
Consolidated language utilities module that combines functionality from:
- language_validator.py
- language_config.py
- language_systems.py
- language_types.py
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import re
from dataclasses import dataclass
from datetime import datetime, UTC
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ScriptType(Enum):
    LATIN = "latin"
    BAYBAYIN = "baybayin"
    MIXED = "mixed"

class LanguageFamily(Enum):
    AUSTRONESIAN = "austronesian"
    MALAYO_POLYNESIAN = "malayo-polynesian"
    PHILIPPINE = "philippine"

class ScriptFeature(Enum):
    DIACRITICS = "diacritics"
    LIGATURES = "ligatures"
    VIRAMA = "virama"
    VOWEL_MARKS = "vowel_marks"

class BaybayinCharType(Enum):
    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"

@dataclass
class BaybayinChar:
    char: str
    char_type: BaybayinCharType
    default_sound: str
    possible_sounds: List[str]
    
    def __post_init__(self):
        if not self.char:
            raise ValueError("Character cannot be empty")
        if not isinstance(self.char_type, BaybayinCharType):
            raise ValueError(f"Invalid character type: {self.char_type}")

@dataclass
class LanguageMetadata:
    code: str
    name: str
    native_name: str
    family: LanguageFamily
    scripts: List[ScriptType]
    region: str
    iso_639_1: str
    iso_639_2: str
    iso_639_3: str
    glottolog: str
    features: Optional[Set[ScriptFeature]] = None
    script_data: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if self.features is None:
            self.features = set()
        if self.script_data is None:
            self.script_data = {}
        if self.last_updated is None:
            self.last_updated = datetime.now(UTC)

@dataclass
class ScriptConversionRule:
    from_script: ScriptType
    to_script: ScriptType
    pattern: str
    replacement: str
    context: Optional[str] = None
    priority: int = 0

class LanguageValidator:
    """Validates language codes and related data."""
    
    def __init__(self):
        self.valid_codes = {
            'tl': LanguageMetadata(
                code='tl',
                name='Tagalog',
                native_name='Tagalog',
                family=LanguageFamily.PHILIPPINE,
                scripts=[ScriptType.LATIN, ScriptType.BAYBAYIN],
                region='Philippines',
                iso_639_1='tl',
                iso_639_2='tgl',
                iso_639_3='tgl',
                glottolog='taga1270',
                features={ScriptFeature.VOWEL_MARKS, ScriptFeature.VIRAMA},
                script_data={
                    'baybayin_vowels': ['ᜀ', 'ᜁ', 'ᜂ'],
                    'baybayin_consonants': ['ᜃ', 'ᜄ', 'ᜅ', 'ᜆ', 'ᜇ', 'ᜈ', 'ᜉ', 'ᜊ', 'ᜋ', 'ᜌ', 'ᜎ', 'ᜏ', 'ᜐ', 'ᜑ'],
                    'baybayin_vowel_marks': ['ᜒ', 'ᜓ'],
                    'baybayin_virama': '᜔'
                }
            ),
            'ceb': LanguageMetadata(
                code='ceb',
                name='Cebuano',
                native_name='Sinugbuanon',
                family=LanguageFamily.PHILIPPINE,
                scripts=[ScriptType.LATIN],
                region='Philippines',
                iso_639_1='',
                iso_639_2='ceb',
                iso_639_3='ceb',
                glottolog='cebu1242',
                features=set(),
                script_data={}
            )
        }
        
        self.script_patterns = {
            ScriptType.LATIN: re.compile(r'^[a-zA-Z\s\-\']+$'),
            ScriptType.BAYBAYIN: re.compile(r'^[\u1700-\u171F\s]+$'),
            ScriptType.MIXED: re.compile(r'^[a-zA-Z\u1700-\u171F\s\-\']+$')
        }

        self.conversion_rules = {
            ('tl', ScriptType.LATIN, ScriptType.BAYBAYIN): [
                ScriptConversionRule(
                    from_script=ScriptType.LATIN,
                    to_script=ScriptType.BAYBAYIN,
                    pattern=r'([kg]a)',
                    replacement='ᜃ',
                    priority=1
                ),
                # Add more conversion rules here
            ]
        }

    def is_valid_code(self, code: str) -> bool:
        """Check if a language code is valid."""
        return code in self.valid_codes

    def get_metadata(self, code: str) -> Optional[LanguageMetadata]:
        """Get metadata for a language code."""
        return self.valid_codes.get(code)

    def detect_script(self, text: str) -> ScriptType:
        """Detect the script used in a text."""
        for script_type, pattern in self.script_patterns.items():
            if pattern.match(text):
                return script_type
        return ScriptType.MIXED

    def validate_text(self, text: str, code: str, allowed_scripts: Optional[List[ScriptType]] = None) -> bool:
        """Validate text against allowed scripts for a language."""
        if not self.is_valid_code(code):
            return False
            
        metadata = self.valid_codes[code]
        script = self.detect_script(text)
        
        if allowed_scripts:
            valid_scripts = set(allowed_scripts)
        else:
            valid_scripts = set(metadata.scripts)
            
        return script in valid_scripts

    def convert_script(self, text: str, from_script: ScriptType, to_script: ScriptType, 
                      lang_code: str) -> Optional[str]:
        """Convert text from one script to another."""
        if not self.validate_text(text, lang_code, [from_script]):
            return None
            
        key = (lang_code, from_script, to_script)
        if key not in self.conversion_rules:
            return None
            
        result = text
        for rule in sorted(self.conversion_rules[key], key=lambda r: r.priority, reverse=True):
            if rule.context:
                # Apply context-sensitive rules
                context_pattern = re.compile(f"(?<={rule.context}){rule.pattern}")
                result = context_pattern.sub(rule.replacement, result)
            else:
                # Apply context-free rules
                result = re.sub(rule.pattern, rule.replacement, result)
                
        return result

    def get_script_features(self, code: str, script: ScriptType) -> Set[ScriptFeature]:
        """Get supported features for a language and script combination."""
        metadata = self.get_metadata(code)
        if not metadata or script not in metadata.scripts:
            return set()
        return metadata.features

class LanguageSystem:
    """Manages language-specific operations and metadata."""
    
    def __init__(self):
        self.valid_codes = ['tl', 'ceb']  # Tagalog and Cebuano
        self._metadata = {
            'tl': {
                'name': 'Tagalog',
                'native_name': 'Tagalog',
                'family': 'Austronesian',
                'iso_639_1': 'tl',
                'iso_639_2': 'tgl',
                'has_baybayin': True,
                'writing_systems': ['Latin', 'Baybayin'],
                'region': 'Philippines'
            },
            'ceb': {
                'name': 'Cebuano',
                'native_name': 'Sinugbuanon',
                'family': 'Austronesian',
                'iso_639_1': 'ceb',
                'iso_639_2': 'ceb',
                'has_baybayin': False,
                'writing_systems': ['Latin'],
                'region': 'Philippines'
            }
        }

    def is_valid_language(self, code: str) -> bool:
        """Check if a language code is valid."""
        return code in self.valid_codes

    def get_language_metadata(self, code: str) -> Optional[Dict]:
        """Get metadata for a specific language."""
        return self._metadata.get(code)

    def get_all_languages(self) -> List[Dict]:
        """Get metadata for all supported languages."""
        return [
            {'code': code, **metadata}
            for code, metadata in self._metadata.items()
        ]

    def supports_baybayin(self, code: str) -> bool:
        """Check if a language supports Baybayin script."""
        metadata = self.get_language_metadata(code)
        return metadata.get('has_baybayin', False) if metadata else False

# Initialize global instances
language_validator = LanguageValidator()
language_system = LanguageSystem()

# Export commonly used functions
is_valid_language = language_validator.is_valid_code
validate_text = language_validator.validate_text
get_language_metadata = language_system.get_language_metadata
get_related_languages = language_system.get_related_languages
supports_script = language_system.supports_script
get_supported_scripts = language_system.get_supported_scripts
convert_script = language_system.convert_script
get_script_features = language_system.get_script_features
get_script_data = language_system.get_script_data 