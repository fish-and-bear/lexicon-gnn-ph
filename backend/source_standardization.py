"""
Centralized source standardization for the dictionary system.
"""

from enum import Enum
from typing import Optional, Dict

class DictionarySource(Enum):
    KAIKKI = "kaikki"
    KWF = "kwf"
    ROOT_WORDS = "root_words" 
    TAGALOG_WORDS = "tagalog_words"

class SourceStandardization:
    """Standardized names for dictionary sources."""
    
    FILE_TO_DISPLAY: Dict[str, str] = {
        'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
        'kaikki.jsonl': 'kaikki.org (Tagalog)', 
        'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
        'root_words_with_associated_words_cleaned.json': 'tagalog.com',
        'tagalog-words.json': 'diksiyonaryo.ph'
    }

    SOURCE_TO_ENUM: Dict[str, DictionarySource] = {
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
    def get_source_enum(filename: str) -> Optional[DictionarySource]:
        """Get DictionarySource enum for filename."""
        return SourceStandardization.SOURCE_TO_ENUM.get(filename)

    @staticmethod
    def get_standardized_source_sql() -> str:
        """Returns SQL CASE statement for standardized sources."""
        sql_parts = []
        for file_name, display_name in SourceStandardization.FILE_TO_DISPLAY.items():
            sql_parts.append(f"WHEN sources = '{file_name}' THEN '{display_name}'")
        
        return f"""
            CASE 
                {' '.join(sql_parts)}
                ELSE sources
            END
        """

    @staticmethod
    def standardize_sources(sources: str) -> str:
        """Standardize a comma-separated list of sources."""
        if not sources:
            return ""
        
        source_list = [s.strip() for s in sources.split(',')]
        standardized = []
        
        for source in source_list:
            display_name = SourceStandardization.get_display_name(source)
            if display_name not in standardized:
                standardized.append(display_name)
                
        return ', '.join(sorted(standardized)) 