from enum import Enum
from typing import Optional

class DictionarySource(Enum):
    KAIKKI = "kaikki"
    KWF = "kwf"
    ROOT_WORDS = "root_words" 
    TAGALOG_WORDS = "tagalog_words"

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
    def get_source_enum(filename: str) -> Optional[DictionarySource]:
        """Get DictionarySource enum for filename."""
        return SourceStandardization.SOURCE_TO_ENUM.get(filename)

    @staticmethod
    def get_standardized_source_sql() -> str:
        """Returns SQL CASE statement for standardized sources."""
        return """
            CASE 
                WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE sources
            END
        """ 