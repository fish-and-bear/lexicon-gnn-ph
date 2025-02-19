"""
Centralized source standardization for the dictionary system.
Provides standardization of dictionary sources, validation, and caching.
"""

from enum import Enum
from typing import Optional, Dict, List, Set
from functools import lru_cache
import logging
from prometheus_client import Counter, Histogram
import time

logger = logging.getLogger(__name__)

# Metrics
SOURCE_ERRORS = Counter('source_standardization_errors_total', 'Total source standardization errors', ['operation'])
SOURCE_OPERATION_DURATION = Histogram('source_standardization_duration_seconds', 'Source standardization operation duration')

class DictionarySource(Enum):
    """Enumeration of dictionary sources with metadata."""
    KAIKKI = "kaikki"
    KWF = "kwf"
    ROOT_WORDS = "root_words" 
    TAGALOG_WORDS = "tagalog_words"

    @property
    def display_name(self) -> str:
        """Get display name for the source."""
        return {
            self.KAIKKI: "kaikki.org",
            self.KWF: "KWF Diksiyonaryo ng Wikang Filipino",
            self.ROOT_WORDS: "tagalog.com",
            self.TAGALOG_WORDS: "diksiyonaryo.ph"
        }[self]

    @property
    def priority(self) -> int:
        """Get priority level for the source (lower is higher priority)."""
        return {
            self.KWF: 1,  # Official source
            self.KAIKKI: 2,  # Comprehensive source
            self.ROOT_WORDS: 3,
            self.TAGALOG_WORDS: 4
        }[self]

class SourceValidationError(Exception):
    """Raised when source validation fails."""
    pass

class SourceStandardization:
    """Standardized names for dictionary sources with caching and validation."""
    
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
    @lru_cache(maxsize=128)
    def get_display_name(filename: str) -> str:
        """Get standardized display name for source with caching."""
        start_time = time.time()
        try:
            if not filename:
                raise SourceValidationError("Empty filename provided")
            
            display_name = SourceStandardization.FILE_TO_DISPLAY.get(filename)
            if not display_name:
                logger.warning(f"Unknown source filename: {filename}")
                display_name = filename
            
            return display_name
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='get_display_name').inc()
            logger.error(f"Error getting display name for {filename}: {str(e)}")
            return filename
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    @lru_cache(maxsize=128)
    def get_source_enum(filename: str) -> Optional[DictionarySource]:
        """Get DictionarySource enum for filename with caching."""
        start_time = time.time()
        try:
            if not filename:
                raise SourceValidationError("Empty filename provided")
            
            source_enum = SourceStandardization.SOURCE_TO_ENUM.get(filename)
            if not source_enum:
                logger.warning(f"No enum mapping for source: {filename}")
            
            return source_enum
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='get_source_enum').inc()
            logger.error(f"Error getting source enum for {filename}: {str(e)}")
            return None
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    def get_standardized_source_sql() -> str:
        """Returns SQL CASE statement for standardized sources with validation."""
        start_time = time.time()
        try:
        sql_parts = []
        for file_name, display_name in SourceStandardization.FILE_TO_DISPLAY.items():
                # Escape single quotes in display name
                display_name = display_name.replace("'", "''")
            sql_parts.append(f"WHEN sources = '{file_name}' THEN '{display_name}'")
        
            if not sql_parts:
                raise SourceValidationError("No source mappings available")
            
        return f"""
            CASE 
                {' '.join(sql_parts)}
                ELSE sources
            END
        """
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='get_standardized_source_sql').inc()
            logger.error(f"Error generating standardized source SQL: {str(e)}")
            return "sources"  # Fallback to original column
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    def standardize_sources(sources: str) -> str:
        """Standardize a comma-separated list of sources with validation and deduplication."""
        start_time = time.time()
        try:
        if not sources:
            return ""
        
            # Split and clean source list
            source_list = [s.strip() for s in sources.split(',') if s.strip()]
            if not source_list:
                return ""
            
            # Get display names and deduplicate
            standardized: Set[str] = set()
        for source in source_list:
            display_name = SourceStandardization.get_display_name(source)
                if display_name:
                    standardized.add(display_name)
            
            # Sort by source priority if possible
            def get_source_priority(display_name: str) -> int:
                for filename, enum_source in SourceStandardization.SOURCE_TO_ENUM.items():
                    if SourceStandardization.FILE_TO_DISPLAY.get(filename) == display_name:
                        return enum_source.priority
                return 999  # Default low priority for unknown sources
            
            return ', '.join(sorted(standardized, key=get_source_priority))
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='standardize_sources').inc()
            logger.error(f"Error standardizing sources {sources}: {str(e)}")
            return sources  # Return original on error
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    def validate_source(source: str) -> bool:
        """Validate if a source is recognized."""
        start_time = time.time()
        try:
            if not source:
                return False
            
            # Check if source is in our mappings
            return (source in SourceStandardization.FILE_TO_DISPLAY or 
                   source in [s.display_name for s in DictionarySource])
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='validate_source').inc()
            logger.error(f"Error validating source {source}: {str(e)}")
            return False
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    def get_source_priority(source: str) -> int:
        """Get priority level for a source."""
        start_time = time.time()
        try:
            if not source:
                return 999
            
            # Try to get enum for source
            for filename, enum_source in SourceStandardization.SOURCE_TO_ENUM.items():
                if (SourceStandardization.FILE_TO_DISPLAY.get(filename) == source or 
                    filename == source):
                    return enum_source.priority
            
            return 999  # Default low priority for unknown sources
            
        except Exception as e:
            SOURCE_ERRORS.labels(operation='get_source_priority').inc()
            logger.error(f"Error getting priority for source {source}: {str(e)}")
            return 999
        finally:
            SOURCE_OPERATION_DURATION.observe(time.time() - start_time)

    @staticmethod
    def get_all_sources() -> List[str]:
        """Get list of all recognized sources."""
        return sorted(
            set(SourceStandardization.FILE_TO_DISPLAY.values()),
            key=SourceStandardization.get_source_priority
        ) 