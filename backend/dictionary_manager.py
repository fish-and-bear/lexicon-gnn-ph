#!/usr/bin/env python3
"""
dictionary_manager.py

A comprehensive tool for managing Filipino dictionary data with support for 
Baybayin script, etymologies, and relations.

Usage examples:
  python dictionary_manager.py migrate
  python dictionary_manager.py test
  python dictionary_manager.py verify
  python dictionary_manager.py update --file new_data.json
  python dictionary_manager.py inspect --word "word"
  python dictionary_manager.py leaderboard
  python dictionary_manager.py stats
  python dictionary_manager.py help
  python dictionary_manager.py explore
"""

import argparse
import psycopg2
import psycopg2.extras
from psycopg2.errors import UniqueViolation
import json
import unidecode
import io
from tqdm import tqdm
import sys
import re  # Removed duplicate import
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.align import Align
from rich.console import Group
from rich.rule import Rule
import codecs
import locale
from typing import Optional, List, Tuple, Dict, Any, Set, Callable
from dataclasses import dataclass
import functools
from enum import Enum
from language_systems import LanguageSystem
from dictionary_processor import DictionaryProcessor
from language_types import *
from source_standardization import SourceStandardization

# -------------------------------------------------------------------
# Initialize LanguageSystem
# -------------------------------------------------------------------
lsys = LanguageSystem()
processor = DictionaryProcessor(lsys)

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
load_dotenv()

def setup_logging():
    """Configure logging with proper Unicode handling."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler with UTF-8 encoding
    file_path = f'{log_dir}/dictionary_manager_{timestamp}.log'
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler with Unicode support
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Platform-specific Unicode handling
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding='utf-8')
            console_handler.stream = codecs.getwriter('utf-8')(sys.stdout.buffer)
    except Exception:
        def safe_encode(msg):
            try:
                return str(msg).encode(console_handler.stream.encoding, 'replace').decode(console_handler.stream.encoding)
            except:
                return str(msg).encode('ascii', 'replace').decode('ascii')
        
        original_emit = console_handler.emit
        def safe_emit(record):
            record.msg = safe_encode(record.msg)
            original_emit(record)
        console_handler.emit = safe_emit
    
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()

# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

# Validate database configuration
if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
    print("Error: Missing database configuration!")
    print("Please ensure you have a .env file with the following variables:")
    print("DB_NAME - The name of your PostgreSQL database")
    print("DB_USER - Your PostgreSQL username")
    print("DB_PASSWORD - Your PostgreSQL password")
    print("DB_HOST - Your database host (usually 'localhost')")
    sys.exit(1)

# -------------------------------------------------------------------
# Core Database Connection
# -------------------------------------------------------------------
def get_connection():
    """
    Establish a connection to the PostgreSQL database with proper error handling.
    
    Returns:
        psycopg2.extensions.connection: Database connection object
        
    Raises:
        SystemExit: If connection fails
    """
    try:
        logger.info("Attempting database connection...")
        logger.info(f"Database configuration:")
        logger.info(f"  Database Name: {DB_NAME}")
        logger.info(f"  User: {DB_USER}")
        logger.info(f"  Host: {DB_HOST}")
        logger.info(f"  Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'Not set'}")

        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST
        )
        
        # Disable autocommit for transaction control
        conn.autocommit = False
        
        logger.info("Successfully established database connection")
        return conn

    except psycopg2.OperationalError as e:
        logger.error("Database connection failed!")
        logger.error("Please check:")
        logger.error("1. PostgreSQL service is running")
        logger.error("2. Database credentials in .env file are correct")
        logger.error("3. Database exists and is accessible")
        logger.error(f"Error details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during database connection: {str(e)}")
        sys.exit(1)

# -------------------------------------------------------------------
# Transaction Management Decorator
# -------------------------------------------------------------------
def with_transaction(commit=True):
    """
    Decorator for database operations that handles transactions and rollbacks.
    
    Args:
        commit (bool): Whether to commit the transaction
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection
            savepoint_name = f"sp_{func.__name__}"
            try:
                # Create savepoint for nested transactions
                cur.execute(f"SAVEPOINT {savepoint_name}")
                result = func(cur, *args, **kwargs)
                if commit:
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                return result
            except Exception as e:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                if not isinstance(e, UniqueViolation):
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# -------------------------------------------------------------------
# Database Schema Creation / Update
# -------------------------------------------------------------------

# Complete table creation SQL with all necessary tables and constraints
TABLE_CREATION_SQL = r"""
-- First create the parts_of_speech table since it has no dependencies
CREATE TABLE IF NOT EXISTS parts_of_speech (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) NOT NULL UNIQUE,
    name_en VARCHAR(64) NOT NULL,
    name_tl VARCHAR(64) NOT NULL,
    description TEXT,
    CONSTRAINT parts_of_speech_code_uniq UNIQUE (code)
);

-- Create search indexes to improve performance
CREATE INDEX IF NOT EXISTS idx_parts_of_speech_code ON parts_of_speech(code);
CREATE INDEX IF NOT EXISTS idx_parts_of_speech_name ON parts_of_speech(name_en, name_tl);

-- Next create the words table since other tables depend on it
CREATE TABLE IF NOT EXISTS words (
    id                SERIAL PRIMARY KEY,
    lemma             VARCHAR(255) NOT NULL,
    normalized_lemma  VARCHAR(255) NOT NULL,
    has_baybayin      BOOLEAN DEFAULT FALSE,
    baybayin_form     VARCHAR(255),
    romanized_form    VARCHAR(255),
    language_code     VARCHAR(16)  NOT NULL,
    root_word_id      INT REFERENCES words(id),
    preferred_spelling VARCHAR(255),
    tags              TEXT,
    idioms            JSONB DEFAULT '[]',
    search_text       tsvector,
    pronunciation_data JSONB,
    created_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (language_code, normalized_lemma),
    CONSTRAINT baybayin_form_check CHECK (
        (has_baybayin = FALSE AND baybayin_form IS NULL) OR 
        (has_baybayin = TRUE AND baybayin_form IS NOT NULL)
    ),
    CONSTRAINT baybayin_form_regex CHECK (baybayin_form ~ '^[\u1700-\u171F\s]*$' OR baybayin_form IS NULL)
);

-- Create indexes for words table
CREATE INDEX IF NOT EXISTS idx_words_lemma ON words(lemma);
CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = TRUE;
CREATE INDEX IF NOT EXISTS idx_words_romanized ON words(romanized_form);
CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);

-- Create definitions table
CREATE TABLE IF NOT EXISTS definitions (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    definition_text TEXT NOT NULL,
    original_pos TEXT,
    standardized_pos_id INT REFERENCES parts_of_speech(id),
    examples TEXT,
    usage_notes TEXT,
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for definitions
CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
CREATE INDEX IF NOT EXISTS idx_definitions_pos ON definitions(standardized_pos_id);
CREATE INDEX IF NOT EXISTS idx_definitions_text ON definitions USING gin(to_tsvector('english', definition_text));

-- Create relations table
CREATE TABLE IF NOT EXISTS relations (
    id            SERIAL PRIMARY KEY,
    from_word_id  INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    to_word_id    INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    relation_type VARCHAR(64) NOT NULL,
    sources       TEXT NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT relations_unique UNIQUE (from_word_id, to_word_id, relation_type)
);

-- Create indexes for relations
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- Create etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id                   SERIAL PRIMARY KEY,
    word_id              INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text       TEXT NOT NULL,
    normalized_components TEXT,
    language_codes       TEXT,
    sources             TEXT NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT etymologies_wordid_etymtext_uniq UNIQUE (word_id, etymology_text)
);

-- Create indexes for etymologies
CREATE INDEX IF NOT EXISTS idx_etymologies_word ON etymologies(word_id);
CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));

-- Create definition_relations table
CREATE TABLE IF NOT EXISTS definition_relations (
    id SERIAL PRIMARY KEY,
    definition_id INT NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    relation_type VARCHAR(64) NOT NULL,
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT definition_relations_unique UNIQUE (definition_id, word_id, relation_type)
);

-- Create indexes for definition relations
CREATE INDEX IF NOT EXISTS idx_def_relations_def ON definition_relations(definition_id);
CREATE INDEX IF NOT EXISTS idx_def_relations_word ON definition_relations(word_id);

-- Create affixations table
CREATE TABLE IF NOT EXISTS affixations (
    id SERIAL PRIMARY KEY,
    root_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    affixed_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    affix_type VARCHAR(64) NOT NULL,
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT affixations_unique UNIQUE (root_word_id, affixed_word_id, affix_type)
);

-- Create indexes for affixations
CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);

-- Create timestamp update function if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_timestamp') THEN
        EXECUTE 'CREATE OR REPLACE FUNCTION update_timestamp()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $trigger$ language ''plpgsql''';
    END IF;
END
$$;

-- Create triggers if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_words_timestamp'
        AND tgrelid = 'words'::regclass
    ) THEN
        CREATE TRIGGER update_words_timestamp
            BEFORE UPDATE ON words
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_definitions_timestamp'
        AND tgrelid = 'definitions'::regclass
    ) THEN
        CREATE TRIGGER update_definitions_timestamp
            BEFORE UPDATE ON definitions
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;
END
$$;
"""

def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
    
    try:
        # List of required extensions
        extensions = [
            'pg_trgm',      # For similarity search and fuzzy matching
            'unaccent',     # For handling diacritics
            'fuzzystrmatch' # For Levenshtein distance
        ]
        
        for ext in extensions:
            logger.info(f"Checking extension: {ext}")
            # Check if extension exists
            cur.execute("""
                SELECT COUNT(*) 
                FROM pg_extension 
                WHERE extname = %s
            """, (ext,))
            
            if cur.fetchone()[0] == 0:
                logger.info(f"Installing extension: {ext}")
                try:
                    cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                    conn.commit()
                    logger.info(f"Successfully installed extension: {ext}")
                except Exception as e:
                    logger.warning(f"Could not install extension {ext}: {str(e)}")
                    logger.warning("Some features may not work without this extension")
                    conn.rollback()
            else:
                logger.info(f"Extension already installed: {ext}")
        
        logger.info("Extensions setup completed")
        
    except Exception as e:
        logger.error(f"Error setting up extensions: {str(e)}")
        raise
    finally:
        cur.close()

def create_or_update_tables(conn):
    """
    Create or update the database schema for the Filipino dictionary.
    
    Args:
        conn: Active database connection
    """
    logger.info("Starting table creation/update process.")
    cur = conn.cursor()
    
    try:
        # Execute schema creation SQL
        cur.execute(TABLE_CREATION_SQL)

        # Populate standard POS entries
        pos_entries = [
            ('n', 'Noun', 'Pangngalan', 'Word that refers to a person, place, thing, or idea'),
            ('v', 'Verb', 'Pandiwa', 'Word that expresses action or state of being'),
            ('adj', 'Adjective', 'Pang-uri', 'Word that describes or modifies a noun'),
            ('adv', 'Adverb', 'Pang-abay', 'Word that modifies verbs, adjectives, or other adverbs'),
            ('pron', 'Pronoun', 'Panghalip', 'Word that substitutes for a noun'),
            ('prep', 'Preposition', 'Pang-ukol', 'Word that shows relationship between words'),
            ('conj', 'Conjunction', 'Pangatnig', 'Word that connects words, phrases, or clauses'),
            ('intj', 'Interjection', 'Pandamdam', 'Word expressing emotion'),
            ('det', 'Determiner', 'Pantukoy', 'Word that modifies nouns'),
            ('affix', 'Affix', 'Panlapi', 'Word element attached to base or root'),
            ('baybay', 'Baybayin Script', 'Baybayin', 'Traditional Philippine writing system'),
            ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
        ]

        # Insert POS entries
        for code, name_en, name_tl, desc in pos_entries:
            cur.execute("""
                INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                SET name_en = EXCLUDED.name_en,
                    name_tl = EXCLUDED.name_tl,
                    description = EXCLUDED.description
            """, (code, name_en, name_tl, desc))

        conn.commit()
        logger.info("Tables created or updated successfully.")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {e}")
        raise
    finally:
        cur.close()

# -------------------------------------------------------------------
# Data Structures and Enums
# -------------------------------------------------------------------

class BaybayinCharType(Enum):
    """Enum for different types of Baybayin characters."""
    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"

@dataclass
class BaybayinChar:
    """Represents a single Baybayin character with its properties."""
    char: str
    char_type: BaybayinCharType
    default_sound: str
    possible_sounds: List[str]
    
    def __post_init__(self):
        """Validate the character data."""
        if not self.char:
            raise ValueError("Character cannot be empty")
        if not isinstance(self.char_type, BaybayinCharType):
            raise ValueError(f"Invalid character type: {self.char_type}")

@dataclass
class WordEntry:
    """Data class to represent a word entry with validation."""
    lemma: str
    normalized_lemma: str
    language_code: str
    root_word_id: Optional[int] = None
    preferred_spelling: Optional[str] = None
    tags: Optional[str] = None
    has_baybayin: bool = False
    baybayin_form: Optional[str] = None
    romanized_form: Optional[str] = None

    def __post_init__(self):
        """Validate the word entry data."""
        if not self.lemma or not isinstance(self.lemma, str):
            raise ValueError("Lemma must be a non-empty string")
        if not self.language_code in ('tl', 'ceb'):
            raise ValueError(f"Unsupported language code: {self.language_code}")
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")

# -------------------------------------------------------------------
# POS Mapping and Standardization
# -------------------------------------------------------------------

POS_MAPPING = {
    'noun': {'en': 'Noun', 'tl': 'Pangngalan'},
    'adjective': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'verb': {'en': 'Verb', 'tl': 'Pandiwa'},
    'adverb': {'en': 'Adverb', 'tl': 'Pang-abay'},
    'pronoun': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'preposition': {'en': 'Preposition', 'tl': 'Pang-ukol'},
    'conjunction': {'en': 'Conjunction', 'tl': 'Pangatnig'},
    'interjection': {'en': 'Interjection', 'tl': 'Pandamdam'},
    'affix': {'en': 'Affix', 'tl': 'Panlapi'},
    'pangngalan': {'en': 'Noun', 'tl': 'Pangngalan'},
    'pang-uri': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'pandiwa': {'en': 'Verb', 'tl': 'Pandiwa'},
    'pang-abay': {'en': 'Adverb', 'tl': 'Pang-abay'},
    'panghalip': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'pang-ukol': {'en': 'Preposition', 'tl': 'Pang-ukol'},
    'pangatnig': {'en': 'Conjunction', 'tl': 'Pangatnig'},
    'pandamdam': {'en': 'Interjection', 'tl': 'Pandamdam'},
    'panlapi': {'en': 'Affix', 'tl': 'Panlapi'},
    'pnd': {'en': 'Noun', 'tl': 'Pangngalan'},
    'png': {'en': 'Noun', 'tl': 'Pangngalan'},
    'pnr': {'en': 'Pronoun', 'tl': 'Panghalip'},
    'pnl': {'en': 'Affix', 'tl': 'Panlapi'},
    'pnu': {'en': 'Adjective', 'tl': 'Pang-uri'},
    'pnw': {'en': 'Verb', 'tl': 'Pandiwa'},
    'pny': {'en': 'Adverb', 'tl': 'Pang-abay'},
    'det': {'en': 'Determiner', 'tl': 'Pantukoy'},
    'determiner': {'en': 'Determiner', 'tl': 'Pantukoy'},
    'pantukoy': {'en': 'Determiner', 'tl': 'Pantukoy'},
    'baybayin': {'en': 'Baybayin Script', 'tl': 'Baybayin'},
    'unc': {'en': 'Uncategorized', 'tl': 'Hindi Tiyak'}
}

# -------------------------------------------------------------------
# Core Helper Functions
# -------------------------------------------------------------------

def normalize_lemma(text: str) -> str:
    """
    Normalize text by removing diacritics and converting to lowercase.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    
    Raises:
        ValueError: If input is None or empty
    """
    if not text:
        raise ValueError("Input text cannot be None or empty")
    return unidecode.unidecode(text).lower()

def validate_word_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive word data validation."""
    if not isinstance(data, dict):
        raise ValueError("Word data must be a dictionary")
        
    # Required fields validation
    required_fields = {'lemma', 'language_code'}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Lemma validation
    lemma = data['lemma']
    if not isinstance(lemma, str) or not lemma.strip():
        raise ValueError("Lemma must be a non-empty string")
    if len(lemma) > 255:  # Match database constraint
        raise ValueError("Lemma exceeds maximum length")
    # Removed invalid character check to allow punctuation and diacritics
    
    # Language code validation
    if data['language_code'] not in {'tl', 'ceb'}:
        raise ValueError(f"Unsupported language code: {data['language_code']}")
    
    # Tags validation
    if 'tags' in data:
        if not isinstance(data['tags'], (str, list)):
            raise ValueError("Tags must be string or list")
        if isinstance(data['tags'], list):
            data['tags'] = ','.join(str(tag) for tag in data['tags'])
    
    return data

def has_diacritics(text: str) -> bool:
    """Check if text contains diacritical marks."""
    normalized = normalize_lemma(text)
    return text != normalized

def get_standardized_source(source: str) -> str:
    """Convert internal source filenames to standardized citation format."""
    source_mapping = {
        'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
        'kaikki.jsonl': 'kaikki.org (Tagalog)', 
        'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
        'root_words_with_associated_words_cleaned.json': 'tagalog.com',
        'tagalog-words.json': 'diksiyonaryo.ph'
    }
    # ISSUE: Returns original source if not in mapping
    return source_mapping.get(source, source)  # Could lead to inconsistent source tracking

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

def clean_baybayin_lemma(lemma: str) -> str:
    """Clean Baybayin lemma by removing any prefixes."""
    prefix = "Baybayin spelling of"
    if lemma.lower().startswith(prefix.lower()):
        return lemma[len(prefix):].strip()
    return lemma

def format_word_display(word: str, show_baybayin: bool = True) -> str:
    """
    Format a word for display with proper Baybayin handling.
    """
    has_baybayin = any(ord(c) >= 0x1700 and ord(c) <= 0x171F for c in word)
    if has_baybayin:
        romanized = get_romanized_text(word)
        if show_baybayin:
            return f"[bold cyan]{word}[/] [dim](romanized: {romanized})[/]"
        else:
            return romanized
    return word

def get_root_word_id(cur: "psycopg2.extensions.cursor", 
                     lemma: str, 
                     language_code: str) -> Optional[int]:
    """Get root word ID if it exists."""
    cur.execute("""
        SELECT id FROM words 
        WHERE normalized_lemma = %s 
        AND language_code = %s 
        AND root_word_id IS NULL
    """, (normalize_lemma(lemma), language_code))
    result = cur.fetchone()
    return result[0] if result else None

# -------------------------------------------------------------------
# Baybayin Processing System
# -------------------------------------------------------------------

class BaybayinRomanizer:
    """
    Handles romanization of Baybayin text with proper character combination rules.
    Implements all valid Baybayin syllable patterns and combinations.
    """
    
    # Comprehensive character mappings
    VOWELS = {
        'ᜀ': BaybayinChar('ᜀ', BaybayinCharType.VOWEL, 'a', ['a']),
        'ᜁ': BaybayinChar('ᜁ', BaybayinCharType.VOWEL, 'i', ['i', 'e']),
        'ᜂ': BaybayinChar('ᜂ', BaybayinCharType.VOWEL, 'u', ['u', 'o'])
    }
    
    CONSONANTS = {
        'ᜃ': BaybayinChar('ᜃ', BaybayinCharType.CONSONANT, 'ka', ['ka']),
        'ᜄ': BaybayinChar('ᜄ', BaybayinCharType.CONSONANT, 'ga', ['ga']),
        'ᜅ': BaybayinChar('ᜅ', BaybayinCharType.CONSONANT, 'nga', ['nga']),
        'ᜆ': BaybayinChar('ᜆ', BaybayinCharType.CONSONANT, 'ta', ['ta']),
        'ᜇ': BaybayinChar('ᜇ', BaybayinCharType.CONSONANT, 'da', ['da']),
        'ᜈ': BaybayinChar('ᜈ', BaybayinCharType.CONSONANT, 'na', ['na']),
        'ᜉ': BaybayinChar('ᜉ', BaybayinCharType.CONSONANT, 'pa', ['pa']),
        'ᜊ': BaybayinChar('ᜊ', BaybayinCharType.CONSONANT, 'ba', ['ba']),
        'ᜋ': BaybayinChar('ᜋ', BaybayinCharType.CONSONANT, 'ma', ['ma']),
        'ᜌ': BaybayinChar('ᜌ', BaybayinCharType.CONSONANT, 'ya', ['ya']),
        'ᜎ': BaybayinChar('ᜎ', BaybayinCharType.CONSONANT, 'la', ['la']),
        'ᜏ': BaybayinChar('ᜏ', BaybayinCharType.CONSONANT, 'wa', ['wa']),
        'ᜐ': BaybayinChar('ᜐ', BaybayinCharType.CONSONANT, 'sa', ['sa']),
        'ᜑ': BaybayinChar('ᜑ', BaybayinCharType.CONSONANT, 'ha', ['ha'])
    }
    
    VOWEL_MARKS = {
        'ᜒ': BaybayinChar('ᜒ', BaybayinCharType.VOWEL_MARK, 'i', ['i', 'e']),
        'ᜓ': BaybayinChar('ᜓ', BaybayinCharType.VOWEL_MARK, 'u', ['u', 'o'])
    }
    
    VIRAMA = BaybayinChar('᜔', BaybayinCharType.VIRAMA, '', [])
    
    PUNCTUATION = {
        '᜵': BaybayinChar('᜵', BaybayinCharType.PUNCTUATION, ',', [',']),
        '᜶': BaybayinChar('᜶', BaybayinCharType.PUNCTUATION, '.', ['.'])
    }

    def __init__(self):
        """Initialize the romanizer with validation."""
        # Validate character sets for overlaps
        all_chars = set()
        for char_set in [self.VOWELS, self.CONSONANTS, self.VOWEL_MARKS, 
                        {self.VIRAMA.char: self.VIRAMA}, self.PUNCTUATION]:
            for char in char_set:
                if char in all_chars:
                    raise ValueError(f"Duplicate character in mappings: {char}")
                all_chars.add(char)

    def is_baybayin(self, text: str) -> bool:
        """
        Check if text contains Baybayin characters.
        
        Args:
            text: Input text to check
            
        Returns:
            bool: True if text contains Baybayin characters
        """
        if not text:
            return False
        return any(ord(c) >= 0x1700 and ord(c) <= 0x171F for c in text)

    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
        """
        Get information about a Baybayin character.
        
        Args:
            char: Single character to look up
            
        Returns:
            Optional[BaybayinChar]: Character information or None if not found
        """
        if char in self.VOWELS:
            return self.VOWELS[char]
        elif char in self.CONSONANTS:
            return self.CONSONANTS[char]
        elif char in self.VOWEL_MARKS:
            return self.VOWEL_MARKS[char]
        elif char == self.VIRAMA.char:
            return self.VIRAMA
        elif char in self.PUNCTUATION:
            return self.PUNCTUATION[char]
        return None

    def process_syllable(self, chars: List[str]) -> Tuple[str, int]:
        """
        Process a syllable of Baybayin characters.
        
        Args:
            chars: List of characters in the syllable
            
        Returns:
            Tuple[str, int]: (romanized syllable, number of characters processed)
        """
        if not chars:
            return '', 0

        # Get first character info
        first_char_info = self.get_char_info(chars[0])
        if not first_char_info:
            return chars[0], 1

        # Handle vowels
        if first_char_info.char_type == BaybayinCharType.VOWEL:
            return first_char_info.default_sound, 1

        # Handle consonants
        if first_char_info.char_type == BaybayinCharType.CONSONANT:
            if len(chars) == 1:
                return first_char_info.default_sound, 1

            # Check next character
            next_char_info = self.get_char_info(chars[1]) if len(chars) > 1 else None

            if next_char_info:
                if next_char_info.char_type == BaybayinCharType.VOWEL_MARK:
                    # Consonant + vowel mark
                    base = first_char_info.default_sound[:-1]  # Remove inherent 'a'
                    return base + next_char_info.default_sound, 2
                elif next_char_info.char_type == BaybayinCharType.VIRAMA:
                    # Consonant + virama (kills vowel)
                    return first_char_info.default_sound[:-1], 2

            return first_char_info.default_sound, 1

        # Handle punctuation
        if first_char_info.char_type == BaybayinCharType.PUNCTUATION:
            return first_char_info.default_sound, 1

        return chars[0], 1

    def romanize(self, text: str) -> str:
        """
        Convert Baybayin text to romanized form.
        
        Args:
            text: Baybayin text to romanize
            
        Returns:
            str: Romanized text
        
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text:
            raise ValueError("Input text cannot be empty")

        result = []
        chars = list(text)
        i = 0
        
        while i < len(chars):
            # Skip spaces and preserve them
            if chars[i].isspace():
                result.append(chars[i])
                i += 1
                continue

            # Process next syllable
            romanized, chars_processed = self.process_syllable(chars[i:])
            result.append(romanized)
            i += chars_processed

        return ''.join(result)

    def validate_text(self, text: str) -> bool:
        """
        Validate Baybayin text for correct character combinations.
        
        Args:
            text: Text to validate
            
        Returns:
            bool: True if text is valid Baybayin
        """
        if not text:
            return False

        prev_char_info = None
        for char in text:
            if char.isspace():
                continue

            char_info = self.get_char_info(char)
            if not char_info:
                return False

            # Check invalid combinations
            if prev_char_info:
                # Can't have two vowels in a row
                if (prev_char_info.char_type == BaybayinCharType.VOWEL and 
                    char_info.char_type == BaybayinCharType.VOWEL):
                    return False
                
                # Can't have vowel mark after anything except consonant
                if (char_info.char_type == BaybayinCharType.VOWEL_MARK and 
                    prev_char_info.char_type != BaybayinCharType.CONSONANT):
                    return False
                
                # Can't have virama after anything except consonant
                if (char_info.char_type == BaybayinCharType.VIRAMA and 
                    prev_char_info.char_type != BaybayinCharType.CONSONANT):
                    return False

            prev_char_info = char_info

        return True

def process_baybayin_text(text: str) -> Tuple[str, Optional[str], bool]:
    """
    Process Baybayin text with proper romanization.
    
    Args:
        text: Text that might contain Baybayin
        
    Returns:
        Tuple[str, Optional[str], bool]: (processed_text, romanized_text, has_baybayin)
    """
    if not text:
        return text, None, False
        
    romanizer = BaybayinRomanizer()
    has_baybayin = romanizer.is_baybayin(text)
    
    if not has_baybayin:
        return text, None, False
        
    if not romanizer.validate_text(text):
        logger.warning(f"Invalid Baybayin text detected: {text}")
        return text, None, True
        
    try:
        romanized = romanizer.romanize(text)
        return text, romanized, True
    except ValueError as e:
        logger.error(f"Error romanizing Baybayin text: {str(e)}")
        return text, None, True

def get_romanized_text(text: str) -> str:
    """Convert Baybayin text to romanized form."""
    romanizer = BaybayinRomanizer()
    try:
        return romanizer.romanize(text)
    except ValueError:
        return text

def transliterate_to_baybayin(text: str) -> str:
    """Naively convert a Tagalog word in Latin script to its Baybayin representation."""
    text = text.lower().strip()
    consonants = {
        'k': 'ᜃ', 'g': 'ᜄ', 'ng': 'ᜅ', 't': 'ᜆ', 'd': 'ᜇ', 'n': 'ᜈ',
        'p': 'ᜉ', 'b': 'ᜊ', 'm': 'ᜋ', 'y': 'ᜌ', 'l': 'ᜎ', 'w': 'ᜏ',
        's': 'ᜐ', 'h': 'ᜑ'
    }
    vowels = {
        'a': 'ᜀ', 'i': 'ᜁ', 'u': 'ᜂ'
    }
    vowel_marks = {
        'i': 'ᜒ', 'u': 'ᜓ'
    }
    result = ""
    import re
    # A very naive syllable matcher: matches optional 'ng' or single consonant followed by a vowel
    syllables = re.findall(r'(ng|[kgtdnpmbylswh]?)(a|i|u)', text)
    for cons, vow in syllables:
        if cons:
            bayb_cons = consonants.get(cons, cons)
            if vow == 'a':
                result += bayb_cons
            else:
                result += bayb_cons + vowel_marks.get(vow, '')
        else:
            result += vowels.get(vow, vow)
    return result

def verify_baybayin_data(cur):
    """Verify Baybayin data after migration."""
    # Check for orphaned Baybayin forms
    cur.execute("""
        SELECT id, lemma, baybayin_form 
        FROM words 
        WHERE has_baybayin = TRUE AND baybayin_form IS NULL
    """)
    
    orphaned = cur.fetchall()
    if orphaned:
        logger.warning(f"Found {len(orphaned)} words marked as Baybayin but missing baybayin_form")
        for word_id, lemma, _ in orphaned:
            logger.warning(f"Word ID {word_id}: {lemma}")
            
    # Check for duplicate Baybayin forms
    cur.execute("""
        SELECT baybayin_form, COUNT(*) 
        FROM words 
        WHERE has_baybayin = TRUE 
        GROUP BY baybayin_form 
        HAVING COUNT(*) > 1
    """)
    
    duplicates = cur.fetchall()
    if duplicates:
        logger.warning(f"Found {len(duplicates)} Baybayin forms with multiple entries")
        for baybayin_form, count in duplicates:
            logger.warning(f"Baybayin form {baybayin_form} appears {count} times")

def merge_baybayin_entries(cur, baybayin_id: int, romanized_id: int):
    """
    Merge a Baybayin entry with its romanized equivalent.
    Preserves all information from both entries.
    
    Args:
        cur: Database cursor
        baybayin_id: ID of the Baybayin entry
        romanized_id: ID of the romanized entry
    """
    try:
        # Get Baybayin form and other data before transfer
        cur.execute("""
            SELECT lemma, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """, (baybayin_id,))
        baybayin_result = cur.fetchone()
        if not baybayin_result:
            raise ValueError(f"Baybayin entry {baybayin_id} not found")
            
        baybayin_lemma, baybayin_form, baybayin_rom = baybayin_result
        
        # Transfer all related data with proper error handling
        tables = [
            ('definitions', 'word_id'),
            ('relations', 'from_word_id'),
            ('relations', 'to_word_id'),
            ('etymologies', 'word_id'),
            ('definition_relations', 'word_id'),
            ('affixations', 'root_word_id'),
            ('affixations', 'affixed_word_id')
        ]
        
        for table, column in tables:
            cur.execute(f"""
                UPDATE {table} 
                SET {column} = %s 
                WHERE {column} = %s
            """, (romanized_id, baybayin_id))

        # Update the romanized entry with Baybayin information
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = COALESCE(%s, baybayin_form),
                romanized_form = COALESCE(%s, romanized_form),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (baybayin_form, baybayin_rom or baybayin_lemma, romanized_id))

        # Delete the Baybayin entry
        cur.execute("DELETE FROM words WHERE id = %s", (baybayin_id,))
        
    except Exception as e:
        logger.error(f"Error merging Baybayin entries: {str(e)}")
        raise

def extract_baybayin_text(text: str) -> List[str]:
    """
    Extract valid Baybayin text segments from a mixed string.
    
    Args:
        text: Input text that may contain Baybayin and other characters
        
    Returns:
        List of valid Baybayin text segments
    """
    # Split on non-Baybayin characters
    parts = re.split(r'[^ᜀ-᜔\s]+', text)
    # Filter out empty strings and clean each part
    return [part.strip() for part in parts if part.strip() and re.search(r'[\u1700-\u171F]', part)]

def validate_baybayin_entry(baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
    try:
        romanizer = BaybayinRomanizer()
        
        # Split on non-Baybayin characters and filter parts that contain Baybayin characters
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return False
            
        # Try each valid part without extra strict validation
        for part in sorted(valid_parts, key=len, reverse=True):
            if romanized_form:
                try:
                    generated_rom = romanizer.romanize(part)
                    if normalize_lemma(generated_rom) == normalize_lemma(romanized_form):
                        return True
                except ValueError:
                    continue
            else:
                return True
        
        return False
        
    except Exception:
        return False

def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> None:
    if not baybayin_form:
        return
        
    try:
        romanizer = BaybayinRomanizer()
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        if not valid_parts:
            logger.warning(f"No valid Baybayin segments found for word_id {word_id}: {baybayin_form}")
            return
            
        # Initialize variables before processing
        cleaned_baybayin = None
        romanized_value = None
        for part in sorted(valid_parts, key=len, reverse=True):
            if romanizer.validate_text(part):
                try:
                    romanized_value = romanizer.romanize(part)
                    cleaned_baybayin = part
                    break
                except ValueError:
                    continue
        if not cleaned_baybayin:
            logger.warning(f"Could not process any Baybayin segments for word_id {word_id}")
            return
            
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = %s,
                romanized_form = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (cleaned_baybayin, romanized_value, word_id))
        
    except Exception as e:
        logger.error(f"Error processing Baybayin data for word_id {word_id}: {str(e)}")
        raise

def process_baybayin_entries(cur):
    """
    Process all Baybayin entries, merging them with their romanized equivalents if they exist.
    Each entry is processed in its own transaction to prevent cascading failures.
    """
    logger.info("Processing Baybayin entries...")
    
    # Get all Baybayin entries
    cur.execute("""
        SELECT id, lemma, language_code, normalized_lemma 
        FROM words 
        WHERE lemma ~ '[\u1700-\u171F]'
        ORDER BY id ASC
    """)
    
    baybayin_entries = cur.fetchall()
    conn = cur.connection
    
    for baybayin_id, baybayin_lemma, language_code, current_normalized in baybayin_entries:
        try:
            # Start a transaction for each entry
            cur.execute("BEGIN")
            
            # Split and get valid Baybayin segments
            parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_lemma)
            valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
            
            if not valid_parts:
                logger.warning(f"No valid Baybayin segments found for entry {baybayin_id}: {baybayin_lemma}")
                conn.commit()
                continue
            
            # Try each valid part, starting with the longest
            romanizer = BaybayinRomanizer()
            cleaned_baybayin = None
            romanized = None
            
            for part in sorted(valid_parts, key=len, reverse=True):
                if romanizer.validate_text(part):
                    try:
                        romanized = romanizer.romanize(part)
                        cleaned_baybayin = part
                        break
                    except ValueError:
                        continue
            
            if not cleaned_baybayin or not romanized:
                logger.warning(f"Could not process any Baybayin segments for entry {baybayin_id}")
                conn.commit()
                continue
            
            # Update the entry with cleaned Baybayin form
            logger.info(f"Updating Baybayin entry (ID: {baybayin_id}) with cleaned form")
            cur.execute("""
                UPDATE words 
                SET romanized_form = %s,
                    baybayin_form = %s,
                    has_baybayin = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (romanized, cleaned_baybayin, baybayin_id))
            
            # Commit the transaction
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing Baybayin entry {baybayin_id}: {str(e)}")
            continue

def cleanup_baybayin_data(cur):
    """Clean up and standardize Baybayin data with proper transaction handling."""
    conn = cur.connection
    try:
        # Start a transaction
        cur.execute("BEGIN")
        
        # Remove invalid characters from Baybayin forms
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form,
                '[^ᜀ-᜔\\s]',  -- Only allow Baybayin range and whitespace
                '',
                'g'
            )
            WHERE has_baybayin = TRUE 
            AND baybayin_form IS NOT NULL
        """)
        
        # Normalize whitespace in Baybayin forms
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form, 
                '\s+',  -- Remove extra whitespace
                ' ', 
                'g'
            )
            WHERE has_baybayin = TRUE 
            AND baybayin_form IS NOT NULL
        """)
        
        # Remove empty or invalid Baybayin forms
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE, 
                baybayin_form = NULL
            WHERE has_baybayin = TRUE 
            AND (
                baybayin_form IS NULL 
                OR baybayin_form = ''
                OR baybayin_form !~ '[\u1700-\u171F]'  -- No valid Baybayin characters
            )
        """)
        
        # Remove inconsistent flags
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE,
                baybayin_form = NULL
            WHERE has_baybayin = FALSE 
            AND baybayin_form IS NOT NULL
        """)
        
        # Update search text for Baybayin entries
        cur.execute("""
            UPDATE words
            SET search_text = to_tsvector('english',
                COALESCE(lemma, '') || ' ' ||
                COALESCE(normalized_lemma, '') || ' ' ||
                COALESCE(baybayin_form, '') || ' ' ||
                COALESCE(romanized_form, '')
            )
            WHERE has_baybayin = TRUE
        """)
        
        # Remove duplicate Baybayin forms within same language
        cur.execute("""
            WITH DuplicateBaybayin AS (
                SELECT MIN(id) as keep_id,
                       language_code,
                       baybayin_form
                FROM words
                WHERE has_baybayin = TRUE
                  AND baybayin_form IS NOT NULL
                GROUP BY language_code, baybayin_form
                HAVING COUNT(*) > 1
            )
            UPDATE words w
            SET has_baybayin = FALSE,
                baybayin_form = NULL
            FROM DuplicateBaybayin d
            WHERE w.language_code = d.language_code
              AND w.baybayin_form = d.baybayin_form
              AND w.id != d.keep_id
        """)
        
        # Commit the transaction
        conn.commit()
        logger.info("Baybayin data cleanup completed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during Baybayin cleanup: {str(e)}")
        raise

def check_baybayin_consistency(cur):
    """
    Check consistency of Baybayin data across the database.
    """
    issues = []
    
    # Check for missing romanizations
    cur.execute("""
        SELECT id, lemma
        FROM words
        WHERE has_baybayin = TRUE AND romanized_form IS NULL
    """)
    missing_rom = cur.fetchall()
    if missing_rom:
        issues.append(f"Found {len(missing_rom)} entries missing romanization")
        for word_id, lemma in missing_rom:
            logger.warning(f"Missing romanization for word ID {word_id}: {lemma}")
    
    # Check for inconsistent flags
    cur.execute("""
        SELECT id, lemma
        FROM words
        WHERE (has_baybayin = TRUE AND baybayin_form IS NULL)
           OR (has_baybayin = FALSE AND baybayin_form IS NOT NULL)
    """)
    inconsistent = cur.fetchall()
    if inconsistent:
        issues.append(f"Found {len(inconsistent)} entries with inconsistent Baybayin flags")
        for word_id, lemma in inconsistent:
            logger.warning(f"Inconsistent Baybayin flags for word ID {word_id}: {lemma}")
    
    # Check for invalid Baybayin characters
    cur.execute(r"""
        SELECT id, lemma, baybayin_form
        FROM words
        WHERE baybayin_form ~ '[^ᜀ-᜔\s]'  -- Characters outside Baybayin range
    """)
    invalid_chars = cur.fetchall()
    if invalid_chars:
        issues.append(f"Found {len(invalid_chars)} entries with invalid Baybayin characters")
        for word_id, lemma, baybayin in invalid_chars:
            logger.warning(f"Invalid Baybayin characters in word ID {word_id}: {lemma}")
    
    return issues

def validate_baybayin_entry(baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
    try:
        romanizer = BaybayinRomanizer()
        
        # Split on non-Baybayin characters and filter parts that contain Baybayin characters
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return False
            
        # Try each valid part without extra strict validation
        for part in sorted(valid_parts, key=len, reverse=True):
            if romanized_form:
                try:
                    generated_rom = romanizer.romanize(part)
                    if normalize_lemma(generated_rom) == normalize_lemma(romanized_form):
                        return True
                except ValueError:
                    continue
            else:
                return True
        
        return False
        
    except Exception:
        return False

def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> None:
    if not baybayin_form:
        return
        
    try:
        romanizer = BaybayinRomanizer()
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        if not valid_parts:
            logger.warning(f"No valid Baybayin segments found for word_id {word_id}: {baybayin_form}")
            return
            
        # Initialize variables before processing
        cleaned_baybayin = None
        romanized_value = None
        for part in sorted(valid_parts, key=len, reverse=True):
            if romanizer.validate_text(part):
                try:
                    romanized_value = romanizer.romanize(part)
                    cleaned_baybayin = part
                    break
                except ValueError:
                    continue
        if not cleaned_baybayin:
            logger.warning(f"Could not process any Baybayin segments for word_id {word_id}")
            return
            
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = %s,
                romanized_form = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (cleaned_baybayin, romanized_value, word_id))
        
    except Exception as e:
        logger.error(f"Error processing Baybayin data for word_id {word_id}: {str(e)}")
        raise

# -------------------------------------------------------------------
# Core Data Processing Functions
# -------------------------------------------------------------------

def get_standardized_pos_id(cur, pos_string: str) -> int:
    """Maps input POS string to standardized POS ID using POS_MAPPING."""
    if not pos_string:
        return get_uncategorized_pos_id(cur)

    pos_clean = pos_string.lower().strip()
    
    # Get standardized pos code based on mapping
    std_code = None
    for key, value in POS_MAPPING.items():
        if (pos_clean == key.lower() or 
            pos_clean == value['en'].lower() or 
            pos_clean == value['tl'].lower()):
            if key in ('noun', 'pangngalan', 'pnd', 'png'):
                std_code = 'n'
            elif key in ('verb', 'pandiwa', 'pnw'):
                std_code = 'v'
            elif key in ('adjective', 'pang-uri', 'pnu'):
                std_code = 'adj'
            elif key in ('adverb', 'pang-abay', 'pny'):
                std_code = 'adv'
            elif key in ('pronoun', 'panghalip', 'pnr'):
                std_code = 'pron'
            elif key in ('preposition', 'pang-ukol'):
                std_code = 'prep'
            elif key in ('conjunction', 'pangatnig'):
                std_code = 'conj'
            elif key in ('interjection', 'pandamdam'):
                std_code = 'intj'
            elif key in ('affix', 'panlapi', 'pnl'):
                std_code = 'affix'
            elif key == 'baybayin':
                std_code = 'baybay'
            elif key in ('det', 'determiner', 'pantukoy'):
                std_code = 'det'
            break

    if not std_code:
        return get_uncategorized_pos_id(cur)

    # Get ID from parts_of_speech table
    cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (std_code,))
    result = cur.fetchone()
    return result[0] if result else get_uncategorized_pos_id(cur)

def get_uncategorized_pos_id(cur) -> int:
    """Get the ID for uncategorized POS."""
    cur.execute("SELECT id FROM parts_of_speech WHERE code = 'unc'")
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        # Insert 'unc' POS if it doesn't exist
        cur.execute("""
            INSERT INTO parts_of_speech (code, name_en, name_tl, description)
            VALUES ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
            RETURNING id
        """)
        return cur.fetchone()[0]

@with_transaction(commit=True)
def get_or_create_word_id(
    cur,
    lemma: str,
    language_code: str = "tl",
    **kwargs  # Add this
) -> int:
    """Get or create a word entry."""
    if 'entry_data' in kwargs:
        metadata = processor.process_word_metadata(kwargs['entry_data'], 
            SourceStandardization.get_source_enum(kwargs.get('source', '')))
        kwargs.update(metadata)
    return get_or_create_word_id_base(cur, lemma, language_code, **kwargs)

@with_transaction(commit=True)
def get_or_create_word_id_base(
    cur,
    lemma: str,
    language_code: str = "tl",
    **kwargs
) -> int:
    """Base function to get or create a word entry."""
    
    # Validate word data before processing
    try:
        _ = validate_word_data({"lemma": lemma, "language_code": language_code})
    except ValueError as ve:
        logger.error(f"Invalid word data for '{lemma}': {ve}")
        raise
    
    normalized = normalize_lemma(lemma)
    
    # If Baybayin flag is set, validate baybayin_form and romanized_form
    if kwargs.get('has_baybayin'):
        bb_form = kwargs.get('baybayin_form')
        r_form = kwargs.get('romanized_form')
        if bb_form and not validate_baybayin_entry(bb_form, r_form):
            logger.warning(f"Invalid Baybayin data for word '{lemma}', clearing baybayin fields.")
            kwargs['has_baybayin'] = False
            kwargs['baybayin_form'] = None
            kwargs['romanized_form'] = None
    
    # Try to get existing word first
    cur.execute("""
        SELECT id FROM words 
        WHERE normalized_lemma = %s AND language_code = %s
    """, (normalized, language_code))
    
    result = cur.fetchone()
    if result:
        return result[0]
    
    # Create search text
    search_text = f"{lemma} {normalized}"
    if kwargs.get('baybayin_form'):
        search_text += f" {kwargs['baybayin_form']}"
    if kwargs.get('romanized_form'):
        search_text += f" {kwargs['romanized_form']}"
    
    # Insert new word
    cur.execute("""
        INSERT INTO words (
            lemma, normalized_lemma, language_code, 
            root_word_id, tags, has_baybayin, 
            baybayin_form, romanized_form, search_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))
        ON CONFLICT (normalized_lemma, language_code) 
        DO UPDATE SET 
            lemma = EXCLUDED.lemma,
            search_text = EXCLUDED.search_text,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (
        lemma, normalized, language_code,
        kwargs.get('root_word_id'), kwargs.get('tags'),
        kwargs.get('has_baybayin', False), kwargs.get('baybayin_form'),
        kwargs.get('romanized_form'), search_text
    ))
    
    return cur.fetchone()[0]

@with_transaction(commit=True)
def insert_definition(
    cur,
    word_id: int,
    definition_text: str,
    part_of_speech: str = "",
    examples: str = None,
    usage_notes: str = None,
    category: str = None,
    sources: str = ""
) -> Optional[int]:
    """
    Insert a definition, skipping if it's a Baybayin alternative.
    
    Args:
        cur: Database cursor
        word_id: ID of the word
        definition_text: The definition text
        part_of_speech: Part of speech
        examples: Optional usage examples
        usage_notes: Optional usage notes
        category: Optional category information
        sources: Source identifier
        
    Returns:
        Optional[int]: ID of the inserted definition or None
    """
    if 'Baybayin spelling of' in definition_text:
        return None
        
    # --- New duplicate check added ---
    cur.execute("SELECT id FROM definitions WHERE word_id = %s AND definition_text = %s AND original_pos = %s", 
                (word_id, definition_text, part_of_speech))
    if cur.fetchone():
        return None

    std_pos_id = get_standardized_pos_id(cur, part_of_speech)
    
    # If category exists, add it to usage_notes
    if category:
        usage_notes = f"[{category}] {usage_notes if usage_notes else ''}"
    
    cur.execute("""
        INSERT INTO definitions 
            (word_id, definition_text, original_pos, standardized_pos_id,
             examples, usage_notes, sources)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        word_id,
        definition_text,
        part_of_speech,
        std_pos_id,
        examples,
        usage_notes,
        sources
    ))
    
    return cur.fetchone()[0]

@with_transaction(commit=True)
def insert_relation(
    cur,
    from_word_id: int,
    to_word_id: int,
    relation_type: str,
    sources: str = ""
):
    """Insert a relation between words."""
    cur.execute("""
        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (from_word_id, to_word_id, relation_type) DO UPDATE
        SET sources = CASE 
            WHEN relations.sources IS NULL THEN EXCLUDED.sources
            WHEN EXCLUDED.sources IS NULL THEN relations.sources
            ELSE relations.sources || ', ' || EXCLUDED.sources
        END
    """, (from_word_id, to_word_id, relation_type, sources))

@with_transaction(commit=True)
def insert_definition_relation(
    cur,
    definition_id: int,
    word_id: int,
    relation_type: str,
    sources: str = "auto"
):
    """Insert a relation specific to a definition."""
    cur.execute("""
        INSERT INTO definition_relations 
            (definition_id, word_id, relation_type, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (definition_id, word_id, relation_type) 
        DO UPDATE SET sources = CASE 
            WHEN definition_relations.sources IS NULL THEN EXCLUDED.sources
            WHEN EXCLUDED.sources IS NULL THEN definition_relations.sources
            ELSE definition_relations.sources || ', ' || EXCLUDED.sources
        END
    """, (definition_id, word_id, relation_type, sources))

@with_transaction(commit=True)
def insert_etymology(
    cur,
    word_id: int,
    etymology_text: str,
    normalized_components: Optional[str] = None,
    language_codes: str = "",
    sources: str = ""
) -> None:
    """Insert etymology information for a word."""
    if not etymology_text:
        return
        
    # Skip if it's just describing Baybayin spelling
    if 'Baybayin spelling of' in etymology_text:
        return
        
    cur.execute("""
        INSERT INTO etymologies 
            (word_id, etymology_text, normalized_components, language_codes, sources)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (word_id, etymology_text) DO UPDATE 
        SET normalized_components = EXCLUDED.normalized_components,
            language_codes = EXCLUDED.language_codes,
            sources = CASE 
                WHEN etymologies.sources IS NULL THEN EXCLUDED.sources
                WHEN EXCLUDED.sources IS NULL THEN etymologies.sources
                ELSE etymologies.sources || ', ' || EXCLUDED.sources
            END
    """, (word_id, etymology_text, normalized_components, language_codes, sources))

@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    sources: str
):
    """Insert affixation information."""
    cur.execute("""
        INSERT INTO affixations 
            (root_word_id, affixed_word_id, affix_type, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (root_word_id, affixed_word_id, affix_type)
        DO UPDATE SET sources = CASE 
            WHEN affixations.sources IS NULL THEN EXCLUDED.sources
            WHEN EXCLUDED.sources IS NULL THEN affixations.sources
            ELSE affixations.sources || ', ' || EXCLUDED.sources
        END
    """, (root_id, affixed_id, affix_type, sources))

def batch_get_or_create_word_ids(
    cur: "psycopg2.extensions.cursor",
    entries: List[Tuple[str, str]],
    batch_size: int = 1000
) -> Dict[Tuple[str, str], int]:
    """
    Batch process multiple word entries efficiently.
    
    Args:
        cur: Database cursor
        entries: List of (lemma, language_code) tuples
        batch_size: Size of each batch (default: 1000)
        
    Returns:
        Dictionary mapping (lemma, language_code) to word ID
    """
    result = {}
    
    # Process in batches
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        
        # Prepare normalized forms
        normalized_entries = [
            (lemma, normalize_lemma(lemma), lang_code)
            for lemma, lang_code in batch
        ]
        
        # Check existing entries
        cur.execute("""
            SELECT lemma, language_code, id
            FROM words
            WHERE (normalized_lemma, language_code) IN %s
        """, (tuple((norm, lang) for _, norm, lang in normalized_entries),))
        
        # Process existing entries
        existing = {(lemma, lang): id for lemma, lang, id in cur.fetchall()}
        
        # Prepare entries to insert
        to_insert = [
            (lemma, norm_lemma, lang)
            for lemma, norm_lemma, lang in normalized_entries
            if (lemma, lang) not in existing
        ]
        
        if to_insert:
            # Create search vectors for new entries
            search_vectors = []
            for lemma, norm, lang in to_insert:
                search_text = f"{lemma} {norm}"
                search_vectors.append(search_text)
                
            # Batch insert new entries
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO words 
                    (lemma, normalized_lemma, language_code, search_text)
                VALUES %s
                RETURNING lemma, language_code, id
                """,
                [(lemma, norm, lang, text) 
                 for (lemma, norm, lang), text in zip(to_insert, search_vectors)]
            )
            
            # Add newly inserted entries to results
            for lemma, lang, id in cur.fetchall():
                existing[(lemma, lang)] = id
        
        result.update(existing)
    
    return result

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------

def process_kwf_entry(cur, lemma: str, entry: Dict, src: str = "kwf"):
    """
    Process a KWF dictionary entry with comprehensive information capture.
    
    Args:
        cur: Database cursor
        lemma: Word lemma
        entry: Dictionary entry data
        src: Source identifier (default: "kwf")
    """
    try:
        # Create base word entry
        word_id = get_or_create_word_id(
            cur,
            entry['formatted'],
            language_code="tl",
            tags="KWF"
        )

        # Process definitions by part of speech
        for pos in entry.get('part_of_speech', []):
            pos_definitions = entry.get('definitions', {}).get(pos, [])
            
            for def_entry in pos_definitions:
                try:
                    # Insert main definition
                    def_id = insert_definition(
                        cur,
                        word_id,
                        def_entry['meaning'],
                        pos,
                        category=def_entry.get('category'),
                        sources=src
                    )

                    # Process synonyms for this definition
                    for synonym in def_entry.get('synonyms', []):
                        try:
                            if not synonym or not synonym.strip():
                                continue
                                
                            syn_id = get_or_create_word_id(cur, synonym, language_code="tl")
                            insert_relation(cur, word_id, syn_id, "synonym", src)
                            if def_id:
                                insert_definition_relation(cur, def_id, syn_id, "synonym")
                        except Exception as e:
                            logger.error(f"Error processing synonym {synonym} for {lemma}: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Error processing definition for {lemma}: {str(e)}")
                    continue

        # Process affixation
        for affix_group in entry.get('affixation', []):
            pos_type = affix_group['type']
            for affixed_form in affix_group.get('form', []):
                try:
                    if not affixed_form or not affixed_form.strip():
                        continue
                        
                    # Create word entry for affixed form
                    affixed_id = get_or_create_word_id(
                        cur, 
                        affixed_form,
                        language_code="tl",
                        root_word_id=word_id
                    )
                    # Store affixation type
                    insert_affixation(
                        cur,
                        root_id=word_id,
                        affixed_id=affixed_id,
                        affix_type=pos_type,
                        sources=src
                    )
                except Exception as e:
                    logger.error(f"Error processing affixed form {affixed_form} for {lemma}: {str(e)}")
                    continue

        # Process idioms
        idioms_list = []
        for idiom in entry.get('idioms', []):
            try:
                idiom_text = idiom.get('idiom', '').strip()
                if not idiom_text:
                    continue

                idiom_meaning = idiom.get('meaning', '').strip()
                if not idiom_meaning:
                    logger.warning(f"Idiom '{idiom_text}' in {lemma} has no meaning; skipping.")
                    continue

                idiom_examples = idiom.get('examples', [])
                if not isinstance(idiom_examples, list):
                    idiom_examples = [idiom_examples]

                # Clean up examples
                cleaned_examples = [ex.strip() for ex in idiom_examples if ex and ex.strip()]

                # Create idiom object
                idiom_obj = {
                    "idiom": idiom_text,
                    "meaning": idiom_meaning,
                    "examples": cleaned_examples if cleaned_examples else None
                }
                idioms_list.append(idiom_obj)
            except Exception as e:
                logger.error(f"Error processing idiom '{idiom.get('idiom', '')}' for {lemma}: {str(e)}")
                continue

        # Update the word record with the idioms list if any exist
        if idioms_list:
            try:
                cur.execute("""
                    UPDATE words 
                    SET idioms = %s::jsonb,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (json.dumps(idioms_list), word_id))
            except Exception as e:
                logger.error(f"Error updating idioms for word {lemma}: {str(e)}")

        # Process metadata if available
        metadata = entry.get('metadata', {})
        etymology_text = metadata.get('etymology')
        if etymology_text:
            insert_etymology(
                cur,
                word_id,
                etymology_text,
                normalized_components=None,
                language_codes="",
                sources=src
            )

        pronunciation = metadata.get('pronunciation')
        if pronunciation:
            try:
                cur.execute(
                    "UPDATE words SET pronunciation_data = %s WHERE id = %s",
                    (json.dumps(pronunciation), word_id)
                )
            except Exception as e:
                logger.error(f"Error updating pronunciation for {lemma}: {str(e)}")

        # Process related terms
        related = entry.get('related', {})
        related_terms = related.get('related_terms', [])
        for rel in related_terms:
            term = rel.get('term', '').strip()
            if term:
                try:
                    rel_id = get_or_create_word_id(cur, term, language_code="tl")
                    insert_relation(cur, word_id, rel_id, "related", src)
                except Exception as e:
                    logger.error(f"Error processing related term {term} for {lemma}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing KWF entry '{lemma}': {str(e)}")
        raise

def process_kaikki_jsonl_new(cur, filename: str, check_exists: bool = False):
    """
    Process Kaikki dictionary entries with improved Baybayin handling.
    
    Args:
        cur: Database cursor
        filename: Path to the JSONL file
        check_exists: Whether to check for existing entries
    """
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract Baybayin information from entry using improved parsing."""
        import re
        baybayin = None
        romanized = None

        word = entry.get('word', '')
        # If the word itself is in Baybayin script, use it directly
        if any(0x1700 <= ord(c) <= 0x171F for c in word):
            baybayin = word
            for form in entry.get('forms', []):
                if 'tags' in form and 'romanization' in form['tags']:
                    romanized = form['form']
                    break
            if not romanized:
                try:
                    romanized = get_romanized_text(baybayin)
                except Exception:
                    romanized = word
            return baybayin, romanized

        forms = entry.get('forms', [])
        canonical_form = None
        for form in forms:
            if 'tags' in form:
                if 'canonical' in form['tags']:
                    canonical_form = form['form']
                elif 'Baybayin' in form['tags']:
                    baybayin = form['form']

        if baybayin and not romanized:
            try:
                romanized = get_romanized_text(baybayin)
            except Exception:
                romanized = canonical_form or word

        # Use head_templates if no Baybayin form found in forms
        if not baybayin:
            for template in entry.get('head_templates', []):
                if template.get('name', '').startswith(('tl-', 'ceb-')):
                    expansion = template.get('expansion', '')
                    # Find all sequences of Baybayin characters
                    baybayin_candidates = re.findall(r"([ᜀ-ᜟ]+)", expansion)
                    if baybayin_candidates:
                        # Choose the longest candidate as the Baybayin form
                        baybayin = max(baybayin_candidates, key=len)
                        try:
                            romanized = get_romanized_text(baybayin)
                        except Exception:
                            romanized = canonical_form or word
                        break

        if baybayin:
            logger.debug(f"Extracted Baybayin: {baybayin}, Romanized: {romanized}")

        return baybayin, romanized

    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry."""
        try:
            lemma = entry.get("word", "").strip()
            if not lemma:
                return

            # Handle Baybayin entries
            baybayin_form, romanized = extract_baybayin_info(entry)
            has_baybayin = bool(baybayin_form)
            
            # Determine language code based on filename
            language_code = "tl" if "kaikki.jsonl" in filename else "ceb"
            
            # Create or update word entry with Baybayin information
            word_id = get_or_create_word_id(
                cur,
                lemma,
                language_code=language_code,
                has_baybayin=has_baybayin,
                baybayin_form=baybayin_form,
                romanized_form=romanized if has_baybayin else None
            )

            # Process definitions
            entry_pos = entry.get("pos", "").strip()
            for sense in entry.get("senses", []):
                if all('Baybayin spelling of' in gloss for gloss in sense.get("glosses", [])):
                    continue

                pos = sense.get("pos", entry_pos).strip()
                pos = ", ".join(filter(None, [p.strip() for p in pos.split(",")]))

                examples = []
                for ex in sense.get("examples", []):
                    if isinstance(ex, dict):
                        example_text = ex.get("text", "")
                        if example_text:
                            examples.append(example_text)
                    elif isinstance(ex, str):
                        examples.append(ex)

                for gloss in sense.get("glosses", []):
                    if gloss and not "Baybayin spelling of" in gloss:
                        insert_definition(
                            cur, 
                            word_id, 
                            gloss.strip(),
                            pos,
                            examples="\n".join(examples) if examples else None,
                            usage_notes=None,
                            sources=os.path.basename(filename)
                        )

            # Process etymology
            etymology = entry.get("etymology_text", "")
            if etymology:
                # Use language_systems.py for extraction
                codes, cleaned_ety = lsys.extract_and_remove_language_codes(etymology)
                insert_etymology(
                    cur=cur,
                    word_id=word_id,
                    etymology_text=cleaned_ety,
                    normalized_components=None,
                    language_codes=", ".join(codes) if codes else "",
                    sources=os.path.basename(filename)
                )

            # Process relations
            for derived in entry.get("derived", []):
                der_word = derived.get("word", "") if isinstance(derived, dict) else str(derived)
                if der_word := der_word.strip():
                    der_id = get_or_create_word_id(cur, der_word, language_code)
                    insert_relation(
                        cur,
                        word_id,
                        der_id,
                        "derived_from",
                        os.path.basename(filename)
                    )

            for syn in entry.get("synonyms", []):
                syn_word = syn.get("word", "") if isinstance(syn, dict) else str(syn)
                if syn_word := syn_word.strip():
                    syn_id = get_or_create_word_id(cur, syn_word, language_code)
                    insert_relation(
                        cur,
                        word_id,
                        syn_id,
                        "synonym",
                        os.path.basename(filename)
                    )

            for rel in entry.get("related", []):
                rel_word = rel.get("word", "") if isinstance(rel, dict) else str(rel)
                if rel_word := rel_word.strip():
                    rel_id = get_or_create_word_id(cur, rel_word, language_code)
                    insert_relation(
                        cur,
                        word_id,
                        rel_id,
                        "related",
                        os.path.basename(filename)
                    )

        except Exception as e:
            logger.error(f"Error processing entry '{lemma}': {str(e)}")
            return

    logger.info(f"Starting to process Kaikki file: {filename}")

    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing entries from {filename}"):
                process_entry(cur, json.loads(line))
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise

def process_tagalog_words(cur, filename: str):
    """Process Tagalog words dictionary."""
    logger.info(f"Processing Tagalog words from: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return
    
    src = os.path.basename(filename)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            
        # Ensure entries is a list of dictionaries
        if isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries):
            for entry in tqdm(entries, desc="Processing Tagalog words"):
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        entry['lemma'],
                        language_code="tl",
                        tags=src
                    )
                    
                    # Process definitions
                    if 'definitions' in entry:
                        for definition in entry['definitions']:
                            insert_definition(
                                cur,
                                word_id,
                                definition['text'],
                                definition.get('pos', ''),
                                sources=src
                            )
                            
                    # Process synonyms
                    if 'synonyms' in entry:
                        for synonym in entry['synonyms']:
                            syn_id = get_or_create_word_id(cur, synonym, language_code="tl")
                            insert_relation(cur, word_id, syn_id, "synonym", src)
                    
                    # Process affixations
                    if 'affixation' in entry:
                        for affix_type, affixed_forms in entry['affixation'].items():
                            for affixed_form in affixed_forms:
                                affixed_id = get_or_create_word_id(
                                    cur, 
                                    affixed_form,
                                    language_code="tl",
                                    root_word_id=word_id
                                )
                                insert_affixation(
                                    cur,
                                    root_id=word_id,
                                    affixed_id=affixed_id,
                                    affix_type=affix_type,
                                    sources=src
                                )
                    
                except Exception as e:
                    logger.error(f"Error processing entry: {str(e)}")
                    continue
        else:
            # Handle case where entries is a dictionary mapping lemmas to their data
            if isinstance(entries, dict):
                for lemma, entry_data in tqdm(entries.items(), desc="Processing Tagalog words"):
                    try:
                        word_id = get_or_create_word_id(
                            cur,
                            lemma,
                            language_code="tl",
                            tags=src
                        )
                        
                        # Process definitions
                        if isinstance(entry_data, dict) and 'definitions' in entry_data:
                            for definition in entry_data['definitions']:
                                insert_definition(
                                    cur,
                                    word_id,
                                    definition['text'] if isinstance(definition, dict) else definition,
                                    entry_data.get('pos', ''),
                                    sources=src
                                )
                                
                        # If entry_data is a string, treat it as a definition
                        elif isinstance(entry_data, str):
                            insert_definition(
                                cur,
                                word_id,
                                entry_data,
                                '',  # No POS information available
                                sources=src
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing lemma {lemma}: {str(e)}")
                        continue
            else:
                raise ValueError("Unsupported data format in tagalog-words.json")
    
    except Exception as e:
        logger.error(f"Error processing Tagalog words file: {str(e)}")
        raise

def process_root_words(cur, filename: str):
    """Process root words and their derivatives."""
    logger.info(f"Processing root words from: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return
        
    src = os.path.basename(filename)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for root_lemma, derivatives in tqdm(data.items(), desc="Processing root words"):
            try:
                # Create root word entry
                root_id = get_or_create_word_id(
                    cur,
                    root_lemma,
                    language_code="tl"
                )
                
                # Process derivatives
                for derived_lemma, details in derivatives.items():
                    try:
                        # Create derivative entry
                        derived_id = get_or_create_word_id(
                            cur,
                            derived_lemma,
                            language_code="tl",
                            root_word_id=root_id
                        )
                        
                        # Add definition if available
                        if 'definition' in details:
                            insert_definition(
                                cur,
                                derived_id,
                                details['definition'],
                                details.get('type', ''),
                                sources=src
                            )
                            
                        # Add relation
                        insert_relation(
                            cur,
                            derived_id,
                            root_id,
                            "root_of",
                            src
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing derivative {derived_lemma}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing root word {root_lemma}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing root words file: {str(e)}")
        raise

def process_kwf_dictionary(cur, filename: str):
    """Process the KWF dictionary file."""
    logger.info(f"Processing KWF dictionary from: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return
        
    src = os.path.basename(filename)
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            
        for lemma, entry in tqdm(entries.items(), desc="Processing KWF entries"):
            try:
                process_kwf_entry(cur, lemma, entry, src)
            except Exception as e:
                logger.error(f"Error processing KWF entry '{lemma}': {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing KWF dictionary file: {str(e)}")
        raise

# -------------------------------------------------------------------
# Command Line Interface Functions
# -------------------------------------------------------------------

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Manage dictionary data in PostgreSQL.")
    subparsers = parser.add_subparsers(dest="command")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Create/update schema and load data")
    migrate_parser.add_argument("--check-exists", action="store_true", help="Skip identical entries")
    migrate_parser.add_argument("--force", action="store_true", help="Force migration without confirmation")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity")
    verify_parser.add_argument('--quick', action='store_true', help='Run quick verification')
    verify_parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')

    # Update command
    update_parser = subparsers.add_parser("update", help="Update DB with new data")
    update_parser.add_argument("--file", type=str, help="JSON or JSONL file to use", required=True)
    update_parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")

    # Lookup command
    lookup_parser = subparsers.add_parser("lookup", help="Look up word information")
    lookup_parser.add_argument("word", help="Word to look up")
    lookup_parser.add_argument("--debug", action="store_true", help="Show debug information")
    lookup_parser.add_argument("--format", choices=['text', 'json', 'rich'], default='rich', 
                             help="Output format")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Display dictionary statistics")
    stats_parser.add_argument("--detailed", action="store_true", help="Show detailed statistics")
    stats_parser.add_argument("--export", type=str, help="Export statistics to file")

    # Other commands
    subparsers.add_parser("leaderboard", help="Display top contributors")
    subparsers.add_parser("help", help="Display help information")
    
    purge_parser = subparsers.add_parser("purge", help="Safely delete all data")
    purge_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up dictionary data by removing duplicates and standardizing formats")

    return parser

def migrate_data(args):
    """Migrate dictionary data to PostgreSQL with improved error handling."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        if not args.force:
            console = Console()
            console.print("\n[bold yellow]Warning:[/] This will reset and rebuild the database.")
            console.print("All existing data will be replaced with fresh data from source files.")
            confirmation = input("\nType 'YES' to continue: ")
            if confirmation != "YES":
                logger.info("Migration cancelled")
                return

        # Set up database
        setup_extensions(conn)
        create_or_update_tables(conn)
        
        # Process source files
        source_files = [
            {
                "filename": "data/tagalog-words.json",
                "processor": process_tagalog_words,
                "language_code": "tl"
            },
            {
                "filename": "data/root_words_with_associated_words_cleaned.json",
                "source_enum": SourceStandardization.get_source_enum("root_words_with_associated_words_cleaned.json"),
                "processor": process_root_words,
                "language_code": "tl"
            },
            {
                "filename": "data/kwf_dictionary.json",
                "source_enum": SourceStandardization.get_source_enum("kwf_dictionary.json"),
                "processor": process_kwf_dictionary,
                "language_code": "tl"
            },
            {
                "filename": "data/kaikki.jsonl",
                "source_enum": SourceStandardization.get_source_enum("kaikki.jsonl"),
                "processor": process_kaikki_jsonl_new,
                "language_code": "tl"
            },
            {
                "filename": "data/kaikki-ceb.jsonl",
                "source_enum": SourceStandardization.get_source_enum("kaikki-ceb.jsonl"),
                "processor": process_kaikki_jsonl_new,
                "language_code": "ceb"
            }
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=Console()
        ) as progress:
            for source in source_files:
                filename = source["filename"]
                processor_func = source["processor"]
                
                if not os.path.exists(filename):
                    logger.warning(f"Source file not found: {filename}")
                    continue

                task = progress.add_task(f"Processing {filename}...", total=None)
                
                try:
                    cur.execute("BEGIN")
                    
                    if filename.endswith('.json'):
                        processor_func(cur, filename)
                    elif filename.endswith('.jsonl'):
                        processor_func(cur, filename)
                    else:
                        logger.warning(f"Unsupported file format for: {filename}")
                        continue

                    conn.commit()
                    progress.update(task, description=f"Completed {filename}")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error processing file {filename}: {str(e)}")
                    if not args.force:
                        raise
                    else:
                        logger.warning(f"Skipping file {filename} due to errors.")

        # Post-processing with transaction handling
        try:
            process_baybayin_entries(cur)
            cleanup_baybayin_data(cur)
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during post-processing: {str(e)}")
            raise
            
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def store_processed_entry(cur, word_id: int, processed: Dict) -> None:
    """Store a processed dictionary entry in the database."""
    try:
        # Store definitions with standardized POS
        for definition in processed['data'].get('definitions', []):
            pos = processed['data'].get('pos', '')
            if isinstance(pos, dict):
                pos = f"{pos['english']} ({pos['filipino']})"
                
            insert_definition(
                cur, 
                word_id, 
                definition.get('meaning', definition) if isinstance(definition, dict) else definition,
                pos,
                examples=definition.get('examples'),
                usage_notes=definition.get('usage_notes'),
                category=definition.get('domain'),
                sources=processed['source']
            )
        
        # Store forms with processor's Baybayin validation
        for form in processed['data'].get('forms', []):
            if 'Baybayin' in form.get('tags', []):
                if validate_baybayin_entry(form['form']):
                    cur.execute("""
                        UPDATE words 
                        SET has_baybayin = TRUE, 
                            baybayin_form = %s,
                            romanized_form = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (form['form'], form.get('romanized_form', get_romanized_text(form['form'])), word_id))
        
        # Store metadata using processor's metadata handling
        if 'metadata' in processed['data']:
            metadata = processed['data']['metadata']
            
            # Etymology with source language detection
            if 'etymology' in metadata:
                etymology_data = processor.process_etymology(
                    metadata['etymology'], 
                    processed['source_enum']
                )
                insert_etymology(
                    cur, 
                    word_id, 
                    etymology_data['etymology_text'],
                    etymology_data.get('normalized_components'),
                    etymology_data.get('language_codes'),
                    processed['source']
                )
                
            # Pronunciation with standardization
            if 'pronunciation' in metadata:
                pron_data = processor.standardize_pronunciation(
                    metadata['pronunciation']
                )
                if pron_data:
                    cur.execute("""
                        UPDATE words 
                        SET romanized_form = %s,
                            pronunciation_data = %s
                        WHERE id = %s
                    """, (pron_data['romanized'], json.dumps(pron_data), word_id))
        
        # Store relations with validation
        if 'related_words' in processed['data']:
            for rel_type, related in processed['data']['related_words'].items():
                if isinstance(related, str):
                    related = [related]
                for rel_word in related:
                    # Validate relation type
                    if processor.validate_relation_type(rel_type):
                        rel_id = get_or_create_word_id(cur, rel_word, language_code="tl")
                        insert_relation(cur, word_id, rel_id, rel_type, processed['source'])
                    
    except Exception as e:
        logger.error(f"Error storing processed entry: {str(e)}")
        raise

def verify_database(args):
    """
    Verify database integrity with optional quick test mode and repair.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting database verification")
    conn = get_connection()
    cur = conn.cursor()
    console = Console()
    
    try:
        issues = []
        
        # Basic table counts
        table_stats = Table(title="Table Statistics", box=box.ROUNDED)
        table_stats.add_column("Table", style="cyan")
        table_stats.add_column("Count", justify="right", style="green")
        
        tables = ["words", "definitions", "relations", "etymologies", "affixations", "definition_relations", "parts_of_speech"]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            table_stats.add_row(t, f"{count:,}")
            
        console.print(table_stats)
        console.print()

        if args.quick:
            logger.info("Running quick verification...")
            console.print("[yellow]Sample entries from 'words' table:[/]")
            cur.execute("""
                SELECT id, lemma, language_code, root_word_id 
                FROM words 
                LIMIT 5
            """)
            sample_table = Table(show_header=True)
            sample_table.add_column("ID")
            sample_table.add_column("Lemma")
            sample_table.add_column("Language")
            sample_table.add_column("Root ID")
            
            for row in cur.fetchall():
                sample_table.add_row(*[str(x) for x in row])
                
            console.print(sample_table)
            return

        # Full verification
        logger.info("Running full verification checks...")
        
        # Check data integrity
        integrity_issues = check_data_integrity(cur)
        if integrity_issues:
            issues.extend(integrity_issues)
            
        # Check Baybayin consistency
        baybayin_issues = check_baybayin_consistency(cur)
        if baybayin_issues:
            issues.extend(baybayin_issues)
            
        # Report issues
        if issues:
            console.print("\n[bold red]Found Issues:[/]")
            issues_table = Table(show_header=True)
            issues_table.add_column("Issue")
            issues_table.add_column("Details")
            
            for issue in issues:
                issues_table.add_row(issue, "")
                
            console.print(issues_table)
            
            # Attempt repair if requested
            if args.repair:
                console.print("\n[yellow]Attempting to repair issues...[/]")
                repair_database_issues(cur, issues)
        else:
            console.print("\n[bold green]No issues found![/]")

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("Database verification completed")

def repair_database_issues(cur, issues):
    """Attempt to repair database issues."""
    try:
        # Clean up orphaned relations
        cur.execute("""
            DELETE FROM relations r
            WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
            OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
        """)

        # Clean up duplicate definitions
        cur.execute("""
            WITH DuplicateDefs AS (
                SELECT MIN(id) as keep_id, word_id, definition_text
                FROM definitions
                GROUP BY word_id, definition_text
                HAVING COUNT(*) > 1
            )
            DELETE FROM definitions d
            USING DuplicateDefs dd
            WHERE d.word_id = dd.word_id
            AND d.definition_text = dd.definition_text
            AND d.id != dd.keep_id
        """)

        # Recompute missing search vectors
        cur.execute("""
            UPDATE words
            SET search_text = to_tsvector('english',
                COALESCE(lemma, '') || ' ' ||
                COALESCE(normalized_lemma, '') || ' ' ||
                COALESCE(baybayin_form, '') || ' ' ||
                COALESCE(romanized_form, '')
            )
            WHERE search_text IS NULL
        """)

        # Remove orphaned Baybayin entries
        cur.execute("""
            DELETE FROM words 
            WHERE has_baybayin = TRUE 
              AND baybayin_form IS NULL
        """)

        # Remove inconsistent Baybayin flags
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE, baybayin_form = NULL
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
        """)

        # Commit the transaction
        cur.connection.commit()
        logger.info("Database repairs completed")
        
    except Exception as e:
        cur.connection.rollback()
        logger.error(f"Error during repairs: {str(e)}")

def check_data_integrity(cur) -> List[str]:
    """Check database for integrity issues."""
    issues = []
    
    # Check for orphaned relations
    cur.execute("""
        SELECT COUNT(*) FROM relations r
        WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
           OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
    """)
    if cur.fetchone()[0] > 0:
        issues.append("Found orphaned relations")
        
    # Check for duplicate definitions
    cur.execute("""
        SELECT COUNT(*) 
        FROM (
            SELECT word_id, definition_text, COUNT(*)
            FROM definitions
            GROUP BY word_id, definition_text
            HAVING COUNT(*) > 1
        ) dupes
    """)
    if cur.fetchone()[0] > 0:
        issues.append("Found duplicate definitions")
        
    # Check for missing search vectors
    cur.execute("SELECT COUNT(*) FROM words WHERE search_text IS NULL")
    if cur.fetchone()[0] > 0:
        issues.append("Found words with missing search vectors")
        
    # Check for duplicate Baybayin forms
    cur.execute("""
        SELECT baybayin_form, COUNT(*) 
        FROM words 
        WHERE has_baybayin = TRUE 
        GROUP BY baybayin_form 
        HAVING COUNT(*) > 1
    """)
    duplicates = cur.fetchone()[0]
    if duplicates and duplicates > 0:
        issues.append("Found duplicate Baybayin forms")
        
    return issues

def display_help(args):
    """Display comprehensive help information."""
    console = Console()
    
    # Title
    console.print("\n[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print("[dim]A comprehensive tool for managing Filipino dictionary data[/]\n", 
                 justify="center")
    
    # Basic Usage
    usage_panel = Panel(
        Text.from_markup(
            "python dictionary_manager.py [command] [options]"
        ),
        title="Basic Usage",
        border_style="blue"
    )
    console.print(usage_panel)
    console.print()
    
    # Commands
    commands = [
        {
            "name": "migrate",
            "description": "Create/update schema and load data from sources",
            "options": [
                ("--check-exists", "Skip identical existing entries"),
                ("--force", "Skip confirmation prompt")
            ],
            "example": "python dictionary_manager.py migrate --check-exists",
            "icon": "🔄"
        },
        {
            "name": "lookup",
            "description": "Look up comprehensive information about a word",
            "options": [
                ("word", "The word to look up"),
                ("--format", "Output format (text/json/rich)")
            ],
            "example": "python dictionary_manager.py lookup kamandag",
            "icon": "🔍"
        },
        {
            "name": "stats",
            "description": "Display comprehensive dictionary statistics",
            "options": [
                ("--detailed", "Show detailed statistics"),
                ("--export", "Export statistics to file")
            ],
            "example": "python dictionary_manager.py stats --detailed",
            "icon": "📊"
        },
        {
            "name": "verify",
            "description": "Verify data integrity",
            "options": [
                ("--quick", "Run quick verification"),
                ("--repair", "Attempt to repair issues")
            ],
            "example": "python dictionary_manager.py verify --repair",
            "icon": "✅"
        },
        {
            "name": "purge",
            "description": "Safely delete all data from the database",
            "options": [
                ("--force", "Skip confirmation prompt")
            ],
            "example": "python dictionary_manager.py purge --force",
            "icon": "🗑️"
        }
    ]
    
    # Create command tables
    data_commands = Table(title="Data Management Commands", box=box.ROUNDED, border_style="cyan")
    data_commands.add_column("Command", style="bold yellow")
    data_commands.add_column("Description", style="white")
    data_commands.add_column("Options", style="cyan")
    data_commands.add_column("Example", style="green")
    
    query_commands = Table(title="Query Commands", box=box.ROUNDED, border_style="magenta")
    query_commands.add_column("Command", style="bold yellow")
    query_commands.add_column("Description", style="white")
    query_commands.add_column("Options", style="cyan")
    query_commands.add_column("Example", style="green")
    
    for cmd in commands:
        options_text = "\n".join([f"[cyan]{opt[0]}[/]: {opt[1]}" for opt in cmd["options"]]) or "-"
        row = [
            f"{cmd['icon']} {cmd['name']}",
            cmd["description"],
            options_text,
            f"[dim]{cmd['example']}[/]"
        ]
        
        if cmd["name"] in ["migrate", "update", "purge"]:
            data_commands.add_row(*row)
        else:
            query_commands.add_row(*row)
    
    console.print(data_commands)
    console.print()
    console.print(query_commands)
    console.print()
    
    # Footer
    console.print("\n[dim]For more detailed information, visit the documentation.[/]", justify="center")
    console.print()

def lookup_word(args):
    """Look up a word and display its information."""
    console = Console()
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Get word with processor's normalization
        normalized = normalize_lemma(args.word)
        
        # Get basic word information
        cur.execute("""
            SELECT w.id, w.lemma, w.language_code, w.has_baybayin, w.baybayin_form,
                   w.romanized_form, w.preferred_spelling, w.tags, w.root_word_id,
                   string_agg(DISTINCT d.sources, ', ') as sources,
                   w.pronunciation_data,
                   w.idioms
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            WHERE w.normalized_lemma = %s
            GROUP BY w.id, w.lemma, w.language_code, w.has_baybayin, w.baybayin_form,
                     w.romanized_form, w.preferred_spelling, w.tags, w.root_word_id,
                     w.pronunciation_data, w.idioms
        """, (normalized,))

        result = cur.fetchone()
        if not result:
            # Try processor's fuzzy matching
            suggestions = processor.find_similar_words(cur, args.word)
            if suggestions:
                console.print(f"[bold red]Word '{args.word}' not found.[/]")
                console.print("[yellow]Did you mean:[/]")
                for sugg in suggestions:
                    console.print(f"  • {sugg}")
            else:
                console.print(f"[bold red]Word '{args.word}' not found.[/]")
            return

        # Extract word information
        word_id, lemma, lang_code, has_baybayin, baybayin_form, \
        romanized_form, preferred_spelling, tags, root_word_id, sources, \
        pronunciation_data, idioms = result

        # Format word-level sources (already a string from string_agg)
        sources_text = sources if sources else "-"

        # Display word information
        info = [
            f"[bold]Word:[/] {lemma}",
            f"[bold]Language:[/] {'Tagalog' if lang_code == 'tl' else 'Cebuano'}"
        ]
        baybayin_display = baybayin_form.strip() if baybayin_form and baybayin_form.strip() else "(Not available)"
        romanized_display = romanized_form.strip() if romanized_form and romanized_form.strip() else "(Not available)"
        info.append(f"[bold]Baybayin:[/] {baybayin_display} (Romanized: {romanized_display})")
        if preferred_spelling:
            info.append(f"[bold]Preferred Spelling:[/] {preferred_spelling}")
        if tags:
            info.append(f"[bold]Tags:[/] {tags}")
        if pronunciation_data:
            pron = pronunciation_data.get('romanized', '-') if isinstance(pronunciation_data, dict) else pronunciation_data
            info.append(f"[bold]Pronunciation:[/] {pron}")
        if sources_text and sources_text != "-":
            info.append(f"[bold]Word Sources:[/] {sources_text}")

        console.print(Panel("\n".join(info), title="Word Information", border_style="cyan"))

        # Display idioms if any exist
        if idioms and isinstance(idioms, list) and len(idioms) > 0:
            idiom_table = Table(title="Idioms", box=box.ROUNDED, show_header=True)
            idiom_table.add_column("Idiom", style="cyan")
            idiom_table.add_column("Meaning", style="white")
            idiom_table.add_column("Examples", style="yellow")
            
            for idiom in idioms:
                examples_text = "\n".join(idiom.get('examples', [])) if idiom.get('examples') else "-"
                idiom_table.add_row(
                    idiom.get('idiom', ''),
                    idiom.get('meaning', ''),
                    examples_text
                )
            console.print(idiom_table)

        # Fetch definitions for the word
        cur.execute("""
            SELECT definition_text, original_pos, examples, usage_notes, sources
            FROM definitions
            WHERE word_id = %s
        """, (word_id,))
        definitions = cur.fetchall()

        # Display definitions if any
        if definitions:
            def_table = Table(title="Definitions", box=box.ROUNDED, show_header=True)
            def_table.add_column("No.", justify="right", style="cyan")
            def_table.add_column("Definition", style="white")
            def_table.add_column("Part of Speech", style="yellow")
            def_table.add_column("Usage", style="magenta")
            def_table.add_column("Sources", style="green")
            
            for i, (def_text, pos, examples, usage_notes, src) in enumerate(definitions, 1):
                usage_parts = []
                if examples:
                    usage_parts.append(f"[italic]{examples}[/]")
                if usage_notes:
                    usage_parts.append(usage_notes)
                usage_text = "\n".join(usage_parts) if usage_parts else "-"
                
                # Handle definition sources robustly:
                if src is None:
                    def_sources = "-"
                elif isinstance(src, (list, tuple)):
                    # If all elements are single characters, join them without separator
                    if all(isinstance(x, str) and len(x) == 1 for x in src):
                        def_sources = "".join(src)
                    else:
                        def_sources = ", ".join(src)
                elif isinstance(src, str):
                    def_sources = src
                else:
                    def_sources = str(src)

                def_table.add_row(
                    str(i),
                    def_text,
                    pos if pos else "-",
                    usage_text,
                    def_sources
                )
            console.print(def_table)

        # Display related words
        cur.execute("""
            SELECT r.relation_type, w.lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
        """, (word_id,))
        related = cur.fetchall()
        if related:
            rel_dict = {}
            for rel_type, rel_lemma in related:
                rel_dict.setdefault(rel_type, []).append(rel_lemma)
            
            rel_table = Table(title="Related Words", box=box.ROUNDED, show_header=True)
            rel_table.add_column("Relation Type", style="cyan")
            rel_table.add_column("Words", style="white")
            for rel_type, words in rel_dict.items():
                rel_table.add_row(
                    rel_type.capitalize(),
                    ", ".join(words)
                )
            console.print(rel_table)
        
        # Display Etymologies if any
        cur.execute("""
            SELECT etymology_text, normalized_components, language_codes, sources, created_at
            FROM etymologies
            WHERE word_id = %s
        """, (word_id,))
        etymologies = cur.fetchall()
        if etymologies:
            ety_table = Table(title="Etymologies", box=box.ROUNDED, show_header=True)
            ety_table.add_column("No.", justify="right", style="cyan")
            ety_table.add_column("Etymology", style="white")
            ety_table.add_column("Components", style="yellow")
            ety_table.add_column("Language Codes", style="magenta")
            ety_table.add_column("Sources", style="green")
            ety_table.add_column("Created At", style="dim")
            for j, (ety_text, norm_components, lang_codes, ety_sources, created_at) in enumerate(etymologies, 1):
                ety_table.add_row(
                    str(j),
                    ety_text,
                    norm_components if norm_components else "-",
                    lang_codes if lang_codes else "-",
                    ety_sources if ety_sources else "-",
                    str(created_at)
                )
            console.print(ety_table)
        
        # Display Affixations if any (this word as a root word)
        cur.execute("""
            SELECT a.affix_type, a.affixed_word_id, w.lemma, a.sources, a.created_at
            FROM affixations a
            JOIN words w ON a.affixed_word_id = w.id
            WHERE a.root_word_id = %s
        """, (word_id,))
        affixations = cur.fetchall()
        if affixations:
            affix_table = Table(title="Affixations", box=box.ROUNDED, show_header=True)
            affix_table.add_column("No.", justify="right", style="cyan")
            affix_table.add_column("Affix Type", style="yellow")
            affix_table.add_column("Affixed Word", style="white")
            affix_table.add_column("Sources", style="green")
            affix_table.add_column("Created At", style="dim")
            for k, (affix_type, affixed_word_id, affixed_lemma, affix_sources, affix_created) in enumerate(affixations, 1):
                if affix_sources is None:
                    affix_sources_formatted = "-"
                elif isinstance(affix_sources, (list, tuple)):
                    deduped = list(dict.fromkeys(affix_sources))
                    if all(isinstance(x, str) and len(x) == 1 for x in deduped):
                        affix_sources_formatted = "".join(deduped)
                    else:
                        affix_sources_formatted = ", ".join(deduped)
                elif isinstance(affix_sources, str):
                    sources_list = [s.strip() for s in affix_sources.split(",") if s.strip()]
                    sources_list = list(dict.fromkeys(sources_list))
                    affix_sources_formatted = ", ".join(sources_list) if sources_list else "-"
                else:
                    affix_sources_formatted = str(affix_sources)
                affix_table.add_row(
                    str(k),
                    affix_type,
                    affixed_lemma,
                    affix_sources_formatted,
                    str(affix_created)
                )
            console.print(affix_table)

    except Exception as e:
        logger.error(f"Error during lookup: {str(e)}")
    finally:
        cur.close()
        conn.close()

def display_dictionary_stats(args):
    """Display essential dictionary statistics."""
    console = Console()
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) FROM words")
        total_words = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM definitions")
        total_defs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM relations")
        total_rels = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM etymologies")
        total_etys = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM affixations")
        total_affix = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM definition_relations")
        total_def_rel = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM parts_of_speech")
        total_pos = cur.fetchone()[0]

        stats_table = Table(title="Dictionary Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="bold yellow")
        stats_table.add_column("Count", justify="right", style="cyan")
        stats_table.add_row("Total Words", f"{total_words:,}")
        stats_table.add_row("Total Definitions", f"{total_defs:,}")
        stats_table.add_row("Total Relations", f"{total_rels:,}")
        stats_table.add_row("Total Etymologies", f"{total_etys:,}")
        stats_table.add_row("Total Affixations", f"{total_affix:,}")
        stats_table.add_row("Total Definition Relations", f"{total_def_rel:,}")
        stats_table.add_row("Total Parts of Speech", f"{total_pos:,}")
        console.print(stats_table)

        # Detailed statistics if requested.
        if args.detailed:
            # Implement additional breakdowns here.
            console.print("[dim]Detailed statistics not yet implemented.[/]")
        
        # Export functionality.
        if args.export:
            export_data = {
                "total_words": total_words,
                "total_definitions": total_defs,
                "total_relations": total_rels,
                "total_etymologies": total_etys,
                "total_affixations": total_affix,
                "total_definition_relations": total_def_rel,
                "total_parts_of_speech": total_pos
            }
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            console.print(f"[green]Statistics exported to {args.export}[/]")

    except Exception as e:
        logger.error(f"Error gathering statistics: {str(e)}")
    finally:
        cur.close()
        conn.close()

def display_leaderboard(cur, console):
    """Display top contributors based on contributions."""
    try:
        # Example aggregate query across definitions, etymologies, and relations.
        cur.execute("""
            SELECT source, COUNT(*) as contributions
            FROM (
                SELECT unnest(string_to_array(sources, ', ')) as source FROM definitions
                UNION ALL
                SELECT unnest(string_to_array(sources, ', ')) as source FROM etymologies
                UNION ALL
                SELECT unnest(string_to_array(sources, ', ')) as source FROM relations
            ) as all_sources
            GROUP BY source
            ORDER BY contributions DESC
            LIMIT 10
        """)
        results = cur.fetchall()

        if results:
            table = Table(title="Top Contributors", box=box.ROUNDED)
            table.add_column("Rank", justify="right", style="yellow")
            table.add_column("Source", style="cyan")
            table.add_column("Contributions", justify="right", style="green")
            for i, (source, contributions) in enumerate(results, 1):
                table.add_row(f"#{i}", source, f"{contributions:,}")
            console.print(table)
        else:
            console.print("[yellow]No contributor data available.[/]")

    except Exception as e:
        logger.error(f"Error displaying leaderboard: {str(e)}")

def explore_dictionary():
    """Interactive mode for exploring the dictionary."""
    console = Console()
    console.print("[bold green]Welcome to the Interactive Dictionary Explorer![/]")
    console.print("Type 'exit' to quit.\n")

    while True:
        query = console.input("[bold yellow]Enter a word to look up:[/] ")
        if query.lower() in {"exit", "quit"}:
            console.print("Goodbye!")
            break

        # Reuse lookup_word logic here or call lookup_word with simulated args.
        class Args: pass
        args = Args()
        args.word = query
        args.debug = False
        # Defaulting to rich output for interactive use.
        args.format = 'rich'
        lookup_word(args)

def test_database():
    """Run basic connectivity and simple queries as tests."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        print("Database connectivity test passed.")
        # Optionally add more tests: check table existence, simple inserts, etc.
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        cur.close()
        conn.close()

def update_database(args):
    """Update the database with a new JSON or JSONL file."""
    update_file = args.file
    if not os.path.exists(update_file):
        print(f"File not found: {update_file}")
        sys.exit(1)

    conn = get_connection()
    cur = conn.cursor()
    console = Console()

    src = os.path.basename(update_file)
    dry_run = getattr(args, 'dry_run', False)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        if update_file.endswith('.jsonl'):
            task = progress.add_task(f"Processing {src}...", total=1)
            process_kaikki_jsonl_new(cur, update_file)
            progress.advance(task)
        elif update_file.endswith('.json'):
            with open(update_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                if data and isinstance(data[0], dict) and 'affixation' in data[0]:
                    task = progress.add_task(f"Processing {src} (root words)...", total=len(data))
                    for entry in data:
                        root_lemma = entry.get('lemma', '').strip()
                        if not root_lemma:
                            progress.advance(task)
                            continue
                        derivatives = entry.get('derivatives', {})
                        try:
                            root_id = get_or_create_word_id(cur, root_lemma, language_code="tl", tags=src)
                            for derived_lemma, details in derivatives.items():
                                try:
                                    derived_id = get_or_create_word_id(cur, derived_lemma, language_code="tl", root_word_id=root_id)
                                    if 'definition' in details:
                                        insert_definition(cur, derived_id, details['definition'], details.get('type', ''), sources=src)
                                    insert_relation(cur, derived_id, root_id, "root_of", src)
                                except Exception as e:
                                    logger.error(f"Error processing derivative {derived_lemma}: {str(e)}")
                                    continue
                        except Exception as e:
                            logger.error(f"Error processing root word {root_lemma}: {str(e)}")
                        progress.advance(task)
                elif data and isinstance(data[0], dict) and 'part_of_speech' in data[0]:
                    task = progress.add_task(f"Processing {src} (KWF entries)...", total=len(data))
                    for entry in data:
                        if 'formatted' in entry:
                            process_kwf_entry(cur, entry['formatted'], entry, src=src)
                        progress.advance(task)
                else:
                    task = progress.add_task(f"Processing {src} (Tagalog words)...", total=1)
                    process_tagalog_words(cur, update_file)
                    progress.advance(task)
            elif isinstance(data, dict):
                task = progress.add_task(f"Processing {src} (Tagalog words)...", total=1)
                process_tagalog_words(cur, update_file)
                progress.advance(task)
            else:
                logger.warning(f"Unsupported JSON structure in: {update_file}")
        else:
            logger.warning(f"Unsupported file format for: {update_file}")

    if dry_run:
        conn.rollback()
        console.print("[yellow]Dry run complete. No changes were committed.[/]")
    else:
        conn.commit()
        console.print("[green]Database updated successfully.[/]")

    cur.close()
    conn.close()

def purge_database(args):
    """Purge all data from the database."""
    if not args.force:
        console = Console()
        console.print("\n[bold red]WARNING:[/] This will delete ALL data from the database!")
        confirmation = input("\nType 'PURGE' to continue: ")
        if confirmation != "PURGE":
            logger.info("Purge cancelled")
            return

    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            TRUNCATE TABLE 
                definition_relations, affixations, relations, etymologies, definitions, words, parts_of_speech
            RESTART IDENTITY CASCADE
        """)
        conn.commit()
        logger.info("Database purged successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during purge: {str(e)}")
    finally:
        cur.close()
        conn.close()

def cleanup_relations(cur):
    """Clean up and deduplicate word relations."""
    # Remove duplicate relations
    cur.execute("""
        WITH DupeRels AS (
            SELECT MIN(id) as keep_id,
                   from_word_id,
                   to_word_id,
                   relation_type,
                   array_agg(sources) as all_sources
            FROM relations
            GROUP BY from_word_id, to_word_id, relation_type
            HAVING COUNT(*) > 1
        )
        DELETE FROM relations r
        WHERE EXISTS (
            SELECT 1 FROM DupeRels d
            WHERE r.from_word_id = d.from_word_id
            AND r.to_word_id = d.to_word_id
            AND r.relation_type = d.relation_type
            AND r.id != d.keep_id
        );
    """)
    
    # Standardize relation types
    relation_mapping = {
        'derived from': 'derived_from',
        'root of': 'root_of',
        'synonym of': 'synonym',
        'related to': 'related'
    }
    
    for old, new in relation_mapping.items():
        cur.execute("""
            UPDATE relations
            SET relation_type = %s
            WHERE LOWER(relation_type) = %s
        """, (new, old))

def batch_process_with_progress(items: List[Any], 
                              process_func: Callable, 
                              batch_size: int = 1000) -> None:
    """Process items in batches with progress tracking and error recovery."""
    total_batches = (len(items) + batch_size - 1) // batch_size
    failed_items = []
    
    with tqdm(total=len(items), desc="Processing items") as pbar:
        for batch_num, start_idx in enumerate(range(0, len(items), batch_size)):
            batch = items[start_idx:start_idx + batch_size]
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        for item in batch:
                            try:
                                process_func(cur, item)
                                pbar.update(1)
                            except Exception as e:
                                failed_items.append((item, str(e)))
                                logger.error(f"Error processing item: {str(e)}")
                    conn.commit()
            except Exception as e:
                logger.error(f"Batch {batch_num + 1}/{total_batches} failed: {str(e)}")
                failed_items.extend((item, "Batch failure") for item in batch)
    
    if failed_items:
        logger.error(f"Failed to process {len(failed_items)} items")
        return failed_items

def deduplicate_definitions(cur):
    """Remove duplicate definitions for words while preserving unique information."""
    logger.info("Starting definition deduplication process...")
    
    # Create temporary table to store unique definitions
    cur.execute("""
        CREATE TEMP TABLE unique_definitions AS
        SELECT DISTINCT ON (
            word_id,
            definition_text,
            standardized_pos_id,
            examples,
            usage_notes
        )
            id,
            word_id,
            definition_text,
            original_pos,
            standardized_pos_id,
            examples,
            usage_notes,
            sources,
            created_at,
            updated_at
        FROM definitions
        ORDER BY 
            word_id,
            definition_text,
            standardized_pos_id,
            examples,
            usage_notes,
            created_at DESC;
    """)

    # Delete all existing definitions
    cur.execute("DELETE FROM definitions")
    
    # Reinsert unique definitions
    cur.execute("""
        INSERT INTO definitions
        SELECT * FROM unique_definitions;
    """)
    
    # Drop temporary table
    cur.execute("DROP TABLE unique_definitions")
    
    # Get count of removed duplicates
    cur.execute("SELECT COUNT(*) FROM definitions")
    final_count = cur.fetchone()[0]
    logger.info(f"Definition deduplication complete. {final_count} unique definitions remain.")

def merge_sources(sources):
    """Merge source strings while removing duplicates and formatting properly."""
    if isinstance(sources, str):
        return sources
    unique_sources = set()
    for source in sources:
        if source:
            unique_sources.add(source.strip())
    return ', '.join(sorted(unique_sources))

@with_transaction(commit=True)
def cleanup_dictionary_data(cur):
    """Clean up dictionary data by removing duplicates and standardizing formats."""
    logger.info("Starting dictionary cleanup process...")
    
    # 1. Deduplicate definitions
    deduplicate_definitions(cur)
    
    # 2. Merge sources for identical definitions
    cur.execute("""
        WITH grouped_defs AS (
            SELECT 
                word_id,
                definition_text,
                standardized_pos_id,
                examples,
                usage_notes,
                array_agg(sources) as source_array,
                min(id) as keep_id
            FROM definitions
            GROUP BY 
                word_id,
                definition_text,
                standardized_pos_id,
                examples,
                usage_notes
            HAVING COUNT(*) > 1
        )
        UPDATE definitions d
        SET sources = subquery.merged_sources
        FROM (
            SELECT 
                keep_id,
                array_to_string(source_array, ', ') as merged_sources
            FROM grouped_defs
        ) as subquery
        WHERE d.id = subquery.keep_id;
    """)
    
    # 3. Delete remaining duplicates
    cur.execute("""
        DELETE FROM definitions a
        USING definitions b
        WHERE a.id > b.id
        AND a.word_id = b.word_id
        AND a.definition_text = b.definition_text
        AND COALESCE(a.standardized_pos_id, -1) = COALESCE(b.standardized_pos_id, -1)
        AND COALESCE(a.examples, '') = COALESCE(b.examples, '')
        AND COALESCE(a.usage_notes, '') = COALESCE(b.usage_notes, '');
    """)
    
    logger.info("Dictionary cleanup complete.")

# Add to your command-line interface:
def add_cleanup_command(subparsers):
    """Add cleanup command to CLI parser."""
    cleanup_parser = subparsers.add_parser(
        'cleanup',
        help='Clean up dictionary data by removing duplicates and standardizing formats'
    )
    cleanup_parser.set_defaults(func=handle_cleanup)

def handle_cleanup(args):
    """Handle the cleanup command."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cleanup_dictionary_data(cur)
        conn.commit()
        print("Dictionary cleanup completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during cleanup: {str(e)}")
    finally:
        conn.close()

# Update your main CLI setup to include the new command:
def setup_cli():
    parser = argparse.ArgumentParser(description='Filipino Dictionary Manager')
    subparsers = parser.add_subparsers(dest='command')
    
    # ... existing commands ...
    add_cleanup_command(subparsers)
    
    return parser

def main():
    """Main entry point for the dictionary manager CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize console for rich output
    console = Console()
            
    if args.command == "migrate":
        migrate_data(args)
    elif args.command == "verify":
        verify_database(args)
    elif args.command == "update":
        update_database(args)
    elif args.command == "purge":
        purge_database(args)
    elif args.command == "lookup":
        lookup_word(args)
    elif args.command == "stats":
        display_dictionary_stats(args)
    elif args.command == "leaderboard":
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        display_leaderboard(cur, console)
        cur.close()
        conn.close()
    elif args.command == "help":
        display_help(args)
    elif args.command == "test":
        test_database()
    elif args.command == "explore":
        explore_dictionary()
    elif args.command == "cleanup":
         handle_cleanup(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
