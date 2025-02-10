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
from backend.language_systems import LanguageSystem
from backend.dictionary_processor import DictionaryProcessor
from backend.language_types import *
from backend.source_standardization import SourceStandardization

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
    source_info       JSONB DEFAULT '{}',
    data_hash         TEXT,
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

-- Drop and recreate etymologies table
DROP TABLE IF EXISTS etymologies CASCADE;

-- Create etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id                   SERIAL PRIMARY KEY,
    word_id              INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text       TEXT NOT NULL,
    normalized_components TEXT,
    language_codes       TEXT,
    sources             TEXT NOT NULL,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
        CREATE OR REPLACE FUNCTION update_timestamp()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $trigger$ language 'plpgsql';
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

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_etymologies_timestamp'
        AND tgrelid = 'etymologies'::regclass
    ) THEN
        CREATE TRIGGER update_etymologies_timestamp
            BEFORE UPDATE ON etymologies
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

def standardize_source_names(cur):
    """Standardize source names in all relevant tables."""
    logger.info("Standardizing source names...")
    
    # Update definitions table
    cur.execute("""
        UPDATE definitions
        SET sources = 
            CASE 
                WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE sources
            END
    """)
    
    # Update etymologies table
    cur.execute("""
        UPDATE etymologies
        SET sources = 
            CASE 
                WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE sources
            END
    """)
    
    # Update relations table
    cur.execute("""
        UPDATE relations
        SET sources = 
            CASE 
                WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE sources
            END
    """)
    
    # Update affixations table
    cur.execute("""
        UPDATE affixations
        SET sources = 
            CASE 
                WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE sources
            END
    """)
    
    # Update words table tags
    cur.execute("""
        UPDATE words
        SET tags = 
            CASE 
                WHEN tags = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                WHEN tags = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                WHEN tags = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                WHEN tags = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                WHEN tags = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                ELSE tags
            END
    """)
    
    logger.info("Source names standardization completed")

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
            ('idm', 'Idiom', 'Idyoma', 'Fixed expression with non-literal meaning'),
            ('col', 'Colloquial', 'Kolokyal', 'Informal or conversational usage'),
            ('syn', 'Synonym', 'Singkahulugan', 'Word with similar meaning'),
            ('ant', 'Antonym', 'Di-kasingkahulugan', 'Word with opposite meaning'),
            ('eng', 'English', 'Ingles', 'English loanword or translation'),
            ('spa', 'Spanish', 'Espanyol', 'Spanish loanword or origin'),
            ('tx', 'Texting', 'Texting', 'Text messaging form'),
            ('var', 'Variant', 'Varyant', 'Alternative form or spelling'),
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

        # Standardize source names
        standardize_source_names(cur)

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
            # If not a Baybayin character, return as is
            return chars[0], 1

        # Handle vowels
        if first_char_info.char_type == BaybayinCharType.VOWEL:
            return first_char_info.default_sound, 1

        # Handle consonants
        if first_char_info.char_type == BaybayinCharType.CONSONANT:
            result = first_char_info.default_sound  # Default to consonant + a
            chars_processed = 1

            # Look ahead for modifiers if there are more characters
            if len(chars) > 1:
                next_char_info = self.get_char_info(chars[1])
                if next_char_info:
                    if next_char_info.char_type == BaybayinCharType.VOWEL_MARK:
                        # Consonant + vowel mark (i or u/o)
                        base = result[:-1]  # Remove inherent 'a'
                        result = base + next_char_info.default_sound
                        chars_processed = 2
                    elif next_char_info.char_type == BaybayinCharType.VIRAMA:
                        # Consonant + virama (kills vowel)
                        result = result[:-1]  # Remove inherent 'a'
                        chars_processed = 2
                        # Look ahead for next consonant
                        if len(chars) > 2:
                            next_cons_info = self.get_char_info(chars[2])
                            if next_cons_info and next_cons_info.char_type == BaybayinCharType.CONSONANT:
                                result += next_cons_info.default_sound
                                chars_processed = 3

            return result, chars_processed

        # Handle punctuation
        if first_char_info.char_type == BaybayinCharType.PUNCTUATION:
            return first_char_info.default_sound, 1

        # For any other character type, return as is
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

        try:
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
                if chars_processed == 0:  # Prevent infinite loop
                    result.append(chars[i])
                    i += 1
                else:
                    result.append(romanized)
                    i += chars_processed

            return ''.join(result)
        except Exception as e:
            logger.warning(f"Error romanizing text '{text}': {str(e)}")
            return text  # Return original text if romanization fails

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
    """
    Validate a Baybayin entry with comprehensive checks.
    
    Args:
        baybayin_form: The Baybayin text to validate
        romanized_form: Optional romanized form to check against
        
    Returns:
        bool: True if valid Baybayin entry
    """
    try:
        romanizer = BaybayinRomanizer()
        
        # Split on non-Baybayin characters and filter parts that contain Baybayin characters
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return False
            
        # Try each valid part with validation
        for part in sorted(valid_parts, key=len, reverse=True):
            if not romanizer.validate_text(part):
                continue
                
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
    """Process and store Baybayin data for a word."""
    if not baybayin_form:
        return
        
    try:
        romanizer = BaybayinRomanizer()
        
        # Validate and clean Baybayin form
        if not validate_baybayin_entry(baybayin_form, romanized_form):
            logger.warning(f"Invalid Baybayin form for word_id {word_id}: {baybayin_form}")
            return
            
        # Get the longest valid Baybayin segment
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return
            
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
    """
    Validate a Baybayin entry with comprehensive checks.
    
    Args:
        baybayin_form: The Baybayin text to validate
        romanized_form: Optional romanized form to check against
        
    Returns:
        bool: True if valid Baybayin entry
    """
    try:
        romanizer = BaybayinRomanizer()
        
        # Split on non-Baybayin characters and filter parts that contain Baybayin characters
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return False
            
        # Try each valid part with validation
        for part in sorted(valid_parts, key=len, reverse=True):
            if not romanizer.validate_text(part):
                continue
                
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
    """Process and store Baybayin data for a word."""
    if not baybayin_form:
        return
        
    try:
        romanizer = BaybayinRomanizer()
        
        # Validate and clean Baybayin form
        if not validate_baybayin_entry(baybayin_form, romanized_form):
            logger.warning(f"Invalid Baybayin form for word_id {word_id}: {baybayin_form}")
            return
            
        # Get the longest valid Baybayin segment
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        
        if not valid_parts:
            return
            
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
    """Maps input POS string to standardized POS ID using POS_MAPPINGS."""
    if not pos_string:
        return get_uncategorized_pos_id(cur)

    pos_clean = pos_string.lower().strip(' .')
    
    # Try to find a match in POS_MAPPINGS
    for key, mapping in POS_MAPPINGS.items():
        # Check English term
        if pos_clean == mapping['english'].lower():
            std_code = get_standard_code(key)
            break
            
        # Check Filipino term
        if pos_clean == mapping['filipino'].lower():
            std_code = get_standard_code(key)
            break
            
        # Check abbreviations
        if pos_clean in [abbr.lower().strip('.') for abbr in mapping['abbreviations']]:
            std_code = get_standard_code(key)
            break
            
        # Check variants
        if pos_clean in [var.lower() for var in mapping['variants']]:
            std_code = get_standard_code(key)
            break
    else:
        return get_uncategorized_pos_id(cur)

    # Get ID from parts_of_speech table
    cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (std_code,))
    result = cur.fetchone()
    return result[0] if result else get_uncategorized_pos_id(cur)

def get_standard_code(pos_key: str) -> str:
    """Get standard database code for a POS key."""
    # Map POS keys to standard database codes
    code_mapping = {
        'noun': 'n',
        'verb': 'v',
        'adjective': 'adj',
        'adverb': 'adv',
        'pronoun': 'pron',
        'preposition': 'prep',
        'conjunction': 'conj',
        'interjection': 'intj',
        'article': 'det',
        'affix': 'affix',
        'idiom': 'idm',
        'colloquial': 'col',
        'synonym': 'syn',
        'antonym': 'ant',
        'english': 'eng',
        'spanish': 'spa',
        'texting': 'tx',
        'variant': 'var'
    }
    return code_mapping.get(pos_key, 'unc')

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
        # Update Baybayin information if provided
        if kwargs.get('has_baybayin'):
            cur.execute("""
                UPDATE words 
                SET has_baybayin = %s,
                    baybayin_form = %s,
                    romanized_form = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                kwargs.get('has_baybayin'),
                kwargs.get('baybayin_form'),
                kwargs.get('romanized_form'),
                result[0]
            ))
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

    # Standardize POS
    standardized_pos = standardize_pos(part_of_speech) if part_of_speech else ""
    std_pos_id = get_standardized_pos_id(cur, standardized_pos)
    
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
        standardized_pos,  # Store standardized POS as original_pos
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
    # Don't insert self-referential relations
    if from_word_id == to_word_id:
        return
        
    # Deduplicate sources
    sources = ", ".join(sorted(set(sources.split(", "))))
    
    cur.execute("""
        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (from_word_id, to_word_id, relation_type) DO UPDATE
        SET sources = CASE 
            WHEN relations.sources IS NULL THEN EXCLUDED.sources
            WHEN EXCLUDED.sources IS NULL THEN relations.sources
            ELSE (
                SELECT string_agg(DISTINCT unnest, ', ')
                FROM unnest(string_to_array(relations.sources || ', ' || EXCLUDED.sources, ', '))
            )
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
    
    # Deduplicate sources
    sources = ", ".join(sorted(set(sources.split(", "))))
        
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
                ELSE (
                    SELECT string_agg(DISTINCT unnest, ', ')
                    FROM unnest(string_to_array(etymologies.sources || ', ' || EXCLUDED.sources, ', '))
                )
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
        # Standardize source name
        src = SourceStandardization.get_display_name('kwf_dictionary.json')
        
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

def extract_etymology_components(etymology: str) -> List[str]:
    """
    Extract components from etymology text with proper handling of nested structures.
    
    Args:
        etymology: Etymology text to parse
        
    Returns:
        List of extracted components
    """
    components = []
    
    def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
        """Extract word and its meaning from text with nested parentheses."""
        # Find the word part (before any parentheses)
        word = text.split('(')[0].strip(' "\'')
        
        # Find the meaning part (content within parentheses)
        meaning = None
        paren_count = 0
        meaning_start = -1
        
        for i, char in enumerate(text):
            if char == '(':
                if paren_count == 0:
                    meaning_start = i + 1
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0 and meaning_start != -1:
                    meaning = text[meaning_start:i].strip()
                    break
        
        return word, meaning
    
    try:
        # First clean up the text
        text = etymology.strip()
        if not text:
            return []
            
        # Handle "blend of" pattern
        if "blend of" in text.lower():
            blend_parts = text.split("blend of", 1)[1].split(".")[0].strip()
            # Split on quotes but preserve quoted content
            parts = re.findall(r'"([^"]+)"(?:\s*\([^)]+\))?', blend_parts)
            for part in parts:
                word, meaning = extract_meaning(part)
                if word:
                    comp = word
                    if meaning:
                        comp += f" ({meaning})"
                    components.append(comp)
                    
        # Handle "from" pattern
        elif "from" in text.lower():
            from_parts = text.split("from", 1)[1].split(".")[0].strip()
            # First try to find quoted components
            parts = re.findall(r'"([^"]+)"(?:\s*\([^)]+\))?', from_parts)
            if parts:
                for part in parts:
                    word, meaning = extract_meaning(part)
                    if word:
                        comp = word
                        if meaning:
                            comp += f" ({meaning})"
                        components.append(comp)
            # If no quotes found, try splitting on "+" for compound words
            elif "+" in from_parts:
                for part in from_parts.split("+"):
                    word, meaning = extract_meaning(part)
                    if word:
                        comp = word
                        if meaning:
                            comp += f" ({meaning})"
                        components.append(comp)
                
        # Handle compound words with "+"
        elif "+" in text:
            for part in text.split("+"):
                word, meaning = extract_meaning(part)
                if word:
                    comp = word
                    if meaning:
                        comp += f" ({meaning})"
                    components.append(comp)
            
    except Exception as e:
        logger.debug(f"Error extracting etymology components: {str(e)}")
        
    return [c.strip() for c in components if c.strip()]

def process_kaikki_jsonl_new(cur, filename: str):
    """Process Kaikki dictionary entries with improved Baybayin handling."""
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract Baybayin information from entry using improved parsing."""
        import re
        baybayin = None
        romanized = None
        
        try:
            word = entry.get('word', '')
            if not word:  # Skip if no word is found
                return None, None

            # Skip known non-Tagalog/non-Baybayin words
            if any(word.lower().endswith(suffix) for suffix in ['ismo', 'ista', 'dad', 'cion', 'syon']):
                return None, None
                
            # Skip Spanish/English loanwords and proper nouns
            if any(char in word.lower() for char in 'fjcñvxz'):
                return None, None
                
            # Skip proper nouns (capitalized words)
            if word[0].isupper() and not word.isupper():
                return None, None

            # First check forms array for Baybayin
            forms = entry.get('forms', [])
            if forms and isinstance(forms, list):
                for form in forms:
                    if not isinstance(form, dict):
                        continue
                    if not form.get('tags'):
                        continue
                    if isinstance(form.get('tags'), list) and 'Baybayin' in form['tags']:
                        candidate = form.get('form', '').strip()
                        if candidate and re.match(r'^[ᜀ-ᜟ᜔ᜒᜓ]+$', candidate):
                            baybayin = candidate
                            break

            # If no Baybayin in forms, check head_templates
            if not baybayin:
                head_templates = entry.get('head_templates', [])
                if head_templates and isinstance(head_templates, list):
                    for template in head_templates:
                        if not isinstance(template, dict):
                            continue
                            
                        if not template.get('name', '').startswith(('tl-', 'ceb-')):
                            continue
                            
                        expansion = template.get('expansion', '')
                        if not expansion:
                            continue
                        
                        try:
                            # Look for Baybayin in parentheses after "Baybayin spelling"
                            match = re.search(r'Baybayin spelling[^(]*\((.*?)\)', expansion)
                            if match:
                                candidates = re.findall(r'([ᜀ-ᜟ᜔ᜒᜓ]+)', match.group(1))
                                if candidates:
                                    candidate = candidates[0].strip()
                                    if candidate and re.match(r'^[ᜀ-ᜟ᜔ᜒᜓ]+$', candidate):
                                        baybayin = candidate
                                        break
                            # If no explicit "Baybayin spelling", look for any Baybayin characters
                            else:
                                candidates = re.findall(r'([ᜀ-ᜟ᜔ᜒᜓ]+)', expansion)
                                if candidates:
                                    candidate = candidates[0].strip()
                                    if candidate and re.match(r'^[ᜀ-ᜟ᜔ᜒᜓ]+$', candidate):
                                        baybayin = candidate
                                        break
                        except (IndexError, AttributeError) as e:
                            logger.debug(f"Error extracting Baybayin from template for {word}: {str(e)}")
                            continue

            # Get romanized form if Baybayin exists
            if baybayin:
                # Look for romanized form in forms
                if forms and isinstance(forms, list):
                    for form in forms:
                        if isinstance(form, dict) and form.get('tags'):
                            if isinstance(form.get('tags'), list) and 'canonical' in form.get('tags', []):
                                candidate = form.get('form', '').strip()
                                if candidate:
                                    romanized = candidate
                                    break
                
                # If not found in forms, use the entry word
                if not romanized:
                    romanized = word

            # Final validation
            if baybayin:
                # Validate Baybayin form
                if not re.match(r'^[ᜀ-ᜟ᜔ᜒᜓ]+$', baybayin):
                    return None, None
                    
                # Check for minimum length and maximum length
                if len(baybayin) < 1 or len(baybayin) > 30:  # Add reasonable max length
                    return None, None
                    
                # Check for valid character combinations
                if '᜔᜔' in baybayin:  # Double virama not allowed
                    return None, None
                    
                # Check for valid ending
                if baybayin[-1] not in 'ᜀᜁᜂ' and baybayin[-1] != '᜔':
                    if baybayin[-1] in 'ᜒᜓ':  # Vowel marks must be preceded by consonants
                        if len(baybayin) < 2 or baybayin[-2] not in BaybayinRomanizer.CONSONANTS:
                            return None, None

                # Validate character sequence
                prev_char = None
                for char in baybayin:
                    if char in 'ᜒᜓ' and prev_char not in BaybayinRomanizer.CONSONANTS:
                        return None, None
                    prev_char = char

            return baybayin, romanized

        except Exception as e:
            logger.debug(f"Error in Baybayin extraction for {word if 'word' in locals() else 'unknown word'}: {str(e)}")
            return None, None

    def standardize_entry_pos(pos_str: str) -> str:
        """Standardize POS to Filipino terms using POS_MAPPINGS from language_types."""
        if not pos_str:
            return ""
            
        # Convert to lowercase for case-insensitive matching
        pos_lower = pos_str.lower().strip(' .')
        
        # Direct mapping for common English POS terms
        direct_mapping = {
            'noun': 'Pangngalan',
            'adj': 'Pang-uri',
            'adjective': 'Pang-uri',
            'verb': 'Pandiwa',
            'adverb': 'Pang-abay',
            'pronoun': 'Panghalip',
            'n': 'Pangngalan',
            'v': 'Pandiwa'
        }
        
        if pos_lower in direct_mapping:
            return direct_mapping[pos_lower]
            
        # Check each mapping in POS_MAPPINGS
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
        return pos_str

    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry."""
        try:
            lemma = entry.get("word", "").strip()
            if not lemma:
                return

            # Handle Baybayin entries
            try:
                baybayin_form, romanized = extract_baybayin_info(entry)
                has_baybayin = bool(baybayin_form)
            except Exception as e:
                logger.debug(f"Error extracting Baybayin info for {lemma}: {str(e)}")
                baybayin_form, romanized = None, None
                has_baybayin = False
            
            # Determine language code based on filename
            language_code = "tl" if "kaikki.jsonl" in filename else "ceb"
            
            # Get standardized source name
            source = SourceStandardization.get_display_name(os.path.basename(filename))
            
            # Process pronunciation data
            pronunciation_data = {}
            if 'sounds' in entry:
                pronunciation_data['sounds'] = entry['sounds']
            if 'hyphenation' in entry:
                pronunciation_data['hyphenation'] = entry['hyphenation']
            
            # Get entry-level POS from head_templates or pos field
            entry_pos = None
            if 'head_templates' in entry:
                for template in entry.get('head_templates', []):
                    if isinstance(template, dict):
                        template_name = template.get('name', '').lower()
                        if template_name.startswith(('tl-', 'ceb-')):
                            pos_part = template_name.split('-')[1]
                            entry_pos = standardize_entry_pos(pos_part)
                            # Store template expansion if available
                            if 'expansion' in template:
                                pronunciation_data['template_expansion'] = template['expansion']
                            # Store template args if available
                            if 'args' in template:
                                pronunciation_data['template_args'] = template['args']
                            break
            
            if not entry_pos and 'pos' in entry:
                entry_pos = standardize_entry_pos(entry.get('pos', ''))

            # Collect all tags
            tags = set()
            
            # Add entry-level tags
            if 'tags' in entry:
                entry_tags = entry.get('tags', [])
                if isinstance(entry_tags, list):
                    tags.update(tag.strip() for tag in entry_tags if tag and isinstance(tag, str) and tag.strip())
            
            # Add tags from forms
            if 'forms' in entry:
                for form in entry.get('forms', []):
                    if isinstance(form, dict):
                        # Add form tags
                        form_tags = form.get('tags', [])
                        if isinstance(form_tags, list):
                            tags.update(tag.strip() for tag in form_tags if tag and isinstance(tag, str) and tag.strip())
                        # Add the form itself as a variant
                        if 'form' in form and form['form'] != lemma:
                            tags.add(f"Variant: {form['form']}")

            # Add etymology information as tags
            if 'etymology_text' in entry:
                etymology = entry.get('etymology_text', '')
                if etymology:
                    tags.add(f"Etymology: {etymology}")
                    # Extract source languages
                    if 'Borrowed from' in etymology:
                        match = re.search(r'Borrowed from (\w+)', etymology)
                        if match:
                            tags.add(f"Source Language: {match.group(1)}")
                    elif 'from' in etymology.lower():
                        match = re.search(r'from (\w+)', etymology.lower())
                        if match:
                            tags.add(f"Source Language: {match.group(1)}")

            # Add etymology templates if available
            if 'etymology_templates' in entry:
                for template in entry.get('etymology_templates', []):
                    if isinstance(template, dict):
                        if 'expansion' in template:
                            tags.add(f"Etymology Detail: {template['expansion']}")

            # Add hyphenation if available
            if 'hyphenation' in entry:
                hyphenation = entry.get('hyphenation', [])
                if isinstance(hyphenation, list) and hyphenation:
                    tags.add(f"Hyphenation: {'-'.join(hyphenation)}")

            # Process homophones and sounds
            if 'sounds' in entry:
                for sound in entry.get('sounds', []):
                    if isinstance(sound, dict):
                        if 'homophone' in sound:
                            tags.add(f"Homophone: {sound['homophone']}")
                        if 'rhymes' in sound:
                            tags.add(f"Rhymes: {sound['rhymes']}")
            
            # Process senses for tags and categories
            for sense in entry.get('senses', []):
                if isinstance(sense, dict):
                    # Add sense-level tags
                    sense_tags = sense.get('tags', [])
                    if isinstance(sense_tags, list):
                        tags.update(tag.strip() for tag in sense_tags if tag and isinstance(tag, str) and tag.strip())
                    
                    # Add topics if present
                    if 'topics' in sense:
                        topics = sense.get('topics', [])
                        if isinstance(topics, list):
                            tags.update(f"Topic: {topic}" for topic in topics if topic and isinstance(topic, str))
                    
                    # Add links as related terms
                    if 'links' in sense:
                        links = sense.get('links', [])
                        if isinstance(links, list):
                            for link in links:
                                if isinstance(link, list) and len(link) >= 2:
                                    tags.add(f"Related: {link[0]}")
                    
                    # Add synonyms
                    if 'synonyms' in sense:
                        synonyms = sense.get('synonyms', [])
                        if isinstance(synonyms, list):
                            for syn in synonyms:
                                if isinstance(syn, dict):
                                    syn_word = syn.get('word', '')
                                    syn_tags = syn.get('tags', [])
                                    if syn_word:
                                        tag_prefix = "Synonym"
                                        if syn_tags and isinstance(syn_tags, list):
                                            for tag in syn_tags:
                                                if tag.lower() == 'obsolete':
                                                    tag_prefix = "Obsolete Synonym"
                                                    break
                                        tags.add(f"{tag_prefix}: {syn_word}")
                                elif isinstance(syn, str):
                                    tags.add(f"Synonym: {syn}")
                    
                    # Add relevant categories (exclude maintenance categories)
                    for cat in sense.get('categories', []):
                        if isinstance(cat, dict):
                            cat_name = cat.get("name", "")
                            cat_kind = cat.get("kind", "")
                            cat_parents = cat.get("parents", [])
                            cat_source = cat.get("source", "")
                            
                            # Skip maintenance categories but include topical and other meaningful categories
                            if cat_name and not any(x in cat_name.lower() for x in ["entries with", "terms with"]):
                                if cat_kind:
                                    tags.add(f"{cat_kind}: {cat_name}")
                                else:
                                    tags.add(cat_name)
                                # Add parent categories
                                for parent in cat_parents:
                                    if parent and not any(x in parent.lower() for x in ["entries with", "terms with"]):
                                        tags.add(f"Category: {parent}")
                                # Add source if available
                                if cat_source:
                                    tags.add(f"Category Source: {cat_source}")
            
            # Add pronunciation information as tags
            if 'sounds' in entry:
                for sound in entry.get('sounds', []):
                    if isinstance(sound, dict):
                        # Add IPA pronunciations
                        if 'ipa' in sound:
                            ipa = sound['ipa']
                            if 'tags' in sound and isinstance(sound['tags'], list):
                                dialect = next((tag for tag in sound['tags'] if tag != 'IPA'), None)
                                if dialect:
                                    tags.add(f"IPA ({dialect}): {ipa}")
                                else:
                                    tags.add(f"IPA: {ipa}")
                            else:
                                tags.add(f"IPA: {ipa}")
                        
                        # Add pronunciation tags
                        if 'tags' in sound and isinstance(sound['tags'], list):
                            tags.update(f"Pronunciation: {tag}" for tag in sound['tags'] if tag and isinstance(tag, str))
                        
                        # Add pronunciation notes
                        if 'note' in sound:
                            tags.add(f"Pronunciation Note: {sound['note']}")
            
            # Convert tags to string and remove any empty tags
            tags_str = '; '.join(sorted(tag for tag in tags if tag))
            
            # Create or update word entry
            word_id = get_or_create_word_id(
                cur,
                lemma,
                language_code=language_code,
                has_baybayin=has_baybayin,
                baybayin_form=baybayin_form,
                romanized_form=romanized if has_baybayin else None,
                tags=tags_str,
                pronunciation_data=json.dumps(pronunciation_data) if pronunciation_data else None
            )

            # Process senses
            senses = entry.get("senses", [])
            if isinstance(senses, list):
                for sense in senses:
                    if not isinstance(sense, dict):
                        continue
                        
                    glosses = sense.get("glosses", [])
                    raw_glosses = sense.get("raw_glosses", [])
                    if not isinstance(glosses, list):
                        continue
                    
                    if all('Baybayin spelling of' in str(gloss) for gloss in glosses):
                        continue

                    # Get sense-specific POS, fallback to entry POS
                    sense_pos = sense.get("pos", entry_pos)
                    if sense_pos:
                        sense_pos = standardize_entry_pos(sense_pos)
                    elif entry_pos:
                        sense_pos = entry_pos

                    # Collect examples with context
                    examples = []
                    for ex in sense.get("examples", []):
                        if isinstance(ex, dict):
                            example_text = ex.get("text", "")
                            english = ex.get("english", "")
                            ref = ex.get("ref", "")
                            if example_text:
                                if english:
                                    example_text += f" ({english})"
                                if ref:
                                    example_text += f" [Source: {ref}]"
                                examples.append(example_text)
                        elif isinstance(ex, str):
                            examples.append(ex)

                    # Collect usage notes and categories
                    usage_notes = []
                    categories = []
                    
                    # Add sense tags as usage notes
                    if sense.get("tags"):
                        usage_notes.extend(sense["tags"])
                    
                    # Add raw glosses as usage notes if they provide additional context
                    if raw_glosses:
                        for raw_gloss in raw_glosses:
                            if raw_gloss and raw_gloss not in glosses:
                                usage_notes.append(f"Context: {raw_gloss}")
                    
                    # Add categories
                    if sense.get("categories"):
                        for cat in sense["categories"]:
                            if isinstance(cat, dict) and "name" in cat:
                                if not any(x in cat["name"].lower() for x in ["entries with", "terms with"]):
                                    categories.append(cat["name"])

                    # Process each gloss
                    for gloss in glosses:
                        if gloss and not "Baybayin spelling of" in str(gloss):
                            insert_definition(
                                cur, 
                                word_id, 
                                str(gloss).strip(),
                                sense_pos or entry_pos,  # Use sense_pos if available, fallback to entry_pos
                                examples="\n".join(examples) if examples else None,
                                usage_notes="; ".join(usage_notes) if usage_notes else None,
                                category="; ".join(categories) if categories else None,
                                sources=source
                            )

            # Process etymology
            etymology = entry.get("etymology_text", "")
            if etymology:
                try:
                    # Extract components using improved function
                    components = extract_etymology_components(etymology)
                    
                    # Use language_systems.py for extraction
                    codes, cleaned_ety = lsys.extract_and_remove_language_codes(etymology)
                    
                    # Add additional language detection
                    additional_langs = {
                        'Malay': 'ms',
                        'Sanskrit': 'sa',
                        'Proto-Malayo-Polynesian': 'pmp',
                        'Proto-Austronesian': 'pan',
                        'Spanish': 'es',
                        'Latin': 'la',
                        'Arabic': 'ar',
                        'Chinese': 'zh',
                        'Hokkien': 'nan',
                        'Javanese': 'jv',
                        'Kapampangan': 'pam',
                        'Ilocano': 'ilo',
                        'Bikol': 'bcl',
                        'Cebuano': 'ceb',
                        'Hiligaynon': 'hil',
                        'Waray': 'war'
                    }
                    
                    for lang, code in additional_langs.items():
                        if lang in etymology and code not in codes:
                            codes.append(code)
                    
                    insert_etymology(
                        cur=cur,
                        word_id=word_id,
                        etymology_text=cleaned_ety,
                        normalized_components=", ".join(components) if components else None,
                        language_codes=", ".join(codes) if codes else "",
                        sources=source
                    )
                except Exception as e:
                    logger.debug(f"Error processing etymology for {lemma}: {str(e)}")
                    # Still insert the etymology even if component extraction fails
                    insert_etymology(
                        cur=cur,
                        word_id=word_id,
                        etymology_text=etymology,
                        normalized_components=None,
                        language_codes="",
                        sources=source
                    )

            # Process relations with deduplication and hierarchy
            processed_relations = set()  # Track processed relations
            
            # Process synonyms first as they have highest priority
            synonyms = entry.get("synonyms", [])
            if isinstance(synonyms, list):
                for syn in synonyms:
                    try:
                        if isinstance(syn, dict):
                            syn_word = syn.get("word", "").strip()
                            syn_tags = syn.get("tags", [])
                        else:
                            syn_word = str(syn).strip()
                            syn_tags = []
                            
                        if not syn_word:
                            continue
                            
                        syn_id = get_or_create_word_id(cur, syn_word, language_code)
                        if syn_id != word_id:  # Prevent self-referential relations
                            relation_key = (word_id, syn_id)
                            if relation_key not in processed_relations:
                                # Check if it's an obsolete synonym
                                relation_type = "singkahulugan"  # default: synonym
                                if syn_tags and "obsolete" in [t.lower() for t in syn_tags]:
                                    relation_type = "lumang_singkahulugan"  # obsolete synonym
                                
                                insert_relation(
                                    cur,
                                    word_id,
                                    syn_id,
                                    relation_type,
                                    source
                                )
                                processed_relations.add(relation_key)
                    except Exception as e:
                        logger.debug(f"Error processing synonym for {lemma}: {str(e)}")
                        continue

            # Process derived words
            derived = entry.get("derived", [])
            if isinstance(derived, list):
                for der in derived:
                    try:
                        if isinstance(der, dict):
                            der_word = der.get("word", "").strip()
                            der_tags = der.get("tags", [])
                        else:
                            der_word = str(der).strip()
                            der_tags = []
                            
                        if not der_word:
                            continue
                            
                        der_id = get_or_create_word_id(cur, der_word, language_code)
                        if der_id != word_id:  # Prevent self-referential relations
                            relation_key = (word_id, der_id)
                            if relation_key not in processed_relations:
                                insert_relation(
                                    cur,
                                    word_id,
                                    der_id,
                                    "hinango",  # Use Filipino term for derived
                                    source
                                )
                                processed_relations.add(relation_key)
                    except Exception as e:
                        logger.debug(f"Error processing derived word for {lemma}: {str(e)}")
                        continue

            # Process related words last
            related = entry.get("related", [])
            if isinstance(related, list):
                for rel in related:
                    try:
                        if isinstance(rel, dict):
                            rel_word = rel.get("word", "").strip()
                            rel_tags = rel.get("tags", [])
                        else:
                            rel_word = str(rel).strip()
                            rel_tags = []
                            
                        if not rel_word:
                            continue
                            
                        rel_id = get_or_create_word_id(cur, rel_word, language_code)
                        if rel_id != word_id:  # Prevent self-referential relations
                            relation_key = (word_id, rel_id)
                            if relation_key not in processed_relations:
                                insert_relation(
                                    cur,
                                    word_id,
                                    rel_id,
                                    "kaugnay",  # Use Filipino term for related
                                    source
                                )
                                processed_relations.add(relation_key)
                    except Exception as e:
                        logger.debug(f"Error processing related word for {lemma}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error processing entry '{lemma if 'lemma' in locals() else 'unknown'}': {str(e)}")
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
    
    src = SourceStandardization.get_display_name(os.path.basename(filename))
    
    def standardize_entry_pos(pos_str: str) -> List[str]:
        """Standardize POS to Filipino terms. Returns list of standardized terms."""
        if not pos_str:
            return ['Hindi Tiyak']
            
        # Split on spaces to handle multiple POS values
        pos_parts = pos_str.lower().strip().split()
        standardized = []
        
        # Direct mapping for abbreviations - check this first
        direct_map = {
            'png': 'Pangngalan',
            'pnr': 'Pang-uri',
            'pnw': 'Pandiwa',
            'pny': 'Pang-abay',
            'pnd': 'Pangngalan',
            'adj': 'Pang-uri',
            'n': 'Pangngalan',
            'v': 'Pandiwa'
        }
        
        for pos_part in pos_parts:
            pos_clean = pos_part.strip(' .')
            if not pos_clean:
                continue
                
            # First check direct abbreviation mappings
            if pos_clean in direct_map:
                standardized.append(direct_map[pos_clean])
                continue
                
            # Then check POS_MAPPINGS
            for mapping in POS_MAPPINGS.values():
                # Check abbreviations
                if pos_clean in [abbr.lower().strip('.') for abbr in mapping['abbreviations']]:
                    standardized.append(mapping['filipino'])
                    break
                # Check variants
                elif pos_clean in [var.lower() for var in mapping['variants']]:
                    standardized.append(mapping['filipino'])
                    break
                # Check full terms
                elif pos_clean == mapping['english'].lower() or pos_clean == mapping['filipino'].lower():
                    standardized.append(mapping['filipino'])
                    break
                
        # Only use Hindi Tiyak if we found no valid mappings
        if not standardized:
            standardized = ['Hindi Tiyak']
            
        return list(dict.fromkeys(standardized))  # Remove duplicates while preserving order
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            
        # Handle dictionary format
        if isinstance(entries, dict):
            for lemma, entry_data in tqdm(entries.items(), desc="Processing Tagalog words"):
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        lemma,
                        language_code="tl",
                        tags=src
                    )
                    
                    # Get all standardized POS values
                    pos_values = standardize_entry_pos(entry_data.get('part_of_speech', ''))
                    
                    # Process definitions
                    if 'definitions' in entry_data:
                        for definition in entry_data['definitions']:
                            def_text = definition if isinstance(definition, str) else definition['text']
                            
                            # Insert definition once for each POS value
                            for pos in pos_values:
                                insert_definition(
                                    cur,
                                    word_id,
                                    def_text,
                                    pos,
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
        
    src = SourceStandardization.get_display_name(os.path.basename(filename))
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for root_lemma, derivatives in tqdm(data.items(), desc="Processing root words"):
            try:
                # Create root word entry
                root_id = get_or_create_word_id(
                    cur,
                    root_lemma,
                    language_code="tl",
                    tags=src
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
            logger.info("Starting post-processing steps...")
            
            # Process Baybayin entries
            process_baybayin_entries(cur)
            cleanup_baybayin_data(cur)
            
            # Clean up and standardize data
            logger.info("Running data cleanup and standardization...")
            cleanup_dictionary_data(cur)
            
            conn.commit()
            logger.info("Migration and cleanup completed successfully")
            
        except Exception as e:
            conn.rollback()
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
    """Look up a word in the dictionary and display its information."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Get word ID and basic info with enhanced data
        cur.execute("""
            SELECT w.id, w.lemma, w.language_code, w.tags, w.has_baybayin, w.baybayin_form, 
                   w.romanized_form, w.pronunciation_data, w.root_word_id
            FROM words w
            WHERE w.normalized_lemma = %s
        """, (normalize_lemma(args.word),))
        
        result = cur.fetchone()
        if not result:
            print(f"Word '{args.word}' not found in the dictionary.")
            return
            
        word_id, lemma, language_code, tags, has_baybayin, baybayin_form, romanized_form, pronunciation, root_word_id = result
        
        # Create rich console for output
        console = Console()
        
        # Display word information in a structured panel
        word_info = []
        word_info.append(Text(f"Word: {lemma}", style="bold cyan"))
        word_info.append(Text(f"Language: {'Tagalog' if language_code == 'tl' else 'Cebuano'}", style="blue"))
        
        # Display Baybayin information if available
        if has_baybayin and baybayin_form:
            romanized = f" (Romanized: {romanized_form})" if romanized_form else ""
            word_info.append(Text(f"Baybayin: {baybayin_form}{romanized}", style="magenta"))
        
        # Display pronunciation information
        if pronunciation and isinstance(pronunciation, dict):
            pron_info = []
            if 'sounds' in pronunciation and pronunciation['sounds']:
                for sound in pronunciation['sounds']:
                    if 'ipa' in sound:
                        dialect = f" ({sound['dialect']})" if 'dialect' in sound else ""
                        pron_info.append(f"{sound['ipa']}{dialect}")
                if pron_info:
                    word_info.append(Text(f"Pronunciation (IPA): {', '.join(pron_info)}", style="yellow"))
            
            if 'hyphenation' in pronunciation:
                word_info.append(Text(f"Syllables: {pronunciation['hyphenation']}", style="yellow"))
                
            if 'notes' in pronunciation and pronunciation['notes']:
                word_info.append(Text(f"Pronunciation Notes: {pronunciation['notes']}", style="yellow"))
        
        # Display tags with better organization
        if tags:
            tag_groups = {}
            for tag in tags.split(';'):
                tag = tag.strip()
                if ':' in tag:
                    category, value = tag.split(':', 1)
                    if category not in tag_groups:
                        tag_groups[category] = []
                    tag_groups[category].append(value.strip())
                else:
                    if 'Other' not in tag_groups:
                        tag_groups['Other'] = []
                    tag_groups['Other'].append(tag)
            
            for category, values in sorted(tag_groups.items()):
                word_info.append(Text(f"{category}: {', '.join(sorted(set(values)))}", style="green"))
        
        # Create word information panel
        console.print(Panel(
            "\n".join([str(line) for line in word_info]),
            title="[bold]Word Information[/]",
            expand=True,
            border_style="cyan"
        ))
        
        # Get definitions grouped by POS with enhanced information
        cur.execute("""
            WITH pos_defs AS (
                SELECT 
                    p.name_tl as pos,
                    d.definition_text,
                    d.examples,
                    d.usage_notes,
                    d.sources,
                    d.created_at,
                    ROW_NUMBER() OVER (PARTITION BY p.name_tl ORDER BY d.created_at) as def_num
                FROM definitions d
                JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                WHERE d.word_id = %s
            ),
            grouped_defs AS (
                SELECT 
                    pos,
                    definition_text,
                    examples,
                    usage_notes,
                    string_agg(DISTINCT sources, ', ') as sources,
                    min(def_num) as def_num
                FROM pos_defs
                GROUP BY pos, definition_text, examples, usage_notes
            )
            SELECT *
            FROM grouped_defs
            ORDER BY pos, def_num
        """, (word_id,))
        
        definitions = cur.fetchall()
        
        if definitions:
            console.print("\n[bold]Definitions[/]", justify="center")
            
            table = Table(
                "POS",
                "Definition",
                "Examples",
                "Usage Notes",
                "Sources",
                box=box.ROUNDED,
                expand=True,
                show_lines=True,
                padding=(0, 1)
            )
            
            current_pos = None
            for pos, definition, examples, usage_notes, sources, def_num in definitions:
                # Skip empty definitions
                if not definition or not definition.strip():
                    continue
                
                # Process examples
                example_text = ""
                if examples:
                    example_list = []
                    for ex in examples.split('\n'):
                        ex = ex.strip()
                        if ex and ex not in example_list:
                            example_list.append(f"• {ex}")
                    example_text = "\n".join(example_list)
                
                # Process usage notes
                usage_text = ""
                if usage_notes:
                    usage_list = []
                    for note in usage_notes.split('\n'):
                        note = note.strip()
                        if note and note not in usage_list:
                            usage_list.append(f"• {note}")
                    usage_text = "\n".join(usage_list)
                
                # Process sources - deduplicate and standardize
                source_list = []
                for source in sources.split(','):
                    display_name = SourceStandardization.get_display_name(source.strip())
                    if display_name and display_name not in source_list:
                        source_list.append(display_name)
                
                table.add_row(
                    Text(pos, style="blue") if pos != current_pos else "",
                    Text(definition, style="bold") if def_num == 1 else definition,
                    Text(example_text, style="green") if example_text else "",
                    Text(usage_text, style="yellow") if usage_text else "",
                    Text(", ".join(sorted(source_list)), style="magenta", overflow="fold")
                )
                current_pos = pos
            
            console.print(table)
        
        # Get etymology information with enhanced display
        cur.execute("""
            SELECT DISTINCT ON (etymology_text)
                etymology_text,
                string_agg(DISTINCT normalized_components, '; ') as components,
                string_agg(DISTINCT language_codes, ', ') as langs,
                string_agg(DISTINCT sources, ', ') as sources
            FROM etymologies
            WHERE word_id = %s
            GROUP BY etymology_text
        """, (word_id,))
        
        etymologies = cur.fetchall()
        
        if etymologies:
            console.print("\n[bold]Etymology[/]", justify="center")
            
            for etym_text, components, langs, sources in etymologies:
                if not etym_text.strip():
                    continue
                    
                etymology_panel = []
                etymology_panel.append(Text(etym_text, style="cyan"))
                
                if components:
                    comp_list = []
                    for comp in components.split(';'):
                        comp = comp.strip()
                        if comp and comp not in comp_list:
                            comp_list.append(comp)
                    if comp_list:
                        etymology_panel.append(Text("\nComponents:", style="bold"))
                        for comp in sorted(comp_list):
                            etymology_panel.append(Text(f"• {comp}", style="green"))
                
                if langs:
                    lang_list = sorted(set(lang.strip() for lang in langs.split(',') if lang.strip()))
                    if lang_list:
                        etymology_panel.append(Text("\nLanguages:", style="bold"))
                        etymology_panel.append(Text(", ".join(lang_list), style="blue"))
                
                if sources:
                    source_list = []
                    for source in sources.split(','):
                        display_name = SourceStandardization.get_display_name(source.strip())
                        if display_name and display_name not in source_list:
                            source_list.append(display_name)
                    if source_list:
                        etymology_panel.append(Text("\nSources:", style="bold"))
                        etymology_panel.append(Text(", ".join(sorted(source_list)), style="magenta"))
                
                console.print(Panel(
                    "\n".join([str(line) for line in etymology_panel]),
                    box=box.ROUNDED,
                    expand=True,
                    border_style="blue"
                ))
        
        # Get related words with enhanced grouping
        cur.execute("""
            SELECT DISTINCT ON (r.relation_type, w2.lemma)
                r.relation_type,
                w2.lemma as related_word,
                string_agg(DISTINCT p.name_tl, ', ') as pos_list,
                string_agg(DISTINCT r.sources, ', ') as sources
            FROM relations r
            JOIN words w2 ON r.to_word_id = w2.id
            LEFT JOIN definitions d ON w2.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE r.from_word_id = %s
            GROUP BY r.relation_type, w2.lemma
            ORDER BY r.relation_type, w2.lemma
        """, (word_id,))
        
        relations = cur.fetchall()
        
        if relations:
            console.print("\n[bold]Related Words[/]", justify="center")
            
            # Group and deduplicate related words
            relation_groups = {}
            for rel_type, rel_word, pos_list, sources in relations:
                # Skip duplicate relation types (e.g., 'hinango' if we have 'derived_from')
                if rel_type in ['hinango', 'singkahulugan'] and any(k in relation_groups for k in ['derived_from', 'synonym']):
                    continue
                    
                if rel_type not in relation_groups:
                    relation_groups[rel_type] = []
                pos_info = f" ({pos_list})" if pos_list else ""
                word_info = f"{rel_word}{pos_info}"
                if word_info not in relation_groups[rel_type]:
                    relation_groups[rel_type].append(word_info)
            
            table = Table(
                "Relation Type",
                "Related Words",
                box=box.ROUNDED,
                expand=True,
                show_lines=True,
                padding=(0, 1)
            )
            
            for rel_type, words in sorted(relation_groups.items()):
                table.add_row(
                    Text(rel_type, style="bold yellow"),
                    Text(", ".join(sorted(words)), style="cyan", overflow="fold")
                )
            
            console.print(table)
        
        # Get affixed forms with enhanced display
        cur.execute("""
            SELECT DISTINCT ON (a.affix_type, w2.lemma)
                a.affix_type,
                w2.lemma as affixed_word,
                string_agg(DISTINCT p.name_tl, ', ') as pos_list,
                string_agg(DISTINCT a.sources, ', ') as sources
            FROM affixations a
            JOIN words w2 ON a.affixed_word_id = w2.id
            LEFT JOIN definitions d ON w2.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE a.root_word_id = %s
            GROUP BY a.affix_type, w2.lemma
            ORDER BY a.affix_type, w2.lemma
        """, (word_id,))
        
        affixed = cur.fetchall()
        
        if affixed:
            console.print("\n[bold]Affixed Forms[/]", justify="center")
            
            table = Table(
                "Affix Type",
                "Word",
                "Part of Speech",
                "Sources",
                box=box.ROUNDED,
                expand=True,
                show_lines=True,
                padding=(0, 1)
            )
            
            # Deduplicate and clean up affixed forms
            seen_forms = set()
            for affix_type, word, pos_list, sources in affixed:
                if not word.strip() or word in seen_forms:
                    continue
                seen_forms.add(word)
                
                # Process sources - deduplicate and standardize
                source_list = []
                for source in sources.split(','):
                    display_name = SourceStandardization.get_display_name(source.strip())
                    if display_name and display_name not in source_list:
                        source_list.append(display_name)
                
                table.add_row(
                    Text(affix_type or "", style="yellow"),
                    Text(word, style="cyan"),
                    Text(pos_list or "", style="blue"),
                    Text(", ".join(sorted(source_list)), style="magenta", overflow="fold")
                )
            
            console.print(table)

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

def cleanup_dictionary_data(cur):
    """Clean up dictionary data by removing duplicates and standardizing formats."""
    logger.info("Starting dictionary cleanup process...")
    
    # 1. First standardize all POS values in definitions
    cur.execute("""
        WITH pos_standardization AS (
            SELECT d.id,
                   CASE 
                       WHEN d.original_pos IN ('png', 'n', 'noun', 'pangngalan') THEN 
                           (SELECT id FROM parts_of_speech WHERE code = 'n')
                       WHEN d.original_pos IN ('pnr', 'adj', 'adjective', 'pang-uri') THEN 
                           (SELECT id FROM parts_of_speech WHERE code = 'adj')
                       WHEN d.original_pos IN ('pnw', 'v', 'verb', 'pandiwa') THEN 
                           (SELECT id FROM parts_of_speech WHERE code = 'v')
                       WHEN d.original_pos IN ('pny', 'adv', 'adverb', 'pang-abay') THEN 
                           (SELECT id FROM parts_of_speech WHERE code = 'adv')
                       ELSE standardized_pos_id
                   END as new_pos_id
            FROM definitions d
        )
        UPDATE definitions d
        SET standardized_pos_id = ps.new_pos_id
        FROM pos_standardization ps
        WHERE d.id = ps.id;
    """)

    # 2. Merge and standardize sources
    cur.execute("""
        WITH source_standardization AS (
            SELECT id,
                   string_agg(DISTINCT 
                       CASE 
                           WHEN unnest = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                           WHEN unnest = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                           WHEN unnest = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                           WHEN unnest = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                           WHEN unnest = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                           ELSE unnest
                       END, ', ') as standardized_sources
            FROM (
                SELECT id, unnest(string_to_array(sources, ', ')) as unnest
                FROM definitions
            ) s
            GROUP BY id
        )
        UPDATE definitions d
        SET sources = ss.standardized_sources
        FROM source_standardization ss
        WHERE d.id = ss.id;
    """)

    # 3. Deduplicate definitions while preserving unique information
    cur.execute("""
        WITH grouped_defs AS (
            SELECT 
                word_id,
                definition_text,
                standardized_pos_id,
                string_agg(DISTINCT sources, ' | ') as merged_sources,
                string_agg(DISTINCT examples, ' | ') FILTER (WHERE examples IS NOT NULL AND examples != '') as all_examples,
                string_agg(DISTINCT usage_notes, ' | ') FILTER (WHERE usage_notes IS NOT NULL AND usage_notes != '') as all_notes,
                min(id) as keep_id
            FROM definitions
            GROUP BY word_id, definition_text, standardized_pos_id
            HAVING COUNT(*) > 1
        )
        UPDATE definitions d
        SET 
            sources = g.merged_sources,
            examples = CASE 
                WHEN g.all_examples IS NOT NULL THEN g.all_examples 
                ELSE d.examples 
            END,
            usage_notes = CASE 
                WHEN g.all_notes IS NOT NULL THEN g.all_notes 
                ELSE d.usage_notes 
            END
        FROM grouped_defs g
        WHERE d.id = g.keep_id;
    """)

    # 4. Delete duplicate definitions after merging their information
    cur.execute("""
        WITH duplicates AS (
            SELECT word_id, definition_text, standardized_pos_id, min(id) as keep_id
            FROM definitions
            GROUP BY word_id, definition_text, standardized_pos_id
            HAVING COUNT(*) > 1
        )
        DELETE FROM definitions d
        WHERE EXISTS (
            SELECT 1 FROM duplicates dup
            WHERE d.word_id = dup.word_id
            AND d.definition_text = dup.definition_text
            AND d.standardized_pos_id = dup.standardized_pos_id
            AND d.id != dup.keep_id
        );
    """)

    # 5. Update word tags with proper source information and actual tags
    cur.execute("""
        WITH word_sources AS (
            SELECT 
                d.word_id,
                string_agg(DISTINCT 
                    CASE 
                        WHEN d.sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                        WHEN d.sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                        WHEN d.sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                        WHEN d.sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                        WHEN d.sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                        ELSE d.sources
                    END, ', ') as sources
            FROM definitions d
            GROUP BY d.word_id
        )
        UPDATE words w
        SET tags = ws.sources
        FROM word_sources ws
        WHERE w.id = ws.word_id;
    """)

    # 6. Remove any remaining definitions with "Hindi Tiyak" when better POS exists
    cur.execute("""
        WITH better_pos AS (
            SELECT word_id
            FROM definitions d
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE p.code != 'unc'
        )
        DELETE FROM definitions d
        USING parts_of_speech p
        WHERE d.standardized_pos_id = p.id
        AND p.code = 'unc'
        AND EXISTS (
            SELECT 1 FROM better_pos b
            WHERE b.word_id = d.word_id
        );
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
