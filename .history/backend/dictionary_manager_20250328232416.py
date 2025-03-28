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
  python dictionary_manager.py lookup kamandag
  python dictionary_manager.py leaderboard
  python dictionary_manager.py stats
  python dictionary_manager.py help
  python dictionary_manager.py explore
  python dictionary_manager.py purge
  python dictionary_manager.py cleanup

[FIXED ISSUES - 2023-05-15]:
- Removed duplicate setup_logging function and duplicate logger initialization
- Removed duplicate definition of get_or_create_word_id function
- Removed duplicate declarations of classes (WordEntry, BaybayinRomanizer, etc.)
- Removed duplicate declarations of insert_definition and other functions
- Fixed inconsistent transaction management

[ADDITIONAL FIXES - 2025-03-24]:
- Removed another duplicate WordEntry class definition that was missed in previous cleanup
- Improved transaction management in get_or_create_word_id function by adding @with_transaction decorator
- Removed manual rollback in insert_definition as it's already using the transaction decorator
- Fixed transaction management in batch_get_or_create_word_ids by adding proper decorator
- Enhanced repair_database_issues with proper transaction management
- Removed all manual connection.rollback() calls in favor of the transaction decorator
- Fixed potential deadlocks by ensuring consistent transaction handling
- Improved code organization and consistency

[ADDITIONAL FIXES - 2025-03-28]:
- Added transaction management to additional functions including:
  - display_leaderboard (with read-only transactions)
  - check_data_access and check_query_performance
  - cleanup_relations, cleanup_dictionary_data, deduplicate_definitions
  - cleanup_baybayin_data, purge_database_tables
- Added get_cursor helper function to simplify obtaining a database cursor
- Updated CLI wrapper functions to use transaction-enabled functions
- Improved error handling across all database-related functions
- Ensured consistent use of @with_transaction decorator in all database operations

[ADDITIONAL FIXES - 2025-03-31]:
- Consolidated database configuration code to eliminate redundancy
- Removed duplicate DB_NAME, DB_USER, etc. variables as we already use DB_CONFIG
- Unified connection management between get_db_connection and get_connection
- Standardized error handling for database connections
- Improved transaction management consistency across all functions

If you need to modify this file, please ensure you don't reintroduce duplicates.
Each class and function should appear exactly once in the file.
Always use the @with_transaction decorator for functions that modify the database
instead of manual transaction management.
"""

import argparse
import json
import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2.extras import Json
from psycopg2.errors import UniqueViolation
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set, Iterator
import logging
from enum import Enum, auto
import re
from datetime import datetime, timedelta
import functools
import unidecode
import os
import sys
import unicodedata
from tqdm import tqdm
from dataclasses import dataclass
import enum
import time
import signal
import random
import textwrap
import csv
from dotenv import load_dotenv
import locale
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
from collections import deque

# Load Environment Variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Database connection pool configuration
DB_CONFIG = {
    'dbname': os.getenv("DB_NAME", "fil_dict"),
    'user': os.getenv("DB_USER", "postgres"),
    'password': os.getenv("DB_PASSWORD", "postgres"),
    'host': os.getenv("DB_HOST", "localhost"),
    'port': os.getenv("DB_PORT", "5432")
}

# Validate database configuration
if not all([DB_CONFIG['dbname'], DB_CONFIG['user'], DB_CONFIG['password'], DB_CONFIG['host']]):
    print("Error: Missing database configuration!")
    print("Please ensure you have a .env file with the following variables:")
    print("DB_NAME - The name of your PostgreSQL database")
    print("DB_USER - Your PostgreSQL username")
    print("DB_PASSWORD - Your PostgreSQL password")
    print("DB_HOST - Your database host (usually 'localhost')")
    print("DB_PORT - Your database port (default: 5432)")
    sys.exit(1)

# Initialize connection pool
try:
    connection_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        **DB_CONFIG
    )
    logger.info("Database connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize connection pool: {e}")
    connection_pool = None

def get_db_connection():
    """Get a connection from the pool."""
    if connection_pool:
        try:
            return connection_pool.getconn()
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
    
    # Fallback to direct connection if pool fails
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Failed to establish database connection: {e}")
        raise

def release_db_connection(conn):
    """Return a connection to the pool."""
    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except:
                pass

class DBConnection:
    """Context manager for database connections."""
    
    def __init__(self, autocommit=False):
        self.autocommit = autocommit
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = get_db_connection()
        if self.autocommit:
            self.conn.autocommit = True
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        return self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception occurred, rollback
            if self.conn and not self.autocommit:
                try:
                    self.conn.rollback()
                except:
                    pass
        else:
            # No exception, commit if not autocommit
            if self.conn and not self.autocommit:
                try:
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error committing transaction: {e}")
        
        # Close cursor and release connection
        if self.cursor:
            try:
                self.cursor.close()
            except:
                pass
        
        if self.conn:
            release_db_connection(self.conn)

def with_db_transaction(func):
    """Decorator to handle database transactions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with DBConnection() as cursor:
            # Add cursor as first argument if not already provided
            if args and hasattr(args[0], 'cursor'):
                # If first arg is a class instance with cursor, use that
                return func(*args, **kwargs)
            else:
                # Otherwise provide a cursor
                return func(cursor, *args, **kwargs)
    return wrapper

# -------------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------------
def setup_logging():
    """Configure logging with proper Unicode handling."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_path = f'{log_dir}/dictionary_manager_{timestamp}.log'
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
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
# Core Database Connection (Consolidated - Use get_db_connection instead)
# -------------------------------------------------------------------
def get_connection():
    """Legacy function - use get_db_connection() instead."""
    return get_db_connection()

def get_cursor():
    """Return a cursor from a new connection."""
    conn = get_db_connection()
    return conn.cursor()

# -------------------------------------------------------------------
# Setup Extensions
# -------------------------------------------------------------------
@with_transaction(commit=True)
def setup_extensions(cur):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    try:
        extensions = [
            'pg_trgm',
            'unaccent',
            'fuzzystrmatch',
            'dict_xsyn'
        ]
        for ext in extensions:
            cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = %s", (ext,))
            if cur.fetchone()[0] == 0:
                logger.info(f"Installing extension: {ext}")
                cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
            else:
                logger.info(f"Extension {ext} already installed")
        logger.info("Extensions setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up extensions: {str(e)}")
        raise

# -------------------------------------------------------------------
# Transaction Management Decorator
# -------------------------------------------------------------------

def with_transaction(commit=True):
    """
    Runs a function inside a transaction block.
    If an error occurs, the entire transaction is rolled back to clear
    any aborted state.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection
            # Ensure we're using transactions
            if conn.autocommit:
                conn.autocommit = False
            savepoint_name = f"sp_{func.__name__}"
            try:
                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                except Exception as e:
                    logger.warning(f"Could not create savepoint {savepoint_name}: {e}. Rolling back entire transaction.")
                    conn.rollback()
                    cur.execute("BEGIN")
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                result = func(cur, *args, **kwargs)
                if commit:
                    try:
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    except Exception as e:
                        logger.warning(f"Could not release savepoint {savepoint_name}: {e}")
                    conn.commit()
                return result
            except Exception as ex:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.warning(f"Could not rollback transaction: {rollback_error}")
                raise ex
        return wrapper
    return decorator

# -------------------------------------------------------------------
# Database Schema Creation / Update
# -------------------------------------------------------------------
TABLE_CREATION_SQL = r"""
-- Create timestamp update function if it doesn't exist
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'update_timestamp') THEN
        CREATE OR REPLACE FUNCTION update_timestamp()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $trigger$ language 'plpgsql';
    END IF;
END $$;

-- Create parts_of_speech table
CREATE TABLE IF NOT EXISTS parts_of_speech (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) NOT NULL UNIQUE,
    name_en VARCHAR(64) NOT NULL,
    name_tl VARCHAR(64) NOT NULL,
    description TEXT,
    CONSTRAINT parts_of_speech_code_uniq UNIQUE (code)
);
CREATE INDEX IF NOT EXISTS idx_parts_of_speech_code ON parts_of_speech(code);
CREATE INDEX IF NOT EXISTS idx_parts_of_speech_name ON parts_of_speech(name_en, name_tl);

-- Create words table
CREATE TABLE IF NOT EXISTS words (
    id SERIAL PRIMARY KEY,
    lemma VARCHAR(255) NOT NULL,
    normalized_lemma VARCHAR(255) NOT NULL,
    has_baybayin BOOLEAN DEFAULT FALSE,
    baybayin_form VARCHAR(255),
    romanized_form VARCHAR(255),
    language_code VARCHAR(16) NOT NULL,
    root_word_id INT REFERENCES words(id),
    preferred_spelling VARCHAR(255),
    tags TEXT,
    idioms JSONB DEFAULT '[]',
    pronunciation_data JSONB,
    source_info JSONB DEFAULT '{}',
    data_hash TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (normalized_lemma, language_code),
    CONSTRAINT baybayin_form_check CHECK (
        (has_baybayin = FALSE AND baybayin_form IS NULL) OR 
         (has_baybayin = TRUE AND baybayin_form IS NOT NULL)
    ),
    CONSTRAINT baybayin_form_regex CHECK (baybayin_form ~ '^[\u1700-\u171F\s]*$' OR baybayin_form IS NULL)
);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
         WHERE table_name = 'words' AND column_name = 'search_text'
    ) THEN
        ALTER TABLE words ADD COLUMN search_text TSVECTOR;
        UPDATE words SET search_text = to_tsvector('simple',
            COALESCE(lemma, '') || ' ' ||
            COALESCE(normalized_lemma, '') || ' ' ||
            COALESCE(baybayin_form, '') || ' ' ||
            COALESCE(romanized_form, '')
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_words_lemma ON words(lemma);
CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = TRUE;
CREATE INDEX IF NOT EXISTS idx_words_romanized ON words(romanized_form);
CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);

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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT definitions_unique UNIQUE (word_id, definition_text, standardized_pos_id)
);
CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
CREATE INDEX IF NOT EXISTS idx_definitions_pos ON definitions(standardized_pos_id);
CREATE INDEX IF NOT EXISTS idx_definitions_text ON definitions USING gin(to_tsvector('english', definition_text));

-- Create relations table
CREATE TABLE IF NOT EXISTS relations (
    id SERIAL PRIMARY KEY,
    from_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    to_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    relation_type VARCHAR(64) NOT NULL,
    sources TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT relations_unique UNIQUE (from_word_id, to_word_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_metadata ON relations USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_relations_metadata_strength ON relations((metadata->>'strength'));

DROP TABLE IF EXISTS etymologies CASCADE;
CREATE TABLE IF NOT EXISTS etymologies (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text TEXT NOT NULL,
    normalized_components TEXT,
    language_codes TEXT,
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT etymologies_wordid_etymtext_uniq UNIQUE (word_id, etymology_text)
);
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
CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);

DO $$ BEGIN
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
END $$;
"""

@with_transaction(commit=True)
def create_or_update_tables(cur):
    """Create or update database tables needed for the dictionary."""
    logger.info("Starting table creation/update process.")
    try:
        cur.execute("""
            DROP TABLE IF EXISTS 
                 definition_relations, affixations, relations, etymologies, 
                 definitions, words, parts_of_speech CASCADE;
        """)
        cur.execute(TABLE_CREATION_SQL)
        
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
        for code, name_en, name_tl, desc in pos_entries:
            cur.execute("""
                INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                 SET name_en = EXCLUDED.name_en,
                     name_tl = EXCLUDED.name_tl,
                     description = EXCLUDED.description
            """, (code, name_en, name_tl, desc))
        logger.info("Tables created or updated successfully.")
    except Exception as e:
        logger.error(f"Schema creation error: {str(e)}")
        raise

# -------------------------------------------------------------------
# Data Structures and Enums
# -------------------------------------------------------------------
class BaybayinCharType(Enum):
    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"
    
    @classmethod
    def get_type(cls, char: str) -> 'BaybayinCharType':
        if not char:
            return cls.UNKNOWN
        code_point = ord(char)
        if 0x1700 <= code_point <= 0x1702:
            return cls.VOWEL
        elif 0x1703 <= code_point <= 0x1711:
            return cls.CONSONANT
        elif code_point in (0x1712, 0x1713):
            return cls.VOWEL_MARK
        elif code_point == 0x1714:
            return cls.VIRAMA
        elif 0x1735 <= code_point <= 0x1736:
            return cls.PUNCTUATION
        return cls.UNKNOWN

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
        if not self.default_sound and self.char_type not in (BaybayinCharType.VIRAMA, BaybayinCharType.PUNCTUATION):
            raise ValueError("Default sound required for non-virama characters")
        code_point = ord(self.char)
        if not (0x1700 <= code_point <= 0x171F) and not (0x1735 <= code_point <= 0x1736):
            raise ValueError(f"Invalid Baybayin character: {self.char} (U+{code_point:04X})")
        expected_type = BaybayinCharType.get_type(self.char)
        if expected_type != self.char_type and expected_type != BaybayinCharType.UNKNOWN:
            raise ValueError(f"Character type mismatch for {self.char}: expected {expected_type}, got {self.char_type}")

    def get_sound(self, next_char: Optional['BaybayinChar'] = None) -> str:
        if self.char_type == BaybayinCharType.CONSONANT and next_char:
            if next_char.char_type == BaybayinCharType.VOWEL_MARK:
                return self.default_sound[:-1] + next_char.default_sound
            elif next_char.char_type == BaybayinCharType.VIRAMA:
                return self.default_sound[:-1]
        return self.default_sound

    def __str__(self) -> str:
        return f"{self.char} ({self.char_type.value}, sounds: {self.default_sound})"

@dataclass
class WordEntry:
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
        if not self.lemma or not isinstance(self.lemma, str):
            raise ValueError("Lemma must be a non-empty string")
        if self.language_code not in ('tl', 'ceb'):
            raise ValueError(f"Unsupported language code: {self.language_code}")
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")

class RelationshipCategory(Enum):
    """Categories for organizing relationship types"""
    SEMANTIC = "semantic"       # Meaning-based relationships
    DERIVATIONAL = "derivational"  # Word formation relationships
    VARIANT = "variant"         # Form variations
    TAXONOMIC = "taxonomic"     # Hierarchical relationships
    USAGE = "usage"             # Usage-based relationships
    OTHER = "other"             # Miscellaneous relationships

class RelationshipType(Enum):
    """
    Centralized enum for all relationship types with their properties.
    
    Each relationship type has the following properties:
    - rel_value: The string value stored in the database
    - category: The category this relationship belongs to
    - bidirectional: Whether this relationship applies in both directions
    - inverse: The inverse relationship type (if bidirectional is False)
    - transitive: Whether this relationship is transitive (A->B and B->C implies A->C)
    - strength: Default strength/confidence for this relationship (0-100)
    """
    # Semantic relationships
    SYNONYM = ("synonym", RelationshipCategory.SEMANTIC, True, None, True, 90)
    ANTONYM = ("antonym", RelationshipCategory.SEMANTIC, True, None, False, 90)
    RELATED = ("related", RelationshipCategory.SEMANTIC, True, None, False, 70)
    SIMILAR = ("similar", RelationshipCategory.SEMANTIC, True, None, False, 60)
    
    # Hierarchical/taxonomic relationships
    HYPERNYM = ("hypernym", RelationshipCategory.TAXONOMIC, False, "HYPONYM", True, 85)
    HYPONYM = ("hyponym", RelationshipCategory.TAXONOMIC, False, "HYPERNYM", True, 85)
    MERONYM = ("meronym", RelationshipCategory.TAXONOMIC, False, "HOLONYM", False, 80)
    HOLONYM = ("holonym", RelationshipCategory.TAXONOMIC, False, "MERONYM", False, 80)
    
    # Derivational relationships
    DERIVED_FROM = ("derived_from", RelationshipCategory.DERIVATIONAL, False, "ROOT_OF", False, 95)
    ROOT_OF = ("root_of", RelationshipCategory.DERIVATIONAL, False, "DERIVED_FROM", False, 95)
    
    # Variant relationships
    VARIANT = ("variant", RelationshipCategory.VARIANT, True, None, False, 85)
    SPELLING_VARIANT = ("spelling_variant", RelationshipCategory.VARIANT, True, None, False, 95)
    REGIONAL_VARIANT = ("regional_variant", RelationshipCategory.VARIANT, True, None, False, 90)
    
    # Usage relationships
    COMPARE_WITH = ("compare_with", RelationshipCategory.USAGE, True, None, False, 50)
    SEE_ALSO = ("see_also", RelationshipCategory.USAGE, True, None, False, 40)
    
    # Other relationships
    EQUALS = ("equals", RelationshipCategory.OTHER, True, None, True, 100)
    
    def __init__(self, rel_value, category, bidirectional, inverse, transitive, strength):
        self.rel_value = rel_value
        self.category = category
        self.bidirectional = bidirectional
        self.inverse = inverse
        self.transitive = transitive
        self.strength = strength
    
    @classmethod
    def from_string(cls, relation_str: str) -> 'RelationshipType':
        """Convert a string to a RelationshipType enum value"""
        normalized = relation_str.lower().replace(' ', '_')
        for rel_type in cls:
            if rel_type.rel_value == normalized:
                return rel_type
                
        # Handle legacy/alternative names
        legacy_mapping = {
            # Semantic relationships
            'synonym_of': cls.SYNONYM,
            'antonym_of': cls.ANTONYM,
            'related_to': cls.RELATED,
            'kasingkahulugan': cls.SYNONYM,
            'katulad': cls.SYNONYM,
            'kasalungat': cls.ANTONYM,
            'kabaligtaran': cls.ANTONYM,
            'kaugnay': cls.RELATED,
            
            # Derivational
            'derived': cls.DERIVED_FROM,
            'mula_sa': cls.DERIVED_FROM,
            
            # Variants
            'alternative_spelling': cls.SPELLING_VARIANT,
            'alternate_form': cls.VARIANT,
            'varyant': cls.VARIANT,
            'variants': cls.VARIANT,
            
            # Taxonomy
            'uri_ng': cls.HYPONYM,
            
            # Usage
            'see': cls.SEE_ALSO,
        }
        
        if normalized in legacy_mapping:
            return legacy_mapping[normalized]
            
        # Fall back to RELATED for unknown types
        logger.warning(f"Unknown relationship type: {relation_str}, using RELATED as fallback")
        return cls.RELATED
    
    def get_inverse(self) -> 'RelationshipType':
        """Get the inverse relationship type"""
        if self.bidirectional:
            return self
        if self.inverse:
            return getattr(RelationshipType, self.inverse)
        return RelationshipType.RELATED  # Fallback
    
    def __str__(self):
        return self.rel_value

class RelationshipManager:
    """
    A class to centralize all relationship operations.
    """
    def __init__(self, cursor):
        self.cursor = cursor
    
    def add_relationship(self, from_word_id: int, to_word_id: int, 
                        relationship_type: Union[RelationshipType, str], 
                        sources: str = "", metadata: Dict = None,
                        strength: int = None) -> bool:
        """
        Add a relationship between two words with optional metadata.
        
        Args:
            from_word_id: Source word ID
            to_word_id: Target word ID
            relationship_type: Type of relationship (enum or string)
            sources: Comma-separated list of data sources
            metadata: Optional dictionary of additional metadata
            strength: Optional relationship strength (overrides default)
            
        Returns:
            Success status
        """
        # Skip self-relationships
        if from_word_id == to_word_id:
            logger.warning(f"Skipping self-relationship for word ID {from_word_id}")
            return False
            
        # Get relationship type as RelationshipType enum
        rel_type = relationship_type
        if isinstance(relationship_type, str):
            try:
                rel_type = RelationshipType.from_string(relationship_type)
            except ValueError:
                logger.warning(f"Unknown relationship type: {relationship_type}")
                rel_type = None
                
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        if isinstance(rel_type, RelationshipType):
            # Resolve the relationship type enum to its string value
            rel_type_str = rel_type.rel_value
            
            # Add default strength if not specified
            if strength is None and 'strength' not in metadata:
                metadata['strength'] = rel_type.strength
        else:
            # Use the string value directly if not an enum
            rel_type_str = relationship_type
            
        # Override metadata strength if explicitly provided
        if strength is not None:
            metadata['strength'] = strength
                
        try:
            # Normalize sources to avoid duplicates
            if sources:
                sources = ", ".join(sorted(set(sources.split(", "))))
                
            # Check if relationship already exists
            self.cursor.execute("""
                SELECT id, sources, metadata
                FROM relations
                WHERE from_word_id = %s AND to_word_id = %s AND relation_type = %s
            """, (from_word_id, to_word_id, rel_type_str))
            
            existing = self.cursor.fetchone()
            
            if existing:
                # Update existing relationship
                rel_id, existing_sources, existing_metadata = existing
                
                # Merge sources
                if existing_sources and sources:
                    combined_sources = ", ".join(sorted(set(existing_sources.split(", ") + sources.split(", "))))
                else:
                    combined_sources = sources or existing_sources or ""
                    
                # Merge metadata
                if existing_metadata is None:
                    existing_metadata = {}
                if isinstance(existing_metadata, str):
                    try:
                        existing_metadata = json.loads(existing_metadata)
                    except (json.JSONDecodeError, TypeError):
                        existing_metadata = {}
                        
                combined_metadata = {**existing_metadata, **metadata} if metadata else existing_metadata
                
                self.cursor.execute("""
                    UPDATE relations
                    SET sources = %s,
                        metadata = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (combined_sources, json.dumps(combined_metadata), rel_id))
            else:
                # Insert new relationship
                self.cursor.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (from_word_id, to_word_id, rel_type_str, sources, json.dumps(metadata)))
                
            # Handle bidirectional relationships and inverses
            if isinstance(rel_type, RelationshipType):
                if rel_type.bidirectional:
                    # Create reciprocal relationship with the same type
                    self._ensure_bidirectional_exists(to_word_id, from_word_id, rel_type, sources, metadata)
                elif rel_type.inverse:
                    # Create inverse relationship
                    inverse_type = rel_type.get_inverse()
                    self._ensure_bidirectional_exists(to_word_id, from_word_id, inverse_type, sources, metadata)
                    
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship {from_word_id} -> {to_word_id} ({relationship_type}): {e}")
            return False
            
    def _ensure_bidirectional_exists(self, from_word_id: int, to_word_id: int, 
                                    rel_type: RelationshipType, sources: str = "", 
                                    metadata: Dict = None) -> bool:
        """
        Ensure a bidirectional or inverse relationship exists.
        This is a helper method for add_relationship.
        
        Args:
            from_word_id: Source word ID
            to_word_id: Target word ID
            rel_type: RelationshipType enum
            sources: Source information
            metadata: Metadata dictionary
            
        Returns:
            Success status
        """
        if not isinstance(rel_type, RelationshipType):
            logger.error(f"Invalid relationship type for bidirectional check: {rel_type}")
            return False
            
        rel_type_str = rel_type.rel_value
            
        try:
            # Check if relationship already exists
            self.cursor.execute("""
                SELECT id FROM relations 
                WHERE from_word_id = %s AND to_word_id = %s AND relation_type = %s
            """, (from_word_id, to_word_id, rel_type_str))
            
            if not self.cursor.fetchone():
                # Insert new relationship if it doesn't exist
                self.cursor.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                """, (from_word_id, to_word_id, rel_type_str, sources, json.dumps(metadata or {})))
                
            return True
                
        except Exception as e:
            logger.error(f"Error ensuring bidirectional relationship {from_word_id} -> {to_word_id} ({rel_type_str}): {e}")
            return False
            
    def batch_add_relationships(self, relationships: List[Dict]) -> Tuple[int, int]:
        """
        Add multiple relationships in a single batch operation.
        
        Args:
            relationships: List of dictionaries with the following keys:
                - from_word_id: Source word ID
                - to_word_id: Target word ID
                - relationship_type: RelationshipType or string
                - sources: Optional source string
                - metadata: Optional metadata dictionary
                - strength: Optional strength value
                
        Returns:
            Tuple of (success_count, error_count)
        """
        if not relationships:
            return (0, 0)
            
        success_count = 0
        error_count = 0
        
        # Group by insert/update operation to batch them
        inserts = []
        updates = []
        
        # Bidirectional relationships to create afterward
        bidirectional_rels = []
        
        try:
            # Process each relationship
            for rel in relationships:
                try:
                    from_id = rel['from_word_id']
                    to_id = rel['to_word_id']
                    rel_type = rel['relationship_type']
                    sources = rel.get('sources', '')
                    metadata = rel.get('metadata', {})
                    strength = rel.get('strength')
                    
                    # Skip self-relationships
                    if from_id == to_id:
                        error_count += 1
                        continue
                    
                    # Normalize relationship type
                    if isinstance(rel_type, str):
                        try:
                            rel_type = RelationshipType.from_string(rel_type)
                        except ValueError:
                            # If string doesn't match known type, use as-is
                            pass
                    
                    # Prepare metadata
                    if metadata is None:
                        metadata = {}
                    
                    if isinstance(rel_type, RelationshipType):
                        # Resolve the relationship type enum to its string value
                        rel_type_str = rel_type.rel_value
                        
                        # Add default strength if not specified
                        if strength is None and 'strength' not in metadata:
                            metadata['strength'] = rel_type.strength
                    else:
                        # Use the string value directly
                        rel_type_str = rel_type
                    
                    # Override metadata strength if explicitly provided
                    if strength is not None:
                        metadata['strength'] = strength
                    
                    # Check if relationship already exists
                    self.cursor.execute("""
                        SELECT id FROM relations 
                        WHERE from_word_id = %s AND to_word_id = %s AND relation_type = %s
                    """, (from_id, to_id, rel_type_str))
                    
                    existing = self.cursor.fetchone()
                    
                    if existing:
                        # Update existing relationship
                        updates.append({
                            'id': existing[0],
                            'sources': sources,
                            'metadata': metadata
                        })
                    else:
                        # Insert new relationship
                        inserts.append({
                            'from_id': from_id,
                            'to_id': to_id,
                            'rel_type': rel_type_str,
                            'sources': sources,
                            'metadata': metadata
                        })
                    
                    # Handle bidirectional relationships and inverses
                    if isinstance(rel_type, RelationshipType):
                        if rel_type.bidirectional:
                            # Create reciprocal relationship with the same type
                            bidirectional_rels.append({
                                'from_word_id': to_id,
                                'to_word_id': from_id,
                                'relationship_type': rel_type,
                                'sources': sources,
                                'metadata': metadata
                            })
                        elif rel_type.inverse:
                            # Create inverse relationship
                            inverse_type = rel_type.get_inverse()
                            bidirectional_rels.append({
                                'from_word_id': to_id,
                                'to_word_id': from_id,
                                'relationship_type': inverse_type,
                                'sources': sources,
                                'metadata': metadata
                            })
                    
                    success_count += 1
                
                except Exception as e:
                    logger.error(f"Error processing relationship {rel}: {e}")
                    error_count += 1
            
            # Perform batch inserts
            if inserts:
                insert_values = []
                for rel in inserts:
                    insert_values.append((
                        rel['from_id'],
                        rel['to_id'],
                        rel['rel_type'],
                        rel['sources'],
                        json.dumps(rel['metadata']) if rel['metadata'] else '{}'
                    ))
                
                # Use executemany for better performance
                self.cursor.executemany("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                """, insert_values)
            
            # Perform batch updates
            if updates:
                for rel in updates:
                    # Updates need separate queries because the conditions vary
                    self.cursor.execute("""
                        UPDATE relations
                        SET sources = CASE 
                                WHEN sources IS NULL OR sources = '' THEN %s
                                WHEN %s IS NULL OR %s = '' THEN sources
                                ELSE sources || ', ' || %s
                            END,
                            metadata = CASE
                                WHEN metadata IS NULL THEN %s::jsonb
                                ELSE metadata || %s::jsonb
                            END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (
                        rel['sources'], rel['sources'], rel['sources'], rel['sources'],
                        json.dumps(rel['metadata']), json.dumps(rel['metadata']),
                        rel['id']
                    ))
            
            # Process bidirectional relationships recursively, but avoid infinite recursion
            if bidirectional_rels:
                # Only process relationships that don't already exist
                filtered_rels = []
                for rel in bidirectional_rels:
                    from_id = rel['from_word_id']
                    to_id = rel['to_word_id']
                    rel_type = rel['relationship_type']
                    
                    rel_type_str = rel_type.rel_value if isinstance(rel_type, RelationshipType) else rel_type
                    
                    self.cursor.execute("""
                        SELECT 1 FROM relations 
                        WHERE from_word_id = %s AND to_word_id = %s AND relation_type = %s
                    """, (from_id, to_id, rel_type_str))
                    
                    if not self.cursor.fetchone():
                        filtered_rels.append(rel)
                
                if filtered_rels:
                    bi_success, bi_error = self.batch_add_relationships(filtered_rels)
                    success_count += bi_success
                    error_count += bi_error
            
            return (success_count, error_count)
            
        except Exception as e:
            logger.error(f"Error in batch relationship processing: {e}")
            return (success_count, error_count)
    
    def get_related_words(self, word_id: int, 
                          relationship_types: Optional[List[Union[RelationshipType, str]]] = None,
                          include_metadata: bool = False,
                          category: Optional[RelationshipCategory] = None,
                          transitive: bool = False) -> List[Dict]:
        """
        Get words related to the given word.
        
        Args:
            word_id: The word ID to find relations for
            relationship_types: Specific types of relationships to include, or None for all
            include_metadata: Whether to include relationship metadata
            category: Filter by relationship category
            transitive: Whether to include transitive relationships
            
        Returns:
            List of related words with their relationship information
        """
        # Prepare relationship type filter
        rel_type_filter = ""
        params = [word_id]
        
        if relationship_types:
            type_values = []
            for rt in relationship_types:
                if isinstance(rt, str):
                    type_values.append(str(RelationshipType.from_string(rt)))
                else:
                    type_values.append(str(rt))
            
            rel_type_filter = f"AND r.relation_type IN ({', '.join(['%s'] * len(type_values))})"
            params.extend(type_values)
        elif category:
            # Filter by category
            type_values = [str(rt) for rt in RelationshipType if rt.category == category]
            if type_values:
                rel_type_filter = f"AND r.relation_type IN ({', '.join(['%s'] * len(type_values))})"
                params.extend(type_values)
        
        # Query related words
        select_metadata = ", r.metadata" if include_metadata else ""
        
        query = f"""
            SELECT w.id, w.lemma, w.language_code, r.relation_type, r.sources{select_metadata}
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            {rel_type_filter}
            ORDER BY r.relation_type, w.lemma
        """
        
        try:
            self.cursor.execute(query, params)
            results = []
            
            for row in self.cursor.fetchall():
                if include_metadata:
                    word_id, lemma, lang_code, rel_type, sources, metadata = row
                    results.append({
                        'word_id': word_id,
                        'lemma': lemma,
                        'language_code': lang_code,
                        'relation_type': rel_type,
                        'sources': sources,
                        'metadata': metadata
                    })
                else:
                    word_id, lemma, lang_code, rel_type, sources = row
                    results.append({
                        'word_id': word_id,
                        'lemma': lemma,
                        'language_code': lang_code,
                        'relation_type': rel_type,
                        'sources': sources
                    })
            
            # Handle transitive relationships if requested
            if transitive and results:
                transitive_results = []
                seen_word_ids = {row['word_id'] for row in results}
                seen_word_ids.add(word_id)  # Add the original word ID
                
                for row in results:
                    rel_obj = RelationshipType.from_string(row['relation_type'])
                    if rel_obj.transitive:
                        # Look for transitive relationships
                        trans_results = self.get_transitive_relationships(row['word_id'], rel_obj, seen_word_ids)
                        transitive_results.extend(trans_results)
                        seen_word_ids.update(r['word_id'] for r in trans_results)
                
                results.extend(transitive_results)
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving related words for word ID {word_id}: {e}")
            return []
    
    def get_transitive_relationships(self, word_id: int, rel_type: RelationshipType, 
                                    seen_word_ids: Set[int]) -> List[Dict]:
        """
        Get transitive relationships for a word (e.g., if A->B and B->C, then A->C).
        
        Args:
            word_id: The word ID to find transitive relations for
            rel_type: The relationship type
            seen_word_ids: Set of already seen word IDs to avoid cycles
            
        Returns:
            List of transitively related words
        """
        if not rel_type.transitive:
            return []
            
        try:
            self.cursor.execute("""
                SELECT w.id, w.lemma, w.language_code, r.relation_type, r.sources
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = %s AND r.relation_type = %s
            """, (word_id, str(rel_type)))
            
            results = []
            for row in self.cursor.fetchall():
                to_word_id, to_lemma, to_lang, to_rel_type, sources = row
                if to_word_id in seen_word_ids:
                    continue
                    
                results.append({
                    'word_id': to_word_id,
                    'lemma': to_lemma,
                    'language_code': to_lang,
                    'relation_type': to_rel_type,
                    'sources': sources,
                    'transitive': True
                })
                
                # No need to recurse further to avoid deep dependency chains
                
            return results
        except Exception as e:
            logger.error(f"Error retrieving transitive relationships for word ID {word_id}: {e}")
            return []
    
    def find_relationship_paths(self, from_word_id: int, to_word_id: int, 
                               max_depth: int = 3, 
                               relationship_types: Optional[List[Union[RelationshipType, str]]] = None,
                               prefer_categories: Optional[List[RelationshipCategory]] = None) -> List[List[Dict]]:
        """
        Find paths between two words through relationships using breadth-first search.
        
        Args:
            from_word_id: The source word ID
            to_word_id: The target word ID
            max_depth: Maximum path length to search
            relationship_types: Optional list of relationship types to consider (filters path steps)
            prefer_categories: Optional list of relationship categories to prioritize
            
        Returns:
            List of paths, where each path is a list of relationship steps
        """
        if from_word_id == to_word_id:
            logger.info(f"Source and target words are the same: {from_word_id}")
            return [[{"word_id": from_word_id, "relation_type": "self"}]]
            
        # Initialize data structures for BFS
        global_visited = set()  # Track all visited words for performance
        queue = deque()  # Queue of paths to explore
        found_paths = []  # Paths that reach the target
        
        # Normalize relationship types input
        if relationship_types:
            normalized_types = []
            for rel_type in relationship_types:
                if isinstance(rel_type, str):
                    try:
                        rel_type = RelationshipType.from_string(rel_type)
                    except ValueError:
                        continue
                normalized_types.append(rel_type)
            relationship_types = normalized_types
        
        # Get initial word info
        try:
            self.cursor.execute("SELECT lemma, language_code FROM words WHERE id = %s", (from_word_id,))
            from_word = self.cursor.fetchone()
            if not from_word:
                logger.error(f"Source word ID {from_word_id} not found")
                return []
                
            from_lemma, from_lang = from_word
            
            # Start with the source word
            initial_path = [{
                "word_id": from_word_id,
                "lemma": from_lemma,
                "language_code": from_lang,
                "relation_type": "start",
                "path_visited": {from_word_id}  # Track visited words in this path
            }]
            queue.append(initial_path)
            global_visited.add(from_word_id)
            
            # Perform BFS with limits to prevent excessive computation
            max_iterations = 10000  # Safety limit
            iterations = 0
            
            while queue and len(found_paths) < 10 and iterations < max_iterations:  # Limit paths and iterations
                iterations += 1
                current_path = queue.popleft()
                current_node = current_path[-1]
                current_word_id = current_node["word_id"]
                path_visited = current_node["path_visited"]
                
                # Get all related words
                related_words = self.get_related_words(
                    current_word_id, 
                    relationship_types=relationship_types,
                    include_metadata=True
                )
                
                # Sort by category preference if specified
                if prefer_categories:
                    related_words.sort(
                        key=lambda x: self._get_category_preference_score(x, prefer_categories),
                        reverse=True
                    )
                # Then sort by relationship strength
                related_words.sort(
                    key=lambda x: x.get("metadata", {}).get("strength", 0) 
                    if isinstance(x.get("metadata"), dict) else 0,
                    reverse=True
                )
                
                # Check each related word
                for related in related_words:
                    related_id = related["word_id"]
                    
                    # Skip if already in this path to avoid cycles
                    if related_id in path_visited:
                        continue
                        
                    # Create a new path with this relation
                    new_path_visited = path_visited.copy()
                    new_path_visited.add(related_id)
                    
                    new_node = {
                        "word_id": related_id,
                        "lemma": related["lemma"],
                        "language_code": related["language_code"],
                        "relation_type": related["relation_type"],
                        "relation_id": related.get("relation_id"),
                        "metadata": related.get("metadata", {}),
                        "path_visited": new_path_visited
                    }
                    
                    # Remove path_visited from the node copy used in the result path
                    result_node = {k: v for k, v in new_node.items() if k != 'path_visited'}
                    new_path = current_path[:-1] + [current_node.copy()] + [result_node]
                    
                    # Remove path_visited from the result nodes
                    for node in new_path:
                        if "path_visited" in node:
                            del node["path_visited"]
                    
                    # Check if we've reached the target
                    if related_id == to_word_id:
                        found_paths.append(new_path)
                        # Don't break - we want to find all paths up to max_depth
                    
                    # If we haven't reached max depth, add to queue for further exploration
                    elif len(new_path) < max_depth + 1:  # +1 because path includes starting node
                        # Only add to queue if not globally visited (optimization)
                        if related_id not in global_visited:
                            global_visited.add(related_id)
                            queue.append(current_path[:-1] + [new_node])
            
            if iterations >= max_iterations:
                logger.warning(f"Path finding reached safety limit of {max_iterations} iterations")
            
            # Sort found paths by total strength and path length
            def path_score(path):
                # Calculate weighted score based on relationship types and strengths
                total_strength = 0
                path_length = len(path) - 1  # Exclude start node
                
                if path_length == 0:
                    return 0
                    
                for node in path[1:]:  # Skip the first node (start)
                    # Get strength from metadata or use default from RelationshipType
                    metadata = node.get("metadata", {})
                    if isinstance(metadata, dict):
                        strength = metadata.get("strength", 50)
                    else:
                        strength = 50
                        
                    # Add relationship type bonus
                    rel_type = node.get("relation_type", "")
                    if rel_type == "synonym" or rel_type == "equals":
                        strength += 20
                    elif rel_type in ("derived_from", "root_of"):
                        strength += 15
                    elif rel_type in ("hypernym", "hyponym"):
                        strength += 10
                        
                    total_strength += strength
                
                # Prefer shorter paths with higher average strength
                return total_strength / path_length
                
            found_paths.sort(key=path_score, reverse=True)
            
            # Return with useful info
            if found_paths:
                logger.info(f"Found {len(found_paths)} paths between words {from_word_id} and {to_word_id}")
            else:
                logger.info(f"No paths found between words {from_word_id} and {to_word_id} within depth {max_depth}")
                
            return found_paths
            
        except Exception as e:
            logger.error(f"Error finding relationship paths: {str(e)}")
            logger.exception(e)
            return []
            
    def _get_category_preference_score(self, relation, preferred_categories):
        """Helper method to score relations based on category preference."""
        try:
            rel_type_str = relation.get("relation_type", "")
            if not rel_type_str:
                return 0
                
            rel_type = RelationshipType.from_string(rel_type_str)
            category = rel_type.category
            
            # Check if this category is in preferred list
            for i, preferred in enumerate(preferred_categories):
                if category == preferred:
                    # Earlier categories in the list get higher scores
                    return len(preferred_categories) - i
                    
            return 0
        except Exception:
            return 0

def get_standardized_pos_id(cur, pos_string: str) -> int:
    if not pos_string:
        return get_uncategorized_pos_id(cur)
    pos_key = pos_string.lower().strip()
    code = get_standard_code(pos_key)
    cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (code,))
    result = cur.fetchone()
    return result[0] if result else get_uncategorized_pos_id(cur)

def get_standard_code(pos_key: str) -> str:
    """
    Standardize part of speech codes from various formats.
    Converts common abbreviations and full names to standard codes.
    """
    # Common mappings for parts of speech
    pos_mappings = {
        # Nouns
        'noun': 'n', 'pangngalan': 'n', 'n.': 'n', 'pangngal': 'n', 'pangalan': 'n',
        # Verbs
        'verb': 'v', 'pandiwa': 'v', 'v.': 'v', 'pandw': 'v',
        # Adjectives
        'adjective': 'adj', 'pang-uri': 'adj', 'panguri': 'adj', 'adj.': 'adj', 'p-uri': 'adj',
        # Adverbs
        'adverb': 'adv', 'pang-abay': 'adv', 'pangabay': 'adv', 'adv.': 'adv', 'p-abay': 'adv',
        # Pronouns
        'pronoun': 'pron', 'panghalip': 'pron', 'pron.': 'pron',
        # Prepositions
        'preposition': 'prep', 'pang-ukol': 'prep', 'prep.': 'prep', 'pangukol': 'prep',
        # Conjunctions
        'conjunction': 'conj', 'pangatnig': 'conj', 'conj.': 'conj',
        # Interjections
        'interjection': 'intj', 'pandamdam': 'intj', 'intj.': 'intj',
        # Determiners
        'determiner': 'det', 'pantukoy': 'det', 'det.': 'det',
        # Affixes
        'affix': 'affix', 'panlapi': 'affix',
        # Idioms and expressions
        'idiom': 'idm', 'idyoma': 'idm', 'idm.': 'idm', 'kasabihan': 'idm',
        # Colloquial terms
        'colloquial': 'col', 'kolokyal': 'col', 'col.': 'col', 'slang': 'col',
        # Other categories
        'variant': 'var', 'varyant': 'var', 
        'texting': 'tx', 'text': 'tx',
        'english': 'eng', 'spanish': 'spa'
    }
    
    # Check for direct match in mappings
    cleaned_key = pos_key.replace('-', '').replace('.', '').replace(' ', '')
    if cleaned_key in pos_mappings:
        return pos_mappings[cleaned_key]
    
    # Check for partial matches
    for key, code in pos_mappings.items():
        if cleaned_key.startswith(key) or key.startswith(cleaned_key):
            return code
    
    # Default to uncategorized if no match
    return 'unc'

def get_uncategorized_pos_id(cur) -> int:
    cur.execute("SELECT id FROM parts_of_speech WHERE code = 'unc'")
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        cur.execute("""
            INSERT INTO parts_of_speech (code, name_en, name_tl, description)
            VALUES ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
            RETURNING id
        """)
        return cur.fetchone()[0]

# -------------------------------------------------------------------
# Core Helper Functions
# -------------------------------------------------------------------
def normalize_lemma(text: str) -> str:
    if not text:
        logger.warning("normalize_lemma received empty or None text")
        return ""
    return unidecode.unidecode(text).lower()

def extract_etymology_components(etymology_text):
    """Extract structured components from etymology text"""
    if not etymology_text:
        return None
    
    # Here we could implement more structured etymology parsing
    # For now, just returning a simple structure
    return {
        "original_text": etymology_text,
        "processed": True
    }

def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    match = re.search(r'\(([^)]+)\)', text)
    if match:
        meaning = match.group(1)
        clean_text = text.replace(match.group(0), '').strip()
        return clean_text, meaning
    return text, None

def validate_word_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Word data must be a dictionary")
    required_fields = {'lemma', 'language_code'}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    lemma = data['lemma']
    if not isinstance(lemma, str) or not lemma.strip():
        raise ValueError("Lemma must be a non-empty string")
    if len(lemma) > 255:
        raise ValueError("Lemma exceeds maximum length")
    if data['language_code'] not in {'tl', 'ceb'}:
        raise ValueError(f"Unsupported language code: {data['language_code']}")
    if 'tags' in data:
        if not isinstance(data['tags'], (str, list)):
            raise ValueError("Tags must be string or list")
        if isinstance(data['tags'], list):
            data['tags'] = ','.join(str(tag) for tag in data['tags'])
    return data

def has_diacritics(text: str) -> bool:
    normalized = normalize_lemma(text)
    return text != normalized

class SourceStandardization:
    @staticmethod
    def standardize_sources(source: str) -> str:
        source_mapping = {
            'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
            'kaikki.jsonl': 'kaikki.org (Tagalog)',
            'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
            'root_words_with_associated_words_cleaned.json': 'tagalog.com',
            'tagalog-words.json': 'diksiyonaryo.ph'
        }
        return source_mapping.get(source, source)
    
    @staticmethod
    def get_display_name(source: str) -> str:
        return SourceStandardization.standardize_sources(source)

def get_standardized_source(source: str) -> str:
    return SourceStandardization.standardize_sources(source)

def get_standardized_source_sql() -> str:
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
    prefix = "Baybayin spelling of"
    if lemma.lower().startswith(prefix.lower()):
        return lemma[len(prefix):].strip()
    return lemma

def format_word_display(word: str, show_baybayin: bool = True) -> str:
    has_bb = any(0x1700 <= ord(c) <= 0x171F for c in word)
    if has_bb:
        romanized = get_romanized_text(word)
        if show_baybayin:
            return f"[bold cyan]{word}[/] [dim](romanized: {romanized})[/]"
        else:
            return romanized
    return word

def get_root_word_id(cur: "psycopg2.extensions.cursor", lemma: str, language_code: str) -> Optional[int]:
    cur.execute("""
        SELECT id FROM words 
        WHERE normalized_lemma = %s AND language_code = %s AND root_word_id IS NULL
    """, (normalize_lemma(lemma), language_code))
    result = cur.fetchone()
    return result[0] if result else None

# -------------------------------------------------------------------
# Baybayin Processing System
# -------------------------------------------------------------------
class BaybayinRomanizer:
    """Handles romanization of Baybayin text."""
    
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
        all_chars = set()
        for char_set in [self.VOWELS, self.CONSONANTS, self.VOWEL_MARKS, {self.VIRAMA.char: self.VIRAMA}, self.PUNCTUATION]:
            for char in char_set:
                if char in all_chars:
                    raise ValueError(f"Duplicate character in mappings: {char}")
                all_chars.add(char)
    
    def is_baybayin(self, text: str) -> bool:
        return any(0x1700 <= ord(c) <= 0x171F for c in text)
    
    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
        if char in self.VOWELS:
            return self.VOWELS[char]
        if char in self.CONSONANTS:
            return self.CONSONANTS[char]
        if char in self.VOWEL_MARKS:
            return self.VOWEL_MARKS[char]
        if char == self.VIRAMA.char:
            return self.VIRAMA
        if char in self.PUNCTUATION:
            return self.PUNCTUATION[char]
        return None
    
    def process_syllable(self, chars: List[str]) -> Tuple[str, int]:
        if not chars:
            return '', 0
        first_char = self.get_char_info(chars[0])
        if not first_char:
            return '', 1
        if first_char.char_type == BaybayinCharType.VOWEL:
            return first_char.default_sound, 1
        if first_char.char_type == BaybayinCharType.CONSONANT:
            result = first_char.default_sound
            pos = 1
            if pos < len(chars):
                next_char = self.get_char_info(chars[pos])
                if next_char and next_char.char_type == BaybayinCharType.VOWEL_MARK:
                    result = result[:-1] + next_char.default_sound
                    pos += 1
                elif next_char and next_char.char_type == BaybayinCharType.VIRAMA:
                    result = result[:-1]
                    pos += 1
            return result, pos
        return '', 1
    
    def romanize(self, text: str) -> str:
        if not text:
            return ''
        result = []
        chars = list(text)
        i = 0
        while i < len(chars):
            if chars[i].isspace():
                result.append(' ')
                i += 1
                continue
            char_info = self.get_char_info(chars[i])
            if not char_info:
                i += 1
                continue
            if char_info.char_type == BaybayinCharType.PUNCTUATION:
                result.append(char_info.default_sound)
                i += 1
                continue
            romanized, consumed = self.process_syllable(chars[i:])
            result.append(romanized)
            i += consumed
        return ''.join(result).strip()
    
    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        chars = list(text)
        i = 0
        while i < len(chars):
            if chars[i].isspace():
                i += 1
                continue
            char_info = self.get_char_info(chars[i])
            if not char_info:
                return False
            if char_info.char_type == BaybayinCharType.VOWEL_MARK and (
                i == 0 or not self.get_char_info(chars[i-1]) or 
                self.get_char_info(chars[i-1]).char_type != BaybayinCharType.CONSONANT
            ):
                return False
            if char_info.char_type == BaybayinCharType.VIRAMA and (
                i == 0 or not self.get_char_info(chars[i-1]) or 
                self.get_char_info(chars[i-1]).char_type != BaybayinCharType.CONSONANT
            ):
                return False
            i += 1
        return True

def process_baybayin_text(text: str) -> Tuple[str, Optional[str], bool]:
    if not text:
        return text, None, False
    romanizer = BaybayinRomanizer()
    has_bb = romanizer.is_baybayin(text)
    if not has_bb:
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
    romanizer = BaybayinRomanizer()
    try:
        return romanizer.romanize(text)
    except ValueError:
        return text

def transliterate_to_baybayin(text: str) -> str:
    """
    Transliterate Latin text to Baybayin script.
    Handles all Filipino vowels (a, e, i, o, u) and consonants,
    including final consonants with virama.
    """
    if not text:
        return ""
        
    # Normalize text: lowercase and remove diacritical marks
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                  if not unicodedata.combining(c))
    
    # Define Baybayin character mappings
    consonants = {
        'k': 'ᜃ', 'g': 'ᜄ', 'ng': 'ᜅ', 't': 'ᜆ', 'd': 'ᜇ', 'n': 'ᜈ',
        'p': 'ᜉ', 'b': 'ᜊ', 'm': 'ᜋ', 'y': 'ᜌ', 'l': 'ᜎ', 'w': 'ᜏ',
        's': 'ᜐ', 'h': 'ᜑ'
    }
    vowels = {'a': 'ᜀ', 'i': 'ᜁ', 'e': 'ᜁ', 'u': 'ᜂ', 'o': 'ᜂ'}
    vowel_marks = {'i': 'ᜒ', 'e': 'ᜒ', 'u': 'ᜓ', 'o': 'ᜓ'}
    
    # Process text by analyzing patterns
    result = ""
    i = 0
    
    while i < len(text):
        # Check for 'ng' digraph first
        if i + 1 < len(text) and text[i:i+2] == 'ng':
            if i + 2 < len(text) and text[i+2] in 'aeiou':
                # ng + vowel
                if text[i+2] == 'a':
                    result += consonants['ng']
                else:
                    result += consonants['ng'] + vowel_marks[text[i+2]]
                i += 3
            else:
                # Final 'ng'
                result += consonants['ng'] + '᜔'  # Add virama
                i += 2
        # Handle single consonants
        elif text[i] in 'kgtdnpbmylswh':
            if i + 1 < len(text) and text[i+1] in 'aeiou':
                # Consonant + vowel
                if text[i+1] == 'a':
                    result += consonants[text[i]]
                else:
                    result += consonants[text[i]] + vowel_marks[text[i+1]]
                i += 2
            else:
                # Final consonant
                result += consonants[text[i]] + '᜔'  # Add virama
                i += 1
        # Handle vowels
        elif text[i] in 'aeiou':
            result += vowels[text[i]]
            i += 1
        # Skip other characters
        else:
            i += 1
    
    return result

@with_transaction(commit=True)
def verify_baybayin_data(cur):
    """Verify the consistency of Baybayin data in the database."""
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
