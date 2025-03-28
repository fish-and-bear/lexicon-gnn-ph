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
    'dbname': os.getenv('DB_NAME', 'fil_dict_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

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
# Database Configuration
# -------------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME", "fil_dict_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
    print("Error: Missing database configuration!")
    print("Please ensure you have a .env file with the following variables:")
    print("DB_NAME - The name of your PostgreSQL database")
    print("DB_USER - Your PostgreSQL username")
    print("DB_PASSWORD - Your PostgreSQL password")
    print("DB_HOST - Your database host (usually 'localhost')")
    print("DB_PORT - Your database port (default: 5432)")
    sys.exit(1)

# -------------------------------------------------------------------
# Core Database Connection
# -------------------------------------------------------------------
def get_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', DB_NAME),
        user=os.environ.get('DB_USER', DB_USER),
        password=os.environ.get('DB_PASSWORD', DB_PASSWORD),
        host=os.environ.get('DB_HOST', DB_HOST),
        port=os.environ.get('DB_PORT', DB_PORT)
    )
    conn.autocommit = False  # Ensure transactions are used.
    return conn

def get_cursor():
    """Return a cursor from a new connection."""
    conn = get_connection()
    return conn.cursor()

# -------------------------------------------------------------------
# Setup Extensions
# -------------------------------------------------------------------
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
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
                conn.commit()
            else:
                logger.info(f"Extension {ext} already installed")
    except Exception as e:
        logger.error(f"Error setting up extensions: {str(e)}")
        conn.rollback()
    finally:
        cur.close()

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

def create_or_update_tables(conn):
    logger.info("Starting table creation/update process.")
    cur = conn.cursor()
    try:
        cur.execute("""
            DROP TABLE IF EXISTS 
                 definition_relations, affixations, relations, etymologies, 
                 definitions, words, parts_of_speech CASCADE;
        """)
        cur.execute(TABLE_CREATION_SQL)
        conn.commit()
        
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
        conn.commit()
        logger.info("Tables created or updated successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {str(e)}")
        raise
    finally:
        cur.close()

# Define standard part of speech mappings
POS_MAPPING = {
    # Nouns
    'noun': 'n', 'pangngalan': 'n', 'name': 'n', 'n': 'n', 'pangalan': 'n',
    # Verbs
    'verb': 'v', 'pandiwa': 'v', 'v': 'v', 'action': 'v',
    # Adjectives
    'adjective': 'adj', 'pang-uri': 'adj', 'adj': 'adj', 'quality': 'adj', 'uri': 'adj',
    # Adverbs
    'adverb': 'adv', 'pang-abay': 'adv', 'adv': 'adv', 'manner': 'adv', 'abay': 'adv',
    # Pronouns
    'pronoun': 'pron', 'panghalip': 'pron', 'pron': 'pron', 'halip': 'pron',
    # Prepositions
    'preposition': 'prep', 'pang-ukol': 'prep', 'prep': 'prep', 'ukol': 'prep',
    # Conjunctions
    'conjunction': 'conj', 'pangatnig': 'conj', 'conj': 'conj', 'katnig': 'conj',
    # Interjections
    'interjection': 'intj', 'pandamdam': 'intj', 'intj': 'intj', 'damdam': 'intj',
    # Articles
    'article': 'art', 'pantukoy': 'art', 'art': 'art', 'tukoy': 'art',
    # Others
    'expression': 'expr', 'pahayag': 'expr', 'expr': 'expr',
    'phrase': 'phr', 'parirala': 'phr', 'phr': 'phr',
    'affix': 'aff', 'panlapi': 'aff', 'aff': 'aff', 'lapi': 'aff',
    'prefix': 'pref', 'unlapi': 'pref', 'pref': 'pref',
    'suffix': 'suff', 'hulapi': 'suff', 'suff': 'suff',
    'infix': 'inf', 'gitlapi': 'inf', 'inf': 'inf',
    'particle': 'part', 'kataga': 'part', 'part': 'part',
    'number': 'num', 'bilang': 'num', 'num': 'num',
    'determiner': 'det', 'panuri': 'det', 'det': 'det',
    'auxiliary': 'aux', 'pantulong': 'aux', 'aux': 'aux'
}

def get_standard_code(pos_string: str) -> str:
    """
    Convert a part of speech string to a standardized code.
    
    Args:
        pos_string: The part of speech string to standardize
        
    Returns:
        A standardized code for the part of speech
    """
    if not pos_string:
        return 'unc'  # Uncategorized
        
    # Remove any parenthetical information and clean string
    pos_key = pos_string.lower().strip()
    pos_key = re.sub(r'\([^)]*\)', '', pos_key).strip()
    
    # Check for direct match in mapping
    if pos_key in POS_MAPPING:
        return POS_MAPPING[pos_key]
    
    # Check for partial matches (e.g., "common noun" -> "noun")
    for key, code in POS_MAPPING.items():
        if key in pos_key:
            return code
    
    # Default to uncategorized if no match found
    return 'unc'

def standardize_entry_pos(pos_str: str) -> str:
    """Standardize part-of-speech string in dictionary entries."""
    # Use the existing get_standard_code function
    return get_standard_code(pos_str)
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
                
                # No need to recurse further for now to avoid deep dependency chains
                # For a complete implementation, you could recursively call 
                # get_transitive_relationships here
                
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

@with_transaction(commit=False)
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

@with_transaction(commit=True)
def merge_baybayin_entries(cur, baybayin_id: int, romanized_id: int):
    """Merge a Baybayin entry with its romanized form."""
    try:
        cur.execute("""
            SELECT lemma, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """, (baybayin_id,))
        baybayin_result = cur.fetchone()
        if not baybayin_result:
            raise ValueError(f"Baybayin entry {baybayin_id} not found")
        baybayin_lemma, baybayin_form, baybayin_rom = baybayin_result
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
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = COALESCE(%s, baybayin_form),
                romanized_form = COALESCE(%s, romanized_form),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (baybayin_form, baybayin_rom or baybayin_lemma, romanized_id))
        cur.execute("DELETE FROM words WHERE id = %s", (baybayin_id,))
    except Exception as e:
        logger.error(f"Error merging Baybayin entries: {str(e)}")
        raise

def extract_baybayin_text(text: str) -> List[str]:
    parts = re.split(r'[^ᜀ-᜔\s]+', text)
    return [part.strip() for part in parts if part.strip() and re.search(r'[\u1700-\u171F]', part)]

def validate_baybayin_entry(baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
    try:
        romanizer = BaybayinRomanizer()
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        if not valid_parts:
            return False
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

@with_transaction(commit=True)
def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> None:
    """Process and store Baybayin data for a word."""
    if not baybayin_form:
        return
    try:
        romanizer = BaybayinRomanizer()
        if not validate_baybayin_entry(baybayin_form, romanized_form):
            logger.warning(f"Invalid Baybayin form for word_id {word_id}: {baybayin_form}")
            return
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

@with_transaction(commit=True)
def process_baybayin_entries(cur):
    """Process all Baybayin entries in the database."""
    logger.info("Processing Baybayin entries...")
    cur.execute("""
        SELECT id, lemma, language_code, normalized_lemma 
        FROM words 
        WHERE lemma ~ '[\u1700-\u171F]'
        ORDER BY id ASC
    """)
    baybayin_entries = cur.fetchall()
    conn = cur.connection
    for baybayin_id, baybayin_lemma, language_code, _ in baybayin_entries:
        try:
            cur.execute("BEGIN")
            parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_lemma)
            valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
            if not valid_parts:
                logger.warning(f"No valid Baybayin segments found for entry {baybayin_id}: {baybayin_lemma}")
                conn.commit()
                continue
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
            logger.info(f"Updating Baybayin entry (ID: {baybayin_id}) with cleaned form")
            cur.execute("""
                UPDATE words 
                SET romanized_form = %s,
                    baybayin_form = %s,
                    has_baybayin = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (romanized, cleaned_baybayin, baybayin_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing Baybayin entry {baybayin_id}: {str(e)}")
            continue

@with_transaction(commit=True)
def cleanup_baybayin_data(cur):
    """Clean up Baybayin data in the database."""
    conn = cur.connection
    try:
        cur.execute("BEGIN")
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form,
                '[^ᜀ-᜔\s]',
                '',
                'g'
            )
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
        """)
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form, 
                '\s+',
                ' ',
                'g'
            )
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
        """)
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE, baybayin_form = NULL
            WHERE has_baybayin = TRUE AND (baybayin_form IS NULL OR baybayin_form = '' OR baybayin_form !~ '[\u1700-\u171F]')
        """)
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE,
                baybayin_form = NULL
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
        """)
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
        cur.execute("""
            WITH DuplicateBaybayin AS (
                SELECT MIN(id) as keep_id,
                       language_code,
                       baybayin_form
                FROM words
                WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
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
        conn.commit()
        logger.info("Baybayin data cleanup completed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during Baybayin cleanup: {str(e)}")
        raise

@with_transaction(commit=False)
def check_baybayin_consistency(cur):
    """Check for consistency issues in Baybayin data."""
    issues = []
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
    cur.execute(r"""
        SELECT id, lemma, baybayin_form
        FROM words
        WHERE baybayin_form ~ '[^ᜀ-᜔\s]'
    """)
    invalid_chars = cur.fetchall()
    if invalid_chars:
        issues.append(f"Found {len(invalid_chars)} entries with invalid Baybayin characters")
        for word_id, lemma, baybayin in invalid_chars:
            logger.warning(f"Invalid Baybayin characters in word ID {word_id}: {lemma}")
    return issues

# -------------------------------------------------------------------
# Word Insertion and Update Functions
# -------------------------------------------------------------------
@with_transaction(commit=True)
def get_or_create_word_id(cur, lemma: str, language_code: str = "tl", **kwargs) -> int:
    """Get or create a word in the dictionary and return its ID."""
    if not lemma:
        raise ValueError("Lemma cannot be empty")

    normalized = normalize_lemma(lemma)
    search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {normalized}"))
    
    # Check if word already exists
    cur.execute("""
        SELECT id FROM words
        WHERE normalized_lemma = %s AND language_code = %s
    """, (normalized, language_code))
    
    result = cur.fetchone()
    if result:
        word_id = result[0]
        
        # Update any provided fields
        if kwargs:
            fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['root_word_id', 'has_baybayin', 'baybayin_form', 'romanized_form', 'tags', 'preferred_spelling']:
                    fields.append(f"{key} = %s")
                    values.append(value)
            
            if fields:
                values.append(word_id)
            cur.execute(f"""
            UPDATE words 
                    SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
                """, values)
                
        return word_id
    
    # Word doesn't exist, create it
    fields = ['lemma', 'normalized_lemma', 'language_code', 'search_text']
    values = [lemma, normalized, language_code, search_text]
    placeholders = ['%s', '%s', '%s', 'to_tsvector(\'simple\', %s)']
    
    # Add optional fields if provided
    for key, value in kwargs.items():
        if key in ['root_word_id', 'has_baybayin', 'baybayin_form', 'romanized_form', 'tags', 'preferred_spelling']:
            fields.append(key)
            values.append(value)
            placeholders.append('%s')
    
    # Create the word
    query = f"""
        INSERT INTO words ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
            RETURNING id
    """
    
    cur.execute(query, values)
    word_id = cur.fetchone()[0]
    return word_id

@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str, part_of_speech: str = "",
                      examples: str = None, usage_notes: str = None, category: str = None,
                      tags: str = None, sources: str = "") -> Optional[int]:
    """
    Inserts a definition for a given word.
    Checks for duplicate definitions (by word_id, definition_text, standardized_pos_id)
    to avoid violating the unique constraint.
    """
    try:
        # Skip definitions that are just Baybayin spelling notices.
        if 'Baybayin spelling of' in definition_text:
            return None

        # Verify the word exists.
        cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
        if not cur.fetchone():
            logger.error(f"Cannot insert definition – word ID {word_id} does not exist.")
            return None

        # Get standardized part-of-speech ID (assume your helper function is defined elsewhere)
        std_pos_id = get_standardized_pos_id(cur, part_of_speech)

        # Check for duplicate definitions.
        cur.execute("""
            SELECT id FROM definitions 
            WHERE word_id = %s AND definition_text = %s AND standardized_pos_id = %s
        """, (word_id, definition_text, std_pos_id))
        if cur.fetchone():
            return None

        # Optionally prepend category info to usage_notes.
        if category:
            usage_notes = f"[{category}] {usage_notes if usage_notes else ''}"

        cur.execute("""
            INSERT INTO definitions 
                 (word_id, definition_text, original_pos, standardized_pos_id, 
                  examples, usage_notes, tags, sources)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            word_id,
            definition_text,
            part_of_speech,
            std_pos_id,
            examples,
            usage_notes,
            tags,
            sources
        ))
        return cur.fetchone()[0]

    except UniqueViolation:
        logger.warning(f"Duplicate definition detected for word ID {word_id}: {definition_text[:50]}...")
        # No need for manual rollback here - the decorator will handle it
        return None
    except Exception as e:
        logger.error(f"Error in insert_definition for word ID {word_id}, definition: {definition_text[:50]}...: {e}")
        return None

@with_transaction(commit=True)
def insert_relation(cur, from_word_id: int, to_word_id: int, relation_type: str, sources: str = "", metadata: Dict = None):
    """
    Inserts a relation between two words.
    Does nothing if the same relation already exists.
    
    Args:
        cur: Database cursor
        from_word_id: ID of the source word
        to_word_id: ID of the target word
        relation_type: Type of relationship
        sources: Comma-separated list of data sources
        metadata: Optional dictionary of additional metadata to store (will be serialized as JSONB)
    """
    try:
        # Use the RelationshipManager to handle the relationship
        rel_manager = RelationshipManager(cur)
        return rel_manager.add_relationship(
            from_word_id=from_word_id,
            to_word_id=to_word_id,
            relationship_type=relation_type,
            sources=sources,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error in insert_relation from {from_word_id} to {to_word_id}: {e}")
        
        # Fall back to direct insertion if the RelationshipManager fails
        if from_word_id == to_word_id:
            return False
            
        sources = ", ".join(sorted(set(sources.split(", ")))) if sources else ""
        
        try:
            # If metadata is provided, include it in the insert
            if metadata:
                cur.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) 
                    DO UPDATE SET 
                        sources = CASE 
                            WHEN relations.sources IS NULL THEN EXCLUDED.sources
                            WHEN EXCLUDED.sources IS NULL THEN relations.sources
                            ELSE (
                                SELECT string_agg(DISTINCT unnest, ', ')
                                FROM unnest(string_to_array(relations.sources || ', ' || EXCLUDED.sources, ', '))
                            )
                        END,
                        metadata = COALESCE(relations.metadata, '{}') || EXCLUDED.metadata
                """, (from_word_id, to_word_id, relation_type, sources, json.dumps(metadata)))
            else:
                cur.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                """, (from_word_id, to_word_id, relation_type, sources))
            return True
        except Exception as inner_e:
            logger.error(f"Fallback insertion failed: {inner_e}")
            return False

@with_transaction(commit=True)
def insert_definition_relation(cur, definition_id: int, word_id: int, relation_type: str, sources: str = "auto"):
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
    normalized_components: str = None,
    etymology_structure: str = None,
    language_codes: str = None,
    sources: str = ""
) -> None:
    """Insert etymology data into the etymologies table."""
    if not word_id or not etymology_text:
        return

    try:
        cur.execute(
            """
            INSERT INTO etymologies (
                word_id, etymology_text, normalized_components, etymology_structure, language_codes, sources
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word_id, etymology_text)
            DO UPDATE SET
                normalized_components = COALESCE(etymologies.normalized_components, EXCLUDED.normalized_components),
                etymology_structure = COALESCE(etymologies.etymology_structure, EXCLUDED.etymology_structure),
                language_codes = COALESCE(etymologies.language_codes, EXCLUDED.language_codes),
                sources = array_to_string(ARRAY(
                    SELECT DISTINCT unnest(array_cat(
                        string_to_array(etymologies.sources, ', '),
                        string_to_array(EXCLUDED.sources, ', ')
                    ))
                ), ', ')
            """,
            (word_id, etymology_text, normalized_components, etymology_structure, language_codes, sources)
        )
    except Exception as e:
        logger.error(f"Error inserting etymology for word_id {word_id}: {str(e)}")

@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    sources: str
) -> None:
    """Insert an affixation relationship into the affixations table."""
    if not root_id or not affixed_id or root_id == affixed_id:
        return

    try:
        cur.execute(
            """
            INSERT INTO affixations (
                root_word_id, affixed_word_id, affix_type, sources
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (root_word_id, affixed_word_id, affix_type)
            DO UPDATE SET
                sources = array_to_string(ARRAY(
                    SELECT DISTINCT unnest(array_cat(
                        string_to_array(affixations.sources, ', '),
                        string_to_array(EXCLUDED.sources, ', ')
                    ))
                ), ', ')
            """,
            (root_id, affixed_id, affix_type, sources)
        )
    except Exception as e:
        logger.error(f"Error inserting affixation {root_id} -> {affixed_id}: {str(e)}")

@with_transaction(commit=True)
def batch_get_or_create_word_ids(cur, entries: List[Tuple[str, str]], batch_size: int = 1000) -> Dict[Tuple[str, str], int]:
    """
    Create or get IDs for multiple words in batches.
    
    Args:
        cur: Database cursor
        entries: List of (lemma, language_code) tuples
        batch_size: Number of entries to process in each batch
        
    Returns:
        Dictionary mapping (lemma, language_code) to word_id
    """
    result = {}
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch = list(dict.fromkeys(batch))  # Remove duplicates
        
        # First, let's ensure all existing words are fetched from the database
        normalized_entries = [(lemma, normalize_lemma(lemma), lang_code) for lemma, lang_code in batch]
        placeholders = []
        flat_params = []
        
        for _, norm, lang in normalized_entries:
            placeholders.append("(%s, %s)")
            flat_params.extend([norm, lang])
            
        if not placeholders:
            continue  # Empty batch, skip
            
        query = f"""
            SELECT lemma, language_code, id
            FROM words
            WHERE (normalized_lemma, language_code) IN ({', '.join(placeholders)})
        """
        
        try:
            cur.execute(query, flat_params)
            existing = {(lemma, lang): id for lemma, lang, id in cur.fetchall()}
        except Exception as e:
            logger.error(f"Error fetching existing words: {str(e)}")
            existing = {}
            
        # Identify entries that need to be inserted
        to_insert = []
        for lemma, lang in batch:
            if (lemma, lang) not in existing:
                norm = normalize_lemma(lemma)
                to_insert.append((lemma, norm, lang))
        
        # Insert missing entries with individual transactions
        for lemma, norm, lang in to_insert:
            try:
                search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {norm}"))
                cur.execute("""
                    INSERT INTO words (lemma, normalized_lemma, language_code, tags, search_text)
                    VALUES (%s, %s, %s, %s, to_tsvector('simple', %s))
                    ON CONFLICT ON CONSTRAINT words_lang_lemma_uniq
                    DO UPDATE SET 
                        lemma = EXCLUDED.lemma,
                        tags = EXCLUDED.tags,
                        search_text = to_tsvector('simple', EXCLUDED.lemma || ' ' || EXCLUDED.normalized_lemma),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (lemma, norm, lang, "", search_text))
                
                word_id = cur.fetchone()[0]
                
                # Verify the word was created
                cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
                if cur.fetchone():
                    existing[(lemma, lang)] = word_id
                
            except Exception as e:
                logger.error(f"Error processing entry {lemma}: {str(e)}")
                continue
        
        # Update result with all existing entries
        result.update(existing)
        
    return result

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------
def process_kwf_dictionary(cur, filename: str):
    """
    Process the KWF Dictionary JSON file and store its entries in the database with enhanced data extraction.
    
    The file is expected to be a dictionary where each key is a word and each value is a 
    dictionary with the following (example) structure:
    
    {
        "word_key": {
            "original": "word_original",
            "formatted": "word_formatted",
            "metadata": {
                "etymology": [...],
                "source_language": [...],
                "pronunciation": [...],
                "cross_references": []
            },
            "part_of_speech": ["Pangatnig"],
            "definitions": {
                "Pangatnig": [
                    {
                        "number": null,
                        "categories": ["General"],
                        "meaning": "Definition text...",
                        "sub_definitions": [],
                        "example_sets": [ ... ],
                        "note": null,
                        "see": null,
                        "cross_references": [],
                        "synonyms": [],
                        "synonyms_html": null,
                        "antonyms": [],
                        "antonyms_html": null
                    }
                ]
            },
            "affixation": [],
            "idioms": [],
            "related": { ... },
            "other_sections": {}
        },
        ...
    }
    
    This function uses the standardized source name "KWF Diksiyonaryo ng Wikang Filipino" 
    (via SourceStandardization) when inserting definitions.
    
    It captures rich data including:
    - Etymology and source language
    - Definition categories as tags
    - Synonyms and related terms as relations with metadata
    - Affixation relationships
    - Examples and usage notes
    - Cross-references and "see" references
    """
    standardized_source = SourceStandardization.standardize_sources('kwf_dictionary.json')
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If the loaded data is not a dictionary, assume it's a list of entries
    if not isinstance(data, dict):
        for entry in data:
            if not isinstance(entry, dict) or 'word' not in entry:
                continue
            word = entry['word']
            word_id = get_or_create_word_id(cur, word, language_code='tl')
            definitions = entry.get('definitions', {})
            if isinstance(definitions, dict):
                for pos, def_list in definitions.items():
                    for def_entry in def_list:
                        meaning = def_entry.get('meaning', '')
                        if meaning:
                            insert_definition(
                                cur,
                                word_id,
                                meaning,
                                part_of_speech=pos,
                                sources=standardized_source
                            )
        return  # Done processing the list format
    
    # Process the dictionary format (words as keys)
    for word, entry in data.items():
        if not isinstance(entry, dict):
            continue
            
        # Get both original and formatted versions
        original_word = entry.get('original', word)
        formatted_word = entry.get('formatted', original_word)
        
        # Prefer using the formatted word but fall back to original
        word_id = get_or_create_word_id(cur, formatted_word, language_code='tl')
        
        # Process etymology with structure
        if 'metadata' in entry:
            # Extract and process etymology
            if 'etymology' in entry['metadata'] and entry['metadata']['etymology']:
                etymology_text = ""
                etymology_structure = {
                    'raw_data': entry['metadata']['etymology'],
                    'source_language': entry['metadata'].get('source_language', []),
                    'pronunciation': entry['metadata'].get('pronunciation', [])
                }
                
                # Extract plain text from etymology
                for etym in entry['metadata']['etymology']:
                    if 'value' in etym:
                        if etymology_text:
                            etymology_text += ", "
                        etymology_text += etym['value']
                
                if etymology_text:
                    insert_etymology(
                        cur, 
                        word_id, 
                        etymology_text, 
                        etymology_structure=json.dumps(etymology_structure),
                        sources=standardized_source
                    )
                    
            # Store source language as part of etymology if no etymology text
            elif 'source_language' in entry['metadata'] and entry['metadata']['source_language'] and len(entry['metadata']['source_language']) > 0:
                source_langs = []
                for src_lang in entry['metadata']['source_language']:
                    if isinstance(src_lang, dict) and 'value' in src_lang:
                        source_langs.append(src_lang['value'])
                
                if source_langs:
                    source_text = "From " + ", ".join(source_langs)
                    etymology_structure = {
                        'raw_data': [],
                        'source_language': entry['metadata']['source_language'],
                        'pronunciation': entry['metadata'].get('pronunciation', [])
                    }
                    
                    insert_etymology(
                        cur, 
                        word_id, 
                        source_text, 
                        etymology_structure=json.dumps(etymology_structure),
                        sources=standardized_source
                    )
        
        # Process definitions by part of speech
        definitions = entry.get('definitions', {})
        if isinstance(definitions, dict):
            for pos, def_list in definitions.items():
                # Standardize part of speech
                std_pos = standardize_entry_pos(pos)
                
                if not isinstance(def_list, list):
                    continue
                
                for def_entry in def_list:
                    # Handle "see" references - create relations
                    if def_entry.get('see'):
                        for see_ref in def_entry.get('see', []):
                            if not isinstance(see_ref, dict) or 'term' not in see_ref:
                                continue
                                
                            related_term = see_ref.get('term')
                            if not related_term:
                                continue
                                
                            # Create the related word
                            related_id = get_or_create_word_id(cur, related_term, language_code='tl')
                            
                            # Add metadata for the relationship
                            metadata = {
                                'see_context': def_entry.get('see_context'),
                                'relationship_type': 'see_reference',
                                'link': see_ref.get('link', ''),
                                'broken': see_ref.get('broken', False)
                            }
                            
                            # Insert relation
                            relation_type = 'related'
                            insert_relation(cur, word_id, related_id, relation_type, sources=standardized_source, metadata=metadata)

def process_tagalog_words(cur, filename: str):
    """
    Process tagalog-words.json with the enhanced structure:
    {
        "word1": {
            "word": "word1",
            "pronunciation": "...",
            "alternate_pronunciation": "...",
            "part_of_speech": [["pos1"], ["pos2"]],
            "domains": ["domain1", "domain2"],
            "etymology": {
                "raw": "[ Lang ]",
                "languages": ["Lang"],
                "full_language_names": ["Language"],
                "other_terms": [],
                "terms": []
            },
            "derivative": "derivative1, derivative2",
            "senses": [
                {
                    "counter": "1",
                    "definition": "definition text",
                    "example": {
                        "raw": "example text",
                        "examples": ["example1", "example2"]
                    },
                    "synonyms": ["SYN1", "SYN2"],
                    "references": ["REF1", "REF2"],
                    "variants": ["var1", "var2"],
                    "etymology": {...},
                    "part_of_speech": ["pos"],
                    "affix_forms": ["form1", "form2"],
                    "affix_types": ["type1", "type2"]
                }
            ]
        },
        "word2": { ... }
    }
    """
    logger.info(f"Processing Tagalog words from: {filename}")
    source = SourceStandardization.standardize_sources('tagalog-words.json')
    language_code = 'tl'
    romanizer = BaybayinRomanizer()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Track statistics
        total_entries = len(data)
        processed_entries = 0
        skipped_entries = 0
        definitions_added = 0
        relations_added = 0
        synonyms_added = 0
        references_added = 0
        variants_added = 0
        etymologies_processed = 0
        errors = 0
        
        logger.info(f"Found {total_entries} entries to process")
        
        # Process each word entry
        for lemma, entry_data in tqdm(data.items(), desc="Processing Tagalog words"):
            try:
                # Ensure the entry has the basic required fields
                if not entry_data or not isinstance(entry_data, dict):
                    logger.warning(f"Skipping invalid entry for '{lemma}': not a dictionary")
                    skipped_entries += 1
                    continue
                
                # Create or get word ID
                word_id = get_or_create_word_id(cur, lemma, language_code=language_code)
                
                # Extract tags and domains
                tags = []
                
                # Add domains as tags if present
                if 'domains' in entry_data and entry_data['domains']:
                    tags.extend(entry_data['domains'])
                
                # Update word with tags if present
                if tags:
                    cur.execute("""
                        UPDATE words 
                        SET tags = %s
                        WHERE id = %s
                    """, (", ".join(tags), word_id))
                
                # Add pronunciation data if available
                pronunciation_data = {}
                if 'pronunciation' in entry_data and entry_data['pronunciation']:
                    pronunciation_data['primary'] = entry_data['pronunciation']
                if 'alternate_pronunciation' in entry_data and entry_data['alternate_pronunciation']:
                    pronunciation_data['alternate'] = entry_data['alternate_pronunciation']
                
                if pronunciation_data:
                    cur.execute("""
                        UPDATE words
                        SET pronunciation_data = %s
                        WHERE id = %s
                    """, (json.dumps(pronunciation_data), word_id))
                
                # Process derivative information
                if 'derivative' in entry_data and entry_data['derivative']:
                    derivative_text = entry_data['derivative']
                    # Store as metadata
                    cur.execute("""
                        UPDATE words
                        SET metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{derivative}',
                            %s::jsonb
                        )
                        WHERE id = %s
                    """, (json.dumps(derivative_text), word_id))
                    
                    # Process derivative forms if they exist
                    derivative_forms = derivative_text.split(',')
                    for form in derivative_forms:
                        form = form.strip()
                        if form:
                            # Check if it's a structured form with type
                            form_parts = form.split(' ')
                            if len(form_parts) > 1:
                                derivative_id = get_or_create_word_id(cur, form, language_code=language_code)
                                insert_relation(cur, word_id, derivative_id, 'derived', sources=source)
                
                # Process etymology at word level
                if 'etymology' in entry_data and entry_data['etymology'] and isinstance(entry_data['etymology'], dict):
                    etymology_data = entry_data['etymology']
                    etymology_text = etymology_data.get('raw', '')
                    
                    # Extract structured etymology data
                    etymology_structure = {
                        'languages': etymology_data.get('languages', []),
                        'full_language_names': etymology_data.get('full_language_names', []),
                        'other_terms': etymology_data.get('other_terms', []),
                        'terms': etymology_data.get('terms', [])
                    }
                    
                    if etymology_text:
                        language_codes = ", ".join(etymology_data.get('languages', []))
                        insert_etymology(
                            cur, 
                            word_id, 
                            etymology_text, 
                            etymology_structure=json.dumps(etymology_structure),
                            language_codes=language_codes,
                            sources=source
                        )
                        etymologies_processed += 1
                
                # Process senses (definitions)
                if 'senses' in entry_data and entry_data['senses']:
                    for sense in entry_data['senses']:
                        if not isinstance(sense, dict) or 'definition' not in sense:
                            continue
                        
                        definition_text = sense.get('definition', '').strip()
                        if not definition_text:
                            continue
                        
                        # Add counter information if present
                        counter = sense.get('counter', '')
                        if counter:
                            definition_text = f"[{counter}] {definition_text}"
                        
                        # Extract part of speech for this specific sense
                        pos = ''
                        if 'part_of_speech' in sense and sense['part_of_speech']:
                            pos = ", ".join(sense['part_of_speech'])
                        # Fall back to entry-level part of speech if not defined at sense level
                        elif 'part_of_speech' in entry_data:
                            pos_list = entry_data['part_of_speech']
                            if pos_list and isinstance(pos_list, list):
                                # Flatten potentially nested arrays
                                flat_pos = []
                                for pos_item in pos_list:
                                    if isinstance(pos_item, list):
                                        flat_pos.extend(pos_item)
                                    else:
                                        flat_pos.append(pos_item)
                                pos = ", ".join(flat_pos)
                        
                        # Extract examples if present
                        examples = []
                        example_data = sense.get('example', {})
                        if isinstance(example_data, dict):
                            # Try to get pre-parsed examples
                            if 'examples' in example_data and example_data['examples']:
                                examples.extend(example_data['examples'])
                            # If no pre-parsed examples, use the raw text
                            elif 'raw' in example_data and example_data['raw']:
                                examples.append(example_data['raw'])
                        
                        # Combine examples into a single string
                        examples_text = "; ".join(examples) if examples else None
                        
                        # Extract usage notes if present
                        usage_notes = None
                        if 'usage_notes' in sense and sense['usage_notes']:
                            usage_notes = sense['usage_notes']
                        
                        # Extract category if present
                        category = None
                        if 'category' in sense and sense['category']:
                            category = sense['category']
                        
                        # Get additional tags specific to this sense
                        sense_tags = []
                        if 'tags' in sense and sense['tags']:
                            sense_tags.extend(sense['tags'])
                            
                        # Add counter as tag if present
                        if counter:
                            sense_tags.append(f"counter:{counter}")
                            
                        # Add etymology to tags if present at sense level
                        if 'etymology' in sense and sense['etymology']:
                            sense_etymology = sense['etymology']
                            if isinstance(sense_etymology, dict) and 'raw' in sense_etymology:
                                sense_tags.append(f"etymology:{sense_etymology['raw']}")
                                
                                # Process specific sense etymology
                                etymology_structure = {
                                    'languages': sense_etymology.get('languages', []),
                                    'full_language_names': sense_etymology.get('full_language_names', []),
                                    'other_terms': sense_etymology.get('other_terms', []),
                                    'terms': sense_etymology.get('terms', [])
                                }
                                
                                sense_etymology_text = sense_etymology.get('raw', '')
                                if sense_etymology_text:
                                    language_codes = ", ".join(sense_etymology.get('languages', []))
                                    # Store as metadata
                                    sense_tags.append(f"etymology_languages:{language_codes}")
                        
                        # Combine tags into a single string
                        sense_tags_text = ", ".join(sense_tags) if sense_tags else None
                        
                        # Insert definition
                        definition_id = insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            part_of_speech=pos,
                            examples=examples_text,
                            usage_notes=usage_notes,
                            category=category,
                            tags=sense_tags_text,
                            sources=source
                        )
                        
                        if definition_id:
                            definitions_added += 1
                            
                            # Process synonyms if present
                            if 'synonyms' in sense and sense['synonyms']:
                                for synonym in sense['synonyms']:
                                    if synonym:
                                        # Cleanup synonym text - it could be in all caps or have special formatting
                                        syn_text = synonym.strip()
                                        
                                        # Create the synonym word
                                        syn_id = get_or_create_word_id(cur, syn_text, language_code=language_code)
                                        
                                        # Create both directions of the relationship
                                        insert_relation(cur, word_id, syn_id, 'synonym', sources=source)
                                        insert_relation(cur, syn_id, word_id, 'synonym', sources=source)
                                        
                                        # Also link the definition to the synonym
                                        insert_definition_relation(cur, definition_id, syn_id, 'synonym', sources=source)
                                        
                                        synonyms_added += 1
                            
                            # Process references if present
                            if 'references' in sense and sense['references']:
                                for reference in sense['references']:
                                    if reference:
                                        ref_text = reference.strip()
                                        
                                        # Create the reference word
                                        ref_id = get_or_create_word_id(cur, ref_text, language_code=language_code)
                                        
                                        # Create reference relationship (this is a looser connection than synonym)
                                        insert_relation(cur, word_id, ref_id, 'related', sources=source)
                                        
                                        # Also link the definition to the reference
                                        insert_definition_relation(cur, definition_id, ref_id, 'related', sources=source)
                                        
                                        references_added += 1
                            
                            # Process variants if present
                            if 'variants' in sense and sense['variants']:
                                for variant in sense['variants']:
                                    if variant:
                                        # Clean up variant text - it could include additional information
                                        # like "variant1 Cf variant2" where Cf means "compare with"
                                        variant_parts = variant.split(' Cf ')
                                        variant_text = variant_parts[0].strip()
                                        
                                        # Create the variant word
                                        variant_id = get_or_create_word_id(cur, variant_text, language_code=language_code)
                                        
                                        # Create bidirectional variant relationship
                                        insert_relation(cur, word_id, variant_id, 'variant', sources=source)
                                        insert_relation(cur, variant_id, word_id, 'variant', sources=source)
                                        
                                        # Also link the definition to the variant
                                        insert_definition_relation(cur, definition_id, variant_id, 'variant', sources=source)
                                        
                                        variants_added += 1
                                        
                                        # If there's a "Cf" part, also create a relationship to that word
                                        if len(variant_parts) > 1:
                                            cf_text = variant_parts[1].strip()
                                            cf_id = get_or_create_word_id(cur, cf_text, language_code=language_code)
                                            
                                            # Create a "compare with" relationship
                                            insert_relation(cur, word_id, cf_id, 'compare_with', sources=source)
                                            
                                            # Also link the definition to the comparison
                                            insert_definition_relation(cur, definition_id, cf_id, 'compare_with', sources=source)
                                            
                                            references_added += 1
                            
                            # Process affix forms if present
                            if ('affix_forms' in sense and sense['affix_forms'] and 
                                'affix_types' in sense and sense['affix_types']):
                                
                                forms = sense['affix_forms']
                                types = sense['affix_types']
                                
                                for i, form in enumerate(forms):
                                    if i < len(types):
                                        affix_type = types[i]
                                    else:
                                        affix_type = 'unknown'
                                    
                                    # Normalize form (remove potential dots)
                                    clean_form = form.replace('.', '')
                                    
                                    # Create the affixed word
                                    affixed_id = get_or_create_word_id(cur, clean_form, language_code=language_code)
                                    
                                    # Create affixation relationship
                                    insert_affixation(
                                        cur,
                                        root_id=word_id,
                                        affixed_id=affixed_id,
                                        affix_type=affix_type,
                                        sources=source
                                    )
                                    
                                    # Also create a relation
                                    # Define metadata for the relationship
                                    metadata = {"affix_type": affix_type}
                                    insert_relation(
                                        cur, 
                                        word_id, 
                                        affixed_id, 
                                        'derived', 
                                        sources=source,
                                        metadata=metadata
                                    )
                
                processed_entries += 1
                if processed_entries % 1000 == 0:
                    logger.info(f"Processed {processed_entries}/{total_entries} entries")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error processing entry '{lemma}': {str(e)}")
                continue
                
        # Log statistics
        logger.info(f"Tagalog words processing complete:")
        logger.info(f"  Total entries: {total_entries}")
        logger.info(f"  Processed: {processed_entries}")
        logger.info(f"  Skipped: {skipped_entries}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Definitions added: {definitions_added}")
        logger.info(f"  Synonyms added: {synonyms_added}")
        logger.info(f"  References added: {references_added}")
        logger.info(f"  Variants added: {variants_added}")
        logger.info(f"  Relations added: {relations_added}")
        logger.info(f"  Etymologies processed: {etymologies_processed}")
        
    except Exception as e:
        logger.error(f"Error processing Tagalog words file: {str(e)}")
        raise

def process_root_words_cleaned(cur, filename: str):
    """Process root words data from a cleaned JSON file."""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = 0
    skipped = 0
    errors = 0
    source = get_standardized_source(os.path.basename(filename))
    total = len(data)
    
    for root_word, associated_words in tqdm(data.items(), desc="Processing root words"):
        try:
            # Skip empty root word
            if not root_word:
                skipped += 1
                continue
            
            # Get the root word ID
            root_id = get_or_create_word_id(cur, root_word, 'tl')
            
            # Process associated words (which are the nested dictionaries)
            for assoc_word, word_data in associated_words.items():
                if not assoc_word or assoc_word == root_word:
                    continue
                
                # Create the associated word
                word_type = word_data.get('type', '')
                definition = word_data.get('definition', '')
                
                # Remove the ellipsis from the end of the definition
                if definition.endswith('...'):
                    definition = definition[:-3]
                
                standardized_pos = standardize_entry_pos(word_type)
                
                # Create the word entry
                derived_id = get_or_create_word_id(cur, assoc_word, 'tl', root_word_id=root_id)
                
                # Add definition if available
                if definition:
                    insert_definition(
                        cur, derived_id, definition, part_of_speech=standardized_pos,
                        sources=source
                    )
                
                # Add the relationship if it's not the root word itself
                if assoc_word != root_word:
                    insert_relation(
                        cur, derived_id, root_id, 'derived_from', source
                    )
            
            # Also add the root word's own definition if it exists in the associated words
            if root_word in associated_words:
                root_data = associated_words[root_word]
                root_type = root_data.get('type', '')
                root_definition = root_data.get('definition', '')
                
                # Remove the ellipsis from the end of the root definition
                if root_definition.endswith('...'):
                    root_definition = root_definition[:-3]
                
                standardized_pos = standardize_entry_pos(root_type)
                
                if root_definition:
                    insert_definition(
                        cur, root_id, root_definition, part_of_speech=standardized_pos,
                        sources=source
                    )
            
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{total} root word entries")
                
        except Exception as e:
            errors += 1
            logger.error(f"Error processing root word {root_word}: {str(e)}")
    
    logger.info(f"Root words processing complete: {processed} processed, {skipped} skipped, {errors} errors")
    
def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    lang_map = {
        "Esp": "es", "Eng": "en", "Ch": "zh", "Tsino": "zh", "Jap": "ja",
        "San": "sa", "Sanskrit": "sa", "Tag": "tl", "Mal": "ms", "Arb": "ar"
    }
    return [lang_map[lang] for lang in lang_map if lang in etymology]

@with_transaction(commit=True)
def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Process and create relationships from definition text."""
    synonym_patterns = [r'ka(singkahulugan|tulad|patulad|tumbas) ng\s+(\w+)', r'singkahulugan:\s+(\w+)']
    antonym_patterns = [r'(kasalungat|kabaligtaran) ng\s+(\w+)', r'kabaligtaran:\s+(\w+)']
    
    for pattern in synonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            syn = match.group(2).strip()
            syn_id = get_or_create_word_id(cur, syn, 'tl')
            insert_relation(cur, word_id, syn_id, "synonym", sources=source)
    
    for pattern in antonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            ant = match.group(2).strip()
            ant_id = get_or_create_word_id(cur, ant, 'tl')
            insert_relation(cur, word_id, ant_id, "antonym", sources=source)

# ---------------------------------------------------------------a----
# Robust Relationship Mapping for Kaikki
# -------------------------------------------------------------------
RELATION_MAPPING = {
    "synonym": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "antonym": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "hyponym_of": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of"
    },
    "hypernym_of": {
        "relation_type": "hypernym_of",
        "bidirectional": False,
        "inverse": "hyponym_of"
    },
    "see_also": {
        "relation_type": "see_also",
        "bidirectional": True,
        "inverse": "see_also"
    },
    "compare_with": {
        "relation_type": "compare_with",
        "bidirectional": True,
        "inverse": "compare_with"
    },
    "derived_from": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of"
    },
    "descendant": {
        "relation_type": "descendant_of",
        "bidirectional": False,
        "inverse": "ancestor_of"
    },
    "borrowed": {
        "relation_type": "borrowed_from",
        "bidirectional": False,
        "inverse": "loaned_to"
    },
    "variant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "alt_of": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "abbreviation_of": {
        "relation_type": "abbreviation_of",
        "bidirectional": False,
        "inverse": "has_abbreviation"
    },
    "initialism_of": {
        "relation_type": "initialism_of",
        "bidirectional": False,
        "inverse": "has_initialism"
    },
    "related": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related"
    },
    "derived": {
        "relation_type": "derived",
        "bidirectional": False,
        "inverse": "derives"
    },
    "contraction_of": {
        "relation_type": "contraction_of",
        "bidirectional": False,
        "inverse": "contracts_to"
    },
    "alternate_form": {
        "relation_type": "alternate_form",
        "bidirectional": True,
        "inverse": "alternate_form"
    },
    "regional_form": {
        "relation_type": "regional_form",
        "bidirectional": True,
        "inverse": "regional_form"
    },
    "modern_form": {
        "relation_type": "modern_form",
        "bidirectional": False,
        "inverse": "archaic_form"
    },
    "archaic_form": {
        "relation_type": "archaic_form",
        "bidirectional": False,
        "inverse": "modern_form"
    },
    "obsolete_spelling": {
        "relation_type": "obsolete_spelling",
        "bidirectional": False,
        "inverse": "current_spelling"
    },
    "alternative_spelling": {
        "relation_type": "alternative_spelling",
        "bidirectional": True,
        "inverse": "alternative_spelling"
    },
    "root_of": {
        "relation_type": "root_of",
        "bidirectional": False,
        "inverse": "derived_from"
    },
    "cognate": {
        "relation_type": "cognate",
        "bidirectional": True,
        "inverse": "cognate" 
    },
    # Filipino-specific relationship types
    "kasingkahulugan": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "kasalungat": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "kabaligtaran": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "katulad": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "uri_ng": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of"
    },
    "mula_sa": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of"
    },
    "varyant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "kaugnay": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related"
    }
}

def normalize_relation_type(raw_key: str) -> Tuple[str, bool, Optional[str]]:
    """
    Normalize a relationship type key and return its standard form, whether it's bidirectional,
    and its inverse relationship type.
    
    Args:
        raw_key: The raw relationship type key to normalize.
        
    Returns:
        Tuple containing:
        - Normalized relationship type string
        - Boolean indicating if relationship is bidirectional
        - Optional inverse relationship type (None if no inverse defined)
    """
    if not raw_key:
        return "related", True, "related"  # Default to a generic related relationship
        
    key_lower = raw_key.lower().strip()
    
    # Remove trailing 's' if plural form
    if key_lower.endswith("s") and len(key_lower) > 1:
        key_lower = key_lower[:-1]  # E.g. "synonyms" -> "synonym"
    
    # Special case handling for Tagalog relationship indicators
    if "kasingkahulugan" in key_lower or "katulad ng" in key_lower:
        return "synonym", True, "synonym"
    if "kasalungat" in key_lower or "kabaligtaran" in key_lower:
        return "antonym", True, "antonym" 
    if "uri ng" in key_lower:
        return "hyponym_of", False, "hypernym_of"
    if "mula sa" in key_lower:
        return "derived_from", False, "root_of"
    if "varyant" in key_lower or "variant" in key_lower:
        return "variant", True, "variant"
    
    # Check direct match in mapping
    if key_lower in RELATION_MAPPING:
        info = RELATION_MAPPING[key_lower]
        return info["relation_type"], info.get("bidirectional", False), info.get("inverse")
    
    # Try to find partial matches in keys
    for known_key, info in RELATION_MAPPING.items():
        if known_key in key_lower or key_lower in known_key:
            return info["relation_type"], info.get("bidirectional", False), info.get("inverse")
    
    # If no match found, default to a related relationship but preserve the original key
    logger.info(f"Unknown relationship type: {raw_key}, defaulting to 'related'")
    return "related", True, "related"

@with_transaction(commit=True)
def process_relationships(cur, word_id, data, sources):
    """Process relationship data for a word."""
    if not data or not word_id:
        return
    
    try:
        # Extract relationships from definitions
        definitions = []
        if 'definitions' in data and isinstance(data['definitions'], list):
            definitions = [d.get('text') if isinstance(d, dict) else d for d in data['definitions']]
        elif 'definition' in data and data['definition']:
            definitions = [data['definition']]
        
        # Process each definition for relationships
        for definition in definitions:
            if not definition or not isinstance(definition, str):
                continue
                
            # Look for synonyms in definitions
            syn_patterns = [
                r'ka(singkahulugan|tulad) ng\s+(\w+)',               # Tagalog
                r'syn(onym)?[\.:]?\s+(\w+)',                         # English
                r'same\s+as\s+(\w+)',                                # English
                r'another\s+term\s+for\s+(\w+)',                     # English
                r'another\s+word\s+for\s+(\w+)',                     # English
                r'also\s+called\s+(\w+)',                            # English
                r'also\s+known\s+as\s+(\w+)',                        # English
            ]
            
            for pattern in syn_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    synonym = match[1] if isinstance(match, tuple) and len(match) > 1 else match
                    if synonym and isinstance(synonym, str):
                        # Get or create the synonym word
                        lang_code = data.get('language_code', 'tl')
                        syn_id = get_or_create_word_id(cur, synonym.strip(), language_code=lang_code)
                        
                        # Insert the synonym relationship
                        insert_relation(cur, word_id, syn_id, "synonym", sources=sources)
                        
                        # For bidirectional synonyms
                        insert_relation(cur, syn_id, word_id, "synonym", sources=sources)
            
            # Look for antonyms in definitions
            ant_patterns = [
                r'(kasalungat|kabaligtaran) ng\s+(\w+)',             # Tagalog
                r'ant(onym)?[\.:]?\s+(\w+)',                         # English
                r'opposite\s+of\s+(\w+)',                            # English
                r'contrary\s+to\s+(\w+)',                            # English
            ]
            
            for pattern in ant_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    antonym = match[1] if isinstance(match, tuple) and len(match) > 1 else match
                    if antonym and isinstance(antonym, str):
                        # Get or create the antonym word
                        lang_code = data.get('language_code', 'tl')
                        ant_id = get_or_create_word_id(cur, antonym.strip(), language_code=lang_code)
                        
                        # Insert the antonym relationship
                        insert_relation(cur, word_id, ant_id, "antonym", sources=sources)
                        
                        # For bidirectional antonyms
                        insert_relation(cur, ant_id, word_id, "antonym", sources=sources)
            
            # Look for hypernyms (broader terms) in definitions
            hyper_patterns = [
                r'uri ng\s+(\w+)',                                  # Tagalog
                r'type of\s+(\w+)',                                 # English 
                r'kind of\s+(\w+)',                                 # English
                r'form of\s+(\w+)',                                 # English
                r'variety of\s+(\w+)',                              # English
                r'species of\s+(\w+)',                              # English
                r'member of\s+the\s+(\w+)\s+family',                # English
            ]
            
            for pattern in hyper_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    hypernym = match
                    if hypernym and isinstance(hypernym, str):
                        # Get or create the hypernym word
                        lang_code = data.get('language_code', 'tl')
                        hyper_id = get_or_create_word_id(cur, hypernym.strip(), language_code=lang_code)
                        
                        # Insert the hypernym relationship
                        insert_relation(cur, word_id, hyper_id, "hyponym_of", sources=sources)
                        
                        # Add the inverse relationship
                        insert_relation(cur, hyper_id, word_id, "hypernym_of", sources=sources)
            
            # Look for variations in definitions
            var_patterns = [
                r'(iba\'t ibang|ibang) (anyo|baybay|pagsulat|bigkas) ng\s+(\w+)',  # Tagalog: different form/spelling/pronunciation of
                r'(alternatibo|alternativ|kahalili) ng\s+(\w+)',                    # Tagalog: alternative of
                r'(variant|variation|alt(ernative)?) (form|spelling) of\s+(\w+)',   # English
                r'alternative (to|for)\s+(\w+)',                                    # English
                r'also (written|spelled) as\s+(\w+)',                               # English
                r'(var\.|variant)\s+(\w+)',                                         # English abbreviated
                r'(regional|dialectal) form of\s+(\w+)',                            # English regional variant
                r'(slang|informal) for\s+(\w+)',                                    # English slang variant
                r'commonly (misspelled|written) as\s+(\w+)',                        # English common misspelling
                r'(baryant|lokal na anyo) ng\s+(\w+)',                              # Tagalog regional variant
            ]
            
            for pattern in var_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    # Different patterns have target word in different positions
                    variant = None
                    if len(match) == 3 and isinstance(match, tuple):  # For patterns with 3 capture groups
                        variant = match[2]
                    elif len(match) == 2 and isinstance(match, tuple):  # For patterns with 2 capture groups
                        variant = match[1]
                    elif isinstance(match, str):  # For patterns with 1 capture group
                        variant = match
                        
                    if variant and isinstance(variant, str):
                        # Get or create the variant word
                        lang_code = data.get('language_code', 'tl')
                        var_id = get_or_create_word_id(cur, variant.strip(), language_code=lang_code)
                        
                        # Insert the variant relationship
                        insert_relation(cur, word_id, var_id, "variant", sources=sources)
                        
                        # For bidirectional variant relationship
                        insert_relation(cur, var_id, word_id, "variant", sources=sources)
        
        # Process derivative information
        derivative = data.get('derivative', '')
        if derivative and isinstance(derivative, str):
            # This indicates the word is derived from another root
            mula_sa_patterns = [
                r'mula sa\s+(.+?)(?:\s+na|\s*$)',                   # Tagalog
                r'derived from\s+(?:the\s+)?(\w+)',                 # English
                r'comes from\s+(?:the\s+)?(\w+)',                   # English
                r'root word(?:\s+is)?\s+(\w+)',                     # English
            ]
            
            for pattern in mula_sa_patterns:
                root_match = re.search(pattern, derivative, re.IGNORECASE)
                if root_match:
                    root_word = root_match.group(1).strip()
                    if root_word:
                        # Get or create the root word
                        lang_code = data.get('language_code', 'tl')
                        root_id = get_or_create_word_id(cur, root_word, language_code=lang_code)
                        
                        # Insert the derived_from relationship
                        insert_relation(cur, word_id, root_id, "derived_from", sources=sources)
                        
                        # Add the inverse relationship
                        insert_relation(cur, root_id, word_id, "root_of", sources=sources)
        
        # Process etymology information for potential language relationships
        etymology = data.get('etymology', '')
        if etymology and isinstance(etymology, str):
            # Try to extract language information from etymology
            lang_patterns = {
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Spanish|Esp)[\.:]?\s+(\w+)': 'es',  # Spanish
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:English|Eng)[\.:]?\s+(\w+)': 'en',  # English
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Chinese|Ch|Tsino)[\.:]?\s+(\w+)': 'zh',  # Chinese
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Japanese|Jap)[\.:]?\s+(\w+)': 'ja',  # Japanese
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Sanskrit|San)[\.:]?\s+(\w+)': 'sa',  # Sanskrit
            }
            
            for pattern, lang_code in lang_patterns.items():
                lang_matches = re.findall(pattern, etymology, re.IGNORECASE)
                for lang_word in lang_matches:
                    if lang_word and isinstance(lang_word, str):
                        # Get or create the foreign word
                        foreign_id = get_or_create_word_id(cur, lang_word.strip(), language_code=lang_code)
                        
                        # Insert the etymology relationship
                        insert_relation(cur, word_id, foreign_id, "borrowed_from", sources=sources)
        
        # Process alternate forms and variations
        # Check if there's variations data in metadata
        variations = data.get('variations', [])
        if variations and isinstance(variations, list):
            for variant in variations:
                if isinstance(variant, str) and variant.strip():
                    # Add this explicit variation
                    var_id = get_or_create_word_id(cur, variant.strip(), language_code=data.get('language_code', 'tl'))
                    insert_relation(cur, word_id, var_id, "variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "variant", sources=sources)
                elif isinstance(variant, dict) and 'form' in variant:
                    var_form = variant.get('form', '').strip()
                    if var_form:
                        var_type = variant.get('type', 'variant')
                        var_id = get_or_create_word_id(cur, var_form, language_code=data.get('language_code', 'tl'))
                        
                        # Use specific relationship type if provided, otherwise default to "variant"
                        rel_type = var_type if var_type in ["abbreviation", "misspelling", "regional", "alternate", "dialectal"] else "variant"
                        insert_relation(cur, word_id, var_id, rel_type, sources=sources)
                        insert_relation(cur, var_id, word_id, rel_type, sources=sources)
        
        # Look for variations by checking spelling differences
        # This will detect common spelling variations in Filipino like f/p, e/i, o/u substitutions
        word = data.get('word', '')
        if word and isinstance(word, str) and len(word) > 3:
            # Common letter substitutions in Filipino
            substitutions = [
                ('f', 'p'), ('p', 'f'),  # Filipino/Pilipino
                ('e', 'i'), ('i', 'e'),  # like in leeg/liig (neck)
                ('o', 'u'), ('u', 'o'),  # like in puso/poso (heart)
                ('k', 'c'), ('c', 'k'),  # like in karera/carera (race)
                ('w', 'u'), ('u', 'w'),  # like in uwi/uwi (go home)
                ('j', 'h'), ('h', 'j'),  # like in jahit/hahit
                ('s', 'z'), ('z', 's'),  # like in kasoy/kazoy
                ('ts', 'ch'), ('ch', 'ts'),  # like in tsaa/chaa (tea)
            ]
            
            # Generate possible variations
            potential_variations = []
            for i, char in enumerate(word):
                for orig, repl in substitutions:
                    if char.lower() == orig:
                        var = word[:i] + repl + word[i+1:]
                        potential_variations.append(var)
                    elif char.lower() == orig[0] and i < len(word) - 1 and word[i+1].lower() == orig[1]:
                        var = word[:i] + repl + word[i+2:]
                        potential_variations.append(var)
            
            # Check if these variations actually exist in the database
            for var in potential_variations:
                # Skip if the variation is the same as the original word
                if var.lower() == word.lower():
                    continue
                    
                cur.execute("""
                    SELECT id FROM words 
                    WHERE normalized_lemma = %s AND language_code = %s
                """, (normalize_lemma(var), data.get('language_code', 'tl')))
                
                result = cur.fetchone()
                if result:
                    # Found a real variation in the database
                    var_id = result[0]
                    insert_relation(cur, word_id, var_id, "spelling_variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "spelling_variant", sources=sources)
    
    except Exception as e:
        logger.error(f"Error processing relationships for word_id {word_id}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            tb_str = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"Traceback: {tb_str}")

@with_transaction(commit=True)
def process_direct_relations(cur, word_id, entry, lang_code, source):
    """Process direct relationships specified in the entry."""
    relationship_mappings = {
        'synonyms': ('synonym', True),  # bidirectional
        'antonyms': ('antonym', True),  # bidirectional
        'derived': ('derived_from', False),  # not bidirectional, direction is important
        'related': ('related', True)    # bidirectional
    }
    
    for rel_key, (rel_type, bidirectional) in relationship_mappings.items():
        if rel_key in entry and isinstance(entry[rel_key], list):
            for rel_item in entry[rel_key]:
                # Initialize metadata
                metadata = {}
                
                # Handle both string words and dictionary objects with word property
                if isinstance(rel_item, dict) and 'word' in rel_item:
                    rel_word = rel_item['word']
                    
                    # Extract metadata fields if available
                    if 'strength' in rel_item:
                        metadata['strength'] = rel_item['strength']
                    
                    if 'tags' in rel_item and rel_item['tags']:
                        metadata['tags'] = rel_item['tags']
                    
                    if 'english' in rel_item and rel_item['english']:
                        metadata['english'] = rel_item['english']
                        
                    # Extract any other useful fields
                    for field in ['sense', 'extra', 'notes']:
                        if field in rel_item and rel_item[field]:
                            metadata[field] = rel_item[field]
                            
                elif isinstance(rel_item, str):
                    rel_word = rel_item
                else:
                    continue
                        
                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue
                        
                # For derived words, the entry word is derived from the related word
                if rel_type == 'derived_from':
                    from_id = word_id
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source)
                else:
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source)
                        
                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(cur, to_id, word_id, rel_type, sources=source, metadata=metadata)
                        else:
                            insert_relation(cur, to_id, word_id, rel_type, sources=source)

@with_transaction(commit=True)
def process_relations(cur, from_word_id: int, relations_dict: Dict[str, List[str]], lang_code: str, source: str):
    """
    Process relationship data from dictionary.
    Convert each raw Kaikki relation key using normalize_relation_type,
    then create appropriate relations in the database.
    """
    for raw_key, related_list in relations_dict.items():
        if not related_list:
            continue
        # Normalize relation key
        relation_type, bidirectional, inverse_type = normalize_relation_type(raw_key)
        
        for rel_item in related_list:
            metadata = {}
            
            # Handle both string values and dictionary objects
            if isinstance(rel_item, dict) and 'word' in rel_item:
                rel_word_lemma = rel_item['word']
                
                # Extract metadata fields if available
                if 'strength' in rel_item:
                    metadata['strength'] = rel_item['strength']
                
                if 'tags' in rel_item and rel_item['tags']:
                    metadata['tags'] = rel_item['tags']
                
                if 'english' in rel_item and rel_item['english']:
                    metadata['english'] = rel_item['english']
                    
                # Extract any other useful fields
                for field in ['sense', 'extra', 'notes', 'context']:
                    if field in rel_item and rel_item[field]:
                        metadata[field] = rel_item[field]
            elif isinstance(rel_item, str):
                rel_word_lemma = rel_item
            else:
                continue
                
            to_word_id = get_or_create_word_id(cur, rel_word_lemma, language_code=lang_code)
            
            # Only include metadata if it's not empty
            if metadata:
                insert_relation(cur, from_word_id, to_word_id, relation_type, sources=source, metadata=metadata)
            else:
                insert_relation(cur, from_word_id, to_word_id, relation_type, sources=source)

            if bidirectional and inverse_type:
                # For bidirectional relationships, we might want to copy the metadata
                if metadata:
                    insert_relation(cur, to_word_id, from_word_id, inverse_type, sources=source, metadata=metadata)
                else:
                    insert_relation(cur, to_word_id, from_word_id, inverse_type, sources=source)

@with_transaction(commit=True)
def extract_sense_relations(cur, word_id, sense, lang_code, source):
    """Extract and process relationship data from a word sense."""
    for rel_type in ['synonyms', 'antonyms', 'derived', 'related']:
        if rel_type in sense and isinstance(sense[rel_type], list):
            relation_items = sense[rel_type]
            relationship_type = 'synonym' if rel_type == 'synonyms' else 'antonym' if rel_type == 'antonyms' else 'derived_from' if rel_type == 'derived' else 'related'
            bidirectional = rel_type != 'derived'  # derived relationships are not bidirectional
            
            for item in relation_items:
                # Initialize metadata
                metadata = {}
                
                # Handle both string words and dictionary objects with word property
                if isinstance(item, dict) and 'word' in item:
                    rel_word = item['word']
                    
                    # Extract metadata fields if available
                    if 'strength' in item:
                        metadata['strength'] = item['strength']
                    
                    if 'tags' in item and item['tags']:
                        metadata['tags'] = item['tags']
                    
                    if 'english' in item and item['english']:
                        metadata['english'] = item['english']
                        
                    # Extract sense-specific context if available
                    if 'sense' in item and item['sense']:
                        metadata['sense'] = item['sense']
                        
                    # Extract any other useful fields
                    for field in ['extra', 'notes', 'context']:
                        if field in item and item[field]:
                            metadata[field] = item[field]
                elif isinstance(item, str):
                    rel_word = item
                else:
                    continue
                        
                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue
                        
                # For derived words, the entry word is derived from the related word
                if relationship_type == 'derived_from':
                    from_id = word_id
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, from_id, to_id, relationship_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, from_id, to_id, relationship_type, sources=source)
                else:
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, word_id, to_id, relationship_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, word_id, to_id, relationship_type, sources=source)
                        
                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(cur, to_id, word_id, relationship_type, sources=source, metadata=metadata)
                        else:
                            insert_relation(cur, to_id, word_id, relationship_type, sources=source)
    
def extract_definition_tags(definition_dict):
    """Extract tags from definition data"""
    tags = []
    
    # Extract from directly tagged definitions
    if 'tags' in definition_dict and isinstance(definition_dict['tags'], list):
        tags.extend(definition_dict['tags'])
        
    # Look for tags in glosses field that might indicate usage
    special_tags = ['figuratively', 'figurative', 'colloquial', 'formal', 'informal', 'archaic', 'obsolete', 'rare']
    glosses = definition_dict.get('glosses', [])
    if isinstance(glosses, list) and glosses:
        for gloss in glosses:
            if isinstance(gloss, str):
                # Check if the gloss starts with a tag in parentheses
                tag_match = re.match(r'^\(([^)]+)\)', gloss)
                if tag_match:
                    potential_tag = tag_match.group(1).lower()
                    if potential_tag in special_tags:
                        tags.append(potential_tag)
                        
    return tags if tags else None

def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract Baybayin form and romanized form from an entry."""
    baybayin_form = entry.get('baybayin_form')
    romanized_form = entry.get('romanized_form')
    
    # Return early if no Baybayin form
    if not baybayin_form:
        return None, None
        
    # Validate the Baybayin form
    if validate_baybayin_entry(baybayin_form, romanized_form):
        return baybayin_form, romanized_form
    return None, None

def process_entry(cur, entry: Dict, filename=None):
    """
    Process a single Kaikki entry, with definitions/senses, etymology, relations, etc.
    """
    try:
        word = entry.get("word")
        if not word:
            return
        language_code = entry.get("language_code") or entry.get("lang_code", "tl")

        word_id = get_or_create_word_id(cur, word, language_code=language_code)

        # Extract any Baybayin form
        baybayin_form, romanized_form = extract_baybayin_info(entry)
        if baybayin_form:
            process_baybayin_data(cur, word_id, baybayin_form, romanized_form)

        # Insert definitions from `definitions` or `senses`
        definitions = entry.get('definitions') or entry.get('senses', [])
        for definition in definitions:
            text = None
            pos = entry.get('pos', '')
            examples = None
            usage_notes = None
            tags = None

            if isinstance(definition, dict):
                text = definition.get('text')
                if not text and 'glosses' in definition and isinstance(definition['glosses'], list) and definition['glosses']:
                    text = definition['glosses'][0]
                # sense-level pos override
                if 'pos' in definition:
                    pos = definition['pos']
                if 'examples' in definition:
                    ex_data = definition['examples']
                    if isinstance(ex_data, list):
                        examples = json.dumps(ex_data)
                    else:
                        examples = json.dumps([ex_data])
                if 'usage_notes' in definition:
                    un_data = definition['usage_notes']
                    if isinstance(un_data, list):
                        usage_notes = json.dumps(un_data)
                    else:
                        usage_notes = json.dumps([un_data])
                
                # Extract tags such as "colloquial", "figuratively", etc.
                tags = extract_definition_tags(definition)
                
                # Extract any sense-level relations
                if filename:  # Only call if filename is provided
                    extract_sense_relations(cur, word_id, definition, language_code, 
                                          SourceStandardization.standardize_sources(os.path.basename(filename)))
            else:
                # It's just a string
                text = definition

            if text:
                text = re.sub(r'\.{3,}$', '', text).strip()
                source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
                insert_definition(
                    cur,
                    word_id,
                    text,
                    part_of_speech=standardize_entry_pos(pos),
                    examples=examples,
                    usage_notes=usage_notes,
                    tags=json.dumps(tags) if tags else None,
                    sources=source
                )

        # Insert etymology if present
        if 'etymology' in entry and entry['etymology'].strip():
            ety_text = entry['etymology']
            comps = extract_etymology_components(ety_text)
            
            # Store full etymology structure if available
            etymology_structure = None
            if 'etymology_templates' in entry and isinstance(entry['etymology_templates'], list):
                etymology_structure = json.dumps(entry['etymology_templates'])
            
            source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
            insert_etymology(
                cur,
                word_id,
                ety_text,
                normalized_components=json.dumps(comps) if comps else None,
                etymology_structure=etymology_structure,
                sources=source
            )

        # Process direct relationship arrays
        source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
        process_direct_relations(
            cur, 
            word_id, 
            entry, 
            language_code, 
            source
        )
        
        # Insert relationships with robust mapping (for the 'relations' dict)
        if 'relations' in entry and isinstance(entry['relations'], dict):
            process_relations(
                cur,
                word_id,
                entry['relations'],
                lang_code=language_code,
                source=source
            )
    except Exception as e:
        logger.error(f"Error processing Kaikki entry: {entry.get('word', 'unknown')}. Error: {str(e)}")


def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries."""
    # Check if table structure exists and create required tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pronunciations (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            tags JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, type, value)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_categories (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            category_name TEXT NOT NULL,
            category_kind TEXT,
            parents JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, category_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_templates (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            template_name TEXT NOT NULL,
            args JSONB,
            expansion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, template_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_forms (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            form TEXT NOT NULL,
            is_canonical BOOLEAN DEFAULT FALSE,
            is_primary BOOLEAN DEFAULT FALSE,
            tags JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, form)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_links (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            link_text TEXT NOT NULL,
            link_target TEXT NOT NULL,
            is_wikipedia BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, link_text, link_target)
        )
    """)
    
    # Add new columns to existing tables
    cur.execute("""
        DO $$
        BEGIN
            -- Words table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'badlit_form'
            ) THEN
                ALTER TABLE words ADD COLUMN badlit_form TEXT;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'hyphenation'
            ) THEN
                ALTER TABLE words ADD COLUMN hyphenation JSONB;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_proper_noun'
            ) THEN
                ALTER TABLE words ADD COLUMN is_proper_noun BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_abbreviation'
            ) THEN
                ALTER TABLE words ADD COLUMN is_abbreviation BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_initialism'
            ) THEN
                ALTER TABLE words ADD COLUMN is_initialism BOOLEAN DEFAULT FALSE;
            END IF;
            
            -- Definitions table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'definitions' AND column_name = 'metadata'
            ) THEN
                ALTER TABLE definitions ADD COLUMN metadata JSONB;
            END IF;
        END $$;
    """)
    
    # Extract Baybayin info (existing function)
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Baybayin' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Extract Badlit info (new function for Cebuano)
    def extract_badlit_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Badlit' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Helper function to extract all canonical forms of a word entry
    def extract_canonical_forms(entry: Dict) -> List[str]:
        forms = []
        if 'forms' in entry:
            for form in entry['forms']:
                if 'form' in form and form.get('form') and 'canonical' in form.get('tags', []):
                    forms.append(form['form'])
        return forms
    
    # Standardize entry POS (existing function)  
    def standardize_entry_pos(pos_str: str) -> str:
        if not pos_str:
            return 'unc'  # Default to uncategorized
        pos_key = pos_str.lower().strip()
        return get_standard_code(pos_key)
    
    # Process pronunciation (enhanced version)
    def process_pronunciation(cur, word_id: int, entry: Dict):
        if 'sounds' not in entry:
            return
            
        # Extract pronunciation styles from categories
        pronunciation_styles = set()
        if 'senses' in entry:
            for sense in entry['senses']:
                if 'categories' in sense:
                    for category in sense['categories']:
                        if 'name' in category:
                            category_name = category['name'].lower()
                            if 'pronunciation' in category_name:
                                # Extract the style (e.g., "mabilis", "malumay")
                                if 'with' in category_name:
                                    parts = category_name.split('with')
                                    if len(parts) > 1:
                                        style = parts[1].strip().split()[0]
                                        pronunciation_styles.add(style)
        
        for sound in entry['sounds']:
            if 'ipa' in sound:
                # Store IPA pronunciation
                tags = sound.get('tags', [])
                
                # Associate this with pronunciation styles if any
                metadata = {
                    "styles": list(pronunciation_styles)
                } if pronunciation_styles else None
                
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value, tags, metadata)
                    VALUES (%s, 'ipa', %s, %s, %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (
                    word_id, 
                    sound['ipa'], 
                    json.dumps(tags) if tags else None,
                    json.dumps(metadata) if metadata else None
                ))
            elif 'rhymes' in sound:
                # Store rhyme information
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value)
                    VALUES (%s, 'rhyme', %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (word_id, sound['rhymes']))
    
    # Process sense relationships (enhanced version)
    def process_sense_relationships(cur, word_id: int, sense: Dict):
        # Process synonyms
        if 'synonyms' in sense and isinstance(sense['synonyms'], list):
            for synonym in sense['synonyms']:
                if isinstance(synonym, dict) and 'word' in synonym:
                    syn_word = synonym['word']
                    syn_id = get_or_create_word_id(cur, syn_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in synonym and isinstance(synonym['tags'], list):
                        metadata['tags'] = ','.join(synonym['tags'])
                    insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM.value, "kaikki", metadata)
        
        # Process antonyms
        if 'antonyms' in sense and isinstance(sense['antonyms'], list):
            for antonym in sense['antonyms']:
                if isinstance(antonym, dict) and 'word' in antonym:
                    ant_word = antonym['word']
                    ant_id = get_or_create_word_id(cur, ant_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in antonym and isinstance(antonym['tags'], list):
                        metadata['tags'] = ','.join(antonym['tags'])
                    insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM.value, "kaikki", metadata)
        
        # Process hypernyms
        if 'hypernyms' in sense and isinstance(sense['hypernyms'], list):
            for hypernym in sense['hypernyms']:
                if isinstance(hypernym, dict) and 'word' in hypernym:
                    hyper_word = hypernym['word']
                    hyper_id = get_or_create_word_id(cur, hyper_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hypernym and isinstance(hypernym['tags'], list):
                        metadata['tags'] = ','.join(hypernym['tags'])
                    insert_relation(cur, word_id, hyper_id, RelationshipType.HYPERNYM.value, "kaikki", metadata)
        
        # Process hyponyms
        if 'hyponyms' in sense and isinstance(sense['hyponyms'], list):
            for hyponym in sense['hyponyms']:
                if isinstance(hyponym, dict) and 'word' in hyponym:
                    hypo_word = hyponym['word']
                    hypo_id = get_or_create_word_id(cur, hypo_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hyponym and isinstance(hyponym['tags'], list):
                        metadata['tags'] = ','.join(hyponym['tags'])
                    insert_relation(cur, word_id, hypo_id, RelationshipType.HYPONYM.value, "kaikki", metadata)
        
        # Process holonyms
        if 'holonyms' in sense and isinstance(sense['holonyms'], list):
            for holonym in sense['holonyms']:
                if isinstance(holonym, dict) and 'word' in holonym:
                    holo_word = holonym['word']
                    holo_id = get_or_create_word_id(cur, holo_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in holonym and isinstance(holonym['tags'], list):
                        metadata['tags'] = ','.join(holonym['tags'])
                    insert_relation(cur, word_id, holo_id, RelationshipType.HOLONYM.value, "kaikki", metadata)
        
        # Process meronyms
        if 'meronyms' in sense and isinstance(sense['meronyms'], list):
            for meronym in sense['meronyms']:
                if isinstance(meronym, dict) and 'word' in meronym:
                    mero_word = meronym['word']
                    mero_id = get_or_create_word_id(cur, mero_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in meronym and isinstance(meronym['tags'], list):
                        metadata['tags'] = ','.join(meronym['tags'])
                    insert_relation(cur, word_id, mero_id, RelationshipType.MERONYM.value, "kaikki", metadata)
        
        # Process derived terms
        if 'derived' in sense and isinstance(sense['derived'], list):
            for derived in sense['derived']:
                if isinstance(derived, dict) and 'word' in derived:
                    derived_word = derived['word']
                    derived_id = get_or_create_word_id(cur, derived_word, 'tl')
                    metadata = {'confidence': 95}
                    if 'tags' in derived and isinstance(derived['tags'], list):
                        metadata['tags'] = ','.join(derived['tags'])
                    insert_relation(cur, word_id, derived_id, RelationshipType.ROOT_OF.value, "kaikki", metadata)
        
        # Process "see also" references
        if 'see_also' in sense and isinstance(sense['see_also'], list):
            for see_also in sense['see_also']:
                if isinstance(see_also, dict) and 'word' in see_also:
                    see_also_word = see_also['word']
                    see_also_id = get_or_create_word_id(cur, see_also_word, 'tl')
                    metadata = {'confidence': 70}
                    if 'tags' in see_also and isinstance(see_also['tags'], list):
                        metadata['tags'] = ','.join(see_also['tags'])
                    insert_relation(cur, word_id, see_also_id, RelationshipType.SEE_ALSO.value, "kaikki", metadata)

    def process_form_relationships(cur, word_id: int, entry: Dict, language_code: str):
        """Process relationships based on word forms (variants, spelling, etc.)."""
        if not entry:
            return
            
        # Process alternative forms
        if 'forms' in entry:
            for form in entry.get('forms', []):
                if not form or not isinstance(form, dict):
                    continue
                    
                form_word = form.get('form', '')
                if not form_word:
                    continue
                    
                form_word_id = get_or_create_word_id(cur, form_word, language_code)
                
                # Determine relationship type based on form data
                rel_type = RelationshipType.VARIANT.rel_value
                metadata = {"from_forms": True}
                
                # Check for specific form types
                if 'tags' in form:
                    tags = form.get('tags', [])
                    
                    # Add tags to metadata
                    metadata["tags"] = tags
                    
                    # Determine strength and relationship type based on tags
                    if any(tag in ['standard spelling', 'preferred', 'standard form'] for tag in tags):
                        rel_type = RelationshipType.PREFERRED_SPELLING.rel_value if hasattr(RelationshipType, 'PREFERRED_SPELLING') else RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 95
                    elif any(tag in ['alternative spelling', 'alternate spelling', 'alt form'] for tag in tags):
                        rel_type = RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 90
                    elif any(tag in ['regional', 'dialect'] for tag in tags):
                        rel_type = RelationshipType.REGIONAL_VARIANT.rel_value
                        metadata["strength"] = 85
                    else:
                        metadata["strength"] = 80
                else:
                    metadata["strength"] = 80
                
                # Add source and qualifier if available
                if 'source' in form:
                    metadata["source"] = form.get('source')
                
                if 'qualifier' in form:
                    metadata["qualifier"] = form.get('qualifier')
                    
                insert_relation(
                    cur, 
                    word_id,
                    form_word_id,
                    rel_type,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )
                
        # Process alt_of references 
        if 'alt_of' in entry:
            for alt_of in entry.get('alt_of', []):
                if isinstance(alt_of, str):
                    alt_word = alt_of
                    alt_meta = {}
                elif isinstance(alt_of, list) and len(alt_of) > 0:
                    alt_word = alt_of[0]
                    alt_meta = {"context": alt_of[1]} if len(alt_of) > 1 else {}
                else:
                    continue
                    
                alt_word_id = get_or_create_word_id(cur, alt_word, language_code)
                
                # The current word is an alternative of the alt_word
                metadata = {"strength": 90, "from_alt_of": True}
                metadata.update(alt_meta)
                
                insert_relation(
                    cur, 
                    word_id,
                    alt_word_id,
                    RelationshipType.VARIANT.rel_value,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )


    # Process categories (new function)
    def process_categories(cur, definition_id: int, categories: List[Dict]):
        if not categories:
            return
            
        for category in categories:
            if 'name' not in category:
                continue
                
            category_name = category['name']
            category_kind = category.get('kind', '')
            parents = category.get('parents', [])
            
            cur.execute("""
                INSERT INTO definition_categories (definition_id, category_name, category_kind, parents)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (definition_id, category_name) DO NOTHING
            """, (
                definition_id,
                category_name,
                category_kind,
                json.dumps(parents) if parents else None
            ))
    
    # Process head templates (new function)
    def process_head_templates(cur, word_id: int, templates: List[Dict]):
        if not templates:
            return
            
        for template in templates:
            template_name = template.get('name', '')
            args = template.get('args', {})
            expansion = template.get('expansion', '')
            
            cur.execute("""
                INSERT INTO word_templates (word_id, template_name, args, expansion)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (word_id, template_name) DO NOTHING
            """, (
                word_id,
                template_name,
                json.dumps(args) if args else None,
                expansion
            ))
    
    # Process links (new function)
    def process_links(cur, definition_id: int, links: List[List[str]]):
        if not links:
            return
            
        for link in links:
            if len(link) >= 2:
                text = link[0]
                target = link[1]
                
                # Check if it's a Wikipedia link
                is_wikipedia = False
                if target.startswith('w:'):
                    is_wikipedia = True
                    target = target[2:]  # Remove the w: prefix
                
                cur.execute("""
                    INSERT INTO definition_links (definition_id, link_text, link_target, is_wikipedia)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (definition_id, link_text, link_target) DO NOTHING
                """, (
                    definition_id,
                    text,
                    target,
                    is_wikipedia
                ))
    
    # Process etymology (enhanced version)
    def process_etymology(cur, word_id: int, etymology_text: str, etymology_templates: List[Dict] = None):
        if not etymology_text:
            return
            
        # Extract basic components
        normalized_components = extract_etymology_components(etymology_text)
        
        # Extract source languages from etymology templates
        source_languages = []
        etymology_structure = None
        
        if etymology_templates:
            etymology_structure = json.dumps(etymology_templates)
            
            for template in etymology_templates:
                if template.get('name') in ('bor', 'der', 'inh', 'calque', 'bor+'):
                    args = template.get('args', {})
                    if '2' in args:  # Source language code
                        source_lang = args['2']
                        source_languages.append(source_lang)
                
                # Process borrowing sources
                template_name = template.get('name', '')
                args = template.get('args', {})
                
                if template_name in ('bor', 'bor+') and '2' in args and '3' in args:
                    source_lang = args['2']
                    source_word = args['3']
                    
                    # Skip if source word is proto-language with * prefix
                    if source_word.startswith('*'):
                        continue
                        
                    # Try to find or create the source word
                    source_word_id = get_or_create_word_id(cur, source_word, language_code=source_lang)
                    
                    # Create borrowed_from relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        source_word_id, 
                        'borrowed_from', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
                
                # Process cognates
                elif template_name == 'cog' and '1' in args and '2' in args:
                    cog_lang = args['2']
                    cog_word = args['3'] if '3' in args else args['2']
                    
                    # Skip if cognate word is proto-language with * prefix
                    if cog_word.startswith('*'):
                        continue
                        
                    # Try to find or create the cognate word
                    cog_word_id = get_or_create_word_id(cur, cog_word, language_code=cog_lang)
                    
                    # Create cognate relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        cog_word_id, 
                        'cognate_with', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
        
        # Create or update the etymology record
        cur.execute("""
            INSERT INTO etymologies (
                word_id, 
                etymology_text, 
                normalized_components, 
                etymology_structure,
                source_languages,
                sources
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word_id) DO UPDATE SET
                etymology_text = EXCLUDED.etymology_text,
                normalized_components = EXCLUDED.normalized_components,
                etymology_structure = EXCLUDED.etymology_structure,
                source_languages = EXCLUDED.source_languages,
                updated_at = CURRENT_TIMESTAMP
        """, (
            word_id,
            etymology_text,
            json.dumps(normalized_components) if normalized_components else None,
            etymology_structure,
            json.dumps(source_languages) if source_languages else None,
            SourceStandardization.standardize_sources('kaikki.jsonl')
        ))
    
    # Main process entry function (enhanced version)
    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry."""
        if 'word' not in entry:
            logger.warning("Skipping entry without 'word' field")
            return None
            
        try:
            word = entry['word']
            pos = entry.get('pos', '')
            language_code = entry.get('lang_code', 'tl')  # Default to Tagalog if not specified
            
            # Check if this is a proper noun
            is_proper_noun = False
            if pos == 'prop' or pos == 'proper noun':
                is_proper_noun = True
                pos = 'noun'  # Standardize to noun POS but mark as proper
                
            # Check if this is an abbreviation or initialism
            is_abbreviation = False
            is_initialism = False
            if pos == 'abbrev' or pos == 'abbreviation':
                is_abbreviation = True
                
            if 'tags' in entry:
                tags = entry.get('tags', [])
                if 'abbreviation' in tags or 'abbrev' in tags:
                    is_abbreviation = True
                if 'initialism' in tags or 'acronym' in tags:
                    is_initialism = True
            
            # Get Baybayin form if any
            baybayin_form, romanized_form = extract_baybayin_info(entry)
            
            # Get Badlit form if any
            badlit_form, badlit_romanized = extract_badlit_info(entry)
            
            # Get or create the word
            word_id = get_or_create_word_id(
                cur, 
                word, 
                language_code=language_code,
                has_baybayin=bool(baybayin_form),
                baybayin_form=baybayin_form,
                romanized_form=romanized_form or badlit_romanized,
                badlit_form=badlit_form,
                is_proper_noun=is_proper_noun,
                is_abbreviation=is_abbreviation,
                is_initialism=is_initialism,
                tags=','.join(entry.get('tags', [])) if 'tags' in entry else None
            )
            
            # Process pronunciation information
            process_pronunciation(cur, word_id, entry)
            
            # Process form relationships
            process_form_relationships(cur, word_id, entry, language_code)
            
            # Process etymologies
            if 'etymology_text' in entry:
                process_etymology(cur, word_id, entry['etymology_text'], entry.get('etymology_templates'))
                
            # Process head templates
            if 'head_templates' in entry and entry['head_templates']:
                process_head_templates(cur, word_id, entry['head_templates'])
                
            # Process definitions from senses
            if 'senses' in entry:
                for sense in entry['senses']:
                    if not sense or not isinstance(sense, dict):
                        continue
                        
                    # Skip definitions that are just references to other entries
                    if 'glosses' not in sense and 'raw_glosses' not in sense:
                        # This might be an alt_of entry
                        continue
                        
                    # Get the glosses
                    glosses = sense.get('glosses', sense.get('raw_glosses', []))
                    if not glosses:
                        continue
                    
                    definition_text = '; '.join(glosses) if isinstance(glosses, list) else str(glosses)
                    
                    # Check for qualifiers and add to usage notes
                    usage_notes = None
                    if 'qualifier' in sense:
                        qualifier = sense['qualifier']
                        if qualifier:
                            usage_notes = f"Qualifier: {qualifier}"
                            
                    # Check for domain/topic and add to usage notes
                    if 'topics' in sense:
                        topics = sense['topics']
                        topics_str = ', '.join(topics) if isinstance(topics, list) else str(topics)
                        if topics_str:
                            domain_note = f"Domain: {topics_str}"
                            usage_notes = f"{usage_notes}; {domain_note}" if usage_notes else domain_note
                    
                    # Extract examples if available
                    examples = None
                    if 'examples' in sense and sense['examples']:
                        examples = json.dumps([ex.get('text', ex) for ex in sense['examples'] 
                                             if (isinstance(ex, dict) and 'text' in ex) or isinstance(ex, str)])
                    
                    # Insert the definition
                    definition_id = insert_definition(
                        cur,
                        word_id,
                        definition_text,
                        part_of_speech=pos,
                        examples=examples,
                        usage_notes=usage_notes,
                        sources=SourceStandardization.standardize_sources(filename)
                    )
                    
                    if definition_id:
                        # Process category information
                        if 'categories' in sense and sense['categories']:
                            process_categories(cur, definition_id, sense['categories'])
                            
                        # Process links
                        if 'links' in sense and sense['links']:
                            process_links(cur, definition_id, sense['links'])
                            
                        # Process Wikipedia links
                        if 'wikipedia' in sense and sense['wikipedia']:
                            wiki_links = sense['wikipedia']
                            if isinstance(wiki_links, list):
                                for wiki in wiki_links:
                                    if isinstance(wiki, str):
                                        cur.execute("""
                                            INSERT INTO definition_links (definition_id, link_type, target, source)
                                            VALUES (%s, %s, %s, %s)
                                            ON CONFLICT DO NOTHING
                                        """, (
                                            definition_id,
                                            'wikipedia',
                                            wiki,
                                            SourceStandardization.standardize_sources(filename)
                                        ))
                    
                    # Process relationships
                    process_sense_relationships(cur, word_id, sense)
            
            return word_id
                
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('word', 'unknown')}: {str(e)}")
            return None

    # Process entries from the file
    with open(filename, 'r', encoding='utf-8') as f:
        entries_processed = 0
        errors = 0
        for line in f:
            try:
                entry = json.loads(line)
                process_entry(cur, entry)
                entries_processed += 1
                if entries_processed % 1000 == 0:
                    logger.info(f"Processed {entries_processed} entries from {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {str(e)}")
                errors += 1
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")
                errors += 1
        
        logger.info(f"Completed processing {filename}: {entries_processed} entries processed with {errors} errors")

def process_marayum_json(cur, filename: str):
    """Process Project Marayum dictionary entries.
    
    This function processes dictionary files from Project Marayum in JSON format that end with
    '_processed.json'. Each file represents a different language dictionary.
    
    Args:
        cur: Database cursor
        filename: Path to the Marayum JSON file
    """
    logger.info(f"Processing Marayum dictionary file: {filename}")
    
    try:
        # Read the entire JSON file
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract dictionary information
        dictionary_info = data.get('dictionary_info', {})
        base_language = dictionary_info.get('base_language', '')
        target_language = dictionary_info.get('target_language', 'English')
        dictionary_name = dictionary_info.get('text', os.path.basename(filename))
        
        # Determine the language code from base_language
        # Default to 'tl' (Tagalog) if we can't determine
        language_code = 'tl'  # Default
        
        # Map common Filipino languages to ISO 639 codes
        language_map = {
            'Tagalog': 'tl',
            'Cebuano': 'ceb',
            'Hiligaynon': 'hil',
            'Ilocano': 'ilo',
            'Bikol': 'bik',
            'Waray': 'war',
            'Kapampangan': 'pam',
            'Pangasinan': 'pag',
            'Asi': 'bnc',  # Bontoc language code
            'Kinaray-a': 'krj',
            'Aklanon': 'akl',
            'Chavacano': 'cbk',
            'Masbatenyo': 'bks'  # Butuanon/Surigaonon language code
            # Add more language mappings as needed
        }
        
        if base_language in language_map:
            language_code = language_map[base_language]
        else:
            # Try to extract a code from the filename pattern language-english_processed.json
            filename_base = os.path.basename(filename)
            if '-english_processed.json' in filename_base:
                lang_part = filename_base.split('-english_processed.json')[0]
                logger.info(f"Extracted language from filename: {lang_part}")
                # We'll just use the filename as a code if we can't map it
                if lang_part:
                    language_code = lang_part
        
        logger.info(f"Processing {base_language} dictionary (code: {language_code})")
        
        # Standardized source name for Project Marayum
        source_name = f"Project Marayum - {dictionary_name}"
        
        # Register the source with specific language information
        cur.execute("""
            INSERT INTO sources (name, url, description, last_updated)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (name) DO UPDATE SET
                last_updated = CURRENT_TIMESTAMP,
                description = EXCLUDED.description
            RETURNING id
        """, (
            source_name,
            "https://www.marayum.ph/",
            f"Project Marayum {base_language}-{target_language} Dictionary"
        ))
        
        # Process each word in the data
        words = data.get('words', [])
        words_processed = 0
        relationships_processed = 0
        
        for entry in words:
            try:
                word = entry.get('word', '')
                if not word:
                    logger.warning(f"Skipping entry without word: {entry}")
                    continue
                
                headword = entry.get('headword', word)
                pos = entry.get('pos', '')
                dialect = entry.get('dialect', '')
                etymology = entry.get('etymology', '')
                pronunciation = entry.get('pronunciation', '')
                credits = entry.get('credits', '')
                comment = entry.get('comment', '')
                
                # Extract tags
                tags = []
                if dialect:
                    tags.append(f"dialect:{dialect}")
                if entry.get('is_core', False):
                    tags.append('core')
                
                # Add Project Marayum ID as a tag for cross-reference
                if 'id' in entry:
                    tags.append(f"marayum_id:{entry['id']}")
                    
                # Get or create the word entry
                word_id = get_or_create_word_id(
                    cur,
                    word,
                    language_code=language_code,
                    preferred_spelling=headword if headword != word else None,
                    tags=','.join(tags) if tags else None
                )
                
                # Process etymology if available
                if etymology:
                    insert_etymology(
                        cur,
                        word_id,
                        etymology,
                        sources=SourceStandardization.standardize_sources(source_name)
                    )
                
                # Process pronunciation if available
                if pronunciation:
                    cur.execute("""
                        INSERT INTO pronunciations (word_id, type, value, tags, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (word_id, type, value) DO NOTHING
                    """, (
                        word_id,
                        'ipa' if '[' in pronunciation else 'text',
                        pronunciation,
                        None,
                        json.dumps({"source": "Project Marayum", "file": os.path.basename(filename)})
                    ))
                
                # Process definitions
                definitions = entry.get('definitions', [])
                for definition in definitions:
                    definition_text = definition.get('definition', '')
                    if not definition_text:
                        continue
                    
                    # Process examples if available
                    examples_data = definition.get('examples', [])
                    examples = None
                    if examples_data:
                        # Convert examples to the expected format
                        examples_json = []
                        for ex in examples_data:
                            if isinstance(ex, dict):
                                example_text = ex.get('text', '')
                                translation = ex.get('translation', '')
                                if example_text:
                                    examples_json.append({
                                        'text': example_text,
                                        'translation': translation
                                    })
                            elif isinstance(ex, str):
                                examples_json.append({'text': ex})
                        
                        if examples_json:
                            examples = json.dumps(examples_json)
                    
                    # Insert the definition
                    definition_id = insert_definition(
                        cur,
                        word_id,
                        definition_text,
                        part_of_speech=pos,
                        examples=examples,
                        usage_notes=comment if comment else None,
                        category=None,  # No category information in Marayum data
                        tags=None,  # No additional tags for definitions
                        sources=SourceStandardization.standardize_sources(source_name)
                    )
                    
                    # Add credits as metadata if available
                    if credits and definition_id:
                        cur.execute("""
                            UPDATE definitions
                            SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                            WHERE id = %s
                        """, (
                            json.dumps({"credits": credits}),
                            definition_id
                        ))
                
                # Process see_also relationships
                see_also = entry.get('see_also', [])
                for related in see_also:
                    if isinstance(related, dict) and 'text' in related and 'pk' in related:
                        related_word = related['text']
                        related_id = related['pk']
                        
                        # First, try to find the word by Marayum ID
                        cur.execute("""
                            SELECT id FROM words
                            WHERE tags LIKE %s
                            LIMIT 1
                        """, (f"%marayum_id:{related_id}%",))
                        
                        result = cur.fetchone()
                        related_word_id = result[0] if result else None
                        
                        # If not found by ID, try to find by lemma and language
                        if not related_word_id:
                            related_word_id = get_or_create_word_id(
                                cur,
                                related_word,
                                language_code=language_code
                            )
                        
                        # Add the relationship
                        if related_word_id:
                            # Use RelationshipManager for better relationship handling
                            relationship_manager = RelationshipManager(cur)
                            relationship_manager.add_relationship(
                                word_id,
                                related_word_id,
                                RelationshipType.SEE_ALSO,
                                sources=SourceStandardization.standardize_sources(source_name),
                                metadata={"marayum_entry_id": entry.get('id')}
                            )
                            relationships_processed += 1
                
                words_processed += 1
                if words_processed % 100 == 0:
                    logger.info(f"Processed {words_processed}/{len(words)} words from {os.path.basename(filename)}")
                
            except Exception as e:
                logger.error(f"Error processing entry {entry.get('word', 'unknown')}: {str(e)}")
        
        logger.info(f"Completed processing {filename}: {words_processed} words and {relationships_processed} relationships processed")
    
    except Exception as e:
        logger.error(f"Error processing Marayum file {filename}: {str(e)}")
        raise

def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries."""
    # Check if table structure exists and create required tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pronunciations (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            tags JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, type, value)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_categories (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            category_name TEXT NOT NULL,
            category_kind TEXT,
            parents JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, category_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_templates (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            template_name TEXT NOT NULL,
            args JSONB,
            expansion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, template_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_forms (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            form TEXT NOT NULL,
            is_canonical BOOLEAN DEFAULT FALSE,
            is_primary BOOLEAN DEFAULT FALSE,
            tags JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, form)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_links (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            link_text TEXT NOT NULL,
            link_target TEXT NOT NULL,
            is_wikipedia BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, link_text, link_target)
        )
    """)
    
    # Add new columns to existing tables
    cur.execute("""
        DO $$
        BEGIN
            -- Words table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'badlit_form'
            ) THEN
                ALTER TABLE words ADD COLUMN badlit_form TEXT;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'hyphenation'
            ) THEN
                ALTER TABLE words ADD COLUMN hyphenation JSONB;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_proper_noun'
            ) THEN
                ALTER TABLE words ADD COLUMN is_proper_noun BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_abbreviation'
            ) THEN
                ALTER TABLE words ADD COLUMN is_abbreviation BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_initialism'
            ) THEN
                ALTER TABLE words ADD COLUMN is_initialism BOOLEAN DEFAULT FALSE;
            END IF;
            
            -- Definitions table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'definitions' AND column_name = 'metadata'
            ) THEN
                ALTER TABLE definitions ADD COLUMN metadata JSONB;
            END IF;
        END $$;
    """)
    
    # Extract Baybayin info (existing function)
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Baybayin' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Extract Badlit info (new function for Cebuano)
    def extract_badlit_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Badlit' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Helper function to extract all canonical forms of a word entry
    def extract_canonical_forms(entry: Dict) -> List[str]:
        forms = []
        if 'forms' in entry:
            for form in entry['forms']:
                if 'form' in form and form.get('form') and 'canonical' in form.get('tags', []):
                    forms.append(form['form'])
        return forms
    
    # Standardize entry POS (existing function)  
    def standardize_entry_pos(pos_str: str) -> str:
        if not pos_str:
            return 'unc'  # Default to uncategorized
        pos_key = pos_str.lower().strip()
        return get_standard_code(pos_key)
    
    # Process pronunciation (enhanced version)
    def process_pronunciation(cur, word_id: int, entry: Dict):
        if 'sounds' not in entry:
            return
            
        # Extract pronunciation styles from categories
        pronunciation_styles = set()
        if 'senses' in entry:
            for sense in entry['senses']:
                if 'categories' in sense:
                    for category in sense['categories']:
                        if 'name' in category:
                            category_name = category['name'].lower()
                            if 'pronunciation' in category_name:
                                # Extract the style (e.g., "mabilis", "malumay")
                                if 'with' in category_name:
                                    parts = category_name.split('with')
                                    if len(parts) > 1:
                                        style = parts[1].strip().split()[0]
                                        pronunciation_styles.add(style)
        
        for sound in entry['sounds']:
            if 'ipa' in sound:
                # Store IPA pronunciation
                tags = sound.get('tags', [])
                
                # Associate this with pronunciation styles if any
                metadata = {
                    "styles": list(pronunciation_styles)
                } if pronunciation_styles else None
                
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value, tags, metadata)
                    VALUES (%s, 'ipa', %s, %s, %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (
                    word_id, 
                    sound['ipa'], 
                    json.dumps(tags) if tags else None,
                    json.dumps(metadata) if metadata else None
                ))
            elif 'rhymes' in sound:
                # Store rhyme information
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value)
                    VALUES (%s, 'rhyme', %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (word_id, sound['rhymes']))
    
    # Process sense relationships (enhanced version)
    def process_sense_relationships(cur, word_id: int, sense: Dict):
        # Process synonyms
        if 'synonyms' in sense and isinstance(sense['synonyms'], list):
            for synonym in sense['synonyms']:
                if isinstance(synonym, dict) and 'word' in synonym:
                    syn_word = synonym['word']
                    syn_id = get_or_create_word_id(cur, syn_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in synonym and isinstance(synonym['tags'], list):
                        metadata['tags'] = ','.join(synonym['tags'])
                    insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM.value, "kaikki", metadata)
        
        # Process antonyms
        if 'antonyms' in sense and isinstance(sense['antonyms'], list):
            for antonym in sense['antonyms']:
                if isinstance(antonym, dict) and 'word' in antonym:
                    ant_word = antonym['word']
                    ant_id = get_or_create_word_id(cur, ant_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in antonym and isinstance(antonym['tags'], list):
                        metadata['tags'] = ','.join(antonym['tags'])
                    insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM.value, "kaikki", metadata)
        
        # Process hypernyms
        if 'hypernyms' in sense and isinstance(sense['hypernyms'], list):
            for hypernym in sense['hypernyms']:
                if isinstance(hypernym, dict) and 'word' in hypernym:
                    hyper_word = hypernym['word']
                    hyper_id = get_or_create_word_id(cur, hyper_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hypernym and isinstance(hypernym['tags'], list):
                        metadata['tags'] = ','.join(hypernym['tags'])
                    insert_relation(cur, word_id, hyper_id, RelationshipType.HYPERNYM.value, "kaikki", metadata)
        
        # Process hyponyms
        if 'hyponyms' in sense and isinstance(sense['hyponyms'], list):
            for hyponym in sense['hyponyms']:
                if isinstance(hyponym, dict) and 'word' in hyponym:
                    hypo_word = hyponym['word']
                    hypo_id = get_or_create_word_id(cur, hypo_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hyponym and isinstance(hyponym['tags'], list):
                        metadata['tags'] = ','.join(hyponym['tags'])
                    insert_relation(cur, word_id, hypo_id, RelationshipType.HYPONYM.value, "kaikki", metadata)
        
        # Process holonyms
        if 'holonyms' in sense and isinstance(sense['holonyms'], list):
            for holonym in sense['holonyms']:
                if isinstance(holonym, dict) and 'word' in holonym:
                    holo_word = holonym['word']
                    holo_id = get_or_create_word_id(cur, holo_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in holonym and isinstance(holonym['tags'], list):
                        metadata['tags'] = ','.join(holonym['tags'])
                    insert_relation(cur, word_id, holo_id, RelationshipType.HOLONYM.value, "kaikki", metadata)
        
        # Process meronyms
        if 'meronyms' in sense and isinstance(sense['meronyms'], list):
            for meronym in sense['meronyms']:
                if isinstance(meronym, dict) and 'word' in meronym:
                    mero_word = meronym['word']
                    mero_id = get_or_create_word_id(cur, mero_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in meronym and isinstance(meronym['tags'], list):
                        metadata['tags'] = ','.join(meronym['tags'])
                    insert_relation(cur, word_id, mero_id, RelationshipType.MERONYM.value, "kaikki", metadata)
        
        # Process derived terms
        if 'derived' in sense and isinstance(sense['derived'], list):
            for derived in sense['derived']:
                if isinstance(derived, dict) and 'word' in derived:
                    derived_word = derived['word']
                    derived_id = get_or_create_word_id(cur, derived_word, 'tl')
                    metadata = {'confidence': 95}
                    if 'tags' in derived and isinstance(derived['tags'], list):
                        metadata['tags'] = ','.join(derived['tags'])
                    insert_relation(cur, word_id, derived_id, RelationshipType.ROOT_OF.value, "kaikki", metadata)
        
        # Process "see also" references
        if 'see_also' in sense and isinstance(sense['see_also'], list):
            for see_also in sense['see_also']:
                if isinstance(see_also, dict) and 'word' in see_also:
                    see_also_word = see_also['word']
                    see_also_id = get_or_create_word_id(cur, see_also_word, 'tl')
                    metadata = {'confidence': 70}
                    if 'tags' in see_also and isinstance(see_also['tags'], list):
                        metadata['tags'] = ','.join(see_also['tags'])
                    insert_relation(cur, word_id, see_also_id, RelationshipType.SEE_ALSO.value, "kaikki", metadata)

    def process_form_relationships(cur, word_id: int, entry: Dict, language_code: str):
        """Process relationships based on word forms (variants, spelling, etc.)."""
        if not entry:
            return
            
        # Process alternative forms
        if 'forms' in entry:
            for form in entry.get('forms', []):
                if not form or not isinstance(form, dict):
                    continue
                    
                form_word = form.get('form', '')
                if not form_word:
                    continue
                    
                form_word_id = get_or_create_word_id(cur, form_word, language_code)
                
                # Determine relationship type based on form data
                rel_type = RelationshipType.VARIANT.rel_value
                metadata = {"from_forms": True}
                
                # Check for specific form types
                if 'tags' in form:
                    tags = form.get('tags', [])
                    
                    # Add tags to metadata
                    metadata["tags"] = tags
                    
                    # Determine strength and relationship type based on tags
                    if any(tag in ['standard spelling', 'preferred', 'standard form'] for tag in tags):
                        rel_type = RelationshipType.PREFERRED_SPELLING.rel_value if hasattr(RelationshipType, 'PREFERRED_SPELLING') else RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 95
                    elif any(tag in ['alternative spelling', 'alternate spelling', 'alt form'] for tag in tags):
                        rel_type = RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 90
                    elif any(tag in ['regional', 'dialect'] for tag in tags):
                        rel_type = RelationshipType.REGIONAL_VARIANT.rel_value
                        metadata["strength"] = 85
                    else:
                        metadata["strength"] = 80
                else:
                    metadata["strength"] = 80
                
                # Add source and qualifier if available
                if 'source' in form:
                    metadata["source"] = form.get('source')
                
                if 'qualifier' in form:
                    metadata["qualifier"] = form.get('qualifier')
                    
                insert_relation(
                    cur, 
                    word_id,
                    form_word_id,
                    rel_type,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )
                
        # Process alt_of references 
        if 'alt_of' in entry:
            for alt_of in entry.get('alt_of', []):
                if isinstance(alt_of, str):
                    alt_word = alt_of
                    alt_meta = {}
                elif isinstance(alt_of, list) and len(alt_of) > 0:
                    alt_word = alt_of[0]
                    alt_meta = {"context": alt_of[1]} if len(alt_of) > 1 else {}
                else:
                    continue
                    
                alt_word_id = get_or_create_word_id(cur, alt_word, language_code)
                
                # The current word is an alternative of the alt_word
                metadata = {"strength": 90, "from_alt_of": True}
                metadata.update(alt_meta)
                
                insert_relation(
                    cur, 
                    word_id,
                    alt_word_id,
                    RelationshipType.VARIANT.rel_value,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )


    # Process categories (new function)
    def process_categories(cur, definition_id: int, categories: List[Dict]):
        if not categories:
            return
            
        for category in categories:
            if 'name' not in category:
                continue
                
            category_name = category['name']
            category_kind = category.get('kind', '')
            parents = category.get('parents', [])
            
            cur.execute("""
                INSERT INTO definition_categories (definition_id, category_name, category_kind, parents)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (definition_id, category_name) DO NOTHING
            """, (
                definition_id,
                category_name,
                category_kind,
                json.dumps(parents) if parents else None
            ))
    
    # Process head templates (new function)
    def process_head_templates(cur, word_id: int, templates: List[Dict]):
        if not templates:
            return
            
        for template in templates:
            template_name = template.get('name', '')
            args = template.get('args', {})
            expansion = template.get('expansion', '')
            
            cur.execute("""
                INSERT INTO word_templates (word_id, template_name, args, expansion)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (word_id, template_name) DO NOTHING
            """, (
                word_id,
                template_name,
                json.dumps(args) if args else None,
                expansion
            ))
    
    # Process links (new function)
    def process_links(cur, definition_id: int, links: List[List[str]]):
        if not links:
            return
            
        for link in links:
            if len(link) >= 2:
                text = link[0]
                target = link[1]
                
                # Check if it's a Wikipedia link
                is_wikipedia = False
                if target.startswith('w:'):
                    is_wikipedia = True
                    target = target[2:]  # Remove the w: prefix
                
                cur.execute("""
                    INSERT INTO definition_links (definition_id, link_text, link_target, is_wikipedia)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (definition_id, link_text, link_target) DO NOTHING
                """, (
                    definition_id,
                    text,
                    target,
                    is_wikipedia
                ))
    
    # Process etymology (enhanced version)
    def process_etymology(cur, word_id: int, etymology_text: str, etymology_templates: List[Dict] = None):
        if not etymology_text:
            return
            
        # Extract basic components
        normalized_components = extract_etymology_components(etymology_text)
        
        # Extract source languages from etymology templates
        source_languages = []
        etymology_structure = None
        
        if etymology_templates:
            etymology_structure = json.dumps(etymology_templates)
            
            for template in etymology_templates:
                if template.get('name') in ('bor', 'der', 'inh', 'calque', 'bor+'):
                    args = template.get('args', {})
                    if '2' in args:  # Source language code
                        source_lang = args['2']
                        source_languages.append(source_lang)
                
                # Process borrowing sources
                template_name = template.get('name', '')
                args = template.get('args', {})
                
                if template_name in ('bor', 'bor+') and '2' in args and '3' in args:
                    source_lang = args['2']
                    source_word = args['3']
                    
                    # Skip if source word is proto-language with * prefix
                    if source_word.startswith('*'):
                        continue
                        
                    # Try to find or create the source word
                    source_word_id = get_or_create_word_id(cur, source_word, language_code=source_lang)
                    
                    # Create borrowed_from relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        source_word_id, 
                        'borrowed_from', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
                
                # Process cognates
                elif template_name == 'cog' and '1' in args and '2' in args:
                    cog_lang = args['2']
                    cog_word = args['3'] if '3' in args else args['2']
                    
                    # Skip if cognate word is proto-language with * prefix
                    if cog_word.startswith('*'):
                        continue
                        
                    # Try to find or create the cognate word
                    cog_word_id = get_or_create_word_id(cur, cog_word, language_code=cog_lang)
                    
                    # Create cognate relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        cog_word_id, 
                        'cognate_with', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
        
        # Create or update the etymology record
        cur.execute("""
            INSERT INTO etymologies (
                word_id, 
                etymology_text, 
                normalized_components, 
                etymology_structure,
                source_languages,
                sources
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word_id) DO UPDATE SET
                etymology_text = EXCLUDED.etymology_text,
                normalized_components = EXCLUDED.normalized_components,
                etymology_structure = EXCLUDED.etymology_structure,
                source_languages = EXCLUDED.source_languages,
                updated_at = CURRENT_TIMESTAMP
        """, (
            word_id,
            etymology_text,
            json.dumps(normalized_components) if normalized_components else None,
            etymology_structure,
            json.dumps(source_languages) if source_languages else None,
            SourceStandardization.standardize_sources('kaikki.jsonl')
        ))
    
    # Main process entry function (enhanced version)
    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry."""
        if 'word' not in entry:
            logger.warning("Skipping entry without 'word' field")
            return None
            
        try:
            word = entry['word']
            pos = entry.get('pos', '')
            language_code = entry.get('lang_code', 'tl')  # Default to Tagalog if not specified
            
            # Check if this is a proper noun
            is_proper_noun = False
            if pos == 'prop' or pos == 'proper noun':
                is_proper_noun = True
                pos = 'noun'  # Standardize to noun POS but mark as proper
                
            # Check if this is an abbreviation or initialism
            is_abbreviation = False
            is_initialism = False
            if pos == 'abbrev' or pos == 'abbreviation':
                is_abbreviation = True
                
            if 'tags' in entry:
                tags = entry.get('tags', [])
                if 'abbreviation' in tags or 'abbrev' in tags:
                    is_abbreviation = True
                if 'initialism' in tags or 'acronym' in tags:
                    is_initialism = True
            
            # Get Baybayin form if any
            baybayin_form, romanized_form = extract_baybayin_info(entry)
            
            # Get Badlit form if any
            badlit_form, badlit_romanized = extract_badlit_info(entry)
            
            # Get or create the word
            word_id = get_or_create_word_id(
                cur, 
                word, 
                language_code=language_code,
                has_baybayin=bool(baybayin_form),
                baybayin_form=baybayin_form,
                romanized_form=romanized_form or badlit_romanized,
                badlit_form=badlit_form,
                is_proper_noun=is_proper_noun,
                is_abbreviation=is_abbreviation,
                is_initialism=is_initialism,
                tags=','.join(entry.get('tags', [])) if 'tags' in entry else None
            )
            
            # Process pronunciation information
            process_pronunciation(cur, word_id, entry)
            
            # Process form relationships
            process_form_relationships(cur, word_id, entry, language_code)
            
            # Process etymologies
            if 'etymology_text' in entry:
                process_etymology(cur, word_id, entry['etymology_text'], entry.get('etymology_templates'))
                
            # Process head templates
            if 'head_templates' in entry and entry['head_templates']:
                process_head_templates(cur, word_id, entry['head_templates'])
                
            # Process definitions from senses
            if 'senses' in entry:
                for sense in entry['senses']:
                    if not sense or not isinstance(sense, dict):
                        continue
                        
                    # Skip definitions that are just references to other entries
                    if 'glosses' not in sense and 'raw_glosses' not in sense:
                        # This might be an alt_of entry
                        continue
                        
                    # Get the glosses
                    glosses = sense.get('glosses', sense.get('raw_glosses', []))
                    if not glosses:
                        continue
                    
                    definition_text = '; '.join(glosses) if isinstance(glosses, list) else str(glosses)
                    
                    # Check for qualifiers and add to usage notes
                    usage_notes = None
                    if 'qualifier' in sense:
                        qualifier = sense['qualifier']
                        if qualifier:
                            usage_notes = f"Qualifier: {qualifier}"
                            
                    # Check for domain/topic and add to usage notes
                    if 'topics' in sense:
                        topics = sense['topics']
                        topics_str = ', '.join(topics) if isinstance(topics, list) else str(topics)
                        if topics_str:
                            domain_note = f"Domain: {topics_str}"
                            usage_notes = f"{usage_notes}; {domain_note}" if usage_notes else domain_note
                    
                    # Extract examples if available
                    examples = None
                    if 'examples' in sense and sense['examples']:
                        examples = json.dumps([ex.get('text', ex) for ex in sense['examples'] 
                                             if (isinstance(ex, dict) and 'text' in ex) or isinstance(ex, str)])
                    
                    # Insert the definition
                    definition_id = insert_definition(
                        cur,
                        word_id,
                        definition_text,
                        part_of_speech=pos,
                        examples=examples,
                        usage_notes=usage_notes,
                        sources=SourceStandardization.standardize_sources(filename)
                    )
                    
                    if definition_id:
                        # Process category information
                        if 'categories' in sense and sense['categories']:
                            process_categories(cur, definition_id, sense['categories'])
                            
                        # Process links
                        if 'links' in sense and sense['links']:
                            process_links(cur, definition_id, sense['links'])
                            
                        # Process Wikipedia links
                        if 'wikipedia' in sense and sense['wikipedia']:
                            wiki_links = sense['wikipedia']
                            if isinstance(wiki_links, list):
                                for wiki in wiki_links:
                                    if isinstance(wiki, str):
                                        cur.execute("""
                                            INSERT INTO definition_links (definition_id, link_type, target, source)
                                            VALUES (%s, %s, %s, %s)
                                            ON CONFLICT DO NOTHING
                                        """, (
                                            definition_id,
                                            'wikipedia',
                                            wiki,
                                            SourceStandardization.standardize_sources(filename)
                                        ))
                    
                    # Process relationships
                    process_sense_relationships(cur, word_id, sense)
            
            return word_id
                
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('word', 'unknown')}: {str(e)}")
            return None

    # Process entries from the file
    with open(filename, 'r', encoding='utf-8') as f:
        entries_processed = 0
        errors = 0
        for line in f:
            try:
                entry = json.loads(line)
                process_entry(cur, entry)
                entries_processed += 1
                if entries_processed % 1000 == 0:
                    logger.info(f"Processed {entries_processed} entries from {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {str(e)}")
                errors += 1
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")
                errors += 1
        
        logger.info(f"Completed processing {filename}: {entries_processed} entries processed with {errors} errors")

# -------------------------------------------------------------------
# Command Line Interface Functions
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
    'dbname': os.getenv('DB_NAME', 'fil_dict_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

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
# Database Configuration
# -------------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME", "fil_dict_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
    print("Error: Missing database configuration!")
    print("Please ensure you have a .env file with the following variables:")
    print("DB_NAME - The name of your PostgreSQL database")
    print("DB_USER - Your PostgreSQL username")
    print("DB_PASSWORD - Your PostgreSQL password")
    print("DB_HOST - Your database host (usually 'localhost')")
    print("DB_PORT - Your database port (default: 5432)")
    sys.exit(1)

# -------------------------------------------------------------------
# Core Database Connection
# -------------------------------------------------------------------
def get_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', DB_NAME),
        user=os.environ.get('DB_USER', DB_USER),
        password=os.environ.get('DB_PASSWORD', DB_PASSWORD),
        host=os.environ.get('DB_HOST', DB_HOST),
        port=os.environ.get('DB_PORT', DB_PORT)
    )
    conn.autocommit = False  # Ensure transactions are used.
    return conn

def get_cursor():
    """Return a cursor from a new connection."""
    conn = get_connection()
    return conn.cursor()

# -------------------------------------------------------------------
# Setup Extensions
# -------------------------------------------------------------------
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
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
                conn.commit()
            else:
                logger.info(f"Extension {ext} already installed")
    except Exception as e:
        logger.error(f"Error setting up extensions: {str(e)}")
        conn.rollback()
    finally:
        cur.close()

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

def create_or_update_tables(conn):
    logger.info("Starting table creation/update process.")
    cur = conn.cursor()
    try:
        cur.execute("""
            DROP TABLE IF EXISTS 
                 definition_relations, affixations, relations, etymologies, 
                 definitions, words, parts_of_speech CASCADE;
        """)
        cur.execute(TABLE_CREATION_SQL)
        conn.commit()
        
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
        conn.commit()
        logger.info("Tables created or updated successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {str(e)}")
        raise
    finally:
        cur.close()

# Define standard part of speech mappings
POS_MAPPING = {
    # Nouns
    'noun': 'n', 'pangngalan': 'n', 'name': 'n', 'n': 'n', 'pangalan': 'n',
    # Verbs
    'verb': 'v', 'pandiwa': 'v', 'v': 'v', 'action': 'v',
    # Adjectives
    'adjective': 'adj', 'pang-uri': 'adj', 'adj': 'adj', 'quality': 'adj', 'uri': 'adj',
    # Adverbs
    'adverb': 'adv', 'pang-abay': 'adv', 'adv': 'adv', 'manner': 'adv', 'abay': 'adv',
    # Pronouns
    'pronoun': 'pron', 'panghalip': 'pron', 'pron': 'pron', 'halip': 'pron',
    # Prepositions
    'preposition': 'prep', 'pang-ukol': 'prep', 'prep': 'prep', 'ukol': 'prep',
    # Conjunctions
    'conjunction': 'conj', 'pangatnig': 'conj', 'conj': 'conj', 'katnig': 'conj',
    # Interjections
    'interjection': 'intj', 'pandamdam': 'intj', 'intj': 'intj', 'damdam': 'intj',
    # Articles
    'article': 'art', 'pantukoy': 'art', 'art': 'art', 'tukoy': 'art',
    # Others
    'expression': 'expr', 'pahayag': 'expr', 'expr': 'expr',
    'phrase': 'phr', 'parirala': 'phr', 'phr': 'phr',
    'affix': 'aff', 'panlapi': 'aff', 'aff': 'aff', 'lapi': 'aff',
    'prefix': 'pref', 'unlapi': 'pref', 'pref': 'pref',
    'suffix': 'suff', 'hulapi': 'suff', 'suff': 'suff',
    'infix': 'inf', 'gitlapi': 'inf', 'inf': 'inf',
    'particle': 'part', 'kataga': 'part', 'part': 'part',
    'number': 'num', 'bilang': 'num', 'num': 'num',
    'determiner': 'det', 'panuri': 'det', 'det': 'det',
    'auxiliary': 'aux', 'pantulong': 'aux', 'aux': 'aux'
}

def get_standard_code(pos_string: str) -> str:
    """
    Convert a part of speech string to a standardized code.
    
    Args:
        pos_string: The part of speech string to standardize
        
    Returns:
        A standardized code for the part of speech
    """
    if not pos_string:
        return 'unc'  # Uncategorized
        
    # Remove any parenthetical information and clean string
    pos_key = pos_string.lower().strip()
    pos_key = re.sub(r'\([^)]*\)', '', pos_key).strip()
    
    # Check for direct match in mapping
    if pos_key in POS_MAPPING:
        return POS_MAPPING[pos_key]
    
    # Check for partial matches (e.g., "common noun" -> "noun")
    for key, code in POS_MAPPING.items():
        if key in pos_key:
            return code
    
    # Default to uncategorized if no match found
    return 'unc'

def standardize_entry_pos(pos_str: str) -> str:
    """Standardize part-of-speech string in dictionary entries."""
    # Use the existing get_standard_code function
    return get_standard_code(pos_str)
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
                
                # No need to recurse further for now to avoid deep dependency chains
                # For a complete implementation, you could recursively call 
                # get_transitive_relationships here
                
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

@with_transaction(commit=False)
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

@with_transaction(commit=True)
def merge_baybayin_entries(cur, baybayin_id: int, romanized_id: int):
    """Merge a Baybayin entry with its romanized form."""
    try:
        cur.execute("""
            SELECT lemma, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """, (baybayin_id,))
        baybayin_result = cur.fetchone()
        if not baybayin_result:
            raise ValueError(f"Baybayin entry {baybayin_id} not found")
        baybayin_lemma, baybayin_form, baybayin_rom = baybayin_result
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
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = COALESCE(%s, baybayin_form),
                romanized_form = COALESCE(%s, romanized_form),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (baybayin_form, baybayin_rom or baybayin_lemma, romanized_id))
        cur.execute("DELETE FROM words WHERE id = %s", (baybayin_id,))
    except Exception as e:
        logger.error(f"Error merging Baybayin entries: {str(e)}")
        raise

def extract_baybayin_text(text: str) -> List[str]:
    parts = re.split(r'[^ᜀ-᜔\s]+', text)
    return [part.strip() for part in parts if part.strip() and re.search(r'[\u1700-\u171F]', part)]

def validate_baybayin_entry(baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
    try:
        romanizer = BaybayinRomanizer()
        parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_form)
        valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
        if not valid_parts:
            return False
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

@with_transaction(commit=True)
def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> None:
    """Process and store Baybayin data for a word."""
    if not baybayin_form:
        return
    try:
        romanizer = BaybayinRomanizer()
        if not validate_baybayin_entry(baybayin_form, romanized_form):
            logger.warning(f"Invalid Baybayin form for word_id {word_id}: {baybayin_form}")
            return
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

@with_transaction(commit=True)
def process_baybayin_entries(cur):
    """Process all Baybayin entries in the database."""
    logger.info("Processing Baybayin entries...")
    cur.execute("""
        SELECT id, lemma, language_code, normalized_lemma 
        FROM words 
        WHERE lemma ~ '[\u1700-\u171F]'
        ORDER BY id ASC
    """)
    baybayin_entries = cur.fetchall()
    conn = cur.connection
    for baybayin_id, baybayin_lemma, language_code, _ in baybayin_entries:
        try:
            cur.execute("BEGIN")
            parts = re.split(r'[^ᜀ-᜔\s]+', baybayin_lemma)
            valid_parts = [p.strip() for p in parts if p.strip() and re.search(r'[\u1700-\u171F]', p)]
            if not valid_parts:
                logger.warning(f"No valid Baybayin segments found for entry {baybayin_id}: {baybayin_lemma}")
                conn.commit()
                continue
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
            logger.info(f"Updating Baybayin entry (ID: {baybayin_id}) with cleaned form")
            cur.execute("""
                UPDATE words 
                SET romanized_form = %s,
                    baybayin_form = %s,
                    has_baybayin = TRUE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (romanized, cleaned_baybayin, baybayin_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error processing Baybayin entry {baybayin_id}: {str(e)}")
            continue

@with_transaction(commit=True)
def cleanup_baybayin_data(cur):
    """Clean up Baybayin data in the database."""
    conn = cur.connection
    try:
        cur.execute("BEGIN")
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form,
                '[^ᜀ-᜔\s]',
                '',
                'g'
            )
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
        """)
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form, 
                '\s+',
                ' ',
                'g'
            )
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
        """)
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE, baybayin_form = NULL
            WHERE has_baybayin = TRUE AND (baybayin_form IS NULL OR baybayin_form = '' OR baybayin_form !~ '[\u1700-\u171F]')
        """)
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE,
                baybayin_form = NULL
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
        """)
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
        cur.execute("""
            WITH DuplicateBaybayin AS (
                SELECT MIN(id) as keep_id,
                       language_code,
                       baybayin_form
                FROM words
                WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
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
        conn.commit()
        logger.info("Baybayin data cleanup completed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during Baybayin cleanup: {str(e)}")
        raise

@with_transaction(commit=False)
def check_baybayin_consistency(cur):
    """Check for consistency issues in Baybayin data."""
    issues = []
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
    cur.execute(r"""
        SELECT id, lemma, baybayin_form
        FROM words
        WHERE baybayin_form ~ '[^ᜀ-᜔\s]'
    """)
    invalid_chars = cur.fetchall()
    if invalid_chars:
        issues.append(f"Found {len(invalid_chars)} entries with invalid Baybayin characters")
        for word_id, lemma, baybayin in invalid_chars:
            logger.warning(f"Invalid Baybayin characters in word ID {word_id}: {lemma}")
    return issues

# -------------------------------------------------------------------
# Word Insertion and Update Functions
# -------------------------------------------------------------------
@with_transaction(commit=True)
def get_or_create_word_id(cur, lemma: str, language_code: str = "tl", **kwargs) -> int:
    """Get or create a word in the dictionary and return its ID."""
    if not lemma:
        raise ValueError("Lemma cannot be empty")

    normalized = normalize_lemma(lemma)
    search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {normalized}"))
    
    # Check if word already exists
    cur.execute("""
        SELECT id FROM words
        WHERE normalized_lemma = %s AND language_code = %s
    """, (normalized, language_code))
    
    result = cur.fetchone()
    if result:
        word_id = result[0]
        
        # Update any provided fields
        if kwargs:
            fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['root_word_id', 'has_baybayin', 'baybayin_form', 'romanized_form', 'tags', 'preferred_spelling']:
                    fields.append(f"{key} = %s")
                    values.append(value)
            
            if fields:
                values.append(word_id)
            cur.execute(f"""
            UPDATE words 
                    SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
                """, values)
                
        return word_id
    
    # Word doesn't exist, create it
    fields = ['lemma', 'normalized_lemma', 'language_code', 'search_text']
    values = [lemma, normalized, language_code, search_text]
    placeholders = ['%s', '%s', '%s', 'to_tsvector(\'simple\', %s)']
    
    # Add optional fields if provided
    for key, value in kwargs.items():
        if key in ['root_word_id', 'has_baybayin', 'baybayin_form', 'romanized_form', 'tags', 'preferred_spelling']:
            fields.append(key)
            values.append(value)
            placeholders.append('%s')
    
    # Create the word
    query = f"""
        INSERT INTO words ({', '.join(fields)})
        VALUES ({', '.join(placeholders)})
            RETURNING id
    """
    
    cur.execute(query, values)
    word_id = cur.fetchone()[0]
    return word_id

@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str, part_of_speech: str = "",
                      examples: str = None, usage_notes: str = None, category: str = None,
                      tags: str = None, sources: str = "") -> Optional[int]:
    """
    Inserts a definition for a given word.
    Checks for duplicate definitions (by word_id, definition_text, standardized_pos_id)
    to avoid violating the unique constraint.
    """
    try:
        # Skip definitions that are just Baybayin spelling notices.
        if 'Baybayin spelling of' in definition_text:
            return None

        # Verify the word exists.
        cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
        if not cur.fetchone():
            logger.error(f"Cannot insert definition – word ID {word_id} does not exist.")
            return None

        # Get standardized part-of-speech ID (assume your helper function is defined elsewhere)
        std_pos_id = get_standardized_pos_id(cur, part_of_speech)

        # Check for duplicate definitions.
        cur.execute("""
            SELECT id FROM definitions 
            WHERE word_id = %s AND definition_text = %s AND standardized_pos_id = %s
        """, (word_id, definition_text, std_pos_id))
        if cur.fetchone():
            return None

        # Optionally prepend category info to usage_notes.
        if category:
            usage_notes = f"[{category}] {usage_notes if usage_notes else ''}"

        cur.execute("""
            INSERT INTO definitions 
                 (word_id, definition_text, original_pos, standardized_pos_id, 
                  examples, usage_notes, tags, sources)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            word_id,
            definition_text,
            part_of_speech,
            std_pos_id,
            examples,
            usage_notes,
            tags,
            sources
        ))
        return cur.fetchone()[0]

    except UniqueViolation:
        logger.warning(f"Duplicate definition detected for word ID {word_id}: {definition_text[:50]}...")
        # No need for manual rollback here - the decorator will handle it
        return None
    except Exception as e:
        logger.error(f"Error in insert_definition for word ID {word_id}, definition: {definition_text[:50]}...: {e}")
        return None

@with_transaction(commit=True)
def insert_relation(cur, from_word_id: int, to_word_id: int, relation_type: str, sources: str = "", metadata: Dict = None):
    """
    Inserts a relation between two words.
    Does nothing if the same relation already exists.
    
    Args:
        cur: Database cursor
        from_word_id: ID of the source word
        to_word_id: ID of the target word
        relation_type: Type of relationship
        sources: Comma-separated list of data sources
        metadata: Optional dictionary of additional metadata to store (will be serialized as JSONB)
    """
    try:
        # Use the RelationshipManager to handle the relationship
        rel_manager = RelationshipManager(cur)
        return rel_manager.add_relationship(
            from_word_id=from_word_id,
            to_word_id=to_word_id,
            relationship_type=relation_type,
            sources=sources,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Error in insert_relation from {from_word_id} to {to_word_id}: {e}")
        
        # Fall back to direct insertion if the RelationshipManager fails
        if from_word_id == to_word_id:
            return False
            
        sources = ", ".join(sorted(set(sources.split(", ")))) if sources else ""
        
        try:
            # If metadata is provided, include it in the insert
            if metadata:
                cur.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) 
                    DO UPDATE SET 
                        sources = CASE 
                            WHEN relations.sources IS NULL THEN EXCLUDED.sources
                            WHEN EXCLUDED.sources IS NULL THEN relations.sources
                            ELSE (
                                SELECT string_agg(DISTINCT unnest, ', ')
                                FROM unnest(string_to_array(relations.sources || ', ' || EXCLUDED.sources, ', '))
                            )
                        END,
                        metadata = COALESCE(relations.metadata, '{}') || EXCLUDED.metadata
                """, (from_word_id, to_word_id, relation_type, sources, json.dumps(metadata)))
            else:
                cur.execute("""
                    INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                """, (from_word_id, to_word_id, relation_type, sources))
            return True
        except Exception as inner_e:
            logger.error(f"Fallback insertion failed: {inner_e}")
            return False

@with_transaction(commit=True)
def insert_definition_relation(cur, definition_id: int, word_id: int, relation_type: str, sources: str = "auto"):
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
    normalized_components: str = None,
    etymology_structure: str = None,
    language_codes: str = None,
    sources: str = ""
) -> None:
    """Insert etymology data into the etymologies table."""
    if not word_id or not etymology_text:
        return

    try:
        cur.execute(
            """
            INSERT INTO etymologies (
                word_id, etymology_text, normalized_components, etymology_structure, language_codes, sources
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word_id, etymology_text)
            DO UPDATE SET
                normalized_components = COALESCE(etymologies.normalized_components, EXCLUDED.normalized_components),
                etymology_structure = COALESCE(etymologies.etymology_structure, EXCLUDED.etymology_structure),
                language_codes = COALESCE(etymologies.language_codes, EXCLUDED.language_codes),
                sources = array_to_string(ARRAY(
                    SELECT DISTINCT unnest(array_cat(
                        string_to_array(etymologies.sources, ', '),
                        string_to_array(EXCLUDED.sources, ', ')
                    ))
                ), ', ')
            """,
            (word_id, etymology_text, normalized_components, etymology_structure, language_codes, sources)
        )
    except Exception as e:
        logger.error(f"Error inserting etymology for word_id {word_id}: {str(e)}")

@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    sources: str
) -> None:
    """Insert an affixation relationship into the affixations table."""
    if not root_id or not affixed_id or root_id == affixed_id:
        return

    try:
        cur.execute(
            """
            INSERT INTO affixations (
                root_word_id, affixed_word_id, affix_type, sources
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (root_word_id, affixed_word_id, affix_type)
            DO UPDATE SET
                sources = array_to_string(ARRAY(
                    SELECT DISTINCT unnest(array_cat(
                        string_to_array(affixations.sources, ', '),
                        string_to_array(EXCLUDED.sources, ', ')
                    ))
                ), ', ')
            """,
            (root_id, affixed_id, affix_type, sources)
        )
    except Exception as e:
        logger.error(f"Error inserting affixation {root_id} -> {affixed_id}: {str(e)}")

@with_transaction(commit=True)
def batch_get_or_create_word_ids(cur, entries: List[Tuple[str, str]], batch_size: int = 1000) -> Dict[Tuple[str, str], int]:
    """
    Create or get IDs for multiple words in batches.
    
    Args:
        cur: Database cursor
        entries: List of (lemma, language_code) tuples
        batch_size: Number of entries to process in each batch
        
    Returns:
        Dictionary mapping (lemma, language_code) to word_id
    """
    result = {}
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch = list(dict.fromkeys(batch))  # Remove duplicates
        
        # First, let's ensure all existing words are fetched from the database
        normalized_entries = [(lemma, normalize_lemma(lemma), lang_code) for lemma, lang_code in batch]
        placeholders = []
        flat_params = []
        
        for _, norm, lang in normalized_entries:
            placeholders.append("(%s, %s)")
            flat_params.extend([norm, lang])
            
        if not placeholders:
            continue  # Empty batch, skip
            
        query = f"""
            SELECT lemma, language_code, id
            FROM words
            WHERE (normalized_lemma, language_code) IN ({', '.join(placeholders)})
        """
        
        try:
            cur.execute(query, flat_params)
            existing = {(lemma, lang): id for lemma, lang, id in cur.fetchall()}
        except Exception as e:
            logger.error(f"Error fetching existing words: {str(e)}")
            existing = {}
            
        # Identify entries that need to be inserted
        to_insert = []
        for lemma, lang in batch:
            if (lemma, lang) not in existing:
                norm = normalize_lemma(lemma)
                to_insert.append((lemma, norm, lang))
        
        # Insert missing entries with individual transactions
        for lemma, norm, lang in to_insert:
            try:
                search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {norm}"))
                cur.execute("""
                    INSERT INTO words (lemma, normalized_lemma, language_code, tags, search_text)
                    VALUES (%s, %s, %s, %s, to_tsvector('simple', %s))
                    ON CONFLICT ON CONSTRAINT words_lang_lemma_uniq
                    DO UPDATE SET 
                        lemma = EXCLUDED.lemma,
                        tags = EXCLUDED.tags,
                        search_text = to_tsvector('simple', EXCLUDED.lemma || ' ' || EXCLUDED.normalized_lemma),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (lemma, norm, lang, "", search_text))
                
                word_id = cur.fetchone()[0]
                
                # Verify the word was created
                cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
                if cur.fetchone():
                    existing[(lemma, lang)] = word_id
                
            except Exception as e:
                logger.error(f"Error processing entry {lemma}: {str(e)}")
                continue
        
        # Update result with all existing entries
        result.update(existing)
        
    return result

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------
def process_kwf_dictionary(cur, filename: str):
    """
    Process the KWF Dictionary JSON file and store its entries in the database with enhanced data extraction.
    
    The file is expected to be a dictionary where each key is a word and each value is a 
    dictionary with the following (example) structure:
    
    {
        "word_key": {
            "original": "word_original",
            "formatted": "word_formatted",
            "metadata": {
                "etymology": [...],
                "source_language": [...],
                "pronunciation": [...],
                "cross_references": []
            },
            "part_of_speech": ["Pangatnig"],
            "definitions": {
                "Pangatnig": [
                    {
                        "number": null,
                        "categories": ["General"],
                        "meaning": "Definition text...",
                        "sub_definitions": [],
                        "example_sets": [ ... ],
                        "note": null,
                        "see": null,
                        "cross_references": [],
                        "synonyms": [],
                        "synonyms_html": null,
                        "antonyms": [],
                        "antonyms_html": null
                    }
                ]
            },
            "affixation": [],
            "idioms": [],
            "related": { ... },
            "other_sections": {}
        },
        ...
    }
    
    This function uses the standardized source name "KWF Diksiyonaryo ng Wikang Filipino" 
    (via SourceStandardization) when inserting definitions.
    
    It captures rich data including:
    - Etymology and source language
    - Definition categories as tags
    - Synonyms and related terms as relations with metadata
    - Affixation relationships
    - Examples and usage notes
    - Cross-references and "see" references
    """
    standardized_source = SourceStandardization.standardize_sources('kwf_dictionary.json')
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If the loaded data is not a dictionary, assume it's a list of entries
    if not isinstance(data, dict):
        for entry in data:
            if not isinstance(entry, dict) or 'word' not in entry:
                continue
            word = entry['word']
            word_id = get_or_create_word_id(cur, word, language_code='tl')
            definitions = entry.get('definitions', {})
            if isinstance(definitions, dict):
                for pos, def_list in definitions.items():
                    for def_entry in def_list:
                        meaning = def_entry.get('meaning', '')
                        if meaning:
                            insert_definition(
                                cur,
                                word_id,
                                meaning,
                                part_of_speech=pos,
                                sources=standardized_source
                            )
        return  # Done processing the list format
    
    # Process the dictionary format (words as keys)
    for word, entry in data.items():
        if not isinstance(entry, dict):
            continue
            
        # Get both original and formatted versions
        original_word = entry.get('original', word)
        formatted_word = entry.get('formatted', original_word)
        
        # Prefer using the formatted word but fall back to original
        word_id = get_or_create_word_id(cur, formatted_word, language_code='tl')
        
        # Process etymology with structure
        if 'metadata' in entry:
            # Extract and process etymology
            if 'etymology' in entry['metadata'] and entry['metadata']['etymology']:
                etymology_text = ""
                etymology_structure = {
                    'raw_data': entry['metadata']['etymology'],
                    'source_language': entry['metadata'].get('source_language', []),
                    'pronunciation': entry['metadata'].get('pronunciation', [])
                }
                
                # Extract plain text from etymology
                for etym in entry['metadata']['etymology']:
                    if 'value' in etym:
                        if etymology_text:
                            etymology_text += ", "
                        etymology_text += etym['value']
                
                if etymology_text:
                    insert_etymology(
                        cur, 
                        word_id, 
                        etymology_text, 
                        etymology_structure=json.dumps(etymology_structure),
                        sources=standardized_source
                    )
                    
            # Store source language as part of etymology if no etymology text
            elif 'source_language' in entry['metadata'] and entry['metadata']['source_language'] and len(entry['metadata']['source_language']) > 0:
                source_langs = []
                for src_lang in entry['metadata']['source_language']:
                    if isinstance(src_lang, dict) and 'value' in src_lang:
                        source_langs.append(src_lang['value'])
                
                if source_langs:
                    source_text = "From " + ", ".join(source_langs)
                    etymology_structure = {
                        'raw_data': [],
                        'source_language': entry['metadata']['source_language'],
                        'pronunciation': entry['metadata'].get('pronunciation', [])
                    }
                    
                    insert_etymology(
                        cur, 
                        word_id, 
                        source_text, 
                        etymology_structure=json.dumps(etymology_structure),
                        sources=standardized_source
                    )
        
        # Process definitions by part of speech
        definitions = entry.get('definitions', {})
        if isinstance(definitions, dict):
            for pos, def_list in definitions.items():
                # Standardize part of speech
                std_pos = standardize_entry_pos(pos)
                
                if not isinstance(def_list, list):
                    continue
                
                for def_entry in def_list:
                    # Handle "see" references - create relations
                    if def_entry.get('see'):
                        for see_ref in def_entry.get('see', []):
                            if not isinstance(see_ref, dict) or 'term' not in see_ref:
                                continue
                                
                            related_term = see_ref.get('term')
                            if not related_term:
                                continue
                                
                            # Create the related word
                            related_id = get_or_create_word_id(cur, related_term, language_code='tl')
                            
                            # Add metadata for the relationship
                            metadata = {
                                'see_context': def_entry.get('see_context'),
                                'relationship_type': 'see_reference',
                                'link': see_ref.get('link', ''),
                                'broken': see_ref.get('broken', False)
                            }
                            
                            # Insert relation
                            relation_type = 'related'
                            insert_relation(cur, word_id, related_id, relation_type, sources=standardized_source, metadata=metadata)

def process_tagalog_words(cur, filename: str):
    """
    Process tagalog-words.json with the enhanced structure:
    {
        "word1": {
            "word": "word1",
            "pronunciation": "...",
            "alternate_pronunciation": "...",
            "part_of_speech": [["pos1"], ["pos2"]],
            "domains": ["domain1", "domain2"],
            "etymology": {
                "raw": "[ Lang ]",
                "languages": ["Lang"],
                "full_language_names": ["Language"],
                "other_terms": [],
                "terms": []
            },
            "derivative": "derivative1, derivative2",
            "senses": [
                {
                    "counter": "1",
                    "definition": "definition text",
                    "example": {
                        "raw": "example text",
                        "examples": ["example1", "example2"]
                    },
                    "synonyms": ["SYN1", "SYN2"],
                    "references": ["REF1", "REF2"],
                    "variants": ["var1", "var2"],
                    "etymology": {...},
                    "part_of_speech": ["pos"],
                    "affix_forms": ["form1", "form2"],
                    "affix_types": ["type1", "type2"]
                }
            ]
        },
        "word2": { ... }
    }
    """
    logger.info(f"Processing Tagalog words from: {filename}")
    source = SourceStandardization.standardize_sources('tagalog-words.json')
    language_code = 'tl'
    romanizer = BaybayinRomanizer()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Track statistics
        total_entries = len(data)
        processed_entries = 0
        skipped_entries = 0
        definitions_added = 0
        relations_added = 0
        synonyms_added = 0
        references_added = 0
        variants_added = 0
        etymologies_processed = 0
        errors = 0
        
        logger.info(f"Found {total_entries} entries to process")
        
        # Process each word entry
        for lemma, entry_data in tqdm(data.items(), desc="Processing Tagalog words"):
            try:
                # Ensure the entry has the basic required fields
                if not entry_data or not isinstance(entry_data, dict):
                    logger.warning(f"Skipping invalid entry for '{lemma}': not a dictionary")
                    skipped_entries += 1
                    continue
                
                # Create or get word ID
                word_id = get_or_create_word_id(cur, lemma, language_code=language_code)
                
                # Extract tags and domains
                tags = []
                
                # Add domains as tags if present
                if 'domains' in entry_data and entry_data['domains']:
                    tags.extend(entry_data['domains'])
                
                # Update word with tags if present
                if tags:
                    cur.execute("""
                        UPDATE words 
                        SET tags = %s
                        WHERE id = %s
                    """, (", ".join(tags), word_id))
                
                # Add pronunciation data if available
                pronunciation_data = {}
                if 'pronunciation' in entry_data and entry_data['pronunciation']:
                    pronunciation_data['primary'] = entry_data['pronunciation']
                if 'alternate_pronunciation' in entry_data and entry_data['alternate_pronunciation']:
                    pronunciation_data['alternate'] = entry_data['alternate_pronunciation']
                
                if pronunciation_data:
                    cur.execute("""
                        UPDATE words
                        SET pronunciation_data = %s
                        WHERE id = %s
                    """, (json.dumps(pronunciation_data), word_id))
                
                # Process derivative information
                if 'derivative' in entry_data and entry_data['derivative']:
                    derivative_text = entry_data['derivative']
                    # Store as metadata
                    cur.execute("""
                        UPDATE words
                        SET metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{derivative}',
                            %s::jsonb
                        )
                        WHERE id = %s
                    """, (json.dumps(derivative_text), word_id))
                    
                    # Process derivative forms if they exist
                    derivative_forms = derivative_text.split(',')
                    for form in derivative_forms:
                        form = form.strip()
                        if form:
                            # Check if it's a structured form with type
                            form_parts = form.split(' ')
                            if len(form_parts) > 1:
                                derivative_id = get_or_create_word_id(cur, form, language_code=language_code)
                                insert_relation(cur, word_id, derivative_id, 'derived', sources=source)
                
                # Process etymology at word level
                if 'etymology' in entry_data and entry_data['etymology'] and isinstance(entry_data['etymology'], dict):
                    etymology_data = entry_data['etymology']
                    etymology_text = etymology_data.get('raw', '')
                    
                    # Extract structured etymology data
                    etymology_structure = {
                        'languages': etymology_data.get('languages', []),
                        'full_language_names': etymology_data.get('full_language_names', []),
                        'other_terms': etymology_data.get('other_terms', []),
                        'terms': etymology_data.get('terms', [])
                    }
                    
                    if etymology_text:
                        language_codes = ", ".join(etymology_data.get('languages', []))
                        insert_etymology(
                            cur, 
                            word_id, 
                            etymology_text, 
                            etymology_structure=json.dumps(etymology_structure),
                            language_codes=language_codes,
                            sources=source
                        )
                        etymologies_processed += 1
                
                # Process senses (definitions)
                if 'senses' in entry_data and entry_data['senses']:
                    for sense in entry_data['senses']:
                        if not isinstance(sense, dict) or 'definition' not in sense:
                            continue
                        
                        definition_text = sense.get('definition', '').strip()
                        if not definition_text:
                            continue
                        
                        # Add counter information if present
                        counter = sense.get('counter', '')
                        if counter:
                            definition_text = f"[{counter}] {definition_text}"
                        
                        # Extract part of speech for this specific sense
                        pos = ''
                        if 'part_of_speech' in sense and sense['part_of_speech']:
                            pos = ", ".join(sense['part_of_speech'])
                        # Fall back to entry-level part of speech if not defined at sense level
                        elif 'part_of_speech' in entry_data:
                            pos_list = entry_data['part_of_speech']
                            if pos_list and isinstance(pos_list, list):
                                # Flatten potentially nested arrays
                                flat_pos = []
                                for pos_item in pos_list:
                                    if isinstance(pos_item, list):
                                        flat_pos.extend(pos_item)
                                    else:
                                        flat_pos.append(pos_item)
                                pos = ", ".join(flat_pos)
                        
                        # Extract examples if present
                        examples = []
                        example_data = sense.get('example', {})
                        if isinstance(example_data, dict):
                            # Try to get pre-parsed examples
                            if 'examples' in example_data and example_data['examples']:
                                examples.extend(example_data['examples'])
                            # If no pre-parsed examples, use the raw text
                            elif 'raw' in example_data and example_data['raw']:
                                examples.append(example_data['raw'])
                        
                        # Combine examples into a single string
                        examples_text = "; ".join(examples) if examples else None
                        
                        # Extract usage notes if present
                        usage_notes = None
                        if 'usage_notes' in sense and sense['usage_notes']:
                            usage_notes = sense['usage_notes']
                        
                        # Extract category if present
                        category = None
                        if 'category' in sense and sense['category']:
                            category = sense['category']
                        
                        # Get additional tags specific to this sense
                        sense_tags = []
                        if 'tags' in sense and sense['tags']:
                            sense_tags.extend(sense['tags'])
                            
                        # Add counter as tag if present
                        if counter:
                            sense_tags.append(f"counter:{counter}")
                            
                        # Add etymology to tags if present at sense level
                        if 'etymology' in sense and sense['etymology']:
                            sense_etymology = sense['etymology']
                            if isinstance(sense_etymology, dict) and 'raw' in sense_etymology:
                                sense_tags.append(f"etymology:{sense_etymology['raw']}")
                                
                                # Process specific sense etymology
                                etymology_structure = {
                                    'languages': sense_etymology.get('languages', []),
                                    'full_language_names': sense_etymology.get('full_language_names', []),
                                    'other_terms': sense_etymology.get('other_terms', []),
                                    'terms': sense_etymology.get('terms', [])
                                }
                                
                                sense_etymology_text = sense_etymology.get('raw', '')
                                if sense_etymology_text:
                                    language_codes = ", ".join(sense_etymology.get('languages', []))
                                    # Store as metadata
                                    sense_tags.append(f"etymology_languages:{language_codes}")
                        
                        # Combine tags into a single string
                        sense_tags_text = ", ".join(sense_tags) if sense_tags else None
                        
                        # Insert definition
                        definition_id = insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            part_of_speech=pos,
                            examples=examples_text,
                            usage_notes=usage_notes,
                            category=category,
                            tags=sense_tags_text,
                            sources=source
                        )
                        
                        if definition_id:
                            definitions_added += 1
                            
                            # Process synonyms if present
                            if 'synonyms' in sense and sense['synonyms']:
                                for synonym in sense['synonyms']:
                                    if synonym:
                                        # Cleanup synonym text - it could be in all caps or have special formatting
                                        syn_text = synonym.strip()
                                        
                                        # Create the synonym word
                                        syn_id = get_or_create_word_id(cur, syn_text, language_code=language_code)
                                        
                                        # Create both directions of the relationship
                                        insert_relation(cur, word_id, syn_id, 'synonym', sources=source)
                                        insert_relation(cur, syn_id, word_id, 'synonym', sources=source)
                                        
                                        # Also link the definition to the synonym
                                        insert_definition_relation(cur, definition_id, syn_id, 'synonym', sources=source)
                                        
                                        synonyms_added += 1
                            
                            # Process references if present
                            if 'references' in sense and sense['references']:
                                for reference in sense['references']:
                                    if reference:
                                        ref_text = reference.strip()
                                        
                                        # Create the reference word
                                        ref_id = get_or_create_word_id(cur, ref_text, language_code=language_code)
                                        
                                        # Create reference relationship (this is a looser connection than synonym)
                                        insert_relation(cur, word_id, ref_id, 'related', sources=source)
                                        
                                        # Also link the definition to the reference
                                        insert_definition_relation(cur, definition_id, ref_id, 'related', sources=source)
                                        
                                        references_added += 1
                            
                            # Process variants if present
                            if 'variants' in sense and sense['variants']:
                                for variant in sense['variants']:
                                    if variant:
                                        # Clean up variant text - it could include additional information
                                        # like "variant1 Cf variant2" where Cf means "compare with"
                                        variant_parts = variant.split(' Cf ')
                                        variant_text = variant_parts[0].strip()
                                        
                                        # Create the variant word
                                        variant_id = get_or_create_word_id(cur, variant_text, language_code=language_code)
                                        
                                        # Create bidirectional variant relationship
                                        insert_relation(cur, word_id, variant_id, 'variant', sources=source)
                                        insert_relation(cur, variant_id, word_id, 'variant', sources=source)
                                        
                                        # Also link the definition to the variant
                                        insert_definition_relation(cur, definition_id, variant_id, 'variant', sources=source)
                                        
                                        variants_added += 1
                                        
                                        # If there's a "Cf" part, also create a relationship to that word
                                        if len(variant_parts) > 1:
                                            cf_text = variant_parts[1].strip()
                                            cf_id = get_or_create_word_id(cur, cf_text, language_code=language_code)
                                            
                                            # Create a "compare with" relationship
                                            insert_relation(cur, word_id, cf_id, 'compare_with', sources=source)
                                            
                                            # Also link the definition to the comparison
                                            insert_definition_relation(cur, definition_id, cf_id, 'compare_with', sources=source)
                                            
                                            references_added += 1
                            
                            # Process affix forms if present
                            if ('affix_forms' in sense and sense['affix_forms'] and 
                                'affix_types' in sense and sense['affix_types']):
                                
                                forms = sense['affix_forms']
                                types = sense['affix_types']
                                
                                for i, form in enumerate(forms):
                                    if i < len(types):
                                        affix_type = types[i]
                                    else:
                                        affix_type = 'unknown'
                                    
                                    # Normalize form (remove potential dots)
                                    clean_form = form.replace('.', '')
                                    
                                    # Create the affixed word
                                    affixed_id = get_or_create_word_id(cur, clean_form, language_code=language_code)
                                    
                                    # Create affixation relationship
                                    insert_affixation(
                                        cur,
                                        root_id=word_id,
                                        affixed_id=affixed_id,
                                        affix_type=affix_type,
                                        sources=source
                                    )
                                    
                                    # Also create a relation
                                    # Define metadata for the relationship
                                    metadata = {"affix_type": affix_type}
                                    insert_relation(
                                        cur, 
                                        word_id, 
                                        affixed_id, 
                                        'derived', 
                                        sources=source,
                                        metadata=metadata
                                    )
                
                processed_entries += 1
                if processed_entries % 1000 == 0:
                    logger.info(f"Processed {processed_entries}/{total_entries} entries")
                    
            except Exception as e:
                errors += 1
                logger.error(f"Error processing entry '{lemma}': {str(e)}")
                continue
                
        # Log statistics
        logger.info(f"Tagalog words processing complete:")
        logger.info(f"  Total entries: {total_entries}")
        logger.info(f"  Processed: {processed_entries}")
        logger.info(f"  Skipped: {skipped_entries}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Definitions added: {definitions_added}")
        logger.info(f"  Synonyms added: {synonyms_added}")
        logger.info(f"  References added: {references_added}")
        logger.info(f"  Variants added: {variants_added}")
        logger.info(f"  Relations added: {relations_added}")
        logger.info(f"  Etymologies processed: {etymologies_processed}")
        
    except Exception as e:
        logger.error(f"Error processing Tagalog words file: {str(e)}")
        raise

def process_root_words_cleaned(cur, filename: str):
    """Process root words data from a cleaned JSON file."""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = 0
    skipped = 0
    errors = 0
    source = get_standardized_source(os.path.basename(filename))
    total = len(data)
    
    for root_word, associated_words in tqdm(data.items(), desc="Processing root words"):
        try:
            # Skip empty root word
            if not root_word:
                skipped += 1
                continue
            
            # Get the root word ID
            root_id = get_or_create_word_id(cur, root_word, 'tl')
            
            # Process associated words (which are the nested dictionaries)
            for assoc_word, word_data in associated_words.items():
                if not assoc_word or assoc_word == root_word:
                    continue
                
                # Create the associated word
                word_type = word_data.get('type', '')
                definition = word_data.get('definition', '')
                
                # Remove the ellipsis from the end of the definition
                if definition.endswith('...'):
                    definition = definition[:-3]
                
                standardized_pos = standardize_entry_pos(word_type)
                
                # Create the word entry
                derived_id = get_or_create_word_id(cur, assoc_word, 'tl', root_word_id=root_id)
                
                # Add definition if available
                if definition:
                    insert_definition(
                        cur, derived_id, definition, part_of_speech=standardized_pos,
                        sources=source
                    )
                
                # Add the relationship if it's not the root word itself
                if assoc_word != root_word:
                    insert_relation(
                        cur, derived_id, root_id, 'derived_from', source
                    )
            
            # Also add the root word's own definition if it exists in the associated words
            if root_word in associated_words:
                root_data = associated_words[root_word]
                root_type = root_data.get('type', '')
                root_definition = root_data.get('definition', '')
                
                # Remove the ellipsis from the end of the root definition
                if root_definition.endswith('...'):
                    root_definition = root_definition[:-3]
                
                standardized_pos = standardize_entry_pos(root_type)
                
                if root_definition:
                    insert_definition(
                        cur, root_id, root_definition, part_of_speech=standardized_pos,
                        sources=source
                    )
            
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{total} root word entries")
                
        except Exception as e:
            errors += 1
            logger.error(f"Error processing root word {root_word}: {str(e)}")
    
    logger.info(f"Root words processing complete: {processed} processed, {skipped} skipped, {errors} errors")
    
def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    lang_map = {
        "Esp": "es", "Eng": "en", "Ch": "zh", "Tsino": "zh", "Jap": "ja",
        "San": "sa", "Sanskrit": "sa", "Tag": "tl", "Mal": "ms", "Arb": "ar"
    }
    return [lang_map[lang] for lang in lang_map if lang in etymology]

@with_transaction(commit=True)
def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Process and create relationships from definition text."""
    synonym_patterns = [r'ka(singkahulugan|tulad|patulad|tumbas) ng\s+(\w+)', r'singkahulugan:\s+(\w+)']
    antonym_patterns = [r'(kasalungat|kabaligtaran) ng\s+(\w+)', r'kabaligtaran:\s+(\w+)']
    
    for pattern in synonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            syn = match.group(2).strip()
            syn_id = get_or_create_word_id(cur, syn, 'tl')
            insert_relation(cur, word_id, syn_id, "synonym", sources=source)
    
    for pattern in antonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            ant = match.group(2).strip()
            ant_id = get_or_create_word_id(cur, ant, 'tl')
            insert_relation(cur, word_id, ant_id, "antonym", sources=source)

# ---------------------------------------------------------------a----
# Robust Relationship Mapping for Kaikki
# -------------------------------------------------------------------
RELATION_MAPPING = {
    "synonym": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "antonym": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "hyponym_of": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of"
    },
    "hypernym_of": {
        "relation_type": "hypernym_of",
        "bidirectional": False,
        "inverse": "hyponym_of"
    },
    "see_also": {
        "relation_type": "see_also",
        "bidirectional": True,
        "inverse": "see_also"
    },
    "compare_with": {
        "relation_type": "compare_with",
        "bidirectional": True,
        "inverse": "compare_with"
    },
    "derived_from": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of"
    },
    "descendant": {
        "relation_type": "descendant_of",
        "bidirectional": False,
        "inverse": "ancestor_of"
    },
    "borrowed": {
        "relation_type": "borrowed_from",
        "bidirectional": False,
        "inverse": "loaned_to"
    },
    "variant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "alt_of": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "abbreviation_of": {
        "relation_type": "abbreviation_of",
        "bidirectional": False,
        "inverse": "has_abbreviation"
    },
    "initialism_of": {
        "relation_type": "initialism_of",
        "bidirectional": False,
        "inverse": "has_initialism"
    },
    "related": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related"
    },
    "derived": {
        "relation_type": "derived",
        "bidirectional": False,
        "inverse": "derives"
    },
    "contraction_of": {
        "relation_type": "contraction_of",
        "bidirectional": False,
        "inverse": "contracts_to"
    },
    "alternate_form": {
        "relation_type": "alternate_form",
        "bidirectional": True,
        "inverse": "alternate_form"
    },
    "regional_form": {
        "relation_type": "regional_form",
        "bidirectional": True,
        "inverse": "regional_form"
    },
    "modern_form": {
        "relation_type": "modern_form",
        "bidirectional": False,
        "inverse": "archaic_form"
    },
    "archaic_form": {
        "relation_type": "archaic_form",
        "bidirectional": False,
        "inverse": "modern_form"
    },
    "obsolete_spelling": {
        "relation_type": "obsolete_spelling",
        "bidirectional": False,
        "inverse": "current_spelling"
    },
    "alternative_spelling": {
        "relation_type": "alternative_spelling",
        "bidirectional": True,
        "inverse": "alternative_spelling"
    },
    "root_of": {
        "relation_type": "root_of",
        "bidirectional": False,
        "inverse": "derived_from"
    },
    "cognate": {
        "relation_type": "cognate",
        "bidirectional": True,
        "inverse": "cognate" 
    },
    # Filipino-specific relationship types
    "kasingkahulugan": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "kasalungat": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "kabaligtaran": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym"
    },
    "katulad": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym"
    },
    "uri_ng": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of"
    },
    "mula_sa": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of"
    },
    "varyant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant"
    },
    "kaugnay": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related"
    }
}

def normalize_relation_type(raw_key: str) -> Tuple[str, bool, Optional[str]]:
    """
    Normalize a relationship type key and return its standard form, whether it's bidirectional,
    and its inverse relationship type.
    
    Args:
        raw_key: The raw relationship type key to normalize.
        
    Returns:
        Tuple containing:
        - Normalized relationship type string
        - Boolean indicating if relationship is bidirectional
        - Optional inverse relationship type (None if no inverse defined)
    """
    if not raw_key:
        return "related", True, "related"  # Default to a generic related relationship
        
    key_lower = raw_key.lower().strip()
    
    # Remove trailing 's' if plural form
    if key_lower.endswith("s") and len(key_lower) > 1:
        key_lower = key_lower[:-1]  # E.g. "synonyms" -> "synonym"
    
    # Special case handling for Tagalog relationship indicators
    if "kasingkahulugan" in key_lower or "katulad ng" in key_lower:
        return "synonym", True, "synonym"
    if "kasalungat" in key_lower or "kabaligtaran" in key_lower:
        return "antonym", True, "antonym" 
    if "uri ng" in key_lower:
        return "hyponym_of", False, "hypernym_of"
    if "mula sa" in key_lower:
        return "derived_from", False, "root_of"
    if "varyant" in key_lower or "variant" in key_lower:
        return "variant", True, "variant"
    
    # Check direct match in mapping
    if key_lower in RELATION_MAPPING:
        info = RELATION_MAPPING[key_lower]
        return info["relation_type"], info.get("bidirectional", False), info.get("inverse")
    
    # Try to find partial matches in keys
    for known_key, info in RELATION_MAPPING.items():
        if known_key in key_lower or key_lower in known_key:
            return info["relation_type"], info.get("bidirectional", False), info.get("inverse")
    
    # If no match found, default to a related relationship but preserve the original key
    logger.info(f"Unknown relationship type: {raw_key}, defaulting to 'related'")
    return "related", True, "related"

@with_transaction(commit=True)
def process_relationships(cur, word_id, data, sources):
    """Process relationship data for a word."""
    if not data or not word_id:
        return
    
    try:
        # Extract relationships from definitions
        definitions = []
        if 'definitions' in data and isinstance(data['definitions'], list):
            definitions = [d.get('text') if isinstance(d, dict) else d for d in data['definitions']]
        elif 'definition' in data and data['definition']:
            definitions = [data['definition']]
        
        # Process each definition for relationships
        for definition in definitions:
            if not definition or not isinstance(definition, str):
                continue
                
            # Look for synonyms in definitions
            syn_patterns = [
                r'ka(singkahulugan|tulad) ng\s+(\w+)',               # Tagalog
                r'syn(onym)?[\.:]?\s+(\w+)',                         # English
                r'same\s+as\s+(\w+)',                                # English
                r'another\s+term\s+for\s+(\w+)',                     # English
                r'another\s+word\s+for\s+(\w+)',                     # English
                r'also\s+called\s+(\w+)',                            # English
                r'also\s+known\s+as\s+(\w+)',                        # English
            ]
            
            for pattern in syn_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    synonym = match[1] if isinstance(match, tuple) and len(match) > 1 else match
                    if synonym and isinstance(synonym, str):
                        # Get or create the synonym word
                        lang_code = data.get('language_code', 'tl')
                        syn_id = get_or_create_word_id(cur, synonym.strip(), language_code=lang_code)
                        
                        # Insert the synonym relationship
                        insert_relation(cur, word_id, syn_id, "synonym", sources=sources)
                        
                        # For bidirectional synonyms
                        insert_relation(cur, syn_id, word_id, "synonym", sources=sources)
            
            # Look for antonyms in definitions
            ant_patterns = [
                r'(kasalungat|kabaligtaran) ng\s+(\w+)',             # Tagalog
                r'ant(onym)?[\.:]?\s+(\w+)',                         # English
                r'opposite\s+of\s+(\w+)',                            # English
                r'contrary\s+to\s+(\w+)',                            # English
            ]
            
            for pattern in ant_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    antonym = match[1] if isinstance(match, tuple) and len(match) > 1 else match
                    if antonym and isinstance(antonym, str):
                        # Get or create the antonym word
                        lang_code = data.get('language_code', 'tl')
                        ant_id = get_or_create_word_id(cur, antonym.strip(), language_code=lang_code)
                        
                        # Insert the antonym relationship
                        insert_relation(cur, word_id, ant_id, "antonym", sources=sources)
                        
                        # For bidirectional antonyms
                        insert_relation(cur, ant_id, word_id, "antonym", sources=sources)
            
            # Look for hypernyms (broader terms) in definitions
            hyper_patterns = [
                r'uri ng\s+(\w+)',                                  # Tagalog
                r'type of\s+(\w+)',                                 # English 
                r'kind of\s+(\w+)',                                 # English
                r'form of\s+(\w+)',                                 # English
                r'variety of\s+(\w+)',                              # English
                r'species of\s+(\w+)',                              # English
                r'member of\s+the\s+(\w+)\s+family',                # English
            ]
            
            for pattern in hyper_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    hypernym = match
                    if hypernym and isinstance(hypernym, str):
                        # Get or create the hypernym word
                        lang_code = data.get('language_code', 'tl')
                        hyper_id = get_or_create_word_id(cur, hypernym.strip(), language_code=lang_code)
                        
                        # Insert the hypernym relationship
                        insert_relation(cur, word_id, hyper_id, "hyponym_of", sources=sources)
                        
                        # Add the inverse relationship
                        insert_relation(cur, hyper_id, word_id, "hypernym_of", sources=sources)
            
            # Look for variations in definitions
            var_patterns = [
                r'(iba\'t ibang|ibang) (anyo|baybay|pagsulat|bigkas) ng\s+(\w+)',  # Tagalog: different form/spelling/pronunciation of
                r'(alternatibo|alternativ|kahalili) ng\s+(\w+)',                    # Tagalog: alternative of
                r'(variant|variation|alt(ernative)?) (form|spelling) of\s+(\w+)',   # English
                r'alternative (to|for)\s+(\w+)',                                    # English
                r'also (written|spelled) as\s+(\w+)',                               # English
                r'(var\.|variant)\s+(\w+)',                                         # English abbreviated
                r'(regional|dialectal) form of\s+(\w+)',                            # English regional variant
                r'(slang|informal) for\s+(\w+)',                                    # English slang variant
                r'commonly (misspelled|written) as\s+(\w+)',                        # English common misspelling
                r'(baryant|lokal na anyo) ng\s+(\w+)',                              # Tagalog regional variant
            ]
            
            for pattern in var_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    # Different patterns have target word in different positions
                    variant = None
                    if len(match) == 3 and isinstance(match, tuple):  # For patterns with 3 capture groups
                        variant = match[2]
                    elif len(match) == 2 and isinstance(match, tuple):  # For patterns with 2 capture groups
                        variant = match[1]
                    elif isinstance(match, str):  # For patterns with 1 capture group
                        variant = match
                        
                    if variant and isinstance(variant, str):
                        # Get or create the variant word
                        lang_code = data.get('language_code', 'tl')
                        var_id = get_or_create_word_id(cur, variant.strip(), language_code=lang_code)
                        
                        # Insert the variant relationship
                        insert_relation(cur, word_id, var_id, "variant", sources=sources)
                        
                        # For bidirectional variant relationship
                        insert_relation(cur, var_id, word_id, "variant", sources=sources)
        
        # Process derivative information
        derivative = data.get('derivative', '')
        if derivative and isinstance(derivative, str):
            # This indicates the word is derived from another root
            mula_sa_patterns = [
                r'mula sa\s+(.+?)(?:\s+na|\s*$)',                   # Tagalog
                r'derived from\s+(?:the\s+)?(\w+)',                 # English
                r'comes from\s+(?:the\s+)?(\w+)',                   # English
                r'root word(?:\s+is)?\s+(\w+)',                     # English
            ]
            
            for pattern in mula_sa_patterns:
                root_match = re.search(pattern, derivative, re.IGNORECASE)
                if root_match:
                    root_word = root_match.group(1).strip()
                    if root_word:
                        # Get or create the root word
                        lang_code = data.get('language_code', 'tl')
                        root_id = get_or_create_word_id(cur, root_word, language_code=lang_code)
                        
                        # Insert the derived_from relationship
                        insert_relation(cur, word_id, root_id, "derived_from", sources=sources)
                        
                        # Add the inverse relationship
                        insert_relation(cur, root_id, word_id, "root_of", sources=sources)
        
        # Process etymology information for potential language relationships
        etymology = data.get('etymology', '')
        if etymology and isinstance(etymology, str):
            # Try to extract language information from etymology
            lang_patterns = {
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Spanish|Esp)[\.:]?\s+(\w+)': 'es',  # Spanish
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:English|Eng)[\.:]?\s+(\w+)': 'en',  # English
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Chinese|Ch|Tsino)[\.:]?\s+(\w+)': 'zh',  # Chinese
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Japanese|Jap)[\.:]?\s+(\w+)': 'ja',  # Japanese
                r'(?:from|borrowed from)\s+(?:the\s+)?(?:Sanskrit|San)[\.:]?\s+(\w+)': 'sa',  # Sanskrit
            }
            
            for pattern, lang_code in lang_patterns.items():
                lang_matches = re.findall(pattern, etymology, re.IGNORECASE)
                for lang_word in lang_matches:
                    if lang_word and isinstance(lang_word, str):
                        # Get or create the foreign word
                        foreign_id = get_or_create_word_id(cur, lang_word.strip(), language_code=lang_code)
                        
                        # Insert the etymology relationship
                        insert_relation(cur, word_id, foreign_id, "borrowed_from", sources=sources)
        
        # Process alternate forms and variations
        # Check if there's variations data in metadata
        variations = data.get('variations', [])
        if variations and isinstance(variations, list):
            for variant in variations:
                if isinstance(variant, str) and variant.strip():
                    # Add this explicit variation
                    var_id = get_or_create_word_id(cur, variant.strip(), language_code=data.get('language_code', 'tl'))
                    insert_relation(cur, word_id, var_id, "variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "variant", sources=sources)
                elif isinstance(variant, dict) and 'form' in variant:
                    var_form = variant.get('form', '').strip()
                    if var_form:
                        var_type = variant.get('type', 'variant')
                        var_id = get_or_create_word_id(cur, var_form, language_code=data.get('language_code', 'tl'))
                        
                        # Use specific relationship type if provided, otherwise default to "variant"
                        rel_type = var_type if var_type in ["abbreviation", "misspelling", "regional", "alternate", "dialectal"] else "variant"
                        insert_relation(cur, word_id, var_id, rel_type, sources=sources)
                        insert_relation(cur, var_id, word_id, rel_type, sources=sources)
        
        # Look for variations by checking spelling differences
        # This will detect common spelling variations in Filipino like f/p, e/i, o/u substitutions
        word = data.get('word', '')
        if word and isinstance(word, str) and len(word) > 3:
            # Common letter substitutions in Filipino
            substitutions = [
                ('f', 'p'), ('p', 'f'),  # Filipino/Pilipino
                ('e', 'i'), ('i', 'e'),  # like in leeg/liig (neck)
                ('o', 'u'), ('u', 'o'),  # like in puso/poso (heart)
                ('k', 'c'), ('c', 'k'),  # like in karera/carera (race)
                ('w', 'u'), ('u', 'w'),  # like in uwi/uwi (go home)
                ('j', 'h'), ('h', 'j'),  # like in jahit/hahit
                ('s', 'z'), ('z', 's'),  # like in kasoy/kazoy
                ('ts', 'ch'), ('ch', 'ts'),  # like in tsaa/chaa (tea)
            ]
            
            # Generate possible variations
            potential_variations = []
            for i, char in enumerate(word):
                for orig, repl in substitutions:
                    if char.lower() == orig:
                        var = word[:i] + repl + word[i+1:]
                        potential_variations.append(var)
                    elif char.lower() == orig[0] and i < len(word) - 1 and word[i+1].lower() == orig[1]:
                        var = word[:i] + repl + word[i+2:]
                        potential_variations.append(var)
            
            # Check if these variations actually exist in the database
            for var in potential_variations:
                # Skip if the variation is the same as the original word
                if var.lower() == word.lower():
                    continue
                    
                cur.execute("""
                    SELECT id FROM words 
                    WHERE normalized_lemma = %s AND language_code = %s
                """, (normalize_lemma(var), data.get('language_code', 'tl')))
                
                result = cur.fetchone()
                if result:
                    # Found a real variation in the database
                    var_id = result[0]
                    insert_relation(cur, word_id, var_id, "spelling_variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "spelling_variant", sources=sources)
    
    except Exception as e:
        logger.error(f"Error processing relationships for word_id {word_id}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            tb_str = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"Traceback: {tb_str}")

@with_transaction(commit=True)
def process_direct_relations(cur, word_id, entry, lang_code, source):
    """Process direct relationships specified in the entry."""
    relationship_mappings = {
        'synonyms': ('synonym', True),  # bidirectional
        'antonyms': ('antonym', True),  # bidirectional
        'derived': ('derived_from', False),  # not bidirectional, direction is important
        'related': ('related', True)    # bidirectional
    }
    
    for rel_key, (rel_type, bidirectional) in relationship_mappings.items():
        if rel_key in entry and isinstance(entry[rel_key], list):
            for rel_item in entry[rel_key]:
                # Initialize metadata
                metadata = {}
                
                # Handle both string words and dictionary objects with word property
                if isinstance(rel_item, dict) and 'word' in rel_item:
                    rel_word = rel_item['word']
                    
                    # Extract metadata fields if available
                    if 'strength' in rel_item:
                        metadata['strength'] = rel_item['strength']
                    
                    if 'tags' in rel_item and rel_item['tags']:
                        metadata['tags'] = rel_item['tags']
                    
                    if 'english' in rel_item and rel_item['english']:
                        metadata['english'] = rel_item['english']
                        
                    # Extract any other useful fields
                    for field in ['sense', 'extra', 'notes']:
                        if field in rel_item and rel_item[field]:
                            metadata[field] = rel_item[field]
                            
                elif isinstance(rel_item, str):
                    rel_word = rel_item
                else:
                    continue
                        
                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue
                        
                # For derived words, the entry word is derived from the related word
                if rel_type == 'derived_from':
                    from_id = word_id
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source)
                else:
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source)
                        
                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(cur, to_id, word_id, rel_type, sources=source, metadata=metadata)
                        else:
                            insert_relation(cur, to_id, word_id, rel_type, sources=source)

@with_transaction(commit=True)
def process_relations(cur, from_word_id: int, relations_dict: Dict[str, List[str]], lang_code: str, source: str):
    """
    Process relationship data from dictionary.
    Convert each raw Kaikki relation key using normalize_relation_type,
    then create appropriate relations in the database.
    """
    for raw_key, related_list in relations_dict.items():
        if not related_list:
            continue
        # Normalize relation key
        relation_type, bidirectional, inverse_type = normalize_relation_type(raw_key)
        
        for rel_item in related_list:
            metadata = {}
            
            # Handle both string values and dictionary objects
            if isinstance(rel_item, dict) and 'word' in rel_item:
                rel_word_lemma = rel_item['word']
                
                # Extract metadata fields if available
                if 'strength' in rel_item:
                    metadata['strength'] = rel_item['strength']
                
                if 'tags' in rel_item and rel_item['tags']:
                    metadata['tags'] = rel_item['tags']
                
                if 'english' in rel_item and rel_item['english']:
                    metadata['english'] = rel_item['english']
                    
                # Extract any other useful fields
                for field in ['sense', 'extra', 'notes', 'context']:
                    if field in rel_item and rel_item[field]:
                        metadata[field] = rel_item[field]
            elif isinstance(rel_item, str):
                rel_word_lemma = rel_item
            else:
                continue
                
            to_word_id = get_or_create_word_id(cur, rel_word_lemma, language_code=lang_code)
            
            # Only include metadata if it's not empty
            if metadata:
                insert_relation(cur, from_word_id, to_word_id, relation_type, sources=source, metadata=metadata)
            else:
                insert_relation(cur, from_word_id, to_word_id, relation_type, sources=source)

            if bidirectional and inverse_type:
                # For bidirectional relationships, we might want to copy the metadata
                if metadata:
                    insert_relation(cur, to_word_id, from_word_id, inverse_type, sources=source, metadata=metadata)
                else:
                    insert_relation(cur, to_word_id, from_word_id, inverse_type, sources=source)

@with_transaction(commit=True)
def extract_sense_relations(cur, word_id, sense, lang_code, source):
    """Extract and process relationship data from a word sense."""
    for rel_type in ['synonyms', 'antonyms', 'derived', 'related']:
        if rel_type in sense and isinstance(sense[rel_type], list):
            relation_items = sense[rel_type]
            relationship_type = 'synonym' if rel_type == 'synonyms' else 'antonym' if rel_type == 'antonyms' else 'derived_from' if rel_type == 'derived' else 'related'
            bidirectional = rel_type != 'derived'  # derived relationships are not bidirectional
            
            for item in relation_items:
                # Initialize metadata
                metadata = {}
                
                # Handle both string words and dictionary objects with word property
                if isinstance(item, dict) and 'word' in item:
                    rel_word = item['word']
                    
                    # Extract metadata fields if available
                    if 'strength' in item:
                        metadata['strength'] = item['strength']
                    
                    if 'tags' in item and item['tags']:
                        metadata['tags'] = item['tags']
                    
                    if 'english' in item and item['english']:
                        metadata['english'] = item['english']
                        
                    # Extract sense-specific context if available
                    if 'sense' in item and item['sense']:
                        metadata['sense'] = item['sense']
                        
                    # Extract any other useful fields
                    for field in ['extra', 'notes', 'context']:
                        if field in item and item[field]:
                            metadata[field] = item[field]
                elif isinstance(item, str):
                    rel_word = item
                else:
                    continue
                        
                # Skip empty strings
                if not rel_word or not isinstance(rel_word, str):
                    continue
                        
                # For derived words, the entry word is derived from the related word
                if relationship_type == 'derived_from':
                    from_id = word_id
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, from_id, to_id, relationship_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, from_id, to_id, relationship_type, sources=source)
                else:
                    to_id = get_or_create_word_id(cur, rel_word, language_code=lang_code)
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(cur, word_id, to_id, relationship_type, sources=source, metadata=metadata)
                    else:
                        insert_relation(cur, word_id, to_id, relationship_type, sources=source)
                        
                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(cur, to_id, word_id, relationship_type, sources=source, metadata=metadata)
                        else:
                            insert_relation(cur, to_id, word_id, relationship_type, sources=source)
    
def extract_definition_tags(definition_dict):
    """Extract tags from definition data"""
    tags = []
    
    # Extract from directly tagged definitions
    if 'tags' in definition_dict and isinstance(definition_dict['tags'], list):
        tags.extend(definition_dict['tags'])
        
    # Look for tags in glosses field that might indicate usage
    special_tags = ['figuratively', 'figurative', 'colloquial', 'formal', 'informal', 'archaic', 'obsolete', 'rare']
    glosses = definition_dict.get('glosses', [])
    if isinstance(glosses, list) and glosses:
        for gloss in glosses:
            if isinstance(gloss, str):
                # Check if the gloss starts with a tag in parentheses
                tag_match = re.match(r'^\(([^)]+)\)', gloss)
                if tag_match:
                    potential_tag = tag_match.group(1).lower()
                    if potential_tag in special_tags:
                        tags.append(potential_tag)
                        
    return tags if tags else None

def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract Baybayin form and romanized form from an entry."""
    baybayin_form = entry.get('baybayin_form')
    romanized_form = entry.get('romanized_form')
    
    # Return early if no Baybayin form
    if not baybayin_form:
        return None, None
        
    # Validate the Baybayin form
    if validate_baybayin_entry(baybayin_form, romanized_form):
        return baybayin_form, romanized_form
    return None, None

def process_entry(cur, entry: Dict, filename=None):
    """
    Process a single Kaikki entry, with definitions/senses, etymology, relations, etc.
    """
    try:
        word = entry.get("word")
        if not word:
            return
        language_code = entry.get("language_code") or entry.get("lang_code", "tl")

        word_id = get_or_create_word_id(cur, word, language_code=language_code)

        # Extract any Baybayin form
        baybayin_form, romanized_form = extract_baybayin_info(entry)
        if baybayin_form:
            process_baybayin_data(cur, word_id, baybayin_form, romanized_form)

        # Insert definitions from `definitions` or `senses`
        definitions = entry.get('definitions') or entry.get('senses', [])
        for definition in definitions:
            text = None
            pos = entry.get('pos', '')
            examples = None
            usage_notes = None
            tags = None

            if isinstance(definition, dict):
                text = definition.get('text')
                if not text and 'glosses' in definition and isinstance(definition['glosses'], list) and definition['glosses']:
                    text = definition['glosses'][0]
                # sense-level pos override
                if 'pos' in definition:
                    pos = definition['pos']
                if 'examples' in definition:
                    ex_data = definition['examples']
                    if isinstance(ex_data, list):
                        examples = json.dumps(ex_data)
                    else:
                        examples = json.dumps([ex_data])
                if 'usage_notes' in definition:
                    un_data = definition['usage_notes']
                    if isinstance(un_data, list):
                        usage_notes = json.dumps(un_data)
                    else:
                        usage_notes = json.dumps([un_data])
                
                # Extract tags such as "colloquial", "figuratively", etc.
                tags = extract_definition_tags(definition)
                
                # Extract any sense-level relations
                if filename:  # Only call if filename is provided
                    extract_sense_relations(cur, word_id, definition, language_code, 
                                          SourceStandardization.standardize_sources(os.path.basename(filename)))
            else:
                # It's just a string
                text = definition

            if text:
                text = re.sub(r'\.{3,}$', '', text).strip()
                source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
                insert_definition(
                    cur,
                    word_id,
                    text,
                    part_of_speech=standardize_entry_pos(pos),
                    examples=examples,
                    usage_notes=usage_notes,
                    tags=json.dumps(tags) if tags else None,
                    sources=source
                )

        # Insert etymology if present
        if 'etymology' in entry and entry['etymology'].strip():
            ety_text = entry['etymology']
            comps = extract_etymology_components(ety_text)
            
            # Store full etymology structure if available
            etymology_structure = None
            if 'etymology_templates' in entry and isinstance(entry['etymology_templates'], list):
                etymology_structure = json.dumps(entry['etymology_templates'])
            
            source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
            insert_etymology(
                cur,
                word_id,
                ety_text,
                normalized_components=json.dumps(comps) if comps else None,
                etymology_structure=etymology_structure,
                sources=source
            )

        # Process direct relationship arrays
        source = SourceStandardization.standardize_sources(os.path.basename(filename)) if filename else ""
        process_direct_relations(
            cur, 
            word_id, 
            entry, 
            language_code, 
            source
        )
        
        # Insert relationships with robust mapping (for the 'relations' dict)
        if 'relations' in entry and isinstance(entry['relations'], dict):
            process_relations(
                cur,
                word_id,
                entry['relations'],
                lang_code=language_code,
                source=source
            )
    except Exception as e:
        logger.error(f"Error processing Kaikki entry: {entry.get('word', 'unknown')}. Error: {str(e)}")


def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries."""
    # Check if table structure exists and create required tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pronunciations (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            tags JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, type, value)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_categories (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            category_name TEXT NOT NULL,
            category_kind TEXT,
            parents JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, category_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_templates (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            template_name TEXT NOT NULL,
            args JSONB,
            expansion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, template_name)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS word_forms (
            id SERIAL PRIMARY KEY,
            word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
            form TEXT NOT NULL,
            is_canonical BOOLEAN DEFAULT FALSE,
            is_primary BOOLEAN DEFAULT FALSE,
            tags JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (word_id, form)
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS definition_links (
            id SERIAL PRIMARY KEY,
            definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
            link_text TEXT NOT NULL,
            link_target TEXT NOT NULL,
            is_wikipedia BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (definition_id, link_text, link_target)
        )
    """)
    
    # Add new columns to existing tables
    cur.execute("""
        DO $$
        BEGIN
            -- Words table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'badlit_form'
            ) THEN
                ALTER TABLE words ADD COLUMN badlit_form TEXT;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'hyphenation'
            ) THEN
                ALTER TABLE words ADD COLUMN hyphenation JSONB;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_proper_noun'
            ) THEN
                ALTER TABLE words ADD COLUMN is_proper_noun BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_abbreviation'
            ) THEN
                ALTER TABLE words ADD COLUMN is_abbreviation BOOLEAN DEFAULT FALSE;
            END IF;
            
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'words' AND column_name = 'is_initialism'
            ) THEN
                ALTER TABLE words ADD COLUMN is_initialism BOOLEAN DEFAULT FALSE;
            END IF;
            
            -- Definitions table enhancements
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'definitions' AND column_name = 'metadata'
            ) THEN
                ALTER TABLE definitions ADD COLUMN metadata JSONB;
            END IF;
        END $$;
    """)
    
    # Extract Baybayin info (existing function)
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Baybayin' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Extract Badlit info (new function for Cebuano)
    def extract_badlit_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        if 'forms' not in entry:
            return None, None
            
        for form in entry['forms']:
            if 'Badlit' in form.get('tags', []):
                return form['form'], form.get('romanized_form')
        return None, None
    
    # Helper function to extract all canonical forms of a word entry
    def extract_canonical_forms(entry: Dict) -> List[str]:
        forms = []
        if 'forms' in entry:
            for form in entry['forms']:
                if 'form' in form and form.get('form') and 'canonical' in form.get('tags', []):
                    forms.append(form['form'])
        return forms
    
    # Standardize entry POS (existing function)  
    def standardize_entry_pos(pos_str: str) -> str:
        if not pos_str:
            return 'unc'  # Default to uncategorized
        pos_key = pos_str.lower().strip()
        return get_standard_code(pos_key)
    
    # Process pronunciation (enhanced version)
    def process_pronunciation(cur, word_id: int, entry: Dict):
        if 'sounds' not in entry:
            return
            
        # Extract pronunciation styles from categories
        pronunciation_styles = set()
        if 'senses' in entry:
            for sense in entry['senses']:
                if 'categories' in sense:
                    for category in sense['categories']:
                        if 'name' in category:
                            category_name = category['name'].lower()
                            if 'pronunciation' in category_name:
                                # Extract the style (e.g., "mabilis", "malumay")
                                if 'with' in category_name:
                                    parts = category_name.split('with')
                                    if len(parts) > 1:
                                        style = parts[1].strip().split()[0]
                                        pronunciation_styles.add(style)
        
        for sound in entry['sounds']:
            if 'ipa' in sound:
                # Store IPA pronunciation
                tags = sound.get('tags', [])
                
                # Associate this with pronunciation styles if any
                metadata = {
                    "styles": list(pronunciation_styles)
                } if pronunciation_styles else None
                
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value, tags, metadata)
                    VALUES (%s, 'ipa', %s, %s, %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (
                    word_id, 
                    sound['ipa'], 
                    json.dumps(tags) if tags else None,
                    json.dumps(metadata) if metadata else None
                ))
            elif 'rhymes' in sound:
                # Store rhyme information
                cur.execute("""
                    INSERT INTO pronunciations (word_id, type, value)
                    VALUES (%s, 'rhyme', %s)
                    ON CONFLICT (word_id, type, value) DO NOTHING
                """, (word_id, sound['rhymes']))
    
    # Process sense relationships (enhanced version)
    def process_sense_relationships(cur, word_id: int, sense: Dict):
        # Process synonyms
        if 'synonyms' in sense and isinstance(sense['synonyms'], list):
            for synonym in sense['synonyms']:
                if isinstance(synonym, dict) and 'word' in synonym:
                    syn_word = synonym['word']
                    syn_id = get_or_create_word_id(cur, syn_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in synonym and isinstance(synonym['tags'], list):
                        metadata['tags'] = ','.join(synonym['tags'])
                    insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM.value, "kaikki", metadata)
        
        # Process antonyms
        if 'antonyms' in sense and isinstance(sense['antonyms'], list):
            for antonym in sense['antonyms']:
                if isinstance(antonym, dict) and 'word' in antonym:
                    ant_word = antonym['word']
                    ant_id = get_or_create_word_id(cur, ant_word, 'tl')
                    metadata = {'confidence': 90}
                    if 'tags' in antonym and isinstance(antonym['tags'], list):
                        metadata['tags'] = ','.join(antonym['tags'])
                    insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM.value, "kaikki", metadata)
        
        # Process hypernyms
        if 'hypernyms' in sense and isinstance(sense['hypernyms'], list):
            for hypernym in sense['hypernyms']:
                if isinstance(hypernym, dict) and 'word' in hypernym:
                    hyper_word = hypernym['word']
                    hyper_id = get_or_create_word_id(cur, hyper_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hypernym and isinstance(hypernym['tags'], list):
                        metadata['tags'] = ','.join(hypernym['tags'])
                    insert_relation(cur, word_id, hyper_id, RelationshipType.HYPERNYM.value, "kaikki", metadata)
        
        # Process hyponyms
        if 'hyponyms' in sense and isinstance(sense['hyponyms'], list):
            for hyponym in sense['hyponyms']:
                if isinstance(hyponym, dict) and 'word' in hyponym:
                    hypo_word = hyponym['word']
                    hypo_id = get_or_create_word_id(cur, hypo_word, 'tl')
                    metadata = {'confidence': 85}
                    if 'tags' in hyponym and isinstance(hyponym['tags'], list):
                        metadata['tags'] = ','.join(hyponym['tags'])
                    insert_relation(cur, word_id, hypo_id, RelationshipType.HYPONYM.value, "kaikki", metadata)
        
        # Process holonyms
        if 'holonyms' in sense and isinstance(sense['holonyms'], list):
            for holonym in sense['holonyms']:
                if isinstance(holonym, dict) and 'word' in holonym:
                    holo_word = holonym['word']
                    holo_id = get_or_create_word_id(cur, holo_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in holonym and isinstance(holonym['tags'], list):
                        metadata['tags'] = ','.join(holonym['tags'])
                    insert_relation(cur, word_id, holo_id, RelationshipType.HOLONYM.value, "kaikki", metadata)
        
        # Process meronyms
        if 'meronyms' in sense and isinstance(sense['meronyms'], list):
            for meronym in sense['meronyms']:
                if isinstance(meronym, dict) and 'word' in meronym:
                    mero_word = meronym['word']
                    mero_id = get_or_create_word_id(cur, mero_word, 'tl')
                    metadata = {'confidence': 80}
                    if 'tags' in meronym and isinstance(meronym['tags'], list):
                        metadata['tags'] = ','.join(meronym['tags'])
                    insert_relation(cur, word_id, mero_id, RelationshipType.MERONYM.value, "kaikki", metadata)
        
        # Process derived terms
        if 'derived' in sense and isinstance(sense['derived'], list):
            for derived in sense['derived']:
                if isinstance(derived, dict) and 'word' in derived:
                    derived_word = derived['word']
                    derived_id = get_or_create_word_id(cur, derived_word, 'tl')
                    metadata = {'confidence': 95}
                    if 'tags' in derived and isinstance(derived['tags'], list):
                        metadata['tags'] = ','.join(derived['tags'])
                    insert_relation(cur, word_id, derived_id, RelationshipType.ROOT_OF.value, "kaikki", metadata)
        
        # Process "see also" references
        if 'see_also' in sense and isinstance(sense['see_also'], list):
            for see_also in sense['see_also']:
                if isinstance(see_also, dict) and 'word' in see_also:
                    see_also_word = see_also['word']
                    see_also_id = get_or_create_word_id(cur, see_also_word, 'tl')
                    metadata = {'confidence': 70}
                    if 'tags' in see_also and isinstance(see_also['tags'], list):
                        metadata['tags'] = ','.join(see_also['tags'])
                    insert_relation(cur, word_id, see_also_id, RelationshipType.SEE_ALSO.value, "kaikki", metadata)

    def process_form_relationships(cur, word_id: int, entry: Dict, language_code: str):
        """Process relationships based on word forms (variants, spelling, etc.)."""
        if not entry:
            return
            
        # Process alternative forms
        if 'forms' in entry:
            for form in entry.get('forms', []):
                if not form or not isinstance(form, dict):
                    continue
                    
                form_word = form.get('form', '')
                if not form_word:
                    continue
                    
                form_word_id = get_or_create_word_id(cur, form_word, language_code)
                
                # Determine relationship type based on form data
                rel_type = RelationshipType.VARIANT.rel_value
                metadata = {"from_forms": True}
                
                # Check for specific form types
                if 'tags' in form:
                    tags = form.get('tags', [])
                    
                    # Add tags to metadata
                    metadata["tags"] = tags
                    
                    # Determine strength and relationship type based on tags
                    if any(tag in ['standard spelling', 'preferred', 'standard form'] for tag in tags):
                        rel_type = RelationshipType.PREFERRED_SPELLING.rel_value if hasattr(RelationshipType, 'PREFERRED_SPELLING') else RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 95
                    elif any(tag in ['alternative spelling', 'alternate spelling', 'alt form'] for tag in tags):
                        rel_type = RelationshipType.SPELLING_VARIANT.rel_value
                        metadata["strength"] = 90
                    elif any(tag in ['regional', 'dialect'] for tag in tags):
                        rel_type = RelationshipType.REGIONAL_VARIANT.rel_value
                        metadata["strength"] = 85
                    else:
                        metadata["strength"] = 80
                else:
                    metadata["strength"] = 80
                
                # Add source and qualifier if available
                if 'source' in form:
                    metadata["source"] = form.get('source')
                
                if 'qualifier' in form:
                    metadata["qualifier"] = form.get('qualifier')
                    
                insert_relation(
                    cur, 
                    word_id,
                    form_word_id,
                    rel_type,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )
                
        # Process alt_of references 
        if 'alt_of' in entry:
            for alt_of in entry.get('alt_of', []):
                if isinstance(alt_of, str):
                    alt_word = alt_of
                    alt_meta = {}
                elif isinstance(alt_of, list) and len(alt_of) > 0:
                    alt_word = alt_of[0]
                    alt_meta = {"context": alt_of[1]} if len(alt_of) > 1 else {}
                else:
                    continue
                    
                alt_word_id = get_or_create_word_id(cur, alt_word, language_code)
                
                # The current word is an alternative of the alt_word
                metadata = {"strength": 90, "from_alt_of": True}
                metadata.update(alt_meta)
                
                insert_relation(
                    cur, 
                    word_id,
                    alt_word_id,
                    RelationshipType.VARIANT.rel_value,
                    sources=SourceStandardization.standardize_sources(filename),
                    metadata=metadata
                )


    # Process categories (new function)
    def process_categories(cur, definition_id: int, categories: List[Dict]):
        if not categories:
            return
            
        for category in categories:
            if 'name' not in category:
                continue
                
            category_name = category['name']
            category_kind = category.get('kind', '')
            parents = category.get('parents', [])
            
            cur.execute("""
                INSERT INTO definition_categories (definition_id, category_name, category_kind, parents)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (definition_id, category_name) DO NOTHING
            """, (
                definition_id,
                category_name,
                category_kind,
                json.dumps(parents) if parents else None
            ))
    
    # Process head templates (new function)
    def process_head_templates(cur, word_id: int, templates: List[Dict]):
        if not templates:
            return
            
        for template in templates:
            template_name = template.get('name', '')
            args = template.get('args', {})
            expansion = template.get('expansion', '')
            
            cur.execute("""
                INSERT INTO word_templates (word_id, template_name, args, expansion)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (word_id, template_name) DO NOTHING
            """, (
                word_id,
                template_name,
                json.dumps(args) if args else None,
                expansion
            ))
    
    # Process links (new function)
    def process_links(cur, definition_id: int, links: List[List[str]]):
        if not links:
            return
            
        for link in links:
            if len(link) >= 2:
                text = link[0]
                target = link[1]
                
                # Check if it's a Wikipedia link
                is_wikipedia = False
                if target.startswith('w:'):
                    is_wikipedia = True
                    target = target[2:]  # Remove the w: prefix
                
                cur.execute("""
                    INSERT INTO definition_links (definition_id, link_text, link_target, is_wikipedia)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (definition_id, link_text, link_target) DO NOTHING
                """, (
                    definition_id,
                    text,
                    target,
                    is_wikipedia
                ))
    
    # Process etymology (enhanced version)
    def process_etymology(cur, word_id: int, etymology_text: str, etymology_templates: List[Dict] = None):
        if not etymology_text:
            return
            
        # Extract basic components
        normalized_components = extract_etymology_components(etymology_text)
        
        # Extract source languages from etymology templates
        source_languages = []
        etymology_structure = None
        
        if etymology_templates:
            etymology_structure = json.dumps(etymology_templates)
            
            for template in etymology_templates:
                if template.get('name') in ('bor', 'der', 'inh', 'calque', 'bor+'):
                    args = template.get('args', {})
                    if '2' in args:  # Source language code
                        source_lang = args['2']
                        source_languages.append(source_lang)
                
                # Process borrowing sources
                template_name = template.get('name', '')
                args = template.get('args', {})
                
                if template_name in ('bor', 'bor+') and '2' in args and '3' in args:
                    source_lang = args['2']
                    source_word = args['3']
                    
                    # Skip if source word is proto-language with * prefix
                    if source_word.startswith('*'):
                        continue
                        
                    # Try to find or create the source word
                    source_word_id = get_or_create_word_id(cur, source_word, language_code=source_lang)
                    
                    # Create borrowed_from relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        source_word_id, 
                        'borrowed_from', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
                
                # Process cognates
                elif template_name == 'cog' and '1' in args and '2' in args:
                    cog_lang = args['2']
                    cog_word = args['3'] if '3' in args else args['2']
                    
                    # Skip if cognate word is proto-language with * prefix
                    if cog_word.startswith('*'):
                        continue
                        
                    # Try to find or create the cognate word
                    cog_word_id = get_or_create_word_id(cur, cog_word, language_code=cog_lang)
                    
                    # Create cognate relationship
                    cur.execute("""
                        INSERT INTO relations (from_word_id, to_word_id, relation_type, sources)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (from_word_id, to_word_id, relation_type) DO NOTHING
                    """, (
                        word_id, 
                        cog_word_id, 
                        'cognate_with', 
                        SourceStandardization.standardize_sources('kaikki.jsonl')
                    ))
        
        # Create or update the etymology record
        cur.execute("""
            INSERT INTO etymologies (
                word_id, 
                etymology_text, 
                normalized_components, 
                etymology_structure,
                source_languages,
                sources
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (word_id) DO UPDATE SET
                etymology_text = EXCLUDED.etymology_text,
                normalized_components = EXCLUDED.normalized_components,
                etymology_structure = EXCLUDED.etymology_structure,
                source_languages = EXCLUDED.source_languages,
                updated_at = CURRENT_TIMESTAMP
        """, (
            word_id,
            etymology_text,
            json.dumps(normalized_components) if normalized_components else None,
            etymology_structure,
            json.dumps(source_languages) if source_languages else None,
            SourceStandardization.standardize_sources('kaikki.jsonl')
        ))
    
    # Main process entry function (enhanced version)
    def process_entry(cur, entry: Dict):
        """Process a single dictionary entry."""
        if 'word' not in entry:
            logger.warning("Skipping entry without 'word' field")
            return None
            
        try:
            word = entry['word']
            pos = entry.get('pos', '')
            language_code = entry.get('lang_code', 'tl')  # Default to Tagalog if not specified
            
            # Check if this is a proper noun
            is_proper_noun = False
            if pos == 'prop' or pos == 'proper noun':
                is_proper_noun = True
                pos = 'noun'  # Standardize to noun POS but mark as proper
                
            # Check if this is an abbreviation or initialism
            is_abbreviation = False
            is_initialism = False
            if pos == 'abbrev' or pos == 'abbreviation':
                is_abbreviation = True
                
            if 'tags' in entry:
                tags = entry.get('tags', [])
                if 'abbreviation' in tags or 'abbrev' in tags:
                    is_abbreviation = True
                if 'initialism' in tags or 'acronym' in tags:
                    is_initialism = True
            
            # Get Baybayin form if any
            baybayin_form, romanized_form = extract_baybayin_info(entry)
            
            # Get Badlit form if any
            badlit_form, badlit_romanized = extract_badlit_info(entry)
            
            # Get or create the word
            word_id = get_or_create_word_id(
                cur, 
                word, 
                language_code=language_code,
                has_baybayin=bool(baybayin_form),
                baybayin_form=baybayin_form,
                romanized_form=romanized_form or badlit_romanized,
                badlit_form=badlit_form,
                is_proper_noun=is_proper_noun,
                is_abbreviation=is_abbreviation,
                is_initialism=is_initialism,
                tags=','.join(entry.get('tags', [])) if 'tags' in entry else None
            )
            
            # Process pronunciation information
            process_pronunciation(cur, word_id, entry)
            
            # Process form relationships
            process_form_relationships(cur, word_id, entry, language_code)
            
            # Process etymologies
            if 'etymology_text' in entry:
                process_etymology(cur, word_id, entry['etymology_text'], entry.get('etymology_templates'))
                
            # Process head templates
            if 'head_templates' in entry and entry['head_templates']:
                process_head_templates(cur, word_id, entry['head_templates'])
                
            # Process definitions from senses
            if 'senses' in entry:
                for sense in entry['senses']:
                    if not sense or not isinstance(sense, dict):
                        continue
                        
                    # Skip definitions that are just references to other entries
                    if 'glosses' not in sense and 'raw_glosses' not in sense:
                        # This might be an alt_of entry
                        continue
                        
                    # Get the glosses
                    glosses = sense.get('glosses', sense.get('raw_glosses', []))
                    if not glosses:
                        continue
                    
                    definition_text = '; '.join(glosses) if isinstance(glosses, list) else str(glosses)
                    
                    # Check for qualifiers and add to usage notes
                    usage_notes = None
                    if 'qualifier' in sense:
                        qualifier = sense['qualifier']
                        if qualifier:
                            usage_notes = f"Qualifier: {qualifier}"
                            
                    # Check for domain/topic and add to usage notes
                    if 'topics' in sense:
                        topics = sense['topics']
                        topics_str = ', '.join(topics) if isinstance(topics, list) else str(topics)
                        if topics_str:
                            domain_note = f"Domain: {topics_str}"
                            usage_notes = f"{usage_notes}; {domain_note}" if usage_notes else domain_note
                    
                    # Extract examples if available
                    examples = None
                    if 'examples' in sense and sense['examples']:
                        examples = json.dumps([ex.get('text', ex) for ex in sense['examples'] 
                                             if (isinstance(ex, dict) and 'text' in ex) or isinstance(ex, str)])
                    
                    # Insert the definition
                    definition_id = insert_definition(
                        cur,
                        word_id,
                        definition_text,
                        part_of_speech=pos,
                        examples=examples,
                        usage_notes=usage_notes,
                        sources=SourceStandardization.standardize_sources(filename)
                    )
                    
                    if definition_id:
                        # Process category information
                        if 'categories' in sense and sense['categories']:
                            process_categories(cur, definition_id, sense['categories'])
                            
                        # Process links
                        if 'links' in sense and sense['links']:
                            process_links(cur, definition_id, sense['links'])
                            
                        # Process Wikipedia links
                        if 'wikipedia' in sense and sense['wikipedia']:
                            wiki_links = sense['wikipedia']
                            if isinstance(wiki_links, list):
                                for wiki in wiki_links:
                                    if isinstance(wiki, str):
                                        cur.execute("""
                                            INSERT INTO definition_links (definition_id, link_type, target, source)
                                            VALUES (%s, %s, %s, %s)
                                            ON CONFLICT DO NOTHING
                                        """, (
                                            definition_id,
                                            'wikipedia',
                                            wiki,
                                            SourceStandardization.standardize_sources(filename)
                                        ))
                    
                    # Process relationships
                    process_sense_relationships(cur, word_id, sense)
            
            return word_id
                
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('word', 'unknown')}: {str(e)}")
            return None

    # Process entries from the file
    with open(filename, 'r', encoding='utf-8') as f:
        entries_processed = 0
        errors = 0
        for line in f:
            try:
                entry = json.loads(line)
                process_entry(cur, entry)
                entries_processed += 1
                if entries_processed % 1000 == 0:
                    logger.info(f"Processed {entries_processed} entries from {filename}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {str(e)}")
                errors += 1
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")
                errors += 1
        
        logger.info(f"Completed processing {filename}: {entries_processed} entries processed with {errors} errors")

# -------------------------------------------------------------------
# Command Line Interface Functions
# -------------------------------------------------------------------
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage dictionary data in PostgreSQL.")
    subparsers = parser.add_subparsers(dest="command")
    migrate_parser = subparsers.add_parser("migrate", help="Create/update schema and load data")
    migrate_parser.add_argument("--check-exists", action="store_true", help="Skip identical existing entries")
    migrate_parser.add_argument("--force", action="store_true", help="Force migration without confirmation")
    migrate_parser.add_argument("--data-dir", type=str, help="Directory containing dictionary data files")
    migrate_parser.add_argument("--sources", type=str, help="Comma-separated list of source names to process")
    migrate_parser.add_argument("--file", type=str, help="Specific data file to process")
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity")
    verify_parser.add_argument('--quick', action='store_true', help='Run quick verification')
    verify_parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')
    update_parser = subparsers.add_parser("update", help="Update DB with new data")
    update_parser.add_argument("--file", type=str, required=True, help="JSON or JSONL file to use")
    update_parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    lookup_parser = subparsers.add_parser("lookup", help="Look up word information")
    lookup_parser.add_argument("word", help="Word to look up")
    lookup_parser.add_argument("--debug", action="store_true", help="Show debug information")
    lookup_parser.add_argument("--format", choices=['text', 'json', 'rich'], default='rich', help="Output format")
    stats_parser = subparsers.add_parser("stats", help="Display dictionary statistics")
    stats_parser.add_argument("--detailed", action="store_true", help="Show detailed statistics")
    stats_parser.add_argument("--export", type=str, help="Export statistics to file")
    subparsers.add_parser("leaderboard", help="Display top contributors")
    subparsers.add_parser("help", help="Display help information")
    subparsers.add_parser("test", help="Run database connectivity tests")
    subparsers.add_parser("explore", help="Interactive dictionary explorer")
    purge_parser = subparsers.add_parser("purge", help="Safely delete all data")
    purge_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    subparsers.add_parser("cleanup", help="Clean up dictionary data by removing duplicates and standardizing formats")
    subparsers.add_parser("migrate-relationships", help="Migrate existing relationships to the new RelationshipType system")
    return parser

def migrate_data(args):
    """Migrate dictionary data from various sources."""
    sources = [
        {
            "name": "Tagalog Words",
            "file": "tagalog-words.json",
            "handler": process_tagalog_words,
            "required": False
        },
        {
            "name": "Root Words",
            "file": "root_words_with_associated_words_cleaned.json",
            "handler": process_root_words_cleaned,
            "required": False
        },
        {
            "name": "KWF Dictionary",
            "file": "kwf_dictionary.json",
            "handler": process_kwf_dictionary,
            "required": True  # Fixed: Mark KWF Dictionary as required so it is always processed.
        },
        {
            "name": "Kaikki.org (Tagalog)",
            "file": "kaikki.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False
        },
        {
            "name": "Kaikki.org (Cebuano)",
            "file": "kaikki-ceb.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False
        }
    ]
    
    # Get data directory from args if provided, or use defaults
    if hasattr(args, 'data_dir') and args.data_dir:
        data_dirs = [args.data_dir]
    else:
        data_dirs = ["data", os.path.join("..", "data")]
    
    # Filter sources if specific ones are requested via --sources
    if hasattr(args, 'sources') and args.sources:
        requested_sources = [s.lower() for s in args.sources.split(',')]
        sources = [s for s in sources if s["name"].lower() in requested_sources or s["file"].lower() in requested_sources]
        for source in sources:
            source["required"] = True
    
    # Custom file overrides existing sources
    if hasattr(args, 'file') and args.file:
        filename = args.file
        if filename.endswith('.jsonl'):
            handler = process_kaikki_jsonl
        elif 'root_words' in filename.lower():
            handler = process_root_words_cleaned
        elif 'kwf' in filename.lower():
            handler = process_kwf_dictionary
        else:
            handler = process_tagalog_words
        basename = os.path.basename(filename)
        source_found = False
        for source in sources:
            if source["file"] == basename:
                source["file"] = filename  # Use full path
                source["required"] = True
                source_found = True
                break
        if not source_found:
            sources.append({
                "name": f"Custom ({basename})",
                "file": filename,
                "handler": handler,
                "required": True
            })
    
    conn = None
    cur = None
    console = Console()
    try:
        conn = get_connection()
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        console.print(f"\n[bold red]Failed to connect to the database:[/] {str(e)}")
        console.print("Please check your database configuration and ensure the database exists.")
        console.print(f"Current settings: DB_NAME={DB_NAME}, DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_USER={DB_USER}")
        console.print("\n[bold]To fix this issue:[/]\n1. Make sure PostgreSQL is running\n2. Create the database if it doesn't exist (e.g., `createdb {DB_NAME}`)\n3. Verify your .env settings.")
        return

    try:
        cur = conn.cursor()
        console.print("[bold]Setting up database schema...[/]")
        create_or_update_tables(conn)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            for source in sources:
                # Look for file in provided data directories if not an absolute path
                if os.path.isabs(source["file"]):
                    filename = source["file"]
                else:
                    filename = None
                    for data_dir in data_dirs:
                        potential_path = os.path.join(data_dir, source["file"])
                        if os.path.exists(potential_path):
                            filename = potential_path
                            break
                if not filename or not os.path.exists(filename):
                    if source["required"]:
                        logger.error(f"Required file not found: {source['file']}")
                        sys.exit(1)
                    else:
                        logger.warning(f"Optional file not found: {source['file']}")
                        continue
                task = progress.add_task(f"Processing {source['name']}...", total=1)
                try:
                    source["handler"](cur, filename)
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Error processing {source['name']}: {str(e)}")
                    if source["required"]:
                        raise
            conn.commit()
        console.print("[green]Data migration completed successfully.[/]")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error during migration: {str(e)}")
        console.print(f"\n[bold red]Migration failed:[/] {str(e)}")
    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def verify_database(args):
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        issues = []
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
        
        # Display relation types and counts
        console.print("[bold]Relation Types and Counts[/]")
        cur.execute("""
            SELECT relation_type, COUNT(*) as count
            FROM relations
            GROUP BY relation_type
            ORDER BY count DESC
        """)
        relation_stats = cur.fetchall()
        if relation_stats:
            rel_table = Table(box=box.ROUNDED)
            rel_table.add_column("Relation Type", style="yellow")
            rel_table.add_column("Count", justify="right", style="green")
            
            for rel_type, count in relation_stats:
                rel_table.add_row(rel_type or "Unspecified", f"{count:,}")
            
            console.print(rel_table)
        else:
            console.print("[italic]No relations found[/]")
        
        console.print()
        
        # Display language distributions
        console.print("[bold]Language Distribution[/]")
        cur.execute("""
            SELECT language_code, COUNT(*) as count
            FROM words
            GROUP BY language_code
            ORDER BY count DESC
        """)
        lang_stats = cur.fetchall()
        if lang_stats:
            lang_table = Table(box=box.ROUNDED)
            lang_table.add_column("Language", style="magenta")
            lang_table.add_column("Count", justify="right", style="green")
            
            for lang, count in lang_stats:
                lang_name = "Tagalog" if lang == "tl" else "Cebuano" if lang == "ceb" else lang
                lang_table.add_row(lang_name, f"{count:,}")
            
            console.print(lang_table)
        
        # Display parts of speech distribution
        console.print()
        console.print("[bold]Parts of Speech Distribution[/]")
        cur.execute("""
            SELECT p.name_tl, COUNT(*) as count
            FROM definitions d
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY p.name_tl
            ORDER BY COUNT(*) DESC
        """)
        pos_stats = cur.fetchall()
        if pos_stats:
            pos_table = Table(box=box.ROUNDED)
            pos_table.add_column("Part of Speech", style="blue")
            pos_table.add_column("Count", justify="right", style="green")
            
            for pos, count in pos_stats:
                pos_table.add_row(pos or "Uncategorized", f"{count:,}")
            
            console.print(pos_table)
        
        # Display definition sources
        console.print()
        console.print("[bold]Definition Sources[/]")
        cur.execute("""
            SELECT
                CASE 
                    WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                    WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                    WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                    WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                    WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                    ELSE sources
                END as source_name,
                COUNT(*) as count
            FROM definitions
            WHERE sources IS NOT NULL
            GROUP BY sources
            ORDER BY count DESC
        """)
        source_stats = cur.fetchall()
        if source_stats:
            source_table = Table(box=box.ROUNDED)
            source_table.add_column("Source", style="cyan")
            source_table.add_column("Count", justify="right", style="green")
            
            for source, count in source_stats:
                source_table.add_row(source or "Unknown", f"{count:,}")
            
            console.print(source_table)
        
        # Explain empty tables
        console.print()
        console.print("[bold]Notes on Empty Tables[/]")
        console.print(Panel("""
- [bold]affixations[/bold]: This table is for storing information about word affixation patterns, which are linguistic processes where affixes (prefixes, suffixes, infixes) are added to root words to create new words. These are populated by specialized affix analysis functions.

- [bold]definition_relations[/bold]: This table stores semantic relationships between definitions (rather than between words). These are typically populated during advanced linguistic analysis.

Both tables might be empty if no specialized linguistic analysis has been performed on the dataset yet.
""", title="Table Explanations", border_style="blue"))
        
        if args.quick:
            console.print("[yellow]Sample entries from 'words' table:[/]")
            cur.execute("SELECT id, lemma, language_code, root_word_id FROM words LIMIT 5")
            sample_table = Table(show_header=True)
            sample_table.add_column("ID")
            sample_table.add_column("Lemma")
            sample_table.add_column("Language")
            sample_table.add_column("Root ID")
            for row in cur.fetchall():
                sample_table.add_row(*[str(x) for x in row])
            console.print(sample_table)
            return
        
        def check_data_integrity(cur) -> List[str]:
            integrity_issues = []
            cur.execute("""
                SELECT COUNT(*) FROM relations r
                WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
                   OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
            """)
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found orphaned relations")
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
                integrity_issues.append("Found duplicate definitions")
            cur.execute("SELECT COUNT(*) FROM words WHERE search_text IS NULL")
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found words with missing search vectors")
            cur.execute("""
                SELECT baybayin_form, COUNT(*)
                FROM words
                WHERE has_baybayin = TRUE
                GROUP BY baybayin_form
                HAVING COUNT(*) > 1
            """)
            dupes = cur.fetchall()
            if dupes and len(dupes) > 0:
                integrity_issues.append("Found duplicate Baybayin forms")
            return integrity_issues
        
        integrity_issues = check_data_integrity(cur)
        if integrity_issues:
            issues.extend(integrity_issues)
        baybayin_issues = check_baybayin_consistency(cur)
        if baybayin_issues:
            issues.extend(baybayin_issues)
        if issues:
            console.print("\n[bold red]Found Issues:[/]")
            issues_table = Table(show_header=True)
            issues_table.add_column("Issue")
            issues_table.add_column("Details")
            for issue in issues:
                issues_table.add_row(issue, "")
            console.print(issues_table)
            if args.repair:
                console.print("\n[yellow]Attempting to repair issues...[/]")
                repair_database_issues(cur, issues)
        else:
            console.print("\n[bold green]No issues found![/]")
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        logger.info("Database verification completed")

@with_transaction(commit=True)
def repair_database_issues(cur, issues):
    """
    Repair common database issues by cleaning up orphaned records, fixing inconsistencies,
    and removing duplicate definitions.
    """
    # Delete relations with missing words
    cur.execute("""
        DELETE FROM relations r
        WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
        OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
    """)
    
    # Delete duplicate definitions
    cur.execute("""
        WITH DuplicateDefs AS (
            SELECT word_id, definition_text, COUNT(*)
            FROM definitions
            GROUP BY word_id, definition_text
            HAVING COUNT(*) > 1
        )
        DELETE FROM definitions d
        USING DuplicateDefs dd
        WHERE d.word_id = dd.word_id
        AND d.definition_text = dd.definition_text
        AND d.id != (SELECT MIN(id) FROM definitions WHERE word_id = dd.word_id AND definition_text = dd.definition_text)
    """)
    
    # Update missing search_text values
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
    
    # Fix Baybayin inconsistencies
    cur.execute("""
        DELETE FROM words 
        WHERE has_baybayin = TRUE AND baybayin_form IS NULL
    """)
    
    cur.execute("""
        UPDATE words
        SET has_baybayin = FALSE,
            baybayin_form = NULL
        WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
    """)
    
    logger.info("Database repairs completed")
    return True

def display_help(args):
    console = Console()
    console.print("\n[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print("[dim]A comprehensive tool for managing Filipino dictionary data[/]\n", justify="center")
    usage_panel = Panel(Text.from_markup("python dictionary_manager.py [command] [options]"),
                        title="Basic Usage", border_style="blue")
    console.print(usage_panel)
    console.print()
    commands = [
        {"name": "migrate", "description": "Create/update schema and load data from sources",
         "options": [("--check-exists", "Skip identical existing entries"), ("--force", "Skip confirmation prompt")],
         "example": "python dictionary_manager.py migrate --check-exists", "icon": "🔄"},
        {"name": "lookup", "description": "Look up comprehensive information about a word",
         "options": [("word", "The word to look up"), ("--format", "Output format (text/json/rich)")],
         "example": "python dictionary_manager.py lookup kamandag", "icon": "🔍"},
        {"name": "stats", "description": "Display comprehensive dictionary statistics",
         "options": [("--detailed", "Show detailed statistics"), ("--export", "Export statistics to file")],
         "example": "python dictionary_manager.py stats --detailed", "icon": "📊"},
        {"name": "verify", "description": "Verify data integrity",
         "options": [("--quick", "Run quick verification"), ("--repair", "Attempt to repair issues")],
         "example": "python dictionary_manager.py verify --repair", "icon": "✅"},
        {"name": "purge", "description": "Safely delete all data from the database",
         "options": [("--force", "Skip confirmation prompt")],
         "example": "python dictionary_manager.py purge --force", "icon": "🗑️"}
    ]
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
        row = [f"{cmd['icon']} {cmd['name']}", cmd["description"], options_text, f"[dim]{cmd['example']}[/]"]
        if cmd["name"] in ["migrate", "update", "purge"]:
            data_commands.add_row(*row)
        else:
            query_commands.add_row(*row)
    console.print(data_commands)
    console.print()
    console.print(query_commands)
    console.print()
    console.print("\n[dim]For more detailed information, visit the documentation.[/]", justify="center")
    console.print()

def lookup_word(args):
    """
    Look up a word in the dictionary and display its details (basic info,
    pronunciation, definitions, etymology, and related words) in a clear,
    organized format using Rich's layout features.
    """
    conn = None
    console = Console()
    try:
        logger.info(f"Looking up word: {args}")
        
        # Check if args has the word attribute
        if not hasattr(args, 'word'):
            logger.error("args does not have 'word' attribute")
            console.print("[bold red]Error: Missing word argument[/]")
            return
            
        if args.word is None:
            logger.error("args.word is None")
            console.print("[bold red]Error: Word argument is None[/]")
            return
            
        logger.info(f"Word argument value: '{args.word}'")
        
        conn = get_connection()
        cur = conn.cursor()
        
        try:
            normalized_word = normalize_lemma(args.word)
            logger.info(f"Normalized word: {normalized_word}")
        except Exception as e:
            logger.error(f"Error in normalize_lemma: {e}")
            console.print(f"[bold red]Error normalizing word: {e}[/]")
            return
        
        # Check if we can execute a simple query first
        try:
            cur.execute("SELECT 1")
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            
        cur.execute("""
            WITH word_sources AS (
                SELECT DISTINCT w.id, 
                       array_agg(DISTINCT d.sources) as def_sources,
                       array_agg(DISTINCT e.sources) as ety_sources,
                       w.tags, w.source_info
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN etymologies e ON w.id = e.word_id
                WHERE w.normalized_lemma = %s
                GROUP BY w.id, w.tags, w.source_info
            )
            SELECT w.id, w.lemma, w.language_code, w.has_baybayin, w.baybayin_form,
                   w.romanized_form, w.preferred_spelling, w.pronunciation_data, w.idioms,
                   ws.def_sources, ws.ety_sources, ws.tags, ws.source_info, w.root_word_id
            FROM words w
            JOIN word_sources ws ON w.id = ws.id
        """, (normalized_word,))
        word_data = cur.fetchone()
        logger.info(f"Word data fetched: {word_data is not None}")
        if not word_data:
            console.print(f"[bold red]Word '{args.word}' not found.[/]")
            return  # Added return to prevent proceeding when word is not found
        
        logger.info("Beginning to unpack word data")
        
        try:
            (word_id, lemma, language_code, has_baybayin, baybayin_form,
             romanized_form, preferred_spelling, pronunciation_data, idioms, def_sources,
             ety_sources, tags, source_info, root_word_id) = word_data
            logger.info("Successfully unpacked word data")
            
            # Log the values for debugging
            logger.info(f"word_id: {word_id}, lemma: {lemma}, language_code: {language_code}")
            logger.info(f"has_baybayin: {has_baybayin}, baybayin_form: {baybayin_form}, romanized_form: {romanized_form}")
            logger.info(f"preferred_spelling: {preferred_spelling}, root_word_id: {root_word_id}")
            logger.info(f"def_sources type: {type(def_sources)}, ety_sources type: {type(ety_sources)}")
            logger.info(f"tags type: {type(tags)}, source_info type: {type(source_info)}")
        except Exception as e:
            logger.error(f"Error unpacking word data: {e}")
            console.print(f"[bold red]Error unpacking word data: {e}[/]")
            return

        # Build Basic Information Tree
        logger.info("Building basic information tree")
        basic_tree = Tree(f"[bold blue]{lemma}[/] [dim](ID: {word_id})[/]")
        basic_tree.add(f"Language: {'Tagalog' if language_code=='tl' else 'Cebuano'}")
        
        logger.info("Adding preferred spelling if available")
        if preferred_spelling:
            basic_tree.add(f"Preferred Spelling: {preferred_spelling}")
            
        logger.info("Adding Baybayin form if available")    
        if has_baybayin:
            basic_tree.add(f"Baybayin Form: {baybayin_form}")
            basic_tree.add(f"Romanized: {romanized_form or 'Not available'}")
            
        logger.info("Adding root word if available")
        if root_word_id:
            cur.execute("SELECT lemma FROM words WHERE id = %s", (root_word_id,))
            root_result = cur.fetchone()
            if root_result:
                basic_tree.add(f"Root Word: {root_result[0]}")
                
        # Gather sources from tags and source_info
        logger.info("Gathering sources - starting")
        sources_set = set()
        
        logger.info("Processing tags")
        if tags:
            logger.info(f"Tags content: {tags}")
            sources_set.update(s.strip() for s in tags.split(','))
            
        logger.info("Processing source_info")
        if source_info:
            logger.info(f"Source info content: {source_info}")
            try:
                si = source_info if isinstance(source_info, dict) else json.loads(source_info)
                sources_set.update(s.strip() for s in si.get("sources", []))
            except Exception as e:
                logger.error(f"Error processing source_info: {e}")
                
        logger.info("Processing def_sources")
        if def_sources:
            logger.info(f"def_sources content: {def_sources}")
            for s in def_sources:
                if s is None:
                    logger.warning("Found None value in def_sources")
                    continue
                try:
                    logger.info(f"Processing def_source: {s}, type: {type(s)}")
                    if isinstance(s, str):
                        sources_set.update(src.strip() for src in s.split(','))
                except Exception as e:
                    logger.error(f"Error processing def_source: {e}")
                    
        logger.info("Processing ety_sources")
        if ety_sources:
            logger.info(f"ety_sources content: {ety_sources}")
            for s in ety_sources:
                if s is None:
                    logger.warning("Found None value in ety_sources")
                    continue
                try:
                    logger.info(f"Processing ety_source: {s}, type: {type(s)}")
                    if isinstance(s, str):
                        sources_set.update(src.strip() for src in s.split(','))
                except Exception as e:
                    logger.error(f"Error processing ety_source: {e}")
                    
        if sources_set:
            basic_tree.add(f"Sources: {', '.join(sorted(sources_set))}")

        # Pronunciation Panel (if available)
        pron_panel = None
        if pronunciation_data:
            try:
                pron_data = pronunciation_data if isinstance(pronunciation_data, dict) else json.loads(pronunciation_data)
                pron_lines = []
                if "ipa" in pron_data:
                    pron_lines.append(f"IPA: {pron_data['ipa']}")
                if "audio" in pron_data:
                    pron_lines.append(f"Audio: {pron_data['audio']}")
                if "hyphenation" in pron_data:
                    pron_lines.append(f"Hyphenation: {pron_data['hyphenation']}")
                if "sounds" in pron_data:
                    for sound in pron_data.get("sounds", []):
                        if isinstance(sound, dict) and "ipa" in sound:
                            dialect = sound.get("dialect", "Standard")
                            pron_lines.append(f"IPA ({dialect}): {sound['ipa']}")
                if pron_lines:
                    pron_panel = Panel("\n".join(pron_lines), title="Pronunciation", border_style="green")
            except Exception as e:
                logger.error(f"Error processing pronunciation data: {e}")

        # Definitions Table
        cur.execute("""
            SELECT p.name_tl, d.definition_text, d.examples, d.usage_notes, d.sources, d.original_pos
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_tl, d.created_at
        """, (word_id,))
        definitions = cur.fetchall()
        def_table = None
        if definitions:
            def_table = Table(title="Definitions", box=box.ROUNDED, show_lines=True, expand=True)
            def_table.add_column("POS", style="blue", no_wrap=True)
            def_table.add_column("Definition", style="white")
            def_table.add_column("Examples", style="green")
            def_table.add_column("Usage Notes", style="yellow")
            def_table.add_column("Sources", style="magenta")
            for pos, definition, examples, usage_notes, sources, original_pos in definitions:
                pos_disp = pos if pos else "Uncategorized"
                if original_pos and original_pos.lower() not in pos_disp.lower():
                    pos_disp = f"{pos_disp} ({original_pos})"
                ex_text = ""
                if examples:
                    try:
                        ex_list = json.loads(examples)
                        ex_text = "\n".join(f"• {ex.strip()}" for ex in ex_list if ex.strip())
                    except Exception:
                        ex_text = examples
                note_text = usage_notes if usage_notes else ""
                def_table.add_row(pos_disp, definition, ex_text, note_text, sources)
        
        # Etymology Panels
        cur.execute("""
            SELECT etymology_text, normalized_components, language_codes, sources
            FROM etymologies
            WHERE word_id = %s
            ORDER BY created_at
        """, (word_id,))
        ety_rows = cur.fetchall()
        ety_panels = []
        if ety_rows:
            for ety_text, components, langs, sources in ety_rows:
                if not ety_text.strip():
                    continue
                lines = [f"[cyan]{ety_text}[/]"]
                if components:
                    try:
                        comps = json.loads(components)
                        if isinstance(comps, list) and comps:
                            lines.append("[bold]Components:[/]")
                            for comp in sorted(set(c.strip() for c in comps if c.strip())):
                                lines.append(f"  • {comp}")
                    except Exception:
                        pass
                if langs:
                    lines.append(f"[bold]Languages:[/] {langs}")
                if sources:
                    lines.append(f"[bold]Sources:[/] {sources}")
                ety_panels.append(Panel("\n".join(lines), border_style="blue"))
        
        # Relations Table
        cur.execute("""
            SELECT r.relation_type, w.lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            ORDER BY r.relation_type, w.lemma
        """, (word_id,))
        rel_rows = cur.fetchall()
        rel_table = None
        if rel_rows:
            rel_table = Table(title="Related Words", box=box.ROUNDED, expand=True)
            rel_table.add_column("Relation Type", style="bold yellow")
            rel_table.add_column("Word", style="cyan")
            for rel_type, rel_word in rel_rows:
                rel_table.add_row(rel_type, rel_word)

        # Assemble final output using a grid layout
        grid = Table.grid(expand=True)
        grid.add_row(Panel(basic_tree, border_style="bright_blue", title="Basic Information"))
        if pron_panel:
            grid.add_row(pron_panel)
        if def_table:
            grid.add_row(def_table)
        if ety_panels:
            ety_group = Group(*ety_panels)
            grid.add_row(Panel(ety_group, title="Etymology", border_style="blue"))
        if rel_table:
            grid.add_row(rel_table)
            
        console.print(Panel(grid, title="Lookup Result", border_style="bright_blue"))
        input("\nPress Enter to continue...")
    except Exception as e:
        logger.error(f"Error during lookup: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")
    finally:
        if conn:
            conn.close()

def display_dictionary_stats(args):
    """
    Display a comprehensive set of dictionary statistics in a visually
    appealing layout. Shows basic counts, sources, parts of speech and
    relation type breakdown.
    """
    conn = None
    console = Console()
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Basic statistics
        basic_queries = {
            "Total Words": "SELECT COUNT(*) FROM words",
            "Tagalog Words": "SELECT COUNT(*) FROM words WHERE language_code = 'tl'",
            "Cebuano Words": "SELECT COUNT(*) FROM words WHERE language_code = 'ceb'",
            "Baybayin Words": "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE",
            "Definitions": "SELECT COUNT(*) FROM definitions",
            "Relations": "SELECT COUNT(*) FROM relations",
            "Etymologies": "SELECT COUNT(*) FROM etymologies"
        }
        basic_stats = {}
        for label, query in basic_queries.items():
            cur.execute(query)
            basic_stats[label] = cur.fetchone()[0]
        basic_table = Table(title="Basic Statistics", box=box.ROUNDED)
        basic_table.add_column("Metric", style="cyan")
        basic_table.add_column("Count", justify="right", style="green")
        for label, count in basic_stats.items():
            basic_table.add_row(label, f"{count:,}")
        
        # Definition sources statistics
        source_stats_query = """
            SELECT 
                 CASE
                      WHEN sources = 'kaikki-ceb.jsonl' THEN 'kaikki.org (Cebuano)'
                     WHEN sources = 'kaikki.jsonl' THEN 'kaikki.org (Tagalog)'
                     WHEN sources = 'kwf_dictionary.json' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                     WHEN sources = 'root_words_with_associated_words_cleaned.json' THEN 'tagalog.com'
                     WHEN sources = 'tagalog-words.json' THEN 'diksiyonaryo.ph'
                     ELSE sources
                 END as source_name,
                 COUNT(*) as count
            FROM definitions
            GROUP BY source_name
            ORDER BY count DESC
        """
        cur.execute(source_stats_query)
        source_stats = cur.fetchall()
        source_table = Table(title="Definition Sources", box=box.ROUNDED)
        source_table.add_column("Source", style="yellow")
        source_table.add_column("Count", justify="right", style="green")
        for source, count in source_stats:
            source_table.add_row(source, f"{count:,}")
        
        # Parts of Speech distribution
        pos_query = """
            SELECT p.name_tl, COUNT(*) 
            FROM definitions d
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY p.name_tl
            ORDER BY COUNT(*) DESC
        """
        cur.execute(pos_query)
        pos_stats = cur.fetchall()
        pos_table = Table(title="Parts of Speech", box=box.ROUNDED)
        pos_table.add_column("Part of Speech", style="magenta")
        pos_table.add_column("Count", justify="right", style="green")
        for pos, count in pos_stats:
            pos_table.add_row(pos or "Uncategorized", f"{count:,}")
        
        # Relation types distribution
        rel_query = """
            SELECT relation_type, COUNT(*) 
            FROM relations
            GROUP BY relation_type
            ORDER BY COUNT(*) DESC
        """
        cur.execute(rel_query)
        rel_stats = cur.fetchall()
        rel_table = Table(title="Relation Types", box=box.ROUNDED)
        rel_table.add_column("Relation Type", style="blue")
        rel_table.add_column("Count", justify="right", style="green")
        for rel_type, count in rel_stats:
            rel_table.add_row(rel_type, f"{count:,}")
        
        # Compose final layout
        grid = Table.grid(expand=True)
        grid.add_row(basic_table)
        if args.detailed:
            grid.add_row(source_table)
            grid.add_row(pos_table)
            grid.add_row(rel_table)
        
        console.print(Panel(grid, title="Dictionary Statistics", border_style="bright_blue"))
        
        # Optionally export statistics
        if args.export:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "basic_stats": basic_stats,
                "source_stats": {source: count for source, count in source_stats},
                "pos_stats": {pos if pos else "Uncategorized": count for pos, count in pos_stats},
                "relation_stats": {rel_type: count for rel_type, count in rel_stats}
            }
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            console.print(f"\n[green]Statistics exported to {args.export}[/]")
        
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        console.print(f"[red]An error occurred: {str(e)}[/]")
    finally:
        if conn:
            conn.close()

@with_transaction(commit=False)
def display_leaderboard(cur, console):
    """
    Display a leaderboard of dictionary contributors.
    This version normalizes the sources (using LOWER and TRIM) so that if the
    sources field contains extra data or multiple comma-separated values, the
    expected standardized names are still returned. In particular, KWF Dictionary
    entries will appear as 'KWF Diksiyonaryo ng Wikang Filipino' if the source
    text contains either 'kwf_dictionary.json' or 'kwf diksiyonaryo'.
    """
    try:
        console.print("\n[bold cyan]📊 Dictionary Contributors Leaderboard[/]", justify="center")
        
        # The CASE statement uses LOWER(TRIM(sources)) to catch matches even when the sources field 
        # contains additional text (like comma-separated values). Adjust the search strings as needed.
        cur.execute("""
            SELECT 
                 CASE
                      WHEN LOWER(TRIM(sources)) ILIKE '%kaikki-ceb.jsonl%' THEN 'kaikki.org (Cebuano)'
                     WHEN LOWER(TRIM(sources)) ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                     WHEN LOWER(TRIM(sources)) ILIKE '%kwf_dictionary.json%' 
                          OR LOWER(TRIM(sources)) ILIKE '%kwf diksiyonaryo%' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                     WHEN LOWER(TRIM(sources)) ILIKE '%root_words_with_associated_words_cleaned.json%' THEN 'tagalog.com'
                     WHEN LOWER(TRIM(sources)) ILIKE '%tagalog-words.json%' THEN 'diksiyonaryo.ph'
                     ELSE sources
                 END AS source_name,
                 COUNT(*) AS def_count,
                 COUNT(DISTINCT word_id) AS unique_count
            FROM definitions
            GROUP BY source_name
            ORDER BY def_count DESC
            LIMIT 10;
        """)
        results = cur.fetchall()
        
        if results:
            table = Table(title="Top Contributors", box=box.ROUNDED, border_style="cyan")
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Source", style="yellow")
            table.add_column("Definitions", justify="right", style="green")
            table.add_column("Unique Words", justify="right", style="blue")
            
            for i, (source, def_count, unique_count) in enumerate(results, start=1):
                medal = ""
                if i == 1:
                    medal = "🥇 "
                elif i == 2:
                    medal = "🥈 "
                elif i == 3:
                    medal = "🥉 "
                table.add_row(
                    f"{medal}{i}",
                    source,
                    f"{def_count:,}",
                    f"{unique_count:,}"
                )
            console.print(table)
            
            # Query for recently active sources (past 30 days)
            cur.execute("""
                SELECT 
                     CASE
                          WHEN LOWER(TRIM(sources)) ILIKE '%kaikki-ceb.jsonl%' THEN 'kaikki.org (Cebuano)'
                         WHEN LOWER(TRIM(sources)) ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                         WHEN LOWER(TRIM(sources)) ILIKE '%kwf_dictionary.json%' 
                              OR LOWER(TRIM(sources)) ILIKE '%kwf diksiyonaryo%' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                         WHEN LOWER(TRIM(sources)) ILIKE '%root_words_with_associated_words_cleaned.json%' THEN 'tagalog.com'
                         WHEN LOWER(TRIM(sources)) ILIKE '%tagalog-words.json%' THEN 'diksiyonaryo.ph'
                         ELSE sources
                     END AS source_name,
                     MAX(created_at) AS last_activity,
                     COUNT(*) AS entries_past_month
                FROM definitions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY source_name
                ORDER BY last_activity DESC
                LIMIT 5;
            """)
            recent_results = cur.fetchall()
            
            if recent_results:
                recent_table = Table(title="Recently Active Sources (30d)", box=box.ROUNDED, border_style="magenta")
                recent_table.add_column("Source", style="yellow")
                recent_table.add_column("Last Activity", style="cyan")
                recent_table.add_column("Entries (30d)", justify="right", style="green")
                
                for source, last_activity, recent_count in recent_results:
                    activity_date = last_activity.strftime("%Y-%m-%d %H:%M")
                    recent_table.add_row(source, activity_date, f"{recent_count:,}")
                
                console.print("\n")
                console.print(recent_table)
        else:
            console.print("[yellow]No contributor data available yet.[/]")
    except Exception as e:
        logger.error(f"Error displaying leaderboard: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")

@with_transaction(commit=True)
def purge_database_tables(cur):
    """
    Safely delete all data from all dictionary tables.
    This version uses the transaction decorator for better reliability.
    """
    tables = [
        "definition_relations",
        "affixations",
        "relations",
        "etymologies",
        "definitions",
        "words",
        "parts_of_speech"
    ]
    
    for table in tables:
        print(f"Purging {table}...")
        cur.execute(f"DELETE FROM {table}")
    
    return True

def purge_database(args):
    """Safely delete all data from the database."""
    if not args.force:
        confirmation = input("WARNING: This will delete ALL dictionary data. Type 'DELETE ALL' to confirm: ")
        if confirmation != "DELETE ALL":
            print("Operation cancelled.")
            return
            
    try:
        conn = get_connection()
        cur = conn.cursor()
        purge_database_tables(cur)
        print("Database purged successfully.")
    except Exception as e:
        logger.error(f"Error during database purge: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        if conn:
            conn.close()

@with_transaction(commit=True)
def cleanup_relations(cur):
    """Clean up relation data by removing duplicates and standardizing relation types."""
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
    
    # Standardize relation types - more comprehensive mapping
    relation_mapping = {
        # Basic standardization
        'derived from': 'derived_from',
        'root of': 'root_of',
        'synonym of': 'synonym',
        'related to': 'related',
        
        # Redundancy reduction
        'kaugnay': 'related',  # Filipino for "related"
        'see_also': 'compare_with',
        'alternate_form': 'variant',
        'alternative_spelling': 'variant',
        'regional_form': 'variant',
        
        # Standardize Filipino terms to English equivalents
        'kasingkahulugan': 'synonym',
        'katulad': 'synonym',
        'kasalungat': 'antonym',
        'kabaligtaran': 'antonym',
        'uri_ng': 'hyponym_of',
        'mula_sa': 'derived_from',
        'varyant': 'variant',
        
        # Fix capitalization and plural forms
        'synonyms': 'synonym',
        'antonyms': 'antonym',
        'variants': 'variant',
        'Synonym': 'synonym',
        'Antonym': 'antonym',
        'Related': 'related'
    }
    
    # Apply the mapping
    for old, new in relation_mapping.items():
        cur.execute("""
            UPDATE relations
            SET relation_type = %s
            WHERE LOWER(relation_type) = %s
        """, (new, old))

    # Log the results
    cur.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type ORDER BY COUNT(*) DESC")
    relation_counts = cur.fetchall()
    logger.info(f"Relation types after cleanup:")
    for rel_type, count in relation_counts:
        logger.info(f"  {rel_type}: {count}")
    
    return True

@with_transaction(commit=True)
def deduplicate_definitions(cur):
    """Remove duplicate definitions while preserving the most recent versions."""
    logger.info("Starting definition deduplication process...")
    
    # Create temporary table with unique definitions
    cur.execute("""
        CREATE TEMP TABLE unique_definitions AS
        SELECT DISTINCT ON (
            word_id,
            definition_text,
            standardized_pos_id,
            examples,
            usage_notes
        ) * 
        FROM definitions
        ORDER BY word_id, definition_text, standardized_pos_id, examples, usage_notes, created_at DESC;
    """)
    
    # Replace definitions table content with deduplicated data
    cur.execute("DELETE FROM definitions")
    cur.execute("""
        INSERT INTO definitions
        SELECT * FROM unique_definitions;
    """)
    cur.execute("DROP TABLE unique_definitions")
    
    # Log results
    cur.execute("SELECT COUNT(*) FROM definitions")
    final_count = cur.fetchone()[0]
    logger.info(f"Definition deduplication complete. {final_count} unique definitions remain.")

    return final_count

@with_transaction(commit=True)
def cleanup_dictionary_data(cur):
    """Perform comprehensive cleanup of dictionary data."""
    logger.info("Starting dictionary cleanup process...")
    
    # Standardize parts of speech
    cur.execute("""
        WITH pos_standardization AS (
            SELECT d.id,
                   CASE
                       WHEN d.original_pos IN ('png', 'n', 'noun', 'pangngalan') THEN (SELECT id FROM parts_of_speech WHERE code = 'n')
                       WHEN d.original_pos IN ('pnr', 'adj', 'adjective', 'pang-uri') THEN (SELECT id FROM parts_of_speech WHERE code = 'adj')
                       WHEN d.original_pos IN ('pnw', 'v', 'verb', 'pandiwa') THEN (SELECT id FROM parts_of_speech WHERE code = 'v')
                       WHEN d.original_pos IN ('pny', 'adv', 'adverb', 'pang-abay') THEN (SELECT id FROM parts_of_speech WHERE code = 'adv')
                       ELSE standardized_pos_id
                   END as new_pos_id
            FROM definitions d
        )
        UPDATE definitions d
        SET standardized_pos_id = ps.new_pos_id
        FROM pos_standardization ps
        WHERE d.id = ps.id;
    """)
    
    # Standardize sources
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
    
    # Merge duplicate definitions 
    cur.execute("""
        WITH grouped_defs AS (
            SELECT word_id, definition_text, standardized_pos_id,
                   string_agg(DISTINCT sources, ' | ') as merged_sources,
                   string_agg(DISTINCT examples, ' | ') FILTER (WHERE examples IS NOT NULL AND examples != '') as all_examples,
                   string_agg(DISTINCT usage_notes, ' | ') FILTER (WHERE usage_notes IS NOT NULL AND usage_notes != '') as all_notes,
                   min(id) as keep_id
            FROM definitions
            GROUP BY word_id, definition_text, standardized_pos_id
            HAVING COUNT(*) > 1
        )
        UPDATE definitions d
        SET sources = g.merged_sources,
            examples = COALESCE(g.all_examples, d.examples),
            usage_notes = COALESCE(g.all_notes, d.usage_notes)
        FROM grouped_defs g
        WHERE d.id = g.keep_id;
    """)
    
    # Remove remaining duplicates
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
    
    # Update word tags with sources
    cur.execute("""
        WITH word_sources AS (
            SELECT d.word_id,
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

def store_processed_entry(cur, word_id: int, processed: Dict) -> None:
    try:
        source = processed.get('source', '')
        if not source:
            logger.warning(f"No source provided for word_id {word_id}, using 'unknown'")
            source = 'unknown'
        
        standardized_source = SourceStandardization.standardize_sources(source)
        if not standardized_source:
            logger.warning(f"Could not standardize source '{source}' for word_id {word_id}")
            standardized_source = source
        
        if 'data' in processed and 'definitions' in processed['data']:
            for definition in processed['data']['definitions']:
                try:
                    insert_definition(
                        cur,
                        word_id,
                        definition['text'],
                        part_of_speech=definition.get('pos', ''),
                        examples=json.dumps(definition.get('examples', [])),
                        usage_notes=json.dumps(definition.get('usage_notes', [])),
                        category=definition.get('domain'),
                        tags=json.dumps(definition.get('tags', [])),
                        sources=standardized_source
                    )
                except Exception as e:
                    logger.error(f"Error storing definition for word_id {word_id}: {str(e)}")
        if 'data' in processed and 'forms' in processed['data']:
            for form in processed['data']['forms']:
                if 'Baybayin' in form.get('tags', []):
                    try:
                        process_baybayin_data(
                            cur,
                            word_id,
                            form['form'],
                            form.get('romanized_form', get_romanized_text(form['form']))
                        )
                    except Exception as e:
                        logger.error(f"Error storing Baybayin form for word_id {word_id}: {str(e)}")
        if 'data' in processed and 'metadata' in processed['data']:
            metadata = processed['data']['metadata']
            if 'etymology' in metadata:
                try:
                    etymology_data = metadata['etymology']
                    insert_etymology(
                        cur,
                        word_id,
                        etymology_data,
                        normalized_components=json.dumps(extract_etymology_components(etymology_data)),
                        etymology_structure=None,
                        language_codes=",".join(extract_language_codes(etymology_data)),
                        sources=standardized_source
                    )
                except Exception as e:
                    logger.error(f"Error storing etymology for word_id {word_id}: {str(e)}")
            if 'pronunciation' in metadata:
                try:
                    pron_data = metadata['pronunciation']
                    cur.execute("""
                        UPDATE words 
                        SET pronunciation_data = %s
                        WHERE id = %s
                    """, (json.dumps(pron_data), word_id))
                except Exception as e:
                    logger.error(f"Error storing pronunciation for word_id {word_id}: {str(e)}")
        if 'data' in processed and 'related_words' in processed['data']:
            for rel_type, related in processed['data']['related_words'].items():
                for rel_word in related:
                    try:
                        rel_word_id = get_or_create_word_id(
                            cur,
                            rel_word,
                            language_code=processed.get('language_code', 'tl')
                        )
                        insert_relation(
                            cur,
                            word_id,
                            rel_word_id,
                            rel_type,
                            sources=standardized_source
                        )
                    except Exception as e:
                        logger.error(f"Error storing relation for word_id {word_id} and related word {rel_word}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in store_processed_entry for word_id {word_id}: {str(e)}")
        raise

def explore_dictionary():
    """Launches interactive dictionary explorer."""
    console = Console()
    console.print("\n[bold cyan]🔍 Interactive Dictionary Explorer[/]", justify="center")
    console.print("[dim]Navigate Filipino dictionary data with ease[/]\n", justify="center")
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        while True:
            console.print("\n[bold]Choose an option:[/]")
            options = [
                "1. Search for a word",
                "2. Browse random words",
                "3. Explore Baybayin words",
                "4. View word relations",
                "5. Show statistics",
                "0. Exit explorer"
            ]
            
            for option in options:
                console.print(f"  {option}")
            
            choice = input("\nEnter your choice (0-5): ")
            
            if choice == "0":
                break
            elif choice == "1":
                search_term = input("Enter search term: ")
                if not search_term.strip():
                    continue
                
                cur.execute("""
                    SELECT id, lemma, language_code, has_baybayin
                    FROM words
                    WHERE 
                         lemma ILIKE %s OR
                         normalized_lemma ILIKE %s OR
                         search_text @@ plainto_tsquery('simple', %s)
                    ORDER BY 
                         CASE WHEN lemma ILIKE %s THEN 0
                              WHEN lemma ILIKE %s THEN 1
                              ELSE 2
                         END,
                         length(lemma)
                    LIMIT 20
                """, (
                    f"{search_term}%",
                    f"{search_term}%",
                    search_term,
                    search_term,
                    f"{search_term}%"
                ))
                
                results = cur.fetchall()
                
                if not results:
                    console.print("[yellow]No matches found.[/]")
                    continue
                
                result_table = Table(title=f"Search Results for '{search_term}'", box=box.ROUNDED)
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Language", style="green")
                result_table.add_column("Baybayin", style="magenta")
                
                for word_id, lemma, lang_code, has_baybayin in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        "Tagalog" if lang_code == "tl" else "Cebuano",
                        "✓" if has_baybayin else ""
                    )
                
                console.print(result_table)
                
                word_choice = input("\nEnter word ID to view details (or press Enter to return): ")
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)
            
            elif choice == "2":
                cur.execute("""
                    SELECT id, lemma, language_code, has_baybayin
                    FROM words
                    ORDER BY RANDOM()
                    LIMIT 10
                """)
                
                results = cur.fetchall()
                
                if not results:
                    console.print("[yellow]No words found in the database.[/]")
                    continue
                
                result_table = Table(title="Random Words", box=box.ROUNDED)
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Language", style="green")
                result_table.add_column("Baybayin", style="magenta")
                
                for word_id, lemma, lang_code, has_baybayin in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        "Tagalog" if lang_code == "tl" else "Cebuano",
                        "✓" if has_baybayin else ""
                    )
                
                console.print(result_table)
                
                word_choice = input("\nEnter word ID to view details (or press Enter to return): ")
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)
            
            elif choice == "3":
                cur.execute("""
                    SELECT id, lemma, baybayin_form, romanized_form
                    FROM words
                    WHERE has_baybayin = TRUE
                    ORDER BY RANDOM()
                    LIMIT 10
                """)
                
                results = cur.fetchall()
                
                if not results:
                    console.print("[yellow]No Baybayin words found in the database.[/]")
                    continue
                
                result_table = Table(title="Baybayin Words", box=box.ROUNDED)
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Baybayin", style="magenta")
                result_table.add_column("Romanized", style="green")
                
                for word_id, lemma, baybayin, romanized in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        baybayin or "",
                        romanized or ""
                    )
                
                console.print(result_table)
                
                word_choice = input("\nEnter word ID to view details (or press Enter to return): ")
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)
            
            elif choice == "4":
                word_input = input("Enter word to find relations: ")
                if not word_input.strip():
                    continue
                
                cur.execute("""
                    SELECT id FROM words
                    WHERE lemma = %s OR normalized_lemma = %s
                    LIMIT 1
                """, (word_input, normalize_lemma(word_input)))
                
                result = cur.fetchone()
                
                if not result:
                    console.print(f"[yellow]Word '{word_input}' not found.[/]")
                    continue
                
                word_id = result[0]
                
                cur.execute("""
                    SELECT r.relation_type, w.id, w.lemma
                    FROM relations r
                    JOIN words w ON r.to_word_id = w.id
                    WHERE r.from_word_id = %s
                    ORDER BY r.relation_type, w.lemma
                """, (word_id,))
                
                relations = cur.fetchall()
                
                if not relations:
                    console.print(f"[yellow]No relations found for '{word_input}'.[/]")
                    continue
                
                relation_groups = {}
                for rel_type, rel_id, rel_word in relations:
                    if rel_type not in relation_groups:
                        relation_groups[rel_type] = []
                    relation_groups[rel_type].append((rel_id, rel_word))
                
                for rel_type, words in relation_groups.items():
                    console.print(f"\n[bold]{rel_type.title()}:[/]")
                    for rel_id, rel_word in words:
                        console.print(f"  {rel_word} (ID: {rel_id})")
                
                word_choice = input("\nEnter word ID to view details (or press Enter to return): ")
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)
            
            elif choice == "5":
                cur.execute("SELECT COUNT(*) FROM words")
                word_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM words WHERE language_code = 'tl'")
                tagalog_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM words WHERE language_code = 'ceb'")
                cebuano_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE")
                baybayin_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM definitions")
                def_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM relations")
                rel_count = cur.fetchone()[0]
                
                stats_table = Table(title="Quick Statistics", box=box.ROUNDED)
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Count", justify="right", style="green")
                
                stats_table.add_row("Total Words", f"{word_count:,}")
                stats_table.add_row("Tagalog Words", f"{tagalog_count:,}")
                stats_table.add_row("Cebuano Words", f"{cebuano_count:,}")
                stats_table.add_row("Baybayin Words", f"{baybayin_count:,}")
                stats_table.add_row("Definitions", f"{def_count:,}")
                stats_table.add_row("Relations", f"{rel_count:,}")
                
                console.print(stats_table)
                input("\nPress Enter to continue...")
            else:
                console.print("[yellow]Invalid choice. Please try again.[/]")
    except Exception as e:
        logger.error(f"Error in dictionary explorer: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")
    finally:
        if conn:
            conn.close()
        console.print("\n[bold green]Thank you for using the Dictionary Explorer![/]")

@with_transaction(commit=False)
def lookup_by_id(cur, word_id: int, console: Console):
    """Look up a word by its ID and display its information."""
    try:
        cur.execute("""
            SELECT lemma, language_code, has_baybayin, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """, (word_id,))
        
        result = cur.fetchone()
        
        if not result:
            console.print(f"[yellow]Word with ID {word_id} not found.[/]")
            return
        
        lemma, language_code, has_baybayin, baybayin_form, romanized_form = result
        
        console.print(f"\n[bold]Word Information - ID: {word_id}[/]")
        console.print(f"Lemma: {lemma}")
        console.print(f"Language: {'Tagalog' if language_code == 'tl' else 'Cebuano'}")
        
        if has_baybayin and baybayin_form:
            console.print(f"Baybayin Form: {baybayin_form}")
            if romanized_form:
                console.print(f"Romanized Form: {romanized_form}")
        
        cur.execute("""
            SELECT p.name_tl as pos, d.definition_text
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_tl, d.created_at
        """, (word_id,))
        
        definitions = cur.fetchall()
        
        if definitions:
            console.print("\n[bold]Definitions:[/]")
            current_pos = None
            for pos, definition in definitions:
                if pos != current_pos:
                    console.print(f"\n[cyan]{pos or 'Uncategorized'}[/]")
                    current_pos = pos
                console.print(f"• {definition}")
        
        cur.execute("""
            SELECT r.relation_type, w.lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            ORDER BY r.relation_type, w.lemma
        """, (word_id,))
        
        relations = cur.fetchall()
        
        if relations:
            console.print("\n[bold]Related Words:[/]")
            current_type = None
            for rel_type, rel_word in relations:
                if rel_type != current_type:
                    console.print(f"\n[magenta]{rel_type.title()}[/]")
                    current_type = rel_type
                console.print(f"• {rel_word}")
        
        input("\nPress Enter to continue...")
    
    except Exception as e:
        logger.error(f"Error looking up word ID {word_id}: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")

def test_database():
    """Run database connectivity tests."""
    console = Console()
    console.print("\n[bold cyan]🧪 Database Connection Tests[/]", justify="center")
    
    tests = [
        ("Database Connection", lambda: get_connection()),
        ("PostgreSQL Version", lambda: check_pg_version()),
        ("Tables Existence", lambda: check_tables_exist()),
        ("Extensions", lambda: check_extensions()),
        ("Data Access", lambda: check_data_access()),
        ("Query Performance", lambda: check_query_performance())
    ]
    
    test_table = Table(box=box.ROUNDED)
    test_table.add_column("Test", style="cyan")
    test_table.add_column("Status", style="bold")
    test_table.add_column("Details", style="dim")
    
    conn = None
    
    try:
        for test_name, test_func in tests:
            try:
                with Progress(SpinnerColumn(), TextColumn(f"Running {test_name} test..."), console=console) as progress:
                    task = progress.add_task("Testing", total=1)
                    result, details = test_func()
                    progress.update(task, completed=1)
                    
                    if result:
                        test_table.add_row(test_name, "[green]PASS[/]", details)
                    else:
                        test_table.add_row(test_name, "[red]FAIL[/]", details)
            except Exception as e:
                test_table.add_row(test_name, "[red]ERROR[/]", str(e))
    finally:
        if conn:
            conn.close()
    
    console.print(test_table)

def check_pg_version():
    """Check PostgreSQL version."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        major_version = int(version.split()[1].split('.')[0])
        return major_version >= 10, version
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def check_extensions():
    """Check if required extensions are installed."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        extensions = ['pg_trgm', 'unaccent', 'fuzzystrmatch']
        missing_extensions = []
        
        for ext in extensions:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_extension WHERE extname = %s
                )
            """, (ext,))
            
            if not cur.fetchone()[0]:
                missing_extensions.append(ext)
        
        if missing_extensions:
            return False, f"Missing extensions: {', '.join(missing_extensions)}"
        else:
            return True, f"All {len(extensions)} required extensions installed"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

@with_transaction(commit=True)
def check_data_access():
    """Check if data can be accessed."""
    try:
        cur = get_cursor()
        cur.execute("SELECT COUNT(*) FROM words")
        word_count = cur.fetchone()[0]
        
        test_word = f"test_word_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        normalized = normalize_lemma(test_word)
        
        cur.execute("""
            INSERT INTO words (lemma, normalized_lemma, language_code)
            VALUES (%s, %s, 'tl')
            RETURNING id
        """, (test_word, normalized))
        
        test_id = cur.fetchone()[0]
        cur.execute("DELETE FROM words WHERE id = %s", (test_id,))
        return True, f"Successfully read, wrote, and deleted data. Word count: {word_count:,}"
    except Exception as e:
        return False, str(e)
    
def check_tables_exist():
    """Check if all required tables exist in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'words'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'definitions'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'relations'
                ) AS tables_exist
            """)
            result = cur.fetchone()[0]
            return result

@with_transaction(commit=False)
def check_query_performance():
    """Check query performance."""
    try:
        cur = get_cursor()
        queries = [
            ("Simple Select", "SELECT COUNT(*) FROM words"),
            ("Join Query", """
                SELECT COUNT(*) 
                FROM words w
                JOIN definitions d ON w.id = d.word_id
            """),
            ("Index Usage", """
                SELECT COUNT(*) 
                FROM words 
                WHERE normalized_lemma LIKE 'a%'
            """)
        ]
        results = []
        for name, query in queries:
            start_time = datetime.now()
            cur.execute(query)
            result = cur.fetchone()[0]
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            results.append(f"{name}: {duration:.2f}ms")
        return True, "; ".join(results)
    except Exception as e:
        return False, str(e)

# -------------------------------------------------------------------
# CLI Wrapper Functions
# -------------------------------------------------------------------
def create_argument_parser_cli() -> argparse.ArgumentParser:
    return create_argument_parser()

def migrate_data_cli(args):
    try:
        migrate_data(args)
    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]Migration failed:[/] {str(e)}")
        if "database" in str(e).lower() and "exist" in str(e).lower():
            console.print("\n[bold yellow]The database may not exist.[/] You can create it using:")
            console.print("   [bold]python create_database.py[/]")
            console.print("Then run the migration again.")

def verify_database_cli(args):
    verify_database(args)

def purge_database_cli(args):
    """Run purge operations on the dictionary database."""
    try:
        cur = get_cursor()
        console = Console()
        console.print("[bold blue]Starting dictionary purge process...[/]")
        
        console.print("[yellow]Purging all dictionary data...[/]")
        purge_database_tables(cur)
        
        console.print("[bold green]Dictionary purge completed successfully.[/]")
    except Exception as e:
        print(f"Error during purge: {str(e)}")

def lookup_word_cli(args):
    lookup_word(args)

def display_dictionary_stats_cli(args):
    display_dictionary_stats(args)

def display_leaderboard_cli(args):
    """Display a leaderboard of dictionary contributors."""
    console = Console()
    display_leaderboard(get_cursor(), console)

def explore_dictionary_cli(args):
    explore_dictionary()

def test_database_cli(args):
    test_database()

def display_help_cli(args):
    display_help(args)

def cleanup_database_cli(args):
    """Run cleanup routines on the dictionary database."""
    try:
        cur = get_cursor()
        console = Console()
        console.print("[bold blue]Starting dictionary cleanup process...[/]")
        
        console.print("[yellow]Deduplicating definitions...[/]")
        deduplicate_definitions(cur)
        
        console.print("[yellow]Cleaning up relations...[/]")
        cleanup_relations(cur)
        
        console.print("[yellow]Cleaning up Baybayin data...[/]")
        cleanup_baybayin_data(cur)
        
        console.print("[yellow]Standardizing formats...[/]")
        cleanup_dictionary_data(cur)
        
        console.print("[bold green]Dictionary cleanup completed successfully.[/]")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main():
    parser = create_argument_parser_cli()
    args = parser.parse_args()
    if args.command == "migrate":
        migrate_data_cli(args)
    elif args.command == "verify":
        verify_database_cli(args)
    elif args.command == "purge":
        purge_database_cli(args)
    elif args.command == "lookup":
        lookup_word_cli(args)
    elif args.command == "stats":
        display_dictionary_stats_cli(args)
    elif args.command == "leaderboard":
        display_leaderboard_cli(args)
    elif args.command == "help":
        display_help_cli(args)
    elif args.command == "test":
        test_database_cli(args)
    elif args.command == "explore":
        explore_dictionary_cli(args)
    elif args.command == "cleanup":
        cleanup_database_cli(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()