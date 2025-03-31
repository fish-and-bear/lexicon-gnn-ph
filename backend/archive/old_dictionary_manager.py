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
import re
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

# Assume these backend modules are available
from backend.language_systems import LanguageSystem
from backend.dictionary_processor import DictionaryProcessor
from backend.language_types import *
from backend.source_standardization import SourceStandardization

# -------------------------------------------------------------------
# Initialize LanguageSystem and Processor
# -------------------------------------------------------------------
lsys = LanguageSystem()
processor = DictionaryProcessor(lsys)

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
load_dotenv()

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
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

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
# Setup Extensions
# -------------------------------------------------------------------
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
    try:
        extensions = [
            'pg_trgm',      # For similarity search and fuzzy matching
            'unaccent',     # For handling diacritics
            'fuzzystrmatch',# For Levenshtein distance
            'dict_xsyn'     # For text search dictionary
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
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection
            savepoint_name = f"sp_{func.__name__}"
            try:
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
TABLE_CREATION_SQL = r"""
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

-- Add search_text column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'words' AND column_name = 'search_text'
    ) THEN
        ALTER TABLE words ADD COLUMN search_text TSVECTOR;
        -- Initialize search_text for existing rows
        UPDATE words SET search_text = to_tsvector('simple',
            COALESCE(lemma, '') || ' ' ||
            COALESCE(normalized_lemma, '') || ' ' ||
            COALESCE(baybayin_form, '') || ' ' ||
            COALESCE(romanized_form, '')
        );
    END IF;
END
$$;

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
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT relations_unique UNIQUE (from_word_id, to_word_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

-- Drop and recreate etymologies table
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

-- Create triggers for updating timestamps
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

def create_or_update_tables(conn):
    logger.info("Starting table creation/update process.")
    cur = conn.cursor()
    try:
        # Drop all tables in correct order
        cur.execute("""
            DROP TABLE IF EXISTS 
                definition_relations, affixations, relations, etymologies, 
                definitions, words, parts_of_speech CASCADE;
        """)
        
        # Create tables with new schema
        cur.execute(TABLE_CREATION_SQL)
        conn.commit()  # Commit table creation first

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

# -------------------------------------------------------------------
# POS Mapping and Standardization
# -------------------------------------------------------------------
POS_MAPPING = {
    'noun': {'en': 'Noun', 'tl': 'Pangngalan', 'abbreviations': ['n', 'png'], 'variants': []},
    'adjective': {'en': 'Adjective', 'tl': 'Pang-uri', 'abbreviations': ['adj', 'pnr'], 'variants': []},
    'verb': {'en': 'Verb', 'tl': 'Pandiwa', 'abbreviations': ['v', 'pnw'], 'variants': []},
    'adverb': {'en': 'Adverb', 'tl': 'Pang-abay', 'abbreviations': ['adv', 'pny'], 'variants': []},
    'pronoun': {'en': 'Pronoun', 'tl': 'Panghalip', 'abbreviations': ['pron'], 'variants': []},
    'preposition': {'en': 'Preposition', 'tl': 'Pang-ukol', 'abbreviations': ['prep'], 'variants': []},
    'conjunction': {'en': 'Conjunction', 'tl': 'Pangatnig', 'abbreviations': ['conj'], 'variants': []},
    'interjection': {'en': 'Interjection', 'tl': 'Pandamdam', 'abbreviations': ['intj'], 'variants': []},
    'affix': {'en': 'Affix', 'tl': 'Panlapi', 'abbreviations': ['affix', 'pnl'], 'variants': []},
    'idiom': {'en': 'Idiom', 'tl': 'Idyoma', 'abbreviations': ['idm'], 'variants': []},
    'colloquial': {'en': 'Colloquial', 'tl': 'Kolokyal', 'abbreviations': ['col'], 'variants': []},
    'synonym': {'en': 'Synonym', 'tl': 'Singkahulugan', 'abbreviations': ['syn'], 'variants': []},
    'antonym': {'en': 'Antonym', 'tl': 'Di-kasingkahulugan', 'abbreviations': ['ant'], 'variants': []},
    'english': {'en': 'English', 'tl': 'Ingles', 'abbreviations': ['eng'], 'variants': []},
    'spanish': {'en': 'Spanish', 'tl': 'Espanyol', 'abbreviations': ['spa'], 'variants': []},
    'texting': {'en': 'Texting', 'tl': 'Texting', 'abbreviations': ['tx'], 'variants': []},
    'variant': {'en': 'Variant', 'tl': 'Varyant', 'abbreviations': ['var'], 'variants': []},
    'uncategorized': {'en': 'Uncategorized', 'tl': 'Hindi Tiyak', 'abbreviations': ['unc'], 'variants': []}
}

def standardize_pos(pos: str) -> str:
    if not pos:
        return ""
    pos_lower = pos.lower().strip()
    for key, mapping in POS_MAPPING.items():
        if pos_lower in {mapping['en'].lower(), mapping['tl'].lower(), key.lower()} or \
           pos_lower in {abbr.lower().strip('.') for abbr in mapping['abbreviations']} or \
           pos_lower in {var.lower() for var in mapping['variants']}:
            return mapping['tl']
    return pos

def get_standard_code(pos_key: str) -> str:
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
    pos_key = pos_key.lower().strip()
    if pos_key in code_mapping:
        return code_mapping[pos_key]
    for key, mapping in POS_MAPPING.items():
        if pos_key == mapping['tl'].lower():
            return code_mapping.get(key, 'unc')
    return 'unc'

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
        raise ValueError("Input text cannot be None or empty")
    return unidecode.unidecode(text).lower()

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

def get_standardized_source(source: str) -> str:
    source_mapping = {
        'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
        'kaikki.jsonl': 'kaikki.org (Tagalog)', 
        'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
        'root_words_with_associated_words_cleaned.json': 'tagalog.com',
        'tagalog-words.json': 'diksiyonaryo.ph'
    }
    return source_mapping.get(source, source)

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
        if not text:
            return False
        return any(0x1700 <= ord(c) <= 0x171F for c in text)

    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
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
        if not chars:
            return '', 0
        first_info = self.get_char_info(chars[0])
        if not first_info:
            return chars[0], 1
        if first_info.char_type == BaybayinCharType.VOWEL:
            return first_info.default_sound, 1
        if first_info.char_type == BaybayinCharType.CONSONANT:
            result = first_info.default_sound
            chars_processed = 1
            if len(chars) > 1:
                next_info = self.get_char_info(chars[1])
                if next_info:
                    if next_info.char_type == BaybayinCharType.VOWEL_MARK:
                        base = result[:-1]
                        result = base + next_info.default_sound
                        chars_processed = 2
                    elif next_info.char_type == BaybayinCharType.VIRAMA:
                        result = result[:-1]
                        chars_processed = 2
                        if len(chars) > 2:
                            next2 = self.get_char_info(chars[2])
                            if next2 and next2.char_type == BaybayinCharType.CONSONANT:
                                result += next2.default_sound
                                chars_processed = 3
            return result, chars_processed
        if first_info.char_type == BaybayinCharType.PUNCTUATION:
            return first_info.default_sound, 1
        return chars[0], 1

    def romanize(self, text: str) -> str:
        if not text:
            raise ValueError("Input text cannot be empty")
        try:
            result = []
            chars = list(text)
            i = 0
            while i < len(chars):
                if chars[i].isspace():
                    result.append(chars[i])
                    i += 1
                    continue
                romanized, nproc = self.process_syllable(chars[i:])
                if nproc == 0:
                    result.append(chars[i])
                    i += 1
                else:
                    result.append(romanized)
                    i += nproc
            return ''.join(result)
        except Exception as e:
            logger.warning(f"Error romanizing text '{text}': {str(e)}")
            return text

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        prev_info = None
        for char in text:
            if char.isspace():
                continue
            info = self.get_char_info(char)
            if not info:
                return False
            if prev_info:
                if prev_info.char_type == BaybayinCharType.VOWEL and info.char_type == BaybayinCharType.VOWEL:
                    return False
                if info.char_type == BaybayinCharType.VOWEL_MARK and prev_info.char_type != BaybayinCharType.CONSONANT:
                    return False
                if info.char_type == BaybayinCharType.VIRAMA and prev_info.char_type != BaybayinCharType.CONSONANT:
                    return False
            prev_info = info
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
    text = text.lower().strip()
    consonants = {
        'k': 'ᜃ', 'g': 'ᜄ', 'ng': 'ᜅ', 't': 'ᜆ', 'd': 'ᜇ', 'n': 'ᜈ',
        'p': 'ᜉ', 'b': 'ᜊ', 'm': 'ᜋ', 'y': 'ᜌ', 'l': 'ᜎ', 'w': 'ᜏ',
        's': 'ᜐ', 'h': 'ᜑ'
    }
    vowels = {'a': 'ᜀ', 'i': 'ᜁ', 'u': 'ᜂ'}
    vowel_marks = {'i': 'ᜒ', 'u': 'ᜓ'}
    result = ""
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

def merge_baybayin_entries(cur, baybayin_id: int, romanized_id: int):
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

def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> None:
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

def process_baybayin_entries(cur):
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

def cleanup_baybayin_data(cur):
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

def check_baybayin_consistency(cur):
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
def get_or_create_word_id_base(cur, lemma: str, language_code: str = "tl", **kwargs) -> int:
    try:
        _ = validate_word_data({"lemma": lemma, "language_code": language_code})
    except ValueError as ve:
        logger.error(f"Invalid word data for '{lemma}': {ve}")
        raise

    normalized = normalize_lemma(lemma)
    
    # Pre-validate Baybayin data before attempting any DB operations
    if kwargs.get('has_baybayin'):
        bb_form = kwargs.get('baybayin_form')
        r_form = kwargs.get('romanized_form')
        
        # Skip Baybayin validation for obvious non-Baybayin words
        should_skip_baybayin = any([
            len(lemma) == 1,  # Single letters
            lemma.isupper(),  # All caps words
            any(char in lemma.lower() for char in 'fjcñvxz'),  # Contains non-Baybayin compatible letters
            any(lemma.lower().endswith(suffix) for suffix in ['ismo', 'ista', 'dad', 'cion', 'syon']),  # Spanish suffixes
            lemma[0].isupper() and not lemma.isupper(),  # Proper nouns
            any(word in lemma.lower() for word in ['http', 'www', '.com', '.org', '.net']),  # URLs
            lemma.startswith(('-', '_', '.')),  # Special characters
            any(char.isdigit() for char in lemma),  # Contains numbers
        ])
        
        if should_skip_baybayin:
            kwargs['has_baybayin'] = False
            kwargs['baybayin_form'] = None
            kwargs['romanized_form'] = None
        elif bb_form and not validate_baybayin_entry(bb_form, r_form):
            # Only log warning for words that might actually be Baybayin
            logger.warning(f"Invalid Baybayin data for word '{lemma}', clearing baybayin fields.")
            kwargs['has_baybayin'] = False
            kwargs['baybayin_form'] = None
            kwargs['romanized_form'] = None

    cur.execute("""
        SELECT id FROM words 
        WHERE normalized_lemma = %s AND language_code = %s
    """, (normalized, language_code))
    result = cur.fetchone()
    
    if result:
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

    search_text = lemma + " " + normalized
    cur.execute("""
        INSERT INTO words (
            lemma, normalized_lemma, language_code, 
            root_word_id, tags, has_baybayin, 
            baybayin_form, romanized_form, search_text
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s)
        )
        ON CONFLICT ON CONSTRAINT words_lang_lemma_uniq
        DO UPDATE SET 
            lemma = EXCLUDED.lemma,
            tags = EXCLUDED.tags,
            search_text = to_tsvector('simple', EXCLUDED.lemma || ' ' || EXCLUDED.normalized_lemma),
            updated_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (
        lemma,
        normalized,
        language_code,
        kwargs.get('root_word_id'),
        kwargs.get('tags'),
        kwargs.get('has_baybayin', False),
        kwargs.get('baybayin_form'),
        kwargs.get('romanized_form'),
        search_text
    ))
    return cur.fetchone()[0]

@with_transaction(commit=True)
def get_or_create_word_id(cur, lemma: str, language_code: str = "tl", **kwargs) -> int:
    if 'entry_data' in kwargs:
        metadata = processor.process_word_metadata(kwargs['entry_data'],
                                                   SourceStandardization.get_source_enum(kwargs.get('source', '')))
        kwargs.update(metadata)
    return get_or_create_word_id_base(cur, lemma, language_code, **kwargs)

@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str, part_of_speech: str = "",
                      examples: str = None, usage_notes: str = None, category: str = None,
                      sources: str = "") -> Optional[int]:
    if 'Baybayin spelling of' in definition_text:
        return None
    cur.execute("""
        SELECT id FROM definitions WHERE word_id = %s AND definition_text = %s AND original_pos = %s
    """, (word_id, definition_text, part_of_speech))
    if cur.fetchone():
        return None
    std_pos_id = get_standardized_pos_id(cur, part_of_speech)
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
def insert_relation(cur, from_word_id: int, to_word_id: int, relation_type: str, sources: str = ""):
    if from_word_id == to_word_id:
        return
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
def insert_etymology(cur, word_id: int, etymology_text: str, normalized_components: Optional[str] = None,
                     language_codes: str = "", sources: str = "") -> None:
    if not etymology_text:
        return
    if 'Baybayin spelling of' in etymology_text:
        return
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
def insert_affixation(cur, root_id: int, affixed_id: int, affix_type: str, sources: str):
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

def batch_get_or_create_word_ids(cur, entries: List[Tuple[str, str]], batch_size: int = 1000) -> Dict[Tuple[str, str], int]:
    result = {}
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        # Remove duplicates from the batch
        batch = list(dict.fromkeys(batch))
        normalized_entries = [(lemma, normalize_lemma(lemma), lang_code) for lemma, lang_code in batch]
        
        # Get existing entries
        cur.execute("""
            SELECT lemma, language_code, id
            FROM words
            WHERE (normalized_lemma, language_code) IN %s
        """, (tuple((norm, lang) for _, norm, lang in normalized_entries),))
        existing = {(lemma, lang): id for lemma, lang, id in cur.fetchall()}
        
        # Filter out entries that already exist
        to_insert = [(lemma, norm, lang) for lemma, norm, lang in normalized_entries if (lemma, lang) not in existing]
        
        if to_insert:
            # Process each entry individually to avoid tsvector issues
            for lemma, norm, lang in to_insert:
                try:
                    # Clean and prepare search text
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
                    existing[(lemma, lang)] = word_id
                except Exception as e:
                    logger.error(f"Error processing entry {lemma}: {str(e)}")
                    continue
                
        result.update(existing)
    return result

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------
def process_kwf_entry(cur, lemma: str, entry: Dict, src: str = "kwf"):
    try:
        src = SourceStandardization.get_display_name('kwf_dictionary.json')
        word_id = get_or_create_word_id(cur, entry['formatted'], language_code="tl", tags="KWF")
        for pos in entry.get('part_of_speech', []):
            pos_definitions = entry.get('definitions', {}).get(pos, [])
            for def_entry in pos_definitions:
                try:
                    insert_definition(
                        cur,
                        word_id,
                        def_entry['meaning'],
                        pos,
                        category=def_entry.get('category'),
                        sources=src
                    )
                    for synonym in def_entry.get('synonyms', []):
                        try:
                            if not synonym or not synonym.strip():
                                continue
                            syn_id = get_or_create_word_id(cur, synonym, language_code="tl")
                            insert_relation(cur, word_id, syn_id, "synonym", src)
                        except Exception as e:
                            logger.error(f"Error processing synonym {synonym} for {lemma}: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Error processing definition for {lemma}: {str(e)}")
                    continue
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
                cleaned_examples = [ex.strip() for ex in idiom_examples if ex and ex.strip()]
                idiom_obj = {
                    "idiom": idiom_text,
                    "meaning": idiom_meaning,
                    "examples": cleaned_examples if cleaned_examples else None
                }
                idioms_list.append(idiom_obj)
            except Exception as e:
                logger.error(f"Error processing idiom '{idiom.get('idiom', '')}' for {lemma}: {str(e)}")
                continue
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
    components = []
    def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
        word = text.split('(')[0].strip(' "\'')
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
        text = etymology.strip()
        if not text:
            return []
        if "blend of" in text.lower():
            blend_parts = text.split("blend of", 1)[1].split(".")[0].strip()
            parts = re.findall(r'"([^"]+)"(?:\s*\([^)]+\))?', blend_parts)
            for part in parts:
                word, meaning = extract_meaning(part)
                if word:
                    comp = word + (f" ({meaning})" if meaning else "")
                    components.append(comp)
        elif "from" in text.lower():
            from_parts = text.split("from", 1)[1].split(".")[0].strip()
            parts = re.findall(r'"([^"]+)"(?:\s*\([^)]+\))?', from_parts)
            if parts:
                for part in parts:
                    word, meaning = extract_meaning(part)
                    if word:
                        comp = word + (f" ({meaning})" if meaning else "")
                        components.append(comp)
            elif "+" in from_parts:
                for part in from_parts.split("+"):
                    word, meaning = extract_meaning(part)
                    if word:
                        comp = word + (f" ({meaning})" if meaning else "")
                        components.append(comp)
        elif "+" in text:
            for part in text.split("+"):
                word, meaning = extract_meaning(part)
                if word:
                    comp = word + (f" ({meaning})" if meaning else "")
                    components.append(comp)
    except Exception as e:
        logger.debug(f"Error extracting etymology components: {str(e)}")
    return [c.strip() for c in components if c.strip()]

def process_kaikki_jsonl_new(cur, filename: str):
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        import re
        baybayin = None
        romanized = None
        try:
            word = entry.get('word', '')
            if not word:
                return None, None
            if any(word.lower().endswith(suffix) for suffix in ['ismo', 'ista', 'dad', 'cion', 'syon']):
                return None, None
            if any(char in word.lower() for char in 'fjcñvxz'):
                return None, None
            if word[0].isupper() and not word.isupper():
                return None, None
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
                            match = re.search(r'Baybayin spelling[^(]*\((.*?)\)', expansion)
                            if match:
                                candidates = re.findall(r'([ᜀ-ᜟ᜔ᜒᜓ]+)', match.group(1))
                                if candidates:
                                    candidate = candidates[0].strip()
                                    if candidate and re.match(r'^[ᜀ-ᜟ᜔ᜒᜓ]+$', candidate):
                                        baybayin = candidate
                                        break
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
            if baybayin:
                if forms and isinstance(forms, list):
                    for form in forms:
                        if isinstance(form, dict) and form.get('tags'):
                            if isinstance(form.get('tags'), list) and 'canonical' in form.get('tags', []):
                                candidate = form.get('form', '').strip()
                                if candidate:
                                    romanized = candidate
                                    break
                if not romanized:
                    romanized = word
            return baybayin, romanized
        except Exception as e:
            logger.debug(f"Error in Baybayin extraction for {word if 'word' in locals() else 'unknown'}: {str(e)}")
            return None, None

    def standardize_entry_pos(pos_str: str) -> str:
        if not pos_str:
            return "Hindi Tiyak"
        return standardize_pos(pos_str)
    
    def process_entry(cur, entry: Dict):
        try:
            lemma = entry.get("word", "").strip()
            if not lemma:
                return
            try:
                baybayin_form, romanized = extract_baybayin_info(entry)
                has_baybayin = bool(baybayin_form)
            except Exception as e:
                logger.debug(f"Error extracting Baybayin info for {lemma}: {str(e)}")
                baybayin_form, romanized = None, None
                has_baybayin = False
            language_code = "tl" if "kaikki.jsonl" in filename else "ceb"
            source = SourceStandardization.get_display_name(os.path.basename(filename))
            pronunciation_data = {}
            if 'sounds' in entry:
                pronunciation_data['sounds'] = entry['sounds']
            if 'hyphenation' in entry:
                pronunciation_data['hyphenation'] = entry['hyphenation']
            entry_pos = None
            if 'head_templates' in entry:
                for template in entry.get('head_templates', []):
                    if isinstance(template, dict):
                        template_name = template.get('name', '').lower()
                        if template_name.startswith(('tl-', 'ceb-')):
                            pos_part = template_name.split('-')[1]
                            entry_pos = standardize_entry_pos(pos_part)
                            if 'expansion' in template:
                                pronunciation_data['template_expansion'] = template['expansion']
                            if 'args' in template:
                                pronunciation_data['template_args'] = template['args']
                            break
            if not entry_pos and 'pos' in entry:
                entry_pos = standardize_entry_pos(entry.get('pos', ''))
            tags = set()
            if 'tags' in entry:
                entry_tags = entry.get('tags', [])
                if isinstance(entry_tags, list):
                    tags.update(tag.strip() for tag in entry_tags if tag and tag.strip())
            if 'forms' in entry:
                for form in entry.get('forms', []):
                    if isinstance(form, dict):
                        form_tags = form.get('tags', [])
                        if isinstance(form_tags, list):
                            tags.update(tag.strip() for tag in form_tags if tag and tag.strip())
                        if 'form' in form and form['form'] != lemma:
                            tags.add(f"Variant: {form['form']}")
            if 'etymology_text' in entry:
                etymology = entry.get('etymology_text', '')
                if etymology:
                    tags.add(f"Etymology: {etymology}")
                    if 'Borrowed from' in etymology:
                        match = re.search(r'Borrowed from (\w+)', etymology)
                        if match:
                            tags.add(f"Source Language: {match.group(1)}")
                    elif 'from' in etymology.lower():
                        match = re.search(r'from (\w+)', etymology.lower())
                        if match:
                            tags.add(f"Source Language: {match.group(1)}")
            if 'etymology_templates' in entry:
                for template in entry.get('etymology_templates', []):
                    if isinstance(template, dict):
                        if 'expansion' in template:
                            tags.add(f"Etymology Detail: {template['expansion']}")
            if 'hyphenation' in entry:
                hyphenation = entry.get('hyphenation', [])
                if isinstance(hyphenation, list) and hyphenation:
                    tags.add(f"Hyphenation: {'-'.join(hyphenation)}")
            if 'sounds' in entry:
                for sound in entry.get('sounds', []):
                    if isinstance(sound, dict):
                        if 'homophone' in sound:
                            tags.add(f"Homophone: {sound['homophone']}")
                        if 'rhymes' in sound:
                            tags.add(f"Rhymes: {sound['rhymes']}")
            for sense in entry.get("senses", []):
                if isinstance(sense, dict):
                    sense_tags = sense.get("tags", [])
                    if isinstance(sense_tags, list):
                        tags.update(tag.strip() for tag in sense_tags if tag and tag.strip())
                    if 'topics' in sense:
                        topics = sense.get("topics", [])
                        if isinstance(topics, list):
                            tags.update(f"Topic: {topic}" for topic in topics if topic and isinstance(topic, str))
                    if 'links' in sense:
                        links = sense.get("links", [])
                        if isinstance(links, list):
                            for link in links:
                                if isinstance(link, list) and len(link) >= 2:
                                    tags.add(f"Related: {link[0]}")
                    if 'synonyms' in sense:
                        synonyms = sense.get("synonyms", [])
                        if isinstance(synonyms, list):
                            for syn in synonyms:
                                if isinstance(syn, dict):
                                    syn_word = syn.get("word", "")
                                    syn_tags = syn.get("tags", [])
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
                    if 'categories' in sense:
                        for cat in sense.get("categories", []):
                            if isinstance(cat, dict) and "name" in cat:
                                if cat["name"] and not any(x in cat["name"].lower() for x in ["entries with", "terms with"]):
                                    if cat.get("kind"):
                                        tags.add(f"{cat['kind']}: {cat['name']}")
                                    else:
                                        tags.add(cat["name"])
                                    for parent in cat.get("parents", []):
                                        if parent and not any(x in parent.lower() for x in ["entries with", "terms with"]):
                                            tags.add(f"Category: {parent}")
                                    if cat.get("source"):
                                        tags.add(f"Category Source: {cat['source']}")
                    if 'sounds' in sense:
                        for sound in sense.get("sounds", []):
                            if isinstance(sound, dict):
                                if 'ipa' in sound:
                                    dialect = f" ({sound.get('dialect')})" if sound.get('dialect') else ""
                                    tags.add(f"IPA{dialect}: {sound['ipa']}")
                                if 'tags' in sound and isinstance(sound['tags'], list):
                                    tags.update(f"Pronunciation: {tag}" for tag in sound['tags'] if tag and isinstance(tag, str))
                                if 'note' in sound:
                                    tags.add(f"Pronunciation Note: {sound['note']}")
            if 'pos' in entry and not entry_pos:
                entry_pos = standardize_entry_pos(entry.get('pos', ''))
            tags_str = '; '.join(sorted(tag for tag in tags if tag))
            word_id = get_or_create_word_id(cur, lemma, language_code=language_code,
                                             has_baybayin=bool(baybayin_form),
                                             baybayin_form=baybayin_form,
                                             romanized_form=romanized if baybayin_form else None,
                                             tags=tags_str,
                                             pronunciation_data=json.dumps(pronunciation_data) if pronunciation_data else None)
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
                    sense_pos = sense.get("pos", entry_pos)
                    if sense_pos:
                        sense_pos = standardize_entry_pos(sense_pos)
                    elif entry_pos:
                        sense_pos = entry_pos
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
                    usage_notes = []
                    categories = []
                    if sense.get("tags"):
                        usage_notes.extend(sense["tags"])
                    if raw_glosses:
                        for raw_gloss in raw_glosses:
                            if raw_gloss and raw_gloss not in glosses:
                                usage_notes.append(f"Context: {raw_gloss}")
                    if sense.get("categories"):
                        for cat in sense["categories"]:
                            if isinstance(cat, dict) and "name" in cat:
                                if cat["name"] and not any(x in cat["name"].lower() for x in ["entries with", "terms with"]):
                                    categories.append(cat["name"])
                    for gloss in glosses:
                        if gloss and "Baybayin spelling of" not in str(gloss):
                            insert_definition(cur, word_id, str(gloss).strip(), sense_pos or entry_pos,
                                              examples="\n".join(examples) if examples else None,
                                              usage_notes="; ".join(usage_notes) if usage_notes else None,
                                              category="; ".join(categories) if categories else None,
                                              sources=source)
            etymology = entry.get("etymology_text", "")
            if etymology:
                try:
                    components = extract_etymology_components(etymology)
                    codes, cleaned_ety = lsys.extract_and_remove_language_codes(etymology)
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
                    insert_etymology(cur=cur, word_id=word_id, etymology_text=cleaned_ety,
                                     normalized_components=", ".join(components) if components else None,
                                     language_codes=", ".join(codes) if codes else "",
                                     sources=source)
                except Exception as e:
                    logger.debug(f"Error processing etymology for {lemma}: {str(e)}")
                    insert_etymology(cur=cur, word_id=word_id, etymology_text=etymology,
                                     normalized_components=None,
                                     language_codes="",
                                     sources=source)
            related = entry.get("related", [])
            if isinstance(related, list):
                processed_relations = set()
                for rel in related:
                    try:
                        if isinstance(rel, dict):
                            rel_word = rel.get("word", "").strip()
                        else:
                            rel_word = str(rel).strip()
                        if not rel_word:
                            continue
                        rel_id = get_or_create_word_id(cur, rel_word, language_code)
                        if rel_id != word_id:
                            relation_key = (word_id, rel_id)
                            if relation_key not in processed_relations:
                                insert_relation(cur, word_id, rel_id, "kaugnay", source)
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
    logger.info(f"Processing Tagalog words from: {filename}")
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return
    src = SourceStandardization.get_display_name(os.path.basename(filename))
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        if isinstance(entries, dict):
            with tqdm(total=len(entries), desc="Processing Tagalog words") as pbar:
                for lemma, entry_data in entries.items():
                    try:
                        # Create word entry
                        normalized = normalize_lemma(lemma)
                        # Clean and prepare search text
                        search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {normalized}"))
                        
                        # Insert word directly with error handling
                        try:
                            cur.execute("""
                                INSERT INTO words (lemma, normalized_lemma, language_code, tags, search_text)
                                VALUES (%s, %s, %s, %s, to_tsvector('simple', %s))
                                ON CONFLICT ON CONSTRAINT words_lang_lemma_uniq
                                DO UPDATE SET 
                                    lemma = EXCLUDED.lemma,
                                    tags = EXCLUDED.tags,
                                    search_text = EXCLUDED.search_text,
                                    updated_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """, (lemma, normalized, 'tl', src, search_text))
                            
                            word_id = cur.fetchone()[0]
                            
                            # Process definitions with duplicate handling
                            pos_values = standardize_entry_pos(entry_data.get('part_of_speech', ''))
                            if 'definitions' in entry_data:
                                for definition in entry_data['definitions']:
                                    def_text = definition if isinstance(definition, str) else definition.get('text', '')
                                    if not def_text.strip():  # Skip empty definitions
                                        continue
                                        
                                    for pos in pos_values:
                                        try:
                                            # Get standardized POS ID
                                            pos_id = get_standardized_pos_id(cur, pos)
                                            
                                            # Insert definition with ON CONFLICT DO NOTHING
                                            cur.execute("""
                                                INSERT INTO definitions 
                                                    (word_id, definition_text, original_pos, standardized_pos_id, sources)
                                                VALUES (%s, %s, %s, %s, %s)
                                                ON CONFLICT (word_id, definition_text, standardized_pos_id)
                                                DO NOTHING
                                            """, (word_id, def_text, pos, pos_id, src))
                                            
                                        except Exception as e:
                                            logger.debug(f"Skipping duplicate definition for {lemma} ({def_text}): {str(e)}")
                                            continue
                            
                            cur.connection.commit()
                            
                        except Exception as e:
                            logger.error(f"Error processing word {lemma}: {str(e)}")
                            cur.connection.rollback()
                            continue
                            
                        pbar.update(1)
                        
                    except Exception as e:
                        cur.connection.rollback()
                        logger.error(f"Error processing lemma {lemma}: {str(e)}")
                        continue
        else:
            raise ValueError("Unsupported data format in tagalog-words.json")
    except Exception as e:
        logger.error(f"Error processing Tagalog words file: {str(e)}")
        raise

def standardize_entry_pos(pos_str: str) -> List[str]:
    if not pos_str:
        return ['Hindi Tiyak']
    pos_parts = pos_str.lower().strip().split()
    standardized = []
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
        if pos_clean in direct_map:
            standardized.append(direct_map[pos_clean])
            continue
        standardized.append(standardize_pos(pos_clean))
    if not standardized:
        standardized = ['Hindi Tiyak']
    return list(dict.fromkeys(standardized))

def process_root_words(cur, filename: str):
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
                continue
    except Exception as e:
        logger.error(f"Error processing root words file: {str(e)}")
        raise

def process_kwf_dictionary(cur, filename: str):
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
                process_kwf_entry(cur, lemma, entry, src=src)
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
    parser = argparse.ArgumentParser(description="Manage dictionary data in PostgreSQL.")
    subparsers = parser.add_subparsers(dest="command")
    migrate_parser = subparsers.add_parser("migrate", help="Create/update schema and load data")
    migrate_parser.add_argument("--check-exists", action="store_true", help="Skip identical entries")
    migrate_parser.add_argument("--force", action="store_true", help="Force migration without confirmation")
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
    return parser

def migrate_data(args):
    conn = None
    cur = None
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
        setup_extensions(conn)
        create_or_update_tables(conn)
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
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=Console()) as progress:
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
        try:
            logger.info("Starting post-processing steps...")
            process_baybayin_entries(cur)
            cleanup_baybayin_data(cur)
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
        from typing import List  # ensure List is imported
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
            dupes = cur.fetchone()
            if dupes and dupes[1] > 0:
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

def repair_database_issues(cur, issues):
    try:
        cur.execute("""
            DELETE FROM relations r
            WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
            OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
        """)
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
        cur.connection.commit()
        logger.info("Database repairs completed")
    except Exception as e:
        cur.connection.rollback()
        logger.error(f"Error during repairs: {str(e)}")

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
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Normalize the word in Python before using in SQL
        normalized_word = normalize_lemma(args.word)
        
        # Get word details with all sources
        cur.execute("""
            WITH word_sources AS (
                SELECT DISTINCT w.id, 
                    array_agg(DISTINCT d.sources) as definition_sources,
                    array_agg(DISTINCT e.sources) as etymology_sources,
                    w.tags,
                    w.source_info
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN etymologies e ON w.id = e.word_id
                WHERE w.normalized_lemma = %s
                GROUP BY w.id, w.tags, w.source_info
            )
            SELECT w.id, w.lemma, w.language_code, w.has_baybayin, w.baybayin_form, 
                   w.romanized_form, w.preferred_spelling, w.pronunciation_data, w.idioms,
                   ws.definition_sources, ws.etymology_sources, ws.tags, ws.source_info,
                   w.root_word_id
            FROM words w
            JOIN word_sources ws ON w.id = ws.id
        """, (normalized_word,))
        
        word_data = cur.fetchone()
        if not word_data:
            print(f"Word '{args.word}' not found.")
            return

        word_id, lemma, language_code, has_baybayin, baybayin_form, romanized_form, \
        preferred_spelling, pronunciation_data, idioms, definition_sources, etymology_sources, \
        tags, source_info, root_word_id = word_data

        # Process and categorize all sources
        all_sources = set()
        if tags:
            all_sources.update(tag.strip() for tag in tags.split(','))
        if source_info:
            all_sources.update(s.strip() for s in source_info.get('sources', []))
        for sources in definition_sources:
            if sources:
                all_sources.update(s.strip() for s in sources.split(','))
        for sources in etymology_sources:
            if sources:
                all_sources.update(s.strip() for s in sources.split(','))

        # Categorize sources
        source_categories = {
            'Dictionaries': set(),
            'Online Resources': set(),
            'Academic Sources': set()
        }

        for source in all_sources:
            display_name = SourceStandardization.get_display_name(source)
            if not display_name:
                continue
            if any(kw in display_name.lower() for kw in ['dictionary', 'diksiyonaryo']):
                source_categories['Dictionaries'].add(display_name)
            elif any(kw in display_name.lower() for kw in ['.org', '.com', '.net', 'online']):
                source_categories['Online Resources'].add(display_name)
            else:
                source_categories['Academic Sources'].add(display_name)

        # Display word information
        console = Console()
        console.print("\n[bold]Word Information[/]", justify="center")
        
        word_info = []
        # Basic word info
        word_info.append(f"Word: {format_word_display(lemma, has_baybayin)}")
        word_info.append(f"Language: {'Tagalog' if language_code == 'tl' else 'Cebuano'}")
        
        # Add preferred spelling if available
        if preferred_spelling:
            word_info.append(f"Preferred Spelling: {preferred_spelling}")

        # Add Baybayin information if available
        if has_baybayin and baybayin_form:
            word_info.append(f"Baybayin Form: {baybayin_form}")
            if romanized_form:
                word_info.append(f"Romanized Form: {romanized_form}")

        # Add pronunciation data if available
        if pronunciation_data and isinstance(pronunciation_data, dict):
            if 'ipa' in pronunciation_data:
                word_info.append(f"IPA: {pronunciation_data['ipa']}")
            if 'audio' in pronunciation_data:
                word_info.append(f"Audio: {pronunciation_data['audio']}")
            if 'hyphenation' in pronunciation_data:
                word_info.append(f"Hyphenation: {pronunciation_data['hyphenation']}")
            if 'sounds' in pronunciation_data:
                for sound in pronunciation_data['sounds']:
                    if isinstance(sound, dict):
                        if 'ipa' in sound:
                            dialect = f" ({sound.get('dialect', 'Standard')})"
                            word_info.append(f"IPA{dialect}: {sound['ipa']}")

        # Add root word if available
        if root_word_id:
            cur.execute("SELECT lemma FROM words WHERE id = %s", (root_word_id,))
            root_result = cur.fetchone()
            if root_result:
                word_info.append(f"Root Word: {root_result[0]}")

        # Add non-empty source categories
        for category, sources in source_categories.items():
            if sources:
                word_info.append(f"{category}: {', '.join(sorted(sources))}")

        # Add idioms if available
        if idioms and idioms != '[]':
            try:
                if isinstance(idioms, str):
                    idioms = json.loads(idioms)
                if isinstance(idioms, list):
                    idiom_entries = []
                    for idiom in idioms:
                        if isinstance(idiom, dict):
                            idiom_text = idiom.get('text', '') or idiom.get('idiom', '')
                            idiom_meaning = idiom.get('meaning', '')
                            if idiom_text and idiom_meaning:
                                idiom_entries.append(f"{idiom_text} - {idiom_meaning}")
                    if idiom_entries:
                        word_info.append("\nIdioms:")
                        word_info.extend(f"• {entry}" for entry in idiom_entries)
            except json.JSONDecodeError:
                pass

        console.print(Panel("\n".join(word_info), box=box.ROUNDED, expand=True))

        # Get and display definitions
        cur.execute("""
            SELECT p.name_tl as pos, d.definition_text, d.examples, d.usage_notes, d.sources,
                   d.original_pos
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_tl, d.created_at
        """, (word_id,))
        
        definitions = cur.fetchall()
        if definitions:
            console.print("\n[bold]Definitions[/]", justify="center")
            table = Table("POS", "Definition", "Examples", "Usage Notes", "Sources", 
                         box=box.ROUNDED, expand=True, show_lines=True, padding=(0, 1))
            
            current_pos = None
            for pos, definition, examples, usage_notes, sources, original_pos in definitions:
                # Format examples
                example_text = ""
                if examples:
                    example_list = [f"• {ex.strip()}" for ex in examples.split('\n') if ex.strip()]
                    example_text = "\n".join(example_list)
                
                # Format usage notes
                usage_text = ""
                if usage_notes:
                    usage_list = [f"• {note.strip()}" for note in usage_notes.split('\n') if note.strip()]
                    usage_text = "\n".join(usage_list)
                
                # Format sources
                source_list = []
                if sources:
                    for source in sources.split(','):
                        display_name = SourceStandardization.get_display_name(source.strip())
                        if display_name and display_name not in source_list:
                            source_list.append(display_name)
                
                # Display POS with original if different
                display_pos = pos or "Hindi Tiyak"
                if original_pos and original_pos.lower() not in (pos or "").lower():
                    display_pos = f"{display_pos} ({original_pos})"
                
                table.add_row(
                    Text(display_pos, style="blue") if display_pos != current_pos else "",
                    Text(definition, style="bold"),
                    Text(example_text, style="green") if example_text else "",
                    Text(usage_text, style="yellow") if usage_text else "",
                    Text(", ".join(sorted(source_list)), style="magenta", overflow="fold")
                )
                current_pos = display_pos
            
            console.print(table)

        # Get and display etymologies
        cur.execute("""
            SELECT etymology_text, normalized_components, language_codes, sources
            FROM etymologies
            WHERE word_id = %s
            ORDER BY created_at
        """, (word_id,))
        
        etymologies = cur.fetchall()
        if etymologies:
            console.print("\n[bold]Etymology[/]", justify="center")
            for etymology_text, components, langs, sources in etymologies:
                if not etymology_text.strip():
                    continue
                
                etymology_panel = []
                etymology_panel.append(Text(etymology_text, style="cyan"))
                
                if components:
                    comp_list = sorted(set(comp.strip() for comp in components.split(';') if comp.strip()))
                    if comp_list:
                        etymology_panel.append(Text("\nComponents:", style="bold"))
                        etymology_panel.extend(Text(f"• {comp}", style="green") for comp in comp_list)
                
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
                
                console.print(Panel("\n".join(str(line) for line in etymology_panel), 
                                  box=box.ROUNDED, expand=True, border_style="blue"))

        # Get and display relations with deduplication
        cur.execute("""
            SELECT DISTINCT r.relation_type, w2.lemma, p.name_tl as pos, w2.id
            FROM relations r
            JOIN words w2 ON r.to_word_id = w2.id
            LEFT JOIN definitions d ON w2.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE r.from_word_id = %s
            ORDER BY r.relation_type, w2.lemma
        """, (word_id,))
        
        relations = cur.fetchall()
        if relations:
            console.print("\n[bold]Related Words[/]", justify="center")
            relation_groups = {}
            seen_words = set()  # Track unique word-POS combinations
            
            for rel_type, rel_word, pos, rel_id in relations:
                word_pos = f"{rel_word} ({pos})" if pos else rel_word
                word_key = (rel_word, pos)  # Create a unique key for each word-POS combination
                
                if word_key not in seen_words:
                    relation_groups.setdefault(rel_type, set()).add(word_pos)
                    seen_words.add(word_key)
            
            table = Table("Relation Type", "Related Words", 
                         box=box.ROUNDED, expand=True, show_lines=True, padding=(0, 1))
            for rel_type, words in sorted(relation_groups.items()):
                table.add_row(
                    Text(rel_type, style="bold yellow"),
                    Text(", ".join(sorted(words)), style="cyan", overflow="fold")
                )
            console.print(table)

    except Exception as e:
        logger.error(f"Error during word lookup: {str(e)}")
        print(f"An error occurred while looking up the word: {str(e)}")
    finally:
        if conn:
            conn.close()

def display_dictionary_stats(args):
    console = Console()
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
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
        if args.detailed:
            console.print("[dim]Detailed statistics not yet implemented.[/]")
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

def display_leaderboard(cur, console):
    try:
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
    console = Console()
    console.print("[bold green]Welcome to the Interactive Dictionary Explorer![/]")
    console.print("Type 'exit' to quit.\n")
    while True:
        query = console.input("[bold yellow]Enter a word to look up:[/] ")
        if query.lower() in {"exit", "quit"}:
            console.print("Goodbye!")
            break
        class Args: pass
        args = Args()
        args.word = query
        args.debug = False
        args.format = 'rich'
        lookup_word(args)

def test_database():
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        print("Database connectivity test passed.")
    except Exception as e:
        print(f"Test failed: {e}")
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

def update_database(args):
    update_file = args.file
    if not os.path.exists(update_file):
        print(f"File not found: {update_file}")
        sys.exit(1)
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        src = os.path.basename(update_file)
        dry_run = getattr(args, 'dry_run', False)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=Console()) as progress:
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
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error updating database: {str(e)}")
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

def purge_database(args):
    if not args.force:
        console = Console()
        console.print("\n[bold red]WARNING:[/] This will delete ALL data from the database!")
        confirmation = input("\nType 'PURGE' to continue: ")
        if confirmation != "PURGE":
            logger.info("Purge cancelled")
            return
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            TRUNCATE TABLE 
                definition_relations, affixations, relations, etymologies, definitions, words, parts_of_speech
            RESTART IDENTITY CASCADE
        """)
        conn.commit()
        logger.info("Database purged successfully")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error during purge: {str(e)}")
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

def cleanup_relations(cur):
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
    logger.info("Starting definition deduplication process...")
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
    cur.execute("DELETE FROM definitions")
    cur.execute("""
        INSERT INTO definitions
        SELECT * FROM unique_definitions;
    """)
    cur.execute("DROP TABLE unique_definitions")
    cur.execute("SELECT COUNT(*) FROM definitions")
    final_count = cur.fetchone()[0]
    logger.info(f"Definition deduplication complete. {final_count} unique definitions remain.")

def cleanup_dictionary_data(cur):
    logger.info("Starting dictionary cleanup process...")
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
        for definition in processed['data'].get('definitions', []):
            pos = processed['data'].get('pos', "")
            if isinstance(pos, dict):
                pos = f"{pos['english']} ({pos['filipino']})"
            insert_definition(cur, word_id, definition.get('meaning', definition) if isinstance(definition, dict) else definition,
                              pos, examples=definition.get('examples'),
                              usage_notes=definition.get('usage_notes'),
                              category=definition.get('domain'),
                              sources=processed['source'])
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
        if 'metadata' in processed['data']:
            metadata = processed['data']['metadata']
            if 'etymology' in metadata:
                etymology_data = processor.process_etymology(metadata['etymology'], processed['source_enum'])
                insert_etymology(cur, word_id, etymology_data['etymology_text'],
                                 etymology_data.get('normalized_components'),
                                 etymology_data.get('language_codes'),
                                 processed['source'])
            if 'pronunciation' in metadata:
                pron_data = processor.standardize_pronunciation(metadata['pronunciation'])
                if pron_data:
                    cur.execute("""
                        UPDATE words 
                        SET romanized_form = %s,
                            pronunciation_data = %s
                        WHERE id = %s
                    """, (pron_data['romanized'], json.dumps(pron_data), word_id))
        if 'related_words' in processed['data']:
            for rel_type, related in processed['data']['related_words'].items():
                if isinstance(related, str):
                    related = [related]
                for rel_word in related:
                    if processor.validate_relation_type(rel_type):
                        rel_id = get_or_create_word_id(cur, rel_word, language_code="tl")
                        insert_relation(cur, word_id, rel_id, rel_type, processed['source'])
    except Exception as e:
        logger.error(f"Error storing processed entry: {str(e)}")
        raise

# -------------------------------------------------------------------
# CLI Wrapper Functions
# -------------------------------------------------------------------
def create_argument_parser_cli() -> argparse.ArgumentParser:
    return create_argument_parser()

def migrate_data_cli(args):
    migrate_data(args)

def verify_database_cli(args):
    verify_database(args)

def update_database_cli(args):
    update_database(args)

def purge_database_cli(args):
    purge_database(args)

def cleanup_database_cli(args):
    # Here, we simply call the handle_cleanup function
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cleanup_dictionary_data(cur)
        conn.commit()
        print("Dictionary cleanup completed successfully.")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error during cleanup: {str(e)}")
    finally:
        if conn:
            conn.close()

def lookup_word_cli(args):
    lookup_word(args)

def display_dictionary_stats_cli(args):
    display_dictionary_stats(args)

def display_leaderboard_cli(args):
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        display_leaderboard(cur, Console())
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def explore_dictionary_cli(args):
    explore_dictionary()

def test_database_cli(args):
    test_database()

def display_help_cli(args):
    display_help(args)

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
    elif args.command == "update":
        update_database_cli(args)
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
