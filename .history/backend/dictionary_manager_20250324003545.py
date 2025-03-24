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
from datetime import datetime, UTC
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
import time

# Load Environment Variables
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
DB_NAME = os.getenv("DB_NAME", "fil_dict")
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
    """
    Get a database connection with proper error handling.
    
    Returns:
        psycopg2.extensions.connection: A database connection object
        
    Raises:
        psycopg2.OperationalError: If database connection fails
        Exception: For other unexpected errors
    """
    retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(retries):
        try:
            logger.info("Attempting database connection...")
            logger.info(f"Database configuration:")
            logger.info(f"  Database Name: {DB_NAME}")
            logger.info(f"  User: {DB_USER}")
            logger.info(f"  Host: {DB_HOST}")
            logger.info(f"  Port: {DB_PORT}")
            logger.info(f"  Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'Not set'}")
            
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
                connect_timeout=3,
                application_name='dictionary_manager'
            )
            
            conn.autocommit = False
            
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
                cur.execute("SET idle_in_transaction_session_timeout = '60s'")
            
            logger.info("Successfully established database connection")
            return conn
            
        except psycopg2.OperationalError as e:
            if attempt < retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            
            logger.error("Database connection failed after all retries!")
            logger.error("Please check:")
            logger.error("1. PostgreSQL service is running")
            logger.error("2. Database credentials in .env file are correct")
            logger.error("3. Database exists and is accessible")
            logger.error(f"Error details: {str(e)}")
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {str(e)}")
            sys.exit(1)
            
    raise Exception("Failed to establish database connection after all retries")

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
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT relations_unique UNIQUE (from_word_id, to_word_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

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

def extract_etymology_components(etymology: str) -> List[str]:
    if not etymology:
        return []
    components = re.findall(r'<([^>]+)>', etymology)
    if not components:
        components = re.findall(r'\[([^\]]+)\]', etymology)
    return components

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
    def get_source_enum(source_str: str) -> str:
        return source_str
    
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
    
    if kwargs.get('has_baybayin'):
        bb_form = kwargs.get('baybayin_form')
        r_form = kwargs.get('romanized_form')
        should_skip_baybayin = any([
            len(lemma) == 1,
            lemma.isupper(),
            any(char in lemma.lower() for char in 'fjcñvxz'),
            any(lemma.lower().endswith(suffix) for suffix in ['ismo', 'ista', 'dad', 'cion', 'syon']),
            lemma[0].isupper() and not lemma.isupper(),
            any(word in lemma.lower() for word in ['http', 'www', '.com', '.org', '.net']),
            lemma.startswith(('-', '_', '.')),
            any(char.isdigit() for char in lemma),
        ])
        
        if should_skip_baybayin:
            kwargs['has_baybayin'] = False
            kwargs['baybayin_form'] = None
            kwargs['romanized_form'] = None
        elif bb_form and not validate_baybayin_entry(bb_form, r_form):
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
    if 'entry_data' in kwargs and 'processor' in globals():
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
    cur.execute("""
        INSERT INTO etymologies (word_id, etymology_text, components, language_codes, sources)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (word_id, etymology_text) DO UPDATE
        SET components = EXCLUDED.components,
            language_codes = EXCLUDED.language_codes,
            sources = EXCLUDED.sources,
            updated_at = CURRENT_TIMESTAMP
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
        batch = list(dict.fromkeys(batch))
        normalized_entries = [(lemma, normalize_lemma(lemma), lang_code) for lemma, lang_code in batch]
        
        placeholders = ",".join([f"(%s, %s)" for _ in normalized_entries])
        query = f"""
            SELECT lemma, language_code, id
            FROM words
            WHERE (normalized_lemma, language_code) IN ({placeholders})
        """
        flat_params = []
        for _, norm, lang in normalized_entries:
            flat_params.extend([norm, lang])
        
        cur.execute(query, flat_params)
        existing = {(lemma, lang): id for lemma, lang, id in cur.fetchall()}
        
        to_insert = [(lemma, norm, lang) for lemma, norm, lang in normalized_entries
                      if (lemma, lang) not in existing]
        
        if to_insert:
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
                    existing[(lemma, lang)] = word_id
                except Exception as e:
                    logger.error(f"Error processing entry {lemma}: {str(e)}")
                    continue
        
        result.update(existing)
    return result

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------
def process_kwf_entry(cur, word: str, entry: Dict[str, Any]) -> None:
    """
    Process a single entry from kwf_dictionary.json and store it in the database.
    
    Args:
        cur: Database cursor
        word: The normalized word (key in the JSON)
        entry: The dictionary entry data
    """
    if not word or not isinstance(entry, dict):
        logger.warning(f"Invalid KWF entry: word={word}, entry={entry}")
        return

    try:
        # Step 1: Validate and prepare word data
        original_word = entry.get('original', word)
        formatted_word = entry.get('formatted', word)
        language_code = 'tl'  # Assuming Tagalog for KWF dictionary

        # Prepare metadata
        metadata = entry.get('metadata', {})
        etymology_data = metadata.get('etymology', [])
        source_language_data = metadata.get('source_language', [])
        pronunciation_data = metadata.get('pronunciation', [])
        metadata_cross_references = metadata.get('cross_references', [])

        # Since KWF dictionary doesn't provide Baybayin forms, set to None
        has_baybayin = False
        baybayin_form = None
        romanized_form = None

        # Prepare pronunciation data as a dictionary
        pronunciation_dict = {
            "pronunciations": [
                {
                    "value": p.get('value', ''),
                    "context": p.get('context', []),
                    "html": p.get('html', '')
                } for p in pronunciation_data
            ]
        }

        # Step 2: Insert or get word ID
        word_data = {
            'lemma': formatted_word,
            'language_code': language_code,
            'has_baybayin': has_baybayin,
            'baybayin_form': baybayin_form,
            'romanized_form': romanized_form,
            'pronunciation_data': json.dumps(pronunciation_dict) if pronunciation_dict['pronunciations'] else None,
            'source': 'kwf_dictionary.json'
        }
        word_id = get_or_create_word_id(cur, formatted_word, language_code=language_code, **word_data)

        # Step 3: Process definitions
        definitions = entry.get('definitions', {})
        for pos, def_list in definitions.items():
            standardized_pos = standardize_pos(pos)
            for defn in def_list:
                if not isinstance(defn, dict):
                    logger.warning(f"Invalid definition for {word}: {defn}")
                    continue

                definition_text = defn.get('meaning', '')
                if definition_text is None and not defn.get('see'):
                    continue  # Skip definitions that are just "see" references without a meaning

                # Prepare examples
                example_sets = defn.get('example_sets', [])
                examples = []
                for ex_set in example_sets:
                    for ex in ex_set.get('examples', []):
                        examples.append({
                            'text': ex.get('text', ''),
                            'html': ex.get('html', ''),
                            'cross_references': ex.get('cross_references', [])
                        })
                examples_json = json.dumps(examples) if examples else None

                # Prepare usage notes (using 'note' and 'see_context')
                usage_notes = []
                if defn.get('note'):
                    usage_notes.append(defn['note'])
                if defn.get('see_context'):
                    usage_notes.append(f"See context: {defn['see_context']}")
                usage_notes_json = json.dumps(usage_notes) if usage_notes else None

                # Prepare categories (domain)
                categories = defn.get('categories', ['General'])
                domain = categories[0] if categories else 'General'

                # Insert definition
                insert_definition(
                    cur,
                    word_id,
                    definition_text,
                    part_of_speech=standardized_pos,
                    examples=examples_json,
                    usage_notes=usage_notes_json,
                    category=domain,
                    sources='kwf_dictionary.json'
                )

                # Process sub-definitions as separate definitions
                for sub_def in defn.get('sub_definitions', []):
                    sub_meaning = sub_def.get('meaning', '')
                    if not sub_meaning:
                        continue
                    sub_label = sub_def.get('label', '')
                    sub_usage_notes = [f"Sub-definition {sub_label}"] if sub_label else []
                    sub_usage_notes_json = json.dumps(sub_usage_notes) if sub_usage_notes else None
                    insert_definition(
                        cur,
                        word_id,
                        sub_meaning,
                        part_of_speech=standardized_pos,
                        examples=None,
                        usage_notes=sub_usage_notes_json,
                        category=domain,
                        sources='kwf_dictionary.json'
                    )

                # Process "see" references as relations
                for see_ref in defn.get('see', []):
                    see_term = see_ref.get('term', '')
                    if see_term:
                        see_word_id = get_or_create_word_id(
                            cur,
                            see_term,
                            language_code=language_code,
                            source='kwf_dictionary.json'
                        )
                        insert_relation(
                            cur,
                            word_id,
                            see_word_id,
                            'see',
                            sources='kwf_dictionary.json'
                        )

                # Process cross-references as relations
                for xref in defn.get('cross_references', []):
                    xref_term = xref.get('term', '')
                    if xref_term:
                        xref_word_id = get_or_create_word_id(
                            cur,
                            xref_term,
                            language_code=language_code,
                            source='kwf_dictionary.json'
                        )
                        insert_relation(
                            cur,
                            word_id,
                            xref_word_id,
                            'cross_reference',
                            sources='kwf_dictionary.json'
                        )

                # Process synonyms as relations
                for syn in defn.get('synonyms', []):
                    syn_term = syn.get('term', '')
                    if syn_term:
                        syn_word_id = get_or_create_word_id(
                            cur,
                            syn_term,
                            language_code=language_code,
                            source='kwf_dictionary.json'
                        )
                        insert_relation(
                            cur,
                            word_id,
                            syn_word_id,
                            'synonym',
                            sources='kwf_dictionary.json'
                        )

                # Process antonyms as relations
                for ant in defn.get('antonyms', []):
                    ant_term = ant.get('term', '')
                    if ant_term:
                        ant_word_id = get_or_create_word_id(
                            cur,
                            ant_term,
                            language_code=language_code,
                            source='kwf_dictionary.json'
                        )
                        insert_relation(
                            cur,
                            word_id,
                            ant_word_id,
                            'antonym',
                            sources='kwf_dictionary.json'
                        )

        # Step 4: Process etymology
        etymology_texts = [e.get('value', '') for e in etymology_data]
        etymology = '; '.join(etymology_texts) if etymology_texts else ''
        if etymology:
            components = extract_etymology_components(etymology)
            # Extract language codes from source_language
            language_codes = ','.join([sl.get('value', '') for sl in source_language_data])
            insert_etymology(
                cur,
                word_id,
                etymology,
                normalized_components=json.dumps(components) if components else None,
                language_codes=language_codes,
                sources='kwf_dictionary.json'
            )

        # Step 5: Process affixation as derived forms (relations)
        for affix in entry.get('affixation', []):
            forms = affix.get('form', [])
            for form in forms:
                if form:
                    form_word_id = get_or_create_word_id(
                        cur,
                        form,
                        language_code=language_code,
                        source='kwf_dictionary.json'
                    )
                    insert_relation(
                        cur,
                        word_id,
                        form_word_id,
                        'derived',
                        sources='kwf_dictionary.json'
                    )
            # Store affixation details as usage notes in the word
            affix_notes = {
                'forms': forms,
                'types': affix.get('types', []),
                'note': affix.get('note', ''),
                'examples': affix.get('examples', []),
                'cross_references': affix.get('cross_references', [])
            }
            cur.execute("""
                UPDATE words
                SET usage_notes = COALESCE(usage_notes, '[]') || %s::jsonb
                WHERE id = %s
            """, (json.dumps([json.dumps(affix_notes)]), word_id))

        # Step 6: Process idioms
        for idiom in entry.get('idioms', []):
            idiom_phrase = idiom.get('idiom', '')
            if not idiom_phrase:
                continue
            # Create a new word entry for the idiom
            idiom_word_id = get_or_create_word_id(
                cur,
                idiom_phrase,
                language_code=language_code,
                source='kwf_dictionary.json'
            )
            # Insert the idiom as a definition
            idiom_meaning = idiom.get('meaning', '')
            if idiom_meaning:
                idiom_examples = idiom.get('examples', [])
                examples_json = json.dumps([{'text': ex, 'html': idiom.get('examples_html', [''])[i], 'cross_references': idiom.get('example_cross_references', [[]])[i]} for i, ex in enumerate(idiom_examples)]) if idiom_examples else None
                insert_definition(
                    cur,
                    idiom_word_id,
                    idiom_meaning,
                    part_of_speech='Idyoma',
                    examples=examples_json,
                    usage_notes=None,
                    category='Idiomatic',
                    sources='kwf_dictionary.json'
                )
            # Link the idiom to the main word
            insert_relation(
                cur,
                word_id,
                idiom_word_id,
                'idiom',
                sources='kwf_dictionary.json'
            )
            # Process cross-references in idioms
            for xref in idiom.get('cross_references', []):
                xref_term = xref.get('term', '')
                if xref_term:
                    xref_word_id = get_or_create_word_id(
                        cur,
                        xref_term,
                        language_code=language_code,
                        source='kwf_dictionary.json'
                    )
                    insert_relation(
                        cur,
                        idiom_word_id,
                        xref_word_id,
                        'cross_reference',
                        sources='kwf_dictionary.json'
                    )

        # Step 7: Process related data
        related = entry.get('related', {})
        # Related terms
        for rel_term in related.get('related_terms', []):
            term = rel_term.get('term', '')
            if term:
                rel_word_id = get_or_create_word_id(
                    cur,
                    term,
                    language_code=language_code,
                    source='kwf_dictionary.json'
                )
                insert_relation(
                    cur,
                    word_id,
                    rel_word_id,
                    'related',
                    sources='kwf_dictionary.json'
                )
        # Antonyms (already processed in definitions, but ensure consistency)
        for ant in related.get('antonyms', []):
            ant_term = ant.get('term', '')
            if ant_term:
                ant_word_id = get_or_create_word_id(
                    cur,
                    ant_term,
                    language_code=language_code,
                    source='kwf_dictionary.json'
                )
                insert_relation(
                    cur,
                    word_id,
                    ant_word_id,
                    'antonym',
                    sources='kwf_dictionary.json'
                )

        # Step 8: Process other sections as additional notes
        other_sections = entry.get('other_sections', {})
        if other_sections:
            other_notes = []
            for section, content in other_sections.items():
                section_data = {
                    'section': section,
                    'content': content.get('content', []),
                    'subsections': content.get('subsections', [])
                }
                other_notes.append(section_data)
            cur.execute("""
                UPDATE words
                SET usage_notes = COALESCE(usage_notes, '[]') || %s::jsonb
                WHERE id = %s
            """, (json.dumps([json.dumps(note) for note in other_notes]), word_id))

        # Step 9: Process metadata cross-references
        for xref in metadata_cross_references:
            xref_term = xref.get('term', '')
            if xref_term:
                xref_word_id = get_or_create_word_id(
                    cur,
                    xref_term,
                    language_code=language_code,
                    source='kwf_dictionary.json'
                )
                insert_relation(
                    cur,
                    word_id,
                    xref_word_id,
                    'cross_reference',
                    sources='kwf_dictionary.json'
                )

        # Step 10: Update source info in words table
        cur.execute("""
            UPDATE words
            SET source_info = COALESCE(source_info, '{}') || jsonb_build_object('kwf_dictionary', %s),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (json.dumps({'processed_at': datetime.now(UTC).isoformat()}), word_id))

        logger.info(f"Successfully processed KWF entry for word: {word} (ID: {word_id})")

    except Exception as e:
        logger.error(f"Error processing KWF entry for word {word}: {str(e)}")
        raise

def process_tagalog_words(cur, filename: str):
    """
    Process tagalog-words.json with the structure:
    {
        "word1": {
            "pronunciation": "...",
            "part_of_speech": "...",
            "etymology": "...",
            "derivative": "...",
            "definitions": [
                "definition1",
                "definition2",
                ...
            ]
        },
        "word2": {
            ...
        }
    }
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            # Format 1: Dictionary with word keys
            for word, entry in data.items():
                if not word or not isinstance(entry, dict):
                    logger.warning(f"Skipping invalid entry: {word}")
                    continue
                
                word_id = get_or_create_word_id(cur, word, language_code='tl')
                
                # Process definitions
                definitions = entry.get('definitions', [])
                if definitions and isinstance(definitions, list):
                    for definition in definitions:
                        if definition:
                            # Trim any trailing ellipsis from definition
                            definition = re.sub(r'\.{3,}', '', definition).strip()
                            insert_definition(
                                cur,
                                word_id,
                                definition,
                                part_of_speech=entry.get('part_of_speech', ''),
                                sources=SourceStandardization.standardize_sources('tagalog-words.json')
                            )
                
                # Process etymology if available
                etymology = entry.get('etymology', '')
                if etymology:
                    etymological_components = extract_etymology_components(etymology)
                    normalized_components = json.dumps(etymological_components) if etymological_components else None
                    insert_etymology(
                        cur, 
                        word_id, 
                        etymology, 
                        normalized_components=normalized_components,
                        sources=SourceStandardization.standardize_sources('tagalog-words.json')
                    )
        else:
            # Format 2: List of word entries
            for entry in data:
                if not isinstance(entry, dict) or 'word' not in entry:
                    logger.warning(f"Skipping invalid entry without word: {entry}")
                    continue
                
                word = entry['word']
                definition = entry.get('definition', '')
                word_id = get_or_create_word_id(cur, word, language_code='tl')
                
                if definition:
                    # Trim any trailing ellipsis from definition
                    definition = re.sub(r'\.{3,}', '', definition).strip()
                    insert_definition(
                        cur,
                        word_id,
                        definition,
                        sources=SourceStandardization.standardize_sources('tagalog-words.json')
                    )

def process_root_words(cur, filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            word = entry['word']
            definitions = entry.get('definitions', [])
            word_id = get_or_create_word_id(cur, word, language_code='tl')
            for definition in definitions:
                insert_definition(
                    cur,
                    word_id,
                    definition,
                    sources=SourceStandardization.standardize_sources('root_words_with_associated_words_cleaned.json')
                )

def process_root_words_cleaned(cur, filename: str):
    """
    Process root words from a JSON file with the following structure:
    {
      "root_word1": {
          "variant1": {"type": "noun", "definition": "definition text..."},
          "variant2": {"type": "verb", "definition": "another definition..."},
          ...
      },
      "root_word2": {
          "variantA": {"type": "adjective", "definition": "definition text..."},
          ...
      },
      ...
    }
    
    For each base (root) word, a word entry is created. For each variant,
    if the variant differs from the base, a separate word entry is created and 
    its definition is inserted (with any trailing ellipsis trimmed). Additionally, 
    a relation is inserted indicating the variant is derived from the base.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for base_word, variants in data.items():
            if not isinstance(variants, dict):
                logger.warning(f"Skipping invalid entry for root word {base_word}: variants is not a dictionary")
                continue
                
            base_word_id = get_or_create_word_id(cur, base_word, language_code='tl')
            
            for variant, details in variants.items():
                if not isinstance(details, dict):
                    logger.warning(f"Skipping invalid variant {variant} for root word {base_word}: details is not a dictionary")
                    continue
                    
                # Trim ellipsis from definition (remove three or more dots)
                definition_text = details.get("definition", "")
                if not definition_text:
                    logger.warning(f"Skipping empty definition for variant {variant} of root word {base_word}")
                    continue
                    
                definition_text = re.sub(r'\.{3,}', '', definition_text).strip()
                pos = details.get("type", "")
                
                if variant == base_word:
                    # This is the root word definition
                    insert_definition(
                        cur,
                        base_word_id,
                        definition_text,
                        part_of_speech=standardize_pos(pos),
                        sources=SourceStandardization.standardize_sources('root_words_with_associated_words_cleaned.json')
                    )
                else:
                    # This is a variant/derived word
                    try:
                        variant_id = get_or_create_word_id(cur, variant, language_code='tl')
                        variant_definition = definition_text
                        insert_definition(
                            cur,
                            variant_id,
                            variant_definition,
                            part_of_speech=standardize_pos(pos),
                            sources=SourceStandardization.standardize_sources('root_words_with_associated_words_cleaned.json')
                        )
                        # Record the derivative relationship: variant derived from base.
                        insert_relation(
                            cur, 
                            variant_id, 
                            base_word_id, 
                            "derived", 
                            sources=SourceStandardization.standardize_sources('root_words_with_associated_words_cleaned.json')
                        )
                    except Exception as e:
                        logger.error(f"Error processing variant {variant} of root word {base_word}: {str(e)}")

def process_kwf_dictionary(cur, filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            word = entry['word']
            definitions = entry.get('definitions', [])
            word_id = get_or_create_word_id(cur, word, language_code='tl')
            for definition in definitions:
                insert_definition(
                    cur,
                    word_id,
                    definition,
                    sources=SourceStandardization.standardize_sources('kwf_dictionary.json')
                )

def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries (both Tagalog and Cebuano)."""
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        forms = entry.get('forms')
        if not forms:
            return None, None
        for form in forms:
            tags = form.get('tags', [])
            if 'Baybayin' in tags:
                return form.get('form'), form.get('romanized_form')
        return None, None

    def standardize_entry_pos(pos_str: str) -> str:
        return standardize_pos(pos_str)

    def process_entry(cur, entry: Dict):
        try:
            word = entry['word']
            language_code = entry.get('language_code') or entry.get('lang_code', 'tl')
            word_id = get_or_create_word_id(cur, word, language_code=language_code)
            baybayin_form, romanized_form = extract_baybayin_info(entry)
            if baybayin_form:
                process_baybayin_data(cur, word_id, baybayin_form, romanized_form)
            definitions = entry.get('definitions') or entry.get('senses', [])
            for definition in definitions:
                if isinstance(definition, dict):
                    text = definition.get('text')
                    if not text and 'glosses' in definition and isinstance(definition['glosses'], list) and definition['glosses']:
                        text = definition['glosses'][0]
                    pos = definition.get('pos', entry.get('pos', ''))
                    examples = definition.get('examples')
                    if examples and not isinstance(examples, list):
                        examples = [examples]
                    usage_notes = definition.get('usage_notes')
                else:
                    text = definition
                    pos = entry.get('pos', '')
                    examples = None
                    usage_notes = None
                if text:
                    # Trim any trailing ellipsis from definition text
                    text = re.sub(r'\.{3,}', '', text).strip()
                    insert_definition(
                        cur,
                        word_id,
                        text,
                        part_of_speech=standardize_entry_pos(pos),
                        examples=json.dumps(examples) if examples else None,
                        usage_notes=json.dumps(usage_notes) if usage_notes else None,
                        sources=SourceStandardization.standardize_sources('kaikki.jsonl')
                    )
            if 'etymology' in entry and entry['etymology'].strip():
                etymology_text = entry['etymology']
                normalized_components = extract_etymology_components(etymology_text)
                insert_etymology(
                    cur,
                    word_id,
                    etymology_text,
                    normalized_components=json.dumps(normalized_components) if normalized_components else None,
                    sources=SourceStandardization.standardize_sources('kaikki.jsonl')
                )
            if 'relations' in entry and isinstance(entry['relations'], dict):
                for rel_type, related_words in entry['relations'].items():
                    for rel_word in related_words:
                        rel_word_id = get_or_create_word_id(cur, rel_word, language_code=language_code)
                        insert_relation(
                            cur,
                            word_id,
                            rel_word_id,
                            rel_type,
                            sources=SourceStandardization.standardize_sources('kaikki.jsonl')
                        )
        except Exception as e:
            logger.error(f"Error processing entry {entry.get('word', 'unknown')}: {str(e)}")

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                process_entry(cur, entry)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing line: {str(e)}")

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
            "required": False
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
    data_dirs = []
    if hasattr(args, 'data_dir') and args.data_dir:
        data_dirs.append(args.data_dir)
    else:
        data_dirs = ["data", os.path.join("..", "data")]
    
    # Filter sources if specific ones are requested
    if hasattr(args, 'sources') and args.sources:
        requested_sources = [s.lower() for s in args.sources.split(',')]
        sources = [s for s in sources if s["name"].lower() in requested_sources or s["file"].lower() in requested_sources]
        # Mark requested sources as required
        for source in sources:
            source["required"] = True
    
    # Custom file overrides existing sources
    if hasattr(args, 'file') and args.file:
        # Determine handler based on file extension
        filename = args.file
        handler = None
        if filename.endswith('.jsonl'):
            handler = process_kaikki_jsonl
        elif 'root_words' in filename.lower():
            handler = process_root_words_cleaned
        elif 'kwf' in filename.lower():
            handler = process_kwf_dictionary
        else:
            handler = process_tagalog_words
            
        # Add or replace source
        basename = os.path.basename(filename)
        source_found = False
        for source in sources:
            if source["file"] == basename:
                source["file"] = filename  # Use the full path
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
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        
        console.print("[bold]Setting up database schema...[/]")
        create_or_update_tables(conn)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=Console()) as progress:
            for source in sources:
                # Check if the file is an absolute path
                if os.path.isabs(source["file"]):
                    filename = source["file"]
                else:
                    # Try to find the file in the data directories
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
                handler_func = source["handler"]
                try:
                    handler_func(cur, filename)
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
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

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
            ORDER BY count DESC
            LIMIT 15
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
        
        normalized_word = normalize_lemma(args.word)
        
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
        
        all_sources = set()
        if tags:
            all_sources.update(tag.strip() for tag in tags.split(','))
        if source_info:
            try:
                if isinstance(source_info, str):
                    source_info = json.loads(source_info)
                all_sources.update(s.strip() for s in source_info.get('sources', []))
            except (json.JSONDecodeError, TypeError):
                pass
        
        if definition_sources:
            for sources in definition_sources:
                if sources:
                    all_sources.update(s.strip() for s in sources.split(','))
        
        if etymology_sources:
            for sources in etymology_sources:
                if sources:
                    all_sources.update(s.strip() for s in sources.split(','))
        
        source_categories = {
            'Dictionaries': set(),
            'Online Resources': set(),
            'Academic Sources': set()
        }
        
        for source in all_sources:
            if not source:
                continue
            display_name = SourceStandardization.get_display_name(source)
            if not display_name:
                continue
            if any(kw in display_name.lower() for kw in ['dictionary', 'diksiyonaryo']):
                source_categories['Dictionaries'].add(display_name)
            elif any(kw in display_name.lower() for kw in ['.org', '.com', '.net', 'online']):
                source_categories['Online Resources'].add(display_name)
            else:
                source_categories['Academic Sources'].add(display_name)
        
        console = Console()
        console.print("\n[bold]Word Information[/]", justify="center")
        
        word_info = []
        word_info.append(f"Word: {format_word_display(lemma, has_baybayin)}")
        word_info.append(f"Language: {'Tagalog' if language_code == 'tl' else 'Cebuano'}")
        
        if preferred_spelling:
            word_info.append(f"Preferred Spelling: {preferred_spelling}")
        
        if has_baybayin and baybayin_form:
            word_info.append(f"Baybayin Form: {baybayin_form}")
            if romanized_form:
                word_info.append(f"Romanized Form: {romanized_form}")
        
        if pronunciation_data:
            try:
                if isinstance(pronunciation_data, str):
                    pronunciation_data = json.loads(pronunciation_data)
                if isinstance(pronunciation_data, dict):
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
            except (json.JSONDecodeError, TypeError):
                pass
        
        if root_word_id:
            cur.execute("SELECT lemma FROM words WHERE id = %s", (root_word_id,))
            root_result = cur.fetchone()
            if root_result:
                word_info.append(f"Root Word: {root_result[0]}")
        
        for category, sources in source_categories.items():
            if sources:
                word_info.append(f"{category}: {', '.join(sorted(sources))}")
        
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
                example_text = ""
                if examples:
                    try:
                        example_list = json.loads(examples)
                        if isinstance(example_list, list):
                            example_text = "\n".join(f"• {ex.strip()}" for ex in example_list if ex.strip())
                        else:
                            example_text = f"• {str(example_list).strip()}"
                    except json.JSONDecodeError:
                        example_list = [f"• {ex.strip()}" for ex in examples.split('\n') if ex.strip()]
                        example_text = "\n".join(example_list)
                usage_text = ""
                if usage_notes:
                    try:
                        usage_list = json.loads(usage_notes)
                        if isinstance(usage_list, list):
                            usage_text = "\n".join(f"• {note.strip()}" for note in usage_list if note.strip())
                        else:
                            usage_text = f"• {str(usage_list).strip()}"
                    except json.JSONDecodeError:
                        usage_list = [f"• {note.strip()}" for note in usage_notes.split('\n') if note.strip()]
                        usage_text = "\n".join(usage_list)
                source_list = []
                if sources:
                    for source in sources.split(','):
                        display_name = SourceStandardization.get_display_name(source.strip())
                        if display_name and display_name not in source_list:
                            source_list.append(display_name)
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
                    try:
                        comp_list = json.loads(components)
                        if isinstance(comp_list, list):
                            comp_list = sorted(set(comp.strip() for comp in comp_list if comp.strip()))
                            if comp_list:
                                etymology_panel.append(Text("\nComponents:", style="bold"))
                                etymology_panel.extend(Text(f"• {comp}", style="green") for comp in comp_list)
                    except json.JSONDecodeError:
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
                console.print(Panel(Group(*etymology_panel), box=box.ROUNDED, expand=True, border_style="blue"))
        
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
            seen_words = set()
            for rel_type, rel_word, pos, rel_id in relations:
                word_pos = f"{rel_word} ({pos})" if pos else rel_word
                word_key = (rel_word, pos)
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

def update_database(args):
    """Update dictionary with new data."""
    if not args.file or not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        return
    
    filename = args.file
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in ['.json', '.jsonl']:
        print(f"Error: Unsupported file format '{file_ext}'. Please use JSON or JSONL files.")
        return
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        if file_ext == '.json':
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                print(f"Error: {filename} must contain a dictionary of entries.")
                return

            if args.dry_run:
                print(f"Dry run: Found {len(data)} entries to process")
                return
            
            if 'kwf' in filename.lower():
                logger.info(f"Processing KWF dictionary data from {filename}")
                with tqdm(total=len(data), desc="Processing KWF entries") as pbar:
                    for word, entry in data.items():
                        if not word or not entry:
                            logger.warning(f"Skipping invalid entry: word={word}, entry={entry}")
                            pbar.update(1)
                            continue
                        process_kwf_entry(cur, word, entry)
                        pbar.update(1)
            else:
                logger.info(f"Processing generic JSON data from {filename}")
                for word, entry in tqdm(data.items(), desc="Processing entries"):
                    word_id = get_or_create_word_id(cur, word, language_code='tl')
                    for pos, definitions in entry.get('definitions', {}).items():
                        for definition in definitions:
                            if isinstance(definition, dict) and 'meaning' in definition:
                                insert_definition(
                                    cur,
                                    word_id,
                                    definition['meaning'],
                                    part_of_speech=definition.get('pos', ''),
                                    sources=os.path.basename(filename)
                                )
                            else:
                                insert_definition(
                                    cur,
                                    word_id,
                                    definition,
                                    sources=os.path.basename(filename)
                                )
        else:
            if 'kaikki' in filename.lower():
                process_kaikki_jsonl(cur, filename)
            else:
                if args.dry_run:
                    count = sum(1 for _ in open(filename, 'r', encoding='utf-8'))
                    print(f"Dry run: Found {count} entries to process")
                    return
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Processing entries"):
                        try:
                            entry = json.loads(line)
                            word = entry.get('word', '')
                            if not word:
                                continue
                            language_code = entry.get('language_code', 'tl')
                            word_id = get_or_create_word_id(cur, word, language_code=language_code)
                            for definition in entry.get('definitions', []):
                                if isinstance(definition, dict) and 'text' in definition:
                                    insert_definition(
                                        cur,
                                        word_id,
                                        definition['text'],
                                        part_of_speech=definition.get('pos', ''),
                                        sources=os.path.basename(filename)
                                    )
                                else:
                                    insert_definition(
                                        cur,
                                        word_id,
                                        definition,
                                        sources=os.path.basename(filename)
                                    )
                        except json.JSONDecodeError:
                            continue
        conn.commit()
        print(f"Successfully updated database with data from {filename}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error during update: {str(e)}")
    finally:
        if conn:
            conn.close()

def display_dictionary_stats(args):
    """Display dictionary statistics."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        console = Console()
        
        stats_table = Table(title="Dictionary Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right", style="green")
        
        stats = [
            ("Total Words", "SELECT COUNT(*) FROM words"),
            ("Tagalog Words", "SELECT COUNT(*) FROM words WHERE language_code = 'tl'"),
            ("Cebuano Words", "SELECT COUNT(*) FROM words WHERE language_code = 'ceb'"),
            ("Baybayin Words", "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE"),
            ("Definitions", "SELECT COUNT(*) FROM definitions"),
            ("Relations", "SELECT COUNT(*) FROM relations"),
            ("Etymologies", "SELECT COUNT(*) FROM etymologies"),
        ]
        
        for label, query in stats:
            cur.execute(query)
            count = cur.fetchone()[0]
            stats_table.add_row(label, f"{count:,}")
        
        console.print(stats_table)
        
        if args.detailed:
            console.print("\n[bold]Data Sources[/]")
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
            source_results = cur.fetchall()
            
            if source_results:
                source_table = Table(box=box.ROUNDED)
                source_table.add_column("Source", style="yellow")
                source_table.add_column("Definitions", justify="right", style="green")
                
                for source, count in source_results:
                    source_table.add_row(source, f"{count:,}")
                
                console.print(source_table)
            
            console.print("\n[bold]Parts of Speech Distribution[/]")
            pos_query = """
                SELECT p.name_tl, COUNT(*)
                FROM definitions d
                JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                GROUP BY p.name_tl
                ORDER BY COUNT(*) DESC
            """
            cur.execute(pos_query)
            pos_results = cur.fetchall()
            
            if pos_results:
                pos_table = Table(box=box.ROUNDED)
                pos_table.add_column("Part of Speech", style="magenta")
                pos_table.add_column("Count", justify="right", style="green")
                
                for pos, count in pos_results:
                    pos_table.add_row(pos or "Uncategorized", f"{count:,}")
                
                console.print(pos_table)
            
            console.print("\n[bold]Relation Types[/]")
            rel_query = """
                SELECT relation_type, COUNT(*)
                FROM relations
                GROUP BY relation_type
                ORDER BY COUNT(*) DESC
            """
            cur.execute(rel_query)
            rel_results = cur.fetchall()
            
            if rel_results:
                rel_table = Table(box=box.ROUNDED)
                rel_table.add_column("Relation Type", style="blue")
                rel_table.add_column("Count", justify="right", style="green")
                
                for rel_type, count in rel_results:
                    rel_table.add_row(rel_type, f"{count:,}")
                
                console.print(rel_table)
        
        if args.export:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "basic_stats": {},
                "source_stats": {},
                "pos_stats": {},
                "relation_stats": {}
            }
            
            for label, query in stats:
                cur.execute(query)
                count = cur.fetchone()[0]
                export_data["basic_stats"][label] = count
            
            if args.detailed:
                cur.execute(source_stats_query)
                for source, count in cur.fetchall():
                    export_data["source_stats"][source] = count
            
            cur.execute(pos_query)
            for pos, count in cur.fetchall():
                export_data["pos_stats"][pos or "Uncategorized"] = count
            
            cur.execute(rel_query)
            for rel_type, count in cur.fetchall():
                export_data["relation_stats"][rel_type] = count
            
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"\n[green]Statistics exported to {args.export}[/]")
        
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        if conn:
            conn.close()

def display_leaderboard(cur, console):
    """Display top contributors based on definitions and entries."""
    try:
        console.print("\n[bold cyan]📊 Dictionary Contributors Leaderboard[/]", justify="center")
        
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
                 COUNT(*) as definition_count,
                 COUNT(DISTINCT word_id) as word_count
            FROM definitions
            GROUP BY source_name
            ORDER BY definition_count DESC
            LIMIT 10
        """)
        
        results = cur.fetchall()
        
        if results:
            table = Table(title="Top Contributors", box=box.ROUNDED, border_style="cyan")
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Source", style="yellow")
            table.add_column("Definitions", justify="right", style="green")
            table.add_column("Unique Words", justify="right", style="blue")
            
            for i, (source, def_count, word_count) in enumerate(results, 1):
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
                    f"{word_count:,}"
                )
            
            console.print(table)
            
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
                     MAX(created_at) as last_activity,
                     COUNT(*) as entries_past_month
                FROM definitions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY source_name
                ORDER BY last_activity DESC
                LIMIT 5
            """)
            
            recent_results = cur.fetchall()
            
            if recent_results:
                recent_table = Table(title="Recently Active Sources", box=box.ROUNDED, border_style="magenta")
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
        conn.commit()
        print("Database purged successfully.")
    except Exception as e:
        logger.error(f"Error during database purge: {str(e)}")
        print(f"An error occurred: {str(e)}")
    finally:
        if conn:
            conn.close()

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

def lookup_by_id(cur, word_id: int, console: Console):
    """Look up word by ID and display details."""
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

def check_tables_exist():
    """Check if required tables exist."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        tables = ["words", "definitions", "parts_of_speech", "relations", "etymologies"]
        missing_tables = []
        
        for table in tables:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                     WHERE table_schema = 'public' 
                     AND table_name = %s
                )
            """, (table,))
            
            if not cur.fetchone()[0]:
                missing_tables.append(table)
        
        if missing_tables:
            return False, f"Missing tables: {', '.join(missing_tables)}"
        else:
            return True, f"All {len(tables)} required tables exist"
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

def check_data_access():
    """Check if data can be accessed."""
    conn = get_connection()
    try:
        cur = conn.cursor()
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
        conn.commit()
        return True, f"Successfully read, wrote, and deleted data. Word count: {word_count:,}"
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()

def check_query_performance():
    """Check query performance."""
    conn = get_connection()
    try:
        cur = conn.cursor()
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
    finally:
        conn.close()

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
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cleanup_dictionary_data(cur)
            deduplicate_definitions(cur)
            cleanup_relations(cur)
            cleanup_baybayin_data(cur)
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

def cleanup_database_cli(args):
    """Run cleanup routines on the dictionary database."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
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
            
            conn.commit()
            console.print("[bold green]Dictionary cleanup completed successfully.[/]")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error during cleanup: {str(e)}")
    finally:
        if conn:
            conn.close()

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
