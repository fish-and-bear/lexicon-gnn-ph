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
    """Get a connection to the database."""
    return psycopg2.connect(
        dbname=os.environ.get('DB_NAME', DB_NAME),
        user=os.environ.get('DB_USER', DB_USER),
        password=os.environ.get('DB_PASSWORD', DB_PASSWORD),
        host=os.environ.get('DB_HOST', DB_HOST),
        port=os.environ.get('DB_PORT', DB_PORT)
    )

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
    A decorator for database functions that need transaction handling.
    
    Args:
        commit (bool): Whether to commit the transaction after function execution
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection
            savepoint_name = f"sp_{func.__name__}_{time.time_ns()}"
            
            # Store original autocommit state
            original_autocommit = conn.autocommit
            conn.autocommit = False
            
            # Check if we're already in a transaction
            in_transaction = False
            try:
                # Check transaction status
                cur.execute("SELECT pg_typeof(txid_current())")
                in_transaction = True
            except Exception:
                # If we can't execute the query, assume no transaction
                pass
            
            try:
                if in_transaction:
                    # We're in a transaction, create a savepoint
                    try:
                        cur.execute(f"SAVEPOINT {savepoint_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create savepoint in {func.__name__}: {str(e)}")
                        # Continue without savepoint
                
                # Execute the function
                result = func(cur, *args, **kwargs)
                
                # Commit or release savepoint if needed
                if commit:
                    if in_transaction:
                        try:
                            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        except Exception as e:
                            logger.warning(f"Failed to release savepoint in {func.__name__}: {str(e)}")
                            # Not fatal, continue
                    else:
                        conn.commit()
                
                return result
            except Exception as e:
                # Handle transaction error
                try:
                    if in_transaction:
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                            logger.info(f"Rolled back to savepoint in {func.__name__}")
                        except Exception as rollback_error:
                            logger.error(f"Failed to rollback to savepoint in {func.__name__}: {str(rollback_error)}")
                            # If we can't rollback to savepoint, try a full rollback
                            try:
                                conn.rollback()
                                logger.info(f"Performed full rollback in {func.__name__}")
                            except Exception as full_rollback_error:
                                logger.error(f"Failed to perform full rollback in {func.__name__}: {str(full_rollback_error)}")
                    else:
                        conn.rollback()
                        logger.info(f"Rolled back transaction in {func.__name__}")
                except Exception as rollback_error:
                    logger.error(f"Unhandled error during rollback in {func.__name__}: {str(rollback_error)}")
                
                # Re-raise the original error
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
            finally:
                # Restore original autocommit state
                conn.autocommit = original_autocommit
        
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
    """
    Validates if a Baybayin entry is properly formatted.
    
    Args:
        baybayin_form: The Baybayin text to validate
        romanized_form: Optional romanized form to check consistency
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Quick validation first
    if not baybayin_form or len(baybayin_form) == 0:
        return False
        
    # Validate the full Baybayin character range (Baybayin Unicode block: U+1700 to U+171F)
    BAYBAYIN_PATTERN = re.compile(r'^[\u1700-\u171F\u1735\u1736\s]+$')
    if not BAYBAYIN_PATTERN.match(baybayin_form):
        return False
    
    # Additional validation rules
    romanizer = BaybayinRomanizer()
    
    # Ensure all characters in the string are valid Baybayin
    if not all(romanizer.get_char_info(char) is not None for char in baybayin_form if not char.isspace()):
        return False
    
    # Skip entries with single character, as they're often problematic
    if len(baybayin_form.strip()) == 1:
        return False
        
    # Validate that romanized form (if provided) is plausible
    if romanized_form:
        # Compare lengths as a sanity check (Baybayin should generally be shorter)
        if len(baybayin_form.strip()) > len(romanized_form.strip()) * 1.5:
            return False
            
        # For an additional check, try to romanize the Baybayin and see if it's similar
        # to the provided romanized form
        try:
            derived_romanized = romanizer.romanize(baybayin_form)
            if derived_romanized:
                # Simplified comparison - both versions should share some characters
                common_chars = set(derived_romanized.lower()) & set(romanized_form.lower())
                if len(common_chars) < min(2, len(romanized_form) // 3):
                    return False
        except Exception:
            # If romanization fails, that's also a sign the Baybayin might be invalid
            return False
    
    # If we made it here, the entry is valid
    return True

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
    """
    Get or create a word entry and return its ID.
    
    Args:
        cur: Database cursor
        lemma: The word to look up or create
        language_code: ISO language code (default: "tl" for Tagalog)
        **kwargs: Additional word attributes
    
    Returns:
        int: Word ID
        
    Raises:
        ValueError: If word creation fails
    """
    if not lemma or not isinstance(lemma, str):
        logger.error(f"Invalid lemma: {lemma}")
        return None
    
    conn = cur.connection
    try:
        normalized = normalize_lemma(lemma)
        
        # First try to find the word
        try:
            cur.execute("""
                SELECT id FROM words 
                WHERE normalized_lemma = %s AND language_code = %s
            """, (normalized, language_code))
            result = cur.fetchone()
            if result:
                return result[0]
        except Exception as e:
            logger.error(f"Error searching for word {lemma}: {str(e)}")
            # If we can't search, we need to abort this operation
            return None
        
        # Word doesn't exist, try to create it
        try:
            # Prepare data for insertion
            search_text = lemma + " " + normalized
            
            # Extract relevant fields from kwargs
            root_word_id = kwargs.get('root_word_id')
            
            # Handle Baybayin-related fields
            has_baybayin = kwargs.get('has_baybayin', False)
            baybayin_form = kwargs.get('baybayin_form')
            romanized_form = kwargs.get('romanized_form', lemma)
            
            # Only set Baybayin fields if they pass validation
            if has_baybayin and baybayin_form:
                # Validate Baybayin before inserting
                if not validate_baybayin_entry(baybayin_form, romanized_form):
                    has_baybayin = False
                    baybayin_form = None
                    romanized_form = lemma
            
            # Create the word
            query = """
                INSERT INTO words (
                    lemma, normalized_lemma, language_code, 
                    root_word_id, has_baybayin, baybayin_form, romanized_form,
                    search_text
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s)
                ) RETURNING id
            """
            
            cur.execute(query, (
                lemma, normalized, language_code, root_word_id, 
                has_baybayin, baybayin_form, romanized_form, search_text
            ))
            
            word_id = cur.fetchone()[0]
            return word_id
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in get_or_create_word_id for lemma '{lemma}': {error_msg}")
            
            # Don't raise the error - we'll handle it in the decorator
            # which will perform the rollback
            return None
            
    except Exception as e:
        logger.error(f"Error in get_or_create_word_id: {str(e)}")
        return None

@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str, part_of_speech: str = "",
                      examples: str = None, usage_notes: str = None, category: str = None,
                      sources: str = "") -> Optional[int]:
    """
    Insert a definition for a word into the database.
    
    Args:
        cur: Database cursor
        word_id: ID of the word to add definition for
        definition_text: The definition text
        part_of_speech: Part of speech for this definition
        examples: JSON string of example sentences
        usage_notes: Additional usage notes
        category: Category/domain of the definition
        sources: Source of the definition
        
    Returns:
        ID of the inserted definition or None if skipped
    """
    try:
        # Skip Baybayin spelling definitions
        if 'Baybayin spelling of' in definition_text:
            return None
            
        # Verify the word exists in the database
        cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
        if not cur.fetchone():
            logger.error(f"Cannot insert definition - word ID {word_id} does not exist in database.")
            return None
            
        # Check for duplicate definitions
        cur.execute("""
            SELECT id FROM definitions WHERE word_id = %s AND definition_text = %s AND original_pos = %s
        """, (word_id, definition_text, part_of_speech))
        if cur.fetchone():
            return None
            
        # Get standardized part of speech ID
        std_pos_id = get_standardized_pos_id(cur, part_of_speech)
        
        # Format usage notes with category
        if category:
            usage_notes = f"[{category}] {usage_notes if usage_notes else ''}"
            
        # Insert the definition
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
        
    except Exception as e:
        logger.error(f"Error in insert_definition for word_id {word_id}, definition '{definition_text[:50]}...': {str(e)}")
        return None

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
def insert_etymology(
    cur,
    word_id: int,
    etymology_text: str,
    normalized_components: str,
    language_codes: str,
    sources: str
) -> None:
    """Insert etymology data into the etymologies table."""
    if not word_id or not etymology_text:
        return

    try:
        cur.execute(
            """
            INSERT INTO etymologies (
                word_id, etymology_text, normalized_components, language_codes, sources
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (word_id, etymology_text)
            DO UPDATE SET
                normalized_components = COALESCE(etymologies.normalized_components, EXCLUDED.normalized_components),
                language_codes = COALESCE(etymologies.language_codes, EXCLUDED.language_codes),
                sources = array_to_string(ARRAY(
                    SELECT DISTINCT unnest(array_cat(
                        string_to_array(etymologies.sources, ', '),
                        string_to_array(EXCLUDED.sources, ', ')
                    ))
                ), ', ')
            """,
            (word_id, etymology_text, normalized_components, language_codes, sources)
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

def batch_get_or_create_word_ids(cur, entries: List[Tuple[str, str]], batch_size: int = 1000) -> Dict[Tuple[str, str], int]:
    """
    Efficiently get or create multiple word IDs in batches.
    
    Args:
        cur: Database cursor
        entries: List of (lemma, language_code) tuples
        batch_size: Number of entries to process per batch
        
    Returns:
        Dictionary mapping (lemma, language_code) to word_id
    """
    if not entries:
        return {}
    
    # Remove duplicates
    unique_entries = list(set(entries))
    result_dict = {}
    conn = cur.connection
    
    # Process in batches to avoid large transactions
    for i in range(0, len(unique_entries), batch_size):
        batch = unique_entries[i:i+batch_size]
        batch_dict = {}
        
        try:
            # Store original autocommit state
            original_autocommit = conn.autocommit
            conn.autocommit = False
            
            # First check which words already exist
            placeholders = []
            query_params = []
            
            for lemma, lang_code in batch:
                normalized = normalize_lemma(lemma)
                placeholders.append(f"(%s, %s)")
                query_params.extend([normalized, lang_code])
            
            query = f"""
                SELECT id, normalized_lemma, language_code
                FROM words
                WHERE (normalized_lemma, language_code) IN 
                ({", ".join(placeholders)})
            """
            
            cur.execute(query, query_params)
            existing_words = cur.fetchall()
            
            # Map existing words
            existing_dict = {}
            for word_id, normalized, lang_code in existing_words:
                # Find matching entries
                for lemma, code in batch:
                    if normalize_lemma(lemma) == normalized and code == lang_code:
                        existing_dict[(lemma, code)] = word_id
            
            # Create words that don't exist yet
            to_create = [(lemma, code) for lemma, code in batch if (lemma, code) not in existing_dict]
            
            if to_create:
                # For each word to create, use individual inserts with error handling
                for lemma, lang_code in to_create:
                    try:
                        # Create savepoint for each word
                        savepoint_name = f"sp_batch_create_{time.time_ns()}"
                        cur.execute(f"SAVEPOINT {savepoint_name}")
                        
                        normalized = normalize_lemma(lemma)
                        search_text = lemma + " " + normalized
                        
                        cur.execute("""
                            INSERT INTO words (lemma, normalized_lemma, language_code, search_text)
                            VALUES (%s, %s, %s, to_tsvector('simple', %s))
                            RETURNING id
                        """, (lemma, normalized, lang_code, search_text))
                        
                        word_id = cur.fetchone()[0]
                        existing_dict[(lemma, lang_code)] = word_id
                        
                        # Release savepoint
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    except Exception as e:
                        # Roll back to savepoint
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                            logger.warning(f"Failed to create word '{lemma}' in batch: {str(e)}")
                        except Exception as rollback_error:
                            logger.error(f"Failed to rollback savepoint: {str(rollback_error)}")
                            # If rollback fails, we need to abort the whole batch
                            conn.rollback()
                            logger.error("Rolling back entire batch due to transaction error")
                            break
            
            # Commit batch
            conn.commit()
            
            # Add batch results to main result dictionary
            result_dict.update(existing_dict)
            
        except Exception as batch_error:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(batch_error)}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            # Restore original autocommit state
            conn.autocommit = original_autocommit
    
    return result_dict

# -------------------------------------------------------------------
# Dictionary Entry Processing
# -------------------------------------------------------------------
def process_kwf_entry(cur, word: str, entry: Dict[str, Any]) -> None:
    """
    Process a single entry from kwf_dictionary.json and store it in the database,
    with comprehensive extraction of all word relationships.
    
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
        source = 'kwf_dictionary.json'
        standardized_source = SourceStandardization.standardize_sources(source)

        # Since KWF dictionary doesn't provide Baybayin forms, set to None
        has_baybayin = False
        baybayin_form = None
        romanized_form = None

        # Step 2: Insert the main word into the database
        word_id = get_or_create_word_id(
            cur,
            lemma=original_word,
            language_code=language_code,
            has_baybayin=has_baybayin,
            baybayin_form=baybayin_form,
            romanized_form=romanized_form,
            source=source
        )

        if not word_id:
            logger.error(f"Failed to create word ID for '{original_word}'")
            return

        # Step 3: Process metadata (etymology, pronunciation, etc.)
        metadata = entry.get('metadata', {})
        etymology_data = metadata.get('etymology', [])
        source_language_data = metadata.get('source_language', [])
        pronunciation_data = metadata.get('pronunciation', [])
        metadata_cross_references = metadata.get('cross_references', [])

        # Insert etymology
        for etym in etymology_data:
            etymology_text = etym.get('value', '')
            if etymology_text:
                language_codes = ', '.join([lang.get('value', '') for lang in source_language_data if lang.get('value')])
                insert_etymology(
                    cur,
                    word_id=word_id,
                    etymology_text=etymology_text,
                    normalized_components=json.dumps({'context': etym.get('context', [])}),
                    language_codes=language_codes,
                    sources=standardized_source
                )

        # Insert pronunciation data
        if pronunciation_data:
            pronunciation_dict = {
                "pronunciations": [
                    {
                        "value": p.get('value', ''),
                        "context": p.get('context', []),
                        "html": p.get('html', '')
                    } for p in pronunciation_data
                ]
            }
            cur.execute(
                "UPDATE words SET pronunciation_data = %s WHERE id = %s",
                (json.dumps(pronunciation_dict), word_id)
            )

        # Step 4: Process definitions and their relationships
        definitions = entry.get('definitions', {})
        part_of_speech_list = entry.get('part_of_speech', ['Unknown'])

        for pos in part_of_speech_list:
            if pos not in definitions:
                continue
            for def_entry in definitions[pos]:
                # Insert the definition
                meaning = def_entry.get('meaning', '')
                if not meaning or 'see' in def_entry:  # Skip "see" definitions
                    continue

                examples = []
                for example_set in def_entry.get('example_sets', []):
                    for example in example_set.get('examples', []):
                        examples.append(example.get('text', ''))

                usage_notes = def_entry.get('note', '')
                categories = def_entry.get('categories', ['General'])
                if categories:
                    usage_notes = f"[{' | '.join(categories)}] {usage_notes}" if usage_notes else f"[{' | '.join(categories)}]"

                definition_id = insert_definition(
                    cur,
                    word_id=word_id,
                    definition_text=meaning,
                    part_of_speech=pos,
                    examples=json.dumps(examples),
                    usage_notes=usage_notes,
                    sources=standardized_source
                )

                if not definition_id:
                    continue

                # Process definition-level relationships
                # Synonyms
                for syn in def_entry.get('synonyms', []):
                    syn_term = syn.get('term', '')
                    if syn_term:
                        syn_id = get_or_create_word_id(cur, lemma=syn_term, language_code=language_code)
                        if syn_id:
                            insert_definition_relation(
                                cur,
                                definition_id=definition_id,
                                word_id=syn_id,
                                relation_type='synonym',
                                sources=standardized_source
                            )
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=syn_id,
                                relation_type='synonym',
                                sources=standardized_source
                            )

                # Antonyms
                for ant in def_entry.get('antonyms', []):
                    ant_term = ant.get('term', '')
                    if ant_term:
                        ant_id = get_or_create_word_id(cur, lemma=ant_term, language_code=language_code)
                        if ant_id:
                            insert_definition_relation(
                                cur,
                                definition_id=definition_id,
                                word_id=ant_id,
                                relation_type='antonym',
                                sources=standardized_source
                            )
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=ant_id,
                                relation_type='antonym',
                                sources=standardized_source
                            )

                # Cross-references in definitions
                for xref in def_entry.get('cross_references', []):
                    xref_term = xref.get('term', '')
                    if xref_term and not xref.get('broken', True):  # Skip broken links
                        xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                        if xref_id:
                            insert_definition_relation(
                                cur,
                                definition_id=definition_id,
                                word_id=xref_id,
                                relation_type='cross_reference',
                                sources=standardized_source
                            )
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=xref_id,
                                relation_type='cross_reference',
                                sources=standardized_source
                            )

                # "See" references
                for see_ref in def_entry.get('see', []):
                    see_term = see_ref.get('term', '')
                    if see_term and not see_ref.get('broken', True):
                        see_id = get_or_create_word_id(cur, lemma=see_term, language_code=language_code)
                        if see_id:
                            insert_definition_relation(
                                cur,
                                definition_id=definition_id,
                                word_id=see_id,
                                relation_type='see_also',
                                sources=standardized_source
                            )
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=see_id,
                                relation_type='see_also',
                                sources=standardized_source
                            )

        # Step 5: Process metadata cross-references
        for xref in metadata_cross_references:
            xref_term = xref.get('term', '')
            if xref_term and not xref.get('broken', True):
                xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                if xref_id:
                    insert_relation(
                        cur,
                        from_word_id=word_id,
                        to_word_id=xref_id,
                        relation_type='cross_reference',
                        sources=standardized_source
                    )

        # Step 6: Process affixation (derived forms)
        for affix in entry.get('affixation', []):
            forms = affix.get('form', [])
            affix_types = affix.get('types', [])
            affix_type = ', '.join(affix_types) if affix_types else 'derived'
            for form in forms:
                if form:
                    affixed_id = get_or_create_word_id(cur, lemma=form, language_code=language_code)
                    if affixed_id:
                        insert_affixation(
                            cur,
                            root_id=word_id,
                            affixed_id=affixed_id,
                            affix_type=affix_type,
                            sources=standardized_source
                        )
                        insert_relation(
                            cur,
                            from_word_id=word_id,
                            to_word_id=affixed_id,
                            relation_type='derived',
                            sources=standardized_source
                        )
                        # Also insert the reverse relation
                        insert_relation(
                            cur,
                            from_word_id=affixed_id,
                            to_word_id=word_id,
                            relation_type='derived_from',
                            sources=standardized_source
                        )

            # Cross-references in affixation
            for xref in affix.get('cross_references', []):
                xref_term = xref.get('term', '')
                if xref_term and not xref.get('broken', True):
                    xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                    if xref_id:
                        insert_relation(
                            cur,
                            from_word_id=word_id,
                            to_word_id=xref_id,
                            relation_type='cross_reference',
                            sources=standardized_source
                        )

        # Step 7: Process idioms
        idioms = entry.get('idioms', [])
        idiom_list = []
        for idiom_entry in idioms:
            idiom_text = idiom_entry.get('idiom', '')
            if not idiom_text:
                continue

            # Create a new word entry for the idiom
            idiom_id = get_or_create_word_id(
                cur,
                lemma=idiom_text,
                language_code=language_code,
                tags='idiom'
            )

            if not idiom_id:
                continue

            # Insert the idiom as a definition
            meaning = idiom_entry.get('meaning', '')
            if meaning:
                insert_definition(
                    cur,
                    word_id=idiom_id,
                    definition_text=meaning,
                    part_of_speech='Idyoma',
                    examples=json.dumps(idiom_entry.get('examples', [])),
                    usage_notes='',
                    sources=standardized_source
                )

            # Link the idiom to the main word
            insert_relation(
                cur,
                from_word_id=word_id,
                to_word_id=idiom_id,
                relation_type='idiom',
                sources=standardized_source
            )

            # Cross-references in idioms
            for xref in idiom_entry.get('cross_references', []):
                xref_term = xref.get('term', '')
                if xref_term and not xref.get('broken', True):
                    xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                    if xref_id:
                        insert_relation(
                            cur,
                            from_word_id=idiom_id,
                            to_word_id=xref_id,
                            relation_type='cross_reference',
                            sources=standardized_source
                        )

            # Cross-references in idiom examples
            for xref_list in idiom_entry.get('example_cross_references', []):
                for xref in xref_list:
                    xref_term = xref.get('term', '')
                    if xref_term and not xref.get('broken', True):
                        xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                        if xref_id:
                            insert_relation(
                                cur,
                                from_word_id=idiom_id,
                                to_word_id=xref_id,
                                relation_type='cross_reference',
                                sources=standardized_source
                            )

            idiom_list.append({
                'text': idiom_text,
                'meaning': meaning
            })

        # Update the main word with the idioms list
        if idiom_list:
            cur.execute(
                "UPDATE words SET idioms = %s WHERE id = %s",
                (json.dumps(idiom_list), word_id)
            )

        # Step 8: Process related terms
        related_data = entry.get('related', {})
        for rel_term in related_data.get('related_terms', []):
            rel_term_text = rel_term.get('term', '')
            if rel_term_text and not rel_term.get('broken', True):
                rel_id = get_or_create_word_id(cur, lemma=rel_term_text, language_code=language_code)
                if rel_id:
                    insert_relation(
                        cur,
                        from_word_id=word_id,
                        to_word_id=rel_id,
                        relation_type='related',
                        sources=standardized_source
                    )

        # Antonyms from related section
        for ant_term in related_data.get('antonyms', []):
            ant_term_text = ant_term.get('term', '')
            if ant_term_text and not ant_term.get('broken', True):
                ant_id = get_or_create_word_id(cur, lemma=ant_term_text, language_code=language_code)
                if ant_id:
                    insert_relation(
                        cur,
                        from_word_id=word_id,
                        to_word_id=ant_id,
                        relation_type='antonym',
                        sources=standardized_source
                    )

        # Step 9: Process other sections (unrecognized sections)
        other_sections = entry.get('other_sections', {})
        for section_name, section_data in other_sections.items():
            usage_notes = []
            for content in section_data.get('content', []):
                usage_notes.append(content.get('text', ''))
                for xref in content.get('cross_references', []):
                    xref_term = xref.get('term', '')
                    if xref_term and not xref.get('broken', True):
                        xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                        if xref_id:
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=xref_id,
                                relation_type='cross_reference',
                                sources=standardized_source
                            )
            for subsection in section_data.get('subsections', []):
                usage_notes.append(f"{subsection.get('title', 'Section')}: {'; '.join(subsection.get('content', []))}")
                for xref in subsection.get('cross_references', []):
                    xref_term = xref.get('term', '')
                    if xref_term and not xref.get('broken', True):
                        xref_id = get_or_create_word_id(cur, lemma=xref_term, language_code=language_code)
                        if xref_id:
                            insert_relation(
                                cur,
                                from_word_id=word_id,
                                to_word_id=xref_id,
                                relation_type='cross_reference',
                                sources=standardized_source
                            )
            if usage_notes:
                cur.execute(
                    "UPDATE words SET tags = COALESCE(tags, '') || %s WHERE id = %s",
                    (f", {section_name}: {'; '.join(usage_notes)}", word_id)
                )

        logger.info(f"Successfully processed KWF entry for word: {original_word} (ID: {word_id})")

    except Exception as e:
        logger.error(f"Error processing KWF entry for word '{word}': {str(e)}")
        raise

def process_tagalog_words(cur, filename: str):
    """
    Process tagalog-words.json with structure:
    {
        "word1": {
            "pronunciation": "...",
            "part_of_speech": "...",
            "etymology": "...",
            "derivative": "...",
            "definitions": ["definition1", "definition2", ...]
        },
        ...
    }
    or list format: [{"word": "...", "definition": "...", "part_of_speech": "...", "etymology": "..."}, ...]
    """
    logger.info(f"Processing Tagalog words from: {filename}")
    source = SourceStandardization.standardize_sources('tagalog-words.json')
    language_code = 'tl'
    romanizer = BaybayinRomanizer()

    # Words with letters that shouldn't have Baybayin forms
    NON_BAYBAYIN_LETTERS = "fjcñvxz"
    SKIP_BAYBAYIN_ENDINGS = ['ismo', 'ista', 'dad', 'cion', 'syon']

    def validate_for_baybayin(word):
        """Check if a word should have a Baybayin form"""
        if not word:
            return False
            
        # Skip words that definitely can't be converted to proper Baybayin
        if (len(word) == 1 or 
            word.isupper() or
            any(char in word.lower() for char in NON_BAYBAYIN_LETTERS) or
            any(word.lower().endswith(suffix) for suffix in SKIP_BAYBAYIN_ENDINGS) or
            (word[0].isupper() and not word.isupper()) or
            any(word in word.lower() for word in ['http', 'www', '.com', '.org', '.net']) or
            word.startswith(('-', '_', '.')) or
            any(char.isdigit() for char in word)):
            return False
            
        return True

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prepare batch word entries
        word_entries = []
        if isinstance(data, dict):
            total_entries = len(data)
            logger.info(f"Found {total_entries} dictionary-style entries")
            for word, entry in data.items():
                if not word or not isinstance(entry, dict):
                    logger.warning(f"Skipping invalid entry: {word}")
                    continue
                word_entries.append((word, language_code))
        else:
            total_entries = len(data)
            logger.info(f"Found {total_entries} list-style entries")
            for entry in data:
                if not isinstance(entry, dict) or 'word' not in entry:
                    logger.warning(f"Skipping invalid entry: {entry}")
                    continue
                word_entries.append((entry['word'], language_code))

        # First, we need to ensure all words exist in the database
        # Insert or get all words (with proper transaction handling)
        conn = cur.connection
        try:
            # Explicitly begin a transaction for word creation
            conn.autocommit = False
            logger.info(f"Creating {len(word_entries)} words...")
            word_ids = {}
            
            for i, (word, lang) in enumerate(word_entries):
                try:
                    # Use individual transactions to ensure each word is properly committed
                    cur.execute("BEGIN")
                    
                    # Check if word already exists
                    normalized = normalize_lemma(word)
                    cur.execute("""
                        SELECT id FROM words WHERE normalized_lemma = %s AND language_code = %s
                    """, (normalized, lang))
                    result = cur.fetchone()
                    
                    if result:
                        # Word exists, use its ID
                        word_ids[(word, lang)] = result[0]
                        cur.execute("COMMIT")
                    else:
                        # Create new word (without Baybayin for now)
                        search_text = word + " " + normalized
                        cur.execute("""
                            INSERT INTO words (lemma, normalized_lemma, language_code, search_text)
                            VALUES (%s, %s, %s, to_tsvector('simple', %s))
                            RETURNING id
                        """, (word, normalized, lang, search_text))
                        
                        word_id = cur.fetchone()[0]
                        word_ids[(word, lang)] = word_id
                        
                        # Commit this word insertion
                        cur.execute("COMMIT")
                        
                        # Double-check that the word was actually created
                        cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
                        if not cur.fetchone():
                            logger.error(f"Failed to create word {word} (ID: {word_id})")
                            word_ids.pop((word, lang), None)
                    
                    # Progress logging
                    if (i + 1) % 1000 == 0:
                        logger.info(f"Processed {i + 1}/{len(word_entries)} words")
                        
                except Exception as e:
                    logger.error(f"Error creating word {word}: {str(e)}")
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass  # Ignore rollback errors
            
            logger.info(f"Successfully created/found {len(word_ids)} words")

            # Now process the entries with their definitions, etc.
            with tqdm(total=total_entries, desc="Processing Tagalog words") as pbar:
                entries = data.items() if isinstance(data, dict) else [(entry['word'], entry) for entry in data]
                for word, entry in entries:
                    if (word, language_code) not in word_ids:
                        logger.warning(f"No word_id for {word}, skipping")
                        pbar.update(1)
                        continue
                    
                    try:
                        # Start transaction for this word's definitions/metadata
                        cur.execute("BEGIN")
                        word_id = word_ids[(word, language_code)]
                        
                        # Double-check the word ID is valid
                        cur.execute("SELECT id FROM words WHERE id = %s", (word_id,))
                        if not cur.fetchone():
                            logger.error(f"Word ID {word_id} for '{word}' not found in database. Skipping.")
                            cur.execute("ROLLBACK")
                            pbar.update(1)
                            continue

                        # Extract fields
                        pos = standardize_pos(entry.get('part_of_speech', ''))
                        definitions = entry.get('definitions', entry.get('definition', []))
                        if isinstance(definitions, str):
                            definitions = [definitions]
                        etymology = entry.get('etymology', '')
                        derivative = entry.get('derivative', '')
                        pronunciation = entry.get('pronunciation', word)

                        # Baybayin handling with proper validation
                        has_baybayin = False
                        baybayin_form = None
                        romanized_form = word
                        
                        # Only attempt Baybayin conversion if the word is valid for it
                        if validate_for_baybayin(word):
                            if romanizer.is_baybayin(word):
                                # Word is already in Baybayin
                                has_baybayin = True
                                baybayin_form = word
                                romanized_form = romanizer.romanize(word)
                            elif not has_diacritics(word):
                                # Try to transliterate to Baybayin
                                try:
                                    temp_baybayin = transliterate_to_baybayin(word)
                                    if temp_baybayin and validate_baybayin_entry(temp_baybayin, word):
                                        has_baybayin = True
                                        baybayin_form = temp_baybayin
                                except Exception as bay_error:
                                    logger.debug(f"Failed to generate Baybayin for '{word}': {str(bay_error)}")
                                    has_baybayin = False
                                    baybayin_form = None

                        # Update word with pronunciation only (skip Baybayin if it failed validation)
                        try:
                            if has_baybayin and baybayin_form:
                                # Update with Baybayin
                                cur.execute("""
                                    UPDATE words 
                                    SET has_baybayin = %s,
                                        baybayin_form = %s,
                                        romanized_form = %s,
                                        pronunciation_data = %s,
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = %s
                                """, (has_baybayin, baybayin_form, romanized_form, 
                                      json.dumps({"hyphenation": pronunciation}), word_id))
                            else:
                                # Update without Baybayin
                                cur.execute("""
                                    UPDATE words 
                                    SET pronunciation_data = %s,
                                        updated_at = CURRENT_TIMESTAMP
                                    WHERE id = %s
                                """, (json.dumps({"hyphenation": pronunciation}), word_id))
                        except Exception as update_error:
                            logger.error(f"Error updating word '{word}': {str(update_error)}")
                            # Continue processing definitions and other data

                        # Process definitions
                        for def_text in definitions:
                            if not def_text or not def_text.strip():
                                continue
                            def_text = re.sub(r'\s*\.{3,}\s*', '', def_text).strip()  # Clean ellipses
                            try:
                                insert_definition(cur, word_id, def_text, part_of_speech=pos, sources=source)
                            except Exception as def_error:
                                logger.error(f"Error inserting definition for word '{word}' (ID: {word_id}): {str(def_error)}")

                        # Process etymology
                        if etymology and etymology.strip():
                            lang_codes = extract_language_codes(etymology)
                            components = extract_etymology_components(etymology)
                            try:
                                insert_etymology(
                                    cur,
                                    word_id,
                                    etymology,
                                    normalized_components=json.dumps(components) if components else None,
                                    language_codes=",".join(lang_codes),
                                    sources=source
                                )
                            except Exception as etym_error:
                                logger.error(f"Error inserting etymology for word '{word}': {str(etym_error)}")

                        # Process derivatives
                        if derivative and derivative.strip():
                            if "mula sa" in derivative.lower():
                                root_match = re.search(r'mula sa\s+(.+?)(?:\s+na|\s*$)', derivative)
                                if root_match:
                                    root_word = root_match.group(1).strip()
                                    try:
                                        # Create a separate transaction for the root word
                                        cur.execute("SAVEPOINT derivatives")
                                        root_id = get_or_create_word_id(cur, root_word, language_code)
                                        if root_id:
                                            insert_relation(cur, word_id, root_id, "derived_from", sources=source)
                                        cur.execute("RELEASE SAVEPOINT derivatives")
                                    except Exception as rel_error:
                                        logger.error(f"Error processing derivation for '{word}': {str(rel_error)}")
                                        cur.execute("ROLLBACK TO SAVEPOINT derivatives")
                            else:
                                # Treat as affixed forms
                                derivs = [d.strip() for d in derivative.split(',') if d.strip()]
                                for deriv in derivs:
                                    try:
                                        # Create a separate transaction for each derivation
                                        cur.execute("SAVEPOINT affixation")
                                        deriv_id = get_or_create_word_id(cur, deriv, language_code)
                                        if deriv_id:
                                            insert_affixation(cur, word_id, deriv_id, "derived", sources=source)
                                        cur.execute("RELEASE SAVEPOINT affixation")
                                    except Exception as deriv_error:
                                        logger.error(f"Error processing affixation for '{word}': {str(deriv_error)}")
                                        cur.execute("ROLLBACK TO SAVEPOINT affixation")

                        # Extract relationships from definitions
                        for def_text in definitions:
                            if def_text and def_text.strip():
                                try:
                                    cur.execute("SAVEPOINT def_relations")
                                    process_definition_relations(cur, word_id, def_text, source)
                                    cur.execute("RELEASE SAVEPOINT def_relations")
                                except Exception as rel_error:
                                    logger.error(f"Error processing definition relations for '{word}': {str(rel_error)}")
                                    cur.execute("ROLLBACK TO SAVEPOINT def_relations")

                        # Commit this word's processed data
                        cur.execute("COMMIT")
                    except Exception as e:
                        logger.error(f"Error processing {word}: {str(e)}")
                        try:
                            cur.execute("ROLLBACK")
                        except Exception:
                            pass  # Ignore rollback errors
                    finally:
                        pbar.update(1)
        finally:
            # Restore autocommit behavior
            conn.autocommit = True

        logger.info("Completed processing tagalog-words.json")
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {str(e)}")
        try:
            cur.connection.rollback()
        except Exception:
            pass  # Ignore rollback errors
        raise

def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    lang_map = {
        "Esp": "es", "Eng": "en", "Ch": "zh", "Tsino": "zh", "Jap": "ja",
        "San": "sa", "Sanskrit": "sa", "Tag": "tl", "Mal": "ms", "Arb": "ar"
    }
    return [lang_map[lang] for lang in lang_map if lang in etymology]

def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Extract synonyms and antonyms from definition text."""
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
RELATION_MAPPING: Dict[str, Dict[str, Any]] = {
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
    "hypernym": {
        "relation_type": "hypernym_of",
        "bidirectional": False,
        "inverse": "hyponym_of"
    },
    "hyponym": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of"
    },
    "meronym": {
        "relation_type": "meronym_of",
        "bidirectional": False,
        "inverse": "holonym_of"
    },
    "holonym": {
        "relation_type": "holonym_of",
        "bidirectional": False,
        "inverse": "meronym_of"
    },
    "derived": {
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
    # Extend as needed
}

def normalize_relation_type(raw_key: str) -> Tuple[str, bool, Optional[str]]:
    key_lower = raw_key.lower().strip()
    if key_lower.endswith("s"):
        key_lower = key_lower[:-1]  # E.g. "synonyms" -> "synonym"
    if key_lower in RELATION_MAPPING:
        info = RELATION_MAPPING[key_lower]
        return info["relation_type"], info.get("bidirectional", False), info.get("inverse")
    else:
        # Unrecognized: keep as raw
        return (raw_key, False, None)

def process_relationships(cur, word_id, data, sources):
    """
    Process and insert relationships between words based on the provided data.
    
    Args:
        cur: Database cursor
        word_id: The ID of the word being processed
        data: Dictionary containing word data
        sources: Source information for the relationships
    """
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
    """Process entries from Kaikki.org's jsonl files."""
    logger.info(f"Processing Kaikki entries from: {filename}")
    source = SourceStandardization.standardize_sources('kaikki')
    
    # No need to change autocommit at this level - let the decorator handle it
    
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        forms = entry.get('forms', [])
        for form in forms:
            if form.get('form', '').strip() and ('ᜀ' in form.get('form', '') or '᜔' in form.get('form', '')):
                baybayin = form.get('form', '').strip()
                romanized = entry.get('word', '')
                return baybayin, romanized
        return None, None
        
    def standardize_entry_pos(pos_str: str) -> str:
        return standardize_pos(pos_str)
    
    def process_relations(cur, from_word_id: int, relations_dict: Dict[str, List[str]], lang_code: str, source: str):
        """Process relations from a Kaikki entry."""
        for rel_type, targets in relations_dict.items():
            normalized_type, is_reverse, _ = normalize_relation_type(rel_type)
            
            if not normalized_type:
                continue
                
            for target in targets:
                try:
                    # Create savepoint for each relation
                    savepoint_name = f"sp_kaikki_rel_{time.time_ns()}"
                    try:
                        cur.execute(f"SAVEPOINT {savepoint_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create savepoint for relation: {str(e)}")
                        continue
                        
                    # Get target word ID
                    try:
                        to_word_id = get_or_create_word_id(cur, target, lang_code)
                    except Exception as e:
                        logger.error(f"Error getting word ID for relation target '{target}': {str(e)}")
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        except Exception:
                            pass  # Ignore rollback errors
                        continue
                    
                    if not to_word_id:
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        except Exception:
                            pass  # Ignore rollback errors
                        continue
                    
                    # Insert the relation with correct direction
                    try:
                        if is_reverse:
                            insert_relation(cur, to_word_id, from_word_id, normalized_type, sources=source)
                        else:
                            insert_relation(cur, from_word_id, to_word_id, normalized_type, sources=source)
                        
                        try:
                            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                        except Exception:
                            pass  # Ignore release errors
                    except Exception as rel_error:
                        logger.error(f"Error inserting relation '{normalized_type}' from {from_word_id} to {to_word_id}: {str(rel_error)}")
                        try:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        except Exception:
                            pass  # Ignore rollback errors
                except Exception as e:
                    logger.error(f"Unexpected error processing relation: {str(e)}")
    
    def process_entry(cur, entry: Dict):
        """Process a single entry from Kaikki."""
        # Extract basic information
        word = entry.get('word', '').strip()
        if not word:
            logger.warning("Skipping entry without word")
            return
            
        lang_code = 'tl'  # Tagalog
        pos = standardize_entry_pos(entry.get('pos', ''))
        
        # Extract Baybayin if available
        baybayin_form, romanized_form = extract_baybayin_info(entry)
        has_baybayin = bool(baybayin_form)
        
        # Create the word - this function has its own transaction handling
        try:
            word_id = get_or_create_word_id(
                cur, 
                word, 
                language_code=lang_code,
                has_baybayin=has_baybayin,
                baybayin_form=baybayin_form,
                romanized_form=romanized_form if has_baybayin else word
            )
            
            if not word_id:
                logger.error(f"Failed to create word '{word}'")
                return
        except Exception as e:
            logger.error(f"Error in get_or_create_word_id for lemma '{word}': {str(e)}")
            return
        
        # Process senses (definitions)
        senses = entry.get('senses', [])
        for sense in senses:
            glosses = sense.get('glosses', [])
            if not glosses:
                continue
                
            definition = "; ".join(glosses)
            
            # Extract examples
            examples = []
            for example in sense.get('examples', []):
                ex_text = example.get('text', '')
                if ex_text:
                    examples.append(ex_text)
            
            # Insert definition - this function has its own transaction handling
            try:
                definition_id = insert_definition(
                    cur,
                    word_id=word_id,
                    definition_text=definition,
                    part_of_speech=pos,
                    examples=json.dumps(examples) if examples else None,
                    sources=source
                )
                
                if not definition_id:
                    logger.warning(f"Failed to insert definition for '{word}': {definition}")
                    continue
            except Exception as def_error:
                logger.error(f"Error inserting definition for '{word}': {str(def_error)}")
                continue
        
        # Process relations
        relations = entry.get('relations', {})
        if relations:
            try:
                process_relations(cur, word_id, relations, lang_code, source)
            except Exception as rel_error:
                logger.error(f"Error processing relations for '{word}': {str(rel_error)}")
        
        # Process etymology
        etymology = entry.get('etymology_text', '')
        if etymology:
            try:
                lang_codes = extract_language_codes(etymology)
                components = extract_etymology_components(etymology)
                
                # This function has its own transaction handling
                insert_etymology(
                    cur,
                    word_id,
                    etymology,
                    normalized_components=json.dumps(components) if components else None,
                    language_codes=",".join(lang_codes),
                    sources=source
                )
            except Exception as etym_error:
                logger.error(f"Error inserting etymology for '{word}': {str(etym_error)}")
    
    try:
        # Try using jsonlines module (preferred)
        try:
            import jsonlines
            console = Console()
            console.print("[blue]Using jsonlines module to process Kaikki data.[/]")
            
            total_entries = 0
            processed_entries = 0
            error_entries = 0
            
            # First, count total entries
            with jsonlines.open(filename) as reader:
                for _ in reader:
                    total_entries += 1
            
            # Process entries with progress bar
            with tqdm(total=total_entries, desc="Processing Kaikki.org (Tagalog)") as pbar:
                with jsonlines.open(filename) as reader:
                    for entry in reader:
                        try:
                            process_entry(cur, entry)
                            processed_entries += 1
                        except Exception as e:
                            logger.error(f"Failed to process entry: {str(e)}")
                            error_entries += 1
                        finally:
                            pbar.update(1)
            
            logger.info(f"Processed {processed_entries} entries from Kaikki.org with {error_entries} errors")
            
        except ImportError:
            # Fallback to manual line-by-line JSON parsing
            console = Console()
            console.print("[yellow]jsonlines module not available. Using fallback method.[/]")
            
            total_entries = 0
            processed_entries = 0
            error_entries = 0
            
            # First, count total entries
            with open(filename, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_entries += 1
            
            # Process entries
            with tqdm(total=total_entries, desc="Processing Kaikki.org (Tagalog)") as pbar:
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if line.strip():
                                entry = json.loads(line)
                                process_entry(cur, entry)
                                processed_entries += 1
                        except Exception as e:
                            logger.error(f"Failed to process entry: {str(e)}")
                            error_entries += 1
                        finally:
                            pbar.update(1)
            
            logger.info(f"Processed {processed_entries} entries from Kaikki.org with {error_entries} errors")
    except Exception as e:
        logger.error(f"Error processing Kaikki file: {str(e)}")

# -------------------------------------------------------------------
# Command Line Interface Functions
# -------------------------------------------------------------------
def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the dictionary manager."""
    parser = argparse.ArgumentParser(description='Filipino Dictionary Manager')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Migrate data
    migrate_parser = subparsers.add_parser('migrate', help='Migrate dictionary data')
    migrate_parser.add_argument('--source', nargs='*', help='Specific source(s) to migrate (space-separated list)', 
                              choices=["root-words", "root-words-cleaned", "tagalog-words", "kwf-dictionary", "kaikki"])
    migrate_parser.add_argument('--file', help='Custom file to process')
    migrate_parser.add_argument('--data-dir', help='Directory containing data files')
    migrate_parser.add_argument('--exit-on-error', action='store_true', help='Exit on first error')
    migrate_parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup routines')
    migrate_parser.set_defaults(func=migrate_data_cli)

    # Verify database
    verify_parser = subparsers.add_parser('verify', help='Verify database integrity')
    verify_parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    verify_parser.set_defaults(func=verify_database_cli)

    # Update database
    update_parser = subparsers.add_parser('update', help='Update database schema')
    update_parser.set_defaults(func=update_database_cli)

    # Purge database
    purge_parser = subparsers.add_parser('purge', help='Purge all dictionary data')
    purge_parser.add_argument('--confirm', action='store_true', 
                              help='Confirm purging without additional prompt')
    purge_parser.set_defaults(func=purge_database_cli)

    # Cleanup database
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up database')
    cleanup_parser.add_argument('--relations', action='store_true', 
                               help='Clean up relations')
    cleanup_parser.add_argument('--definitions', action='store_true', 
                               help='Deduplicate definitions')
    cleanup_parser.add_argument('--baybayin', action='store_true', 
                               help='Clean up Baybayin data')
    cleanup_parser.add_argument('--all', action='store_true', 
                               help='Run all cleanup routines')
    cleanup_parser.set_defaults(func=cleanup_database_cli)

    # Lookup word
    lookup_parser = subparsers.add_parser('lookup', help='Look up a word')
    lookup_parser.add_argument('word', help='Word to look up')
    lookup_parser.add_argument('--id', action='store_true', 
                              help='Look up by ID instead of word')
    lookup_parser.add_argument('--lang', choices=['tl', 'ceb'], default='tl',
                              help='Language code (default: tl)')
    lookup_parser.set_defaults(func=lookup_word_cli)

    # Display dictionary stats
    stats_parser = subparsers.add_parser('stats', help='Display dictionary statistics')
    stats_parser.set_defaults(func=display_dictionary_stats_cli)

    # Display leaderboard
    leaderboard_parser = subparsers.add_parser('leaderboard', help='Display contributor leaderboard')
    leaderboard_parser.set_defaults(func=display_leaderboard_cli)

    # Explore dictionary
    explore_parser = subparsers.add_parser('explore', help='Interactive dictionary explorer')
    explore_parser.set_defaults(func=explore_dictionary_cli)

    # Test database
    test_parser = subparsers.add_parser('test', help='Run database tests')
    test_parser.set_defaults(func=test_database_cli)

    # Display help
    help_parser = subparsers.add_parser('help', help='Display help')
    help_parser.set_defaults(func=display_help_cli)

    return parser

def migrate_data(args):
    """
    Process and migrate data from various sources into the dictionary database.
    
    Args:
        args: Command-line arguments including source specifications
    """
    console = Console()
    
    # Determine data directory
    data_dirs = []
    if hasattr(args, 'data_dir') and args.data_dir:
        data_dirs.append(args.data_dir)
    else:
        # Try common data directories
        data_dirs = ["data", "../data", os.path.join(os.path.dirname(__file__), "data"), 
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")]
    
    # Make sure at least one data directory exists
    existing_data_dirs = [d for d in data_dirs if os.path.isdir(d)]
    if not existing_data_dirs:
        console.print(f"[bold red]Error: Could not find a valid data directory.[/]")
        console.print("Please specify a valid data directory with --data-dir or create a 'data' directory.")
        return
    
    # Use the first existing data directory
    data_dir = existing_data_dirs[0]
    console.print(f"[blue]Using data directory: {data_dir}[/]")
    
    # Connect to the database
    try:
        conn = get_connection()
        conn.autocommit = True
        
        # Create tables if they don't exist
        create_or_update_tables(conn)
        
        # Set up extensions
        setup_extensions(conn)
        
        # Now process each source in a separate transaction
        with conn.cursor() as cur:
            # Fix any existing transaction issues first
            try:
                cur.execute("ROLLBACK")
                logger.info("Cleared any existing transactions")
            except Exception:
                pass
            
            # Process each source
            if args.source:
                source_list = args.source
            else:
                source_list = [
                    "root-words-cleaned",
                    "tagalog-words", 
                    "kwf-dictionary",
                    "kaikki"
                ]
            
            for source in source_list:
                try:
                    conn.autocommit = True  # Reset to autocommit mode for each source
                    console.print(f"[bold blue]Processing source: {source}[/bold blue]")
                    
                    if source == "root-words":
                        file_path = os.path.join(data_dir, "root-words-enhanced.json")
                        if os.path.exists(file_path):
                            process_root_words(cur, file_path)
                        else:
                            console.print(f"[yellow]File not found: {file_path}[/]")
                            continue
                    
                    elif source == "root-words-cleaned":
                        file_path = os.path.join(data_dir, "root-words-cleaned.json")
                        if os.path.exists(file_path):
                            process_root_words_cleaned(cur, file_path)
                        else:
                            console.print(f"[yellow]File not found: {file_path}[/]")
                            continue
                    
                    elif source == "tagalog-words":
                        file_path = os.path.join(data_dir, "tagalog-words.json")
                        if os.path.exists(file_path):
                            process_tagalog_words(cur, file_path)
                        else:
                            console.print(f"[yellow]File not found: {file_path}[/]")
                            continue
                    
                    elif source == "kwf-dictionary":
                        file_path = os.path.join(data_dir, "kwf-dictionary.json")
                        if os.path.exists(file_path):
                            process_kwf_dictionary(cur, file_path)
                        else:
                            console.print(f"[yellow]File not found: {file_path}[/]")
                            continue
                    
                    elif source == "kaikki":
                        # Try both possible Kaikki filenames
                        kaikki_filenames = ["kaikki-tl.jsonl", "kaikki.jsonl"]
                        kaikki_file_found = False
                        
                        for kaikki_file in kaikki_filenames:
                            file_path = os.path.join(data_dir, kaikki_file)
                            if os.path.exists(file_path):
                                try:
                                    # Check if jsonlines module is available
                                    try:
                                        import jsonlines
                                    except ImportError:
                                        console.print("[yellow]The 'jsonlines' module is required for processing Kaikki data.[/]")
                                        console.print("[yellow]Installing jsonlines module...[/]")
                                        try:
                                            import subprocess
                                            subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonlines"])
                                            import jsonlines
                                            console.print("[green]Successfully installed jsonlines module.[/]")
                                        except Exception as install_error:
                                            console.print(f"[red]Failed to install jsonlines: {str(install_error)}[/]")
                                            console.print("[yellow]Please install manually: pip install jsonlines[/]")
                                            continue
                                    
                                    process_kaikki_jsonl(cur, file_path)
                                    kaikki_file_found = True
                                    break
                                except Exception as kaikki_error:
                                    logger.error(f"Error processing Kaikki file {file_path}: {str(kaikki_error)}")
                                    console.print(f"[red]Error processing Kaikki file: {str(kaikki_error)}[/]")
                                    continue
                        
                        if not kaikki_file_found:
                            console.print(f"[yellow]No Kaikki files found in {data_dir}[/]")
                            continue
                    
                    console.print(f"[green]Completed processing source: {source}[/green]")
                except Exception as e:
                    conn.autocommit = True  # Ensure we're in autocommit mode
                    
                    # Try to clean up any pending transaction
                    try:
                        cur.execute("ROLLBACK")
                    except Exception:
                        pass
                    
                    logger.error(f"Error processing source {source}: {str(e)}")
                    console.print(f"[bold red]Error processing source {source}: {str(e)}[/bold red]")
                    
                    if hasattr(args, 'exit_on_error') and args.exit_on_error:
                        console.print("[yellow]Exiting due to error (--exit-on-error flag is set)[/yellow]")
                        return
                    
                    console.print("[yellow]Continuing with next source...[/yellow]")
                    continue
            
            # Run cleanup and consistency checks
            try:
                console.print("[bold blue]Running cleanup routines...[/bold blue]")
                
                if not hasattr(args, 'no_cleanup') or not args.no_cleanup:
                    console.print("[yellow]Cleaning up dictionary data...[/yellow]")
                    cleanup_dictionary_data(cur)
                
                    console.print("[yellow]Cleaning up Baybayin data...[/yellow]")
                    cleanup_baybayin_data(cur)
                    
                    console.print("[yellow]Checking Baybayin consistency...[/yellow]")
                    check_baybayin_consistency(cur)
                
                console.print("[green]Cleanup completed.[/green]")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                console.print(f"[bold red]Error during cleanup: {str(cleanup_error)}[/bold red]")
                
                if hasattr(args, 'exit_on_error') and args.exit_on_error:
                    return
                
            console.print("[bold green]Migration process completed.[/bold green]")
    
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        console.print(f"[bold red]Database connection error: {str(e)}[/bold red]")
    finally:
        if 'conn' in locals() and conn:
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
    """CLI wrapper for migrate_data function."""
    try:
        # Check database connection first
        conn = get_connection()
        conn.close()
        
        # If connection successful, run migration
        migrate_data(args)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error during migration:[/] {str(e)}")
        logger.error(f"Error in migrate_data_cli: {str(e)}")
        sys.exit(1)

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
