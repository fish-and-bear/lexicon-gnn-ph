#!/usr/bin/env python3
"""
dictionary_manager.py

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
import codecs
import locale
from typing import Optional
from language_systems import LanguageSystem
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import functools
from psycopg2.errors import UniqueViolation
import psycopg2.extras
import re
from enum import Enum

# -------------------------------------------------------------------
# (NEW) Instantiate a LanguageSystem from language_systems
# -------------------------------------------------------------------
lsys = LanguageSystem()

load_dotenv()

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
# 1. Database Credentials (adjust for your environment)
# -------------------------------------------------------------------
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

# After loading env vars, add this check
if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
    print("Error: Missing database configuration!")
    print("Please ensure you have a .env file with the following variables:")
    print("DB_NAME - The name of your PostgreSQL database")
    print("DB_USER - Your PostgreSQL username")
    print("DB_PASSWORD - Your PostgreSQL password")
    print("DB_HOST - Your database host (usually 'localhost')")
    sys.exit(1)

# -------------------------------------------------------------------
# 1. DATABASE CONNECTION
# -------------------------------------------------------------------
def get_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        logger.info("Attempting database connection...")
        logger.info(f"Database configuration:")
        logger.info(f"  Database Name: {DB_NAME}")
        logger.info(f"  User: {DB_USER}")
        logger.info(f"  Host: {DB_HOST}")
        logger.info(f"  Password: {'*' * len(DB_PASSWORD) if DB_PASSWORD else 'Not set'}")

        if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST]):
            logger.error("Missing database configuration!")
            raise ValueError("Missing database configuration. Please check your .env file.")

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
# 2. SCHEMA CREATION / UPDATE
# -------------------------------------------------------------------
TABLE_CREATION_SQL = """

-- Parts of speech reference table
CREATE TABLE IF NOT EXISTS parts_of_speech (
    id SERIAL PRIMARY KEY,
    code VARCHAR(32) NOT NULL UNIQUE,
    name_en VARCHAR(64) NOT NULL,
    name_tl VARCHAR(64) NOT NULL
);

-- words table
CREATE TABLE IF NOT EXISTS words (
    id                SERIAL PRIMARY KEY,
    lemma             VARCHAR(255) NOT NULL,
    normalized_lemma  VARCHAR(255) NOT NULL,
    has_baybayin      BOOLEAN DEFAULT FALSE,
    baybayin_form     VARCHAR(255),
    romanized_form    VARCHAR(255), -- Added column
    language_code     VARCHAR(16)  NOT NULL,
    root_word_id      INT REFERENCES words(id),
    preferred_spelling VARCHAR(255),
    tags              TEXT,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (language_code, normalized_lemma)
);

-- (NEW) Composite index for faster lookups by (language_code, normalized_lemma)
CREATE INDEX IF NOT EXISTS idx_words_lang_norm_lemma 
    ON words(language_code, normalized_lemma);
CREATE INDEX IF NOT EXISTS idx_words_baybayin 
    ON words(baybayin_form) WHERE has_baybayin = TRUE;

-- Modified definitions table
CREATE TABLE IF NOT EXISTS definitions (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id),
    definition_text TEXT NOT NULL,
    original_pos TEXT,                          -- Keep original POS string
    standardized_pos_id INT REFERENCES parts_of_speech(id), -- Reference to standard POS
    examples TEXT,
    usage_notes TEXT,
    sources TEXT NOT NULL
);

-- Create index for faster POS lookups
CREATE INDEX IF NOT EXISTS idx_definitions_std_pos 
    ON definitions(standardized_pos_id);

-- relations table
CREATE TABLE IF NOT EXISTS relations (
    id            SERIAL PRIMARY KEY,
    from_word_id  INT NOT NULL REFERENCES words(id),
    to_word_id    INT NOT NULL REFERENCES words(id),
    relation_type VARCHAR(64) NOT NULL,
    sources       TEXT NOT NULL
);

-- etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id                   SERIAL PRIMARY KEY,
    word_id              INT NOT NULL REFERENCES words(id),
    original_text        TEXT NOT NULL,
    normalized_components TEXT,
    language_codes       TEXT,
    sources              TEXT NOT NULL,
    -- Here's the new line:
    CONSTRAINT etymologies_wordid_originaltext_uniq UNIQUE (word_id, original_text)
);
"""

def create_or_update_tables(conn):
    """
    Create or update the database schema for the Filipino dictionary.
    
    Schema includes:
    - words: Main word entries with language and metadata
    - definitions: Word meanings with examples and usage notes
    - relations: Word relationships (synonyms, roots, derivatives)
    - etymologies: Word origins and language evolution
    
    Args:
        conn: Active database connection
    """
    logger.info("Starting table creation/update process.")
    cur = conn.cursor()
    try:
        cur.execute(TABLE_CREATION_SQL)

        # Populate standard POS entries
        pos_entries = [
            ('n', 'Noun', 'Pangngalan'),
            ('v', 'Verb', 'Pandiwa'),
            ('adj', 'Adjective', 'Pang-uri'),
            ('adv', 'Adverb', 'Pang-abay'),
            ('pron', 'Pronoun', 'Panghalip'),
            ('prep', 'Preposition', 'Pang-ukol'),
            ('conj', 'Conjunction', 'Pangatnig'),
            ('intj', 'Interjection', 'Pandamdam'),
            ('affix', 'Affix', 'Panlapi'),
            ('baybay', 'Baybayin Script', 'Baybayin'),
            ('unc', 'Uncategorized', 'Hindi Tiyak')
        ]

        cur.execute("TRUNCATE parts_of_speech RESTART IDENTITY CASCADE")
        for code, name_en, name_tl in pos_entries:
            cur.execute("""
                INSERT INTO parts_of_speech (code, name_en, name_tl)
                VALUES (%s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                SET name_en = EXCLUDED.name_en,
                    name_tl = EXCLUDED.name_tl
            """, (code, name_en, name_tl))

        conn.commit()
        logger.info("Tables created or updated successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {e}")
        raise
    finally:
        cur.close()

# -------------------------------------------------------------------
# 3. Insert Helpers
# -------------------------------------------------------------------
# Reuse the same POS_MAPPING dict from your migration:
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
    'baybayin': {'en': 'Baybayin Script', 'tl': 'Baybayin'},
    '': {'en': 'Uncategorized', 'tl': 'Hindi Tiyak'}
}

def entry_exists_and_identical(cur, lemma, normalized_lemma, language_code, entry_data):
    """
    Check if an entry already exists and is identical to avoid duplicate processing.
    """
    logger.debug(f"Checking if entry exists: {lemma}")
    
    try:
        # First check if word exists
        cur.execute("""
            SELECT w.id, w.tags, w.preferred_spelling, w.root_word_id
            FROM words w
            WHERE w.normalized_lemma = %s AND w.language_code = %s
         LIMIT 1
        """, (normalized_lemma, language_code))
        
        existing = cur.fetchone()
        if not existing:
            return False, None
        
        word_id = existing[0]
        
        # Check definitions
        cur.execute("""
            SELECT COUNT(*), array_agg(definition_text), array_agg(part_of_speech)
            FROM definitions
            WHERE word_id = %s
        """, (word_id,))
        def_info = cur.fetchone()
        
        # Check etymologies
        cur.execute("""
            SELECT COUNT(*), array_agg(original_text), array_agg(language_codes)
            FROM etymologies
            WHERE word_id = %s
        """, (word_id,))
        ety_info = cur.fetchone()
        
        # Check relations
        cur.execute("""
            SELECT COUNT(*), array_agg(relation_type)
            FROM relations
            WHERE from_word_id = %s OR to_word_id = %s
        """, (word_id, word_id))
        rel_info = cur.fetchone()
        
        # Compare with new data
        def compare_entry():
            # Compare definitions
            new_defs = entry_data.get('definitions', [])
            if def_info[0] != len(new_defs):
                logger.debug(f"Different number of definitions for {lemma}")
                return False
                
            # Compare etymologies
            new_ety = entry_data.get('etymology')
            if bool(new_ety) != bool(ety_info[0]):
                logger.debug(f"Etymology presence mismatch for {lemma}")
                return False
                
            # Compare basic metadata
            if entry_data.get('preferred_spelling') != existing[2]:
                logger.debug(f"Different preferred spelling for {lemma}")
                return False
                
            if entry_data.get('tags') != existing[1]:
                logger.debug(f"Different tags for {lemma}")
                return False
            
            logger.info(f"Entry {lemma} exists and is identical - skipping")
            return True
            
        return True, compare_entry()
        
    except Exception as e:
        logger.error(f"Error checking entry existence: {str(e)}")
        return False, None

def clean_baybayin_lemma(lemma: str) -> str:
    """Clean Baybayin lemma by removing any prefixes."""
    prefix = "Baybayin spelling of"
    if lemma.lower().startswith(prefix.lower()):
        return lemma[len(prefix):].strip()
    return lemma

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

    def __post_init__(self):
        """Validate the word entry data."""
        if not self.lemma or not isinstance(self.lemma, str):
            raise ValueError("Lemma must be a non-empty string")
        if not self.language_code in ('tl', 'ceb'):
            raise ValueError(f"Unsupported language code: {self.language_code}")
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")

def with_transaction(commit=True):
    """
    Improved transaction decorator with commit control and proper cleanup.
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
    """
    Comprehensive input validation for word data.
    """
    if not isinstance(data, dict):
        raise ValueError("Word data must be a dictionary")
        
    required_fields = ['lemma', 'language_code']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
            
    # Lemma validation
    if not data['lemma'] or not isinstance(data['lemma'], str):
        raise ValueError("Lemma must be a non-empty string")
    data['lemma'] = data['lemma'].strip()
    
    # Language code validation
    if data['language_code'] not in ('tl', 'ceb'):
        raise ValueError(f"Unsupported language code: {data['language_code']}")
        
    # Baybayin validation
    if data.get('has_baybayin'):
        if not data.get('baybayin_form'):
            raise ValueError("Baybayin form required when has_baybayin is True")
            
    return data

def validate_word_entry(
    lemma: str,
    language_code: str,
    root_word_id: Optional[int] = None,
    preferred_spelling: Optional[str] = None,
    tags: Optional[str] = None,
    has_baybayin: bool = False,
    baybayin_form: Optional[str] = None
) -> WordEntry:
    """
    Validate word entry data and create a WordEntry instance.
    
    Args:
        lemma: The word lemma
        language_code: Language code (tl or ceb)
        root_word_id: Optional ID of root word
        preferred_spelling: Optional preferred spelling
        tags: Optional tags
        has_baybayin: Whether entry has Baybayin form
        baybayin_form: Optional Baybayin form
        
    Returns:
        WordEntry instance with validated data
        
    Raises:
        ValueError: If validation fails
    """
    try:
        normalized_lemma = normalize_lemma(lemma)
        return WordEntry(
            lemma=lemma,
            normalized_lemma=normalized_lemma,
            language_code=language_code,
            root_word_id=root_word_id,
            preferred_spelling=preferred_spelling,
            tags=tags,
            has_baybayin=has_baybayin,
            baybayin_form=baybayin_form
        )
    except ValueError as e:
        raise ValueError(f"Word entry validation failed: {str(e)}")

@with_transaction(commit=True)
def get_or_create_word_id(
    cur: psycopg2.extensions.cursor,
    lemma: str,
    language_code: str = "tl",
    root_word_id: Optional[int] = None,
    preferred_spelling: Optional[str] = None,
    tags: Optional[str] = None,
    check_exists: bool = False,
    has_baybayin: bool = False,
    baybayin_form: Optional[str] = None,
    entry_data: Optional[Dict[str, Any]] = None
) -> int:
    """
    Get existing word ID or create new word entry with proper validation and error handling.
    
    Args:
        cur: Database cursor
        lemma: Word lemma
        language_code: Language code (default: "tl")
        root_word_id: Optional ID of root word
        preferred_spelling: Optional preferred spelling
        tags: Optional tags
        check_exists: Whether to check for existing identical entries
        has_baybayin: Whether entry has Baybayin form
        baybayin_form: Optional Baybayin form
        entry_data: Optional additional entry data for comparison
        
    Returns:
        Word ID (either existing or newly created)
        
    Raises:
        ValueError: If input validation fails
        psycopg2.Error: If database operation fails
    """
    try:
        # Validate input and create WordEntry
        word_entry = validate_word_entry(
            lemma=lemma,
            language_code=language_code,
            root_word_id=root_word_id,
            preferred_spelling=preferred_spelling,
            tags=tags,
            has_baybayin=has_baybayin,
            baybayin_form=baybayin_form
        )

        # Check for existing entry
        cur.execute("""
            SELECT id 
            FROM words 
            WHERE normalized_lemma = %s AND language_code = %s
            LIMIT 1
        """, (word_entry.normalized_lemma, word_entry.language_code))
        
        existing = cur.fetchone()
        
        if existing:
            existing_id = existing[0]
            
            # If entry exists and has Baybayin form, update it
            if word_entry.has_baybayin and word_entry.baybayin_form:
                cur.execute("""
                    UPDATE words
                    SET has_baybayin = TRUE,
                        baybayin_form = COALESCE(baybayin_form, %s)
                    WHERE id = %s
                """, (word_entry.baybayin_form, existing_id))
                
            # If checking for identical entries
            if check_exists and entry_data:
                is_identical, compare_result = entry_exists_and_identical(
                    cur, 
                    word_entry.lemma,
                    word_entry.normalized_lemma,
                    word_entry.language_code,
                    entry_data
                )
                if is_identical and compare_result:
                    logger.info(f"Skipping identical entry for {word_entry.lemma}")
                    return existing_id
                
            return existing_id

        # Insert new entry
        cur.execute("""
            INSERT INTO words (
                lemma, normalized_lemma, has_baybayin, baybayin_form,
                language_code, root_word_id, preferred_spelling, tags
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            word_entry.lemma,
            word_entry.normalized_lemma,
            word_entry.has_baybayin,
            word_entry.baybayin_form,
            word_entry.language_code,
            word_entry.root_word_id,
            word_entry.preferred_spelling,
            word_entry.tags
        ))
        
        new_id = cur.fetchone()[0]
        logger.debug(f"Created new word entry for {word_entry.lemma} with ID {new_id}")
        return new_id

    except UniqueViolation:
        # Handle race condition where entry was created between our check and insert
        cur.execute("""
            SELECT id 
            FROM words 
            WHERE normalized_lemma = %s AND language_code = %s
            LIMIT 1
        """, (word_entry.normalized_lemma, word_entry.language_code))
        return cur.fetchone()[0]
        
    except ValueError as e:
        logger.error(f"Validation error for lemma '{lemma}': {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error processing lemma '{lemma}': {str(e)}")
        raise

def batch_process_entries(cur, entries: List[Dict], batch_size: int = 1000):
    """
    Process entries in efficient batches with error recovery.
    """
    processed = []
    failed = []
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        
        try:
            # Prepare batch data
            values = [(
                entry['lemma'],
                normalize_lemma(entry['lemma']),
                entry.get('language_code', 'tl'),
                entry.get('root_word_id'),
                entry.get('has_baybayin', False),
                entry.get('baybayin_form')
            ) for entry in batch]
            
            # Bulk insert using executemany
            cur.executemany("""
                INSERT INTO words (
                    lemma, normalized_lemma, language_code,
                    root_word_id, has_baybayin, baybayin_form
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (language_code, normalized_lemma) 
                DO UPDATE SET 
                    has_baybayin = EXCLUDED.has_baybayin,
                    baybayin_form = COALESCE(words.baybayin_form, EXCLUDED.baybayin_form)
                RETURNING id
            """, values)
            
            processed.extend(cur.fetchall())
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            failed.extend(batch)
            
    return processed, failed

def batch_get_or_create_word_ids(
    cur: psycopg2.extensions.cursor,
    entries: list[tuple[str, str]],
    batch_size: int = 1000
) -> dict[tuple[str, str], int]:
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
            # Batch insert new entries
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO words (lemma, normalized_lemma, language_code)
                VALUES %s
                RETURNING lemma, language_code, id
                """,
                [(lemma, norm, lang) for lemma, norm, lang in to_insert]
            )
            
            # Add newly inserted entries to results
            for lemma, lang, id in cur.fetchall():
                existing[(lemma, lang)] = id
        
        result.update(existing)
    
    return result

def has_diacritics(text):
    """Check if text contains diacritical marks."""
    normalized = normalize_lemma(text)
    return text != normalized

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
    return cur.fetchone()[0]

def insert_definition(
    cur,
    word_id: int,
    definition_text: str,
    part_of_speech: str = "",
    examples: str = None,
    usage_notes: str = None,
    sources: str = ""
) -> int:
    """Insert a definition, skipping if it's a Baybayin alternative."""
    if 'Baybayin spelling of' in definition_text:
        return None
        
    std_pos_id = get_standardized_pos_id(cur, part_of_speech)
    
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

def insert_relation(cur,
                    from_word_id: int,
                    to_word_id: int,
                    relation_type: str,
                    sources=""):
    """
    Creates a row in 'relations'. For synonyms, 'relation_type' might be "synonym", etc.
    For derived forms, it might be "derived", "root_of", etc.
    """
    sql = """
    INSERT INTO relations
        (from_word_id, to_word_id, relation_type, sources)
    VALUES (%s, %s, %s, %s)
    """
    cur.execute(sql, (from_word_id, to_word_id, relation_type, sources))

def insert_etymology(cur, word_id, etymology_text, sources=""):
    """Process etymology with Baybayin handling."""
    if not etymology_text:
        return
        
    # Skip if it's just describing Baybayin spelling
    if 'Baybayin spelling of' in etymology_text:
        return
        
    codes, cleaned_ety = lsys.extract_and_remove_language_codes(etymology_text)
    
    cur.execute("""
        INSERT INTO etymologies
            (word_id, original_text, language_codes, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (word_id, original_text) DO UPDATE
        SET language_codes = EXCLUDED.language_codes,
            sources = COALESCE(etymologies.sources || ' | ' || EXCLUDED.sources, EXCLUDED.sources)
        RETURNING id
    """, (
        word_id,
        cleaned_ety,
        ", ".join(codes) if codes else "",
        sources
    ))

# -------------------------------------------------------------------
# BAYBAYIN PROCESSING
# -------------------------------------------------------------------

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

def cleanup_baybayin_data(cur):
    """Clean up problematic Baybayin data."""
    # Remove Baybayin flag from entries without Baybayin form
    cur.execute("""
        UPDATE words 
        SET has_baybayin = FALSE 
        WHERE has_baybayin = TRUE AND baybayin_form IS NULL
    """)
    
    # Merge duplicate Baybayin entries
    cur.execute("""
        WITH dupes AS (
            SELECT DISTINCT w1.id as id1, w2.id as id2
            FROM words w1
            JOIN words w2 ON w1.baybayin_form = w2.baybayin_form
            WHERE w1.has_baybayin AND w2.has_baybayin
            AND w1.id < w2.id
        )
        SELECT id1, id2 FROM dupes
    """)
    
    for word1_id, word2_id in cur.fetchall():
        logger.info(f"Merging duplicate Baybayin entries: {word1_id} and {word2_id}")
        cur.execute("""
            UPDATE definitions SET word_id = %s WHERE word_id = %s;
            UPDATE relations SET from_word_id = %s WHERE from_word_id = %s;
            UPDATE relations SET to_word_id = %s WHERE to_word_id = %s;
            DELETE FROM words WHERE id = %s;
        """, (word1_id, word2_id, word1_id, word2_id, word1_id, word2_id, word2_id))

class BaybayinCharType(Enum):
    """Enum for different types of Baybayin characters"""
    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"

@dataclass
class BaybayinChar:
    """Represents a single Baybayin character with its properties"""
    char: str
    char_type: BaybayinCharType
    default_sound: str
    possible_sounds: List[str]
    
    def __post_init__(self):
        if not self.char:
            raise ValueError("Character cannot be empty")
        if not isinstance(self.char_type, BaybayinCharType):
            raise ValueError(f"Invalid character type: {self.char_type}")

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
   
# -------------------------------------------------------------------
# 4. Processing Tagalog Words (New)
# -------------------------------------------------------------------
def process_tagalog_words_file(cur, filename, check_exists=False):
    """Process Tagalog dictionary entries from a JSON file."""
    logger.info(f"Starting to process Tagalog words file: {filename}")
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    src = os.path.basename(filename)
    logger.info(f"Using source identifier: {src}")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON data with {len(data)} entries")

            for lemma, info in tqdm(data.items(), desc="Processing Tagalog words"):
                try:
                    logger.debug(f"Processing lemma: {lemma}")
                    word_id = get_or_create_word_id(cur, lemma, "tl", check_exists=check_exists)
                    logger.debug(f"Got word_id: {word_id} for lemma: {lemma}")

                    part_of_speech = info.get("part_of_speech", "").strip()
                    ety = info.get("etymology", "").strip()
                    derivative = info.get("derivative", "").strip()
                    defs_list = info.get("definitions", [])

                    if ety:
                        logger.debug(f"Processing etymology for {lemma}")
                        codes, cleaned_ety = lsys.extract_and_remove_language_codes(ety)
                        insert_etymology(
                            cur,
                            word_id,
                            original_text=cleaned_ety,
                            language_codes=", ".join(codes) if codes else "",
                            sources=src,
                        )

                    logger.debug(f"Processing {len(defs_list)} definitions for {lemma}")
                    for def_txt in defs_list:
                        insert_definition(
                            cur,
                            word_id,
                            def_txt.strip(),
                            part_of_speech,
                            examples=None,
                            usage_notes=None,
                            sources=src,
                        )

                    if derivative:
                        logger.debug(f"Processing derivatives for {lemma}: {derivative}")
                        possible_derivatives = [d.strip() for d in derivative.split(",")]
                        for der_lemma in possible_derivatives:
                            if der_lemma:
                                der_id = get_or_create_word_id(cur, der_lemma, "tl")
                                insert_relation(cur, word_id, der_id, "derivative", src)

                except Exception as e:
                    logger.error(f"Error processing lemma {lemma}: {str(e)}")
                    continue

            logger.info(f"Completed processing Tagalog words file: {filename}")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise

# -------------------------------------------------------------------
# 5. Processing Root Words
# -------------------------------------------------------------------
def process_root_words_file(cur, filename, check_exists=False):
    """Process root words and their derivatives from a JSON file."""
    logger.info(f"Starting to process Root Words file: {filename}")
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    src = os.path.basename(filename)
    logger.info(f"Using source identifier: {src}")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON data with {len(data)} entries")

            for root_lemma, assoc_dict in tqdm(data.items(), desc="Processing root words"):
                try:
                    logger.debug(f"Processing root lemma: {root_lemma}")
                    root_id = get_or_create_word_id(cur, root_lemma, "tl", check_exists=check_exists)
                    logger.debug(f"Got root_id: {root_id} for lemma: {root_lemma}")

                    for derived_lemma, details in assoc_dict.items():
                        logger.debug(f"Processing derived lemma: {derived_lemma}")
                        derived_id = get_or_create_word_id(
                            cur, derived_lemma, "tl", root_word_id=root_id
                        )

                        definition_txt = details.get("definition", "")
                        part_of_speech = details.get("type", "")

                        if definition_txt:
                            logger.debug(f"Inserting definition for {derived_lemma}")
                            insert_definition(
                                cur,
                                derived_id,
                                definition_txt,
                                part_of_speech,
                                examples=None,
                                usage_notes=None,
                                sources=src,
                            )

                        logger.debug(f"Creating root_of relation: {derived_lemma} -> {root_lemma}")
                        insert_relation(cur, derived_id, root_id, "root_of", src)

                except Exception as e:
                    logger.error(f"Error processing root lemma {root_lemma}: {str(e)}")
                    continue

            logger.info(f"Completed processing Root Words file: {filename}")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise

# -------------------------------------------------------------------
# 6. KWF Dictionary
# -------------------------------------------------------------------
def process_kwf_file(cur, filename, check_exists=False):
    """Comprehensive processing of KWF Dictionary with verbose logging."""
    logger.info(f"Starting to process KWF Dictionary: {filename}")
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    src = os.path.basename(filename)
    logger.info(f"Using source identifier: {src}")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded KWF data with {len(data)} entries")

            for lemma, entry in tqdm(data.items(), desc="Processing KWF entries"):
                try:
                    logger.debug(f"Processing entry: {lemma}")
                    formatted_lemma = entry.get("formatted", lemma)

                    # Prepare entry data for comparison
                    entry_data = {
                        'preferred_spelling': formatted_lemma if formatted_lemma != lemma else None,
                        'etymology': entry.get('metadata', {}).get('etymology'),
                        'definitions': [
                            {'text': d.get('meaning'), 'pos': pos}
                            for pos, defs in entry.get('definitions', {}).items()
                            for d in defs
                        ],
                        'tags': entry.get('metadata', {}).get('pronunciation'),
                    }

                    word_id = get_or_create_word_id(
                        cur,
                        formatted_lemma,
                        "tl",
                        entry_data=entry_data,
                        check_exists=check_exists
                    )

                    # Process metadata
                    meta = entry.get("metadata", {})
                    ety_text = meta.get("etymology")
                    source_lang = meta.get("source_language")

                    if ety_text or source_lang:
                        combined_ety = []
                        if source_lang:
                            combined_ety.append(f"From {source_lang}")
                        if ety_text:
                            combined_ety.append(ety_text)

                        full_ety = ": ".join(combined_ety)
                        if full_ety:
                            logger.debug(f"Processing etymology for {lemma}: {full_ety}")
                            codes, cleaned_ety = lsys.extract_and_remove_language_codes(full_ety)
                            insert_etymology(
                                cur,
                                word_id,
                                original_text=cleaned_ety,
                                language_codes=", ".join(codes) if codes else "",
                                sources=src
                            )

                    pronunciation = meta.get("pronunciation")
                    if pronunciation:
                        logger.debug(f"Storing pronunciation for {lemma}")
                        cur.execute("""
                            UPDATE words 
                            SET tags = CASE 
                                WHEN tags IS NULL THEN %s
                                ELSE tags || E'\n' || %s
                            END
                            WHERE id = %s
                        """, (f"pronunciation: {pronunciation}", f"pronunciation: {pronunciation}", word_id))

                    defs_dict = entry.get("definitions", {})
                    for pos_label, def_list in defs_dict.items():
                        def_list_sorted = sorted(def_list,
                                                 key=lambda x: int(x["meaning"].split('.')[0])
                                                 if x["meaning"].strip().split('.')[0].isdigit()
                                                 else float('inf'))

                        for def_obj in def_list_sorted:
                            category = def_obj.get("category", "").strip()
                            meaning = def_obj.get("meaning", "").strip()
                            synonyms = def_obj.get("synonyms", [])

                            if meaning:
                                full_def = f"[{category}] {meaning}" if category else meaning
                                logger.debug(f"Inserting definition for {lemma}: {full_def}")
                                insert_definition(
                                    cur,
                                    word_id,
                                    definition_text=full_def,
                                    part_of_speech=pos_label,
                                    examples=None,
                                    usage_notes=category,
                                    sources=src
                                )

                            for syn_lemma in synonyms:
                                syn_lemma_clean = syn_lemma.strip()
                                if syn_lemma_clean:
                                    logger.debug(f"Processing synonym for {lemma}: {syn_lemma_clean}")
                                    syn_id = get_or_create_word_id(cur, syn_lemma_clean, "tl")
                                    insert_relation(cur, word_id, syn_id, "synonym", src)

                    related = entry.get("related", {})
                    related_terms = related.get("related_terms", [])
                    for term_obj in related_terms:
                        term = term_obj.get("term", "").strip()
                        if term:
                            logger.debug(f"Processing related term for {lemma}: {term}")
                            term_id = get_or_create_word_id(cur, term, "tl")
                            insert_relation(cur, word_id, term_id, "related", src)

                except Exception as e:
                    logger.error(f"Error processing KWF entry {lemma}: {str(e)}")
                    continue

            logger.info("Completed processing KWF Dictionary")
    except Exception as e:
        logger.error(f"Error processing KWF file: {str(e)}")
        raise

# -------------------------------------------------------------------
# MIGRATE SUBCOMMAND
# -------------------------------------------------------------------
 
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
    
    try:
        # List of required extensions
        extensions = [
            'pg_trgm',      # For similarity search and fuzzy matching
            'unaccent',     # For handling diacritics
            'fuzzystrmatch' # For levenshtein distance
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

def migrate_data(args):
    """Migrate data with proper Baybayin handling."""
    logger.info("Starting fresh data migration")
    
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Drop existing tables
        cur.execute("""
            DROP TABLE IF EXISTS 
                etymologies, relations, definitions, words CASCADE;
        """)
        conn.commit()
        
        # Create fresh tables with Baybayin support
        create_or_update_tables(conn)
        
        # Process each source file
        source_files = [
            ("Processing Tagalog words", "data/tagalog-words.json", process_tagalog_words_file),
            ("Processing Root words", "data/root_words_with_associated_words_cleaned.json", process_root_words_file),
            ("Processing KWF Dictionary", "data/kwf_dictionary.json", process_kwf_file),
            ("Processing Kaikki (Tagalog)", "data/kaikki.jsonl", process_kaikki_jsonl_new),
            ("Processing Kaikki (Cebuano)", "data/kaikki-ceb.jsonl", process_kaikki_jsonl_new)
        ]
        
        for message, filename, processor in source_files:
            logger.info(message)
            processor(cur, filename, args.check_exists)
            conn.commit()
        
        # Verify and clean up Baybayin data
        verify_baybayin_data(cur)
        cleanup_baybayin_data(cur)
        conn.commit()
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------
# VERIFY SUBCOMMAND
# -------------------------------------------------------------------
def verify_database(args):
    """
    Verify database integrity with optional quick test mode.
    
    Args:
        args: Command line arguments including optional --quick flag
    """
    logger.info("Starting database verification")
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Quick test section (from former test_database)
        tables = ["words", "definitions", "relations", "etymologies"]
        logger.info("Checking table counts...")
        
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            logger.info(f"Table {t} count = {count}")

        if args.quick:
            logger.info("Running in quick test mode...")
            logger.info("Sampling 5 entries from 'words' table:")
            cur.execute("""
                SELECT id, lemma, language_code, root_word_id 
                FROM words 
                LIMIT 5
            """)
            rows = cur.fetchall()
            for r in rows:
                logger.info(f"Sample entry: {r}")
            return

        # Full verification section (original verify_database)
        logger.info("Running full verification checks...")
        
        # Check for duplicates in words
        logger.info("Checking for duplicate entries in words table...")
        cur.execute("""
            SELECT language_code, normalized_lemma, COUNT(*)
              FROM words
             GROUP BY language_code, normalized_lemma
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()
        if duplicates:
            logger.warning("Found duplicates in (lang_code, normalized_lemma):")
            for d in duplicates:
                logger.warning(f"  {d}")
        else:
            logger.info("No duplicates found in words table")

        # Missing references in relations
        logger.info("Checking for missing references in relations table...")
        cur.execute("""
            SELECT r.id
              FROM relations r
              LEFT JOIN words w1 ON r.from_word_id = w1.id
              LEFT JOIN words w2 ON r.to_word_id = w2.id
             WHERE w1.id IS NULL OR w2.id IS NULL
        """)
        missing_refs = cur.fetchall()
        if missing_refs:
            logger.warning("Relations with missing word references:")
            for mr in missing_refs:
                logger.warning(f"  relation_id: {mr[0]}")
        else:
            logger.info("No missing references in relations table")

        # Add more thorough checks
        logger.info("Checking definition integrity...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM definitions 
            WHERE definition_text IS NULL OR definition_text = ''
        """)
        empty_defs = cur.fetchone()[0]
        if empty_defs:
            logger.warning(f"Found {empty_defs} empty definitions")
            
        logger.info("Checking etymology integrity...")
        cur.execute("""
            SELECT COUNT(*) 
            FROM etymologies 
            WHERE original_text IS NULL OR original_text = ''
        """)
        empty_etys = cur.fetchone()[0]
        if empty_etys:
            logger.warning(f"Found {empty_etys} empty etymologies")

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("Database verification completed")
# -------------------------------------------------------------------
# UPDATE SUBCOMMAND
# -------------------------------------------------------------------
def update_database(args):
    """
    For small incremental updates from a new JSON file:
    We read the JSON, and for each lemma, we might add definitions or synonyms, etc.
    """
    update_file = args.file
    if not update_file:
        print("Please provide a --file <update.json> for 'update' subcommand.")
        sys.exit(1)
    if not os.path.exists(update_file):
        print(f"File not found: {update_file}")
        sys.exit(1)

    print(f"Updating DB with data from: {update_file}")
    conn = get_connection()
    cur = conn.cursor()
    try:
        with open(update_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Sample structure:
        # {
        #   "bagong_salita": {
        #       "definitions": ["A new definition."],
        #       "synonyms": ["kasingkahulugan_something"],
        #       "root_word": "ugat_lemma"
        #   }
        # }
        src = os.path.basename(update_file)
        for lemma, entry in data.items():
            word_id = get_or_create_word_id(cur, lemma, "tl")
            # Insert definitions
            for def_txt in entry.get("definitions", []):
                insert_definition(cur, word_id, def_txt, "", None, None, src)
            # Insert synonyms
            for syn_lemma in entry.get("synonyms", []):
                syn_id = get_or_create_word_id(cur, syn_lemma, "tl")
                insert_relation(cur, word_id, syn_id, "synonym", src)
            # Possibly set root word
            if "root_word" in entry:
                root_lemma = entry["root_word"]
                root_id = get_or_create_word_id(cur, root_lemma, "tl")
                upd_sql = "UPDATE words SET root_word_id=%s WHERE id=%s"
                cur.execute(upd_sql, (root_id, word_id))
        conn.commit()
        print("Database updated successfully.")
    except Exception as e:
        conn.rollback()
        print("Error during update:", e)
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------
# PURGE SUBCOMMAND
# -------------------------------------------------------------------
def purge_database(args):
    """
    Safely delete all data while preserving schema. Handles Baybayin and language codes.
    """
    logger.warning("Starting database purge process")
    
    if not args.force:
        console = Console()
        console.print("\n[red]WARNING:[/] This will delete ALL data including:", style="bold")
        console.print("• Word entries and definitions")
        console.print("• Relationships and etymologies")
        console.print("• Baybayin romanizations")
        console.print("• Language code mappings")
        console.print("• All source attributions")
        
        confirmation = input("\nType 'YES' to confirm: ")
        if confirmation != "YES":
            logger.info("Purge cancelled")
            return

    conn = get_connection()
    cur = conn.cursor()
    
    try:
        logger.info("Disabling triggers...")
        cur.execute("SET session_replication_role = 'replica';")
        
        # Delete data in the correct order
        logger.info("Deleting data from all tables...")
        tables = [
            'etymologies',  # Contains language codes
            'relations',
            'definitions',
            'words'  # Contains Baybayin data
        ]
        
        for table in tables:
            logger.info(f"Purging table: {table}")
            cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
            
        # Reset sequences
        logger.info("Resetting sequences...")
        for table in tables:
            cur.execute(f"ALTER SEQUENCE {table}_id_seq RESTART WITH 1;")
            
        # Check if Baybayin columns exist before trying to reset them
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words' 
            AND column_name IN ('romanized_form', 'has_baybayin')
        """)
        existing_columns = [col[0] for col in cur.fetchall()]
        
        if existing_columns:
            logger.info("Resetting Baybayin-related columns...")
            if 'romanized_form' in existing_columns:
                cur.execute("ALTER TABLE words ALTER COLUMN romanized_form DROP NOT NULL")
            if 'has_baybayin' in existing_columns:
                cur.execute("ALTER TABLE words ALTER COLUMN has_baybayin SET DEFAULT FALSE")
        
        # Check if language_codes column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'etymologies' 
            AND column_name = 'language_codes'
        """)
        if cur.fetchone():
            logger.info("Updating language codes documentation...")
            cur.execute("""
                COMMENT ON COLUMN etymologies.language_codes IS 'Standardized language codes'
            """)
        
        logger.info("Re-enabling triggers...")
        cur.execute("SET session_replication_role = 'origin';")
        
        conn.commit()
        logger.info("Database purge completed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during database purge: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("Database connection closed")

# -------------------------------------------------------------------
# LOOKUP SUBCOMMAND
# -------------------------------------------------------------------
def display_word_info(cur, word_id, lemma):
    """
    Display comprehensive information about a word entry.
    """
    console = Console()

    try:
        # Query for word information, including POS names from the parts_of_speech table
        cur.execute("""
            SELECT 
                w.lemma,
                w.language_code,
                w.romanized_form,
                w.has_baybayin,
                w.tags,
                ARRAY_AGG(DISTINCT d.definition_text) FILTER (WHERE d.definition_text IS NOT NULL) AS definitions,
                STRING_AGG(DISTINCT p.name_en, ', ') FILTER (WHERE p.name_en IS NOT NULL) AS pos_names,
                ARRAY_AGG(DISTINCT e.original_text) FILTER (WHERE e.original_text IS NOT NULL) AS etymologies,
                ARRAY_AGG(DISTINCT e.language_codes) FILTER (WHERE e.language_codes IS NOT NULL) AS etymology_langs,
                STRING_AGG(DISTINCT e.sources, ', ') FILTER (WHERE e.sources IS NOT NULL) AS etymology_sources
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            LEFT JOIN etymologies e ON w.id = e.word_id
            WHERE w.id = %s
            GROUP BY w.id, w.lemma, w.language_code, w.romanized_form, w.has_baybayin, w.tags
        """, (word_id,))
        
        word_info = cur.fetchone()

        if not word_info:
            console.print(f"[red]Word with ID {word_id} not found.[/]")
            return

        # Build the word header
        header = Text()
        header.append(f"📚 ", style="bold yellow")
        header.append(format_word_display(lemma), style="bold cyan")
        lang_display = {'tl': 'Tagalog', 'ceb': 'Cebuano'}.get(word_info[1], word_info[1])
        header.append(f"\nLanguage: {lang_display}", style="dim")
        if word_info[2]:
            header.append(f"\nPreferred Form: {word_info[2]}", style="italic cyan")
        header.append(f"\nDefinitions: {len(word_info[5])}", style="green")
        header.append(f" • Parts of Speech: {word_info[6] or '-'}", style="green")
        
        console.print(Panel(header, title="Word Information", border_style="cyan", padding=(1, 2)))

        # Etymology Section
        if word_info[7]:
            etymology_panel = Text()
            etymology_panel.append("Etymology: ", style="bold yellow")
            etymology_panel.append(", ".join(word_info[7]), style="white")
            if word_info[8]:
                etymology_panel.append("\nLanguages: ", style="bold yellow")
                etymology_panel.append(" → ".join(word_info[8]), style="cyan")
            if word_info[9]:
                etymology_panel.append("\nSources: ", style="bold yellow")
                etymology_panel.append(word_info[9], style="dim")
            console.print(Panel(etymology_panel, title="Etymology", border_style="yellow", padding=(1, 2)))

        # Definitions grouped by part of speech
        cur.execute("""
            SELECT 
                p.name_en AS pos_name,
                d.definition_text,
                d.usage_notes,
                d.examples,
                d.sources
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_en, d.id
        """, (word_id,))
        definitions = cur.fetchall()

        if definitions:
            current_pos = None
            def_table = None

            for pos, definition, usage, examples, source in definitions:
                if pos != current_pos:
                    if def_table:
                        console.print(Panel(def_table, title=f"Definitions ({current_pos})", border_style="magenta", padding=(1, 2)))
                    def_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, show_lines=True)
                    def_table.add_column("№", style="dim", width=4)
                    def_table.add_column("Definition", style="white", width=50)
                    def_table.add_column("Usage", style="yellow", width=20)
                    def_table.add_column("Source", style="dim", width=20)
                    current_pos = pos

                def_row = [
                    f"#{len(def_table.rows) + 1}",
                    definition or "-",
                    usage or "-",
                    source or "-"
                ]
                def_table.add_row(*def_row)

            if def_table:
                console.print(Panel(def_table, title=f"Definitions ({current_pos})", border_style="magenta", padding=(1, 2)))

    except Exception as e:
        console.print(f"[red]Error processing word information: {e}[/]")
    finally:
        cur.close()


def lookup_word(args):
    """
    Look up comprehensive information about a word with optional detailed view.
    
    Args:
        args: Command line arguments including word to look up and optional --detailed flag
    """
    if not args.word:
        logger.error("Please provide a word to look up")
        return
    
    conn = get_connection()
    cur = conn.cursor()
    console = Console()
    
    try:
        normalized_search = normalize_lemma(args.word)
        
        # Enhanced query to show more details about matches
        cur.execute(f"""
            SELECT 
                w.id, 
                w.lemma, 
                w.normalized_lemma,
                w.preferred_spelling,
                w.language_code,
                COUNT(d.id) as def_count,
                STRING_AGG(DISTINCT {get_standardized_source_sql()}, ', ') as sources,
                w.tags,
                w.root_word_id,
                w.romanized_form,
                w.has_baybayin
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            WHERE w.normalized_lemma = %s
            GROUP BY w.id, w.lemma, w.normalized_lemma, w.preferred_spelling, 
                     w.language_code, w.tags, w.root_word_id, w.romanized_form,
                     w.has_baybayin
            ORDER BY w.lemma
        """, (normalized_search,))
        
        exact_matches = cur.fetchall()
        
        if exact_matches:
            if len(exact_matches) > 1:
                console.print("\n[yellow]Multiple forms found:[/]")
                
                if args.detailed:
                    # Detailed technical view (former inspect_word_entries style)
                    for entry in exact_matches:
                        panel = Panel(
                            Text.from_markup(f"""
                                [bold]Word ID:[/] {entry[0]}
                                [bold]Lemma:[/] {entry[1]}
                                [bold]Normalized:[/] {entry[2]}
                                [bold]Preferred Spelling:[/] {entry[3] or '-'}
                                [bold]Language:[/] {entry[4]}
                                [bold]Definition Count:[/] {entry[5]}
                                [bold]Sources:[/] {entry[6] or '-'}
                                [bold]Tags:[/] {entry[7] or '-'}
                                [bold]Root Word ID:[/] {entry[8] or '-'}
                                [bold]Romanized Form:[/] {entry[9] or '-'}
                                [bold]Has Baybayin:[/] {entry[10]}
                            """.strip()),
                            title=f"Entry: {entry[1]}",
                            border_style="cyan"
                        )
                        console.print(panel)
                        console.print()
                else:
                    # User-friendly view (original lookup style)
                    console.print("[dim]ID | Word | Language | Definitions | Sources[/]")
                    for i, entry in enumerate(exact_matches, 1):
                        lang_display = {
                            'tl': 'Tagalog',
                            'ceb': 'Cebuano'
                        }.get(entry[4], entry[4])
                        
                        console.print(f"  {i}. [cyan]{entry[1]}[/] ({lang_display})")
                        console.print(f"     ID: {entry[0]}")
                        console.print(f"     Definitions: {entry[5]}")
                        console.print(f"     Sources: {entry[6]}")
                        if entry[3]:  # preferred_spelling
                            console.print(f"     Preferred: {entry[3]}")
                        console.print()
                
                # Let user choose which entry to view
                while True:
                    choice = input("\nEnter number to view details (or 'all' to view all): ").strip()
                    if choice.lower() == 'all':
                        for word_id, lemma, *_ in exact_matches:
                            console.print("\n[bold yellow]═══════════════════════════════════[/]")
                            display_word_info(cur, word_id, lemma)
                        break
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(exact_matches):
                            display_word_info(cur, exact_matches[idx][0], exact_matches[idx][1])
                            break
                        else:
                            console.print("[red]Invalid number. Try again.[/]")
                    except ValueError:
                        console.print("[red]Please enter a valid number or 'all'[/]")
            else:
                # Single exact match
                display_word_info(cur, exact_matches[0][0], exact_matches[0][1])
                
        else:
            # No exact matches, try fuzzy matching
            cur.execute("""
                WITH similarities AS (
                    SELECT id, 
                           lemma, 
                           normalized_lemma,
                           similarity(normalized_lemma, %s) as sim_score,
                           levenshtein(normalized_lemma, %s) as lev_distance
                    FROM words
                    WHERE normalized_lemma ILIKE '%%' || %s || '%%'
                       OR %s ILIKE '%%' || normalized_lemma || '%%'
                       OR levenshtein(normalized_lemma, %s) <= 2
                )
                SELECT id, lemma, normalized_lemma, sim_score
                FROM similarities
                WHERE sim_score > 0.3
                ORDER BY sim_score DESC, lev_distance ASC
                LIMIT 10
            """, (normalized_search, normalized_search, normalized_search, 
                  normalized_search, normalized_search))
            
            suggestions = cur.fetchall()
            
            if suggestions:
                console.print("\n[yellow]Word not found. Did you mean one of these?[/]")
                console.print("[grey70]Enter the number to view details:[/]\n")
                
                # Group suggestions by normalized form
                grouped_suggestions = {}
                for word_id, lemma, norm_lemma, score in suggestions:
                    if norm_lemma not in grouped_suggestions:
                        grouped_suggestions[norm_lemma] = []
                    grouped_suggestions[norm_lemma].append((word_id, lemma, score))
                
                # Display grouped suggestions
                suggestion_map = {}  # To map numbers to word_ids
                counter = 1
                
                for norm_lemma, variants in grouped_suggestions.items():
                    if len(variants) > 1:
                        console.print(f"[cyan]{norm_lemma}[/] variants:")
                        for word_id, lemma, score in variants:
                            console.print(f"  {counter}. [green]{lemma}[/]")
                            suggestion_map[counter] = (word_id, lemma)
                            counter += 1
                        console.print()
                    else:
                        word_id, lemma, score = variants[0]
                        console.print(f"{counter}. [green]{lemma}[/]")
                        suggestion_map[counter] = (word_id, lemma)
                        counter += 1
                
                # Let user choose from suggestions
                while True:
                    choice = input("\nEnter number (or 'q' to quit): ").strip()
                    if choice.lower() == 'q':
                        break
                    try:
                        num = int(choice)
                        if num in suggestion_map:
                            word_id, lemma = suggestion_map[num]
                            display_word_info(cur, word_id, lemma)
                            break
                        else:
                            console.print("[red]Invalid number. Try again.[/]")
                    except ValueError:
                        console.print("[red]Please enter a valid number or 'q'[/]")
            else:
                logger.error(f"No matches found for '{args.word}'")
                return
        
    except Exception as e:
        logger.error(f"Error during word lookup: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------
# STATS SUBCOMMAND
# -------------------------------------------------------------------

def display_dictionary_stats(args):
    """Display essential dictionary statistics."""
    console = Console()
    console.print("\n[bold cyan]📊 Dictionary Statistics[/]\n")
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        stats = {}
        
        # Basic counts - fast query
        cur.execute("SELECT COUNT(*) FROM words")
        stats['total_words'] = cur.fetchone()[0]
        
        # Language distribution - optimized query
        cur.execute("""
            SELECT 
                w.language_code,
                COUNT(DISTINCT w.id) as word_count,
                COUNT(DISTINCT d.id) as def_count,
                ROUND(AVG(CASE WHEN d.id IS NOT NULL THEN 1 ELSE 0 END)::numeric * 100, 1) as pct_with_defs
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            GROUP BY w.language_code
            ORDER BY word_count DESC
        """)
        stats['languages'] = cur.fetchall()
        
        # Create language distribution panel
        lang_table = Table(title="📚 Language Distribution", box=box.ROUNDED, border_style="cyan")
        lang_table.add_column("Language", style="bold yellow")
        lang_table.add_column("Words", justify="right", style="cyan")
        lang_table.add_column("Definitions", justify="right", style="green")
        lang_table.add_column("% With Defs", justify="right", style="magenta")
        
        total_words = sum(lang[1] for lang in stats['languages'])
        
        for lang_code, words, defs, pct_defs in stats['languages']:
            lang_display = {'tl': 'Tagalog', 'ceb': 'Cebuano'}.get(lang_code, lang_code)
            percentage = (words / total_words) * 100
            
            lang_table.add_row(
                f"{lang_display} ({percentage:.1f}%)",
                f"{words:,}",
                f"{defs:,}",
                f"{pct_defs}%"
            )
        
        # Add totals row
        lang_table.add_row(
            "[bold]Total[/]",
            f"[bold]{total_words:,}[/]",
            f"[bold]{sum(lang[2] for lang in stats['languages']):,}[/]",
            f"[bold]{sum(lang[1] * lang[3] for lang in stats['languages']) / total_words:.1f}%[/]"
        )
        
        console.print(Panel(lang_table, border_style="cyan", padding=(1, 2)))
        console.print()
        
        # Source distribution - simplified query
        cur.execute(f"""
            SELECT {get_standardized_source_sql()} as source,
                   COUNT(*) as count,
                   ROUND(COUNT(*)::numeric / 
                       (SELECT COUNT(*)::numeric FROM definitions) * 100, 1) as percentage
            FROM definitions 
            GROUP BY sources 
            ORDER BY count DESC
        """)
        stats['sources'] = cur.fetchall()
        
        # Source distribution table
        source_table = Table(title="Source Distribution", box=box.ROUNDED, border_style="yellow")
        source_table.add_column("Source", style="bold yellow")
        source_table.add_column("Definitions", justify="right", style="cyan")
        source_table.add_column("%", justify="right", style="magenta")
        
        for source, count, pct in stats['sources']:
            source_table.add_row(
                source,
                f"{count:,}",
                f"{pct}%"
            )
        
        # Quick stats table
        quick_stats = Table(show_header=False, box=box.ROUNDED, border_style="blue")
        quick_stats.add_column("Metric", style="bold yellow")
        quick_stats.add_column("Value", style="cyan")
        
        # Basic counts - fast queries
        cur.execute("SELECT COUNT(*) FROM definitions")
        total_defs = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM relations")
        total_rels = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM etymologies")
        total_etys = cur.fetchone()[0]
        
        quick_stats.add_row("Total Words", f"{stats['total_words']:,}")
        quick_stats.add_row("Total Definitions", f"{total_defs:,}")
        quick_stats.add_row("Total Relations", f"{total_rels:,}")
        quick_stats.add_row("Total Etymologies", f"{total_etys:,}")
        
        # Layout everything
        console.print(Panel(quick_stats, title="📊 Quick Statistics", border_style="cyan"))
        console.print()
        console.print(source_table)
        console.print()
        
        # Add Baybayin statistics
        cur.execute(r"""
            WITH baybayin_stats AS (
                SELECT 
                    COUNT(*) as total_baybayin,
                    SUM(CASE 
                        WHEN normalized_lemma NOT IN (
                            SELECT normalized_lemma 
                            FROM words w2 
                            WHERE NOT (w2.lemma ~ '[\u1700-\u171F]')
                        ) THEN 1 
                        ELSE 0 
                    END) as standalone_count
                FROM words
                WHERE lemma ~ '[\u1700-\u171F]'
            )
            SELECT 
                total_baybayin,
                standalone_count,
                CAST(
                    (standalone_count::float * 100 / NULLIF(total_baybayin, 0)) 
                    AS NUMERIC(10,1)
                ) as standalone_pct
            FROM baybayin_stats
        """)
        baybayin_stats = cur.fetchone()
        
        if baybayin_stats and baybayin_stats[0] > 0:
            baybayin_table = Table(title="📜 Baybayin Statistics", box=box.ROUNDED, border_style="magenta")
            baybayin_table.add_column("Metric", style="bold yellow")
            baybayin_table.add_column("Count", justify="right", style="cyan")
            baybayin_table.add_column("Percentage", justify="right", style="green")
            
            total_baybayin = baybayin_stats[0]
            standalone = baybayin_stats[1]
            standalone_pct = baybayin_stats[2]
            
            baybayin_table.add_row(
                "Total Baybayin Words",
                f"{total_baybayin:,}",
                f"{(total_baybayin / stats['total_words'] * 100):.1f}% of all words"
            )
            baybayin_table.add_row(
                "Standalone Baybayin",
                f"{standalone:,}",
                f"{standalone_pct}% of Baybayin words"
            )
            baybayin_table.add_row(
                "With Romanized Pairs",
                f"{total_baybayin - standalone:,}",
                f"{100 - standalone_pct:.1f}% of Baybayin words"
            )
            
            console.print(Panel(baybayin_table, border_style="magenta", padding=(1, 2)))
            console.print()
        
    except Exception as e:
        logger.error(f"Error gathering statistics: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------
# LEADERBOARD SUBCOMMAND
# -------------------------------------------------------------------

def display_leaderboard(args):
    """
    Display various Top 10 rankings from the dictionary, leveraging standardized_pos_id.
    """
    console = Console()
    console.print("\n[bold cyan]📊 Dictionary Leaderboards[/]\n")
    
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Most Defined Words (by language and standardized_pos_id)
        cur.execute("""
            SELECT 
                w.lemma, 
                w.language_code, 
                COUNT(DISTINCT d.id) as def_count,
                STRING_AGG(DISTINCT p.name_en, ', ') as pos_list
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY w.lemma, w.language_code
            ORDER BY def_count DESC
            LIMIT 10
        """)
        most_defined = cur.fetchall()
        
        # Most Connected Words (by relations)
        cur.execute("""
            WITH relation_counts AS (
                SELECT w.lemma, w.language_code,
                       COUNT(DISTINCT r.id) as rel_count,
                       STRING_AGG(DISTINCT r.relation_type, ', ') as rel_types
                FROM words w
                LEFT JOIN relations r ON w.id = r.from_word_id OR w.id = r.to_word_id
                GROUP BY w.lemma, w.language_code
                ORDER BY rel_count DESC
                LIMIT 10
            )
            SELECT * FROM relation_counts
        """)
        most_connected = cur.fetchall()
        
        # Words with Most Examples
        cur.execute("""
            SELECT w.lemma, w.language_code,
                   COUNT(DISTINCT CASE WHEN d.examples IS NOT NULL THEN d.id END) as example_count
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            GROUP BY w.lemma, w.language_code
            ORDER BY example_count DESC
            LIMIT 10
        """)
        most_examples = cur.fetchall()
        
        # Most Common Root Words
        cur.execute("""
            SELECT w.lemma, w.language_code, COUNT(*) as derived_count
            FROM words w
            LEFT JOIN words derived ON derived.root_word_id = w.id
            GROUP BY w.lemma, w.language_code
            ORDER BY derived_count DESC
            LIMIT 10
        """)
        common_roots = cur.fetchall()
        
        # Words with Most Synonyms
        cur.execute("""
            SELECT w.lemma, w.language_code, COUNT(DISTINCT r.id) as synonym_count
            FROM words w
            LEFT JOIN relations r ON (w.id = r.from_word_id OR w.id = r.to_word_id)
            WHERE r.relation_type = 'synonym'
            GROUP BY w.lemma, w.language_code
            ORDER BY synonym_count DESC
            LIMIT 10
        """)
        most_synonyms = cur.fetchall()
        
        # Words with Richest Etymology
        cur.execute("""
            SELECT w.lemma, w.language_code,
                   COUNT(DISTINCT e.id) as ety_count,
                   STRING_AGG(DISTINCT NULLIF(TRIM(e.language_codes), ''), ', ') as lang_codes
            FROM words w
            LEFT JOIN etymologies e ON w.id = e.word_id
            GROUP BY w.lemma, w.language_code
            ORDER BY ety_count DESC
            LIMIT 10
        """)
        rich_etymology = cur.fetchall()

        # Render tables for each leaderboard
        # Most Defined Words Table
        create_table(console, "🏆 Most Defined Words", most_defined,
                     ["Rank", "Word", "Language", "Definitions", "Parts of Speech"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2]), row[3]))
        
        # Most Connected Words Table
        create_table(console, "🔗 Most Connected Words", most_connected,
                     ["Rank", "Word", "Language", "Relations", "Types"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2]), row[3]))
        
        # Words with Most Examples Table
        create_table(console, "📝 Words with Most Examples", most_examples,
                     ["Rank", "Word", "Language", "Examples"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2])))
        
        # Most Common Root Words Table
        create_table(console, "🌱 Most Common Root Words", common_roots,
                     ["Rank", "Word", "Language", "Derived Words"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2])))
        
        # Words with Most Synonyms Table
        create_table(console, "🔄 Words with Most Synonyms", most_synonyms,
                     ["Rank", "Word", "Language", "Synonyms"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2])))
        
        # Words with Richest Etymology Table
        create_table(console, "📚 Words with Richest Etymology", rich_etymology,
                     ["Rank", "Word", "Language", "Sources", "Origin Languages"],
                     lambda i, row: (f"#{i}", row[0], row[1], str(row[2]), row[3]))
    
    except Exception as e:
        logger.error(f"Error gathering leaderboard data: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()


def create_table(console, title, data, headers, row_formatter):
    """Helper to render tables."""
    table = Table(title=title, box=box.ROUNDED)
    for header in headers:
        table.add_column(header)
    
    for i, row in enumerate(data, 1):
        table.add_row(*row_formatter(i, row))
    
    console.print(table)
    console.print()

# -------------------------------------------------------------------
# HELP SUBCOMMAND
# -------------------------------------------------------------------

def display_help(args):
    """
    Display comprehensive help information about all commands.
    """
    console = Console()
    
    # Title
    console.print("\n[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print("[dim]A comprehensive tool for managing Filipino dictionary data[/]\n", justify="center")
    
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
                ("--check-exists", "Skip identical existing entries")
            ],
            "example": "python dictionary_manager.py migrate --check-exists",
            "icon": "🔄"
        },
        {
            "name": "lookup",
            "description": "Look up comprehensive information about a word",
            "options": [
                ("word", "The word to look up")
            ],
            "example": "python dictionary_manager.py lookup kamandag",
            "icon": "🔍"
        },
        {
            "name": "stats",
            "description": "Display comprehensive dictionary statistics",
            "options": [],
            "example": "python dictionary_manager.py stats",
            "icon": "📊"
        },
        {
            "name": "leaderboard",
            "description": "Display Top 10 rankings for various dictionary aspects",
            "options": [],
            "example": "python dictionary_manager.py leaderboard",
            "icon": "🏆"
        },
        {
            "name": "inspect",
            "description": "Inspect word entries in detail",
            "options": [
                ("word", "The word to inspect")
            ],
            "example": "python dictionary_manager.py inspect kulit",
            "icon": "🔬"
        },
        {
            "name": "verify",
            "description": "Verify data integrity",
            "options": [],
            "example": "python dictionary_manager.py verify",
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
        },
        {
            "name": "update",
            "description": "Update database with a new JSON file",
            "options": [
                ("--file", "JSON file to use for update")
            ],
            "example": "python dictionary_manager.py update --file new_data.json",
            "icon": "📝"
        }
    ]
    
    # Create command tables grouped by type
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
    
    # Additional Information
    info_text = """
[bold yellow]Additional Information:[/]

[cyan]1. Database Configuration[/]
   • Set up your database credentials in .env file
   • Required variables: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST

[cyan]2. Data Sources[/]
   • [green]KWF Diksiyonaryo ng Wikang Filipino[/]: Official KWF dictionary
   • [green]kaikki.org[/]: Tagalog and Cebuano entries
   • [green]diksiyonaryo.ph[/]: Online Filipino dictionary
   • [green]tagalog.com[/]: Root words and derivatives

[cyan]3. Features[/]
   • Comprehensive word information with etymology
   • Cross-references and relationships
   • Statistical analysis and leaderboards
   • Data integrity verification
   • Source attribution for all entries

[cyan]4. Logging[/]
   • All operations are logged in [green]logs/dictionary_manager_*.log[/]
   • Detailed debug information available in log files
    """
    
    console.print(Panel(Text.from_markup(info_text), 
                       title="Additional Information",
                       border_style="blue",
                       padding=(1, 2)))
    
    # Footer
    console.print("\n[dim]For more detailed information, visit the documentation.[/]", justify="center")
    console.print()

# -------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------
def main():
    """
    Main CLI entry point with enhanced error handling and help display.
    """
    console = Console()
    
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            """Enhanced error handling with command suggestions."""
            console = Console()
            
            console.print("\n[red bold]Error:[/] Invalid command\n")
            
            # Show available commands in a nice table
            command_table = Table(title="📋 Available Commands", 
                                box=box.ROUNDED, border_style="cyan")
            command_table.add_column("Command", style="bold yellow")
            command_table.add_column("Description", style="white")
            
            commands = [
                ("migrate", "Create/update schema and load data"),
                ("lookup", "Search for a word and its details"),
                ("stats", "Show dictionary statistics"),
                ("leaderboard", "Display Top 10 rankings"),
                ("verify", "Check data integrity"),
                ("purge", "Clear all dictionary data"),
                ("update", "Add new data from file"),
                ("help", "Show detailed help information")
            ]
            
            for cmd, desc in commands:
                command_table.add_row(cmd, desc)
            
            console.print(command_table)
            
            # Show example usage
            console.print("\n[bold cyan]Example usage:[/]")
            console.print("  python dictionary_manager.py lookup kaibigan")
            console.print("  python dictionary_manager.py stats")
            console.print("\n[bold cyan]For more information:[/]")
            console.print("  python dictionary_manager.py help\n")
            sys.exit(2)
    
    parser = CustomArgumentParser(description="Manage dictionary data in PostgreSQL.")
    subparsers = parser.add_subparsers(dest="command")

    migrate_parser = subparsers.add_parser("migrate", help="Create/update schema and load data from sources.")
    migrate_parser.add_argument("--check-exists", action="store_true", 
                              help="Check for and skip identical existing entries")
    
    # Update verify parser to include quick flag
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity.")
    verify_parser.add_argument('--quick', action='store_true', 
                             help='Run quick verification only')
    
    update_parser = subparsers.add_parser("update", help="Update DB with a new JSON file.")
    update_parser.add_argument("--file", type=str, help="JSON file to use for update", required=False)

    # Add the new purge command
    purge_parser = subparsers.add_parser("purge", help="WARNING: Safely delete all data from the database")
    purge_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')

    # Update lookup parser
    lookup_parser = subparsers.add_parser("lookup", 
        help="Look up comprehensive information about a word")
    lookup_parser.add_argument("word", help="Word to look up")
    lookup_parser.add_argument('--detailed', action='store_true',
                             help='Show detailed technical information')

    # Add the stats command
    stats_parser = subparsers.add_parser("stats", help="Display comprehensive dictionary statistics")

    # Add inspect command
    inspect_parser = subparsers.add_parser("inspect", 
        help="Inspect word entries in detail")
    inspect_parser.add_argument("word", help="Word to inspect")

    # Add the leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Display various Top 10 rankings from the dictionary")

    # Add help command
    help_parser = subparsers.add_parser("help", 
        help="Display comprehensive help information")

    args = parser.parse_args()
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
        display_leaderboard(args)
    elif args.command == "help":
        display_help(args)
    else:
        parser.print_help()

def get_standardized_source(source):
    """
    Convert internal source filenames to standardized citation format.
    
    Maps internal source identifiers to proper citation formats:
    - kaikki.org entries are labeled by language
    - Official dictionaries get full names
    - Online sources get proper website names
    
    Args:
        source: Internal source filename
    
    Returns:
        Properly formatted citation string
    """
    source_mapping = {
        'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
        'kaikki.jsonl': 'kaikki.org (Tagalog)', 
        'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
        'root_words_with_associated_words_cleaned.json': 'tagalog.com',
        'tagalog-words.json': 'diksiyonaryo.ph'
    }
    return source_mapping.get(source, source)

def get_standardized_source_sql():
    """
    Returns SQL CASE statement for standardized sources.
    """
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

def format_word_display(word, show_baybayin=True):
    """
    Format a word for display with proper Baybayin handling.
    
    Args:
        word: The word to format
        show_baybayin: Whether to show Baybayin script (default: True)
    
    Returns:
        Formatted word string with romanization if needed
    """
    has_baybayin = any(ord(c) >= 0x1700 and ord(c) <= 0x171F for c in word)
    if has_baybayin:
        romanized = get_romanized_text(word)
        if show_baybayin:
            return f"[bold cyan]{word}[/] [dim](romanized: {romanized})[/]"
        else:
            return romanized
    return word

def clean_language_codes(codes):
    """
    Clean and standardize language codes.
    
    Args:
        codes: String of comma-separated language codes
        
    Returns:
        Cleaned, standardized language codes string
    """
    if not codes:
        return None
        
    # Standard language code mappings
    lang_map = {
        'Tag': 'tl',    # Tagalog
        'Tgl': 'tl',
        'Fil': 'tl',
        'Ceb': 'ceb',   # Cebuano
        'Esp': 'es',    # Spanish
        'Español': 'es',
        'Ing': 'en',    # English
        'Skt': 'sa',    # Sanskrit
        'Bik': 'bik',   # Bikol
        'Hil': 'hil',   # Hiligaynon
        'War': 'war',   # Waray
        'Ilk': 'ilk',   # Ilokano
        'Ifu': 'ifg',   # Ifugao
        'Jap': 'ja',    # Japanese
        'Seb': 'ceb',   # Another form of Cebuano
        'Kap': 'kap',   # Kapampangan
        'ST': 'tl',     # Probably meant Tagalog
    }
    
    # Split, clean, and standardize codes
    cleaned = []
    for code in codes.split(','):
        code = code.strip()
        if code and code != '-':
            # Map to standard code if exists
            code = lang_map.get(code, code)
            cleaned.append(code)
    
    return ', '.join(sorted(set(cleaned))) if cleaned else None

def merge_baybayin_entries(cur, baybayin_id, romanized_id):
    """
    Merge a Baybayin entry with its romanized equivalent.
    Preserves all information from both entries.
    
    Args:
        cur: Database cursor
        baybayin_id: ID of the Baybayin entry
        romanized_id: ID of the romanized entry
    """
    # Transfer definitions
    cur.execute("""
        UPDATE definitions 
        SET word_id = %s 
        WHERE word_id = %s
    """, (romanized_id, baybayin_id))
    
    # Transfer relations
    cur.execute("""
        UPDATE relations 
        SET from_word_id = %s 
        WHERE from_word_id = %s
    """, (romanized_id, baybayin_id))
    
    cur.execute("""
        UPDATE relations 
        SET to_word_id = %s 
        WHERE to_word_id = %s
    """, (romanized_id, baybayin_id))
    
    # Transfer etymologies
    cur.execute("""
        UPDATE etymologies 
        SET word_id = %s 
        WHERE word_id = %s
    """, (romanized_id, baybayin_id))
    
    # Update the romanized entry to indicate it has a Baybayin form
    cur.execute("""
        UPDATE words 
        SET has_baybayin = TRUE,
            romanized_form = COALESCE(romanized_form, 
                (SELECT lemma FROM words WHERE id = %s))
        WHERE id = %s
    """, (baybayin_id, romanized_id))
    
    # Delete the redundant Baybayin entry
    cur.execute("DELETE FROM words WHERE id = %s", (baybayin_id,))

def process_baybayin_entries(cur):
    """
    Process all Baybayin entries, merging them with their romanized equivalents if they exist.
    """
    logger.info("Processing Baybayin entries...")
    
    # Get all Baybayin entries
    cur.execute("""
        SELECT id, lemma 
        FROM words 
        WHERE lemma ~ '[\u1700-\u171F]'
    """)
    baybayin_entries = cur.fetchall()
    
    for baybayin_id, baybayin_lemma in baybayin_entries:
        # Get romanized form
        processed_text, romanized, has_baybayin = process_baybayin_text(baybayin_lemma)
        if not romanized:
            continue
            
        # Look for existing entry with equivalent normalized form
        normalized_romanized = normalize_lemma(romanized)
        cur.execute("""
            SELECT id 
            FROM words 
            WHERE normalized_lemma = %s 
              AND id != %s
              AND NOT (lemma ~ '[\u1700-\u171F]')
            ORDER BY id ASC
            LIMIT 1
        """, (normalized_romanized, baybayin_id))
        
        existing = cur.fetchone()
        
        if existing:
            # Use repr for logging to avoid encoding issues
            logger.info(f"Merging Baybayin entry (ID: {baybayin_id}) with romanized form: {romanized}")
            merge_baybayin_entries(cur, baybayin_id, existing[0])
        else:
            logger.info(f"Updating standalone Baybayin entry (ID: {baybayin_id})")
            cur.execute("""
                UPDATE words 
                SET romanized_form = %s,
                    has_baybayin = TRUE,
                    normalized_lemma = %s
                WHERE id = %s
            """, (romanized, normalized_romanized, baybayin_id))
    
def handle_invalid_command():
    """Handle invalid or missing command arguments."""
    console = Console()
    console.print("\n[bold red]❌ Invalid or missing command[/]")
    console.print("\nAvailable commands:")
    
    commands = {
        "migrate": "Reset and rebuild the database",
        "stats": "View dictionary statistics",
        "inspect": "Look up word information",
        "explore": "Interactive dictionary exploration",
        "verify": "Run integrity checks",
        "help": "Show help information"
    }
    
    for cmd, desc in commands.items():
        console.print(f"[yellow]•[/] [bold cyan]{cmd}[/] - {desc}")
    
    console.print("\nFor detailed help, run: [bold]python dictionary_manager.py help[/]\n")

def standardize_language_codes(codes_str: str) -> str:
    """Use the external language_systems.py to standardize codes."""
    return lsys.standardize_language_codes(codes_str)

def display_etymology_stats(cur, console):
    """Display statistics about word etymologies."""
    cur.execute("""
        SELECT 
            w.lemma,
            w.language_code,
            COUNT(DISTINCT e.id) as source_count,
            STRING_AGG(DISTINCT e.language_codes, ', ') as origins
        FROM words w
        JOIN etymologies e ON w.id = e.word_id
        GROUP BY w.lemma, w.language_code
        HAVING COUNT(DISTINCT e.id) > 5
        ORDER BY source_count DESC, w.lemma
        LIMIT 10
    """)
    
    results = cur.fetchall()
    
    if results:
        table = Table(title="📚 Words with Richest Etymology", box=box.ROUNDED)
        table.add_column("Rank", style="bold yellow", justify="right")
        table.add_column("Word", style="bold cyan")
        table.add_column("Language", style="green")
        table.add_column("Sources", style="magenta", justify="right")
        table.add_column("Origin Languages", style="white", no_wrap=False)
        
        for i, (word, lang, count, origins) in enumerate(results, 1):
            standardized_origins = standardize_language_codes(origins)
            table.add_row(
                f"#{i}",
                word,
                {'tl': 'Tagalog', 'ceb': 'Cebuano'}.get(lang, lang),
                str(count),
                standardized_origins
            )
        
        console.print("\n", table, "\n")

def standardize_existing_etymologies(cur):
    """Update existing etymology entries with standardized language codes."""
    logger.info("Standardizing existing etymology language codes...")
    
    cur.execute("SELECT id, language_codes FROM etymologies")
    etymologies = cur.fetchall()
    
    for ety_id, codes in etymologies:
        standardized = standardize_language_codes(codes)
        cur.execute("""
            UPDATE etymologies 
            SET language_codes = %s 
            WHERE id = %s
        """, (standardized, ety_id))
    
    logger.info(f"Standardized {len(etymologies)} etymology entries")

def process_kaikki_jsonl_new(cur, filename, check_exists=False):
    """Process Kaikki dictionary entries from JSONL file with proper Baybayin handling."""
    logger.info(f"Starting to process Kaikki file: {filename}")

    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    src = os.path.basename(filename)
    lang_code = "tl" if "kaikki.jsonl" in filename else "ceb"

    def extract_baybayin_info(entry):
        """Extract Baybayin information from entry."""
        # Check directly provided Baybayin form first
        baybayin = entry.get("baybayin") or entry.get("script", {}).get("baybayin")
        
        # Check pronunciation for Baybayin characters
        pronunciation = entry.get("pronunciation", "")
        if pronunciation and any(char in pronunciation for char in "ᜀᜁᜂᜃᜄᜅᜆᜇᜈᜉᜊᜋᜌᜎᜏ"):
            baybayin = baybayin or pronunciation

        # Check senses for "Baybayin spelling of X"
        romanized = entry.get("romanized")
        if romanized:
            return baybayin, romanized

        for sense in entry.get("senses", []):
            # Check alt_of field first
            for alt in sense.get("alt_of", []):
                if isinstance(alt, dict) and "word" in alt:
                    romanized = alt["word"]
                    break
            if romanized:
                break
            
            # Check glosses
            for gloss in sense.get("glosses", []):
                if "Baybayin spelling of" in gloss:
                    romanized = gloss.replace("Baybayin spelling of", "").strip()
                    break
            if romanized:
                break

        return baybayin, romanized

    def process_entry(cur, entry):
        """Process a single dictionary entry."""
        try:
            lemma = entry.get("word", "").strip()
            if not lemma:
                return

            # Handle Baybayin entries
            baybayin_form, romanized = extract_baybayin_info(entry)
            has_baybayin = bool(baybayin_form)
            
            # If this is a Baybayin entry, look for existing romanized form
            if has_baybayin and romanized:
                # Check for existing romanized entry
                cur.execute("""
                    SELECT id FROM words 
                    WHERE normalized_lemma = %s AND language_code = %s
                    AND NOT has_baybayin
                """, (normalize_lemma(romanized), lang_code))
                
                result = cur.fetchone()
                if result:
                    # Update existing entry with Baybayin form
                    romanized_id = result[0]
                    cur.execute("""
                        UPDATE words 
                        SET has_baybayin = TRUE,
                            baybayin_form = %s
                        WHERE id = %s
                    """, (baybayin_form, romanized_id))
                    return
                else:
                    # Create romanized entry with Baybayin form
                    word_id = get_or_create_word_id(
                        cur,
                        romanized,
                        lang_code=lang_code,
                        has_baybayin=True,
                        baybayin_form=baybayin_form
                    )
                    return

            # For non-Baybayin entries, process normally
            is_root = entry.get("is_root", False) or ("root word" in entry.get("tags", []))
            root_word_id = None if is_root else get_root_word_id(cur, lemma, lang_code)

            word_id = get_or_create_word_id(
                cur,
                lemma,
                lang_code=lang_code,
                root_word_id=root_word_id,
                check_exists=check_exists
            )

            # Process definitions (skip if pure Baybayin entry)
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
                            sources=src
                        )

            # Process etymology
            etymology = entry.get("etymology_text", "")
            if etymology:
                codes, cleaned_ety = lsys.extract_and_remove_language_codes(etymology)
                insert_etymology(
                    cur,
                    word_id,
                    cleaned_ety,
                    language_codes=", ".join(codes) if codes else "",
                    sources=src,
                )

            # Process relations
            for derived in entry.get("derived", []):
                der_word = derived.get("word", "") if isinstance(derived, dict) else str(derived)
                if der_word := der_word.strip():
                    der_id = get_or_create_word_id(cur, der_word, lang_code)
                    insert_relation(cur, word_id, der_id, "derived_from", src)

            for syn in entry.get("synonyms", []):
                syn_word = syn.get("word", "") if isinstance(syn, dict) else str(syn)
                if syn_word := syn_word.strip():
                    syn_id = get_or_create_word_id(cur, syn_word, lang_code)
                    insert_relation(cur, word_id, syn_id, "synonym", src)

            for rel in entry.get("related", []):
                rel_word = rel.get("word", "") if isinstance(rel, dict) else str(rel)
                if rel_word := rel_word.strip():
                    rel_id = get_or_create_word_id(cur, rel_word, lang_code)
                    insert_relation(cur, word_id, rel_id, "related", src)

            # Process hyponyms/hypernyms
            for hyper in entry.get("hypernyms", []):
                hyper_word = hyper.get("word", "") if isinstance(hyper, dict) else str(hyper)
                if hyper_word := hyper_word.strip():
                    hyper_id = get_or_create_word_id(cur, hyper_word, lang_code)
                    insert_relation(cur, word_id, hyper_id, "hypernym", src)

            for hypo in entry.get("hyponyms", []):
                hypo_word = hypo.get("word", "") if isinstance(hypo, dict) else str(hypo)
                if hypo_word := hypo_word.strip():
                    hypo_id = get_or_create_word_id(cur, hypo_word, lang_code)
                    insert_relation(cur, word_id, hypo_id, "hyponym", src)

        except Exception as e:
            logger.error(f"Error processing entry '{lemma}': {str(e)}")
            return

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {lang_code} entries"):
                process_entry(cur, json.loads(line))
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise

def get_root_word_id(cur, lemma: str, lang_code: str) -> Optional[int]:
    """Get root word ID if it exists."""
    cur.execute("""
        SELECT id FROM words 
        WHERE normalized_lemma = %s 
        AND language_code = %s 
        AND root_word_id IS NULL
    """, (normalize_lemma(lemma), lang_code))
    result = cur.fetchone()
    return result[0] if result else None

def get_romanized_text(text: str) -> str:
    """Convert Baybayin text to romanized form."""
    romanizer = BaybayinRomanizer()
    try:
        return romanizer.romanize(text)
    except ValueError:
        return text

if __name__ == "__main__":
    main()
