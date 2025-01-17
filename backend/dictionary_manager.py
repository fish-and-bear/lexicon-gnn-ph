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
    name_tl VARCHAR(64) NOT NULL,
    CONSTRAINT pos_code_uniq UNIQUE (code)
);

-- words table
CREATE TABLE IF NOT EXISTS words (
    id                SERIAL PRIMARY KEY,
    lemma             VARCHAR(255) NOT NULL,
    normalized_lemma  VARCHAR(255) NOT NULL,
    romanized_form    VARCHAR(255),
    has_baybayin      BOOLEAN DEFAULT FALSE,
    language_code     VARCHAR(16)  NOT NULL,
    root_word_id      INT REFERENCES words(id),
    preferred_spelling VARCHAR(255),
    tags              TEXT,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (language_code, normalized_lemma)
);

-- (NEW) Composite index for faster lookups by (language_code, normalized_lemma)
CREATE INDEX IF NOT EXISTS idx_words_lang_norm_lemma 
    ON words(language_code, normalized_lemma);

-- Add index for Baybayin search
CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(has_baybayin);

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

def normalize_lemma(lemma: str) -> str:
    """
    Remove diacritics and convert to lowercase, e.g. "sumagíp" -> "sumagip"
    """
    if not lemma:
        return ""
    return unidecode.unidecode(lemma).lower()

def clean_lemma_if_baybayin_spelling(lemma: str) -> str:
    """Strip the prefix 'Baybayin spelling of' if present."""
    prefix = "Baybayin spelling of"
    if lemma.lower().startswith(prefix.lower()):
        return lemma[len(prefix):].strip()
    return lemma

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

def get_or_create_word_id(
    cur,
    lemma,
    lang_code="tl",
    root_word_id=None,
    preferred_spelling=None,
    tags=None,
    entry_data=None,
    check_exists=False,
    has_baybayin=None,
    romanized_form=None
):
    """
    Retrieve or create a row in the 'words' table. 
    This version assumes that if 'has_baybayin' is not provided, it's False, 
    i.e., do NOT attempt to auto-detect Baybayin unless explicitly told.

    Args:
        cur (cursor): Database cursor
        lemma (str): The original lemma or word form
        lang_code (str): e.g. "tl", "ceb", ...
        root_word_id (int): If this lemma is derived from another word, pass that ID
        preferred_spelling (str): If there is a known official or standardized spelling
        tags (str): Arbitrary tags or metadata
        entry_data (dict): Additional metadata for possible comparisons (optional)
        check_exists (bool): Whether to check for identical existing entries 
                             before inserting (optional)
        has_baybayin (bool): If explicitly True, we mark the entry as Baybayin
        romanized_form (str): If has_baybayin=True, you can provide a known romanization

    Returns:
        int: The ID of the record in the 'words' table.
    """
    # Default to not Baybayin if not explicitly stated
    if has_baybayin is None:
        has_baybayin = False

    # Normalize lemma (remove diacritics, lowercase)
    norm_lemma = normalize_lemma(lemma)

    # If has_baybayin is false, don't do any auto-detection
    processed_lemma = lemma
    if has_baybayin and romanized_form is None:
        # If the caller said it's Baybayin but didn't provide a romanized form,
        # you could do a fallback or detection. For example:
        processed_lemma, auto_romanized, _ = process_baybayin_text(lemma)
        romanized_form = auto_romanized
    else:
        # Otherwise, no change to the lemma
        processed_lemma = lemma

    # Check if there's an existing row with the same normalized lemma & language
    cur.execute("""
        SELECT id, lemma, preferred_spelling, romanized_form
          FROM words
         WHERE normalized_lemma = %s
           AND language_code = %s
         LIMIT 1
    """, (norm_lemma, lang_code))
    existing = cur.fetchone()

    if existing:
        existing_id = existing[0]
        existing_lemma = existing[1]
        existing_pref = existing[2]
        existing_rom = existing[3]

        # If the caller explicitly says "Yes, it has Baybayin" but existing doesn't,
        # update that entry with the new romanized form.
        if has_baybayin and not existing_rom:
            cur.execute("""
                UPDATE words
                   SET romanized_form = %s,
                       has_baybayin = TRUE
                 WHERE id = %s
            """, (romanized_form, existing_id))
        return existing_id

    # Otherwise, insert a new row
    insert_sql = """
        INSERT INTO words
            (lemma, normalized_lemma, romanized_form, has_baybayin,
             language_code, root_word_id, preferred_spelling, tags)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    cur.execute(
        insert_sql,
        (
            processed_lemma,
            norm_lemma,
            romanized_form,
            has_baybayin,
            lang_code,
            root_word_id,
            preferred_spelling,
            tags,
        )
    )
    new_id = cur.fetchone()[0]
    return new_id

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

def insert_definition(cur,
                     word_id: int,
                     definition_text: str,
                     part_of_speech: str = "",
                     examples: str = None,
                     usage_notes: str = None,
                     sources: str = "") -> int:
    """Insert a definition with both original and standardized POS."""
    # Get standardized POS ID while keeping original
    std_pos_id = get_standardized_pos_id(cur, part_of_speech)
    
    # Insert new definition
    cur.execute("""
        INSERT INTO definitions 
            (word_id, definition_text, original_pos, standardized_pos_id,
             examples, usage_notes, sources)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        word_id,
        definition_text,
        part_of_speech,  # Keep original POS string
        std_pos_id,      # Store standardized reference
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

def insert_etymology(cur, word_id, original_text, language_codes="", normalized_components=None, sources=""):
    """Insert etymology with standardized language codes."""
    try:
        # Standardize language codes before insertion
        standardized_codes = standardize_language_codes(language_codes)
        
        cur.execute("""
    INSERT INTO etymologies
        (word_id, original_text, normalized_components, language_codes, sources)
            VALUES 
                (%s, %s, %s, %s, %s)
            ON CONFLICT (word_id, original_text) DO UPDATE
            SET 
                normalized_components = EXCLUDED.normalized_components,
                language_codes = EXCLUDED.language_codes,
                sources = CASE 
                    WHEN etymologies.sources IS NULL THEN EXCLUDED.sources
                    WHEN EXCLUDED.sources IS NULL THEN etymologies.sources
                    ELSE etymologies.sources || ' | ' || EXCLUDED.sources
                END
            RETURNING id
        """, (
            word_id,
            original_text,
            normalized_components,
            standardized_codes,
            sources
        ))
        
        return cur.fetchone()[0]
    except Exception as e:
        logger.error(f"Error inserting etymology: {str(e)}")
        raise

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

def process_baybayin_entries(cur):
    """
    Process and clean Baybayin entries in the database.
    - Fix entries with lemmas starting with "Baybayin spelling of".
    - Merge or update Baybayin entries as needed.
    """
    logger.info("Processing Baybayin entries...")

    # Step 1: Fix lemmas starting with "Baybayin spelling of"
    cur.execute("""
        SELECT id, lemma
        FROM words
        WHERE lower(lemma) LIKE 'baybayin spelling of%'
    """)
    rows = cur.fetchall()

    for row_id, old_lemma in rows:
        new_lemma = clean_lemma_if_baybayin_spelling(old_lemma)
        if not new_lemma:
            continue
        norm_lemma = normalize_lemma(new_lemma)

        # Check for existing word with the cleaned lemma
        cur.execute("""
            SELECT id FROM words 
            WHERE normalized_lemma = %s 
              AND language_code = 'tl'
            LIMIT 1
        """, (norm_lemma,))
        existing = cur.fetchone()

        if existing:
            existing_id = existing[0]
            # Merge the problematic entry with the existing one
            logger.info(f"Merging word ID {row_id} ('{old_lemma}') into existing ID {existing_id}")
            merge_baybayin_entries(cur, row_id, existing_id)
        else:
            logger.info(f"Updating lemma for word ID {row_id} from '{old_lemma}' to '{new_lemma}'")
            cur.execute("""
                UPDATE words
                   SET lemma = %s,
                       normalized_lemma = %s
                 WHERE id = %s
            """, (new_lemma, norm_lemma, row_id))

    # Step 2: Process standard Baybayin entries (existing logic)
    cur.execute("""
        SELECT id, lemma 
        FROM words 
        WHERE lemma ~ '[\u1700-\u171F]'
    """)
    baybayin_entries = cur.fetchall()

    for baybayin_id, baybayin_lemma in baybayin_entries:
        processed_text, romanized, has_baybayin = process_baybayin_text(baybayin_lemma)
        if not romanized:
            continue

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
            logger.info(f"Merging Baybayin entry ID {baybayin_id} ('{baybayin_lemma}') with Romanized ID {existing[0]}")
            merge_baybayin_entries(cur, baybayin_id, existing[0])
        else:
            logger.info(f"Updating standalone Baybayin entry ID {baybayin_id}")
            cur.execute("""
                UPDATE words 
                SET romanized_form = %s,
                    has_baybayin = TRUE,
                    normalized_lemma = %s
                WHERE id = %s
            """, (romanized, normalized_romanized, baybayin_id))

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
    """
    Migrate data with complete database reset for clean installation.
    """
    logger.info("Starting fresh data migration process")
        
    conn = get_connection()
    cur = conn.cursor()

    try:
        # First drop all tables to ensure clean slate
        logger.info("Dropping all existing tables...")
        cur.execute("""
            DROP TABLE IF EXISTS 
                etymologies,
                relations, 
                definitions,
                words CASCADE;
                
            DROP TYPE IF EXISTS word_type CASCADE;
        """)
        conn.commit()
        
        # Set up extensions
        setup_extensions(conn)
        
        # Create fresh tables
        logger.info("Creating new database tables")
        create_or_update_tables(conn)
        
        # Proceed with data migration
        logger.info("Beginning data processing")
        
        # Process each source file
        source_files = [
            ("1. Processing Tagalog words", "data/tagalog-words.json", process_tagalog_words_file),
            ("2. Processing Root words", "data/root_words_with_associated_words_cleaned.json", process_root_words_file),
            ("3. Processing KWF Dictionary", "data/kwf_dictionary.json", process_kwf_file),
            ("4. Processing Kaikki (Tagalog)", "data/kaikki.jsonl", process_kaikki_jsonl_new),
            ("5. Processing Kaikki (Cebuano)", "data/kaikki-ceb.jsonl", process_kaikki_jsonl_new)
        ]
        
        for message, filename, processor in source_files:
            logger.info(message)
            processor(cur, filename, args.check_exists)
            
            # After each file is processed, handle any Baybayin entries
            logger.info("Processing Baybayin entries from recent import...")
            process_baybayin_entries(cur)
        conn.commit()

        # After all data is processed
        logger.info("Standardizing etymology language codes...")
        standardize_existing_etymologies(cur)
        conn.commit()
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()
        logger.info("Database connection closed")

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
    elif args.command == "explore":
        explore_dictionary(args)
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

def get_romanized_text(text):
    """
    Convert Baybayin characters to their romanized form with full character support.
    """
    baybayin_to_roman = {
        # Basic vowels
        'ᜀ': 'a',     # A
        'ᜁ': 'i/e',   # I/E
        'ᜂ': 'o/u',   # O/U
        
        # Basic consonants (with default 'a' vowel)
        'ᜃ': 'ka',    # KA
        'ᜄ': 'ga',    # GA
        'ᜅ': 'nga',   # NGA
        'ᜆ': 'ta',    # TA
        'ᜇ': 'da',    # DA
        'ᜈ': 'na',    # NA
        'ᜉ': 'pa',    # PA
        'ᜊ': 'ba',    # BA
        'ᜋ': 'ma',    # MA
        'ᜌ': 'ya',    # YA
        'ᜎ': 'la',    # LA
        'ᜏ': 'wa',    # WA
        'ᜐ': 'sa',    # SA
        'ᜑ': 'ha',    # HA
        
        # Vowel markers
        'ᜒ': 'i',     # I/E marker
        'ᜓ': 'u',     # U/O marker
        '᜔': '',      # Virama (removes inherent 'a' vowel)
        
        # Special characters
        '᜵': ',',     # Comma
        '᜶': '.',     # Period
        ' ': ' ',     # Space
    }
    
    # Process text character by character
    result = []
    i = 0
    while i < len(text):
        # Try three-character combinations first
        if i + 2 < len(text):
            three_chars = text[i:i+3]
            if three_chars in baybayin_to_roman:
                result.append(baybayin_to_roman[three_chars])
                i += 3
                continue
        
        # Try two-character combinations
        if i + 1 < len(text):
            two_chars = text[i:i+2]
            # Handle consonant + vowel marker combinations
            if text[i] in 'ᜃᜄᜅᜆᜇᜈᜉᜊᜋᜌᜎᜏᜐᜑ':
                if text[i+1] == 'ᜒ':  # I/E marker
                    result.append(baybayin_to_roman[text[i]].replace('a', 'i'))
                    i += 2
                    continue
                elif text[i+1] == 'ᜓ':  # U/O marker
                    result.append(baybayin_to_roman[text[i]].replace('a', 'u'))
                    i += 2
                    continue
                elif text[i+1] == '᜔':  # Virama
                    result.append(baybayin_to_roman[text[i]][0])  # Just the consonant
                    i += 2
                    continue
        
        # Single character
        if text[i] in baybayin_to_roman:
            result.append(baybayin_to_roman[text[i]])
        else:
            result.append(text[i])
        i += 1
    
    return ''.join(result)

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

def process_baybayin_text(text):
    """
    Process Baybayin text for storage and display.
    
    Args:
        text: Text that might contain Baybayin
        
    Returns:
        tuple: (processed_text, romanized_text, has_baybayin)
    """
    has_baybayin = any(ord(c) >= 0x1700 and ord(c) <= 0x171F for c in text)
    if not has_baybayin:
        return text, None, False
        
    romanized = get_romanized_text(text)
    return text, romanized, True

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

def explore_dictionary(args):
    """
    Interactive dictionary exploration with rich visual interface.
    """
    console = Console()
    conn = get_connection()
    cur = conn.cursor()

    try:
        while True:
            console.clear()
            
            # Create a visually appealing header
            header = Panel(
                Align.center(
                    Text.from_markup(
                        "\n[bold cyan]📚 Filipino Dictionary Explorer[/]\n"
                        "[dim]Discover the richness of Filipino languages[/]\n"
                    )
                ),
                box=box.ROUNDED,
                border_style="cyan",
                padding=(1, 2)
            )
            console.print(header)
            
            # Show quick stats in a horizontal layout
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT w.id) as words,
                    COUNT(DISTINCT d.id) as definitions,
                    COUNT(DISTINCT e.id) as etymologies,
                    COUNT(DISTINCT CASE WHEN w.lemma ~ '[\u1700-\u171F]' THEN w.id END) as baybayin
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN etymologies e ON w.id = e.word_id
            """)
            stats = cur.fetchone()
            
            stats_layout = Table.grid(padding=2)
            stats_layout.add_column(justify="center")
            stats_layout.add_column(justify="center")
            stats_layout.add_column(justify="center")
            stats_layout.add_column(justify="center")
            
            stats_layout.add_row(
                f"[bold yellow]📝 {stats[0]:,}[/]\n[dim]Words[/]",
                f"[bold green]📖 {stats[1]:,}[/]\n[dim]Definitions[/]",
                f"[bold magenta]🌱 {stats[2]:,}[/]\n[dim]Etymologies[/]",
                f"[bold cyan]🔤 {stats[3]:,}[/]\n[dim]Baybayin[/]"
            )
            
            console.print(Panel(stats_layout, border_style="blue"))
            console.print()
            
            # Create an attractive menu layout
            menu_grid = Table.grid(padding=1)
            menu_grid.add_column(ratio=1)
            menu_grid.add_column(ratio=1)
            
            # Left column - Basic exploration
            basic_menu = Table(
                title="Basic Exploration",
                box=box.ROUNDED,
                border_style="cyan",
                show_header=False
            )
            basic_menu.add_column("Option", style="bold yellow")
            basic_menu.add_column("Description", style="white")
            
            basic_options = [
                ("1", "Browse by Language", "🌏"),
                ("2", "Search by Word Type", "📑"),
                ("3", "Recent Additions", "✨"),
                ("4", "Popular Words", "⭐")
            ]
            
            for num, desc, icon in basic_options:
                basic_menu.add_row(f"{icon} [{num}]", desc)
            
            # Right column - Advanced features
            advanced_menu = Table(
                title="Advanced Features",
                box=box.ROUNDED,
                border_style="magenta",
                show_header=False
            )
            advanced_menu.add_column("Option", style="bold yellow")
            advanced_menu.add_column("Description", style="white")
            
            advanced_options = [
                ("5", "Etymology Search", "🔍"),
                ("6", "Baybayin Browser", "🔤"),
                ("7", "Word Relationships", "🔄"),
                ("8", "Pattern Explorer", "📊")
            ]
            
            for num, desc, icon in advanced_options:
                advanced_menu.add_row(f"{icon} [{num}]", desc)
            
            menu_grid.add_row(basic_menu, advanced_menu)
            console.print(menu_grid)
            
            # Help footer
            footer = Table.grid()
            footer.add_column()
            footer.add_row("[dim]Enter a number to explore or 'q' to quit[/]")
            footer.add_row("[dim]Type 'help' for detailed information[/]")
            console.print("\n", Align.center(footer))
            
            # Get user choice with enhanced feedback
            choice = input("\n[→] Your choice: ").strip().lower()
            
            if choice == 'q':
                console.print("\n[yellow]Thanks for exploring! Goodbye![/]")
                break
            elif choice == 'help':
                display_explorer_help(console)
                input("\nPress Enter to continue...")
                continue
            
            # Process numeric choices
            try:
                choice_num = int(choice)
                if choice_num == 1:
                    browse_by_language_pos(cur, console)
                elif choice_num == 2:
                    browse_by_word_type(cur, console)
                elif choice_num == 3:
                    view_recent_additions(cur, console)
                elif choice_num == 4:
                    view_popular_words(cur, console)
                elif choice_num == 5:
                    search_by_etymology(cur, console)
                elif choice_num == 6:
                    browse_baybayin(cur, console)
                elif choice_num == 7:
                    explore_relationships(cur, console)
                elif choice_num == 8:
                    explore_patterns(cur, console)
                else:
                    console.print("[red]Invalid option. Please try again.[/]")
                    input("\nPress Enter to continue...")
            except ValueError:
                console.print("[red]Please enter a valid number or command.[/]")
                input("\nPress Enter to continue...")

    finally:
        cur.close()
        conn.close()

def display_explorer_help(console):
    """Display detailed help for the explorer interface."""
    help_text = """
[bold cyan]Dictionary Explorer Help[/]

[bold yellow]Basic Exploration:[/]
  🌏 [bold]Browse by Language[/]
     • View words by language and part of speech
     • Filter and sort by various criteria
     • Explore word definitions and usage

  📑 [bold]Search by Word Type[/]
     • Find nouns, verbs, adjectives, etc.
     • See common patterns and forms
     • Explore word formation rules

  ✨ [bold]Recent Additions[/]
     • Latest entries in the dictionary
     • New words and definitions
     • Track dictionary growth

  ⭐ [bold]Popular Words[/]
     • Most frequently referenced words
     • Common root words
     • Words with rich definitions

[bold yellow]Advanced Features:[/]
  🔍 [bold]Etymology Search[/]
     • Trace word origins
     • Explore language influences
     • Find cognates and borrowed words

  🔤 [bold]Baybayin Browser[/]
     • Explore Baybayin script
     • View romanized pairs
     • Learn character meanings

  🔄 [bold]Word Relationships[/]
     • Find related words
     • Explore word families
     • Discover semantic connections

  📊 [bold]Pattern Explorer[/]
     • Analyze word formation
     • Find similar patterns
     • Study language structure

[bold yellow]Navigation Tips:[/]
  • Use numbers [1-8] to select options
  • Press 'b' to go back in any menu
  • Type 'q' to quit the explorer
  • Type 'help' for this information
    """
    
    console.print(Panel(
        Text.from_markup(help_text),
        title="Explorer Help",
        border_style="cyan",
        padding=(1, 2)
    ))

def browse_by_language_pos(cur, console):
    """Browse words by language and part of speech."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Browse by Language & Part of Speech[/]\n")
        
        # Get languages with word counts
        cur.execute("""
            SELECT 
                language_code,
                COUNT(*) as words,
                COUNT(DISTINCT d.standardized_pos_id) as pos_count
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            GROUP BY language_code
            ORDER BY words DESC
        """)
        
        # Display languages
        lang_table = Table(box=box.ROUNDED)
        lang_table.add_column("Language", style="bold yellow")
        lang_table.add_column("Words", justify="right", style="cyan")
        lang_table.add_column("Parts of Speech", justify="right", style="green")
        
        for lang, words, pos in cur.fetchall():
            lang_name = {'tl': 'Tagalog', 'ceb': 'Cebuano'}.get(lang, lang)
            lang_table.add_row(lang, f"{words:,}", f"{pos:,}")
        
        console.print(lang_table)
        
        # Get language choice
        lang = input("\nEnter language code (or 'b' to go back): ").strip().lower()
        if lang == 'b':
            break
            
        # Show parts of speech for chosen language
        cur.execute("""
            SELECT part_of_speech, COUNT(*) as count
            FROM words w
            JOIN definitions d ON w.id = d.word_id
            WHERE language_code = %s AND part_of_speech IS NOT NULL
            GROUP BY part_of_speech
            ORDER BY count DESC
        """, (lang,))
        
        pos_results = cur.fetchall()
        if not pos_results:
            console.print("[yellow]No parts of speech found for this language[/]")
            continue
            
        pos_table = Table(box=box.ROUNDED)
        pos_table.add_column("Part of Speech", style="bold yellow")
        pos_table.add_column("Count", justify="right", style="cyan")
        
        for pos, count in pos_results:
            pos_table.add_row(pos, f"{count:,}")
        
        console.print("\n", pos_table)
        
        # Get POS choice and show words
        pos = input("\nEnter part of speech (or 'b' to go back): ").strip()
        if pos == 'b':
            continue
            
        cur.execute("""
            SELECT DISTINCT w.lemma, d.definition_text
            FROM words w
            JOIN definitions d ON w.id = d.word_id
            WHERE w.language_code = %s AND d.standardized_pos_id = %s
            ORDER BY w.lemma
            LIMIT 50
        """, (lang, pos))
        
        words_table = Table(box=box.ROUNDED)
        words_table.add_column("Word", style="bold yellow")
        words_table.add_column("Definition", style="white")
        
        for word, definition in cur.fetchall():
            words_table.add_row(
                format_word_display(word),
                Text(definition[:100] + "..." if len(definition) > 100 else definition)
            )
        
        console.print("\n", words_table)
        input("\nPress Enter to continue...")

def browse_by_word_type(cur, console):
    """Browse words by their part of speech and type."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Browse by Word Type[/]\n")
        
        # Get all parts of speech with counts
        cur.execute("""
            SELECT 
                part_of_speech,
                COUNT(*) as count,
                COUNT(DISTINCT w.language_code) as lang_count
            FROM definitions d
            JOIN words w ON w.id = d.word_id
            WHERE part_of_speech IS NOT NULL
            GROUP BY part_of_speech
            ORDER BY count DESC
        """)
        
        # Display POS table
        pos_table = Table(box=box.ROUNDED)
        pos_table.add_column("Part of Speech", style="bold yellow")
        pos_table.add_column("Words", justify="right", style="cyan")
        pos_table.add_column("Languages", justify="right", style="green")
        
        for pos, count, langs in cur.fetchall():
            pos_table.add_row(pos, f"{count:,}", f"{langs}")
        
        console.print(pos_table)
        
        # Get POS choice
        pos = input("\nEnter part of speech (or 'b' to go back): ").strip()
        if pos == 'b':
            break
            
        # Show words for chosen POS
        cur.execute("""
            SELECT 
                w.lemma,
                w.language_code,
                d.definition_text
            FROM words w
            JOIN definitions d ON w.id = d.word_id
            WHERE d.standardized_pos_id = %s
            ORDER BY w.language_code, w.lemma
            LIMIT 50
        """, (pos,))
        
        words_table = Table(box=box.ROUNDED)
        words_table.add_column("Word", style="bold yellow")
        words_table.add_column("Language", style="cyan")
        words_table.add_column("Definition", style="white")
        
        for word, lang, definition in cur.fetchall():
            words_table.add_row(
                format_word_display(word),
                lang,
                Text(definition[:100] + "..." if len(definition) > 100 else definition)
            )
        
        console.print("\n", words_table)
        input("\nPress Enter to continue...")

def view_recent_additions(cur, console):
    """Display recently added dictionary entries."""
    console.clear()
    console.print("\n[bold cyan]Recent Dictionary Additions[/]\n")
    
    cur.execute("""
        SELECT 
            w.lemma,
            w.language_code,
            d.definition_text,
            d.standardized_pos_id,
            w.id
        FROM words w
        LEFT JOIN definitions d ON w.id = d.word_id
        ORDER BY w.id DESC
        LIMIT 50
    """)
    
    results_table = Table(box=box.ROUNDED)
    results_table.add_column("Word", style="bold yellow")
    results_table.add_column("Language", style="cyan")
    results_table.add_column("Definition", style="white")
    results_table.add_column("Type", style="green")
    
    for word, lang, definition, pos, _ in cur.fetchall():
        results_table.add_row(
            format_word_display(word),
            lang,
            Text(definition[:100] + "..." if definition and len(definition) > 100 else (definition or "-")),
            pos or "-"
        )
    
    console.print(results_table)
    input("\nPress Enter to continue...")

def view_popular_words(cur, console):
    """Display most frequently referenced words."""
    console.clear()
    console.print("\n[bold cyan]Popular Dictionary Words[/]\n")
    
    cur.execute("""
        SELECT 
            w.lemma,
            w.language_code,
            COUNT(DISTINCT d.id) as def_count,
            COUNT(DISTINCT r.id) as rel_count,
            STRING_AGG(DISTINCT d.definition_text, ' | ') as definitions
        FROM words w
        LEFT JOIN definitions d ON w.id = d.word_id
        LEFT JOIN relations r ON w.id = r.from_word_id OR w.id = r.to_word_id
        GROUP BY w.id, w.lemma, w.language_code
        ORDER BY (COUNT(DISTINCT d.id) + COUNT(DISTINCT r.id)) DESC
        LIMIT 50
    """)
    
    results_table = Table(box=box.ROUNDED)
    results_table.add_column("Word", style="bold yellow")
    results_table.add_column("Language", style="cyan")
    results_table.add_column("Definitions", justify="right", style="green")
    results_table.add_column("Relations", justify="right", style="magenta")
    results_table.add_column("Sample Definition", style="white")
    
    for word, lang, def_count, rel_count, defs in cur.fetchall():
        first_def = defs.split(' | ')[0] if defs else "-"
        results_table.add_row(
            format_word_display(word),
            lang,
            str(def_count),
            str(rel_count),
            Text(first_def[:100] + "..." if len(first_def) > 100 else first_def)
        )
    
    console.print(results_table)
    input("\nPress Enter to continue...")

def search_by_etymology(cur, console):
    """Search words by their etymology and language origins."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Search by Etymology Origins[/]\n")
        
        # Get available language origins
        cur.execute("""
            WITH RECURSIVE split_langs AS (
                SELECT DISTINCT 
                    regexp_split_to_table(language_codes, ',\s*') as lang_code,
                    COUNT(*) as word_count
                FROM etymologies
                WHERE language_codes IS NOT NULL
                GROUP BY lang_code
            )
            SELECT 
                lang_code,
                word_count,
                ROUND(word_count::numeric / 
                    (SELECT COUNT(DISTINCT word_id) FROM etymologies) * 100, 1) as percentage
            FROM split_langs
            ORDER BY word_count DESC
        """)
        
        # Display language origins table
        origins_table = Table(box=box.ROUNDED)
        origins_table.add_column("Language", style="bold yellow")
        origins_table.add_column("Words", justify="right", style="cyan")
        origins_table.add_column("Percentage", justify="right", style="green")
        
        for lang, count, pct in cur.fetchall():
            origins_table.add_row(lang.strip(), f"{count:,}", f"{pct}%")
        
        console.print(origins_table)
        
        # Get language choice
        lang_choice = input("\nEnter language code to explore (or 'b' to go back): ").strip()
        if lang_choice.lower() == 'b':
            break
            
        # Show words with this origin
        cur.execute("""
            SELECT 
                w.lemma,
                w.language_code,
                e.original_text,
                e.language_codes
            FROM words w
            JOIN etymologies e ON w.id = e.word_id
            WHERE e.language_codes ILIKE %s
            ORDER BY w.lemma
            LIMIT 50
        """, (f'%{lang_choice}%',))
        
        results_table = Table(box=box.ROUNDED)
        results_table.add_column("Word", style="bold yellow")
        results_table.add_column("Language", style="cyan")
        results_table.add_column("Etymology", style="white")
        results_table.add_column("Path", style="green")
        
        for word, lang, ety, langs in cur.fetchall():
            results_table.add_row(
                format_word_display(word),
                lang,
                Text(ety[:100] + "..." if len(ety) > 100 else ety),
                langs
            )
        
        if results_table.row_count:
            console.print("\n", results_table)
        else:
            console.print(f"\n[yellow]No words found with {lang_choice} origin[/]")
        
        input("\nPress Enter to continue...")

def browse_baybayin(cur, console):
    """Browse and explore Baybayin entries."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Browse Baybayin Entries[/]\n")
        
        # Get Baybayin statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE romanized_form IS NOT NULL) as with_romanized,
                COUNT(*) FILTER (WHERE 
                    normalized_lemma NOT IN (
                        SELECT normalized_lemma 
                        FROM words w2 
                        WHERE NOT (w2.lemma ~ '[\u1700-\u171F]')
                    )
                ) as standalone
            FROM words
            WHERE lemma ~ '[\u1700-\u171F]'
        """)
        
        stats = cur.fetchone()
        
        # Show statistics
        stats_table = Table(title="Baybayin Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="bold yellow")
        stats_table.add_column("Count", justify="right", style="cyan")
        
        stats_table.add_row("Total Entries", f"{stats[0]:,}")
        stats_table.add_row("With Romanization", f"{stats[1]:,}")
        stats_table.add_row("Standalone Entries", f"{stats[2]:,}")
        
        console.print(stats_table)
        console.print()
        
        # Show options
        options = [
            ("1", "View All Baybayin Entries"),
            ("2", "View Standalone Entries"),
            ("3", "View Entries with Romanized Pairs"),
            ("4", "Search Baybayin"),
            ("b", "Back to Main Menu")
        ]
        
        for opt, desc in options:
            console.print(f"[cyan]{opt}[/] {desc}")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'b':
            break
            
        # Build query based on choice
        query = """
            SELECT 
                w.lemma,
                w.romanized_form,
                w.language_code,
                STRING_AGG(DISTINCT d.definition_text, ' | ') as definitions,
                EXISTS (
                    SELECT 1 
                    FROM words w2 
                    WHERE w2.normalized_lemma = w.normalized_lemma 
                    AND NOT (w2.lemma ~ '[\u1700-\u171F]')
                ) as has_pair
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            WHERE w.lemma ~ '[\u1700-\u171F]'
        """
        
        if choice == '2':  # Standalone only
            query += " AND NOT EXISTS (SELECT 1 FROM words w2 WHERE w2.normalized_lemma = w.normalized_lemma AND NOT (w2.lemma ~ '[\u1700-\u171F]'))"
        elif choice == '3':  # With pairs only
            query += " AND EXISTS (SELECT 1 FROM words w2 WHERE w2.normalized_lemma = w.normalized_lemma AND NOT (w2.lemma ~ '[\u1700-\u171F]'))"
        elif choice == '4':  # Search
            search = input("\nEnter search term: ").strip()
            if not search:
                continue
            query += f" AND (w.lemma LIKE '%{search}%' OR w.romanized_form LIKE '%{search}%')"
        
        query += " GROUP BY w.lemma, w.romanized_form, w.language_code ORDER BY w.lemma LIMIT 50"
        
        cur.execute(query)
        entries = cur.fetchall()
        
        if entries:
            results_table = Table(box=box.ROUNDED)
            results_table.add_column("Baybayin", style="bold cyan")
            results_table.add_column("Romanized", style="yellow")
            results_table.add_column("Language", style="green")
            results_table.add_column("Definitions", style="white")
            results_table.add_column("Status", style="magenta")
            
            for baybayin, roman, lang, defs, has_pair in entries:
                status = "Paired" if has_pair else "Standalone"
                results_table.add_row(
                    baybayin,
                    roman or "-",
                    lang,
                    Text(defs[:100] + "..." if defs and len(defs) > 100 else (defs or "-")),
                    status
                )
            
            console.print("\n", results_table)
        else:
            console.print("\n[yellow]No entries found[/]")
        
        input("\nPress Enter to continue...")

def explore_relationships(cur, console):
    """Explore word relationships and connections."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Explore Word Relationships[/]\n")
        
        # Get relationship types and counts
        cur.execute("""
            SELECT relation_type, COUNT(*) as count
            FROM relations
            GROUP BY relation_type
            ORDER BY count DESC
        """)
        
        # Display relationship types
        types_table = Table(box=box.ROUNDED)
        types_table.add_column("Relationship", style="bold yellow")
        types_table.add_column("Count", justify="right", style="cyan")
        
        for rel_type, count in cur.fetchall():
            types_table.add_row(rel_type, f"{count:,}")
        
        console.print(types_table)
        
        # Get relationship choice
        rel_type = input("\nEnter relationship type (or 'b' to go back): ").strip()
        if rel_type.lower() == 'b':
            break
            
        # Show word pairs with this relationship
        cur.execute("""
            SELECT 
                w1.lemma as from_word,
                w2.lemma as to_word,
                w1.language_code,
                STRING_AGG(DISTINCT d.definition_text, ' | ') as definitions
            FROM relations r
            JOIN words w1 ON r.from_word_id = w1.id
            JOIN words w2 ON r.to_word_id = w2.id
            LEFT JOIN definitions d ON w1.id = d.word_id
            WHERE r.relation_type = %s
            GROUP BY w1.lemma, w2.lemma, w1.language_code
            ORDER BY w1.lemma
            LIMIT 50
        """, (rel_type,))
        
        pairs_table = Table(box=box.ROUNDED)
        pairs_table.add_column("From", style="bold yellow")
        pairs_table.add_column("To", style="bold cyan")
        pairs_table.add_column("Language", style="green")
        pairs_table.add_column("Definition", style="white")
        
        for from_word, to_word, lang, defs in cur.fetchall():
            pairs_table.add_row(
                format_word_display(from_word),
                format_word_display(to_word),
                lang,
                Text(defs[:100] + "..." if defs and len(defs) > 100 else (defs or "-"))
            )
        
        console.print("\n", pairs_table)
        input("\nPress Enter to continue...")

def explore_patterns(cur, console):
    """Explore word formation patterns and structures."""
    while True:
        console.clear()
        console.print("\n[bold cyan]Word Pattern Explorer[/]\n")
        
        # Show pattern options
        pattern_table = Table(box=box.ROUNDED)
        pattern_table.add_column("Pattern", style="bold yellow")
        pattern_table.add_column("Description", style="white")
        pattern_table.add_column("Example", style="cyan")
        
        patterns = [
            ("Prefix", "Words starting with...", "mag-, pag-, ka-"),
            ("Suffix", "Words ending with...", "-an, -in, -han"),
            ("Length", "Words of length...", "5 letters, 10+ chars"),
            ("Infix", "Words containing...", "-um-, -in-"),
            ("Reduplicated", "Repeated syllables", "kaka-, papa-")
        ]
        
        for pat, desc, ex in patterns:
            pattern_table.add_row(pat, desc, ex)
        
        console.print(pattern_table)
        
        pattern = input("\nEnter pattern type (or 'b' to go back): ").strip().lower()
        if pattern == 'b':
            break
            
        search = input("Enter specific pattern to search: ").strip()
        if not search:
            continue
            
        # Build query based on pattern type
        query = """
            SELECT 
                w.lemma,
                w.language_code,
                STRING_AGG(DISTINCT d.definition_text, ' | ') as definitions
            FROM words w
            LEFT JOIN definitions d ON w.id = d.word_id
            WHERE """
            
        if pattern.startswith('p'):  # prefix
            query += "w.lemma LIKE %s"
            search = f"{search}%"
        elif pattern.startswith('s'):  # suffix
            query += "w.lemma LIKE %s"
            search = f"%{search}"
        elif pattern.startswith('l'):  # length
            try:
                length = int(search)
                query += "LENGTH(w.lemma) = %s"
                search = length
            except ValueError:
                console.print("[red]Please enter a valid number for length[/]")
                continue
        elif pattern.startswith('i'):  # infix
            query += "w.lemma LIKE %s"
            search = f"%{search}%"
        elif pattern.startswith('r'):  # reduplicated
            query += "w.lemma ~ %s"
            search = f"^{search}.*{search}"
            
        query += " GROUP BY w.lemma, w.language_code ORDER BY w.lemma LIMIT 50"
        
        cur.execute(query, (search,))
        results = cur.fetchall()
        
        if results:
            results_table = Table(box=box.ROUNDED)
            results_table.add_column("Word", style="bold yellow")
            results_table.add_column("Language", style="cyan")
            results_table.add_column("Definition", style="white")
            
            for word, lang, defs in results:
                results_table.add_row(
                    format_word_display(word),
                    lang,
                    Text(defs[:100] + "..." if defs and len(defs) > 100 else (defs or "-"))
                )
            
            console.print("\n", results_table)
        else:
            console.print("\n[yellow]No matching words found[/]")
            
        input("\nPress Enter to continue...")

def display_help():
    """Display help information about available commands."""
    console = Console()
    
    console.print("\n[bold cyan]📚 Dictionary Manager Help[/]\n")
    
    commands = Table(box=box.ROUNDED)
    commands.add_column("Command", style="bold yellow")
    commands.add_column("Description", style="white")
    commands.add_column("Example", style="cyan")
    
    commands.add_row(
        "migrate",
        "Reset and rebuild the database with fresh data",
        "python dictionary_manager.py migrate"
    )
    commands.add_row(
        "stats",
        "Display dictionary statistics and metrics",
        "python dictionary_manager.py stats"
    )
    commands.add_row(
        "inspect",
        "Look up detailed information about a word",
        "python dictionary_manager.py inspect --word \"bahay\""
    )
    commands.add_row(
        "explore",
        "Interactive exploration of the dictionary",
        "python dictionary_manager.py explore"
    )
    commands.add_row(
        "verify",
        "Run database integrity checks",
        "python dictionary_manager.py verify"
    )
    commands.add_row(
        "help",
        "Show this help information",
        "python dictionary_manager.py help"
    )
    
    console.print(commands)
    console.print("\n[bold green]For more information, visit: https://github.com/jrdndj/fil-relex[/]\n")

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
    """
    Process Kaikki dictionary entries from a JSONL file.
    Properly extracts synonyms, derived terms, etc., even if they are dictionaries.
    """
    logger.info(f"Starting to process Kaikki file: {filename}")

    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        return

    src = os.path.basename(filename)
    # Determine language code: "tl" if the filename is "kaikki.jsonl", otherwise "ceb"
    lang_code = "tl" if "kaikki.jsonl" in filename else "ceb"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {lang_code} entries"):
                try:
                    entry = json.loads(line)
                    lemma = entry.get("word", "").strip()
                    if not lemma:
                        continue

                    # Get the entry-level POS first
                    entry_pos = entry.get("pos", "").strip()

                    # Check for Baybayin script
                    has_baybayin = False
                    baybayin_form = entry.get("baybayin") or entry.get("script", {}).get("baybayin")
                    romanized = entry.get("romanized")

                    pronunciation = entry.get("pronunciation", "")
                    if pronunciation and any(char in pronunciation for char in "ᜀᜁᜂᜃᜄᜅᜆᜇᜈᜉᜊᜋᜌᜎᜏ"):
                        baybayin_form = baybayin_form or pronunciation

                    if baybayin_form:
                        has_baybayin = True
                        romanized = romanized or lemma

                    is_root = entry.get("is_root", False) or ("root word" in entry.get("tags", []))
                    root_word_id = None
                    if not is_root:
                        root_word_id = get_root_word_id(cur, lemma, lang_code)

                    word_id = get_or_create_word_id(
                        cur,
                        lemma,
                        lang_code,
                        has_baybayin=has_baybayin,
                        romanized_form=romanized,
                        root_word_id=root_word_id,
                        check_exists=check_exists,
                    )

                    if baybayin_form or pronunciation:
                        tag_lines = []
                        if baybayin_form:
                            tag_lines.append(f"baybayin: {baybayin_form}")
                        if pronunciation and pronunciation != baybayin_form:
                            tag_lines.append(f"pronunciation: {pronunciation}")

                        if tag_lines:
                            joined_tags = "\n".join(tag_lines)
                            cur.execute(
                                """
                                UPDATE words 
                                   SET tags = CASE 
                                       WHEN tags IS NULL THEN %s
                                       ELSE tags || E'\n' || %s
                                   END
                                 WHERE id = %s
                                """,
                                (joined_tags, joined_tags, word_id),
                            )

                    for sense in entry.get("senses", []):
                        pos = sense.get("pos", entry_pos).strip()
                        if has_baybayin and not pos:
                            pos = "Baybayin"

                        if not pos:
                            logger.debug(f"No part of speech found for sense in entry: {lemma}")
                            continue

                        pos = ", ".join(filter(None, [p.strip() for p in pos.split(",")]))

                        glosses = sense.get("glosses", [])
                        raw_examples = sense.get("examples", [])

                        examples = []
                        for ex in raw_examples:
                            if isinstance(ex, dict):
                                example_text = ex.get("text", "")
                                if example_text:
                                    examples.append(example_text)
                            elif isinstance(ex, str):
                                examples.append(ex)

                        sense_tags = sense.get("tags", [])

                        for gloss in glosses:
                            if gloss:
                                insert_definition(
                                    cur, 
                                    word_id, 
                                    gloss.strip(),
                                    pos,
                                    examples="\n".join(examples) if examples else None,
                                    usage_notes=", ".join(sense_tags) if sense_tags else None,
                                    sources=src
                                )

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

                    def extract_wordlist(lst):
                        results = []
                        for item in lst:
                            if isinstance(item, dict):
                                w = item.get("word", "").strip()
                            else:
                                w = str(item).strip()
                            if w:
                                results.append(w)
                        return results

                    derived_list = extract_wordlist(entry.get("derived", []))
                    for derived_lemma in derived_list:
                        derived_id = get_or_create_word_id(cur, derived_lemma, lang_code)
                        insert_relation(cur, word_id, derived_id, "derived_from", src)

                    related_list = extract_wordlist(entry.get("related", []))
                    for related_lemma in related_list:
                        related_id = get_or_create_word_id(cur, related_lemma, lang_code)
                        insert_relation(cur, word_id, related_id, "related", src)

                    synonyms_list = extract_wordlist(entry.get("synonyms", []))
                    for syn_lemma in synonyms_list:
                        syn_id = get_or_create_word_id(cur, syn_lemma, lang_code)
                        insert_relation(cur, word_id, syn_id, "synonym", src)

                    alt_forms = entry.get("forms", [])
                    for form_obj in alt_forms:
                        if isinstance(form_obj, dict):
                            form_str = form_obj.get("form", "")
                        else:
                            form_str = str(form_obj)
                        form_str = form_str.strip()
                        if form_str and form_str != lemma:
                            alt_id = get_or_create_word_id(cur, form_str, lang_code, preferred_spelling=lemma)
                            insert_relation(cur, word_id, alt_id, "alternative_form", src)

                    head_templates = entry.get("head_templates", [])
                    for template in head_templates:
                        template_name = template.get("name", "")
                        template_expansion = template.get("expansion", "")
                        if template_name and template_expansion:
                            tag_line = f"template: {template_name} - {template_expansion}"
                            cur.execute(
                                """
                                UPDATE words 
                                   SET tags = CASE 
                                       WHEN tags IS NULL THEN %s
                                       ELSE tags || E'\n' || %s
                                   END
                                 WHERE id = %s
                                """,
                                (tag_line, tag_line, word_id),
                            )

                    hyphenation = entry.get("hyphenation", [])
                    if hyphenation:
                        hyph_str = " ".join(hyphenation)
                        cur.execute(
                            """
                            UPDATE words 
                               SET tags = CASE 
                                   WHEN tags IS NULL THEN %s
                                   ELSE tags || E'\n' || %s
                               END
                             WHERE id = %s
                            """,
                            (f"hyphenation: {hyph_str}", f"hyphenation: {hyph_str}", word_id),
                        )

                    categories = entry.get("categories", [])
                    for cat_obj in categories:
                        if isinstance(cat_obj, dict):
                            category_name = cat_obj.get("name", "")
                        else:
                            category_name = str(cat_obj)
                        if category_name:
                            cur.execute(
                                """
                                UPDATE words 
                                   SET tags = CASE 
                                       WHEN tags IS NULL THEN %s
                                       ELSE tags || E'\n' || %s
                                   END
                                 WHERE id = %s
                                """,
                                (
                                    f"category: {category_name}",
                                    f"category: {category_name}",
                                    word_id
                                ),
                            )

                    hypernyms_list = extract_wordlist(entry.get("hypernyms", []))
                    for hypernym_lemma in hypernyms_list:
                        hypernym_id = get_or_create_word_id(cur, hypernym_lemma, lang_code)
                        insert_relation(cur, word_id, hypernym_id, "hypernym", src)

                    hyponyms_list = extract_wordlist(entry.get("hyponyms", []))
                    for hyponym_lemma in hyponyms_list:
                        hyponym_id = get_or_create_word_id(cur, hyponym_lemma, lang_code)
                        insert_relation(cur, word_id, hyponym_id, "hyponym", src)

                except Exception as e:
                    logger.error(f"Error processing entry '{lemma}': {str(e)}")
                    continue

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

if __name__ == "__main__":
    main()
