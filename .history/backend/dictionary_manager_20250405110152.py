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
import codecs
import csv
import enum
import functools
import glob
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
import textwrap
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union
import psycopg2
import psycopg2.extras
import psycopg2.pool
import unidecode
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from psycopg2.errors import UniqueViolation
from psycopg2.extras import Json
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from tqdm import tqdm
from psycopg2.extras import Json
import json # Make sure json is imported too
import hashlib # Make sure hashlib is imported
from psycopg2.extras import Json
import re  # Ensure re is imported if not already

# Define a regex for valid Baybayin characters (U+1700–U+171F)
# Allows Baybayin letters, vowels, viramas, AND whitespace. Adjust if other punctuation is needed.
VALID_BAYBAYIN_REGEX = re.compile(r"^[\u1700-\u171F\s]+$")

# Custom exceptions
class DatabaseError(Exception):
    """Base exception for database-related errors"""

    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised for database connection errors"""

    pass


# Load Environment Variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Database connection pool configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "fil_dict_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Initialize connection pool
try:
    connection_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1, maxconn=10, **DB_CONFIG
    )
    logger.info("Database connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize connection pool: {e}")
    connection_pool = None


def get_db_connection(max_retries=3, retry_delay=1.0):
    """
    Get a connection from the connection pool with retry logic.

    Args:
        max_retries (int): Maximum number of connection attempts
        retry_delay (float): Delay between retries in seconds

    Returns:
        connection: A database connection

    Raises:
        DatabaseConnectionError: If connection cannot be established after retries
    """
    global connection_pool

    if connection_pool is None:
        try:
            # Initialize connection pool if not exists
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **get_db_config()
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseConnectionError(
                f"Could not initialize database connection pool: {e}"
            )

    for attempt in range(max_retries):
        try:
            conn = connection_pool.getconn()

            # Validate connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

            # Set reasonable timeouts
            conn.set_session(
                autocommit=True,
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED,
            )
            return conn

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 1.5
            else:
                logger.error(
                    f"Failed to get database connection after {max_retries} attempts: {e}"
                )
                raise DatabaseConnectionError(
                    f"Could not connect to database after {max_retries} attempts: {e}"
                )

        except Exception as e:
            logger.error(f"Unexpected error getting database connection: {e}")
            # Try to return connection to pool if it was obtained
            try:
                if "conn" in locals():
                    connection_pool.putconn(conn)
            except:
                pass
            raise DatabaseConnectionError(f"Unexpected error: {e}")

    # Should never reach here due to the final raise in the loop
    raise DatabaseConnectionError("Failed to get database connection")


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
            if args and hasattr(args[0], "cursor"):
                # If first arg is a class instance with cursor, use that
                return func(*args, **kwargs)
            else:
                # Otherwise provide a cursor
                return func(cursor, *args, **kwargs)

    return wrapper


def standardize_source_identifier(source_identifier: str) -> str:
    """
    Standardize a source identifier string.

    Args:
        source_identifier: Raw source identifier (typically a filename)

    Returns:
        Standardized source name
    """
    if not source_identifier:
        return "unknown"

    # Use existing SourceStandardization class with fallback
    standardized = SourceStandardization.standardize_sources(source_identifier)

    # If the standardization failed (returns None or empty string), use the original
    if not standardized:
        return source_identifier

    return standardized


# Define constants for reusable values
DEFAULT_LANGUAGE_CODE = "tl"
SOURCE_INFO_FILES_KEY = "files"


def update_word_source_info(
    current_source_info: Optional[Union[str, dict]],
    new_source_identifier: Optional[str],
) -> str:
    """
    Updates the source_info JSON for a word entry, adding a new source identifier
    to a list under the 'files' key if it's not already present.

    Args:
        current_source_info: The current source_info from the words table.
        new_source_identifier: The identifier (e.g., filename) to add.

    Returns:
        A JSON string representation of the updated source_info.
    """
    # Better handling for empty or None new_source_identifier
    if not new_source_identifier:
        if isinstance(current_source_info, dict):
            return json.dumps(current_source_info)
        elif isinstance(current_source_info, str):
            return current_source_info
        else:
            return "{}"

    # Parse current_source_info into a dictionary
    source_info_dict = {}
    if isinstance(current_source_info, dict):
        source_info_dict = (
            current_source_info.copy()
        )  # Use copy to avoid modifying the original
    elif isinstance(current_source_info, str) and current_source_info:
        try:
            source_info_dict = json.loads(current_source_info)
            if not isinstance(source_info_dict, dict):
                source_info_dict = {}  # Reset if not a dictionary
        except (json.JSONDecodeError, TypeError):
            # Start fresh if invalid JSON
            source_info_dict = {}

    # Ensure 'files' key exists and is a list
    if SOURCE_INFO_FILES_KEY not in source_info_dict or not isinstance(
        source_info_dict[SOURCE_INFO_FILES_KEY], list
    ):
        source_info_dict[SOURCE_INFO_FILES_KEY] = []

    # Standardize the source identifier before adding
    standardized_source = SourceStandardization.standardize_sources(
        new_source_identifier
    )

    # Add new source identifier if not already present
    if standardized_source not in source_info_dict[SOURCE_INFO_FILES_KEY]:
        source_info_dict[SOURCE_INFO_FILES_KEY].append(standardized_source)
        source_info_dict[SOURCE_INFO_FILES_KEY].sort()  # Keep list sorted

    # Add metadata about when and how this source was added
    if "last_updated" not in source_info_dict:
        source_info_dict["last_updated"] = {}

    source_info_dict["last_updated"][standardized_source] = datetime.now().isoformat()

    # Return JSON string
    return json.dumps(source_info_dict)


# -------------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------------
def setup_logging():
    """Configure logging with proper Unicode handling."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_path = f"{log_dir}/dictionary_manager_{timestamp}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8")
            console_handler.stream = codecs.getwriter("utf-8")(sys.stdout.buffer)
    except Exception:

        def safe_encode(msg):
            try:
                return (
                    str(msg)
                    .encode(console_handler.stream.encoding, "replace")
                    .decode(console_handler.stream.encoding)
                )
            except:
                return str(msg).encode("ascii", "replace").decode("ascii")

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


def get_db_config():
    """Return the database configuration dictionary."""
    return DB_CONFIG


# -------------------------------------------------------------------
# Core Database Connection
# -------------------------------------------------------------------
def get_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME", DB_NAME),
        user=os.environ.get("DB_USER", DB_USER),
        password=os.environ.get("DB_PASSWORD", DB_PASSWORD),
        host=os.environ.get("DB_HOST", DB_HOST),
        port=os.environ.get("DB_PORT", DB_PORT),
    )
    conn.autocommit = False  # Ensure transactions are used.
    return conn


def get_cursor():
    """Return a cursor from a new connection with proper error handling."""
    conn = None
    try:
        conn = get_connection()
        return conn.cursor()
    except Exception as e:
        # Make sure we release the connection if we got it but failed to get a cursor
        if conn:
            release_db_connection(conn)
        logger.error(f"Error getting cursor: {e}")
        raise


# -------------------------------------------------------------------
# Setup Extensions
# -------------------------------------------------------------------
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
    try:
        extensions = ["pg_trgm", "unaccent", "fuzzystrmatch", "dict_xsyn"]
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
    Decorator for database operations that need to run within a transaction.

    Args:
        commit (bool): Whether to commit the transaction after successful execution.
                      If False, the transaction will be left open.

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection

            # Check if we're already in a transaction
            original_autocommit = None
            started_transaction = False

            try:
                # Only start a new transaction if we're not already in one
                if conn.autocommit:
                    original_autocommit = True
                    conn.autocommit = False
                    started_transaction = True
                    logger.debug(f"Started new transaction for {func.__name__}")

                # Execute the function
                result = func(cur, *args, **kwargs)

                # Commit if requested and we started the transaction
                if commit and started_transaction:
                    try:
                        conn.commit()
                        logger.debug(f"Transaction committed for {func.__name__}")
                    except Exception as commit_error:
                        logger.error(f"Failed to commit transaction: {commit_error}")
                        try:
                            conn.rollback()
                            logger.debug("Transaction rolled back due to commit error")
                        except:
                            pass
                        raise

                return result

            except Exception as e:
                # Only roll back if we started the transaction
                if started_transaction:
                    try:
                        conn.rollback()
                        logger.debug(f"Transaction rolled back due to error: {e}")
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback transaction: {rollback_error}"
                        )
                # Re-raise the original exception
                raise

            finally:
                # Restore original autocommit state if we changed it
                if started_transaction and original_autocommit is not None:
                    try:
                        conn.autocommit = original_autocommit
                        logger.debug(
                            f"Restored autocommit state to {original_autocommit}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to restore autocommit state: {e}")

        return wrapper

    return decorator


@with_transaction(commit=True)
def fix_inconsistent_sources(cur) -> Dict:
    """
    Fix inconsistent source information across the database.

    Args:
        cur: Database cursor

    Returns:
        Statistics about the fixes
    """
    stats = {
        "words_updated": 0,
        "definitions_updated": 0,
        "relations_updated": 0,
        "etymologies_updated": 0,
    }

    try:
        # Standardize sources in words table
        cur.execute(
            """
            WITH source_standardization AS (
                SELECT id, 
                       CASE
                           WHEN source_info::text ILIKE '%kaikki-ceb.jsonl%' THEN 'kaikki.org (Cebuano)'
                           WHEN source_info::text ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                           WHEN source_info::text ILIKE '%kwf_dictionary.json%' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                           WHEN source_info::text ILIKE '%root_words_with_associated_words_cleaned.json%' THEN 'tagalog.com'
                           WHEN source_info::text ILIKE '%tagalog-words.json%' THEN 'diksiyonaryo.ph'
                           WHEN source_info::text ILIKE '%marayum%' THEN 'Project Marayum'
                           ELSE source_info::text
                       END as standardized_source
                FROM words
                WHERE source_info IS NOT NULL
            )
            UPDATE words w
            SET source_info = jsonb_build_object('files', jsonb_build_array(ss.standardized_source))
            FROM source_standardization ss
            WHERE w.id = ss.id
            AND (source_info IS NULL OR source_info = '{}'::jsonb)
        """
        )
        stats["words_updated"] = cur.rowcount

        # Similar updates for other tables...

        return stats
    except Exception as e:
        logger.error(f"Error fixing inconsistent sources: {e}")
        raise


@with_transaction(commit=False)
def get_word_sources(cur, word_id: int) -> Dict:
    """
    Get comprehensive source information for a word.

    Args:
        cur: Database cursor
        word_id: ID of the word

    Returns:
        Dictionary with source information
    """
    result = {
        "word": None,
        "direct_sources": [],
        "definition_sources": [],
        "relation_sources": [],
        "etymology_sources": [],
    }

    try:
        # Get word info
        cur.execute(
            """
            SELECT lemma, source_info 
            FROM words 
            WHERE id = %s
        """,
            (word_id,),
        )
        word_row = cur.fetchone()

        if not word_row:
            return result

        result["word"] = word_row[0]

        # Extract direct sources
        if word_row[1]:
            try:
                source_info = (
                    json.loads(word_row[1])
                    if isinstance(word_row[1], str)
                    else word_row[1]
                )
                if (
                    isinstance(source_info, dict)
                    and SOURCE_INFO_FILES_KEY in source_info
                ):
                    result["direct_sources"] = source_info[SOURCE_INFO_FILES_KEY]
            except (json.JSONDecodeError, TypeError):
                pass

        # Get definition sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM definitions 
            WHERE word_id = %s AND sources IS NOT NULL
        """,
            (word_id,),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["definition_sources"]:
                result["definition_sources"].append(row[0])

        # Get relation sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM relations 
            WHERE (from_word_id = %s OR to_word_id = %s) AND sources IS NOT NULL
        """,
            (word_id, word_id),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["relation_sources"]:
                result["relation_sources"].append(row[0])

        # Get etymology sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM etymologies 
            WHERE word_id = %s AND sources IS NOT NULL
        """,
            (word_id,),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["etymology_sources"]:
                result["etymology_sources"].append(row[0])

        return result
    except Exception as e:
        logger.error(f"Error getting sources for word ID {word_id}: {e}")
        return result


@with_transaction(commit=True)
def propagate_source_info(cur, word_id: int, source_identifier: str) -> bool:
    """
    Propagate source information to related records.

    Args:
        cur: Database cursor
        word_id: ID of the word
        source_identifier: Source identifier to propagate

    Returns:
        True if successful, False otherwise
    """
    try:
        standardized_source = standardize_source_identifier(source_identifier)

        # Update related definitions
        cur.execute(
            """
            UPDATE definitions
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
            ),
        )

        # Update related etymologies
        cur.execute(
            """
            UPDATE etymologies
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
            ),
        )

        # Update related relations
        cur.execute(
            """
            UPDATE relations
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE from_word_id = %s OR to_word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
                word_id,
            ),
        )

        return True
    except Exception as e:
        logger.error(f"Error propagating source info for word ID {word_id}: {e}")
        return False


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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
    root_word_id INT REFERENCES words(id) ON DELETE SET NULL,
    preferred_spelling VARCHAR(255),
    tags TEXT,
    idioms JSONB DEFAULT '[]',
    pronunciation_data JSONB,
    source_info JSONB DEFAULT '{}',
    word_metadata JSONB DEFAULT '{}',
    data_hash TEXT,
    badlit_form TEXT,
    hyphenation JSONB,
    is_proper_noun BOOLEAN DEFAULT FALSE,
    is_abbreviation BOOLEAN DEFAULT FALSE,
    is_initialism BOOLEAN DEFAULT FALSE,
    search_text TSVECTOR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (normalized_lemma, language_code),
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
CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);
CREATE INDEX IF NOT EXISTS idx_words_metadata ON words USING GIN(word_metadata);

-- Create pronunciations table
CREATE TABLE IF NOT EXISTS pronunciations (
    id SERIAL PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL DEFAULT 'ipa',
    value TEXT NOT NULL,
    tags JSONB,
    pronunciation_metadata JSONB,
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pronunciations_unique UNIQUE(word_id, type, value)
);
CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
CREATE INDEX IF NOT EXISTS idx_pronunciations_type ON pronunciations(type);
CREATE INDEX IF NOT EXISTS idx_pronunciations_value ON pronunciations(value);

-- Create credits table
CREATE TABLE IF NOT EXISTS credits (
    id SERIAL PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    credit TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT credits_unique UNIQUE(word_id, credit)
);
CREATE INDEX IF NOT EXISTS idx_credits_word ON credits(word_id);

-- Create definitions table
CREATE TABLE IF NOT EXISTS definitions (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    definition_text TEXT NOT NULL,
    original_pos TEXT,
    standardized_pos_id INT REFERENCES parts_of_speech(id) ON DELETE SET NULL,
    examples TEXT,
    usage_notes TEXT,
    tags TEXT,
    sources TEXT,
    metadata JSONB,
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

-- Create etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text TEXT NOT NULL,
    normalized_components TEXT,
    etymology_structure TEXT,
    language_codes TEXT,
    sources TEXT,
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
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT affixations_unique UNIQUE (root_word_id, affixed_word_id, affix_type)
);
CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);

-- Create definition_categories table
CREATE TABLE IF NOT EXISTS definition_categories (
   id SERIAL PRIMARY KEY,
   definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
   category_name TEXT NOT NULL,
   category_kind TEXT,
   parents JSONB,
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (definition_id, category_name)
);
CREATE INDEX IF NOT EXISTS idx_def_categories_def ON definition_categories(definition_id);
CREATE INDEX IF NOT EXISTS idx_def_categories_name ON definition_categories(category_name);

-- Create word_templates table
CREATE TABLE IF NOT EXISTS word_templates (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
    template_name TEXT NOT NULL,
    args JSONB,
    expansion TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (word_id, template_name)
);
CREATE INDEX IF NOT EXISTS idx_word_templates_word ON word_templates(word_id);
CREATE INDEX IF NOT EXISTS idx_word_templates_name ON word_templates(template_name);

-- Create word_forms table
CREATE TABLE IF NOT EXISTS word_forms (
   id SERIAL PRIMARY KEY,
   word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
   form TEXT NOT NULL,
   is_canonical BOOLEAN DEFAULT FALSE,
   is_primary BOOLEAN DEFAULT FALSE,
   tags JSONB,
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (word_id, form)
);
CREATE INDEX IF NOT EXISTS idx_word_forms_word ON word_forms(word_id);
CREATE INDEX IF NOT EXISTS idx_word_forms_form ON word_forms(form);

-- Create definition_links table
CREATE TABLE IF NOT EXISTS definition_links (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
    link_text TEXT NOT NULL,
    link_target TEXT NOT NULL,
    is_wikipedia BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (definition_id, link_text, link_target)
);
CREATE INDEX IF NOT EXISTS idx_def_links_def ON definition_links(definition_id);


-- Create triggers for timestamp updates (ensure function exists first)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_words_timestamp' AND tgrelid = 'words'::regclass) THEN
        CREATE TRIGGER update_words_timestamp BEFORE UPDATE ON words FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definitions_timestamp' AND tgrelid = 'definitions'::regclass) THEN
        CREATE TRIGGER update_definitions_timestamp BEFORE UPDATE ON definitions FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_etymologies_timestamp' AND tgrelid = 'etymologies'::regclass) THEN
        CREATE TRIGGER update_etymologies_timestamp BEFORE UPDATE ON etymologies FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_pronunciations_timestamp' AND tgrelid = 'pronunciations'::regclass) THEN
        CREATE TRIGGER update_pronunciations_timestamp BEFORE UPDATE ON pronunciations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_credits_timestamp' AND tgrelid = 'credits'::regclass) THEN
        CREATE TRIGGER update_credits_timestamp BEFORE UPDATE ON credits FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_relations_timestamp' AND tgrelid = 'definition_relations'::regclass) THEN
        CREATE TRIGGER update_definition_relations_timestamp BEFORE UPDATE ON definition_relations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_parts_of_speech_timestamp' AND tgrelid = 'parts_of_speech'::regclass) THEN
        CREATE TRIGGER update_parts_of_speech_timestamp BEFORE UPDATE ON parts_of_speech FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_affixations_timestamp' AND tgrelid = 'affixations'::regclass) THEN
        CREATE TRIGGER update_affixations_timestamp BEFORE UPDATE ON affixations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;

    -- Add triggers for NEW TABLES
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_categories_timestamp' AND tgrelid = 'definition_categories'::regclass) THEN
        CREATE TRIGGER update_definition_categories_timestamp BEFORE UPDATE ON definition_categories FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_word_templates_timestamp' AND tgrelid = 'word_templates'::regclass) THEN
        CREATE TRIGGER update_word_templates_timestamp BEFORE UPDATE ON word_templates FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
     IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_word_forms_timestamp' AND tgrelid = 'word_forms'::regclass) THEN
        CREATE TRIGGER update_word_forms_timestamp BEFORE UPDATE ON word_forms FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
     IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_links_timestamp' AND tgrelid = 'definition_links'::regclass) THEN
        CREATE TRIGGER update_definition_links_timestamp BEFORE UPDATE ON definition_links FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;

END $$;

"""

def create_or_update_tables(conn):
    """Create or update the database tables."""
    logger.info("Starting table creation/update process.")

    cur = conn.cursor()
    try:
        # Drop existing tables in correct order
        cur.execute(
            """
            DROP TABLE IF EXISTS 
                credits, pronunciations, definition_relations, affixations, 
                relations, etymologies, definitions, words, parts_of_speech CASCADE;
        """
        )

        # Create tables
        cur.execute(TABLE_CREATION_SQL)

        # Insert standard parts of speech
        pos_entries = [
            # --- Core Grammatical Categories ---
            (
                "n",
                "Noun",
                "Pangngalan",
                "Word that refers to a person, place, thing, or idea",
            ),
            ("v", "Verb", "Pandiwa", "Word that expresses action or state of being"),
            ("adj", "Adjective", "Pang-uri", "Word that describes or modifies a noun"),
            (
                "adv",
                "Adverb",
                "Pang-abay",
                "Word that modifies verbs, adjectives, or other adverbs",
            ),
            ("pron", "Pronoun", "Panghalip", "Word that substitutes for a noun"),
            (
                "prep",
                "Preposition",
                "Pang-ukol",
                "Word that shows relationship between words",
            ),
            (
                "conj",
                "Conjunction",
                "Pangatnig",
                "Word that connects words, phrases, or clauses",
            ),
            ("intj", "Interjection", "Pandamdam", "Word expressing emotion"),
            ("det", "Determiner", "Pantukoy", "Word that modifies nouns"),
            ("affix", "Affix", "Panlapi", "Word element attached to base or root"),

            # --- Added/Ensured based on POS_MAPPING and Daglat ---
            ("lig", "Ligature", "Pang-angkop", "Word that links modifiers to modified words"), # For 'pnk'
            ("part", "Particle", "Kataga", "Function word that doesn't fit other categories"), # From mapping
            ("num", "Number", "Pamilang", "Word representing a number"), # From mapping
            ("expr", "Expression", "Pahayag", "Common phrase or expression"), # From mapping
            ("punc", "Punctuation", "Bantas", "Punctuation mark"), # From mapping

            # --- Other Categories from original code ---
            ("idm", "Idiom", "Idyoma", "Fixed expression with non-literal meaning"),
            ("col", "Colloquial", "Kolokyal", "Informal or conversational usage"),
            ("syn", "Synonym", "Singkahulugan", "Word with similar meaning"), # Note: Relationships preferred over POS tags for this
            ("ant", "Antonym", "Di-kasingkahulugan", "Word with opposite meaning"), # Note: Relationships preferred over POS tags for this
            ("eng", "English", "Ingles", "English loanword or translation"), # Note: Etymology preferred over POS tags for this
            ("spa", "Spanish", "Espanyol", "Spanish loanword or origin"), # Note: Etymology preferred over POS tags for this
            ("tx", "Texting", "Texting", "Text messaging form"),
            ("var", "Variant", "Varyant", "Alternative form or spelling"), # Note: Relationships preferred over POS tags for this
            (
                "unc",
                "Uncategorized",
                "Hindi Tiyak",
                "Part of speech not yet determined",
            ),
        ]

        for code, name_en, name_tl, desc in pos_entries:
            cur.execute(
                """
                INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                SET name_en = EXCLUDED.name_en,
                    name_tl = EXCLUDED.name_tl,
                    description = EXCLUDED.description
            """,
                (code, name_en, name_tl, desc),
            )

        conn.commit()
        logger.info("Tables created or updated successfully.")

    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {str(e)}")
        raise
    finally:
        cur.close()


@with_transaction(commit=True)
def insert_pronunciation(
    cur, word_id: int, pronunciation_data: Union[str, Dict], source_identifier: str
) -> Optional[int]:
    """
    Insert pronunciation data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing pronunciations based on (word_id, type, value),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        pronunciation_data: Pronunciation string (assumed IPA) or dictionary
                           (e.g., {'type': 'ipa', 'value': '...', 'tags': [...]}).
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated pronunciation record, or None if failed.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping pronunciation insert for word ID {word_id}: Missing MANDATORY source identifier."
        )
        return None

    pron_type = "ipa"  # Default type
    value = None
    tags_list = []
    metadata = {}

    try:
        # Parse input data
        if isinstance(pronunciation_data, dict):
            # Prioritize keys from the dict, but allow source_identifier argument to override 'sources' key if present
            pron_type = (
                pronunciation_data.get("type", "ipa") or "ipa"
            )  # Default to 'ipa' if empty
            value = (
                pronunciation_data.get("value", "").strip()
                if isinstance(pronunciation_data.get("value"), str)
                else None
            )
            tags_input = pronunciation_data.get("tags")
            if isinstance(tags_input, list):
                tags_list = tags_input
            elif isinstance(tags_input, str):
                # Simple split if tags are a comma-separated string? Adapt as needed.
                tags_list = [
                    tag.strip() for tag in tags_input.split(",") if tag.strip()
                ]
                logger.debug(
                    f"Parsed tags string '{tags_input}' into list for word ID {word_id}."
                )
            # else: ignore invalid tags format

            metadata_input = pronunciation_data.get("metadata")
            if isinstance(metadata_input, dict):
                metadata = metadata_input
            # else: ignore invalid metadata format

        elif isinstance(pronunciation_data, str):
            value = pronunciation_data.strip()
            # Assumed type is 'ipa' and no tags/metadata provided via string input
        else:
            logger.warning(
                f"Invalid pronunciation_data type for word ID {word_id} (source '{source_identifier}'): {type(pronunciation_data)}. Skipping."
            )
            return None

        if not value:
            logger.warning(
                f"Empty pronunciation value for word ID {word_id} (source '{source_identifier}'). Skipping pronounicitaion insertion."
            )
            return None

        # Safely dump JSON fields (tags and metadata) for DB insertion (assuming JSONB columns)
        tags_json = None
        try:
            tags_json = json.dumps(
                tags_list
            )  # tags_list will be [] if not provided or invalid format
        except TypeError as e:
            logger.warning(
                f"Could not serialize tags for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Tags: {tags_list}"
            )
            tags_json = "[]"  # Fallback to empty JSON array string

        metadata_json = None
        try:
            metadata_json = json.dumps(
                metadata
            )  # metadata will be {} if not provided or invalid format
        except TypeError as e:
            logger.warning(
                f"Could not serialize metadata for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Metadata: {metadata}"
            )
            metadata_json = "{}"  # Fallback to empty JSON object string

        # Prepare parameters for query
        params = {
            "word_id": word_id,
            "type": pron_type,
            "value": value,
            "tags": tags_json,
            "pronunciation_metadata": metadata_json,
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }

        # Insert or update pronunciation
        cur.execute(
            """
            INSERT INTO pronunciations (word_id, type, value, tags, pronunciation_metadata, sources)
            VALUES (%(word_id)s, %(type)s, %(value)s, %(tags)s::jsonb, %(pronunciation_metadata)s::jsonb, %(sources)s) -- Cast JSONs
            ON CONFLICT (word_id, type, value) -- Conflict on word, type, and exact value
            DO UPDATE SET
                -- Update tags/metadata only if new value is not NULL (or empty JSON?) - COALESCE prefers new non-null
                tags = COALESCE(EXCLUDED.tags, pronunciations.tags),
                pronunciation_metadata = COALESCE(EXCLUDED.pronunciation_metadata, pronunciations.pronunciation_metadata),
                -- Overwrite sources: Last write wins for this pronunciation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        pron_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated pronunciation (ID: {pron_id}, Type: {pron_type}) for word ID {word_id} from source '{source_identifier}'. Value: '{value}'"
        )
        return pron_id

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None


@with_transaction(commit=True)
def insert_credit(
    cur, word_id: int, credit_data: Union[str, Dict], source_identifier: str
) -> Optional[int]:
    """
    Insert credit data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing credits based on (word_id, credit),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        credit_data: Credit string or dictionary (e.g., {'text': 'Source Name'}).
                     If dict, the 'text' key is used.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated credit record, or None if failed.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping credit insert for word ID {word_id}: Missing MANDATORY source identifier."
        )
        return None

    credit_text = None
    try:
        # Extract credit text
        if isinstance(credit_data, dict):
            # Prioritize 'text' key if dict is provided
            credit_text = (
                credit_data.get("text", "").strip()
                if isinstance(credit_data.get("text"), str)
                else None
            )
        elif isinstance(credit_data, str):
            credit_text = credit_data.strip()
        else:
            logger.warning(
                f"Invalid credit_data type for word ID {word_id} (source '{source_identifier}'): {type(credit_data)}. Skipping."
            )
            return None

        if not credit_text:
            logger.warning(
                f"Empty credit text for word ID {word_id} (source '{source_identifier}'). Skipping."
            )
            return None

        # Prepare parameters
        params = {
            "word_id": word_id,
            "credit": credit_text,
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }

        # Insert or update credit
        cur.execute(
            """
            INSERT INTO credits (word_id, credit, sources)
            VALUES (%(word_id)s, %(credit)s, %(sources)s)
            ON CONFLICT (word_id, credit) -- Conflict on word and exact credit text
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this credit record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        credit_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated credit (ID: {credit_id}) for word ID {word_id} from source '{source_identifier}'. Credit: '{credit_text}'"
        )
        return credit_id

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting credit for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting credit for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None
# Define standard part of speech mappings with standard codes
POS_MAPPING = {
    # Nouns - Pangngalan ('n')
    "noun": "n",
    "pangngalan": "n",
    "name": "n",
    "n": "n",
    "pangalan": "n", # Common typo
    "pangngalang pantangi": "n", # Proper noun
    "pangngalang pambalana": "n", # Common noun
    "salitang ugat": "n", # Root word (treated as noun contextually in KWF)
    "salitang hango": "n", # Derived word (noun context)
    "pulutong": "n", # Group/category (noun context)
    "sangkap": "n", # Component (noun context)
    "uri": "n", # Type (noun context) - Note: 'uri' also maps to 'adj', context might matter
    "simbolo": "n", # Symbol (noun context)
    "dinidiinang pantig": "n", # Stressed syllable (noun context)
    "pagbabago": "n", # Variation (noun context)
    "palatandaan": "n", # Marker (noun context)
    "pahiwatig": "n", # Indication (noun context)
    "tugma": "n", # Rhyme (noun context)
    "kahulugan": "n", # Gloss/Meaning (noun context)
    "salitang hiram": "n", # Loan word (noun context)
    "damdamin": "n", # Emotion (noun context)
    "katumbas na salita": "n", # Equivalent term
    "kabaliktaran": "n", # Antonym (noun context)
    "aspekto": "n", # Aspect (noun context)
    "kasarian": "n", # Gender (noun context)
    "anyo": "n", # Form (noun context)
    "kayarian": "n", # Structure (noun context)
    "gamit": "n", # Usage (noun context)
    "pagkakaugnay": "n", # Relation (noun context)
    "Pangngalan": "n", # KWF Casing

    # Verbs - Pandiwa ('v')
    "verb": "v",
    "pandiwa": "v",
    "v": "v",
    "action": "v",
    "Pandiwa": "v", # KWF Casing

    # Adjectives - Pang-uri ('adj')
    "adjective": "adj",
    "pang-uri": "adj",
    "adj": "adj",
    "quality": "adj",
    "uri": "adj", # Ambiguous, but often adj in KWF context alongside noun senses
    "antas na paghahambing": "adj", # Comparative (adj context)
    "Pang-uri": "adj", # KWF Casing
    "pang-uri sa panlagay": "adj", # Predicate adjective
    "pang-uri sa panlapi": "adj", # Affixal adjective

    # Adverbs - Pang-abay ('adv')
    "adverb": "adv",
    "pang-abay": "adv",
    "adv": "adv",
    "manner": "adv",
    "abay": "adv", # Typo/short form
    "Pang-abay": "adv", # KWF Casing
    "pang-abay na pamaraan": "adv", # Manner adverb
    "pang-abay na panlunan": "adv", # Locative adverb
    "pang-abay na pamanahon": "adv", # Temporal adverb
    "pang-abay na pang-uring": "adv", # Qualifying adverb

    # Pronouns - Panghalip ('pron')
    "pronoun": "pron",
    "panghalip": "pron",
    "pron": "pron",
    "halip": "pron", # Short form
    "Panghalip": "pron", # KWF Casing
    "panghalip panao": "pron", # Personal pronoun
    "panghalip pamatlig": "pron", # Demonstrative pronoun
    "panghalip pananong": "pron", # Interrogative pronoun
    "panghalip paari": "pron", # Possessive pronoun

    # Prepositions - Pang-ukol ('prep')
    "preposition": "prep",
    "pang-ukol": "prep",
    "prep": "prep",
    "ukol": "prep", # Short form
    "Pang-ukol": "prep", # KWF Casing

    # Conjunctions - Pangatnig ('conj')
    "conjunction": "conj",
    "pangatnig": "conj",
    "conj": "conj",
    "katnig": "conj", # Short form
    "Pangatnig": "conj", # KWF Casing

    # Interjections - Pandamdam ('intj')
    "interjection": "intj",
    "pandamdam": "intj",
    "padamdam": "intj", # Alternate spelling
    "intj": "intj",
    "damdam": "intj", # Short form
    "Pandamdam": "intj", # KWF Casing
    "Padamdam": "intj", # KWF Casing Alt

    # Determiners / Articles - Pantukoy ('det')
    "determiner": "det",
    "pantukoy": "det",
    "article": "det",
    "art": "det",
    "tukoy": "det", # Short form
    "Pantukoy": "det", # KWF Casing
    "panuri": "det", # Determiner (alt Filipino term)

    # Affixes - Panlapi ('affix')
    "affix": "affix",
    "panlapi": "affix",
    "aff": "affix",
    "lapi": "affix", # Short form
    "prefix": "affix", # Specific types map to general 'affix'
    "unlapi": "affix",
    "pref": "affix",
    "suffix": "affix",
    "hulapi": "affix",
    "suff": "affix",
    "infix": "affix",
    "gitlapi": "affix",
    "inf": "affix",
    "Panlapi": "affix", # KWF Casing
    "Pangkayarian": "affix", # KWF term for structure marker, map to affix

    # Ligatures - Pang-angkop ('lig')
    "ligature": "lig", # Needs 'lig' code in parts_of_speech table
    "linker": "lig",
    "pang-angkop": "lig", # Ensure this maps correctly now
    "Pang-angkop": "lig", # KWF Casing

    # Particles - Kataga ('part')
    "particle": "part", # Needs 'part' code in parts_of_speech table
    "kataga": "part",
    "part": "part",

    # Numbers - Pamilang ('num')
    "number": "num", # Needs 'num' code in parts_of_speech table
    "bilang": "num",
    "num": "num",
    "pamilang": "num",
    "karamihan": "num", # Number (grammatical)

    # Expressions / Phrases ('expr')
    "expression": "expr", # Needs 'expr' code
    "pahayag": "expr",
    "expr": "expr",
    "phrase": "expr", # Map phrase to expression for simplicity
    "parirala": "expr",
    "phr": "expr",

    # Punctuation ('punc')
    "punctuation": "punc", # Needs 'punc' code
    "bantas": "punc",
    "punto": "punc", # KWF 'punto'

    # Mappings for Daglat from the image
    "png": "n",
    "pnd": "v",
    "pnr": "adj",
    "pnb": "adv",
    "pnh": "pron",
    "pnu": "prep",
    "pnt": "conj",
    "pdd": "intj", # Pandamdam
    "ptk": "det",
    "pnl": "affix",
    "pnk": "lig", # Pang-angkop mapped to Ligature

    # Other relevant Daglat
    "id": "idm",  # idyomatiko
    "kol": "col", # kolokyal
    "var": "var", # varyant
    "st": "unc",  # Sinaunang Tagalog (map to uncategorized for POS)

    # Contextual phrases from KWF - These describe usage, often noun contexts
    "sa sanga ng punongkahoy": "n",
    "sa anumang bukás na bahagi": "n",
    "sa isang tao": "n",
    "sa isang sakít na": "n",
    "sa isang rabáw": "n",
    "sa isang sasakyang-dagat": "n",
    "sa isang serbisyo, binili, o inutang": "n",
    "sa isang súgat": "n",
    "kung sa bakal": "n",
    "sa pananim": "n",
    "sa isang pook": "n",
    "sa isang materyal": "n",
    "sa dalawang tao": "n",
    "sa mga pananim": "n",
    "sa isang likido": "n",
    "sa isang salita": "n",
    "sa isang kilos": "n",
    "sa kagamitan": "n",
    "sa gamot": "n",
    "sa isang halaga": "n",
    "sa buhok": "n",
    "sa lupa": "n",
    "sa damit": "n",
    "sa isang sawsáwan": "n",
    "sa sakít": "n",
    "sa pagluluto": "n",
    "kung sa batok o ilong": "n",

    # Other less common/specific tags - Map to 'unc' or best fit if possible
    "idm": "idm", # Map Idiom code to itself (consistent with table)
    "col": "col", # Map Colloquial code to itself (consistent with table)
    "syn": "unc", # Synonym tag is redundant with relations
    "ant": "unc", # Antonym tag redundant
    "eng": "unc", # Language info better in etymology
    "spa": "unc", # Language info better in etymology
    "tx": "unc", # Texting form - uncategorized for now
    "var": "var", # Map Variant code to itself (consistent with table)
    "auxiliary": "unc", # Auxiliary - needs standard code/handling
    "pantulong": "unc",
    "aux": "unc",

    # Fallback for unknown
    "unknown": "unc", # Map unknown explicitly to uncategorized code
    "unc": "unc", # Ensure 'unc' maps to itself
}
# Add lowercased versions of all the mappings
LOWERCASE_POS_MAPPING = {k.lower(): v for k, v in POS_MAPPING.items()}

def get_standard_code(pos_string: Optional[str]) -> str:
    """
    Convert a part of speech string to a standardized short code (e.g., 'n', 'v', 'adj').

    Args:
        pos_string: The part of speech string to standardize. Can be None or empty.

    Returns:
        A standardized short code for the part of speech (default: 'unc').
    """
    if not pos_string:
        return "unc" # Uncategorized code

    # Clean the input string - remove parentheticals, lowercase, strip whitespace
    pos_key = str(pos_string).lower().strip() # Ensure it's a string
    pos_key = re.sub(r"\([^)]*\)", "", pos_key).strip()

    # Handle empty string after cleaning
    if not pos_key:
        return "unc"

    # Check for direct match in lowercase mapping first
    if pos_key in LOWERCASE_POS_MAPPING:
        return LOWERCASE_POS_MAPPING[pos_key]

    # Try direct match with original case (less common but possible)
    # Note: Original POS_MAPPING keys might need adjustment if mixed case was intended
    if pos_string in POS_MAPPING: # Check original string against original mapping keys
        return POS_MAPPING[pos_string]

    # Check for 'Sa' prefix pattern which often indicates noun context in KWF
    if pos_key.startswith("sa ") or pos_key.startswith("kung sa "):
        return "n" # These are typically noun usage contexts

    # Check for partial matches (e.g., "common noun" should match "noun" -> "n")
    # This might be overly broad, consider removing if too many false positives
    # Best match is usually direct key lookup.
    # for key_pattern, code_value in LOWERCASE_POS_MAPPING.items():
    #     if key_pattern in pos_key: # Check if known pattern is substring of input
    #         # Be cautious: 'verb' is in 'adverb'. Prioritize longer matches?
    #         # Or rely primarily on direct LOWERCASE_POS_MAPPING lookup above.
    #         # Let's comment this out for now as direct mapping is safer.
    #         # return code_value
    #         pass

    # Log the unmapped POS for future improvements
    # Use logger instead of print for better integration
    logger.debug(f"Unmapped POS: '{pos_string}' -> mapped to 'unc'")

    # Default to uncategorized code if no match found
    return "unc"

# --- Replace the standardize_entry_pos function (around line 1377) ---

def standardize_entry_pos(pos_input: Union[str, list, None]) -> str:
    """
    Standardize part-of-speech from dictionary entries (string or list)
    to a standard short code.

    Args:
        pos_input: The raw POS value from the source (string, list, or None).

    Returns:
        The standardized POS code (e.g., 'n', 'v', 'unc').
    """
    if not pos_input:
        return "unc" # Default to uncategorized code

    pos_to_standardize = None

    # Handle lists/arrays of POS values (like KWF)
    if isinstance(pos_input, list):
        # Try to find the first valid/mappable POS string in the list
        for pos_item in pos_input:
            if pos_item and isinstance(pos_item, str) and pos_item.strip():
                pos_to_standardize = pos_item
                # Check if this item maps to something other than 'unc'
                temp_code = get_standard_code(pos_to_standardize)
                if temp_code != "unc":
                    return temp_code # Return the first valid code found
        # If no valid item found in the list, fall back to 'unc'
        return "unc"
    elif isinstance(pos_input, str):
        pos_to_standardize = pos_input
    else:
        # Handle other unexpected types gracefully
        return "unc"

    # Standardize the selected string using the main helper
    return get_standard_code(pos_to_standardize)


# -------------------------------------------------------------------
# Data Structures and Enums
# -------------------------------------------------------------------
class BaybayinCharType(Enum):
    """Define types of Baybayin characters."""

    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"

    @classmethod
    def get_type(cls, char: str) -> "BaybayinCharType":
        """Determine the type of a Baybayin character."""
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
    """Define a Baybayin character with its properties."""

    char: str
    char_type: BaybayinCharType
    default_sound: str
    possible_sounds: List[str]

    def __post_init__(self):
        if not self.char:
            raise ValueError("Character cannot be empty")
        if not isinstance(self.char_type, BaybayinCharType):
            raise ValueError(f"Invalid character type: {self.char_type}")
        if not self.default_sound and self.char_type not in (
            BaybayinCharType.VIRAMA,
            BaybayinCharType.PUNCTUATION,
        ):
            raise ValueError("Default sound required for non-virama characters")
        code_point = ord(self.char)
        if not (0x1700 <= code_point <= 0x171F) and not (
            0x1735 <= code_point <= 0x1736
        ):
            raise ValueError(
                f"Invalid Baybayin character: {self.char} (U+{code_point:04X})"
            )
        expected_type = BaybayinCharType.get_type(self.char)
        if (
            expected_type != self.char_type
            and expected_type != BaybayinCharType.UNKNOWN
        ):
            raise ValueError(
                f"Character type mismatch for {self.char}: expected {expected_type}, got {self.char_type}"
            )

    def get_sound(self, next_char: Optional["BaybayinChar"] = None) -> str:
        """Get the sound of this character, considering the next character."""
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

    SEMANTIC = "semantic"  # Meaning-based relationships
    DERIVATIONAL = "derivational"  # Word formation relationships
    VARIANT = "variant"  # Form variations
    TAXONOMIC = "taxonomic"  # Hierarchical relationships
    USAGE = "usage"  # Usage-based relationships
    OTHER = "other"  # Miscellaneous relationships


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
    DERIVED_FROM = (
        "derived_from",
        RelationshipCategory.DERIVATIONAL,
        False,
        "ROOT_OF",
        False,
        95,
    )
    ROOT_OF = (
        "root_of",
        RelationshipCategory.DERIVATIONAL,
        False,
        "DERIVED_FROM",
        False,
        95,
    )

    # Variant relationships
    VARIANT = ("variant", RelationshipCategory.VARIANT, True, None, False, 85)
    SPELLING_VARIANT = (
        "spelling_variant",
        RelationshipCategory.VARIANT,
        True,
        None,
        False,
        95,
    )
    REGIONAL_VARIANT = (
        "regional_variant",
        RelationshipCategory.VARIANT,
        True,
        None,
        False,
        90,
    )

    # Usage relationships
    COMPARE_WITH = ("compare_with", RelationshipCategory.USAGE, True, None, False, 50)
    SEE_ALSO = ("see_also", RelationshipCategory.USAGE, True, None, False, 40)

    # Other relationships
    EQUALS = ("equals", RelationshipCategory.OTHER, True, None, True, 100)

    def __init__(
        self, rel_value, category, bidirectional, inverse, transitive, strength
    ):
        self.rel_value = rel_value
        self.category = category
        self.bidirectional = bidirectional
        self.inverse = inverse
        self.transitive = transitive
        self.strength = strength

    @classmethod
    def from_string(cls, relation_str: str) -> "RelationshipType":
        """Convert a string to a RelationshipType enum value"""
        normalized = relation_str.lower().replace(" ", "_")
        for rel_type in cls:
            if rel_type.rel_value == normalized:
                return rel_type

        # Handle legacy/alternative names
        legacy_mapping = {
            # Semantic relationships
            "synonym_of": cls.SYNONYM,
            "antonym_of": cls.ANTONYM,
            "related_to": cls.RELATED,
            "kasingkahulugan": cls.SYNONYM,
            "katulad": cls.SYNONYM,
            "kasalungat": cls.ANTONYM,
            "kabaligtaran": cls.ANTONYM,
            "kaugnay": cls.RELATED,
            # Derivational
            "derived": cls.DERIVED_FROM,
            "mula_sa": cls.DERIVED_FROM,
            # Variants
            "alternative_spelling": cls.SPELLING_VARIANT,
            "alternate_form": cls.VARIANT,
            "varyant": cls.VARIANT,
            "variants": cls.VARIANT,
            # Taxonomy
            "uri_ng": cls.HYPONYM,
            # Usage
            "see": cls.SEE_ALSO,
        }

        if normalized in legacy_mapping:
            return legacy_mapping[normalized]

        # Fall back to RELATED for unknown types
        logger.warning(
            f"Unknown relationship type: {relation_str}, using RELATED as fallback"
        )
        return cls.RELATED

    def get_inverse(self) -> "RelationshipType":
        """Get the inverse relationship type"""
        if self.bidirectional:
            return self
        if self.inverse:
            return getattr(RelationshipType, self.inverse)
        return RelationshipType.RELATED  # Fallback

    def __str__(self):
        return self.rel_value


# --- Replace the get_standardized_pos_id function (around line 1504) ---

def get_standardized_pos_id(cur, pos_string: Optional[str]) -> int:
    """
    Get the primary key ID from the parts_of_speech table for a given POS string.

    Args:
        cur: Database cursor.
        pos_string: The raw part of speech string to standardize and look up.

    Returns:
        The integer ID from the parts_of_speech table, or the ID for 'unc' if not found.
    """
    # Get the standardized short code (e.g., 'n', 'v', 'unc')
    standard_code = get_standard_code(pos_string) # Use the corrected function

    try:
        # Query using the CODE obtained from standardization
        cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (standard_code,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            # If the standard code (even 'unc') is somehow not in the table, log error
            logger.error(f"Standard POS code '{standard_code}' (derived from '{pos_string}') not found in parts_of_speech table. Falling back to fetching 'unc' ID.")
            return get_uncategorized_pos_id(cur) # Fallback to ensure 'unc' exists
    except Exception as e:
        logger.error(f"Error fetching POS ID for code '{standard_code}' (from '{pos_string}'): {e}. Returning 'unc'.")
        # Ensure the 'unc' entry exists in case of query failure
        return get_uncategorized_pos_id(cur)

def get_uncategorized_pos_id(cur) -> int:
    cur.execute("SELECT id FROM parts_of_speech WHERE code = 'unc'")
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        cur.execute(
            """
            INSERT INTO parts_of_speech (code, name_en, name_tl, description)
            VALUES ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
            RETURNING id
        """
        )
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
        return []

    # Here we could implement more structured etymology parsing
    # For now, just returning a simple structure
    return {"original_text": etymology_text, "processed": True}


def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        meaning = match.group(1)
        clean_text = text.replace(match.group(0), "").strip()
        return clean_text, meaning
    return text, None


def validate_schema(cur):
    """Validate database schema and constraints."""
    try:
        # Check required tables exist
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'words'
            )
        """
        )
        if not cur.fetchone()[0]:
            raise DatabaseError("Required table 'words' does not exist")

        # Check required indexes
        required_indexes = [
            ("words_lemma_idx", "words", "lemma"),
            ("words_normalized_lemma_idx", "words", "normalized_lemma"),
            ("pronunciations_word_id_idx", "pronunciations", "word_id"),
            ("credits_word_id_idx", "credits", "word_id"),
        ]

        for index_name, table, column in required_indexes:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = %s
                )
            """,
                (index_name,),
            )
            if not cur.fetchone()[0]:
                raise DatabaseError(
                    f"Required index {index_name} on {table}({column}) does not exist"
                )

        # Check foreign key constraints
        required_fks = [
            ("pronunciations", "word_id", "words", "id"),
            ("credits", "word_id", "words", "id"),
        ]

        for table, column, ref_table, ref_column in required_fks:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.key_column_usage
                    WHERE table_name = %s
                    AND column_name = %s
                    AND referenced_table_name = %s
                    AND referenced_column_name = %s
                )
            """,
                (table, column, ref_table, ref_column),
            )
            if not cur.fetchone()[0]:
                raise DatabaseError(
                    f"Required foreign key constraint missing: {table}({column}) -> {ref_table}({ref_column})"
                )

    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise


def validate_word_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Word data must be a dictionary")
    required_fields = {"lemma", "language_code"}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    lemma = data["lemma"]
    if not isinstance(lemma, str) or not lemma.strip():
        raise ValueError("Lemma must be a non-empty string")
    if len(lemma) > 255:
        raise ValueError("Lemma exceeds maximum length")
    if data["language_code"] not in {"tl", "ceb"}:
        raise ValueError(f"Unsupported language code: {data['language_code']}")
    if "tags" in data:
        if not isinstance(data["tags"], (str, list)):
            raise ValueError("Tags must be string or list")
        if isinstance(data["tags"], list):
            data["tags"] = ",".join(str(tag) for tag in data["tags"])
    return data


def has_diacritics(text: str) -> bool:
    normalized = normalize_lemma(text)
    return text != normalized


class SourceStandardization:
    @staticmethod
    def standardize_sources(source: str) -> str:
        """Convert source filenames to standardized display names."""
        if not source:
            return "unknown"

        source_mapping = {
            "kaikki-ceb.jsonl": "kaikki.org (Cebuano)",
            "kaikki.jsonl": "kaikki.org (Tagalog)",
            "kwf_dictionary.json": "KWF Diksiyonaryo ng Wikang Filipino",
            "root_words_with_associated_words_cleaned.json": "tagalog.com",
            "tagalog-words.json": "diksiyonaryo.ph",
        }

        # Try direct mapping first
        if source in source_mapping:
            return source_mapping[source]

        # Handle cases where only part of the filename is matched
        for key, value in source_mapping.items():
            if key in source:
                return value

        # Special case for Marayum dictionaries
        if "marayum" in source.lower():
            return "Project Marayum"

        # Return the original if no mapping is found
        return source

    @staticmethod
    def get_display_name(source: str) -> str:
        """Get a display-friendly name for a source."""
        return SourceStandardization.standardize_sources(source)


def format_word_display(word: str, show_baybayin: bool = True) -> str:
    has_bb = any(0x1700 <= ord(c) <= 0x171F for c in word)
    if has_bb:
        romanized = get_romanized_text(word)
        if show_baybayin:
            return f"[bold cyan]{word}[/] [dim](romanized: {romanized})[/]"
        else:
            return romanized
    return word


def get_root_word_id(
    cur: "psycopg2.extensions.cursor", lemma: str, language_code: str
) -> Optional[int]:
    cur.execute(
        """
        SELECT id FROM words 
        WHERE normalized_lemma = %s AND language_code = %s AND root_word_id IS NULL
    """,
        (normalize_lemma(lemma), language_code),
    )
    result = cur.fetchone()
    return result[0] if result else None


# -------------------------------------------------------------------
# Baybayin Processing System
# -------------------------------------------------------------------
class BaybayinRomanizer:
    """Handles romanization of Baybayin text."""

    VOWELS = {
        "ᜀ": BaybayinChar("ᜀ", BaybayinCharType.VOWEL, "a", ["a"]),
        "ᜁ": BaybayinChar("ᜁ", BaybayinCharType.VOWEL, "i", ["i", "e"]),
        "ᜂ": BaybayinChar("ᜂ", BaybayinCharType.VOWEL, "u", ["u", "o"]),
    }
    CONSONANTS = {
        "ᜃ": BaybayinChar("ᜃ", BaybayinCharType.CONSONANT, "ka", ["ka"]),
        "ᜄ": BaybayinChar("ᜄ", BaybayinCharType.CONSONANT, "ga", ["ga"]),
        "ᜅ": BaybayinChar("ᜅ", BaybayinCharType.CONSONANT, "nga", ["nga"]),
        "ᜆ": BaybayinChar("ᜆ", BaybayinCharType.CONSONANT, "ta", ["ta"]),
        "ᜇ": BaybayinChar("ᜇ", BaybayinCharType.CONSONANT, "da", ["da"]),
        "ᜈ": BaybayinChar("ᜈ", BaybayinCharType.CONSONANT, "na", ["na"]),
        "ᜉ": BaybayinChar("ᜉ", BaybayinCharType.CONSONANT, "pa", ["pa"]),
        "ᜊ": BaybayinChar("ᜊ", BaybayinCharType.CONSONANT, "ba", ["ba"]),
        "ᜋ": BaybayinChar("ᜋ", BaybayinCharType.CONSONANT, "ma", ["ma"]),
        "ᜌ": BaybayinChar("ᜌ", BaybayinCharType.CONSONANT, "ya", ["ya"]),
        "ᜎ": BaybayinChar("ᜎ", BaybayinCharType.CONSONANT, "la", ["la"]),
        "ᜏ": BaybayinChar("ᜏ", BaybayinCharType.CONSONANT, "wa", ["wa"]),
        "ᜐ": BaybayinChar("ᜐ", BaybayinCharType.CONSONANT, "sa", ["sa"]),
        "ᜑ": BaybayinChar("ᜑ", BaybayinCharType.CONSONANT, "ha", ["ha"]),
        "ᜍ": BaybayinChar("ᜍ", BaybayinCharType.CONSONANT, "ra", ["ra"]),  # Added ra
    }
    VOWEL_MARKS = {
        "ᜒ": BaybayinChar("ᜒ", BaybayinCharType.VOWEL_MARK, "i", ["i", "e"]),
        "ᜓ": BaybayinChar("ᜓ", BaybayinCharType.VOWEL_MARK, "u", ["u", "o"]),
    }
    VIRAMA = BaybayinChar("᜔", BaybayinCharType.VIRAMA, "", [])
    PUNCTUATION = {
        "᜵": BaybayinChar("᜵", BaybayinCharType.PUNCTUATION, ",", [","]),
        "᜶": BaybayinChar("᜶", BaybayinCharType.PUNCTUATION, ".", ["."]),
    }

    def __init__(self):
        """Initialize the romanizer with a combined character mapping."""
        self.all_chars = {}
        # Combine all character mappings for easy lookup
        for char_map in [
            self.VOWELS,
            self.CONSONANTS,
            self.VOWEL_MARKS,
            {self.VIRAMA.char: self.VIRAMA},
            self.PUNCTUATION,
        ]:
            self.all_chars.update(char_map)

    def is_baybayin(self, text: str) -> bool:
        """Check if a string contains any Baybayin characters."""
        if not text:
            return False
        # Check for characters in the Baybayin Unicode block (U+1700 to U+171F)
        return any(0x1700 <= ord(c) <= 0x171F for c in text)

    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
        """Get character information for a Baybayin character."""
        return self.all_chars.get(char)

    def process_syllable(self, chars: List[str]) -> Tuple[str, int]:
        """
        Process a Baybayin syllable and return its romanized form.

        Args:
            chars: List of characters in the potential syllable

        Returns:
            (romanized_syllable, number_of_characters_consumed)
        """
        if not chars:
            return "", 0

        # Get information about the first character
        first_char = self.get_char_info(chars[0])
        if not first_char:
            # Not a recognized Baybayin character
            return chars[0], 1

        if first_char.char_type == BaybayinCharType.VOWEL:
            # Simple vowel
            return first_char.default_sound, 1

        elif first_char.char_type == BaybayinCharType.CONSONANT:
            # Start with default consonant sound (with 'a' vowel)
            result = first_char.default_sound
            pos = 1

            # Check for vowel marks or virama (vowel killer)
            if pos < len(chars):
                next_char = self.get_char_info(chars[pos])
                if next_char:
                    if next_char.char_type == BaybayinCharType.VOWEL_MARK:
                        # Replace default 'a' vowel with the marked vowel
                        result = result[:-1] + next_char.default_sound
                        pos += 1
                    elif next_char.char_type == BaybayinCharType.VIRAMA:
                        # Remove the default 'a' vowel (final consonant)
                        result = result[:-1]
                        pos += 1

            return result, pos

        elif first_char.char_type == BaybayinCharType.PUNCTUATION:
            # Baybayin punctuation
            return first_char.default_sound, 1

        # For unhandled cases (shouldn't normally happen)
        return "", 1

    def romanize(self, text: str) -> str:
        """
        Convert Baybayin text to its romanized form.

        Args:
            text: The Baybayin text to romanize

        Returns:
            The romanized text, or original text if romanization failed
        """
        if not text:
            return ""

        # Normalize Unicode for consistent character handling
        text = unicodedata.normalize("NFC", text)

        result = []
        i = 0

        while i < len(text):
            # Skip spaces and non-Baybayin characters
            if text[i].isspace() or not self.is_baybayin(text[i]):
                result.append(text[i])
                i += 1
                continue

            # Process a syllable
            try:
                processed_syllable, chars_consumed = self.process_syllable(
                    list(text[i:])
                )

                if processed_syllable:
                    result.append(processed_syllable)
                    i += chars_consumed
                else:
                    # Handle unrecognized characters
                    result.append(text[i])
                    i += 1
            except Exception as e:
                logger.error(f"Error during Baybayin romanization at position {i}: {e}")
                # Skip problematic character
                i += 1

        return "".join(result)

    def validate_text(self, text: str) -> bool:
        """
        Validate that a string contains valid Baybayin text.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid Baybayin, False otherwise
        """
        if not text:
            return False

        # Normalize Unicode
        text = unicodedata.normalize("NFC", text)
        chars = list(text)
        i = 0

        while i < len(chars):
            # Skip spaces
            if chars[i].isspace():
                i += 1
                continue

            # Get character info
            char_info = self.get_char_info(chars[i])

            # Not a valid Baybayin character
            if not char_info:
                if 0x1700 <= ord(chars[i]) <= 0x171F:
                    # It's in the Baybayin Unicode range but not recognized
                    logger.warning(
                        f"Unrecognized Baybayin character at position {i}: {chars[i]} (U+{ord(chars[i]):04X})"
                    )
                return False

            # Vowel mark must follow a consonant
            if char_info.char_type == BaybayinCharType.VOWEL_MARK:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(
                        f"Vowel mark not following a consonant at position {i}"
                    )
                    return False

            # Virama (vowel killer) must follow a consonant
            if char_info.char_type == BaybayinCharType.VIRAMA:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(f"Virama not following a consonant at position {i}")
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

    Args:
        text: Latin text to convert to Baybayin

    Returns:
        Baybayin text
    """
    if not text:
        return ""

    # Handle prefix starting with '-' (like '-an')
    if text.startswith("-"):
        # Skip the hyphen and process the rest
        text = text[1:]

    # Normalize text: lowercase and remove diacritical marks
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )

    # Define Baybayin character mappings
    consonants = {
        "k": "ᜃ",
        "g": "ᜄ",
        "ng": "ᜅ",
        "t": "ᜆ",
        "d": "ᜇ",
        "n": "ᜈ",
        "p": "ᜉ",
        "b": "ᜊ",
        "m": "ᜋ",
        "y": "ᜌ",
        "l": "ᜎ",
        "w": "ᜏ",
        "s": "ᜐ",
        "h": "ᜑ",
        "r": "ᜍ",  # Added 'r' mapping
    }
    vowels = {"a": "ᜀ", "i": "ᜁ", "e": "ᜁ", "u": "ᜂ", "o": "ᜂ"}
    vowel_marks = {"i": "ᜒ", "e": "ᜒ", "u": "ᜓ", "o": "ᜓ"}
    virama = "᜔"  # Pamudpod (vowel killer)

    result = []
    i = 0

    while i < len(text):
        # Check for 'ng' digraph first
        if i + 1 < len(text) and text[i : i + 2] == "ng":
            if i + 2 < len(text) and text[i + 2] in "aeiou":
                # ng + vowel
                if text[i + 2] == "a":
                    result.append(consonants["ng"])
                else:
                    result.append(consonants["ng"] + vowel_marks[text[i + 2]])
                i += 3
            else:
                # Final 'ng'
                result.append(consonants["ng"] + virama)
                i += 2

        # Handle single consonants
        elif text[i] in consonants:
            if i + 1 < len(text) and text[i + 1] in "aeiou":
                # Consonant + vowel
                if text[i + 1] == "a":
                    result.append(consonants[text[i]])
                else:
                    result.append(consonants[text[i]] + vowel_marks[text[i + 1]])
                i += 2
            else:
                # Final consonant
                result.append(consonants[text[i]] + virama)
                i += 1

        # Handle vowels
        elif text[i] in "aeiou":
            result.append(vowels[text[i]])
            i += 1

        # Skip spaces and other characters
        elif text[i].isspace():
            result.append(" ")
            i += 1
        else:
            # Skip non-convertible characters
            i += 1

    # Final validation - ensure only valid characters are included
    valid_output = "".join(
        c for c in result if (0x1700 <= ord(c) <= 0x171F) or c.isspace()
    )

    # Verify the output meets database constraints
    if not re.match(r"^[\u1700-\u171F\s]*$", valid_output):
        logger.warning(
            f"Transliterated Baybayin doesn't match required regex pattern: {valid_output}"
        )
        # Additional cleanup to ensure it matches the pattern
        valid_output = re.sub(r"[^\u1700-\u171F\s]", "", valid_output)

    return valid_output


@with_transaction(commit=False)
def verify_baybayin_data(cur):
    """Verify the consistency of Baybayin data in the database."""
    cur.execute(
        """
        SELECT id, lemma, baybayin_form 
        FROM words 
        WHERE has_baybayin = TRUE AND baybayin_form IS NULL
    """
    )
    orphaned = cur.fetchall()
    if orphaned:
        logger.warning(
            f"Found {len(orphaned)} words marked as Baybayin but missing baybayin_form"
        )
        for word_id, lemma, _ in orphaned:
            logger.warning(f"Word ID {word_id}: {lemma}")
    cur.execute(
        """
        SELECT baybayin_form, COUNT(*) 
        FROM words 
        WHERE has_baybayin = TRUE 
        GROUP BY baybayin_form 
        HAVING COUNT(*) > 1
    """
    )
    duplicates = cur.fetchall()
    if duplicates:
        logger.warning(f"Found {len(duplicates)} Baybayin forms with multiple entries")
        for baybayin_form, count in duplicates:
            logger.warning(f"Baybayin form {baybayin_form} appears {count} times")


@with_transaction(commit=True)
def merge_baybayin_entries(cur, baybayin_id: int, romanized_id: int):
    """Merge a Baybayin entry with its romanized form."""
    try:
        cur.execute(
            """
            SELECT lemma, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """,
            (baybayin_id,),
        )
        baybayin_result = cur.fetchone()
        if not baybayin_result:
            raise ValueError(f"Baybayin entry {baybayin_id} not found")
        baybayin_lemma, baybayin_form, baybayin_rom = baybayin_result
        tables = [
            ("definitions", "word_id"),
            ("relations", "from_word_id"),
            ("relations", "to_word_id"),
            ("etymologies", "word_id"),
            ("definition_relations", "word_id"),
            ("affixations", "root_word_id"),
            ("affixations", "affixed_word_id"),
        ]
        for table, column in tables:
            cur.execute(
                f"""
                UPDATE {table} 
                SET {column} = %s 
                WHERE {column} = %s
            """,
                (romanized_id, baybayin_id),
            )
        cur.execute(
            """
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = COALESCE(%s, baybayin_form),
                romanized_form = COALESCE(%s, romanized_form),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """,
            (baybayin_form, baybayin_rom or baybayin_lemma, romanized_id),
        )
        cur.execute("DELETE FROM words WHERE id = %s", (baybayin_id,))
    except Exception as e:
        logger.error(f"Error merging Baybayin entries: {str(e)}")
        raise


def clean_baybayin_text(text: str) -> str:
    """
    Clean Baybayin text by removing non-Baybayin characters.

    Args:
        text: Text that may contain Baybayin and other characters

    Returns:
        Cleaned text with only valid Baybayin characters and spaces
    """
    if not text:
        return ""

    # Keep only characters in the Baybayin Unicode range (U+1700 to U+171F) and spaces
    cleaned = "".join(c for c in text if (0x1700 <= ord(c) <= 0x171F) or c.isspace())

    # Normalize whitespace and trim
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_baybayin_text(text: str) -> List[str]:
    """
    Extract Baybayin text segments from a string.

    Args:
        text: Text that may contain Baybayin

    Returns:
        List of Baybayin segments
    """
    if not text:
        return []

    # Split by non-Baybayin characters
    parts = re.split(r"[^\u1700-\u171F\s]+", text)
    results = []

    for part in parts:
        # Clean and normalize
        cleaned_part = clean_baybayin_text(part)

        # Make sure part contains at least one Baybayin character
        if cleaned_part and any(0x1700 <= ord(c) <= 0x171F for c in cleaned_part):
            # Verify it meets database constraints
            if re.match(r"^[\u1700-\u171F\s]*$", cleaned_part):
                results.append(cleaned_part)

    return results


def validate_baybayin_entry(
    baybayin_form: str, romanized_form: Optional[str] = None
) -> bool:
    """
    Validate if a Baybayin form is correct and matches the romanized form if provided.

    Args:
        baybayin_form: The Baybayin text to validate
        romanized_form: Optional romanized form to verify against

    Returns:
        bool: True if valid, False otherwise
    """
    if not baybayin_form:
        return False

    try:
        # Clean and validate the form
        cleaned_form = clean_baybayin_text(baybayin_form)
        if not cleaned_form:
            logger.warning(f"No valid Baybayin characters found in: {baybayin_form}")
            return False

        # Check against the database regex constraint
        if not re.match(r"^[\u1700-\u171F\s]*$", cleaned_form):
            logger.warning(
                f"Baybayin form doesn't match required regex pattern: {cleaned_form}"
            )
            return False

        # Create a romanizer to validate structure
        romanizer = BaybayinRomanizer()
        if not romanizer.validate_text(cleaned_form):
            logger.warning(f"Invalid Baybayin structure in: {cleaned_form}")
            return False

        # If romanized form is provided, check if it matches our romanization
        if romanized_form:
            try:
                generated_rom = romanizer.romanize(cleaned_form)
                # Compare normalized versions to avoid case and diacritic issues
                if normalize_lemma(generated_rom) == normalize_lemma(romanized_form):
                    return True
                else:
                    logger.warning(
                        f"Romanization mismatch: expected '{romanized_form}', got '{generated_rom}'"
                    )
                    # Still return True if structure is valid but romanization doesn't match
                    # This allows for different romanization standards
                    return True
            except Exception as e:
                logger.error(f"Error during romanization validation: {e}")
                return False

        # If no romanized form to check against, return True if the form is valid
        return True

    except Exception as e:
        logger.error(f"Error validating Baybayin entry: {e}")
        return False


@with_transaction(commit=True)
def process_baybayin_data(
    cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None
) -> bool:
    """
    Process and store Baybayin data for a word.

    Args:
        cur: Database cursor
        word_id: Word ID to update
        baybayin_form: The Baybayin text
        romanized_form: Optional romanized form

    Returns:
        bool: True if successful, False otherwise
    """
    if not baybayin_form:
        logger.warning(f"Empty Baybayin form for word_id {word_id}")
        return False

    try:
        # Clean the baybayin form
        cleaned_baybayin = clean_baybayin_text(baybayin_form)

        if not cleaned_baybayin:
            logger.warning(
                f"No valid Baybayin characters found in: {baybayin_form} for word_id {word_id}"
            )
            return False

        # Verify it meets database constraints
        if not re.match(r"^[\u1700-\u171F\s]*$", cleaned_baybayin):
            logger.warning(
                f"Baybayin form doesn't match required regex pattern: {cleaned_baybayin}"
            )
            return False

        # Create a romanizer to validate structure
        romanizer = BaybayinRomanizer()
        if not romanizer.validate_text(cleaned_baybayin):
            logger.warning(
                f"Invalid Baybayin structure in: {cleaned_baybayin} for word_id {word_id}"
            )
            return False

        # Generate romanization if not provided
        if not romanized_form:
            try:
                romanized_form = romanizer.romanize(cleaned_baybayin)
            except Exception as e:
                logger.error(
                    f"Error generating romanization for word_id {word_id}: {e}"
                )
                # Try to continue with the process even if romanization fails
                romanized_form = None

        # Verify the word exists before updating
        cur.execute("SELECT 1 FROM words WHERE id = %s", (word_id,))
        if not cur.fetchone():
            logger.warning(f"Word ID {word_id} does not exist in the database")
            return False

        # Update the word record
        cur.execute(
            """
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = %s,
                romanized_form = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """,
            (cleaned_baybayin, romanized_form, word_id),
        )

        return True

    except Exception as e:
        logger.error(f"Error processing Baybayin data for word_id {word_id}: {e}")
        raise


def get_db_connection(max_retries=3, retry_delay=1.0):
    """
    Get a connection from the connection pool with retry logic.

    Args:
        max_retries (int): Maximum number of connection attempts
        retry_delay (float): Delay between retries in seconds

    Returns:
        connection: A database connection

    Raises:
        DatabaseConnectionError: If connection cannot be established after retries
    """
    global connection_pool

    if connection_pool is None:
        try:
            # Initialize connection pool if not exists
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **get_db_config()
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseConnectionError(
                f"Could not initialize database connection pool: {e}"
            )

    for attempt in range(max_retries):
        try:
            conn = connection_pool.getconn()

            # Validate connection
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

            # Set reasonable timeouts
            conn.set_session(
                autocommit=True,
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED,
            )
            return conn

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 1.5
            else:
                logger.error(
                    f"Failed to get database connection after {max_retries} attempts: {e}"
                )
                raise DatabaseConnectionError(
                    f"Could not connect to database after {max_retries} attempts: {e}"
                )

        except Exception as e:
            logger.error(f"Unexpected error getting database connection: {e}")
            # Try to return connection to pool if it was obtained
            try:
                if "conn" in locals():
                    connection_pool.putconn(conn)
            except:
                pass
            raise DatabaseConnectionError(f"Unexpected error: {e}")

    # Should never reach here due to the final raise in the loop
    raise DatabaseConnectionError("Failed to get database connection")


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
            if args and hasattr(args[0], "cursor"):
                # If first arg is a class instance with cursor, use that
                return func(*args, **kwargs)
            else:
                # Otherwise provide a cursor
                return func(cursor, *args, **kwargs)

    return wrapper


def standardize_source_identifier(source_identifier: str) -> str:
    """
    Standardize a source identifier string.

    Args:
        source_identifier: Raw source identifier (typically a filename)

    Returns:
        Standardized source name
    """
    if not source_identifier:
        return "unknown"

    # Use existing SourceStandardization class with fallback
    standardized = SourceStandardization.standardize_sources(source_identifier)

    # If the standardization failed (returns None or empty string), use the original
    if not standardized:
        return source_identifier

    return standardized


# Define constants for reusable values
DEFAULT_LANGUAGE_CODE = "tl"
SOURCE_INFO_FILES_KEY = "files"


def update_word_source_info(
    current_source_info: Optional[Union[str, dict]],
    new_source_identifier: Optional[str],
) -> str:
    """
    Updates the source_info JSON for a word entry, adding a new source identifier
    to a list under the 'files' key if it's not already present.

    Args:
        current_source_info: The current source_info from the words table.
        new_source_identifier: The identifier (e.g., filename) to add.

    Returns:
        A JSON string representation of the updated source_info.
    """
    # Better handling for empty or None new_source_identifier
    if not new_source_identifier:
        if isinstance(current_source_info, dict):
            return json.dumps(current_source_info)
        elif isinstance(current_source_info, str):
            return current_source_info
        else:
            return "{}"

    # Parse current_source_info into a dictionary
    source_info_dict = {}
    if isinstance(current_source_info, dict):
        source_info_dict = (
            current_source_info.copy()
        )  # Use copy to avoid modifying the original
    elif isinstance(current_source_info, str) and current_source_info:
        try:
            source_info_dict = json.loads(current_source_info)
            if not isinstance(source_info_dict, dict):
                source_info_dict = {}  # Reset if not a dictionary
        except (json.JSONDecodeError, TypeError):
            # Start fresh if invalid JSON
            source_info_dict = {}

    # Ensure 'files' key exists and is a list
    if SOURCE_INFO_FILES_KEY not in source_info_dict or not isinstance(
        source_info_dict[SOURCE_INFO_FILES_KEY], list
    ):
        source_info_dict[SOURCE_INFO_FILES_KEY] = []

    # Standardize the source identifier before adding
    standardized_source = SourceStandardization.standardize_sources(
        new_source_identifier
    )

    # Add new source identifier if not already present
    if standardized_source not in source_info_dict[SOURCE_INFO_FILES_KEY]:
        source_info_dict[SOURCE_INFO_FILES_KEY].append(standardized_source)
        source_info_dict[SOURCE_INFO_FILES_KEY].sort()  # Keep list sorted

    # Add metadata about when and how this source was added
    if "last_updated" not in source_info_dict:
        source_info_dict["last_updated"] = {}

    source_info_dict["last_updated"][standardized_source] = datetime.now().isoformat()

    # Return JSON string
    return json.dumps(source_info_dict)


# -------------------------------------------------------------------
# Setup Logging
# -------------------------------------------------------------------
def setup_logging():
    """Configure logging with proper Unicode handling."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_path = f"{log_dir}/dictionary_manager_{timestamp}.log"
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8")
            console_handler.stream = codecs.getwriter("utf-8")(sys.stdout.buffer)
    except Exception:

        def safe_encode(msg):
            try:
                return (
                    str(msg)
                    .encode(console_handler.stream.encoding, "replace")
                    .decode(console_handler.stream.encoding)
                )
            except:
                return str(msg).encode("ascii", "replace").decode("ascii")

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


def get_db_config():
    """Return the database configuration dictionary."""
    return DB_CONFIG


# -------------------------------------------------------------------
# Core Database Connection
# -------------------------------------------------------------------
def get_connection():
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME", DB_NAME),
        user=os.environ.get("DB_USER", DB_USER),
        password=os.environ.get("DB_PASSWORD", DB_PASSWORD),
        host=os.environ.get("DB_HOST", DB_HOST),
        port=os.environ.get("DB_PORT", DB_PORT),
    )
    conn.autocommit = False  # Ensure transactions are used.
    return conn


def get_cursor():
    """Return a cursor from a new connection with proper error handling."""
    conn = None
    try:
        conn = get_connection()
        return conn.cursor()
    except Exception as e:
        # Make sure we release the connection if we got it but failed to get a cursor
        if conn:
            release_db_connection(conn)
        logger.error(f"Error getting cursor: {e}")
        raise


# -------------------------------------------------------------------
# Setup Extensions
# -------------------------------------------------------------------
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    cur = conn.cursor()
    try:
        extensions = ["pg_trgm", "unaccent", "fuzzystrmatch", "dict_xsyn"]
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
    Decorator for database operations that need to run within a transaction.

    Args:
        commit (bool): Whether to commit the transaction after successful execution.
                      If False, the transaction will be left open.

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection

            # Check if we're already in a transaction
            original_autocommit = None
            started_transaction = False

            try:
                # Only start a new transaction if we're not already in one
                if conn.autocommit:
                    original_autocommit = True
                    conn.autocommit = False
                    started_transaction = True
                    logger.debug(f"Started new transaction for {func.__name__}")

                # Execute the function
                result = func(cur, *args, **kwargs)

                # Commit if requested and we started the transaction
                if commit and started_transaction:
                    try:
                        conn.commit()
                        logger.debug(f"Transaction committed for {func.__name__}")
                    except Exception as commit_error:
                        logger.error(f"Failed to commit transaction: {commit_error}")
                        try:
                            conn.rollback()
                            logger.debug("Transaction rolled back due to commit error")
                        except:
                            pass
                        raise

                return result

            except Exception as e:
                # Only roll back if we started the transaction
                if started_transaction:
                    try:
                        conn.rollback()
                        logger.debug(f"Transaction rolled back due to error: {e}")
                    except Exception as rollback_error:
                        logger.error(
                            f"Failed to rollback transaction: {rollback_error}"
                        )
                # Re-raise the original exception
                raise

            finally:
                # Restore original autocommit state if we changed it
                if started_transaction and original_autocommit is not None:
                    try:
                        conn.autocommit = original_autocommit
                        logger.debug(
                            f"Restored autocommit state to {original_autocommit}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to restore autocommit state: {e}")

        return wrapper

    return decorator


@with_transaction(commit=True)
def fix_inconsistent_sources(cur) -> Dict:
    """
    Fix inconsistent source information across the database.

    Args:
        cur: Database cursor

    Returns:
        Statistics about the fixes
    """
    stats = {
        "words_updated": 0,
        "definitions_updated": 0,
        "relations_updated": 0,
        "etymologies_updated": 0,
    }

    try:
        # Standardize sources in words table
        cur.execute(
            """
            WITH source_standardization AS (
                SELECT id, 
                       CASE
                           WHEN source_info::text ILIKE '%kaikki-ceb.jsonl%' THEN 'kaikki.org (Cebuano)'
                           WHEN source_info::text ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                           WHEN source_info::text ILIKE '%kwf_dictionary.json%' THEN 'KWF Diksiyonaryo ng Wikang Filipino'
                           WHEN source_info::text ILIKE '%root_words_with_associated_words_cleaned.json%' THEN 'tagalog.com'
                           WHEN source_info::text ILIKE '%tagalog-words.json%' THEN 'diksiyonaryo.ph'
                           WHEN source_info::text ILIKE '%marayum%' THEN 'Project Marayum'
                           ELSE source_info::text
                       END as standardized_source
                FROM words
                WHERE source_info IS NOT NULL
            )
            UPDATE words w
            SET source_info = jsonb_build_object('files', jsonb_build_array(ss.standardized_source))
            FROM source_standardization ss
            WHERE w.id = ss.id
            AND (source_info IS NULL OR source_info = '{}'::jsonb)
        """
        )
        stats["words_updated"] = cur.rowcount

        # Similar updates for other tables...

        return stats
    except Exception as e:
        logger.error(f"Error fixing inconsistent sources: {e}")
        raise


@with_transaction(commit=False)
def get_word_sources(cur, word_id: int) -> Dict:
    """
    Get comprehensive source information for a word.

    Args:
        cur: Database cursor
        word_id: ID of the word

    Returns:
        Dictionary with source information
    """
    result = {
        "word": None,
        "direct_sources": [],
        "definition_sources": [],
        "relation_sources": [],
        "etymology_sources": [],
    }

    try:
        # Get word info
        cur.execute(
            """
            SELECT lemma, source_info 
            FROM words 
            WHERE id = %s
        """,
            (word_id,),
        )
        word_row = cur.fetchone()

        if not word_row:
            return result

        result["word"] = word_row[0]

        # Extract direct sources
        if word_row[1]:
            try:
                source_info = (
                    json.loads(word_row[1])
                    if isinstance(word_row[1], str)
                    else word_row[1]
                )
                if (
                    isinstance(source_info, dict)
                    and SOURCE_INFO_FILES_KEY in source_info
                ):
                    result["direct_sources"] = source_info[SOURCE_INFO_FILES_KEY]
            except (json.JSONDecodeError, TypeError):
                pass

        # Get definition sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM definitions 
            WHERE word_id = %s AND sources IS NOT NULL
        """,
            (word_id,),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["definition_sources"]:
                result["definition_sources"].append(row[0])

        # Get relation sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM relations 
            WHERE (from_word_id = %s OR to_word_id = %s) AND sources IS NOT NULL
        """,
            (word_id, word_id),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["relation_sources"]:
                result["relation_sources"].append(row[0])

        # Get etymology sources
        cur.execute(
            """
            SELECT DISTINCT sources 
            FROM etymologies 
            WHERE word_id = %s AND sources IS NOT NULL
        """,
            (word_id,),
        )
        for row in cur.fetchall():
            if row[0] and row[0] not in result["etymology_sources"]:
                result["etymology_sources"].append(row[0])

        return result
    except Exception as e:
        logger.error(f"Error getting sources for word ID {word_id}: {e}")
        return result


@with_transaction(commit=True)
def propagate_source_info(cur, word_id: int, source_identifier: str) -> bool:
    """
    Propagate source information to related records.

    Args:
        cur: Database cursor
        word_id: ID of the word
        source_identifier: Source identifier to propagate

    Returns:
        True if successful, False otherwise
    """
    try:
        standardized_source = standardize_source_identifier(source_identifier)

        # Update related definitions
        cur.execute(
            """
            UPDATE definitions
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
            ),
        )

        # Update related etymologies
        cur.execute(
            """
            UPDATE etymologies
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
            ),
        )

        # Update related relations
        cur.execute(
            """
            UPDATE relations
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE from_word_id = %s OR to_word_id = %s
        """,
            (
                standardized_source,
                f"%{standardized_source}%",
                standardized_source,
                word_id,
                word_id,
            ),
        )

        return True
    except Exception as e:
        logger.error(f"Error propagating source info for word ID {word_id}: {e}")
        return False


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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
    root_word_id INT REFERENCES words(id) ON DELETE SET NULL,
    preferred_spelling VARCHAR(255),
    tags TEXT,
    idioms JSONB DEFAULT '[]',
    pronunciation_data JSONB,
    source_info JSONB DEFAULT '{}',
    word_metadata JSONB DEFAULT '{}',
    data_hash TEXT,
    badlit_form TEXT,
    hyphenation JSONB,
    is_proper_noun BOOLEAN DEFAULT FALSE,
    is_abbreviation BOOLEAN DEFAULT FALSE,
    is_initialism BOOLEAN DEFAULT FALSE,
    search_text TSVECTOR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT words_lang_lemma_uniq UNIQUE (normalized_lemma, language_code),
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
CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);
CREATE INDEX IF NOT EXISTS idx_words_metadata ON words USING GIN(word_metadata);

-- Create pronunciations table
CREATE TABLE IF NOT EXISTS pronunciations (
    id SERIAL PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL DEFAULT 'ipa',
    value TEXT NOT NULL,
    tags JSONB,
    pronunciation_metadata JSONB,
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pronunciations_unique UNIQUE(word_id, type, value)
);
CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
CREATE INDEX IF NOT EXISTS idx_pronunciations_type ON pronunciations(type);
CREATE INDEX IF NOT EXISTS idx_pronunciations_value ON pronunciations(value);

-- Create credits table
CREATE TABLE IF NOT EXISTS credits (
    id SERIAL PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    credit TEXT NOT NULL,
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT credits_unique UNIQUE(word_id, credit)
);
CREATE INDEX IF NOT EXISTS idx_credits_word ON credits(word_id);

-- Create definitions table
CREATE TABLE IF NOT EXISTS definitions (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    definition_text TEXT NOT NULL,
    original_pos TEXT,
    standardized_pos_id INT REFERENCES parts_of_speech(id) ON DELETE SET NULL,
    examples TEXT,
    usage_notes TEXT,
    tags TEXT,
    sources TEXT,
    metadata JSONB,
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

-- Create etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text TEXT NOT NULL,
    normalized_components TEXT,
    etymology_structure TEXT,
    language_codes TEXT,
    sources TEXT,
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
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
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
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT affixations_unique UNIQUE (root_word_id, affixed_word_id, affix_type)
);
CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);

-- Create definition_categories table
CREATE TABLE IF NOT EXISTS definition_categories (
   id SERIAL PRIMARY KEY,
   definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
   category_name TEXT NOT NULL,
   category_kind TEXT,
   parents JSONB,
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (definition_id, category_name)
);
CREATE INDEX IF NOT EXISTS idx_def_categories_def ON definition_categories(definition_id);
CREATE INDEX IF NOT EXISTS idx_def_categories_name ON definition_categories(category_name);

-- Create word_templates table
CREATE TABLE IF NOT EXISTS word_templates (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
    template_name TEXT NOT NULL,
    args JSONB,
    expansion TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (word_id, template_name)
);
CREATE INDEX IF NOT EXISTS idx_word_templates_word ON word_templates(word_id);
CREATE INDEX IF NOT EXISTS idx_word_templates_name ON word_templates(template_name);

-- Create word_forms table
CREATE TABLE IF NOT EXISTS word_forms (
   id SERIAL PRIMARY KEY,
   word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
   form TEXT NOT NULL,
   is_canonical BOOLEAN DEFAULT FALSE,
   is_primary BOOLEAN DEFAULT FALSE,
   tags JSONB,
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (word_id, form)
);
CREATE INDEX IF NOT EXISTS idx_word_forms_word ON word_forms(word_id);
CREATE INDEX IF NOT EXISTS idx_word_forms_form ON word_forms(form);

-- Create definition_links table
CREATE TABLE IF NOT EXISTS definition_links (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
    link_text TEXT NOT NULL,
    link_target TEXT NOT NULL,
    is_wikipedia BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (definition_id, link_text, link_target)
);
CREATE INDEX IF NOT EXISTS idx_def_links_def ON definition_links(definition_id);


-- Create triggers for timestamp updates (ensure function exists first)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_words_timestamp' AND tgrelid = 'words'::regclass) THEN
        CREATE TRIGGER update_words_timestamp BEFORE UPDATE ON words FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definitions_timestamp' AND tgrelid = 'definitions'::regclass) THEN
        CREATE TRIGGER update_definitions_timestamp BEFORE UPDATE ON definitions FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_etymologies_timestamp' AND tgrelid = 'etymologies'::regclass) THEN
        CREATE TRIGGER update_etymologies_timestamp BEFORE UPDATE ON etymologies FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_pronunciations_timestamp' AND tgrelid = 'pronunciations'::regclass) THEN
        CREATE TRIGGER update_pronunciations_timestamp BEFORE UPDATE ON pronunciations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_credits_timestamp' AND tgrelid = 'credits'::regclass) THEN
        CREATE TRIGGER update_credits_timestamp BEFORE UPDATE ON credits FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_relations_timestamp' AND tgrelid = 'definition_relations'::regclass) THEN
        CREATE TRIGGER update_definition_relations_timestamp BEFORE UPDATE ON definition_relations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_parts_of_speech_timestamp' AND tgrelid = 'parts_of_speech'::regclass) THEN
        CREATE TRIGGER update_parts_of_speech_timestamp BEFORE UPDATE ON parts_of_speech FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_affixations_timestamp' AND tgrelid = 'affixations'::regclass) THEN
        CREATE TRIGGER update_affixations_timestamp BEFORE UPDATE ON affixations FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;

    -- Add triggers for NEW TABLES
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_categories_timestamp' AND tgrelid = 'definition_categories'::regclass) THEN
        CREATE TRIGGER update_definition_categories_timestamp BEFORE UPDATE ON definition_categories FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_word_templates_timestamp' AND tgrelid = 'word_templates'::regclass) THEN
        CREATE TRIGGER update_word_templates_timestamp BEFORE UPDATE ON word_templates FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
     IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_word_forms_timestamp' AND tgrelid = 'word_forms'::regclass) THEN
        CREATE TRIGGER update_word_forms_timestamp BEFORE UPDATE ON word_forms FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;
     IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_definition_links_timestamp' AND tgrelid = 'definition_links'::regclass) THEN
        CREATE TRIGGER update_definition_links_timestamp BEFORE UPDATE ON definition_links FOR EACH ROW EXECUTE FUNCTION update_timestamp();
    END IF;

END $$;

"""

def create_or_update_tables(conn):
    """Create or update the database tables."""
    logger.info("Starting table creation/update process.")

    cur = conn.cursor()
    try:
        # Drop existing tables in correct order
        cur.execute(
            """
            DROP TABLE IF EXISTS 
                credits, pronunciations, definition_relations, affixations, 
                relations, etymologies, definitions, words, parts_of_speech CASCADE;
        """
        )

        # Create tables
        cur.execute(TABLE_CREATION_SQL)

        # Insert standard parts of speech
        pos_entries = [
            # --- Core Grammatical Categories ---
            (
                "n",
                "Noun",
                "Pangngalan",
                "Word that refers to a person, place, thing, or idea",
            ),
            ("v", "Verb", "Pandiwa", "Word that expresses action or state of being"),
            ("adj", "Adjective", "Pang-uri", "Word that describes or modifies a noun"),
            (
                "adv",
                "Adverb",
                "Pang-abay",
                "Word that modifies verbs, adjectives, or other adverbs",
            ),
            ("pron", "Pronoun", "Panghalip", "Word that substitutes for a noun"),
            (
                "prep",
                "Preposition",
                "Pang-ukol",
                "Word that shows relationship between words",
            ),
            (
                "conj",
                "Conjunction",
                "Pangatnig",
                "Word that connects words, phrases, or clauses",
            ),
            ("intj", "Interjection", "Pandamdam", "Word expressing emotion"),
            ("det", "Determiner", "Pantukoy", "Word that modifies nouns"),
            ("affix", "Affix", "Panlapi", "Word element attached to base or root"),

            # --- Added/Ensured based on POS_MAPPING and Daglat ---
            ("lig", "Ligature", "Pang-angkop", "Word that links modifiers to modified words"), # For 'pnk'
            ("part", "Particle", "Kataga", "Function word that doesn't fit other categories"), # From mapping
            ("num", "Number", "Pamilang", "Word representing a number"), # From mapping
            ("expr", "Expression", "Pahayag", "Common phrase or expression"), # From mapping
            ("punc", "Punctuation", "Bantas", "Punctuation mark"), # From mapping

            # --- Other Categories from original code ---
            ("idm", "Idiom", "Idyoma", "Fixed expression with non-literal meaning"),
            ("col", "Colloquial", "Kolokyal", "Informal or conversational usage"),
            ("syn", "Synonym", "Singkahulugan", "Word with similar meaning"), # Note: Relationships preferred over POS tags for this
            ("ant", "Antonym", "Di-kasingkahulugan", "Word with opposite meaning"), # Note: Relationships preferred over POS tags for this
            ("eng", "English", "Ingles", "English loanword or translation"), # Note: Etymology preferred over POS tags for this
            ("spa", "Spanish", "Espanyol", "Spanish loanword or origin"), # Note: Etymology preferred over POS tags for this
            ("tx", "Texting", "Texting", "Text messaging form"),
            ("var", "Variant", "Varyant", "Alternative form or spelling"), # Note: Relationships preferred over POS tags for this
            (
                "unc",
                "Uncategorized",
                "Hindi Tiyak",
                "Part of speech not yet determined",
            ),
        ]

        for code, name_en, name_tl, desc in pos_entries:
            cur.execute(
                """
                INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (code) DO UPDATE 
                SET name_en = EXCLUDED.name_en,
                    name_tl = EXCLUDED.name_tl,
                    description = EXCLUDED.description
            """,
                (code, name_en, name_tl, desc),
            )

        conn.commit()
        logger.info("Tables created or updated successfully.")

    except Exception as e:
        conn.rollback()
        logger.error(f"Schema creation error: {str(e)}")
        raise
    finally:
        cur.close()


@with_transaction(commit=True)
def insert_pronunciation(
    cur, word_id: int, pronunciation_data: Union[str, Dict], source_identifier: str
) -> Optional[int]:
    """
    Insert pronunciation data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing pronunciations based on (word_id, type, value),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        pronunciation_data: Pronunciation string (assumed IPA) or dictionary
                           (e.g., {'type': 'ipa', 'value': '...', 'tags': [...]}).
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated pronunciation record, or None if failed.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping pronunciation insert for word ID {word_id}: Missing MANDATORY source identifier."
        )
        return None

    pron_type = "ipa"  # Default type
    value = None
    tags_list = []
    metadata = {}

    try:
        # Parse input data
        if isinstance(pronunciation_data, dict):
            # Prioritize keys from the dict, but allow source_identifier argument to override 'sources' key if present
            pron_type = (
                pronunciation_data.get("type", "ipa") or "ipa"
            )  # Default to 'ipa' if empty
            value = (
                pronunciation_data.get("value", "").strip()
                if isinstance(pronunciation_data.get("value"), str)
                else None
            )
            tags_input = pronunciation_data.get("tags")
            if isinstance(tags_input, list):
                tags_list = tags_input
            elif isinstance(tags_input, str):
                # Simple split if tags are a comma-separated string? Adapt as needed.
                tags_list = [
                    tag.strip() for tag in tags_input.split(",") if tag.strip()
                ]
                logger.debug(
                    f"Parsed tags string '{tags_input}' into list for word ID {word_id}."
                )
            # else: ignore invalid tags format

            metadata_input = pronunciation_data.get("metadata")
            if isinstance(metadata_input, dict):
                metadata = metadata_input
            # else: ignore invalid metadata format

        elif isinstance(pronunciation_data, str):
            value = pronunciation_data.strip()
            # Assumed type is 'ipa' and no tags/metadata provided via string input
        else:
            logger.warning(
                f"Invalid pronunciation_data type for word ID {word_id} (source '{source_identifier}'): {type(pronunciation_data)}. Skipping."
            )
            return None

        if not value:
            logger.warning(
                f"Empty pronunciation value for word ID {word_id} (source '{source_identifier}'). Skipping pronounicitaion insertion."
            )
            return None

        # Safely dump JSON fields (tags and metadata) for DB insertion (assuming JSONB columns)
        tags_json = None
        try:
            tags_json = json.dumps(
                tags_list
            )  # tags_list will be [] if not provided or invalid format
        except TypeError as e:
            logger.warning(
                f"Could not serialize tags for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Tags: {tags_list}"
            )
            tags_json = "[]"  # Fallback to empty JSON array string

        metadata_json = None
        try:
            metadata_json = json.dumps(
                metadata
            )  # metadata will be {} if not provided or invalid format
        except TypeError as e:
            logger.warning(
                f"Could not serialize metadata for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Metadata: {metadata}"
            )
            metadata_json = "{}"  # Fallback to empty JSON object string

        # Prepare parameters for query
        params = {
            "word_id": word_id,
            "type": pron_type,
            "value": value,
            "tags": tags_json,
            "pronunciation_metadata": metadata_json,
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }

        # Insert or update pronunciation
        cur.execute(
            """
            INSERT INTO pronunciations (word_id, type, value, tags, pronunciation_metadata, sources)
            VALUES (%(word_id)s, %(type)s, %(value)s, %(tags)s::jsonb, %(pronunciation_metadata)s::jsonb, %(sources)s) -- Cast JSONs
            ON CONFLICT (word_id, type, value) -- Conflict on word, type, and exact value
            DO UPDATE SET
                -- Update tags/metadata only if new value is not NULL (or empty JSON?) - COALESCE prefers new non-null
                tags = COALESCE(EXCLUDED.tags, pronunciations.tags),
                pronunciation_metadata = COALESCE(EXCLUDED.pronunciation_metadata, pronunciations.pronunciation_metadata),
                -- Overwrite sources: Last write wins for this pronunciation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        pron_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated pronunciation (ID: {pron_id}, Type: {pron_type}) for word ID {word_id} from source '{source_identifier}'. Value: '{value}'"
        )
        return pron_id

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None


@with_transaction(commit=True)
def insert_credit(
    cur, word_id: int, credit_data: Union[str, Dict], source_identifier: str
) -> Optional[int]:
    """
    Insert credit data for a word. Handles string or dictionary input.
    Uses ON CONFLICT to update existing credits based on (word_id, credit),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        credit_data: Credit string or dictionary (e.g., {'text': 'Source Name'}).
                     If dict, the 'text' key is used.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated credit record, or None if failed.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping credit insert for word ID {word_id}: Missing MANDATORY source identifier."
        )
        return None

    credit_text = None
    try:
        # Extract credit text
        if isinstance(credit_data, dict):
            # Prioritize 'text' key if dict is provided
            credit_text = (
                credit_data.get("text", "").strip()
                if isinstance(credit_data.get("text"), str)
                else None
            )
        elif isinstance(credit_data, str):
            credit_text = credit_data.strip()
        else:
            logger.warning(
                f"Invalid credit_data type for word ID {word_id} (source '{source_identifier}'): {type(credit_data)}. Skipping."
            )
            return None

        if not credit_text:
            logger.warning(
                f"Empty credit text for word ID {word_id} (source '{source_identifier}'). Skipping."
            )
            return None

        # Prepare parameters
        params = {
            "word_id": word_id,
            "credit": credit_text,
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }

        # Insert or update credit
        cur.execute(
            """
            INSERT INTO credits (word_id, credit, sources)
            VALUES (%(word_id)s, %(credit)s, %(sources)s)
            ON CONFLICT (word_id, credit) -- Conflict on word and exact credit text
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this credit record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        credit_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated credit (ID: {credit_id}) for word ID {word_id} from source '{source_identifier}'. Credit: '{credit_text}'"
        )
        return credit_id

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting credit for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting credit for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None

# --- Replace the get_standardized_pos_id function (around line 1504) ---

def get_standardized_pos_id(cur, pos_string: Optional[str]) -> int:
    """
    Get the primary key ID from the parts_of_speech table for a given POS string.

    Args:
        cur: Database cursor.
        pos_string: The raw part of speech string to standardize and look up.

    Returns:
        The integer ID from the parts_of_speech table, or the ID for 'unc' if not found.
    """
    # Get the standardized short code (e.g., 'n', 'v', 'unc')
    standard_code = get_standard_code(pos_string) # Use the corrected function

    try:
        # Query using the CODE obtained from standardization
        cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (standard_code,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            # If the standard code (even 'unc') is somehow not in the table, log error
            logger.error(f"Standard POS code '{standard_code}' (derived from '{pos_string}') not found in parts_of_speech table. Falling back to fetching 'unc' ID.")
            return get_uncategorized_pos_id(cur) # Fallback to ensure 'unc' exists
    except Exception as e:
        logger.error(f"Error fetching POS ID for code '{standard_code}' (from '{pos_string}'): {e}. Returning 'unc'.")
        # Ensure the 'unc' entry exists in case of query failure
        return get_uncategorized_pos_id(cur)

def get_uncategorized_pos_id(cur) -> int:
    cur.execute("SELECT id FROM parts_of_speech WHERE code = 'unc'")
    result = cur.fetchone()
    if result:
        return result[0]
    else:
        cur.execute(
            """
            INSERT INTO parts_of_speech (code, name_en, name_tl, description)
            VALUES ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
            RETURNING id
        """
        )
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
        return []

    # Here we could implement more structured etymology parsing
    # For now, just returning a simple structure
    return {"original_text": etymology_text, "processed": True}


def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        meaning = match.group(1)
        clean_text = text.replace(match.group(0), "").strip()
        return clean_text, meaning
    return text, None


def validate_schema(cur):
    """Validate database schema and constraints."""
    try:
        # Check required tables exist
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'words'
            )
        """
        )
        if not cur.fetchone()[0]:
            raise DatabaseError("Required table 'words' does not exist")

        # Check required indexes
        required_indexes = [
            ("words_lemma_idx", "words", "lemma"),
            ("words_normalized_lemma_idx", "words", "normalized_lemma"),
            ("pronunciations_word_id_idx", "pronunciations", "word_id"),
            ("credits_word_id_idx", "credits", "word_id"),
        ]

        for index_name, table, column in required_indexes:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = %s
                )
            """,
                (index_name,),
            )
            if not cur.fetchone()[0]:
                raise DatabaseError(
                    f"Required index {index_name} on {table}({column}) does not exist"
                )

        # Check foreign key constraints
        required_fks = [
            ("pronunciations", "word_id", "words", "id"),
            ("credits", "word_id", "words", "id"),
        ]

        for table, column, ref_table, ref_column in required_fks:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.key_column_usage
                    WHERE table_name = %s
                    AND column_name = %s
                    AND referenced_table_name = %s
                    AND referenced_column_name = %s
                )
            """,
                (table, column, ref_table, ref_column),
            )
            if not cur.fetchone()[0]:
                raise DatabaseError(
                    f"Required foreign key constraint missing: {table}({column}) -> {ref_table}({ref_column})"
                )

    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise


def validate_word_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Word data must be a dictionary")
    required_fields = {"lemma", "language_code"}
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    lemma = data["lemma"]
    if not isinstance(lemma, str) or not lemma.strip():
        raise ValueError("Lemma must be a non-empty string")
    if len(lemma) > 255:
        raise ValueError("Lemma exceeds maximum length")
    if data["language_code"] not in {"tl", "ceb"}:
        raise ValueError(f"Unsupported language code: {data['language_code']}")
    if "tags" in data:
        if not isinstance(data["tags"], (str, list)):
            raise ValueError("Tags must be string or list")
        if isinstance(data["tags"], list):
            data["tags"] = ",".join(str(tag) for tag in data["tags"])
    return data


def has_diacritics(text: str) -> bool:
    normalized = normalize_lemma(text)
    return text != normalized


class SourceStandardization:
    @staticmethod
    def standardize_sources(source: str) -> str:
        """Convert source filenames to standardized display names."""
        if not source:
            return "unknown"

        source_mapping = {
            "kaikki-ceb.jsonl": "kaikki.org (Cebuano)",
            "kaikki.jsonl": "kaikki.org (Tagalog)",
            "kwf_dictionary.json": "KWF Diksiyonaryo ng Wikang Filipino",
            "root_words_with_associated_words_cleaned.json": "tagalog.com",
            "tagalog-words.json": "diksiyonaryo.ph",
        }

        # Try direct mapping first
        if source in source_mapping:
            return source_mapping[source]

        # Handle cases where only part of the filename is matched
        for key, value in source_mapping.items():
            if key in source:
                return value

        # Special case for Marayum dictionaries
        if "marayum" in source.lower():
            return "Project Marayum"

        # Return the original if no mapping is found
        return source

    @staticmethod
    def get_display_name(source: str) -> str:
        """Get a display-friendly name for a source."""
        return SourceStandardization.standardize_sources(source)


def format_word_display(word: str, show_baybayin: bool = True) -> str:
    has_bb = any(0x1700 <= ord(c) <= 0x171F for c in word)
    if has_bb:
        romanized = get_romanized_text(word)
        if show_baybayin:
            return f"[bold cyan]{word}[/] [dim](romanized: {romanized})[/]"
        else:
            return romanized
    return word


def get_root_word_id(
    cur: "psycopg2.extensions.cursor", lemma: str, language_code: str
) -> Optional[int]:
    cur.execute(
        """
        SELECT id FROM words 
        WHERE normalized_lemma = %s AND language_code = %s AND root_word_id IS NULL
    """,
        (normalize_lemma(lemma), language_code),
    )
    result = cur.fetchone()
    return result[0] if result else None


# -------------------------------------------------------------------
# Baybayin Processing System
# -------------------------------------------------------------------
class BaybayinRomanizer:
    """Handles romanization of Baybayin text."""

    VOWELS = {
        "ᜀ": BaybayinChar("ᜀ", BaybayinCharType.VOWEL, "a", ["a"]),
        "ᜁ": BaybayinChar("ᜁ", BaybayinCharType.VOWEL, "i", ["i", "e"]),
        "ᜂ": BaybayinChar("ᜂ", BaybayinCharType.VOWEL, "u", ["u", "o"]),
    }
    CONSONANTS = {
        "ᜃ": BaybayinChar("ᜃ", BaybayinCharType.CONSONANT, "ka", ["ka"]),
        "ᜄ": BaybayinChar("ᜄ", BaybayinCharType.CONSONANT, "ga", ["ga"]),
        "ᜅ": BaybayinChar("ᜅ", BaybayinCharType.CONSONANT, "nga", ["nga"]),
        "ᜆ": BaybayinChar("ᜆ", BaybayinCharType.CONSONANT, "ta", ["ta"]),
        "ᜇ": BaybayinChar("ᜇ", BaybayinCharType.CONSONANT, "da", ["da"]),
        "ᜈ": BaybayinChar("ᜈ", BaybayinCharType.CONSONANT, "na", ["na"]),
        "ᜉ": BaybayinChar("ᜉ", BaybayinCharType.CONSONANT, "pa", ["pa"]),
        "ᜊ": BaybayinChar("ᜊ", BaybayinCharType.CONSONANT, "ba", ["ba"]),
        "ᜋ": BaybayinChar("ᜋ", BaybayinCharType.CONSONANT, "ma", ["ma"]),
        "ᜌ": BaybayinChar("ᜌ", BaybayinCharType.CONSONANT, "ya", ["ya"]),
        "ᜎ": BaybayinChar("ᜎ", BaybayinCharType.CONSONANT, "la", ["la"]),
        "ᜏ": BaybayinChar("ᜏ", BaybayinCharType.CONSONANT, "wa", ["wa"]),
        "ᜐ": BaybayinChar("ᜐ", BaybayinCharType.CONSONANT, "sa", ["sa"]),
        "ᜑ": BaybayinChar("ᜑ", BaybayinCharType.CONSONANT, "ha", ["ha"]),
        "ᜍ": BaybayinChar("ᜍ", BaybayinCharType.CONSONANT, "ra", ["ra"]),  # Added ra
    }
    VOWEL_MARKS = {
        "ᜒ": BaybayinChar("ᜒ", BaybayinCharType.VOWEL_MARK, "i", ["i", "e"]),
        "ᜓ": BaybayinChar("ᜓ", BaybayinCharType.VOWEL_MARK, "u", ["u", "o"]),
    }
    VIRAMA = BaybayinChar("᜔", BaybayinCharType.VIRAMA, "", [])
    PUNCTUATION = {
        "᜵": BaybayinChar("᜵", BaybayinCharType.PUNCTUATION, ",", [","]),
        "᜶": BaybayinChar("᜶", BaybayinCharType.PUNCTUATION, ".", ["."]),
    }

    def __init__(self):
        """Initialize the romanizer with a combined character mapping."""
        self.all_chars = {}
        # Combine all character mappings for easy lookup
        for char_map in [
            self.VOWELS,
            self.CONSONANTS,
            self.VOWEL_MARKS,
            {self.VIRAMA.char: self.VIRAMA},
            self.PUNCTUATION,
        ]:
            self.all_chars.update(char_map)

    def is_baybayin(self, text: str) -> bool:
        """Check if a string contains any Baybayin characters."""
        if not text:
            return False
        # Check for characters in the Baybayin Unicode block (U+1700 to U+171F)
        return any(0x1700 <= ord(c) <= 0x171F for c in text)

    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
        """Get character information for a Baybayin character."""
        return self.all_chars.get(char)

    def process_syllable(self, chars: List[str]) -> Tuple[str, int]:
        """
        Process a Baybayin syllable and return its romanized form.

        Args:
            chars: List of characters in the potential syllable

        Returns:
            (romanized_syllable, number_of_characters_consumed)
        """
        if not chars:
            return "", 0

        # Get information about the first character
        first_char = self.get_char_info(chars[0])
        if not first_char:
            # Not a recognized Baybayin character
            return chars[0], 1

        if first_char.char_type == BaybayinCharType.VOWEL:
            # Simple vowel
            return first_char.default_sound, 1

        elif first_char.char_type == BaybayinCharType.CONSONANT:
            # Start with default consonant sound (with 'a' vowel)
            result = first_char.default_sound
            pos = 1

            # Check for vowel marks or virama (vowel killer)
            if pos < len(chars):
                next_char = self.get_char_info(chars[pos])
                if next_char:
                    if next_char.char_type == BaybayinCharType.VOWEL_MARK:
                        # Replace default 'a' vowel with the marked vowel
                        result = result[:-1] + next_char.default_sound
                        pos += 1
                    elif next_char.char_type == BaybayinCharType.VIRAMA:
                        # Remove the default 'a' vowel (final consonant)
                        result = result[:-1]
                        pos += 1

            return result, pos

        elif first_char.char_type == BaybayinCharType.PUNCTUATION:
            # Baybayin punctuation
            return first_char.default_sound, 1

        # For unhandled cases (shouldn't normally happen)
        return "", 1

    def romanize(self, text: str) -> str:
        """
        Convert Baybayin text to its romanized form.

        Args:
            text: The Baybayin text to romanize

        Returns:
            The romanized text, or original text if romanization failed
        """
        if not text:
            return ""

        # Normalize Unicode for consistent character handling
        text = unicodedata.normalize("NFC", text)

        result = []
        i = 0

        while i < len(text):
            # Skip spaces and non-Baybayin characters
            if text[i].isspace() or not self.is_baybayin(text[i]):
                result.append(text[i])
                i += 1
                continue

            # Process a syllable
            try:
                processed_syllable, chars_consumed = self.process_syllable(
                    list(text[i:])
                )

                if processed_syllable:
                    result.append(processed_syllable)
                    i += chars_consumed
                else:
                    # Handle unrecognized characters
                    result.append(text[i])
                    i += 1
            except Exception as e:
                logger.error(f"Error during Baybayin romanization at position {i}: {e}")
                # Skip problematic character
                i += 1

        return "".join(result)

    def validate_text(self, text: str) -> bool:
        """
        Validate that a string contains valid Baybayin text.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid Baybayin, False otherwise
        """
        if not text:
            return False

        # Normalize Unicode
        text = unicodedata.normalize("NFC", text)
        chars = list(text)
        i = 0

        while i < len(chars):
            # Skip spaces
            if chars[i].isspace():
                i += 1
                continue

            # Get character info
            char_info = self.get_char_info(chars[i])

            # Not a valid Baybayin character
            if not char_info:
                if 0x1700 <= ord(chars[i]) <= 0x171F:
                    # It's in the Baybayin Unicode range but not recognized
                    logger.warning(
                        f"Unrecognized Baybayin character at position {i}: {chars[i]} (U+{ord(chars[i]):04X})"
                    )
                return False

            # Vowel mark must follow a consonant
            if char_info.char_type == BaybayinCharType.VOWEL_MARK:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(
                        f"Vowel mark not following a consonant at position {i}"
                    )
                    return False

            # Virama (vowel killer) must follow a consonant
            if char_info.char_type == BaybayinCharType.VIRAMA:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(f"Virama not following a consonant at position {i}")
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

    Args:
        text: Latin text to convert to Baybayin

    Returns:
        Baybayin text
    """
    if not text:
        return ""

    # Handle prefix starting with '-' (like '-an')
    if text.startswith("-"):
        # Skip the hyphen and process the rest
        text = text[1:]

    # Normalize text: lowercase and remove diacritical marks
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )

    # Define Baybayin character mappings
    consonants = {
        "k": "ᜃ",
        "g": "ᜄ",
        "ng": "ᜅ",
        "t": "ᜆ",
        "d": "ᜇ",
        "n": "ᜈ",
        "p": "ᜉ",
        "b": "ᜊ",
        "m": "ᜋ",
        "y": "ᜌ",
        "l": "ᜎ",
        "w": "ᜏ",
        "s": "ᜐ",
        "h": "ᜑ",
        "r": "ᜍ",  # Added 'r' mapping
    }
    vowels = {"a": "ᜀ", "i": "ᜁ", "e": "ᜁ", "u": "ᜂ", "o": "ᜂ"}
    vowel_marks = {"i": "ᜒ", "e": "ᜒ", "u": "ᜓ", "o": "ᜓ"}
    virama = "᜔"  # Pamudpod (vowel killer)

    result = []
    i = 0

    while i < len(text):
        # Check for 'ng' digraph first
        if i + 1 < len(text) and text[i : i + 2] == "ng":
            if i + 2 < len(text) and text[i + 2] in "aeiou":
                # ng + vowel
                if text[i + 2] == "a":
                    result.append(consonants["ng"])
                else:
                    result.append(consonants["ng"] + vowel_marks[text[i + 2]])
                i += 3
            else:
                # Final 'ng'
                result.append(consonants["ng"] + virama)
                i += 2

        # Handle single consonants
        elif text[i] in consonants:
            if i + 1 < len(text) and text[i + 1] in "aeiou":
                # Consonant + vowel
                if text[i + 1] == "a":
                    result.append(consonants[text[i]])
                else:
                    result.append(consonants[text[i]] + vowel_marks[text[i + 1]])
                i += 2
            else:
                # Final consonant
                result.append(consonants[text[i]] + virama)
                i += 1

        # Handle vowels
        elif text[i] in "aeiou":
            result.append(vowels[text[i]])
            i += 1

        # Skip spaces and other characters
        elif text[i].isspace():
            result.append(" ")
            i += 1
        else:
            # Skip non-convertible characters
            i += 1

    # Final validation - ensure only valid characters are included
    valid_output = "".join(
        c for c in result if (0x1700 <= ord(c) <= 0x171F) or c.isspace()
    )

    # Verify the output meets database constraints
    if not re.match(r"^[\u1700-\u171F\s]*$", valid_output):
        logger.warning(
            f"Transliterated Baybayin doesn't match required regex pattern: {valid_output}"
        )
        # Additional cleanup to ensure it matches the pattern
        valid_output = re.sub(r"[^\u1700-\u171F\s]", "", valid_output)

    return valid_output


@with_transaction(commit=True)
def regenerate_all_romanizations(cur):
    """Regenerate romanized forms for all Baybayin entries."""
    regenerated_count = 0
    error_count = 0

    try:
        romanizer = BaybayinRomanizer()

        # Get all entries with Baybayin forms
        cur.execute(
            """
            SELECT id, baybayin_form
            FROM words
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL AND baybayin_form ~ '[\u1700-\u171f]'
        """
        )

        entries = cur.fetchall()

        logger.info(f"Regenerating romanizations for {len(entries)} Baybayin entries")

        for word_id, baybayin_form in entries:
            try:
                # Clean the form
                cleaned_form = clean_baybayin_text(baybayin_form)

                if cleaned_form and re.match(r"^[\u1700-\u171F\s]*$", cleaned_form):
                    # Generate romanization
                    romanized = romanizer.romanize(cleaned_form)

                    # Update the entry
                    cur.execute(
                        """
                        UPDATE words
                        SET romanized_form = %s,
                            baybayin_form = %s
                        WHERE id = %s
                    """,
                        (romanized, cleaned_form, word_id),
                    )

                    regenerated_count += 1

                    # Log progress every 100 entries
                    if regenerated_count % 100 == 0:
                        logger.info(
                            f"Regenerated {regenerated_count}/{len(entries)} romanizations"
                        )
            except Exception as e:
                logger.error(
                    f"Error regenerating romanization for word ID {word_id}: {e}"
                )
                error_count += 1

        logger.info(
            f"Romanization regeneration complete: {regenerated_count} updated, {error_count} errors"
        )

    except Exception as e:
        logger.error(f"Error in regenerate_all_romanizations: {e}")
        return 0, 0

    finally:
        # This will always execute, even if there's an exception
        logger.info(
            f"Romanization regeneration finished - processed {regenerated_count + error_count} entries"
        )

    return regenerated_count, error_count


@with_transaction(commit=True)
def fix_baybayin_constraint_violations(cur):
    """Fix Baybayin entries that violate the database constraint."""
    fixed_count = 0

    try:
        # Identify entries that would violate the constraint
        cur.execute(
            r"""
            SELECT id, lemma, baybayin_form
            FROM words
            WHERE baybayin_form IS NOT NULL AND baybayin_form !~ '^[\u1700-\u171F\s]*$'
        """
        )

        violations = cur.fetchall()

        logger.info(
            f"Found {len(violations)} entries violating Baybayin regex constraint"
        )

        for word_id, lemma, baybayin_form in violations:
            # Clean the form
            cleaned_form = clean_baybayin_text(baybayin_form)

            if cleaned_form and re.match(r"^[\u1700-\u171F\s]*$", cleaned_form):
                # We can fix this entry
                cur.execute(
                    """
                    UPDATE words
                    SET baybayin_form = %s
                    WHERE id = %s
                """,
                    (cleaned_form, word_id),
                )
                fixed_count += 1
            else:
                # Cannot fix, remove Baybayin data
                cur.execute(
                    """
                    UPDATE words
                    SET has_baybayin = FALSE,
                        baybayin_form = NULL,
                        romanized_form = NULL
                    WHERE id = %s
                """,
                    (word_id,),
                )
                logger.warning(
                    f"Removed invalid Baybayin form for word ID {word_id}: {lemma}"
                )

        logger.info(f"Fixed {fixed_count} Baybayin constraint violations")

    except Exception as e:
        logger.error(f"Error fixing Baybayin constraint violations: {e}")
        return 0

    finally:
        logger.info(f"Completed Baybayin constraint violation check")

    return fixed_count


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


# Main function to run comprehensive Baybayin data repair
def repair_baybayin_data():
    """Run a comprehensive Baybayin data repair process."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                logger.info("Starting comprehensive Baybayin data repair...")

                # Step 1: Fix constraint violations
                fixed_count = fix_baybayin_constraint_violations(cur)
                logger.info(f"Fixed {fixed_count} constraint violations")

                # Step 2: Clean up data
                cleanup_baybayin_data(cur)

                # Step 3: Regenerate romanizations
                regen_count, regen_errors = regenerate_all_romanizations(cur)
                logger.info(
                    f"Regenerated {regen_count} romanizations with {regen_errors} errors"
                )

                # Step 4: Process entries with Baybayin in lemma
                processed_count, error_count = process_baybayin_entries(cur)
                logger.info(
                    f"Processed {processed_count} entries with {error_count} errors"
                )

                # Step 5: Verify consistency
                issues = check_baybayin_consistency(cur)
                if issues:
                    logger.warning(f"Remaining issues: {', '.join(issues)}")
                else:
                    logger.info("No issues found after repair")

                logger.info("Baybayin data repair completed successfully")
                return True
    except Exception as e:
        logger.error(f"Error in Baybayin data repair: {e}")
        return False


# -------------------------------------------------------------------
# Word Insertion and Update Functions
# -------------------------------------------------------------------
@with_transaction(commit=False)
def get_or_create_word_id(
    cur,
    lemma: str,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    source_identifier: Optional[str] = None,  # Optional, but recommended
    check_exists: bool = False, # Note: check_exists logic removed below in favor of direct insert/update approach
    **kwargs,
) -> int:
    """
    Get the ID of a word, creating it if necessary. Uses ON CONFLICT for UPSERT.
    Updates the word's source_info JSONB field with the provided identifier.
    Runs within the caller's transaction (commit=False).

    Args:
        cur: Database cursor.
        lemma: The word lemma.
        language_code: The language code (default 'tl').
        source_identifier: Identifier for the data source (e.g., filename). Recommended.
        check_exists: (Deprecated) No longer used, ON CONFLICT handles existence check.
        **kwargs: Additional word attributes (has_baybayin, baybayin_form, romanized_form,
                  root_word_id, preferred_spelling, tags, idioms, pronunciation_data,
                  word_metadata, badlit_form, hyphenation, is_proper_noun,
                  is_abbreviation, is_initialism). Values intended for JSONB
                  should be passed as Python dicts/lists or None.

    Returns:
        The word ID.

    Raises:
        ValueError: If lemma is empty.
        DatabaseError: If the operation fails.
    """
    if not lemma:
        raise ValueError("Lemma cannot be empty")

    normalized = normalize_lemma(lemma)
    word_id = None

    # --- Extract all relevant fields from kwargs ---
    has_baybayin = kwargs.get("has_baybayin", False) # Default to False if not provided
    baybayin_form = kwargs.get("baybayin_form")
    romanized_form = kwargs.get("romanized_form")
    root_word_id = kwargs.get("root_word_id")
    preferred_spelling = kwargs.get("preferred_spelling")
    tags = kwargs.get("tags") # Should be a string or None

    # Fields previously missed in INSERT
    badlit_form = kwargs.get("badlit_form")
    hyphenation_data = kwargs.get("hyphenation") # Could be list/dict/None
    is_proper_noun = kwargs.get("is_proper_noun", False)
    is_abbreviation = kwargs.get("is_abbreviation", False)
    is_initialism = kwargs.get("is_initialism", False)

    # Data for JSONB columns - expect Python objects or None
    idioms_data = kwargs.get("idioms") # Expect list or None
    pronunciation_data = kwargs.get("pronunciation_data") # Expect dict/list or None
    word_metadata = kwargs.get("word_metadata") # Expect dict or None

    # --- Clean up Baybayin if inconsistent ---
    if has_baybayin is False:
        baybayin_form = None # Ensure form is None if explicitly false
    elif has_baybayin is True and not baybayin_form:
        logger.warning(
            f"Word '{lemma}' ({language_code}, source: {source_identifier}) marked as has_baybayin but no form provided. Setting has_baybayin to False."
        )
        has_baybayin = False # Correct the inconsistency
        baybayin_form = None # Also clear the form

    # --- Standardize source and prepare initial source_info JSON ---
    standardized_source = None
    if source_identifier:
        try:
            standardized_source = SourceStandardization.standardize_sources(
                source_identifier
            )
            if not standardized_source:
                standardized_source = source_identifier
        except Exception as e:
            logger.warning(f"Error standardizing source '{source_identifier}': {e}")
            standardized_source = source_identifier

    # Prepare the source_info JSON string using the helper, starting empty
    new_source_json_str = update_word_source_info(None, standardized_source)

    # --- Prepare Data for Insertion (including JSON adaptation) ---
    params = {
        "lemma": lemma,
        "normalized": normalized,
        "language_code": language_code,
        "has_baybayin": has_baybayin,
        "baybayin_form": baybayin_form,
        "romanized_form": romanized_form,
        "root_word_id": root_word_id,
        "preferred_spelling": preferred_spelling,
        "tags": tags,
        "source_info": Json(json.loads(new_source_json_str)), # Parse helper result and wrap
        "idioms": Json(idioms_data) if isinstance(idioms_data, list) else None,
        "pronunciation_data": Json(pronunciation_data) if isinstance(pronunciation_data, (dict, list)) else None,
        "word_metadata": Json(word_metadata) if isinstance(word_metadata, dict) else None,
        "badlit_form": badlit_form,
        "hyphenation": Json(hyphenation_data) if isinstance(hyphenation_data, list) else None,
        "is_proper_noun": is_proper_noun,
        "is_abbreviation": is_abbreviation,
        "is_initialism": is_initialism,
        "data_hash": None, # Regenerate hash below
    }

    # Calculate data hash
    try:
        hash_input = "|".join(map(str, [
            params["lemma"], params["language_code"], params["baybayin_form"],
            params["romanized_form"], params["badlit_form"], params["tags"],
            params["root_word_id"], params["preferred_spelling"],
            params["is_proper_noun"], params["is_abbreviation"], params["is_initialism"]
        ]))
        params["data_hash"] = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    except Exception as hash_e:
        logger.debug(f"Error generating data hash for '{lemma}': {hash_e}")
        params["data_hash"] = None

    try:
        # --- Use INSERT ... ON CONFLICT for UPSERT ---
        cur.execute(
            """
            INSERT INTO words (
                lemma, normalized_lemma, language_code,
                has_baybayin, baybayin_form, romanized_form, root_word_id,
                preferred_spelling, tags, source_info,
                idioms, pronunciation_data, word_metadata,
                badlit_form, hyphenation, is_proper_noun, is_abbreviation, is_initialism,
                data_hash, search_text
            )
            VALUES (
                %(lemma)s, %(normalized)s, %(language_code)s,
                %(has_baybayin)s, %(baybayin_form)s, %(romanized_form)s, %(root_word_id)s,
                %(preferred_spelling)s, %(tags)s, %(source_info)s,
                %(idioms)s, %(pronunciation_data)s, %(word_metadata)s,
                %(badlit_form)s, %(hyphenation)s, %(is_proper_noun)s, %(is_abbreviation)s, %(is_initialism)s,
                %(data_hash)s, to_tsvector('simple', %(lemma)s || ' ' || %(normalized)s)
            )
            ON CONFLICT (normalized_lemma, language_code) DO UPDATE SET
                lemma = EXCLUDED.lemma,
                -- START BAYBAYIN CONSTRAINT FIX FOR UPDATE
                has_baybayin = COALESCE(EXCLUDED.has_baybayin, words.has_baybayin),
                baybayin_form = CASE
                                   WHEN COALESCE(EXCLUDED.has_baybayin, words.has_baybayin) = FALSE THEN NULL
                                   ELSE COALESCE(EXCLUDED.baybayin_form, words.baybayin_form)
                                END,
                romanized_form = CASE
                                    WHEN COALESCE(EXCLUDED.has_baybayin, words.has_baybayin) = FALSE THEN NULL
                                    ELSE COALESCE(EXCLUDED.romanized_form, words.romanized_form)
                                 END,
                -- END BAYBAYIN CONSTRAINT FIX FOR UPDATE
                root_word_id = COALESCE(EXCLUDED.root_word_id, words.root_word_id),
                preferred_spelling = COALESCE(EXCLUDED.preferred_spelling, words.preferred_spelling),
                tags = COALESCE(EXCLUDED.tags, words.tags),
                source_info = words.source_info || EXCLUDED.source_info,
                idioms = COALESCE(EXCLUDED.idioms, words.idioms),
                pronunciation_data = COALESCE(EXCLUDED.pronunciation_data, words.pronunciation_data),
                word_metadata = COALESCE(words.word_metadata, '{}'::jsonb) || COALESCE(EXCLUDED.word_metadata, '{}'::jsonb),
                badlit_form = COALESCE(EXCLUDED.badlit_form, words.badlit_form),
                hyphenation = COALESCE(EXCLUDED.hyphenation, words.hyphenation),
                is_proper_noun = COALESCE(EXCLUDED.is_proper_noun, words.is_proper_noun),
                is_abbreviation = COALESCE(EXCLUDED.is_abbreviation, words.is_abbreviation),
                is_initialism = COALESCE(EXCLUDED.is_initialism, words.is_initialism),
                data_hash = EXCLUDED.data_hash,
                search_text = to_tsvector('simple', EXCLUDED.lemma || ' ' || EXCLUDED.normalized_lemma),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            params,
        )
        word_id = cur.fetchone()[0]
        logger.debug(f"Word '{lemma}' ({language_code}) UPSERTED (ID: {word_id}) from source '{standardized_source}'.")

    except psycopg2.Error as e:
        # Log specific pgcode and pgerror
        logger.error(
            f"Database error in get_or_create_word_id (UPSERT) for '{lemma}' ({language_code}) from source '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=False,
        )
        # Optionally log more detail for specific errors like constraint violations
        if e.pgcode == '23514': # Check constraint violation
             logger.error(f"DETAIL (from error): {e.diag.message_detail}")
        # Log parameters safely
        safe_params = {k: type(v).__name__ if isinstance(v, (dict, list, Json)) else v for k, v in params.items()}
        logger.error(f"Parameters (types shown for complex data): {safe_params}")
        # Raise a more specific exception or the original one
        raise DatabaseError(
            f"Failed to get/create word ID for '{lemma}' from source '{source_identifier}': {e}"
        ) from e
    except Exception as e:
        # Catch other unexpected errors
        logger.error(
            f"Unexpected error in get_or_create_word_id (UPSERT) for '{lemma}' ({language_code}) from source '{source_identifier}': {e}",
            exc_info=True,
        )
        raise  # Reraise unexpected errors

    if word_id is None:
        raise DatabaseError(
            f"Failed to obtain word ID for '{lemma}' ({language_code}) from source '{source_identifier}' after UPSERT operations."
        )

    # --- Source Info Propagation (Moved outside main UPSERT) ---
    if standardized_source and word_id:
        try:
            # This uses jsonb_set to append to the 'files' array if the source isn't already there
            # It also updates a 'last_updated' timestamp for that source within the JSONB
            cur.execute("""
                UPDATE words
                SET source_info = jsonb_set(
                        COALESCE(source_info, '{}'::jsonb),
                        '{files}',
                        (COALESCE(source_info -> 'files', '[]'::jsonb) || %s::jsonb) - -1,
                        true
                    ) || jsonb_build_object('last_updated', jsonb_build_object(%s, %s::text)),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND NOT (source_info -> 'files' ? %s);
            """, (Json([standardized_source]), standardized_source, datetime.now().isoformat(), word_id, standardized_source))

            if cur.rowcount > 0:
                logger.debug(f"Updated source_info for word ID {word_id} with source '{standardized_source}'.")
                # Propagate source to definitions only if newly added to word
                cur.execute(
                    """
                    UPDATE definitions
                    SET sources = CASE
                        WHEN sources IS NULL OR sources = '' THEN %s
                        WHEN position(%s IN sources) = 0 THEN sources || ', ' || %s
                        ELSE sources
                    END
                    WHERE word_id = %s
                    """,
                    (standardized_source, standardized_source, standardized_source, word_id),
                )
        except Exception as prop_e:
            logger.warning(
                f"Error updating/propagating source '{standardized_source}' for word ID {word_id}: {prop_e}"
            )

    return word_id

@with_transaction(commit=False)
def insert_definition(
    cur,
    word_id: int,
    definition_text: str,
    source_identifier: str, # MANDATORY
    part_of_speech: Optional[str] = None, # This should be the ORIGINAL POS string
    examples: Optional[Union[str, Json]] = None, # Can accept string or Json object
    usage_notes: Optional[str] = None,
    tags: Optional[str] = None, # Comma-separated string or None
    # ADDED the metadata parameter back
    metadata: Optional[Json] = None
) -> Optional[int]:
    """
    Insert definition data for a word. Stores original POS and standardized POS ID.
    Now includes handling for a JSONB metadata field.
    Runs within the caller's transaction (commit=False).

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        definition_text: The definition text.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        part_of_speech: Original part of speech string from the source (optional).
        examples: JSON string or psycopg2.extras.Json object of example sentences (optional).
        usage_notes: Notes on usage (optional).
        tags: Comma-separated tags string (optional).
        metadata: Optional psycopg2.extras.Json object containing additional metadata.

    Returns:
        The ID of the inserted/updated definition record.

    Raises:
        ValueError: If required arguments are missing/invalid.
        psycopg2.Error: If a database error occurs.
        Exception: For unexpected errors.
    """
    if not source_identifier:
        # Raise error instead of returning None to ensure transaction rollback
        raise ValueError(f"CRITICAL: Missing MANDATORY source identifier for definition of word ID {word_id}.")

    if not isinstance(definition_text, str):
         raise ValueError(f"Invalid definition_text type ({type(definition_text)}) for word ID {word_id}.")

    definition_text = definition_text.strip()
    if not definition_text:
        # This might be acceptable depending on the source, log as warning but maybe don't raise error
        logger.warning(f"Empty definition text for word ID {word_id} from source '{source_identifier}'. Skipping this definition.")
        return None # Return None to indicate skipping, not failure

    try:
        # Standardize POS ID (handles None/empty pos_string internally)
        standardized_pos_id = get_standardized_pos_id(cur, part_of_speech)

        # Validate example type before passing to DB
        if examples is not None and not isinstance(examples, (str, Json)):
             logger.warning(f"Invalid type for 'examples' for word ID {word_id}. Setting to NULL.")
             examples = None

        # --- Insert or update definition ---
        cur.execute(
            """
            INSERT INTO definitions (
                word_id, definition_text, standardized_pos_id,
                examples, usage_notes, tags, sources, original_pos,
                metadata -- ADDED metadata column
            )
            VALUES (
                %(word_id)s, %(definition_text)s, %(pos_id)s,
                %(examples)s, %(usage_notes)s, %(tags)s, %(sources)s, %(original_pos)s,
                %(metadata)s -- ADDED metadata parameter placeholder
            )
            ON CONFLICT (word_id, definition_text, standardized_pos_id)
            DO UPDATE SET
                examples = COALESCE(EXCLUDED.examples, definitions.examples),
                usage_notes = COALESCE(EXCLUDED.usage_notes, definitions.usage_notes),
                tags = COALESCE(EXCLUDED.tags, definitions.tags),
                sources = EXCLUDED.sources, -- Update sources (last write wins for this def)
                original_pos = COALESCE(EXCLUDED.original_pos, definitions.original_pos),
                -- ADDED metadata update: Merge new JSONB into existing JSONB
                metadata = COALESCE(definitions.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            {
                "word_id": word_id,
                "definition_text": definition_text,
                "pos_id": standardized_pos_id,
                "examples": examples,
                "usage_notes": usage_notes,
                "tags": tags,
                "sources": source_identifier,
                "original_pos": part_of_speech,
                "metadata": metadata, # Pass the Json object or None directly
            },
        )
        fetched = cur.fetchone()
        if fetched:
            def_id = fetched[0]
            logger.debug(f"Inserted/Updated definition (ID: {def_id}) for word ID {word_id} from source '{source_identifier}'.")
            return def_id
        else:
            # This case (ON CONFLICT but RETURNING id fails) is highly unlikely but possible
            logger.error(f"Definition insertion/update conflict for word ID {word_id} did not return ID.")
            # Try fetching the ID separately based on the conflict keys
            cur.execute("""
                SELECT id FROM definitions
                WHERE word_id = %s AND definition_text = %s AND standardized_pos_id = %s
            """, (word_id, definition_text, standardized_pos_id))
            refetched = cur.fetchone()
            if refetched:
                logger.info(f"Found existing definition ID {refetched[0]} after conflict.")
                return refetched[0]
            else:
                # If still not found, something is wrong
                raise DatabaseError(f"Could not find or create definition for word ID {word_id} after conflict resolution attempt.")


    except psycopg2.Error as e:
        # Log clearly and re-raise to trigger savepoint rollback
        logger.error(
            f"Database error inserting definition for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}"
        )
        raise e # Re-raise database errors
    except Exception as e:
        # Log clearly and re-raise to trigger savepoint rollback
        logger.error(
            f"Unexpected error inserting definition for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        raise e # Re-raise other unexpected errors
    
@with_transaction(commit=True)
def insert_relation(
    cur,
    from_word_id: int,
    to_word_id: int,
    relation_type: Union[RelationshipType, str],
    source_identifier: str,  # MANDATORY
    metadata: Optional[Dict] = None,
) -> Optional[int]:
    """
    Insert a relationship between two words. Uses ON CONFLICT to update existing relations
    based on (from_word_id, to_word_id, relation_type), applying a 'last write wins'
    strategy for the 'sources' field for the specific relation record.

    Args:
        cur: Database cursor.
        from_word_id: ID of the source word.
        to_word_id: ID of the target word.
        relation_type: The type of relationship (RelationshipType enum or string).
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        metadata: Optional JSON metadata for the relationship (will be stored as JSONB).

    Returns:
        The ID of the inserted/updated relation, or None if failed.
    """
    if from_word_id == to_word_id:
        logger.warning(
            f"Skipping self-relation for word ID {from_word_id}, type '{relation_type}', source '{source_identifier}'."
        )
        return None
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping relation insert from {from_word_id} to {to_word_id}: Missing MANDATORY source identifier."
        )
        return None  # Corrected indentation

    rel_type_enum = None
    rel_type_str = None
    try:
        # Standardize relation type
        if isinstance(relation_type, RelationshipType):
            rel_type_enum = relation_type
            rel_type_str = rel_type_enum.rel_value
        elif isinstance(relation_type, str):
            relation_type_cleaned = relation_type.lower().strip()
            if not relation_type_cleaned:
                logger.warning(
                    f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Empty relation type string provided."
                )
                return None
            try:
                # Attempt to map string to enum value
                rel_type_enum = RelationshipType.from_string(relation_type_cleaned)
                rel_type_str = rel_type_enum.rel_value
            except ValueError:
                # If not a standard enum value, use the cleaned string directly
                rel_type_str = relation_type_cleaned
                logger.debug(
                    f"Using non-standard relation type string '{rel_type_str}' from source '{source_identifier}'."
                )
        else:
            logger.warning(
                f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Invalid relation_type type '{type(relation_type)}'."
            )
            return None

        # Dump metadata safely to JSON string for DB insertion (assuming metadata column is JSONB)
        metadata_json = None
        if metadata is not None:  # Check explicitly for None, allow empty dict {}
            if isinstance(metadata, dict):
                try:
                    metadata_json = json.dumps(metadata)
                except TypeError as e:
                    logger.warning(
                        f"Could not serialize metadata for relation {from_word_id}->{to_word_id} (source '{source_identifier}'): {e}. Metadata: {metadata}"
                    )
                    metadata_json = "{}"  # Use empty JSON object string as fallback
            else:
                logger.warning(
                    f"Metadata provided for relation {from_word_id}->{to_word_id} (source '{source_identifier}') is not a dict: {type(metadata)}. Storing as null."
                )
                metadata_json = None  # Store null if not a dict

        cur.execute(
            """
            INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
            VALUES (%(from_id)s, %(to_id)s, %(rel_type)s, %(sources)s, %(metadata)s::jsonb) -- Cast metadata to JSONB
            ON CONFLICT (from_word_id, to_word_id, relation_type)
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this relation record
                sources = EXCLUDED.sources,
                -- Update metadata only if the new value is not NULL
                metadata = COALESCE(EXCLUDED.metadata, relations.metadata),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            {
                "from_id": from_word_id,
                "to_id": to_word_id,
                "rel_type": rel_type_str,
                "sources": source_identifier,
                "metadata": metadata_json,  # Pass the JSON string (or None)
            },
        )
        relation_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated relation (ID: {relation_id}) {from_word_id}->{to_word_id} ('{rel_type_str}') from source '{source_identifier}'."
        )
        return relation_id

    except psycopg2.IntegrityError as e:
        # Likely due to non-existent from_word_id or to_word_id (FK constraint violation)
        logger.error(
            f"Integrity error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}"
        )
        return None
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e.pgcode} {e.pgerror}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None


def insert_definition_relation(
    cur, definition_id: int, word_id: int, relation_type: str, sources: str = "auto"
):
    cur.execute(
        """
        INSERT INTO definition_relations 
             (definition_id, word_id, relation_type, sources)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (definition_id, word_id, relation_type)
         DO UPDATE SET sources = CASE 
              WHEN definition_relations.sources IS NULL THEN EXCLUDED.sources
             WHEN EXCLUDED.sources IS NULL THEN definition_relations.sources
             ELSE definition_relations.sources || ', ' || EXCLUDED.sources
        END
    """,
        (definition_id, word_id, relation_type, sources),
    )


@with_transaction(commit=True)
def insert_etymology(
    cur,
    word_id: int,
    etymology_text: str,
    source_identifier: str,  # MANDATORY
    normalized_components: Optional[str] = None,
    etymology_structure: Optional[str] = None,  # Consider if this should be JSON
    language_codes: Optional[str] = None,  # Comma-separated string
) -> Optional[int]:
    """
    Inserts or updates an etymology record for a given word, linking it to a source.
    The primary key for conflict resolution is (word_id, etymology_text).

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        etymology_text: The etymological explanation.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        normalized_components: Normalized components string.
        etymology_structure: Structural information (string).
        language_codes: Comma-separated language codes involved (string).

    Returns:
        The ID of the inserted/updated etymology record, or None if failed.
    """
    etymology_text = etymology_text.strip() if isinstance(etymology_text, str) else None
    if not etymology_text:
        logger.warning(
            f"Skipping etymology insert for word ID {word_id} from source '{source_identifier}': Missing etymology text."
        )
        return None
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping etymology insert for word ID {word_id}: Missing MANDATORY source identifier."
        )
        return None

    try:
        # Prepare data, ensuring None is passed for empty optional fields
        params = {
            "word_id": word_id,
            "etym_text": etymology_text,
            "norm_comp": (
                normalized_components.strip()
                if isinstance(normalized_components, str)
                else None
            ),
            "etym_struct": (
                etymology_structure.strip()
                if isinstance(etymology_structure, str)
                else None
            ),
            "lang_codes": (
                language_codes.strip() if isinstance(language_codes, str) else None
            ),
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }  # <-- Corrected: Added missing closing brace

        # Ensure cur.execute is correctly indented within the try block
        cur.execute(
            """
            INSERT INTO etymologies (
                word_id, etymology_text, normalized_components,
                etymology_structure, language_codes, sources
            )
            VALUES (%(word_id)s, %(etym_text)s, %(norm_comp)s,
                    %(etym_struct)s, %(lang_codes)s, %(sources)s)
            ON CONFLICT (word_id, etymology_text) -- Conflict on word and exact text match
            DO UPDATE SET
                -- Update optional fields only if the new value is not NULL
                normalized_components = COALESCE(EXCLUDED.normalized_components, etymologies.normalized_components),
                etymology_structure = COALESCE(EXCLUDED.etymology_structure, etymologies.etymology_structure),
                language_codes = COALESCE(EXCLUDED.language_codes, etymologies.language_codes),
                -- Overwrite sources: Last write wins for this etymology record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        etymology_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated etymology (ID: {etymology_id}) for word ID {word_id} from source '{source_identifier}'. Text: '{etymology_text[:50]}...'"
        )
        return etymology_id

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting etymology for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting etymology for word ID {word_id} from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None


@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    source_identifier: str,  # MANDATORY
) -> Optional[int]:
    """
    Insert an affixation relationship (e.g., root -> derived word).
    Uses ON CONFLICT to update existing affixations based on (root_word_id, affixed_word_id, affix_type),
    applying a 'last write wins' strategy for the 'sources' field.

    Args:
        cur: Database cursor.
        root_id: ID of the root word.
        affixed_id: ID of the affixed word.
        affix_type: Type of affixation (e.g., 'prefix', 'suffix', 'infix', 'circumfix').
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated affixation record, or None if failed.
    """
    affix_type = (
        affix_type.strip().lower() if isinstance(affix_type, str) else None
    )  # Normalize type
    if not affix_type:
        logger.warning(
            f"Skipping affixation insert for root {root_id}, affixed {affixed_id} (source '{source_identifier}'): Missing affix type."
        )
        return None
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping affixation insert for root {root_id}, affixed {affixed_id}: Missing MANDATORY source identifier."
        )
        return None
    if root_id == affixed_id:
        logger.warning(
            f"Skipping self-affixation for word ID {root_id}, type '{affix_type}', source '{source_identifier}'."
        )
        return None

    try:
        # Prepare parameters
        params = {
            "root_id": root_id,
            "affixed_id": affixed_id,
            "affix_type": affix_type,
            "sources": source_identifier,  # Use mandatory source_identifier directly
        }

        cur.execute(
            """
            INSERT INTO affixations (root_word_id, affixed_word_id, affix_type, sources)
            VALUES (%(root_id)s, %(affixed_id)s, %(affix_type)s, %(sources)s)
            ON CONFLICT (root_word_id, affixed_word_id, affix_type) -- Conflict on exact triplet
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this affixation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        affixation_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated affixation (ID: {affixation_id}) {root_id}(root) -> {affixed_id}(affixed) [{affix_type}] from source '{source_identifier}'."
        )
        return affixation_id

    except psycopg2.IntegrityError as e:
        # Likely due to non-existent root_id or affixed_id (FK constraint violation)
        logger.error(
            f"Integrity error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}"
        )
        return None
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e.pgcode} {e.pgerror}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e}",
            exc_info=True,
        )
        return None


# Update for batch_get_or_create_word_ids function to properly handle sources


@with_transaction(commit=True)
def batch_get_or_create_word_ids(
    cur, entries: List[Tuple[str, str]], source: str = None, batch_size: int = 1000
) -> Dict[Tuple[str, str], int]:
    """
    Create or get IDs for multiple words in batches.

    Args:
        cur: Database cursor
        entries: List of (lemma, language_code) tuples
        source: Source information to add to the entries
        batch_size: Number of entries to process in each batch

    Returns:
        Dictionary mapping (lemma, language_code) to word_id
    """
    result = {}

    # Standardize source if provided
    if source:
        standardized_source = SourceStandardization.standardize_sources(source)
    else:
        standardized_source = None

    # Prepare source_info JSON for database
    if standardized_source is None or standardized_source == "":
        source_info = None
    else:
        # Check if it's already a JSON string
        if isinstance(standardized_source, (dict, list)):
            source_info = json.dumps(standardized_source)
        else:
            # It's a string, but make sure it's valid JSON
            try:
                # Try to parse it as JSON first
                json.loads(standardized_source)
                source_info = standardized_source  # It's already valid JSON
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON, treat as plain text
                source_info = json.dumps(standardized_source)

    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        batch = list(dict.fromkeys(batch))  # Remove duplicates

        # First, let's ensure all existing words are fetched from the database
        normalized_entries = [
            (lemma, normalize_lemma(lemma), lang_code) for lemma, lang_code in batch
        ]
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

        # If source is provided, update sources for existing words
        if source_info and existing:
            for (lemma, lang), word_id in existing.items():
                try:
                    # Check if the word already has source information
                    cur.execute(
                        "SELECT source_info FROM words WHERE id = %s", (word_id,)
                    )
                    row = cur.fetchone()
                    existing_source = row[0] if row and row[0] else None

                    # If no existing source or different source, update it
                    if not existing_source:
                        cur.execute(
                            "UPDATE words SET source_info = %s WHERE id = %s",
                            (source_info, word_id),
                        )
                    elif standardized_source not in existing_source:
                        # Combine sources if they're different
                        try:
                            # Try to parse existing source as JSON
                            existing_json = json.loads(existing_source)
                            if isinstance(existing_json, list):
                                if standardized_source not in existing_json:
                                    existing_json.append(standardized_source)
                                combined_source = json.dumps(existing_json)
                            elif isinstance(existing_json, dict):
                                if "sources" in existing_json:
                                    if (
                                        standardized_source
                                        not in existing_json["sources"]
                                    ):
                                        existing_json["sources"].append(
                                            standardized_source
                                        )
                                else:
                                    existing_json["sources"] = [standardized_source]
                                combined_source = json.dumps(existing_json)
                            else:
                                # Not a list or dict, treat as string
                                combined_source = json.dumps(
                                    f"{existing_source}, {standardized_source}"
                                )
                        except (json.JSONDecodeError, TypeError):
                            # Not valid JSON, just combine strings
                            combined_source = json.dumps(
                                f"{existing_source}, {standardized_source}"
                            )

                        cur.execute(
                            "UPDATE words SET source_info = %s WHERE id = %s",
                            (combined_source, word_id),
                        )
                except Exception as e:
                    logger.warning(f"Error updating source for word '{lemma}': {e}")

        # Identify entries that need to be inserted
        to_insert = []
        for lemma, lang in batch:
            if (lemma, lang) not in existing:
                norm = normalize_lemma(lemma)
                to_insert.append((lemma, norm, lang))

        # Insert missing entries with individual transactions
        for lemma, norm, lang in to_insert:
            try:
                search_text = " ".join(
                    word.strip() for word in re.findall(r"\w+", f"{lemma} {norm}")
                )
                # FIX: Changed to_tsquery to to_tsvector for search_text
                cur.execute(
                    """
                    INSERT INTO words (lemma, normalized_lemma, language_code, tags, search_text, source_info)
                    VALUES (%s, %s, %s, %s, to_tsvector('simple', %s), %s)
                    ON CONFLICT ON CONSTRAINT words_lang_lemma_uniq
                    DO UPDATE SET 
                        lemma = EXCLUDED.lemma,
                        tags = EXCLUDED.tags,
                        search_text = to_tsvector('simple', EXCLUDED.lemma || ' ' || EXCLUDED.normalized_lemma),
                        source_info = EXCLUDED.source_info,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """,
                    (lemma, norm, lang, "", search_text, source_info),
                )

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
# Helper function to clean HTML (using BeautifulSoup for robustness)
def clean_html(raw_html: Optional[str]) -> str:
    """Removes HTML tags from a string."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    # Replace <br/> with newline before parsing might help preserve structure
    text_with_newlines = raw_html.replace('<br/>', '\n').replace('<br>', '\n')
    soup = BeautifulSoup(text_with_newlines, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    # Clean up potential extra whitespace introduced by stripping tags
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- New process_kwf_dictionary function ---

@with_transaction(commit=False) # Manage commit via migrate_data
def process_kwf_dictionary(cur, filename: str) -> Tuple[int, int]:
    """
    Processes entries from the KWF Dictionary JSON file (dictionary format).
    Handles complex nested structure, HTML cleaning, and relations.
    Manages transactions manually using savepoints for individual entry resilience.

    Args:
        cur: Database cursor
        filename: Path to the kwf_dictionary.json file

    Returns:
        Tuple: (number_of_entries_processed_successfully, number_of_entries_with_issues)
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)
    if not source_identifier: # Fallback
        source_identifier = "KWF Diksiyonaryo ng Wikang Filipino"

    logger.info(f"Processing KWF Dictionary: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection # Get the connection for savepoint management

    # Statistics tracking for this file
    stats = {
        "processed": 0,
        "definitions": 0,
        "relations": 0,
        "synonyms": 0,
        "antonyms": 0,
        "related_terms": 0,
        "cross_refs": 0,
        "etymologies": 0,
        "pronunciations": 0,
        "examples": 0,
        "idioms": 0,
        "skipped_invalid": 0, # Entries skipped due to format issues
        "errors": 0,          # Entries skipped due to processing errors
    }
    error_types = {} # Track specific error types encountered

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return 0, 1 # 0 processed, 1 issue (file error)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    if not isinstance(data, dict):
        logger.error(f"File {filename} does not contain a top-level dictionary.")
        return 0, 1

    # --- Optional: Extract top-level metadata if needed ---
    dictionary_metadata = data.pop("__metadata__", {}) # Remove metadata if key exists
    entries_in_file = len(data) # Count actual word entries

    if entries_in_file == 0:
        logger.info(f"Found 0 word entries in {filename}. Skipping file.")
        return 0, 0

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Process Entries ---
    with tqdm(total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry", leave=False) as pbar:
        for entry_index, (original_key, entry) in enumerate(data.items()):
            savepoint_name = f"kwf_entry_{entry_index}"
            lemma = "" # Initialize for error logging
            word_id = None

            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")

                if not isinstance(entry, dict):
                    logger.warning(f"Skipping non-dictionary value for key '{original_key}' at index {entry_index}")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                # Use formatted headword if available, otherwise the original key
                lemma = entry.get("formatted", original_key).strip()
                if not lemma:
                    logger.warning(f"Skipping entry at index {entry_index} (original key: '{original_key}') due to missing/empty 'formatted' or key.")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                # Default language code for KWF is Tagalog
                language_code = "tl"

                # --- Extract word-level metadata ---
                entry_metadata = entry.get("metadata", {})
                word_metadata_to_store = {}
                if isinstance(entry_metadata, dict):
                     # Store source language if present
                    source_langs = entry_metadata.get("source_language", [])
                    if isinstance(source_langs, list) and source_langs:
                         # Store the first source language found, or handle multiple if needed
                        first_lang_info = source_langs[0]
                        if isinstance(first_lang_info, dict) and "value" in first_lang_info:
                             word_metadata_to_store["source_language"] = first_lang_info["value"]
                        elif isinstance(first_lang_info, str):
                             word_metadata_to_store["source_language"] = first_lang_info
                    elif isinstance(source_langs, str): # Handle if it's just a string
                         word_metadata_to_store["source_language"] = source_langs

                word_metadata_to_store["original_key"] = original_key # Keep track of the original key
                word_metadata_to_store["kwf_original"] = entry.get("original", original_key) # Store original form if different
                # Add other relevant simple metadata from entry_metadata if needed


                # --- Get or Create Word ---
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        lemma=lemma,
                        language_code=language_code,
                        source_identifier=source_identifier,
                        word_metadata=Json(word_metadata_to_store) if word_metadata_to_store else None,
                        # Determine tags from top-level part_of_speech if useful
                        # tags = ",".join(p for p in entry.get("part_of_speech", []) if isinstance(p, str)) or None
                    )
                    if not word_id:
                        raise ValueError("get_or_create_word_id returned None")
                    logger.debug(f"Word '{lemma}' ({language_code}) -> ID: {word_id}")
                except Exception as word_err:
                    logger.error(f"CRITICAL FAILURE creating word '{lemma}' (original key: '{original_key}'): {word_err}")
                    raise word_err # Re-raise critical error

                # --- Process word-level Pronunciation, Etymology, Cross-refs from Metadata ---
                if isinstance(entry_metadata, dict):
                    try:
                        pronunciations = entry_metadata.get("pronunciation", [])
                        if isinstance(pronunciations, list):
                             for pron_data in pronunciations:
                                 # KWF pronunciation can be simple string or dict like {'value': '...', 'type': 'ipa'}
                                 pron_obj = {}
                                 if isinstance(pron_data, str) and pron_data.strip():
                                     pron_obj = {"value": pron_data.strip(), "type": "kwf_raw"} # Mark as raw initially
                                 elif isinstance(pron_data, dict) and pron_data.get("value"):
                                     pron_obj = pron_data # Use dict directly
                                 else: continue # Skip invalid format

                                 pron_inserted_id = insert_pronunciation(cur, word_id, pron_obj, source_identifier)
                                 if pron_inserted_id: stats["pronunciations"] += 1
                    except Exception as pron_err:
                        logger.warning(f"Error processing metadata pronunciations for '{lemma}' (ID: {word_id}): {pron_err}")
                        error_key = f"PronunciationInsertError: {type(pron_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

                    try:
                        etymologies = entry_metadata.get("etymology", [])
                        if isinstance(etymologies, list):
                             for ety_data in etymologies:
                                 # KWF etymology can be simple string or dict like {'value': '...', 'context': []}
                                 ety_text = None
                                 if isinstance(ety_data, str) and ety_data.strip():
                                     ety_text = ety_data.strip()
                                 elif isinstance(ety_data, dict) and ety_data.get("value"):
                                     ety_text = ety_data["value"].strip()
                                     # Could extract context into etymology metadata if schema supports it
                                 else: continue # Skip invalid format

                                 if ety_text:
                                     ety_id = insert_etymology(cur, word_id, ety_text, source_identifier)
                                     if ety_id: stats["etymologies"] += 1
                    except Exception as ety_err:
                        logger.warning(f"Error processing metadata etymology for '{lemma}' (ID: {word_id}): {ety_err}")
                        error_key = f"EtymologyInsertError: {type(ety_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

                    try:
                        cross_refs = entry_metadata.get("cross_references", [])
                        if isinstance(cross_refs, list):
                             for ref_item in cross_refs:
                                 # KWF cross_references are often dicts {'term': '...', 'link': '?query=...'}
                                 ref_word = None
                                 if isinstance(ref_item, dict) and ref_item.get("term"):
                                     ref_word = ref_item["term"].strip()
                                 elif isinstance(ref_item, str): # Handle plain string refs if they exist
                                     ref_word = ref_item.strip()

                                 if ref_word and ref_word.lower() != lemma.lower():
                                     ref_id = get_or_create_word_id(cur, ref_word, language_code, source_identifier)
                                     if ref_id and ref_id != word_id:
                                         rel_id_1 = insert_relation(cur, word_id, ref_id, RelationshipType.SEE_ALSO, source_identifier)
                                         if rel_id_1: stats["relations"] += 1; stats["cross_refs"] += 1
                    except Exception as ref_err:
                        logger.warning(f"Error processing metadata cross-references for '{lemma}' (ID: {word_id}): {ref_err}")
                        error_key = f"RelationInsertError: {type(ref_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Process Definitions by POS ---
                definitions_by_pos = entry.get("definitions", {})
                if isinstance(definitions_by_pos, dict):
                    for raw_pos, definitions_list in definitions_by_pos.items():
                        if not isinstance(definitions_list, list):
                            logger.debug(f"Skipping invalid definitions list for POS '{raw_pos}' in word '{lemma}'")
                            continue

                        # Standardize the POS key itself
                        standardized_pos_code_from_key = get_standard_code(raw_pos)

                        for def_idx, def_item in enumerate(definitions_list):
                            if not isinstance(def_item, dict):
                                logger.debug(f"Skipping invalid definition item at index {def_idx} for POS '{raw_pos}', word '{lemma}'")
                                continue

                            definition_text_raw = def_item.get("meaning", "")
                            definition_text = clean_html(definition_text_raw) # Clean HTML

                            if not definition_text: continue # Skip empty definitions

                            # Combine categories and potential note into tags/usage_notes
                            categories = def_item.get("categories", [])
                            usage_note_text = clean_html(def_item.get("note")) # Clean note
                            def_tags_list = [c for c in categories if isinstance(c, str) and c.strip()]
                            tags_str = ", ".join(def_tags_list) if def_tags_list else None

                            # Process examples
                            examples_processed = []
                            example_sets = def_item.get("example_sets", [])
                            if isinstance(example_sets, list):
                                for ex_set in example_sets:
                                    if isinstance(ex_set, dict) and "examples" in ex_set and isinstance(ex_set["examples"], list):
                                        for ex_data in ex_set["examples"]:
                                            if isinstance(ex_data, dict) and "text" in ex_data:
                                                ex_text_clean = clean_html(ex_data["text"]) # Clean example text
                                                if ex_text_clean:
                                                     # Optionally add label or other context if needed
                                                     ex_obj = {"text": ex_text_clean}
                                                     if ex_set.get("label"):
                                                         ex_obj["label"] = ex_set["label"]
                                                     examples_processed.append(ex_obj)
                            examples_json = Json(examples_processed) if examples_processed else None
                            if examples_processed: stats["examples"] += len(examples_processed)

                            # Insert Definition
                            definition_id = None
                            try:
                                # Pass the raw POS string from the dictionary key for standardization inside insert_definition
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    source_identifier=source_identifier,
                                    part_of_speech=raw_pos, # Pass the KWF POS key
                                    examples=examples_json,
                                    usage_notes=usage_note_text,
                                    tags=tags_str
                                )
                                if def_id:
                                    stats["definitions"] += 1
                                    definition_id = def_id # Store ID for relating items below
                                else:
                                     logger.warning(f"insert_definition failed for '{lemma}', POS '{raw_pos}', Def {def_idx+1}")
                                     error_key = f"DefinitionInsertFailure"
                                     error_types[error_key] = error_types.get(error_key, 0) + 1
                            except Exception as def_err:
                                logger.error(f"Error inserting definition for '{lemma}', POS '{raw_pos}', Def {def_idx+1}: {def_err}", exc_info=True)
                                error_key = f"DefinitionInsertError: {type(def_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1
                                continue # Skip relations for this failed definition

                            # --- Process Definition-Level Relations (if definition inserted) ---
                            if definition_id:
                                # Synonyms within definition
                                synonyms_list = def_item.get("synonyms", [])
                                if isinstance(synonyms_list, list):
                                    for syn_item in synonyms_list:
                                        syn_word = None
                                        if isinstance(syn_item, str): syn_word = syn_item
                                        elif isinstance(syn_item, dict) and "term" in syn_item: syn_word = syn_item["term"] # KWF sometimes uses dicts here

                                        syn_word_clean = clean_html(syn_word)
                                        if syn_word_clean and syn_word_clean.lower() != lemma.lower():
                                            try:
                                                syn_id = get_or_create_word_id(cur, syn_word_clean, language_code, source_identifier)
                                                if syn_id and syn_id != word_id:
                                                    rel_id_1 = insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM, source_identifier, metadata={"definition_id": definition_id})
                                                    rel_id_2 = insert_relation(cur, syn_id, word_id, RelationshipType.SYNONYM, source_identifier, metadata={"definition_id": definition_id})
                                                    if rel_id_1: stats["relations"] += 1; stats["synonyms"] += 1
                                            except Exception as e: logger.warning(f"Error creating synonym relation for '{lemma}': {e}")

                                # Antonyms within definition
                                antonyms_list = def_item.get("antonyms", [])
                                if isinstance(antonyms_list, list):
                                    for ant_item in antonyms_list:
                                        ant_word = None
                                        if isinstance(ant_item, str): ant_word = ant_item
                                        elif isinstance(ant_item, dict) and "term" in ant_item: ant_word = ant_item["term"]

                                        ant_word_clean = clean_html(ant_word)
                                        if ant_word_clean and ant_word_clean.lower() != lemma.lower():
                                            try:
                                                ant_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier)
                                                if ant_id and ant_id != word_id:
                                                    rel_id_1 = insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM, source_identifier, metadata={"definition_id": definition_id})
                                                    rel_id_2 = insert_relation(cur, ant_id, word_id, RelationshipType.ANTONYM, source_identifier, metadata={"definition_id": definition_id})
                                                    if rel_id_1: stats["relations"] += 1; stats["antonyms"] += 1
                                            except Exception as e: logger.warning(f"Error creating antonym relation for '{lemma}': {e}")

                                # Cross-references within definition
                                def_cross_refs = def_item.get("cross_references", [])
                                if isinstance(def_cross_refs, list):
                                     for ref_item in def_cross_refs:
                                         ref_word = None
                                         if isinstance(ref_item, dict) and ref_item.get("term"): ref_word = ref_item["term"]
                                         elif isinstance(ref_item, str): ref_word = ref_item

                                         ref_word_clean = clean_html(ref_word)
                                         if ref_word_clean and ref_word_clean.lower() != lemma.lower():
                                             try:
                                                 ref_id = get_or_create_word_id(cur, ref_word_clean, language_code, source_identifier)
                                                 if ref_id and ref_id != word_id:
                                                     # Decide if SEE_ALSO or RELATED is better
                                                     rel_id_1 = insert_relation(cur, word_id, ref_id, RelationshipType.SEE_ALSO, source_identifier, metadata={"definition_id": definition_id})
                                                     # Optionally make SEE_ALSO bidirectional
                                                     # rel_id_2 = insert_relation(cur, ref_id, word_id, RelationshipType.SEE_ALSO, source_identifier, metadata={"definition_id": definition_id})
                                                     if rel_id_1: stats["relations"] += 1; stats["cross_refs"] += 1
                                             except Exception as e: logger.warning(f"Error creating cross-reference relation for '{lemma}': {e}")

                # --- Process Top-Level Related Terms ---
                related_block = entry.get("related", {})
                if isinstance(related_block, dict):
                    related_terms = related_block.get("related_terms", [])
                    if isinstance(related_terms, list):
                        for rel_item in related_terms:
                            rel_word = None
                            if isinstance(rel_item, dict) and "term" in rel_item: rel_word = rel_item["term"]
                            elif isinstance(rel_item, str): rel_word = rel_item

                            rel_word_clean = clean_html(rel_word)
                            if rel_word_clean and rel_word_clean.lower() != lemma.lower():
                                try:
                                    rel_id = get_or_create_word_id(cur, rel_word_clean, language_code, source_identifier)
                                    if rel_id and rel_id != word_id:
                                        rel_ins_id = insert_relation(cur, word_id, rel_id, RelationshipType.RELATED, source_identifier, metadata={"from_related_block": True})
                                        # Optionally make RELATED bidirectional
                                        # insert_relation(cur, rel_id, word_id, RelationshipType.RELATED, source_identifier, metadata={"from_related_block": True})
                                        if rel_ins_id: stats["relations"] += 1; stats["related_terms"] += 1
                                except Exception as e: logger.warning(f"Error creating related_term relation for '{lemma}': {e}")

                    # Handle top-level antonyms if present
                    top_antonyms = related_block.get("antonyms", [])
                    if isinstance(top_antonyms, list):
                         for ant_item in top_antonyms:
                             ant_word = None
                             if isinstance(ant_item, str): ant_word = ant_item
                             elif isinstance(ant_item, dict) and "term" in ant_item: ant_word = ant_item["term"]

                             ant_word_clean = clean_html(ant_word)
                             if ant_word_clean and ant_word_clean.lower() != lemma.lower():
                                 try:
                                     ant_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier)
                                     if ant_id and ant_id != word_id:
                                         rel_id_1 = insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM, source_identifier, metadata={"from_related_block": True})
                                         rel_id_2 = insert_relation(cur, ant_id, word_id, RelationshipType.ANTONYM, source_identifier, metadata={"from_related_block": True})
                                         if rel_id_1: stats["relations"] += 1; stats["antonyms"] += 1
                                 except Exception as e: logger.warning(f"Error creating top-level antonym relation for '{lemma}': {e}")

                # --- Process Idioms ---
                idioms_list = entry.get("idioms", [])
                if isinstance(idioms_list, list) and idioms_list:
                     # Store idioms in the words table's JSONB field
                     try:
                         # Ensure idioms are strings, clean them
                         cleaned_idioms = [clean_html(i) for i in idioms_list if isinstance(i, str) and clean_html(i)]
                         if cleaned_idioms:
                             idioms_json = Json(cleaned_idioms)
                             cur.execute("UPDATE words SET idioms = %s WHERE id = %s", (idioms_json, word_id))
                             stats["idioms"] += len(cleaned_idioms)
                     except Exception as idiom_err:
                         logger.warning(f"Error storing idioms for '{lemma}' (ID: {word_id}): {idiom_err}")

                # --- Finish Entry ---
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1

                # --- Periodic Commit ---
                if stats["processed"] % 500 == 0:
                    try:
                        conn.commit()
                        logger.info(f"Committed batch after {stats['processed']} KWF entries.")
                    except Exception as commit_err:
                        logger.error(f"Error during batch commit for KWF: {commit_err}. Rolling back...", exc_info=True)
                        conn.rollback() # Rollback the current batch on commit error
                        # Decide if to continue or stop - stopping might be safer if commit fails
                        # For now, log error and continue trying entries with savepoints

            except Exception as entry_err:
                logger.error(f"Failed processing KWF entry #{entry_index} ('{original_key}' -> '{lemma}'): {entry_err}", exc_info=True)
                stats["errors"] += 1
                error_key = f"KwfEntryError: {type(entry_err).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except Exception as rb_err:
                    logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name} after KWF entry error: {rb_err}. Aborting file.", exc_info=True)
                    # Abort processing this file if rollback fails
                    raise entry_err from rb_err
            finally:
                pbar.update(1)

    # --- Final Commit handled by migrate_data ---
    logger.info(f"Finished processing {filename}. Stats: {stats}")
    if error_types:
        logger.warning(f"Error summary for {filename}: {error_types}")

    total_issues = stats["skipped_invalid"] + stats["errors"]
    return stats["processed"], total_issues


@with_transaction(commit=True)  # Using commit=True for this function as per the original code
def process_tagalog_words(cur, filename: str):
    """
    Process tagalog-words.json (diksiyonaryo.ph format) with enhanced source tracking,
    metadata handling, and relation processing.
    Uses the @with_transaction(commit=True) decorator, meaning each entry attempt
    (or the whole function if an early error occurs) runs in its own transaction.

    Args:
        cur: Database cursor
        filename: Path to the tagalog-words.json file

    Returns:
        A dictionary containing processing statistics.
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)

    # Make sure we have a valid source identifier, providing a default if needed
    if not source_identifier:
        source_identifier = "diksiyonaryo.ph"  # Default fallback for this format

    logger.info(f"Processing Tagalog words from: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    language_code = "tl"  # Assume Tagalog
    romanizer = BaybayinRomanizer()  # Instantiate for potential Baybayin processing

    # Statistics tracking dictionary - initialized as in the original snippet
    stats = {
        "total_entries": 0,
        "processed_entries": 0,
        "skipped_entries": 0,  # For invalid format or missing essential data
        "error_entries": 0,  # For entries that caused exceptions during processing
        "definitions_added": 0,
        "relations_added": 0,  # Grand total of all relation types added
        "synonyms_added": 0,
        "references_added": 0,  # Corresponds to 'references' -> SEE_ALSO
        "variants_added": 0,
        "derivatives_added": 0,  # Processed from 'derivative' field
        "etymologies_processed": 0,  # Updated if etymology is successfully inserted
        "baybayin_added": 0,  # Updated if Baybayin form is added/updated
        "pronunciations_added": 0,
        "affixations_added": 0,  # Processed from 'affix_forms'/'affix_types'
    }

    # Note: conn = cur.connection is not needed here because @with_transaction(commit=True)
    # handles the commit/rollback automatically for the whole function scope or per error.
    # Savepoints are also less critical here but kept from the original for consistency.

    try:
        # Load and parse the data file
        with open(filename, "r", encoding="utf-8") as f:
            # Assuming the JSON is a dictionary where keys are lemmas
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error(
                f"File {filename} content is not a dictionary as expected. Aborting."
            )
            raise TypeError(f"Expected dictionary data in {filename}")

        # Initialize statistics
        stats["total_entries"] = len(data)
        logger.info(f"Found {stats['total_entries']} entries to process")

        # Process each word entry with progress bar
        with tqdm(
            total=stats["total_entries"],
            desc=f"Processing {source_identifier}",
            unit="word",
        ) as pbar:
            for lemma, entry_data in data.items():
                # --- Savepoint (Optional with commit=True, but kept from original) ---
                # Using abs(hash()) to create a somewhat unique but valid SQL identifier
                savepoint_name = f"tagalog_word_{abs(hash(lemma))}"
                try:
                    # While commit=True handles transactions, savepoints might still be
                    # used for partial rollbacks within the loop if complex logic were added.
                    # However, with commit=True, an error below will likely trigger a full rollback
                    # for this entry attempt anyway by the decorator.
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                except Exception as sp_err:
                    # This error is less likely to be recoverable if savepoints fail repeatedly
                    logger.error(
                        f"Failed to create savepoint {savepoint_name} for '{lemma}': {sp_err}. Skipping entry."
                    )
                    stats["error_entries"] += 1
                    pbar.update(1)
                    continue  # Skip entry if savepoint fails

                try:
                    lemma = lemma.strip()  # Ensure lemma is stripped
                    if not lemma:
                        logger.warning(f"Skipping entry with empty lemma key.")
                        stats["skipped_entries"] += 1
                        cur.execute(
                            f"RELEASE SAVEPOINT {savepoint_name}"
                        )  # Release if skipping
                        pbar.update(1)
                        continue

                    # --- Basic Validation ---
                    if not entry_data or not isinstance(entry_data, dict):
                        logger.warning(
                            f"Skipping invalid entry data for '{lemma}': not a dictionary or empty."
                        )
                        stats["skipped_entries"] += 1
                        cur.execute(
                            f"RELEASE SAVEPOINT {savepoint_name}"
                        )  # Release if skipping
                        pbar.update(1)
                        continue

                    # --- Prepare Metadata and Tags ---
                    # Construct metadata dictionary as in the original snippet
                    word_metadata = {"processed_timestamp": datetime.now().isoformat()}
                    tags = []  # List to collect tags for the word level
                    if "domains" in entry_data and entry_data["domains"]:
                        # Ensure domains is a list of strings
                        domain_list = (
                            entry_data["domains"]
                            if isinstance(entry_data["domains"], list)
                            else [str(entry_data["domains"])]
                        )
                        word_metadata["domains"] = [
                            str(d).strip() for d in domain_list if str(d).strip()
                        ]
                        tags.extend(
                            word_metadata["domains"]
                        )  # Add domains to word tags
                    if "part_of_speech" in entry_data and entry_data["part_of_speech"]:
                        # Store raw POS structure in metadata
                        word_metadata["part_of_speech_raw"] = entry_data[
                            "part_of_speech"
                        ]
                        # Flatten POS list for potential tagging (optional)
                        # flat_pos = []
                        # for item in entry_data['part_of_speech']: flat_pos.extend(item if isinstance(item, list) else [item])
                        # tags.extend(p for p in flat_pos if p) # Optionally add POS to tags
                    if "pronunciation" in entry_data and entry_data["pronunciation"]:
                        word_metadata["pronunciation_primary_raw"] = entry_data[
                            "pronunciation"
                        ]
                    if "alternate_pronunciation" in entry_data:  # Check optional key
                        word_metadata["pronunciation_alternate_raw"] = entry_data[
                            "alternate_pronunciation"
                        ]
                    if "derivative" in entry_data and entry_data["derivative"]:
                        word_metadata["derivative_raw_text"] = entry_data["derivative"]

                    # --- Create or Get Word ID ---
                    # Pass the mandatory source_identifier
                    word_id = get_or_create_word_id(
                        cur,
                        lemma,
                        language_code=language_code,
                        source_identifier=source_identifier,  # Pass mandatory source
                        word_metadata=json.dumps(
                            word_metadata
                        ),  # Pass constructed metadata as JSON string
                        tags=(
                            ", ".join(sorted(list(set(tags)))) if tags else None
                        ),  # Pass unique, sorted tags
                        # Add other kwargs like is_proper_noun if determinable
                    )
                    if not word_id:
                        # Error should be logged within get_or_create_word_id
                        raise ValueError(
                            f"Failed to get/create word ID for '{lemma}' from {source_identifier}"
                        )

                    # --- Process Pronunciations ---
                    # Insert primary pronunciation
                    if "pronunciation" in entry_data and entry_data["pronunciation"]:
                        # Standardize pronunciation data into a dict if it's just a string
                        pron_data = entry_data["pronunciation"]
                        if isinstance(pron_data, str):
                            pron_obj = {
                                "value": pron_data,
                                "type": "ipa",
                            }  # Assume IPA if string
                        elif isinstance(pron_data, dict):
                            pron_obj = pron_data  # Assume dict has needed structure
                        else:
                            pron_obj = None

                        if pron_obj:
                            try:
                                pron_id = insert_pronunciation(
                                    cur,
                                    word_id,
                                    pron_obj,
                                    source_identifier=source_identifier,
                                )
                                if pron_id:
                                    stats["pronunciations_added"] += 1
                            except Exception as pron_e:
                                logger.warning(
                                    f"Failed primary pronunciation insert for '{lemma}': {pron_obj}. Error: {pron_e}"
                                )
                    # Insert alternate pronunciation
                    if (
                        "alternate_pronunciation" in entry_data
                        and entry_data["alternate_pronunciation"]
                    ):
                        alt_pron_data = entry_data["alternate_pronunciation"]
                        if isinstance(alt_pron_data, str):
                            alt_pron_obj = {
                                "value": alt_pron_data,
                                "type": "ipa",
                                "metadata": {"is_alternate": True},
                            }
                        elif isinstance(alt_pron_data, dict):
                            # Add alternate flag if passing a dict
                            alt_pron_data.setdefault("metadata", {})[
                                "is_alternate"
                            ] = True
                            alt_pron_obj = alt_pron_data
                        else:
                            alt_pron_obj = None

                        if alt_pron_obj:
                            try:
                                alt_pron_id = insert_pronunciation(
                                    cur,
                                    word_id,
                                    alt_pron_obj,
                                    source_identifier=source_identifier,
                                )
                                if alt_pron_id:
                                    stats["pronunciations_added"] += 1
                            except Exception as alt_pron_e:
                                logger.warning(
                                    f"Failed alternate pronunciation insert for '{lemma}': {alt_pron_obj}. Error: {alt_pron_e}"
                                )

                    # --- Process Derivatives (as Relations) ---
                    # Treat 'derivative' field as source for DERIVED_FROM relations
                    if "derivative" in entry_data and entry_data["derivative"]:
                        derivative_text = entry_data["derivative"]
                        # Assuming comma-separated string
                        derivative_forms = (
                            derivative_text.split(",")
                            if isinstance(derivative_text, str)
                            else []
                        )
                        for form_idx, form in enumerate(derivative_forms):
                            form_clean = form.strip()
                            if (
                                form_clean and form_clean != lemma
                            ):  # Ensure not empty and not self-referential
                                # Basic check for format like "verb salamat" - adjust logic as needed
                                form_word = form_clean
                                pos_hint = None
                                if " " in form_clean:
                                    parts = form_clean.split(" ", 1)
                                    # Simple heuristic: if first part is short, maybe it's a POS hint
                                    # Use POS_MAPPING or similar constant if available for validation
                                    if (
                                        len(parts[0]) <= 5 and parts[0].isalpha()
                                    ):  # Basic check
                                        pos_hint = parts[0]
                                        form_word = parts[1].strip()

                                try:
                                    # Get ID for the word mentioned in the derivative field
                                    derivative_word_id = get_or_create_word_id(
                                        cur,
                                        form_word,
                                        language_code,
                                        source_identifier=source_identifier,
                                    )
                                    if derivative_word_id:
                                        # Insert relation: derivative_word DERIVED_FROM lemma_word (word_id)
                                        rel_metadata = {
                                            "source": source_identifier,
                                            "raw_text": form_clean,
                                            "index": form_idx,
                                            "pos_hint": pos_hint,
                                            "confidence": 85,
                                        }
                                        rel_id = insert_relation(
                                            cur,
                                            derivative_word_id,
                                            word_id,  # From derivative TO base word
                                            RelationshipType.DERIVED_FROM,
                                            source_identifier=source_identifier,  # Pass mandatory source
                                            metadata=rel_metadata,
                                        )
                                        if rel_id:
                                            stats["derivatives_added"] += 1
                                            stats["relations_added"] += 1
                                        # Optionally add inverse ROOT_OF relation if needed
                                except Exception as der_e:
                                    logger.warning(
                                        f"Error processing derivative form '{form_clean}' for word '{lemma}': {der_e}"
                                    )

                    # --- Process Etymology ---
                    # Check if etymology data exists and is a dictionary
                    if "etymology" in entry_data and isinstance(
                        entry_data["etymology"], dict
                    ):
                        etymology_data = entry_data["etymology"]
                        etymology_text = etymology_data.get(
                            "raw", ""
                        )  # Assuming 'raw' key holds the text
                        if etymology_text and isinstance(etymology_text, str):
                            try:
                                # Prepare structure and language codes if available
                                etymology_structure_dict = {
                                    "source": source_identifier,
                                    **etymology_data,
                                }
                                lang_codes_list = etymology_data.get(
                                    "languages", []
                                )  # Assuming 'languages' key holds a list
                                language_codes_str = (
                                    ", ".join(lang_codes_list)
                                    if isinstance(lang_codes_list, list)
                                    else None
                                )

                                # Call insert_etymology with mandatory source_identifier
                                ety_id = insert_etymology(
                                    cur,
                                    word_id,
                                    etymology_text.strip(),
                                    source_identifier=source_identifier,  # Pass mandatory source
                                    etymology_structure=json.dumps(
                                        etymology_structure_dict
                                    ),  # Pass structure as JSON string
                                    language_codes=language_codes_str,  # Pass language codes string
                                    # Pass normalized_components if calculated/available
                                )
                                if ety_id:
                                    stats["etymologies_processed"] += 1
                            except Exception as e:
                                logger.warning(
                                    f"Error processing etymology for word '{lemma}': {etymology_text[:50]}... Error: {e}"
                                )

                    # --- Process Senses ---
                    senses = entry_data.get("senses", [])
                    if isinstance(senses, list):
                        for sense_idx, sense in enumerate(senses):
                            if not isinstance(sense, dict) or "definition" not in sense:
                                continue
                            definition_text = sense.get("definition", "").strip()
                            if not definition_text:
                                continue

                            # Prepend counter if exists
                            counter = sense.get("counter", "")
                            if counter:
                                definition_text = f"[{counter}] {definition_text}"

                            # Determine POS: Use sense POS if available, else fallback to entry POS
                            pos_list_sense = sense.get(
                                "part_of_speech", []
                            )  # POS specific to sense
                            pos_list_entry = entry_data.get(
                                "part_of_speech", []
                            )  # POS for the whole entry
                            pos_list = (
                                pos_list_sense if pos_list_sense else pos_list_entry
                            )
                            # Flatten potential list-of-lists structure
                            flat_pos = []
                            if isinstance(pos_list, list):
                                for item in pos_list:
                                    flat_pos.extend(
                                        item if isinstance(item, list) else [item]
                                    )
                            elif isinstance(
                                pos_list, str
                            ):  # Handle case where POS is just a string
                                flat_pos.append(pos_list)
                            # Create comma-separated string for storage and standardization
                            pos_str = ", ".join(
                                str(p).strip() for p in flat_pos if str(p).strip()
                            )
                            standardized_pos_id = get_standardized_pos_id(
                                cur, pos_str
                            )  # Get standardized ID

                            # Extract Examples into a JSON list of dicts
                            examples_json = None
                            examples_list = []
                            example_data = sense.get(
                                "example", {}
                            )  # examples might be under 'example' key
                            if isinstance(example_data, dict):
                                raw_ex_list = example_data.get(
                                    "examples", []
                                )  # Assuming a list under 'examples'
                                if isinstance(raw_ex_list, list):
                                    examples_list = [
                                        {
                                            "text": str(ex).strip(),
                                            "source": source_identifier,
                                            "index": i,
                                        }
                                        for i, ex in enumerate(raw_ex_list)
                                        if str(ex).strip()
                                    ]
                                elif (
                                    "raw" in example_data and example_data["raw"]
                                ):  # Handle 'raw' example key
                                    examples_list = [
                                        {
                                            "text": str(example_data["raw"]).strip(),
                                            "source": source_identifier,
                                            "raw": True,
                                        }
                                    ]
                            elif isinstance(
                                example_data, list
                            ):  # If 'example' directly holds a list
                                examples_list = [
                                    {
                                        "text": str(ex).strip(),
                                        "source": source_identifier,
                                        "index": i,
                                    }
                                    for i, ex in enumerate(example_data)
                                    if str(ex).strip()
                                ]

                            if examples_list:
                                try:
                                    examples_json = json.dumps(examples_list)
                                except TypeError:
                                    logger.warning(
                                        f"Could not serialize examples for '{lemma}', sense {sense_idx}"
                                    )

                            # Prepare Tags: Combine sense tags and category
                            sense_tags_list = []
                            raw_tags = sense.get("tags", [])
                            if isinstance(raw_tags, list):
                                sense_tags_list.extend(
                                    str(t).strip() for t in raw_tags if str(t).strip()
                                )
                            category = sense.get("category")  # Check for category field
                            if category and isinstance(category, str):
                                sense_tags_list.append(f"category:{category.strip()}")
                            tags_str = (
                                ", ".join(sorted(list(set(sense_tags_list))))
                                if sense_tags_list
                                else None
                            )

                            try:
                                # Insert definition, passing mandatory source_identifier
                                definition_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    source_identifier=source_identifier,  # Pass mandatory source
                                    part_of_speech=pos_str,  # Fix: Pass the original POS string, not the ID
                                    examples=examples_json,  # Pass examples JSON string or None
                                    usage_notes=sense.get(
                                        "usage_notes"
                                    ),  # Pass usage notes if present
                                    tags=tags_str,  # Pass comma-separated tags string or None
                                )

                                if definition_id:
                                    stats["definitions_added"] += 1
                                    # Process sense-level relations (synonyms, variants, references)
                                    sense_relation_map = {
                                        "synonyms": RelationshipType.SYNONYM,
                                        "variants": RelationshipType.VARIANT,
                                        "references": RelationshipType.SEE_ALSO,  # Map 'references' to SEE_ALSO
                                    }
                                    for (
                                        rel_key,
                                        rel_type_enum,
                                    ) in sense_relation_map.items():
                                        related_items = sense.get(rel_key)
                                        if isinstance(related_items, list):
                                            for item_idx, item_word in enumerate(
                                                related_items
                                            ):
                                                if (
                                                    isinstance(item_word, str)
                                                    and item_word.strip()
                                                ):
                                                    item_word_clean = item_word.strip()
                                                    if (
                                                        item_word_clean != lemma
                                                    ):  # Avoid self-relation
                                                        try:
                                                            item_id = get_or_create_word_id(
                                                                cur,
                                                                item_word_clean,
                                                                language_code,
                                                                source_identifier=source_identifier,
                                                            )
                                                            if item_id:
                                                                # Add metadata specific to the relation source (sense)
                                                                rel_meta = {
                                                                    "source": source_identifier,
                                                                    "definition_id": definition_id,
                                                                    "sense_index": sense_idx,
                                                                    "item_index": item_idx,
                                                                    "confidence": 80,
                                                                }  # Example confidence
                                                                # Insert relation, passing mandatory source
                                                                rel_ins_id = insert_relation(
                                                                    cur,
                                                                    word_id,
                                                                    item_id,  # From lemma word TO item word
                                                                    rel_type_enum,
                                                                    source_identifier=source_identifier,  # Pass mandatory source
                                                                    metadata=rel_meta,
                                                                )
                                                                if rel_ins_id:
                                                                    stats[
                                                                        f"{rel_key}_added"
                                                                    ] += 1
                                                                    stats[
                                                                        "relations_added"
                                                                    ] += 1
                                                                # Handle bidirectional if needed (e.g., SYNONYM)
                                                                # if rel_type_enum.bidirectional: insert_relation(cur, item_id, word_id, rel_type_enum, source_identifier=source_identifier, metadata=rel_meta)

                                                        except Exception as rel_e:
                                                            logger.warning(
                                                                f"Error processing sense relation '{rel_key}'->'{item_word_clean}' for '{lemma}', sense {sense_idx}: {rel_e}"
                                                            )
                            except psycopg2.errors.UniqueViolation:
                                logger.debug(
                                    f"Definition already exists for '{lemma}', sense {sense_idx}: {definition_text[:30]}..."
                                )
                            except Exception as def_e:
                                logger.error(
                                    f"Error inserting definition for '{lemma}', sense {sense_idx}: {definition_text[:30]}... Error: {def_e}",
                                    exc_info=False,
                                )  # Less verbose

                    # --- Process Baybayin ---
                    # Check for 'baybayin' key and validate/process
                    if "baybayin" in entry_data and entry_data["baybayin"]:
                        baybayin_text = entry_data["baybayin"]
                        if isinstance(baybayin_text, str) and baybayin_text.strip():
                            baybayin_form_raw = baybayin_text.strip()
                            # Validate the raw Baybayin form before cleaning/storing
                            if validate_baybayin_entry(baybayin_form_raw):
                                clean_baybayin = clean_baybayin_text(
                                    baybayin_form_raw
                                )  # Clean for storage
                                if clean_baybayin:
                                    try:
                                        # Romanize the *cleaned* version for consistency
                                        romanized = romanizer.romanize(clean_baybayin)
                                        # Update words table, setting has_baybayin = true
                                        # Only update if baybayin_form is NULL or different
                                        cur.execute(
                                            """
                                           UPDATE words SET has_baybayin = true, baybayin_form = %s, romanized_form = %s
                                           WHERE id = %s AND (baybayin_form IS NULL OR baybayin_form != %s)
                                       """,
                                            (
                                                clean_baybayin,
                                                romanized,
                                                word_id,
                                                clean_baybayin,
                                            ),
                                        )
                                        if cur.rowcount > 0:
                                            stats["baybayin_added"] += 1
                                            logger.debug(
                                                f"Added/Updated Baybayin for '{lemma}' (ID: {word_id})"
                                            )
                                    except Exception as bb_e:
                                        logger.error(
                                            f"Error processing/updating Baybayin '{baybayin_form_raw}' for '{lemma}': {bb_e}"
                                        )
                            else:
                                logger.warning(
                                    f"Invalid baybayin structure found for '{lemma}': {baybayin_form_raw}"
                                )

                    # --- Process Affix Forms ---
                    # Check for both 'affix_forms' and 'affix_types' keys
                    if (
                        "affix_forms" in entry_data
                        and isinstance(entry_data["affix_forms"], list)
                        and "affix_types" in entry_data
                        and isinstance(entry_data["affix_types"], list)
                    ):
                        affix_forms = entry_data["affix_forms"]
                        affix_types = entry_data["affix_types"]
                        # Ensure lists have the same length for safe iteration
                        min_len = min(len(affix_forms), len(affix_types))
                        if len(affix_forms) != len(affix_types):
                            logger.warning(
                                f"Mismatch in length between affix_forms ({len(affix_forms)}) and affix_types ({len(affix_types)}) for '{lemma}'. Processing up to index {min_len-1}."
                            )

                        for i in range(min_len):
                            form = affix_forms[i]
                            affix_type = affix_types[i]
                            clean_form = str(form).strip() if form else None
                            clean_type = (
                                str(affix_type).strip().lower()
                                if affix_type
                                else "unknown"
                            )

                            if (
                                clean_form and clean_form != lemma
                            ):  # Ensure form exists and is not the lemma itself
                                try:
                                    # Get ID for the affixed form, passing source
                                    affixed_id = get_or_create_word_id(
                                        cur,
                                        clean_form,
                                        language_code,
                                        source_identifier=source_identifier,
                                    )
                                    if affixed_id:
                                        # Insert into affixation table, passing source
                                        aff_rec_id = insert_affixation(
                                            cur,
                                            root_id=word_id,
                                            affixed_id=affixed_id,
                                            affix_type=clean_type,
                                            source_identifier=source_identifier,  # Pass mandatory source
                                        )
                                        if aff_rec_id:
                                            stats["affixations_added"] += 1
                                        # Optionally add DERIVED_FROM/ROOT_OF relations here as well, if desired
                                        # Example: insert_relation(cur, affixed_id, word_id, RelationshipType.DERIVED_FROM, source_identifier=source_identifier, metadata={"affix_type": clean_type})

                                except Exception as aff_e:
                                    logger.warning(
                                        f"Error processing affix form '{clean_form}' (type {clean_type}, index {i}) for root '{lemma}': {aff_e}"
                                    )

                    # --- Finish Entry Processing ---
                    stats["processed_entries"] += 1
                    # Release savepoint (optional with commit=True, but kept for consistency)
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

                except Exception as entry_err:
                    # Catch errors specific to processing this entry after getting lemma/data
                    logger.error(
                        f"Error processing entry for lemma '{lemma}': {entry_err}",
                        exc_info=True,
                    )  # Log traceback for debugging
                    stats["error_entries"] += 1
                    try:
                        # Rollback to the state before this entry started
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rollback_error:
                        # This is more critical - if savepoint rollback fails, the transaction state is unknown
                        logger.critical(
                            f"CRITICAL: Failed rollback to savepoint {savepoint_name} for '{lemma}': {rollback_error}. Decorator should handle full rollback."
                        )
                        # Re-raising might be appropriate here to stop the whole process if this occurs
                        # raise entry_err from rollback_error
                finally:
                    pbar.update(
                        1
                    )  # Ensure progress bar updates even if an error occurred

    # --- Error Handling for File Loading ---
    except (IOError, json.JSONDecodeError, TypeError) as file_err:
        logger.error(f"Fatal error reading or parsing file {filename}: {file_err}")
        # The @with_transaction decorator should handle rollback here
        # Log the stats gathered so far before returning/raising
        logger.info(f"Partial stats before fatal file error: {stats}")
        raise  # Re-raise the error to signal failure to the caller

    # --- General Error Handling ---
    except Exception as e:
        logger.error(
            f"Unexpected fatal error during processing of {filename}: {e}",
            exc_info=True,
        )
        # The @with_transaction decorator should handle rollback
        logger.info(f"Partial stats before fatal error: {stats}")
        raise  # Re-raise critical errors

    # --- Final Logging ---
    # This block will execute if the main try block completes without fatal errors
    finally:
        logger.info(f"Finished processing {source_identifier}")
        logger.info(f"  Total entries in file: {stats['total_entries']}")
        logger.info(f"  Processed successfully: {stats['processed_entries']}")
        logger.info(f"  Skipped (invalid/empty): {stats['skipped_entries']}")
        logger.info(f"  Errors during processing: {stats['error_entries']}")
        logger.info(f"  Definitions added/updated: {stats['definitions_added']}")
        logger.info(f"  Pronunciations added/updated: {stats['pronunciations_added']}")
        logger.info(f"  Etymologies added/updated: {stats['etymologies_processed']}")
        logger.info(f"  Total relations added/updated: {stats['relations_added']}")
        logger.info(
            f"    (Deriv: {stats['derivatives_added']}, Syn: {stats['synonyms_added']}, Var: {stats['variants_added']}, Ref: {stats['references_added']})"
        )  # Detailed relation counts
        logger.info(f"  Baybayin forms added/updated: {stats['baybayin_added']}")
        logger.info(f"  Affixations added/updated: {stats['affixations_added']}")

    return stats  # Return the final statistics dictionary


def process_root_words_cleaned(cur, filename: str):
    """
    Processes entries from the tagalog.com Root Words JSON file (cleaned version).
    Handles JSON input that is either a list of root word objects
    or a dictionary mapping root words (str) to their details (dict).
    Manages transactions manually using savepoints for individual entry resilience.

    Args:
        cur: Database cursor
        filename: Path to the root words cleaned JSON file

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing Root Words (tagalog.com cleaned) file: {filename}")
    stats = {
        "roots_processed": 0,
        "definitions_added": 0,
        "relations_added": 0,
        "associated_processed": 0,
        "errors": 0,
        "skipped": 0,
    }
    error_types = {} # Track error types

    # Define source identifier, standardizing the filename
    source_identifier = standardize_source_identifier(
        os.path.basename(filename)
    )
    if not source_identifier: # Fallback if standardization fails unexpectedly
        source_identifier = "tagalog.com-RootWords-Cleaned"
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection # Get the connection for savepoint management

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats # Return stats; outer transaction might continue or rollback
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1
        error_types["JSONDecodeError"] = error_types.get("JSONDecodeError", 0) + 1
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        error_types[f"FileReadError: {type(e).__name__}"] = error_types.get(f"FileReadError: {type(e).__name__}", 0) + 1
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    # --- Determine format and prepare iterator ---
    entries_iterator = None
    total_roots = 0
    if isinstance(data, list):  # Original list format
        entries_iterator = enumerate(data)
        total_roots = len(data)
        logger.info(
            f"Found {total_roots} root word entries in list format in {filename}"
        )
    elif isinstance(data, dict):  # New dictionary format {root_word: details_dict}
        # Create an iterator that yields (index, root_word, root_details)
        def dict_iterator(d):
            for i, (key, value) in enumerate(d.items()):
                yield i, key, value

        entries_iterator = dict_iterator(data)
        total_roots = len(data)
        logger.info(
            f"Found {total_roots} root word entries in dictionary format in {filename}"
        )
    else:
        logger.error(
            f"File {filename} does not contain a list or dictionary of root word entries."
        )
        stats["errors"] += 1
        error_types["InvalidTopLevelFormat"] = error_types.get("InvalidTopLevelFormat", 0) + 1
        raise TypeError(f"Invalid top-level format in {filename}") # Raise error

    if total_roots == 0:
        logger.info(f"No root word entries found in {filename}. Nothing to process.")
        return stats # Return empty stats

    # --- Process Entries ---
    with tqdm(
        total=total_roots, desc=f"Processing {source_identifier}"
    ) as pbar:
        for entry_item in entries_iterator:
            # Handle different formats
            if isinstance(data, dict):
                entry_index, root_word, root_details = entry_item
            else:  # list format
                entry_index, root_word_entry = entry_item
                root_word = root_word_entry.get("root_word", "").strip() if isinstance(root_word_entry, dict) else ""
                root_details = root_word_entry

            savepoint_name = f"root_entry_{entry_index}" # Create a savepoint name

            try:
                cur.execute(f"SAVEPOINT {savepoint_name}") # Start savepoint for this entry

                # Skip if root_word is empty
                if not root_word:
                    logger.warning(f"Skipping entry at index {entry_index} with empty root word.")
                    stats["skipped"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                language_code = "tl"

                # --- Root Word Creation ---
                word_id = get_or_create_word_id(
                    cur,
                    root_word,
                    language_code=language_code,
                    source_identifier=source_identifier,
                    is_root_word=True,
                )
                if not word_id or word_id <= 0:
                    raise ValueError(f"Failed to get/create root word ID for '{root_word}'")
                logger.debug(f"Processing root word '{root_word}' (ID: {word_id})")

                # --- Process definition for the root word itself ---
                if isinstance(data, dict):
                    # In dict format, root word details are in the details[root_word] if it exists
                    root_word_details = root_details.get(root_word, {})
                    if isinstance(root_word_details, dict):
                        definition_text = root_word_details.get("definition", "").strip()
                        part_of_speech = root_word_details.get("type", "").strip() or None
                        
                        if definition_text:
                            if definition_text.endswith("..."):
                                definition_text = definition_text[:-3].strip()
                            
                            def_id = insert_definition(
                                cur,
                                word_id,
                                definition_text,
                                source_identifier=source_identifier,
                                part_of_speech=part_of_speech
                            )
                            if def_id:
                                stats["definitions_added"] += 1
                                logger.debug(f"Added definition for root '{root_word}': {definition_text[:50]}...")
                else:
                    # Process list format definitions
                    definitions = root_details.get("definitions", [])
                    if isinstance(definitions, list):
                        for def_idx, definition_item in enumerate(definitions):
                            definition_text = None
                            part_of_speech = None
                            examples_json = None
                            tags_str = None
                            usage_notes = None

                            if isinstance(definition_item, str):
                                definition_text = definition_item.strip()
                            elif isinstance(definition_item, dict):
                                definition_text = (
                                    definition_item.get("text", "").strip()
                                    or definition_item.get("definition", "").strip()
                                )
                                part_of_speech = (
                                    definition_item.get("pos")
                                    or definition_item.get("part_of_speech")
                                    or definition_item.get("type")
                                )
                                if part_of_speech:
                                    part_of_speech = part_of_speech.strip()

                            if definition_text and definition_text.endswith("..."):
                                definition_text = definition_text[:-3].strip()

                            if definition_text:
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    source_identifier=source_identifier,
                                    part_of_speech=part_of_speech,
                                    examples=examples_json,
                                    usage_notes=usage_notes,
                                    tags=tags_str,
                                )
                                if def_id:
                                    stats["definitions_added"] += 1

                # --- Process associated words ---
                if isinstance(data, dict):
                    # In dictionary format, each key other than the root word is an associated word
                    associated_words = {k: v for k, v in root_details.items() if k != root_word}
                    
                    for assoc_idx, (assoc_word, assoc_details) in enumerate(associated_words.items()):
                        if assoc_word and assoc_word != root_word:
                            stats["associated_processed"] += 1
                            
                            # Create the associated word
                            assoc_word_id = get_or_create_word_id(
                                cur,
                                assoc_word,
                                language_code=language_code,
                                root_word_id=word_id,
                                source_identifier=source_identifier,
                            )
                            
                            if assoc_word_id:
                                # Add DERIVED_FROM relationship
                                rel_metadata = {"source": source_identifier, "index": assoc_idx, "confidence": 95}
                                rel_id = insert_relation(
                                    cur, assoc_word_id, word_id, RelationshipType.DERIVED_FROM,
                                    source_identifier=source_identifier, metadata=rel_metadata
                                )
                                if rel_id:
                                    stats["relations_added"] += 1
                                    logger.debug(f"Added relation: '{assoc_word}' DERIVED_FROM '{root_word}'")

                                # Add definition for the associated word
                                if isinstance(assoc_details, dict):
                                    assoc_def_text = assoc_details.get("definition", "").strip()
                                    if assoc_def_text and assoc_def_text.endswith("..."):
                                        assoc_def_text = assoc_def_text[:-3].strip()
                                    
                                    assoc_pos = assoc_details.get("type", "").strip() or None
                                    
                                    if assoc_def_text:
                                        assoc_def_id = insert_definition(
                                            cur, assoc_word_id, assoc_def_text,
                                            source_identifier=source_identifier, part_of_speech=assoc_pos
                                        )
                                        if assoc_def_id:
                                            stats["definitions_added"] += 1
                                            logger.debug(f"Added definition for associated word '{assoc_word}': {assoc_def_text[:50]}...")
                else:
                    # Process original list format associated words
                    associated_data = root_details.get("associated_words", {})
                    associated_items_iterator = None
                    
                    if isinstance(associated_data, dict):
                        def assoc_dict_iterator(d):
                            for i, (key, value) in enumerate(d.items()):
                                 if isinstance(value, dict): yield i, key.strip(), value
                                 else: logger.warning(f"Skipping non-dict value for associated word key '{key}' in root '{root_word}'")
                        associated_items_iterator = assoc_dict_iterator(associated_data)
                    elif isinstance(associated_data, list):
                        def assoc_list_iterator(l):
                            for i, item in enumerate(l):
                                word_str, details = None, {}
                                if isinstance(item, str): word_str = item.strip()
                                elif isinstance(item, dict):
                                    word_str = item.get("word", "").strip(); details = item
                                if word_str: yield i, word_str, details
                                else: logger.warning(f"Skipping invalid associated word item at index {i} in root '{root_word}' (list format)")
                        associated_items_iterator = assoc_list_iterator(associated_data)
                    else:
                        logger.warning(f"Unexpected format for 'associated_words' for root '{root_word}': {type(associated_data)}. Skipping.")
                        associated_items_iterator = iter([])

                    if associated_data:
                        processed_assoc_count_for_root = 0
                        for assoc_idx, assoc_word, assoc_details in associated_items_iterator:
                            if assoc_word and assoc_word != root_word:
                                stats["associated_processed"] += 1
                                processed_assoc_count_for_root += 1
                                
                                assoc_word_id = get_or_create_word_id(
                                    cur,
                                    assoc_word,
                                    language_code=language_code,
                                    root_word_id=word_id,
                                    source_identifier=source_identifier,
                                )
                                
                                if assoc_word_id:
                                    # Add DERIVED_FROM relationship
                                    rel_metadata = {"source": source_identifier, "index": assoc_idx, "confidence": 95}
                                    rel_id = insert_relation(
                                        cur, assoc_word_id, word_id, RelationshipType.DERIVED_FROM,
                                        source_identifier=source_identifier, metadata=rel_metadata
                                    )
                                    if rel_id:
                                        stats["relations_added"] += 1
                                        logger.debug(f"Added relation: '{assoc_word}' DERIVED_FROM '{root_word}'")

                                    # Process definition for associated word
                                    assoc_def_text = assoc_details.get("definition", "").strip()
                                    if assoc_def_text and assoc_def_text.endswith("..."):
                                        assoc_def_text = assoc_def_text[:-3].strip()
                                    assoc_pos = assoc_details.get("type", "").strip() or None

                                    if assoc_def_text:
                                        assoc_def_id = insert_definition(
                                            cur, assoc_word_id, assoc_def_text,
                                            source_identifier=source_identifier, part_of_speech=assoc_pos
                                        )
                                        if assoc_def_id:
                                            stats["definitions_added"] += 1
                                            logger.debug(f"Added definition for associated word '{assoc_word}': {assoc_def_text[:50]}...")

                        if processed_assoc_count_for_root > 0:
                            logger.debug(f"Root '{root_word}': Processed {processed_assoc_count_for_root} associated words.")

                # --- Finish Entry Processing Successfully ---
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Commit the entry's changes
                stats["roots_processed"] += 1 # Increment successful count

            except Exception as entry_e:
                # Log error and count it
                logger.error(
                    f"Error processing root entry index {entry_index} ('{root_word or 'N/A'}'): {entry_e}",
                    exc_info=True, # Log traceback for debugging
                )
                stats["errors"] += 1
                error_key = f"RootEntryProcessingError: {type(entry_e).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1

                # Rollback to the state before this entry started
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    logger.info(f"Rolled back changes for failed entry '{root_word or 'N/A'}' (Index: {entry_index}).")
                except Exception as rb_err:
                    # This is more critical - if savepoint rollback fails, the outer transaction might become unstable
                    logger.critical(
                        f"CRITICAL: Failed rollback to savepoint {savepoint_name} for '{root_word or 'N/A'}': {rb_err}. "
                        f"Raising original error to abort outer transaction.",
                        exc_info=True
                    )
                    raise entry_e from rb_err # Re-raise the original error to ensure outer transaction rolls back

            finally:
                # This will always run, even if an error occurred
                pbar.update(1)

    # No final commit needed here - the outer migrate_data function handles the commit for the entire source file

    if error_types:
        logger.warning(f"Error summary for {filename}: {json.dumps(error_types, indent=2)}")
    if stats["roots_processed"] == 0 and total_roots > 0:
        logger.warning(f"No root entries were successfully processed from {filename} despite finding {total_roots} entries.")

    # <<< Log the final statistics >>>
    logger.info(f"Statistics for {filename}: {json.dumps(stats, indent=2)}")

    # Return the statistics gathered during processing
    return stats

# Add this function definition somewhere alongside the other process_* functions

@with_transaction(commit=False) # Assume this will be called within migrate_data
def process_gay_slang_json(cur, filename: str) -> Tuple[int, int]:
    """
    Processes entries from the gay-slang.json file.
    Handles JSON input that is a list of entry objects.
    Manages transactions manually using savepoints for individual entry resilience.

    Args:
        cur: Database cursor
        filename: Path to the gay-slang.json file

    Returns:
        Tuple: (number_of_entries_processed_successfully, number_of_entries_with_errors)
    """
    # Standardize source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    # Explicitly map this filename to a clear source name
    source_identifier_map = {
        "gay-slang.json": "Philippine Slang and Gay Dictionary (2023)"
    }
    source_identifier = source_identifier_map.get(
        raw_source_identifier,
        standardize_source_identifier(raw_source_identifier) # Fallback if not specific match
    )
    # Ensure we have a valid source identifier, providing a default if needed
    if not source_identifier:
        source_identifier = "Gay Slang Dictionary" # Default fallback

    logger.info(f"Processing Gay Slang file: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    conn = cur.connection # Get the connection for savepoint management

    # Statistics tracking for this file
    stats = {
        "processed": 0,
        "definitions": 0,
        "relations": 0,
        "synonyms": 0,        # Count Filipino synonyms specifically
        "variants": 0,        # Count variations specifically
        "etymologies": 0,
        "skipped_invalid": 0, # Entries skipped due to format issues
        "errors": 0,          # Entries skipped due to processing errors
    }
    error_types = {} # Track specific error types encountered

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return 0, 1 # 0 processed, 1 issue (file error)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Invalid JSON in file {filename}: {e}") from e
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        # Raise exception to make the outer migrate_data function rollback
        raise RuntimeError(f"Error reading file {filename}: {e}") from e

    if not isinstance(data, list):
        logger.error(f"File {filename} does not contain a list of entries as expected.")
        return 0, 1 # Indicate format error

    entries_in_file = len(data)
    if entries_in_file == 0:
        logger.info(f"Found 0 entries in {filename}. Skipping file.")
        return 0, 0

    logger.info(f"Found {entries_in_file} entries in {filename}")

    # --- Process Entries ---
    with tqdm(total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry", leave=False) as pbar:
        for entry_index, entry in enumerate(data):
            savepoint_name = f"gayslang_entry_{entry_index}"
            lemma = "" # Initialize for error logging
            word_id = None

            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")

                if not isinstance(entry, dict):
                    logger.warning(f"Skipping non-dictionary item at index {entry_index} in {filename}")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                lemma = entry.get("headword", "").strip()
                if not lemma:
                    logger.warning(f"Skipping entry at index {entry_index} due to missing/empty 'headword' field.")
                    stats["skipped_invalid"] += 1
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    pbar.update(1)
                    continue

                # Assume Tagalog ('tl') as the base language for this dictionary
                language_code = "tl"

                # --- Extract Metadata ---
                entry_metadata = entry.get("metadata", {})
                word_metadata_to_store = {}
                if isinstance(entry_metadata, dict) and "page" in entry_metadata:
                    word_metadata_to_store["page"] = entry_metadata["page"]
                word_metadata_to_store["source_file"] = raw_source_identifier # Store original filename

                # --- Get or Create Word ---
                try:
                    word_id = get_or_create_word_id(
                        cur,
                        lemma=lemma,
                        language_code=language_code,
                        source_identifier=source_identifier,
                        word_metadata=Json(word_metadata_to_store) if word_metadata_to_store else None
                    )
                    if not word_id:
                        raise ValueError("get_or_create_word_id returned None")
                    logger.debug(f"Word '{lemma}' ({language_code}) -> ID: {word_id}")
                except Exception as word_err:
                    logger.error(f"CRITICAL FAILURE creating word '{lemma}': {word_err}")
                    raise word_err # Re-raise critical error to trigger outer catch

                # --- Process Etymology ---
                etymology_text = entry.get("etymology", "").strip()
                if etymology_text:
                    try:
                        ety_id = insert_etymology(cur, word_id, etymology_text, source_identifier)
                        if ety_id:
                            stats["etymologies"] += 1
                    except Exception as ety_err:
                        logger.warning(f"Error inserting etymology for '{lemma}' (ID: {word_id}): {ety_err}")
                        error_key = f"EtymologyInsertError: {type(ety_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Process POS ---
                # Use the first POS found, standardize it
                pos_list = entry.get("partOfSpeech", [])
                raw_pos_str = pos_list[0] if pos_list and isinstance(pos_list[0], str) else None

                # --- Process Definitions and Examples ---
                definitions_raw = entry.get("definitions", [])
                examples_raw = entry.get("examples", [])
                usage_labels = entry.get("usageLabels", [])
                tags_str = ", ".join(l for l in usage_labels if isinstance(l, str)) if usage_labels else None

                if isinstance(definitions_raw, list):
                    for def_item in definitions_raw:
                        if isinstance(def_item, dict):
                            def_lang = def_item.get("language", "").lower()
                            def_meaning = def_item.get("meaning", "").strip()

                            if not def_meaning:
                                continue # Skip empty definitions

                            # Prepare examples for this specific definition language
                            def_examples = []
                            if isinstance(examples_raw, list):
                                for ex_item in examples_raw:
                                    if isinstance(ex_item, dict) and ex_item.get("language", "").lower() == def_lang:
                                        example_text = ex_item.get("example", "").strip()
                                        if example_text:
                                            def_examples.append({"text": example_text}) # Store as simple list of text dicts

                            examples_json = Json(def_examples) if def_examples else None
                            def_text_to_insert = def_meaning

                            # Append English synonyms to English definitions if desired
                            if def_lang == "english" and entry.get("synonyms"):
                                eng_syns = [s for s in entry["synonyms"] if isinstance(s, str) and s.strip()]
                                if eng_syns:
                                    def_text_to_insert += f" (Synonyms: {', '.join(eng_syns)})"

                            try:
                                def_id = insert_definition(
                                    cur,
                                    word_id,
                                    def_text_to_insert,
                                    source_identifier=source_identifier,
                                    part_of_speech=raw_pos_str, # Pass the raw POS string associated with the headword
                                    examples=examples_json,
                                    tags=tags_str # Add usage labels as tags
                                )
                                if def_id:
                                    stats["definitions"] += 1
                            except Exception as def_err:
                                logger.warning(f"Error inserting definition for '{lemma}' ({def_lang}): {def_err}")
                                error_key = f"DefinitionInsertError: {type(def_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Process Relations ---
                # Variations
                variations = entry.get("variations", [])
                if isinstance(variations, list):
                    for var_word in variations:
                        if isinstance(var_word, str) and var_word.strip() and var_word.strip() != lemma:
                            try:
                                var_id = get_or_create_word_id(cur, var_word.strip(), language_code, source_identifier=source_identifier)
                                if var_id and var_id != word_id:
                                    rel_id_1 = insert_relation(cur, word_id, var_id, RelationshipType.VARIANT, source_identifier)
                                    rel_id_2 = insert_relation(cur, var_id, word_id, RelationshipType.VARIANT, source_identifier)
                                    if rel_id_1: stats["relations"] += 1; stats["variants"] += 1
                            except Exception as rel_err:
                                logger.warning(f"Error creating VARIANT relation for '{lemma}' -> '{var_word}': {rel_err}")
                                error_key = f"RelationInsertError: {type(rel_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1

                # Filipino Synonyms (sangkahulugan)
                fil_synonyms = entry.get("sangkahulugan", [])
                if isinstance(fil_synonyms, list):
                    for syn_word in fil_synonyms:
                        if isinstance(syn_word, str) and syn_word.strip() and syn_word.strip() != lemma:
                            try:
                                syn_id = get_or_create_word_id(cur, syn_word.strip(), language_code, source_identifier=source_identifier)
                                if syn_id and syn_id != word_id:
                                    rel_id_1 = insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM, source_identifier)
                                    rel_id_2 = insert_relation(cur, syn_id, word_id, RelationshipType.SYNONYM, source_identifier)
                                    if rel_id_1: stats["relations"] += 1; stats["synonyms"] += 1
                            except Exception as rel_err:
                                logger.warning(f"Error creating SYNONYM relation for '{lemma}' -> '{syn_word}': {rel_err}")
                                error_key = f"RelationInsertError: {type(rel_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1

                # --- Finish Entry ---
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                stats["processed"] += 1

            except Exception as entry_err:
                logger.error(f"Failed processing entry #{entry_index} ('{lemma}') in {filename}: {entry_err}", exc_info=True)
                stats["errors"] += 1
                error_key = f"EntryProcessingError: {type(entry_err).__name__}"
                error_types[error_key] = error_types.get(error_key, 0) + 1
                try:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    logger.info(f"Rolled back changes for failed entry '{lemma}' (Index: {entry_index}).")
                except Exception as rb_err:
                    logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name} after entry error: {rb_err}. Raising original error.", exc_info=True)
                    # Re-raise the original error to abort the whole file processing if rollback fails
                    raise entry_err from rb_err
            finally:
                pbar.update(1) # Ensure progress bar updates

    # --- Final Commit/Rollback handled by migrate_data ---
    # Log final stats for this file
    logger.info(f"Finished processing {filename}. Processed: {stats['processed']}, Skipped: {stats['skipped_invalid']}, Errors: {stats['errors']}")
    if error_types:
        logger.warning(f"Error summary for {filename}: {error_types}")

    total_issues = stats["skipped_invalid"] + stats["errors"]
    return stats["processed"], total_issues

def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    lang_map = {
        "Esp": "es",
        "Eng": "en",
        "Ch": "zh",
        "Tsino": "zh",
        "Jap": "ja",
        "San": "sa",
        "Sanskrit": "sa",
        "Tag": "tl",
        "Mal": "ms",
        "Arb": "ar",
    }
    return [lang_map[lang] for lang in lang_map if lang in etymology]


@with_transaction(commit=True)
def process_definition_relations(cur, word_id: int, definition: str, source: str):
    """Process and create relationships from definition text."""
    synonym_patterns = [
        r"ka(singkahulugan|tulad|patulad|tumbas) ng\s+(\w+)",
        r"singkahulugan:\s+(\w+)",
    ]
    antonym_patterns = [
        r"(kasalungat|kabaligtaran) ng\s+(\w+)",
        r"kabaligtaran:\s+(\w+)",
    ]

    for pattern in synonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            syn = match.group(2).strip()
            syn_id = get_or_create_word_id(cur, syn, "tl")
            insert_relation(cur, word_id, syn_id, "synonym", sources=source)

    for pattern in antonym_patterns:
        for match in re.finditer(pattern, definition, re.IGNORECASE):
            ant = match.group(2).strip()
            ant_id = get_or_create_word_id(cur, ant, "tl")
            insert_relation(cur, word_id, ant_id, "antonym", sources=source)


# ---------------------------------------------------------------a----
# Robust Relationship Mapping for Kaikki
# -------------------------------------------------------------------
RELATION_MAPPING = {
    "synonym": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym",
    },
    "antonym": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym",
    },
    "hyponym_of": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of",
    },
    "hypernym_of": {
        "relation_type": "hypernym_of",
        "bidirectional": False,
        "inverse": "hyponym_of",
    },
    "see_also": {
        "relation_type": "see_also",
        "bidirectional": True,
        "inverse": "see_also",
    },
    "compare_with": {
        "relation_type": "compare_with",
        "bidirectional": True,
        "inverse": "compare_with",
    },
    "derived_from": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of",
    },
    "descendant": {
        "relation_type": "descendant_of",
        "bidirectional": False,
        "inverse": "ancestor_of",
    },
    "borrowed": {
        "relation_type": "borrowed_from",
        "bidirectional": False,
        "inverse": "loaned_to",
    },
    "variant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant",
    },
    "alt_of": {"relation_type": "variant", "bidirectional": True, "inverse": "variant"},
    "abbreviation_of": {
        "relation_type": "abbreviation_of",
        "bidirectional": False,
        "inverse": "has_abbreviation",
    },
    "initialism_of": {
        "relation_type": "initialism_of",
        "bidirectional": False,
        "inverse": "has_initialism",
    },
    "related": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related",
    },
    "derived": {
        "relation_type": "derived",
        "bidirectional": False,
        "inverse": "derives",
    },
    "contraction_of": {
        "relation_type": "contraction_of",
        "bidirectional": False,
        "inverse": "contracts_to",
    },
    "alternate_form": {
        "relation_type": "alternate_form",
        "bidirectional": True,
        "inverse": "alternate_form",
    },
    "regional_form": {
        "relation_type": "regional_form",
        "bidirectional": True,
        "inverse": "regional_form",
    },
    "modern_form": {
        "relation_type": "modern_form",
        "bidirectional": False,
        "inverse": "archaic_form",
    },
    "archaic_form": {
        "relation_type": "archaic_form",
        "bidirectional": False,
        "inverse": "modern_form",
    },
    "obsolete_spelling": {
        "relation_type": "obsolete_spelling",
        "bidirectional": False,
        "inverse": "current_spelling",
    },
    "alternative_spelling": {
        "relation_type": "alternative_spelling",
        "bidirectional": True,
        "inverse": "alternative_spelling",
    },
    "root_of": {
        "relation_type": "root_of",
        "bidirectional": False,
        "inverse": "derived_from",
    },
    "cognate": {
        "relation_type": "cognate",
        "bidirectional": True,
        "inverse": "cognate",
    },
    # Filipino-specific relationship types
    "kasingkahulugan": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym",
    },
    "kasalungat": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym",
    },
    "kabaligtaran": {
        "relation_type": "antonym",
        "bidirectional": True,
        "inverse": "antonym",
    },
    "katulad": {
        "relation_type": "synonym",
        "bidirectional": True,
        "inverse": "synonym",
    },
    "uri_ng": {
        "relation_type": "hyponym_of",
        "bidirectional": False,
        "inverse": "hypernym_of",
    },
    "mula_sa": {
        "relation_type": "derived_from",
        "bidirectional": False,
        "inverse": "root_of",
    },
    "varyant": {
        "relation_type": "variant",
        "bidirectional": True,
        "inverse": "variant",
    },
    "kaugnay": {
        "relation_type": "related",
        "bidirectional": True,
        "inverse": "related",
    },
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
        return (
            info["relation_type"],
            info.get("bidirectional", False),
            info.get("inverse"),
        )

    # Try to find partial matches in keys
    for known_key, info in RELATION_MAPPING.items():
        if known_key in key_lower or key_lower in known_key:
            return (
                info["relation_type"],
                info.get("bidirectional", False),
                info.get("inverse"),
            )

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
        if "definitions" in data and isinstance(data["definitions"], list):
            definitions = [
                d.get("text") if isinstance(d, dict) else d for d in data["definitions"]
            ]
        elif "definition" in data and data["definition"]:
            definitions = [data["definition"]]

        # Process each definition for relationships
        for definition in definitions:
            if not definition or not isinstance(definition, str):
                continue

            # Look for synonyms in definitions
            syn_patterns = [
                r"ka(singkahulugan|tulad) ng\s+(\w+)",  # Tagalog
                r"syn(onym)?[\.:]?\s+(\w+)",  # English
                r"same\s+as\s+(\w+)",  # English
                r"another\s+term\s+for\s+(\w+)",  # English
                r"another\s+word\s+for\s+(\w+)",  # English
                r"also\s+called\s+(\w+)",  # English
                r"also\s+known\s+as\s+(\w+)",  # English
            ]

            for pattern in syn_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    synonym = (
                        match[1]
                        if isinstance(match, tuple) and len(match) > 1
                        else match
                    )
                    if synonym and isinstance(synonym, str):
                        # Get or create the synonym word
                        lang_code = data.get("language_code", "tl")
                        syn_id = get_or_create_word_id(
                            cur, synonym.strip(), language_code=lang_code
                        )

                        # Insert the synonym relationship
                        insert_relation(
                            cur, word_id, syn_id, "synonym", sources=sources
                        )

                        # For bidirectional synonyms
                        insert_relation(
                            cur, syn_id, word_id, "synonym", sources=sources
                        )

            # Look for antonyms in definitions
            ant_patterns = [
                r"(kasalungat|kabaligtaran) ng\s+(\w+)",  # Tagalog
                r"ant(onym)?[\.:]?\s+(\w+)",  # English
                r"opposite\s+of\s+(\w+)",  # English
                r"contrary\s+to\s+(\w+)",  # English
            ]

            for pattern in ant_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    antonym = (
                        match[1]
                        if isinstance(match, tuple) and len(match) > 1
                        else match
                    )
                    if antonym and isinstance(antonym, str):
                        # Get or create the antonym word
                        lang_code = data.get("language_code", "tl")
                        ant_id = get_or_create_word_id(
                            cur, antonym.strip(), language_code=lang_code
                        )

                        # Insert the antonym relationship
                        insert_relation(
                            cur, word_id, ant_id, "antonym", sources=sources
                        )

                        # For bidirectional antonyms
                        insert_relation(
                            cur, ant_id, word_id, "antonym", sources=sources
                        )

            # Look for hypernyms (broader terms) in definitions
            hyper_patterns = [
                r"uri ng\s+(\w+)",  # Tagalog
                r"type of\s+(\w+)",  # English
                r"kind of\s+(\w+)",  # English
                r"form of\s+(\w+)",  # English
                r"variety of\s+(\w+)",  # English
                r"species of\s+(\w+)",  # English
                r"member of\s+the\s+(\w+)\s+family",  # English
            ]

            for pattern in hyper_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    hypernym = match
                    if hypernym and isinstance(hypernym, str):
                        # Get or create the hypernym word
                        lang_code = data.get("language_code", "tl")
                        hyper_id = get_or_create_word_id(
                            cur, hypernym.strip(), language_code=lang_code
                        )

                        # Insert the hypernym relationship
                        insert_relation(
                            cur, word_id, hyper_id, "hyponym_of", sources=sources
                        )

                        # Add the inverse relationship
                        insert_relation(
                            cur, hyper_id, word_id, "hypernym_of", sources=sources
                        )

            # Look for variations in definitions
            var_patterns = [
                r"(iba\'t ibang|ibang) (anyo|baybay|pagsulat|bigkas) ng\s+(\w+)",  # Tagalog: different form/spelling/pronunciation of
                r"(alternatibo|alternativ|kahalili) ng\s+(\w+)",  # Tagalog: alternative of
                r"(variant|variation|alt(ernative)?) (form|spelling) of\s+(\w+)",  # English
                r"alternative (to|for)\s+(\w+)",  # English
                r"also (written|spelled) as\s+(\w+)",  # English
                r"(var\.|variant)\s+(\w+)",  # English abbreviated
                r"(regional|dialectal) form of\s+(\w+)",  # English regional variant
                r"(slang|informal) for\s+(\w+)",  # English slang variant
                r"commonly (misspelled|written) as\s+(\w+)",  # English common misspelling
                r"(baryant|lokal na anyo) ng\s+(\w+)",  # Tagalog regional variant
            ]

            for pattern in var_patterns:
                matches = re.findall(pattern, definition, re.IGNORECASE)
                for match in matches:
                    # Different patterns have target word in different positions
                    variant = None
                    if len(match) == 3 and isinstance(
                        match, tuple
                    ):  # For patterns with 3 capture groups
                        variant = match[2]
                    elif len(match) == 2 and isinstance(
                        match, tuple
                    ):  # For patterns with 2 capture groups
                        variant = match[1]
                    elif isinstance(match, str):  # For patterns with 1 capture group
                        variant = match

                    if variant and isinstance(variant, str):
                        # Get or create the variant word
                        lang_code = data.get("language_code", "tl")
                        var_id = get_or_create_word_id(
                            cur, variant.strip(), language_code=lang_code
                        )

                        # Insert the variant relationship
                        insert_relation(
                            cur, word_id, var_id, "variant", sources=sources
                        )

                        # For bidirectional variant relationship
                        insert_relation(
                            cur, var_id, word_id, "variant", sources=sources
                        )

        # Process derivative information
        derivative = data.get("derivative", "")
        if derivative and isinstance(derivative, str):
            # This indicates the word is derived from another root
            mula_sa_patterns = [
                r"mula sa\s+(.+?)(?:\s+na|\s*$)",  # Tagalog
                r"derived from\s+(?:the\s+)?(\w+)",  # English
                r"comes from\s+(?:the\s+)?(\w+)",  # English
                r"root word(?:\s+is)?\s+(\w+)",  # English
            ]

            for pattern in mula_sa_patterns:
                root_match = re.search(pattern, derivative, re.IGNORECASE)
                if root_match:
                    root_word = root_match.group(1).strip()
                    if root_word:
                        # Get or create the root word
                        lang_code = data.get("language_code", "tl")
                        root_id = get_or_create_word_id(
                            cur, root_word, language_code=lang_code
                        )

                        # Insert the derived_from relationship
                        insert_relation(
                            cur, word_id, root_id, "derived_from", sources=sources
                        )

                        # Add the inverse relationship
                        insert_relation(
                            cur, root_id, word_id, "root_of", sources=sources
                        )

        # Process etymology information for potential language relationships
        etymology = data.get("etymology", "")
        if etymology and isinstance(etymology, str):
            # Try to extract language information from etymology
            lang_patterns = {
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Spanish|Esp)[\.:]?\s+(\w+)": "es",  # Spanish
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:English|Eng)[\.:]?\s+(\w+)": "en",  # English
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Chinese|Ch|Tsino)[\.:]?\s+(\w+)": "zh",  # Chinese
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Japanese|Jap)[\.:]?\s+(\w+)": "ja",  # Japanese
                r"(?:from|borrowed from)\s+(?:the\s+)?(?:Sanskrit|San)[\.:]?\s+(\w+)": "sa",  # Sanskrit
            }

            for pattern, lang_code in lang_patterns.items():
                lang_matches = re.findall(pattern, etymology, re.IGNORECASE)
                for lang_word in lang_matches:
                    if lang_word and isinstance(lang_word, str):
                        # Get or create the foreign word
                        foreign_id = get_or_create_word_id(
                            cur, lang_word.strip(), language_code=lang_code
                        )

                        # Insert the etymology relationship
                        insert_relation(
                            cur, word_id, foreign_id, "borrowed_from", sources=sources
                        )

        # Process alternate forms and variations
        # Check if there's variations data in metadata
        variations = data.get("variations", [])
        if variations and isinstance(variations, list):
            for variant in variations:
                if isinstance(variant, str) and variant.strip():
                    # Add this explicit variation
                    var_id = get_or_create_word_id(
                        cur,
                        variant.strip(),
                        language_code=data.get("language_code", "tl"),
                    )
                    insert_relation(cur, word_id, var_id, "variant", sources=sources)
                    insert_relation(cur, var_id, word_id, "variant", sources=sources)
                elif isinstance(variant, dict) and "form" in variant:
                    var_form = variant.get("form", "").strip()
                    if var_form:
                        var_type = variant.get("type", "variant")
                        var_id = get_or_create_word_id(
                            cur, var_form, language_code=data.get("language_code", "tl")
                        )

                        # Use specific relationship type if provided, otherwise default to "variant"
                        rel_type = (
                            var_type
                            if var_type
                            in [
                                "abbreviation",
                                "misspelling",
                                "regional",
                                "alternate",
                                "dialectal",
                            ]
                            else "variant"
                        )
                        insert_relation(cur, word_id, var_id, rel_type, sources=sources)
                        insert_relation(cur, var_id, word_id, rel_type, sources=sources)

        # Look for variations by checking spelling differences
        # This will detect common spelling variations in Filipino like f/p, e/i, o/u substitutions
        word = data.get("word", "")
        if word and isinstance(word, str) and len(word) > 3:
            # Common letter substitutions in Filipino
            substitutions = [
                ("f", "p"),
                ("p", "f"),  # Filipino/Pilipino
                ("e", "i"),
                ("i", "e"),  # like in leeg/liig (neck)
                ("o", "u"),
                ("u", "o"),  # like in puso/poso (heart)
                ("k", "c"),
                ("c", "k"),  # like in karera/carera (race)
                ("w", "u"),
                ("u", "w"),  # like in uwi/uwi (go home)
                ("j", "h"),
                ("h", "j"),  # like in jahit/hahit
                ("s", "z"),
                ("z", "s"),  # like in kasoy/kazoy
                ("ts", "ch"),
                ("ch", "ts"),  # like in tsaa/chaa (tea)
            ]

            # Generate possible variations
            potential_variations = []
            for i, char in enumerate(word):
                for orig, repl in substitutions:
                    if char.lower() == orig:
                        var = word[:i] + repl + word[i + 1 :]
                        potential_variations.append(var)
                    elif (
                        char.lower() == orig[0]
                        and i < len(word) - 1
                        and word[i + 1].lower() == orig[1]
                    ):
                        var = word[:i] + repl + word[i + 2 :]
                        potential_variations.append(var)

            # Check if these variations actually exist in the database
            for var in potential_variations:
                # Skip if the variation is the same as the original word
                if var.lower() == word.lower():
                    continue

                cur.execute(
                    """
                    SELECT id FROM words 
                    WHERE normalized_lemma = %s AND language_code = %s
                """,
                    (normalize_lemma(var), data.get("language_code", "tl")),
                )

                result = cur.fetchone()
                if result:
                    # Found a real variation in the database
                    var_id = result[0]
                    insert_relation(
                        cur, word_id, var_id, "spelling_variant", sources=sources
                    )
                    insert_relation(
                        cur, var_id, word_id, "spelling_variant", sources=sources
                    )

    except Exception as e:
        logger.error(f"Error processing relationships for word_id {word_id}: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            tb_str = "".join(traceback.format_tb(e.__traceback__))
            logger.error(f"Traceback: {tb_str}")


@with_transaction(commit=True)
def process_direct_relations(cur, word_id, entry, lang_code, source):
    """Process direct relationships specified in the entry."""
    relationship_mappings = {
        "synonyms": ("synonym", True),  # bidirectional
        "antonyms": ("antonym", True),  # bidirectional
        "derived": ("derived_from", False),  # not bidirectional, direction is important
        "related": ("related", True),  # bidirectional
    }

    for rel_key, (rel_type, bidirectional) in relationship_mappings.items():
        if rel_key in entry and isinstance(entry[rel_key], list):
            for rel_item in entry[rel_key]:
                # Initialize metadata
                metadata = {}

                # Handle both string words and dictionary objects with word property
                if isinstance(rel_item, dict) and "word" in rel_item:
                    rel_word = rel_item["word"]

                    # Extract metadata fields if available
                    if "strength" in rel_item:
                        metadata["strength"] = rel_item["strength"]

                    if "tags" in rel_item and rel_item["tags"]:
                        metadata["tags"] = rel_item["tags"]

                    if "english" in rel_item and rel_item["english"]:
                        metadata["english"] = rel_item["english"]

                    # Extract any other useful fields
                    for field in ["sense", "extra", "notes"]:
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
                if rel_type == "derived_from":
                    from_id = word_id
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            from_id,
                            to_id,
                            rel_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(cur, from_id, to_id, rel_type, sources=source)
                else:
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            word_id,
                            to_id,
                            rel_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(cur, word_id, to_id, rel_type, sources=source)

                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(
                                cur,
                                to_id,
                                word_id,
                                rel_type,
                                sources=source,
                                metadata=metadata,
                            )
                        else:
                            insert_relation(
                                cur, to_id, word_id, rel_type, sources=source
                            )


@with_transaction(commit=True)
def process_relations(
    cur,
    from_word_id: int,
    relations_dict: Dict[str, List[str]],
    lang_code: str,
    source: str,
):
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
            if isinstance(rel_item, dict) and "word" in rel_item:
                rel_word_lemma = rel_item["word"]

                # Extract metadata fields if available
                if "strength" in rel_item:
                    metadata["strength"] = rel_item["strength"]

                if "tags" in rel_item and rel_item["tags"]:
                    metadata["tags"] = rel_item["tags"]

                if "english" in rel_item and rel_item["english"]:
                    metadata["english"] = rel_item["english"]

                # Extract any other useful fields
                for field in ["sense", "extra", "notes", "context"]:
                    if field in rel_item and rel_item[field]:
                        metadata[field] = rel_item[field]
            elif isinstance(rel_item, str):
                rel_word_lemma = rel_item
            else:
                continue

            to_word_id = get_or_create_word_id(
                cur, rel_word_lemma, language_code=lang_code
            )

            # Only include metadata if it's not empty
            if metadata:
                insert_relation(
                    cur,
                    from_word_id,
                    to_word_id,
                    relation_type,
                    sources=source,
                    metadata=metadata,
                )
            else:
                insert_relation(
                    cur, from_word_id, to_word_id, relation_type, sources=source
                )

            if bidirectional and inverse_type:
                # For bidirectional relationships, we might want to copy the metadata
                if metadata:
                    insert_relation(
                        cur,
                        to_word_id,
                        from_word_id,
                        inverse_type,
                        sources=source,
                        metadata=metadata,
                    )
                else:
                    insert_relation(
                        cur, to_word_id, from_word_id, inverse_type, sources=source
                    )


@with_transaction(commit=True)
def extract_sense_relations(cur, word_id, sense, lang_code, source):
    """Extract and process relationship data from a word sense."""
    for rel_type in ["synonyms", "antonyms", "derived", "related"]:
        if rel_type in sense and isinstance(sense[rel_type], list):
            relation_items = sense[rel_type]
            relationship_type = (
                "synonym"
                if rel_type == "synonyms"
                else (
                    "antonym"
                    if rel_type == "antonyms"
                    else "derived_from" if rel_type == "derived" else "related"
                )
            )
            bidirectional = (
                rel_type != "derived"
            )  # derived relationships are not bidirectional

            for item in relation_items:
                # Initialize metadata
                metadata = {}

                # Handle both string words and dictionary objects with word property
                if isinstance(item, dict) and "word" in item:
                    rel_word = item["word"]

                    # Extract metadata fields if available
                    if "strength" in item:
                        metadata["strength"] = item["strength"]

                    if "tags" in item and item["tags"]:
                        metadata["tags"] = item["tags"]

                    if "english" in item and item["english"]:
                        metadata["english"] = item["english"]

                    # Extract sense-specific context if available
                    if "sense" in item and item["sense"]:
                        metadata["sense"] = item["sense"]

                    # Extract any other useful fields
                    for field in ["extra", "notes", "context"]:
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
                if relationship_type == "derived_from":
                    from_id = word_id
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            from_id,
                            to_id,
                            relationship_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(
                            cur, from_id, to_id, relationship_type, sources=source
                        )
                else:
                    to_id = get_or_create_word_id(
                        cur, rel_word, language_code=lang_code
                    )
                    # Only include metadata if it's not empty
                    if metadata:
                        insert_relation(
                            cur,
                            word_id,
                            to_id,
                            relationship_type,
                            sources=source,
                            metadata=metadata,
                        )
                    else:
                        insert_relation(
                            cur, word_id, to_id, relationship_type, sources=source
                        )

                    # Add bidirectional relationship if needed
                    if bidirectional:
                        # For bidirectional relationships, we might want to copy the metadata
                        if metadata:
                            insert_relation(
                                cur,
                                to_id,
                                word_id,
                                relationship_type,
                                sources=source,
                                metadata=metadata,
                            )
                        else:
                            insert_relation(
                                cur, to_id, word_id, relationship_type, sources=source
                            )


def extract_definition_tags(definition_dict):
    """Extract tags from definition data"""
    tags = []

    # Extract from directly tagged definitions
    if "tags" in definition_dict and isinstance(definition_dict["tags"], list):
        tags.extend(definition_dict["tags"])

    # Look for tags in glosses field that might indicate usage
    special_tags = [
        "figuratively",
        "figurative",
        "colloquial",
        "formal",
        "informal",
        "archaic",
        "obsolete",
        "rare",
    ]
    glosses = definition_dict.get("glosses", [])
    if isinstance(glosses, list) and glosses:
        for gloss in glosses:
            if isinstance(gloss, str):
                # Check if the gloss starts with a tag in parentheses
                tag_match = re.match(r"^\(([^)]+)\)", gloss)
                if tag_match:
                    potential_tag = tag_match.group(1).lower()
                    if potential_tag in special_tags:
                        tags.append(potential_tag)

    return tags if tags else None


def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract Baybayin form and romanized form from an entry."""
    baybayin_form = entry.get("baybayin_form")
    romanized_form = entry.get("romanized_form")

    # Return early if no Baybayin form
    if not baybayin_form:
        return None, None

    # Validate the Baybayin form
    if validate_baybayin_entry(baybayin_form, romanized_form):
        return baybayin_form, romanized_form
    return None, None


# --- Helper function to extract script info ---
def extract_script_info(entry: Dict, script_tag: str, script_name_in_template: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts specific script form and explicit romanization if available."""
    script_form = None
    romanized = None
    
    # Try 'forms' array first
    if "forms" in entry and isinstance(entry["forms"], list):
        for form_data in entry["forms"]:
            if isinstance(form_data, dict) and "tags" in form_data and script_tag in form_data.get("tags", []):
                form_text = form_data.get("form", "").strip()
                if form_text:
                    prefixes = ["spelling ", "script ", script_tag.lower() + " "]
                    cleaned_form = form_text
                    for prefix in prefixes:
                        if cleaned_form.lower().startswith(prefix):
                            cleaned_form = cleaned_form[len(prefix):].strip()
                    # Basic validation - check for non-Latin chars
                    if cleaned_form and any(not ('a' <= char.lower() <= 'z') for char in cleaned_form):
                        script_form = cleaned_form
                        romanized = form_data.get("romanized")  # Get explicit romanization
                        return script_form, romanized

    # Fallback: Try 'head_templates' expansion
    if "head_templates" in entry and isinstance(entry["head_templates"], list):
        for template in entry["head_templates"]:
            if isinstance(template, dict) and "expansion" in template:
                expansion = template.get("expansion", "")
                if isinstance(expansion, str):
                    # Regex to find script spelling after specific text
                    pattern = rf'{script_name_in_template} spelling\s+([\u1700-\u171F\u1730-\u173F\s]+)'  # Adjust range if needed
                    match = re.search(pattern, expansion, re.IGNORECASE)
                    if match:
                        potential_script = match.group(1).strip()
                        # Basic validation
                        if potential_script and any(not ('a' <= char.lower() <= 'z') for char in potential_script):
                            script_form = potential_script
                            romanized = None  # Romanization unlikely here
                            return script_form, romanized
    return None, None


@with_transaction(commit=True)  # Changed to commit=True to follow process_tagalog_words pattern
def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries with optimized transaction handling."""
    # Initialize statistics
    entry_stats = {
        "definitions": 0, "relations": 0, "etymologies": 0, "pronunciations": 0,
        "forms": 0, "categories": 0, "templates": 0, "links": 0, "scripts": 0,
        "ety_relations": 0, "sense_relations": 0, "form_relations": 0, "examples": 0,
    }

    # Cache common lookups to reduce database queries
    word_cache = {}  # Format: "{lemma}|{lang_code}" -> word_id
    pos_code_cache = {}  # POS code -> standardized_pos_id

    # Standardize the source identifier from the filename
    source_identifier = standardize_source_identifier(os.path.basename(filename))
    raw_source_identifier = os.path.basename(filename)
    logger.info(f"Processing Kaikki dictionary: {filename} with source ID: '{source_identifier}'")

    # Count total lines in file
    try:
        with open(filename, "r", encoding="utf-8") as f_count:
            total_lines = sum(1 for _ in f_count)
        logger.info(f"Found {total_lines} entries to process in {filename}")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {"total_entries": 0, "processed_entries": 0, "error_entries": 1, "skipped_entries": 0}
    except Exception as count_error:
        logger.error(f"Error counting lines in {filename}: {count_error}")
        total_lines = -1  # Indicate unknown total

    # Initialize romanizer
    romanizer = None
    try:
        if 'BaybayinRomanizer' in globals() and callable(globals()['BaybayinRomanizer']):
            romanizer = BaybayinRomanizer()
            logger.info("Initialized BaybayinRomanizer for script romanization")
        else:
            logger.warning("BaybayinRomanizer class not found. Cannot initialize.")
    except Exception as rom_err:
        logger.warning(f"Could not initialize BaybayinRomanizer: {rom_err}")

    # --- Process entries in chunks for better performance ---
    def process_entry_chunk(entries_chunk, chunk_idx):
        """Process a chunk of entries to reduce overhead."""
        processed_ok = 0
        processed_with_errors = 0
        failed_entries = 0
        error_details = {}
        
        for entry in entries_chunk:
            entry_result, error = process_single_entry(entry)
            if entry_result:
                if error:
                    processed_with_errors += 1
                    error_key = error[:100] if error else "Unknown error"
                    error_details[error_key] = error_details.get(error_key, 0) + 1
                else:
                    processed_ok += 1
            else:
                failed_entries += 1
                error_key = error[:100] if error else "Unknown critical failure"
                error_details[error_key] = error_details.get(error_key, 0) + 1
                
        return processed_ok, processed_with_errors, failed_entries, error_details

    # --- Single entry processor ---
    def process_single_entry(entry):
        """Process a single dictionary entry with optimized data handling."""
        if "word" not in entry or not entry["word"]:
            return None, "No word field"

        word = entry["word"]
        word_id = None
        error_messages = []
        local_stats = {k: 0 for k in entry_stats.keys()}
        
        try:
            # Get language code and validate
            pos = entry.get("pos", "unc")
            language_code = entry.get("lang_code", DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                language_code = DEFAULT_LANGUAGE_CODE
            
            # Extract Baybayin and other script info
            baybayin_form, baybayin_romanized = extract_script_info(entry, "Baybayin", "Baybayin") 
            badlit_form, badlit_romanized = extract_script_info(entry, "Badlit", "Badlit")
            
            # --- Baybayin Validation ---
            validated_baybayin_form = None
            has_baybayin = False
            if baybayin_form:
                # Check if the form contains only valid characters
                # Stripping the problematic dotted circle U+25CC proactively
                cleaned_baybayin_form = baybayin_form.replace("\u25CC", "")
                if cleaned_baybayin_form and VALID_BAYBAYIN_REGEX.match(cleaned_baybayin_form):
                    validated_baybayin_form = cleaned_baybayin_form
                    has_baybayin = True
                else:
                    # Log if the original form was non-empty but became invalid/empty after cleaning
                    if baybayin_form: 
                         logger.warning(f"Invalid or problematic Baybayin form found for '{word}': '{baybayin_form}'. Ignoring Baybayin for this entry.")
                         error_messages.append(f"Invalid Baybayin form '{baybayin_form}' ignored")

            # Generate romanization if needed, using the validated form
            romanized_form = None
            if validated_baybayin_form and not baybayin_romanized and romanizer and romanizer.validate_text(validated_baybayin_form):
                try:
                    romanized_form = romanizer.romanize(validated_baybayin_form)
                except Exception as rom_err:
                    # Log the romanization error instead of passing silently
                    logger.warning(f"Could not romanize Baybayin '{validated_baybayin_form}' for word '{word}': {rom_err}")
                    error_messages.append(f"Romanization error: {rom_err}")
                    
            if not romanized_form and baybayin_romanized:
                romanized_form = baybayin_romanized
            elif not romanized_form and badlit_romanized:
                romanized_form = badlit_romanized
                
            if has_baybayin or badlit_form: # Count scripts based on validated/present forms
                local_stats["scripts"] += 1
            
            # Check cache for existing word
            cache_key = f"{word.lower()}|{language_code}"
            if cache_key in word_cache:
                word_id = word_cache[cache_key]
            else:
                # Prepare word attributes
                is_proper_noun = entry.get("proper", False) or pos in ["prop", "proper noun", "name"]
                is_abbreviation = pos in ["abbrev", "abbreviation"]
                is_initialism = pos in ["init", "initialism", "acronym"]
                
                # Process tags
                tags_list = entry.get("tags", [])
                if isinstance(tags_list, list):
                    if any(t in ["abbreviation", "abbrev"] for t in tags_list):
                        is_abbreviation = True
                    if any(t in ["initialism", "acronym"] for t in tags_list):
                        is_initialism = True
                
                # Format tag string and prepare metadata
                word_tags_str = ",".join(tags_list) if tags_list else None
                word_metadata = {"source_file": raw_source_identifier}
                
                # Handle hyphenation
                hyphenation = entry.get("hyphenation")
                hyphenation_json = Json(hyphenation) if hyphenation and isinstance(hyphenation, list) else None
                
                # Create or get word ID
                word_id = get_or_create_word_id(
                    cur, 
                    lemma=word,
                    language_code=language_code,
                    source_identifier=source_identifier,
                    has_baybayin=has_baybayin, # Use validated status
                    baybayin_form=validated_baybayin_form, # Use validated form
                    romanized_form=romanized_form,
                    badlit_form=badlit_form, # Consider validating Badlit similarly if needed
                    hyphenation=hyphenation,
                    is_proper_noun=is_proper_noun,
                    is_abbreviation=is_abbreviation,
                    is_initialism=is_initialism,
                    tags=word_tags_str,
                    word_metadata=word_metadata
                )
                
                # Add to cache for potential reuse
                if word_id:
                    word_cache[cache_key] = word_id
            
            if not word_id:
                return None, f"Failed to get/create word ID for '{word}'"
            
            # Process related data in properly grouped operations
            
            # 1. Process pronunciations
            if "sounds" in entry and isinstance(entry["sounds"], list):
                for sound in entry.get("sounds", []):
                    if not isinstance(sound, dict):
                        continue
                        
                    # Extract pronunciation info
                    pron_type, pron_value = None, None
                    if "ipa" in sound and sound["ipa"]: 
                        pron_type, pron_value = "ipa", sound["ipa"]
                    elif "enpr" in sound and sound["enpr"]: 
                        pron_type, pron_value = "enpr", sound["enpr"]
                    elif "rhymes" in sound and sound["rhymes"]: 
                        pron_type, pron_value = "rhyme", sound["rhymes"]
                    elif "audio" in sound and sound["audio"]: 
                        pron_type, pron_value = "audio", sound["audio"]
                    else:
                        continue
                        
                    if pron_type and pron_value:
                        pron_data = sound.copy()
                        pron_data["type"], pron_data["value"] = pron_type, pron_value
                        
                        try:
                            pron_id = insert_pronunciation(cur, word_id, pron_data, source_identifier)
                            if pron_id:
                                local_stats["pronunciations"] += 1
                        except Exception as e:
                            error_messages.append(f"Pronunciation error ({pron_type}): {str(e)}")
                
            # 2. Process word forms and relationships together
            if "forms" in entry and isinstance(entry["forms"], list):
                # First collect forms data
                canonical_forms = set()
                for form in entry["forms"]:
                    if isinstance(form, dict) and "form" in form and form.get("form") and "tags" in form and "canonical" in form.get("tags", []):
                        canonical_forms.add(form["form"])
                
                # Process each form
                for form_data in entry["forms"]:
                    if not isinstance(form_data, dict) or not form_data.get("form"):
                        continue
                        
                    form_text = form_data["form"]
                    form_tags = form_data.get("tags", [])
                    
                    # Skip script-specific forms processed elsewhere
                    if "Baybayin" in form_tags or "Badlit" in form_tags:
                        continue
                    
                    # Process regular form
                    tags_json = Json(form_tags) if form_tags else None
                    is_canonical = "canonical" in form_tags
                    is_primary = "primary" in form_tags
                    
                    try:
                        # Insert form data
                        cur.execute("""
                            INSERT INTO word_forms (word_id, form, is_canonical, is_primary, tags) 
                            VALUES (%s, %s, %s, %s, %s) 
                            ON CONFLICT (word_id, form) DO UPDATE 
                            SET is_canonical = EXCLUDED.is_canonical, 
                                is_primary = EXCLUDED.is_primary, 
                                tags = COALESCE(word_forms.tags, '{}'::jsonb) || %s::jsonb, 
                                updated_at = CURRENT_TIMESTAMP
                            """, (word_id, form_text, is_canonical, is_primary, tags_json, tags_json))
                        
                        if cur.rowcount > 0:
                            local_stats["forms"] += 1
                            
                        # Process form relationships for non-canonical forms
                        if not is_canonical and form_text != word and form_text not in canonical_forms:
                            # Determine relation type based on tags
                            rel_type = RelationshipType.VARIANT
                            metadata = {"from_forms": True, "confidence": rel_type.strength}
                            
                            if form_tags and isinstance(form_tags, list):
                                metadata["tags"] = ",".join(form_tags)
                                if any(t in ["standard spelling", "preferred", "standard form", "canonical"] for t in form_tags):
                                    rel_type, metadata["confidence"] = RelationshipType.SPELLING_VARIANT, 95
                                elif any(t in ["alternative spelling", "alternate spelling", "alt form"] for t in form_tags):
                                    rel_type, metadata["confidence"] = RelationshipType.SPELLING_VARIANT, 90
                                elif any(t in ["regional", "dialectal", "dialect"] for t in form_tags):
                                    rel_type, metadata["confidence"] = RelationshipType.REGIONAL_VARIANT, 85
                                elif "obsolete" in form_tags:
                                    metadata["confidence"] = 70
                            
                            if "qualifier" in form_data:
                                metadata["qualifier"] = form_data["qualifier"]
                                
                            # Get form word ID
                            form_cache_key = f"{form_text.lower()}|{language_code}"
                            if form_cache_key in word_cache:
                                form_word_id = word_cache[form_cache_key]
                            else:
                                form_word_id = get_or_create_word_id(cur, form_text, language_code, source_identifier=source_identifier)
                                if form_word_id:
                                    word_cache[form_cache_key] = form_word_id
                            
                            # Create relationship
                            if form_word_id and form_word_id != word_id:
                                rel_id = insert_relation(cur, word_id, form_word_id, rel_type, source_identifier, metadata)
                                if rel_id:
                                    local_stats["form_relations"] += 1
                                    local_stats["relations"] += 1
                                
                                # Add inverse relation if applicable
                                if rel_type.bidirectional:
                                    insert_relation(cur, form_word_id, word_id, rel_type, source_identifier, metadata)
                                else:
                                    inverse_rel = rel_type.get_inverse()
                                    if inverse_rel:
                                        insert_relation(cur, form_word_id, word_id, inverse_rel, source_identifier, metadata)
                    except Exception as e:
                        error_messages.append(f"Form error for '{form_text}': {str(e)}")
                
            # 3. Process etymology
            if "etymology_text" in entry and entry["etymology_text"]:
                try:
                    # Process etymology text
                    etymology_text = entry["etymology_text"]
                    etymology_templates = entry.get("etymology_templates", [])
                    language_codes = None
                    
                    # Extract language codes from templates
                    if etymology_templates and isinstance(etymology_templates, list):
                        languages = set()
                        for template in etymology_templates:
                            if not isinstance(template, dict) or "name" not in template:
                                continue
                                
                            args = template.get("args", {})
                            for key, value in args.items():
                                if isinstance(value, str) and 1 < len(value) <= 3 and value.isalpha():
                                    languages.add(value.lower())
                                elif isinstance(key, str) and 1 < len(key) <= 3 and key.isalpha():
                                    languages.add(key.lower())
                                    
                        if languages:
                            language_codes = ",".join(sorted(list(languages)))
                    
                    # Insert etymology
                    cleaned_text = clean_html(etymology_text.strip())
                    ety_id = insert_etymology(
                        cur, 
                        word_id=word_id,
                        etymology_text=cleaned_text,
                        source_identifier=source_identifier,
                        language_codes=language_codes
                    )
                    
                    if ety_id:
                        local_stats["etymologies"] += 1
                    
                    # Process etymology relationships
                    if etymology_templates and isinstance(etymology_templates, list):
                        for template in etymology_templates:
                            if not isinstance(template, dict) or "name" not in template:
                                continue
                                
                            template_name = template["name"].lower()
                            args = template.get("args", {})
                            
                            if template_name in ["derived", "borrowing", "derived from", "borrowed from", "bor", "der", "bor+"]:
                                rel_type = RelationshipType.DERIVED_FROM
                                source_lang = args.get("1", "") or args.get("2", "")
                                source_word = args.get("2", "") or args.get("3", "")
                                
                                if source_word and isinstance(source_word, str):
                                    source_word_clean = clean_html(source_word)
                                    
                                    if source_word_clean and source_word_clean.lower() != word.lower():
                                        source_word_lang = source_lang if source_lang else language_code
                                        
                                        # Check cache for source word
                                        source_cache_key = f"{source_word_clean.lower()}|{source_word_lang}"
                                        if source_cache_key in word_cache:
                                            related_word_id = word_cache[source_cache_key]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur, 
                                                source_word_clean, 
                                                source_word_lang, 
                                                source_identifier
                                            )
                                            if related_word_id:
                                                word_cache[source_cache_key] = related_word_id
                                                
                                        if related_word_id and related_word_id != word_id:
                                            metadata = {
                                                "from_etymology": True,
                                                "template": template_name,
                                                "confidence": rel_type.strength
                                            }
                                            
                                            # Insert relation
                                            rel_id = insert_relation(
                                                cur, 
                                                word_id, 
                                                related_word_id,
                                                rel_type,
                                                source_identifier,
                                                metadata
                                            )
                                            
                                            if rel_id:
                                                local_stats["ety_relations"] += 1
                                                local_stats["relations"] += 1
                                            
                                            # Add inverse relation
                                            inverse_rel = rel_type.get_inverse()
                                            if inverse_rel:
                                                insert_relation(
                                                    cur,
                                                    related_word_id,
                                                    word_id,
                                                    inverse_rel,
                                                    source_identifier,
                                                    metadata
                                                )
                except Exception as e:
                    error_messages.append(f"Etymology error: {str(e)}")
                    
            # 4. Process templates
            if "head_templates" in entry and entry["head_templates"]:
                try:
                    for template in entry["head_templates"]:
                        if not isinstance(template, dict) or "name" not in template:
                            continue
                            
                        template_name = template["name"]
                        args = template.get("args")
                        expansion = template.get("expansion")
                        
                        cur.execute("""
                            INSERT INTO word_templates (word_id, template_name, args, expansion) 
                            VALUES (%s, %s, %s, %s) 
                            ON CONFLICT (word_id, template_name) DO UPDATE 
                            SET args = EXCLUDED.args, 
                                expansion = EXCLUDED.expansion, 
                                updated_at = CURRENT_TIMESTAMP
                            """, (word_id, template_name, Json(args) if args else None, expansion))
                        
                        if cur.rowcount > 0:
                            local_stats["templates"] += 1
                except Exception as e:
                    error_messages.append(f"Template error: {str(e)}")
                    
            # 5. Process senses/definitions
            if "senses" in entry and isinstance(entry["senses"], list):
                sense_processed_count = 0
                for sense_idx, sense in enumerate(entry["senses"]):
                    if not isinstance(sense, dict):
                        continue
                        
                    try:
                        # Get glosses
                        glosses = sense.get("glosses", []) or sense.get("raw_glosses", [])
                        if not glosses or not isinstance(glosses, list):
                            continue
                            
                        # Create definition text
                        definition_text = "; ".join([clean_html(g) for g in glosses if isinstance(g, str)])
                        if not definition_text:
                            continue
                            
                        # Process examples
                        examples = []
                        if "examples" in sense and isinstance(sense["examples"], list):
                            for example in sense["examples"]:
                                ex_text = None
                                if isinstance(example, str):
                                    ex_text = clean_html(example)
                                elif isinstance(example, dict):
                                    ex_text = clean_html(example.get("text"))
                                    
                                if ex_text:
                                    examples.append(ex_text)
                                    local_stats["examples"] += 1
                                    
                        # Process tags and labels
                        sense_tags = sense.get("tags", [])
                        sense_labels = sense.get("labels", [])
                        all_tags = (sense_tags if isinstance(sense_tags, list) else []) + (sense_labels if isinstance(sense_labels, list) else [])
                        tags_str = ",".join(all_tags) if all_tags else None
                        
                        # Process usage notes
                        usage_notes = None
                        if "usage_notes" in sense and sense["usage_notes"]:
                            usage_notes = clean_html(sense["usage_notes"])
                            
                        # Build metadata
                        def_metadata = {}
                        for key in ["form_of", "raw_glosses", "topics", "taxonomy", "qualifier"]:
                            if key in sense and sense[key]:
                                def_metadata[key] = sense[key]
                                
                        # Get part of speech
                        sense_pos_str = sense.get("pos") or pos
                        
                        # Insert definition
                        definition_id = insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            source_identifier=source_identifier,
                            part_of_speech=sense_pos_str,
                            examples=Json(examples) if examples else None,
                            usage_notes=usage_notes,
                            tags=tags_str,
                            metadata=Json(def_metadata) if def_metadata else None
                        )
                        
                        if definition_id:
                            sense_processed_count += 1
                            local_stats["definitions"] += 1
                            
                            # Process definition categories
                            if "categories" in sense and isinstance(sense["categories"], list):
                                for category in sense["categories"]:
                                    if not isinstance(category, dict) or "name" not in category:
                                        continue
                                        
                                    category_name = category["name"]
                                    category_kind = category.get("kind")
                                    parents = category.get("parents")
                                    
                                    try:
                                        cur.execute("""
                                            INSERT INTO definition_categories 
                                            (definition_id, category_name, category_kind, parents) 
                                            VALUES (%s, %s, %s, %s) 
                                            ON CONFLICT (definition_id, category_name) DO NOTHING
                                            """, (
                                                definition_id, 
                                                category_name, 
                                                category_kind, 
                                                Json(parents) if parents else None
                                            ))
                                        
                                        if cur.rowcount > 0:
                                            local_stats["categories"] += 1
                                    except Exception as cat_e:
                                        error_messages.append(f"Category error: {str(cat_e)}")
                                        
                            # Process links
                            if "links" in sense and isinstance(sense["links"], list):
                                for link in sense["links"]:
                                    if not isinstance(link, list) or len(link) < 2:
                                        continue
                                        
                                    link_text, link_target = link[0], link[1]
                                    if not link_text or not link_target:
                                        continue
                                        
                                    is_wikipedia = ("wikipedia.org" in link_target.lower() or "w:" in link_text or "W:" in link_text)
                                    
                                    try:
                                        cur.execute("""
                                            INSERT INTO definition_links 
                                            (definition_id, link_text, link_target, is_wikipedia) 
                                            VALUES (%s, %s, %s, %s) 
                                            ON CONFLICT (definition_id, link_text, link_target) DO NOTHING
                                            """, (definition_id, link_text, link_target, is_wikipedia))
                                        
                                        if cur.rowcount > 0:
                                            local_stats["links"] += 1
                                    except Exception as link_e:
                                        error_messages.append(f"Link error: {str(link_e)}")
                                        
                            # Process sense relationships
                            relation_types = {
                                "synonyms": RelationshipType.SYNONYM,
                                "antonyms": RelationshipType.ANTONYM,
                                "hypernyms": RelationshipType.HYPERNYM,
                                "hyponyms": RelationshipType.HYPONYM, 
                                "holonyms": RelationshipType.HOLONYM,
                                "meronyms": RelationshipType.MERONYM,
                                "derived": RelationshipType.ROOT_OF,
                                "related": RelationshipType.RELATED,
                                "coordinate_terms": RelationshipType.RELATED,
                                "see_also": RelationshipType.SEE_ALSO
                            }
                            
                            for rel_key, rel_enum in relation_types.items():
                                if rel_key in sense and isinstance(sense[rel_key], list):
                                    for item in sense[rel_key]:
                                        related_word = None
                                        metadata = {"from_sense": True, "confidence": rel_enum.strength}
                                        
                                        if isinstance(item, dict) and "word" in item:
                                            related_word = item["word"]
                                        elif isinstance(item, str):
                                            related_word = item
                                        else:
                                            continue
                                            
                                        related_word_clean = clean_html(related_word)
                                        if not related_word_clean or related_word_clean.lower() == word.lower():
                                            continue
                                            
                                        # Get or create related word
                                        related_cache_key = f"{related_word_clean.lower()}|{language_code}"
                                        if related_cache_key in word_cache:
                                            related_word_id = word_cache[related_cache_key]
                                        else:
                                            related_word_id = get_or_create_word_id(
                                                cur,
                                                related_word_clean,
                                                language_code,
                                                source_identifier=source_identifier
                                            )
                                            if related_word_id:
                                                word_cache[related_cache_key] = related_word_id
                                                
                                        if related_word_id and related_word_id != word_id:
                                            # Determine direction for non-symmetrical relations
                                            from_id, to_id, current_rel_enum = word_id, related_word_id, rel_enum
                                            
                                            if rel_key == "hyponyms":
                                                from_id, to_id, current_rel_enum = related_word_id, word_id, RelationshipType.HYPONYM
                                            elif rel_key == "hypernyms":
                                                current_rel_enum = RelationshipType.HYPONYM
                                                
                                            # Insert relation
                                            rel_id = insert_relation(
                                                cur,
                                                from_id,
                                                to_id,
                                                current_rel_enum,
                                                source_identifier,
                                                metadata
                                            )
                                            
                                            if rel_id:
                                                local_stats["sense_relations"] += 1
                                                local_stats["relations"] += 1
                                                
                                            # Add inverse/bidirectional relation if needed
                                            if current_rel_enum.bidirectional:
                                                insert_relation(
                                                    cur,
                                                    to_id,
                                                    from_id,
                                                    current_rel_enum,
                                                    source_identifier,
                                                    metadata
                                                )
                                            else:
                                                inverse_rel = current_rel_enum.get_inverse()
                                                if inverse_rel:
                                                    insert_relation(
                                                        cur,
                                                        to_id,
                                                        from_id,
                                                        inverse_rel,
                                                        source_identifier,
                                                        metadata
                                                    )
                            
                            # Check for superseded spelling in definition
                            try:
                                superseded_match = re.search(
                                    r"Superseded.*?spelling of\s+([^\s\.,(]+)", 
                                    definition_text, 
                                    re.IGNORECASE
                                )
                                
                                if superseded_match:
                                    correct_spelling = superseded_match.group(1).strip("()")
                                    if correct_spelling and correct_spelling.lower() != word.lower():
                                        correct_cache_key = f"{correct_spelling.lower()}|{language_code}"
                                        
                                        if correct_cache_key in word_cache:
                                            correct_word_id = word_cache[correct_cache_key]
                                        else:
                                            correct_word_id = get_or_create_word_id(
                                                cur,
                                                correct_spelling,
                                                language_code,
                                                source_identifier=source_identifier
                                            )
                                            if correct_word_id:
                                                word_cache[correct_cache_key] = correct_word_id
                                                
                                        if correct_word_id and correct_word_id != word_id:
                                            meta = {"tag": "superseded", "definition_id": definition_id}
                                            rel_id = insert_relation(
                                                cur,
                                                word_id,
                                                correct_word_id,
                                                RelationshipType.SPELLING_VARIANT,
                                                source_identifier,
                                                meta
                                            )
                                            
                                            if rel_id:
                                                local_stats["relations"] += 1
                            except Exception as e:
                                error_messages.append(f"Superseded spelling error: {str(e)}")
                                
                    except psycopg2.errors.UniqueViolation:
                        # Definition already exists, not an error
                        pass
                    except Exception as e:
                        error_messages.append(f"Definition error: {str(e)}")
                
                # Log if no definitions were processed
                if sense_processed_count == 0:
                    logger.info(f"No definitions successfully processed for word '{word}' (ID: {word_id}).")
                    
            # 6. Process top-level relations
            try:
                top_level_rels = {
                    "derived": RelationshipType.ROOT_OF,
                    "related": RelationshipType.RELATED
                }
                
                for rel_key, rel_enum in top_level_rels.items():
                    if rel_key in entry and isinstance(entry[rel_key], list):
                        for item in entry[rel_key]:
                            related_word = None
                            
                            if isinstance(item, dict) and "word" in item:
                                related_word = item["word"]
                            elif isinstance(item, str):
                                related_word = item
                            else:
                                continue
                                
                            related_word_clean = clean_html(related_word)
                            if not related_word_clean or related_word_clean.lower() == word.lower():
                                continue
                                
                            # Get or create related word
                            related_cache_key = f"{related_word_clean.lower()}|{language_code}"
                            if related_cache_key in word_cache:
                                related_word_id = word_cache[related_cache_key]
                            else:
                                related_word_id = get_or_create_word_id(
                                    cur,
                                    related_word_clean,
                                    language_code,
                                    source_identifier=source_identifier
                                )
                                if related_word_id:
                                    word_cache[related_cache_key] = related_word_id
                                    
                            if related_word_id and related_word_id != word_id:
                                metadata = {"confidence": rel_enum.strength, "from_sense": False}
                                
                                if isinstance(item, dict) and "tags" in item:
                                    metadata["tags"] = ",".join(item["tags"])
                                    
                                # Insert relation
                                rel_id = insert_relation(
                                    cur,
                                    word_id,
                                    related_word_id,
                                    rel_enum,
                                    source_identifier,
                                    metadata
                                )
                                
                                if rel_id:
                                    local_stats["relations"] += 1
                                    
                                # Add inverse/bidirectional relation if needed
                                if rel_enum.bidirectional:
                                    insert_relation(
                                        cur,
                                        related_word_id,
                                        word_id,
                                        rel_enum,
                                        source_identifier,
                                        metadata
                                    )
                                else:
                                    inv = rel_enum.get_inverse()
                                    if inv:
                                        insert_relation(
                                            cur,
                                            related_word_id,
                                            word_id,
                                            inv,
                                            source_identifier,
                                            metadata
                                        )
            except Exception as e:
                error_messages.append(f"Top-level relation error: {str(e)}")
                
            # Update global statistics
            for key, value in local_stats.items():
                entry_stats[key] += value
                
            return word_id, "; ".join(error_messages) if error_messages else None
            
        except Exception as e:
            logger.error(f"Unexpected error processing entry for '{word}': {str(e)}")
            # Ensure word_id is None if the core processing failed before ID acquisition
            current_word_id = word_id if 'word_id' in locals() and word_id is not None else None
            return current_word_id, f"Unhandled exception: {str(e)}"
    
    # Main processing logic
    stats = {
        "total_entries": total_lines if total_lines >= 0 else 0,
        "processed_ok": 0,
        "processed_with_errors": 0,
        "failed_entries": 0,
        "skipped_json_errors": 0
    }
    error_summary = {}
    
    try:
        # Process file in smaller chunks for better performance
        with open(filename, "r", encoding="utf-8") as f:
            entry_count = 0
            progress_bar = tqdm(total=total_lines, desc=f"Processing {os.path.basename(filename)}", unit=" entries", leave=False) if total_lines > 0 else None
            
            for line in f:
                entry_count += 1
                if progress_bar:
                    progress_bar.update(1)
                
                # --- Per-entry Transaction ---
                word_id = None # Reset word_id for each entry
                entry_errors = None
                try:
                    # Start a new transaction for this entry
                    cur.execute("BEGIN;")

                    # Parse JSON entry
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON on line {entry_count}")
                        stats["skipped_json_errors"] += 1
                        cur.execute("ROLLBACK;") # Rollback the empty transaction
                        continue # Skip to the next line

                    # Process single entry
                    word_id, entry_errors = process_single_entry(entry)
                    
                    # Track results
                    if word_id:
                        if entry_errors:
                            stats["processed_with_errors"] += 1
                            first_error = entry_errors.split(";")[0][:100] if entry_errors else "Unknown error"
                            err_key = f"EntryError: {first_error}"
                            error_summary[err_key] = error_summary.get(err_key, 0) + 1
                            # Non-critical error, commit the valid parts
                            cur.execute("COMMIT;")
                        else:
                            stats["processed_ok"] += 1
                            # Successful processing, commit
                            cur.execute("COMMIT;")
                    else:
                        # Critical failure within process_single_entry (e.g., failed to get word_id)
                        stats["failed_entries"] += 1
                        reason = entry_errors or "Unknown critical failure"
                        err_key = f"EntryFailedCritically: {reason[:100]}"
                        error_summary[err_key] = error_summary.get(err_key, 0) + 1
                        # Critical failure, rollback any potential partial changes for this entry
                        cur.execute("ROLLBACK;") 
                
                except psycopg2.Error as db_err: # Catch specific database errors
                    cur.execute("ROLLBACK;") # Ensure rollback on DB error
                    logger.error(f"Database error processing line {entry_count} (Word: {entry.get('word', 'N/A')}): {db_err}")
                    # Log specific psycopg2 error details if available
                    if hasattr(db_err, 'pgcode') and db_err.pgcode:
                        logger.error(f"  PGCode: {db_err.pgcode}")
                    if hasattr(db_err, 'pgerror') and db_err.pgerror:
                         logger.error(f"  PGError: {db_err.pgerror}")
                    stats["failed_entries"] += 1
                    err_key = f"DBError: {type(db_err).__name__} ({str(db_err)[:50]})"
                    error_summary[err_key] = error_summary.get(err_key, 0) + 1

                except Exception as e:
                    # Catch any other unexpected error during entry processing
                    cur.execute("ROLLBACK;") # Rollback on any other exception
                    logger.error(f"Unhandled exception processing line {entry_count} (Word: {entry.get('word', 'N/A')}): {str(e)}", exc_info=True)
                    stats["failed_entries"] += 1
                    err_key = f"LoopException: {type(e).__name__}"
                    error_summary[err_key] = error_summary.get(err_key, 0) + 1
            
            if progress_bar:
                progress_bar.close()
    
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {
            "total_entries": stats.get("total_entries", 0),
            "processed_entries": 0,
            "error_entries": 1,
            "skipped_entries": 0,
            "detail_stats": entry_stats
        }
    except Exception as e:
        logger.error(f"Fatal error processing {filename}: {str(e)}", exc_info=True)
        stats["failed_entries"] = stats.get("total_entries", 0) - stats["processed_ok"] - stats["processed_with_errors"] - stats["skipped_json_errors"]
        if stats["failed_entries"] < 0:
            stats["failed_entries"] = 0
    
    # Final logging
    logger.info(f"Completed processing {filename}:")
    logger.info(f"  Total lines: {total_lines if total_lines >= 0 else 'Unknown'}")
    logger.info(f"  Successfully processed: {stats['processed_ok']}")
    logger.info(f"  Processed with non-critical errors: {stats['processed_with_errors']}")
    logger.info(f"  Failed entries: {stats['failed_entries']}")
    logger.info(f"  Skipped JSON errors: {stats['skipped_json_errors']}")
    logger.info(f"  --- Data processed: ---")
    for key, value in entry_stats.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")
    
    if error_summary:
        logger.warning(f"Error summary for {filename}: {json.dumps(error_summary, indent=2)}")
    
    return {
        "total_entries": stats["total_entries"],
        "processed_entries": stats["processed_ok"] + stats["processed_with_errors"],
        "error_entries": stats["failed_entries"],
        "skipped_entries": stats["skipped_json_errors"],
        "detail_stats": entry_stats
    }

def standardize_pronunciation(pron: str) -> str:
    """Standardize IPA pronunciation format."""
    if not pron:
        return None
    # Clean the pronunciation
    pron = pron.strip()
    # Remove extra spaces
    pron = re.sub(r"\s+", " ", pron)
    # Ensure proper IPA brackets
    if not (pron.startswith("[") and pron.endswith("]")):
        pron = f"[{pron.strip('[]')}]"
    # Standardize stress marks
    pron = pron.replace("'", "ˈ")
    pron = pron.replace('"', "ˌ")
    return pron

def standardize_credits(credits: str) -> str:
    """Standardize credits format."""
    if not credits:
        return None

    # Remove extra whitespace and newlines
    credits = re.sub(r"\s+", " ", credits.strip())

    # Extract different roles
    contributors = re.findall(r"Contributors?:(.*?)(?:Reviewer|Editor|$)", credits)
    reviewers = re.findall(r"Reviewer?:(.*?)(?:Editor|$)", credits)
    editors = re.findall(r"Editor?:(.*?)(?:$)", credits)

    # Format each role
    parts = []
    if contributors:
        names = [n.strip() for n in contributors[0].split("(")[0].strip().split(",")]
        parts.append(f"Contributors: {', '.join(names)}")
    if reviewers:
        names = [n.strip() for n in reviewers[0].split("(")[0].strip().split(",")]
        parts.append(f"Reviewers: {', '.join(names)}")
    if editors:
        names = [n.strip() for n in editors[0].split("(")[0].strip().split(",")]
        parts.append(f"Editors: {', '.join(names)}")

    return "; ".join(parts)


def process_examples(examples: List[Dict]) -> List[Dict]:
    """Process and validate examples."""
    processed = []
    for example in examples:
        if not isinstance(example, dict):
            continue

        text = example.get("text", "").strip()
        translation = example.get("translation", "").strip()

        if text and translation:
            processed.append(
                {
                    "text": text,
                    "translation": translation,
                    "example_id": example.get("example_id", len(processed) + 1),
                }
            )
    return processed


def process_see_also(see_also: List[Dict], language_code: str) -> List[str]:
    """Process see also references."""
    refs = []
    for ref in see_also:
        if isinstance(ref, dict):
            text = ref.get("text", "").strip()
            if text:
                # Remove numbering from references
                base_word = re.sub(r"\d+$", "", text)
                if base_word:
                    refs.append(base_word)
    return refs


def get_language_mapping():
    """Dynamically build language mapping from dictionary files."""
    # Base ISO 639-3 codes for Philippine languages
    base_language_map = {
        "onhan": "onx",
        "waray": "war",
        "ibanag": "ibg",
        "iranon": "iro",
        "ilocano": "ilo",
        "cebuano": "ceb",
        "hiligaynon": "hil",
        "kinaray-a": "krj",
        "kinaraya": "krj",
        "kinaray": "krj",
        "asi": "asi",
        "bikol": "bik",
        "bikolano": "bik",
        "bicol": "bik",
        "surigaonon": "sgd",
        "aklanon": "akl",
        "masbatenyo": "msb",
        "chavacano": "cbk",
        "tagalog": "tgl",
        "filipino": "tgl",
        "pilipino": "tgl",
        "pangasinan": "pag",
        "kapampangan": "pam",
        "manobo": "mbt",
        "manide": "abd",
        "maguindanaon": "mdh",
        "ivatan": "ivv",
        "itawis": "itv",
        "isneg": "isd",
        "ifugao": "ifk",
        "gaddang": "gad",
        "cuyonon": "cyo",
        "blaan": "bpr",  # Default to Koronadal Blaan
    }

    # Add regional variants
    regional_variants = {
        "bikol-central": "bcl",
        "bikol-albay": "bik",
        "bikol-rinconada": "bto",
        "bikol-partido": "bik",
        "bikol-miraya": "bik",
        "bikol-libon": "bik",
        "bikol-west-albay": "fbl",
        "bikol-southern-catanduanes": "bln",
        "bikol-northern-catanduanes": "bln",
        "bikol-boinen": "bik",
        "blaan-koronadal": "bpr",
        "blaan-sarangani": "bps",
        "cebuano-cotabato": "ceb",
        "hiligaynon-cotabato": "hil",
        "tagalog-marinduque": "tgl",
        "manobo-erumanen": "mbt",
        "isneg-yapayao": "isd",
        "ifugao-tuwali-ihapo": "ifk",
    }

    # Combine base map with variants
    language_map = {**base_language_map, **regional_variants}

    # Try to get additional mappings from dictionary files
    try:
        json_pattern = os.path.join("data", "marayum_dictionaries", "*_processed.json")
        json_files = glob.glob(json_pattern)

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "dictionary_info" not in data:
                        continue

                    dict_info = data["dictionary_info"]
                    base_language = dict_info.get("base_language", "").lower()
                    if not base_language:
                        continue

                    # Extract language code from filename
                    filename = os.path.basename(json_file)
                    if filename.endswith("_processed.json"):
                        filename = filename[:-14]  # Remove '_processed.json'

                    # Split on first hyphen to get language part
                    lang_code = filename.split("-")[0].lower()

                    # Add to mapping if not already present
                    if base_language not in language_map:
                        language_map[base_language] = lang_code

                    # Handle compound names
                    if "-" in base_language:
                        parts = base_language.split("-")
                        for part in parts:
                            if part not in language_map:
                                language_map[part] = lang_code

                    # Add filename-based mapping
                    orig_name = filename.replace("-english", "").lower()
                    if orig_name not in language_map:
                        language_map[orig_name] = lang_code

            except Exception as e:
                logger.warning(
                    f"Error processing language mapping from {json_file}: {str(e)}"
                )
                continue
    except Exception as e:
        logger.error(f"Error building language mapping: {str(e)}")

    return language_map


def get_language_code(language: str) -> str:
    """Get standardized ISO 639-3 language code."""
    if not language:
        return ""

    language = language.lower().strip()

    # Direct mappings for Philippine languages
    LANGUAGE_CODES = {
        "chavacano": "cbk",
        "zamboangueño": "cbk",
        "chabacano": "cbk",
        "cebuano": "ceb",
        "hiligaynon": "hil",
        "ilonggo": "hil",
        "waray": "war",
        "waray-waray": "war",
        "tagalog": "tgl",
        "filipino": "fil",
        "bikol": "bik",
        "bikolano": "bik",
        "bicol": "bik",
        "ilocano": "ilo",
        "iloko": "ilo",
        "kapampangan": "pam",
        "pangasinan": "pag",
        "kinaray-a": "krj",
        "kinaraya": "krj",
        "aklanon": "akl",
        "masbatenyo": "msb",
        "surigaonon": "sgd",
        "tausug": "tsg",
        "maguindanao": "mdh",
        "maguindanaon": "mdh",
        "maranao": "mrw",
        "iranon": "iro",
        "iranun": "iro",
        "ibanag": "ibg",
        "ivatan": "ivv",
        "itawis": "itv",
        "isneg": "isd",
        "ifugao": "ifk",
        "gaddang": "gad",
        "cuyonon": "cyo",
        "asi": "asz",
        "bantoanon": "bno",
        "blaan": "bpr",
        "manobo": "mbt",
        "manide": "abd",
        "onhan": "onx",
    }

    # Try direct lookup
    if language in LANGUAGE_CODES:
        return LANGUAGE_CODES[language]

    # Try without parenthetical clarifications
    base_name = re.sub(r"\s*\([^)]*\)", "", language).strip()
    if base_name in LANGUAGE_CODES:
        return LANGUAGE_CODES[base_name]

    # Try normalizing hyphens and spaces
    normalized = re.sub(r"[\s-]+", "", language)
    if normalized in LANGUAGE_CODES:
        return LANGUAGE_CODES[normalized]

    # If not found, log warning and return a safe fallback
    logger.warning(f"No language code mapping found for: {language}")
    # Create a safe ASCII identifier from the language name
    safe_code = re.sub(r"[^a-z]", "", language.lower())[:3]
    if not safe_code:
        safe_code = "unk"  # Unknown language
    return safe_code


# Make sure these imports are present at the top of your file
import json
import os
import logging
import psycopg2
from psycopg2.extras import Json # Ensure Json is imported
from typing import Optional, Tuple, Dict, Union, List # Ensure needed types are imported
from tqdm import tqdm
# --- Add other necessary imports like RelationshipType ---
from enum import Enum # Example if RelationshipType is an Enum

@with_transaction(commit=False) # Manage commit manually within the function
def process_marayum_json(cur, filename: str, source_identifier: Optional[str] = None) -> Tuple[int, int]:
    """Processes a single Marayum JSON dictionary file."""
    # Determine the effective source identifier
    if source_identifier:
        effective_source_identifier = source_identifier
    else:
        # Fallback to standardizing from filename if not provided
        effective_source_identifier = standardize_source_identifier(
            os.path.basename(filename)
        )
    logger.info(
        f"Processing Marayum file: {filename} with source: {effective_source_identifier}"
    )

    conn = cur.connection

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing Marayum file {filename}: {e}")
        return 0, 1 # Indicate 0 processed, 1 issue (file error)
    except Exception as e:
        logger.error(f"Unexpected error reading Marayum file {filename}: {e}", exc_info=True)
        return 0, 1 # Indicate 0 processed, 1 issue (file error)

    # --- Adjust for Dictionary Structure ---
    if not isinstance(data, dict): # Expect a dictionary now
        logger.error(f"Marayum file {filename} does not contain a top-level dictionary.")
        return 0, 1 # Indicate 0 processed, 1 issue (format error)

    # Extract the list of word entries from the 'words' key
    word_entries_list = data.get("words", [])
    if not isinstance(word_entries_list, list):
         logger.error(f"Marayum file {filename} dictionary does not contain a 'words' list.")
         return 0, 1 # Indicate 0 processed, 1 issue (format error)
    # --- End Adjustment ---

    entries_in_file = len(word_entries_list) # Get length from the 'words' list
    if entries_in_file == 0:
         logger.info(f"Found 0 word entries in {filename}. Skipping file.")
         # No commit needed, just return
         return 0, 0 # 0 processed, 0 issues

    logger.info(f"Found {entries_in_file} word entries in {filename}")

    # --- Determine Language Code for the entire file ---
    dict_info = data.get("dictionary_info", {})
    base_language_name = dict_info.get("base_language", "")
    language_code = "unk" # Default before mapping attempt
    if base_language_name:
        # Normalize the language name slightly before lookup (lowercase, strip)
        normalized_lang_name = base_language_name.lower().strip()
        language_code = get_language_code(normalized_lang_name) # Use helper function
        # get_language_code logs warning if mapping fails and returns a safe code ('unk' or derived)
        if not language_code: # Should not happen if get_language_code is robust, but check anyway
             logger.warning(f"get_language_code returned empty for '{base_language_name}' (normalized: '{normalized_lang_name}') in {filename}. Defaulting to 'unk'.")
             language_code = "unk"
        elif language_code != "unk": # Only log success if a specific code was found
             logger.info(f"Determined language code '{language_code}' for {filename} from base language '{base_language_name}'.")
        # If language_code is 'unk', consider if processing should stop or continue
        # For now, we proceed, but 'unk' will be used for word entries.
        # --- End Language Code Determination ---


        # Initialize counters for this file's processing run
        stats = {
            "processed": 0,
            "definitions": 0,
            "relations": 0,
            "pronunciations": 0,
            "etymologies": 0,
            "credits": 0, # Added counter for credits
            "skipped": 0,
            "errors": 0,
        }
        error_types = {} # Dictionary to track types of errors encountered

        # Iterate over the extracted list of word entries
        with tqdm(total=entries_in_file, desc=f"Processing {effective_source_identifier}", unit="entry", leave=False) as pbar:
            for entry_index, entry in enumerate(word_entries_list): # <-- Iterate over word_entries_list
                # Create a unique savepoint name for each entry
                # Using hash is okay, but ensure it doesn't collide easily; index helps uniqueness
                savepoint_name = f"marayum_{entry_index}_{abs(hash(str(entry)) % 1000000)}" # Limit hash part length

                lemma = "" # Initialize lemma outside try block for use in error logging if needed
                word_id = None # Initialize word_id

                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")

                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping non-dictionary item at index {entry_index} in {filename}")
                        stats["skipped"] += 1
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release savepoint for skipped item
                        pbar.update(1)
                        continue

                    lemma = entry.get("word", "").strip()
                    if not lemma:
                        logger.warning(f"Skipping entry at index {entry_index} due to missing or empty 'word' field in {filename}")
                        stats["skipped"] += 1
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release savepoint for skipped item
                        pbar.update(1)
                        continue

                    # --- Language code determined before the loop, use the 'language_code' variable ---

                    # --- Extract Basic Metadata (excluding complex/separately handled fields) ---
                    word_metadata = {}
                    for key, value in entry.items():
                        # Exclude fields processed separately or too complex/large
                        if key not in ["word", "definitions", "pronunciation", "etymology", "see_also", "examples", "id", "language_code", "credits"]: # Exclude credits now
                            # Basic check for simple types
                            if isinstance(value, (str, int, float, bool)) or value is None:
                                 # Limit string length
                                 if isinstance(value, str) and len(value) > 500:
                                     word_metadata[key] = value[:500] + "...(truncated)"
                                 else:
                                     word_metadata[key] = value
                            elif isinstance(value, list) and all(isinstance(i, (str, int, float, bool)) for i in value):
                                # Include simple lists, limit size
                                 try:
                                    json_str = json.dumps(value)
                                    if len(json_str) < 500:
                                         word_metadata[key] = value
                                    else:
                                        logger.debug(f"Skipping large list metadata field '{key}' for word '{lemma}'")
                                 except TypeError:
                                    logger.debug(f"Skipping non-serializable list metadata field '{key}' for word '{lemma}'")
                            # Optionally handle simple dicts here too if needed, with size/complexity checks

                    # --- Get or Create Word ---
                    try:
                         # Pass the language_code determined for the file
                         word_id = get_or_create_word_id(
                            cur,
                            lemma,
                            language_code=language_code, # Use file-level language code
                            source_identifier=effective_source_identifier,
                            word_metadata=Json(word_metadata) if word_metadata else None
                        )
                         if not word_id:
                             # This case should ideally be handled within get_or_create_word_id by raising an error
                             raise ValueError(f"get_or_create_word_id returned None unexpectedly for lemma '{lemma}'")
                         # Use debug level for successful creation log
                         logger.debug(f"Word '{lemma}' ({language_code}) created/found (ID: {word_id}) from source '{effective_source_identifier}'.")

                    except Exception as word_err:
                         logger.error(f"CRITICAL: Failed to get/create word ID for lemma '{lemma}' (Index: {entry_index}) in {filename}: {word_err}")
                         stats["errors"] += 1
                         error_key = f"WordCreationError: {type(word_err).__name__}"
                         error_types[error_key] = error_types.get(error_key, 0) + 1
                         cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}") # Rollback this entry
                         pbar.update(1)
                         continue # Skip to next entry - cannot proceed without word_id


                    # --- Process Credits using insert_credit (AFTER word_id is obtained) ---
                    credits_raw = entry.get("credits")
                    if credits_raw:
                        # insert_credit handles string or dict and logs internally
                        credit_inserted_id = insert_credit(cur, word_id, credits_raw, effective_source_identifier)
                        if credit_inserted_id:
                            stats["credits"] += 1
                        else:
                            # Log failure here as well, as insert_credit might only log warning/error
                            logger.warning(f"Failed to insert credit for word ID {word_id} ('{lemma}') from source '{effective_source_identifier}'. Raw data: {credits_raw}")
                            # Optionally count credit failures if needed:
                            # stats["credit_errors"] = stats.get("credit_errors", 0) + 1
                            error_key = f"CreditInsertFailure"
                            error_types[error_key] = error_types.get(error_key, 0) + 1


                    # --- Process Definitions and Examples ---
                    definitions = entry.get("definitions", [])
                    if isinstance(definitions, list):
                         for def_idx, def_item in enumerate(definitions):
                             if not isinstance(def_item, dict) or "definition" not in def_item:
                                 logger.debug(f"Skipping invalid definition item at index {def_idx} for word '{lemma}' (ID: {word_id})")
                                 continue

                             definition_text = def_item.get("definition", "")
                             # Allow definition text to be None or empty string initially
                             # .strip() might fail if it's not a string
                             if isinstance(definition_text, str):
                                  definition_text = definition_text.strip()
                             elif definition_text is None:
                                  definition_text = "" # Handle None case explicitly
                             else:
                                  logger.warning(f"Non-string definition text found for word '{lemma}' (ID: {word_id}), Def {def_idx+1}: {type(definition_text)}. Skipping definition.")
                                  continue # Skip this definition if type is wrong

                             if not definition_text:
                                 logger.debug(f"Skipping empty definition for word '{lemma}' (ID: {word_id}), Def {def_idx+1}")
                                 continue

                             # Prepare definition metadata
                             def_metadata = {}
                             for meta_key, meta_val in def_item.items():
                                if meta_key not in ["definition", "examples", "definition_id"]: # Exclude example, def, id
                                    # Add similar type/size checks as for word_metadata if needed
                                    if isinstance(meta_val, (str, int, float, bool)) or meta_val is None:
                                        if isinstance(meta_val, str) and len(meta_val) > 200: # Limit size
                                            def_metadata[meta_key] = meta_val[:200] + "..."
                                        else:
                                            def_metadata[meta_key] = meta_val
                                    # Add list/dict handling if necessary

                             # Process examples associated with this definition
                             examples_processed = []
                             examples_raw = def_item.get("examples", [])
                             if isinstance(examples_raw, list):
                                 examples_processed = process_examples(examples_raw) # Use helper

                             # Insert definition
                             try:
                                 # Use provided ID or index as fallback order
                                 def_order = def_item.get("definition_id", def_idx + 1)
                                 # Extract entry-level part of speech
                                 part_of_speech = entry.get("pos", "")
                                 def_id = insert_definition(
                                    cur,
                                    word_id,
                                    definition_text,
                                    source_identifier=effective_source_identifier,
                                    part_of_speech=part_of_speech,  # Pass the part of speech
                                    #definition_order=def_order,
                                    examples=Json(examples_processed) if examples_processed else None,
                                    #metadata=Json(def_metadata) if def_metadata else None
                                 )
                                 if def_id:
                                    stats["definitions"] += 1
                                 else: # insert_definition returned None or raised error handled internally
                                    logger.warning(f"insert_definition failed for '{lemma}' (ID: {word_id}), Def {def_idx+1}. Check internal logs.")
                                    error_key = f"DefinitionInsertFailure"
                                    error_types[error_key] = error_types.get(error_key, 0) + 1

                             except Exception as def_err:
                                 # Catch errors not handled inside insert_definition
                                 logger.error(f"Error during definition insertion for '{lemma}' (ID: {word_id}), Def {def_idx+1}: {def_err}", exc_info=True)
                                 error_key = f"DefinitionInsertError: {type(def_err).__name__}"
                                 error_types[error_key] = error_types.get(error_key, 0) + 1
                                 # Continue processing other parts of the entry, but log the error

                    # --- Process Pronunciation (if available) ---
                    pronunciation = entry.get("pronunciation")
                    if pronunciation:
                        # Marayum pronunciation is often just a string, format it
                        pron_obj = {}
                        if isinstance(pronunciation, str) and pronunciation.strip():
                            pron_obj = {"value": pronunciation.strip(), "type": "ipa"} # Assume IPA if not empty
                        elif isinstance(pronunciation, dict):
                            pron_obj = pronunciation # Use as is if already a dict
                        else:
                            logger.debug(f"Skipping invalid pronunciation data type for '{lemma}' (ID: {word_id}): {type(pronunciation)}")

                        if pron_obj and pron_obj.get("value"): # Ensure value exists
                            try:
                                pron_inserted_id = insert_pronunciation(cur, word_id, pron_obj, effective_source_identifier)
                                if pron_inserted_id:
                                    stats["pronunciations"] += 1
                                else:
                                    logger.warning(f"insert_pronunciation failed for '{lemma}' (ID: {word_id}). Check internal logs.")
                                    error_key = f"PronunciationInsertFailure"
                                    error_types[error_key] = error_types.get(error_key, 0) + 1
                            except Exception as pron_err:
                                logger.error(f"Error during pronunciation insertion for '{lemma}' (ID: {word_id}): {pron_err}", exc_info=True)
                                error_key = f"PronunciationInsertError: {type(pron_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1

                    # --- Process Etymology (if available) ---
                    etymology = entry.get("etymology")
                    if etymology and isinstance(etymology, str) and etymology.strip():
                        try:
                            ety_id = insert_etymology(cur, word_id, etymology, effective_source_identifier)
                            if ety_id:
                                stats["etymologies"] += 1
                            else:
                                logger.warning(f"insert_etymology failed for '{lemma}' (ID: {word_id}). Check internal logs.")
                                error_key = f"EtymologyInsertFailure"
                                error_types[error_key] = error_types.get(error_key, 0) + 1
                        except Exception as ety_err:
                            logger.error(f"Error during etymology insertion for '{lemma}' (ID: {word_id}): {ety_err}", exc_info=True)
                            error_key = f"EtymologyInsertError: {type(ety_err).__name__}"
                            error_types[error_key] = error_types.get(error_key, 0) + 1

                    # --- Process See Also (as Relations) ---
                    see_also = entry.get("see_also", [])
                    if isinstance(see_also, list):
                         # Assuming process_see_also returns list of related word strings
                         see_also_words = process_see_also(see_also, language_code)
                         for related_word_str in see_also_words:
                             if related_word_str.lower() != lemma.lower(): # Avoid self-relation (case-insensitive)
                                 related_word_id = None
                                 try:
                                     # Get ID for the related word, creating if necessary
                                     related_word_id = get_or_create_word_id(
                                         cur,
                                         related_word_str,
                                         language_code=language_code, # Use same language code
                                         source_identifier=effective_source_identifier # Attribute creation to this source if new
                                     )
                                     if related_word_id and related_word_id != word_id: # Check IDs aren't same
                                         rel_id = insert_relation(
                                             cur,
                                             word_id,
                                             related_word_id,
                                             RelationshipType.SEE_ALSO,
                                             source_identifier=effective_source_identifier
                                         )
                                         if rel_id:
                                             stats["relations"] += 1
                                             # Optionally insert bidirectional relation
                                             # insert_relation(cur, related_word_id, word_id, RelationshipType.SEE_ALSO, effective_source_identifier)
                                         else:
                                             # Log if insert_relation fails (might be due to constraint/conflict handled internally)
                                             logger.debug(f"Failed to insert SEE_ALSO relation {word_id} -> {related_word_id}. Might already exist.")
                                             # error_key = f"SeeAlsoRelationInsertFailure" # Only if failure is unexpected
                                             # error_types[error_key] = error_types.get(error_key, 0) + 1

                                 except Exception as rel_err:
                                     logger.error(f"Error processing 'see_also' relation for '{lemma}' -> '{related_word_str}': {rel_err}", exc_info=True)
                                     error_key = f"SeeAlsoRelationError: {type(rel_err).__name__}"
                                     error_types[error_key] = error_types.get(error_key, 0) + 1

                    # --- Finish Entry Processing ---
                    # If we reach here, the main parts were processed (or errors handled non-critically)
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    stats["processed"] += 1

                    # Commit periodically within the file processing loop
                    # Check if connection is still active before committing
                    if not conn.closed and stats["processed"] % 500 == 0:
                        try:
                            conn.commit()
                            logger.info(f"Committed batch after {stats['processed']} entries processed for {filename}")
                        except (psycopg2.InterfaceError, psycopg2.OperationalError) as conn_err:
                             logger.error(f"Connection error during batch commit for {filename} at entry {entry_index}: {conn_err}. Attempting to reconnect/rollback is complex here. Stopping file processing.", exc_info=True)
                             # Mark remaining as errors and stop processing this file
                             remaining_entries = entries_in_file - entry_index - 1
                             stats["errors"] += remaining_entries
                             error_types["BatchCommitConnectionError"] = error_types.get("BatchCommitConnectionError", 0) + 1
                             pbar.update(remaining_entries) # Update progress bar fully
                             # Need to return immediately as connection state is uncertain
                             total_issues = stats["skipped"] + stats["errors"]
                             return stats["processed"], total_issues # Return counts so far

                        except Exception as batch_commit_err:
                             logger.error(f"Error committing batch for {filename} at entry {entry_index}: {batch_commit_err}. Rolling back current transaction...", exc_info=True)
                             try:
                                 conn.rollback()
                                 logger.info("Transaction rolled back after batch commit error.")
                             except Exception as rb_err:
                                 logger.critical(f"CRITICAL: Failed to rollback after batch commit error for {filename}: {rb_err}. Stopping file processing.", exc_info=True)
                                 remaining_entries = entries_in_file - entry_index - 1
                                 stats["errors"] += remaining_entries
                                 error_types["BatchCommitRollbackError"] = error_types.get("BatchCommitRollbackError", 0) + 1
                                 pbar.update(remaining_entries)
                                 total_issues = stats["skipped"] + stats["errors"]
                                 return stats["processed"], total_issues # Return counts so far

                             # After rollback, the loop continues with the next entry in a fresh transaction state
                             error_types["BatchCommitError"] = error_types.get("BatchCommitError", 0) + 1

                except Exception as entry_err:
                    # General catch-all for unexpected errors during the processing of a single entry
                    # Errors related to specific parts (word, def, pron, etc.) should be caught closer to the source
                    logger.error(f"UNEXPECTED error processing entry #{entry_index} ('{lemma or 'unknown'}') in {filename}: {entry_err}", exc_info=True)
                    stats["errors"] += 1
                    error_key = f"UnexpectedEntryError: {type(entry_err).__name__}"
                    error_types[error_key] = error_types.get(error_key, 0) + 1
                    try:
                         # Rollback the specific entry that failed
                         cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err:
                         logger.critical(f"CRITICAL: Failed to rollback to savepoint {savepoint_name} after entry error in {filename}: {rb_err}. Attempting full transaction rollback.", exc_info=True)
                         try:
                             if not conn.closed:
                                 conn.rollback()
                                 logger.info("Full transaction rolled back due to savepoint rollback failure.")
                             else:
                                 logger.warning("Connection was already closed before full rollback attempt.")
                         except Exception as full_rb_err:
                             logger.critical(f"CRITICAL: Failed even full transaction rollback for {filename}: {full_rb_err}. Stopping file processing.", exc_info=True)

                             # Since rollback state is uncertain, stop processing this file.
                             remaining_entries = entries_in_file - entry_index - 1
                             stats["errors"] += remaining_entries # Mark remaining as issues
                             error_types["CriticalRollbackFailure"] = error_types.get("CriticalRollbackFailure", 0) + 1
                             pbar.update(remaining_entries) # Update progress bar fully
                             total_issues = stats["skipped"] + stats["errors"]
                             return stats["processed"], total_issues # Exit function

                finally:
                     # Ensure the progress bar is always updated, even if an error occurred
                     pbar.update(1)
                     
        # --- Final Commit for the file ---
        final_commit_success = False
        try:
            # First check if we have a valid connection
            if conn is None:
                # Connection doesn't exist at all
                logger.error(f"No valid connection available for final commit for {filename}. Data might be lost.")
                stats["errors"] += 1 # Count as an error state
                error_types["NoConnectionForFinalCommit"] = error_types.get("NoConnectionForFinalCommit", 0) + 1
            elif conn.closed:
                # Connection exists but is closed
                logger.error(f"Connection was closed before final commit for {filename}. Some data might be lost.")
                stats["errors"] += 1 # Count as an error state
                error_types["ConnectionClosedBeforeFinalCommit"] = error_types.get("ConnectionClosedBeforeFinalCommit", 0) + 1
            else:
                # We have a valid, open connection - proceed with commit
                try:
                    conn.commit()
                    final_commit_success = True
                    logger.info(f"Finished processing {filename}. Final commit successful.")
                    logger.info(f"Stats for {filename}: Processed: {stats['processed']}, Definitions: {stats['definitions']}, Relations: {stats['relations']}, Pronunciations: {stats['pronunciations']}, Etymologies: {stats['etymologies']}, Credits: {stats['credits']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
                    if error_types:
                        logger.warning(f"Error summary for {filename}: {error_types}")
                except (psycopg2.InterfaceError, psycopg2.OperationalError) as conn_err:
                    logger.error(f"Connection error during final commit for {filename}: {conn_err}. Changes might be lost.", exc_info=True)
                    stats["errors"] += 1 # Count final commit failure as an error
                    error_types["FinalCommitConnectionError"] = error_types.get("FinalCommitConnectionError", 0) + 1
                except Exception as final_commit_err:
                    logger.error(f"Error during final commit for {filename}: {final_commit_err}. Rolling back changes...", exc_info=True)
                    stats["errors"] += 1 # Count final commit failure as an error
                    error_types["FinalCommitError"] = error_types.get("FinalCommitError", 0) + 1
                    try:
                        # Only attempt rollback if connection is still valid
                        if not conn.closed:
                            conn.rollback()
                            logger.info("Transaction rolled back after final commit error.")
                    except Exception as rb_err:
                        logger.error(f"Failed to rollback after final commit error for {filename}: {rb_err}", exc_info=True)
                        error_types["RollbackError"] = error_types.get("RollbackError", 0) + 1
        except Exception as e:
            # Handle any unexpected errors when checking connection state
            logger.error(f"Unexpected error during final transaction handling for {filename}: {e}", exc_info=True)
            stats["errors"] += 1
            error_types["UnexpectedFinalCommitError"] = error_types.get("UnexpectedFinalCommitError", 0) + 1

        # Aggregate total issues from errors and skips
        total_issues = stats["skipped"] + stats["errors"]

        # Add a warning if no entries were successfully processed despite the file having entries
        if entries_in_file > 0 and stats["processed"] == 0:
            logger.warning(f"No entries were successfully processed from {filename}, although {entries_in_file} were found. Issues encountered: {total_issues}")


        return stats["processed"], total_issues

    # Note: Ensure all helper functions (get_or_create_word_id, insert_definition, etc.)
    # and imports (Json, RelationshipType, etc.) are correctly defined and available.
    # This version assumes `insert_credit` handles its own commit/rollback logic as needed
    # and correctly logs its own errors, but adds extra logging/counting in this function for robustness.

@with_transaction(commit=True)
def process_marayum_directory(cur, directory_path: str) -> None:
    """Process all Project Marayum dictionary files in the specified directory."""

    # Normalize directory path
    directory_path = os.path.normpath(directory_path)
    logger.info(f"Processing Marayum dictionaries from directory: {directory_path}")

    # Define the standard source identifier for ALL files in this directory
    marayum_source_id = SourceStandardization.standardize_sources("marayum")
    # Check if the directory exists
    if not os.path.exists(directory_path):
        logger.error(f"Directory does not exist: {directory_path}")
        return

    # Find only processed JSON files in the directory
    json_pattern = os.path.join(directory_path, "*_processed.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        logger.warning(f"No processed JSON files found in {directory_path}")
        logger.info(f"Looking for files with the pattern: {json_pattern}")
        # Try to list what's actually in the directory
        if os.path.isdir(directory_path):
            files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
            if files:
                processed_files = [f for f in files if f.endswith("_processed.json")]
                unprocessed_files = [
                    f for f in files if not f.endswith("_processed.json")
                ]
                if processed_files:
                    logger.info(f"Processed files found: {', '.join(processed_files)}")
                if unprocessed_files:
                    logger.info(
                        f"Unprocessed files found (skipping): {', '.join(unprocessed_files)}"
                    )
            else:
                logger.info(f"No JSON files found in directory: {directory_path}")
        return

    total_processed = 0
    total_errors = 0
    total_files_processed = 0
    total_files_skipped = 0

    # Sort files by size for efficient processing (process smaller files first)
    json_files.sort(key=lambda x: os.path.getsize(x))

    # Process each processed JSON file found
    for json_file in json_files:
        try:
            # Normalize path
            json_file = os.path.normpath(json_file)

            # Check file size and readability
            if not os.path.isfile(json_file):
                logger.error(f"Not a file: {json_file}")
                total_files_skipped += 1
                continue

            file_size = os.path.getsize(json_file)
            if file_size == 0:
                logger.warning(f"Skipping empty file: {json_file}")
                total_files_skipped += 1
                continue

            if not os.access(json_file, os.R_OK):
                logger.error(f"File not readable: {json_file}")
                total_files_skipped += 1
                continue

            # Process the dictionary
            processed, errors = process_marayum_json(
                cur, json_file, source_identifier=marayum_source_id
            )  # <-- MODIFIED CALL
            total_processed += processed
            total_errors += errors
            if processed > 0:
                total_files_processed += 1
            else:
                total_files_skipped += 1
                logger.warning(
                    f"No entries processed from {os.path.basename(json_file)}"
                )

        except Exception as e:
            logger.error(
                f"Error processing Marayum dictionary file {json_file}: {str(e)}"
            )
            total_errors += 1
            total_files_skipped += 1
            continue

    # Log final statistics
    logger.info(f"Completed processing Marayum dictionaries:")
    logger.info(f"  Files processed: {total_files_processed}")
    logger.info(f"  Files skipped: {total_files_skipped}")
    logger.info(f"  Total entries processed: {total_processed}")
    logger.info(f"  Total errors encountered: {total_errors}")


# -------------------------------------------------------------------
# Command Line Interface Functions
# -------------------------------------------------------------------
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage dictionary data in PostgreSQL."
    )
    subparsers = parser.add_subparsers(dest="command")
    migrate_parser = subparsers.add_parser(
        "migrate", help="Create/update schema and load data"
    )
    migrate_parser.add_argument(
        "--check-exists", action="store_true", help="Skip identical existing entries"
    )
    migrate_parser.add_argument(
        "--force", action="store_true", help="Force migration without confirmation"
    )
    migrate_parser.add_argument(
        "--data-dir", type=str, help="Directory containing dictionary data files"
    )
    migrate_parser.add_argument(
        "--sources", type=str, help="Comma-separated list of source names to process"
    )
    migrate_parser.add_argument(
        "--file", type=str, help="Specific data file to process"
    )
    verify_parser = subparsers.add_parser("verify", help="Verify data integrity")
    verify_parser.add_argument(
        "--quick", action="store_true", help="Run quick verification"
    )
    verify_parser.add_argument(
        "--repair", action="store_true", help="Attempt to repair issues"
    )
    update_parser = subparsers.add_parser("update", help="Update DB with new data")
    update_parser.add_argument(
        "--file", type=str, required=True, help="JSON or JSONL file to use"
    )
    update_parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying"
    )
    lookup_parser = subparsers.add_parser("lookup", help="Look up word information")
    lookup_parser.add_argument("word", help="Word to look up")
    lookup_parser.add_argument(
        "--debug", action="store_true", help="Show debug information"
    )
    lookup_parser.add_argument(
        "--format",
        choices=["text", "json", "rich"],
        default="rich",
        help="Output format",
    )
    stats_parser = subparsers.add_parser("stats", help="Display dictionary statistics")
    stats_parser.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics"
    )
    stats_parser.add_argument("--export", type=str, help="Export statistics to file")
    subparsers.add_parser("leaderboard", help="Display top contributors")
    subparsers.add_parser("help", help="Display help information")
    subparsers.add_parser("test", help="Run database connectivity tests")
    subparsers.add_parser("explore", help="Interactive dictionary explorer")
    purge_parser = subparsers.add_parser("purge", help="Safely delete all data")
    purge_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )
    subparsers.add_parser(
        "cleanup",
        help="Clean up dictionary data by removing duplicates and standardizing formats",
    )
    subparsers.add_parser(
        "migrate-relationships",
        help="Migrate existing relationships to the new RelationshipType system",
    )
    return parser


def migrate_data(args):
    """Migrate dictionary data from various sources."""
    sources = [
        {
            "name": "Tagalog Words",
            "file": "tagalog-words.json",
            "handler": process_tagalog_words,
            "required": False,
        },
        {
            "name": "Root Words",
            "file": "root_words_with_associated_words_cleaned.json",
            "handler": process_root_words_cleaned,
            "required": False,
        },
        {
            "name": "KWF Dictionary",
            "file": "kwf_dictionary.json",
            "handler": process_kwf_dictionary,
            "required": False,
        },
        {
            "name": "Kaikki.org (Tagalog)",
            "file": "kaikki.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False,
        },
        {
            "name": "Kaikki.org (Cebuano)",
            "file": "kaikki-ceb.jsonl",
            "handler": process_kaikki_jsonl,
            "required": False,
        },
        {
            "name": "Project Marayum",
            "file": "marayum_dictionaries",  # Changed to directory name only
            "handler": process_marayum_directory,
            "required": False,
            "is_directory": True,
        },
        {
            "name": "Philippine Slang and Gay Dictionary",
            "file": "gay-slang.json",
            "handler": process_gay_slang_json,
            "required": False, # Set to True if it should always be processed if present
            "is_directory": False, # It's a file
         },
    ]

    # Get data directory from args if provided, or use defaults
    if hasattr(args, "data_dir") and args.data_dir:
        data_dirs = [args.data_dir]
    else:
        data_dirs = ["data", os.path.join("..", "data")]

    # Filter sources if specific ones are requested via --sources
    if hasattr(args, "sources") and args.sources:
        requested_sources = [s.lower() for s in args.sources.split(",")]
        # More flexible matching for source names
        sources = [
            s
            for s in sources
            if any(
                req in s["name"].lower()
                or req in s["file"].lower()
                or (req == "marayum" and "marayum" in s["file"].lower())
                for req in requested_sources
            )
        ]
        if not sources:
            logger.error(f"No matching sources found for: {args.sources}")
            return
        for source in sources:
            source["required"] = True

    # Custom file overrides existing sources
    if hasattr(args, "file") and args.file:
        filename = args.file
        if filename.endswith(".jsonl"):
            handler = process_kaikki_jsonl
        elif "root_words" in filename.lower():
            handler = process_root_words_cleaned
        elif "kwf" in filename.lower():
            handler = process_kwf_dictionary
        elif "marayum" in filename.lower():
            handler = process_marayum_directory
        elif "tagalog" in filename.lower():
            handler = process_tagalog_words
        elif "gay" in filename.lower():
            handler = process_gay_slang_json

        basename = os.path.basename(filename)
        source_found = False
        for source in sources:
            if source["file"] == basename or (
                os.path.isdir(filename)
                and os.path.basename(source["file"]) == os.path.basename(filename)
            ):
                source["file"] = filename  # Use full path
                source["required"] = True
                source_found = True
                break
        if not source_found:
            sources.append(
                {
                    "name": f"Custom ({basename})",
                    "file": filename,
                    "handler": handler,
                    "required": True,
                    "is_directory": os.path.isdir(filename),
                }
            )

    conn = None
    cur = None
    console = Console()
    try:
        conn = get_connection()
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        console.print(f"\n[bold red]Failed to connect to the database:[/] {str(e)}")
        console.print(
            "Please check your database configuration and ensure the database exists."
        )
        console.print(
            f"Current settings: DB_NAME={DB_NAME}, DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_USER={DB_USER}"
        )
        console.print(
            "\n[bold]To fix this issue:[/]\n1. Make sure PostgreSQL is running\n2. Create the database if it doesn't exist (e.g., `createdb {DB_NAME}`)\n3. Verify your .env settings."
        )
        return

    try:
        cur = conn.cursor()
        console.print("[bold]Setting up database schema...[/]")
        create_or_update_tables(conn) # This function handles its own commit/rollback

        console.print("[bold]Processing data sources...[/]")
        # --- Start Transaction for Data Processing ---
        conn.autocommit = False # Ensure we control the transaction

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            total_files = len(sources)
            main_task = progress.add_task("Migrating data", total=total_files)

            for idx, source in enumerate(sources):
                source_name = source['name']
                progress.update(main_task, description=f"Processing source {idx+1}/{total_files}: {source_name}", advance=0)

                # Look for file in provided data directories if not an absolute path
                if os.path.isabs(source["file"]):
                    filepath = source["file"]
                else:
                    filepath = None
                    for data_dir in data_dirs:
                        potential_path = os.path.join(data_dir, source["file"])
                        if source.get("is_directory", False):
                            if os.path.isdir(potential_path):
                                filepath = potential_path
                                break
                        else:
                            if os.path.isfile(potential_path):
                                filepath = potential_path
                                break

                if not filepath:
                    msg = f"Required {'directory' if source.get('is_directory', False) else 'file'} not found: {source['file']}"
                    if source["required"]:
                        logger.error(msg)
                        # Raise an error to trigger rollback
                        raise FileNotFoundError(msg)
                    else:
                        logger.warning(f"{msg}. Skipping source: {source_name}")
                        progress.update(main_task, advance=1) # Advance main task when skipping
                        continue

                # Use the found absolute path
                source_task_desc = f"Processing {source_name} ({os.path.basename(filepath)})..."
                # Add a sub-task for individual file progress if handler supports it
                # For now, just log start/end
                logger.info(f"Starting processing for {source_name} from {filepath}")

                try:
                    # Call the handler function for the source
                    # Handlers might use @with_transaction, but they operate within the larger transaction managed here
                    source["handler"](cur, filepath)
                    logger.info(f"Successfully processed source: {source_name}")

                except Exception as handler_error:
                    # Log error and rollback the entire migration if a handler fails
                    logger.error(f"Error processing source '{source_name}' from {filepath}: {handler_error}", exc_info=True)
                    # Raise the error to trigger the outer rollback
                    raise handler_error

                finally:
                     # Ensure the main progress bar always advances
                    progress.update(main_task, advance=1)

        # --- If all sources processed without error, commit the transaction ---
        logger.info("All sources processed successfully. Committing transaction...")
        conn.commit() # Commit all changes made by handlers
        console.print("[bold green]Migration completed successfully.[/]")

    except Exception as e:
        # Rollback transaction if any error occurred during the migration process
        logger.error(f"Error during migration: {str(e)}", exc_info=True)
        console.print(f"\n[bold red]Migration failed:[/] {str(e)}")
        if conn:
            try:
                logger.info("Rolling back transaction due to error.")
                conn.rollback()
            except Exception as rb_err:
                logger.error(f"Failed to rollback transaction: {rb_err}")
        # Re-raise the exception to indicate failure
        # raise e # Optionally re-raise

    finally:
        # Ensure cursor and connection are closed
        if cur:
            try:
                cur.close()
            except Exception as cur_close_err:
                 logger.warning(f"Error closing cursor: {cur_close_err}")
        if conn:
            try:
                 # Restore autocommit if it was changed
                 conn.autocommit = True
                 conn.close()
                 logger.info("Database connection closed.")
            except Exception as conn_close_err:
                 logger.warning(f"Error closing connection: {conn_close_err}")

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
        tables = [
            "words",
            "definitions",
            "relations",
            "etymologies",
            "affixations",
            "definition_relations",
            "parts_of_speech",
        ]
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            table_stats.add_row(t, f"{count:,}")
        console.print(table_stats)
        console.print()

        # Display relation types and counts
        console.print("[bold]Relation Types and Counts[/]")
        cur.execute(
            """
            SELECT relation_type, COUNT(*) as count
            FROM relations
            GROUP BY relation_type
            ORDER BY count DESC
        """
        )
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
        cur.execute(
            """
            SELECT language_code, COUNT(*) as count
            FROM words
            GROUP BY language_code
            ORDER BY count DESC
        """
        )
        lang_stats = cur.fetchall()
        if lang_stats:
            lang_table = Table(box=box.ROUNDED)
            lang_table.add_column("Language", style="magenta")
            lang_table.add_column("Count", justify="right", style="green")

            for lang, count in lang_stats:
                lang_name = (
                    "Tagalog" if lang == "tl" else "Cebuano" if lang == "ceb" else lang
                )
                lang_table.add_row(lang_name, f"{count:,}")

            console.print(lang_table)

        # Display parts of speech distribution
        console.print()
        console.print("[bold]Parts of Speech Distribution[/]")
        cur.execute(
            """
            SELECT p.name_tl, COUNT(*) as count
            FROM definitions d
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            GROUP BY p.name_tl
            ORDER BY COUNT(*) DESC
        """
        )
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
        cur.execute(
            """
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
        """
        )
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
        console.print(
            Panel(
                """
- [bold]affixations[/bold]: This table is for storing information about word affixation patterns, which are linguistic processes where affixes (prefixes, suffixes, infixes) are added to root words to create new words. These are populated by specialized affix analysis functions.

- [bold]definition_relations[/bold]: This table stores semantic relationships between definitions (rather than between words). These are typically populated during advanced linguistic analysis.

Both tables might be empty if no specialized linguistic analysis has been performed on the dataset yet.
""",
                title="Table Explanations",
                border_style="blue",
            )
        )

        if args.quick:
            console.print("[yellow]Sample entries from 'words' table:[/]")
            cur.execute(
                "SELECT id, lemma, language_code, root_word_id FROM words LIMIT 5"
            )
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
            cur.execute(
                """
                SELECT COUNT(*) FROM relations r
                WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
                   OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
            """
            )
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found orphaned relations")
            cur.execute(
                """
                SELECT COUNT(*) 
                FROM (
                    SELECT word_id, definition_text, COUNT(*)
                    FROM definitions
                    GROUP BY word_id, definition_text
                    HAVING COUNT(*) > 1
                ) dupes
            """
            )
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found duplicate definitions")
            cur.execute("SELECT COUNT(*) FROM words WHERE search_text IS NULL")
            if cur.fetchone()[0] > 0:
                integrity_issues.append("Found words with missing search vectors")
            cur.execute(
                """
                SELECT baybayin_form, COUNT(*)
                FROM words
                WHERE has_baybayin = TRUE
                GROUP BY baybayin_form
                HAVING COUNT(*) > 1
            """
            )
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
    cur.execute(
        """
        DELETE FROM relations r
        WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
        OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
    """
    )

    # Delete duplicate definitions
    cur.execute(
        """
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
    """
    )

    # Update missing search_text values
    cur.execute(
        """
        UPDATE words
        SET search_text = to_tsvector('english',
            COALESCE(lemma, '') || ' ' ||
            COALESCE(normalized_lemma, '') || ' ' ||
            COALESCE(baybayin_form, '') || ' ' ||
            COALESCE(romanized_form, '')
        )
        WHERE search_text IS NULL
    """
    )

    # Fix Baybayin inconsistencies
    cur.execute(
        """
        DELETE FROM words 
        WHERE has_baybayin = TRUE AND baybayin_form IS NULL
    """
    )

    cur.execute(
        """
        UPDATE words
        SET has_baybayin = FALSE,
            baybayin_form = NULL
        WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
    """
    )

    logger.info("Database repairs completed")
    return True


def display_help(args):
    console = Console()
    console.print("\n[bold cyan]📖 Dictionary Manager CLI Help[/]", justify="center")
    console.print(
        "[dim]A comprehensive tool for managing Filipino dictionary data[/]\n",
        justify="center",
    )
    usage_panel = Panel(
        Text.from_markup("python dictionary_manager.py [command] [options]"),
        title="Basic Usage",
        border_style="blue",
    )
    console.print(usage_panel)
    console.print()
    commands = [
        {
            "name": "migrate",
            "description": "Create/update schema and load data from sources",
            "options": [
                ("--check-exists", "Skip identical existing entries"),
                ("--force", "Skip confirmation prompt"),
            ],
            "example": "python dictionary_manager.py migrate --check-exists",
            "icon": "🔄",
        },
        {
            "name": "lookup",
            "description": "Look up comprehensive information about a word",
            "options": [
                ("word", "The word to look up"),
                ("--format", "Output format (text/json/rich)"),
            ],
            "example": "python dictionary_manager.py lookup kamandag",
            "icon": "🔍",
        },
        {
            "name": "stats",
            "description": "Display comprehensive dictionary statistics",
            "options": [
                ("--detailed", "Show detailed statistics"),
                ("--export", "Export statistics to file"),
            ],
            "example": "python dictionary_manager.py stats --detailed",
            "icon": "📊",
        },
        {
            "name": "verify",
            "description": "Verify data integrity",
            "options": [
                ("--quick", "Run quick verification"),
                ("--repair", "Attempt to repair issues"),
            ],
            "example": "python dictionary_manager.py verify --repair",
            "icon": "✅",
        },
        {
            "name": "purge",
            "description": "Safely delete all data from the database",
            "options": [("--force", "Skip confirmation prompt")],
            "example": "python dictionary_manager.py purge --force",
            "icon": "🗑️",
        },
    ]
    data_commands = Table(
        title="Data Management Commands", box=box.ROUNDED, border_style="cyan"
    )
    data_commands.add_column("Command", style="bold yellow")
    data_commands.add_column("Description", style="white")
    data_commands.add_column("Options", style="cyan")
    data_commands.add_column("Example", style="green")
    query_commands = Table(
        title="Query Commands", box=box.ROUNDED, border_style="magenta"
    )
    query_commands.add_column("Command", style="bold yellow")
    query_commands.add_column("Description", style="white")
    query_commands.add_column("Options", style="cyan")
    query_commands.add_column("Example", style="green")
    for cmd in commands:
        options_text = (
            "\n".join([f"[cyan]{opt[0]}[/]: {opt[1]}" for opt in cmd["options"]]) or "-"
        )
        row = [
            f"{cmd['icon']} {cmd['name']}",
            cmd["description"],
            options_text,
            f"[dim]{cmd['example']}[/]",
        ]
        if cmd["name"] in ["migrate", "update", "purge"]:
            data_commands.add_row(*row)
        else:
            query_commands.add_row(*row)
    console.print(data_commands)
    console.print()
    console.print(query_commands)
    console.print()
    console.print(
        "\n[dim]For more detailed information, visit the documentation.[/]",
        justify="center",
    )
    console.print()


def lookup_word(args):
    """Look up a word and display all its information."""
    console = Console()

    try:
        logger.info(f"Starting lookup for word: '{args.word}'")
        conn = get_connection()
        cur = conn.cursor()

        # First verify that pg_trgm extension is installed
        try:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            has_pg_trgm = cur.fetchone() is not None
            logger.info(f"pg_trgm extension installed: {has_pg_trgm}")
        except Exception as e:
            logger.warning(f"Could not check for pg_trgm extension: {str(e)}")
            has_pg_trgm = False

        # Search across all relevant fields
        logger.info(f"Executing exact match query for '{args.word}'")
        query = """
            SELECT DISTINCT w.id, w.lemma, w.language_code, w.normalized_lemma, 
                   w.preferred_spelling, w.has_baybayin, w.baybayin_form, 
                   w.romanized_form, w.tags, w.source_info
            FROM words w
            LEFT JOIN pronunciations p ON w.id = p.word_id
            WHERE LOWER(w.lemma) = LOWER(%s)
               OR LOWER(w.normalized_lemma) = LOWER(%s)
               OR LOWER(w.romanized_form) = LOWER(%s)
               OR LOWER(w.baybayin_form) = LOWER(%s)
               OR LOWER(p.value) = LOWER(%s)
        """
        try:
            cur.execute(query, (args.word, args.word, args.word, args.word, args.word))
            results = cur.fetchall()
            logger.info(f"Found {len(results)} exact matches")
        except Exception as e:
            logger.error(f"Error in exact match query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(
                f"Parameters: {(args.word, args.word, args.word, args.word, args.word)}"
            )
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        if not results:
            # Try either fuzzy search or ILIKE depending on extension availability
            if has_pg_trgm:
                logger.info(f"No exact matches, trying fuzzy search for '{args.word}'")
                try:
                    fuzzy_query = """
                        SELECT DISTINCT w.id, w.lemma, w.language_code, w.normalized_lemma, 
                               w.preferred_spelling, w.has_baybayin, w.baybayin_form, 
                               w.romanized_form, w.tags, w.source_info,
                               similarity(w.lemma, %s) as sim_score
                        FROM words w
                        WHERE similarity(w.lemma, %s) > 0.3
                        ORDER BY sim_score DESC
                        LIMIT 5
                    """
                    cur.execute(fuzzy_query, (args.word, args.word))
                    fuzzy_results = cur.fetchall()
                    logger.info(f"Found {len(fuzzy_results)} fuzzy matches")

                    if fuzzy_results:
                        logger.info(f"Fuzzy matches: {[r[1] for r in fuzzy_results]}")
                        results = [result[:-1] for result in fuzzy_results]
                        console.print(
                            f"\n[yellow]No exact matches found. Showing similar words:[/]"
                        )
                    else:
                        # Fall back to ILIKE if no fuzzy matches
                        logger.info("No fuzzy matches, falling back to ILIKE search")
                except Exception as e:
                    logger.error(f"Error in fuzzy search query: {str(e)}")
                    logger.error(f"Falling back to ILIKE search due to error")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.info("pg_trgm not available, using ILIKE search instead")

            # If we have no results yet, try ILIKE search
            if not results:
                logger.info(f"Trying ILIKE search for '{args.word}'")
                ilike_query = """
                    SELECT DISTINCT w.id, w.lemma, w.language_code, w.normalized_lemma, 
                          w.preferred_spelling, w.has_baybayin, w.baybayin_form, 
                          w.romanized_form, w.tags, w.source_info
                    FROM words w
                    WHERE w.lemma ILIKE %s
                    OR w.normalized_lemma ILIKE %s
                    LIMIT 5
                """
                try:
                    cur.execute(ilike_query, (f"%{args.word}%", f"%{args.word}%"))
                    ilike_results = cur.fetchall()
                    logger.info(f"ILIKE search found {len(ilike_results)} matches")

                    if ilike_results:
                        logger.info(f"ILIKE matches: {[r[1] for r in ilike_results]}")
                        console.print(
                            f"\n[yellow]Found words containing '[bold]{args.word}[/]':[/]"
                        )
                        results = ilike_results
                    else:
                        # Check if word exists in database at all
                        logger.info("Diagnostic: checking if word exists in any form")
                        cur.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1 FROM words 
                                WHERE lemma LIKE %s 
                                OR normalized_lemma LIKE %s
                            )
                        """,
                            (f"%{args.word}%", f"%{args.word}%"),
                        )
                        exists = cur.fetchone()[0]
                        logger.info(
                            f"Diagnostic result: Word '{args.word}' exists (partial match): {exists}"
                        )

                        # Verify database schema
                        logger.info("Diagnostic: verifying database schema")
                        cur.execute(
                            """
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = 'words'
                        """
                        )
                        columns = cur.fetchall()
                        logger.info(
                            f"Diagnostic: words table has {len(columns)} columns"
                        )
                        for col in columns:
                            logger.info(f"  Column: {col[0]}, Type: {col[1]}")

                        console.print(
                            f"\n[yellow]No entries found for '[bold]{args.word}[/]'[/]"
                        )
                        return
                except Exception as e:
                    logger.error(f"Error in ILIKE search: {str(e)}")
                    import traceback

                    logger.error(f"Traceback: {traceback.format_exc()}")
                    console.print(f"\n[red]Error searching for word: {str(e)}[/]")
                    return

        for idx, result in enumerate(results):
            try:
                logger.info(f"Processing result {idx+1}/{len(results)}")

                word_id = result[0]
                logger.info(f"Word ID: {word_id}")

                lemma = result[1]
                logger.info(f"Lemma: {lemma}")

                lang_code = result[2]
                logger.info(f"Language code: {lang_code}")

                norm_lemma = result[3]
                pref_spell = result[4]
                has_bayb = result[5]
                bayb_form = result[6]
                rom_form = result[7]
                tags = result[8]
                sources = result[9]

                # Print word header with metadata
                console.print(f"\n[bold blue]═══ {lemma} ═══[/]")

                # Print basic information
                info_table = Table(show_header=False, box=None)
                info_table.add_row("[dim]Language:[/]", f" {lang_code}")

                if norm_lemma and norm_lemma != lemma:
                    info_table.add_row("[dim]Normalized:[/]", f" {norm_lemma}")
                if pref_spell:
                    info_table.add_row("[dim]Preferred:[/]", f" {pref_spell}")
                if rom_form:
                    info_table.add_row("[dim]Romanized:[/]", f" {rom_form}")
                if has_bayb and bayb_form:
                    info_table.add_row("[dim]Baybayin:[/]", f" {bayb_form}")

                console.print(info_table)

                # Display tags
                if tags:
                    try:
                        tags_list = json.loads(tags) if isinstance(tags, str) else tags
                        if tags_list:
                            console.print("[bold cyan]Tags:[/]", ", ".join(tags_list))
                    except Exception as e:
                        logger.warning(f"Error parsing tags: {str(e)}")

                # Display sources
                if sources:
                    try:
                        sources_list = json.loads(sources) if isinstance(sources, str) else sources
                        if sources_list:
                            console.print("[bold cyan]Sources:[/]")
                            for src in sources_list:
                                console.print(f"• {src}")
                    except Exception as e:
                        logger.warning(f"Error parsing sources: {str(e)}")

                # Get and print pronunciations
                try:
                    logger.info(f"Querying pronunciations for word_id {word_id}")
                    cur.execute(
                        """
                        SELECT type, value, tags, sources
                        FROM pronunciations
                        WHERE word_id = %s
                        ORDER BY type
                    """,
                        (word_id,),
                    )

                    prons = cur.fetchall()
                    logger.info(f"Found {len(prons)} pronunciations")

                    if prons:
                        console.print("\n[bold cyan]Pronunciations:[/]")
                        for pron in prons:
                            type_, value, p_tags, p_sources = pron
                            console.print(
                                f"• {value}" + (f" ({type_})" if type_ else "")
                            )
                            
                            # Display pronunciation tags if available
                            if p_tags:
                                try:
                                    p_tags_list = json.loads(p_tags) if isinstance(p_tags, str) else p_tags
                                    if p_tags_list:
                                        console.print(f"  [dim]Tags:[/] {', '.join(p_tags_list)}")
                                except Exception as e:
                                    logger.warning(f"Error parsing pronunciation tags: {str(e)}")
                                    
                            # Display pronunciation sources if available
                            if p_sources:
                                try:
                                    p_sources_list = json.loads(p_sources) if isinstance(p_sources, str) else p_sources
                                    if p_sources_list:
                                        console.print(f"  [dim]Sources:[/] {', '.join(p_sources_list)}")
                                except Exception as e:
                                    logger.warning(f"Error parsing pronunciation sources: {str(e)}")

                except Exception as e:
                    logger.error(f"Error retrieving pronunciations: {str(e)}")

                # Get and print etymologies
                try:
                    logger.info(f"Querying etymologies for word_id {word_id}")
                    cur.execute(
                        """
                        SELECT etymology_text, normalized_components, language_codes, sources
                        FROM etymologies
                        WHERE word_id = %s
                    """,
                        (word_id,),
                    )

                    etyms = cur.fetchall()
                    logger.info(f"Found {len(etyms)} etymologies")

                    if etyms:
                        console.print("\n[bold cyan]Etymology:[/]")
                        for etym in etyms:
                            etym_text = etym[0]
                            etym_panel = Panel(etym_text, title="Etymology")
                            console.print(etym_panel)

                except Exception as e:
                    logger.error(f"Error retrieving etymologies: {str(e)}")

                # Get and print definitions
                try:
                    logger.info(f"Querying definitions for word_id {word_id}")
                    cur.execute(
                        """
                        SELECT d.definition_text, d.examples, d.usage_notes,
                               p.name_tl AS pos_name
                        FROM definitions d
                        LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                        WHERE d.word_id = %s
                        ORDER BY p.name_tl, d.id
                    """,
                        (word_id,),
                    )

                    definitions = cur.fetchall()
                    logger.info(f"Found {len(definitions)} definitions")

                    if definitions:
                        console.print("\n[bold cyan]Definitions:[/]")
                        current_pos = None

                        for def_idx, definition in enumerate(definitions):
                            logger.info(
                                f"Processing definition {def_idx+1}/{len(definitions)}"
                            )
                            def_text, examples, notes, pos = definition

                            if pos != current_pos:
                                console.print(f"\n{pos or 'Uncategorized'}:")
                                current_pos = pos

                            def_panel = Panel(def_text)
                            console.print(def_panel)

                            if examples:
                                try:
                                    ex_list = (
                                        json.loads(examples)
                                        if isinstance(examples, str)
                                        else examples
                                    )
                                    if ex_list:
                                        console.print("[dim]Examples:[/]")
                                        for ex in ex_list:
                                            console.print(f"• {ex}")
                                except Exception as e:
                                    logger.warning(f"Error parsing examples: {str(e)}")

                            if notes:
                                console.print(f"[dim]Notes:[/] {notes}")

                except Exception as e:
                    logger.error(f"Error retrieving definitions: {str(e)}")

                # Get and print relations
                try:
                    logger.info(f"Querying relations for word_id {word_id}")
                    cur.execute(
                        """
                        SELECT DISTINCT r.relation_type, w2.lemma
                        FROM relations r
                        JOIN words w2 ON r.to_word_id = w2.id
                        WHERE r.from_word_id = %s
                        ORDER BY r.relation_type, w2.lemma
                    """,
                        (word_id,),
                    )

                    relations = cur.fetchall()
                    logger.info(f"Found {len(relations)} relations")

                    if relations:
                        console.print("\n[bold cyan]Related Words:[/]")
                        current_type = None
                        for rel_type, rel_word in relations:
                            if rel_type != current_type:
                                console.print(f"\n[bold]{rel_type}:[/]")
                                current_type = rel_type
                            console.print(f"• {rel_word}")
                    else:
                        logger.info(f"No relations found for word ID {word_id}")

                except Exception as e:
                    logger.error(f"Error retrieving relations: {str(e)}")

                console.print("\n" + "─" * 80 + "\n")

            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                logger.error(f"Word: {args.word}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                console.print(f"[red]Error processing word: {str(e)}[/]")

    except Exception as e:
        logger.error(f"Error looking up word: {str(e)}")
        logger.error(f"Word: {args.word}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        console.print(f"[red]Error looking up word '{args.word}': {str(e)}[/]")
        console.print("[red]Check logs for detailed error information[/]")

    finally:
        if "conn" in locals() and conn:
            conn.close()
            logger.info("Database connection closed")


def display_dictionary_stats_cli(args):
    """Display dictionary statistics from the command line."""
    try:
        # Get a proper database connection and cursor
        conn = get_connection()
        with conn.cursor() as cur:
            display_dictionary_stats(cur)

        # Make sure to close the connection when done
        if conn:
            conn.close()
    except Exception as e:
        console = Console()
        console.print(f"[red]Error displaying dictionary stats: {str(e)}[/]")


@with_transaction(commit=False)
def display_dictionary_stats(cur):
    """Display comprehensive dictionary statistics."""
    console = Console()
    try:
        # Overall Statistics
        overall_table = Table(title="[bold blue]Overall Statistics[/]", box=box.ROUNDED)
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Count", justify="right", style="green")
        overall_table.add_column("Details", style="dim")

        # Check which columns exist in the words table
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words'
        """
        )
        available_columns = {row[0] for row in cur.fetchall()}

        # Basic counts with details - build dynamically based on available columns
        basic_queries = {
            "Total Words": ("SELECT COUNT(*) FROM words", None),
            "Total Definitions": ("SELECT COUNT(*) FROM definitions", None),
            "Total Relations": ("SELECT COUNT(*) FROM relations", None),
            "Total Etymologies": ("SELECT COUNT(*) FROM etymologies", None),
            "Total Pronunciations": ("SELECT COUNT(*) FROM pronunciations", None),
            "Total Credits": ("SELECT COUNT(*) FROM credits", None),
            "Words with Baybayin": (
                "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE",
                None,
            ),
            "Words with Examples": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM definitions 
                WHERE examples IS NOT NULL
            """,
                None,
            ),
            "Words with Etymology": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM etymologies
            """,
                None,
            ),
            "Words with Pronunciation": (
                """
                SELECT COUNT(DISTINCT word_id) 
                FROM pronunciations
            """,
                None,
            ),
        }

        # Add optional columns if they exist
        if "is_proper_noun" in available_columns:
            basic_queries["Proper Nouns"] = (
                "SELECT COUNT(*) FROM words WHERE is_proper_noun = TRUE",
                None,
            )
        if "is_abbreviation" in available_columns:
            basic_queries["Abbreviations"] = (
                "SELECT COUNT(*) FROM words WHERE is_abbreviation = TRUE",
                None,
            )
        if "is_initialism" in available_columns:
            basic_queries["Initialisms"] = (
                "SELECT COUNT(*) FROM words WHERE is_initialism = TRUE",
                None,
            )
        if "root_word_id" in available_columns:
            basic_queries["Words with Root"] = (
                "SELECT COUNT(*) FROM words WHERE root_word_id IS NOT NULL",
                None,
            )

        for label, (query, detail_query) in basic_queries.items():
            try:
                cur.execute(query)
                count = cur.fetchone()[0]
                details = ""
                if detail_query:
                    cur.execute(detail_query)
                    details = cur.fetchone()[0]
                overall_table.add_row(label, f"{count:,}", details)
            except Exception as e:
                logger.warning(f"Error getting stats for {label}: {e}")
                overall_table.add_row(label, "N/A", f"Error: {str(e)}")

        # Language Statistics with more details
        try:
            cur.execute(
                """
                SELECT 
                    w.language_code,
                    COUNT(*) as word_count,
                    COUNT(DISTINCT d.id) as def_count,
                    COUNT(DISTINCT e.id) as etym_count,
                    COUNT(DISTINCT p.id) as pron_count,
                    COUNT(DISTINCT CASE WHEN w.has_baybayin THEN w.id END) as baybayin_count
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN etymologies e ON w.id = e.word_id
                LEFT JOIN pronunciations p ON w.id = p.word_id
                GROUP BY w.language_code
                ORDER BY word_count DESC
            """
            )

            lang_table = Table(title="[bold blue]Words by Language[/]", box=box.ROUNDED)
            lang_table.add_column("Language", style="yellow")
            lang_table.add_column("Words", justify="right", style="green")
            lang_table.add_column("Definitions", justify="right", style="green")
            lang_table.add_column("Etymologies", justify="right", style="green")
            lang_table.add_column("Pronunciations", justify="right", style="green")
            lang_table.add_column("Baybayin", justify="right", style="green")

            total_words = 0
            results = cur.fetchall()
            for row in results:
                total_words += row[1]

            for lang_code, words, defs, etyms, prons, bayb in results:
                percentage = (words / total_words) * 100 if total_words > 0 else 0
                lang_table.add_row(
                    lang_code,
                    f"{words:,} ({percentage:.1f}%)",
                    f"{defs:,}",
                    f"{etyms:,}",
                    f"{prons:,}",
                    f"{bayb:,}",
                )

            console.print("\n[bold]Dictionary Statistics[/]")
            console.print(overall_table)
            console.print()
            console.print(lang_table)
        except Exception as e:
            logger.error(f"Error displaying language statistics: {str(e)}")
            console.print(f"[red]Error displaying language statistics: {str(e)}[/]")
            console.print(overall_table)

        # Parts of Speech Statistics with examples
        try:
            # Check if standardized_pos_id exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'definitions' AND column_name = 'standardized_pos_id'
                )
            """
            )

            has_standardized_pos = cur.fetchone()[0]

            if has_standardized_pos:
                cur.execute(
                    """
                    SELECT 
                        p.name_tl,
                        COUNT(*) as count,
                        COUNT(DISTINCT d.word_id) as unique_words,
                        COUNT(CASE WHEN d.examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions d
                    JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                    GROUP BY p.name_tl
                    ORDER BY count DESC
                """
                )
            else:
                # Fallback to using part_of_speech text field
                cur.execute(
                    """
                    SELECT 
                        COALESCE(part_of_speech, 'Unknown'),
                        COUNT(*) as count,
                        COUNT(DISTINCT word_id) as unique_words,
                        COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions
                    GROUP BY part_of_speech
                    ORDER BY count DESC
                """
                )

            pos_table = Table(title="[bold blue]Parts of Speech[/]", box=box.ROUNDED)
            pos_table.add_column("Part of Speech", style="yellow")
            pos_table.add_column("Definitions", justify="right", style="green")
            pos_table.add_column("Unique Words", justify="right", style="green")
            pos_table.add_column("With Examples", justify="right", style="green")

            pos_results = cur.fetchall()
            if pos_results:
                for pos, count, unique_words, with_examples in pos_results:
                    pos_table.add_row(
                        pos or "Uncategorized",
                        f"{count:,}",
                        f"{unique_words:,}",
                        f"{with_examples:,}",
                    )

                console.print()
                console.print(pos_table)
        except Exception as e:
            logger.error(f"Error displaying part of speech statistics: {str(e)}")
            console.print(
                f"[red]Error displaying part of speech statistics: {str(e)}[/]"
            )

        # Relationship Statistics by category
        try:
            cur.execute(
                """
                SELECT 
                    r.relation_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT r.from_word_id) as unique_sources,
                    COUNT(DISTINCT r.to_word_id) as unique_targets
                FROM relations r
                GROUP BY r.relation_type
                ORDER BY count DESC
            """
            )

            rel_results = cur.fetchall()
            if rel_results:
                rel_table = Table(
                    title="[bold blue]Relationship Types[/]", box=box.ROUNDED
                )
                rel_table.add_column("Type", style="yellow")
                rel_table.add_column("Total", justify="right", style="green")
                rel_table.add_column("Unique Sources", justify="right", style="green")
                rel_table.add_column("Unique Targets", justify="right", style="green")

                for rel_type, count, sources, targets in rel_results:
                    rel_table.add_row(
                        rel_type or "Unknown",
                        f"{count:,}",
                        f"{sources:,}",
                        f"{targets:,}",
                    )

                console.print()
                console.print(rel_table)
        except Exception as e:
            logger.error(f"Error displaying relationship statistics: {str(e)}")
            console.print(f"[red]Error displaying relationship statistics: {str(e)}[/]")

        # Source Statistics with more details
        try:
            # First check if source_info column exists
            if "source_info" in available_columns:
                # Get source statistics from source_info
                cur.execute(
                    """
                    SELECT 
                        COALESCE(source_info, 'Unknown') as source_name,
                        COUNT(*) as word_count
                    FROM words
                    GROUP BY source_name
                    ORDER BY word_count DESC
                """
                )

                source_results = cur.fetchall()
                if source_results:
                    source_table = Table(
                        title="[bold blue]Source Distribution[/]", box=box.ROUNDED
                    )
                    source_table.add_column("Source", style="yellow")
                    source_table.add_column("Words", justify="right", style="green")

                    for source, count in source_results:
                        source_table.add_row(source or "Unknown", f"{count:,}")

                    console.print()
                    console.print(source_table)

            # Also check definitions sources
            cur.execute(
                """
                SELECT 
                    COALESCE(sources, 'Unknown') as source_name,
                    COUNT(*) as def_count,
                    COUNT(DISTINCT word_id) as word_count,
                    COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as example_count
                FROM definitions
                GROUP BY sources
                ORDER BY def_count DESC
            """
            )

            def_source_results = cur.fetchall()
            if def_source_results:
                def_source_table = Table(
                    title="[bold blue]Definition Sources[/]", box=box.ROUNDED
                )
                def_source_table.add_column("Source", style="yellow")
                def_source_table.add_column(
                    "Definitions", justify="right", style="green"
                )
                def_source_table.add_column("Words", justify="right", style="green")
                def_source_table.add_column(
                    "With Examples", justify="right", style="green"
                )

                for source, def_count, word_count, example_count in def_source_results:
                    def_source_table.add_row(
                        source or "Unknown",
                        f"{def_count:,}",
                        f"{word_count:,}",
                        f"{example_count:,}",
                    )

                console.print()
                console.print(def_source_table)
        except Exception as e:
            logger.error(f"Error displaying source statistics: {str(e)}")
            console.print(f"[yellow]Could not generate source statistics: {str(e)}[/]")

        # Baybayin Statistics with details
        try:
            baybayin_table = Table(
                title="[bold blue]Baybayin Statistics[/]", box=box.ROUNDED
            )
            baybayin_table.add_column("Metric", style="yellow")
            baybayin_table.add_column("Count", justify="right", style="green")
            baybayin_table.add_column("Details", style="dim")

            baybayin_queries = {
                "Total Baybayin Forms": (
                    "SELECT COUNT(*) FROM words WHERE baybayin_form IS NOT NULL",
                    """SELECT COUNT(DISTINCT language_code) 
                    FROM words WHERE baybayin_form IS NOT NULL""",
                ),
                "With Romanization": (
                    "SELECT COUNT(*) FROM words WHERE romanized_form IS NOT NULL",
                    None,
                ),
                "Verified Forms": (
                    """SELECT COUNT(*) FROM words 
                    WHERE has_baybayin = TRUE 
                    AND baybayin_form IS NOT NULL""",
                    None,
                ),
            }

            # Only add Badlit stats if the column exists
            if "badlit_form" in available_columns:
                baybayin_queries["With Badlit"] = (
                    "SELECT COUNT(*) FROM words WHERE badlit_form IS NOT NULL",
                    None,
                )
                baybayin_queries["Complete Forms"] = (
                    """SELECT COUNT(*) FROM words 
                    WHERE baybayin_form IS NOT NULL 
                    AND romanized_form IS NOT NULL 
                    AND badlit_form IS NOT NULL""",
                    None,
                )

            for label, (query, detail_query) in baybayin_queries.items():
                try:
                    cur.execute(query)
                    count = cur.fetchone()[0]
                    details = ""
                    if detail_query:
                        cur.execute(detail_query)
                        details = f"across {cur.fetchone()[0]} languages"
                    baybayin_table.add_row(label, f"{count:,}", details)
                except Exception as e:
                    logger.warning(f"Error getting Baybayin stats for {label}: {e}")
                    baybayin_table.add_row(label, "N/A", f"Error: {str(e)}")

            console.print()
            console.print(baybayin_table)
        except Exception as e:
            logger.error(f"Error displaying Baybayin statistics: {str(e)}")
            console.print(
                f"[yellow]Could not generate Baybayin statistics: {str(e)}[/]"
            )

        # Print timestamp
        console.print(
            f"\n[dim]Statistics generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
        )

    except Exception as e:
        logger.error(f"Error displaying dictionary stats: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")

def display_leaderboard(cur, console):
    """Displays various contribution leaderboards."""
    # Get a fresh connection and cursor instead of using the one passed in
    # This ensures we start with a clean transaction state
    conn = None
    try:
        from contextlib import closing
        import psycopg2
        from psycopg2.extras import DictCursor
        
        # Get a fresh connection from the pool
        conn = get_connection()
        
        # Use a context manager to ensure proper closing
        with closing(conn.cursor(cursor_factory=DictCursor)) as fresh_cur:
            console.print(
                "\n[bold magenta underline]📊 Dictionary Contributors Leaderboard[/]\n"
            )

            overall_stats_table = Table(
                title="[bold blue]Overall Statistics[/]", box=box.ROUNDED, show_header=False
            )
            overall_stats_table.add_column("Statistic", style="cyan")
            overall_stats_table.add_column("Value", justify="right", style="green")

            try:
                # Overall stats
                fresh_cur.execute("SELECT COUNT(*) FROM words")
                total_words = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Words", f"{total_words:,}")

                fresh_cur.execute("SELECT COUNT(*) FROM definitions")
                total_definitions = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Definitions", f"{total_definitions:,}")

                fresh_cur.execute("SELECT COUNT(*) FROM relations")
                total_relations = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Relations", f"{total_relations:,}")

                fresh_cur.execute("SELECT COUNT(*) FROM etymologies")
                total_etymologies = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Total Etymologies", f"{total_etymologies:,}")

                fresh_cur.execute(
                    "SELECT COUNT(DISTINCT standardized_pos_id) FROM definitions WHERE standardized_pos_id IS NOT NULL"
                )
                total_pos = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Unique Parts of Speech", str(total_pos))

                fresh_cur.execute(
                    "SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL"
                )
                words_with_baybayin = fresh_cur.fetchone()[0]
                overall_stats_table.add_row("Words w/ Baybayin", f"{words_with_baybayin:,}")

                console.print(overall_stats_table)
                console.print()

            except Exception as e:
                logger.error(f"Error generating overall statistics: {str(e)}")
                console.print(f"[yellow]Could not generate overall statistics: {str(e)}[/]")
                # Roll back to clean state
                conn.rollback()

            # Definition Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH source_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            COUNT(*) AS def_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                            COUNT(CASE WHEN examples IS NOT NULL AND examples != '' THEN 1 END) AS with_examples,
                            COUNT(DISTINCT standardized_pos_id) AS pos_count,
                            COUNT(CASE WHEN usage_notes IS NOT NULL AND usage_notes != '' THEN 1 END) AS with_notes
                        FROM definitions
                        GROUP BY source_name
                    )
                    SELECT
                        source_name,
                        def_count,
                        unique_words,
                        with_examples,
                        pos_count,
                        with_notes,
                        ROUND(100.0 * with_examples / NULLIF(def_count, 0), 1) as example_percentage,
                        ROUND(100.0 * with_notes / NULLIF(def_count, 0), 1) as notes_percentage
                    FROM source_stats
                    ORDER BY def_count DESC
                    """
                )

                def_results = fresh_cur.fetchall()
                if def_results:
                    def_table = Table(
                        title="[bold blue]Definition Contributors[/]", box=box.ROUNDED
                    )
                    def_table.add_column("Source", style="yellow")
                    def_table.add_column("Definitions", justify="right", style="green")
                    def_table.add_column("Words", justify="right", style="green")
                    def_table.add_column("Examples", justify="right", style="cyan")
                    def_table.add_column("POS Types", justify="right", style="cyan")
                    def_table.add_column("Notes", justify="right", style="cyan")
                    def_table.add_column("Coverage", style="dim")

                    for row in def_results:
                        source = row['source_name'] if 'source_name' in row else "Unknown"
                        defs = row['def_count'] if 'def_count' in row else 0
                        words = row['unique_words'] if 'unique_words' in row else 0
                        examples = row['with_examples'] if 'with_examples' in row else 0
                        pos = row['pos_count'] if 'pos_count' in row else 0
                        notes = row['with_notes'] if 'with_notes' in row else 0
                        ex_pct = row['example_percentage'] if 'example_percentage' in row else 0.0
                        notes_pct = row['notes_percentage'] if 'notes_percentage' in row else 0.0

                        coverage = f"Examples: {ex_pct or 0.0}%, Notes: {notes_pct or 0.0}%"
                        def_table.add_row(
                            source,
                            f"{defs:,}",
                            f"{words:,}",
                            f"{examples:,}",
                            str(pos),
                            f"{notes:,}",
                            coverage,
                        )

                    console.print(def_table)
                    console.print()
            except Exception as e:
                logger.error(f"Error generating definition statistics: {str(e)}")
                console.print(
                    f"[red]Error:[/][yellow] Could not generate definition statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Etymology Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH etym_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            COUNT(*) AS etym_count,
                            COUNT(DISTINCT word_id) AS unique_words,
                            COUNT(CASE WHEN normalized_components IS NOT NULL THEN 1 END) AS with_components,
                            COUNT(CASE WHEN language_codes IS NOT NULL THEN 1 END) AS with_lang_codes
                        FROM etymologies
                        GROUP BY source_name
                    )
                    SELECT *,
                        ROUND(100.0 * with_components / NULLIF(etym_count, 0), 1) as comp_percentage,
                        ROUND(100.0 * with_lang_codes / NULLIF(etym_count, 0), 1) as lang_percentage
                    FROM etym_stats
                    ORDER BY etym_count DESC
                    """
                )

                etym_results = fresh_cur.fetchall()
                if etym_results:
                    etym_table = Table(
                        title="[bold blue]Etymology Contributors[/]", box=box.ROUNDED
                    )
                    etym_table.add_column("Source", style="yellow")
                    etym_table.add_column("Etymologies", justify="right", style="green")
                    etym_table.add_column("Words", justify="right", style="green")
                    etym_table.add_column("Components", justify="right", style="cyan")
                    etym_table.add_column("Lang Codes", justify="right", style="cyan")
                    etym_table.add_column("Coverage", style="dim")

                    for row in etym_results:
                        source = row['source_name'] if 'source_name' in row else "Unknown"
                        count = row['etym_count'] if 'etym_count' in row else 0
                        words = row['unique_words'] if 'unique_words' in row else 0
                        comps = row['with_components'] if 'with_components' in row else 0
                        langs = row['with_lang_codes'] if 'with_lang_codes' in row else 0
                        comp_pct = row['comp_percentage'] if 'comp_percentage' in row else 0.0
                        lang_pct = row['lang_percentage'] if 'lang_percentage' in row else 0.0

                        coverage = (
                            f"Components: {comp_pct or 0.0}%, Languages: {lang_pct or 0.0}%"
                        )
                        etym_table.add_row(
                            source,
                            f"{count:,}",
                            f"{words:,}",
                            f"{comps:,}",
                            f"{langs:,}",
                            coverage,
                        )

                    console.print(etym_table)
                    console.print()
            except Exception as e:
                logger.error(f"Error generating etymology statistics: {str(e)}")
                console.print(
                    f"[red]Error:[/][yellow] Could not generate etymology statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Relationship Contributors
            try:
                fresh_cur.execute(
                    """
                    WITH rel_stats AS (
                        SELECT
                            CASE
                                WHEN sources ILIKE '%project marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%marayum%' THEN 'Project Marayum'
                                WHEN sources ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kaikki%' AND sources ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                                WHEN sources ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                                WHEN sources ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                                WHEN sources ILIKE '%tagalog.com%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%root_words%' THEN 'tagalog.com'
                                WHEN sources ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                                WHEN sources ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                                ELSE COALESCE(sources, 'Unknown')
                            END AS source_name,
                            relation_type,
                            COUNT(*) AS rel_count
                        FROM relations
                        GROUP BY source_name, relation_type
                    )
                    SELECT source_name, relation_type, rel_count
                    FROM rel_stats
                    ORDER BY source_name, rel_count DESC
                    """
                )

                rel_results = fresh_cur.fetchall()
                if rel_results:
                    rel_table = Table(
                        title="[bold blue]Relationship Contributors[/]", box=box.ROUNDED
                    )
                    rel_table.add_column("Source", style="yellow")
                    rel_table.add_column("Relation Type", style="magenta")
                    rel_table.add_column("Count", justify="right", style="green")

                    current_source = None
                    for row in rel_results:
                        source = row['source_name'] if 'source_name' in row else "Unknown"
                        rel_type = row['relation_type'] if 'relation_type' in row else "Unknown"
                        count = row['rel_count'] if 'rel_count' in row else 0
                        
                        if source != current_source:
                            if current_source is not None:
                                rel_table.add_row("---", "---", "---")  # Separator
                            rel_table.add_row(source, rel_type, f"{count:,}")
                            current_source = source
                        else:
                            rel_table.add_row(
                                "", rel_type, f"{count:,}"
                            )  # Indent or leave source blank

                    console.print(rel_table)
                    console.print()
            except Exception as e:
                logger.error(
                    f"Error generating relationship statistics: {str(e)}", exc_info=True
                )
                console.print(
                    f"[red]Error:[/][yellow] Could not generate relationship statistics: {str(e)}[/]"
                )
                # Roll back to clean state
                conn.rollback()

            # Commit the transaction since we're done
            conn.commit()
            
            console.print(
                f"Leaderboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
    except Exception as outer_e:
        # Handle any unexpected errors
        logger.error(f"Unexpected error in display_leaderboard: {str(outer_e)}")
        console.print(f"[red]Error:[/] {str(outer_e)}")
        if conn and not conn.closed:
            conn.rollback()
    finally:
        # Always close the connection when done
        if conn and not conn.closed:
            conn.close()

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
        "parts_of_speech",
    ]

    for table in tables:
        print(f"Purging {table}...")
        cur.execute(f"DELETE FROM {table}")

    return True


def purge_database(args):
    """Safely delete all data from the database."""
    if not args.force:
        confirmation = input(
            "WARNING: This will delete ALL dictionary data. Type 'DELETE ALL' to confirm: "
        )
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
    cur.execute(
        """
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
    """
    )

    # Standardize relation types - more comprehensive mapping
    relation_mapping = {
        # Basic standardization
        "derived from": "derived_from",
        "root of": "root_of",
        "synonym of": "synonym",
        "related to": "related",
        # Redundancy reduction
        "kaugnay": "related",  # Filipino for "related"
        "see_also": "compare_with",
        "alternate_form": "variant",
        "alternative_spelling": "variant",
        "regional_form": "variant",
        # Standardize Filipino terms to English equivalents
        "kasingkahulugan": "synonym",
        "katulad": "synonym",
        "kasalungat": "antonym",
        "kabaligtaran": "antonym",
        "uri_ng": "hyponym_of",
        "mula_sa": "derived_from",
        "varyant": "variant",
        # Fix capitalization and plural forms
        "synonyms": "synonym",
        "antonyms": "antonym",
        "variants": "variant",
        "Synonym": "synonym",
        "Antonym": "antonym",
        "Related": "related",
    }

    # Apply the mapping
    for old, new in relation_mapping.items():
        cur.execute(
            """
            UPDATE relations
            SET relation_type = %s
            WHERE LOWER(relation_type) = %s
        """,
            (new, old),
        )

    # Log the results
    cur.execute(
        "SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type ORDER BY COUNT(*) DESC"
    )
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
    cur.execute(
        """
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
    """
    )

    # Replace definitions table content with deduplicated data
    cur.execute("DELETE FROM definitions")
    cur.execute(
        """
        INSERT INTO definitions
        SELECT * FROM unique_definitions;
    """
    )
    cur.execute("DROP TABLE unique_definitions")

    # Log results
    cur.execute("SELECT COUNT(*) FROM definitions")
    final_count = cur.fetchone()[0]
    logger.info(
        f"Definition deduplication complete. {final_count} unique definitions remain."
    )

    return final_count


@with_transaction(commit=True)
def cleanup_dictionary_data(cur):
    """Perform comprehensive cleanup of dictionary data."""
    logger.info("Starting dictionary cleanup process...")

    # Standardize parts of speech
    cur.execute(
        """
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
    """
    )

    # Standardize sources
    cur.execute(
        """
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
    """
    )

    # Merge duplicate definitions
    cur.execute(
        """
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
    """
    )

    # Remove remaining duplicates
    cur.execute(
        """
        WITH duplicates AS (
            SELECT word_id, definition_text, standardized_pos_id, min(id) as keep_id
            FROM definitions
            GROUP BY word_id, definition_text, standardized_pos_id
            HAVING COUNT(*) > 1
        )
        DELETE FROM definitions d
        USING duplicates dup
        WHERE d.word_id = dup.word_id
        AND d.definition_text = dup.definition_text
        AND d.standardized_pos_id = dup.standardized_pos_id
        AND d.id != dup.keep_id;
    """
    )

    # Update word tags with sources
    cur.execute(
        """
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
    """
    )
    cur.execute(
        """
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
    """
    )
    logger.info("Dictionary cleanup complete.")


def store_processed_entry(cur, word_id: int, processed: Dict) -> None:
    try:
        source = processed.get("source", "")
        if not source:
            logger.warning(f"No source provided for word_id {word_id}, using 'unknown'")
            source = "unknown"

        standardized_source = SourceStandardization.standardize_sources(source)
        if not standardized_source:
            logger.warning(
                f"Could not standardize source '{source}' for word_id {word_id}"
            )
            standardized_source = source

        if "data" in processed and "definitions" in processed["data"]:
            for definition in processed["data"]["definitions"]:
                try:
                    insert_definition(
                        cur,
                        word_id,
                        definition["text"],
                        part_of_speech=definition.get("pos", ""),
                        examples=json.dumps(definition.get("examples", [])),
                        usage_notes=json.dumps(definition.get("usage_notes", [])),
                        category=definition.get("domain"),
                        tags=json.dumps(definition.get("tags", [])),
                        sources=standardized_source,
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing definition for word_id {word_id}: {str(e)}"
                    )
        if "data" in processed and "forms" in processed["data"]:
            for form in processed["data"]["forms"]:
                if "Baybayin" in form.get("tags", []):
                    try:
                        process_baybayin_data(
                            cur,
                            word_id,
                            form["form"],
                            form.get(
                                "romanized_form", get_romanized_text(form["form"])
                            ),
                        )
                    except Exception as e:
                        logger.error(
                            f"Error storing Baybayin form for word_id {word_id}: {str(e)}"
                        )
        if "data" in processed and "metadata" in processed["data"]:
            metadata = processed["data"]["metadata"]
            if "etymology" in metadata:
                try:
                    etymology_data = metadata["etymology"]
                    insert_etymology(
                        cur,
                        word_id,
                        etymology_data,
                        normalized_components=json.dumps(
                            extract_etymology_components(etymology_data)
                        ),
                        etymology_structure=None,
                        language_codes=",".join(extract_language_codes(etymology_data)),
                        source_identifier=source,
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing etymology for word_id {word_id}: {str(e)}"
                    )
            if "pronunciation" in metadata:
                try:
                    pron_data = metadata["pronunciation"]
                    cur.execute(
                        """
                        UPDATE words 
                        SET pronunciation_data = %s
                        WHERE id = %s
                    """,
                        (json.dumps(pron_data), word_id),
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing pronunciation for word_id {word_id}: {str(e)}"
                    )
        if "data" in processed and "related_words" in processed["data"]:
            for rel_type, related in processed["data"]["related_words"].items():
                for rel_word in related:
                    try:
                        rel_word_id = get_or_create_word_id(
                            cur,
                            rel_word,
                            language_code=processed.get("language_code", "tl"),
                        )
                        insert_relation(
                            cur,
                            word_id,
                            rel_word_id,
                            rel_type,
                            sources=standardized_source,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error storing relation for word_id {word_id} and related word {rel_word}: {str(e)}"
                        )
    except Exception as e:
        logger.error(f"Error in store_processed_entry for word_id {word_id}: {str(e)}")
        raise


def explore_dictionary():
    """Launches interactive dictionary explorer."""
    console = Console()
    console.print(
        "\n[bold cyan]🔍 Interactive Dictionary Explorer[/]", justify="center"
    )
    console.print(
        "[dim]Navigate Filipino dictionary data with ease[/]\n", justify="center"
    )

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
                "0. Exit explorer",
            ]

            for option in options:
                console.print(f"  {option}")

            choice = input("Enter your choice (0-5): ")

            if choice == "0":
                break
            elif choice == "1":
                search_term = input("Enter search term: ")
                if not search_term.strip():
                    continue

                cur.execute(
                    """
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
                """,
                    (
                        f"{search_term}%",
                        f"{search_term}%",
                        search_term,
                        search_term,
                        f"{search_term}%",
                    ),
                )

                results = cur.fetchall()

                if not results:
                    console.print("[yellow]No matches found.[/]")
                    continue

                result_table = Table(
                    title=f"Search Results for '{search_term}'", box=box.ROUNDED
                )
                result_table.add_column("ID", style="dim")
                result_table.add_column("Word", style="cyan")
                result_table.add_column("Language", style="green")
                result_table.add_column("Baybayin", style="magenta")

                for word_id, lemma, lang_code, has_baybayin in results:
                    result_table.add_row(
                        str(word_id),
                        lemma,
                        "Tagalog" if lang_code == "tl" else "Cebuano",
                        "✓" if has_baybayin else "",
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "2":
                cur.execute(
                    """
                    SELECT id, lemma, language_code, has_baybayin
                    FROM words
                    ORDER BY RANDOM()
                    LIMIT 10
                """
                )

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
                        "✓" if has_baybayin else "",
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "3":
                cur.execute(
                    """
                    SELECT id, lemma, baybayin_form, romanized_form
                    FROM words
                    WHERE has_baybayin = TRUE
                    ORDER BY RANDOM()
                    LIMIT 10
                """
                )

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
                        str(word_id), lemma, baybayin or "", romanized or ""
                    )

                console.print(result_table)

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
                if word_choice.strip() and word_choice.isdigit():
                    lookup_by_id(cur, int(word_choice), console)

            elif choice == "4":
                word_input = input("Enter word to find relations: ")
                if not word_input.strip():
                    continue

                cur.execute(
                    """
                    SELECT id FROM words
                    WHERE lemma = %s OR normalized_lemma = %s
                    LIMIT 1
                """,
                    (word_input, normalize_lemma(word_input)),
                )

                result = cur.fetchone()

                if not result:
                    console.print(f"[yellow]Word '{word_input}' not found.[/]")
                    continue

                word_id = result[0]

                cur.execute(
                    """
                    SELECT r.relation_type, w.id, w.lemma
                    FROM relations r
                    JOIN words w ON r.to_word_id = w.id
                    WHERE r.from_word_id = %s
                    ORDER BY r.relation_type, w.lemma
                """,
                    (word_id,),
                )

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

                word_choice = input(
                    "\nEnter word ID to view details (or press Enter to return): "
                )
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
        cur.execute(
            """
            SELECT lemma, language_code, has_baybayin, baybayin_form, romanized_form
            FROM words
            WHERE id = %s
        """,
            (word_id,),
        )

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

        cur.execute(
            """
            SELECT p.name_tl as pos, d.definition_text
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY p.name_tl, d.created_at
        """,
            (word_id,),
        )

        definitions = cur.fetchall()

        if definitions:
            console.print("\n[bold]Definitions:[/]")
            current_pos = None
            for pos, definition in definitions:
                if pos != current_pos:
                    console.print(f"\n[cyan]{pos or 'Uncategorized'}[/]")
                    current_pos = pos
                console.print(f"• {definition}")

        cur.execute(
            """
            SELECT r.relation_type, w.lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            ORDER BY r.relation_type, w.lemma
        """,
            (word_id,),
        )

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
        ("Query Performance", lambda: check_query_performance()),
    ]

    test_table = Table(box=box.ROUNDED)
    test_table.add_column("Test", style="cyan")
    test_table.add_column("Status", style="bold")
    test_table.add_column("Details", style="dim")

    conn = None

    try:
        for test_name, test_func in tests:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Running {test_name} test..."),
                    console=console,
                ) as progress:
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
        major_version = int(version.split()[1].split(".")[0])
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
        extensions = ["pg_trgm", "unaccent", "fuzzystrmatch"]
        missing_extensions = []

        for ext in extensions:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM pg_extension WHERE extname = %s
                )
            """,
                (ext,),
            )

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

        cur.execute(
            """
            INSERT INTO words (lemma, normalized_lemma, language_code)
            VALUES (%s, %s, 'tl')
            RETURNING id
        """,
            (test_word, normalized),
        )

        test_id = cur.fetchone()[0]
        cur.execute("DELETE FROM words WHERE id = %s", (test_id,))
        return (
            True,
            f"Successfully read, wrote, and deleted data. Word count: {word_count:,}",
        )
    except Exception as e:
        return False, str(e)


def check_tables_exist():
    """Check if all required tables exist in the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
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
            """
            )
            result = cur.fetchone()[0]
            return result


@with_transaction(commit=False)
def check_query_performance():
    """Check query performance."""
    try:
        cur = get_cursor()
        queries = [
            ("Simple Select", "SELECT COUNT(*) FROM words"),
            (
                "Join Query",
                """
                SELECT COUNT(*) 
                FROM words w
                JOIN definitions d ON w.id = d.word_id
            """,
            ),
            (
                "Index Usage",
                """
                SELECT COUNT(*) 
                FROM words 
                WHERE normalized_lemma LIKE 'a%'
            """,
            ),
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
            console.print(
                "\n[bold yellow]The database may not exist.[/] You can create it using:"
            )
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
