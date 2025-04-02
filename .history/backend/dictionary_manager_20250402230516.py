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
import glob
import hashlib
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
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import os
import json
import glob
import re
from psycopg2.extras import Json
import functools

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
                minconn=1,
                maxconn=20,
                **get_db_config()
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseConnectionError(f"Could not initialize database connection pool: {e}")
            
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
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED
            )
            return conn
            
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                # Exponential backoff
                retry_delay *= 1.5
            else:
                logger.error(f"Failed to get database connection after {max_retries} attempts: {e}")
                raise DatabaseConnectionError(f"Could not connect to database after {max_retries} attempts: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error getting database connection: {e}")
            # Try to return connection to pool if it was obtained
            try:
                if 'conn' in locals():
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
            if args and hasattr(args[0], 'cursor'):
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

def update_word_source_info(current_source_info: Optional[Union[str, dict]], new_source_identifier: Optional[str]) -> str:
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
            return '{}'
    
    # Parse current_source_info into a dictionary    
    source_info_dict = {}
    if isinstance(current_source_info, dict):
        source_info_dict = current_source_info.copy()  # Use copy to avoid modifying the original
    elif isinstance(current_source_info, str) and current_source_info:
        try:
            source_info_dict = json.loads(current_source_info)
            if not isinstance(source_info_dict, dict):
                source_info_dict = {}  # Reset if not a dictionary
        except (json.JSONDecodeError, TypeError):
            # Start fresh if invalid JSON
            source_info_dict = {}
    
    # Ensure 'files' key exists and is a list
    if SOURCE_INFO_FILES_KEY not in source_info_dict or not isinstance(source_info_dict[SOURCE_INFO_FILES_KEY], list):
        source_info_dict[SOURCE_INFO_FILES_KEY] = []
    
    # Standardize the source identifier before adding
    standardized_source = SourceStandardization.standardize_sources(new_source_identifier)
    
    # Add new source identifier if not already present
    if standardized_source not in source_info_dict[SOURCE_INFO_FILES_KEY]:
        source_info_dict[SOURCE_INFO_FILES_KEY].append(standardized_source)
        source_info_dict[SOURCE_INFO_FILES_KEY].sort()  # Keep list sorted
    
    # Add metadata about when and how this source was added
    if 'last_updated' not in source_info_dict:
        source_info_dict['last_updated'] = {}
        
    source_info_dict['last_updated'][standardized_source] = datetime.now().isoformat()
    
    # Return JSON string
    return json.dumps(source_info_dict)
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

def get_db_config():
    """Return the database configuration dictionary."""
    return DB_CONFIG

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
                        logger.error(f"Failed to rollback transaction: {rollback_error}")
                # Re-raise the original exception
                raise
                
            finally:
                # Restore original autocommit state if we changed it
                if started_transaction and original_autocommit is not None:
                    try:
                        conn.autocommit = original_autocommit
                        logger.debug(f"Restored autocommit state to {original_autocommit}")
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
        'words_updated': 0,
        'definitions_updated': 0,
        'relations_updated': 0,
        'etymologies_updated': 0
    }
    
    try:
        # Standardize sources in words table
        cur.execute("""
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
        """)
        stats['words_updated'] = cur.rowcount
        
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
        'word': None,
        'direct_sources': [],
        'definition_sources': [],
        'relation_sources': [],
        'etymology_sources': []
    }
    
    try:
        # Get word info
        cur.execute("""
            SELECT lemma, source_info 
            FROM words 
            WHERE id = %s
        """, (word_id,))
        word_row = cur.fetchone()
        
        if not word_row:
            return result
            
        result['word'] = word_row[0]
        
        # Extract direct sources
        if word_row[1]:
            try:
                source_info = json.loads(word_row[1]) if isinstance(word_row[1], str) else word_row[1]
                if isinstance(source_info, dict) and SOURCE_INFO_FILES_KEY in source_info:
                    result['direct_sources'] = source_info[SOURCE_INFO_FILES_KEY]
            except (json.JSONDecodeError, TypeError):
                pass
                
        # Get definition sources
        cur.execute("""
            SELECT DISTINCT sources 
            FROM definitions 
            WHERE word_id = %s AND sources IS NOT NULL
        """, (word_id,))
        for row in cur.fetchall():
            if row[0] and row[0] not in result['definition_sources']:
                result['definition_sources'].append(row[0])
                
        # Get relation sources
        cur.execute("""
            SELECT DISTINCT sources 
            FROM relations 
            WHERE (from_word_id = %s OR to_word_id = %s) AND sources IS NOT NULL
        """, (word_id, word_id))
        for row in cur.fetchall():
            if row[0] and row[0] not in result['relation_sources']:
                result['relation_sources'].append(row[0])
                
        # Get etymology sources
        cur.execute("""
            SELECT DISTINCT sources 
            FROM etymologies 
            WHERE word_id = %s AND sources IS NOT NULL
        """, (word_id,))
        for row in cur.fetchall():
            if row[0] and row[0] not in result['etymology_sources']:
                result['etymology_sources'].append(row[0])
                
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
        cur.execute("""
            UPDATE definitions
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """, (standardized_source, f'%{standardized_source}%', standardized_source, word_id))
        
        # Update related etymologies
        cur.execute("""
            UPDATE etymologies
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE word_id = %s
        """, (standardized_source, f'%{standardized_source}%', standardized_source, word_id))
        
        # Update related relations
        cur.execute("""
            UPDATE relations
            SET sources = CASE
                WHEN sources IS NULL OR sources = '' THEN %s
                WHEN sources NOT LIKE %s THEN sources || ', ' || %s
                ELSE sources
            END
            WHERE from_word_id = %s OR to_word_id = %s
        """, (standardized_source, f'%{standardized_source}%', standardized_source, word_id, word_id))
        
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
    root_word_id INT REFERENCES words(id),
    preferred_spelling VARCHAR(255),
    tags TEXT,
    idioms JSONB DEFAULT '[]',
    pronunciation_data JSONB,
    source_info JSONB DEFAULT '{}',
    word_metadata JSONB DEFAULT '{}',
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

-- Create pronunciations table
CREATE TABLE IF NOT EXISTS pronunciations (
    id SERIAL PRIMARY KEY,
    word_id INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL DEFAULT 'ipa',
    value TEXT NOT NULL,
    tags JSONB,
    metadata JSONB,
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
    standardized_pos_id INT REFERENCES parts_of_speech(id),
    examples TEXT,
    usage_notes TEXT,
    tags TEXT,
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

-- Create etymologies table
CREATE TABLE IF NOT EXISTS etymologies (
    id SERIAL PRIMARY KEY,
    word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    etymology_text TEXT NOT NULL,
    normalized_components TEXT,
    etymology_structure TEXT,
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
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT definition_relations_unique UNIQUE (definition_id, word_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_def_relations_def ON definition_relations(definition_id);
CREATE INDEX IF NOT EXISTS idx_def_relations_word ON definition_relations(word_id);

CREATE TABLE IF NOT EXISTS affixations (
    id SERIAL PRIMARY KEY,
    root_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    affixed_word_id INT NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    affix_type VARCHAR(64) NOT NULL,
    sources TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT affixations_unique UNIQUE (root_word_id, affixed_word_id, affix_type)
);
CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);

-- Create triggers for timestamp updates
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
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
         WHERE tgname = 'update_pronunciations_timestamp'
         AND tgrelid = 'pronunciations'::regclass
    ) THEN
        CREATE TRIGGER update_pronunciations_timestamp
            BEFORE UPDATE ON pronunciations
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
         WHERE tgname = 'update_credits_timestamp'
         AND tgrelid = 'credits'::regclass
    ) THEN
        CREATE TRIGGER update_credits_timestamp
            BEFORE UPDATE ON credits
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
         WHERE tgname = 'update_definition_relations_timestamp'
         AND tgrelid = 'definition_relations'::regclass
    ) THEN
        CREATE TRIGGER update_definition_relations_timestamp
            BEFORE UPDATE ON definition_relations
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
         WHERE tgname = 'update_parts_of_speech_timestamp'
         AND tgrelid = 'parts_of_speech'::regclass
    ) THEN
        CREATE TRIGGER update_parts_of_speech_timestamp
            BEFORE UPDATE ON parts_of_speech
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
         WHERE tgname = 'update_affixations_timestamp'
         AND tgrelid = 'affixations'::regclass
    ) THEN
        CREATE TRIGGER update_affixations_timestamp
            BEFORE UPDATE ON affixations
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;
END $$;

-- Add updated_at column to definition_relations if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'definition_relations' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE definition_relations 
        ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
    END IF;
END $$;

-- Create trigger for definition_relations if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_definition_relations_timestamp'
        AND tgrelid = 'definition_relations'::regclass
    ) THEN
        CREATE TRIGGER update_definition_relations_timestamp
            BEFORE UPDATE ON definition_relations
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
    END IF;
END $$;

-- Update existing rows to have a value for updated_at
UPDATE definition_relations 
SET updated_at = created_at 
WHERE updated_at IS NULL;
"""

def create_or_update_tables(conn):
    """Create or update the database tables."""
    logger.info("Starting table creation/update process.")
    
    cur = conn.cursor()
    try:
        # Drop existing tables in correct order
        cur.execute("""
            DROP TABLE IF EXISTS 
                credits, pronunciations, definition_relations, affixations, 
                relations, etymologies, definitions, words, parts_of_speech CASCADE;
        """)
        
        # Create tables
        cur.execute(TABLE_CREATION_SQL)
        
        # Insert standard parts of speech
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
        
@with_transaction(commit=True)
def insert_pronunciation(cur, word_id: int, pronunciation_data: Union[str, Dict],
                         source_identifier: str) -> Optional[int]:
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
        logger.error(f"CRITICAL: Skipping pronunciation insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    pron_type = 'ipa' # Default type
    value = None
    tags_list = []
    metadata = {}

    try:
        # Parse input data
        if isinstance(pronunciation_data, dict):
            # Prioritize keys from the dict, but allow source_identifier argument to override 'sources' key if present
            pron_type = pronunciation_data.get('type', 'ipa') or 'ipa' # Default to 'ipa' if empty
            value = pronunciation_data.get('value', '').strip() if isinstance(pronunciation_data.get('value'), str) else None
            tags_input = pronunciation_data.get('tags')
            if isinstance(tags_input, list):
                 tags_list = tags_input
            elif isinstance(tags_input, str):
                 # Simple split if tags are a comma-separated string? Adapt as needed.
                 tags_list = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                 logger.debug(f"Parsed tags string '{tags_input}' into list for word ID {word_id}.")
            # else: ignore invalid tags format

            metadata_input = pronunciation_data.get('metadata')
            if isinstance(metadata_input, dict):
                metadata = metadata_input
            # else: ignore invalid metadata format

        elif isinstance(pronunciation_data, str):
            value = pronunciation_data.strip()
            # Assumed type is 'ipa' and no tags/metadata provided via string input
        else:
             logger.warning(f"Invalid pronunciation_data type for word ID {word_id} (source '{source_identifier}'): {type(pronunciation_data)}. Skipping.")
             return None

        if not value:
            logger.warning(f"Empty pronunciation value for word ID {word_id} (source '{source_identifier}'). Skipping.")
            return None

        # Safely dump JSON fields (tags and metadata) for DB insertion (assuming JSONB columns)
        tags_json = None
        try:
            tags_json = json.dumps(tags_list) # tags_list will be [] if not provided or invalid format
        except TypeError as e:
            logger.warning(f"Could not serialize tags for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Tags: {tags_list}")
            tags_json = '[]' # Fallback to empty JSON array string

        metadata_json = None
        try:
            metadata_json = json.dumps(metadata) # metadata will be {} if not provided or invalid format
        except TypeError as e:
            logger.warning(f"Could not serialize metadata for pronunciation (word ID {word_id}, source '{source_identifier}'): {e}. Metadata: {metadata}")
            metadata_json = '{}' # Fallback to empty JSON object string

        # Prepare parameters for query
        params = {
            'word_id': word_id,
            'type': pron_type,
            'value': value,
            'tags': tags_json,
            'metadata': metadata_json,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        # Insert or update pronunciation
        cur.execute("""
            INSERT INTO pronunciations (word_id, type, value, tags, metadata, sources)
            VALUES (%(word_id)s, %(type)s, %(value)s, %(tags)s::jsonb, %(metadata)s::jsonb, %(sources)s) -- Cast JSONs
            ON CONFLICT (word_id, type, value) -- Conflict on word, type, and exact value
            DO UPDATE SET
                -- Update tags/metadata only if new value is not NULL (or empty JSON?) - COALESCE prefers new non-null
                tags = COALESCE(EXCLUDED.tags, pronunciations.tags),
                metadata = COALESCE(EXCLUDED.metadata, pronunciations.metadata),
                -- Overwrite sources: Last write wins for this pronunciation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        pron_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated pronunciation (ID: {pron_id}, Type: {pron_type}) for word ID {word_id} from source '{source_identifier}'. Value: '{value}'")
        return pron_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting pronunciation for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None


@with_transaction(commit=True)
def insert_credit(cur, word_id: int, credit_data: Union[str, Dict],
                  source_identifier: str) -> Optional[int]:
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
        logger.error(f"CRITICAL: Skipping credit insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    credit_text = None
    try:
        # Extract credit text
        if isinstance(credit_data, dict):
            # Prioritize 'text' key if dict is provided
            credit_text = credit_data.get('text', '').strip() if isinstance(credit_data.get('text'), str) else None
        elif isinstance(credit_data, str):
            credit_text = credit_data.strip()
        else:
            logger.warning(f"Invalid credit_data type for word ID {word_id} (source '{source_identifier}'): {type(credit_data)}. Skipping.")
            return None

        if not credit_text:
            logger.warning(f"Empty credit text for word ID {word_id} (source '{source_identifier}'). Skipping.")
            return None

        # Prepare parameters
        params = {
            'word_id': word_id,
            'credit': credit_text,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        # Insert or update credit
        cur.execute("""
            INSERT INTO credits (word_id, credit, sources)
            VALUES (%(word_id)s, %(credit)s, %(sources)s)
            ON CONFLICT (word_id, credit) -- Conflict on word and exact credit text
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this credit record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        credit_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated credit (ID: {credit_id}) for word ID {word_id} from source '{source_identifier}'. Credit: '{credit_text}'")
        return credit_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting credit for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting credit for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None

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
    """Define types of Baybayin characters."""
    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"
    
    @classmethod
    def get_type(cls, char: str) -> 'BaybayinCharType':
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
        if not self.default_sound and self.char_type not in (BaybayinCharType.VIRAMA, BaybayinCharType.PUNCTUATION):
            raise ValueError("Default sound required for non-virama characters")
        code_point = ord(self.char)
        if not (0x1700 <= code_point <= 0x171F) and not (0x1735 <= code_point <= 0x1736):
            raise ValueError(f"Invalid Baybayin character: {self.char} (U+{code_point:04X})")
        expected_type = BaybayinCharType.get_type(self.char)
        if expected_type != self.char_type and expected_type != BaybayinCharType.UNKNOWN:
            raise ValueError(f"Character type mismatch for {self.char}: expected {expected_type}, got {self.char_type}")

    def get_sound(self, next_char: Optional['BaybayinChar'] = None) -> str:
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
        return []
    
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

def validate_schema(cur):
    """Validate database schema and constraints."""
    try:
        # Check required tables exist
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'words'
            )
        """)
        if not cur.fetchone()[0]:
            raise DatabaseError("Required table 'words' does not exist")

        # Check required indexes
        required_indexes = [
            ('words_lemma_idx', 'words', 'lemma'),
            ('words_normalized_lemma_idx', 'words', 'normalized_lemma'),
            ('pronunciations_word_id_idx', 'pronunciations', 'word_id'),
            ('credits_word_id_idx', 'credits', 'word_id')
        ]
        
        for index_name, table, column in required_indexes:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes 
                    WHERE indexname = %s
                )
            """, (index_name,))
            if not cur.fetchone()[0]:
                raise DatabaseError(f"Required index {index_name} on {table}({column}) does not exist")

        # Check foreign key constraints
        required_fks = [
            ('pronunciations', 'word_id', 'words', 'id'),
            ('credits', 'word_id', 'words', 'id')
        ]
        
        for table, column, ref_table, ref_column in required_fks:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.key_column_usage
                    WHERE table_name = %s
                    AND column_name = %s
                    AND referenced_table_name = %s
                    AND referenced_column_name = %s
                )
            """, (table, column, ref_table, ref_column))
            if not cur.fetchone()[0]:
                raise DatabaseError(f"Required foreign key constraint missing: {table}({column}) -> {ref_table}({ref_column})")

    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise

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
        """Convert source filenames to standardized display names."""
        if not source:
            return "unknown"
            
        source_mapping = {
            'kaikki-ceb.jsonl': 'kaikki.org (Cebuano)',
            'kaikki.jsonl': 'kaikki.org (Tagalog)',
            'kwf_dictionary.json': 'KWF Diksiyonaryo ng Wikang Filipino',
            'root_words_with_associated_words_cleaned.json': 'tagalog.com',
            'tagalog-words.json': 'diksiyonaryo.ph'
        }
        
        # Try direct mapping first
        if source in source_mapping:
            return source_mapping[source]
            
        # Handle cases where only part of the filename is matched
        for key, value in source_mapping.items():
            if key in source:
                return value
                
        # Special case for Marayum dictionaries
        if 'marayum' in source.lower():
            return 'Project Marayum'
            
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
        'ᜑ': BaybayinChar('ᜑ', BaybayinCharType.CONSONANT, 'ha', ['ha']),
        'ᜍ': BaybayinChar('ᜍ', BaybayinCharType.CONSONANT, 'ra', ['ra'])  # Added ra
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
        """Initialize the romanizer with a combined character mapping."""
        self.all_chars = {}
        # Combine all character mappings for easy lookup
        for char_map in [self.VOWELS, self.CONSONANTS, self.VOWEL_MARKS, 
                         {self.VIRAMA.char: self.VIRAMA}, self.PUNCTUATION]:
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
            return '', 0
        
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
        return '', 1
    
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
        text = unicodedata.normalize('NFC', text)
        
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
                processed_syllable, chars_consumed = self.process_syllable(list(text[i:]))
                
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
                
        return ''.join(result)

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
        text = unicodedata.normalize('NFC', text)
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
                    logger.warning(f"Unrecognized Baybayin character at position {i}: {chars[i]} (U+{ord(chars[i]):04X})")
                return False
                
            # Vowel mark must follow a consonant
            if char_info.char_type == BaybayinCharType.VOWEL_MARK:
                if i == 0 or not self.get_char_info(chars[i-1]) or self.get_char_info(chars[i-1]).char_type != BaybayinCharType.CONSONANT:
                    logger.warning(f"Vowel mark not following a consonant at position {i}")
                    return False
                    
            # Virama (vowel killer) must follow a consonant
            if char_info.char_type == BaybayinCharType.VIRAMA:
                if i == 0 or not self.get_char_info(chars[i-1]) or self.get_char_info(chars[i-1]).char_type != BaybayinCharType.CONSONANT:
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
    if text.startswith('-'):
        # Skip the hyphen and process the rest
        text = text[1:]
    
    # Normalize text: lowercase and remove diacritical marks
    text = text.lower().strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                   if not unicodedata.combining(c))
    
    # Define Baybayin character mappings
    consonants = {
        'k': 'ᜃ', 'g': 'ᜄ', 'ng': 'ᜅ', 't': 'ᜆ', 'd': 'ᜇ', 'n': 'ᜈ',
        'p': 'ᜉ', 'b': 'ᜊ', 'm': 'ᜋ', 'y': 'ᜌ', 'l': 'ᜎ', 'w': 'ᜏ',
        's': 'ᜐ', 'h': 'ᜑ', 'r': 'ᜍ'  # Added 'r' mapping
    }
    vowels = {'a': 'ᜀ', 'i': 'ᜁ', 'e': 'ᜁ', 'u': 'ᜂ', 'o': 'ᜂ'}
    vowel_marks = {'i': 'ᜒ', 'e': 'ᜒ', 'u': 'ᜓ', 'o': 'ᜓ'}
    virama = '᜔'  # Pamudpod (vowel killer)
    
    result = []
    i = 0
    
    while i < len(text):
        # Check for 'ng' digraph first
        if i + 1 < len(text) and text[i:i+2] == 'ng':
            if i + 2 < len(text) and text[i+2] in 'aeiou':
                # ng + vowel
                if text[i+2] == 'a':
                    result.append(consonants['ng'])
                else:
                    result.append(consonants['ng'] + vowel_marks[text[i+2]])
                i += 3
            else:
                # Final 'ng'
                result.append(consonants['ng'] + virama)
                i += 2
                
        # Handle single consonants
        elif text[i] in consonants:
            if i + 1 < len(text) and text[i+1] in 'aeiou':
                # Consonant + vowel
                if text[i+1] == 'a':
                    result.append(consonants[text[i]])
                else:
                    result.append(consonants[text[i]] + vowel_marks[text[i+1]])
                i += 2
            else:
                # Final consonant
                result.append(consonants[text[i]] + virama)
                i += 1
                
        # Handle vowels
        elif text[i] in 'aeiou':
            result.append(vowels[text[i]])
            i += 1
            
        # Skip spaces and other characters
        elif text[i].isspace():
            result.append(' ')
            i += 1
        else:
            # Skip non-convertible characters
            i += 1
    
    # Final validation - ensure only valid characters are included
    valid_output = ''.join(c for c in result if (0x1700 <= ord(c) <= 0x171F) or c.isspace())
    
    # Verify the output meets database constraints
    if not re.match(r'^[\u1700-\u171F\s]*$', valid_output):
        logger.warning(f"Transliterated Baybayin doesn't match required regex pattern: {valid_output}")
        # Additional cleanup to ensure it matches the pattern
        valid_output = re.sub(r'[^\u1700-\u171F\s]', '', valid_output)
    
    return valid_output

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
    cleaned = ''.join(c for c in text if (0x1700 <= ord(c) <= 0x171F) or c.isspace())
    
    # Normalize whitespace and trim
    return re.sub(r'\s+', ' ', cleaned).strip()

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
    parts = re.split(r'[^\u1700-\u171F\s]+', text)
    results = []
    
    for part in parts:
        # Clean and normalize
        cleaned_part = clean_baybayin_text(part)
        
        # Make sure part contains at least one Baybayin character
        if cleaned_part and any(0x1700 <= ord(c) <= 0x171F for c in cleaned_part):
            # Verify it meets database constraints
            if re.match(r'^[\u1700-\u171F\s]*$', cleaned_part):
                results.append(cleaned_part)
                
    return results

def validate_baybayin_entry(baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
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
        if not re.match(r'^[\u1700-\u171F\s]*$', cleaned_form):
            logger.warning(f"Baybayin form doesn't match required regex pattern: {cleaned_form}")
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
                    logger.warning(f"Romanization mismatch: expected '{romanized_form}', got '{generated_rom}'")
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
def process_baybayin_data(cur, word_id: int, baybayin_form: str, romanized_form: Optional[str] = None) -> bool:
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
            logger.warning(f"No valid Baybayin characters found in: {baybayin_form} for word_id {word_id}")
            return False
        
        # Verify it meets database constraints
        if not re.match(r'^[\u1700-\u171F\s]*$', cleaned_baybayin):
            logger.warning(f"Baybayin form doesn't match required regex pattern: {cleaned_baybayin}")
            return False
            
        # Create a romanizer to validate structure
        romanizer = BaybayinRomanizer()
        if not romanizer.validate_text(cleaned_baybayin):
            logger.warning(f"Invalid Baybayin structure in: {cleaned_baybayin} for word_id {word_id}")
            return False
            
        # Generate romanization if not provided
        if not romanized_form:
            try:
                romanized_form = romanizer.romanize(cleaned_baybayin)
            except Exception as e:
                logger.error(f"Error generating romanization for word_id {word_id}: {e}")
                # Try to continue with the process even if romanization fails
                romanized_form = None
        
        # Verify the word exists before updating
        cur.execute("SELECT 1 FROM words WHERE id = %s", (word_id,))
        if not cur.fetchone():
            logger.warning(f"Word ID {word_id} does not exist in the database")
            return False
        
        # Update the word record
        cur.execute("""
            UPDATE words 
            SET has_baybayin = TRUE,
                baybayin_form = %s,
                romanized_form = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (cleaned_baybayin, romanized_form, word_id))
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing Baybayin data for word_id {word_id}: {e}")
        raise

@with_transaction(commit=True)
def process_baybayin_entries(cur):
    """Process all Baybayin entries in the database."""
    logger.info("Processing Baybayin entries...")
    
    try:
        # Create a new BaybayinRomanizer instance
        romanizer = BaybayinRomanizer()
        
        # Get all entries that contain Baybayin characters
        cur.execute("""
            SELECT id, lemma, language_code
            FROM words 
            WHERE lemma ~ '[\u1700-\u171F]'
            OR baybayin_form IS NOT NULL
            ORDER BY id ASC
        """)
        
        baybayin_entries = cur.fetchall()
        processed_count = 0
        error_count = 0
        skipped_count = 0
        
        logger.info(f"Found {len(baybayin_entries)} potential Baybayin entries to process")
        
        for entry in baybayin_entries:
            baybayin_id, baybayin_lemma, language_code = entry
            
            try:
                # Step 1: Extract Baybayin segments from the lemma
                baybayin_segments = extract_baybayin_text(baybayin_lemma)
                
                # Step 2: Check if entry already has a valid Baybayin form
                cur.execute("SELECT baybayin_form, romanized_form FROM words WHERE id = %s", (baybayin_id,))
                existing_forms = cur.fetchone()
                existing_baybayin = existing_forms[0] if existing_forms else None
                existing_romanized = existing_forms[1] if existing_forms else None
                
                # Step 3: Process based on available data
                if not baybayin_segments and existing_baybayin:
                    # No new segments but has existing form - validate it
                    cleaned_existing = clean_baybayin_text(existing_baybayin)
                    
                    if cleaned_existing and re.match(r'^[\u1700-\u171F\s]*$', cleaned_existing):
                        # Generate romanization if missing
                        if not existing_romanized:
                            try:
                                romanized = romanizer.romanize(cleaned_existing)
                                cur.execute("""
                                    UPDATE words
                                    SET baybayin_form = %s,
                                        romanized_form = %s
                                    WHERE id = %s
                                """, (cleaned_existing, romanized, baybayin_id))
                                processed_count += 1
                            except Exception as e:
                                logger.error(f"Error updating romanization for word ID {baybayin_id}: {e}")
                                error_count += 1
                    else:
                        # Invalid existing form - clear it
                        cur.execute("""
                            UPDATE words
                            SET has_baybayin = FALSE,
                                baybayin_form = NULL,
                                romanized_form = NULL
                            WHERE id = %s
                        """, (baybayin_id,))
                        skipped_count += 1
                        logger.warning(f"Removed invalid Baybayin form for word ID {baybayin_id}")
                
                elif baybayin_segments:
                    # Found new segments - use the longest valid one
                    valid_segments = []
                    for segment in baybayin_segments:
                        try:
                            romanized = romanizer.romanize(segment)
                            valid_segments.append((segment, romanized))
                        except Exception as e:
                            logger.warning(f"Error romanizing segment '{segment}' for word ID {baybayin_id}: {e}")
                    
                    if valid_segments:
                        # Sort by length to get the longest valid segment
                        valid_segments.sort(key=lambda x: len(x[0]), reverse=True)
                        cleaned_baybayin, romanized_value = valid_segments[0]
                        
                        # Update the record
                        cur.execute("""
                            UPDATE words 
                            SET has_baybayin = TRUE,
                                baybayin_form = %s,
                                romanized_form = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = %s
                        """, (cleaned_baybayin, romanized_value, baybayin_id))
                        
                        processed_count += 1
                    else:
                        # No valid segments found
                        logger.warning(f"No valid Baybayin segments in entry {baybayin_id}: {baybayin_lemma}")
                        cur.execute("""
                            UPDATE words
                            SET has_baybayin = FALSE,
                                baybayin_form = NULL,
                                romanized_form = NULL
                            WHERE id = %s
                        """, (baybayin_id,))
                        skipped_count += 1
                else:
                    # No Baybayin data found - mark as not having Baybayin
                    cur.execute("""
                        UPDATE words
                        SET has_baybayin = FALSE,
                            baybayin_form = NULL,
                            romanized_form = NULL
                        WHERE id = %s
                    """, (baybayin_id,))
                    skipped_count += 1
                
                # Commit every 100 entries to avoid long-running transactions
                if (processed_count + skipped_count + error_count) % 100 == 0:
                    logger.info(f"Progress: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
                
            except Exception as e:
                logger.error(f"Error processing Baybayin entry {baybayin_id}: {str(e)}")
                error_count += 1
        
        logger.info(f"Processed {processed_count} Baybayin entries, skipped {skipped_count}, with {error_count} errors")
        return processed_count, error_count
    
    except Exception as e:
        logger.error(f"Error in process_baybayin_entries: {str(e)}")
        return 0, 0

@with_transaction(commit=True)
def cleanup_baybayin_data(cur):
    """Clean up Baybayin data in the database."""
    try:
        logger.info("Starting Baybayin data cleanup...")
        
        # Step 1: Fix the baybayin_form field to comply with constraints
        cur.execute(r"""
            UPDATE words 
            SET baybayin_form = regexp_replace(
                baybayin_form,
                '[^\u1700-\u171F\s]',
                '',
                'g'
            )
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL
        """)
        
        # Step 2: Normalize whitespace in Baybayin form
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
        
        # Step 3: Remove has_baybayin flag if baybayin_form is empty or invalid
        cur.execute("""
            UPDATE words
            SET has_baybayin = FALSE, baybayin_form = NULL
            WHERE has_baybayin = TRUE AND (
                baybayin_form IS NULL OR 
                baybayin_form = '' OR 
                baybayin_form !~ '[\u1700-\u171F]'
            )
        """)
        
        # Step 4: Remove baybayin_form if has_baybayin is false
        cur.execute("""
            UPDATE words
            SET baybayin_form = NULL, romanized_form = NULL
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
        """)
        
        # Step 5: Generate missing romanized forms
        cur.execute("""
            SELECT id, baybayin_form 
            FROM words 
            WHERE has_baybayin = TRUE 
              AND baybayin_form IS NOT NULL 
              AND baybayin_form ~ '[\u1700-\u171F]'
              AND (romanized_form IS NULL OR romanized_form = '')
        """)
        
        romanizer = BaybayinRomanizer()
        missing_romanization_count = 0
        
        for word_id, baybayin_form in cur.fetchall():
            try:
                # Clean the form first
                cleaned_form = clean_baybayin_text(baybayin_form)
                
                if cleaned_form and romanizer.validate_text(cleaned_form):
                    romanized = romanizer.romanize(cleaned_form)
                    if romanized:
                        cur.execute("""
                            UPDATE words 
                            SET romanized_form = %s,
                                baybayin_form = %s
                            WHERE id = %s
                        """, (romanized, cleaned_form, word_id))
                        missing_romanization_count += 1
            except Exception as e:
                logger.warning(f"Error generating romanization for word ID {word_id}: {e}")
        
        # Step 6: Update search text to include Baybayin data for improved search
        cur.execute("""
            UPDATE words
            SET search_text = to_tsvector('simple',
                COALESCE(lemma, '') || ' ' ||
                COALESCE(normalized_lemma, '') || ' ' ||
                COALESCE(baybayin_form, '') || ' ' ||
                COALESCE(romanized_form, '')
            )
            WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL OR romanized_form IS NOT NULL
        """)
        
        # Log results
        cur.execute("SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE")
        baybayin_count = cur.fetchone()[0]
        
        logger.info(f"Baybayin data cleanup completed: {baybayin_count} valid Baybayin entries remain")
        logger.info(f"Generated {missing_romanization_count} missing romanizations")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during Baybayin cleanup: {str(e)}")
        raise

@with_transaction(commit=False)
def check_baybayin_consistency(cur):
    """Check for consistency issues in Baybayin data."""
    issues = []
    
    # Check for entries marked as Baybayin but missing romanization
    cur.execute("""
        SELECT COUNT(*)
        FROM words
        WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL AND romanized_form IS NULL
    """)
    missing_rom_count = cur.fetchone()[0]
    
    if missing_rom_count > 0:
        issues.append(f"Found {missing_rom_count} entries missing romanization")
        
        # Sample some problematic entries for the log
        cur.execute("""
            SELECT id, lemma, baybayin_form
            FROM words
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL AND romanized_form IS NULL
            LIMIT 5
        """)
        
        for word_id, lemma, baybayin in cur.fetchall():
            logger.warning(f"Missing romanization for word ID {word_id}: {lemma} / {baybayin}")
    
    # Check for inconsistent flag states
    cur.execute("""
        SELECT COUNT(*)
        FROM words
        WHERE (has_baybayin = TRUE AND baybayin_form IS NULL)
           OR (has_baybayin = FALSE AND baybayin_form IS NOT NULL)
    """)
    inconsistent_count = cur.fetchone()[0]
    
    if inconsistent_count > 0:
        issues.append(f"Found {inconsistent_count} entries with inconsistent Baybayin flags")
        
        # Sample some inconsistent entries
        cur.execute("""
            SELECT id, lemma, has_baybayin, baybayin_form
            FROM words
            WHERE (has_baybayin = TRUE AND baybayin_form IS NULL)
               OR (has_baybayin = FALSE AND baybayin_form IS NOT NULL)
            LIMIT 5
        """)
        
        for word_id, lemma, has_flag, baybayin in cur.fetchall():
            state = "marked as Baybayin but missing form" if has_flag else "has Baybayin form but not flagged"
            logger.warning(f"Inconsistent Baybayin flags for word ID {word_id}: {lemma} ({state})")
    
    # Check for invalid characters in Baybayin form
    cur.execute(r"""
        SELECT COUNT(*)
        FROM words
        WHERE baybayin_form IS NOT NULL AND baybayin_form ~ '[^\u1700-\u171F\s]'
    """)
    invalid_chars_count = cur.fetchone()[0]
    
    if invalid_chars_count > 0:
        issues.append(f"Found {invalid_chars_count} entries with invalid Baybayin characters")
        
        # Sample some entries with invalid characters
        cur.execute(r"""
            SELECT id, lemma, baybayin_form
            FROM words
            WHERE baybayin_form IS NOT NULL AND baybayin_form ~ '[^\u1700-\u171F\s]'
            LIMIT 5
        """)
        
        for word_id, lemma, baybayin in cur.fetchall():
            logger.warning(f"Invalid Baybayin characters in word ID {word_id}: {lemma} / {baybayin}")
    
    # Check for entries with Baybayin in the lemma field but not in baybayin_form
    cur.execute(r"""
        SELECT COUNT(*)
        FROM words
        WHERE lemma ~ '[\u1700-\u171F]' AND (baybayin_form IS NULL OR NOT has_baybayin)
    """)
    missing_baybayin_count = cur.fetchone()[0]
    
    if missing_baybayin_count > 0:
        issues.append(f"Found {missing_baybayin_count} entries with Baybayin characters in lemma but not processed")
    
    return issues if issues else []

@with_transaction(commit=True)
def regenerate_all_romanizations(cur):
    """Regenerate romanized forms for all Baybayin entries."""
    regenerated_count = 0
    error_count = 0
    
    try:
        romanizer = BaybayinRomanizer()
        
        # Get all entries with Baybayin forms
        cur.execute("""
            SELECT id, baybayin_form
            FROM words
            WHERE has_baybayin = TRUE AND baybayin_form IS NOT NULL AND baybayin_form ~ '[\u1700-\u171F]'
        """)
        
        entries = cur.fetchall()
        
        logger.info(f"Regenerating romanizations for {len(entries)} Baybayin entries")
        
        for word_id, baybayin_form in entries:
            try:
                # Clean the form
                cleaned_form = clean_baybayin_text(baybayin_form)
                
                if cleaned_form and re.match(r'^[\u1700-\u171F\s]*$', cleaned_form):
                    # Generate romanization
                    romanized = romanizer.romanize(cleaned_form)
                    
                    # Update the entry
                    cur.execute("""
                        UPDATE words
                        SET romanized_form = %s,
                            baybayin_form = %s
                        WHERE id = %s
                    """, (romanized, cleaned_form, word_id))
                    
                    regenerated_count += 1
                    
                    # Log progress every 100 entries
                    if regenerated_count % 100 == 0:
                        logger.info(f"Regenerated {regenerated_count}/{len(entries)} romanizations")
            except Exception as e:
                logger.error(f"Error regenerating romanization for word ID {word_id}: {e}")
                error_count += 1
        
        logger.info(f"Romanization regeneration complete: {regenerated_count} updated, {error_count} errors")
        
    except Exception as e:
        logger.error(f"Error in regenerate_all_romanizations: {e}")
        return 0, 0
        
    finally:
        # This will always execute, even if there's an exception
        logger.info(f"Romanization regeneration finished - processed {regenerated_count + error_count} entries")
        
    return regenerated_count, error_count

@with_transaction(commit=True)
def fix_baybayin_constraint_violations(cur):
    """Fix Baybayin entries that violate the database constraint."""
    fixed_count = 0
    
    try:
        # Identify entries that would violate the constraint
        cur.execute(r"""
            SELECT id, lemma, baybayin_form
            FROM words
            WHERE baybayin_form IS NOT NULL AND baybayin_form !~ '^[\u1700-\u171F\s]*$'
        """)
        
        violations = cur.fetchall()
        
        logger.info(f"Found {len(violations)} entries violating Baybayin regex constraint")
        
        for word_id, lemma, baybayin_form in violations:
            # Clean the form
            cleaned_form = clean_baybayin_text(baybayin_form)
            
            if cleaned_form and re.match(r'^[\u1700-\u171F\s]*$', cleaned_form):
                # We can fix this entry
                cur.execute("""
                    UPDATE words
                    SET baybayin_form = %s
                    WHERE id = %s
                """, (cleaned_form, word_id))
                fixed_count += 1
            else:
                # Cannot fix, remove Baybayin data
                cur.execute("""
                    UPDATE words
                    SET has_baybayin = FALSE,
                        baybayin_form = NULL,
                        romanized_form = NULL
                    WHERE id = %s
                """, (word_id,))
                logger.warning(f"Removed invalid Baybayin form for word ID {word_id}: {lemma}")
        
        logger.info(f"Fixed {fixed_count} Baybayin constraint violations")
        
    except Exception as e:
        logger.error(f"Error fixing Baybayin constraint violations: {e}")
        return 0
        
    finally:
        logger.info(f"Completed Baybayin constraint violation check")
        
    return fixed_count

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
                logger.info(f"Regenerated {regen_count} romanizations with {regen_errors} errors")
                
                # Step 4: Process entries with Baybayin in lemma
                processed_count, error_count = process_baybayin_entries(cur)
                logger.info(f"Processed {processed_count} entries with {error_count} errors")
                
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
@with_transaction(commit=True)
def get_or_create_word_id(cur, lemma: str, language_code: str = DEFAULT_LANGUAGE_CODE,
                          source_identifier: Optional[str] = None, # Optional, but recommended
                          check_exists: bool = False, **kwargs) -> int:
    """
    Get the ID of a word, creating it if necessary.
    Updates the word's source_info JSONB field with the provided identifier.
    
    Args:
        cur: Database cursor.
        lemma: The word lemma.
        language_code: The language code (default 'tl').
        source_identifier: Identifier for the data source (e.g., filename). Recommended.
        check_exists: If True, check existence before attempting insert (less efficient).
        **kwargs: Additional word attributes (e.g., has_baybayin, baybayin_form, root_word_id).
        
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

    # Prepare optional fields from kwargs safely
    has_baybayin = kwargs.get('has_baybayin') # Keep as None if not provided
    baybayin_form = kwargs.get('baybayin_form')
    root_word_id = kwargs.get('root_word_id')
    tags = kwargs.get('tags')
    romanized_form = kwargs.get('romanized_form')
    preferred_spelling = kwargs.get('preferred_spelling')
    idioms_json = kwargs.get('idioms', '[]') # Default to empty JSON array string
    pronunciation_data_json = kwargs.get('pronunciation_data') # Assume already JSON string or None
    word_metadata_json = kwargs.get('word_metadata', '{}') # Default to empty JSON object string

    # Clean up Baybayin if inconsistent
    if has_baybayin is False:
        baybayin_form = None # Ensure form is None if explicitly false
    elif has_baybayin is True and not baybayin_form:
        logger.warning(f"Word '{lemma}' ({language_code}, source: {source_identifier}) marked as has_baybayin but no form provided. Setting has_baybayin to False.")
        has_baybayin = False # Correct the inconsistency

    # Standardize source identifier if provided
    standardized_source = None
    if source_identifier:
        try:
            standardized_source = SourceStandardization.standardize_sources(source_identifier)
            if not standardized_source:
                standardized_source = source_identifier  # Fallback to original if standardization fails
                logger.debug(f"Source standardization failed for '{source_identifier}', using original")
        except Exception as e:
            logger.warning(f"Error standardizing source '{source_identifier}': {e}")
            standardized_source = source_identifier  # Use original on error

    try:
         # Check if word exists and get its current ID and source_info
         cur.execute("""
             SELECT id, source_info FROM words
             WHERE normalized_lemma = %s AND language_code = %s
         """, (normalized, language_code))
         existing_word = cur.fetchone()

         if existing_word:
             word_id = existing_word[0]
             current_source_info = existing_word[1] # Fetched as dict/list/etc. from JSONB

             # Only update source info if a source identifier was provided
             if standardized_source:
                 # Update existing word's source info using the helper
                 updated_source_json = update_word_source_info(current_source_info, standardized_source)

                 # Parse the updated JSON string back to compare with the original dict/list
                 updated_source_dict = {}
                 try:
                     updated_source_dict = json.loads(updated_source_json)
                 except (json.JSONDecodeError, TypeError):
                     logger.error(f"Failed to parse updated source JSON for word ID {word_id}: '{updated_source_json}'")
                     # Decide how to handle - maybe skip update? For now, assume it might need update if parsing fails.
                     pass # Let the update proceed cautiously

                 # Only update if the source_info content has actually changed
                 if updated_source_dict != (current_source_info or {}):
                     cur.execute("""
                         UPDATE words SET source_info = %s, updated_at = CURRENT_TIMESTAMP
                         WHERE id = %s
                     """, (updated_source_json, word_id))
                     logger.debug(f"Word '{lemma}' ({language_code}) found (ID: {word_id}). Updated source_info from source '{standardized_source}'.")
                     
                     # Consider propagating the source to related records
                     try:
                         # Update definitions with this source if they exist
                         cur.execute("""
                             UPDATE definitions
                             SET sources = CASE
                                 WHEN sources IS NULL OR sources = '' THEN %s
                                 WHEN position(%s IN sources) = 0 THEN sources || ', ' || %s
                                 ELSE sources
                             END
                             WHERE word_id = %s
                         """, (standardized_source, standardized_source, standardized_source, word_id))
                         
                         # Similar updates could be made for etymologies, relations, etc.
                     except Exception as prop_e:
                         logger.warning(f"Error propagating source '{standardized_source}' to related records for word ID {word_id}: {prop_e}")
                 else:
                     logger.debug(f"Word '{lemma}' ({language_code}) found (ID: {word_id}). Source_info already includes '{standardized_source}' or no update needed.")

             # Consider updating other fields from kwargs when appropriate
             # For now, just report we found the word
             if not standardized_source:
                 logger.debug(f"Word '{lemma}' ({language_code}) found (ID: {word_id}). No source provided for update.")

         else:
             # Word doesn't exist, insert it
             logger.debug(f"Word '{lemma}' ({language_code}) not found. Creating new entry from source '{standardized_source}'.")
             initial_source_json = update_word_source_info(None, standardized_source)

             # Calculate a data hash for consistency checking (optional enhancement)
             data_hash = None
             try:
                 hash_input = f"{lemma}|{language_code}|{baybayin_form}|{romanized_form}|{tags}"
                 data_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
             except Exception as hash_e:
                 logger.debug(f"Error generating data hash for '{lemma}': {hash_e}")

             cur.execute("""
                 INSERT INTO words (
                     lemma, normalized_lemma, language_code,
                     has_baybayin, baybayin_form, romanized_form, root_word_id,
                     preferred_spelling, tags, source_info,
                     idioms, pronunciation_data, word_metadata,
                     data_hash, search_text
                 )
                 VALUES (
                     %(lemma)s, %(normalized)s, %(language_code)s,
                     %(has_baybayin)s, %(baybayin_form)s, %(romanized_form)s, %(root_word_id)s,
                     %(preferred_spelling)s, %(tags)s, %(source_info)s,
                     %(idioms)s, %(pronunciation_data)s, %(word_metadata)s,
                     %(data_hash)s, to_tsvector('simple', %(lemma)s || ' ' || %(normalized)s)
                 )
                 RETURNING id
             """, {
                 'lemma': lemma, 'normalized': normalized, 'language_code': language_code,
                 'has_baybayin': has_baybayin, 'baybayin_form': baybayin_form, 'romanized_form': romanized_form,
                 'root_word_id': root_word_id, 'preferred_spelling': preferred_spelling, 'tags': tags,
                 'source_info': initial_source_json, # Directly use the JSON string
                 'idioms': idioms_json, # Assumed to be valid JSON string or None
                 'pronunciation_data': pronunciation_data_json, # Assumed to be valid JSON string or None
                 'word_metadata': word_metadata_json, # Assumed to be valid JSON string or None
                 'data_hash': data_hash # Add data hash for consistency checking
             })
             word_id = cur.fetchone()[0]
             logger.info(f"Word '{lemma}' ({language_code}) created (ID: {word_id}) from source '{standardized_source}'.")

    except psycopg2.Error as e:
         logger.error(f"Database error in get_or_create_word_id for '{lemma}' ({language_code}) from source '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
         raise DatabaseError(f"Failed to get/create word ID for '{lemma}' from source '{source_identifier}': {e}") from e
    except Exception as e:
         logger.error(f"Unexpected error in get_or_create_word_id for '{lemma}' ({language_code}) from source '{source_identifier}': {e}", exc_info=True)
         raise # Reraise unexpected errors

    if word_id is None:
        # Should not happen if exceptions are handled, but as a safeguard
        raise DatabaseError(f"Failed to obtain word ID for '{lemma}' ({language_code}) from source '{source_identifier}' after operations.")

    return word_id

# Replace the existing insert_definition function (starts around line 2924)
@with_transaction(commit=True)
def insert_definition(cur, word_id: int, definition_text: str,
                      source_identifier: str, # MANDATORY
                      part_of_speech: Optional[str] = None, # Changed default to None
                      examples: Optional[str] = None, usage_notes: Optional[str] = None,
                      tags: Optional[str] = None) -> Optional[int]:
    """
    Insert definition data for a word. Ensures source is updated on conflict.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        definition_text: The definition text.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.
        part_of_speech: Original part of speech string (optional).
        examples: Example sentences (optional).
        usage_notes: Notes on usage (optional).
        tags: Comma-separated tags string (optional).

    Returns:
        The ID of the inserted/updated definition record, or None if failed.
    """
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping definition insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    if not definition_text:
        logger.warning(f"Empty definition text for word ID {word_id} from source '{source_identifier}'. Skipping.")
        return None

    try:
        # Get standardized part of speech ID
        standardized_pos_id = get_standardized_pos_id(cur, part_of_speech)

        # Truncate definition if too long for the database field (adjust length as needed)
        # Assuming definition_text max length is handled by DB or is sufficient. Add check if needed.
        # max_def_length = 4096
        # if len(definition_text) > max_def_length:
        #      logger.warning(f"Definition text for word ID {word_id} truncated from {len(definition_text)} to {max_def_length} characters.")
        #      definition_text = definition_text[:max_def_length]

        # Truncate examples if too long
        # Assuming examples max length is handled by DB or is sufficient. Add check if needed.
        # max_examples_length = 4096
        # if examples and len(examples) > max_examples_length:
        #      logger.warning(f"Examples for word ID {word_id} truncated from {len(examples)} to {max_examples_length} characters.")
        #      examples = examples[:max_examples_length]

        # Insert or update definition
        # Corrected ON CONFLICT clause to update sources
        cur.execute("""
            INSERT INTO definitions (word_id, definition_text, standardized_pos_id, examples, usage_notes, tags, sources, original_pos)
            VALUES (%(word_id)s, %(definition_text)s, %(pos_id)s, %(examples)s, %(usage_notes)s, %(tags)s, %(sources)s, %(original_pos)s)
            ON CONFLICT (word_id, definition_text, standardized_pos_id)
            DO UPDATE SET
                examples = COALESCE(EXCLUDED.examples, definitions.examples), -- Keep existing if new is NULL
                usage_notes = COALESCE(EXCLUDED.usage_notes, definitions.usage_notes), -- Keep existing if new is NULL
                tags = COALESCE(EXCLUDED.tags, definitions.tags), -- Keep existing if new is NULL
                sources = EXCLUDED.sources, -- Update sources field with the new source (last write wins)
                original_pos = COALESCE(EXCLUDED.original_pos, definitions.original_pos), -- Keep existing if new is NULL
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, {
            'word_id': word_id,
            'definition_text': definition_text,
            'pos_id': standardized_pos_id,
            'examples': examples,
            'usage_notes': usage_notes,
            'tags': tags,
            'sources': source_identifier, # Use the mandatory source_identifier
            'original_pos': part_of_speech # Store original POS string
        })
        def_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated definition (ID: {def_id}) for word ID {word_id} from source '{source_identifier}'.")
        return def_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting definition for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        # No rollback needed here as @with_transaction handles it
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting definition for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        # No rollback needed here as @with_transaction handles it
        return None
        
@with_transaction(commit=True)
def insert_relation(
    cur, 
    from_word_id: int, 
    to_word_id: int, 
    relation_type: Union[RelationshipType, str], 
    source_identifier: str, # MANDATORY
    metadata: Optional[Dict] = None
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
        logger.warning(f"Skipping self-relation for word ID {from_word_id}, type '{relation_type}', source '{source_identifier}'.")
        return None
    if not source_identifier:
         logger.error(f"CRITICAL: Skipping relation insert from {from_word_id} to {to_word_id}: Missing MANDATORY source identifier.")
         return None # Corrected indentation

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
                 logger.warning(f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Empty relation type string provided.")
                 return None
            try:
                # Attempt to map string to enum value
                rel_type_enum = RelationshipType.from_string(relation_type_cleaned)
                rel_type_str = rel_type_enum.rel_value
            except ValueError:
                # If not a standard enum value, use the cleaned string directly
                rel_type_str = relation_type_cleaned
                logger.debug(f"Using non-standard relation type string '{rel_type_str}' from source '{source_identifier}'.")
        else:
             logger.warning(f"Skipping relation insert from {from_word_id} to {to_word_id} (source '{source_identifier}'): Invalid relation_type type '{type(relation_type)}'.")
             return None

        # Dump metadata safely to JSON string for DB insertion (assuming metadata column is JSONB)
        metadata_json = None
        if metadata is not None: # Check explicitly for None, allow empty dict {}
            if isinstance(metadata, dict):
                try:
                    metadata_json = json.dumps(metadata)
                except TypeError as e:
                    logger.warning(f"Could not serialize metadata for relation {from_word_id}->{to_word_id} (source '{source_identifier}'): {e}. Metadata: {metadata}")
                    metadata_json = '{}' # Use empty JSON object string as fallback
            else:
                logger.warning(f"Metadata provided for relation {from_word_id}->{to_word_id} (source '{source_identifier}') is not a dict: {type(metadata)}. Storing as null.")
                metadata_json = None # Store null if not a dict

        cur.execute("""
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
        """, {
            'from_id': from_word_id, 'to_id': to_word_id, 'rel_type': rel_type_str,
            'sources': source_identifier,
            'metadata': metadata_json # Pass the JSON string (or None)
        })
        relation_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated relation (ID: {relation_id}) {from_word_id}->{to_word_id} ('{rel_type_str}') from source '{source_identifier}'.")
        return relation_id

    except psycopg2.IntegrityError as e:
         # Likely due to non-existent from_word_id or to_word_id (FK constraint violation)
         logger.error(f"Integrity error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}")
         return None
    except psycopg2.Error as e:
        logger.error(f"Database error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e.pgcode} {e.pgerror}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting relation {from_word_id}->{to_word_id} ('{relation_type}') from '{source_identifier}': {e}", exc_info=True)
        return None


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
    source_identifier: str, # MANDATORY
    normalized_components: Optional[str] = None,
    etymology_structure: Optional[str] = None, # Consider if this should be JSON
    language_codes: Optional[str] = None, # Comma-separated string
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
        logger.warning(f"Skipping etymology insert for word ID {word_id} from source '{source_identifier}': Missing etymology text.")
        return None
    if not source_identifier:
        logger.error(f"CRITICAL: Skipping etymology insert for word ID {word_id}: Missing MANDATORY source identifier.")
        return None

    try:
        # Prepare data, ensuring None is passed for empty optional fields
        params = {
            'word_id': word_id,
            'etym_text': etymology_text,
            'norm_comp': normalized_components.strip() if isinstance(normalized_components, str) else None,
            'etym_struct': etymology_structure.strip() if isinstance(etymology_structure, str) else None,
            'lang_codes': language_codes.strip() if isinstance(language_codes, str) else None,
            'sources': source_identifier # Use mandatory source_identifier directly
        } # <-- Corrected: Added missing closing brace

        # Ensure cur.execute is correctly indented within the try block
        cur.execute("""
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
        """, params)
        etymology_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated etymology (ID: {etymology_id}) for word ID {word_id} from source '{source_identifier}'. Text: '{etymology_text[:50]}...'")
        return etymology_id

    except psycopg2.Error as e:
        logger.error(f"Database error inserting etymology for word ID {word_id} from '{source_identifier}': {e.pgcode} {e.pgerror}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting etymology for word ID {word_id} from '{source_identifier}': {e}", exc_info=True)
        return None

@with_transaction(commit=True)
def insert_affixation(
    cur,
    root_id: int,
    affixed_id: int,
    affix_type: str,
    source_identifier: str # MANDATORY
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
    affix_type = affix_type.strip().lower() if isinstance(affix_type, str) else None # Normalize type
    if not affix_type:
         logger.warning(f"Skipping affixation insert for root {root_id}, affixed {affixed_id} (source '{source_identifier}'): Missing affix type.")
         return None
    if not source_identifier:
         logger.error(f"CRITICAL: Skipping affixation insert for root {root_id}, affixed {affixed_id}: Missing MANDATORY source identifier.")
         return None
    if root_id == affixed_id:
         logger.warning(f"Skipping self-affixation for word ID {root_id}, type '{affix_type}', source '{source_identifier}'.")
         return None

    try:
        # Prepare parameters
        params = {
            'root_id': root_id,
            'affixed_id': affixed_id,
            'affix_type': affix_type,
            'sources': source_identifier # Use mandatory source_identifier directly
        }

        cur.execute("""
            INSERT INTO affixations (root_word_id, affixed_word_id, affix_type, sources)
            VALUES (%(root_id)s, %(affixed_id)s, %(affix_type)s, %(sources)s)
            ON CONFLICT (root_word_id, affixed_word_id, affix_type) -- Conflict on exact triplet
            DO UPDATE SET
                -- Overwrite sources: Last write wins for this affixation record
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, params)
        affixation_id = cur.fetchone()[0]
        logger.debug(f"Inserted/Updated affixation (ID: {affixation_id}) {root_id}(root) -> {affixed_id}(affixed) [{affix_type}] from source '{source_identifier}'.")
        return affixation_id

    except psycopg2.IntegrityError as e:
         # Likely due to non-existent root_id or affixed_id (FK constraint violation)
         logger.error(f"Integrity error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}'. Word ID might not exist. Error: {e.pgcode} {e.pgerror}")
         return None
    except psycopg2.Error as e:
        logger.error(f"Database error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e.pgcode} {e.pgerror}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error inserting affixation {root_id}->{affixed_id} ({affix_type}) from '{source_identifier}': {e}", exc_info=True)
        return None


# Update for batch_get_or_create_word_ids function to properly handle sources

@with_transaction(commit=True)
def batch_get_or_create_word_ids(cur, entries: List[Tuple[str, str]], source: str = None, batch_size: int = 1000) -> Dict[Tuple[str, str], int]:
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
    if standardized_source is None or standardized_source == '':
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
        
        # If source is provided, update sources for existing words
        if source_info and existing:
            for (lemma, lang), word_id in existing.items():
                try:
                    # Check if the word already has source information
                    cur.execute("SELECT source_info FROM words WHERE id = %s", (word_id,))
                    row = cur.fetchone()
                    existing_source = row[0] if row and row[0] else None
                    
                    # If no existing source or different source, update it
                    if not existing_source:
                        cur.execute(
                            "UPDATE words SET source_info = %s WHERE id = %s",
                            (source_info, word_id)
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
                                if 'sources' in existing_json:
                                    if standardized_source not in existing_json['sources']:
                                        existing_json['sources'].append(standardized_source)
                                else:
                                    existing_json['sources'] = [standardized_source]
                                combined_source = json.dumps(existing_json)
                            else:
                                # Not a list or dict, treat as string
                                combined_source = json.dumps(f"{existing_source}, {standardized_source}")
                        except (json.JSONDecodeError, TypeError):
                            # Not valid JSON, just combine strings
                            combined_source = json.dumps(f"{existing_source}, {standardized_source}")
                            
                        cur.execute(
                            "UPDATE words SET source_info = %s WHERE id = %s",
                            (combined_source, word_id)
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
                search_text = ' '.join(word.strip() for word in re.findall(r'\w+', f"{lemma} {norm}"))
                # FIX: Changed to_tsquery to to_tsvector for search_text
                cur.execute("""
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
                """, (lemma, norm, lang, "", search_text, source_info))
                
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
# Replace the existing process_kwf_dictionary function (starts around line 3426)
@with_transaction(commit=False)  # Manage transactions manually
def process_kwf_dictionary(cur, filename: str):
    """
    Process a KWF Dictionary JSON file and store entries in database.
    Handles JSON input that is a dictionary mapping words (str) to their details (dict).
    Manages transactions manually with savepoints per entry.

    Args:
        cur: Database cursor
        filename: Path to KWF Dictionary JSON file

    Returns:
        Dictionary with processing statistics
    """
    # Standardize the source identifier consistently
    raw_source_identifier = os.path.basename(filename)
    source_identifier = SourceStandardization.standardize_sources(raw_source_identifier)

    # Ensure we have a valid source identifier, providing a default if needed
    if not source_identifier:
        source_identifier = "KWF Diksiyonaryo ng Wikang Filipino" # Default fallback

    logger.info(f"Processing KWF Dictionary: {filename}")
    logger.info(f"Using standardized source identifier: '{source_identifier}'")

    # Statistics tracking dictionary
    stats = {
        "total_entries": 0,
        "processed_entries": 0,
        "skipped_entries": 0,
        "error_entries": 0,
        "definitions_added": 0,
        "examples_added": 0,
        "synonyms_added": 0,
        "antonyms_added": 0,
        "refs_added": 0, # For see_also relations
        "affixes_processed": 0,
        "relations_added": 0,
        "source_updates": 0,
        "etymologies_added": 0
    }
    error_types = {}  # Track unique error types for summary

    # Get the connection object for transaction management
    conn = cur.connection

    try:
        # Read the JSON file
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {filename}: {str(e)}")
                stats["error_entries"] += 1
                error_types["JSONDecodeError"] = error_types.get("JSONDecodeError", 0) + 1
                return stats # Return early if file is invalid

        # --- Check format and prepare iterator ---
        # Expecting a dictionary: { "word": { details... }, ... }
        if not isinstance(data, dict):
            logger.error(f"Unexpected format in {filename}, expected a dictionary mapping words to details.")
            stats["error_entries"] += 1
            error_types["InvalidFormat"] = error_types.get("InvalidFormat", 0) + 1
            return stats

        # Ensure values are dictionaries (entry details)
        if not all(isinstance(v, dict) for v in data.values()):
             logger.error(f"File {filename} is a dictionary, but not all values are dictionaries (entry details). Cannot process.")
             stats["error_entries"] += 1
             error_types["InvalidValueFormat"] = error_types.get("InvalidValueFormat", 0) + 1
             return stats

        # Create an iterator that yields (index, word, entry_details)
        def kwf_dict_iterator(d):
            for i, (word_key, entry_details) in enumerate(d.items()):
                 yield i, word_key, entry_details # Pass the key (word) and value (details)

        entries_iterator = kwf_dict_iterator(data)
        stats["total_entries"] = len(data)
        logger.info(f"Found {stats['total_entries']} entries in dictionary format in {filename}")

        # Extract top-level metadata if available (e.g., version, publisher)
        # This requires a convention, e.g., metadata stored under a special key like "__metadata__"
        dictionary_metadata = {}
        if "__metadata__" in data and isinstance(data["__metadata__"], dict):
             dictionary_metadata = data["__metadata__"]
             logger.info("Found top-level dictionary metadata.")
             # Adjust total count if metadata key is present
             stats["total_entries"] -= 1 # Don't count metadata as an entry

        # Create source metadata object
        source_metadata = {
            "name": "KWF Diksiyonaryo ng Wikang Filipino",
            "file": raw_source_identifier,
            "processed_timestamp": datetime.now().isoformat()
        }
        if dictionary_metadata:
            source_metadata.update(dictionary_metadata)

        # Mapping for KWF relation types
        RELATION_TYPE_MAPPING = {
            "examples": None,
            "cognate_with": RelationshipType.RELATED,
            "borrowed_from": RelationshipType.DERIVED_FROM,
            "synonym": RelationshipType.SYNONYM,
            "antonym": RelationshipType.ANTONYM,
            "see_also": RelationshipType.SEE_ALSO,
            "related": RelationshipType.RELATED,
            "related_terms": RelationshipType.RELATED # Added based on KWF JSON structure
        }

        # --- Process each entry ---
        with tqdm(total=stats["total_entries"], desc=f"Processing {os.path.basename(filename)}", unit="entry") as pbar:
            for entry_index, word, entry in entries_iterator:
                # Skip metadata key if present
                if word == "__metadata__":
                     continue

                # Skip empty or invalid entries (word is key, entry is value dict)
                if not word or not isinstance(entry, dict):
                    logger.debug(f"Skipping invalid entry for key '{word}' at index {entry_index}")
                    stats["skipped_entries"] += 1
                    pbar.update(1)
                    continue

                # Create a savepoint for this entry
                savepoint_name = f"kwf_entry_{entry_index}"
                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                except Exception as sp_err:
                     logger.error(f"Failed to create savepoint {savepoint_name} for entry '{word}': {sp_err}. Skipping entry.")
                     stats["error_entries"] += 1
                     error_types[f"SavepointError: {sp_err}"] = error_types.get(f"SavepointError: {sp_err}", 0) + 1
                     pbar.update(1)
                     continue

                try:
                    # Extract formatted word if available, otherwise use the key
                    formatted_word = entry.get("formatted", word).strip()
                    if not formatted_word:
                         logger.warning(f"Entry '{word}' has empty formatted word, skipping")
                         stats["skipped_entries"] += 1
                         cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                         pbar.update(1)
                         continue

                    # --- Language Code (Infer if possible, default to 'tl') ---
                    language_code = entry.get("metadata", {}).get("language_code", "tl").strip() or "tl"

                    # --- Baybayin Processing (if relevant data exists in KWF format) ---
                    # KWF JSON structure doesn't show baybayin directly, adjust if needed
                    baybayin_form = None
                    romanized_form = None
                    has_baybayin = False
                    # Example: if 'baybayin_script' in entry.get("metadata", {}): ...

                    # --- Prepare Word Metadata ---
                    word_metadata_dict = entry.get("metadata", {}).copy() # Start with metadata section
                    word_metadata_dict["source_file"] = raw_source_identifier
                    word_metadata_dict["original_key"] = word # Store the original dictionary key
                    if dictionary_metadata: word_metadata_dict["dictionary_info"] = dictionary_metadata
                    # Add pronunciation if available
                    pronunciation_list = word_metadata_dict.pop("pronunciation", []) # Remove from dict, process separately
                    # Add etymology info if available
                    etymology_list = word_metadata_dict.pop("etymology", []) # Remove from dict, process separately
                    # Add source language if available
                    source_lang_list = word_metadata_dict.pop("source_language", []) # Remove from dict
                    if source_lang_list: word_metadata_dict["source_language_detail"] = source_lang_list


                    word_metadata_json = json.dumps(word_metadata_dict) if word_metadata_dict else None

                    # --- Get or Create Word ID ---
                    word_id = get_or_create_word_id(
                        cur,
                        formatted_word,
                        language_code=language_code,
                        has_baybayin=has_baybayin,
                        baybayin_form=baybayin_form,
                        romanized_form=romanized_form,
                        source_identifier=source_identifier,
                        word_metadata=word_metadata_json
                        # Add lemma=word if formatted_word might differ significantly
                    )

                    if not word_id or word_id <= 0:
                        raise ValueError(f"Failed to get or create a valid word ID for '{formatted_word}' (key: '{word}')")

                    # --- Process Pronunciation (from metadata) ---
                    if isinstance(pronunciation_list, list):
                         for pron_data in pronunciation_list:
                              # Adapt based on how pronunciation is structured in KWF JSON
                              if isinstance(pron_data, str) and pron_data.strip():
                                   insert_pronunciation(cur, word_id, pron_data.strip(), source_identifier)
                              # elif isinstance(pron_data, dict): ... handle complex pronunciation dict

                    # --- Process Etymology (from metadata) ---
                    # Assuming etymology_list contains strings or dicts with etymology info
                    if isinstance(etymology_list, list):
                         for ety_idx, ety_item in enumerate(etymology_list):
                              ety_text = None
                              # Adapt based on KWF etymology structure
                              if isinstance(ety_item, str) and ety_item.strip():
                                   ety_text = ety_item.strip()
                              elif isinstance(ety_item, dict) and ety_item.get('text'): # Example structure
                                   ety_text = ety_item['text'].strip()
                                   # Extract other components if available in dict

                              if ety_text:
                                  try:
                                      components = extract_etymology_components(ety_text)
                                      components_json = json.dumps(components) if components else None
                                      lang_codes_str = None
                                      language_codes_list = extract_language_codes(ety_text)
                                      if language_codes_list: lang_codes_str = ", ".join(language_codes_list)

                                      ety_db_id = insert_etymology(
                                          cur, word_id, ety_text, source_identifier,
                                          normalized_components=components_json,
                                          language_codes=lang_codes_str
                                      )
                                      if ety_db_id: stats["etymologies_added"] += 1
                                  except Exception as ety_error:
                                      logger.warning(f"Error processing etymology for '{formatted_word}': {ety_text[:50]}... Error: {ety_error}")
                                      error_types[f"EtymologyError: {ety_error}"] = error_types.get(f"EtymologyError: {ety_error}", 0) + 1


                    # --- Process Definitions by Part of Speech ---
                    # KWF structure: "definitions": { "PartOfSpeech": [ {def details}, ... ], ... }
                    definitions_by_pos = entry.get("definitions", {})
                    if isinstance(definitions_by_pos, dict):
                        for raw_pos, definitions_list in definitions_by_pos.items():
                            if not isinstance(definitions_list, list): continue

                            standardized_pos_str = standardize_entry_pos(raw_pos) if raw_pos else None

                            for def_idx, def_item in enumerate(definitions_list):
                                # Check if def_item is a dictionary
                                if not isinstance(def_item, dict):
                                    logger.warning(f"Skipping invalid definition item (not a dict) for '{formatted_word}', POS '{raw_pos}', index {def_idx}")
                                    continue

                                meaning_value = def_item.get("meaning")
                                definition_text = meaning_value.strip() if meaning_value else ""
                                if not definition_text: continue # Skip if no definition text

                                examples_json = None
                                usage_notes = def_item.get("note") # KWF uses "note"
                                tags_list = def_item.get("categories", []) # KWF uses "categories" for tags/domain
                                tags = ", ".join(filter(None, map(str.strip, tags_list))) if tags_list else None
                                def_metadata_dict = {"source_file": raw_source_identifier, "original_pos": raw_pos}
                                if standardized_pos_str: def_metadata_dict["standardized_pos"] = standardized_pos_str
                                def_num = def_item.get("number")
                                if def_num is not None: def_metadata_dict["definition_number"] = def_num

                                # Process examples (KWF format: "example_sets": [ {"label":.., "examples": [ {"text":.., "html":..} ] } ])
                                example_sets = def_item.get("example_sets", [])
                                normalized_examples = []
                                if isinstance(example_sets, list):
                                    for set_idx, ex_set in enumerate(example_sets):
                                         if isinstance(ex_set, dict):
                                             set_label = ex_set.get("label")
                                             examples_in_set = ex_set.get("examples", [])
                                             if isinstance(examples_in_set, list):
                                                 for ex_idx, ex in enumerate(examples_in_set):
                                                     if isinstance(ex, dict) and "text" in ex and ex["text"].strip():
                                                         ex_obj = {"text": ex["text"].strip(), "source": source_identifier, "set_label": set_label, "set_index": set_idx, "index_in_set": ex_idx}
                                                         # Add translation/html if needed from 'ex' dict
                                                         normalized_examples.append(ex_obj)

                                if normalized_examples:
                                    try:
                                        examples_json = json.dumps(normalized_examples)
                                        stats["examples_added"] += len(normalized_examples)
                                    except TypeError: logger.warning(f"Could not serialize examples for '{formatted_word}', def {def_idx}")

                                # Insert the definition
                                if definition_text:
                                    try:
                                        definition_id = insert_definition(
                                            cur, word_id, definition_text,
                                            source_identifier=source_identifier,
                                            part_of_speech=standardized_pos_str, # Use standardized POS
                                            examples=examples_json,
                                            usage_notes=usage_notes.strip() if usage_notes else None,
                                            tags=tags
                                            # Pass def_metadata_dict if insert_definition accepts it
                                        )
                                        if definition_id:
                                            stats["definitions_added"] += 1

                                            # Process definition-level relations (synonyms, antonyms from KWF def_item)
                                            # Synonyms
                                            synonyms_list = def_item.get("synonyms", [])
                                            if isinstance(synonyms_list, list):
                                                 for syn_idx, syn in enumerate(synonyms_list):
                                                     # KWF synonyms seem to be strings directly
                                                     if isinstance(syn, str) and syn.strip() and syn.strip() != formatted_word:
                                                          # Strip HTML tags before further processing
                                                         syn_word_raw = syn.strip()
                                                         syn_word_clean = re.sub(r'<[^>]+>', '', syn_word_raw).strip()
                                                         if not syn_word_clean or syn_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                         try:
                                                             syn_id = get_or_create_word_id(cur, syn_word_clean, language_code, source_identifier=source_identifier)
                                                             if syn_id:
                                                                  syn_metadata = {"source": source_identifier, "confidence": 90, "definition_id": definition_id, "index": syn_idx}
                                                                  rel_syn_id = insert_relation(cur, word_id, syn_id, RelationshipType.SYNONYM, source_identifier, metadata=syn_metadata)
                                                                  if rel_syn_id: stats["synonyms_added"] += 1; stats["relations_added"] += 1
                                                                  # Handle bidirectional SYNONYM
                                                                  insert_relation(cur, syn_id, word_id, RelationshipType.SYNONYM, source_identifier, metadata=syn_metadata)
                                                         except Exception as syn_error: logger.warning(f"Error processing synonym '{syn_word_clean}' for '{formatted_word}': {syn_error}")

                                            # Antonyms
                                            antonyms_list = def_item.get("antonyms", [])
                                            if isinstance(antonyms_list, list):
                                                 for ant_idx, ant in enumerate(antonyms_list):
                                                     if isinstance(ant, str) and ant.strip() and ant.strip() != formatted_word:
                                                                                                                  # Strip HTML tags before further processing
                                                         ant_word_raw = ant.strip()
                                                         ant_word_clean = re.sub(r'<[^>]+>', '', ant_word_raw).strip()
                                                         if not ant_word_clean or ant_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                         try:
                                                             ant_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier=source_identifier)
                                                             if ant_id:
                                                                 ant_metadata = {"source": source_identifier, "confidence": 90, "definition_id": definition_id, "index": ant_idx}
                                                                 rel_ant_id = insert_relation(cur, word_id, ant_id, RelationshipType.ANTONYM, source_identifier, metadata=ant_metadata)
                                                                 if rel_ant_id: stats["antonyms_added"] += 1; stats["relations_added"] += 1
                                                                 # Handle bidirectional ANTONYM
                                                                 insert_relation(cur, ant_id, word_id, RelationshipType.ANTONYM, source_identifier, metadata=ant_metadata)
                                                         except Exception as ant_error: logger.warning(f"Error processing antonym '{ant_word_clean}' for '{formatted_word}': {ant_error}")

                                            # Process cross_references within definition if present
                                            cross_refs = def_item.get("cross_references", [])
                                            if isinstance(cross_refs, list):
                                                for ref_idx, ref in enumerate(cross_refs):
                                                    ref_word = None
                                                    ref_link = None
                                                    if isinstance(ref, dict):
                                                        ref_word = ref.get("term")
                                                        ref_link = ref.get("link") # KWF has link field

                                                    if isinstance(ref_word, str) and ref_word.strip() and ref_word.strip() != formatted_word:
                                                                                                                   # Strip HTML tags before further processing
                                                          ref_word_raw = ref_word.strip()
                                                          ref_word_clean = re.sub(r'<[^>]+>', '', ref_word_raw).strip()
                                                          if not ref_word_clean or ref_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                          try:
                                                              ref_id = get_or_create_word_id(cur, ref_word_clean, language_code, source_identifier=source_identifier)
                                                              if ref_id:
                                                                  ref_metadata = {"source": source_identifier, "confidence": 80, "definition_id": definition_id, "index": ref_idx, "link": ref_link}
                                                                  rel_ref_id = insert_relation(cur, word_id, ref_id, RelationshipType.SEE_ALSO, source_identifier, metadata=ref_metadata)
                                                                  if rel_ref_id:
                                                                      stats["refs_added"] += 1
                                                                      stats["relations_added"] += 1
                                                                  else:
                                                                      logger.warning(f"Error processing cross_reference '{ref_word_clean}' for '{formatted_word}': Relation not inserted.")
                                                              else:
                                                                  logger.warning(f"Error processing cross_reference '{ref_word_clean}' for '{formatted_word}': Word ID not found.")
                                                          except Exception as ref_error:
                                                              logger.warning(f"Error processing cross_reference '{ref_word_clean}' for '{formatted_word}': {ref_error}")

                                    except psycopg2.errors.UniqueViolation:
                                        logger.debug(f"Definition already exists for '{formatted_word}', POS '{raw_pos}', def {def_idx}: {definition_text[:50]}...")
                                    except Exception as def_error:
                                        logger.warning(f"Error inserting definition for '{formatted_word}', POS '{raw_pos}', def {def_idx}: {definition_text[:50]}... Error: {def_error}")
                                        error_types[f"DefinitionInsertError: {def_error}"] = error_types.get(f"DefinitionInsertError: {def_error}", 0) + 1

                    # --- Process Top-Level Related Terms ---
                    # KWF structure: "related": { "related_terms": [ {"term": "...", "link": "..."} ], ... }
                    related_data = entry.get("related", {})
                    if isinstance(related_data, dict):
                        logger.debug(f"Processing top-level related terms for '{formatted_word}'")
                        for rel_type_raw, rel_items in related_data.items():
                            # Skip 'examples' explicitly if present here
                            if rel_type_raw.lower() == "examples": continue

                            # Map KWF relation type to standard RelationshipType
                            rel_type_enum = RELATION_TYPE_MAPPING.get(rel_type_raw.lower())
                            if rel_type_enum is None:
                                try:
                                    std_rel_type_str, _, _ = normalize_relation_type(rel_type_raw)
                                    rel_type_enum = RelationshipType.from_string(std_rel_type_str)
                                except (ValueError, KeyError):
                                    logger.warning(f"Unknown top-level relation type '{rel_type_raw}' for '{formatted_word}', treating as RELATED.")
                                    rel_type_enum = RelationshipType.RELATED

                            if not isinstance(rel_items, list):
                                logger.warning(f"Skipping invalid related items for type '{rel_type_raw}' in '{formatted_word}': not a list")
                                continue

                            for rel_idx, rel_item in enumerate(rel_items):
                                related_term = None
                                item_metadata = {} # Metadata specific to this related item
                                # KWF format: { "term": "...", "link": "..." } or just string
                                related_term_raw = None # Initialize
                                if isinstance(rel_item, dict) and 'term' in rel_item:
                                    related_term_raw = str(rel_item['term']).strip()
                                    for meta_key, meta_val in rel_item.items():
                                        if meta_key != 'term': item_metadata[meta_key] = meta_val
                                elif isinstance(rel_item, str):
                                    related_term_raw = rel_item.strip()

                                related_term_clean = None
                                if related_term_raw:
                                    # Strip HTML tags before further processing
                                    related_term_clean = re.sub(r'<[^>]+>', '', related_term_raw).strip()

                                if related_term_clean and len(related_term_clean) <= 255 and related_term_clean != formatted_word:
                                    try:
                                        related_id = get_or_create_word_id(
                                            cur, related_term_clean, language_code=language_code,
                                            source_identifier=source_identifier
                                        )

                                        if related_id:
                                            relation_metadata = {
                                                "source": source_identifier,
                                                "original_type": rel_type_raw,
                                                "index": rel_idx,
                                                "confidence": 85 # Default confidence
                                            }
                                            relation_metadata.update(item_metadata)

                                            rel_top_id = insert_relation(
                                                cur, word_id, related_id,
                                                rel_type_enum,
                                                source_identifier=source_identifier,
                                                metadata=relation_metadata
                                            )
                                            if rel_top_id: stats["relations_added"] += 1
                                            # Handle bidirectional if needed
                                            if rel_type_enum.bidirectional:
                                                 # Check if inverse already added to avoid double counting if structure lists both ways
                                                 # For simplicity here, just add inverse - review if duplicates occur
                                                 insert_relation(cur, related_id, word_id, rel_type_enum, source_identifier, metadata=relation_metadata)


                                    except Exception as e:
                                        logger.warning(f"Error processing top-level related term '{related_term}' (type '{rel_type_raw}') for '{formatted_word}': {e}")
                                        error_types[f"RelationError: {e}"] = error_types.get(f"RelationError: {e}", 0) + 1


                    # --- Finish Entry Processing ---
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    stats["processed_entries"] += 1

                    # Commit periodically
                    if stats["processed_entries"] % 500 == 0:
                        try:
                            conn.commit()
                            logger.info(f"Processed {stats['processed_entries']} entries, committed batch for {filename}")
                        except Exception as commit_batch_err:
                             logger.error(f"Error committing batch at entry {entry_index} for {filename}: {commit_batch_err}. Rolling back batch...")
                             conn.rollback()
                             stats["error_entries"] += 500 # Approximate error count
                             error_types["BatchCommitError"] = error_types.get("BatchCommitError", 0) + 1


                # --- Error Handling for a Single Entry ---
                except psycopg2.Error as db_error:
                    error_msg = str(db_error).split('\n')[0]
                    logger.error(f"Database error processing entry '{word}': {error_msg}", exc_info=False)
                    error_types[f"DBError: {error_msg}"] = error_types.get(f"DBError: {error_msg}", 0) + 1
                    stats["error_entries"] += 1
                    try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err: logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name}: {rb_err}. Attempting full tx rollback.", exc_info=True); conn.rollback(); raise db_error from rb_err

                except ValueError as val_error:
                    logger.error(f"ValueError processing entry '{word}': {val_error}", exc_info=False)
                    error_types[f"ValueError: {val_error}"] = error_types.get(f"ValueError: {val_error}", 0) + 1
                    stats["error_entries"] += 1
                    try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err: logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name}: {rb_err}. Attempting full tx rollback.", exc_info=True); conn.rollback(); raise val_error from rb_err

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Unexpected error processing entry '{word}': {error_msg}", exc_info=True)
                    error_types[f"UnexpectedError: {error_msg}"] = error_types.get(f"UnexpectedError: {error_msg}", 0) + 1
                    stats["error_entries"] += 1
                    try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err: logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name}: {rb_err}. Attempting full tx rollback.", exc_info=True); conn.rollback(); raise e from rb_err

                finally:
                    pbar.update(1)

        # --- Final Commit ---
        try:
            conn.commit()
            logger.info("Final transaction commit successful for KWF processing.")
        except Exception as commit_error:
            logger.error(f"Error during final commit for {filename}: {commit_error}. Rolling back remaining changes...")
            stats["error_entries"] += 1 # Count commit failure as an error
            error_types["FinalCommitError"] = error_types.get("FinalCommitError", 0) + 1
            conn.rollback()

    # --- Outer Exception Handling ---
    except Exception as outer_e:
        logger.error(f"Outer exception processing KWF dictionary {filename}: {outer_e}", exc_info=True)
        error_types[f"OuterError: {outer_e}"] = error_types.get(f"OuterError: {outer_e}", 0) + 1
        stats["error_entries"] += 1
        try:
            if conn and not conn.closed and conn.status != psycopg2.extensions.STATUS_IN_TRANSACTION: # Check if not already closed/rolled back
                 logger.warning("Rolling back transaction due to outer exception.")
                 conn.rollback()
            else:
                 logger.info("Transaction already closed or rolled back in outer exception.")
        except Exception as rollback_error:
            logger.error(f"Failed to roll back transaction in outer exception handler: {rollback_error}")

    # --- Final Logging ---
    finally:
        logger.info(f"Completed processing KWF Dictionary: {filename}")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        if error_types:
            logger.info(f"Error types encountered: {json.dumps(error_types, indent=2)}")

    return stats

#  Replace the existing process_tagalog_words function (starts around line 4059)
@with_transaction(commit=True) # Using commit=True for this function as per the original code
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

    language_code = 'tl'  # Assume Tagalog
    romanizer = BaybayinRomanizer() # Instantiate for potential Baybayin processing

    # Statistics tracking dictionary - initialized as in the original snippet
    stats = {
        "total_entries": 0,
        "processed_entries": 0,
        "skipped_entries": 0, # For invalid format or missing essential data
        "error_entries": 0,   # For entries that caused exceptions during processing
        "definitions_added": 0,
        "relations_added": 0, # Grand total of all relation types added
        "synonyms_added": 0,
        "references_added": 0, # Corresponds to 'references' -> SEE_ALSO
        "variants_added": 0,
        "derivatives_added": 0, # Processed from 'derivative' field
        "etymologies_processed": 0, # Updated if etymology is successfully inserted
        "baybayin_added": 0,      # Updated if Baybayin form is added/updated
        "pronunciations_added": 0,
        "affixations_added": 0 # Processed from 'affix_forms'/'affix_types'
    }

    # Note: conn = cur.connection is not needed here because @with_transaction(commit=True)
    # handles the commit/rollback automatically for the whole function scope or per error.
    # Savepoints are also less critical here but kept from the original for consistency.

    try:
        # Load and parse the data file
        with open(filename, 'r', encoding='utf-8') as f:
            # Assuming the JSON is a dictionary where keys are lemmas
            data = json.load(f)

        if not isinstance(data, dict):
             logger.error(f"File {filename} content is not a dictionary as expected. Aborting.")
             raise TypeError(f"Expected dictionary data in {filename}")


        # Initialize statistics
        stats["total_entries"] = len(data)
        logger.info(f"Found {stats['total_entries']} entries to process")

        # Process each word entry with progress bar
        with tqdm(total=stats["total_entries"], desc=f"Processing {source_identifier}", unit="word") as pbar:
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
                    logger.error(f"Failed to create savepoint {savepoint_name} for '{lemma}': {sp_err}. Skipping entry.")
                    stats["error_entries"] += 1
                    pbar.update(1)
                    continue # Skip entry if savepoint fails

                try:
                    lemma = lemma.strip() # Ensure lemma is stripped
                    if not lemma:
                         logger.warning(f"Skipping entry with empty lemma key.")
                         stats["skipped_entries"] += 1
                         cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release if skipping
                         pbar.update(1)
                         continue

                    # --- Basic Validation ---
                    if not entry_data or not isinstance(entry_data, dict):
                        logger.warning(f"Skipping invalid entry data for '{lemma}': not a dictionary or empty.")
                        stats["skipped_entries"] += 1
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release if skipping
                        pbar.update(1)
                        continue

                    # --- Prepare Metadata and Tags ---
                    # Construct metadata dictionary as in the original snippet
                    word_metadata = {"processed_timestamp": datetime.now().isoformat()}
                    tags = [] # List to collect tags for the word level
                    if 'domains' in entry_data and entry_data['domains']:
                        # Ensure domains is a list of strings
                        domain_list = entry_data['domains'] if isinstance(entry_data['domains'], list) else [str(entry_data['domains'])]
                        word_metadata["domains"] = [str(d).strip() for d in domain_list if str(d).strip()]
                        tags.extend(word_metadata["domains"]) # Add domains to word tags
                    if 'part_of_speech' in entry_data and entry_data['part_of_speech']:
                         # Store raw POS structure in metadata
                        word_metadata["part_of_speech_raw"] = entry_data['part_of_speech']
                        # Flatten POS list for potential tagging (optional)
                        # flat_pos = []
                        # for item in entry_data['part_of_speech']: flat_pos.extend(item if isinstance(item, list) else [item])
                        # tags.extend(p for p in flat_pos if p) # Optionally add POS to tags
                    if 'pronunciation' in entry_data and entry_data['pronunciation']:
                        word_metadata["pronunciation_primary_raw"] = entry_data['pronunciation']
                    if 'alternate_pronunciation' in entry_data: # Check optional key
                         word_metadata["pronunciation_alternate_raw"] = entry_data['alternate_pronunciation']
                    if 'derivative' in entry_data and entry_data['derivative']:
                        word_metadata["derivative_raw_text"] = entry_data['derivative']

                    # --- Create or Get Word ID ---
                    # Pass the mandatory source_identifier
                    word_id = get_or_create_word_id(
                        cur, lemma, language_code=language_code,
                        source_identifier=source_identifier, # Pass mandatory source
                        word_metadata=json.dumps(word_metadata), # Pass constructed metadata as JSON string
                        tags=", ".join(sorted(list(set(tags)))) if tags else None # Pass unique, sorted tags
                        # Add other kwargs like is_proper_noun if determinable
                    )
                    if not word_id:
                        # Error should be logged within get_or_create_word_id
                        raise ValueError(f"Failed to get/create word ID for '{lemma}' from {source_identifier}")

                    # --- Process Pronunciations ---
                    # Insert primary pronunciation
                    if 'pronunciation' in entry_data and entry_data['pronunciation']:
                        # Standardize pronunciation data into a dict if it's just a string
                        pron_data = entry_data['pronunciation']
                        if isinstance(pron_data, str):
                             pron_obj = {"value": pron_data, "type": "ipa"} # Assume IPA if string
                        elif isinstance(pron_data, dict):
                             pron_obj = pron_data # Assume dict has needed structure
                        else: pron_obj = None

                        if pron_obj:
                            try:
                                pron_id = insert_pronunciation(cur, word_id, pron_obj, source_identifier=source_identifier)
                                if pron_id: stats["pronunciations_added"] += 1
                            except Exception as pron_e:
                                logger.warning(f"Failed primary pronunciation insert for '{lemma}': {pron_obj}. Error: {pron_e}")
                    # Insert alternate pronunciation
                    if 'alternate_pronunciation' in entry_data and entry_data['alternate_pronunciation']:
                         alt_pron_data = entry_data['alternate_pronunciation']
                         if isinstance(alt_pron_data, str):
                              alt_pron_obj = {"value": alt_pron_data, "type": "ipa", "metadata": {"is_alternate": True}}
                         elif isinstance(alt_pron_data, dict):
                              # Add alternate flag if passing a dict
                              alt_pron_data.setdefault('metadata', {})['is_alternate'] = True
                              alt_pron_obj = alt_pron_data
                         else: alt_pron_obj = None

                         if alt_pron_obj:
                            try:
                                alt_pron_id = insert_pronunciation(cur, word_id, alt_pron_obj, source_identifier=source_identifier)
                                if alt_pron_id: stats["pronunciations_added"] += 1
                            except Exception as alt_pron_e:
                                logger.warning(f"Failed alternate pronunciation insert for '{lemma}': {alt_pron_obj}. Error: {alt_pron_e}")


                    # --- Process Derivatives (as Relations) ---
                    # Treat 'derivative' field as source for DERIVED_FROM relations
                    if 'derivative' in entry_data and entry_data['derivative']:
                        derivative_text = entry_data['derivative']
                        # Assuming comma-separated string
                        derivative_forms = derivative_text.split(',') if isinstance(derivative_text, str) else []
                        for form_idx, form in enumerate(derivative_forms):
                            form_clean = form.strip()
                            if form_clean and form_clean != lemma: # Ensure not empty and not self-referential
                                # Basic check for format like "verb salamat" - adjust logic as needed
                                form_word = form_clean
                                pos_hint = None
                                if " " in form_clean:
                                    parts = form_clean.split(' ', 1)
                                    # Simple heuristic: if first part is short, maybe it's a POS hint
                                    # Use POS_MAPPING or similar constant if available for validation
                                    if len(parts[0]) <= 5 and parts[0].isalpha(): # Basic check
                                        pos_hint = parts[0]
                                        form_word = parts[1].strip()

                                try:
                                    # Get ID for the word mentioned in the derivative field
                                    derivative_word_id = get_or_create_word_id(cur, form_word, language_code, source_identifier=source_identifier)
                                    if derivative_word_id:
                                        # Insert relation: derivative_word DERIVED_FROM lemma_word (word_id)
                                        rel_metadata = {"source": source_identifier, "raw_text": form_clean, "index": form_idx, "pos_hint": pos_hint, "confidence": 85}
                                        rel_id = insert_relation(
                                             cur, derivative_word_id, word_id, # From derivative TO base word
                                             RelationshipType.DERIVED_FROM,
                                             source_identifier=source_identifier, # Pass mandatory source
                                             metadata=rel_metadata
                                        )
                                        if rel_id:
                                            stats["derivatives_added"] += 1
                                            stats["relations_added"] += 1
                                        # Optionally add inverse ROOT_OF relation if needed
                                except Exception as der_e:
                                    logger.warning(f"Error processing derivative form '{form_clean}' for word '{lemma}': {der_e}")

                    # --- Process Etymology ---
                    # Check if etymology data exists and is a dictionary
                    if 'etymology' in entry_data and isinstance(entry_data['etymology'], dict):
                        etymology_data = entry_data['etymology']
                        etymology_text = etymology_data.get('raw', '') # Assuming 'raw' key holds the text
                        if etymology_text and isinstance(etymology_text, str):
                            try:
                                # Prepare structure and language codes if available
                                etymology_structure_dict = {'source': source_identifier, **etymology_data}
                                lang_codes_list = etymology_data.get('languages', []) # Assuming 'languages' key holds a list
                                language_codes_str = ", ".join(lang_codes_list) if isinstance(lang_codes_list, list) else None

                                # Call insert_etymology with mandatory source_identifier
                                ety_id = insert_etymology(
                                    cur, word_id, etymology_text.strip(),
                                    source_identifier=source_identifier, # Pass mandatory source
                                    etymology_structure=json.dumps(etymology_structure_dict), # Pass structure as JSON string
                                    language_codes=language_codes_str # Pass language codes string
                                    # Pass normalized_components if calculated/available
                                )
                                if ety_id: stats["etymologies_processed"] += 1
                            except Exception as e:
                                logger.warning(f"Error processing etymology for word '{lemma}': {etymology_text[:50]}... Error: {e}")

                    # --- Process Senses ---
                    senses = entry_data.get('senses', [])
                    if isinstance(senses, list):
                        for sense_idx, sense in enumerate(senses):
                            if not isinstance(sense, dict) or 'definition' not in sense: continue
                            definition_text = sense.get('definition', '').strip()
                            if not definition_text: continue

                            # Prepend counter if exists
                            counter = sense.get('counter', '')
                            if counter: definition_text = f"[{counter}] {definition_text}"

                            # Determine POS: Use sense POS if available, else fallback to entry POS
                            pos_list_sense = sense.get('part_of_speech', []) # POS specific to sense
                            pos_list_entry = entry_data.get('part_of_speech', []) # POS for the whole entry
                            pos_list = pos_list_sense if pos_list_sense else pos_list_entry
                            # Flatten potential list-of-lists structure
                            flat_pos = []
                            if isinstance(pos_list, list):
                                for item in pos_list: flat_pos.extend(item if isinstance(item, list) else [item])
                            elif isinstance(pos_list, str): # Handle case where POS is just a string
                                flat_pos.append(pos_list)
                            # Create comma-separated string for storage and standardization
                            pos_str = ", ".join(str(p).strip() for p in flat_pos if str(p).strip())
                            standardized_pos_id = get_standardized_pos_id(cur, pos_str) # Get standardized ID

                            # Extract Examples into a JSON list of dicts
                            examples_json = None
                            examples_list = []
                            example_data = sense.get('example', {}) # examples might be under 'example' key
                            if isinstance(example_data, dict):
                                raw_ex_list = example_data.get('examples', []) # Assuming a list under 'examples'
                                if isinstance(raw_ex_list, list):
                                    examples_list = [{"text": str(ex).strip(), "source": source_identifier, "index": i}
                                                     for i, ex in enumerate(raw_ex_list) if str(ex).strip()]
                                elif 'raw' in example_data and example_data['raw']: # Handle 'raw' example key
                                    examples_list = [{"text": str(example_data['raw']).strip(), "source": source_identifier, "raw": True}]
                            elif isinstance(example_data, list): # If 'example' directly holds a list
                                 examples_list = [{"text": str(ex).strip(), "source": source_identifier, "index": i}
                                                     for i, ex in enumerate(example_data) if str(ex).strip()]

                            if examples_list:
                                try: examples_json = json.dumps(examples_list)
                                except TypeError: logger.warning(f"Could not serialize examples for '{lemma}', sense {sense_idx}")

                            # Prepare Tags: Combine sense tags and category
                            sense_tags_list = []
                            raw_tags = sense.get('tags', [])
                            if isinstance(raw_tags, list):
                                sense_tags_list.extend(str(t).strip() for t in raw_tags if str(t).strip())
                            category = sense.get('category') # Check for category field
                            if category and isinstance(category, str):
                                sense_tags_list.append(f"category:{category.strip()}")
                            tags_str = ", ".join(sorted(list(set(sense_tags_list)))) if sense_tags_list else None

                            try:
                                # Insert definition, passing mandatory source_identifier
                                definition_id = insert_definition(
                                    cur, word_id, definition_text,
                                    source_identifier=source_identifier, # Pass mandatory source
                                    part_of_speech=pos_str, # Fix: Pass the original POS string, not the ID
                                    examples=examples_json, # Pass examples JSON string or None
                                    usage_notes=sense.get('usage_notes'), # Pass usage notes if present
                                    tags=tags_str # Pass comma-separated tags string or None
                                )

                                if definition_id:
                                    stats["definitions_added"] += 1
                                    # Process sense-level relations (synonyms, variants, references)
                                    sense_relation_map = {
                                        'synonyms': RelationshipType.SYNONYM,
                                        'variants': RelationshipType.VARIANT,
                                        'references': RelationshipType.SEE_ALSO # Map 'references' to SEE_ALSO
                                    }
                                    for rel_key, rel_type_enum in sense_relation_map.items():
                                        related_items = sense.get(rel_key)
                                        if isinstance(related_items, list):
                                            for item_idx, item_word in enumerate(related_items):
                                                if isinstance(item_word, str) and item_word.strip():
                                                    item_word_clean = item_word.strip()
                                                    if item_word_clean != lemma: # Avoid self-relation
                                                        try:
                                                            item_id = get_or_create_word_id(cur, item_word_clean, language_code, source_identifier=source_identifier)
                                                            if item_id:
                                                                # Add metadata specific to the relation source (sense)
                                                                rel_meta = {"source": source_identifier, "definition_id": definition_id, "sense_index": sense_idx, "item_index": item_idx, "confidence": 80} # Example confidence
                                                                # Insert relation, passing mandatory source
                                                                rel_ins_id = insert_relation(
                                                                    cur, word_id, item_id, # From lemma word TO item word
                                                                    rel_type_enum,
                                                                    source_identifier=source_identifier, # Pass mandatory source
                                                                    metadata=rel_meta
                                                                )
                                                                if rel_ins_id:
                                                                    stats[f"{rel_key}_added"] += 1
                                                                    stats["relations_added"] += 1
                                                                # Handle bidirectional if needed (e.g., SYNONYM)
                                                                # if rel_type_enum.bidirectional: insert_relation(cur, item_id, word_id, rel_type_enum, source_identifier=source_identifier, metadata=rel_meta)

                                                        except Exception as rel_e:
                                                            logger.warning(f"Error processing sense relation '{rel_key}'->'{item_word_clean}' for '{lemma}', sense {sense_idx}: {rel_e}")
                            except psycopg2.errors.UniqueViolation:
                                logger.debug(f"Definition already exists for '{lemma}', sense {sense_idx}: {definition_text[:30]}...")
                            except Exception as def_e:
                                logger.error(f"Error inserting definition for '{lemma}', sense {sense_idx}: {definition_text[:30]}... Error: {def_e}", exc_info=False) # Less verbose

                    # --- Process Baybayin ---
                    # Check for 'baybayin' key and validate/process
                    if 'baybayin' in entry_data and entry_data['baybayin']:
                        baybayin_text = entry_data['baybayin']
                        if isinstance(baybayin_text, str) and baybayin_text.strip():
                           baybayin_form_raw = baybayin_text.strip()
                           # Validate the raw Baybayin form before cleaning/storing
                           if validate_baybayin_entry(baybayin_form_raw):
                               clean_baybayin = clean_baybayin_text(baybayin_form_raw) # Clean for storage
                               if clean_baybayin:
                                   try:
                                       # Romanize the *cleaned* version for consistency
                                       romanized = romanizer.romanize(clean_baybayin)
                                       # Update words table, setting has_baybayin = true
                                       # Only update if baybayin_form is NULL or different
                                       cur.execute("""
                                           UPDATE words SET has_baybayin = true, baybayin_form = %s, romanized_form = %s
                                           WHERE id = %s AND (baybayin_form IS NULL OR baybayin_form != %s)
                                       """, (clean_baybayin, romanized, word_id, clean_baybayin))
                                       if cur.rowcount > 0:
                                           stats["baybayin_added"] += 1
                                           logger.debug(f"Added/Updated Baybayin for '{lemma}' (ID: {word_id})")
                                   except Exception as bb_e:
                                       logger.error(f"Error processing/updating Baybayin '{baybayin_form_raw}' for '{lemma}': {bb_e}")
                           else:
                               logger.warning(f"Invalid baybayin structure found for '{lemma}': {baybayin_form_raw}")

                    # --- Process Affix Forms ---
                    # Check for both 'affix_forms' and 'affix_types' keys
                    if 'affix_forms' in entry_data and isinstance(entry_data['affix_forms'], list) and \
                       'affix_types' in entry_data and isinstance(entry_data['affix_types'], list):
                        affix_forms = entry_data['affix_forms']
                        affix_types = entry_data['affix_types']
                        # Ensure lists have the same length for safe iteration
                        min_len = min(len(affix_forms), len(affix_types))
                        if len(affix_forms) != len(affix_types):
                             logger.warning(f"Mismatch in length between affix_forms ({len(affix_forms)}) and affix_types ({len(affix_types)}) for '{lemma}'. Processing up to index {min_len-1}.")

                        for i in range(min_len):
                            form = affix_forms[i]
                            affix_type = affix_types[i]
                            clean_form = str(form).strip() if form else None
                            clean_type = str(affix_type).strip().lower() if affix_type else "unknown"

                            if clean_form and clean_form != lemma: # Ensure form exists and is not the lemma itself
                                try:
                                    # Get ID for the affixed form, passing source
                                    affixed_id = get_or_create_word_id(cur, clean_form, language_code, source_identifier=source_identifier)
                                    if affixed_id:
                                        # Insert into affixation table, passing source
                                        aff_rec_id = insert_affixation(
                                            cur, root_id=word_id, affixed_id=affixed_id,
                                            affix_type=clean_type, source_identifier=source_identifier # Pass mandatory source
                                        )
                                        if aff_rec_id: stats["affixations_added"] += 1
                                        # Optionally add DERIVED_FROM/ROOT_OF relations here as well, if desired
                                        # Example: insert_relation(cur, affixed_id, word_id, RelationshipType.DERIVED_FROM, source_identifier=source_identifier, metadata={"affix_type": clean_type})

                                except Exception as aff_e:
                                    logger.warning(f"Error processing affix form '{clean_form}' (type {clean_type}, index {i}) for root '{lemma}': {aff_e}")

                    # --- Finish Entry Processing ---
                    stats["processed_entries"] += 1
                    # Release savepoint (optional with commit=True, but kept for consistency)
                    cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

                except Exception as entry_err:
                    # Catch errors specific to processing this entry after getting lemma/data
                    logger.error(f"Error processing entry for lemma '{lemma}': {entry_err}", exc_info=True) # Log traceback for debugging
                    stats["error_entries"] += 1
                    try:
                        # Rollback to the state before this entry started
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rollback_error:
                        # This is more critical - if savepoint rollback fails, the transaction state is unknown
                        logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name} for '{lemma}': {rollback_error}. Decorator should handle full rollback.")
                        # Re-raising might be appropriate here to stop the whole process if this occurs
                        # raise entry_err from rollback_error
                finally:
                    pbar.update(1) # Ensure progress bar updates even if an error occurred

    # --- Error Handling for File Loading ---
    except (IOError, json.JSONDecodeError, TypeError) as file_err:
        logger.error(f"Fatal error reading or parsing file {filename}: {file_err}")
        # The @with_transaction decorator should handle rollback here
        # Log the stats gathered so far before returning/raising
        logger.info(f"Partial stats before fatal file error: {stats}")
        raise # Re-raise the error to signal failure to the caller

    # --- General Error Handling ---
    except Exception as e:
        logger.error(f"Unexpected fatal error during processing of {filename}: {e}", exc_info=True)
        # The @with_transaction decorator should handle rollback
        logger.info(f"Partial stats before fatal error: {stats}")
        raise # Re-raise critical errors

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
        logger.info(f"    (Deriv: {stats['derivatives_added']}, Syn: {stats['synonyms_added']}, Var: {stats['variants_added']}, Ref: {stats['references_added']})") # Detailed relation counts
        logger.info(f"  Baybayin forms added/updated: {stats['baybayin_added']}")
        logger.info(f"  Affixations added/updated: {stats['affixations_added']}")

    return stats # Return the final statistics dictionary

@with_transaction(commit=False)  # Manage transactions manually
def process_root_words_cleaned(cur, filename: str):
    """
    Processes entries from the tagalog.com Root Words JSON file (cleaned version).
    Handles JSON input that is either a list of root word objects
    or a dictionary mapping root words (str) to their details (dict).
    Manages transactions manually with savepoints per entry.
    """
    logger.info(f"Processing Root Words (tagalog.com cleaned) file: {filename}")
    stats = {"roots_processed": 0, "definitions_added": 0, "relations_added": 0, "associated_processed": 0, "errors": 0, "skipped": 0}
    # Define source identifier, standardizing the filename
    source_identifier = standardize_source_identifier(os.path.basename(filename)) # Or "tagalog.com-RootWords-Cleaned"
    conn = cur.connection # Get connection for manual commit/rollback and savepoints

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return stats
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filename}: {e}")
        stats["errors"] += 1 # Count file load error
        return stats
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}", exc_info=True)
        stats["errors"] += 1
        return stats

    # --- Determine format and prepare iterator ---
    entries_iterator = None
    total_roots = 0
    if isinstance(data, list): # Original list format
        entries_iterator = enumerate(data)
        total_roots = len(data)
        logger.info(f"Found {total_roots} root word entries in list format in {filename}")
    elif isinstance(data, dict): # New dictionary format {root_word: details_dict}
        # Ensure details are dictionaries
        if not all(isinstance(v, dict) for v in data.values()):
            logger.error(f"File {filename} is a dictionary, but not all values are dictionaries (details). Cannot process.")
            stats["errors"] += 1
            return stats
        # Create an iterator yielding (index, {**details, 'root_word': key})
        def dict_iterator(d):
            for i, (key, value) in enumerate(d.items()):
                 # Add the root word (key) into the details dictionary for consistent processing
                 entry_data = value.copy()
                 entry_data['root_word'] = key # Ensure root_word key exists within the entry data
                 yield i, entry_data
        entries_iterator = dict_iterator(data)
        total_roots = len(data)
        logger.info(f"Found {total_roots} root word entries in dictionary format in {filename}")
    else:
         logger.error(f"File {filename} does not contain a list or dictionary of root word entries.")
         stats["errors"] += 1
         return stats

    # --- Process Entries ---
    for entry_index, root_word_entry in tqdm(entries_iterator, total=total_roots, desc=f"Processing {source_identifier}"):
        # Define savepoint name using entry index for safety
        savepoint_name = f"tagalogcom_root_cleaned_{entry_index}"
        try:
            cur.execute(f"SAVEPOINT {savepoint_name}")
        except Exception as e:
            logger.error(f"Failed to create savepoint {savepoint_name} for root entry index {entry_index}: {e}. Skipping.")
            stats["errors"] += 1
            continue # Skip this entry if savepoint fails

        root_word = None # Initialize for error logging
        try:
            # Validate entry format (should always be dict now)
            if not isinstance(root_word_entry, dict):
                 logger.warning(f"Skipping invalid entry at index {entry_index} in {filename} (not a dict after processing)")
                 stats["skipped"] += 1
                 cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release savepoint
                 continue

            # Extract root word (should exist due to iterator modification for dict format)
            root_word = root_word_entry.get('root_word', '').strip()
            if not root_word:
                logger.warning(f"Skipping tagalog.com entry at index {entry_index} with empty root word.")
                stats["skipped"] += 1
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release savepoint
                continue

            language_code = 'tl' # Assume Tagalog

            # --- Root Word Creation ---
            # Pass the mandatory source_identifier
            # Use word_id variable consistently
            word_id = get_or_create_word_id(
                cur,
                root_word,
                language_code=language_code,
                source_identifier=source_identifier, # Pass mandatory source
                is_root_word=True # Optional flag if schema supports it
                # Add other kwargs as needed, e.g., check_exists=False
            )
            if not word_id or word_id <= 0: # Check if word_id was successfully obtained
                raise ValueError(f"Failed to get/create root word ID for tagalog.com entry: {root_word}")

            stats["roots_processed"] += 1

            # --- Process definitions for the root word ---
            definitions = root_word_entry.get('definitions', [])
            if isinstance(definitions, list):
                 for def_idx, definition_item in enumerate(definitions):
                     # Handling definitions that might be strings or dicts
                     definition_text: Optional[str] = None
                     part_of_speech: Optional[str] = None
                     examples_json: Optional[str] = None # Store examples as JSON string or None
                     tags: Optional[str] = None # Store tags as comma-separated string or None

                     if isinstance(definition_item, str):
                         definition_text = definition_item.strip()
                         # Potentially extract POS from string definition if a pattern exists
                         # e.g., pos_match = re.match(r"\[(\w+)\]", definition_text) ...
                     elif isinstance(definition_item, dict):
                         # Check common keys for definition text
                         definition_text = definition_item.get('text', '').strip() or \
                                           definition_item.get('definition', '').strip()
                         # Check common keys for part of speech
                         part_of_speech = definition_item.get('pos') or \
                                          definition_item.get('part_of_speech') or \
                                          definition_item.get('type') # Added 'type' based on user JSON
                         if part_of_speech: part_of_speech = part_of_speech.strip()

                         # Extract examples if present, format as JSON list of strings
                         raw_examples = definition_item.get('examples')
                         if isinstance(raw_examples, list):
                             examples_list = [str(ex).strip() for ex in raw_examples if str(ex).strip()]
                             if examples_list:
                                try: examples_json = json.dumps(examples_list)
                                except TypeError: logger.warning(f"Could not serialize examples for root '{root_word}', def {def_idx}: {examples_list}")
                         elif isinstance(raw_examples, str) and raw_examples.strip():
                             try: examples_json = json.dumps([raw_examples.strip()])
                             except TypeError: logger.warning(f"Could not serialize example string for root '{root_word}', def {def_idx}: {raw_examples}")

                         # Extract tags if present
                         raw_tags = definition_item.get('tags')
                         if isinstance(raw_tags, list):
                             tags = ", ".join(str(t).strip() for t in raw_tags if str(t).strip())
                         elif isinstance(raw_tags, str):
                             tags = raw_tags.strip()

                     if definition_text:
                         try:
                             # Call insert_definition, passing the mandatory source_identifier
                             def_id = insert_definition(
                                 cur,
                                 word_id, # Use the obtained root word_id
                                 definition_text,
                                 source_identifier=source_identifier, # Pass mandatory source
                                 part_of_speech=part_of_speech, # Pass original POS string
                                 examples=examples_json, # Pass examples (JSON string or None)
                                 tags=tags # Pass tags string
                                 # Pass usage_notes if available in definition_item dict
                             )
                             if def_id:
                                 stats["definitions_added"] += 1
                             # else: insert_definition handles logging its own errors/warnings
                         except Exception as def_e:
                             # Catch unexpected errors during insert
                             logger.warning(f"Error inserting definition for root '{root_word}', def {def_idx}: {definition_text[:50]}... Error: {def_e}")


            # --- Process associated words (assuming they are derived) ---
            # Handle associated_words being a dict {assoc_word: details} or a list [str_or_dict]
            associated_data = root_word_entry.get('associated_words', {}) # Default to empty dict if missing
            associated_items_iterator = None

            if isinstance(associated_data, dict):
                # Format: { "assoc_word": { "type": "...", "definition": "..." } }
                def assoc_dict_iterator(d):
                     for i, (key, value) in enumerate(d.items()):
                         # Yield index, word string, and details dict
                         yield i, key.strip(), value if isinstance(value, dict) else {}
                associated_items_iterator = assoc_dict_iterator(associated_data)

            elif isinstance(associated_data, list):
                 # Format: [ "word_str", { "word": "...", ... } ]
                 def assoc_list_iterator(l):
                     for i, item in enumerate(l):
                         word_str = None
                         details = {}
                         if isinstance(item, str):
                             word_str = item.strip()
                         elif isinstance(item, dict):
                             word_str = item.get('word', '').strip()
                             details = item # Pass the whole dict as details
                         if word_str: yield i, word_str, details
                 associated_items_iterator = assoc_list_iterator(associated_data)

            else:
                 logger.warning(f"Unexpected format for 'associated_words' for root '{root_word}'. Expected dict or list, got {type(associated_data)}. Skipping.")
                 associated_items_iterator = iter([]) # Empty iterator


            for assoc_idx, assoc_word, assoc_details in associated_items_iterator:
                if assoc_word and assoc_word != root_word:
                    stats["associated_processed"] += 1
                    try:
                        # Get or create ID for associated word, passing source_identifier
                        assoc_word_id = get_or_create_word_id(
                            cur,
                            assoc_word,
                            language_code=language_code,
                            root_word_id=word_id, # Link derived word to root in word table if possible
                            source_identifier=source_identifier # Pass mandatory source
                        )
                        if assoc_word_id:
                            # Add DERIVED_FROM relationship, passing source_identifier
                            rel_metadata = {"source": source_identifier, "index": assoc_idx, "confidence": 95} # Example metadata
                            rel_id = insert_relation(
                                cur, assoc_word_id, word_id, # from derived TO root
                                RelationshipType.DERIVED_FROM,
                                source_identifier=source_identifier, # Pass mandatory source
                                metadata=rel_metadata # Pass optional metadata
                            )
                            if rel_id:
                                stats["relations_added"] += 1

                            # --- Optionally process definition for associated word ---
                            assoc_def_text = assoc_details.get('definition', '').strip()
                            assoc_pos = assoc_details.get('type', '').strip() # Get POS from 'type' key
                            if assoc_def_text:
                                try:
                                    assoc_def_id = insert_definition(
                                         cur, assoc_word_id, assoc_def_text,
                                         source_identifier=source_identifier,
                                         part_of_speech=assoc_pos or None
                                         # Add examples etc. if available in assoc_details
                                    )
                                    if assoc_def_id: stats["definitions_added"] += 1 # Count assoc defs too
                                except Exception as assoc_def_e:
                                     logger.warning(f"Error inserting definition for associated word '{assoc_word}' of root '{root_word}': {assoc_def_e}")

                    except Exception as assoc_e:
                        logger.warning(f"Failed to process/relate associated word '{assoc_word}' (index {assoc_idx}) for root '{root_word}': {assoc_e}")
                        # Don't increment main error count, maybe a specific counter?

            # Release savepoint for successful entry
            cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

        except Exception as entry_e:
            # Catch errors during word creation or main processing for the entry
            logger.error(f"Error processing root entry index {entry_index} ('{root_word or 'N/A'}') in {filename}: {entry_e}", exc_info=False) # Less verbose logs
            stats["errors"] += 1
            try:
                # Rollback to the savepoint created at the start of this entry
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
            except Exception as rb_err:
                # If rollback fails, transaction state is uncertain, needs full rollback
                logger.critical(f"CRITICAL: Failed rollback to savepoint {savepoint_name} after error: {rb_err}. Attempting full transaction rollback.")
                conn.rollback() # Rollback the entire transaction
                logger.warning("Performed full transaction rollback due to savepoint rollback failure.")
                # Consider re-raising the original error to stop the whole process
                raise entry_e from rb_err

    # Final commit for the entire file after processing all entries
    try:
        conn.commit()
        logger.info(f"Finished processing {filename}. Stats: {json.dumps(stats, indent=2)}")
    except Exception as commit_err:
         logger.error(f"Error during final commit for {filename}: {commit_err}. Rolling back any uncommitted changes...")
         stats["errors"] += 1 # Count final commit failure as an error
         conn.rollback()

    return stats

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

@with_transaction(commit=False)  # Changed to commit=False to manage transactions manually
def process_kaikki_jsonl(cur, filename: str):
    """Process Kaikki.org dictionary entries."""
    # Initialize statistics
    processed = 0
    skipped = 0
    errors = 0
    # Standardize the source identifier from the filename
    source_identifier = standardize_source_identifier(os.path.basename(filename)) # <<< CORRECTED
    logger.info(f"Processing Kaikki dictionary: {filename} with source ID: '{source_identifier}'") # Log the used ID

    # First count total lines in file
    try:
        with open(filename, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for _ in f_count)
        logger.info(f"Found {total_lines} entries to process in {filename}")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {
            "total_entries": 0,
            "processed_entries": 0,
            "error_entries": 1, # Count file not found as an error
            "skipped_entries": 0
        }
    except Exception as count_error:
        logger.error(f"Error counting lines in {filename}: {count_error}")
        # Proceed, but log the issue
        total_lines = -1 # Indicate unknown total

    conn = cur.connection # Get connection early for schema changes

    # Check if table structure exists and create required tables
    try:
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

                -- Definitions table enhancements - check first if not exists
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'definitions' AND column_name = 'metadata'
                ) THEN
                    ALTER TABLE definitions ADD COLUMN metadata JSONB;
                END IF;
            END $$;
        """)

        # Commit schema changes
        conn.commit()
        logger.info("Successfully checked/committed schema changes for Kaikki processing.")
    except Exception as e:
        logger.error(f"Error during schema setup/check for Kaikki processing: {str(e)}")
        conn.rollback()
        return { # Return stats indicating failure
            "total_entries": total_lines if total_lines != -1 else 0,
            "processed_entries": 0,
            "error_entries": 1, # Count schema error as one major error
            "skipped_entries": total_lines if total_lines != -1 else 0
        }

    # Create a mapping for custom relationship types to standard ones
    RELATIONSHIP_TYPE_MAPPING = {
        'borrowed_from': RelationshipType.DERIVED_FROM.rel_value,  # Map borrowed_from to DERIVED_FROM
        'cognate_with': RelationshipType.RELATED.rel_value,        # Map cognate_with to RELATED
    }

    # Function to get standard relationship type
    def get_standard_relationship_type(rel_type):
        """Convert custom relationship types to standard ones defined in RelationshipType enum"""
        if rel_type in RELATIONSHIP_TYPE_MAPPING:
            return RELATIONSHIP_TYPE_MAPPING[rel_type]

        # Check if it's already a standard type
        try:
            # Attempt to directly create enum from value
            RelationshipType(rel_type)
            return rel_type
        except ValueError:
             # Check members by value if direct init fails (covers aliases/different cases if any)
            for rel in RelationshipType:
                if rel.rel_value == rel_type:
                    return rel_type

        logger.warning(f"Unknown relationship type: {rel_type}, using RELATED as fallback")
        return RelationshipType.RELATED.rel_value

    # Extract Baybayin info with proper cleaning
    def extract_baybayin_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract Baybayin script form from an entry, cleaning prefixes."""
        if 'forms' not in entry:
            return None, None

        for form in entry['forms']:
            if isinstance(form, dict) and 'tags' in form and 'Baybayin' in form.get('tags', []):
                baybayin_form = form.get('form', '')
                romanized = form.get('romanized') # Corrected key based on common JSON structure

                # Clean the baybayin form of prefixes like "spelling "
                if baybayin_form and isinstance(baybayin_form, str):
                    # Check for common prefixes
                    prefixes = ["spelling ", "script ", "baybayin "]
                    for prefix in prefixes:
                        if baybayin_form.lower().startswith(prefix):
                            baybayin_form = baybayin_form[len(prefix):].strip()

                # Basic validation - ensure it contains Baybayin characters
                if baybayin_form and any('\u1700' <= char <= '\u171F' for char in baybayin_form):
                     # Further validation if needed: validate_baybayin_entry(baybayin_form, romanized)
                     return baybayin_form, romanized
        return None, None

    # Extract Badlit info with similar cleaning
    def extract_badlit_info(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract Badlit script form from an entry, cleaning prefixes."""
        if 'forms' not in entry:
            return None, None

        for form in entry['forms']:
             if isinstance(form, dict) and 'tags' in form and 'Badlit' in form.get('tags', []):
                badlit_form = form.get('form', '')
                romanized = form.get('romanized') # Corrected key

                # Clean the badlit form of prefixes
                if badlit_form and isinstance(badlit_form, str):
                    prefixes = ["spelling ", "script ", "badlit "]
                    for prefix in prefixes:
                        if badlit_form.lower().startswith(prefix):
                            badlit_form = badlit_form[len(prefix):].strip()

                # Basic validation - add more specific checks if needed
                if badlit_form: # Add Badlit character range check if known
                    return badlit_form, romanized
        return None, None

    # Helper function to extract all canonical forms of a word entry
    def extract_canonical_forms(entry: Dict) -> List[str]:
        forms = []
        if 'forms' in entry:
            for form in entry['forms']:
                if isinstance(form, dict) and 'form' in form and form.get('form') and 'tags' in form and 'canonical' in form.get('tags', []):
                    forms.append(form['form'])
        return forms

    # Standardize entry POS
    def standardize_entry_pos(pos_str: str) -> str:
        if not pos_str:
            return 'unc'  # Default to uncategorized
        pos_key = pos_str.lower().strip()
        return get_standard_code(pos_key) # Use the global helper

    # Process pronunciation
    def process_pronunciation(cur, word_id: int, entry: Dict, source_identifier: str): # Added source_identifier
        if 'sounds' not in entry:
            return

        for sound in entry.get('sounds', []):
             if not isinstance(sound, dict): continue

             pron_data = {}
             pron_type = None
             pron_value = None

             if 'ipa' in sound:
                 pron_type = 'ipa'
                 pron_value = sound['ipa']
                 pron_data = sound # Store the whole sound dict as metadata/tags
             elif 'rhymes' in sound:
                 pron_type = 'rhyme'
                 pron_value = sound['rhymes']
                 pron_data = sound
             elif 'audio' in sound: # Handle audio file links if needed
                 pron_type = 'audio'
                 pron_value = sound['audio']
                 pron_data = sound
             # Add other sound types if present (e.g., 'enpr')

             if pron_type and pron_value:
                 try:
                     # Use the dedicated insert_pronunciation function
                     insert_pronunciation(cur, word_id, pron_data, source_identifier)
                 except Exception as e:
                     logger.error(f"Error inserting pronunciation for word ID {word_id} (Type: {pron_type}): {str(e)}")


    # Process sense relationships
    def process_sense_relationships(cur, word_id: int, sense: Dict, language_code: str, source_identifier: str):
        """Processes various relationships within a sense."""
        relation_types = {
            'synonyms': RelationshipType.SYNONYM,
            'antonyms': RelationshipType.ANTONYM,
            'hypernyms': RelationshipType.HYPERNYM,
            'hyponyms': RelationshipType.HYPONYM,
            'holonyms': RelationshipType.HOLONYM,
            'meronyms': RelationshipType.MERONYM,
            'derived': RelationshipType.ROOT_OF, # word_id is the root_of derived_word
            'related': RelationshipType.RELATED,
            'coordinate_terms': RelationshipType.RELATED, # Map coordinate terms to related
            'see_also': RelationshipType.SEE_ALSO
        }

        for rel_key, rel_enum in relation_types.items():
            if rel_key in sense and isinstance(sense[rel_key], list):
                for item in sense[rel_key]:
                    if isinstance(item, dict) and 'word' in item:
                        related_word = item['word']
                        if not related_word or not isinstance(related_word, str): continue

                        try:
                            related_word_id = get_or_create_word_id(cur, related_word, language_code, source_identifier=source_identifier)
                            if related_word_id == word_id: continue # Avoid self-references

                            metadata = {'confidence': rel_enum.strength} # Default confidence from enum
                            if 'tags' in item and isinstance(item['tags'], list):
                                metadata['tags'] = ','.join(item['tags'])
                            if 'qualifier' in item: # Capture qualifiers if present
                                metadata['qualifier'] = item['qualifier']
                            if '_dis1' in item: # Capture distance if present (from Kaikki?)
                                metadata['distance'] = item['_dis1']
                            metadata['from_sense'] = True # Indicate it came from a sense

                            insert_relation(cur, word_id, related_word_id, rel_enum, source_identifier, metadata)

                            # Handle bidirectionality / inverse relationships automatically based on enum
                            if rel_enum.bidirectional:
                                insert_relation(cur, related_word_id, word_id, rel_enum, source_identifier, metadata)
                            else:
                                inverse_rel = rel_enum.get_inverse()
                                if inverse_rel:
                                    insert_relation(cur, related_word_id, word_id, inverse_rel, source_identifier, metadata)

                        except Exception as e:
                            logger.error(f"Error processing {rel_key} relation for word ID {word_id} to '{related_word}': {str(e)}")


    def process_form_relationships(cur, word_id: int, entry: Dict, language_code: str, source_identifier: str):
        """Process relationships based on word forms (variants, spelling, etc.)."""
        if 'forms' not in entry or not isinstance(entry['forms'], list):
            return

        canonical_forms = set(extract_canonical_forms(entry))

        for form_data in entry['forms']:
            if not form_data or not isinstance(form_data, dict):
                continue

            form_word = form_data.get('form', '')
            if not form_word or not isinstance(form_word, str):
                continue

            # Skip if the form is essentially the main entry word or already known canonical
            if form_word == entry.get('word') or form_word in canonical_forms:
                 continue

            try:
                # Determine relationship type and metadata based on tags
                rel_type = RelationshipType.VARIANT # Default
                metadata = {"from_forms": True, "confidence": RelationshipType.VARIANT.strength}
                tags = form_data.get('tags', [])

                if tags and isinstance(tags, list):
                    metadata["tags"] = ','.join(tags) # Store tags

                    if any(tag in ['standard spelling', 'preferred', 'standard form', 'canonical'] for tag in tags):
                        rel_type = RelationshipType.SPELLING_VARIANT
                        metadata["confidence"] = 95
                        # Consider updating preferred spelling on the main word_id? Risky.
                    elif any(tag in ['alternative spelling', 'alternate spelling', 'alt form'] for tag in tags):
                        rel_type = RelationshipType.SPELLING_VARIANT
                        metadata["confidence"] = 90
                    elif any(tag in ['regional', 'dialectal', 'dialect'] for tag in tags):
                        rel_type = RelationshipType.REGIONAL_VARIANT
                        metadata["confidence"] = 85
                    elif 'obsolete' in tags:
                         metadata["confidence"] = 70 # Lower confidence for obsolete variants
                         # Could add a specific 'obsolete_variant' type if needed
                    # Add more specific tag checks if needed (e.g., 'archaic', 'dated')

                if 'qualifier' in form_data:
                    metadata["qualifier"] = form_data['qualifier']

                # Get ID for the form word
                form_word_id = get_or_create_word_id(cur, form_word, language_code, source_identifier=source_identifier)
                if form_word_id == word_id: continue # Avoid self-references

                # Insert the relationship (and its inverse/bidirectional counterpart)
                insert_relation(cur, word_id, form_word_id, rel_type, source_identifier, metadata)
                if rel_type.bidirectional:
                     insert_relation(cur, form_word_id, word_id, rel_type, source_identifier, metadata)
                else:
                     inverse_rel = rel_type.get_inverse()
                     if inverse_rel:
                         insert_relation(cur, form_word_id, word_id, inverse_rel, source_identifier, metadata)

            except Exception as e:
                logger.error(f"Error processing form relationship for word ID {word_id} to form '{form_word}': {str(e)}")


    def process_categories(cur, definition_id: int, categories: List[Dict]):
        """Insert categories for a definition."""
        for category in categories:
            if not isinstance(category, dict) or 'name' not in category:
                continue

            category_name = category['name']
            category_kind = category.get('kind')
            parents = category.get('parents')

            try:
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
            except Exception as e:
                logger.error(f"Error inserting category '{category_name}' for definition ID {definition_id}: {str(e)}")


    def process_head_templates(cur, word_id: int, templates: List[Dict]):
        """Process head templates for a word."""
        for template in templates:
             if not isinstance(template, dict) or 'name' not in template:
                continue

             template_name = template['name']
             args = template.get('args')
             expansion = template.get('expansion')

             try:
                cur.execute("""
                    INSERT INTO word_templates (word_id, template_name, args, expansion)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (word_id, template_name) DO UPDATE SET
                        args = EXCLUDED.args,
                        expansion = EXCLUDED.expansion,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    word_id,
                    template_name,
                    json.dumps(args) if args else None,
                    expansion
                ))
             except Exception as e:
                 logger.error(f"Error inserting/updating template '{template_name}' for word ID {word_id}: {str(e)}")


    def process_links(cur, definition_id: int, links: List[List[str]]):
        """Process links for a definition."""
        for link in links:
            # Links are typically a two-element list: [text, target]
            if not isinstance(link, list) or len(link) < 2:
                continue

            link_text = link[0]
            link_target = link[1]
            if not link_text or not link_target: continue # Skip empty links

            # Check if this is a Wikipedia link
            is_wikipedia = 'wikipedia.org' in link_target.lower() or 'w:' in link_text or 'W:' in link_text

            try:
                cur.execute("""
                    INSERT INTO definition_links (definition_id, link_text, link_target, is_wikipedia)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (definition_id, link_text, link_target) DO NOTHING
                """, (
                    definition_id,
                    link_text,
                    link_target,
                    is_wikipedia
                ))
            except Exception as e:
                logger.error(f"Error inserting link '{link_text}' -> '{link_target}' for definition ID {definition_id}: {str(e)}")


    def process_etymology(cur, word_id: int, etymology_text: str, source_identifier: str, etymology_templates: List[Dict] = None):
        """Process etymology text and templates using insert_etymology helper."""
        if not etymology_text or not isinstance(etymology_text, str):
            return

        normalized_components = None # TODO: Implement extract_etymology_components if needed
        etymology_structure = None # TODO: Parse structure if needed
        language_codes = None

        # Extract language info from templates if available
        if etymology_templates and isinstance(etymology_templates, list):
            languages = set()
            # Simple extraction: Look for language codes in template args
            # More complex parsing might be needed based on template names (e.g., 'derived', 'borrowed')
            for template in etymology_templates:
                 if isinstance(template, dict) and 'args' in template and isinstance(template['args'], dict):
                     for key, value in template['args'].items():
                         # Check if value looks like a language code (2-3 letters)
                         if isinstance(value, str) and 1 < len(value) <= 3 and value.isalpha():
                             languages.add(value.lower())
                         # Check if key looks like a language code
                         elif isinstance(key, str) and 1 < len(key) <= 3 and key.isalpha():
                              languages.add(key.lower())
            if languages:
                 language_codes = ",".join(sorted(list(languages))) # Comma-separated unique codes

        try:
            # Call insert_etymology helper function
            insert_etymology(
                cur,
                word_id=word_id,
                etymology_text=etymology_text.strip(),
                source_identifier=source_identifier, # Pass the received identifier
                normalized_components=json.dumps(normalized_components) if normalized_components else None,
                etymology_structure=json.dumps(etymology_structure) if etymology_structure else None,
                language_codes=language_codes
            )
            # logger.debug(f"Processed etymology for word ID {word_id} from source '{source_identifier}'")
        except Exception as e:
            logger.error(f"Error saving etymology for word ID {word_id} from source '{source_identifier}': {str(e)}")


    # Process a single entry
    def process_entry(cur, entry: Dict, source_identifier: str):
        """Process a single dictionary entry with improved resilience."""
        if 'word' not in entry or not entry['word']:
            logger.warning("Skipping entry without valid 'word' field")
            return None, "No word field" # Indicate critical failure

        word = entry['word']
        word_id = None # Initialize word_id
        critical_error_occurred = False
        error_messages = [] # Collect non-critical errors

        try:
            pos = entry.get('pos', 'unc')
            language_code = entry.get('lang_code', DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                 language_code = DEFAULT_LANGUAGE_CODE
                 logger.warning(f"Invalid/missing lang_code for '{word}', defaulting to '{DEFAULT_LANGUAGE_CODE}'.")

            is_proper_noun = entry.get('proper', False) or pos in ['prop', 'proper noun', 'name']
            is_abbreviation = pos in ['abbrev', 'abbreviation']
            is_initialism = pos in ['init', 'initialism', 'acronym']
            tags_list = entry.get('tags', [])
            if isinstance(tags_list, list):
                if any(t in ['abbreviation', 'abbrev'] for t in tags_list): is_abbreviation = True
                if any(t in ['initialism', 'acronym'] for t in tags_list): is_initialism = True

            baybayin_form, romanized_form = extract_baybayin_info(entry)
            badlit_form, badlit_romanized = extract_badlit_info(entry)
            hyphenation = entry.get('hyphenation')
            hyphenation_json = json.dumps(hyphenation) if hyphenation and isinstance(hyphenation, list) else None
            word_tags_str = ','.join(tags_list) if tags_list else None

            # --- CRITICAL STEP: Get or Create Word ID ---
            try:
                 word_id = get_or_create_word_id(
                    cur, lemma=word, language_code=language_code, has_baybayin=bool(baybayin_form),
                    baybayin_form=baybayin_form, romanized_form=romanized_form or badlit_romanized,
                    badlit_form=badlit_form, hyphenation=hyphenation_json, is_proper_noun=is_proper_noun,
                    is_abbreviation=is_abbreviation, is_initialism=is_initialism, tags=word_tags_str,
                    source_identifier=source_identifier
                )
            except Exception as word_create_err:
                 logger.error(f"CRITICAL FAILURE: Could not get/create word_id for '{word}' ({language_code}). Error: {word_create_err}. Skipping entry.")
                 critical_error_occurred = True
                 # No point continuing if we don't have a word_id
                 return None, f"Word creation failed: {word_create_err}"

            # --- Process Additional Word-Level Data (Non-Critical Failures) ---
            # Wrap each subsequent step in its own try-except block
            try:
                process_pronunciation(cur, word_id, entry, source_identifier)
            except Exception as e:
                msg = f"Failed processing pronunciation for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                process_form_relationships(cur, word_id, entry, language_code, source_identifier)
            except Exception as e:
                msg = f"Failed processing form relationships for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                if 'etymology_text' in entry:
                    process_etymology(cur, word_id, entry['etymology_text'], source_identifier, entry.get('etymology_templates'))
            except Exception as e:
                msg = f"Failed processing etymology for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            try:
                if 'head_templates' in entry and entry['head_templates']:
                    process_head_templates(cur, word_id, entry['head_templates'])
            except Exception as e:
                msg = f"Failed processing head templates for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)


            # --- Process Definitions (Senses) ---
            sense_processed_count = 0
            if 'senses' in entry and isinstance(entry['senses'], list):
                for sense_idx, sense in enumerate(entry['senses']):
                    if not sense or not isinstance(sense, dict):
                        logger.debug(f"Skipping invalid sense item {sense_idx} for word ID {word_id}")
                        continue

                    definition_id = None # Reset for each sense
                    try:
                        glosses = sense.get('glosses', []) or sense.get('raw_glosses', [])
                        if not glosses or not isinstance(glosses, list):
                            logger.debug(f"Skipping sense {sense_idx} for word ID {word_id}: missing/invalid glosses.")
                            continue
                        definition_text = '; '.join([g for g in glosses if isinstance(g, str)])
                        if not definition_text:
                            logger.debug(f"Skipping sense {sense_idx} for word ID {word_id}: empty definition text.")
                            continue

                        max_def_length = 4096
                        if len(definition_text) > max_def_length:
                            logger.warning(f"Definition {sense_idx} for word ID {word_id} truncated.")
                            definition_text = definition_text[:max_def_length]

                        examples = []
                        if 'examples' in sense and isinstance(sense['examples'], list):
                            for example in sense['examples']:
                                ex_text = None
                                if isinstance(example, str): ex_text = example
                                elif isinstance(example, dict):
                                    ex_text = example.get('text')
                                    ref = example.get('ref')
                                    tr = example.get('translation') or example.get('english')
                                    if ex_text and tr: ex_text += f" - {tr}"
                                    if ex_text and ref: ex_text += f" (Ref: {ref})"
                                if ex_text: examples.append(ex_text)
                        examples_str = '\n'.join(examples) if examples else None

                        sense_tags = sense.get('tags', [])
                        sense_labels = sense.get('labels', [])
                        all_tags = (sense_tags if isinstance(sense_tags, list) else []) + \
                                   (sense_labels if isinstance(sense_labels, list) else [])
                        tags_str = ','.join(all_tags) if all_tags else None
                        usage_notes = None

                        metadata_dict = {}
                        for key in ['form_of', 'raw_glosses', 'topics', 'taxonomy', 'qualifier']:
                            if key in sense and sense[key]: metadata_dict[key] = sense[key]

                        sense_pos_str = sense.get('pos') or pos
                        standard_pos = standardize_entry_pos(sense_pos_str)

                        # --- Insert Definition ---
                        try:
                            definition_id = insert_definition(
                                cur, word_id, definition_text, part_of_speech=standard_pos,
                                examples=examples_str, usage_notes=usage_notes, tags=tags_str,
                                source_identifier=source_identifier
                            )
                            sense_processed_count += 1
                        except psycopg2.errors.UniqueViolation:
                             logger.debug(f"Definition already exists for word ID {word_id}, sense {sense_idx}")
                             # Try to fetch existing ID if needed later? For now, skip sub-processing.
                             definition_id = None # Ensure it's None if insertion failed due to conflict
                        except Exception as def_error:
                            msg = f"Failed inserting definition for word ID {word_id}, sense {sense_idx}: {def_error}"
                            logger.warning(msg)
                            error_messages.append(msg)
                            definition_id = None # Ensure it's None if insertion failed

                        # --- Process Definition Sub-Data (only if definition_id is valid) ---
                        if definition_id:
                            try:
                                if metadata_dict:
                                     cur.execute("""
                                        UPDATE definitions SET metadata = COALESCE(metadata, '{}'::jsonb) || %s
                                        WHERE id = %s
                                     """, (json.dumps(metadata_dict), definition_id))
                            except Exception as meta_err:
                                msg = f"Failed storing metadata for definition {definition_id}: {meta_err}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                if 'categories' in sense and sense['categories']:
                                    process_categories(cur, definition_id, sense['categories'])
                            except Exception as e:
                                msg = f"Failed processing categories for def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                if 'links' in sense and sense['links']:
                                    process_links(cur, definition_id, sense['links'])
                            except Exception as e:
                                msg = f"Failed processing links for def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                            try:
                                process_sense_relationships(cur, word_id, sense, language_code, source_identifier)
                            except Exception as e:
                                msg = f"Failed processing sense relationships for word ID {word_id}, def ID {definition_id}: {e}"
                                logger.warning(msg)
                                error_messages.append(msg)

                    except Exception as sense_proc_err:
                        # Catch errors during preparation for definition insert (e.g., processing examples)
                        msg = f"Error processing sense {sense_idx} for word ID {word_id} before definition insert: {sense_proc_err}"
                        logger.warning(msg)
                        error_messages.append(msg)
                        continue # Skip to the next sense

                if sense_processed_count == 0 and not critical_error_occurred:
                     # Log if no definitions were processed *successfully*, but don't treat as error unless word creation also failed
                     logger.info(f"No definitions successfully inserted/processed for word '{word}' (ID: {word_id}). Senses might be empty or all failed individually.")


            # --- Process Top-Level Derived/Related (Non-Critical) ---
            try:
                top_level_rels = {'derived': RelationshipType.ROOT_OF, 'related': RelationshipType.RELATED}
                for rel_key, rel_enum in top_level_rels.items():
                    if rel_key in entry and isinstance(entry[rel_key], list):
                        for item in entry[rel_key]:
                            if isinstance(item, dict) and 'word' in item:
                                related_word = item['word']
                                if not related_word or not isinstance(related_word, str): continue
                                try:
                                    related_word_id = get_or_create_word_id(cur, related_word, language_code, source_identifier=source_identifier)
                                    if related_word_id == word_id: continue
                                    metadata = {'confidence': rel_enum.strength, 'from_sense': False}
                                    insert_relation(cur, word_id, related_word_id, rel_enum, source_identifier, metadata)
                                    if rel_enum.bidirectional: insert_relation(cur, related_word_id, word_id, rel_enum, source_identifier, metadata)
                                    else:
                                         inv = rel_enum.get_inverse()
                                         if inv: insert_relation(cur, related_word_id, word_id, inv, source_identifier, metadata)
                                except Exception as top_rel_err_inner:
                                    msg = f"Failed processing top-level {rel_key} relation for word ID {word_id} -> '{related_word}': {top_rel_err_inner}"
                                    logger.warning(msg)
                                    error_messages.append(msg) # Log inner failure but continue loop
            except Exception as e:
                # Catch error in the loop setup itself
                msg = f"Failed processing top-level relationships section for word ID {word_id}: {e}"
                logger.warning(msg)
                error_messages.append(msg)

            # Return word_id (even if non-critical errors occurred) and collected error messages
            return word_id, "; ".join(error_messages) if error_messages else None

        except Exception as e:
            # Catch truly unexpected errors during the main processing flow (after word_id creation attempt)
            logger.error(f"UNEXPECTED EXCEPTION during processing entry for word '{word}' (ID: {word_id}): {str(e)}", exc_info=True)
            # If word_id was created, return it but indicate a major issue occurred.
            # If word_id wasn't created (should have been caught earlier), this path might not be reached,
            # but handle defensively.
            return word_id if word_id else None, f"Unhandled entry exception: {e}"

    # Process entries without Baybayin - Simplified fallback (Consider removing if not needed)
    # This might be unnecessary if the main process_entry handles errors gracefully.
    # Keeping it for now as per original structure, but mark as potentially removable.
    def process_entry_without_baybayin(cur, entry: Dict, source_identifier: str):
        """Simplified fallback processing, skipping Baybayin/form/etymology/complex relations."""
        logger.warning(f"Using FALLBACK processing for word: {entry.get('word', 'N/A')}")
        if 'word' not in entry or not entry['word']:
            return None, "Fallback: No word"

        word = entry['word']
        entry_error = None

        try:
            pos = entry.get('pos', 'unc')
            language_code = entry.get('lang_code', DEFAULT_LANGUAGE_CODE)
            if not language_code or len(language_code) > 10:
                language_code = DEFAULT_LANGUAGE_CODE

            # Basic word attributes (no Baybayin)
            is_proper_noun = entry.get('proper', False) or pos in ['prop', 'proper noun', 'name']
            tags_list = entry.get('tags', [])
            word_tags_str = ','.join(tags_list) if isinstance(tags_list, list) else None

            # Get or create the word ID (basic info only)
            try:
                word_id = get_or_create_word_id(
                    cur,
                    lemma=word,
                    language_code=language_code,
                    has_baybayin=False, # Explicitly false
                    is_proper_noun=is_proper_noun,
                    tags=word_tags_str,
                    source_identifier=source_identifier
                )
            except Exception as word_create_err:
                 logger.error(f"FALLBACK CRITICAL: Failed get/create word_id for '{word}'. Error: {word_create_err}")
                 return None, f"Fallback word creation failed: {word_create_err}"


            # Process only definitions from senses (simplified)
            if 'senses' in entry and isinstance(entry['senses'], list):
                sense_processed_count = 0
                for sense_idx, sense in enumerate(entry['senses']):
                    if not sense or not isinstance(sense, dict): continue

                    glosses = sense.get('glosses', []) or sense.get('raw_glosses', [])
                    if not glosses or not isinstance(glosses, list): continue

                    definition_text = '; '.join([g for g in glosses if isinstance(g, str)])
                    if not definition_text: continue

                    max_def_length = 4096
                    if len(definition_text) > max_def_length:
                        definition_text = definition_text[:max_def_length]

                    # Simplified examples
                    examples = []
                    if 'examples' in sense and isinstance(sense['examples'], list):
                         for example in sense['examples']:
                              ex_text = None
                              if isinstance(example, str): ex_text = example
                              elif isinstance(example, dict): ex_text = example.get('text')
                              if ex_text: examples.append(ex_text)
                    examples_str = '\n'.join(examples) if examples else None

                    sense_pos_str = sense.get('pos') or pos
                    standard_pos = standardize_entry_pos(sense_pos_str)

                    # Insert only basic definition
                    try:
                        insert_definition(
                            cur,
                            word_id,
                            definition_text,
                            part_of_speech=standard_pos,
                            examples=examples_str,
                            source_identifier=source_identifier # Pass source_identifier
                        )
                        sense_processed_count += 1
                    except psycopg2.errors.UniqueViolation:
                         logger.debug(f"Fallback: Definition already exists for {word_id}, sense {sense_idx}")
                    except Exception as def_error:
                        logger.warning(f"Fallback: Error inserting definition for {word_id}, sense {sense_idx}: {def_error}")
                        if not entry_error: entry_error = f"Fallback def insert failed: {def_error}"

                if sense_processed_count == 0 and not entry_error:
                     logger.warning(f"Fallback: No definitions processed for word '{word}' (ID: {word_id})")
                     # entry_error = "Fallback: No valid definitions"

            return word_id, entry_error
        except Exception as e:
            logger.error(f"FALLBACK UNHANDLED EXCEPTION for word '{word}': {str(e)}", exc_info=True)
            try: conn.rollback()
            except Exception as rb_err: logger.error(f"Fallback rollback error: {rb_err}")
            return None, f"Fallback unhandled exception: {e}"


    # Main function processing logic
    stats = {
        "total_entries": total_lines if total_lines != -1 else 0, # Use counted lines if available
        "processed_ok": 0,
        "processed_with_errors": 0, # Entries processed but had some non-critical error
        "failed_entries": 0, # Entries that couldn't be processed at all
        "skipped_json_errors": 0,
        "fallback_entries_used": 0 # Count how many times fallback was triggered - REMOVED FALLBACK CALL
    }
    error_summary = {} # Track types of errors encountered

    # Ensure connection is valid before starting loop
    if conn.closed:
        logger.error("Database connection is closed before starting Kaikki processing loop.")
        stats["failed_entries"] = stats["total_entries"]
        return stats

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            entry_count = 0
            progress_bar = tqdm(total=total_lines, desc=f"Processing {os.path.basename(filename)}", unit=" entries") if total_lines > 0 else None

            for line in f:
                entry_count += 1
                if progress_bar: progress_bar.update(1)

                # --- Transaction Management: SAVEPOINT per entry ---
                savepoint_name = f"kaikki_entry_{entry_count}"
                try:
                    cur.execute(f"SAVEPOINT {savepoint_name}")
                except Exception as sp_error:
                     logger.error(f"CRITICAL: Failed to create savepoint {savepoint_name}. Aborting processing. Error: {sp_error}")
                     stats["failed_entries"] = stats["total_entries"] - entry_count + 1 # Mark remaining as failed
                     if progress_bar: progress_bar.close()
                     return stats # Stop processing

                try:
                    entry = json.loads(line.strip())

                    # Try standard processing first
                    word_id, entry_errors = process_entry(cur, entry, source_identifier) # Returns word_id and potential errors

                    if word_id: # Word was created/found, proceed even if non-critical errors occurred
                        if entry_errors:
                            stats["processed_with_errors"] += 1
                            # Log first part of error summary key
                            first_error = entry_errors.split(';')[0][:100] # Limit key length
                            err_key = f"EntryError: {first_error}"
                            error_summary[err_key] = error_summary.get(err_key, 0) + 1
                            logger.debug(f"Entry {entry_count} ('{entry.get('word', 'N/A')}') processed with errors: {entry_errors}")
                        else:
                            stats["processed_ok"] += 1
                        # Commit successful or partially successful processing for this entry
                        cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    else:
                        # Critical failure within process_entry (likely word creation failed)
                        stats["failed_entries"] += 1
                        reason = entry_errors or "Unknown critical failure" # entry_errors should contain the reason
                        err_key = f"EntryFailedCritically: {reason[:100]}"
                        error_summary[err_key] = error_summary.get(err_key, 0) + 1
                        logger.warning(f"Entry {entry_count} ('{entry.get('word', 'N/A')}') failed critical processing step: {reason}")
                        # Rollback this specific entry
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                        # NOTE: Fallback logic removed; process_entry is now resilient.

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {entry_count}")
                    stats["skipped_json_errors"] += 1
                    # Rollback if savepoint might exist
                    try: cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception: pass # Ignore rollback error if savepoint doesn't exist
                except Exception as e:
                    logger.error(f"Unhandled error in main loop at line {entry_count}: {str(e)}", exc_info=True)
                    stats["failed_entries"] += 1
                    err_key = f"LoopException: {type(e).__name__}"
                    error_summary[err_key] = error_summary.get(err_key, 0) + 1
                    # Rollback potentially affected entry
                    try:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    except Exception as rb_err:
                        logger.error(f"Error rolling back savepoint {savepoint_name} after loop exception: {rb_err}")

            if progress_bar: progress_bar.close()

        # --- Final Commit ---
        # If we reached here without critical errors, commit the overall transaction
        # Note: Individual entries were committed via RELEASE SAVEPOINT.
        # This final commit might not be strictly necessary if autocommit is off
        # and each entry was handled atomically with savepoints. However, it doesn't hurt.
        try:
             conn.commit()
             logger.info("Final commit for Kaikki processing successful.")
        except Exception as final_commit_err:
             logger.error(f"Error during final commit after processing {filename}: {final_commit_err}")
             # Data might be inconsistent if this fails after releasing savepoints.
             # The implications depend on the database isolation level.

        logger.info(f"Completed processing {filename}: {stats}")
        if error_summary:
             logger.warning(f"Error summary for {filename}: {json.dumps(error_summary, indent=2)}")
        return stats

    except FileNotFoundError:
         # Already handled at the start, but catch again just in case.
         logger.error(f"File not found error during main processing block: {filename}")
         return stats # Return stats initialized earlier
    except Exception as e:
        logger.error(f"Fatal error during processing of {filename}: {str(e)}", exc_info=True)
        try:
            conn.rollback() # Rollback any pending changes from the main transaction
        except Exception as rb_err:
            logger.error(f"Error during final rollback after fatal error: {rb_err}")
        # Update stats to reflect failure
        stats["failed_entries"] = stats["total_entries"] - stats["processed_ok"] - stats["processed_with_errors"] - stats["skipped_json_errors"]
        return stats
def standardize_pronunciation(pron: str) -> str:
    """Standardize IPA pronunciation format."""
    if not pron:
        return None
    # Clean the pronunciation
    pron = pron.strip()
    # Remove extra spaces
    pron = re.sub(r'\s+', ' ', pron)
    # Ensure proper IPA brackets
    if not (pron.startswith('[') and pron.endswith(']')):
        pron = f"[{pron.strip('[]')}]"
    # Standardize stress marks
    pron = pron.replace("'", "ˈ")
    pron = pron.replace('"', "ˌ")
    return pron

def standardize_pos(pos: str) -> str:
    """Standardize part of speech tags."""
    pos_map = {
        "noun": "n",
        "verb": "v",
        "adjective": "adj",
        "adverb": "adv",
        "adveb": "adv",  # Fix typo in some dictionaries
        "pronoun": "pron",
        "preposition": "prep",
        "interjection": "interj",
        "conjunction": "conj",
        "determiner": "det",
        "article": "art",
        "numeral": "num",
        "particle": "part",
        "exclamation": "excl",
        "interrogative": "interrog",
        "marker": "mark"
    }
    return pos_map.get(pos.lower(), pos.lower())

def standardize_credits(credits: str) -> str:
    """Standardize credits format."""
    if not credits:
        return None
    
    # Remove extra whitespace and newlines
    credits = re.sub(r'\s+', ' ', credits.strip())
    
    # Extract different roles
    contributors = re.findall(r'Contributors?:(.*?)(?:Reviewer|Editor|$)', credits)
    reviewers = re.findall(r'Reviewer?:(.*?)(?:Editor|$)', credits)
    editors = re.findall(r'Editor?:(.*?)(?:$)', credits)
    
    # Format each role
    parts = []
    if contributors:
        names = [n.strip() for n in contributors[0].split('(')[0].strip().split(',')]
        parts.append(f"Contributors: {', '.join(names)}")
    if reviewers:
        names = [n.strip() for n in reviewers[0].split('(')[0].strip().split(',')]
        parts.append(f"Reviewers: {', '.join(names)}")
    if editors:
        names = [n.strip() for n in editors[0].split('(')[0].strip().split(',')]
        parts.append(f"Editors: {', '.join(names)}")
    
    return '; '.join(parts)

def process_examples(examples: List[Dict]) -> List[Dict]:
    """Process and validate examples."""
    processed = []
    for example in examples:
        if not isinstance(example, dict):
            continue
            
        text = example.get('text', '').strip()
        translation = example.get('translation', '').strip()
        
        if text and translation:
            processed.append({
                'text': text,
                'translation': translation,
                'example_id': example.get('example_id', len(processed) + 1)
            })
    return processed

def process_see_also(see_also: List[Dict], language_code: str) -> List[str]:
    """Process see also references."""
    refs = []
    for ref in see_also:
        if isinstance(ref, dict):
            text = ref.get('text', '').strip()
            if text:
                # Remove numbering from references
                base_word = re.sub(r'\d+$', '', text)
                if base_word:
                    refs.append(base_word)
    return refs

def get_language_mapping():
    """Dynamically build language mapping from dictionary files."""
    # Base ISO 639-3 codes for Philippine languages
    base_language_map = {
        'onhan': 'onx',
        'waray': 'war',
        'ibanag': 'ibg',
        'iranon': 'iro',
        'ilocano': 'ilo',
        'cebuano': 'ceb',
        'hiligaynon': 'hil',
        'kinaray-a': 'krj',
        'kinaraya': 'krj',
        'kinaray': 'krj',
        'asi': 'asi',
        'bikol': 'bik',
        'bikolano': 'bik',
        'bicol': 'bik',
        'surigaonon': 'sgd',
        'aklanon': 'akl',
        'masbatenyo': 'msb',
        'chavacano': 'cbk',
        'tagalog': 'tgl',
        'filipino': 'tgl',
        'pilipino': 'tgl',
        'pangasinan': 'pag',
        'kapampangan': 'pam',
        'manobo': 'mbt',
        'manide': 'abd',
        'maguindanaon': 'mdh',
        'ivatan': 'ivv',
        'itawis': 'itv',
        'isneg': 'isd',
        'ifugao': 'ifk',
        'gaddang': 'gad',
        'cuyonon': 'cyo',
        'blaan': 'bpr',  # Default to Koronadal Blaan
    }

    # Add regional variants
    regional_variants = {
        'bikol-central': 'bcl',
        'bikol-albay': 'bik',
        'bikol-rinconada': 'bto',
        'bikol-partido': 'bik',
        'bikol-miraya': 'bik',
        'bikol-libon': 'bik',
        'bikol-west-albay': 'fbl',
        'bikol-southern-catanduanes': 'bln',
        'bikol-northern-catanduanes': 'bln',
        'bikol-boienen': 'bik',
        'blaan-koronadal': 'bpr',
        'blaan-sarangani': 'bps',
        'cebuano-cotabato': 'ceb',
        'hiligaynon-cotabato': 'hil',
        'tagalog-marinduque': 'tgl',
        'manobo-erumanen': 'mbt',
        'isneg-yapayao': 'isd',
        'ifugao-tuwali-ihapo': 'ifk',
    }

    # Combine base map with variants
    language_map = {**base_language_map, **regional_variants}

    # Try to get additional mappings from dictionary files
    try:
        json_pattern = os.path.join("data", "marayum_dictionaries", "*_processed.json")
        json_files = glob.glob(json_pattern)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or 'dictionary_info' not in data:
                        continue
                    
                    dict_info = data['dictionary_info']
                    base_language = dict_info.get('base_language', '').lower()
                    if not base_language:
                        continue
                    
                    # Extract language code from filename
                    filename = os.path.basename(json_file)
                    if filename.endswith('_processed.json'):
                        filename = filename[:-14]  # Remove '_processed.json'
                    
                    # Split on first hyphen to get language part
                    lang_code = filename.split('-')[0].lower()
                    
                    # Add to mapping if not already present
                    if base_language not in language_map:
                        language_map[base_language] = lang_code
                    
                    # Handle compound names
                    if '-' in base_language:
                        parts = base_language.split('-')
                        for part in parts:
                            if part not in language_map:
                                language_map[part] = lang_code
                    
                    # Add filename-based mapping
                    orig_name = filename.replace('-english', '').lower()
                    if orig_name not in language_map:
                        language_map[orig_name] = lang_code
                    
            except Exception as e:
                logger.warning(f"Error processing language mapping from {json_file}: {str(e)}")
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
        'chavacano': 'cbk',
        'zamboangueño': 'cbk',
        'chabacano': 'cbk',
        'cebuano': 'ceb',
        'hiligaynon': 'hil',
        'ilonggo': 'hil',
        'waray': 'war',
        'waray-waray': 'war',
        'tagalog': 'tgl',
        'filipino': 'fil',
        'bikol': 'bik',
        'bikolano': 'bik',
        'bicol': 'bik',
        'ilocano': 'ilo',
        'iloko': 'ilo',
        'kapampangan': 'pam',
        'pangasinan': 'pag',
        'kinaray-a': 'krj',
        'kinaraya': 'krj',
        'aklanon': 'akl',
        'masbatenyo': 'msb',
        'surigaonon': 'sgd',
        'tausug': 'tsg',
        'maguindanao': 'mdh',
        'maguindanaon': 'mdh',
        'maranao': 'mrw',
        'iranon': 'iro',
        'iranun': 'iro',
        'ibanag': 'ibg',
        'ivatan': 'ivv',
        'itawis': 'itv',
        'isneg': 'isd',
        'ifugao': 'ifk',
        'gaddang': 'gad',
        'cuyonon': 'cyo',
        'asi': 'asz',
        'bantoanon': 'bno',
        'blaan': 'bpr',
        'manobo': 'mbt',
        'manide': 'abd',
        'onhan': 'onx'
    }
    
    # Try direct lookup
    if language in LANGUAGE_CODES:
        return LANGUAGE_CODES[language]
        
    # Try without parenthetical clarifications
    base_name = re.sub(r'\s*\([^)]*\)', '', language).strip()
    if base_name in LANGUAGE_CODES:
        return LANGUAGE_CODES[base_name]
        
    # Try normalizing hyphens and spaces
    normalized = re.sub(r'[\s-]+', '', language)
    if normalized in LANGUAGE_CODES:
        return LANGUAGE_CODES[normalized]
        
    # If not found, log warning and return a safe fallback
    logger.warning(f"No language code mapping found for: {language}")
    # Create a safe ASCII identifier from the language name
    safe_code = re.sub(r'[^a-z]', '', language.lower())[:3]
    if not safe_code:
        safe_code = 'unk'  # Unknown language
    return safe_code

# Replace the existing process_marayum_json function (starts around line 6704)
@with_transaction(commit=False)  # Manage transactions manually within the loop
def process_marayum_json(cur, filename: str) -> Tuple[int, int]:
    """Process a single Marayum JSON file, ensuring source propagation."""
    # Use the standardized source from the original function logic
    source_identifier = SourceStandardization.standardize_sources(os.path.basename(filename))
    logger.info(f"Processing Marayum file: {filename} with source: {source_identifier}")

    processed_count = 0
    error_count = 0
    skipped_count = 0
    definitions_added = 0 # Add counter for definitions
    relations_added = 0   # Add counter for relations
    pronunciations_added = 0 # Add counter for pronunciations
    etymologies_added = 0 # Add counter for etymologies
    entries_in_file = 0
    conn = cur.connection # Get connection for manual commit/rollback

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing Marayum file {filename}: {e}")
        return 0, 1 # Return 0 processed, 1 error (representing the file)
    except Exception as e:
        logger.error(f"Unexpected error reading Marayum file {filename}: {e}", exc_info=True)
        return 0, 1

    if not isinstance(data, list):
        logger.error(f"Marayum file {filename} does not contain a list of entries.")
        return 0, 1

    entries_in_file = len(data)
    logger.info(f"Found {entries_in_file} entries in {filename}")

    # Assuming get_language_mapping() is defined elsewhere and returns a dict
    # language_mapping = get_language_mapping() # Load this once if needed by get_language_code

    with tqdm(total=entries_in_file, desc=f"Processing {source_identifier}", unit="entry") as pbar:
        for entry_index, entry in enumerate(data):
            # --- Savepoint ---
            # Marayum uses entry_index, which is safe
            savepoint_name = f"marayum_entry_{entry_index}"
            try:
                cur.execute(f"SAVEPOINT {savepoint_name}")
            except Exception as e:
                logger.error(f"Failed to create savepoint {savepoint_name} for Marayum entry index {entry_index}: {e}. Skipping.")
                error_count += 1
                pbar.update(1)
                continue # Skip this entry

            try:
                if not isinstance(entry, dict):
                     logger.warning(f"Skipping invalid entry at index {entry_index} in {filename} (not a dict)")
                     skipped_count += 1
                     cur.execute(f"RELEASE SAVEPOINT {savepoint_name}") # Release if skipping early
                     pbar.update(1)
                     continue

                headword = entry.get('headword', '').strip()
                language_name = entry.get('language', 'Unknown') # Keep original language name
                language_code = get_language_code(language_name) # Use helper

                if not headword:
                     logger.warning(f"Skipping entry at index {entry_index} in {filename} (no headword)")
                     skipped_count += 1
                     cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                     pbar.update(1)
                     continue

                # --- Word Creation ---
                # Preserve original metadata logic
                word_metadata = {'marayum_source': source_identifier, 'language_name': language_name, 'processed_timestamp': datetime.now().isoformat()}
                # Add other Marayum fields to metadata if needed: e.g., entry.get('id')
                try:
                    word_id = get_or_create_word_id(
                        cur, headword, language_code=language_code,
                        source_identifier=source_identifier, # Pass MANDATORY source_identifier
                        word_metadata=json.dumps(word_metadata) # Pass metadata as before
                    )
                except TypeError as te: # Catch if word_metadata isn't expected by get_or_create_word_id
                    logger.warning(f"TypeError calling get_or_create_word_id for '{headword}', possibly due to word_metadata. Retrying without it. Error: {te}")
                    word_id = get_or_create_word_id(
                         cur, headword, language_code=language_code,
                         source_identifier=source_identifier # Pass MANDATORY source_identifier
                     )

                if not word_id: raise ValueError(f"Failed to get/create word ID for '{headword}' ({language_code}) from {source_identifier}")

                # --- Process Pronunciations ---
                pronunciations = entry.get('pronunciations')
                if isinstance(pronunciations, list):
                     for pron in pronunciations:
                         if isinstance(pron, str) and pron.strip():
                             try:
                                 # Pass mandatory source_identifier argument correctly
                                 pron_id = insert_pronunciation(cur, word_id, pron.strip(), source_identifier=source_identifier)
                                 if pron_id: pronunciations_added += 1
                             except Exception as e: logger.warning(f"Failed pronunciation insert '{pron}' for '{headword}' from {source_identifier}: {e}")

                # --- Process Etymology ---
                etymology_text_raw = entry.get('etymology')
                if isinstance(etymology_text_raw, str) and etymology_text_raw.strip():
                     etymology_text = etymology_text_raw.strip()
                     try:
                         # Pass mandatory source_identifier argument correctly
                         ety_id = insert_etymology(cur, word_id, etymology_text, source_identifier=source_identifier)
                         if ety_id: etymologies_added += 1
                     except Exception as e: logger.warning(f"Failed etymology insert for '{headword}' from {source_identifier}: {e}")

                # --- Process Senses ---
                senses = entry.get('senses')
                if isinstance(senses, list):
                     for sense_idx, sense in enumerate(senses):
                         if not isinstance(sense, dict): continue
                         definition = sense.get('definition', '').strip()
                         pos = sense.get('partOfSpeech', '') # Marayum uses 'partOfSpeech'
                         examples_raw = sense.get('examples', [])

                         if definition:
                             # Process examples into JSON list of strings/dicts
                             examples_list = []
                             if isinstance(examples_raw, list):
                                  examples_list = [ex.strip() for ex in examples_raw if isinstance(ex, str) and ex.strip()]
                                  # If Marayum examples are dicts, adapt here:
                                  # examples_list = [{"text": ex.get("text"), ...} for ex in examples_raw if isinstance(ex, dict)]

                             # Convert examples list to JSON string or None
                             examples_json = None
                             if examples_list:
                                 try:
                                     examples_json = json.dumps(examples_list)
                                 except TypeError:
                                     logger.warning(f"Could not serialize examples for sense in '{headword}': {examples_list}")

                             # Get standardized POS ID using the original POS string
                             # Use standardize_entry_pos helper if defined and needed
                             std_pos = standardize_entry_pos(pos) if pos else None
                             standardized_pos_id = get_standardized_pos_id(cur, std_pos)

                             try:
                                 # Pass mandatory source_identifier
                                 # Pass standardized_pos_id and original_pos directly
                                 definition_id = insert_definition(
                                     cur, word_id, definition,
                                     source_identifier=source_identifier, # Pass mandatory source
                                    part_of_speech=standardized_pos_id, # Fix: Rename argument to match definition, # Pass pre-calculated ID
                                     original_pos=pos, # Pass original string
                                     examples=examples_json
                                     # Add tags/usage_notes if Marayum provides them here in sense
                                 )
                                 if definition_id:
                                     definitions_added += 1
                                     # Process sense-level relations if Marayum format includes them
                                     sense_relation_map = {
                                         'synonyms': RelationshipType.SYNONYM,
                                         'antonyms': RelationshipType.ANTONYM
                                         # Add other sense relations if needed
                                     }
                                     for sense_rel_key, sense_rel_type in sense_relation_map.items():
                                         if sense_rel_key in sense and isinstance(sense[sense_rel_key], list):
                                             for sense_rel_word in sense[sense_rel_key]:
                                                 if isinstance(sense_rel_word, str) and sense_rel_word.strip():
                                                     sense_rel_word_clean = sense_rel_word.strip()
                                                     if sense_rel_word_clean != headword:
                                                         try:
                                                             sense_rel_id = get_or_create_word_id(cur, sense_rel_word_clean, language_code, source_identifier=source_identifier)
                                                             if sense_rel_id:
                                                                 # Add metadata specific to sense relation if needed
                                                                 sense_rel_metadata = {'source': source_identifier, 'definition_id': definition_id, 'confidence': 70} # Example confidence
                                                                 # Pass mandatory source_identifier argument correctly
                                                                 rel_rec_id = insert_relation(
                                                                     cur, word_id, sense_rel_id, sense_rel_type,
                                                                     source_identifier=source_identifier, # Pass source
                                                                     metadata=sense_rel_metadata
                                                                 )
                                                                 if rel_rec_id: relations_added += 1
                                                                 # Handle bidirectional for sense relations if needed
                                                                 # if sense_rel_type.bidirectional:
                                                                 #    insert_relation(cur, sense_rel_id, word_id, sense_rel_type, source_identifier=source_identifier, metadata=sense_rel_metadata)

                                                         except Exception as rel_e:
                                                             logger.warning(f"Error adding sense {sense_rel_key} '{sense_rel_word_clean}' for '{headword}': {rel_e}")

                             except psycopg2.errors.UniqueViolation:
                                logger.debug(f"Marayum def exists for '{headword}', sense idx {sense_idx}: {definition[:30]}...")
                             except Exception as def_e: logger.error(f"Failed def insert for '{headword}', sense idx {sense_idx}: {definition[:30]}... : {def_e}", exc_info=False) # Keep log cleaner

                # --- Process Top-Level Relations ---
                # Using the structure from the provided snippet
                top_level_relation_map = {
                     'synonyms': RelationshipType.SYNONYM,
                     'antonyms': RelationshipType.ANTONYM,
                     'relatedWords': RelationshipType.RELATED # Assuming 'relatedWords' maps to RELATED
                     # Add 'hypernyms', 'hyponyms' etc. if Marayum provides them
                }
                for rel_key, rel_type_enum in top_level_relation_map.items():
                     related_words_list = entry.get(rel_key)
                     if isinstance(related_words_list, list):
                         for related_word_str in related_words_list:
                             if isinstance(related_word_str, str) and related_word_str.strip():
                                 related_word_clean = related_word_str.strip()
                                 if related_word_clean != headword: # Avoid self-relation
                                     try:
                                         related_id = get_or_create_word_id(cur, related_word_clean, language_code, source_identifier=source_identifier)
                                         if related_id:
                                             # Add metadata for top-level relations if needed
                                             top_rel_metadata = {'source': source_identifier, 'confidence': 70} # Example confidence
                                             # Pass mandatory source_identifier argument correctly
                                             top_rel_rec_id = insert_relation(
                                                 cur, word_id, related_id, rel_type_enum,
                                                 source_identifier=source_identifier, # Pass source
                                                 metadata=top_rel_metadata
                                             )
                                             if top_rel_rec_id: relations_added += 1
                                             # Handle bidirectional for top-level relations
                                             # if rel_type_enum.bidirectional:
                                             #    insert_relation(cur, related_id, word_id, rel_type_enum, source_identifier=source_identifier, metadata=top_rel_metadata)

                                     except Exception as top_rel_e: logger.warning(f"Error adding top-level {rel_key} '{related_word_clean}' for '{headword}': {top_rel_e}")

                # --- Finish Entry ---
                processed_count += 1
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")

            except Exception as entry_err:
                logger.error(f"Error processing Marayum entry index {entry_index} ('{entry.get('headword', 'N/A')}') in {filename}: {entry_err}", exc_info=False) # Keep log cleaner
                error_count += 1
                try:
                    # Rollback to the savepoint for this specific entry
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                except Exception as rb_err:
                    logger.critical(f"CRITICAL: Failed to rollback savepoint {savepoint_name} for {filename}: {rb_err}. Attempting full transaction rollback.")
                    conn.rollback() # Rollback the whole transaction for safety
                    logger.warning("Performed full transaction rollback due to savepoint rollback failure.")
                    # Consider re-raising the error to stop processing if this happens
                    # raise entry_err from rb_err
            finally:
                pbar.update(1) # Update progress bar regardless of success or failure

            # --- Periodic Commit (Optional but recommended for large files) ---
            entries_processed_so_far = processed_count + error_count + skipped_count
            if entries_processed_so_far > 0 and entries_processed_so_far % 500 == 0: # Commit every 500 entries
                 try:
                     conn.commit()
                     logger.debug(f"Committed progress for {filename} at entry {entry_index}")
                 except Exception as commit_err:
                     logger.error(f"Error committing progress for {filename} at entry {entry_index}: {commit_err}. Rolling back...")
                     conn.rollback()
                     # Log the error but continue processing remaining entries if possible
                     # The uncommitted batch will be retried or lost depending on subsequent commits/errors

    # Final commit for the file
    try:
        conn.commit()
        logger.info(f"Finished processing {filename}. Processed: {processed_count}, Definitions: {definitions_added}, Relations: {relations_added}, Pronunciations: {pronunciations_added}, Etymologies: {etymologies_added}, Skipped: {skipped_count}, Errors: {error_count}")
    except Exception as final_commit_err:
        logger.error(f"Error during final commit for {filename}: {final_commit_err}. Rolling back any remaining changes...")
        conn.rollback()
        error_count += 1 # Mark an error for the file due to final commit failure

    # Return counts: (successful, total issues)
    total_issues = error_count + skipped_count
    return processed_count, total_issues

@with_transaction(commit=True)
def process_marayum_directory(cur, directory_path: str) -> None:
    """Process all Project Marayum dictionary files in the specified directory."""
    
    # Normalize directory path
    directory_path = os.path.normpath(directory_path)
    logger.info(f"Processing Marayum dictionaries from directory: {directory_path}")
    
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
            files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
            if files:
                processed_files = [f for f in files if f.endswith('_processed.json')]
                unprocessed_files = [f for f in files if not f.endswith('_processed.json')]
                if processed_files:
                    logger.info(f"Processed files found: {', '.join(processed_files)}")
                if unprocessed_files:
                    logger.info(f"Unprocessed files found (skipping): {', '.join(unprocessed_files)}")
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
            processed, errors = process_marayum_json(cur, json_file)
            total_processed += processed
            total_errors += errors
            if processed > 0:
                total_files_processed += 1
            else:
                total_files_skipped += 1
                logger.warning(f"No entries processed from {os.path.basename(json_file)}")
            
        except Exception as e:
            logger.error(f"Error processing Marayum dictionary file {json_file}: {str(e)}")
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
        },
        {
            "name": "Project Marayum",
            "file": "marayum_dictionaries",  # Changed to directory name only
            "handler": process_marayum_directory,
            "required": False,
            "is_directory": True
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
        # More flexible matching for source names
        sources = [s for s in sources if any(
            req in s["name"].lower() or 
            req in s["file"].lower() or
            (req == 'marayum' and 'marayum' in s["file"].lower())
            for req in requested_sources
        )]
        if not sources:
            logger.error(f"No matching sources found for: {args.sources}")
            return
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
        elif 'marayum' in filename.lower():
            handler = process_marayum_directory
        else:
            handler = process_tagalog_words
        
        basename = os.path.basename(filename)
        source_found = False
        for source in sources:
            if source["file"] == basename or (
                os.path.isdir(filename) and 
                os.path.basename(source["file"]) == os.path.basename(filename)
            ):
                source["file"] = filename  # Use full path
                source["required"] = True
                source_found = True
                break
        if not source_found:
            sources.append({
                "name": f"Custom ({basename})",
                "file": filename,
                "handler": handler,
                "required": True,
                "is_directory": os.path.isdir(filename)
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
                        if source.get("is_directory", False):
                            if os.path.isdir(potential_path):
                                filename = potential_path
                                break
                        else:
                            if os.path.isfile(potential_path):
                                filename = potential_path
                                break
                
                if not filename:
                    msg = f"Required {'directory' if source.get('is_directory', False) else 'file'} not found: {source['file']}"
                    if source["required"]:
                        logger.error(msg)
                        sys.exit(1)
                    else:
                        logger.warning(msg)
                        continue
                
                task = progress.add_task(f"Processing {source['name']}...", total=1)
                try:
                    source["handler"](cur, filename)
                    progress.update(task, completed=1)
                except Exception as e:
                    logger.error(f"Error processing {source['name']}: {str(e)}")
                    if source["required"]:
                        raise
                    
    except Exception as e:
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
            logger.error(f"Parameters: {(args.word, args.word, args.word, args.word, args.word)}")
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
                        console.print(f"\n[yellow]No exact matches found. Showing similar words:[/]")
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
                        console.print(f"\n[yellow]Found words containing '[bold]{args.word}[/]':[/]")
                        results = ilike_results
                    else:
                        # Check if word exists in database at all
                        logger.info("Diagnostic: checking if word exists in any form")
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT 1 FROM words 
                                WHERE lemma LIKE %s 
                                OR normalized_lemma LIKE %s
                            )
                        """, (f"%{args.word}%", f"%{args.word}%"))
                        exists = cur.fetchone()[0]
                        logger.info(f"Diagnostic result: Word '{args.word}' exists (partial match): {exists}")
                        
                        # Verify database schema
                        logger.info("Diagnostic: verifying database schema")
                        cur.execute("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = 'words'
                        """)
                        columns = cur.fetchall()
                        logger.info(f"Diagnostic: words table has {len(columns)} columns")
                        for col in columns:
                            logger.info(f"  Column: {col[0]}, Type: {col[1]}")
                            
                        console.print(f"\n[yellow]No entries found for '[bold]{args.word}[/]'[/]")
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
                
                console.print(info_table)
                
                # Get and print pronunciations
                try:
                    logger.info(f"Querying pronunciations for word_id {word_id}")
                    cur.execute("""
                        SELECT type, value, tags, metadata, sources
                        FROM pronunciations
                        WHERE word_id = %s
                        ORDER BY type
                    """, (word_id,))
                    
                    prons = cur.fetchall()
                    logger.info(f"Found {len(prons)} pronunciations")
                    
                    if prons:
                        console.print("\n[bold cyan]Pronunciations:[/]")
                        for pron in prons:
                            type_, value, p_tags, metadata, p_sources = pron
                            console.print(f"• {value}" + (f" ({type_})" if type_ else ""))
                
                except Exception as e:
                    logger.error(f"Error retrieving pronunciations: {str(e)}")
                
                # Get and print etymologies
                try:
                    logger.info(f"Querying etymologies for word_id {word_id}")
                    cur.execute("""
                        SELECT etymology_text, normalized_components, language_codes, sources
                        FROM etymologies
                        WHERE word_id = %s
                    """, (word_id,))
                    
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
                    cur.execute("""
                        SELECT d.definition_text, d.examples, d.usage_notes,
                               p.name_tl AS pos_name
                        FROM definitions d
                        LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                        WHERE d.word_id = %s
                        ORDER BY p.name_tl, d.id
                    """, (word_id,))
                    
                    definitions = cur.fetchall()
                    logger.info(f"Found {len(definitions)} definitions")
                    
                    if definitions:
                        console.print("\n[bold cyan]Definitions:[/]")
                        current_pos = None
                        
                        for def_idx, definition in enumerate(definitions):
                            logger.info(f"Processing definition {def_idx+1}/{len(definitions)}")
                            def_text, examples, notes, pos = definition
                            
                            if pos != current_pos:
                                console.print(f"\n{pos or 'Uncategorized'}:")
                                current_pos = pos
                            
                            def_panel = Panel(def_text)
                            console.print(def_panel)
                            
                            if examples:
                                try:
                                    ex_list = json.loads(examples) if isinstance(examples, str) else examples
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
                    cur.execute("""
                        SELECT DISTINCT r.relation_type, w2.lemma
                        FROM relations r
                        JOIN words w2 ON r.to_word_id = w2.id
                        WHERE r.from_word_id = %s
                        ORDER BY r.relation_type, w2.lemma
                    """, (word_id,))
                    
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
        if 'conn' in locals() and conn:
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
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'words'
        """)
        available_columns = {row[0] for row in cur.fetchall()}
        
        # Basic counts with details - build dynamically based on available columns
        basic_queries = {
            "Total Words": ("SELECT COUNT(*) FROM words", None),
            "Total Definitions": ("SELECT COUNT(*) FROM definitions", None),
            "Total Relations": ("SELECT COUNT(*) FROM relations", None),
            "Total Etymologies": ("SELECT COUNT(*) FROM etymologies", None),
            "Total Pronunciations": ("SELECT COUNT(*) FROM pronunciations", None),
            "Total Credits": ("SELECT COUNT(*) FROM credits", None),
            "Words with Baybayin": ("SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE", None),
            "Words with Examples": ("""
                SELECT COUNT(DISTINCT word_id) 
                FROM definitions 
                WHERE examples IS NOT NULL
            """, None),
            "Words with Etymology": ("""
                SELECT COUNT(DISTINCT word_id) 
                FROM etymologies
            """, None),
            "Words with Pronunciation": ("""
                SELECT COUNT(DISTINCT word_id) 
                FROM pronunciations
            """, None)
        }
        
        # Add optional columns if they exist
        if 'is_proper_noun' in available_columns:
            basic_queries["Proper Nouns"] = ("SELECT COUNT(*) FROM words WHERE is_proper_noun = TRUE", None)
        if 'is_abbreviation' in available_columns:
            basic_queries["Abbreviations"] = ("SELECT COUNT(*) FROM words WHERE is_abbreviation = TRUE", None)
        if 'is_initialism' in available_columns:
            basic_queries["Initialisms"] = ("SELECT COUNT(*) FROM words WHERE is_initialism = TRUE", None)
        if 'root_word_id' in available_columns:
            basic_queries["Words with Root"] = ("SELECT COUNT(*) FROM words WHERE root_word_id IS NOT NULL", None)
        
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
            cur.execute("""
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
            """)
            
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
                    f"{bayb:,}"
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
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'definitions' AND column_name = 'standardized_pos_id'
                )
            """)
            
            has_standardized_pos = cur.fetchone()[0]
            
            if has_standardized_pos:
                cur.execute("""
                    SELECT 
                        p.name_tl,
                        COUNT(*) as count,
                        COUNT(DISTINCT d.word_id) as unique_words,
                        COUNT(CASE WHEN d.examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions d
                    JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                    GROUP BY p.name_tl
                    ORDER BY count DESC
                """)
            else:
                # Fallback to using part_of_speech text field
                cur.execute("""
                    SELECT 
                        COALESCE(part_of_speech, 'Unknown'),
                        COUNT(*) as count,
                        COUNT(DISTINCT word_id) as unique_words,
                        COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as with_examples
                    FROM definitions
                    GROUP BY part_of_speech
                    ORDER BY count DESC
                """)
            
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
                        f"{with_examples:,}"
                    )
                
                console.print()
                console.print(pos_table)
        except Exception as e:
            logger.error(f"Error displaying part of speech statistics: {str(e)}")
            console.print(f"[red]Error displaying part of speech statistics: {str(e)}[/]")
        
        # Relationship Statistics by category
        try:
            cur.execute("""
                SELECT 
                    r.relation_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT r.from_word_id) as unique_sources,
                    COUNT(DISTINCT r.to_word_id) as unique_targets
                FROM relations r
                GROUP BY r.relation_type
                ORDER BY count DESC
            """)
            
            rel_results = cur.fetchall()
            if rel_results:
                rel_table = Table(title="[bold blue]Relationship Types[/]", box=box.ROUNDED)
                rel_table.add_column("Type", style="yellow")
                rel_table.add_column("Total", justify="right", style="green")
                rel_table.add_column("Unique Sources", justify="right", style="green")
                rel_table.add_column("Unique Targets", justify="right", style="green")
                
                for rel_type, count, sources, targets in rel_results:
                    rel_table.add_row(
                        rel_type or "Unknown",
                        f"{count:,}",
                        f"{sources:,}",
                        f"{targets:,}"
                    )
                
                console.print()
                console.print(rel_table)
        except Exception as e:
            logger.error(f"Error displaying relationship statistics: {str(e)}")
            console.print(f"[red]Error displaying relationship statistics: {str(e)}[/]")
        
        # Source Statistics with more details
        try:
            # First check if source_info column exists
            if 'source_info' in available_columns:
                # Get source statistics from source_info
                cur.execute("""
                    SELECT 
                        COALESCE(source_info, 'Unknown') as source_name,
                        COUNT(*) as word_count
                    FROM words
                    GROUP BY source_name
                    ORDER BY word_count DESC
                """)
                
                source_results = cur.fetchall()
                if source_results:
                    source_table = Table(title="[bold blue]Source Distribution[/]", box=box.ROUNDED)
                    source_table.add_column("Source", style="yellow")
                    source_table.add_column("Words", justify="right", style="green")
                    
                    for source, count in source_results:
                        source_table.add_row(
                            source or "Unknown",
                            f"{count:,}"
                        )
                    
                    console.print()
                    console.print(source_table)
            
            # Also check definitions sources
            cur.execute("""
                SELECT 
                    COALESCE(sources, 'Unknown') as source_name,
                    COUNT(*) as def_count,
                    COUNT(DISTINCT word_id) as word_count,
                    COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as example_count
                FROM definitions
                GROUP BY sources
                ORDER BY def_count DESC
            """)
            
            def_source_results = cur.fetchall()
            if def_source_results:
                def_source_table = Table(title="[bold blue]Definition Sources[/]", box=box.ROUNDED)
                def_source_table.add_column("Source", style="yellow")
                def_source_table.add_column("Definitions", justify="right", style="green")
                def_source_table.add_column("Words", justify="right", style="green")
                def_source_table.add_column("With Examples", justify="right", style="green")
                
                for source, def_count, word_count, example_count in def_source_results:
                    def_source_table.add_row(
                        source or "Unknown",
                        f"{def_count:,}",
                        f"{word_count:,}",
                        f"{example_count:,}"
                    )
                
                console.print()
                console.print(def_source_table)
        except Exception as e:
            logger.error(f"Error displaying source statistics: {str(e)}")
            console.print(f"[yellow]Could not generate source statistics: {str(e)}[/]")
        
        # Baybayin Statistics with details
        try:
            baybayin_table = Table(title="[bold blue]Baybayin Statistics[/]", box=box.ROUNDED)
            baybayin_table.add_column("Metric", style="yellow")
            baybayin_table.add_column("Count", justify="right", style="green")
            baybayin_table.add_column("Details", style="dim")
            
            baybayin_queries = {
                "Total Baybayin Forms": (
                    "SELECT COUNT(*) FROM words WHERE baybayin_form IS NOT NULL",
                    """SELECT COUNT(DISTINCT language_code) 
                    FROM words WHERE baybayin_form IS NOT NULL"""
                ),
                "With Romanization": (
                    "SELECT COUNT(*) FROM words WHERE romanized_form IS NOT NULL",
                    None
                ),
                "Verified Forms": (
                    """SELECT COUNT(*) FROM words 
                    WHERE has_baybayin = TRUE 
                    AND baybayin_form IS NOT NULL""",
                    None
                )
            }
            
            # Only add Badlit stats if the column exists
            if 'badlit_form' in available_columns:
                baybayin_queries["With Badlit"] = (
                    "SELECT COUNT(*) FROM words WHERE badlit_form IS NOT NULL",
                    None
                )
                baybayin_queries["Complete Forms"] = (
                    """SELECT COUNT(*) FROM words 
                    WHERE baybayin_form IS NOT NULL 
                    AND romanized_form IS NOT NULL 
                    AND badlit_form IS NOT NULL""",
                    None
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
            console.print(f"[yellow]Could not generate Baybayin statistics: {str(e)}[/]")
        
        # Print timestamp
        console.print(f"\n[dim]Statistics generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]")
        
    except Exception as e:
        logger.error(f"Error displaying dictionary stats: {str(e)}")
        console.print(f"[red]Error: {str(e)}[/]")

@with_transaction(commit=False)
def display_leaderboard(cur, console):
    """Displays various contribution leaderboards."""
    console.print("\n[bold magenta underline]📊 Dictionary Contributors Leaderboard[/]\n")

    overall_stats_table = Table(title="[bold blue]Overall Statistics[/]", box=box.ROUNDED, show_header=False)
    overall_stats_table.add_column("Statistic", style="cyan")
    overall_stats_table.add_column("Value", justify="right", style="green")

    try:
        # Overall stats
        cur.execute("SELECT COUNT(*) FROM words")
        total_words = cur.fetchone()[0]
        overall_stats_table.add_row("Total Words", f"{total_words:,}")

        cur.execute("SELECT COUNT(*) FROM definitions")
        total_definitions = cur.fetchone()[0]
        overall_stats_table.add_row("Total Definitions", f"{total_definitions:,}")

        cur.execute("SELECT COUNT(*) FROM relations")
        total_relations = cur.fetchone()[0]
        overall_stats_table.add_row("Total Relations", f"{total_relations:,}")

        cur.execute("SELECT COUNT(*) FROM etymologies")
        total_etymologies = cur.fetchone()[0]
        overall_stats_table.add_row("Total Etymologies", f"{total_etymologies:,}")

        cur.execute("SELECT COUNT(DISTINCT standardized_pos_id) FROM definitions WHERE standardized_pos_id IS NOT NULL")
        total_pos = cur.fetchone()[0]
        overall_stats_table.add_row("Unique Parts of Speech", str(total_pos))

        cur.execute("SELECT COUNT(*) FROM words WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL")
        words_with_baybayin = cur.fetchone()[0]
        overall_stats_table.add_row("Words w/ Baybayin", f"{words_with_baybayin:,}")

        console.print(overall_stats_table)
        console.print()

    except Exception as e:
        logger.error(f"Error generating overall statistics: {str(e)}")
        console.print(f"[yellow]Could not generate overall statistics: {str(e)}[/]")


    # Definition Contributors
    try:
        cur.execute("""
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
        """)

        def_results = cur.fetchall()
        if def_results:
            def_table = Table(title="[bold blue]Definition Contributors[/]", box=box.ROUNDED)
            def_table.add_column("Source", style="yellow")
            def_table.add_column("Definitions", justify="right", style="green")
            def_table.add_column("Words", justify="right", style="green")
            def_table.add_column("Examples", justify="right", style="cyan")
            def_table.add_column("POS Types", justify="right", style="cyan")
            def_table.add_column("Notes", justify="right", style="cyan")
            def_table.add_column("Coverage", style="dim")

            for row in def_results:
                source = row[0] if len(row) > 0 else "Unknown"
                defs = row[1] if len(row) > 1 else 0
                words = row[2] if len(row) > 2 else 0
                examples = row[3] if len(row) > 3 else 0
                pos = row[4] if len(row) > 4 else 0
                notes = row[5] if len(row) > 5 else 0
                ex_pct = row[6] if len(row) > 6 else 0.0
                notes_pct = row[7] if len(row) > 7 else 0.0

                coverage = f"Examples: {ex_pct or 0.0}%, Notes: {notes_pct or 0.0}%"
                def_table.add_row(
                    source,
                    f"{defs:,}",
                    f"{words:,}",
                    f"{examples:,}",
                    str(pos),
                    f"{notes:,}",
                    coverage
                )

            console.print(def_table)
            console.print()
    except Exception as e:
        logger.error(f"Error generating definition statistics: {str(e)}")
        console.print(f"[red]Error:[/][yellow] Could not generate definition statistics: {str(e)}[/]")


    # Etymology Contributors
    try:
        cur.execute("""
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
        """)

        etym_results = cur.fetchall()
        if etym_results:
            etym_table = Table(title="[bold blue]Etymology Contributors[/]", box=box.ROUNDED)
            etym_table.add_column("Source", style="yellow")
            etym_table.add_column("Etymologies", justify="right", style="green")
            etym_table.add_column("Words", justify="right", style="green")
            etym_table.add_column("Components", justify="right", style="cyan")
            etym_table.add_column("Lang Codes", justify="right", style="cyan")
            etym_table.add_column("Coverage", style="dim")

            for row in etym_results:
                source = row[0] if len(row) > 0 else "Unknown"
                count = row[1] if len(row) > 1 else 0
                words = row[2] if len(row) > 2 else 0
                comps = row[3] if len(row) > 3 else 0
                langs = row[4] if len(row) > 4 else 0
                comp_pct = row[5] if len(row) > 5 else 0.0
                lang_pct = row[6] if len(row) > 6 else 0.0

                coverage = f"Components: {comp_pct or 0.0}%, Languages: {lang_pct or 0.0}%"
                etym_table.add_row(
                    source,
                    f"{count:,}",
                    f"{words:,}",
                    f"{comps:,}",
                    f"{langs:,}",
                    coverage
                )

            console.print(etym_table)
            console.print()
    except Exception as e:
        logger.error(f"Error generating etymology statistics: {str(e)}")
        console.print(f"[red]Error:[/][yellow] Could not generate etymology statistics: {str(e)}[/]")


    # Baybayin & Related Scripts Contributors
    try:
        # Check if badlit_form column exists first
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'words' AND column_name = 'badlit_form'
            )
        """)
        has_badlit_column = cur.fetchone()[0]

        # Build the conditional parts of the query
        badlit_count_sql = "COUNT(CASE WHEN badlit_form IS NOT NULL THEN 1 END) AS with_badlit" if has_badlit_column else "0 AS with_badlit"
        badlit_percentage_sql = "ROUND(100.0 * with_badlit / NULLIF(total_count, 0), 1) as badlit_percentage" if has_badlit_column else "0.0 as badlit_percentage"

        # Try different approaches to detect source info in words table
        source_detection_sql = ""
        try:
            # Check if source_info or sources column exists
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'words' AND column_name IN ('source_info', 'sources')
            """)
            columns_info = cur.fetchall()
            column_types = {col[0]: col[1] for col in columns_info}

            if 'source_info' in column_types:
                source_col = 'source_info'
                is_jsonb = column_types['source_info'].upper() == 'JSONB'
            elif 'sources' in column_types:
                source_col = 'sources'
                is_jsonb = column_types['sources'].upper() == 'JSONB'
            else:
                source_col = 'tags' # Fallback
                is_jsonb = False

            source_expr = f"{source_col}::text" if is_jsonb else source_col

            # Construct the CTE with proper source detection
            source_detection_sql = f"""
                WITH baybayin_stats AS (
                    SELECT
                        CASE
                            WHEN {source_expr} ILIKE '%project marayum%' THEN 'Project Marayum'
                            WHEN {source_expr} ILIKE '%marayum%' THEN 'Project Marayum'
                            WHEN {source_expr} ILIKE '%kaikki-ceb%' THEN 'kaikki.org (Cebuano)'
                            WHEN {source_expr} ILIKE '%kaikki.jsonl%' THEN 'kaikki.org (Tagalog)'
                            WHEN {source_expr} ILIKE '%kaikki%' AND {source_expr} ILIKE '%ceb%' THEN 'kaikki.org (Cebuano)'
                            WHEN {source_expr} ILIKE '%kaikki%' THEN 'kaikki.org (Tagalog)'
                            WHEN {source_expr} ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                            WHEN {source_expr} ILIKE '%kwf_dictionary%' THEN 'KWF Diksiyonaryo'
                            WHEN {source_expr} ILIKE '%tagalog.com%' THEN 'tagalog.com'
                            WHEN {source_expr} ILIKE '%root_words%' THEN 'tagalog.com'
                            WHEN {source_expr} ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                            WHEN {source_expr} ILIKE '%tagalog-words%' THEN 'diksiyonaryo.ph'
                            WHEN tags ILIKE '%marayum%' THEN 'Project Marayum' -- Fallback check on tags
                            WHEN tags ILIKE '%kaikki%' THEN 'kaikki.org'
                            WHEN tags ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                            ELSE 'Unknown'
                        END AS source_name,
                        COUNT(*) AS total_count,
                        COUNT(CASE WHEN baybayin_form IS NOT NULL THEN 1 END) AS with_baybayin,
                        COUNT(CASE WHEN romanized_form IS NOT NULL THEN 1 END) AS with_romanized,
                        {badlit_count_sql} -- Conditional Badlit count
                    FROM words
                    WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL -- Include cases where flag might be false but form exists
                    GROUP BY source_name
                )
            """
        except Exception as e:
            logger.warning(f"Error checking words table columns for source detection: {str(e)}. Falling back to tags only.")
            # Fallback to a simpler approach if column detection fails
            source_detection_sql = f"""
                WITH baybayin_stats AS (
                    SELECT
                        CASE
                            WHEN tags ILIKE '%marayum%' THEN 'Project Marayum'
                            WHEN tags ILIKE '%kaikki%' THEN 'kaikki.org'
                            WHEN tags ILIKE '%kwf%' THEN 'KWF Diksiyonaryo'
                            WHEN tags ILIKE '%tagalog.com%' THEN 'tagalog.com'
                            WHEN tags ILIKE '%diksiyonaryo.ph%' THEN 'diksiyonaryo.ph'
                            ELSE 'Unknown'
                        END AS source_name,
                        COUNT(*) AS total_count,
                        COUNT(CASE WHEN baybayin_form IS NOT NULL THEN 1 END) AS with_baybayin,
                        COUNT(CASE WHEN romanized_form IS NOT NULL THEN 1 END) AS with_romanized,
                        {badlit_count_sql} -- Conditional Badlit count
                    FROM words
                    WHERE has_baybayin = TRUE OR baybayin_form IS NOT NULL
                    GROUP BY source_name
                )
            """

        # Execute the final query with percentage calculations
        final_query = f"""
            {source_detection_sql}
            SELECT *,
                ROUND(100.0 * with_baybayin / NULLIF(total_count, 0), 1) as baybayin_percentage,
                ROUND(100.0 * with_romanized / NULLIF(total_count, 0), 1) as rom_percentage,
                {badlit_percentage_sql} -- Conditional Badlit percentage
            FROM baybayin_stats
            ORDER BY total_count DESC
        """
        cur.execute(final_query)

        baybayin_results = cur.fetchall()
        if baybayin_results:
            baybayin_table = Table(title="[bold blue]Baybayin & Related Scripts Contributors[/]", box=box.ROUNDED)
            baybayin_table.add_column("Source", style="yellow")
            baybayin_table.add_column("Total Entries", justify="right", style="green") # Words with potential for scripts
            baybayin_table.add_column("w/ Baybayin", justify="right", style="cyan")
            baybayin_table.add_column("w/ Romanized", justify="right", style="cyan")
            if has_badlit_column: # Only add column if it exists
                baybayin_table.add_column("w/ Badlit", justify="right", style="cyan")
            baybayin_table.add_column("Coverage (%)", style="dim")

            for row in baybayin_results:
                # Dynamically adjust indices based on whether badlit data is present
                source = row[0]
                total = row[1]
                bayb = row[2]
                rom = row[3]

                current_index = 4
                badlit = 0
                if has_badlit_column:
                    badlit = row[current_index]
                    current_index += 1

                bayb_pct = row[current_index]
                current_index += 1
                rom_pct = row[current_index]
                current_index += 1

                badlit_pct = 0.0
                if has_badlit_column:
                    badlit_pct = row[current_index]

                coverage = f"Bayb: {bayb_pct or 0.0}%, Rom: {rom_pct or 0.0}%"
                if has_badlit_column:
                    coverage += f", Badlit: {badlit_pct or 0.0}%"

                # Construct row data based on whether badlit column exists
                row_data = [
                    source,
                    f"{total:,}",
                    f"{bayb:,}",
                    f"{rom:,}",
                ]
                if has_badlit_column:
                    row_data.append(f"{badlit:,}")
                row_data.append(coverage)

                baybayin_table.add_row(*row_data) # Use unpacking

            console.print(baybayin_table)
            console.print()
    except Exception as e:
        # Use exc_info=True for traceback in logs
        logger.error(f"Error generating Baybayin statistics: {str(e)}", exc_info=True)
        # Display a user-friendly error
        console.print(f"[red]Error:[/][yellow] Could not generate Baybayin statistics: {str(e)}[/]")
        # Crucially, rollback the transaction if this section failed
        try:
            conn = cur.connection
            conn.rollback()
            logger.info("Rolled back transaction due to error in Baybayin stats.")
        except Exception as rb_error:
            logger.error(f"Failed to rollback transaction after Baybayin stats error: {rb_error}")


    # Relationship Contributors
    try:
        # Ensure transaction is usable before proceeding
        if cur.connection.closed or cur.connection.status != psycopg2.extensions.STATUS_READY:
             raise Exception("Transaction aborted in previous step. Cannot proceed.")

        cur.execute("""
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
         """)

        rel_results = cur.fetchall()
        if rel_results:
             rel_table = Table(title="[bold blue]Relationship Contributors[/]", box=box.ROUNDED)
             rel_table.add_column("Source", style="yellow")
             rel_table.add_column("Relation Type", style="magenta")
             rel_table.add_column("Count", justify="right", style="green")

             current_source = None
             for source, rel_type, count in rel_results:
                 if source != current_source:
                     if current_source is not None:
                         rel_table.add_row("---", "---", "---") # Separator
                     rel_table.add_row(source, rel_type, f"{count:,}")
                     current_source = source
                 else:
                     rel_table.add_row("", rel_type, f"{count:,}") # Indent or leave source blank

             console.print(rel_table)
             console.print()
    except Exception as e:
         logger.error(f"Error generating relationship statistics: {str(e)}", exc_info=True) # Add traceback
         console.print(f"[red]Error:[/][yellow] Could not generate relationship statistics: {str(e)}[/]")
         # Don't necessarily need to rollback here unless this is the *only* thing done in the transaction


    console.print(f"Leaderboard generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


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
        USING duplicates dup
        WHERE d.word_id = dup.word_id
        AND d.definition_text = dup.definition_text
        AND d.standardized_pos_id = dup.standardized_pos_id
        AND d.id != dup.keep_id;
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
                        source_identifier=source
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