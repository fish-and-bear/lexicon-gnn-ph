#!/usr/bin/env python3
"""
Db helpers for dictionary manager
"""
import json
import logging
import os
import psycopg2
import psycopg2.pool
import psycopg2.extras
from psycopg2.extras import Json, execute_values
from typing import (
    Optional,
    Tuple,
    Dict,
    Union,
    List,
    Any,
)
from contextlib import contextmanager
import functools
import time
import hashlib
from datetime import datetime
import re # Add re import
from tqdm import tqdm # Import tqdm for progress bars
import psycopg2.errorcodes
import json # Also ensure this is imported for the next fix
# Try to import url maker for DATABASE_URL parsing
try:
    from sqlalchemy.engine.url import make_url
except ImportError:
    make_url = None
    logging.warning("SQLAlchemy not found, DATABASE_URL parsing may be limited.")

# Absolute imports for enums and text helpers
from backend.dictionary_manager.enums import RelationshipType
from backend.dictionary_manager.text_helpers import (
    SourceStandardization,
    standardize_source_identifier,
    normalize_lemma,
    get_standard_code,
    remove_trailing_numbers,
    extract_parenthesized_text,
    BaybayinRomanizer,
    process_baybayin_text,
    clean_html,
    clean_baybayin_text, # <-- Import the correct helper
    get_language_code,
)

logger = logging.getLogger(__name__)

# Moved from dictionary_manager.py
# Define default language code
DEFAULT_LANGUAGE_CODE = "tl"
# VALID_BAYBAYIN_REGEX = re.compile(r'^[\u1700-\u171F\u1735\u1736\s]+$') # Original with +
VALID_BAYBAYIN_REGEX = re.compile(r'^[\u1700-\u171F\u1735\u1736\s]*$') # Use * to match SQL constraint
# Generate the negative regex pattern from the valid one for easier use in SQL queries
INVALID_BAYBAYIN_CHARS_SQL_PATTERN = r'[^\\u1700-\\u171F\\u1735\\u1736\\s]'

# Custom exceptions
class DatabaseError(Exception):
    """Base exception for database-related errors"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Exception raised for database connection errors"""
    pass

# Database connection pool configuration (copied from dictionary_manager)
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "fil_dict_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}

# Initialize connection pool (copied from dictionary_manager)
try:
    connection_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1, maxconn=10, **DB_CONFIG
    )
    logger.info("Database connection pool initialized for db_helpers")
except Exception as e:
    logger.error(f"Failed to initialize connection pool in db_helpers: {e}")
    connection_pool = None


# --- Core Connection Functions --- (Copied from dictionary_manager.py)
def get_connection():
    """Establish a database connection using env vars or DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    conn = None
    if not database_url:
        logger.debug("DATABASE_URL not set, using individual DB_* variables.")
        try:
             conn = psycopg2.connect(
                 dbname=os.getenv("DB_NAME", DB_CONFIG['dbname']),
                 user=os.getenv("DB_USER", DB_CONFIG['user']),
                 password=os.getenv("DB_PASSWORD", DB_CONFIG['password']),
                 host=os.getenv("DB_HOST", DB_CONFIG['host']),
                 port=os.getenv("DB_PORT", DB_CONFIG['port']),
                 client_encoding='UTF8' # <<< Explicitly set encoding
             )
        except psycopg2.OperationalError as e:
             logger.error(f"Failed to connect using DB_* variables: {e}")
             raise DatabaseConnectionError(f"DB connect failed: {e}") from e
    else:
        if make_url is None:
            logger.error("DATABASE_URL provided, but SQLAlchemy not installed for parsing.")
            raise ImportError("SQLAlchemy is required to parse DATABASE_URL")
        try:
            logger.debug(f"Attempting connection using DATABASE_URL: {database_url[:20]}...")
            url = make_url(database_url)
            conn_args = url.translate_connect_args(database_driver="psycopg2")
            if "username" in conn_args:
                conn_args["user"] = conn_args.pop("username")
            conn_args['client_encoding'] = 'UTF8' # <<< Explicitly set encoding
            conn = psycopg2.connect(**conn_args)
            logger.debug("Connection successful using DATABASE_URL.")
        except Exception as e:
            logger.error(f"Failed to connect using DATABASE_URL ({database_url[:20]}...): {e}", exc_info=True)
            raise DatabaseConnectionError(f"DATABASE_URL connect failed: {e}") from e

    conn.autocommit = False # Default to False for manual transaction control
    return conn

def get_db_connection(max_retries=3, retry_delay=1.0):
    """Get a connection from the pool with retry logic."""
    if not connection_pool:
        raise DatabaseConnectionError("Connection pool is not initialized.")

    last_exception = None
    for attempt in range(max_retries):
        try:
            conn = connection_pool.getconn()
            if conn:
                logger.debug(f"Connection acquired (attempt {attempt + 1}/{max_retries})")
                if conn.closed:
                    logger.warning("Pool returned closed connection. Discarding & retrying...")
                    connection_pool.putconn(conn, close=True)
                    last_exception = DatabaseConnectionError("Pool returned closed connection")
                    continue
                conn.autocommit = False # Ensure default is False
                return conn
            else:
                last_exception = DatabaseConnectionError("Connection pool exhausted")
        except psycopg2.OperationalError as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            last_exception = e
            time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logger.error(f"Unexpected error getting db connection: {e}", exc_info=True)
            last_exception = DatabaseConnectionError(f"Unexpected error: {e}")
            break

    logger.error(f"Failed to get database connection after {max_retries} attempts.")
    if isinstance(last_exception, DatabaseError):
        raise last_exception
    elif last_exception:
        raise DatabaseConnectionError(f"Connection failed: {last_exception}") from last_exception
    else:
        raise DatabaseConnectionError("Unknown error acquiring connection")

def release_db_connection(conn):
    """Return a connection to the pool."""
    if connection_pool and conn:
        try:
            connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
            try:
                conn.close()
            except Exception as close_err:
                logger.error(f"Failed to close connection after putconn error: {close_err}")

# Moved from dictionary_manager.py
class DBConnection:
    """Context manager for database connections using the pool."""
    def __init__(self, autocommit=False):
        self.autocommit = autocommit
        self.conn = None
        self.cursor = None

    def __enter__(self):
        try:
            self.conn = get_db_connection()
            if self.autocommit:
                self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            return self.cursor
        except Exception as e:
            logger.error(f"DBConnection __enter__ failed: {e}", exc_info=True)
            if self.conn:
                release_db_connection(self.conn)
                self.conn = None
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: # Only proceed if connection was successful
            try:
                if exc_type is not None:
                    if not self.autocommit:
                        try:
                            self.conn.rollback()
                            logger.debug("Transaction rolled back due to exception in DBConnection context.")
                        except Exception as rb_err:
                            logger.error(f"Rollback failed in DBConnection __exit__: {rb_err}")
                else:
                    if not self.autocommit:
                        try:
                            self.conn.commit()
                        except Exception as commit_err:
                            logger.error(f"Commit failed in DBConnection __exit__: {commit_err}")
                            try: self.conn.rollback()
                            except: pass
                            raise DatabaseError("Commit failed") from commit_err
            finally:
                if self.cursor:
                    try:
                        self.cursor.close()
                    except Exception:
                        pass
                release_db_connection(self.conn)
                self.conn = None

# Moved from dictionary_manager.py
def with_transaction(commit=True):
    """
    Decorator for database operations that need to run within a transaction.
    Handles nested calls correctly.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cur, *args, **kwargs):
            conn = cur.connection
            original_autocommit = None
            started_transaction = False
            
            # Check if the connection is in a failed transaction state
            if hasattr(conn, 'info') and conn.info and conn.info.transaction_status == psycopg2.extensions.TRANSACTION_STATUS_INERROR:
                logger.warning(f"Connection is in aborted transaction state before executing {func.__name__}. Attempting rollback.")
                try:
                    conn.rollback()
                    logger.info(f"Successfully rolled back aborted transaction before {func.__name__}")
                except Exception as rb_error:
                    logger.error(f"Failed to rollback aborted transaction before {func.__name__}: {rb_error}")
                    raise DatabaseError(f"Cannot proceed with {func.__name__} - transaction aborted and rollback failed") from rb_error
            
            try:
                if conn.autocommit:
                    original_autocommit = True
                    conn.autocommit = False
                    started_transaction = True
                    logger.debug(f"Started new transaction for {func.__name__}")

                result = func(cur, *args, **kwargs)

                if commit and started_transaction:
                    try:
                        conn.commit()
                        logger.debug(f"Transaction committed for {func.__name__}")
                    except Exception as commit_error:
                        logger.error(f"Failed to commit transaction for {func.__name__}: {commit_error}")
                        try: 
                            conn.rollback()
                            logger.info(f"Rolled back transaction after commit failure for {func.__name__}")
                        except Exception as rb_error: 
                            logger.error(f"Failed to rollback after commit error for {func.__name__}: {rb_error}")
                        raise
                return result
            except Exception as e:
                # Check if this is a transaction error
                if isinstance(e, psycopg2.errors.InFailedSqlTransaction):
                    logger.error(f"Transaction already aborted before {func.__name__} execution completed")
                elif started_transaction:
                    logger.error(f"Error in {func.__name__}, rolling back transaction: {e}", exc_info=True)
                    try: 
                        conn.rollback()
                        logger.debug(f"Transaction rolled back for {func.__name__}")
                    except Exception as rb_error:
                        logger.error(f"Failed to rollback transaction for {func.__name__}: {rb_error}")
                        # We still want to raise the original error
                raise
            finally:
                if started_transaction and original_autocommit is not None:
                    try: 
                        conn.autocommit = original_autocommit
                        logger.debug(f"Restored autocommit={original_autocommit} after {func.__name__}")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore autocommit state after {func.__name__}: {restore_error}")
        return wrapper
    return decorator

# Moved from dictionary_manager.py
def get_cursor():
    """Return a cursor from a new connection with proper error handling."""
    conn = None
    try:
        conn = get_connection()
        return conn.cursor() # Use default cursor factory unless DictCursor needed
    except Exception as e:
        if conn:
            release_db_connection(conn) # Release original non-pooled connection
        logger.error(f"Error getting cursor: {e}", exc_info=True)
        raise DatabaseConnectionError("Failed to get cursor") from e


# Moved from dictionary_manager.py
# Needs a connection passed in, likely managed by the calling context.
def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    logger.info("Setting up PostgreSQL extensions...")
    if conn.closed:
        logger.error("Cannot setup extensions: connection is closed.")
        return
    with conn.cursor() as cur:
        try:
            extensions = ["pg_trgm", "unaccent", "fuzzystrmatch", "dict_xsyn"]
            for ext in extensions:
                cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = %s", (ext,))
                if cur.fetchone()[0] == 0:
                    logger.info(f"Installing extension: {ext}")
                    # Use autocommit for CREATE EXTENSION
                    original_autocommit = conn.autocommit
                    conn.autocommit = True
                    try:
                         cur.execute(f"CREATE EXTENSION IF NOT EXISTS \"{ext}\"")
                    finally:
                         conn.autocommit = original_autocommit
                else:
                    logger.debug(f"Extension {ext} already installed")
            # No final commit needed if extensions were done with autocommit=True
        except Exception as e:
            logger.error(f"Error setting up extensions: {str(e)}", exc_info=True)
            # Rollback might not be needed if autocommit was used, but doesn't hurt
            try: conn.rollback()
            except Exception: pass
            raise DatabaseError("Extension setup failed") from e

@with_transaction # Apply decorator (commit=True by default)
def add_linguistic_note(
    cur, word_id: int, note_type: str, note_value: str, note_source: str
):
    """
    Appends a structured note to the word_metadata['linguistic_notes'] JSONB array.
    Handles NULL metadata or missing 'linguistic_notes' array gracefully.
    Assumes it is called within an existing transaction.

    Args:
        cur: Active database cursor.
        word_id: The ID of the word to update.
        note_type: The type/category of the note (e.g., "Language Code").
        note_value: The actual value of the note (e.g., "en", "archaic").
        note_source: Where this note was found (e.g., "etymology_cog").
    """
    if not all([word_id, note_type, note_value, note_source]):
        logger.warning(
            f"Skipping add_linguistic_note for word ID {word_id}: Missing required arguments."
        )
        return

    try:
        # Ensure all parts are strings before creating JSON
        note_object_json = json.dumps(
            {
                "type": str(note_type).strip(),
                "value": str(note_value).strip(),
                "source": str(note_source).strip(),
            }
        )

        # Use jsonb_set with create_missing=True for robustness
        sql = """
            UPDATE words
            SET word_metadata = jsonb_set(
                COALESCE(word_metadata, '{}'::jsonb),
                '{linguistic_notes}', -- Path to the array
                COALESCE(word_metadata->'linguistic_notes', '[]'::jsonb) || %s::jsonb, -- Append new note
                true -- create_missing = true: Creates the array if 'linguistic_notes' path doesn't exist
            )
            WHERE id = %s;
        """
        cur.execute(sql, (note_object_json, word_id))
        # Logging successful update is handled by the calling function context if needed

    except psycopg2.Error as db_err:
        logger.error(
            f"Database error adding linguistic note for word ID {word_id}: {db_err.pgcode} {db_err.pgerror}",
            exc_info=True,
        )
        # Reraise to allow the calling transaction to handle rollback
        raise db_err
    except Exception as e:
        logger.error(
            f"Unexpected error adding linguistic note for word ID {word_id}: {e}",
            exc_info=True,
        )
        # Reraise to allow the calling transaction to handle rollback
        raise e

# Moved from dictionary_manager.py
# Note: This function does NOT use the transaction decorator directly.
def update_word_source_info(
    current_source_info: Optional[Union[str, dict]],
    new_source_identifier: Optional[str],
) -> str:
    """
    Updates the source_info JSON for a word entry.
    """
    SOURCE_INFO_FILES_KEY = "files"
    if not new_source_identifier:
        if isinstance(current_source_info, dict): return json.dumps(current_source_info)
        elif isinstance(current_source_info, str): return current_source_info
        else: return "{}"

    source_info_dict = {}
    if isinstance(current_source_info, dict):
        source_info_dict = current_source_info.copy()
    elif isinstance(current_source_info, str) and current_source_info:
        try:
            loaded = json.loads(current_source_info)
            if isinstance(loaded, dict): source_info_dict = loaded
        except (json.JSONDecodeError, TypeError): source_info_dict = {}

    if SOURCE_INFO_FILES_KEY not in source_info_dict or not isinstance(source_info_dict[SOURCE_INFO_FILES_KEY], list):
        source_info_dict[SOURCE_INFO_FILES_KEY] = []

    standardized_source = SourceStandardization.standardize_sources(new_source_identifier)

    if standardized_source not in source_info_dict[SOURCE_INFO_FILES_KEY]:
        source_info_dict[SOURCE_INFO_FILES_KEY].append(standardized_source)
        source_info_dict[SOURCE_INFO_FILES_KEY].sort()

    if "last_updated" not in source_info_dict or not isinstance(source_info_dict["last_updated"], dict):
        source_info_dict["last_updated"] = {}

    source_info_dict["last_updated"][standardized_source] = datetime.now().isoformat()
    return json.dumps(source_info_dict)

# Moved from dictionary_manager.py
@with_transaction(commit=True)
def fix_inconsistent_sources(cur) -> Dict:
    """
    Fix inconsistent source information across the database.
    """
    stats = {
        "words_updated": 0,
        "definitions_updated": 0,
        "relations_updated": 0,
        "etymologies_updated": 0,
    }
    try:
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
            -- Added condition to only update if source_info is empty or null
            -- If it already has data, update_word_source_info should be used during inserts.
        """
        )
        stats["words_updated"] = cur.rowcount
        logger.info(f"Standardized source_info for {stats['words_updated']} words.")
        # Add similar logic for other tables if their sources need standardizing
        return stats
    except Exception as e:
        logger.error(f"Error fixing inconsistent sources: {e}", exc_info=True)
        raise

# Moved from dictionary_manager.py
# Reads data, doesn't need transaction decorator
def get_word_sources(cur, word_id: int) -> Dict:
    """
    Get comprehensive source information for a word.
    """
    result = {
        "word": None, "direct_sources": [], "definition_sources": [],
        "relation_sources": [], "etymology_sources": [],
    }
    try:
        cur.execute("SELECT lemma, source_info FROM words WHERE id = %s", (word_id,))
        word_row = cur.fetchone()
        if not word_row: return result

        result["word"] = word_row[0] # Assumes default cursor
        source_info_raw = word_row[1]
        if source_info_raw:
            try:
                source_info = source_info_raw
                if isinstance(source_info_raw, str):
                    source_info = json.loads(source_info_raw)
                if isinstance(source_info, dict) and "files" in source_info and isinstance(source_info["files"], list):
                    result["direct_sources"] = source_info["files"]
                elif isinstance(source_info, list): result["direct_sources"] = source_info
                elif isinstance(source_info, str): result["direct_sources"] = [source_info]
            except (json.JSONDecodeError, TypeError): pass

        # Get sources from related tables
        tables_info = [
            ("definitions", "definition_sources", f"WHERE word_id = {word_id}"), # Corrected WHERE clause
            ("etymologies", "etymology_sources", f"WHERE word_id = {word_id}"), # Corrected WHERE clause
            ("relations", "relation_sources", f"WHERE (from_word_id = {word_id} OR to_word_id = {word_id})")
        ]
        for table, key, where_clause in tables_info:
            # Check if 'sources' column exists
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s AND column_name = 'sources'", (table,))
            if cur.fetchone():
                # Construct the query carefully
                query = f"""
                    SELECT DISTINCT
                        CASE
                            WHEN jsonb_typeof(sources) = 'array' THEN jsonb_array_elements_text(sources)
                            WHEN jsonb_typeof(sources) = 'string' THEN sources::text
                            ELSE sources::text -- Fallback for older TEXT type
                        END as source
                    FROM {table}
                    {where_clause} AND sources IS NOT NULL
                """
                cur.execute(query) # Removed word_id param as it's already in where_clause
                result[key] = [row[0].strip() for row in cur.fetchall() if row[0] and row[0].strip()]

        # Deduplicate
        for key in ["direct_sources", "definition_sources", "relation_sources", "etymology_sources"]:
             result[key] = sorted(list(set(s for s in result[key] if s))) # Filter out empty strings
        return result
    except Exception as e:
        logger.error(f"Error getting sources for word ID {word_id}: {e}", exc_info=True)
        return result

# Moved from dictionary_manager.py
@with_transaction(commit=True)
def propagate_source_info(cur, word_id: int, source_identifier: str) -> bool:
    """
    Propagate source information to related records using JSONB operations.
    Assumes target 'sources' columns are JSONB arrays.
    """
    try:
        standardized_source = standardize_source_identifier(source_identifier)
        if not standardized_source or standardized_source == 'unknown':
             logger.warning(f"Skipping source propagation for word ID {word_id}: Invalid source '{source_identifier}'")
             return False
        source_json = Json([standardized_source])

        # Update definitions
        cur.execute("""
            UPDATE definitions SET sources = COALESCE(sources, '[]'::jsonb) || %s
            WHERE word_id = %s AND NOT (sources @> %s::jsonb);
        """, (source_json, word_id, source_json))
        # Update etymologies
        cur.execute("""
            UPDATE etymologies SET sources = COALESCE(sources, '[]'::jsonb) || %s
            WHERE word_id = %s AND NOT (sources @> %s::jsonb);
        """, (source_json, word_id, source_json))
        # Update relations
        cur.execute("""
            UPDATE relations SET sources = COALESCE(sources, '[]'::jsonb) || %s
            WHERE (from_word_id = %s OR to_word_id = %s) AND NOT (sources @> %s::jsonb);
        """, (source_json, word_id, word_id, source_json))
        # Update pronunciations (metadata)
        cur.execute("""
             UPDATE pronunciations SET pronunciation_metadata = jsonb_set(
                 COALESCE(pronunciation_metadata, '{}'::jsonb), '{sources}',
                 COALESCE(pronunciation_metadata->'sources', '[]'::jsonb) || %s::jsonb)
             WHERE word_id = %s AND NOT (pronunciation_metadata->'sources' @> %s::jsonb);
         """, (source_json, word_id, source_json))

        logger.info(f"Propagated source '{standardized_source}' for word ID {word_id}")
        return True
    except Exception as e:
        logger.error(f"Error propagating source info for word ID {word_id}, source '{source_identifier}': {e}", exc_info=True)
        raise

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
    -- Updated regex to match Python VALID_BAYBAYIN_REGEX (U+1700–U+171F, U+1735, U+1736, spaces)
    CONSTRAINT baybayin_form_regex CHECK (baybayin_form ~ '^[\\u1700-\\u171F\\u1735\\u1736\\s]*$' OR baybayin_form IS NULL)
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
    -- sources TEXT, -- Removed, stored in pronunciation_metadata
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
    examples TEXT, -- Consider moving to definition_examples table
    usage_notes TEXT,
    tags TEXT, -- Should probably be normalized
    sources TEXT, -- Should probably be normalized or JSONB
    definition_metadata JSONB DEFAULT '{}', -- Added missing metadata column
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

-- Create definition_links table
CREATE TABLE IF NOT EXISTS definition_links (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    link_text TEXT NOT NULL,
    tags TEXT, -- The required column
    link_metadata JSONB DEFAULT '{}'::jsonb, -- Correct column
    sources TEXT, -- Required column
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT definition_links_unique UNIQUE (definition_id, link_text) -- Correct constraint
);
CREATE INDEX IF NOT EXISTS idx_def_links_def ON definition_links(definition_id);
-- Add gin index for link_text if needed and extension exists
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        CREATE INDEX IF NOT EXISTS idx_def_links_type_trgm ON definition_links USING gin(link_text gin_trgm_ops);
    END IF;
END $$;
-- Trigger for updated_at
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_definition_links_timestamp') THEN
        NULL; -- Trigger exists, do nothing
    ELSE
        CREATE TRIGGER trigger_update_definition_links_timestamp
        BEFORE UPDATE ON definition_links
        FOR EACH ROW EXECUTE PROCEDURE update_timestamp();
    END IF;
END $$;

-- Create definition_relations table
CREATE TABLE IF NOT EXISTS definition_relations (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
    word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
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
   sources TEXT, -- Added based on analysis
   -- category_metadata JSONB DEFAULT '{}'::jsonb, -- Column definition moved to ALTER TABLE
   created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
   UNIQUE (definition_id, category_name)
);

-- Add category_metadata column if it doesn't exist (handles existing tables)
ALTER TABLE definition_categories ADD COLUMN IF NOT EXISTS category_metadata JSONB DEFAULT '{}'::jsonb;

-- Indices for definition_categories
CREATE INDEX IF NOT EXISTS idx_definition_categories_definition_id ON definition_categories(definition_id);

-- Create definition_examples table
CREATE TABLE IF NOT EXISTS definition_examples (
    id SERIAL PRIMARY KEY,
    definition_id INTEGER NOT NULL REFERENCES definitions(id) ON DELETE CASCADE,
    example_text TEXT NOT NULL,
    translation TEXT,
    example_type VARCHAR(50),
    reference TEXT,
    metadata JSONB,
    sources TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT definition_examples_unique UNIQUE (definition_id, example_text) -- Consider if example_text alone should be unique per definition
);

CREATE INDEX IF NOT EXISTS idx_def_examples_def ON definition_examples(definition_id);
-- Trigger for updated_at (use the generic update_timestamp function)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_definition_examples_timestamp') THEN
        CREATE TRIGGER trigger_update_definition_examples_timestamp
        BEFORE UPDATE ON definition_examples
        FOR EACH ROW EXECUTE PROCEDURE update_timestamp();
    END IF;
END $$;


-- Create word_templates table
CREATE TABLE IF NOT EXISTS word_templates (
    id SERIAL PRIMARY KEY,
    word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
    template_name VARCHAR(255) NOT NULL,
    args JSONB DEFAULT '{}'::jsonb,
    expansion TEXT,
    sources TEXT, -- Changed from source_identifier based on other tables
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT word_templates_unique UNIQUE (word_id, template_name)
);
CREATE INDEX IF NOT EXISTS idx_word_templates_word ON word_templates(word_id);
CREATE INDEX IF NOT EXISTS idx_word_templates_name ON word_templates(template_name);
-- Trigger for updated_at (copying pattern from other tables)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'trigger_update_word_templates_timestamp') THEN
        CREATE TRIGGER trigger_update_word_templates_timestamp
        BEFORE UPDATE ON word_templates
        FOR EACH ROW EXECUTE PROCEDURE update_timestamp();
    END IF;
END $$;

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
-- Drop potentially corrupted index before recreating
DROP INDEX IF EXISTS idx_word_forms_form;
CREATE INDEX IF NOT EXISTS idx_word_forms_word ON word_forms(word_id);
CREATE INDEX IF NOT EXISTS idx_word_forms_form ON word_forms(form);


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


def check_baybayin_consistency(cur):
    """Check for consistency issues in Baybayin data."""
    issues = []
    try:
        # Check for entries missing romanization
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

        # Check for inconsistent Baybayin flags
        cur.execute(
            """
            SELECT id, lemma
            FROM words
            WHERE (has_baybayin = TRUE AND baybayin_form IS NULL)
               OR (has_baybayin = FALSE AND baybayin_form IS NOT NULL)
            """
        )
        inconsistent = cur.fetchall()
        if inconsistent:
            issues.append(
                f"Found {len(inconsistent)} entries with inconsistent Baybayin flags"
            )
            for word_id, lemma in inconsistent:
                logger.warning(
                    f"Inconsistent Baybayin flags for word ID {word_id}: {lemma}"
                )

        # Check for invalid Baybayin characters using the SQL pattern derived from VALID_BAYBAYIN_REGEX
        # The regex checks if the string contains *any* character NOT in the allowed set.
        cur.execute(
            f"""
            SELECT id, lemma, baybayin_form
            FROM words
            WHERE baybayin_form IS NOT NULL AND baybayin_form ~ '{INVALID_BAYBAYIN_CHARS_SQL_PATTERN}'
            """
        )
        invalid_chars = cur.fetchall()
        if invalid_chars:
            issues.append(
                f"Found {len(invalid_chars)} entries with invalid Baybayin characters"
            )
            for word_id, lemma, baybayin in invalid_chars:
                logger.warning(f"Invalid Baybayin characters in word ID {word_id}: {lemma} ({baybayin})") # Added baybayin form to log

    except psycopg2.Error as db_err: # Be more specific with exception type
        logger.error(f"Database error during Baybayin consistency check: {db_err}", exc_info=True)
        issues.append(f"Database error during check: {db_err}")
    except Exception as e:
        logger.error(f"Unexpected error during Baybayin consistency check: {e}", exc_info=True)
        issues.append(f"Unexpected error during check: {e}")

    return issues

@with_transaction(commit=True)
def repair_database_issues(cur, issues=None): # issues parameter is optional now
    """
    Repair common database issues by cleaning up orphaned records, fixing inconsistencies,
    and removing duplicate definitions.
    """
    try:
        # Delete relations with missing words
        cur.execute(
            """
            DELETE FROM relations r
            WHERE NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.from_word_id)
            OR NOT EXISTS (SELECT 1 FROM words w WHERE w.id = r.to_word_id)
            """
        )
        logger.debug(f"Deleted {cur.rowcount} orphaned relations.")

        # Delete duplicate definitions (keeping the one with the lowest ID)
        cur.execute(
            """
            DELETE FROM definitions d
            WHERE d.id IN (
                SELECT id
                FROM (
                    SELECT
                        id,
                        ROW_NUMBER() OVER(PARTITION BY word_id, definition_text ORDER BY id ASC) as rn
                    FROM definitions
                ) t
                WHERE t.rn > 1
            );
            """
        )
        logger.debug(f"Deleted {cur.rowcount} duplicate definitions.")


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
        logger.debug(f"Updated {cur.rowcount} missing search_text values.")

        # Fix Baybayin inconsistencies
        cur.execute(
            """
            UPDATE words
            SET has_baybayin = FALSE,
                baybayin_form = NULL
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
            """
        )
        logger.debug(f"Corrected {cur.rowcount} entries with baybayin_form but has_baybayin=FALSE.")

        # Ensure has_baybayin flag is set correctly
        cur.execute(
            """
            UPDATE words
            SET has_baybayin = TRUE
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL
            """
        )
        logger.debug(f"Ensured has_baybayin=TRUE for {cur.rowcount} entries where baybayin_form existed.")
        # Note: Logic to handle missing romanized_form (e.g., generating it) would go here or elsewhere.
        # Deleting words based solely on missing romanization is generally discouraged.

        logger.info("Database repairs attempted.")
        return True # Indicate completion

    except psycopg2.Error as db_err:
        logger.error(f"Database error during repair: {db_err}", exc_info=True)
        return False # Indicate failure
    except Exception as e:
        logger.error(f"Unexpected error during repair: {e}", exc_info=True)
        return False # Indicate failure

@with_transaction(commit=True)
def cleanup_baybayin_data(cur):
    """Clean up and normalize Baybayin data in the words table."""
    stats = {
        "removed_invalid_chars": 0,
        "corrected_flags": 0,
        "generated_romanization": 0,
        "removed_empty": 0,
    }
    try:
        # 1. Remove non-Baybayin characters using the SQL pattern derived from VALID_BAYBAYIN_REGEX
        cur.execute(f"""
            UPDATE words
            SET baybayin_form = regexp_replace(baybayin_form, '{INVALID_BAYBAYIN_CHARS_SQL_PATTERN}', '', 'g')
            WHERE baybayin_form IS NOT NULL AND baybayin_form ~ '{INVALID_BAYBAYIN_CHARS_SQL_PATTERN}';
        """)
        stats["removed_invalid_chars"] = cur.rowcount
        logger.info(f"Removed invalid characters from {stats['removed_invalid_chars']} baybayin forms.")

        # 2. Trim whitespace
        cur.execute(r"""
            UPDATE words
            SET baybayin_form = trim(regexp_replace(baybayin_form, '\\s+', ' ', 'g')) -- Also collapse multiple spaces
            WHERE baybayin_form LIKE ' %' OR baybayin_form LIKE '% ' OR baybayin_form LIKE '%  %';
        """)

        # 3. Set baybayin_form to NULL if it becomes empty after cleaning
        cur.execute(r"""
            UPDATE words
            SET baybayin_form = NULL, has_baybayin = FALSE
            WHERE has_baybayin = TRUE AND (baybayin_form = '' OR baybayin_form IS NULL);
        """)
        stats["removed_empty"] = cur.rowcount
        logger.info(f"Removed {stats['removed_empty']} empty or NULL baybayin forms.")

        # 4. Correct has_baybayin flag inconsistencies
        cur.execute(r"""
            UPDATE words
            SET has_baybayin = TRUE
            WHERE has_baybayin = FALSE AND baybayin_form IS NOT NULL AND baybayin_form != '';
        """)
        stats["corrected_flags"] += cur.rowcount
        cur.execute(r"""
            UPDATE words
            SET has_baybayin = FALSE
            WHERE has_baybayin = TRUE AND (baybayin_form IS NULL OR baybayin_form = '');
        """)
        stats["corrected_flags"] += cur.rowcount
        logger.info(f"Corrected has_baybayin flags for {stats['corrected_flags']} entries.")

        # 5. Attempt to generate missing romanized_form (if BaybayinRomanizer is available)
        try:
            romanizer = BaybayinRomanizer()
            cur.execute("SELECT id, baybayin_form FROM words WHERE has_baybayin = TRUE AND romanized_form IS NULL")
            missing_romanization = cur.fetchall()
            updated_count = 0
            for word_id, bb_form in missing_romanization:
                # Add specific try-except around romanization call
                try:
                    romanized = romanizer.romanize(bb_form)
                    if romanized:
                        cur.execute("UPDATE words SET romanized_form = %s WHERE id = %s", (romanized, word_id))
                        updated_count += 1
                except ValueError as rom_val_err: # Catch specific romanizer error
                    logger.warning(f"Romanizer validation error for word ID {word_id}, Baybayin: '{bb_form}': {rom_val_err}")
                except Exception as rom_err: # Catch other errors during romanization
                    logger.warning(f"Could not generate romanization for word ID {word_id}, Baybayin: '{bb_form}': {rom_err}")
            stats["generated_romanization"] = updated_count
            logger.info(f"Attempted to generate missing romanized forms for {len(missing_romanization)} entries, succeeded for {stats['generated_romanization']}.")
        except ImportError:
            logger.warning("BaybayinRomanizer not found in text_helpers. Skipping romanization generation.")
        except Exception as gen_err:
            logger.error(f"Error during romanization generation step: {gen_err}", exc_info=True)

        logger.info("Baybayin data cleanup finished.")
        return stats

    except psycopg2.Error as db_err:
        logger.error(f"Database error during Baybayin cleanup: {db_err}", exc_info=True)
        # Rollback handled by @with_transaction
        raise # Re-raise to signal failure
    except Exception as e:
        logger.error(f"Unexpected error during Baybayin cleanup: {e}", exc_info=True)
        # Rollback handled by @with_transaction
        raise # Re-raise to signal failure

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

# --- End Inserted Functions ---

def process_relations_batch(cur, relations_batch, stats, word_id_cache=None):
    """
    Process a batch of relations collected during word processing.
    
    Args:
        cur: Database cursor
        relations_batch: List of relation dictionaries
        stats: Statistics dictionary to update
        word_id_cache: Cache of word IDs (optional, no longer required)
    """
    if not relations_batch:
        logger.debug("No relations to process in batch")
        return
        
    # Initialize stats if not present
    stats["relations_processed"] = stats.get("relations_processed", 0) + len(relations_batch)
    stats["relations_failed"] = stats.get("relations_failed", 0) 
    stats["relations_inserted"] = stats.get("relations_inserted", 0)
    
    logger.info(f"Processing {len(relations_batch)} relations from batch")
    
    # Process each relation in the batch
    for relation_data in relations_batch:
        try:
            # Validate required keys
            required_keys = ["from_word", "to_word", "relation_type", "source"]
            missing_keys = [key for key in required_keys if key not in relation_data]
            if missing_keys:
                logger.error(f"Missing keys in relation data: {missing_keys} (Data: {relation_data})")
                stats["relations_failed"] += 1
                continue
                    
            from_word_id = relation_data["from_word"]
            to_word = relation_data["to_word"]
            relation_type = relation_data["relation_type"]
            source = relation_data["source"]
            metadata = relation_data.get("metadata", {})
            def_id = relation_data.get("def_id")
            
            # First verify from_word_id exists in the database, don't rely on cache
            cur.execute("SELECT id FROM words WHERE id = %s", (from_word_id,))
            if cur.fetchone() is None:
                logger.warning(f"From word ID {from_word_id} not found in database, skipping relation")
                stats["relations_failed"] += 1
                continue
                
            # If to_word is a string (lemma), try to find or create its ID
            to_word_id = None
            if isinstance(to_word, int):
                to_word_id = to_word
                # Verify that the to_word_id actually exists
                cur.execute("SELECT id FROM words WHERE id = %s", (to_word_id,))
                if cur.fetchone() is None:
                    logger.warning(f"To word ID {to_word_id} not found in database, skipping relation")
                    stats["relations_failed"] += 1
                    continue
            elif isinstance(to_word, str) and to_word.strip():
                # First check directly in the database
                clean_to_word = to_word.strip()
                normalized = normalize_lemma(clean_to_word)
                cur.execute(
                    "SELECT id FROM words WHERE lemma = %s OR normalized_lemma = %s LIMIT 1",
                    (clean_to_word, normalized)
                )
                result = cur.fetchone()
                if result:
                    to_word_id = result[0]
                else:
                    # Not found, create new word
                    try:
                        to_word_id = get_or_create_word_id(cur, clean_to_word, "tl", source)
                    except Exception as word_err:
                        logger.error(f"Error creating to_word '{clean_to_word}': {word_err}")
                        stats["relations_failed"] += 1
                        continue
            
            if not to_word_id:
                logger.warning(f"Could not resolve to_word: {to_word}")
                stats["relations_failed"] += 1
                continue
                
            # Insert the relation using the correct parameter names for insert_relation
            try:
                relation_id = insert_relation(
                    cur,
                    from_word_id,
                    to_word_id,
                    relation_type,
                    source_identifier=source,  # Using the parameter name expected by insert_relation
                    metadata=metadata
                )
                
                if relation_id:
                    stats["relations_inserted"] += 1
                else:
                    logger.error(f"insert_relation returned None for {from_word_id}->{to_word_id} ({relation_type})")
                    stats["relations_failed"] += 1
            except Exception as rel_err:
                logger.error(f"Error inserting relation {from_word_id}->{to_word_id} ({relation_type}): {rel_err}")
                stats["relations_failed"] += 1
                
        except Exception as e:
            logger.error(f"Error processing relation data: {relation_data}. Error: {e}", exc_info=True)
            stats["relations_failed"] += 1
            
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
    Appends the source_identifier to the sources column on conflict if not already present.

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
            "sources": source_identifier.strip(), # Ensure source is stripped
        }

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
                -- Append sources: Add new source if not already present in the comma-separated list
                sources = CASE
                              WHEN etymologies.sources IS NULL THEN EXCLUDED.sources
                              WHEN EXCLUDED.sources IS NULL THEN etymologies.sources
                              WHEN string_to_array(etymologies.sources, ', ') @> ARRAY[EXCLUDED.sources] THEN etymologies.sources
                              ELSE etymologies.sources || ', ' || EXCLUDED.sources
                          END,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """,
            params,
        )
        etymology_id_tuple = cur.fetchone()
        if etymology_id_tuple:
             etymology_id = etymology_id_tuple[0]
             logger.debug(
                 f"Inserted/Updated etymology (ID: {etymology_id}) for word ID {word_id} from source '{source_identifier}'. Text: '{etymology_text[:50]}...'")
             return etymology_id
        else:
             # This case should ideally not happen with RETURNING id, but handle defensively
             logger.warning(f"Upsert for etymology word_id={word_id}, text='{etymology_text[:50]}...' did not return an ID.")
             # Attempt to refetch if needed, though the transaction might rollback anyway
             # cur.execute("SELECT id FROM etymologies WHERE word_id=%(word_id)s AND etymology_text=%(etym_text)s", params)
             # refetched_id = cur.fetchone()
             # return refetched_id[0] if refetched_id else None
             return None # Indicate potential issue

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
def insert_pronunciation(
    cur,
    word_id: int,
    pronunciation_type: str,
    value: str,
    tags: Optional[Union[List[str], Dict]] = None,
    metadata: Optional[Dict] = None,
    source_identifier: Optional[str] = None,
) -> Optional[int]:
    """
    Inserts or updates a pronunciation record for a given word.
    Uses ON CONFLICT on (word_id, type, value) to update tags and metadata.
    Source information is stored within the 'pronunciation_metadata' JSONB column.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        pronunciation_type: Type of pronunciation (e.g., 'IPA', 'audio').
        value: The pronunciation value (e.g., phonetic transcription, audio URL).
        tags: List of strings or dictionary to be stored in the 'tags' JSONB column.
        metadata: Dictionary for additional metadata (stored in 'pronunciation_metadata' JSONB column).
        source_identifier: Identifier for the data source (stored within metadata).

    Returns:
        The ID of the inserted/updated pronunciation record, or None if failed.
    """
    if not value or not isinstance(value, str):
        logger.warning(
            f"Skipping pronunciation insert for word ID {word_id}: Missing or invalid value."
        )
        return None

    pron_type = pronunciation_type.strip().lower() if pronunciation_type else None
    if not pron_type:
        logger.warning(
            f"Skipping pronunciation insert for word ID {word_id}: Missing type."
        )
        return None

    # --- Prepare Tags (JSONB) ---
    tags_json = None
    if isinstance(tags, list):
        # Ensure items are strings and handle potential non-string items gracefully
        processed_tags = [str(t).strip() for t in tags if t is not None]
        tags_data = {"tags": processed_tags} if processed_tags else {}
    elif isinstance(tags, dict):
        tags_data = tags
    else:
        tags_data = {}  # Default to empty dict

    try:
        # Only dump if tags_data is not empty
        tags_json = json.dumps(tags_data, default=str) if tags_data else None
    except TypeError as e:
        logger.error(
            f"Could not serialize tags for pronunciation (Word ID {word_id}, Type {pron_type}): {e}. Tags: {tags_data}",
            exc_info=True,
        )
        tags_json = None  # Proceed without tags

    # --- Prepare Metadata (JSONB) ---
    # Start with a copy of input metadata or an empty dict
    pronunciation_metadata = metadata.copy() if isinstance(metadata, dict) else {}

    # Add/Update source identifier within metadata
    if source_identifier:
        source_identifier = source_identifier.strip()
        if source_identifier:  # Ensure source_identifier is not empty after stripping
            existing_sources = pronunciation_metadata.get("sources", [])
            # Ensure existing_sources is a list
            if not isinstance(existing_sources, list):
                logger.warning(
                    f"Existing 'sources' in pronunciation metadata for word ID {word_id} (Type: {pron_type}) is not a list ({type(existing_sources)}). Overwriting with new source."
                )
                existing_sources = []

            # Add the new source if it's not already present
            if source_identifier not in existing_sources:
                existing_sources.append(source_identifier)

            # Update the metadata dictionary
            pronunciation_metadata["sources"] = existing_sources

    metadata_json = None
    try:
        # Only dump if pronunciation_metadata is not empty
        metadata_json = (
            json.dumps(pronunciation_metadata, default=str)
            if pronunciation_metadata
            else None
        )
    except TypeError as e:
        logger.error(
            f"Could not serialize metadata for pronunciation (Word ID {word_id}, Type {pron_type}): {e}. Metadata: {pronunciation_metadata}",
            exc_info=True,
        )
        metadata_json = None  # Proceed without metadata

    # --- Database Operation ---
    params = {
        "word_id": word_id,
        "type": pron_type,
        "value": value.strip(),
        # "tags": tags_json,  # Pass None if serialization failed or data was empty
        "tags": Json(tags_data) if tags_data else None, # Use Json adapter
        # "metadata": metadata_json,  # Pass None if serialization failed or data was empty
        "metadata": Json(pronunciation_metadata) if pronunciation_metadata else None, # Use Json adapter
        # "sources": source_identifier, # Removed: Stored in metadata
    }

    try:
        # Note: Using explicit NULLIF for JSONB fields in VALUES to handle None params correctly
        cur.execute(
            """
            INSERT INTO pronunciations (word_id, type, value, tags, pronunciation_metadata)
            VALUES (
                %(word_id)s, %(type)s, %(value)s,
                %(tags)s,
                %(metadata)s
                -- sources column removed
            )
            ON CONFLICT (word_id, type, value) DO UPDATE SET
                -- Merge tags and metadata JSONB fields using || operator
                -- Handles cases where existing or excluded values might be NULL
                tags = COALESCE(pronunciations.tags, '{}'::jsonb) || COALESCE(EXCLUDED.tags, '{}'::jsonb),
                pronunciation_metadata = COALESCE(pronunciations.pronunciation_metadata, '{}'::jsonb) || COALESCE(EXCLUDED.pronunciation_metadata, '{}'::jsonb),
                -- sources update removed
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            params,
        )
        pronunciation_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated pronunciation (ID: {pronunciation_id}) for word ID {word_id}. Type: {pron_type}, Value: '{value[:50]}...'"
        )
        return pronunciation_id

    except psycopg2.IntegrityError as e:
        logger.error(
            f"Integrity error inserting pronunciation for word ID {word_id} (Type: {pron_type}, Value: '{value[:50]}...'). Error: {e.pgcode} {e.pgerror}"
        )
        # Rollback might be handled by decorator, but can be explicit
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after IntegrityError: {rb_e}")
        return None
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting pronunciation for word ID {word_id} (Type: {pron_type}): {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after DB Error: {rb_e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting pronunciation for word ID {word_id} (Type: {pron_type}): {e}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after Unexpected Error: {rb_e}")
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

@with_transaction(commit=True)
def insert_definition(
    cur,
    word_id: int,
    definition_text: str,
    part_of_speech: Optional[str] = None, # Accept raw part_of_speech
    # notes: Optional[str] = None, # Removed - No 'notes' column in schema
    # examples: Optional[List[Dict]] = None, # Removed: Examples are handled by insert_definition_example
    usage_notes: Optional[str] = None,
    # cultural_notes: Optional[str] = None, # Removed - No 'cultural_notes' column in schema
    # etymology_notes: Optional[str] = None, # Removed - No 'etymology_notes' column in schema
    # scientific_name: Optional[str] = None, # Removed - No 'scientific_name' column in schema
    # verified: Optional[bool] = False, # Removed - No 'verified' column in schema
    # verification_notes: Optional[str] = None, # Removed - No 'verification_notes' column in schema
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    # popularity_score: Optional[float] = 0.0, # Removed - column doesn't exist in schema
    sources: Optional[str] = None,
) -> Optional[int]:
    """
    Inserts or updates a definition for a word.
    Uses ON CONFLICT to update existing definitions based on (word_id, definition_text, standardized_pos_id).
    Examples are handled separately by insert_definition_example.

    Args:
        cur: Database cursor.
        word_id: ID of the word.
        definition_text: The text of the definition.
        part_of_speech: Raw part of speech string from source.
        # original_pos: Original part of speech tag from source. Removed.
        # standardized_pos_id: Foreign key to the standardized part_of_speech table. Removed.
        # notes: General notes about the definition. Removed.
        # examples: List of example dictionaries. Removed.
        usage_notes: Notes on usage.
        # cultural_notes: Cultural context notes. Removed.
        # etymology_notes: Notes related to etymology within the definition context. Removed.
        # scientific_name: Scientific name if applicable. Removed.
        # verified: Boolean flag indicating verification status. Removed.
        # verification_notes: Notes regarding verification. Removed.
        tags: List of tags associated with the definition (stored as comma-separated text).
        metadata: JSONB dictionary for additional metadata.
        # popularity_score: Numeric score indicating definition popularity/importance. Removed.
        sources: Comma-separated string of source identifiers.

    Returns:
        The ID of the inserted/updated definition record, or None if failed.
    """
    if not definition_text or not isinstance(definition_text, str):
        logger.warning(
            f"Skipping definition insert for word ID {word_id}: Missing or invalid definition text."
        )
        return None

    # --- Determine original_pos and standardized_pos_id from part_of_speech ---
    original_pos = part_of_speech.strip() if part_of_speech else None
    standardized_pos_id = None
    if original_pos:
        try:
            standardized_pos_id = get_standardized_pos_id(cur, original_pos)
        except Exception as pos_err:
            logger.error(f"Error getting standardized POS ID for '{original_pos}' (Word ID {word_id}): {pos_err}", exc_info=True)
            # Optionally fall back to 'unc' or None depending on desired behavior
            standardized_pos_id = get_uncategorized_pos_id(cur) # Fallback to 'unc' ID

    # Prepare data, ensuring None is passed for empty optional fields
    tags_string = ",".join(tags) if tags else None # Corrected: join tags, not empty string


    params = {
        "word_id": word_id,
        "def_text": definition_text.strip(),
        "orig_pos": original_pos, # Use determined original_pos
        "std_pos_id": standardized_pos_id, # Use determined standardized_pos_id
        # "notes": notes.strip() if notes else None, # Removed
        # "examples": examples_json, # Removed
        "usage_notes": usage_notes.strip() if usage_notes else None,
        # "cult_notes": cultural_notes.strip() if cultural_notes else None, # Removed
        # "etym_notes": etymology_notes.strip() if etymology_notes else None, # Removed
        # "sci_name": scientific_name.strip() if scientific_name else None, # Removed
        # "verif_notes": verification_notes.strip() if verification_notes else None, # REMOVED key
        "tags": tags_string,
        "metadata": Json(metadata) if metadata else None, # Wrap metadata in Json
        # "pop_score": float(popularity_score) if popularity_score is not None else 0.0, # REMOVED - column doesn't exist
        "sources": sources.strip() if sources else None,
    }

    sql_insert = """
        INSERT INTO definitions (
            word_id, definition_text, original_pos, standardized_pos_id, -- notes,
            usage_notes, -- cultural_notes,
            -- etymology_notes,
            -- scientific_name,
            tags,
            definition_metadata, -- Corrected column name
            sources
            -- popularity_score,
            -- examples
        )
        VALUES (
            %(word_id)s, %(def_text)s, %(orig_pos)s, %(std_pos_id)s, -- Removed notes placeholder
            %(usage_notes)s, -- cultural_notes removed below
            -- etymology_notes removed below
            -- sci_name removed
            %(tags)s,
            %(metadata)s, -- Parameter name is still metadata
            %(sources)s
            -- popularity_score removed
            -- examples value removed
        )
        ON CONFLICT (word_id, definition_text, standardized_pos_id) DO UPDATE SET
            original_pos = EXCLUDED.original_pos,
            -- notes = EXCLUDED.notes,
            usage_notes = EXCLUDED.usage_notes,
            -- cultural_notes = EXCLUDED.cultural_notes,
            -- etymology_notes = EXCLUDED.etymology_notes,
            -- scientific_name = EXCLUDED.scientific_name,
            tags = EXCLUDED.tags,
            definition_metadata = EXCLUDED.definition_metadata, -- Corrected column name
            sources = EXCLUDED.sources
            -- popularity_score = EXCLUDED.popularity_score,
            -- examples = EXCLUDED.examples
        RETURNING id;
        """

    try:
        cur.execute(sql_insert, params)
        definition_id = cur.fetchone()[0]
        logger.debug(
            f"Inserted/Updated definition (ID: {definition_id}) for word ID {word_id}. Text: '{definition_text[:50]}...'")
        return definition_id

    except psycopg2.IntegrityError as e:
        # Handle potential FK violation for standardized_pos_id
        if e.pgcode == "23503":  # foreign_key_violation
            logger.warning(
                f"Integrity error inserting definition for word ID {word_id}. Standardized POS ID {standardized_pos_id} likely doesn't exist. Setting to NULL. Error: {e.pgerror}"
            )
            params["std_pos_id"] = None  # Retry with NULL POS ID
            try:
                # Re-execute the same SQL but with updated params where std_pos_id is None
                cur.execute(sql_insert, params)
                definition_id = cur.fetchone()[0]
                logger.debug(
                    f"Inserted/Updated definition (ID: {definition_id}) for word ID {word_id} with NULL POS ID after FK error."
                )
                return definition_id
            except psycopg2.Error as retry_e:
                logger.error(
                    f"Database error on retry inserting definition for word ID {word_id} with NULL POS ID: {retry_e.pgcode} {retry_e.pgerror}",
                    exc_info=True,
                )
                cur.connection.rollback()  # Rollback the retry attempt
                return None
        else:
            logger.error(
                f"Integrity error inserting definition for word ID {word_id}: {e.pgcode} {e.pgerror}",
                exc_info=True,
            )
            # Consider rolling back explicitly here if the transaction decorator doesn't handle it
            # try: cur.connection.rollback()
            # except Exception: pass
            return None  # Indicate failure
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting definition for word ID {word_id}: {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        # Consider rolling back
        # try: cur.connection.rollback()
        # except Exception: pass
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting definition for word ID {word_id}: {e}",
            exc_info=True,
        )
        # Consider rolling back
        # try: cur.connection.rollback()
        # except Exception: pass
        return None



@with_transaction(commit=False) # Keep commit=False as it's likely called within insert_definition
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
    standard_code = get_standard_code(pos_string)  # Use the corrected function

    try:
        # Query using the CODE obtained from standardization
        cur.execute("SELECT id FROM parts_of_speech WHERE code = %s", (standard_code,))
        result = cur.fetchone()
        if result:
            return result[0]
        else:
            # If the standard code (even 'unc') is somehow not in the table, log error
            logger.error(
                f"Standard POS code '{standard_code}' (derived from '{pos_string}') not found in parts_of_speech table. Falling back to fetching 'unc' ID."
            )
            return get_uncategorized_pos_id(cur)  # Fallback to ensure 'unc' exists
    except Exception as e:
        logger.error(
            f"Error fetching POS ID for code '{standard_code}' (from '{pos_string}'): {e}. Returning 'unc'."
        )
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

# @with_transaction(commit=True) # Commit changes after setting up POS
# def setup_parts_of_speech(cur):
#     """Insert standard parts of speech into the database if they don't exist."""
#     logger.info("Setting up standard parts of speech...")
#     standard_pos = [
#         # code, name_en, name_tl, description
#         ('n', 'Noun', 'Pangngalan', 'Represents a person, place, thing, or idea.'),
#         ('v', 'Verb', 'Pandiwa', 'Represents an action or state of being.'),
#         ('adj', 'Adjective', 'Pang-uri', 'Describes or modifies a noun.'),
#         ('adv', 'Adverb', 'Pang-abay', 'Describes or modifies a verb, adjective, or other adverb.'),
#         ('pron', 'Pronoun', 'Panghalip', 'Replaces a noun.'),
#         ('prep', 'Preposition', 'Pang-ukol', 'Shows relationship between a noun/pronoun and other words.'),
#         ('conj', 'Conjunction', 'Pangatnig', 'Connects words, phrases, or clauses.'),
#         ('intj', 'Interjection', 'Pandamdam', 'Expresses strong emotion.'),
#         ('det', 'Determiner', 'Pantukoy', 'Introduces a noun (e.g., articles.'),
#         ('affix', 'Affix', 'Panlapi', 'Morpheme added to a word base (prefix, suffix, infix).'),
#         ('lig', 'Ligature', 'Pang-angkop', 'Connects words grammatically (e.g., na, ng).'),
#         ('part', 'Particle', 'Kataga', 'Function word that doesn\'t fit other categories.'),
#         ('num', 'Number', 'Pamilang', 'Represents a quantity.'),
#         ('expr', 'Expression', 'Pahayag', 'A fixed phrase or idiom.'),
#         ('punc', 'Punctuation', 'Bantas', 'Marks used to structure writing.'),
#         ('idm', 'Idiom', 'Idyoma', 'Idiomatic expression.'),
#         ('col', 'Colloquial', 'Kolokyal', 'Informal language.'),
#         ('var', 'Variant', 'Baryant', 'Alternative form or spelling.'),
#         ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined or ambiguous.'),
#         # Add any other essential codes here if needed
#     ]
#     try:
#         insert_query = """
#             INSERT INTO parts_of_speech (code, name_en, name_tl, description)
#             VALUES (%s, %s, %s, %s)
#             ON CONFLICT (code) DO NOTHING;
#         """
#         execute_values(cur, insert_query, standard_pos, template=None, page_size=100)
#         logger.info(f"Ensured standard parts of speech entries exist (processed {len(standard_pos)} codes).")
#     except Exception as e:
#         logger.error(f"Error setting up parts of speech: {e}", exc_info=True)
#         raise # Re-raise to allow transaction rollback

# --- Add functions from backup --- 
def create_or_update_tables(conn):
    """Create or update tables based on the TABLE_CREATION_SQL string."""
    logger.info("Starting table creation/update process.")
    try:
        with conn.cursor() as cur:
            logger.info("Creating required PostgreSQL extensions...") # Indent this line
            cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            # Add pg_vector extension creation if needed
            # cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # *** FIX: Explicitly drop the potentially outdated constraint first ***
            try:
                logger.info("Dropping existing baybayin_form_regex constraint on words table (if it exists)...")
                cur.execute("ALTER TABLE words DROP CONSTRAINT IF EXISTS baybayin_form_regex;")
                logger.info("Constraint baybayin_form_regex dropped successfully or did not exist.")
            except psycopg2.Error as drop_err:
                 # Log error if dropping fails for reasons other than 'does not exist' (which is handled by IF EXISTS)
                 # Note: UndefinedTable error might occur if 'words' doesn't exist yet, which is fine here.
                if drop_err.pgcode != psycopg2.errorcodes.UNDEFINED_TABLE:
                     logger.warning(f"Could not drop constraint baybayin_form_regex (may be expected if table doesn't exist): {drop_err}")
                else:
                    logger.info("Words table doesn't exist yet, skipping constraint drop.")


            logger.info("Executing main table creation SQL...")
            cur.execute(TABLE_CREATION_SQL)
            logger.info("Main table creation SQL executed.")
        conn.commit()
        logger.info("Tables created or updated successfully.")
    except psycopg2.Error as e:
        logger.error(f"Database error during table creation: {e}", exc_info=True)
        conn.rollback() # Rollback on error
        raise DatabaseError(f"Failed to create/update tables: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during table creation: {e}", exc_info=True)
        conn.rollback() # Rollback on error
        raise DatabaseError(f"Unexpected error during table creation: {e}") from e

@with_transaction(commit=True)
def insert_relation(
    cur,
    from_word_id: int,
    to_word_id: int,
    relation_type: str,
    source_identifier: str,
    metadata: Optional[Dict] = None,
) -> Optional[int]:
    """
    Create (or merge) a relation row in `relations`.
    – `source_identifier` is appended to the comma‑separated `sources` column.
    – `metadata` is kept as JSONB and deep‑merged on conflict.

    Returns the relation's id, or None on failure/skip.
    """
    # ---------- quick guards -------------------------------------------------
    if not source_identifier:
        logger.warning("insert_relation‑skip: source_identifier is required")
        return None
    if from_word_id == to_word_id:
        logger.debug("insert_relation‑skip: self‑relation")
        return None
    if not relation_type or not isinstance(relation_type, str):
        logger.warning("insert_relation‑skip: empty relation_type")
        return None

    relation_type = relation_type.strip().lower()
    source_identifier = source_identifier.strip()

    # ---------- build params -------------------------------------------------
    json_metadata = Json(metadata or {})          # never NULL → avoids jsonb || NULL
    params = dict(
        from_id=from_word_id,
        to_id=to_word_id,
        rel_type=relation_type,
        sources=source_identifier,
        metadata=json_metadata,
    )

    # ---------- upsert -------------------------------------------------------
    upsert_sql = """
            INSERT INTO relations (from_word_id, to_word_id, relation_type, sources, metadata)
            VALUES (%(from_id)s, %(to_id)s, %(rel_type)s, %(sources)s, %(metadata)s)
    ON CONFLICT (from_word_id, to_word_id, relation_type) DO UPDATE
    SET
        -- append source if it is new
                sources = CASE
                    WHEN relations.sources IS NULL
                         OR relations.sources = '' THEN EXCLUDED.sources
                    WHEN string_to_array(relations.sources, ', ')
                         @> ARRAY[EXCLUDED.sources] THEN relations.sources
                              ELSE relations.sources || ', ' || EXCLUDED.sources
                          END,
        -- deep‑merge JSONB safely even when either side is NULL
        metadata = COALESCE(relations.metadata, '{}'::jsonb)
                   || COALESCE(EXCLUDED.metadata, '{}'::jsonb),
                updated_at = CURRENT_TIMESTAMP
    RETURNING id;
    """

    try:
        cur.execute(upsert_sql, params)
        rid = cur.fetchone()[0]
        # Indent the logger call correctly under the try block where rid is available
        logger.debug(
            "relation %s → %s [%s] (id %s) recorded from " "%s",
            from_word_id,
            to_word_id,
            relation_type,
            rid,
            source_identifier,
        )
        return rid

    except psycopg2.IntegrityError as e:            # FK problems etc.
        logger.error("insert_relation FK/unique failure: %s", e.pgerror.strip())
    except psycopg2.Error as e:                     # any other PG error
        logger.error("insert_relation PG error: %s", e.pgerror.strip(), exc_info=True)
    except Exception as e:                          # python‑side
        logger.exception("insert_relation unexpected error: %s", e)

    # Ensure None is returned if any exception occurred
    return None


@with_transaction(commit=True)
def insert_definition_example(
    cur, definition_id: int, example_data: Dict, source_identifier: str
) -> Optional[int]:
    """
    Insert an example associated with a definition.
    Stores romanization within the 'metadata' JSON column.

    Args:
        cur: Database cursor.
        definition_id: ID of the definition this example belongs to.
        example_data: Dictionary containing example data (e.g., {"text": "...", "translation": "...", "type": "...", "ref": "...", "roman": "..."}).
                       Expected keys: 'text' (required), 'translation', 'type', 'ref', 'roman'.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted definition example record, or None if failed.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping definition example insert for def ID {definition_id}: Missing MANDATORY source identifier."
        )
        return None

    example_text = example_data.get("text")
    if not example_text or not isinstance(example_text, str):
        logger.warning(
            f"Skipping definition example for def ID {definition_id} (source '{source_identifier}'): Missing or invalid 'text'. Data: {example_data}"
        )
        return None

    # Extract other fields safely
    translation = example_data.get("translation")
    example_type = example_data.get("type")
    reference = example_data.get("ref")  # Note: source data uses 'ref'
    romanization = example_data.get(
        "roman"
    )  # Note: source data uses 'roman' for romanization

    # Prepare metadata (start with existing if provided, or empty)
    metadata = example_data.get("metadata", {})
    if not isinstance(metadata, dict):
        logger.warning(
            f"Metadata for example (Def ID {definition_id}) is not a dict. Initializing empty. Data: {metadata}"
        )
        metadata = {}
    else:
        metadata = metadata.copy()  # Work on a copy

    # Add romanization to metadata if present and not empty
    if romanization and isinstance(romanization, str):
        roman_strip = romanization.strip()
        if roman_strip:
            metadata["romanization"] = roman_strip

    # Add any other fields from example_data not explicitly handled into metadata
    standard_keys = {"text", "translation", "type", "ref", "roman", "metadata"}
    for k, v in example_data.items():
        if k not in standard_keys and k not in metadata:
            try:
                # Ensure value is serializable before adding
                json.dumps({f"extra_{k}": v})
                metadata[f"extra_{k}"] = v
            except TypeError:
                logger.warning(
                    f"Could not serialize extra example data key '{k}' for def ID {definition_id}. Skipping field."
                )

    metadata_json = None
    try:
        # Only serialize if metadata dict is not empty
        metadata_json = json.dumps(metadata, default=str) if metadata else None
    except TypeError as e:
        logger.error(
            f"Could not serialize metadata for example (Def ID {definition_id}): {e}. Metadata: {metadata}",
            exc_info=True,
        )
        metadata_json = None  # Proceed without metadata on error

    # --- Database Operation ---
    params = {
        "def_id": definition_id,
        "ex_text": example_text.strip(),
        "trans": translation.strip() if isinstance(translation, str) else None,
        "ex_type": example_type.strip() if isinstance(example_type, str) else None,
        "ref": reference.strip() if isinstance(reference, str) else None,
        # "roman": removed - stored in metadata
        "metadata": metadata_json,  # Pass None if empty or serialization failed
        "sources": source_identifier.strip(),  # Use the mandatory source_identifier for the sources column
    }

    try:
        # Note: Using explicit NULLIF for JSONB fields in VALUES
        cur.execute(
            """
            INSERT INTO definition_examples (definition_id, example_text, translation, example_type, reference, metadata, sources)
            VALUES (
                %(def_id)s, %(ex_text)s, %(trans)s, %(ex_type)s, %(ref)s,
                NULLIF(%(metadata)s, NULL)::jsonb,
                %(sources)s
                -- romanization column removed
            )
            ON CONFLICT (definition_id, example_text) DO UPDATE SET
                translation = COALESCE(EXCLUDED.translation, definition_examples.translation),
                example_type = COALESCE(EXCLUDED.example_type, definition_examples.example_type),
                reference = COALESCE(EXCLUDED.reference, definition_examples.reference),
                -- Merge metadata using || operator
                metadata = COALESCE(definition_examples.metadata, '{}'::jsonb) || COALESCE(NULLIF(%(metadata)s, NULL)::jsonb, '{}'::jsonb),
                -- Append sources
                sources = CASE
                              WHEN definition_examples.sources IS NULL THEN EXCLUDED.sources
                              WHEN EXCLUDED.sources IS NULL THEN definition_examples.sources
                              WHEN string_to_array(definition_examples.sources, ', ') @> ARRAY[EXCLUDED.sources] THEN definition_examples.sources
                              ELSE definition_examples.sources || ', ' || EXCLUDED.sources
                          END,
                -- romanization_text update removed
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            params,
        )
        example_id_tuple = cur.fetchone()
        if example_id_tuple:
            example_id = example_id_tuple[0]
            logger.info(
                f"Inserted/Updated definition example (ID: {example_id}) for def ID {definition_id} from source '{source_identifier}'. Text: '{example_text[:50]}...'"
            )
            return example_id
        else:
            logger.error(
                f"Failed to get definition example ID after upsert for def ID {definition_id}, text: '{example_text[:50]}...'."
            )
            cur.connection.rollback()
            return None

    except psycopg2.IntegrityError as e:
        logger.error(
            f"Database IntegrityError inserting example for def ID {definition_id} ('{example_text[:50]}...') from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(
                f"Rollback failed after IntegrityError inserting example: {rb_e}"
            )
        return None
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting example for def ID {definition_id} ('{example_text[:50]}...') from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after DB error inserting example: {rb_e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting example for def ID {definition_id} ('{example_text[:50]}...') from '{source_identifier}': {e}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.error(
                f"Failed to rollback transaction after unexpected error inserting example: {rb_e}"
            )
        return None

@with_transaction(commit=True)          # keep same decorator
def get_or_create_word_id(
    cur,
    lemma: str,
    language_code: str = DEFAULT_LANGUAGE_CODE,
    source_identifier: Optional[str] = None,
    preserve_numbers: bool = False,
    **kwargs,
) -> Optional[int]:
    """
    Get the ID of a word by lemma and language, creating it if it doesn't exist.
    Updates source_info for existing words.

    Args:
        cur: Database cursor.
        lemma: The word lemma.
        language_code: ISO language code (default: 'tl').
        source_identifier: Identifier for the data source (used for tracking).
        preserve_numbers: If True, don't remove trailing numbers from the lemma.
        **kwargs: Additional fields for the words table if creating a new word.

    Returns:
        The integer ID of the word, or None if an error occurred.
    """
    if not lemma or not isinstance(lemma, str):
        logger.warning("get_or_create_word_id called with empty or invalid lemma")
        return None

    original_lemma = lemma.strip()
    if not original_lemma:
        logger.warning("get_or_create_word_id called with empty lemma after stripping")
        return None

    # Normalize lemma and language code
    processed_lemma = original_lemma if preserve_numbers else remove_trailing_numbers(original_lemma)
    normalized = normalize_lemma(processed_lemma)
    # Safeguard: Ensure language_code defaults correctly if None is passed
    lang_code_input = language_code or DEFAULT_LANGUAGE_CODE
    # --- CHANGE HERE: Use get_language_code instead of get_standard_code ---
    # lang_code = get_standard_code(lang_code_input, code_type='language') or DEFAULT_LANGUAGE_CODE
    lang_code = get_language_code(lang_code_input) or DEFAULT_LANGUAGE_CODE # Use the specific language helper

    # Check cache first (optional)
    # cache_key = f"word_id:{normalized}:{lang_code}"
    # cached_id = cache.get(cache_key)
    # if cached_id:
    #     logger.debug(f"Cache hit for {normalized} ({lang_code}): {cached_id}")
    #     return cached_id

    # Query database
    try:
        # Prioritize exact normalized match
        cur.execute(
            "SELECT id, source_info FROM words WHERE normalized_lemma = %s AND language_code = %s",
            (normalized, lang_code),
        )
        result = cur.fetchone()

        if result:
            word_id, current_source_info = result
            logger.debug(
                f"Found existing word '{processed_lemma}' ({lang_code}) with ID: {word_id}"
            )
            # Update source info if a new source is provided
            if source_identifier:
                updated_source_info_json = update_word_source_info(
                    current_source_info, source_identifier
                )
                # Check if update is needed
                if updated_source_info_json != (current_source_info if isinstance(current_source_info, str) else json.dumps(current_source_info)):
                    cur.execute(
                        "UPDATE words SET source_info = %s::jsonb, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                        (updated_source_info_json, word_id),
                    )
                    logger.debug(f"Updated source info for word ID {word_id}")
            # Update cache (optional)
            # cache.set(cache_key, word_id, timeout=3600)
            return word_id
        else:
            # Word not found, create it
            logger.debug(f"Word '{processed_lemma}' ({lang_code}) not found. Creating new entry...")
            # --- Prepare Baybayin, Romanized, etc. from kwargs --- #
            has_baybayin = bool(kwargs.get("has_baybayin")) # Default to False
            if not has_baybayin and "has_baybayin" in kwargs: # Allow explicit False
                has_baybayin = bool(kwargs.get("has_baybayin"))

            baybayin_form_input = kwargs.get("baybayin_form") # Get the raw input
            baybayin_form_to_store = None # Initialize variable for storing validated form

            if has_baybayin and baybayin_form_input and isinstance(baybayin_form_input, str):
                # Clean the input form first using the correct helper
                # Ensure clean_baybayin_text is imported from text_helpers
                cleaned_bb_string = clean_baybayin_text(baybayin_form_input)

                if cleaned_bb_string: # Check if cleaning resulted in non-empty string
                    # Now validate the cleaned string with the regex
                    if not VALID_BAYBAYIN_REGEX.match(cleaned_bb_string):
                        logger.warning(f"Provided baybayin_form '{baybayin_form_input}' for lemma '{processed_lemma}' is invalid after cleaning/regex check ('{cleaned_bb_string}'). Setting to NULL.")
                        # baybayin_form_to_store remains None
                        has_baybayin = False
                    else:
                        # Validation passed, store the cleaned string
                        baybayin_form_to_store = cleaned_bb_string
                        logger.debug(f"Validated Baybayin form '{baybayin_form_to_store}' for '{processed_lemma}'")
                else: # Cleaning resulted in empty string
                    logger.warning(f"Provided baybayin_form '{baybayin_form_input}' for lemma '{processed_lemma}' became empty after cleaning. Setting flag to FALSE.")
                    # baybayin_form_to_store remains None
                    has_baybayin = False

            elif has_baybayin: # has_baybayin=True but no form provided or invalid type
                logger.warning(f"has_baybayin=True but no valid string baybayin_form provided for '{processed_lemma}'. Setting flag to FALSE.")
                has_baybayin = False # Corrected typo, ensure aligned with logger
                # baybayin_form_to_store remains None
            # else: # has_baybayin=False, baybayin_form_to_store remains None

            # Prepare Romanized form (uses baybayin_form_to_store)
            # --- FIX Indentation START --- #
            romanized_form = kwargs.get("romanized_form")
            if has_baybayin and not romanized_form and baybayin_form_to_store: # Check against the validated form to store
                try:
                    # Ensure BaybayinRomanizer is imported from text_helpers
                    romanizer = BaybayinRomanizer()
                    romanized_form = romanizer.romanize(baybayin_form_to_store) # Romanize the cleaned, validated form
                    logger.debug(f"Generated romanized form '{romanized_form}' for '{baybayin_form_to_store}'")
                except Exception as rom_err:
                    logger.warning(f"Could not generate romanized form for '{baybayin_form_to_store}': {rom_err}")
            # --- FIX Indentation END --- #

            # --- Prepare other fields --- #
            # --- Aggressive Final Cleaning for DB --- #
            if baybayin_form_to_store:
                allowed_chars = set(chr(i) for i in range(0x1700, 0x171F + 1)) | {chr(0x1735), chr(0x1736), ' '}
                original_len = len(baybayin_form_to_store)
                baybayin_form_to_store = ''.join(c for c in baybayin_form_to_store if c in allowed_chars)
                if len(baybayin_form_to_store) != original_len:
                    logger.warning(f"Aggressive cleaning removed characters from Baybayin form for '{processed_lemma}'. Original len: {original_len}, New len: {len(baybayin_form_to_store)}. Storing: '{baybayin_form_to_store}'")
                # If cleaning makes it empty, nullify it and the flag
                if not baybayin_form_to_store:
                    logger.warning(f"Baybayin form for '{processed_lemma}' became empty after aggressive cleaning. Setting to NULL.")
                    baybayin_form_to_store = None
                    has_baybayin = False

            # Ensure update_word_source_info is defined/imported
            source_info_json = update_word_source_info(None, source_identifier) # Initial source info
            # Ensure Json is imported from psycopg2.extras
            word_metadata_json = json.dumps(kwargs.get("word_metadata", {}), default=str)

            insert_params = {
                "lemma": processed_lemma,
                "norm_lemma": normalized,
                "lang": lang_code,
                "has_bb": has_baybayin,
                "bb_form": baybayin_form_to_store, # Use the validated/cleaned form
                "rom_form": romanized_form,
                "root_id": kwargs.get("root_word_id"),
                "pref_spell": kwargs.get("preferred_spelling"),
                "tags": kwargs.get("tags"),
                "idioms": Json(kwargs.get("idioms", [])),
                "pron_data": Json(kwargs.get("pronunciation_data", {})),
                "source_info": source_info_json,
                "metadata": word_metadata_json,
                "badlit": kwargs.get("badlit_form"),
                "hyphen": Json(kwargs.get("hyphenation", {})),
                "is_proper": kwargs.get("is_proper_noun", False),
                "is_abbr": kwargs.get("is_abbreviation", False),
                "is_init": kwargs.get("is_initialism", False),
            }

            insert_sql = """
            INSERT INTO words (
                    lemma, normalized_lemma, language_code, has_baybayin,
                    baybayin_form, romanized_form, root_word_id,
                    preferred_spelling, tags, idioms, pronunciation_data,
                    source_info, word_metadata, badlit_form, hyphenation,
                    is_proper_noun, is_abbreviation, is_initialism,
                    search_text -- Added search_text column
            )
            VALUES (
                    %(lemma)s, %(norm_lemma)s, %(lang)s, %(has_bb)s,
                    %(bb_form)s, %(rom_form)s, %(root_id)s,
                    %(pref_spell)s, %(tags)s, %(idioms)s, %(pron_data)s,
                    %(source_info)s::jsonb, %(metadata)s::jsonb, %(badlit)s, %(hyphen)s,
                    %(is_proper)s, %(is_abbr)s, %(is_init)s,
                    -- Generate search_text on insert
                    to_tsvector('english', COALESCE(%(lemma)s, '') || ' ' || COALESCE(%(norm_lemma)s, '') || ' ' || COALESCE(%(bb_form)s, '') || ' ' || COALESCE(%(rom_form)s, ''))
                )
                RETURNING id;
            """
            cur.execute(insert_sql, insert_params)
            new_word_id = cur.fetchone()[0]
            logger.info(
                f"Created new word '{processed_lemma}' ({lang_code}) with ID: {new_word_id} from source '{source_identifier}'"
            )
            # Update cache (optional)
            # cache.set(cache_key, new_word_id, timeout=3600)
            return new_word_id

    except psycopg2.IntegrityError as e:
        # Specific handling for unique constraint violation
        # Ensure psycopg2.errorcodes is imported
        if e.pgcode == psycopg2.errorcodes.UNIQUE_VIOLATION: # Correct check
            logger.warning(f"Unique constraint violation during insert attempt for '{normalized}' ({lang_code}). Retrying fetch. Error: {e.pgerror}")
            # Rollback the failed insert attempt before retrying fetch
            try:
                cur.connection.rollback()
            except Exception as rb_err:
                logger.error(f"Rollback failed after unique violation for '{normalized}': {rb_err}")
                return None # Cannot safely proceed

            # --- FIX Indentation START --- #
            # Retry fetching the ID
            try:
                cur.execute(
                    "SELECT id FROM words WHERE normalized_lemma = %s AND language_code = %s",
                    (normalized, lang_code),
                )
                result = cur.fetchone()
                if result:
                    # TODO: Consider updating source_info here as well if needed after failed insert
                    logger.info(f"Successfully fetched ID {result[0]} for '{normalized}' ({lang_code}) after unique violation.")
                    return result[0]
                else:
                    logger.error(f"Could not fetch ID for '{normalized}' ({lang_code}) even after unique violation. Data inconsistency suspected.")
                    return None
            except Exception as fetch_err:
                logger.error(f"Error fetching ID for '{normalized}' ({lang_code}) after unique violation: {fetch_err}", exc_info=True)
                return None
            # --- FIX Indentation END --- #
        else:
            # Handle other integrity errors (e.g., check constraints, FK violations)
            logger.error(
                f"Database integrity error processing word '{original_lemma}' ({lang_code}): {e.pgcode} {e.pgerror}",
                exc_info=True, # Include traceback for DB errors
            )
            # Rollback handled by decorator
            return None
    except psycopg2.Error as e:
        # Handle other database errors
        logger.error(
            f"Database error processing word '{original_lemma}' ({lang_code}): {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        # Rollback handled by decorator
        return None
    except Exception as e:
        # Handle unexpected errors
        logger.error(
            f"Unexpected error processing word '{original_lemma}' ({lang_code}): {e}",
            exc_info=True,
        )
        # Rollback handled by decorator
        return None


def batch_get_or_create_word_ids(
    cur, entries: List[Tuple[str, str]], source: str = None, batch_size: int = 1000
) -> Dict[Tuple[str, str], int]:
    """
    Create or get IDs for multiple words in batches.
    Uses the @with_transaction decorator implicitly via calls to get_or_create_word_id.
    Ensures proper source attribution.

    Args:
        cur: Database cursor.
        entries: List of (lemma, language_code) tuples.
        source: Source information to add to the entries.
        batch_size: Number of entries to process in each batch (primarily for logging/progress).

    Returns:
        Dictionary mapping (lemma, language_code) to word_id.
    """
    result = {}
    processed_count = 0
    total_entries = len(entries)
    unique_entries = list(dict.fromkeys(entries)) # Process unique entries
    total_unique = len(unique_entries)
    logger.info(f"Batch getting/creating IDs for {total_unique} unique entries ({total_entries} total requested)...")
    
    # Track errors to report at the end
    error_count = 0
    error_types = {}

    # --- MODIFICATION: Use DictCursor for pre-fetch --- 
    from psycopg2.extras import DictCursor # Import DictCursor
    # Create a new cursor with DictCursor factory for this specific operation
    with cur.connection.cursor(cursor_factory=DictCursor) as dict_cur:
        # Pre-normalize lemmas and check for existing words in a larger batch if possible
        # This reduces individual lookups inside the loop
        normalized_map = { (lemma, lang): normalize_lemma(lemma) for lemma, lang in unique_entries }
        existing_words_cache = {}
        try:
            # Fetch existing words based on normalized lemma and language code
            if unique_entries:
                query_params = []
                placeholders = []
                for (lemma, lang) in unique_entries:
                    norm_lemma = normalized_map[(lemma, lang)]
                    query_params.extend([norm_lemma, lang])
                    placeholders.append("(%s, %s)")

                if placeholders:
                    fetch_query = f"""
                        SELECT id, lemma, language_code, normalized_lemma, source_info
                        FROM words
                        WHERE (normalized_lemma, language_code) IN ({', '.join(placeholders)})
                    """
                    # Use the DictCursor for execution
                    dict_cur.execute(fetch_query, query_params)
                    for row in dict_cur.fetchall():
                        # Cache by (original_lemma, lang_code) using the retrieved lemma for mapping back
                        # Note: This assumes the original lemma used for lookup is the same as stored lemma if normalized matches
                        # A more robust approach might cache by normalized_lemma+lang if conflicts are possible.
                        existing_words_cache[(row['lemma'], row['language_code'])] = {
                            'id': row['id'],
                            'source_info': row['source_info']
                        }
            logger.info(f"Cached {len(existing_words_cache)} existing words from the batch.")
        except Exception as e:
            logger.error(f"Error pre-fetching existing words: {e}", exc_info=True)
            # Proceed without cache, will rely on individual lookups
    # --- END MODIFICATION ---

    # Process entries, potentially using the cache
    # Continue using the original cursor (cur) for subsequent operations like get_or_create_word_id
    for lemma, lang_code in tqdm(unique_entries, desc="Batch Get/Create Words", unit="word"):
        processed_count += 1
        entry_key = (lemma, lang_code)
        # Create a savepoint for this entry
        savepoint_name = f"batch_word_{processed_count}"
        try:
            # Create a savepoint before each word processing
            cur.execute(f"SAVEPOINT {savepoint_name}")
            
            word_id = None
            # Check cache first
            if entry_key in existing_words_cache:
                word_id = existing_words_cache[entry_key]['id']
                existing_source_info = existing_words_cache[entry_key]['source_info']
                # Update source info if needed for existing cached word
                if source:
                    updated_source_json = update_word_source_info(existing_source_info, source)
                    # Check if source info actually changed to avoid unnecessary DB write
                    if updated_source_json != json.dumps(existing_source_info):
                        cur.execute("UPDATE words SET source_info = %s WHERE id = %s", (updated_source_json, word_id))
                        logger.debug(f"Updated source for cached word '{lemma}' (ID: {word_id})")

            # If not in cache, use the single get_or_create function
            else:
                word_id = get_or_create_word_id(cur, lemma, lang_code, source_identifier=source)

            if word_id is not None:
                result[entry_key] = word_id
                # Successfully processed this entry, release savepoint
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            else:
                logger.error(f"Failed to get or create ID for entry: ({lemma}, {lang_code}) from source '{source}'")
                # Rollback just this entry to keep the transaction valid
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                error_count += 1
                error_key = "WordLookupFailed"
                error_types[error_key] = error_types.get(error_key, 0) + 1

        except Exception as e:
            logger.error(f"Error processing batch entry ({lemma}, {lang_code}): {e}", exc_info=True)
            error_count += 1
            error_key = f"EntryError: {type(e).__name__}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
            # Try to rollback to the savepoint to maintain transaction validity
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                cur.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            except Exception as rb_error:
                logger.error(f"Failed to rollback to savepoint for {lemma}, {lang_code}: {rb_error}")
                # If we can't rollback to savepoint, the transaction might be in a bad state
                # but we'll continue anyway and let the caller decide what to do
                
            # Continue with the next entry in the batch

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors while batch processing. Error types: {error_types}")
    
    logger.info(f"Batch processing complete. Successfully processed {len(result)} unique entries.")
    return result

@with_transaction(commit=True)
def insert_word_entry(cur, entry: Dict, source_identifier: str) -> Dict:
    """Inserts a full word entry, including definitions, relations, etc.

    Args:
        cur: Database cursor.
        entry: Dictionary representing the word entry.
        source_identifier: Identifier for the data source.

    Returns:
        Dictionary containing statistics about the insertion process.
    """
    stats = {
        "words": 0,
        "definitions": 0,
        "relations": 0,
        "etymologies": 0,
        "pronunciations": 0,
        "examples": 0,
        "credits": 0,
        "skipped": 0,
        "errors": 0,
        "cross_refs": 0, # Added for tracking metadata refs
    }
    error_types = {}

    try:
        lemma = entry.get("word")
        if not lemma or not isinstance(lemma, str):
            logger.warning(f"Skipping entry due to missing or invalid 'word' field from source '{source_identifier}'. Entry: {entry}")
            stats["skipped"] += 1
            return stats

        # Basic cleaning - often done earlier, but good as a safeguard
        lemma = clean_html(lemma).strip()
        if not lemma:
            logger.warning(f"Skipping entry due to empty 'word' field after cleaning from source '{source_identifier}'. Entry: {entry}")
            stats["skipped"] += 1
            return stats

        language_code = entry.get("lang_code", DEFAULT_LANGUAGE_CODE)
        if not language_code:
            language_code = DEFAULT_LANGUAGE_CODE # Fallback

        # --- Get or Create Word ID --- (Using the enhanced function)
        # Pass the full entry data if available for potential use during creation
        word_id = get_or_create_word_id(cur, lemma, language_code, source_identifier)
        if word_id is None:
            logger.error(f"Failed to get or create word ID for '{lemma}' ({language_code}) from source '{source_identifier}'. Skipping entry.")
            stats["errors"] += 1
            error_key = f"WordIDError"
            error_types[error_key] = error_types.get(error_key, 0) + 1
            return {**stats, "error_details": error_types} # Return stats including errors
        stats["words"] += 1 # Count successfully retrieved/created words

        # --- Process Etymologies --- (Depends on insert_etymology)
        etymologies = entry.get("etymology_text", [])
        if isinstance(etymologies, str): # Handle single string case
            etymologies = [etymologies]
        if isinstance(etymologies, list):
            for etym_text_raw in etymologies:
                etym_text = clean_html(etym_text_raw).strip()
                if etym_text:
                    try:
                        # Check if insert_etymology function exists before calling
                        if 'insert_etymology' in globals() and callable(globals()['insert_etymology']):
                            etym_id = insert_etymology(cur, word_id, etym_text, source_identifier)
                            if etym_id:
                                stats["etymologies"] += 1
                        else:
                            logger.warning("insert_etymology function not found, skipping etymology insertion.")
                    except Exception as etym_err:
                        logger.error(f"Error inserting etymology for '{lemma}' (ID: {word_id}): {etym_err}", exc_info=True)
                        error_key = f"EtymologyInsertError: {type(etym_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

        # --- Process Pronunciations --- (Depends on insert_pronunciation)
        pronunciations = entry.get("sounds", [])
        if isinstance(pronunciations, list):
            for pron_data in pronunciations:
                if isinstance(pron_data, dict):
                    ipa = pron_data.get("ipa")
                    audio = pron_data.get("audio")
                    pron_metadata = {"note": clean_html(pron_data.get("note"))} # Example metadata
                    pron_tags = pron_data.get("tags")

                    # Check if insert_pronunciation exists before calling
                    if 'insert_pronunciation' in globals() and callable(globals()['insert_pronunciation']):
                        if ipa:
                            try:
                                pron_id_ipa = insert_pronunciation(cur, word_id, 'ipa', ipa, pron_tags, pron_metadata, source_identifier)
                                if pron_id_ipa: stats["pronunciations"] += 1
                            except Exception as pron_err:
                                logger.error(f"Error inserting IPA pronunciation for '{lemma}' (ID: {word_id}): {pron_err}", exc_info=True)
                                error_key = f"PronunciationInsertError: {type(pron_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1
                        if audio:
                            try:
                                pron_id_audio = insert_pronunciation(cur, word_id, 'audio', audio, pron_tags, pron_metadata, source_identifier)
                                if pron_id_audio: stats["pronunciations"] += 1
                            except Exception as pron_err:
                                 logger.error(f"Error inserting audio pronunciation for '{lemma}' (ID: {word_id}): {pron_err}", exc_info=True)
                                 error_key = f"PronunciationInsertError: {type(pron_err).__name__}"
                                 error_types[error_key] = error_types.get(error_key, 0) + 1
                    else:
                         logger.warning("insert_pronunciation function not found, skipping pronunciation insertion.")

        # --- Process Credits --- (Depends on insert_credit)
        credits_list = entry.get("credits", [])
        if isinstance(credits_list, list):
            for credit_item in credits_list:
                if credit_item: # Can be string or dict
                    try:
                         # Check if insert_credit exists before calling
                        if 'insert_credit' in globals() and callable(globals()['insert_credit']):
                            credit_id = insert_credit(cur, word_id, credit_item, source_identifier)
                            if credit_id: stats["credits"] += 1
                        else:
                             logger.warning("insert_credit function not found, skipping credit insertion.")
                    except Exception as credit_err:
                        logger.error(f"Error inserting credit for '{lemma}' (ID: {word_id}): {credit_err}", exc_info=True)
                        error_key = f"CreditInsertError: {type(credit_err).__name__}"
                        error_types[error_key] = error_types.get(error_key, 0) + 1

        # --- Process Word-Level Relations --- (Depends on insert_relation)
        # Example: Processing 'related', 'derived', etc., if present at the top level
        if 'insert_relation' in globals() and callable(globals()['insert_relation']):
            for rel_key in ["related", "synonyms", "antonyms", "derived_from", "hypernyms", "hyponyms", "cognate_of", "doublet_of"]:
                rel_items = entry.get(rel_key, [])
                if isinstance(rel_items, list):
                    rel_type_enum = RelationshipType.from_string(rel_key) # Get enum from string
                    for item in rel_items:
                        target_lemma = None
                        if isinstance(item, str):
                            target_lemma = item
                        elif isinstance(item, dict) and ('word' in item or 'term' in item):
                            target_lemma = item.get('word') or item.get('term')

                        target_lemma_clean = clean_html(target_lemma).strip()
                        if target_lemma_clean and target_lemma_clean.lower() != lemma.lower():
                            try:
                                target_id = get_or_create_word_id(cur, target_lemma_clean, language_code, source_identifier)
                                if target_id and target_id != word_id:
                                    rel_id = insert_relation(cur, word_id, target_id, rel_type_enum.rel_value, source_identifier)
                                    if rel_id: stats["relations"] += 1
                            except Exception as rel_err:
                                logger.error(f"Error inserting relation ({rel_key}) for '{lemma}' -> '{target_lemma_clean}': {rel_err}", exc_info=True)
                                error_key = f"RelationInsertError: {type(rel_err).__name__}"
                                error_types[error_key] = error_types.get(error_key, 0) + 1
        else:
            logger.warning("insert_relation function not found, skipping word-level relation insertion.")


        # --- Process Definitions (Senses) --- (Depends on insert_definition and insert_definition_example)
        senses = entry.get("senses", [])
        if isinstance(senses, list):
            for sense_idx, sense in enumerate(senses):
                if not isinstance(sense, dict): continue

                raw_pos = sense.get("pos") # Get raw POS string from sense
                # Handle multiple glosses - taking the first non-empty one
                glosses = sense.get("glosses", [])
                definition_text = ""
                if isinstance(glosses, list):
                    for gloss in glosses:
                        cleaned_gloss = clean_html(gloss).strip()
                        if cleaned_gloss:
                            definition_text = cleaned_gloss
                            break
                if not definition_text:
                    continue # Skip sense if no valid gloss found

                categories = sense.get("categories", []) # Often contains usage tags
                examples = sense.get("examples", [])
                sense_tags = sense.get("tags", []) # Other tags specific to this sense
                raw_glosses = sense.get("raw_glosses", []) # Can be used for usage notes
                usage_notes = clean_html(raw_glosses[0] if raw_glosses else "")

                # Combine categories and sense_tags into a single list for simplicity for insert_definition
                all_tags = list(set([t for t in categories + sense_tags if isinstance(t, str) and t.strip()]))

                definition_id = None
                try:
                    # Check if insert_definition exists
                    if 'insert_definition' in globals() and callable(globals()['insert_definition']):
                        def_id = insert_definition(
                            cur,
                            word_id=word_id,
                            definition_text=definition_text,
                            part_of_speech=raw_pos, # Pass raw POS
                            usage_notes=usage_notes,
                            category=None, # Use tags parameter instead
                            sources=source_identifier,
                            tags=all_tags # Pass combined tags
                        )
                        if def_id:
                            stats["definitions"] += 1
                            definition_id = def_id
                        else:
                            logger.warning(f"insert_definition failed for '{lemma}', POS '{raw_pos}', Sense {sense_idx+1}")
                            error_key = f"DefinitionInsertFailure"
                            error_types[error_key] = error_types.get(error_key, 0) + 1
                    else:
                         logger.warning("insert_definition function not found, skipping definition insertion.")
                         continue # Skip examples if definition failed/skipped

                except Exception as def_err:
                    logger.error(f"Error inserting definition for '{lemma}', POS '{raw_pos}', Sense {sense_idx+1}: {def_err}", exc_info=True)
                    error_key = f"DefinitionInsertError: {type(def_err).__name__}"
                    error_types[error_key] = error_types.get(error_key, 0) + 1
                    continue # Skip examples if definition failed

                # Process examples for this definition (if definition was inserted)
                if definition_id:
                    if 'insert_definition_example' in globals() and callable(globals()['insert_definition_example']):
                        if isinstance(examples, list):
                            for ex_data in examples:
                                if isinstance(ex_data, dict) and ex_data.get("text"):
                                    try:
                                        ex_id = insert_definition_example(cur, definition_id, ex_data, source_identifier)
                                        if ex_id: stats["examples"] += 1
                                    except Exception as ex_err:
                                        logger.error(f"Error inserting example for def ID {definition_id}: {ex_err}", exc_info=True)
                                        error_key = f"ExampleInsertError: {type(ex_err).__name__}"
                                        error_types[error_key] = error_types.get(error_key, 0) + 1
                    else:
                        logger.warning("insert_definition_example function not found, skipping example insertion.")

    except Exception as e:
        logger.critical(f"CRITICAL error processing entry for '{entry.get('word', 'UNKNOWN')}' from source '{source_identifier}': {e}", exc_info=True)
        stats["errors"] += 1
        error_key = f"TopLevelError: {type(e).__name__}"
        error_types[error_key] = error_types.get(error_key, 0) + 1
        # Depending on strategy, might want to rollback explicitly here if not using @with_transaction
        # try: cur.connection.rollback()
        # except Exception: pass

    # Add error details to stats before returning
    if error_types:
        stats["error_details"] = error_types

    return stats

# --- Add other functions from backup below ---

@with_transaction(commit=False) # Read-only, so no commit needed
def get_word_details(cur, word_id: int) -> Optional[Dict]:
    """Fetch comprehensive details for a word ID."""
    try:
        # Fetch word details
        cur.execute("SELECT * FROM words WHERE id = %s", (word_id,))
        word_data = cur.fetchone()
        if not word_data:
            return None

        word_details = dict(word_data)

        # Fetch definitions
        cur.execute(
            """
            SELECT d.*, p.code as pos_code
            FROM definitions d
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE d.word_id = %s
            ORDER BY d.id ASC
            """,
            (word_id,)
        )
        word_details['definitions'] = [dict(row) for row in cur.fetchall()]

        # Fetch etymologies
        cur.execute("SELECT * FROM etymologies WHERE word_id = %s", (word_id,))
        word_details['etymologies'] = [dict(row) for row in cur.fetchall()]

        # Fetch pronunciations
        cur.execute("SELECT * FROM pronunciations WHERE word_id = %s", (word_id,))
        word_details['pronunciations'] = [dict(row) for row in cur.fetchall()]

        # Fetch credits
        cur.execute("SELECT * FROM credits WHERE word_id = %s", (word_id,))
        word_details['credits'] = [dict(row) for row in cur.fetchall()]

        # Fetch outgoing relations
        cur.execute(
            """
            SELECT r.*, w.lemma as to_lemma
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = %s
            """,
            (word_id,)
        )
        word_details['outgoing_relations'] = [dict(row) for row in cur.fetchall()]

        # Fetch incoming relations
        cur.execute(
            """
            SELECT r.*, w.lemma as from_lemma
            FROM relations r
            JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = %s
            """,
            (word_id,)
        )
        word_details['incoming_relations'] = [dict(row) for row in cur.fetchall()]

        # --- Add fetching for NEW tables ---

        # Fetch word forms
        cur.execute("SELECT * FROM word_forms WHERE word_id = %s ORDER BY is_canonical DESC, id ASC", (word_id,))
        word_details['forms'] = [dict(row) for row in cur.fetchall()]

        # Fetch definition examples (nested under definitions)
        for definition in word_details.get('definitions', []):
            cur.execute("SELECT * FROM definition_examples WHERE definition_id = %s ORDER BY id ASC", (definition['id'],))
            definition['examples'] = [dict(row) for row in cur.fetchall()]

        # Fetch definition categories (nested under definitions)
        for definition in word_details.get('definitions', []):
            cur.execute("SELECT * FROM definition_categories WHERE definition_id = %s ORDER BY id ASC", (definition['id'],))
            definition['categories'] = [dict(row) for row in cur.fetchall()]

        # Fetch definition links (nested under definitions)
        for definition in word_details.get('definitions', []):
            cur.execute("SELECT * FROM definition_links WHERE definition_id = %s ORDER BY id ASC", (definition['id'],))
            definition['links'] = [dict(row) for row in cur.fetchall()]

        # Fetch affixations (where this word is the root)
        cur.execute("SELECT af.*, w.lemma as affixed_lemma FROM affixations af JOIN words w ON af.affixed_word_id = w.id WHERE af.root_word_id = %s", (word_id,))
        word_details['root_affixations'] = [dict(row) for row in cur.fetchall()]

        # Fetch affixations (where this word is affixed)
        cur.execute("SELECT af.*, w.lemma as root_lemma FROM affixations af JOIN words w ON af.root_word_id = w.id WHERE af.affixed_word_id = %s", (word_id,))
        word_details['affixed_affixations'] = [dict(row) for row in cur.fetchall()]

        # Fetch word templates
        cur.execute("SELECT * FROM word_templates WHERE word_id = %s ORDER BY id ASC", (word_id,))
        word_details['templates'] = [dict(row) for row in cur.fetchall()]

        return word_details

    except Exception as e:
        logger.error(f"Error fetching details for word ID {word_id}: {e}", exc_info=True)
        return None

# --- Add other functions from backup below ---

def verify_database_schema(conn):
    """Verify the database schema matches expected columns."""
    logger.info("Verifying database schema...")
    cur = conn.cursor()
    # Define the complete expected schema based on TABLE_CREATION_SQL
    expected_schema = {
        "parts_of_speech": {
            "id", "code", "name_en", "name_tl", "description",
            "created_at", "updated_at"
        },
        "words": {
            "id", "lemma", "normalized_lemma", "has_baybayin", "baybayin_form",
            "romanized_form", "language_code", "root_word_id", "preferred_spelling",
            "tags", "idioms", "pronunciation_data", "source_info", "word_metadata",
            "data_hash", "badlit_form", "hyphenation", "is_proper_noun",
            "is_abbreviation", "is_initialism", "search_text",
            "created_at", "updated_at"
        },
        "pronunciations": {
            "id", "word_id", "type", "value", "tags",
            "pronunciation_metadata", # 'sources' column removed, now in metadata
            "created_at", "updated_at"
        },
        "credits": {
            "id", "word_id", "credit", "sources", "created_at", "updated_at"
        },
        "definitions": {
            "id", "word_id", "definition_text", "original_pos", "standardized_pos_id",
            "examples", "usage_notes", "tags", "sources", "definition_metadata",
            "created_at", "updated_at"
        },
        "relations": {
            "id", "from_word_id", "to_word_id", "relation_type", "sources",
            "metadata", "created_at", "updated_at"
        },
        "etymologies": {
            "id", "word_id", "etymology_text", "normalized_components",
            "etymology_structure", "language_codes", "sources",
            "created_at", "updated_at"
        },
        "definition_links": {
            "id", "definition_id", "link_text", "tags", "link_metadata",
            "sources", "created_at", "updated_at"
        },
        "definition_relations": {
            "id", "definition_id", "word_id", "relation_type", "sources",
            "created_at", "updated_at"
        },
        "affixations": {
             "id", "root_word_id", "affixed_word_id", "affix_type", "sources",
             "created_at", "updated_at"
        },
        "definition_categories": {
           "id", "definition_id", "category_name", "category_kind", "parents",
           "sources", "category_metadata", # Added category_metadata
           "created_at", "updated_at"
        },
        "definition_examples": {
            "id", "definition_id", "example_text", "translation", "example_type",
            "reference", "metadata", "sources", "created_at", "updated_at"
        },
        "word_templates": {
            "id", "word_id", "template_name", "args", "expansion",
            "sources", # Added sources column based on TABLE_CREATION_SQL
            "created_at", "updated_at"
        },
        "word_forms": {
           "id", "word_id", "form", "is_canonical", "is_primary", "tags",
           "created_at", "updated_at"
        }
    }
    issues_found = False

    try:
        for table, expected_cols in expected_schema.items():
            logger.debug(f"Verifying table: {table}")
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (table,)
            )
            actual_cols = {row[0] for row in cur.fetchall()}

            missing_cols = expected_cols - actual_cols
            extra_cols = actual_cols - expected_cols

            if missing_cols:
                logger.error(f"Table '{table}' is missing expected columns: {missing_cols}")
                issues_found = True
            if extra_cols:
                logger.warning(f"Table '{table}' has extra columns: {extra_cols}")

        if issues_found:
            logger.error("Database schema verification failed. Please run migration or update schema.")
            return False
        else:
            logger.info("Database schema verification successful.")
            return True
    except Exception as e:
        logger.error(f"Error during schema verification: {e}", exc_info=True)
        return False
    finally:
        cur.close()

# --- Add other functions from backup below ---

@with_transaction(commit=True)
def insert_definition_category(
    cur: psycopg2.extensions.cursor,
    definition_id: int,
    category_data: Union[str, Dict],
    source_identifier: str,
) -> Optional[int]:
    """
    Upserts a category link for a definition.
    Handles string category names or dictionary structures.
    Merges metadata and appends unique sources on conflict.
    """
    if not source_identifier:
        logger.warning("insert_definition_category skipped: source_identifier missing.")
        return None

    cat_name: Optional[str] = None
    cat_kind: Optional[str] = None
    cat_parents: Optional[List[str]] = None
    cat_source: Optional[str] = None
    cat_metadata: Optional[Dict] = None

    # --- Type Handling & Extraction ---
    if isinstance(category_data, str):
        cat_name = category_data.strip()
        cat_metadata = {} # Initialize empty metadata for string input
    elif isinstance(category_data, dict):
        cat_name = category_data.get("name")
        cat_kind = category_data.get("kind")
        cat_parents = category_data.get("parents") # Should be a list or None
        cat_source = category_data.get("source")
        cat_metadata = category_data.copy() # Work on a copy
        # Remove core fields from metadata if they exist
        cat_metadata.pop("name", None)
        cat_metadata.pop("kind", None)
        cat_metadata.pop("parents", None)
        cat_metadata.pop("source", None)
    else:
        logger.warning(f"Invalid category_data type for Def ID {definition_id}: {type(category_data)}. Skipping category.")
        # --- FIX Indentation START --- #
        return None # Align this with the logger.warning above
        # --- FIX Indentation END --- #

    if not cat_name:
        logger.warning(f"Category name missing for Def ID {definition_id}. Skipping category. Data: {category_data}")

    # --- Validation & Preparation ---
    if cat_parents is not None and not isinstance(cat_parents, list):
        logger.warning(f"Category 'parents' for '{cat_name}' (Def ID {definition_id}) is not a list: {cat_parents}. Converting to list.")
        # Attempt conversion, default to empty list on failure
        try:
            cat_parents = list(cat_parents) if hasattr(cat_parents, '__iter__') and not isinstance(cat_parents, str) else [str(cat_parents)]
        except Exception:
            logger.error(f"Could not convert category parents to list for '{cat_name}'. Setting to empty.", exc_info=True)
            cat_parents = []

    sources_str = source_identifier.strip()

    # Ensure metadata is a dict
    if cat_metadata is None:
        cat_metadata = {}
    elif not isinstance(cat_metadata, dict):
        logger.warning(f"Category metadata for '{cat_name}' is not a dict: {cat_metadata}. Resetting to empty dict.")
        cat_metadata = {}

    # Add original source from category data to metadata if different
    if cat_source and cat_source != source_identifier:
        if 'original_sources' not in cat_metadata:
            cat_metadata['original_sources'] = []
        # Ensure original_sources is a list before appending
        if isinstance(cat_metadata.get('original_sources'), list):
            if cat_source not in cat_metadata['original_sources']:
                 cat_metadata['original_sources'].append(cat_source)
    else:
            logger.warning(f"'original_sources' in metadata for '{cat_name}' was not a list. Overwriting with new source.")
            cat_metadata['original_sources'] = [cat_source]

    # --- Parameter Assembly (Using Json Adapter) ---
    params = {
        "def_id": definition_id,
        "name": cat_name,
        "kind": cat_kind, # Let DB handle NULL if None
        "parents": Json(cat_parents or []), # Use Json adapter for list -> JSONB
        "sources": sources_str,
        "metadata": Json(cat_metadata or {}) # Use Json adapter for dict -> JSONB
    }

    # --- Detailed Type Logging --- #
    param_types = {k: type(v).__name__ for k, v in params.items()}
    logger.debug(f"Parameter types for insert_definition_category: {param_types}")
    logger.debug(f"Parameter values (metadata truncated): def_id={params['def_id']}, name='{params['name']}', kind='{params['kind']}', sources='{params['sources']}'")
    # Avoid logging potentially large JSON objects directly unless necessary
    # logger.debug(f"Full Params: {params}")

    # --- SQL Query (No ::jsonb casts in VALUES) ---
    upsert_sql = r"""
        INSERT INTO definition_categories (
            definition_id, category_name, category_kind, parents, sources, category_metadata
        )
        VALUES (
            %(def_id)s, %(name)s, %(kind)s, %(parents)s, %(sources)s, %(metadata)s
        )
            ON CONFLICT (definition_id, category_name)
            DO UPDATE SET
            -- *** FIX: Use correct column name from EXCLUDED ***
            category_kind = COALESCE(EXCLUDED.category_kind, definition_categories.category_kind),
                parents = CASE
                              WHEN EXCLUDED.parents IS NOT NULL THEN EXCLUDED.parents
                              ELSE definition_categories.parents
                          END,
            sources = CASE
                        WHEN definition_categories.sources IS NULL
                             OR definition_categories.sources = '' THEN EXCLUDED.sources
                        -- Check if the source already exists as a whole word/item
                        WHEN string_to_array(definition_categories.sources, '; ') @> ARRAY[EXCLUDED.sources] THEN
                             definition_categories.sources
                        ELSE definition_categories.sources || '; ' || EXCLUDED.sources
                      END,
            category_metadata =
                   COALESCE(definition_categories.category_metadata,'{}'::jsonb)
                   || COALESCE(EXCLUDED.category_metadata,'{}'::jsonb),
                updated_at = CURRENT_TIMESTAMP
        RETURNING id;
    """

    # --- Execution --- ---
    try:
        cur.execute(upsert_sql, params)
        cat_id_tuple = cur.fetchone()
        if cat_id_tuple:
            cat_id = cat_id_tuple[0]
            logger.info(f"Inserted/Updated definition category (ID: {cat_id}) for def ID {definition_id} from source '{source_identifier}'. Name: '{cat_name}'")
            return cat_id
        else:
            logger.warning(f"Upsert for definition category '{cat_name}' (Def ID {definition_id}) did not return an ID.")
            return None # Should not happen with RETURNING id, but safety check

    except psycopg2.Error as pg_err:
        pg_error_msg = str(pg_err.pgerror).strip() if pg_err.pgerror else str(pg_err).strip()
        # Log the formatted message and include SQL/params for context
        log_message = (
            f"definition_category PG-error [{pg_err.pgcode or 'No Code'}]: {pg_error_msg}\n\t" + 
            f"SQL: {upsert_sql}\n\t" +
            f"Params: {params}" # Show the params dict that caused the error
        )
        logger.error(log_message, exc_info=True)

    except Exception as exc:
        logger.exception(
            f"definition_category unexpected error for def ID {definition_id}, cat '{cat_name}': {exc}"
        )

    return None # Return None on any error

@with_transaction(commit=True)
def insert_definition_link(
    cur, definition_id: int, link_data: Union[str, Dict, List], source_identifier: str
) -> Optional[int]: # Allow List input
    """
    Insert a link associated with a definition. Handles string, dictionary, or list input.
    If input is a list, processes each item recursively.
    Uses ON CONFLICT to update existing links based on (definition_id, link_text).
    Stores source information within the 'link_metadata' JSONB column.

    Args:
        cur: Database cursor.
        definition_id: ID of the definition this link belongs to.
        link_data: Link string, dictionary (e.g., {"text": "...", "tags": [...], "metadata": {...}}),
                   or a list containing such items.
                   Expected keys in dict: 'text' (required), fallback '1', fallback 'link', 'tags', 'metadata'.
                   Other keys will be added to metadata.
        source_identifier: Identifier for the data source (e.g., filename). MANDATORY.

    Returns:
        The ID of the inserted/updated definition link record (or the ID of the last processed item if input is a list),
        or None if failed or skipped.
    """
    if not source_identifier:
        logger.error(
            f"CRITICAL: Skipping definition link insert for def ID {definition_id}: Missing MANDATORY source identifier."
        )
        return None

    # --- Handle List Input Recursively ---
    if isinstance(link_data, list):
        logger.debug(f"Processing list of links for def ID {definition_id}. Count: {len(link_data)}")
        last_inserted_id = None
        for index, item in enumerate(link_data):
            try:
                inserted_id = insert_definition_link(cur, definition_id, item, source_identifier)
                if inserted_id is not None:
                    last_inserted_id = inserted_id
            except Exception as list_item_error:
                logger.error(f"Error processing item at index {index} in links list for def ID {definition_id}: {list_item_error}", exc_info=True)
        return last_inserted_id # Return the ID of the last successfully processed link in the list

    # --- Process Single Link (String or Dict) ---
    link_text = None
    tags_list = []
    link_metadata = {}
    # source_identifier will be added to link_metadata

    # --- Data Extraction ---
    if isinstance(link_data, str):
        link_text = link_data.strip()
        # Initialize metadata with the source
        link_metadata = {"sources": [source_identifier]}  # Store source in metadata
        logger.debug(
            f"Processing definition link (string) for def ID {definition_id}: '{link_text}'"
        )
    elif isinstance(link_data, dict):
        # Extract required link text with fallbacks
        link_text = link_data.get("text")
        if not link_text or not isinstance(link_text, str):
            link_text = link_data.get("1") # Fallback 1: '1'
        if not link_text or not isinstance(link_text, str):
            link_text = link_data.get("link") # Fallback 2: 'link'

        if not link_text or not isinstance(link_text, str):
                logger.warning(
                    f"Definition link dict for def ID {definition_id} (source '{source_identifier}') missing required 'text' (or '1' or 'link') key or not a string. Skipping. Data: {link_data}"
                )
                return None
        link_text = link_text.strip()

        # Extract optional tags (store as comma-sep string in DB tags column)
        tags_input = link_data.get("tags")
        if isinstance(tags_input, list):
            tags_list = [
                str(t).strip()
                for t in tags_input
                if t and isinstance(t, (str, int, float))
            ]
        elif isinstance(tags_input, str) and tags_input.strip():
            tags_list = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        # Extract optional metadata (assume it's JSON serializable)
        meta_input = link_data.get("metadata")
        if isinstance(meta_input, dict):
            link_metadata = meta_input.copy()  # Copy to avoid modifying original
        elif meta_input:
             logger.warning(f"Metadata for link '{link_text}' (Def ID {definition_id}) is not a dict: {meta_input}. Ignoring.")

        # Capture other potential fields from link_data into metadata, excluding known ones
        standard_keys = {"text", "1", "link", "tags", "sources", "metadata"}
        for k, v in link_data.items():
            if (
                k not in standard_keys and k not in link_metadata
            ):  # Avoid overwriting explicit metadata
                # Basic check for serializability
                try:
                    json.dumps({f"extra_{k}": v})
                    link_metadata[f"extra_{k}"] = v
                except TypeError:
                    logger.warning(
                        f"Could not serialize extra link data key '{k}' for def ID {definition_id}. Skipping field."
                    )

        # Add/Update source information in metadata
        # Ensure 'sources' in metadata is a list and append the new source_identifier if not present
        existing_sources = link_metadata.get("sources", [])
        if not isinstance(existing_sources, list):
            # Attempt to handle non-list existing data gracefully
            logger.warning(
                f"Existing 'sources' in metadata for link '{link_text}' (Def ID {definition_id}) is not a list: {existing_sources}. Overwriting with new source."
            )
            existing_sources = []
        if source_identifier not in existing_sources:
            existing_sources.append(source_identifier)
        link_metadata["sources"] = (
            existing_sources  # Update metadata with the potentially modified list
        )

        logger.debug(
            f"Processing definition link (dict) for def ID {definition_id}: '{link_text}'"
        )

    else:
        # Handle unexpected types (neither list, str, nor dict)
        logger.warning(
            f"Invalid link_data type for def ID {definition_id}: {type(link_data)}. Skipping. Data: {link_data}"
        )
        return None

    # --- Validation ---
    if not link_text:
        logger.warning(
            f"Empty link text after processing for definition ID {definition_id} (source '{source_identifier}'). Skipping link insertion."
        )
        return None

    # --- Data Preparation for DB ---
    tags_string = ",".join(tags_list) if tags_list else None
    metadata_json = None
    try:
        if link_metadata: # Only serialize if not empty
             metadata_json = json.dumps(link_metadata, default=str)
    except TypeError as e:
        logger.error(
            f"Could not serialize metadata for link '{link_text}' (Def ID {definition_id}): {e}. Metadata: {link_metadata}",
            exc_info=True,
        )
        return None  # Skip if metadata fails serialization

    # --- Database Operation ---
    try:
        cur.execute(
            """
            INSERT INTO definition_links (definition_id, link_text, tags, link_metadata)
            VALUES (%(def_id)s, %(link_text)s, %(tags)s, %(metadata)s::jsonb)
            ON CONFLICT (definition_id, link_text)
            DO UPDATE SET
                tags = CASE
                           WHEN EXCLUDED.tags IS NOT NULL THEN EXCLUDED.tags
                           ELSE definition_links.tags
                       END,
                -- Merge metadata: Prioritize new keys, keep old ones if not overwritten.
                -- The || operator merges JSONB objects, overwriting keys from left with right.
                -- For the 'sources' list specifically, the logic above ensures it's appended correctly before serialization.
                link_metadata = definition_links.link_metadata || EXCLUDED.link_metadata,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            {
                "def_id": definition_id,
                "link_text": link_text,
                "tags": tags_string,  # Pass comma-separated string or None
                "metadata": metadata_json,  # Pass JSON string or None
            },
        )
        link_id_tuple = cur.fetchone()
        if link_id_tuple:
            link_id = link_id_tuple[0]
            logger.info(  # Changed to info
                f"Inserted/Updated definition link (ID: {link_id}) for def ID {definition_id} from source '{source_identifier}'. Link: '{link_text}'"
            )
            return link_id
        else:
            logger.error(
                f"Failed to get definition link ID after upsert for def ID {definition_id}, link: '{link_text}'."
            )
            cur.connection.rollback()
            return None

    except psycopg2.IntegrityError as e:
        logger.error(
            f"Database IntegrityError inserting link for def ID {definition_id} ('{link_text}') from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(
                f"Rollback failed after IntegrityError inserting link: {rb_e}"
            )
        return None
    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting link for def ID {definition_id} ('{link_text}') from '{source_identifier}': {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after DB error inserting link: {rb_e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting link for def ID {definition_id} ('{link_text}') from '{source_identifier}': {e}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.error(
                f"Failed to rollback transaction after unexpected error inserting link: {rb_e}"
            )
        return None

@with_transaction(commit=True)  # Assume same transaction decorator pattern
def insert_word_form(
    cur,
    word_id: int,
    form: str,
    metadata: Optional[Dict] = None,
    source_identifier: Optional[str] = None,  # Optional source tracking
) -> Optional[int]:
    """
    Insert or update a word form record.

    Stores the provided metadata dictionary (which includes original tags, source, etc.)
    into the 'tags' JSONB column. Extracts 'is_canonical' from metadata for its
    dedicated column.

    Uses ON CONFLICT (word_id, form) DO UPDATE to handle existing entries.

    Args:
        cur: Database cursor.
        word_id: ID of the associated word.
        form: The specific word form string.
        metadata: Dictionary containing all metadata related to this form
                  (e.g., {'tags': [...], 'source': '...', 'is_canonical': True}).
        source_identifier: Optional identifier for the data source. (Not currently used in SQL)

    Returns:
        The ID of the inserted/updated word form record, or None if failure.
    """
    if not form or not isinstance(form, str):
        logger.warning(
            f"Skipping word form insertion for word ID {word_id}: Invalid or empty form provided."
        )
        return None
    if word_id is None:
        logger.warning(
            f"Skipping word form insertion for form '{form}': Invalid word_id (None)."
        )
        return None

    form_text = form.strip()
    if not form_text:
        logger.warning(
            f"Skipping word form insertion for word ID {word_id}: Form is empty after stripping."
        )
        return None

    # Prepare metadata for JSONB storage, default to empty dict
    form_metadata = metadata if isinstance(metadata, dict) else {}

    # Extract is_canonical specifically for its own column
    is_canonical = form_metadata.get("is_canonical", False)
    if not isinstance(is_canonical, bool):
        logger.warning(
            f"Invalid type for is_canonical ({type(is_canonical)}) in metadata for form '{form_text}' (Word ID: {word_id}). Defaulting to False."
        )
        is_canonical = False

    # Note: is_primary is not typically in Kaikki form data, defaulting to False in the model.

    metadata_json = None
    try:
        # Serialize the entire metadata dictionary for the 'tags' JSONB column
        metadata_json = json.dumps(
            form_metadata, default=str
        )  # Use default=str for safety
    except TypeError as e:
        logger.error(
            f"Could not serialize metadata for word form '{form_text}' (Word ID {word_id}): {e}. Metadata: {form_metadata}",
            exc_info=True,
        )
        return None  # Skip if metadata can't be serialized

    params = {
        "word_id": word_id,
        "form": form_text,
        "is_canonical": is_canonical,
        "tags_metadata": metadata_json,  # Serialized metadata for the 'tags' column
    }

    try:
        # Assumes 'tags' column is the target for the metadata JSONB.
        # Assumes NO 'sources' column in word_forms table.
        cur.execute(
            """
            INSERT INTO word_forms (word_id, form, is_canonical, tags)
            VALUES (
                %(word_id)s,
                %(form)s,
                %(is_canonical)s,
                CASE WHEN %(tags_metadata)s IS NOT NULL THEN %(tags_metadata)s::jsonb ELSE NULL END
            )
            ON CONFLICT (word_id, form) DO UPDATE
            SET
                is_canonical = EXCLUDED.is_canonical, -- Update based on incoming data
                tags = CASE WHEN EXCLUDED.tags IS NOT NULL THEN EXCLUDED.tags ELSE word_forms.tags END,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            params,
        )
        form_id_tuple = cur.fetchone()
        if form_id_tuple:
            form_id = form_id_tuple[0]
            logger.debug(
                f"Inserted/Updated word form (ID: {form_id}, Form: '{form_text}', Canonical: {is_canonical}) for word ID {word_id}."
            )
            return form_id
        else:
            logger.error(
                f"Failed to get form ID after upsert for word ID {word_id}, Form: '{form_text}'."
            )
            try:
                cur.connection.rollback()
            except Exception:
                pass
            return None

    except psycopg2.Error as e:
        logger.error(
            f"Database error inserting word form '{form_text}' for word ID {word_id}: {e.pgcode} {e.pgerror}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.warning(f"Rollback failed after DB error: {rb_e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error inserting word form '{form_text}' for word ID {word_id}: {e}",
            exc_info=True,
        )
        try:
            cur.connection.rollback()
        except Exception as rb_e:
            logger.error(
                f"Failed to rollback transaction after unexpected error: {rb_e}"
            )
        return None

@with_transaction(commit=True)  # Assume same transaction decorator pattern
def insert_word_template(
    cur,
    word_id: int,
    template_name: str,
    args: Optional[Dict] = None,
    expansion: Optional[str] = None,  # Added expansion based on model
    source_identifier: Optional[str] = None,  # This name is used by the CALLER
) -> Optional[int]:
    """
    Inserts or updates a word template entry associated with a word_id.
    Handles JSONB conversion for arguments.
    Uses 'sources' column in the table.

    Args:
        cur: Database cursor.
        word_id: The ID of the word this template belongs to.
        template_name: The name of the template (e.g., \'tl-infl-noun\').
        args: A dictionary of template arguments (will be stored as JSONB).
        expansion: The expanded form of the template (e.g., the resulting word).
        source_identifier: The source identifier for tracking data origin (mapped to 'sources' column).

    Returns:
        The ID of the inserted or updated template entry, or None on error.
    """
    if not all([word_id, template_name]):
        logger.warning(
            "Skipping insert_word_template: Missing required word_id or template_name."
        )
        return None

    # --- Database Operation ---
    try:
        # Ensure the SQL uses %(...)s placeholders for dictionary substitution
        # and references the correct 'sources' column.
        insert_query = """
            INSERT INTO word_templates (word_id, template_name, args, expansion, sources)
            VALUES (%(word_id)s, %(template_name)s, %(args)s, %(expansion)s, %(sources)s)
            ON CONFLICT (word_id, template_name) DO UPDATE SET
                args = EXCLUDED.args,
                expansion = EXCLUDED.expansion,
                sources = EXCLUDED.sources,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """
        # Prepare arguments dictionary, mapping the input parameter name
        # 'source_identifier' to the dictionary key 'sources' which matches the SQL placeholder.
        args_json = Json(args) if args is not None else None
        params_dict = {
            'word_id': word_id,
            'template_name': template_name.strip(), # Ensure name is stripped
            'args': args_json,
            'expansion': expansion.strip() if expansion else None, # Ensure expansion is stripped
            'sources': source_identifier # Map input parameter to the 'sources' key
        }

        # Execute the insert/update query using the dictionary
        cur.execute(insert_query, params_dict)
        result = cur.fetchone()

        if result:
            template_id = result[0]
            # Corrected log message format
            logger.info(f"Inserted/Updated word template ID {template_id} for word {word_id}.")
            return template_id
        else:
            logger.error(f"Failed to get template ID for word {word_id} and template {template_name}.")
            return None
    except Exception as e:
        logger.error(f"Error inserting/updating word template for word_id {word_id}, template {template_name}: {e}", exc_info=True)
        raise  # Re-raise the exception after logging

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

def check_tables_exist(conn):
    """Check if the core tables have been created.
    
    Args:
        conn: Database connection
        
    Returns:
        bool: True if all required tables exist, False otherwise
    """
    required_tables = ["words", "definitions", "relations"]
    try:
        cursor = conn.cursor()
        for table in required_tables:
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table}'
                )
            """)
            exists = cursor.fetchone()[0]
            if not exists:
                logger.error(f"Required table '{table}' does not exist")
                cursor.close()
                return False
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return False

def check_query_performance(conn):
    """Check the performance of various queries.
    
    Args:
        conn: Database connection
        
    Returns:
        dict: Dictionary of query types and their duration in ms
    """
    results = {}
    try:
        cursor = conn.cursor()
        
        # Test 1: Simple select count
        start_time = datetime.now()
        cursor.execute("SELECT count(*) FROM words")
        word_count = cursor.fetchone()[0]
        duration = (datetime.now() - start_time).total_seconds() * 1000
        results["simple_count"] = duration
        
        # Test 2: Join query count
        start_time = datetime.now()
        cursor.execute("""
            SELECT count(*) 
            FROM words w
            JOIN definitions d ON w.id = d.word_id
        """)
        join_count = cursor.fetchone()[0]
        duration = (datetime.now() - start_time).total_seconds() * 1000
        results["join_count"] = duration
        
        # Test 3: Index usage
        start_time = datetime.now()
        cursor.execute("""
            SELECT count(*) 
            FROM words 
            WHERE lemma = 'test'
        """)
        indexed_count = cursor.fetchone()[0]
        duration = (datetime.now() - start_time).total_seconds() * 1000
        results["indexed_query"] = duration
        
        cursor.close()
        return results
    except Exception as e:
        logger.error(f"Error checking query performance: {e}")
        if 'cursor' in locals():
            cursor.close()
        return {"error": str(e)}

@with_transaction(commit=True)
def purge_database_tables():
    """Safely delete all data from the dictionary tables.
    
    This function deletes data from the following tables in order:
    - definition_relations
    - affixations
    - relations
    - etymologies
    - definitions
    - words
    - parts_of_speech
    
    Returns:
        tuple: (success, message)
    """
    try:
        cur = get_cursor()
        # Order matters due to foreign key constraints
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
            cur.execute(f"DELETE FROM {table}")
            count = cur.rowcount
            logger.info(f"Deleted {count} rows from {table}")
        
        conn = cur.connection
        cur.close()
        return True, "All dictionary data has been purged"
    except Exception as e:
        logger.error(f"Error purging database tables: {e}")
        return False, str(e)
