"""
Database utilities for managing connections and database operations.
Provides connection pooling, retry logic, monitoring, and advanced functionality.
"""

from sqlalchemy import create_engine, text, func, MetaData
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.orm import scoped_session, sessionmaker
import psycopg2
from psycopg2.extensions import (
    ISOLATION_LEVEL_AUTOCOMMIT,
    ISOLATION_LEVEL_READ_COMMITTED,
    ISOLATION_LEVEL_REPEATABLE_READ
)
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import DictCursor, execute_batch
import time
from contextlib import contextmanager
from typing import Optional, Generator, Dict, Any, List, Tuple, Callable
from prometheus_client import Counter, Histogram, Gauge, Summary
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import traceback
import structlog
from functools import wraps
import json
from datetime import datetime, timedelta
from sqlalchemy.ext.declarative import declarative_base
import redis
from flask_sqlalchemy import SQLAlchemy

# Set up structured logging
logger = structlog.get_logger(__name__)

load_dotenv()

# Initialize SQLAlchemy with explicit naming convention
convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
db = SQLAlchemy(metadata=metadata)

# Metrics
DB_ERRORS = Counter('database_errors_total', 'Total database errors', ['operation'])
DB_OPERATION_DURATION = Histogram('database_operation_duration_seconds', 'Database operation duration')
DB_CONNECTIONS = Counter('database_connections_total', 'Total database connections')
DB_CONNECTION_ERRORS = Counter('database_connection_errors_total', 'Total database connection errors')
DB_POOL_SIZE = Gauge('database_pool_size', 'Current database pool size')
DB_POOL_AVAILABLE = Gauge('database_pool_available', 'Available connections in pool')
DB_POOL_IN_USE = Gauge('database_pool_in_use', 'Connections currently in use')
DB_TRANSACTION_COUNT = Counter('database_transaction_total', 'Total database transactions')
DB_TRANSACTION_ERRORS = Counter('database_transaction_errors_total', 'Total transaction errors')
DB_TRANSACTION_DURATION = Histogram('database_transaction_duration_seconds', 'Transaction duration')
DB_QUERY_DURATION = Summary('database_query_duration_seconds', 'Query duration', ['query_type'])
DB_BATCH_OPERATIONS = Counter('database_batch_operations_total', 'Total batch operations', ['operation'])
DB_CACHE_HITS = Counter('database_cache_hits_total', 'Total cache hits')
DB_CACHE_MISSES = Counter('database_cache_misses_total', 'Total cache misses')

# Global variables
connection_pool = None
db_session = None
engine = None

# Query cache
query_cache = {}
CACHE_TIMEOUT = 300  # 5 minutes

# Database URL configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
REDIS_TIMEOUT = int(os.getenv('REDIS_TIMEOUT', 3600))  # 1 hour default
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# Create Redis client only if enabled
redis_client = None
if REDIS_ENABLED:
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.ConnectionError as e:
        logger.warning(f"Redis connection failed (even though enabled): {e}. Caching will be disabled.")
        redis_client = None # Explicitly set to None on error
else:
    logger.info("Redis is disabled via REDIS_ENABLED=false. Caching will be disabled.")

def get_engine_options():
    """Get SQLAlchemy engine options with optimized connection pooling."""
    return {
        'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),         # Increased base pool size
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 30)),   # Allow more overflow connections
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),   # Keep timeout reasonable
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 1800)), # Recycle connections after 30 minutes
        'pool_pre_ping': True,                                   # Health check connections before use
        'connect_args': {
            'connect_timeout': 10,                               # Fail fast on connection attempts
            'application_name': 'fil-relex-api'                  # Identify app in DB logs
        }
    }

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    **get_engine_options(),
    echo=False,
    future=True
)

# Create scoped session
db_session = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
)

# Base class for models
Base = declarative_base(metadata=metadata)
Base.query = db_session.query_property()

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

class TransactionError(DatabaseError):
    """Raised when transaction operations fail."""
    pass

class QueryError(DatabaseError):
    """Raised when query execution fails."""
    pass

def get_base_config() -> Dict[str, Any]:
    """Get base database configuration without database name."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
        'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
        'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 10)),
        'application_name': os.getenv('APP_NAME', 'dictionary_api'),
        'statement_timeout': int(os.getenv('DB_STATEMENT_TIMEOUT', 30000)),  # 30 seconds
        'idle_in_transaction_session_timeout': int(os.getenv('DB_IDLE_TIMEOUT', 60000)),  # 60 seconds
        'tcp_keepalives': 1,
        'tcp_keepalives_idle': 60,
        'tcp_keepalives_interval': 10,
        'tcp_keepalives_count': 3,
        'client_encoding': 'UTF8',
        'timezone': 'UTC',
        'application_type': os.getenv('APP_TYPE', 'dictionary'),
        'max_prepared_transactions': int(os.getenv('DB_MAX_PREPARED_TRANSACTIONS', 0)),
        'max_worker_processes': int(os.getenv('DB_MAX_WORKER_PROCESSES', 8)),
        'max_parallel_workers': int(os.getenv('DB_MAX_PARALLEL_WORKERS', 8)),
        'max_parallel_workers_per_gather': int(os.getenv('DB_MAX_PARALLEL_WORKERS_PER_GATHER', 2)),
        'lc_collate': 'en_US.UTF-8',  # For proper sorting of all languages
        'lc_ctype': 'en_US.UTF-8',    # For proper character handling
        'default_text_search_config': 'pg_catalog.simple',  # For text search across languages
        'standard_conforming_strings': 'on',
        'synchronous_commit': 'off',  # For better performance
        'effective_cache_size': '4GB',  # Adjust based on available memory
        'maintenance_work_mem': '256MB',
        'max_stack_depth': '7MB',
        'temp_file_limit': '1GB',
        'work_mem': '64MB',
        'random_page_cost': 1.1,  # Assuming SSD storage
        'enable_seqscan': 'off',  # Prefer index scans
        'effective_io_concurrency': 200,  # For SSDs
        'jit': 'off',  # Disable JIT compilation for predictable performance
        'gin_fuzzy_search_limit': 100,  # For trigram searches
        'gin_pending_list_limit': '4MB'  # For faster GIN index updates
    }

def check_database_exists(dbname: str) -> bool:
    """Check if a database exists with enhanced error handling."""
    try:
        config = get_base_config()
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database='postgres',
            connect_timeout=config['connect_timeout']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone() is not None
            
        conn.close()
        return exists
    except Exception:
        return False

def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment variables with enhanced settings."""
    # List of possible database names in order of preference
    possible_db_names = ['fil_dict_db', 'fil_dict', 'dictionary']
    
    # Get the first existing database or default to fil_dict_db
    database_name = next(
        (name for name in possible_db_names if check_database_exists(name)),
        os.getenv('DB_NAME', 'fil_dict_db')
    )
    
    config = get_base_config()
    config['database'] = database_name
    return config

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def create_database_if_not_exists() -> bool:
    """Create the database if it doesn't exist, with retry logic."""
    config = get_db_config()
    
    try:
        # Connect to default database to create new database
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database='postgres',
            connect_timeout=config['connect_timeout'],
            application_name=f"{config['application_name']}_init"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
            if not cur.fetchone():
                # Create database with proper encoding and locale
                cur.execute(f"""
                    CREATE DATABASE {config['database']}
                    WITH 
                    OWNER = {config['user']}
                    ENCODING = 'UTF8'
                    LC_COLLATE = 'en_US.UTF-8'
                    LC_CTYPE = 'en_US.UTF-8'
                    TEMPLATE = template0
                    CONNECTION LIMIT = -1;
                    
                    -- Set database configuration
                    ALTER DATABASE {config['database']} SET timezone TO 'UTC';
                    ALTER DATABASE {config['database']} SET max_worker_processes TO {config['max_worker_processes']};
                    ALTER DATABASE {config['database']} SET max_parallel_workers TO {config['max_parallel_workers']};
                    ALTER DATABASE {config['database']} SET max_parallel_workers_per_gather TO {config['max_parallel_workers_per_gather']};
                    ALTER DATABASE {config['database']} SET standard_conforming_strings TO 'on';
                    ALTER DATABASE {config['database']} SET synchronous_commit TO 'off';
                    ALTER DATABASE {config['database']} SET effective_cache_size TO '4GB';
                    ALTER DATABASE {config['database']} SET maintenance_work_mem TO '256MB';
                    ALTER DATABASE {config['database']} SET max_stack_depth TO '7MB';
                    ALTER DATABASE {config['database']} SET temp_file_limit TO '1GB';
                    ALTER DATABASE {config['database']} SET work_mem TO '64MB';
                    ALTER DATABASE {config['database']} SET random_page_cost TO 1.1;
                    ALTER DATABASE {config['database']} SET default_text_search_config TO 'pg_catalog.simple';
                """)
                logger.info(f"Created database {config['database']}")
            return True
        
    except Exception as e:
        logger.error("Database creation failed", error=str(e), traceback=traceback.format_exc())
        DB_ERRORS.labels(operation='create_database').inc()
        raise ConnectionError(f"Failed to create database: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def setup_extensions(conn, required_only=False) -> bool:
    """Set up required PostgreSQL extensions with enhanced functionality."""
    try:
        with conn.cursor() as cur:
            # Essential extensions
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                CREATE EXTENSION IF NOT EXISTS btree_gin;
                CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
                CREATE EXTENSION IF NOT EXISTS hstore;
                CREATE EXTENSION IF NOT EXISTS pgcrypto;
                CREATE EXTENSION IF NOT EXISTS dict_xsyn;
                CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
            """)
            
            # Create text search configurations
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_ts_config WHERE cfgname = 'filipino'
                    ) THEN
                        CREATE TEXT SEARCH CONFIGURATION filipino (COPY = simple);
                        ALTER TEXT SEARCH CONFIGURATION filipino
                            ALTER MAPPING FOR asciiword, word, numword, asciihword, hword, numhword
                            WITH unaccent, simple;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1 FROM pg_ts_config WHERE cfgname = 'baybayin'
                    ) THEN
                        CREATE TEXT SEARCH CONFIGURATION baybayin (COPY = simple);
                        ALTER TEXT SEARCH CONFIGURATION baybayin
                            ALTER MAPPING FOR asciiword, word, numword, asciihword, hword, numhword
                            WITH simple;
                    END IF;
                END;
                $$;
            """)
            
            # Drop only our custom functions, not the pg_trgm ones
            cur.execute("""
                DROP FUNCTION IF EXISTS normalize_search_text(text);
                DROP FUNCTION IF EXISTS is_valid_baybayin(text);
                DROP FUNCTION IF EXISTS update_word_search_vector();
            """)
            
            # Create custom functions
            cur.execute("""
                -- Function to normalize text for search
                CREATE OR REPLACE FUNCTION normalize_search_text(text) RETURNS text
                LANGUAGE sql IMMUTABLE
                AS $$
                    SELECT lower(unaccent($1));
                $$;
                
                -- Function to validate Baybayin text
                CREATE OR REPLACE FUNCTION is_valid_baybayin(text) RETURNS boolean
                LANGUAGE sql IMMUTABLE
                AS $$
                    SELECT $1 ~ '^[\u1700-\u171F[:space:]]*$';
                $$;
                
                -- Function to update search vectors
                CREATE OR REPLACE FUNCTION update_word_search_vector() RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    NEW.search_text = setweight(to_tsvector('filipino', NEW.lemma), 'A') ||
                                    setweight(to_tsvector('filipino', COALESCE(NEW.normalized_lemma, '')), 'B') ||
                                    setweight(to_tsvector('filipino', COALESCE(NEW.romanized_form, '')), 'C');
                    RETURN NEW;
                END;
                $$;
            """)
            
            # Create triggers
            cur.execute("""
                -- Trigger for updating search vectors
                DROP TRIGGER IF EXISTS word_search_update ON words;
                CREATE TRIGGER word_search_update
                    BEFORE INSERT OR UPDATE ON words
                    FOR EACH ROW
                    EXECUTE FUNCTION update_word_search_vector();
                
                -- Trigger for updating timestamps
                DROP TRIGGER IF EXISTS update_timestamp ON words;
                CREATE TRIGGER update_timestamp
                    BEFORE UPDATE ON words
                    FOR EACH ROW
                    EXECUTE FUNCTION update_timestamp();
            """)
            
            # Create indexes
            cur.execute("""
                -- GIN indexes for full text search
                CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
                CREATE INDEX IF NOT EXISTS idx_words_trgm ON words USING gin(lemma gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS idx_words_normalized_trgm ON words USING gin(normalized_lemma gin_trgm_ops);
                
                -- B-tree indexes for sorting and exact matches
                CREATE INDEX IF NOT EXISTS idx_words_lemma ON words(lemma);
                CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
                CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
                CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = true;
                
                -- Indexes for relationships
                CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
                CREATE INDEX IF NOT EXISTS idx_etymologies_word ON etymologies(word_id);
                CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
                CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
                CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
                CREATE INDEX IF NOT EXISTS idx_affixations_root ON affixations(root_word_id);
                CREATE INDEX IF NOT EXISTS idx_affixations_affixed ON affixations(affixed_word_id);
            """)
            
            logger.info("Database extensions and language support setup completed")
            return True
        
    except Exception as e:
        logger.error("Failed to set up extensions",
                    error=str(e),
                    traceback=traceback.format_exc())
        if required_only:
            raise DatabaseError(f"Failed to set up required extensions: {str(e)}")
        else:
            logger.warning("Some extensions could not be installed",
                        error=str(e))
            return False

def init_connection_pool() -> None:
    """Initialize the database connection pool with enhanced monitoring."""
    global connection_pool
    config = get_db_config()
    
    if connection_pool is not None:
        return
    
    try:
        connection_pool = ThreadedConnectionPool(
            config['min_connections'],
            config['max_connections'],
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
            connect_timeout=config['connect_timeout'],
            application_name=config['application_name'],
            options=f'-c statement_timeout={config["statement_timeout"]} -c idle_in_transaction_session_timeout={config["idle_in_transaction_session_timeout"]}',
            keepalives=config['tcp_keepalives'],
            keepalives_idle=config['tcp_keepalives_idle'],
            keepalives_interval=config['tcp_keepalives_interval'],
            keepalives_count=config['tcp_keepalives_count'],
            client_encoding=config['client_encoding']
        )
        DB_POOL_SIZE.set(config['max_connections'])
        DB_POOL_AVAILABLE.set(config['max_connections'])
        DB_POOL_IN_USE.set(0)
        logger.info("Database connection pool initialized", 
                   min_connections=config['min_connections'],
                   max_connections=config['max_connections'])
    except Exception as e:
        logger.error("Failed to initialize connection pool", 
                    error=str(e),
                    traceback=traceback.format_exc())
        DB_ERRORS.labels(operation='init_pool').inc()
        raise ConnectionError(f"Failed to initialize database connection pool: {str(e)}")

def get_connection():
    """
    Get a database connection with automatic reconnect handling.
    
    Returns:
        A database connection from the connection pool with optimized settings
    """
    try:
        # Try to get a connection from the pool
        conn = connection_pool.getconn()
        
        # Test if connection is alive
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            
        # Configure connection settings
        conn.autocommit = False  # Always use transactions
        return conn
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
        # Connection failed or is bad - create a new one
        logger.warning(f"Database connection error: {e}. Reconnecting...")
        try:
            if 'conn' in locals():
                try:
                    conn.close()
                except:
                    pass
            
            # Reinitialize the connection pool
            init_connection_pool()
            
            # Get a fresh connection
            conn = connection_pool.getconn()
            conn.autocommit = False
            return conn
        except Exception as retry_err:
            logger.error(f"Failed to reconnect to database: {retry_err}")
            raise DatabaseError(f"Database connection error: {retry_err}")
    except Exception as e:
        logger.error(f"Unexpected error getting connection: {e}")
        raise DatabaseError(f"Database error: {e}")

@contextmanager
def get_db_connection(isolation_level=ISOLATION_LEVEL_READ_COMMITTED, 
                    retry_count=3, 
                    retry_delay=0.5) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    Get a database connection from the pool with enhanced retry capability.
    
    Args:
        isolation_level: PostgreSQL isolation level to use
        retry_count: Maximum number of times to retry getting a connection
        retry_delay: Delay in seconds between retries
        
    Yields:
        A database connection
    """
    conn = None
    start_time = time.time()
    
    for attempt in range(retry_count):
        try:
            # Get a connection from the pool
            conn = get_connection()
            
            # Set isolation level
            conn.set_session(isolation_level=isolation_level)
            
            # Update metrics
            DB_CONNECTIONS.inc()
            current_available = DB_POOL_AVAILABLE._value.get()
            current_in_use = DB_POOL_IN_USE._value.get()
            DB_POOL_AVAILABLE.set(max(0, current_available - 1))
            DB_POOL_IN_USE.set(current_in_use + 1)
            
            # Ensure connection is valid with a simple query
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            
            # Connection is good, yield it
            yield conn
            break
            
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Handle connection errors
            DB_CONNECTION_ERRORS.inc()
            
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
            
            if attempt < retry_count - 1:
                logger.warning(f"Connection error on attempt {attempt+1}: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"Failed to get database connection after {retry_count} attempts: {e}")
                raise ConnectionError(f"Failed to connect to database after {retry_count} attempts: {e}")
                
        except Exception as e:
            # Handle other exceptions
            DB_CONNECTION_ERRORS.inc()
            logger.error(f"Database connection error: {e}", exc_info=True)
            
            if conn:
                try:
                    conn.close()
                except:
                    pass
                conn = None
                
            raise ConnectionError(f"Failed to connect to database: {e}")
            
        finally:
            if attempt == retry_count - 1:
                # Update duration metric on last attempt
                DB_OPERATION_DURATION.observe(time.time() - start_time)
    
    # Handle connection cleanup in a finally block outside the loop
    try:
        # At this point we have either yielded a connection or thrown an exception
        # If conn is not None, we need to clean up after the with block is done
        if conn is not None:
            try:
                if conn.status in (psycopg2.extensions.STATUS_IN_TRANSACTION,
                                psycopg2.extensions.STATUS_PREPARED):
                    logger.debug("Rolling back unfinished transaction")
                    conn.rollback()
                
                # Return connection to the pool
                connection_pool.putconn(conn)
                
                # Update metrics
                current_available = DB_POOL_AVAILABLE._value.get()
                current_in_use = DB_POOL_IN_USE._value.get()
                DB_POOL_AVAILABLE.set(current_available + 1)
                DB_POOL_IN_USE.set(max(0, current_in_use - 1))
                
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}", exc_info=True)
                try:
                    conn.close()
                except:
                    pass
    finally:
        DB_OPERATION_DURATION.observe(time.time() - start_time)

@contextmanager
def transaction(isolation_level=ISOLATION_LEVEL_READ_COMMITTED, 
              retry_count=3, 
              retry_delay=0.5,
              readonly=False):
    """
    Enhanced transaction context manager with configurable isolation level and retry logic.
    
    Args:
        isolation_level: Transaction isolation level
        retry_count: Number of retries for transaction errors
        retry_delay: Initial delay between retries (will increase with backoff)
        readonly: Whether this is a read-only transaction
        
    Yields:
        Database cursor
    """
    with get_db_connection(isolation_level, retry_count, retry_delay) as conn:
        start_time = time.time()
        DB_TRANSACTION_COUNT.inc()
        
        # Set connection to read-only if specified
        if readonly:
            conn.readonly = True
        
        for attempt in range(retry_count):
            try:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    yield cur
                conn.commit()
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                # Connection error during transaction
                conn.rollback()
                DB_TRANSACTION_ERRORS.inc()
                
                if attempt < retry_count - 1:
                    logger.warning(
                        f"Database transaction error (attempt {attempt+1}/{retry_count}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                else:
                    logger.error(f"Transaction failed after {retry_count} attempts: {e}")
                    raise TransactionError(f"Transaction failed after {retry_count} attempts: {e}")
            
            except Exception as e:
                # Other errors - rollback and raise
                conn.rollback()
                DB_TRANSACTION_ERRORS.inc()
                logger.error(f"Transaction error: {e}", exc_info=True)
                raise TransactionError(f"Transaction failed: {e}")
        
        DB_TRANSACTION_DURATION.observe(time.time() - start_time)

def with_transaction(isolation_level=ISOLATION_LEVEL_READ_COMMITTED, 
                   retry_count=3, 
                   retry_delay=0.5,
                   readonly=False):
    """
    Enhanced decorator to wrap function in a transaction with retry capabilities.
    
    Args:
        isolation_level: Transaction isolation level
        retry_count: Number of retries for transaction errors
        retry_delay: Initial delay between retries (will increase with backoff)
        readonly: Whether this is a read-only transaction
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with transaction(
                isolation_level=isolation_level, 
                retry_count=retry_count, 
                retry_delay=retry_delay,
                readonly=readonly
            ) as cur:
                return func(cur, *args, **kwargs)
        return wrapper
    return decorator

def execute_batch_operation(operation: str, data: List[Tuple], page_size: int = 1000) -> int:
    """Execute a batch operation with progress tracking."""
    DB_BATCH_OPERATIONS.labels(operation=operation).inc()
    start_time = time.time()
    
    try:
        with transaction() as cur:
            execute_batch(cur, operation, data, page_size=page_size)
            row_count = cur.rowcount
            logger.info(f"Batch operation completed",
                       operation=operation,
                       rows=row_count,
                       duration=time.time() - start_time)
            return row_count
    except Exception as e:
        logger.error("Batch operation failed",
                    operation=operation,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise

def cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

def cached(timeout: int = REDIS_TIMEOUT, key_prefix: str = "") -> Callable:
    """
    Decorator for caching function results in Redis.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not redis_client:
                return f(*args, **kwargs)
            
            # Generate cache key
            cache_key_str = f"{key_prefix}:{f.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = redis_client.get(cache_key_str)
            if cached_value:
                try:
                    return json.loads(cached_value)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode cached value for key: {cache_key_str}")
            
            # Get fresh value
            result = f(*args, **kwargs)
            
            # Cache the result
            try:
                redis_client.setex(
                    cache_key_str,
                    timeout,
                    json.dumps(result, default=str)
                )
            except (TypeError, json.JSONEncodeError) as e:
                logger.warning(f"Failed to cache result for key {cache_key_str}: {e}")
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str = "*"):
    """
    Invalidate cache entries matching the pattern.
    
    Args:
        pattern: Redis key pattern to match
    """
    if redis_client:
        try:
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries matching pattern: {pattern}")
        except redis.RedisError as e:
            logger.error(f"Failed to invalidate cache: {e}")

def clear_cache():
    """Clear all cache entries."""
    if redis_client:
        try:
            redis_client.flushdb()
            logger.info("Cache cleared")
        except redis.RedisError as e:
            logger.error(f"Failed to clear cache: {e}")

def execute_query(query: str, params: Optional[Dict] = None, fetch: bool = True) -> Any:
    """Execute a query with timing and monitoring."""
    start_time = time.time()
    query_type = query.split()[0].lower()
    
    try:
        with transaction() as cur:
            cur.execute(query, params or {})
            result = cur.fetchall() if fetch else cur.rowcount
            DB_QUERY_DURATION.labels(query_type=query_type).observe(time.time() - start_time)
            return result
    except Exception as e:
        logger.error("Query execution failed",
                    query=query,
                    params=params,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise QueryError(f"Query execution failed: {str(e)}")

def get_table_stats() -> Dict[str, Any]:
    """Get statistics for all tables."""
    query = """
    SELECT 
        schemaname,
        relname as table_name,
        n_live_tup as row_count,
        n_dead_tup as dead_rows,
        n_mod_since_analyze as modifications,
        last_vacuum,
        last_autovacuum,
        last_analyze,
        last_autoanalyze
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    """
    return execute_query(query)

def get_index_stats() -> Dict[str, Any]:
    """Get statistics for all indexes."""
    query = """
    SELECT 
        schemaname,
        relname as table_name,
        indexrelname as index_name,
        idx_scan as scans,
        idx_tup_read as tuples_read,
        idx_tup_fetch as tuples_fetched
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    """
    return execute_query(query)

def get_query_stats() -> Dict[str, Any]:
    """Get statistics for queries."""
    query = """
    SELECT 
        query,
        calls,
        total_time,
        min_time,
        max_time,
        mean_time,
        rows
    FROM pg_stat_statements
    ORDER BY total_time DESC
    LIMIT 100
    """
    return execute_query(query)

def vacuum_analyze(table_name: Optional[str] = None):
    """Perform VACUUM ANALYZE on specified table or all tables."""
    with get_db_connection(ISOLATION_LEVEL_AUTOCOMMIT) as conn:
        with conn.cursor() as cur:
            if table_name:
                cur.execute(f"VACUUM ANALYZE {table_name}")
                logger.info(f"VACUUM ANALYZE completed", table=table_name)
            else:
                cur.execute("VACUUM ANALYZE")
                logger.info("VACUUM ANALYZE completed on all tables")

def check_db_health() -> Dict[str, Any]:
    """Enhanced database health check with detailed metrics."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check basic connectivity
                cur.execute("SELECT 1")
                
                # Check pool status
                pool_info = {
                    "max_connections": connection_pool.maxconn,
                    "used_connections": len(connection_pool._used),
                    "available_connections": connection_pool.maxconn - len(connection_pool._used)
                }
                
                # Check PostgreSQL statistics
                cur.execute("""
                    SELECT * FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                stats = dict(zip([col[0] for col in cur.description], cur.fetchone()))
                
                # Check table statistics
                cur.execute("""
                    SELECT 
                        schemaname,
                        relname,
                        n_live_tup,
                        n_dead_tup,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                """)
                table_stats = [dict(zip([col[0] for col in cur.description], row))
                             for row in cur.fetchall()]
                
                # Check index health
                cur.execute("""
                    SELECT 
                        schemaname,
                        relname as table_name,
                        indexrelname as index_name,
                        idx_scan as scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                """)
                index_stats = [dict(zip([col[0] for col in cur.description], row))
                             for row in cur.fetchall()]
                
                return {
                    "status": "healthy",
                    "pool": pool_info,
                    "statistics": {
                        "commits": stats.get("xact_commit", 0),
                        "rollbacks": stats.get("xact_rollback", 0),
                        "blocks_read": stats.get("blks_read", 0),
                        "blocks_hit": stats.get("blks_hit", 0),
                        "rows_returned": stats.get("tup_returned", 0),
                        "rows_fetched": stats.get("tup_fetched", 0),
                        "rows_inserted": stats.get("tup_inserted", 0),
                        "rows_updated": stats.get("tup_updated", 0),
                        "rows_deleted": stats.get("tup_deleted", 0)
                    },
                    "tables": table_stats,
                    "indexes": index_stats,
                    "vacuum_status": {
                        "running": bool(stats.get("vacuum_count", 0)),
                        "analyze_count": stats.get("analyze_count", 0),
                        "autovacuum_count": stats.get("autovacuum_count", 0),
                        "autoanalyze_count": stats.get("autoanalyze_count", 0)
                    }
                }
    except Exception as e:
        logger.error("Database health check failed",
                    error=str(e),
                    traceback=traceback.format_exc())
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def init_db(app=None):
    """Initialize database with proper configuration."""
    if app is not None:
        db.init_app(app)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Set up session
    if app is not None:
        with app.app_context():
            db.create_all()
    
    logger.info("Database initialized successfully")

def teardown_db(exception=None):
    """Clean up database resources."""
    try:
        if hasattr(db, 'session'):
            db.session.remove()
    except Exception as e:
        # Log but continue teardown process
        logger.error(f"Error during session teardown: {e}")
    
    # For emergency shutdown, consider disposing the engine
    # Only uncomment if needed for emergency cleanup situations
    # if getattr(g, 'emergency_shutdown', False) and hasattr(db, 'engine'):
    #     db.engine.dispose()

def get_db():
    """Get database session."""
    try:
        return db_session
    except Exception as e:
        logger.error(f"Error getting database session: {e}")
        raise

def is_testing_db(engine) -> bool:
    """Check if we're using a test database (SQLite)."""
    return engine.url.drivername == 'sqlite'

def cached_query(timeout: int = 3600, key_prefix: str = ""):
    """Cache the result of a query for a specified time.
    If Redis is not available or disabled, will pass through to the original function.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Explicitly import the global redis_client inside the wrapper
            # to ensure it's correctly referenced in this scope.
            from backend.database import redis_client, REDIS_ENABLED
            
            # Explicitly check if Redis is enabled AND the client exists
            if not REDIS_ENABLED or redis_client is None:
                logger.debug("Redis disabled or client is None, skipping cache.")
                return f(*args, **kwargs)
                
            # Generate cache key
            try:
                serialized_args = json.dumps(args, sort_keys=True, default=str) if args else ""
                serialized_kwargs = json.dumps(kwargs, sort_keys=True, default=str) if kwargs else ""
                key = f"{key_prefix}:{f.__name__}:{serialized_args}:{serialized_kwargs}"
            except TypeError as e:
                logger.warning(f"Could not serialize cache key args/kwargs: {e}. Skipping cache.")
                return f(*args, **kwargs)
            
            # Try to get from cache
            try:
                cached_result = redis_client.get(key)
                if cached_result:
                    logger.debug(f"Cache hit for key: {key}")
                    DB_CACHE_HITS.inc()
                    return json.loads(cached_result)
            except redis.RedisError as e:
                logger.warning(f"Redis GET error for key {key}: {e}. Proceeding without cache.")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode cached JSON for key {key}: {e}. Proceeding without cache.")
                
            # Execute function and cache result
            logger.debug(f"Cache miss for key: {key}")
            DB_CACHE_MISSES.inc()
            result = f(*args, **kwargs)
            
            # Store result in cache
            try:
                serialized_result = json.dumps(result, default=str)
                redis_client.setex(key, timeout, serialized_result)
                logger.debug(f"Cached result for key: {key}")
            except redis.RedisError as e:
                 logger.warning(f"Redis SETEX error for key {key}: {e}")
            except (TypeError, json.JSONEncodeError) as e:
                logger.warning(f"Error serializing result for caching key {key}: {e}")
                
            return result
        return wrapper
    return decorator

def get_pool_status():
    """Return the current database connection pool status for monitoring."""
    if not hasattr(engine, 'pool'):
        return {'error': 'No connection pool available'}
    
    pool = engine.pool
    return {
        'size': pool.size(),
        'checkedin': pool.checkedin(),
        'checkedout': pool.checkedout(),
        'overflow': pool.overflow(),
        'checkedout_overflow': pool.overflow_checkedout()
    }