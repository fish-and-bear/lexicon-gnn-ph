"""
Database utilities for managing connections and database operations.
Provides connection pooling, retry logic, and monitoring.
"""

from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.orm import scoped_session, sessionmaker
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.pool import ThreadedConnectionPool
import time
from contextlib import contextmanager
from typing import Optional, Generator
from prometheus_client import Counter, Histogram, Gauge
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback
from tenacity import RetryError
import structlog

# Set up structured logging
logger = structlog.get_logger(__name__)

load_dotenv()

# Metrics
DB_ERRORS = Counter('database_errors_total', 'Total database errors', ['operation'])
DB_OPERATION_DURATION = Histogram('database_operation_duration_seconds', 'Database operation duration')
DB_CONNECTIONS = Counter('database_connections_total', 'Total database connections')
DB_CONNECTION_ERRORS = Counter('database_connection_errors_total', 'Total database connection errors')
DB_POOL_SIZE = Gauge('database_pool_size', 'Current database pool size')
DB_POOL_AVAILABLE = Gauge('database_pool_available', 'Available connections in pool')
DB_POOL_IN_USE = Gauge('database_pool_in_use', 'Connections currently in use')

# Global connection pool
connection_pool = None

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'dictionary'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
        'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
        'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 10)),
        'application_name': os.getenv('APP_NAME', 'dictionary_api')
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_database_if_not_exists():
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
                cur.execute(f"CREATE DATABASE {config['database']}")
                logger.info(f"Created database {config['database']}")
        return True
        
    except Exception as e:
        logger.error(f"Database creation failed: {str(e)}")
        DB_ERRORS.labels(operation='create_database').inc()
        raise ConnectionError(f"Failed to create database: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def setup_extensions(conn):
    """Set up required PostgreSQL extensions."""
    try:
        with conn.cursor() as cur:
            # Create extensions if they don't exist
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                CREATE EXTENSION IF NOT EXISTS btree_gin;
                CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
            """)
            logger.info("Database extensions setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up extensions: {str(e)}")
        DB_ERRORS.labels(operation='setup_extensions').inc()
        raise DatabaseError(f"Failed to set up extensions: {str(e)}")

def init_connection_pool():
    """Initialize the database connection pool with monitoring."""
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
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        DB_POOL_SIZE.set(config['max_connections'])
        DB_POOL_AVAILABLE.set(config['max_connections'])
        DB_POOL_IN_USE.set(0)
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}")
        DB_ERRORS.labels(operation='init_pool').inc()
        raise ConnectionError(f"Failed to initialize database connection pool: {str(e)}")

@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection from the pool with automatic cleanup and monitoring."""
    conn = None
    start_time = time.time()
    
    try:
        conn = connection_pool.getconn()
        DB_CONNECTIONS.inc()
        DB_POOL_AVAILABLE.dec()
        DB_POOL_IN_USE.inc()
        
        # Ensure connection is valid
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        
        yield conn
        
    except Exception as e:
        DB_CONNECTION_ERRORS.inc()
        logger.error(f"Database connection error: {str(e)}")
        raise ConnectionError(f"Failed to get database connection: {str(e)}")
    finally:
        if conn is not None:
            connection_pool.putconn(conn)
            DB_POOL_AVAILABLE.inc()
            DB_POOL_IN_USE.dec()
        DB_OPERATION_DURATION.observe(time.time() - start_time)

def create_or_update_tables(conn):
    """Create or update database tables."""
    logger.debug("Creating or updating database tables")
    
    try:
        cur = conn.cursor()
        
        # Create tables if they don't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id SERIAL PRIMARY KEY,
                lemma VARCHAR(255) NOT NULL,
                normalized_lemma VARCHAR(255) NOT NULL,
                language_code VARCHAR(10) NOT NULL,
                root_word_id INTEGER REFERENCES words(id),
                preferred_spelling VARCHAR(255),
                tags TEXT,
                has_baybayin BOOLEAN DEFAULT FALSE,
                baybayin_form TEXT,
                romanized_form TEXT,
                search_text TSVECTOR,
                source_info JSONB,
                pronunciation_data JSONB,
                idioms TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT words_lang_lemma_uniq UNIQUE (language_code, normalized_lemma)
            );
            
            CREATE TABLE IF NOT EXISTS parts_of_speech (
                id SERIAL PRIMARY KEY,
                code VARCHAR(10) NOT NULL UNIQUE,
                name_en VARCHAR(50) NOT NULL,
                name_tl VARCHAR(50) NOT NULL,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS definitions (
                id SERIAL PRIMARY KEY,
                word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                definition_text TEXT NOT NULL,
                original_pos TEXT,
                standardized_pos_id INTEGER REFERENCES parts_of_speech(id),
                examples TEXT,
                usage_notes TEXT,
                sources TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS etymologies (
                id SERIAL PRIMARY KEY,
                word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                etymology_text TEXT NOT NULL,
                normalized_components TEXT,
                language_codes TEXT,
                sources TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS relations (
                id SERIAL PRIMARY KEY,
                from_word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                to_word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                relation_type VARCHAR(50) NOT NULL,
                sources TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (from_word_id, to_word_id, relation_type)
            );
            
            CREATE TABLE IF NOT EXISTS definition_relations (
                id SERIAL PRIMARY KEY,
                definition_id INTEGER REFERENCES definitions(id) ON DELETE CASCADE,
                word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                relation_type VARCHAR(50) NOT NULL,
                sources TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (definition_id, word_id, relation_type)
            );
            
            CREATE TABLE IF NOT EXISTS affixations (
                id SERIAL PRIMARY KEY,
                root_word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                affixed_word_id INTEGER REFERENCES words(id) ON DELETE CASCADE,
                affix_type VARCHAR(50) NOT NULL,
                sources TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (root_word_id, affixed_word_id, affix_type)
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_words_normalized_lemma ON words(normalized_lemma);
            CREATE INDEX IF NOT EXISTS idx_words_language_code ON words(language_code);
            CREATE INDEX IF NOT EXISTS idx_words_search_text ON words USING gin(search_text);
            CREATE INDEX IF NOT EXISTS idx_definitions_word_id ON definitions(word_id);
            CREATE INDEX IF NOT EXISTS idx_etymologies_word_id ON etymologies(word_id);
            CREATE INDEX IF NOT EXISTS idx_relations_from_word_id ON relations(from_word_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to_word_id ON relations(to_word_id);
            CREATE INDEX IF NOT EXISTS idx_definition_relations_definition_id ON definition_relations(definition_id);
            CREATE INDEX IF NOT EXISTS idx_definition_relations_word_id ON definition_relations(word_id);
            CREATE INDEX IF NOT EXISTS idx_affixations_root_word_id ON affixations(root_word_id);
            CREATE INDEX IF NOT EXISTS idx_affixations_affixed_word_id ON affixations(affixed_word_id);
        """)
        
        # Commit the changes
        conn.commit()
        logger.info("Database tables created/updated successfully")
        
    except Exception as e:
        logger.error("Failed to create/update tables",
                    error_msg=str(e),
                    error_type=type(e).__name__,
                    error_traceback=traceback.format_exc())
        raise DatabaseError(f"Failed to create/update tables: {str(e)}")
    finally:
        cur.close()

def init_db():
    """Initialize the database with retries."""
    logger.info("Starting database initialization")
    
    # Create database if it doesn't exist
    create_database_if_not_exists()
    
    # Initialize connection pool
    init_connection_pool()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def init_with_retry():
        try:
            with get_connection() as conn:
                logger.debug("Setting up database extensions")
                setup_extensions(conn)
                
                logger.debug("Creating or updating tables")
                create_or_update_tables(conn)
                
                logger.debug("Verifying database setup")
                verify_database_setup(conn)
                
                logger.info("Database initialization completed successfully")
                return True
        except Exception as e:
            logger.error("Database initialization failed",
                        error_msg=str(e),
                        error_type=type(e).__name__,
                        error_traceback=traceback.format_exc())
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    try:
        init_with_retry()
    except RetryError as e:
        logger.error("Database initialization failed after retries",
                    error_msg=str(e),
                    attempts=3)
        raise DatabaseError(f"Database initialization failed: {str(e)}")

def verify_database_setup(conn):
    """Verify that the database is set up correctly."""
    logger.debug("Starting database verification")
    
    try:
        cur = conn.cursor()
        
        # Check if tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cur.fetchall()]
        logger.debug("Found tables in database", tables=tables)
        
        # Check if extensions are installed
        cur.execute("""
            SELECT extname 
            FROM pg_extension
        """)
        extensions = [row[0] for row in cur.fetchall()]
        logger.debug("Found extensions in database", extensions=extensions)
        
        # Verify basic queries work
        cur.execute("SELECT 1")
        logger.debug("Basic query test successful")
        
        cur.close()
        logger.info("Database verification completed successfully")
    except Exception as e:
        logger.error("Database verification failed",
                    error_msg=str(e),
                    error_type=type(e).__name__,
                    error_traceback=traceback.format_exc())
        raise DatabaseError(f"Database verification failed: {str(e)}")

def close_db():
    """Close all database connections and clean up resources with monitoring."""
    global connection_pool
    if connection_pool is not None:
        try:
            connection_pool.closeall()
            DB_POOL_SIZE.set(0)
            DB_POOL_AVAILABLE.set(0)
            DB_POOL_IN_USE.set(0)
            connection_pool = None
            logger.info("Closed all database connections")
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            DB_ERRORS.labels(operation='close_db').inc()

def check_db_health() -> dict:
    """Check database health and return status information."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check basic connectivity
                cur.execute("SELECT 1")
                
                # Get connection pool stats
                stats = {
                    'status': 'healthy',
                    'pool_size': connection_pool._maxconn,
                    'available_connections': connection_pool._idle_count(),
                    'in_use_connections': connection_pool._in_use,
                }
                
                # Get database size
                cur.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                    AS db_size
                """)
                stats['database_size'] = cur.fetchone()[0]
                
                # Get connection count
                cur.execute("""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE datname = current_database()
                """)
                stats['active_connections'] = cur.fetchone()[0]
                
                return stats
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        DB_ERRORS.labels(operation='health_check').inc()
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

# Initialize database on module import
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database on module import: {str(e)}")
    # Don't raise here - let the application handle initialization errors