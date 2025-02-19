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
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Metrics
DB_ERRORS = Counter('database_errors_total', 'Total database errors', ['operation'])
DB_OPERATION_DURATION = Histogram('database_operation_duration_seconds', 'Database operation duration')
DB_CONNECTIONS = Counter('database_connections_total', 'Total database connections')
DB_CONNECTION_ERRORS = Counter('database_connection_errors_total', 'Total database connection errors')
DB_POOL_SIZE = Counter('database_pool_size', 'Current database pool size')
DB_POOL_AVAILABLE = Counter('database_pool_available', 'Available connections in pool')

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def setup_extensions():
    """Set up required PostgreSQL extensions with retry logic."""
    config = get_db_config()
    
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            connect_timeout=config['connect_timeout'],
            application_name=f"{config['application_name']}_extensions"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Create extensions if they don't exist
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                CREATE EXTENSION IF NOT EXISTS btree_gin;
                CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
            """)
        return True
        
    except Exception as e:
        logger.error(f"Extension setup failed: {str(e)}")
        DB_ERRORS.labels(operation='setup_extensions').inc()
        raise DatabaseError(f"Failed to set up extensions: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

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
        DB_POOL_SIZE.inc(config['min_connections'])
        DB_POOL_AVAILABLE.inc(config['min_connections'])
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
        DB_OPERATION_DURATION.observe(time.time() - start_time)

def init_db():
    """Initialize the database and connection pool with comprehensive setup."""
    try:
        # Create database if it doesn't exist
        create_database_if_not_exists()
        
        # Set up extensions
        setup_extensions()
        
        # Initialize connection pool
        init_connection_pool()
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        DB_ERRORS.labels(operation='init_db').inc()
        raise DatabaseError(f"Failed to initialize database: {str(e)}")

def close_db():
    """Close all database connections and clean up resources with monitoring."""
    global connection_pool
    if connection_pool is not None:
        try:
            connection_pool.closeall()
            DB_POOL_SIZE.set(0)
            DB_POOL_AVAILABLE.set(0)
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
init_db()