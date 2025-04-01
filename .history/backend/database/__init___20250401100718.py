"""
Database initialization and configuration.
"""

import os
import logging
import psycopg2
import time
import threading
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from contextlib import contextmanager
from typing import Dict, Any, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
_pool: Optional[ThreadedConnectionPool] = None
_initialized_features: Set[str] = set()
_state_lock = threading.RLock()  # Reentrant lock for all state changes

class DatabaseError(Exception):
    """Base class for database errors."""
    pass

class DatabaseConnectionError(DatabaseError):
    """Error connecting to the database."""
    pass

class DatabaseConfigError(DatabaseError):
    """Error in database configuration."""
    pass

class DatabaseInitializationError(DatabaseError):
    """Error during database initialization."""
    pass

def is_feature_initialized(feature: str) -> bool:
    """Check if a feature has been initialized."""
    with _state_lock:
        return feature in _initialized_features

def mark_feature_initialized(feature: str) -> None:
    """Mark a feature as initialized."""
    with _state_lock:
        _initialized_features.add(feature)

@contextmanager
def feature_lock(feature: str):
    """Lock for feature initialization."""
    with _state_lock:
        if is_feature_initialized(feature):
            yield False  # Feature already initialized
            return
        yield True  # Feature needs initialization
        mark_feature_initialized(feature)

def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment variables."""
    try:
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'fil_dict_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
            'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
            'statement_timeout': int(os.getenv('DB_STATEMENT_TIMEOUT', 30000)),
            'idle_timeout': int(os.getenv('DB_IDLE_TIMEOUT', 60000))
        }
        
        # Validate configuration
        if not config['host']:
            raise DatabaseConfigError("Database host not configured")
        if not config['database']:
            raise DatabaseConfigError("Database name not configured")
        if not config['user']:
            raise DatabaseConfigError("Database user not configured")
        if not config['password']:
            raise DatabaseConfigError("Database password not configured")
            
        return config
    except ValueError as e:
        raise DatabaseConfigError(f"Invalid database configuration value: {str(e)}")
    except Exception as e:
        raise DatabaseConfigError(f"Failed to get database configuration: {str(e)}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def wait_for_postgres():
    """Wait for PostgreSQL to be ready."""
    config = get_db_config()
    try:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database='postgres',
            user=config['user'],
            password=config['password'],
            connect_timeout=5
        )
        conn.close()
        logger.info("PostgreSQL is ready")
        return True
    except Exception as e:
        logger.warning(f"PostgreSQL not ready yet: {str(e)}")
        raise DatabaseConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")

def check_database_exists(config: Dict[str, Any]) -> bool:
    """Check if database exists."""
    conn = psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database='postgres',
        user=config['user'],
        password=config['password']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
            return cur.fetchone() is not None
    finally:
        conn.close()

def create_database_if_not_exists() -> None:
    """Create the database if it doesn't exist."""
    with feature_lock('database') as needs_init:  # Use feature lock for database initialization
        if not needs_init:
            logger.info("Database already initialized")
            return
            
        config = get_db_config()
        
        try:
            # Wait for PostgreSQL to be ready
            wait_for_postgres()
            
            # Check if database exists
            if check_database_exists(config):
                logger.info(f"Database {config['database']} already exists")
                return
            
            # Connect to default database
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database='postgres',
                user=config['user'],
                password=config['password']
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            try:
                with conn.cursor() as cur:
                    # Terminate existing connections
                    cur.execute(f"""
                        SELECT pg_terminate_backend(pg_stat_activity.pid)
                        FROM pg_stat_activity
                        WHERE pg_stat_activity.datname = %s
                        AND pid <> pg_backend_pid()
                    """, (config['database'],))
                    
                    # Create database with proper encoding and locale
                    cur.execute(f"""
                        CREATE DATABASE {config['database']}
                        WITH OWNER = {config['user']}
                        ENCODING = 'UTF8'
                        LC_COLLATE = 'C'
                        LC_CTYPE = 'C'
                        TEMPLATE = template0;
                    """)
                    logger.info(f"Created database {config['database']}")
                    
                    # Grant privileges and set timezone
                    cur.execute(f"""
                        GRANT ALL PRIVILEGES ON DATABASE {config['database']} TO {config['user']};
                        ALTER DATABASE {config['database']} SET timezone TO 'UTC';
                    """)
                    
                    # Connect to the new database and create extensions
                    conn_new = psycopg2.connect(
                        host=config['host'],
                        port=config['port'],
                        database=config['database'],
                        user=config['user'],
                        password=config['password']
                    )
                    conn_new.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    
                    try:
                        with conn_new.cursor() as cur_new:
                            # Create extensions
                            cur_new.execute("""
                                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                                CREATE EXTENSION IF NOT EXISTS unaccent;
                                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                            """)
                            logger.info("PostgreSQL extensions created successfully")
                    finally:
                        conn_new.close()
            finally:
                conn.close()
        except RetryError as e:
            raise DatabaseConnectionError("PostgreSQL did not become ready in time") from e
        except Exception as e:
            raise DatabaseError(f"Failed to create database: {str(e)}") from e

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_connection_pool() -> None:
    """Initialize the database connection pool with retry logic."""
    global _pool
    
    with feature_lock('pool') as needs_init:  # Use feature lock for pool initialization
        if not needs_init:
            try:
                # Test the pool
                conn = _pool.getconn()
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
                _pool.putconn(conn)
                return
            except Exception:
                # Pool is invalid, close it and create a new one
                try:
                    _pool.closeall()
                except Exception:
                    pass
                _pool = None
        
        config = get_db_config()
        
        try:
            _pool = ThreadedConnectionPool(
                minconn=config['min_connections'],
                maxconn=config['max_connections'],
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password'],
                options=f"-c statement_timeout={config['statement_timeout']} -c timezone=UTC",
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize connection pool: {str(e)}")

@contextmanager
def get_db_connection():
    """Get a database connection from the pool with automatic retry."""
    if _pool is None:
        init_connection_pool()
    
    conn = None
    try:
        with _state_lock:  # Use state lock for connection management
            conn = _pool.getconn()
            # Test the connection
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        # Try to reinitialize the pool
        try:
            if conn:
                with _state_lock:
                    _pool.putconn(conn)
            init_connection_pool()
            with _state_lock:
                conn = _pool.getconn()
            yield conn
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to recover database connection: {str(e)}")
    finally:
        if conn is not None:
            try:
                with _state_lock:
                    _pool.putconn(conn)
            except Exception as e:
                logger.error(f"Failed to return connection to pool: {str(e)}")

def init_app(app):
    """Initialize the database with the Flask app."""
    from models import db
    from models.base import init_app as init_models
    
    try:
        with feature_lock('app') as needs_init:  # Use feature lock for app initialization
            if not needs_init:
                logger.info("App already initialized")
                return
                
            # Create database if it doesn't exist (this also creates extensions)
            create_database_if_not_exists()
            
            # Initialize Flask-SQLAlchemy
            db.init_app(app)
            
            with app.app_context():
                # Initialize models and create tables
                init_models(app)
                
                # Initialize connection pool
                init_connection_pool()
                
                logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise DatabaseInitializationError("Failed to initialize database") from e 