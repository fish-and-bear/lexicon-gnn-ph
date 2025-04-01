"""
Database initialization and configuration.
"""

import os
import logging
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[ThreadedConnectionPool] = None

def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment variables."""
    return {
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

def create_database_if_not_exists() -> None:
    """Create the database if it doesn't exist."""
    config = get_db_config()
    
    # Connect to default database to check if our database exists
    conn = psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database='postgres',
        user=config['user'],
        password=config['password']
    )
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
            exists = cur.fetchone() is not None
            
            if not exists:
                # Create database
                cur.execute(f"CREATE DATABASE {config['database']}")
                logger.info(f"Created database {config['database']}")
    finally:
        conn.close()

def init_connection_pool() -> None:
    """Initialize the database connection pool."""
    global _pool
    
    if _pool is not None:
        return
    
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
            options=f"-c statement_timeout={config['statement_timeout']}"
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {str(e)}")
        raise

@contextmanager
def get_db_connection():
    """Get a database connection from the pool."""
    if _pool is None:
        init_connection_pool()
    
    conn = None
    try:
        conn = _pool.getconn()
        yield conn
    finally:
        if conn is not None:
            _pool.putconn(conn)

def init_app(app):
    """Initialize the database with the Flask app."""
    from models import db
    from models.base import init_app as init_models
    
    # Initialize Flask-SQLAlchemy
    db.init_app(app)
    
    # Initialize models
    init_models(app)
    
    # Initialize connection pool
    init_connection_pool()
    
    logger.info("Database initialized successfully") 