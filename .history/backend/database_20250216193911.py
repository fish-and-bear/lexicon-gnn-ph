from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging
from backend.models import db
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'db_user': os.getenv('DB_USER', 'postgres'),
        'db_password': os.getenv('DB_PASSWORD', 'ta3m1n.!'),
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': os.getenv('DB_PORT', '5432'),
        'db_name': os.getenv('DB_NAME', 'fil_dict_db')
    }

def create_database_if_not_exists():
    """Create the database if it doesn't exist."""
    config = get_db_config()
    
    # Connect to default database to create new database
    conn = psycopg2.connect(
        dbname='postgres',
        user=config['db_user'],
        password=config['db_password'],
        host=config['db_host'],
        port=config['db_port']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        cur = conn.cursor()
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['db_name'],))
        exists = cur.fetchone()
        
        if exists:
            # Terminate existing connections
            cur.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{config['db_name']}'
                AND pid <> pg_backend_pid()
            """)
            
            # Drop the database if it exists
            cur.execute(f"DROP DATABASE {config['db_name']}")
            logger.info(f"Dropped existing database '{config['db_name']}'")

        # Create the database
        cur.execute(f"CREATE DATABASE {config['db_name']}")
        logger.info(f"Created database {config['db_name']}")
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def setup_extensions():
    """Set up required PostgreSQL extensions."""
    config = get_db_config()
    
    # Connect to the target database to create extensions
    conn = psycopg2.connect(
        dbname=config['db_name'],
        user=config['db_user'],
        password=config['db_password'],
        host=config['db_host'],
        port=config['db_port']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        cur = conn.cursor()
        # Create extensions
        cur.execute('CREATE EXTENSION IF NOT EXISTS unaccent')
        cur.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
        cur.execute('CREATE EXTENSION IF NOT EXISTS btree_gin')
        cur.execute('CREATE EXTENSION IF NOT EXISTS fuzzystrmatch')
        logger.info("Created database extensions")
    except Exception as e:
        logger.error(f"Error creating extensions: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def init_db():
    """Initialize the database with required extensions and tables."""
    try:
        # First create database if it doesn't exist
        create_database_if_not_exists()
        
        # Set up extensions
        setup_extensions()
        
        # Create tables using SQLAlchemy
        db.create_all()
        db.session.commit()

        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        db.session.rollback()
        raise