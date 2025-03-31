import pytest
import os
import sys
import logging
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from backend.models import db, PartOfSpeech
from backend.routes import bp

# Load environment variables from .env file
load_dotenv()

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_config():
    """Get database configuration from environment variables with defaults."""
    return {
        'db_name': os.getenv('TEST_DB_NAME', 'fil_dict_test'),
        'db_user': os.getenv('TEST_DB_USER', 'postgres'),
        'db_password': os.getenv('TEST_DB_PASSWORD', 'ta3m1n.!'),
        'db_host': os.getenv('TEST_DB_HOST', 'localhost'),
        'db_port': os.getenv('TEST_DB_PORT', '5432')
    }

def create_test_database():
    """Create the test database if it doesn't exist."""
    config = get_db_config()
    
    # Connect to the default database to create the test database
    try:
        # First try connecting to postgres database
        conn = psycopg2.connect(
            dbname='postgres',
            user=config['db_user'],
            password=config['db_password'],
            host=config['db_host'],
            port=config['db_port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['db_name'],))
        exists = cursor.fetchone()
        
        if exists:
            # Terminate existing connections
            cursor.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{config['db_name']}'
                AND pid <> pg_backend_pid()
            """)
            
            # Drop the test database
            cursor.execute(f"DROP DATABASE {config['db_name']}")
            logger.info(f"Dropped existing test database '{config['db_name']}'")

        # Create the database
        cursor.execute(f"CREATE DATABASE {config['db_name']}")
        logger.info(f"Created test database '{config['db_name']}'")

        # Close the connection to postgres
        cursor.close()
        conn.close()

        # Connect to the new test database to create extensions
        conn = psycopg2.connect(
            dbname=config['db_name'],
            user=config['db_user'],
            password=config['db_password'],
            host=config['db_host'],
            port=config['db_port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Create extensions
        cursor.execute('CREATE EXTENSION IF NOT EXISTS unaccent')
        cursor.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
        cursor.execute('CREATE EXTENSION IF NOT EXISTS btree_gin')
        cursor.execute('CREATE EXTENSION IF NOT EXISTS fuzzystrmatch')
        logger.info("Created database extensions")

    except Exception as e:
        logger.error(f"Error setting up test database: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@pytest.fixture(scope="session")
def app():
    """Create a Flask app instance for testing."""
    from backend.routes import create_app
    
    # Create test database
    create_test_database()
    
    # Use test database URL
    config = get_db_config()
    test_db_url = f"postgresql://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"
    
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': test_db_url,
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'REDIS_ENABLED': False  # Disable Redis caching for tests
    })
    
    # Create the database and tables
    with app.app_context():
        # Drop all tables first
        db.drop_all()
        # Then create all tables
        db.create_all()

        # Add default parts of speech
        default_parts = [
            PartOfSpeech(code='n', name_en='Noun', name_tl='Pangngalan'),
            PartOfSpeech(code='v', name_en='Verb', name_tl='Pandiwa'),
            PartOfSpeech(code='adj', name_en='Adjective', name_tl='Pang-uri'),
            PartOfSpeech(code='adv', name_en='Adverb', name_tl='Pang-abay'),
            PartOfSpeech(code='prep', name_en='Preposition', name_tl='Pang-ukol'),
            PartOfSpeech(code='conj', name_en='Conjunction', name_tl='Pangatnig'),
            PartOfSpeech(code='part', name_en='Particle', name_tl='Kataga'),
            PartOfSpeech(code='intj', name_en='Interjection', name_tl='Padamdamin'),
            PartOfSpeech(code='pron', name_en='Pronoun', name_tl='Panghalip'),
            PartOfSpeech(code='det', name_en='Determiner', name_tl='Pantukoy'),
            PartOfSpeech(code='num', name_en='Numeral', name_tl='Pamilang'),
            PartOfSpeech(code='abbr', name_en='Abbreviation', name_tl='Daglat'),
            PartOfSpeech(code='phr', name_en='Phrase', name_tl='Parirala'),
            PartOfSpeech(code='prefix', name_en='Prefix', name_tl='Unlapi'),
            PartOfSpeech(code='suffix', name_en='Suffix', name_tl='Hulapi'),
            PartOfSpeech(code='infix', name_en='Infix', name_tl='Gitlapi'),
            PartOfSpeech(code='circumfix', name_en='Circumfix', name_tl='Kabilaan'),
            PartOfSpeech(code='root', name_en='Root Word', name_tl='Salitang-ugat'),
        ]

        # Add all parts of speech at once
        db.session.add_all(default_parts)
        db.session.commit()

    yield app

    # Clean up after all tests
    with app.app_context():
        db.drop_all()

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def db_session(app):
    """Create a fresh database session for each test."""
    with app.app_context():
        # Clear all tables before each test
        for table in reversed(db.metadata.sorted_tables):
            db.session.execute(table.delete())
        db.session.commit()

        yield db.session

        # Clean up after the test
        db.session.remove()