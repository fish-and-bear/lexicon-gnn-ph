"""
Test configuration and fixtures for the Filipino Dictionary API tests.
"""

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
from app import create_app
from models import db as _db
from models import Word, Definition, Etymology, Relation
import json
import tempfile
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3

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

# Enable SQLite foreign key support
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

@pytest.fixture(scope='session')
def app():
    """Create a Flask app context for the tests."""
    # Create a temporary file to use as our database file
    db_fd, db_path = tempfile.mkstemp()
    
    # Create the Flask application
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'CACHE_TYPE': 'simple',
        'TESTING_DB': True  # Flag to indicate we're in testing mode
    })
    
    # Establish an application context
    with app.app_context():
        _db.create_all()
        _setup_test_data()
        yield app
        
        # Cleanup after tests
        _db.session.remove()
        _db.drop_all()
        os.close(db_fd)
        os.unlink(db_path)

@pytest.fixture(scope='function')
def db(app):
    """Create a fresh database for each test."""
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

def _setup_test_data():
    """Set up test data in the database."""
    try:
        # Create parts of speech
        noun = PartOfSpeech(code='n', name_en='noun', name_tl='pangngalan')
        verb = PartOfSpeech(code='v', name_en='verb', name_tl='pandiwa')
        _db.session.add_all([noun, verb])
        
        # Create test words
        aso = Word(
            lemma='aso',
            normalized_lemma='aso',
            language_code='tl',
            has_baybayin=True,
            baybayin_form='ᜀᜐᜓ',
            word_metadata={'quality_score': 0.9},
            verification_status='verified'
        )
        
        tuta = Word(
            lemma='tuta',
            normalized_lemma='tuta',
            language_code='tl',
            has_baybayin=True,
            baybayin_form='ᜆᜓᜆ',
            word_metadata={'quality_score': 0.8},
            verification_status='verified'
        )
        
        tahol = Word(
            lemma='tahol',
            normalized_lemma='tahol',
            language_code='tl',
            has_baybayin=True,
            baybayin_form='ᜆᜑᜓᜎ᜔',
            word_metadata={'quality_score': 0.85},
            verification_status='verified'
        )
        
        _db.session.add_all([aso, tuta, tahol])
        
        # Create definitions
        aso_def = Definition(
            word=aso,
            definition_text='domesticated canine',
            standardized_pos=noun,
            examples=[{'text': 'Ang aso ay tumatahol.', 'translation': 'The dog is barking.'}],
            meta_info={'quality_score': 0.9},
            sources='dictionary'
        )
        
        tuta_def = Definition(
            word=tuta,
            definition_text='puppy, young dog',
            standardized_pos=noun,
            examples=[{'text': 'Ang tuta ay naglalaro.', 'translation': 'The puppy is playing.'}],
            meta_info={'quality_score': 0.85},
            sources='dictionary'
        )
        
        tahol_def = Definition(
            word=tahol,
            definition_text='to bark (of a dog)',
            standardized_pos=verb,
            examples=[{'text': 'Tumatahol ang aso.', 'translation': 'The dog is barking.'}],
            meta_info={'quality_score': 0.8},
            sources='dictionary'
        )
        
        _db.session.add_all([aso_def, tuta_def, tahol_def])
        
        # Create etymologies
        aso_etym = Etymology(
            word=aso,
            etymology_text='From Proto-Austronesian *asu',
            language_codes=['poz'],
            normalized_components=json.dumps([
                {'text': '*asu', 'language': 'poz', 'meaning': 'dog'}
            ]),
            confidence_score=0.9
        )
        
        tuta_etym = Etymology(
            word=tuta,
            etymology_text='From Proto-Malayo-Polynesian *tuta',
            language_codes=['poz'],
            normalized_components=json.dumps([
                {'text': '*tuta', 'language': 'poz', 'meaning': 'young animal'}
            ]),
            confidence_score=0.8
        )
        
        _db.session.add_all([aso_etym, tuta_etym])
        
        # Create relations
        aso_tuta = Relation(
            from_word=aso,
            to_word=tuta,
            relation_type='hypernym',
            bidirectional=False,
            confidence_score=0.9
        )
        
        aso_tahol = Relation(
            from_word=aso,
            to_word=tahol,
            relation_type='subject_of',
            bidirectional=False,
            confidence_score=0.85
        )
        
        _db.session.add_all([aso_tuta, aso_tahol])
        
        # Commit all changes
        _db.session.commit()
        
    except Exception as e:
        _db.session.rollback()
        raise Exception(f"Failed to set up test data: {str(e)}")