from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import logging
from backend.models import db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def init_db():
    """Initialize the database with required extensions and tables."""
    # Create extensions
    db.session.execute(text('CREATE EXTENSION IF NOT EXISTS unaccent'))
    db.session.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
    db.session.commit()

    # Create tables
    db.create_all()
    db.session.commit()

    return True

def get_db_config():
    """Get database configuration from environment variables."""
    return {
        'db_user': os.getenv('DB_USER', 'postgres'),
        'db_password': os.getenv('DB_PASSWORD', 'postgres'),
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': os.getenv('DB_PORT', '5432'),
        'db_name': os.getenv('DB_NAME', 'fil_dict')
    }