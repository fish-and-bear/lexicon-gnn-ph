#!/usr/bin/env python3
"""
Migration script to add the languages table to the database.
This creates the languages table based on the Language model definition.
"""

import logging
import sys
import os
from dotenv import load_dotenv
from sqlalchemy import text, create_engine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/migration_add_languages_table.log")
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection from environment variables."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "fil_dict_db")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string, future=True)
        conn = engine.connect()
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def run_migration():
    """Create the languages table if it doesn't exist."""
    logger.info("Starting languages table migration")
    
    try:
        conn = get_db_connection()
        
        # Create languages table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS languages (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            code VARCHAR(10) NOT NULL UNIQUE,
            name_en VARCHAR(50) NOT NULL,
            name_tl VARCHAR(50),
            region VARCHAR(50),
            family VARCHAR(50),
            status VARCHAR(20)
        );
        
        CREATE INDEX IF NOT EXISTS languages_code_idx ON languages (code);
        """))
        
        # Insert some default languages
        conn.execute(text("""
        INSERT INTO languages (code, name_en, name_tl, region, family, status)
        VALUES 
            ('tl', 'Tagalog', 'Tagalog', 'Philippines', 'Austronesian', 'living'),
            ('fil', 'Filipino', 'Filipino', 'Philippines', 'Austronesian', 'living'),
            ('ceb', 'Cebuano', 'Cebuano', 'Philippines', 'Austronesian', 'living'),
            ('ilo', 'Ilokano', 'Ilokano', 'Philippines', 'Austronesian', 'living'),
            ('en', 'English', 'Ingles', 'Global', 'Indo-European', 'living'),
            ('es', 'Spanish', 'Espanyol', 'Spain/Americas', 'Indo-European', 'living'),
            ('ja', 'Japanese', 'Hapon', 'Japan', 'Japonic', 'living'),
            ('zh', 'Chinese', 'Tsino', 'China', 'Sino-Tibetan', 'living'),
            ('san', 'Sanskrit', 'Sanskrit', 'India', 'Indo-European', 'historical')
        ON CONFLICT (code) DO NOTHING;
        """))
        
        # Commit the transaction
        conn.commit()
        
        logger.info("Successfully created languages table and added default languages")
        return True
        
    except Exception as e:
        logger.error(f"Error in languages table migration: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1) 