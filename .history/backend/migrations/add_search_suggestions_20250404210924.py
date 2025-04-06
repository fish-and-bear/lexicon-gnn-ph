#!/usr/bin/env python3
"""
Migration script to add database structures needed for efficient search suggestions.
This adds:
1. search_logs table to track queries
2. pg_trgm extension for trigram search
3. Necessary indexes for fast lookups
4. Materialized view for popular words
"""

import logging
import sys
import psycopg2
from psycopg2.extras import execute_batch
from sqlalchemy import text, create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/migration_search_suggestions_{__name__}.log")
    ]
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection from environment variables."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "dictionary")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string, future=True)
        conn = engine.connect()
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def run_migration():
    """Add database structures needed for efficient search suggestions."""
    logger.info("Starting search suggestions infrastructure migration")
    
    try:
        conn = get_db_connection()
        
        # Create search logs table for tracking queries
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS search_logs (
            id SERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            language VARCHAR(10),
            type VARCHAR(20) NOT NULL, 
            word_id INTEGER REFERENCES words(id) ON DELETE SET NULL,
            user_selected BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS search_logs_query_text_idx ON search_logs (query_text);
        CREATE INDEX IF NOT EXISTS search_logs_created_at_idx ON search_logs (created_at);
        CREATE INDEX IF NOT EXISTS search_logs_word_id_idx ON search_logs (word_id);
        """))
        
        # Check if pg_trgm extension exists
        result = conn.execute(text("""
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'
        );
        """)).scalar()
        
        # Create pg_trgm extension if it doesn't exist
        if not result:
            logger.info("Creating pg_trgm extension")
            conn.execute(text("CREATE EXTENSION pg_trgm;"))
        
        # Check if we have necessary indexes
        conn.execute(text("""
        -- For prefix matching
        CREATE INDEX IF NOT EXISTS words_normalized_lemma_pattern_idx ON words (normalized_lemma text_pattern_ops);
        
        -- For trigram matching
        CREATE INDEX IF NOT EXISTS words_normalized_lemma_trgm_idx ON words USING gin (normalized_lemma gin_trgm_ops);
        
        -- For definition text search
        CREATE INDEX IF NOT EXISTS definitions_text_search_idx ON definitions USING gin (to_tsvector('english', definition_text));
        """))
        
        # Create materialized view for popular words
        conn.execute(text("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS popular_words AS
        SELECT 
            word_id, 
            COUNT(*) as search_count
        FROM search_logs
        WHERE 
            created_at > NOW() - INTERVAL '30 days'
            AND word_id IS NOT NULL
        GROUP BY word_id
        ORDER BY search_count DESC;
        
        CREATE INDEX IF NOT EXISTS popular_words_count_idx ON popular_words (search_count DESC);
        """))
        
        # Refresh the materialized view
        conn.execute(text("REFRESH MATERIALIZED VIEW popular_words;"))
        
        # Commit the transaction
        conn.commit()
        
        logger.info("Successfully completed search suggestions infrastructure migration")
        return True
        
    except Exception as e:
        logger.error(f"Error in search suggestions migration: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1) 