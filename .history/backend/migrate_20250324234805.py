"""
Enhanced migration script to ensure database compatibility
between the dictionary_manager and the Flask API.
"""

from sqlalchemy import create_engine, text
import os
import sys
from dotenv import load_dotenv
import logging
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment variables
db_url = os.getenv('DATABASE_URL')
if not db_url:
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'fil_dict')
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Connect to the database
engine = create_engine(db_url)

def run_dictionary_manager_migration():
    """Run the dictionary_manager migration to create or update tables."""
    try:
        logger.info("Running dictionary_manager migration...")
        result = subprocess.run(
            [sys.executable, "dictionary_manager.py", "migrate"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"dictionary_manager migration failed: {result.stderr}")
            return False
        
        logger.info("dictionary_manager migration successful")
        return True
    except Exception as e:
        logger.error(f"Error running dictionary_manager migration: {e}")
        return False

def verify_schema_compatibility():
    """Verify schema compatibility and make necessary adjustments."""
    try:
        with engine.connect() as conn:
            # Check if all required tables exist
            tables = ['words', 'definitions', 'etymologies', 'relations', 
                     'definition_relations', 'affixations', 'parts_of_speech']
            
            for table in tables:
                result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')"))
                exists = result.scalar()
                if not exists:
                    logger.error(f"Table {table} does not exist")
                    return False
            
            # Ensure constraints match
            # 1. Check for baybayin constraints
            logger.info("Ensuring Baybayin constraints are set correctly")
            conn.execute(text("""
                ALTER TABLE words DROP CONSTRAINT IF EXISTS baybayin_form_check;
                ALTER TABLE words ADD CONSTRAINT baybayin_form_check CHECK (
                    (has_baybayin = FALSE AND baybayin_form IS NULL) OR 
                    (has_baybayin = TRUE AND baybayin_form IS NOT NULL)
                );
                
                ALTER TABLE words DROP CONSTRAINT IF EXISTS baybayin_form_regex;
                ALTER TABLE words ADD CONSTRAINT baybayin_form_regex CHECK (
                    baybayin_form ~ '^[\u1700-\u171F\s]*$' OR baybayin_form IS NULL
                );
            """))
            conn.commit()
            
            # 2. Check for search_text column and update it
            logger.info("Ensuring search_text column exists and is properly configured")
            conn.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'words' AND column_name = 'search_text'
                    ) THEN
                        ALTER TABLE words ADD COLUMN search_text TSVECTOR;
                    END IF;
                END $$;
            """))
            conn.commit()
            
            # Create index if it doesn't exist
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
            """))
            conn.commit()
            
            # Update search_text for any entries where it's missing
            conn.execute(text("""
                UPDATE words 
                SET search_text = to_tsvector('simple',
                    COALESCE(lemma, '') || ' ' ||
                    COALESCE(normalized_lemma, '') || ' ' ||
                    COALESCE(baybayin_form, '') || ' ' ||
                    COALESCE(romanized_form, '')
                )
                WHERE search_text IS NULL;
            """))
            conn.commit()
            
            logger.info("Schema compatibility verification complete")
            return True
    except Exception as e:
        logger.error(f"Error verifying schema compatibility: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database migration process")
    
    # Check if the --recreate flag is present
    recreate = "--recreate" in sys.argv
    
    if recreate:
        logger.info("Recreating database (--recreate flag provided)")
        
        # In recreate mode, first run dictionary_manager migration
        if not run_dictionary_manager_migration():
            logger.error("Failed to recreate database using dictionary_manager")
            sys.exit(1)
    
    # Verify schema compatibility whether in recreate mode or not
    if verify_schema_compatibility():
        logger.info("Migration complete. Database is compatible with both dictionary_manager and the API.")
    else:
        logger.error("Migration failed. Database schema is not compatible.")
        
        # Try running dictionary_manager migration if not already done
        if not recreate:
            logger.info("Attempting to repair schema with dictionary_manager...")
            if run_dictionary_manager_migration() and verify_schema_compatibility():
                logger.info("Schema repaired successfully.")
            else:
                logger.error("Failed to repair schema. Manual intervention required.")
                sys.exit(1)
        else:
            sys.exit(1) 