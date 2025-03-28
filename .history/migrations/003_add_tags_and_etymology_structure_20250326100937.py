#!/usr/bin/env python
"""
Migration to add 'tags' field to the definitions table and
'etymology_structure' field to the etymologies table.

This enables storing richer information from the Kaikki JSON format.
"""

import os
import sys
import psycopg2
import logging
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from backend
from backend.db_connection import get_db_connection
from backend.dictionary_manager import process_data_source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_migration():
    """
    Run the migration to add the new fields to the database schema.
    """
    conn = None
    try:
        logger.info("Starting migration: Adding tags and etymology_structure fields")
        
        # Connect to the database
        conn = get_db_connection()
        conn.autocommit = False
        
        # Create a cursor
        cur = conn.cursor()
        
        # Add tags field to definitions table
        logger.info("Adding 'tags' field to definitions table")
        cur.execute("""
            ALTER TABLE definitions
            ADD COLUMN IF NOT EXISTS tags TEXT;
        """)
        
        # Add etymology_structure field to etymologies table
        logger.info("Adding 'etymology_structure' field to etymologies table")
        cur.execute("""
            ALTER TABLE etymologies
            ADD COLUMN IF NOT EXISTS etymology_structure TEXT;
        """)
        
        # Update the models to include these new fields
        logger.info("Updating the Definition model in SQLAlchemy")
        cur.execute("""
            -- Add any indices or constraints as needed
            CREATE INDEX IF NOT EXISTS idx_definitions_tags ON definitions (tags);
            CREATE INDEX IF NOT EXISTS idx_etymologies_structure ON etymologies (etymology_structure);
        """)
        
        # Commit the transaction
        conn.commit()
        
        logger.info("Migration complete: Added tags and etymology_structure fields")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_migration() 