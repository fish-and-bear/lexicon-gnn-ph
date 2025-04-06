#!/usr/bin/env python3
"""
Migration script to rename the 'metadata' column to 'pronunciation_metadata' in the pronunciations table.

This script performs the following operations:
1. Adds a new column 'pronunciation_metadata' to the pronunciations table
2. Copies data from 'metadata' to 'pronunciation_metadata'
3. Drops the 'metadata' column

Usage:
    python rename_metadata_column.py

Requirements:
    The script requires database connection details in .env file or environment variables:
    - DB_NAME
    - DB_USER
    - DB_PASSWORD
    - DB_HOST
    - DB_PORT
"""

import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get database connection parameters
DB_NAME = os.getenv("DB_NAME", "fil_dict_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        sys.exit(1)

def execute_migration():
    """Execute the migration to rename the metadata column."""
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Check if the metadata column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pronunciations' AND column_name = 'metadata'
        """)
        if cur.fetchone() is None:
            logger.info("Column 'metadata' does not exist in the pronunciations table. "
                      "It might have been already renamed or never existed.")
            return

        # Check if the pronunciation_metadata column already exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'pronunciations' AND column_name = 'pronunciation_metadata'
        """)
        if cur.fetchone() is not None:
            logger.info("Column 'pronunciation_metadata' already exists in the pronunciations table.")
            
            # Check if we still need to migrate data
            cur.execute("""
                SELECT COUNT(*) FROM pronunciations WHERE metadata IS NOT NULL
            """)
            count = cur.fetchone()[0]
            if count == 0:
                logger.info("No data to migrate. Proceeding to drop the old column.")
            else:
                # Migrate remaining data
                logger.info(f"Migrating {count} rows of data from 'metadata' to 'pronunciation_metadata'...")
                cur.execute("""
                    UPDATE pronunciations 
                    SET pronunciation_metadata = metadata 
                    WHERE metadata IS NOT NULL AND (pronunciation_metadata IS NULL OR pronunciation_metadata = '{}')
                """)
                logger.info("Data migration completed.")
        else:
            # Add the new column
            logger.info("Adding 'pronunciation_metadata' column...")
            cur.execute("""
                ALTER TABLE pronunciations ADD COLUMN pronunciation_metadata JSONB
            """)
            logger.info("Column added successfully.")
            
            # Copy data from the old column to the new one
            logger.info("Migrating data from 'metadata' to 'pronunciation_metadata'...")
            cur.execute("""
                UPDATE pronunciations SET pronunciation_metadata = metadata
            """)
            logger.info("Data migration completed.")
            
            # Create index on the new column
            logger.info("Creating index on 'pronunciation_metadata'...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pronunciations_pron_metadata 
                ON pronunciations USING GIN (pronunciation_metadata)
            """)
            logger.info("Index created successfully.")
        
        # Drop the old column
        logger.info("Dropping 'metadata' column...")
        cur.execute("ALTER TABLE pronunciations DROP COLUMN IF EXISTS metadata")
        logger.info("Column dropped successfully.")
        
        # Commit the transaction
        conn.commit()
        logger.info("Migration completed successfully.")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    logger.info("Starting migration to rename 'metadata' to 'pronunciation_metadata'...")
    execute_migration()
    logger.info("Migration process completed.") 