#!/usr/bin/env python
"""
Run the migration to add new fields and re-process Kaikki data.

This script:
1. Runs the migration to add tags and etymology_structure fields
2. Re-processes Kaikki.jsonl files to populate these fields with rich data
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from migrations.003_add_tags_and_etymology_structure import run_migration
from backend.db_connection import get_db_connection
from backend.dictionary_manager import process_kaikki_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the migration and re-process Kaikki data."""
    parser = argparse.ArgumentParser(description="Run migration and re-process Kaikki data")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing Kaikki data files")
    parser.add_argument("--skip-migration", action="store_true", help="Skip running the migration")
    parser.add_argument("--kaikki-file", type=str, help="Specific Kaikki file to process, if not provided will look for kaikki.jsonl")
    args = parser.parse_args()
    
    # Step 1: Run the migration to add new fields
    if not args.skip_migration:
        logger.info("Running migration to add tags and etymology_structure fields")
        run_migration()
        logger.info("Migration completed successfully")
    
    # Step 2: Re-process Kaikki data
    logger.info("Re-processing Kaikki data to populate new fields")
    
    # Find Kaikki files
    kaikki_files = []
    if args.kaikki_file:
        kaikki_files.append(args.kaikki_file)
    else:
        data_dirs = [args.data_dir]
        if not os.path.isabs(args.data_dir):
            data_dirs.append(os.path.join("..", args.data_dir))
            
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.startswith("kaikki") and file.endswith(".jsonl"):
                        kaikki_files.append(os.path.join(data_dir, file))
                break
    
    if not kaikki_files:
        logger.error("No Kaikki files found. Please provide a valid file with --kaikki-file")
        return
    
    # Process each Kaikki file
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for kaikki_file in kaikki_files:
            logger.info(f"Processing {kaikki_file}")
            process_kaikki_jsonl(cur, kaikki_file)
            logger.info(f"Finished processing {kaikki_file}")
    except Exception as e:
        logger.error(f"Error processing Kaikki data: {str(e)}")
        raise
    finally:
        conn.close()
    
    logger.info("Data processing completed successfully")

if __name__ == "__main__":
    main() 