#!/usr/bin/env python
"""
Script to run the migration to add new fields and re-process Kaikki JSONL files.

This script performs two main functions:
1. Runs the migration to add 'tags', 'etymology_structure', and 'metadata' fields
2. Re-processes Kaikki JSONL files to capture rich relationship and etymology information

Usage:
    python run_migration.py [--kaikki-file PATH_TO_KAIKKI_JSONL]
"""
import os
import sys
import argparse
import logging
import glob
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Run migration and re-process Kaikki data")
    parser.add_argument('--kaikki-file', type=str, help='Path to Kaikki JSONL file to re-process')
    args = parser.parse_args()
    
    # Run the migration first
    logger.info("Running migration to add new fields...")
    try:
        # Add the parent directory to sys.path to import modules properly
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import migration module and run it
        from migrations import _003_add_tags_and_etymology_structure
        _003_add_tags_and_etymology_structure.run_migration()
        logger.info("Migration completed successfully")
        
        # Re-process Kaikki data
        from backend.dictionary_manager import process_kaikki_jsonl, get_db_connection
        
        # Find Kaikki JSONL files
        kaikki_files = []
        
        if args.kaikki_file:
            if os.path.exists(args.kaikki_file):
                kaikki_files.append(args.kaikki_file)
            else:
                logger.error(f"Specified Kaikki file not found: {args.kaikki_file}")
                return
        else:
            # Look for Kaikki files in standard locations
            data_dirs = ["data", os.path.join("..", "data")]
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    for file_pattern in ["*kaikki*.jsonl", "*Kaikki*.jsonl"]:
                        matches = glob.glob(os.path.join(data_dir, file_pattern))
                        kaikki_files.extend(matches)
        
        if not kaikki_files:
            logger.warning("No Kaikki JSONL files found to re-process")
            return
            
        # Process each Kaikki file
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                for kaikki_file in kaikki_files:
                    logger.info(f"Re-processing {kaikki_file}...")
                    
                    # Determine language code from filename
                    filename = os.path.basename(kaikki_file).lower()
                    language = "ceb" if "ceb" in filename else "tl"
                    
                    try:
                        process_kaikki_jsonl(cur, kaikki_file)
                        conn.commit()
                        logger.info(f"Successfully processed {kaikki_file}")
                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Error processing {kaikki_file}: {e}")
                        
        finally:
            conn.close()
            
        logger.info("All migrations and data processing completed successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
    except Exception as e:
        logger.error(f"Migration failed with error: {e}")

if __name__ == "__main__":
    main() 