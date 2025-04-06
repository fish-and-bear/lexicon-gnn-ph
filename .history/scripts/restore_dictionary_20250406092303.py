#!/usr/bin/env python
"""
Dictionary database restore utility.

This script helps restore the dictionary database from a backup file
using the import API endpoint.

Usage:
    python restore_dictionary.py /path/to/backup.zip [options]

Options:
    --api-url URL       Base URL of the API (default: http://localhost:5000/api/v2)
    --api-key KEY       API key for authentication
    --dry-run           Simulate the restore without making changes
    --update-existing   Update words that already exist in the database
    --batch-size N      Number of words to process in each batch (default: 100)
    --skip-validation   Skip data validation (use with caution)
"""

import os
import sys
import argparse
import requests
import json
import zipfile
import logging
import time
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("restore.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dictionary_restore")

def extract_backup_data(backup_file):
    """Extract data from backup file (ZIP or JSON)."""
    backup_path = Path(backup_file)
    
    if not backup_path.exists():
        logger.error(f"Backup file not found: {backup_file}")
        return None
    
    if backup_path.suffix.lower() == '.zip':
        try:
            with zipfile.ZipFile(backup_path, 'r') as zip_file:
                # Look for words.json or similar in the ZIP
                json_files = [f for f in zip_file.namelist() if f.endswith('.json') and ('words' in f or 'dictionary' in f)]
                
                if not json_files:
                    logger.error("No dictionary data found in ZIP file")
                    return None
                
                # Use the first matching file
                with zip_file.open(json_files[0]) as json_file:
                    data = json.load(json_file)
                    logger.info(f"Extracted data from {json_files[0]} in ZIP archive")
                    return data
        except Exception as e:
            logger.error(f"Error extracting data from ZIP file: {str(e)}")
            return None
    
    elif backup_path.suffix.lower() == '.json':
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded data from JSON file: {backup_file}")
                return data
        except Exception as e:
            logger.error(f"Error loading JSON file: {str(e)}")
            return None
    
    else:
        logger.error(f"Unsupported file format: {backup_path.suffix}")
        return None

def get_word_data(data):
    """Extract word data from different possible data structures."""
    if isinstance(data, list):
        # Direct array of words
        return data
    elif isinstance(data, dict):
        # Dictionary with metadata and words
        if 'words' in data and isinstance(data['words'], list):
            return data['words']
        elif 'data' in data and isinstance(data['data'], list):
            return data['data']
    
    # If we can't determine the structure, return the original data
    return data

def restore_backup(backup_file, api_url, api_key, dry_run=False, 
                  update_existing=False, batch_size=100, skip_validation=False):
    """
    Restore a database from a backup file.
    
    Args:
        backup_file: Path to the backup file (.zip or .json)
        api_url: Base URL of the API
        api_key: API key for authentication
        dry_run: If True, simulate the restore without making changes
        update_existing: If True, update words that already exist
        batch_size: Number of words to process in each batch
        skip_validation: If True, skip data validation
        
    Returns:
        True if successful, False otherwise
    """
    # Extract data from backup file
    data = extract_backup_data(backup_file)
    if not data:
        return False
    
    # Extract word data from the structure
    words = get_word_data(data)
    
    if not words:
        logger.error("No word data found in backup")
        return False
    
    logger.info(f"Found {len(words)} words in backup")
    
    # Prepare API endpoint
    import_endpoint = f"{api_url.rstrip('/')}/import"
    
    # Prepare headers
    headers = {
        'X-Api-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    # Prepare import parameters
    params = {
        'update_existing': str(update_existing).lower(),
        'skip_validation': str(skip_validation).lower(),
        'batch_size': str(batch_size),
        'dry_run': str(dry_run).lower()
    }
    
    # Perform the import
    try:
        logger.info(f"Starting restore operation (dry_run={dry_run}, update_existing={update_existing})")
        
        # Send the import request
        response = requests.post(
            import_endpoint,
            headers=headers,
            params=params,
            json=words,
            timeout=600  # 10 minute timeout for large imports
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Restore completed: {result.get('message', 'Unknown status')}")
            
            # Log detailed results
            if 'results' in result:
                stats = result['results']
                logger.info(f"Total words: {stats.get('total', 0)}")
                logger.info(f"Created: {stats.get('created', 0)}")
                logger.info(f"Updated: {stats.get('updated', 0)}")
                logger.info(f"Skipped: {stats.get('skipped', 0)}")
                logger.info(f"Failed: {stats.get('failed', 0)}")
                
                # Log errors if any
                if stats.get('errors'):
                    logger.warning(f"Encountered {len(stats['errors'])} errors during restore")
                    for error in stats['errors'][:10]:  # Log first 10 errors
                        logger.warning(f"Error for word '{error.get('lemma', 'Unknown')}': {error.get('error', 'Unknown error')}")
                    
                    if len(stats['errors']) > 10:
                        logger.warning(f"... and {len(stats['errors']) - 10} more errors. See API response for details.")
            
            return True
        else:
            logger.error(f"Restore failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Restore operation failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Dictionary database restore utility')
    parser.add_argument('backup_file', help='Path to backup file (.zip or .json)')
    parser.add_argument('--api-url', default='http://localhost:5000/api/v2', help='Base URL of the API')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    parser.add_argument('--dry-run', action='store_true', help='Simulate the restore without making changes')
    parser.add_argument('--update-existing', action='store_true', help='Update words that already exist')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of words to process in each batch')
    parser.add_argument('--skip-validation', action='store_true', help='Skip data validation (use with caution)')
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size < 1 or args.batch_size > 1000:
        logger.error("Batch size must be between 1 and 1000")
        sys.exit(1)
    
    # Display warning for skip_validation
    if args.skip_validation:
        logger.warning("Data validation will be skipped. This may lead to errors if the data format is incorrect.")
        
    # Confirm operation if not a dry run
    if not args.dry_run:
        print("\nWARNING: This operation will modify the database.")
        print("It is recommended to backup the database before proceeding.")
        print("\nOptions:")
        print(f"  - API URL: {args.api_url}")
        print(f"  - Backup file: {args.backup_file}")
        print(f"  - Update existing: {args.update_existing}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Skip validation: {args.skip_validation}")
        
        confirm = input("\nProceed with restore? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Restore operation aborted.")
            sys.exit(0)
    
    # Perform the restore
    success = restore_backup(
        args.backup_file,
        args.api_url,
        args.api_key,
        dry_run=args.dry_run,
        update_existing=args.update_existing,
        batch_size=args.batch_size,
        skip_validation=args.skip_validation
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 