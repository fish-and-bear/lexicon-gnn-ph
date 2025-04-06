#!/usr/bin/env python
"""
Scheduled backup script for the dictionary database.
Performs regular backups using the export API endpoint.

Usage:
    python backup_dictionary.py [--full] [--incremental] [--output-dir DIR]

Options:
    --full           Perform a full backup (default)
    --incremental    Perform an incremental backup (only words updated since last backup)
    --output-dir     Directory to store backups (default: ./backups)
"""

import os
import sys
import argparse
import requests
import json
import datetime
import logging
import zipfile
import shutil
import time
from pathlib import Path
import hashlib
import croniter
import configparser
from datetime import datetime, timedelta, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backup.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("dictionary_backup")

# Default configuration
DEFAULT_CONFIG = {
    'api': {
        'base_url': 'http://localhost:5000/api/v2',
        'api_key': '',
        'timeout': '300',
    },
    'backup': {
        'output_dir': './backups',
        'retention_full': '90',      # days to keep full backups
        'retention_incremental': '30',  # days to keep incremental backups
        'compress': 'true',
        'backup_schedule': '0 2 * * *',  # 2 AM daily in cron format
        'max_retries': '3',
    }
}

def load_config(config_path='backup_config.ini'):
    """Load configuration from file, create default if not exists."""
    config = configparser.ConfigParser()
    
    # Set default config
    for section, options in DEFAULT_CONFIG.items():
        if not config.has_section(section):
            config.add_section(section)
        for option, value in options.items():
            config.set(section, option, value)
    
    # Try to load from file
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        # Save default config
        with open(config_path, 'w') as f:
            config.write(f)
        logger.info(f"Created default configuration at {config_path}")
    
    return config

def create_backup_filename(backup_type, extension='.zip'):
    """Create a backup filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{backup_type}_backup_{timestamp}{extension}"

def get_last_backup_date(backup_dir, backup_type):
    """Find the date of the last backup of specified type."""
    pattern = f"{backup_type}_backup_"
    
    try:
        # List all files in backup directory
        files = [f for f in os.listdir(backup_dir) if f.startswith(pattern)]
        
        if not files:
            return None
            
        # Sort by name (which includes the timestamp)
        files.sort(reverse=True)
        
        # Extract date from the most recent backup
        latest = files[0]
        
        # Parse the timestamp from the filename
        # Assumes format: type_backup_YYYYMMDD_HHMMSS.ext
        date_str = latest.replace(pattern, '').split('.')[0]
        
        # Try to parse the date
        try:
            timestamp = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            return timestamp
        except ValueError:
            logger.warning(f"Could not parse date from filename: {latest}")
            return None
            
    except Exception as e:
        logger.error(f"Error determining last backup date: {str(e)}")
        return None

def perform_backup(config, backup_type='full'):
    """Perform a backup of the specified type using the export API."""
    api_base_url = config.get('api', 'base_url')
    api_key = config.get('api', 'api_key')
    timeout = int(config.get('api', 'timeout'))
    output_dir = config.get('backup', 'output_dir')
    max_retries = int(config.get('backup', 'max_retries'))
    compress = config.getboolean('backup', 'compress')
    
    # Create backup directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build API request parameters
    endpoint = f"{api_base_url.rstrip('/')}/export"
    headers = {
        'X-Api-Key': api_key,
        'Accept': 'application/json',
    }
    
    params = {
        'format': 'zip' if compress else 'json',
    }
    
    # For incremental backups, determine the last backup date
    if backup_type == 'incremental':
        last_backup = get_last_backup_date(output_dir, 'full')
        if not last_backup:
            last_backup = get_last_backup_date(output_dir, 'incremental')
            
        if last_backup:
            # Add some buffer to prevent missing updates during the backup itself
            last_backup = last_backup - timedelta(minutes=10)
            params['updated_after'] = last_backup.isoformat()
            logger.info(f"Incremental backup for updates since {last_backup.isoformat()}")
        else:
            logger.warning("No previous backup found, performing full backup instead")
            backup_type = 'full'
    
    # Backup filename
    backup_filename = os.path.join(
        output_dir, 
        create_backup_filename(backup_type, '.zip' if compress else '.json')
    )
    
    # Attempt the backup with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting {backup_type} backup (attempt {attempt+1}/{max_retries})")
            
            # Make the API request
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                stream=True,
                timeout=timeout
            )
            
            # Check for success
            if response.status_code == 200:
                # Write response to file
                with open(backup_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Backup completed successfully: {backup_filename}")
                
                # Calculate and store backup hash
                file_hash = calculate_file_hash(backup_filename)
                hash_file = f"{backup_filename}.sha256"
                with open(hash_file, 'w') as f:
                    f.write(file_hash)
                
                logger.info(f"Backup hash: {file_hash}")
                
                # Record successful backup in manifest
                update_backup_manifest(output_dir, backup_type, backup_filename, file_hash)
                
                return backup_filename
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    
        except Exception as e:
            logger.error(f"Backup attempt {attempt+1} failed: {str(e)}")
            
        # Wait before retry
        if attempt < max_retries - 1:
            sleep_time = (attempt + 1) * 5  # Exponential backoff
            logger.info(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    logger.error(f"All backup attempts failed.")
    return None

def calculate_file_hash(filename):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def update_backup_manifest(backup_dir, backup_type, backup_filename, file_hash):
    """Update the backup manifest with information about a new backup."""
    manifest_file = os.path.join(backup_dir, "backup_manifest.json")
    
    # Create or load existing manifest
    if os.path.exists(manifest_file):
        with open(manifest_file, 'r') as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                manifest = {"backups": []}
    else:
        manifest = {"backups": []}
    
    # Add new backup entry
    manifest["backups"].append({
        "type": backup_type,
        "filename": os.path.basename(backup_filename),
        "timestamp": datetime.now().isoformat(),
        "hash": file_hash,
    })
    
    # Save updated manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

def cleanup_old_backups(config):
    """Delete old backups based on retention policy."""
    output_dir = config.get('backup', 'output_dir')
    retention_full = int(config.get('backup', 'retention_full'))
    retention_incremental = int(config.get('backup', 'retention_incremental'))
    
    logger.info(f"Cleaning up old backups (retention: full={retention_full} days, incremental={retention_incremental} days)")
    
    try:
        # Get the cutoff dates
        now = datetime.now()
        full_cutoff = now - timedelta(days=retention_full)
        incr_cutoff = now - timedelta(days=retention_incremental)
        
        # Get a list of all backup files
        files = os.listdir(output_dir)
        
        for filename in files:
            filepath = os.path.join(output_dir, filename)
            
            # Skip non-backup files and directories
            if not os.path.isfile(filepath) or not any(x in filename for x in ['full_backup_', 'incremental_backup_']):
                continue
                
            try:
                # Extract date from filename
                if 'full_backup_' in filename:
                    backup_type = 'full'
                    date_str = filename.replace('full_backup_', '').split('.')[0]
                    cutoff = full_cutoff
                else:
                    backup_type = 'incremental'
                    date_str = filename.replace('incremental_backup_', '').split('.')[0]
                    cutoff = incr_cutoff
                
                # Parse the date
                file_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                
                # Check if the file is older than the cutoff
                if file_date < cutoff:
                    logger.info(f"Removing old {backup_type} backup: {filename}")
                    os.remove(filepath)
                    
                    # Also remove the hash file if it exists
                    hash_file = f"{filepath}.sha256"
                    if os.path.exists(hash_file):
                        os.remove(hash_file)
            
            except Exception as e:
                logger.warning(f"Error processing backup file {filename}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during backup cleanup: {str(e)}")

def verify_backups(config):
    """Verify integrity of backup files using stored hashes."""
    output_dir = config.get('backup', 'output_dir')
    
    try:
        files = os.listdir(output_dir)
        backup_files = [f for f in files if 'backup_' in f and not f.endswith('.sha256')]
        
        for backup_file in backup_files:
            filepath = os.path.join(output_dir, backup_file)
            hash_file = f"{filepath}.sha256"
            
            if os.path.exists(hash_file):
                # Read stored hash
                with open(hash_file, 'r') as f:
                    stored_hash = f.read().strip()
                
                # Calculate current hash
                current_hash = calculate_file_hash(filepath)
                
                if current_hash == stored_hash:
                    logger.info(f"Verified backup integrity: {backup_file}")
                else:
                    logger.warning(f"Backup file corrupted: {backup_file} (hash mismatch)")
            else:
                logger.warning(f"No hash file found for {backup_file}")
    
    except Exception as e:
        logger.error(f"Error verifying backups: {str(e)}")

def is_backup_due(config):
    """Check if a backup is due based on the schedule."""
    try:
        schedule = config.get('backup', 'backup_schedule')
        cron = croniter.croniter(schedule, datetime.now())
        
        # Get the previous scheduled time
        prev_run = cron.get_prev(datetime)
        
        # Check manifest for the last actual run
        manifest_file = os.path.join(config.get('backup', 'output_dir'), "backup_manifest.json")
        
        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                
            if manifest.get('backups'):
                # Get the latest backup timestamp
                backups = sorted(manifest['backups'], key=lambda x: x['timestamp'], reverse=True)
                last_backup = datetime.fromisoformat(backups[0]['timestamp'])
                
                # Check if last backup was before the previous scheduled time
                return last_backup < prev_run
        
        # If no manifest or no backups, assume backup is due
        return True
    
    except Exception as e:
        logger.error(f"Error checking backup schedule: {str(e)}")
        # If there's an error, assume backup is due to be safe
        return True

def main():
    parser = argparse.ArgumentParser(description='Dictionary database backup utility')
    parser.add_argument('--full', action='store_true', default=False, help='Perform a full backup')
    parser.add_argument('--incremental', action='store_true', default=False, help='Perform an incremental backup')
    parser.add_argument('--verify', action='store_true', default=False, help='Verify integrity of existing backups')
    parser.add_argument('--cleanup', action='store_true', default=False, help='Remove old backups according to retention policy')
    parser.add_argument('--config', default='backup_config.ini', help='Path to configuration file')
    parser.add_argument('--output-dir', help='Directory to store backups')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args if provided
    if args.output_dir:
        config.set('backup', 'output_dir', args.output_dir)
    
    # Determine backup type
    if not args.full and not args.incremental:
        backup_type = 'full'  # Default to full backup
    elif args.full:
        backup_type = 'full'
    else:
        backup_type = 'incremental'
    
    # Special operations
    if args.verify:
        verify_backups(config)
        return
    
    if args.cleanup:
        cleanup_old_backups(config)
        return
    
    # Check if backup is scheduled (only if not explicitly requested)
    if not (args.full or args.incremental):
        if not is_backup_due(config):
            logger.info("No backup due according to schedule. Exiting.")
            return
    
    # Perform the backup
    backup_file = perform_backup(config, backup_type)
    
    if backup_file:
        # Cleanup old backups
        cleanup_old_backups(config)
    else:
        logger.error("Backup failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 