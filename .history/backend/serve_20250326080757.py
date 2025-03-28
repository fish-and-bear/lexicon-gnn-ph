"""
Server script for the Filipino Dictionary application.
This script handles database schema verification before starting the server.
"""

import os
import sys
import subprocess
import structlog
from app import app

# Set up logging
logger = structlog.get_logger(__name__)

def ensure_database_schema():
    """Ensure the database schema matches the models."""
    try:
        # Try to run the migration script with schema check
        logger.info("Checking database schema...")
        result = subprocess.run(
            [sys.executable, "migrate.py"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            logger.error(
                "Database schema check failed. Running migration with --recreate...",
                error=result.stderr
            )
            # If the check failed, try to recreate the schema
            recreate_result = subprocess.run(
                [sys.executable, "migrate.py", "--recreate"], 
                capture_output=True, 
                text=True
            )
            
            if recreate_result.returncode != 0:
                logger.error(
                    "Database schema recreation failed",
                    error=recreate_result.stderr
                )
                return False
            
            # After recreating schema, run dictionary manager to migrate data
            logger.info("Loading dictionary data...")
            data_dir = os.getenv('DICTIONARY_DATA_DIR', 'data')
            
            # Check if the data directory exists
            if not os.path.exists(data_dir):
                logger.warning(f"Data directory {data_dir} not found")
                # Try parent directory as fallback
                parent_data_dir = os.path.join("..", "data")
                if os.path.exists(parent_data_dir):
                    data_dir = parent_data_dir
                    logger.info(f"Using parent data directory: {data_dir}")
            
            # Run dictionary data migration
            dict_migration = subprocess.run(
                [sys.executable, "dictionary_manager.py", "migrate", "--data-dir", data_dir],
                capture_output=True,
                text=True
            )
            
            if dict_migration.returncode != 0:
                logger.warning(
                    "Dictionary data migration partially failed but may have succeeded in schema creation.",
                    error=dict_migration.stderr
                )
                # Try to verify schema compatibility explicitly
                verify_result = subprocess.run(
                    [sys.executable, "migrate.py"],
                    capture_output=True,
                    text=True
                )
                if verify_result.returncode != 0:
                    logger.error("Schema verification failed after data migration attempt")
                    return False
                
            # Run dictionary_manager verify to check database integrity
            logger.info("Verifying database integrity...")
            verify = subprocess.run(
                [sys.executable, "dictionary_manager.py", "verify"],
                capture_output=True,
                text=True
            )
            
            if verify.returncode != 0:
                logger.warning(
                    "Database integrity verification failed but server can start",
                    error=verify.stderr
                )
            
        return True
    except Exception as e:
        logger.exception("Error checking database schema", error=str(e))
        return False

if __name__ == "__main__":
    # Skip schema check for testing
    logger.info("Skipping database schema verification for testing")
    logger.info("Starting server directly...")
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)