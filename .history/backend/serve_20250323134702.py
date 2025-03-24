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
        # Try to run the migration script
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
            
        return True
    except Exception as e:
        logger.exception("Error checking database schema", error=str(e))
        return False

if __name__ == "__main__":
    # Check database schema before starting
    if ensure_database_schema():
        logger.info("Database schema verified, starting server...")
        port = int(os.getenv('PORT', 10000))
        host = os.getenv('HOST', '0.0.0.0')
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        app.run(host=host, port=port, debug=debug)
    else:
        logger.error("Failed to ensure database schema. Exiting...")
        sys.exit(1)