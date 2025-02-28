#!/usr/bin/env python3
"""
Script to run the Flask development server with enhanced logging and error handling.
"""
import os
import sys
from pathlib import Path
import logging
import structlog
from datetime import datetime

# Configure logging before anything else
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables with validation and logging."""
    try:
        # Add the parent directory to Python path
        parent_dir = str(Path(__file__).resolve().parent.parent)
        sys.path.insert(0, parent_dir)
        logger.info(f"Added {parent_dir} to Python path")

        # Set and validate environment variables
        env_vars = {
            'PYTHONPATH': parent_dir,
            'FLASK_APP': 'backend.app',
            'FLASK_ENV': 'development',
            'FLASK_DEBUG': '1',
            'DB_NAME': 'fil_dict_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DATABASE_URL': 'postgresql://postgres:postgres@localhost:5432/fil_dict_db',
            'REDIS_ENABLED': 'false'
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"Set {key}={value}")

        return True
    except Exception as e:
        logger.error(f"Failed to set up environment: {str(e)}", exc_info=True)
        return False

def check_database_connection():
    """Check if database is accessible."""
    try:
        import psycopg2
        conn_str = os.environ['DATABASE_URL']
        logger.info("Attempting to connect to database...")
        
        conn = psycopg2.connect(conn_str)
        conn.close()
        logger.info("Successfully connected to database")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}", exc_info=True)
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import flask
        import sqlalchemy
        import redis
        import structlog
        logger.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    startup_time = datetime.now()
    logger.info("Starting deployment process...")

    # Step 1: Set up environment
    if not setup_environment():
        sys.exit(1)

    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Step 3: Check database connection
    if not check_database_connection():
        sys.exit(1)

    # Step 4: Start Flask application
    try:
        from backend.app import app
        logger.info("Successfully imported Flask application")
        
        # Log all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"Route: {rule.rule} [{', '.join(rule.methods)}]")

        # Start the server
        logger.info("Starting Flask server...")
        logger.info(f"Server will be accessible at http://127.0.0.1:10000")
        
        app.run(
            host='127.0.0.1',
            port=10000,
            debug=True,
            use_reloader=True
        )
    except Exception as e:
        logger.error("Failed to start Flask server", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)