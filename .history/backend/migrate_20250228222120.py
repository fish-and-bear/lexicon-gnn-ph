"""
Database migration script for the Filipino Dictionary application.
"""

from flask import Flask
from flask_migrate import Migrate, init, migrate, upgrade
from app import create_app
from models import db
import os
from dotenv import load_dotenv
import structlog

# Set up logging
logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()

def run_migrations():
    """Run database migrations."""
    try:
        # Create Flask app with test config
        app = create_app({
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'TESTING': True
        })
        
        # Initialize migration environment
        migrate = Migrate(app, db)
        
        with app.app_context():
            logger.info("Starting database migrations")
            
            # Create the migration
            init()
            logger.info("Migration environment initialized")
            
            # Generate migration
            migrate()
            logger.info("Migration script generated")
            
            # Apply the migration
            upgrade()
            logger.info("Migration applied successfully")
            
            return True
    except Exception as e:
        logger.error(
            "Migration failed",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

if __name__ == '__main__':
    success = run_migrations()
    exit(0 if success else 1) 