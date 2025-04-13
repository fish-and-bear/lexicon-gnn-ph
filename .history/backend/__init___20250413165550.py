"""
Filipino Dictionary API backend package.
This package contains the Flask application and all its modules.
"""

from backend.database import db
from backend.app import create_app

# Define package version
__version__ = '2.0.0'

# Define exported symbols
__all__ = ['create_app', 'db']

from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from backend.database import init_db, teardown_db
import logging
import os
import sys
import re

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure database
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'max_overflow': 20,
        'pool_timeout': 30,
        'pool_recycle': 1800
    }
    
    # Initialize extensions
    CORS(app)
    
    # Initialize database with the application context
    with app.app_context():
        db.init_app(app)
        # This ensures the DB initialization happens within the application context
        try:
            db.create_all()
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Failed to create database tables: {str(e)}")
    
    migrate = Migrate(app, db)
    
    # Import and register blueprints
    from .routes import bp
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        teardown_db()
    
    return app

# Make backend a Python package

# Fix import issues with source_standardization
import os
import sys
import re

# Define the fallback implementation directly in __init__.py so it's available to all modules
def source_extract_etymology_components(etymology_text):
    """Extract word components from etymology text."""
    if not etymology_text:
        return []
    
    # Skip bracketed language codes
    if re.match(r'^\s*\[\s*(?:Ing|Esp|War|San|Arb|Ch|Jap|Mal|Tsino)\s*\]\s*$', etymology_text, re.IGNORECASE):
        return []
    
    # Extract potential components
    words = re.findall(r'\b[a-zA-Z]{3,}\b', etymology_text.lower())
    return list(set(words))  # Remove duplicates

# Add to sys.modules so it can be imported by other modules in the package
sys.modules['source_standardization'] = type('', (), {
    'extract_etymology_components': source_extract_etymology_components
})