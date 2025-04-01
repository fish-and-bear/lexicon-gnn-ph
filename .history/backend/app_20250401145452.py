"""
Flask application entry point.
"""

import os
import sys

# Add the current directory to Python path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, redirect
from flask_cors import CORS
from flask_migrate import Migrate
import logging
from database import db, init_db, teardown_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure database from .env file
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
    db.init_app(app)
    
    # Import and register blueprints
    from routes import bp
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Root route redirect to test API endpoint
    @app.route('/')
    def index():
        return redirect('/api/v2/test')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        teardown_db()
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)