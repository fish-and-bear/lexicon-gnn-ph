"""
Flask application entry point.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, redirect
from flask_cors import CORS
from flask_migrate import Migrate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    from backend.database import db, init_db, teardown_db
    
    CORS(app)
    init_db(app)
    migrate = Migrate(app, db)
    
    # Import and register blueprints
    from backend.routes import bp
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
    logger.info("Starting Flask application on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)