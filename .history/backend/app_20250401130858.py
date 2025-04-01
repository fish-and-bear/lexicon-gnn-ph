"""
Flask application setup with proper database initialization.
"""

from flask import Flask
from flask_cors import CORS
from routes import bp
from database import db, init_db, teardown_db
import logging
import os

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
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Register teardown
    app.teardown_appcontext(lambda exc: teardown_db())
    
    return app

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run app
    app = create_app()
    app.run(debug=True)