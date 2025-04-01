"""
Flask application for the Filipino Dictionary API.
"""

from flask import Flask
from flask_cors import CORS
from flask_graphql import GraphQLView
from database import db, init_db
from routes import api
from config import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    
    # Initialize database
    with app.app_context():
        init_db(app)
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)