"""
Flask application for the Filipino Dictionary API.
"""

from flask import Flask
from flask_cors import CORS
from flask_healthz import healthz
from flask_talisman import Talisman
from database import init_db, close_db
from schema import schema
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure app
    app.config.from_object('config')
    
    # Initialize extensions
    CORS(app)
    healthz.init_app(app)
    Talisman(app)
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    from routes import api
    app.register_blueprint(api, url_prefix='/api')
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    # Register cleanup
    app.teardown_appcontext(close_db)
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)