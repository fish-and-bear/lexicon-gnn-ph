"""
Flask application factory for the Filipino Dictionary API.
"""

from flask import Flask
from flask_cors import CORS
from database import init_db, close_db
import routes
import logging
import os

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
        }
    })
    
    # Initialize database
    init_db(app)
    
    # Register routes
    app.register_blueprint(routes.bp)
    
    # Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def server_error(error):
        return {"error": "Internal server error"}, 500
    
    # Add cleanup
    @app.teardown_appcontext
    def cleanup(exc):
        close_db()
    
    # Add basic health check endpoint
    @app.route('/')
    def health_check():
        return {"status": "healthy"}, 200
    
    return app