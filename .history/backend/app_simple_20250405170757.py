"""
Simplified Flask application for the Filipino Dictionary API.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from datetime import datetime
from backend.database_simple import db, init_db, teardown_db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure database
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///dictionary.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")
        }
    })
    
    # Initialize database
    init_db(app)
    
    # Register health endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            # Check database connection
            result = db.session.execute("SELECT 1").scalar()
            db_status = "ok" if result == 1 else "error"
        except Exception as e:
            logger.error(f"Database error: {e}")
            db_status = f"error: {str(e)}"
            return jsonify({
                "status": "unhealthy",
                "database": db_status,
                "timestamp": datetime.now().isoformat()
            }), 500
        
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now().isoformat()
        })
    
    # Register API test endpoint
    @app.route('/api/v2/test', methods=['GET'])
    def test_api():
        """Test API endpoint."""
        return jsonify({
            "status": "ok",
            "message": "API is running",
            "timestamp": datetime.now().isoformat()
        })
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        teardown_db()
    
    logger.info("Application created successfully")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=10000, debug=True) 