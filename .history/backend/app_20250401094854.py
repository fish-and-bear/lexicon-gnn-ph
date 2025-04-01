"""
Flask application factory for the Filipino Dictionary API.
"""

from flask import Flask, jsonify
from flask_cors import CORS
from models import db, init_app as init_models
import logging
import os
from pathlib import Path
from datetime import datetime

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    try:
        # Configure Flask-SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = (
            f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'postgres')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('DB_NAME', 'fil_dict_db')}"
        )
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
            'pool_timeout': 30,
            'pool_recycle': 1800,
            'pool_pre_ping': True,
            'max_overflow': 5
        }
        
        # Configure CORS
        CORS(app, resources={
            r"/api/*": {
                "origins": os.getenv('ALLOWED_ORIGINS', '*').split(','),
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization", "X-API-Key"]
            }
        })
        
        # Initialize database and models
        init_models(app)
        logger.info("Database and models initialized successfully")
        
        # Register routes
        try:
            from routes import bp as api_bp
            app.register_blueprint(api_bp)
            logger.info("API routes registered successfully")
        except ImportError as e:
            logger.error(f"Failed to import routes: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to register routes: {str(e)}")
            raise
        
        # Register error handlers
        @app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Not found", "message": str(error)}), 404
        
        @app.errorhandler(500)
        def server_error(error):
            db.session.rollback()
            logger.error(f"Internal server error: {str(error)}")
            return jsonify({"error": "Internal server error"}), 500
        
        @app.errorhandler(Exception)
        def handle_exception(error):
            db.session.rollback()
            logger.error(f"Unhandled exception: {str(error)}")
            return jsonify({"error": "Internal server error"}), 500
        
        # Add basic health check endpoint
        @app.route('/')
        def health_check():
            try:
                # Test database connection
                db.session.execute('SELECT 1')
                db.session.commit()
                return jsonify({
                    "status": "healthy",
                    "database": "connected",
                    "timestamp": datetime.utcnow().isoformat()
                }), 200
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                return jsonify({
                    "status": "unhealthy",
                    "database": "disconnected",
                    "error": str(e)
                }), 503
        
        logger.info("Application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create application: {str(e)}")
        raise