"""
Main application file for the Filipino Dictionary application.
"""

from flask import Flask
from flask_cors import CORS
from models import db
from flask_migrate import Migrate
import os
from dotenv import load_dotenv
import structlog

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = structlog.get_logger(__name__)

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure CORS to allow requests from the frontend
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
    
    # Database URL construction from environment variables
    db_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    
    # Default configuration using environment variables
    app.config.update(
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', db_url),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={
            'pool_size': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
            'max_overflow': int(os.getenv('DB_MAX_CONNECTIONS', 20)) - int(os.getenv('DB_MIN_CONNECTIONS', 5)),
            'pool_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 10))
        },
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
        REDIS_URL=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        DEBUG=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        PORT=int(os.getenv('PORT', 10000)),
        HOST=os.getenv('HOST', '0.0.0.0'),
        ALLOWED_ORIGINS=os.getenv('ALLOWED_ORIGINS', '').split(','),
        RATE_LIMIT_DEFAULT=os.getenv('RATE_LIMIT_DEFAULT', '200 per day'),
        RATE_LIMIT_SEARCH=os.getenv('RATE_LIMIT_SEARCH', '60 per minute'),
        RATE_LIMIT_WRITE=os.getenv('RATE_LIMIT_WRITE', '30 per minute')
    )
    
    # Override config if provided
    if config:
        app.config.update(config)
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)

    # Register blueprints
    from routes import bp as api_bp
    app.register_blueprint(api_bp)

    # Initialize rate limiter
    with app.app_context():
        from routes import init_rate_limiter
        init_rate_limiter(app)

    # Set up error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return {
            "error": {
                "message": "Resource not found",
                "code": "ERR_NOT_FOUND",
                "status_code": 404
            }
        }, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {
            "error": {
                "message": "Internal server error",
                "code": "ERR_INTERNAL",
                "status_code": 500
            }
        }, 500

    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)