import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from backend.database import init_db
import logging
from backend.caching import init_cache
from backend.routes import bp, init_limiter
from backend.models import db

load_dotenv()

def create_app(test_config=None):
    # Initialize the Flask app
    app = Flask(__name__)
    
    # Configure the app
    if test_config is None:
        # Load the default configuration
        app.config.from_mapping(
            SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL'),
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            REDIS_URL=os.getenv('REDIS_URL'),
            TESTING=False,
            REDIS_ENABLED=True
        )
    else:
        # Load the test configuration
        app.config.update(test_config)

    # Configure CORS
    CORS(app, resources={r"/api/*": {"origins": ["https://explorer.hapinas.net", "http://localhost:3000"]}}, supports_credentials=True)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize SQLAlchemy
    db.init_app(app)

    # Initialize the database
    logger.info("Initializing database...")
    try:
        with app.app_context():
            init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

    # Initialize cache
    init_cache(app.config.get('REDIS_URL'))

    # Initialize rate limiter
    init_limiter(app)

    @app.route('/')
    def index():
        return "Welcome to the Word Relationship API!"

    @app.route('/favicon.ico')
    def favicon():
        return '', 204  # No content response

    # Register your blueprint
    app.register_blueprint(bp)

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error('Server Error: %s', error)
        return jsonify({"error": "Internal server error"}), 500

    # Teardown the database session after each request
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()

    # Define any additional routes or logic
    @app.route('/health')
    def health_check():
        """A simple health check route."""
        return {"status": "OK"}

    return app

if __name__ == '__main__':
    # Create and run the Flask app
    app = create_app()
    app.run(host='0.0.0.0', port=10000, debug=True)
