import os
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from database import db_session, init_db
import logging
from caching import multi_level_cache as cache, init_cache
from routes import bp

load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the database
logger.info("Initializing database...")
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")

# Initialize cache
init_cache(os.getenv('REDIS_URL'))

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": os.getenv('ALLOWED_ORIGINS', '*').split(',')}})

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
    db_session.remove()

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
    db_session.remove()

# Define any additional routes or logic
@app.route('/health')
def health_check():
    """A simple health check route."""
    return {"status": "OK"}

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=10000, debug=True)
