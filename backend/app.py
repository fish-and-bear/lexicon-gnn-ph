from flask import Flask, jsonify
from flask_cors import CORS
from routes import bp
from database import db_session, init_db
import logging
from flask_caching import Cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# Configure caching
cache_config = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL')
}
cache = Cache(app, config=cache_config)

# Initialize the database
logger.info("Initializing database...")
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
