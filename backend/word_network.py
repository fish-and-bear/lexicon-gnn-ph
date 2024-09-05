from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix
import os

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Secure headers
Talisman(app, content_security_policy={
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline'",
    'style-src': "'self' 'unsafe-inline'",
})

# CORS settings
CORS(app, resources={r"/api/*": {"origins": os.environ.get("ALLOWED_ORIGINS", "").split(",")}})

# Input validation middleware
@app.before_request
def validate_input():
    # Implement input validation logic here
    pass

# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f"Unhandled exception: {str(e)}")
    # Return a generic error message
    return jsonify({"error": "An unexpected error occurred"}), 500

from flask import Flask, jsonify
from flask_cors import CORS
from routes import bp
from database import db_session, init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins for API endpoints

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

# Error Handlers
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
