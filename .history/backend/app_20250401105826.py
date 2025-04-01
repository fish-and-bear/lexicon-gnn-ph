"""
Flask application for serving the Filipino Dictionary GraphQL API.
"""

from flask import Flask, request, jsonify
from flask_graphql import GraphQLView
from schema import schema
from database import db_session, init_db
from prometheus_client import Counter, Histogram
import time
import logging
from flask_healthz import healthz
from flask_talisman import Talisman
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds')
REQUEST_COUNT = Counter('request_count', 'Total request count')

# Create Flask app
app = Flask(__name__)

# Security headers
Talisman(app, force_https=False)  # Disable HTTPS for local development

# Enable CORS for development
CORS(app)

# Health checks
app.register_blueprint(healthz, url_prefix="/healthz")

def liveness():
    """Health check for kubernetes liveness probe."""
    return True

def readiness():
    """Health check for kubernetes readiness probe."""
    try:
        db_session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database not ready: {e}")
        return False

app.config.update(
    HEALTHZ = {
        "live": "app.liveness",
        "ready": "app.readiness",
    }
)

# Request timing middleware
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        latency = time.time() - request.start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.inc()
    return response

# GraphQL endpoint
app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # Enable GraphiQL interface for testing
    )
)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Database cleanup
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Enable debug mode for development
    )