from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix
from routes import bp
from database import db_session, init_db
import logging
import os
from dotenv import load_dotenv
from marshmallow import ValidationError
import sys
from datetime import datetime, UTC
from sqlalchemy import text
import time
from flask_caching import Cache
from prometheus_client import Counter, Histogram, generate_latest
from functools import wraps
import redis
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from flask_compress import Compress

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('word_network.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Enhanced security headers
Talisman(app, 
    force_https=os.environ.get("FORCE_HTTPS", "true").lower() == "true",
    content_security_policy={
    'default-src': "'self'",
    'script-src': "'self' 'unsafe-inline'",
    'style-src': "'self' 'unsafe-inline'",
        'img-src': "'self' data:",
        'font-src': "'self'",
        'connect-src': "'self' " + os.environ.get("API_URL", "")
    },
    feature_policy={
        'geolocation': "'none'",
        'microphone': "'none'",
        'camera': "'none'"
    }
)

# Enhanced CORS settings
CORS(app, 
    resources={
        r"/api/*": {
            "origins": os.environ.get("ALLOWED_ORIGINS", "").split(","),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Range", "X-Total-Count"],
            "supports_credentials": True,
            "max_age": 600
        }
    }
)

# Initialize monitoring
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
    environment=os.environ.get("FLASK_ENV", "production")
)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
DB_QUERY_LATENCY = Histogram('db_query_duration_seconds', 'Database query latency')
CACHE_HIT_COUNT = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISS_COUNT = Counter('cache_misses_total', 'Total cache misses')

# Initialize Redis and Cache
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=int(os.environ.get("REDIS_DB", 0)),
    decode_responses=True
)

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Request validation and sanitization
@app.before_request
def validate_input():
    """Validate and sanitize incoming requests."""
    try:
        # Validate content type for POST/PUT requests
        if request.method in ['POST', 'PUT']:
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 415

        # Sanitize query parameters
        for key, value in request.args.items():
            if not isinstance(value, str):
                continue
            if len(value) > 1000:  # Prevent extremely long inputs
                return jsonify({"error": f"Query parameter {key} too long"}), 400
            if any(char in value for char in ['<', '>', '{', '}']):  # Basic XSS protection
                return jsonify({"error": f"Invalid characters in parameter {key}"}), 400

    except Exception as e:
        logger.error(f"Input validation error: {str(e)}")
        return jsonify({"error": "Invalid request"}), 400

# Enhanced error handling
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all exceptions with proper logging and user-friendly responses."""
    error_id = os.urandom(8).hex()
    
    if isinstance(e, ValidationError):
        return jsonify({
            "error": "Validation error",
            "details": e.messages,
            "error_id": error_id
        }), 400
    
    if isinstance(e, ConnectionError):
        logger.error(f"Database connection error: {str(e)}")
        return jsonify({
            "error": "Database connection error",
            "error_id": error_id
        }), 503

    # Log the full error with stack trace
    logger.error(f"Unhandled exception {error_id}: {str(e)}", exc_info=True)
    
    # Return a generic error message in production
    return jsonify({
        "error": "An unexpected error occurred",
        "error_id": error_id,
        "details": str(e) if app.debug else None
    }), 500

# Initialize the database with retry logic
def initialize_database():
    """Initialize database with retry logic."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
try:
    init_db()
    logger.info("Database initialized successfully")
            return True
except Exception as e:
            retry_count += 1
            logger.error(f"Database initialization attempt {retry_count} failed: {str(e)}")
            if retry_count == max_retries:
                logger.critical("Failed to initialize database after maximum retries")
                raise
            time.sleep(2 ** retry_count)  # Exponential backoff

# Application routes
@app.route('/')
def index():
    """API root endpoint with version and status information."""
    return jsonify({
        "name": "Word Relationship API",
        "version": os.environ.get("API_VERSION", "2.0.0"),
        "status": "operational",
        "timestamp": datetime.now(UTC).isoformat()
    })

# Performance monitoring decorator
def monitor_performance(endpoint_name):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            status = "200"
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=endpoint_name,
                    status=status
                ).inc()
                REQUEST_LATENCY.observe(duration)
        return wrapped
    return decorator

# Database query monitoring
def monitor_query(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            DB_QUERY_LATENCY.observe(duration)
    return wrapped

# Add rate limiting with Redis
def rate_limit(limit_per_minute):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = f"rate_limit:{request.remote_addr}:{f.__name__}"
            current = redis_client.get(key)
            
            if current is None:
                redis_client.setex(key, 60, 1)
            elif int(current) >= limit_per_minute:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": redis_client.ttl(key)
                }), 429
            else:
                redis_client.incr(key)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Add metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest()

# Enhanced health check
@app.route('/health')
@monitor_performance('health_check')
def health_check():
    """Enhanced health check endpoint for monitoring."""
    try:
        # Check database
        db_start = time.time()
        db_session.execute(text("SELECT 1"))
        db_latency = time.time() - db_start

        # Check Redis
        redis_start = time.time()
        redis_client.ping()
        redis_latency = time.time() - redis_start

        # Check disk space
        disk = os.statvfs('/')
        disk_free = (disk.f_bavail * disk.f_frsize) / (1024 * 1024 * 1024)  # GB

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "components": {
                "database": {
                    "status": "connected",
                    "latency_ms": round(db_latency * 1000, 2)
                },
                "redis": {
                    "status": "connected",
                    "latency_ms": round(redis_latency * 1000, 2)
                },
                "disk": {
                    "free_space_gb": round(disk_free, 2)
                }
            },
            "version": os.environ.get("API_VERSION", "2.0.0")
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }), 503

# Add request context logging
@app.before_request
def log_request_info():
    """Log detailed request information."""
    logger.info(
        "Request: %s %s - Client: %s - User-Agent: %s",
        request.method,
        request.path,
        request.remote_addr,
        request.headers.get('User-Agent')
    )

# Add response compression
Compress(app)

# Add security headers middleware
@app.after_request
def add_security_headers(response):
    """Add additional security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

# Add graceful shutdown
def graceful_shutdown(signal, frame):
    """Handle graceful shutdown of the application."""
    logger.info("Received shutdown signal. Starting graceful shutdown...")
    
    # Close database connections
    try:
        db_session.remove()
        logger.info("Closed database connections")
    except Exception as e:
        logger.error(f"Error closing database connections: {str(e)}")

    # Close Redis connections
    try:
        redis_client.close()
        logger.info("Closed Redis connections")
    except Exception as e:
        logger.error(f"Error closing Redis connections: {str(e)}")

    sys.exit(0)

import signal
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Register blueprints
app.register_blueprint(bp)

# Cleanup
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    # Initialize components
    try:
        initialize_database()
        port = int(os.environ.get("PORT", 10000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        app.run(
            host=host,
            port=port,
            debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}", exc_info=True)
        sys.exit(1)
