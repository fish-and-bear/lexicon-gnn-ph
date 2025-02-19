"""
Main application module for the dictionary API.
Provides application configuration, initialization, and monitoring.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from backend.database import init_db, check_db_health
from backend.caching import init_cache, get_cache_stats
import logging
from backend.routes import bp, init_limiter
from backend.models import db
from backend.language_utils import language_system
from datetime import datetime, UTC
import psutil
import sys
import time
from prometheus_client import Counter, Histogram, Gauge
from flask_talisman import Talisman
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
import structlog
from pythonjsonlogger import jsonlogger
from flask_healthz import healthz
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Metrics
APP_ERRORS = Counter('app_errors_total', 'Total application errors', ['type'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

def setup_logging():
    """Configure structured logging with JSON format."""
    # Create logs directory if it doesn't exist
    log_dir = Path('/app/logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/app/logs/app.log')
        ]
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure JSON logging
    json_handler = logging.StreamHandler()
    json_handler.setFormatter(jsonlogger.JsonFormatter())
    logging.getLogger().addHandler(json_handler)

def create_app(test_config=None):
    """Create and configure the Flask application."""
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
            REDIS_ENABLED=True,
            SUPPORTED_LANGUAGES=['tl', 'ceb'],
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
            PERMANENT_SESSION_LIFETIME=1800,
            MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10MB max file size
            PROPAGATE_EXCEPTIONS=True
        )
    else:
        # Load the test configuration
        app.config.update(test_config)

    # Set up logging
    setup_logging()
    logger = structlog.get_logger()

    # Initialize Sentry for error tracking
    if os.getenv('SENTRY_DSN'):
        sentry_sdk.init(
            dsn=os.getenv('SENTRY_DSN'),
            integrations=[FlaskIntegration()],
            traces_sample_rate=float(os.getenv('SENTRY_SAMPLE_RATE', '0.1')),
            environment=os.getenv('FLASK_ENV', 'production')
        )

    # Configure security headers with Talisman
    Talisman(app,
        force_https=os.getenv('FORCE_HTTPS', 'true').lower() == 'true',
        session_cookie_secure=True,
        content_security_policy={
            'default-src': "'self'",
            'script-src': "'self' 'unsafe-inline'",
            'style-src': "'self' 'unsafe-inline'",
            'img-src': "'self' data:",
            'font-src': "'self'",
            'connect-src': "'self' " + os.getenv('API_URL', '')
        },
        feature_policy={
            'geolocation': "'none'",
            'microphone': "'none'",
            'camera': "'none'"
        }
    )

    # Configure CORS with more secure settings
    CORS(app, 
        resources={
            r"/api/*": {
                "origins": os.getenv('ALLOWED_ORIGINS', '').split(','),
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "expose_headers": ["Content-Range", "X-Total-Count"],
                "supports_credentials": True,
                "max_age": 600
            }
        }
    )

    # Enable response compression
    Compress(app)

    # Configure proxy settings
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Initialize SQLAlchemy
    db.init_app(app)

    # Initialize the database with retry logic
    logger.info("Initializing database...")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def init_database():
        with app.app_context():
            init_db()
        logger.info("Database initialized successfully")

    try:
        init_database()
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        APP_ERRORS.labels(type='database_init').inc()
        raise

    # Initialize cache with error handling
    try:
        with app.app_context():
            init_cache(app.config.get('REDIS_URL'))
            logger.info("Cache initialized successfully")
    except Exception as e:
        logger.error("Error initializing cache", error=str(e))
        APP_ERRORS.labels(type='cache_init').inc()
        # Continue without cache if it fails

    # Initialize rate limiter
    init_limiter(app)

    # Validate language configuration
    for lang in app.config['SUPPORTED_LANGUAGES']:
        if not language_system.is_valid_language(lang):
            logger.error("Invalid language code in configuration", language=lang)
            raise ValueError(f"Invalid language code: {lang}")

    # Register health check endpoints
    app.register_blueprint(healthz, url_prefix="/health")

    def liveness():
        """Check if the application is running."""
        return True

    def readiness():
        """Check if the application is ready to handle requests."""
        try:
            # Check database
            db_health = check_db_health()
            if db_health['status'] != 'healthy':
                raise Exception("Database health check failed")

            # Check cache if enabled
            if app.config.get('REDIS_ENABLED'):
                cache_stats = get_cache_stats()
                if not cache_stats['redis']['connected']:
                    raise Exception("Cache health check failed")

            return True
        except Exception as e:
            logger.error("Readiness check failed", error=str(e))
            return False

    healthz.add_healthz(liveness)
    healthz.add_readiness(readiness)

    @app.route('/')
    def index():
        """API root endpoint with version and status information."""
        return jsonify({
            "name": "Word Relationship API",
            "version": os.getenv("API_VERSION", "2.0.0"),
            "status": "operational",
            "timestamp": datetime.now(UTC).isoformat(),
            "supported_languages": app.config['SUPPORTED_LANGUAGES']
        })

    @app.before_request
    def before_request():
        """Pre-request processing."""
        ACTIVE_REQUESTS.inc()
        request.start_time = time.time()

    @app.after_request
    def after_request(response):
        """Post-request processing."""
        ACTIVE_REQUESTS.dec()
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Calculate request duration
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            REQUEST_DURATION.observe(duration)
        
        # Update system metrics
        MEMORY_USAGE.set(psutil.Process().memory_info().rss)
        CPU_USAGE.set(psutil.cpu_percent())
        
        return response

    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors."""
        APP_ERRORS.labels(type='not_found').inc()
        return jsonify({
            "error": "Resource not found",
            "path": request.path,
            "timestamp": datetime.now(UTC).isoformat()
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        error_id = os.urandom(8).hex()
        logger.error("Internal server error",
            error_id=error_id,
            error=str(error),
            path=request.path,
            method=request.method
        )
        APP_ERRORS.labels(type='internal').inc()
        return jsonify({
            "error": "Internal server error",
            "error_id": error_id,
            "timestamp": datetime.now(UTC).isoformat()
        }), 500

    @app.errorhandler(429)
    def ratelimit_error(error):
        """Handle rate limit errors."""
        APP_ERRORS.labels(type='rate_limit').inc()
        return jsonify({
            "error": "Rate limit exceeded",
            "retry_after": error.description,
            "timestamp": datetime.now(UTC).isoformat()
        }), 429

    # Register blueprint
    app.register_blueprint(bp)

    # Teardown the database session after each request
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db.session.remove()

    return app

# Create the Flask application instance
app = create_app()

if __name__ == '__main__':
    # Run the Flask app directly when the file is executed
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    app.run(
        host=host,
        port=port,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    )
