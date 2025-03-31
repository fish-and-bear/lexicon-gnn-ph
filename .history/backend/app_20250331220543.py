"""
Flask application initialization for the Filipino Dictionary API.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_caching import Cache
from models import db, init_app as init_models
from routes import bp as routes_bp
from security import init_security
from gql.views import graphql_view
import redis
import os
import structlog
from prometheus_client import Counter, Histogram, REGISTRY, CollectorRegistry
from database import init_db, check_db_health
from datetime import datetime
import traceback

# Set up logging
logger = structlog.get_logger(__name__)

# Create a new registry for our metrics
REGISTRY = CollectorRegistry()

# Initialize metrics
SERVER_UPTIME = Histogram('server_uptime_seconds', 'Server uptime in seconds', registry=REGISTRY)
HEALTH_CHECK_STATUS = Counter('health_check_status', 'Health check status (1=healthy, 0=unhealthy)', registry=REGISTRY)
DB_METRICS = Counter('database_metrics', 'Database metrics', ['metric'], registry=REGISTRY)

# Initialize cache
cache = Cache()

def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    if test_config is None:
        # Production configuration
        app.config.update({
            'SQLALCHEMY_DATABASE_URI': os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/fil_dict_db'
            ),
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': {
                'pool_size': int(os.getenv('DB_POOL_SIZE', 20)),
                'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 10)),
                'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),
                'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 1800)),
            },
            'JSON_SORT_KEYS': False,  # Preserve dictionary key order
            'JSON_AS_ASCII': False,    # Proper Unicode support
            'CORS_ORIGINS': os.getenv('CORS_ORIGINS', '*').split(','),
            'RATE_LIMIT_ENABLED': os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            'RATE_LIMIT_DEFAULT': os.getenv('RATE_LIMIT_DEFAULT', '100/minute'),
            'CACHE_TYPE': 'redis',
            'CACHE_REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'CACHE_DEFAULT_TIMEOUT': int(os.getenv('CACHE_TIMEOUT', 300)),
            'MAX_CONTENT_LENGTH': int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)),  # 16MB max-content
            'PROPAGATE_EXCEPTIONS': True,
            'ERROR_INCLUDE_MESSAGE': True,
            'API_TITLE': 'Filipino Dictionary API',
            'API_VERSION': '2.0',
            'OPENAPI_VERSION': '3.0.2',
            'OPENAPI_JSON_PATH': 'api-spec.json',
            'OPENAPI_URL_PREFIX': '/',
            'OPENAPI_REDOC_PATH': '/redoc',
            'OPENAPI_SWAGGER_UI_PATH': '/swagger-ui',
            'OPENAPI_SWAGGER_UI_URL': 'https://cdn.jsdelivr.net/npm/swagger-ui-dist/'
        })
        
        # Initialize Redis
        redis_client = redis.from_url(app.config['CACHE_REDIS_URL'])
    else:
        # Test configuration
        app.config.update(test_config)
        redis_client = None
    
    # Initialize extensions
    CORS(app)
    cache.init_app(app)
    init_models(app)
    
    if not app.config.get('TESTING_DB', False):
        # Only initialize security and database for non-test environments
        if redis_client:
            init_security(app, redis_client)
        
        # Initialize database with all extensions for production
        with app.app_context():
            init_db(required_extensions_only=False)
    else:
        # For testing, initialize database with only required extensions
        with app.app_context():
            init_db(required_extensions_only=True)
    
    # Register blueprints
    app.register_blueprint(routes_bp)
    
    # Register GraphQL endpoint
    app.add_url_rule(
        '/graphql',
        view_func=graphql_view,
        methods=['GET', 'POST', 'OPTIONS']
    )
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({
            "error": "Bad request",
            "message": str(error),
            "status_code": 400,
            "path": request.path
        }), 400

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not found",
            "status_code": 404,
            "path": request.path
        }), 404
    
    @app.errorhandler(429)
    def ratelimit_error(error):
        return jsonify({
            "error": "Too many requests",
            "status_code": 429,
            "path": request.path,
            "retry_after": error.description
        }), 429

    @app.errorhandler(500)
    def internal_error(error):
        logger.error("Internal server error",
                    error=str(error),
                    traceback=traceback.format_exc(),
                    path=request.path,
                    method=request.method)
        return jsonify({
            "error": "Internal server error",
            "status_code": 500,
            "path": request.path
        }), 500
    
    # Health check endpoint
    @app.route('/health')
    @cache.cached(timeout=30)  # Cache for 30 seconds
    def health_check():
        health_status = check_db_health()
        return jsonify({
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "database": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": os.getenv('APP_VERSION', '2.0.0'),
            "environment": os.getenv('FLASK_ENV', 'production')
        }), 200 if health_status["status"] == "healthy" else 503
    
    # Request logging and metrics
    @app.before_request
    def before_request():
        # Store start time for duration calculation
        request._start_time = datetime.utcnow()
        # Log request
        logger.info("Request started",
                   path=request.path,
                   method=request.method,
                   remote_addr=request.remote_addr)

    @app.after_request
    def after_request(response):
        # Calculate request duration
        duration = (datetime.utcnow() - request._start_time).total_seconds()
        
        # Log response
        logger.info("Request completed",
                   path=request.path,
                   method=request.method,
                   status=response.status_code,
                   duration=duration)
        
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run()