"""
Flask application entry point for the Filipino Dictionary API.
Configures the application with database connection, CORS, and GraphQL.
"""

import os
import sys
import time
from flask import Flask, redirect, jsonify, request, g
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_compress import Compress
import logging
import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
pre_chain = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    timestamper,
]

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)
logger = structlog.get_logger("app")

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure database from .env with better defaults
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
        'max_overflow': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30)),
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 1800)),
        'pool_pre_ping': True,  # Add ping before using connections to ensure they're still alive
        'echo': False
    }
    
    # Security settings
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('ENVIRONMENT', 'development') != 'development'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    
    # Performance settings
    app.config['JSON_SORT_KEYS'] = False  # Avoid unnecessary sorting of JSON keys
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Don't prettify JSON in production
    
    # Apply ProxyFix middleware to handle reverse proxies correctly
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    
    # Initialize Flask-Compress for response compression
    compress = Compress(app)
    
    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute", "5 per second"],
        storage_uri=os.getenv("REDIS_URL", "memory://"),
        strategy="fixed-window",
    )
    
    # Import database after app configuration
    from database import db, teardown_db
    
    # Initialize extensions
    cors_resources = {
        r"/api/*": {
            "origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "X-API-Key"]
        }
    }
    CORS(app, resources=cors_resources)
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Configure custom error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource could not be found.',
            'status_code': 404
        }), 404
        
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({
            'error': 'Too many requests',
            'message': 'Rate limit exceeded. Please slow down your requests.',
            'status_code': 429
        }), 429
        
    @app.errorhandler(500)
    def server_error(error):
        logger.error("Internal server error", exc_info=error)
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred on the server.',
            'status_code': 500
        }), 500

    # Performance monitoring middleware
    @app.before_request
    def before_request():
        """Track request start time for performance monitoring."""
        g.start_time = time.time()
        
    @app.after_request
    def after_request(response):
        """Add security headers and log response time."""
        # Add security headers
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Content-Security-Policy', "default-src 'self'")
        
        # Log response time
        if hasattr(g, 'start_time'):
            elapsed_time = time.time() - g.start_time
            logger.info(
                f"Request completed",
                path=request.path,
                method=request.method,
                status=response.status_code,
                time_ms=round(elapsed_time * 1000, 2)
            )
            
        return response
    
    # Import and register blueprints
    from routes import bp
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Initialize and register GraphQL view
    with app.app_context():
        # Run migrations on startup to ensure schema is up to date
        try:
            from flask_migrate import upgrade
            logger.info("Running database migrations...")
            upgrade()
            logger.info("Database migrations completed successfully.")
        except Exception as e:
            logger.error(f"Error running migrations: {str(e)}")
            
        from gql import init_graphql
        from flask_graphql import GraphQLView
        schema, _ = init_graphql()
        
        # Register GraphQL endpoint
        app.add_url_rule(
            '/api/v2/graphql',
            view_func=GraphQLView.as_view(
                'graphql',
                schema=schema,
                graphiql=True
            )
        )
    
    # Root route redirect to API docs or test endpoint
    @app.route('/')
    def index():
        return redirect('/api/v2/test')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        teardown_db()
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 10000))
    debug = os.getenv('ENVIRONMENT', 'development') == 'development'
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}", environment=os.getenv('ENVIRONMENT', 'development'))
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)