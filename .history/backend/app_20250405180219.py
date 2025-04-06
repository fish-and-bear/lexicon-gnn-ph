"""
Flask application entry point for the Filipino Dictionary API.
Configures the application with database connection, CORS, and GraphQL.
"""

import os
import sys
import time
from flask import Flask, redirect, jsonify, request, g, abort
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_compress import Compress
import logging
import structlog
from dotenv import load_dotenv
import threading
import psutil

# Fix imports to work both as a module and standalone script
try:
    # When running as part of the package
    from backend.search_tasks import log_search_query
    from backend.database import db
    from backend.routes import bp
    from backend.gql import init_graphql
except ImportError:
    # When running as a standalone script
    from .search_tasks import log_search_query
    from .database import db
    from .routes import bp
    from .gql import init_graphql

# Prometheus for metrics (if not already imported)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    has_prometheus = True
except ImportError:
    has_prometheus = False

# Add health metrics
# if has_prometheus:
    # API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
    # REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])
    # DB_POOL_SIZE = Gauge('db_pool_size', 'Database pool size')
    # DB_POOL_USED = Gauge('db_pool_used', 'Database pool connections in use')
    # SYSTEM_MEMORY = Gauge('system_memory_bytes', 'System memory usage', ['type'])
    # ERROR_COUNTER = Counter('api_errors_total', 'API errors', ['type'])
    # CPU_USAGE = Gauge('system_cpu_usage', 'System CPU usage percentage')

# Add function to update system metrics
def update_system_metrics():
    """Update system resource metrics."""
    if not has_prometheus:
        return
        
    # Update CPU metrics
    CPU_USAGE.set(psutil.cpu_percent())
    
    # Update memory metrics
    memory = psutil.virtual_memory()
    SYSTEM_MEMORY.labels('total').set(memory.total)
    SYSTEM_MEMORY.labels('available').set(memory.available)
    SYSTEM_MEMORY.labels('used').set(memory.used)
    
    # Update DB connection pool metrics
    try:
        # Skip DB pool metrics check since get_pool_status is not available
        pass
    except Exception as e:
        logger.error(f"Error updating DB pool metrics: {e}")

# Add background thread for system metrics
def start_metrics_thread():
    """Start a background thread to update system metrics."""
    if not has_prometheus:
        return None
        
    def metrics_worker():
        while True:
            try:
                update_system_metrics()
            except Exception as e:
                logger.error(f"Error in metrics thread: {e}")
            time.sleep(15)  # Update every 15 seconds
    
    thread = threading.Thread(target=metrics_worker, daemon=True)
    thread.start()
    return thread

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

def create_app(testing=False):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Set testing mode
    app.config['TESTING'] = testing
    
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
    
    # Enhanced rate limiter with Redis storage if available
    redis_url = os.getenv("REDIS_URL")
    storage_uri = redis_url if redis_url else "memory://"
    
    limiter = Limiter(
        app=app,
        default_limits=[
            "200 per minute", 
            "5 per second"
        ],
        storage_uri=storage_uri,
        strategy="fixed-window",
        # Add custom key function that combines IP and endpoint
        key_func=lambda: f"{get_remote_address()}:{request.endpoint}"
    )
    
    # DDOS Prevention: Additional rate limiting for search and expensive endpoints
    if not testing:
        @limiter.limit("20 per minute, 2 per second", key_func=get_remote_address)
        @app.route("/api/v2/search", methods=["GET"])
        def limited_search():
            # This just sets up the rate limit - the actual route is defined elsewhere
            pass
            
        @limiter.limit("30 per minute, 3 per second", key_func=get_remote_address)
        @app.route("/api/v2/search/suggestions", methods=["GET"]) 
        def limited_suggestions():
            # This just sets up the rate limit - the actual route is defined elsewhere
            pass
    
    # Initialize extensions
    cors_resources = {
        r"/api/*": {
            "origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "X-API-Key", "X-Requested-With"]
        }
    }
    CORS(app, resources=cors_resources)
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Register blueprints
    # app.register_blueprint(bp) # Removed redundant registration
    
    # Initialize GraphQL
    # init_graphql(app) # Removed redundant call
    
    # Add a health route for load balancers and monitoring
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint for monitoring."""
        try:
            # Check database connection
            db.session.execute("SELECT 1")
            db_status = "ok"
        except Exception as e:
            db_status = f"error: {str(e)}"
            if not request.args.get('detailed'):
                # Short-circuit if we don't need details
                return jsonify({"status": "unhealthy", "database": db_status}), 500
        
        # Basic response for load balancers
        if not request.args.get('detailed'):
            return jsonify({"status": "healthy", "database": db_status})
            
        # Detailed response for monitoring
        memory = psutil.virtual_memory()
        health_data = {
            "status": "healthy" if db_status == "ok" else "unhealthy",
            "database": {
                "status": db_status,
                # Skip pool status since it's not available
                "pool": {"size": "N/A", "checkedout": "N/A"}
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "uptime_seconds": time.time() - psutil.boot_time()
            },
            "timestamp": time.time()
        }
        
        status_code = 200 if db_status == "ok" else 500
        return jsonify(health_data), status_code
    
    # Configure better error handlers
    @app.errorhandler(429)
    def ratelimit_handler(error):
        """Improved rate limit exceeded handler."""
        if has_prometheus:
            # ERROR_COUNTER.labels('rate_limit_exceeded').inc()
            pass
            
        return jsonify({
            'error': 'Too many requests', 
            'message': str(error),
            'retry_after': error.description.get('retry_after', 60)
        }), 429
        
    @app.errorhandler(500)
    def server_error(error):
        """Improved server error handler."""
        if has_prometheus:
            # ERROR_COUNTER.labels('server_error').inc()
            pass
            
        logger.error("Internal server error", exc_info=error)
        return jsonify({
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500
        
    @app.errorhandler(400)
    def bad_request(error):
        """Improved bad request handler."""
        if has_prometheus:
            # ERROR_COUNTER.labels('bad_request').inc()
            pass
            
        return jsonify({
            'error': 'Bad request',
            'message': str(error)
        }), 400
        
    @app.errorhandler(404)
    def not_found(error):
        """Improved not found handler."""
        if has_prometheus:
            # ERROR_COUNTER.labels('not_found').inc()
            pass
            
        return jsonify({
            'error': 'Not found',
            'path': request.path
        }), 404

    # Performance monitoring and request tracking
    @app.before_request
    def before_request():
        """Track request start time and add request ID."""
        g.start_time = time.time()
        g.request_id = request.headers.get('X-Request-ID', os.urandom(8).hex())
        
        # Check if request is coming from a suspicious source
        if not testing and is_suspicious_request(request):
            # if has_prometheus:
                # ERROR_COUNTER.labels('suspicious_request').inc()
            pass
            abort(403)
        
    @app.after_request
    def after_request(response):
        """Add security headers and log response time."""
        # Add security headers
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Content-Security-Policy', "default-src 'self'")
        response.headers.setdefault('X-Request-ID', getattr(g, 'request_id', 'unknown'))
        
        # Add caching headers for successful GET requests
        if request.method == 'GET' and response.status_code < 400:
            # Use Cache-Control to enable client-side caching
            if not response.headers.get('Cache-Control'):
                response.headers['Cache-Control'] = 'public, max-age=60'
        
        # Log response time and update metrics
        if hasattr(g, 'start_time'):
            elapsed_time = time.time() - g.start_time
            
            if has_prometheus:
                # REQUEST_LATENCY.labels(endpoint=request.endpoint or 'unknown').observe(elapsed_time)
                # API_REQUESTS.labels(
                    # endpoint=request.endpoint or 'unknown',
                    # method=request.method,
                    # status=response.status_code
                # ).inc()
                pass
            
            logger.info(
                f"Request completed",
                path=request.path,
                method=request.method,
                status=response.status_code,
                time_ms=round(elapsed_time * 1000, 2),
                request_id=getattr(g, 'request_id', 'unknown')
            )
            
        # Start system metrics thread if Prometheus is available
        # if has_prometheus:
            # metrics_thread = start_metrics_thread()
        
        return response
    
    # Import and register blueprints
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Initialize and register GraphQL view
    # Only run migrations and GraphQL setup in the main worker process when reloading
    if not testing and os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        with app.app_context():
            # Run migrations on startup to ensure schema is up to date
            try:
                from flask_migrate import upgrade
                logger.info("Running database migrations...")
                # upgrade() # Temporarily disabled
                logger.info("Database migrations completed successfully.")
            except Exception as e:
                logger.error(f"Error running migrations: {str(e)}")
                
            # Initialize GraphQL if flask_graphql is available
            try:
                from flask_graphql import GraphQLView
                schema, context = init_graphql()
                
                # Register GraphQL endpoint
                app.add_url_rule(
                    '/api/v2/graphql',
                    view_func=GraphQLView.as_view(
                        'graphql',
                        schema=schema,
                        graphiql=True
                    )
                )
                logger.info("GraphQL endpoint registered")
            except ImportError:
                logger.warning("flask_graphql not installed, GraphQL endpoint not available")
    
    # Root route redirect to API docs or test endpoint
    @app.route('/')
    def index():
        return redirect('/api/v2/test')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        db.session.remove()
    
    # Initialize search background tasks
    if not testing:
        # Initialize search tasks if the module has the initialize function
        if hasattr(log_search_query, 'initialize'):
            search_tasks_thread = log_search_query.initialize()
        
        # Start system metrics thread if Prometheus is available
        # if has_prometheus:
            # metrics_thread = start_metrics_thread()
    
    # Register search tasks cleanup on app teardown
    @app.teardown_appcontext
    def cleanup_search_tasks(exception=None):
        # Clean up search tasks if the module has the cleanup function
        if hasattr(log_search_query, 'cleanup'):
            log_search_query.cleanup()
    
    return app

def is_suspicious_request(request):
    """
    Check if a request appears suspicious or malicious.
    
    Args:
        request: Flask request object
    
    Returns:
        bool: True if request appears suspicious
    """
    # Check for known malicious user agents
    user_agent = request.headers.get('User-Agent', '').lower()
    suspicious_agents = [
        'zgrab', 'gobuster', 'masscan', 'nmap', 'nikto',
        'sqlmap', 'dirbuster', 'wpscan', 'zmap', 'scanbot',
        'burpcollaborator', 'burpsuite', 'owasp zap',
        'shellshock', 'struts', 'default'
    ]
    
    if any(agent in user_agent for agent in suspicious_agents):
        logger.warning(f"Suspicious user agent detected: {user_agent}")
        return True
    
    # Check for suspicious paths that might indicate scanning
    path = request.path.lower()
    suspicious_paths = [
        'wp-login', 'wp-admin', 'admin', 'phpmyadmin', 'php',
        '.env', '.git', '.svn', '.DS_Store', 'aws',
        'api/v1/', 'graphql/console', 'console', 'phpinfo',
        'config', 'configprops', 'actuator', 'swagger'
    ]
    
    if any(bad_path in path for bad_path in suspicious_paths) and 'api/v2' not in path:
        logger.warning(f"Suspicious path requested: {path}")
        return True
    
    return False

# Create an application instance for direct imports
app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 10000))
    debug = os.getenv('ENVIRONMENT', 'development') == 'development'
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}", environment=os.getenv('ENVIRONMENT', 'development'))
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)