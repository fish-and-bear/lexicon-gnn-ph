"""
Flask application entry point for the Filipino Dictionary API.
Configures the application with database connection, CORS, and GraphQL.
"""

import os
import sys
import time
from flask import Flask, redirect, jsonify, request, g, abort, make_response, url_for, send_file, current_app, send_from_directory
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter.util import get_remote_address
from flask_compress import Compress
from flask_talisman import Talisman
import logging
import structlog
from dotenv import load_dotenv
import threading
import psutil
import json
import uuid
import traceback
from datetime import datetime, timedelta
from functools import wraps
import re
import psycopg2

# Fix imports to work both as a module and standalone script
# try:
#     # When running as part of the package
#     from backend.search_tasks import log_search_query
#     from backend.database import db
#     from backend.routes import bp
#     # Import gql module instead of the specific function
#     import backend.gql as gql_module
#     from backend.extensions import limiter
#     from backend.baybayin_routes import bp as baybayin_bp
# except ImportError:
#     # When running as a standalone script
# Use explicit relative imports as the app is run from within the backend directory
# from .search_tasks import log_search_query
from .database import db
from .routes import bp
# from . import gql as gql_module # Use relative import for gql
from .extensions import limiter
# from .baybayin_routes import bp as baybayin_bp

# Prometheus for metrics (if not already imported)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    has_prometheus = True
except ImportError:
    has_prometheus = False

# Function to ensure parts_of_speech table exists without dropping data
def ensure_parts_of_speech_table_exists(db_uri):
    """
    Ensure the parts_of_speech table exists without dropping any existing data.
    Only creates the table if it doesn't already exist.
    """
    try:
        # Extract connection parameters from SQLAlchemy URI
        # Format: postgresql://username:password@host:port/dbname
        if '://' in db_uri:
            db_uri = db_uri.split('://', 1)[1]
        if '@' in db_uri:
            auth, conn_str = db_uri.split('@', 1)
            if ':' in auth:
                user, password = auth.split(':', 1)
            else:
                user, password = auth, ''
        else:
            user, password = '***', '***'
            conn_str = db_uri
            
        if '/' in conn_str:
            host_port, dbname = conn_str.split('/', 1)
            if ':' in host_port:
                host, port = host_port.split(':', 1)
                port = int(port)
            else:
                host, port = host_port, 5432
        else:
            host, port = conn_str, 5432
            dbname = 'postgres'
            
        # Connect to the database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname
        )
        conn.autocommit = False
        cur = conn.cursor()
        
        # Check if parts_of_speech table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'parts_of_speech'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            logger.info("Creating parts_of_speech table as it doesn't exist")
            # Create the parts_of_speech table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS parts_of_speech (
                    id SERIAL PRIMARY KEY,
                    code VARCHAR(32) NOT NULL UNIQUE,
                    name_en VARCHAR(64) NOT NULL,
                    name_tl VARCHAR(64) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT parts_of_speech_code_uniq UNIQUE (code)
                );
                CREATE INDEX IF NOT EXISTS idx_parts_of_speech_code ON parts_of_speech(code);
                CREATE INDEX IF NOT EXISTS idx_parts_of_speech_name ON parts_of_speech(name_en, name_tl);
            """)
            
            # Insert standard parts of speech
            pos_entries = [
                # --- Core Grammatical Categories ---
                (
                    "n",
                    "Noun",
                    "Pangngalan",
                    "Word that refers to a person, place, thing, or idea",
                ),
                ("v", "Verb", "Pandiwa", "Word that expresses action or state of being"),
                ("adj", "Adjective", "Pang-uri", "Word that describes or modifies a noun"),
                (
                    "adv",
                    "Adverb",
                    "Pang-abay",
                    "Word that modifies verbs, adjectives, or other adverbs",
                ),
                ("pron", "Pronoun", "Panghalip", "Word that substitutes for a noun"),
                (
                    "prep",
                    "Preposition",
                    "Pang-ukol",
                    "Word that shows relationship between words",
                ),
                (
                    "conj",
                    "Conjunction",
                    "Pangatnig",
                    "Word that connects words, phrases, or clauses",
                ),
                ("intj", "Interjection", "Pandamdam", "Word expressing emotion"),
                ("det", "Determiner", "Pantukoy", "Word that modifies nouns"),
                ("affix", "Affix", "Panlapi", "Word element attached to base or root"),
                # Other important categories
                ("lig", "Ligature", "Pang-angkop", "Word that links modifiers to modified words"),
                ("part", "Particle", "Kataga", "Function word that doesn't fit other categories"),
                ("num", "Number", "Pamilang", "Word representing a number"),
                ("unc", "Uncategorized", "Hindi Tiyak", "Part of speech not yet determined"),
            ]

            for code, name_en, name_tl, desc in pos_entries:
                cur.execute("""
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (code) DO UPDATE 
                    SET name_en = EXCLUDED.name_en,
                        name_tl = EXCLUDED.name_tl,
                        description = EXCLUDED.description
                """, (code, name_en, name_tl, desc))
                
            conn.commit()
            logger.info("parts_of_speech table created and populated with standard entries")
        else:
            # Table exists, check if we have records
            cur.execute("SELECT COUNT(*) FROM parts_of_speech")
            count = cur.fetchone()[0]
            if count == 0:
                logger.warning("parts_of_speech table exists but is empty, adding standard entries")
                # Add standard entries
                cur.execute("""
                    INSERT INTO parts_of_speech (code, name_en, name_tl, description)
                    VALUES 
                    ('n', 'Noun', 'Pangngalan', 'Word that refers to a person, place, thing, or idea'),
                    ('v', 'Verb', 'Pandiwa', 'Word that expresses action or state of being'),
                    ('adj', 'Adjective', 'Pang-uri', 'Word that describes or modifies a noun'),
                    ('unc', 'Uncategorized', 'Hindi Tiyak', 'Part of speech not yet determined')
                    ON CONFLICT (code) DO NOTHING
                """)
                conn.commit()
                logger.info("Added basic entries to empty parts_of_speech table")
            else:
                logger.info(f"parts_of_speech table exists with {count} entries")
                
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error ensuring parts_of_speech table: {str(e)}")
        return False

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
    # CPU_USAGE.set(psutil.cpu_percent())
    
    # Update memory metrics
    memory = psutil.virtual_memory()
    # SYSTEM_MEMORY.labels('total').set(memory.total)
    # SYSTEM_MEMORY.labels('available').set(memory.available)
    # SYSTEM_MEMORY.labels('used').set(memory.used)
    
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
    
    # Configure CORS properly with all necessary headers
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",  # Allow all origins in development
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": "*",  # Allow all headers
            "expose_headers": [
                "Content-Range", "X-Total-Count", "X-Request-ID",
                "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"
            ],
            "supports_credentials": True,
            "max_age": 600
        }
    })
    
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
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize Flask-Compress for response compression
    compress = Compress(app)
    
    # Initialize Talisman for security headers
    Talisman(app)

    # Enhanced rate limiter configuration (storage URI)
    redis_enabled = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    storage_uri = "memory://"
    if redis_enabled:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            storage_uri = redis_url
        else:
            logger.warning("REDIS_ENABLED is true but REDIS_URL is not set. Falling back to memory storage for rate limiter.")

    app.config['RATELIMIT_STORAGE_URI'] = storage_uri # Set config BEFORE initializing limiter
    limiter.init_app(app)
    app.config['RATELIMIT_DEFAULT_LIMITS'] = ["200 per minute", "5 per second"]
    # Configure the key function using app.config
    app.config['RATELIMIT_KEY_FUNC'] = lambda: f"{get_remote_address()}:{request.endpoint}"

    # DDOS Prevention: Additional rate limiting for search and expensive endpoints
    if not testing:
        # @limiter.limit("20 per minute, 2 per second", key_func=get_remote_address)
        # @app.route("/api/v2/search", methods=["GET"])
        # def limited_search():
            # This just sets up the rate limit - the actual route is defined elsewhere
            # pass
            
        @limiter.limit("30 per minute, 3 per second", key_func=get_remote_address)
        @app.route("/api/v2/search/suggestions", methods=["GET"]) 
        def limited_suggestions():
            # This just sets up the rate limit - the actual route is defined elsewhere
            pass
    
    # Only run setup in the main worker process when reloading
    if not testing and os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # Check if we should skip database setup
        skip_db_setup = os.environ.get('SKIP_DB_SETUP', 'false').lower() == 'true'
        
        if not skip_db_setup:
            # Ensure the parts_of_speech table exists before initializing SQLAlchemy
            # This safely creates only that table if needed without dropping existing data
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            ensure_parts_of_speech_table_exists(db_uri)
            
    # Initialize database
    db.init_app(app)
    # Revert to standard initialization, relying on alembic.ini for directory
    migrate = Migrate(app, db)
    
    # Register blueprints
    app.register_blueprint(bp, url_prefix='/api/v2')
    
    # Register ML blueprint
    try:
        from .ml_routes import ml_bp
        app.register_blueprint(ml_bp)
        logger.info("ML routes registered successfully")
    except ImportError as e:
        logger.warning(f"ML routes not available: {e}")
    
    # Initialize GraphQL
    # Only run migrations and GraphQL setup in the main worker process when reloading
    if not testing and os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        try:
            # Execute database migrations
            skip_db_setup = os.environ.get('SKIP_DB_SETUP', 'false').lower() == 'true'
            if not skip_db_setup:
                with app.app_context():
                    logger.info("Running database migrations...")
                    # Execute migrations through code rather than command
                    # This assumes alembic is properly configured
                    try:
                        alembic_ini_path = "alembic.ini"
                        if os.path.exists(alembic_ini_path):
                            from alembic import command
                            from alembic.config import Config
                            alembic_cfg = Config(alembic_ini_path)
                            command.upgrade(alembic_cfg, "head")
                            logger.info("Database migrations completed successfully.")
                        else:
                            logger.warning(f"Alembic config file '{alembic_ini_path}' not found. Skipping migrations.")
                    except Exception as e:
                        logger.error(f"Error running migrations: {str(e)}")
            else:
                logger.info("Database migrations and setup skipped (SKIP_DB_SETUP=true)")
                
            # Initialize GraphQL
            # schema, context = init_graphql()  # Comment out or replace with proper initialization
            # schema, context = gql_module.init_graphql()  # Use the module reference
        except Exception as e:
            logger.error(f"Error initializing GraphQL: {str(e)}")
    
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
        
        # Add CORS headers
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        response.headers.setdefault('Access-Control-Allow-Headers', '*')
        response.headers.setdefault('Access-Control-Expose-Headers', 'Content-Range, X-Total-Count, X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset')
        
        # Add caching headers for successful GET requests
        if request.method == 'GET' and response.status_code < 400:
            if not response.headers.get('Cache-Control'):
                response.headers['Cache-Control'] = 'public, max-age=60'
        
        # Log response time and update metrics
        if hasattr(g, 'start_time'):
            elapsed_time = time.time() - g.start_time
            
            if has_prometheus:
                pass
            
            logger.info(
                f"Request completed",
                path=request.path,
                method=request.method,
                status=response.status_code,
                time_ms=round(elapsed_time * 1000, 2),
                request_id=getattr(g, 'request_id', 'unknown')
            )
        
        return response
    
    # Blueprints already registered above
    # app.register_blueprint(baybayin_bp, url_prefix='/api/v2')
    
    # Register teardown
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Clean up database resources on app shutdown."""
        db.session.remove()
    
    # Initialize search background tasks
    if not testing:
        # Initialize search tasks if the module has the initialize function
        # if hasattr(log_search_query, 'initialize'):
        #     search_tasks_thread = log_search_query.initialize()
        pass # Add pass if this block becomes empty
        
        # Start system metrics thread if Prometheus is available
        # if has_prometheus:
            # metrics_thread = start_metrics_thread()
    
    # Register search tasks cleanup on app teardown
    @app.teardown_appcontext
    def cleanup_search_tasks(exception=None):
        # Clean up search tasks if the module has the cleanup function
        # if hasattr(log_search_query, 'cleanup'):
        #     log_search_query.cleanup()
        pass # Ensure the function is not empty
    
    # Initialize OpenTelemetry for distributed tracing
    # if config.get('JAEGER_HOST'):
    #     trace.set_tracer_provider(TracerProvider())
    #     jaeger_exporter = JaegerExporter(
    #         agent_host_name=config['JAEGER_HOST'],
    #         agent_port=int(config.get('JAEGER_PORT', 6831)),
    #     )
    #     trace.get_tracer_provider().add_span_processor(
    #         BatchSpanProcessor(jaeger_exporter)
    #     )
    #     
    #     # Instrument Flask
    #     FlaskInstrumentor().instrument_app(app)
    #     RequestsInstrumentor().instrument()
    
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