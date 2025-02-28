#!/usr/bin/env python3
"""
Script to run the Flask development server with enhanced logging and error handling.
"""
import os
import sys
from pathlib import Path
import logging
import structlog
from datetime import datetime
import socket
import psutil
import traceback

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / 'server.log', encoding='utf-8')
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
        structlog.dev.ConsoleRenderer()  # Pretty printing for development
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
print("Starting server with enhanced logging...")

def check_port_availability(port: int) -> bool:
    """Check if the port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        print(f"Port {port} is available")
        return True
    except socket.error:
        print(f"Port {port} is not available!")
        process_info = get_process_using_port(port)
        print(f"Process info: {process_info}")
        return False

def find_available_port(start_port: int, max_attempts: int = 5) -> int:
    """Find an available port starting from start_port."""
    print(f"Looking for available port starting from {start_port}...")
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(port):
            print(f"Found available port: {port}")
            return port
    error_msg = f"No available ports found in range {start_port}-{start_port + max_attempts - 1}"
    print(error_msg)
    raise RuntimeError(error_msg)

def get_process_using_port(port: int) -> str:
    """Get information about process using the port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        info = f"Process {proc.name()} (PID: {proc.pid}) is using port {port}"
                        print(info)
                        return info
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return f"No process found using port {port}"
    except Exception as e:
        error_msg = f"Error checking port {port}: {str(e)}"
        print(error_msg)
        return error_msg

def setup_environment():
    """Set up environment variables with validation and logging."""
    try:
        print("Setting up environment variables...")
        # Add the parent directory to Python path
        parent_dir = str(Path(__file__).resolve().parent.parent)
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to Python path")

        # Set and validate environment variables
        env_vars = {
            'PYTHONPATH': parent_dir,
            'FLASK_APP': 'backend.app',
            'FLASK_ENV': 'development',
            'FLASK_DEBUG': '1',
            'DB_NAME': 'fil_dict_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DATABASE_URL': 'postgresql://postgres:postgres@localhost:5432/fil_dict_db',
            'REDIS_ENABLED': 'false'
        }

        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")

        return True
    except Exception as e:
        print(f"Failed to set up environment: {str(e)}")
        traceback.print_exc()
        return False

def check_database_connection():
    """Check if database is accessible."""
    try:
        import psycopg2
        conn_str = os.environ['DATABASE_URL']
        print("Attempting to connect to database...")
        
        conn = psycopg2.connect(conn_str)
        conn.close()
        print("Successfully connected to database")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        print("Checking dependencies...")
        import flask
        import sqlalchemy
        import redis
        import structlog
        print("All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        traceback.print_exc()
        return False

def get_network_info():
    """Get network interface information."""
    try:
        print("Getting network information...")
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Hostname: {hostname}")
        print(f"Local IP: {local_ip}")
        return hostname, local_ip
    except Exception as e:
        print(f"Failed to get network info: {str(e)}")
        traceback.print_exc()
        return "unknown", "unknown"

if __name__ == '__main__':
    startup_time = datetime.now()
    print("\n=== Starting Flask Development Server ===")
    print(f"Start time: {startup_time}")
    print("Current directory:", os.getcwd())
    print("Python version:", sys.version)
    print("=" * 40 + "\n")

    # Step 1: Set up environment
    print("\n[Step 1/6] Setting up environment...")
    if not setup_environment():
        print("Failed to set up environment. Exiting.")
        sys.exit(1)

    # Step 2: Check dependencies
    print("\n[Step 2/6] Checking dependencies...")
    if not check_dependencies():
        print("Failed to verify dependencies. Exiting.")
        sys.exit(1)

    # Step 3: Check database connection
    print("\n[Step 3/6] Checking database connection...")
    if not check_database_connection():
        print("Failed to connect to database. Exiting.")
        sys.exit(1)

    # Step 4: Find available port
    print("\n[Step 4/6] Finding available port...")
    try:
        preferred_port = 10000
        port = find_available_port(preferred_port)
        if port != preferred_port:
            print(f"Warning: Preferred port {preferred_port} not available, using port {port} instead")
    except Exception as e:
        print(f"Failed to find available port: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Get network information
    print("\n[Step 5/6] Getting network information...")
    hostname, local_ip = get_network_info()

    # Step 6: Start Flask application
    print("\n[Step 6/6] Starting Flask application...")
    try:
        from backend.app import app
        print("Successfully imported Flask application")
        
        # Log all registered routes
        print("\nRegistered routes:")
        for rule in app.url_map.iter_rules():
            print(f"  {rule.rule} [{', '.join(rule.methods)}]")

        # Start the server
        print("\nStarting Flask server...")
        print("Server will be accessible at:")
        print(f"  http://127.0.0.1:{port}")
        print(f"  http://{local_ip}:{port}")
        print("\nPress CTRL+C to stop the server\n")
        
        app.run(
            host='0.0.0.0',  # Bind to all interfaces
            port=port,
            debug=True,
            use_reloader=True,
            threaded=True
        )
    except Exception as e:
        print("\nFailed to start Flask server:")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)