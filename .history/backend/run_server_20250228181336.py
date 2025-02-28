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

# Configure logging before anything else
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_port_availability(port: int) -> bool:
    """Check if the port is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except socket.error:
        logger.error(f"Port {port} is not available!")
        process_info = get_process_using_port(port)
        logger.error(process_info)
        return False

def find_available_port(start_port: int, max_attempts: int = 5) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

def get_process_using_port(port: int) -> str:
    """Get information about process using the port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        return f"Process {proc.name()} (PID: {proc.pid}) is using port {port}"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return f"No process found using port {port}"
    except Exception as e:
        return f"Error checking port {port}: {str(e)}"

def setup_environment():
    """Set up environment variables with validation and logging."""
    try:
        # Add the parent directory to Python path
        parent_dir = str(Path(__file__).resolve().parent.parent)
        sys.path.insert(0, parent_dir)
        logger.info(f"Added {parent_dir} to Python path")

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
            logger.info(f"Set {key}={value}")

        return True
    except Exception as e:
        logger.error(f"Failed to set up environment: {str(e)}", exc_info=True)
        return False

def check_database_connection():
    """Check if database is accessible."""
    try:
        import psycopg2
        conn_str = os.environ['DATABASE_URL']
        logger.info("Attempting to connect to database...")
        
        conn = psycopg2.connect(conn_str)
        conn.close()
        logger.info("Successfully connected to database")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}", exc_info=True)
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import flask
        import sqlalchemy
        import redis
        import structlog
        logger.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}", exc_info=True)
        return False

def get_network_info():
    """Get network interface information."""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return hostname, local_ip
    except Exception as e:
        logger.error(f"Failed to get network info: {str(e)}")
        return "unknown", "unknown"

if __name__ == '__main__':
    startup_time = datetime.now()
    logger.info("Starting deployment process...")

    # Step 1: Set up environment
    if not setup_environment():
        sys.exit(1)

    # Step 2: Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Step 3: Check database connection
    if not check_database_connection():
        sys.exit(1)

    # Step 4: Find available port
    try:
        preferred_port = 10000
        port = find_available_port(preferred_port)
        if port != preferred_port:
            logger.warning(f"Preferred port {preferred_port} not available, using port {port} instead")
    except Exception as e:
        logger.error(f"Failed to find available port: {str(e)}")
        sys.exit(1)

    # Step 5: Get network information
    hostname, local_ip = get_network_info()
    logger.info(f"Network Information:")
    logger.info(f"  Hostname: {hostname}")
    logger.info(f"  Local IP: {local_ip}")

    # Step 6: Start Flask application
    try:
        from backend.app import app
        logger.info("Successfully imported Flask application")
        
        # Log all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"Route: {rule.rule} [{', '.join(rule.methods)}]")

        # Start the server
        logger.info("Starting Flask server...")
        logger.info(f"Server will be accessible at:")
        logger.info(f"  http://127.0.0.1:{port}")
        logger.info(f"  http://{local_ip}:{port}")
        
        app.run(
            host='0.0.0.0',  # Bind to all interfaces
            port=port,
            debug=True,
            use_reloader=True,
            threaded=True
        )
    except Exception as e:
        logger.error("Failed to start Flask server", exc_info=True)
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)