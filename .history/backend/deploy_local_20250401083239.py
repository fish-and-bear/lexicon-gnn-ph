"""
Local deployment script for the Filipino Dictionary API.
Sets up the environment, database, and runs the API server.
"""

import os
import sys
import subprocess
import time
import logging
import psycopg2

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db_config, setup_extensions
from serve import main as serve_main
import threading
from typing import Optional
import requests
from prometheus_client import start_http_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'flask',
        'sqlalchemy',
        'psycopg2-binary',
        'prometheus_client',
        'structlog',
        'marshmallow',
        'unidecode',
        'requests'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.info("Installing missing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

def wait_for_port(port: int, host: str = 'localhost', timeout: int = 30) -> bool:
    """Wait for a port to become available."""
    start_time = time.time()
    while True:
        try:
            requests.get(f"http://{host}:{port}/")
            return True
        except requests.ConnectionError:
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)

def setup_environment():
    """Set up the environment variables."""
    os.environ.setdefault('FLASK_ENV', 'development')
    os.environ.setdefault('FLASK_APP', 'app:create_app()')
    os.environ.setdefault('FLASK_DEBUG', '1')
    
    # Database configuration
    os.environ.setdefault('DB_HOST', 'localhost')
    os.environ.setdefault('DB_PORT', '5432')
    os.environ.setdefault('DB_NAME', 'fil_dict_db')
    os.environ.setdefault('DB_USER', 'postgres')
    os.environ.setdefault('DB_PASSWORD', 'postgres')
    
    # API configuration
    os.environ.setdefault('PORT', '10000')
    os.environ.setdefault('METRICS_PORT', '9090')
    os.environ.setdefault('API_KEYS', 'test_key')
    
    # Performance tuning
    os.environ.setdefault('DB_MIN_CONNECTIONS', '5')
    os.environ.setdefault('DB_MAX_CONNECTIONS', '20')
    os.environ.setdefault('DB_STATEMENT_TIMEOUT', '30000')
    os.environ.setdefault('DB_IDLE_TIMEOUT', '60000')

def setup_database():
    """Set up the database with required extensions and initial data."""
    try:
        # Initialize database
        init_db()
        
        # Get database connection
        config = get_db_config()
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        # Set up extensions and indexes
        setup_extensions(conn)
        
        conn.close()
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False

def start_metrics_server(port: int = 9090):
    """Start the Prometheus metrics server."""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
        return False

def start_api_server(port: int = 10000):
    """Start the API server."""
    try:
        serve_main()
        return True
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        return False

def run_health_check() -> bool:
    """Run basic health check on the API."""
    try:
        # Check API
        response = requests.get("http://localhost:10000/api/v2/statistics")
        if not response.ok:
            logger.error("API health check failed")
            return False
        
        # Check metrics
        response = requests.get("http://localhost:9090/metrics")
        if not response.ok:
            logger.error("Metrics health check failed")
            return False
        
        logger.info("Health check passed")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def deploy():
    """Deploy the API locally."""
    logger.info("Starting local deployment")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    check_dependencies()
    
    # Set up environment
    logger.info("Setting up environment...")
    setup_environment()
    
    # Set up database
    logger.info("Setting up database...")
    if not setup_database():
        logger.error("Database setup failed")
        return False
    
    # Start metrics server
    logger.info("Starting metrics server...")
    metrics_thread = threading.Thread(
        target=start_metrics_server,
        args=(int(os.getenv('METRICS_PORT', 9090)),),
        daemon=True
    )
    metrics_thread.start()
    
    # Wait for metrics server
    if not wait_for_port(int(os.getenv('METRICS_PORT', 9090))):
        logger.error("Metrics server failed to start")
        return False
    
    # Start API server
    logger.info("Starting API server...")
    api_thread = threading.Thread(
        target=start_api_server,
        args=(int(os.getenv('PORT', 10000)),),
        daemon=True
    )
    api_thread.start()
    
    # Wait for API server
    if not wait_for_port(int(os.getenv('PORT', 10000))):
        logger.error("API server failed to start")
        return False
    
    # Run health check
    logger.info("Running health check...")
    if not run_health_check():
        logger.error("Health check failed")
        return False
    
    logger.info("Deployment completed successfully")
    logger.info("API server running on http://localhost:10000")
    logger.info("Metrics server running on http://localhost:9090")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return True

if __name__ == "__main__":
    deploy() 