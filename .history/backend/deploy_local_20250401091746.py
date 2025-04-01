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
import threading
import queue

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from database import init_db, get_db_config, setup_extensions
from serve import main as serve_main
from typing import Optional
import requests
from prometheus_client import start_http_server
from flask import Flask
from app import create_app
from models import Base
from sqlalchemy import create_engine

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
            response = requests.get(f"http://{host}:{port}/", timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
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
        # Get database configuration
        config = get_db_config()
        
        # Create database URL
        db_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        
        # Create engine with retry logic
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                engine = create_engine(db_url, pool_pre_ping=True)
                # Test connection
                with engine.connect() as conn:
                    conn.execute("SELECT 1")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Database connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
        
        # Drop all tables and recreate them
        logger.info("Recreating database schema...")
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        
        # Get database connection
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        # Set up extensions and indexes
        logger.info("Setting up database extensions and indexes...")
        setup_extensions(conn)
        
        conn.close()
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False

def run_health_check() -> bool:
    """Run basic health check on the API."""
    try:
        # Check API root endpoint with retries
        max_retries = 3
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.get("http://localhost:10000/", timeout=5)
                if response.status_code == 200:
                    break
                if attempt < max_retries - 1:
                    logger.warning(f"API health check attempt {attempt + 1} failed with status code {response.status_code}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API health check failed with status code {response.status_code}")
                    return False
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API health check attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API health check failed: {str(e)}")
                    return False
        
        # Check metrics with retries
        for attempt in range(max_retries):
            try:
                response = requests.get("http://localhost:9090/metrics", timeout=5)
                if response.status_code == 200:
                    break
                if attempt < max_retries - 1:
                    logger.warning(f"Metrics health check attempt {attempt + 1} failed with status code {response.status_code}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Metrics health check failed with status code {response.status_code}")
                    return False
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Metrics health check attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Metrics health check failed: {str(e)}")
                    return False
        
        logger.info("Health check passed")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def read_output(process, output_queue, is_stderr=False):
    """Read output from a process and put it in a queue."""
    stream = process.stderr if is_stderr else process.stdout
    while True:
        line = stream.readline()
        if not line and process.poll() is not None:
            break
        if line:
            output_queue.put(('stderr' if is_stderr else 'stdout', line.strip()))
    
    # Read any remaining output
    for line in stream:
        if line:
            output_queue.put(('stderr' if is_stderr else 'stdout', line.strip()))

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
    
    # Start metrics server in a separate process
    logger.info("Starting metrics server...")
    metrics_port = int(os.getenv('METRICS_PORT', 9090))
    metrics_process = subprocess.Popen(
        [sys.executable, "-c", "from prometheus_client import start_http_server; start_http_server(9090); import time; time.sleep(3600)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for metrics server
    if not wait_for_port(metrics_port):
        logger.error("Metrics server failed to start")
        metrics_process.terminate()
        return False
    
    # Start API server in a separate process
    logger.info("Starting API server...")
    api_port = int(os.getenv('PORT', 10000))
    
    # Create a temporary script to run the API server
    api_script = os.path.join(backend_dir, 'run_api.py')
    with open(api_script, 'w') as f:
        f.write(f'''import os
import sys
import logging
sys.path.insert(0, "{backend_dir}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app import create_app
    app = create_app()
    logger.info("Starting API server...")
    app.run(host='0.0.0.0', port={api_port}, debug=True)
except Exception as e:
    logger.error(f"Failed to start API server: {str(e)}")
    sys.exit(1)
''')
    
    # Start the API server using the script
    api_process = subprocess.Popen(
        [sys.executable, api_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        text=True
    )
    
    # Set up output reading for both stdout and stderr
    output_queue = queue.Queue()
    stdout_thread = threading.Thread(target=read_output, args=(api_process, output_queue, False))
    stderr_thread = threading.Thread(target=read_output, args=(api_process, output_queue, True))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for API server to start with timeout
    start_time = time.time()
    server_started = False
    while time.time() - start_time < 30:  # 30 second timeout
        try:
            # Check if process has terminated
            if api_process.poll() is not None:
                logger.error("API server process terminated unexpectedly")
                # Get any remaining output
                while True:
                    try:
                        source, line = output_queue.get_nowait()
                        print(f"[{source}] {line}")
                    except queue.Empty:
                        break
                break
                
            # Check for output
            try:
                source, line = output_queue.get_nowait()
                print(f"[{source}] {line}")
                if "Starting API server..." in line:
                    server_started = True
            except queue.Empty:
                pass
                
            # Check if server is responding
            if server_started and wait_for_port(api_port, timeout=1):
                # Give the server a moment to fully initialize
                time.sleep(2)
                break
                
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error while waiting for API server: {str(e)}")
            break
    
    if not wait_for_port(api_port):
        logger.error("API server failed to start")
        api_process.terminate()
        metrics_process.terminate()
        os.remove(api_script)  # Clean up the temporary script
        return False
    
    # Run health check with retries
    logger.info("Running health check...")
    max_health_check_retries = 3
    for attempt in range(max_health_check_retries):
        if run_health_check():
            break
        if attempt < max_health_check_retries - 1:
            logger.warning(f"Health check attempt {attempt + 1} failed, retrying...")
            time.sleep(2)
        else:
            logger.error("Health check failed after all retries")
            api_process.terminate()
            metrics_process.terminate()
            os.remove(api_script)  # Clean up the temporary script
            return False
    
    logger.info("Deployment completed successfully")
    logger.info("API server running on http://localhost:10000")
    logger.info("Metrics server running on http://localhost:9090")
    
    try:
        # Keep the main process alive and show output
        while True:
            try:
                source, line = output_queue.get_nowait()
                print(f"[{source}] {line}")
            except queue.Empty:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        metrics_process.terminate()
        os.remove(api_script)  # Clean up the temporary script
        return True
    finally:
        api_process.terminate()
        metrics_process.terminate()
        if os.path.exists(api_script):
            os.remove(api_script)  # Clean up the temporary script

if __name__ == "__main__":
    deploy() 