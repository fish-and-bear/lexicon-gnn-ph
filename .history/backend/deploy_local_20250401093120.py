"""
Local deployment script for the Filipino Dictionary API.
"""

import os
import sys
import subprocess
import time
import logging
import psycopg2
import threading
import queue
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

from database import get_db_config, create_database_if_not_exists
import requests
from prometheus_client import start_http_server
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        'requests',
        'flask-sqlalchemy',
        'flask-cors',
        'python-dotenv'
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
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    env_file = backend_dir / '.env'
    load_dotenv(env_file)
    
    # Set default environment variables
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
        # Create database if it doesn't exist
        create_database_if_not_exists()
        
        # Create Flask app for database initialization
        app = Flask(__name__)
        
        # Configure Flask-SQLAlchemy
        config = get_db_config()
        app.config['SQLALCHEMY_DATABASE_URI'] = (
            f"postgresql://{config['user']}:{config['password']}@"
            f"{config['host']}:{config['port']}/{config['database']}"
        )
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': config['max_connections'],
            'pool_timeout': 30,
            'pool_recycle': 1800,
            'pool_pre_ping': True,
        }
        
        # Initialize models and create tables
        from models import init_app
        init_app(app)
        
        logger.info("Database setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
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

def create_api_script():
    """Create the API server script."""
    api_script = backend_dir / 'run_api.py'
    api_port = int(os.getenv('PORT', 10000))
    
    with open(api_script, 'w') as f:
        f.write(f'''"""
API server runner script.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from app import create_app
    
    app = create_app()
    logger.info("Starting API server...")
    app.run(
        host='0.0.0.0',
        port={api_port},
        debug=True,
        use_reloader=False  # Disable reloader in subprocess
    )
except Exception as e:
    logger.error(f"Failed to start API server: {{str(e)}}")
    sys.exit(1)
''')
    
    return api_script

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
    
    # Create and start the API server
    api_script = create_api_script()
    api_process = subprocess.Popen(
        [sys.executable, str(api_script)],
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
        api_script.unlink()  # Clean up the temporary script
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
        api_script.unlink()  # Clean up the temporary script
        return True
    finally:
        api_process.terminate()
        metrics_process.terminate()
        if api_script.exists():
            api_script.unlink()  # Clean up the temporary script

if __name__ == "__main__":
    deploy() 