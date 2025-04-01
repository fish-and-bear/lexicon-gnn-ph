"""
Local deployment script for the Filipino Dictionary API.
"""

import os
import sys
import subprocess
import time
import logging
import threading
import queue
import socket
import psutil
from pathlib import Path
import requests

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_port_in_use(port: int, host: str = 'localhost') -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    proc.terminate()
                    proc.wait(timeout=5)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    return False

def pre_flight_check() -> bool:
    """Perform comprehensive pre-flight checks."""
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
            
        # Check if PostgreSQL is running
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=int(os.getenv('DB_PORT', 5432)),
                database='postgres',
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'postgres'),
                connect_timeout=5
            )
            conn.close()
        except Exception as e:
            logger.error(f"PostgreSQL is not running or not accessible: {str(e)}")
            return False
            
        # Check required ports
        api_port = int(os.getenv('PORT', 10000))
        metrics_port = int(os.getenv('METRICS_PORT', 9090))
        
        for port in [api_port, metrics_port]:
            if is_port_in_use(port):
                logger.warning(f"Port {port} is in use. Attempting to kill existing process...")
                if not kill_process_on_port(port):
                    logger.error(f"Failed to free port {port}")
                    return False
                    
        # Check disk space
        free_space = psutil.disk_usage(backend_dir).free
        if free_space < 100 * 1024 * 1024:  # 100MB
            logger.error("Insufficient disk space (less than 100MB available)")
            return False
            
        # Check memory
        available_memory = psutil.virtual_memory().available
        if available_memory < 500 * 1024 * 1024:  # 500MB
            logger.error("Insufficient memory (less than 500MB available)")
            return False
            
        # Check required files exist
        required_files = [
            backend_dir / 'app.py',
            backend_dir / 'models' / '__init__.py',
            backend_dir / 'database' / '__init__.py'
        ]
        for file in required_files:
            if not file.exists():
                logger.error(f"Required file not found: {file}")
                return False
                
        # Check write permissions
        try:
            test_file = backend_dir / '.test_write'
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            logger.error(f"No write permission in backend directory: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Pre-flight check failed: {str(e)}")
        return False

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
        'python-dotenv',
        'tenacity',
        'psutil'
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
    try:
        # Load environment variables from .env file if it exists
        from dotenv import load_dotenv
        env_file = backend_dir / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning(f"No .env file found at {env_file}, using defaults")
        
        # Database configuration
        db_config = {
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_NAME': 'fil_dict_db',
            'DB_USER': 'postgres',
            'DB_PASSWORD': 'postgres',
            'DB_MIN_CONNECTIONS': '5',
            'DB_MAX_CONNECTIONS': '20',
            'DB_STATEMENT_TIMEOUT': '30000',
            'DB_IDLE_TIMEOUT': '60000'
        }
        
        # API configuration
        api_config = {
            'FLASK_ENV': 'development',
            'FLASK_APP': 'app:create_app()',
            'FLASK_DEBUG': '1',
            'PORT': '10000',
            'METRICS_PORT': '9090',
            'API_KEYS': 'test_key'
        }
        
        # Set environment variables with defaults
        for config in [db_config, api_config]:
            for key, default in config.items():
                if key not in os.environ:
                    os.environ[key] = default
                    logger.debug(f"Set {key}={default}")
        
        # Log the configuration
        logger.info("Environment configuration:")
        logger.info(f"Database: postgresql://{os.environ['DB_USER']}:***@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}")
        logger.info(f"API Port: {os.environ['PORT']}")
        logger.info(f"Metrics Port: {os.environ['METRICS_PORT']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up environment: {str(e)}")
        return False

def setup_database():
    """Set up the database with required extensions and initial data."""
    try:
        from database import (
            create_database_if_not_exists,
            init_app,
            get_db_config,
            wait_for_postgres
        )
        from flask import Flask
        
        # Wait for PostgreSQL to be ready
        logger.info("Waiting for PostgreSQL to be ready...")
        try:
            wait_for_postgres()
        except Exception as e:
            logger.error(f"PostgreSQL is not ready: {str(e)}")
            logger.error("Please make sure PostgreSQL is running and accessible")
            return False
        
        # Create database if it doesn't exist
        try:
            create_database_if_not_exists()
        except Exception as e:
            logger.error(f"Failed to create database: {str(e)}")
            logger.error("Please check your PostgreSQL permissions and configuration")
            return False
        
        # Create Flask app for database initialization
        app = Flask(__name__)
        
        try:
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
            
            # Initialize database
            with app.app_context():
                init_app(app)
            
            logger.info("Database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            logger.error("Please check your database configuration and permissions")
            return False
            
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        logger.error("Please make sure all dependencies are installed correctly")
        return False
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        logger.error("Please check the error message above and your configuration")
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
    
    # Run pre-flight checks
    logger.info("Running pre-flight checks...")
    if not pre_flight_check():
        logger.error("Pre-flight checks failed")
        return False
    
    # Check dependencies
    logger.info("Checking dependencies...")
    try:
        check_dependencies()
    except Exception as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False
    
    # Set up environment
    logger.info("Setting up environment...")
    if not setup_environment():
        logger.error("Environment setup failed")
        return False
    
    # Set up database
    logger.info("Setting up database...")
    if not setup_database():
        logger.error("Database setup failed")
        return False
    
    # Start metrics server
    logger.info("Starting metrics server...")
    metrics_port = int(os.getenv('METRICS_PORT', 9090))
    try:
        metrics_process = subprocess.Popen(
            [sys.executable, "-c", "from prometheus_client import start_http_server; start_http_server(9090); import time; time.sleep(3600)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
        return False
    
    # Wait for metrics server
    if not wait_for_port(metrics_port):
        logger.error("Metrics server failed to start")
        metrics_process.terminate()
        return False
    
    # Start API server
    logger.info("Starting API server...")
    api_port = int(os.getenv('PORT', 10000))
    
    try:
        # Create and start the API server
        api_script = create_api_script()
        api_process = subprocess.Popen(
            [sys.executable, str(api_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
            text=True
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}")
        metrics_process.terminate()
        if api_script.exists():
            api_script.unlink()
        return False
    
    # Set up output reading
    output_queue = queue.Queue()
    stdout_thread = threading.Thread(target=read_output, args=(api_process, output_queue, False))
    stderr_thread = threading.Thread(target=read_output, args=(api_process, output_queue, True))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # Wait for API server to start
    start_time = time.time()
    server_started = False
    error_output = []
    
    while time.time() - start_time < 30:  # 30 second timeout
        try:
            # Check if process has terminated
            if api_process.poll() is not None:
                logger.error("API server process terminated unexpectedly")
                # Get any remaining output
                while True:
                    try:
                        source, line = output_queue.get_nowait()
                        error_output.append(f"[{source}] {line}")
                    except queue.Empty:
                        break
                logger.error("API server output:")
                for line in error_output:
                    logger.error(line)
                break
            
            # Check for output
            try:
                source, line = output_queue.get_nowait()
                print(f"[{source}] {line}")
                if "Starting API server..." in line:
                    server_started = True
                if "Error:" in line or "Exception:" in line:
                    error_output.append(f"[{source}] {line}")
            except queue.Empty:
                pass
            
            # Check if server is responding
            if server_started and wait_for_port(api_port, timeout=1):
                time.sleep(2)  # Give the server a moment to fully initialize
                break
            
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error while waiting for API server: {str(e)}")
            break
    
    if not wait_for_port(api_port):
        logger.error("API server failed to start")
        if error_output:
            logger.error("Error output from API server:")
            for line in error_output:
                logger.error(line)
        api_process.terminate()
        metrics_process.terminate()
        if api_script.exists():
            api_script.unlink()
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
                if "Error:" in line or "Exception:" in line:
                    logger.error(f"Error in {source}: {line}")
            except queue.Empty:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        metrics_process.terminate()
        if api_script.exists():
            api_script.unlink()
        return True
    finally:
        api_process.terminate()
        metrics_process.terminate()
        if api_script.exists():
            api_script.unlink()

if __name__ == "__main__":
    try:
        if not deploy():
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error during deployment: {str(e)}")
        sys.exit(1) 