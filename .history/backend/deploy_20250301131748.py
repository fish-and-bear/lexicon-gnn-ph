"""
Deployment script for the Filipino Dictionary API.
This script handles deployment for both development (Waitress) and production (Gunicorn) environments.
"""

import os
import sys
import subprocess
import platform
import structlog
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()

# Determine base directory
BASE_DIR = Path(__file__).resolve().parent

def setup_directories():
    """Create necessary directories."""
    dirs = ['logs', 'certs', 'data']
    for dir_name in dirs:
        dir_path = BASE_DIR.parent / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")

def run_migrations():
    """Run database migrations."""
    try:
        logger.info("Running database migrations")
        from migrate import run_migrations
        success = run_migrations()
        if not success:
            logger.error("Database migration failed")
            return False
        logger.info("Database migrations completed successfully")
        return True
    except Exception as e:
        logger.error(
            "Migration error",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

def start_server():
    """Start the appropriate server based on environment."""
    try:
        # Determine environment
        is_production = os.getenv('FLASK_ENV', 'development') == 'production'
        is_windows = platform.system() == 'Windows'
        
        if is_windows or not is_production:
            return start_waitress()
        else:
            return start_gunicorn()
    except Exception as e:
        logger.error(
            "Failed to start server",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

def start_waitress():
    """Start Waitress server (Windows/development)."""
    try:
        logger.info("Starting Waitress server")
        
        # Build command
        cmd = [
            sys.executable,
            str(BASE_DIR / 'serve.py')
        ]
        
        # Add environment variables
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(BASE_DIR),
            'FLASK_APP': 'app.py',
            'FLASK_ENV': os.getenv('FLASK_ENV', 'development'),
        })
        
        # Start server
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Log process information
        logger.info(
            "Waitress started",
            pid=process.pid,
            command=' '.join(cmd)
        )
        
        return True
    except Exception as e:
        logger.error(
            "Failed to start Waitress",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

def start_gunicorn():
    """Start Gunicorn server (Linux/production)."""
    try:
        logger.info("Starting Gunicorn server")
        
        # Check if supervisor is available
        use_supervisor = os.getenv('USE_SUPERVISOR', 'true').lower() == 'true'
        
        if use_supervisor:
            return start_with_supervisor()
        
        # Build Gunicorn command
        cmd = [
            'gunicorn',
            '--config', str(BASE_DIR / 'gunicorn_config.py'),
            '--chdir', str(BASE_DIR),
            'app:app'
        ]
        
        # Add environment variables
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(BASE_DIR),
            'FLASK_APP': 'app.py',
            'FLASK_ENV': 'production',
        })
        
        # Start Gunicorn
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Log process information
        logger.info(
            "Gunicorn started",
            pid=process.pid,
            command=' '.join(cmd)
        )
        
        return True
    except Exception as e:
        logger.error(
            "Failed to start Gunicorn",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

def start_with_supervisor():
    """Start the application using supervisor."""
    try:
        logger.info("Starting with supervisor")
        
        # Build supervisor command
        cmd = [
            'supervisord',
            '-c', str(BASE_DIR / 'supervisor.conf')
        ]
        
        # Start supervisor
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Log process information
        logger.info(
            "Supervisor started",
            pid=process.pid,
            command=' '.join(cmd)
        )
        
        # Check status
        status_cmd = [
            'supervisorctl',
            '-c', str(BASE_DIR / 'supervisor.conf'),
            'status'
        ]
        
        # Give supervisor a moment to start
        import time
        time.sleep(2)
        
        status = subprocess.run(
            status_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(
            "Supervisor status",
            stdout=status.stdout,
            stderr=status.stderr
        )
        
        return True
    except Exception as e:
        logger.error(
            "Failed to start with supervisor",
            error=str(e),
            error_type=type(e).__name__
        )
        return False

def main():
    """Main deployment function."""
    try:
        # Create necessary directories
        setup_directories()
        
        # Run database migrations
        if not run_migrations():
            return 1
        
        # Start server
        if not start_server():
            return 1
        
        logger.info("Deployment completed successfully")
        return 0
        
    except Exception as e:
        logger.error(
            "Deployment failed",
            error=str(e),
            error_type=type(e).__name__
        )
        return 1

if __name__ == '__main__':
    sys.exit(main()) 