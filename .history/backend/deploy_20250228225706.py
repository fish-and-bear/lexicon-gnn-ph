"""
Deployment script for the Filipino Dictionary API.
"""

import os
import sys
import subprocess
import structlog
from dotenv import load_dotenv
from pathlib import Path

# Set up logging
logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()

def setup_directories():
    """Create necessary directories."""
    dirs = ['logs', 'certs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

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

def start_gunicorn():
    """Start Gunicorn server."""
    try:
        logger.info("Starting Gunicorn server")
        
        # Build Gunicorn command
        cmd = [
            'gunicorn',
            '--config', 'backend/gunicorn_config.py',
            '--chdir', 'backend',
            'app:app'
        ]
        
        # Add environment variables
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': 'backend',
            'FLASK_APP': 'app.py',
            'FLASK_ENV': os.getenv('FLASK_ENV', 'production'),
            'DB_NAME': os.getenv('DB_NAME', 'fil_dict_db'),
            'DB_USER': os.getenv('DB_USER', 'postgres'),
            'DB_PASSWORD': os.getenv('DB_PASSWORD', 'postgres'),
            'DB_HOST': os.getenv('DB_HOST', 'localhost'),
            'DB_PORT': os.getenv('DB_PORT', '5432'),
            'REDIS_ENABLED': os.getenv('REDIS_ENABLED', 'false')
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

def main():
    """Main deployment function."""
    try:
        # Create necessary directories
        setup_directories()
        
        # Run database migrations
        if not run_migrations():
            return 1
        
        # Start Gunicorn server
        if not start_gunicorn():
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