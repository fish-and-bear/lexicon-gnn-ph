"""
Server file for the Filipino Dictionary application.
This module provides a unified server interface that can use either Waitress (Windows/development)
or Gunicorn (Linux/production) based on the environment.
"""

import os
import sys
import platform
import traceback
from dotenv import load_dotenv
from app import app, create_app

# Load environment variables from .env file
load_dotenv()

# For debugging
print("Environment variables loaded:")
print(f"DB_HOST: {os.getenv('DB_HOST', 'Not set')}")
print(f"DB_PORT: {os.getenv('DB_PORT', 'Not set')}")
print(f"DB_NAME: {os.getenv('DB_NAME', 'Not set')}")
print(f"DB_USER: {os.getenv('DB_USER', 'Not set')}")
print(f"DB_PASSWORD: {'*****' if os.getenv('DB_PASSWORD') else 'Not set'}")
print(f"REDIS_URL: {os.getenv('REDIS_URL', 'Not set')}")
print(f"ENVIRONMENT: {os.getenv('FLASK_ENV', 'development')}")
print(f"Platform: {platform.system()}")

def run_with_waitress():
    """Run the application with Waitress (Windows-friendly)."""
    from waitress import serve
    
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '127.0.0.1')
    threads = int(os.getenv('WAITRESS_THREADS', 4))
    
    print(f"Starting Waitress server on http://{host}:{port} with {threads} threads")
    print(f"Routes registered: {[r.rule for r in app.url_map.iter_rules()]}")
    
    serve(app, host=host, port=port, threads=threads)

def run_with_gunicorn():
    """Run the application with Gunicorn (Linux/production)."""
    # We don't actually run Gunicorn here - this would be handled by
    # supervisor or systemd. This function exists for documentation.
    print("Gunicorn should be started via supervisor or directly:")
    print("  gunicorn --config gunicorn_config.py app:app")
    
    # Check if app is properly initialized
    print(f"Routes registered: {[r.rule for r in app.url_map.iter_rules()]}")
    
    # Return success as we're just providing info
    return True

if __name__ == '__main__':
    try:
        # Determine environment
        is_production = os.getenv('FLASK_ENV', 'development') == 'production'
        is_windows = platform.system() == 'Windows'
        
        # Choose server based on environment
        if is_windows or not is_production:
            run_with_waitress()
        else:
            # On Linux production, we'd typically use Gunicorn via supervisor
            # This is just for documentation
            run_with_gunicorn()
            print("For production deployment, use supervisor or systemd to manage Gunicorn.")
            print("See supervisor.conf and gunicorn_config.py for configuration.")
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()