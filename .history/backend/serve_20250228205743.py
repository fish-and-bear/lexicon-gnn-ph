"""
Server file for the Filipino Dictionary application.
This module uses Waitress to serve the Flask application.
"""

from waitress import serve
from app import app, create_app
import os
from dotenv import load_dotenv
import traceback

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

if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', 10000))
        host = os.getenv('HOST', '127.0.0.1')  # Changed from 0.0.0.0 to 127.0.0.1
        threads = int(os.getenv('WAITRESS_THREADS', 4))
        
        print(f"Starting server on http://{host}:{port}")
        
        # Check if app is properly initialized
        print(f"Routes registered: {[r.rule for r in app.url_map.iter_rules()]}")
        
        serve(app, host=host, port=port, threads=threads)
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()