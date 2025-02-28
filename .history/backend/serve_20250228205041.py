"""
Server file for the Filipino Dictionary application.
This module uses Waitress to serve the Flask application.
"""

from waitress import serve
from app import app
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '0.0.0.0')
    threads = int(os.getenv('WAITRESS_THREADS', 4))
    
    print(f"Starting server on http://{host}:{port}")
    serve(app, host=host, port=port, threads=threads)