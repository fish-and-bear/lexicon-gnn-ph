"""
Server file for the Filipino Dictionary application.
This module uses Waitress to serve the Flask application.
"""

from waitress import serve
from app import app

if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:8000")
    serve(app, host='0.0.0.0', port=8000, threads=4)