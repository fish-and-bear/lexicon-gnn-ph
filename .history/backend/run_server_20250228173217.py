#!/usr/bin/env python3
"""
Script to run the Flask development server.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

# Set environment variables
os.environ['PYTHONPATH'] = parent_dir
os.environ['FLASK_APP'] = 'backend.app'
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '1'

# Database configuration
os.environ['DB_NAME'] = 'fil_dict_db'
os.environ['DB_USER'] = 'postgres'
os.environ['DB_PASSWORD'] = 'postgres'
os.environ['DB_HOST'] = 'localhost'
os.environ['DB_PORT'] = '5432'
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/fil_dict_db'

# Redis configuration
os.environ['REDIS_ENABLED'] = 'false'

if __name__ == '__main__':
    try:
        from backend.app import app
        print("Starting Flask server...")
        print("Environment variables:")
        print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
        print(f"FLASK_APP: {os.environ['FLASK_APP']}")
        print(f"DATABASE_URL: {os.environ['DATABASE_URL']}")
        app.run(host='127.0.0.1', port=10000, debug=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)