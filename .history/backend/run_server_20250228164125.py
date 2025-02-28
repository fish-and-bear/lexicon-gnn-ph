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

# API configuration
os.environ['PORT'] = '10000'
os.environ['HOST'] = '0.0.0.0'
os.environ['ALLOWED_ORIGINS'] = '*'

if __name__ == '__main__':
    from backend.app import app
    port = int(os.environ.get('PORT', 10000))
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=True, use_reloader=True) 