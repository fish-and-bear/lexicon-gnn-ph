"""
Script to run the Flask development server.
"""

import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables
load_dotenv()

# Set environment variables
os.environ['PYTHONPATH'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['FLASK_APP'] = 'app.py'
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '1'
os.environ['DATABASE_URL'] = 'postgresql://postgres:postgres@localhost:5432/fil_dict_db'
os.environ['REDIS_ENABLED'] = 'false'

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=10000, debug=True) 