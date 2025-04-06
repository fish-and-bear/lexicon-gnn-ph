"""
Runner script for the Filipino Dictionary API.
"""

import os
import sys
from backend import create_app

if __name__ == '__main__':
    # Create the Flask application
    app = create_app()
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 10000))
    debug = os.getenv('ENVIRONMENT', 'development') == 'development'
    print(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True) 