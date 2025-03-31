"""
Server entry point for the Filipino Dictionary API.
"""

import os
import logging
from app import create_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the server."""
    # Create the Flask application
    app = create_app()
    
    # Get port from environment variable or use default
    port = int(os.getenv('PORT', 10000))
    
    # Run the application
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
