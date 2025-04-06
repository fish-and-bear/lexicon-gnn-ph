"""
Runner script for the Filipino Dictionary API.
This script launches the complete API with all original routes and functionality.
"""

import os
import sys
from flask import Flask
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path to help with imports
sys.path.insert(0, os.path.abspath('.'))

# Import the original app
try:
    from backend.app import create_app
    logger.info("Successfully imported create_app function")
except ImportError as e:
    logger.error(f"Error importing create_app: {e}")
    raise

if __name__ == '__main__':
    try:
        # Create the full application
        app = create_app()
        
        # Set host and port
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 10000))
        
        # Run the application
        logger.info(f"Starting Flask application on http://{host}:{port}")
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True) 
 