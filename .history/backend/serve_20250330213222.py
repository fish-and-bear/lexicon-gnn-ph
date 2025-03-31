"""
Server entry point for the Filipino Dictionary API.
"""

import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from backend.app import app
from backend.migrate import verify_schema_compatibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the server."""
    try:
        # Verify database schema
        if not verify_schema_compatibility():
            logger.error("Database schema verification failed")
            sys.exit(1)
        
        logger.info("Database schema verified successfully")
        
        # Start the server
        port = int(os.getenv('PORT', 10000))
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
