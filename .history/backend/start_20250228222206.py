"""
Entry point for the Filipino Dictionary application.
"""

import os
import sys
import structlog
from dotenv import load_dotenv

# Set up logging
logger = structlog.get_logger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main entry point for the application."""
    try:
        # Add the backend directory to Python path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        # Import and run the server
        from run_server import run_server
        success = run_server()
        
        return 0 if success else 1
    except Exception as e:
        logger.error(
            "Application startup failed",
            error=str(e),
            error_type=type(e).__name__
        )
        return 1

if __name__ == '__main__':
    sys.exit(main()) 