"""
Entry point for running the Flask application.

This script correctly manages imports and handles the application creation
and configuration before running the development server.
"""

import os
import sys
from backend.app import create_app

if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 10000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    # Create the application
    app = create_app()
    
    # Print startup message
    print(f"\n📚 Filipino Dictionary API starting on http://{host}:{port}")
    print(f"✅ Debug mode: {'enabled' if debug else 'disabled'}")
    print(f"💻 Press CTRL+C to stop\n")
    
    # Run the application
    app.run(host=host, port=port, debug=debug) 