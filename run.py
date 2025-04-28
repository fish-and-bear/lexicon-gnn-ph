#!/usr/bin/env python3
"""
Run the Flask app for the dictionary API
"""
import os
import sys

# Set environment variable to skip migrations before importing any modules
os.environ["SKIP_DB_SETUP"] = "true"

# Import our inline source_standardization module first
# This ensures the module is available to all other modules that need it
import backend.inline_source_standardization

# Now import the app
from backend.app import create_app

# Create and run the application
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
    print(f"🔄 Database migrations: disabled")
    print(f"💻 Press CTRL+C to stop\n")
    
    # Run the application - Uncommented for local development
    app.run(host=host, port=port, debug=debug)
    # pass # pass is no longer needed as app.run blocks 