"""
Standalone Flask application for the Filipino Dictionary API.
This version can be run directly without package imports.
"""

import os
import sys
import logging
from flask import Flask, redirect, jsonify
from flask_cors import CORS
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create a simple Flask application with minimal dependencies."""
    app = Flask(__name__)
    CORS(app)
    
    # Root route redirect to test
    @app.route('/')
    def index():
        return redirect('/api/test')
    
    # Test API endpoint
    @app.route('/api/test', methods=['GET'])
    def test_api():
        """Simple test endpoint to verify API is working."""
        return jsonify({
            'status': 'success',
            'message': 'API is working properly!',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    # Health check
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 