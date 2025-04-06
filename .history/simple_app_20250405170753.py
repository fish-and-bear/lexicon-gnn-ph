"""
Simple Flask application for testing the API without dependencies.
"""

from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running with minimal dependencies',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/v2/test', methods=['GET'])
def test_endpoint():
    """Test endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'API test endpoint is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'message': 'Filipino Dictionary API',
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting simple Flask app on http://0.0.0.0:10000")
    app.run(host='0.0.0.0', port=10000, debug=True) 