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
    
    # Words endpoint (demo)
    @app.route('/api/words/<word>', methods=['GET'])
    def get_word(word):
        """Demo word lookup endpoint."""
        sample_words = {
            'aso': {
                'id': 1,
                'lemma': 'aso',
                'normalized_lemma': 'aso',
                'language_code': 'tl',
                'has_baybayin': True,
                'baybayin_form': 'ᜀᜐᜓ',
                'definitions': [
                    {
                        'id': 1,
                        'definition_text': 'A domesticated carnivorous mammal (Canis familiaris) related to the wolves.',
                        'part_of_speech': 'n'
                    }
                ],
                'etymologies': [
                    {
                        'id': 1,
                        'etymology_text': 'From Proto-Malayo-Polynesian *asu',
                        'language_codes': 'pmp'
                    }
                ]
            },
            'bahay': {
                'id': 2,
                'lemma': 'bahay',
                'normalized_lemma': 'bahay',
                'language_code': 'tl',
                'has_baybayin': True,
                'baybayin_form': 'ᜊᜑᜌ᜔',
                'definitions': [
                    {
                        'id': 2,
                        'definition_text': 'A building for human habitation; a dwelling.',
                        'part_of_speech': 'n'
                    }
                ],
                'etymologies': [
                    {
                        'id': 2,
                        'etymology_text': 'From Proto-Malayo-Polynesian *balay',
                        'language_codes': 'pmp'
                    }
                ]
            }
        }
        
        if word.lower() in sample_words:
            return jsonify(sample_words[word.lower()])
        else:
            return jsonify({
                'error': 'Word not found',
                'suggestions': list(sample_words.keys())
            }), 404
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 