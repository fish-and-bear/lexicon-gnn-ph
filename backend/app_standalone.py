"""
Standalone Flask application for the Filipino Dictionary API.
This version can be run directly without package imports.
"""

import os
import sys
import logging
from flask import Flask, redirect, jsonify, request
from flask_cors import CORS
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create a simple Flask application with minimal dependencies."""
    app = Flask(__name__)
    
    # Configure CORS to allow API requests
    cors_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    CORS(app, resources={r"/*": {"origins": cors_origins}})
    
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
        db_status = "not_available"
        
        try:
            # Here you would check database connectivity
            # For now, we'll simulate it
            db_status = "connected"
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            db_status = "error"
            
        return jsonify({
            'status': 'healthy',
            'database': db_status,
            'environment': os.getenv('FLASK_ENV', 'development'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    # Configuration info endpoint
    @app.route('/api/config', methods=['GET'])
    def config_info():
        """Return safe configuration information."""
        config = {
            'app_version': '1.0.0',
            'environment': os.getenv('FLASK_ENV', 'development'),
            'database_host': os.getenv('DB_HOST', 'localhost'),
            'database_name': os.getenv('DB_NAME', 'fil_dict_db'),
            'cors_enabled': True,
            'allowed_origins': cors_origins,
            'redis_enabled': os.getenv('REDIS_ENABLED', 'false').lower() == 'true',
            'cache_expiration': int(os.getenv('CACHE_EXPIRATION', 3600)),
            'features': {
                'search_api': True,
                'word_details': True,
                'baybayin': True,
                'etymology': True,
                'graphql': True
            }
        }
        return jsonify(config)
    
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
            },
            'tubig': {
                'id': 3,
                'lemma': 'tubig',
                'normalized_lemma': 'tubig',
                'language_code': 'tl',
                'has_baybayin': True,
                'baybayin_form': 'ᜆᜓᜊᜒᜄ᜔',
                'definitions': [
                    {
                        'id': 3,
                        'definition_text': 'A colorless, transparent, odorless liquid that forms oceans, lakes, rivers, and rain.',
                        'part_of_speech': 'n'
                    }
                ],
                'etymologies': [
                    {
                        'id': 3,
                        'etymology_text': 'From Proto-Malayo-Polynesian *tubiq',
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
    
    # Search endpoint (demo)
    @app.route('/api/search', methods=['GET'])
    def search():
        """Demo search endpoint."""
        query = request.args.get('q', '')
        language = request.args.get('language', None)
        
        if not query:
            return jsonify({
                'error': 'Missing query parameter',
                'message': 'Please provide a search query with the q parameter'
            }), 400
            
        sample_words = {
            'aso': {
                'id': 1,
                'lemma': 'aso',
                'language_code': 'tl',
                'has_baybayin': True,
                'definitions_short': ['A domesticated carnivorous mammal']
            },
            'bahay': {
                'id': 2,
                'lemma': 'bahay',
                'language_code': 'tl',
                'has_baybayin': True,
                'definitions_short': ['A building for human habitation']
            },
            'tubig': {
                'id': 3,
                'lemma': 'tubig',
                'language_code': 'tl',
                'has_baybayin': True,
                'definitions_short': ['A colorless, transparent liquid']
            }
        }
        
        # Filter by query
        results = {k: v for k, v in sample_words.items() if query.lower() in k.lower()}
        
        # Filter by language if provided
        if language:
            results = {k: v for k, v in results.items() if v['language_code'] == language}
            
        return jsonify({
            'query': query,
            'total': len(results),
            'results': list(results.values())
        })
    
    return app

app = create_app()

if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask application on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 