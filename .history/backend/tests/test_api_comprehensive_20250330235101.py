"""
Comprehensive test suite for the Filipino Dictionary API.
"""

import unittest
import json
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from flask import Flask
from flask_testing import TestCase
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from tests.run_tests import TimedTestCase
from models import db

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPI(TimedTestCase, TestCase):
    """Comprehensive test suite for the Filipino Dictionary API."""
    
    def create_app(self):
        """Create Flask test app."""
        app = create_app(test_config={
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'postgresql://postgres:postgres@localhost:5432/fil_dict_db',
            'REDIS_URL': 'redis://localhost:6379/0',
            'RATE_LIMIT_ENABLED': True
        })
        return app
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Create all database tables
        with self.app.app_context():
            db.create_all()
        self.client = self.app.test_client()
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def test_health_endpoint(self):
        """Test API health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_graphql_introspection(self):
        """Test GraphQL schema introspection."""
        query = """
        query {
            __schema {
                types {
                    name
                }
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('__schema', data['data'])
    
    def test_word_search(self):
        """Test word search functionality."""
        query = """
        query {
            searchWords(query: "maganda") {
                word
                definitions {
                    definition
                }
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('searchWords', data['data'])
    
    def test_word_lookup(self):
        """Test word lookup functionality."""
        query = """
        query {
            getWord(word: "maganda") {
                word
                definitions {
                    definition
                    examples {
                        example
                    }
                }
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('getWord', data['data'])
    
    def test_word_network(self):
        """Test word network functionality."""
        query = """
        query {
            getWordNetwork(word: "maganda") {
                word
                related {
                    word
                    relation
                }
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('getWordNetwork', data['data'])
    
    def test_baybayin_processing(self):
        """Test Baybayin text processing."""
        query = """
        query {
            baybayinWords(text: "ᜋᜄᜈ᜔ᜇ") {
                original
                baybayin
                latin
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('baybayinWords', data['data'])
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        test_cases = [
            {
                'query': """
                query {
                    invalidField {
                        field
                    }
                }
                """,
                'expected_error': 'Cannot query field'
            },
            {
                'query': """
                query {
                    getWord(word: "") {
                        word
                    }
                }
                """,
                'expected_error': 'Invalid word'
            }
        ]
        
        for case in test_cases:
            response = self.client.post(
                '/graphql',
                json={'query': case['query']},
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)  # GraphQL always returns 200
            data = json.loads(response.data)
            self.assertIn('errors', data)
            error_message = data['errors'][0]['message']
            self.assertTrue(
                case['expected_error'] in error_message,
                f"Expected error '{case['expected_error']}' not found in '{error_message}'"
            )
    
    def test_cors_headers(self):
        """Test CORS headers in responses."""
        response = self.client.options(
            '/graphql',
            headers={
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            }
        )
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertIn('Access-Control-Allow-Methods', response.headers)
        self.assertIn('Access-Control-Allow-Headers', response.headers)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        query = """
        query {
            searchWords(query: "maganda") {
                word
            }
        }
        """
        # Make multiple requests quickly
        responses = []
        for _ in range(10):
            response = self.client.post(
                '/graphql',
                json={'query': query},
                headers=self.headers
            )
            responses.append(response.status_code)
        
        # Check if any requests were rate limited
        rate_limited = 429 in responses
        self.assertTrue(rate_limited, "Rate limiting was not applied")
    
    def test_response_format(self):
        """Test consistency of response formats."""
        query = """
        query {
            searchWords(query: "maganda") {
                word
                definitions {
                    definition
                }
            }
        }
        """
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('data', data)
        self.assertIn('searchWords', data['data'])
        self.assertIsInstance(data['data']['searchWords'], list)
        
        # Check word structure
        if data['data']['searchWords']:
            word = data['data']['searchWords'][0]
            self.assertIn('word', word)
            self.assertIn('definitions', word)
            self.assertIsInstance(word['definitions'], list)

if __name__ == '__main__':
    unittest.main() 