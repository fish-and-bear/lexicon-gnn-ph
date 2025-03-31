import unittest
import json
import logging
from typing import Dict, Any
from flask_testing import TestCase
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPI(TestCase):
    """Test suite for the Filipino Dictionary API."""
    
    def create_app(self):
        """Create Flask test app."""
        app = create_app(test_config={
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'REDIS_URL': 'redis://localhost:6379/1'
        })
        return app
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.client = self.app.test_client()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")
    
    def test_graphql_word_search(self):
        """Test word search functionality."""
        query = """
        query SearchWords($query: String!) {
            searchWords(query: $query) {
                word
                definitions {
                    definition
                }
            }
        }
        """
        variables = {"query": "maganda"}
        
        response = self.client.post(
            '/graphql',
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("data", data)
        self.assertIn("searchWords", data["data"])
        results = data["data"]["searchWords"]
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Verify result structure
        for result in results:
            self.assertIn("word", result)
            self.assertIn("definitions", result)
            self.assertIsInstance(result["definitions"], list)
    
    def test_graphql_word_network(self):
        """Test word network functionality."""
        query = """
        query GetWordNetwork($word: String!) {
            getWordNetwork(word: $word) {
                word
                related {
                    word
                    relation
                }
            }
        }
        """
        variables = {"word": "maganda"}
        
        response = self.client.post(
            '/graphql',
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("data", data)
        self.assertIn("getWordNetwork", data["data"])
        result = data["data"]["getWordNetwork"]
        
        # Verify result structure
        self.assertEqual(result["word"], variables["word"])
        self.assertIn("related", result)
        self.assertIsInstance(result["related"], list)
        
        for related in result["related"]:
            self.assertIn("word", related)
            self.assertIn("relation", related)
    
    def test_graphql_parts_of_speech(self):
        """Test parts of speech query."""
        query = """
        query {
            partsOfSpeech {
                name
                description
                examples {
                    word
                    definition
                }
            }
        }
        """
        
        response = self.client.post(
            '/graphql',
            json={"query": query},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("data", data)
        self.assertIn("partsOfSpeech", data["data"])
        results = data["data"]["partsOfSpeech"]
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # Verify result structure
        for pos in results:
            self.assertIn("name", pos)
            self.assertIn("description", pos)
            self.assertIn("examples", pos)
            self.assertIsInstance(pos["examples"], list)
    
    def test_graphql_etymology_components(self):
        """Test etymology components query."""
        query = """
        query GetWordEtymology($word: String!) {
            getWord(word: $word) {
                word
                etymology {
                    components {
                        text
                        language
                    }
                }
            }
        }
        """
        variables = {"word": "maganda"}
        
        response = self.client.post(
            '/graphql',
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("data", data)
        self.assertIn("getWord", data["data"])
        result = data["data"]["getWord"]
        
        # Verify result structure
        self.assertEqual(result["word"], variables["word"])
        self.assertIn("etymology", result)
        etymology = result["etymology"]
        self.assertIn("components", etymology)
        self.assertIsInstance(etymology["components"], list)
        
        for component in etymology["components"]:
            self.assertIn("text", component)
            self.assertIn("language", component)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
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
        variables = {"query": "maganda"}
        
        # Make multiple requests quickly
        responses = []
        for _ in range(10):
            response = self.client.post(
                '/graphql',
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            responses.append(response)
        
        # Check rate limiting headers
        for response in responses:
            self.assertIn("X-RateLimit-Limit", response.headers)
            self.assertIn("X-RateLimit-Remaining", response.headers)
            self.assertIn("X-RateLimit-Reset", response.headers)
        
        # Verify rate limiting was applied
        rate_limited = any(
            response.status_code == 429
            for response in responses
        )
        self.assertTrue(rate_limited, "Rate limiting was not applied")

if __name__ == "__main__":
    unittest.main(verbosity=2) 