import unittest
import requests
import json
import logging
from typing import Dict, Any
from flask_testing import TestCase
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPIBasic(unittest.TestCase):
    """Basic test suite for the Filipino Dictionary API."""
    
    BASE_URL = "http://localhost:10000"
    
    def setUp(self):
        """Set up test environment."""
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = requests.get(f"{self.BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_graphql_introspection(self):
        """Test GraphQL schema introspection."""
        query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    description
                }
            }
        }
        """
        
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("__schema", data["data"])
    
    def test_error_handling(self):
        """Test error handling for invalid queries."""
        # Test invalid GraphQL query
        invalid_query = """
        query InvalidQuery {
            invalidField
        }
        """
        
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": invalid_query},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)  # GraphQL always returns 200
        data = response.json()
        self.assertIn("errors", data)
    
    def test_cors_headers(self):
        """Test CORS headers."""
        response = requests.options(
            f"{self.BASE_URL}/graphql",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertIn("Access-Control-Allow-Methods", response.headers)
        self.assertIn("Access-Control-Allow-Headers", response.headers)

if __name__ == "__main__":
    unittest.main(verbosity=2) 