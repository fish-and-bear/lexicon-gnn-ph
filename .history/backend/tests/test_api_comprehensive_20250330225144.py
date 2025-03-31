import unittest
import requests
import json
import logging
from typing import Dict, Any
from datetime import datetime, timezone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPIComprehensive(unittest.TestCase):
    """Comprehensive test suite for the Filipino Dictionary API."""
    
    BASE_URL = "http://localhost:10000"
    
    def setUp(self):
        """Set up test environment."""
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": "test-api-key"  # Add API key for authenticated endpoints
        }
        self.test_word = "bahay"
        self.test_baybayin = "ᜊᜑᜌ᜔"
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertIn("status", data)
        self.assertIn("features", data)
        self.assertIn("endpoints", data)
        self.assertIn("supported_languages", data)
    
    def test_word_endpoints(self):
        """Test word-related endpoints."""
        # Test getting a specific word
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("lemma", data)
        self.assertIn("language_code", data)
        self.assertIn("definitions", data)
        
        # Test word not found
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/nonexistentword",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 404)
        
        # Test word relations
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}/relations",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("relations", data)
        
        # Test word affixations
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}/affixations",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("affixations", data)
        
        # Test word etymology
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}/etymology",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("etymologies", data)
        
        # Test word pronunciation
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}/pronunciation",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pronunciations", data)
    
    def test_search_endpoint(self):
        """Test search functionality with various modes."""
        search_modes = ["exact", "fuzzy", "phonetic", "baybayin"]
        for mode in search_modes:
            response = requests.get(
                f"{self.BASE_URL}/api/v2/search",
                params={"q": self.test_word, "mode": mode},
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("words", data)
            self.assertIn("total_count", data)
            
            # Test pagination
            response = requests.get(
                f"{self.BASE_URL}/api/v2/search",
                params={
                    "q": self.test_word,
                    "mode": mode,
                    "page": 1,
                    "per_page": 5
                },
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertLessEqual(len(data["words"]), 5)
    
    def test_baybayin_endpoints(self):
        """Test Baybayin-related endpoints."""
        # Test Baybayin processing
        response = requests.post(
            f"{self.BASE_URL}/api/v2/baybayin/process",
            json={
                "text": self.test_baybayin,
                "mode": "romanize",
                "validate_text": True,
                "include_metadata": True,
                "include_pronunciation": True
            },
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("result", data)
        self.assertIn("metadata", data)
        self.assertIn("pronunciation", data)
        
        # Test invalid Baybayin text
        response = requests.post(
            f"{self.BASE_URL}/api/v2/baybayin/process",
            json={
                "text": "invalid_baybayin",
                "mode": "romanize",
                "validate_text": True
            },
            headers=self.headers
        )
        self.assertEqual(response.status_code, 400)
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint."""
        response = requests.get(
            f"{self.BASE_URL}/api/v2/statistics",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_words", data)
        self.assertIn("words_by_language", data)
        self.assertIn("words_by_pos", data)
        self.assertIn("words_with_baybayin", data)
        self.assertIn("words_with_etymology", data)
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test invalid API key
        invalid_headers = self.headers.copy()
        invalid_headers["X-API-Key"] = "invalid-key"
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}",
            headers=invalid_headers
        )
        self.assertEqual(response.status_code, 401)
        
        # Test missing required parameters
        response = requests.get(
            f"{self.BASE_URL}/api/v2/search",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 400)
        
        # Test invalid search mode
        response = requests.get(
            f"{self.BASE_URL}/api/v2/search",
            params={"q": self.test_word, "mode": "invalid_mode"},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 400)
        
        # Test invalid pagination parameters
        response = requests.get(
            f"{self.BASE_URL}/api/v2/search",
            params={
                "q": self.test_word,
                "page": -1,
                "per_page": 0
            },
            headers=self.headers
        )
        self.assertEqual(response.status_code, 400)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        for _ in range(10):
            response = requests.get(
                f"{self.BASE_URL}/api/v2/words/{self.test_word}",
                headers=self.headers
            )
            self.assertIn(response.status_code, [200, 429])
            if response.status_code == 429:
                break
    
    def test_cors_headers(self):
        """Test CORS headers."""
        response = requests.options(
            f"{self.BASE_URL}/api/v2/words/{self.test_word}",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertIn("Access-Control-Allow-Methods", response.headers)
        self.assertIn("Access-Control-Allow-Headers", response.headers)
    
    def test_response_format(self):
        """Test response format consistency."""
        endpoints = [
            f"/api/v2/words/{self.test_word}",
            f"/api/v2/search?q={self.test_word}",
            "/api/v2/statistics"
        ]
        
        for endpoint in endpoints:
            response = requests.get(
                f"{self.BASE_URL}{endpoint}",
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Check common response format
            self.assertIn("data", data)
            self.assertIn("meta", data)
            self.assertIn("status", data)
            self.assertEqual(data["status"], "success")
            
            # Check metadata format
            meta = data["meta"]
            self.assertIn("request_id", meta)
            self.assertIn("timestamp", meta)
            
            # Validate timestamp format
            try:
                datetime.fromisoformat(meta["timestamp"].replace("Z", "+00:00"))
            except ValueError:
                self.fail("Invalid timestamp format")

if __name__ == "__main__":
    unittest.main(verbosity=2) 