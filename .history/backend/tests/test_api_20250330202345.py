import unittest
import requests
import json
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPI(unittest.TestCase):
    """Test suite for the Filipino Dictionary API."""
    
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
    
    def test_graphql_word_search(self):
        """Test word search functionality."""
        query = """
        query SearchWords($query: String) {
            searchWords(query: $query, limit: 5) {
                edges {
                    node {
                        id
                        lemma
                        languageCode
                        hasBaybayin
                        definitions {
                            definitionText
                        }
                    }
                }
            }
        }
        """
        
        variables = {"query": "ganda"}
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("searchWords", data["data"])
    
    def test_graphql_get_word(self):
        """Test getting a specific word."""
        query = """
        query GetWord($word: String!, $language: String) {
            getWord(word: $word, language: $language) {
                id
                lemma
                languageCode
                hasBaybayin
                baybayinForm
                definitions {
                    definitionText
                    partOfSpeech {
                        code
                        nameEn
                    }
                }
                etymologies {
                    etymologyText
                }
                relationships {
                    synonyms
                    antonyms
                    related
                }
            }
        }
        """
        
        variables = {"word": "ganda", "language": "tl"}
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("getWord", data["data"])
    
    def test_graphql_word_network(self):
        """Test word network functionality."""
        query = """
        query GetWordNetwork($word: String!, $depth: Int, $breadth: Int) {
            getWordNetwork(word: $word, depth: $depth, breadth: $breadth) {
                nodes {
                    id
                    group
                    info {
                        word
                        definition
                        derivatives
                        synonyms
                        antonyms
                        baybayin
                        hasBaybayin
                        qualityScore
                    }
                }
                links {
                    source
                    target
                    type
                }
            }
        }
        """
        
        variables = {"word": "ganda", "depth": 2, "breadth": 5}
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("getWordNetwork", data["data"])
    
    def test_graphql_baybayin_words(self):
        """Test Baybayin words query."""
        query = """
        query BaybayinWords($limit: Int, $minQuality: Int) {
            baybayinWords(limit: $limit, minQuality: $minQuality) {
                id
                lemma
                baybayinForm
                hasBaybayin
                definitions {
                    definitionText
                }
            }
        }
        """
        
        variables = {"limit": 5, "minQuality": 50}
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("baybayinWords", data["data"])
    
    def test_graphql_parts_of_speech(self):
        """Test parts of speech query."""
        query = """
        query PartsOfSpeech {
            partsOfSpeech {
                id
                code
                nameEn
                nameTl
                description
                wordCount
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
        self.assertIn("partsOfSpeech", data["data"])
    
    def test_graphql_etymology_components(self):
        """Test etymology components query."""
        query = """
        query EtymologyComponents($word: String!, $language: String) {
            etymologyComponents(word: $word, language: $language)
        }
        """
        
        variables = {"word": "ganda", "language": "tl"}
        response = requests.post(
            f"{self.BASE_URL}/graphql",
            json={"query": query, "variables": variables},
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("data", data)
        self.assertIn("etymologyComponents", data["data"])
    
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
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        query = """
        query SearchWords($query: String) {
            searchWords(query: $query, limit: 1) {
                edges {
                    node {
                        id
                        lemma
                    }
                }
            }
        }
        """
        
        variables = {"query": "test"}
        
        # Make multiple requests quickly
        for _ in range(10):
            response = requests.post(
                f"{self.BASE_URL}/graphql",
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            self.assertIn(response.status_code, [200, 429])  # Either success or rate limit
    
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