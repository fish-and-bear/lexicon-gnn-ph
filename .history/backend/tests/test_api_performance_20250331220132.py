import unittest
import json
import time
import logging
import psutil
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from flask_testing import TestCase
from .run_tests import TimedTestCase
import sys

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPIPerformance(TimedTestCase, TestCase):
    """Performance test suite for the Filipino Dictionary API."""
    
    # Test configuration
    CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 5.0  # seconds
    MEMORY_THRESHOLD = 100 * 1024 * 1024  # 100MB
    
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
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.process = psutil.Process()
    
    def measure_response_time(self, query: str) -> float:
        """Measure response time for a GraphQL query."""
        start_time = time.time()
        response = self.client.post(
            '/graphql',
            json={'query': query},
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)
        end_time = time.time()
        return end_time - start_time
    
    def test_single_request_performance(self):
        """Test performance of single requests to various endpoints."""
        test_cases = [
            {
                'name': 'Word Search',
                'query': """
                query {
                    searchWords(query: "maganda") {
                        word
                        definitions {
                            definition
                        }
                    }
                }
                """
            },
            {
                'name': 'Word Lookup',
                'query': """
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
            },
            {
                'name': 'Word Network',
                'query': """
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
            }
        ]
        
        for case in test_cases:
            response_time = self.measure_response_time(case['query'])
            logger.info(f"{case['name']} response time: {response_time:.3f}s")
            self.assertLess(response_time, self.REQUEST_TIMEOUT)
    
    def test_concurrent_requests(self):
        """Test performance under concurrent load."""
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
        
        def make_request():
            try:
                response = self.client.post(
                    '/graphql',
                    json={'query': query},
                    headers=self.headers
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                return False
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=self.CONCURRENT_REQUESTS) as executor:
            results = list(executor.map(lambda _: make_request(), range(self.CONCURRENT_REQUESTS)))
        
        # Calculate success rate
        successful = sum(1 for r in results if r)
        success_rate = (successful / len(results)) * 100
        
        logger.info(f"Concurrent requests success rate: {success_rate:.1f}%")
        self.assertGreater(success_rate, 95,
                          f"Too many failed requests: {100 - success_rate:.1f}% failed")
    
    def test_search_performance(self):
        """Test search endpoint performance with various modes."""
        test_queries = [
            "maganda",
            "mahal",
            "ganda",
            "m",
            "mag"
        ]
        
        response_times = []
        for query in test_queries:
            graphql_query = f"""
            query {{
                searchWords(query: "{query}") {{
                    word
                    definitions {{
                        definition
                    }}
                }}
            }}
            """
            response_time = self.measure_response_time(graphql_query)
            response_times.append(response_time)
            logger.info(f"Search for '{query}' response time: {response_time:.3f}s")
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        max_time = max(response_times)
        
        logger.info(f"Search performance statistics:")
        logger.info(f"Average time: {avg_time:.3f}s")
        logger.info(f"Median time: {median_time:.3f}s")
        logger.info(f"Maximum time: {max_time:.3f}s")
        
        self.assertLess(avg_time, self.REQUEST_TIMEOUT)
        self.assertLess(max_time, self.REQUEST_TIMEOUT * 2)
    
    def test_baybayin_processing_performance(self):
        """Test Baybayin processing performance."""
        test_cases = [
            {
                'text': "ᜋᜄᜈ᜔ᜇ",
                'description': 'Single word'
            },
            {
                'text': "ᜋᜄᜈ᜔ᜇ ᜈ ᜋᜑᜎ᜔",
                'description': 'Multiple words'
            },
            {
                'text': "ᜋᜄᜈ᜔ᜇ ᜈ ᜋᜑᜎ᜔ ᜀᜅ᜔ ᜊᜑᜌ᜔",
                'description': 'Longer text'
            }
        ]
        
        for case in test_cases:
            query = f"""
            query {{
                baybayinWords(text: "{case['text']}") {{
                    original
                    baybayin
                    latin
                }}
            }}
            """
            response_time = self.measure_response_time(query)
            logger.info(f"Baybayin processing for {case['description']}: {response_time:.3f}s")
            self.assertLess(response_time, self.REQUEST_TIMEOUT)
    
    def test_error_handling_performance(self):
        """Test performance of error handling."""
        test_cases = [
            {
                'query': """
                query {
                    invalidField {
                        field
                    }
                }
                """,
                'description': 'Invalid field'
            },
            {
                'query': """
                query {
                    getWord(word: "") {
                        word
                    }
                }
                """,
                'description': 'Empty word'
            }
        ]
        
        for case in test_cases:
            start_time = time.time()
            response = self.client.post(
                '/graphql',
                json={'query': case['query']},
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)  # GraphQL always returns 200
            data = json.loads(response.data)
            self.assertIn('errors', data)
            response_time = time.time() - start_time
            
            logger.info(f"Error handling for {case['description']}: {response_time:.3f}s")
            self.assertLess(response_time, self.REQUEST_TIMEOUT)
    
    def test_memory_leaks(self):
        """Test for memory leaks during repeated requests."""
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
        
        # Record initial memory usage
        initial_memory = self.process.memory_info().rss
        
        # Make multiple requests
        for i in range(100):
            response = self.client.post(
                '/graphql',
                json={'query': query},
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
            
            # Check memory usage every 10 requests
            if (i + 1) % 10 == 0:
                current_memory = self.process.memory_info().rss
                memory_increase = current_memory - initial_memory
                logger.info(f"Memory increase after {i + 1} requests: {memory_increase / 1024 / 1024:.2f}MB")
                self.assertLess(memory_increase, self.MEMORY_THRESHOLD)

if __name__ == '__main__':
    unittest.main(verbosity=2) 