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
from run_tests import TimedTestCase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPIPerformance(TimedTestCase, TestCase):
    """Performance test suite for the Filipino Dictionary API."""
    
    # Constants
    BASE_URL = 'http://localhost:5000'
    TEST_WORDS = ['bahay', 'aral', 'tao', 'guro', 'aklat']
    CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30
    
    def create_app(self):
        """Create Flask test app."""
        from app import create_app
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    def setUp(self):
        """Set up test environment."""
        self.client = self.app.test_client()
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        # Get current process for memory usage tracking
        self.process = psutil.Process(os.getpid())
    
    def measure_response_time(self, query: str) -> float:
        """Measure response time for a GraphQL query."""
        start_time = time.time()
        response = self.client.post('/graphql',
                                  json={'query': query},
                                  headers=self.headers)
        end_time = time.time()
        self.assertEqual(response.status_code, 200)
        return end_time - start_time
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
            'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
            'percent': self.process.memory_percent()
        }
    
    def test_single_request_performance(self):
        """Test performance of single requests to various endpoints."""
        test_cases = [
            {
                'name': 'Word Lookup',
                'query': """
                {
                    getWord(word: "bahay") {
                        lemma
                        definitions {
                            definitionText
                        }
                        etymology {
                            components {
                                text
                                language
                            }
                        }
                    }
                }
                """
            },
            {
                'name': 'Word Search',
                'query': """
                {
                    searchWords(query: "bahay", limit: 5) {
                        lemma
                        definitions {
                            definitionText
                        }
                    }
                }
                """
            },
            {
                'name': 'Word Network',
                'query': """
                {
                    getWordNetwork(word: "bahay", depth: 2) {
                        nodes {
                            id
                            info {
                                word
                            }
                        }
                        links {
                            source
                            target
                        }
                    }
                }
                """
            }
        ]
        
        for case in test_cases:
            # Measure initial memory
            initial_memory = self.get_memory_usage()
            
            # Measure response time
            response_time = self.measure_response_time(case['query'])
            
            # Measure final memory
            final_memory = self.get_memory_usage()
            
            # Log results
            logger.info(f"Performance test - {case['name']}:")
            logger.info(f"Response time: {response_time:.3f} seconds")
            logger.info(f"Memory usage change: {final_memory['rss'] - initial_memory['rss']:.2f} MB")
            
            # Assert performance requirements
            self.assertLess(response_time, 2.0, 
                          f"{case['name']} request took too long: {response_time:.3f} seconds")
            self.assertLess(final_memory['rss'] - initial_memory['rss'], 50,
                          f"{case['name']} memory increase too high")
    
    def test_concurrent_requests(self):
        """Test performance under concurrent load."""
        query = """
        {
            getWord(word: "bahay") {
                lemma
                definitions {
                    definitionText
                }
            }
        }
        """
        
        def make_request():
            response = self.client.post('/graphql',
                                      json={'query': query},
                                      headers=self.headers)
            return response.status_code == 200
        
        # Measure initial memory
        initial_memory = self.get_memory_usage()
        
        # Make concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.CONCURRENT_REQUESTS) as executor:
            results = list(executor.map(lambda _: make_request(), 
                                      range(self.CONCURRENT_REQUESTS)))
        end_time = time.time()
        
        # Measure final memory
        final_memory = self.get_memory_usage()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_per_request = total_time / self.CONCURRENT_REQUESTS
        success_rate = sum(results) / len(results) * 100
        memory_increase = final_memory['rss'] - initial_memory['rss']
        
        # Log results
        logger.info("Concurrent requests performance test:")
        logger.info(f"Total time: {total_time:.3f} seconds")
        logger.info(f"Average time per request: {avg_time_per_request:.3f} seconds")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Memory usage increase: {memory_increase:.2f} MB")
        
        # Assert performance requirements
        self.assertGreater(success_rate, 95, 
                          f"Too many failed requests: {100 - success_rate:.1f}% failed")
        self.assertLess(avg_time_per_request, 1.0,
                          f"Average request time too high: {avg_time_per_request:.3f} seconds")
        self.assertLess(memory_increase, 100,
                          f"Memory increase too high: {memory_increase:.2f} MB")
    
    def test_search_performance(self):
        """Test search endpoint performance with various modes."""
        search_modes = ['exact', 'fuzzy', 'phonetic']
        response_times = []
        
        for mode in search_modes:
            query = f"""
            {{
                searchWords(
                    query: "bahay",
                    mode: {mode},
                    limit: 10
                ) {{
                    lemma
                    definitions {{
                        definitionText
                    }}
                }}
            }}
            """
            
            # Measure response time
            response_time = self.measure_response_time(query)
            response_times.append(response_time)
            
            # Log results
            logger.info(f"Search performance test - {mode} mode:")
            logger.info(f"Response time: {response_time:.3f} seconds")
            
            # Assert performance requirements
            self.assertLess(response_time, 2.0,
                          f"{mode} search took too long: {response_time:.3f} seconds")
        
        # Calculate and log statistics
        avg_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times)
        logger.info(f"Search performance statistics:")
        logger.info(f"Average time: {avg_time:.3f} seconds")
        logger.info(f"Standard deviation: {std_dev:.3f} seconds")
    
    def test_baybayin_processing_performance(self):
        """Test Baybayin processing performance."""
        test_cases = [
            {
                'name': 'Single Word',
                'query': """
                {
                    baybayinWords(limit: 1) {
                        lemma
                        baybayinForm
                        romanizedForm
                    }
                }
                """
            },
            {
                'name': 'Multiple Words',
                'query': """
                {
                    baybayinWords(limit: 10) {
                        lemma
                        baybayinForm
                        romanizedForm
                    }
                }
                """
            }
        ]
        
        for case in test_cases:
            # Measure response time
            response_time = self.measure_response_time(case['query'])
            
            # Log results
            logger.info(f"Baybayin processing test - {case['name']}:")
            logger.info(f"Response time: {response_time:.3f} seconds")
            
            # Assert performance requirements
            self.assertLess(response_time, 1.0,
                          f"Baybayin processing took too long: {response_time:.3f} seconds")
    
    def test_memory_leaks(self):
        """Test for memory leaks during repeated requests."""
        query = """
        {
            getWord(word: "bahay") {
                lemma
                definitions {
                    definitionText
                }
                etymology {
                    components {
                        text
                        language
                    }
                }
                relationships {
                    synonyms
                    antonyms
                }
            }
        }
        """
        
        # Record initial memory
        initial_memory = self.get_memory_usage()
        memory_readings = []
        
        # Make repeated requests
        for i in range(50):
            response = self.client.post('/graphql',
                                      json={'query': query},
                                      headers=self.headers)
            self.assertEqual(response.status_code, 200)
            
            # Record memory usage
            current_memory = self.get_memory_usage()
            memory_readings.append(current_memory['rss'])
            
            if i > 0 and i % 10 == 0:
                logger.info(f"Memory usage after {i} requests: {current_memory['rss']:.2f} MB")
        
        # Calculate memory growth
        final_memory = self.get_memory_usage()
        memory_growth = final_memory['rss'] - initial_memory['rss']
        
        # Calculate memory growth rate
        memory_growth_rate = memory_growth / len(memory_readings)
        
        # Log results
        logger.info("Memory leak test results:")
        logger.info(f"Initial memory: {initial_memory['rss']:.2f} MB")
        logger.info(f"Final memory: {final_memory['rss']:.2f} MB")
        logger.info(f"Total memory growth: {memory_growth:.2f} MB")
        logger.info(f"Memory growth rate: {memory_growth_rate:.4f} MB/request")
        
        # Assert no significant memory leaks
        self.assertLess(memory_growth_rate, 0.1,
                       f"Possible memory leak detected: {memory_growth_rate:.4f} MB/request")
    
    def test_error_handling_performance(self):
        """Test performance of error handling."""
        error_cases = [
            {
                'name': 'Invalid Word',
                'query': """
                {
                    getWord(word: "") {
                        lemma
                    }
                }
                """
            },
            {
                'name': 'Invalid Query',
                'query': """
                {
                    invalidField {
                        something
                    }
                }
                """
            },
            {
                'name': 'Missing Required Argument',
                'query': """
                {
                    getWord {
                        lemma
                    }
                }
                """
            }
        ]
        
        for case in error_cases:
            # Measure response time
            start_time = time.time()
            response = self.client.post('/graphql',
                                      json={'query': case['query']},
                                      headers=self.headers)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Verify error response
            self.assertIn('errors', json.loads(response.data))
            
            # Log results
            logger.info(f"Error handling test - {case['name']}:")
            logger.info(f"Response time: {response_time:.3f} seconds")
            
            # Assert performance requirements
            self.assertLess(response_time, 0.5,
                          f"Error handling took too long: {response_time:.3f} seconds")

if __name__ == '__main__':
    unittest.main() 