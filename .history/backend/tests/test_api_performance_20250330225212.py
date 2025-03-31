import unittest
import requests
import time
import logging
import psutil
import os
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median, stdev

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPIPerformance(unittest.TestCase):
    """Performance test suite for the Filipino Dictionary API."""
    
    BASE_URL = "http://localhost:10000"
    TEST_WORDS = ["bahay", "ganda", "araw", "tubig", "bato"]
    CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 5  # seconds
    
    def setUp(self):
        """Set up test environment."""
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": "test-api-key"
        }
        self.process = psutil.Process(os.getpid())
    
    def measure_response_time(self, endpoint: str, method: str = "GET", **kwargs) -> float:
        """Measure response time for a single request."""
        start_time = time.time()
        try:
            response = requests.request(
                method,
                f"{self.BASE_URL}{endpoint}",
                headers=self.headers,
                timeout=self.REQUEST_TIMEOUT,
                **kwargs
            )
            response.raise_for_status()
            return time.time() - start_time
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return float('inf')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": self.process.memory_percent()
        }
    
    def test_single_request_performance(self):
        """Test performance of single requests to various endpoints."""
        endpoints = [
            f"/api/v2/words/{word}" for word in self.TEST_WORDS
        ]
        
        results = []
        for endpoint in endpoints:
            response_time = self.measure_response_time(endpoint)
            self.assertLess(response_time, self.REQUEST_TIMEOUT)
            results.append(response_time)
            
            # Log memory usage
            memory_usage = self.get_memory_usage()
            logger.info(f"Memory usage for {endpoint}: {memory_usage}")
        
        # Calculate statistics
        avg_time = mean(results)
        median_time = median(results)
        std_dev = stdev(results) if len(results) > 1 else 0
        
        logger.info(f"Single request performance:")
        logger.info(f"Average response time: {avg_time:.3f}s")
        logger.info(f"Median response time: {median_time:.3f}s")
        logger.info(f"Standard deviation: {std_dev:.3f}s")
    
    def test_concurrent_requests(self):
        """Test performance under concurrent load."""
        endpoint = f"/api/v2/words/{self.TEST_WORDS[0]}"
        
        with ThreadPoolExecutor(max_workers=self.CONCURRENT_REQUESTS) as executor:
            futures = [
                executor.submit(self.measure_response_time, endpoint)
                for _ in range(self.CONCURRENT_REQUESTS)
            ]
            
            results = [future.result() for future in futures]
        
        # Calculate statistics
        avg_time = mean(results)
        median_time = median(results)
        std_dev = stdev(results) if len(results) > 1 else 0
        
        logger.info(f"Concurrent request performance ({self.CONCURRENT_REQUESTS} requests):")
        logger.info(f"Average response time: {avg_time:.3f}s")
        logger.info(f"Median response time: {median_time:.3f}s")
        logger.info(f"Standard deviation: {std_dev:.3f}s")
        
        # Check if any requests failed
        failed_requests = sum(1 for t in results if t == float('inf'))
        self.assertEqual(failed_requests, 0, f"{failed_requests} requests failed")
    
    def test_search_performance(self):
        """Test performance of search endpoint with various modes."""
        search_modes = ["exact", "fuzzy", "phonetic", "baybayin"]
        
        for mode in search_modes:
            results = []
            for word in self.TEST_WORDS:
                response_time = self.measure_response_time(
                    "/api/v2/search",
                    params={"q": word, "mode": mode}
                )
                results.append(response_time)
            
            avg_time = mean(results)
            logger.info(f"Search performance ({mode} mode):")
            logger.info(f"Average response time: {avg_time:.3f}s")
            
            # Log memory usage
            memory_usage = self.get_memory_usage()
            logger.info(f"Memory usage for {mode} search: {memory_usage}")
    
    def test_baybayin_processing_performance(self):
        """Test performance of Baybayin processing endpoint."""
        test_cases = [
            ("ᜊᜑᜌ᜔", "romanize"),
            ("bahay", "transliterate"),
            ("ᜋᜄᜈ᜔ᜇᜅ᜔ ᜂᜋᜄ", "romanize"),
            ("magandang umaga", "transliterate")
        ]
        
        results = []
        for text, mode in test_cases:
            response_time = self.measure_response_time(
                "/api/v2/baybayin/process",
                method="POST",
                json={
                    "text": text,
                    "mode": mode,
                    "validate_text": True,
                    "include_metadata": True,
                    "include_pronunciation": True
                }
            )
            results.append(response_time)
        
        avg_time = mean(results)
        logger.info(f"Baybayin processing performance:")
        logger.info(f"Average response time: {avg_time:.3f}s")
    
    def test_memory_leaks(self):
        """Test for memory leaks during repeated requests."""
        endpoint = f"/api/v2/words/{self.TEST_WORDS[0]}"
        initial_memory = self.get_memory_usage()
        
        # Make multiple requests
        for _ in range(100):
            self.measure_response_time(endpoint)
        
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = {
            "rss": final_memory["rss"] - initial_memory["rss"],
            "vms": final_memory["vms"] - initial_memory["vms"],
            "percent": final_memory["percent"] - initial_memory["percent"]
        }
        
        logger.info(f"Memory usage after 100 requests:")
        logger.info(f"Initial: {initial_memory}")
        logger.info(f"Final: {final_memory}")
        logger.info(f"Increase: {memory_increase}")
        
        # Check for significant memory leaks
        self.assertLess(memory_increase["rss"], 50)  # Less than 50MB increase
        self.assertLess(memory_increase["percent"], 10)  # Less than 10% increase
    
    def test_error_handling_performance(self):
        """Test performance of error handling."""
        # Test invalid word
        response_time = self.measure_response_time("/api/v2/words/nonexistentword")
        self.assertLess(response_time, self.REQUEST_TIMEOUT)
        
        # Test invalid API key
        invalid_headers = self.headers.copy()
        invalid_headers["X-API-Key"] = "invalid-key"
        start_time = time.time()
        response = requests.get(
            f"{self.BASE_URL}/api/v2/words/{self.TEST_WORDS[0]}",
            headers=invalid_headers,
            timeout=self.REQUEST_TIMEOUT
        )
        error_response_time = time.time() - start_time
        self.assertEqual(response.status_code, 401)
        self.assertLess(error_response_time, self.REQUEST_TIMEOUT)
        
        logger.info(f"Error handling performance:")
        logger.info(f"Invalid word response time: {response_time:.3f}s")
        logger.info(f"Invalid API key response time: {error_response_time:.3f}s")

if __name__ == "__main__":
    unittest.main(verbosity=2) 