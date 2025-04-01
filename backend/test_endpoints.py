"""
Test script to verify all API endpoints are working.
Run this script from the command line to check endpoint health.
"""

import requests
import time
import sys

BASE_URL = "http://localhost:10000/api/v2"
TEST_WORD = "aso"  # Simple test word that should exist in most Filipino dictionaries

def test_endpoint(url, name, expected_status=200):
    """Test an endpoint and report the result."""
    start_time = time.time()
    try:
        print(f"Testing {name} endpoint...", end="")
        response = requests.get(url, timeout=5)
        duration = time.time() - start_time
        
        status = "✓" if response.status_code == expected_status else "✗"
        if status == "✓":
            size = len(response.content)
            print(f" {status} ({duration:.2f}s, {size} bytes)")
            return True
        else:
            print(f" {status} Got status {response.status_code} instead of {expected_status}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f" TIMEOUT (>5s)")
        return False
    except Exception as e:
        print(f" ERROR: {str(e)}")
        return False

def run_tests():
    """Run all endpoint tests."""
    print("===== API ENDPOINT TESTS =====")
    print(f"Base URL: {BASE_URL}")
    print("=============================")
    
    # Basic health endpoints
    results = []
    results.append(test_endpoint(f"{BASE_URL}/health", "Health"))
    results.append(test_endpoint(f"{BASE_URL}/test", "Test"))
    
    # Word endpoints
    results.append(test_endpoint(f"{BASE_URL}/words/{TEST_WORD}", "Word lookup"))
    results.append(test_endpoint(f"{BASE_URL}/search?q={TEST_WORD}", "Search"))
    results.append(test_endpoint(f"{BASE_URL}/statistics", "Statistics"))
    
    # More specialized endpoints - these might be slower
    print("\nTesting specialized endpoints with longer timeouts...")
    
    try:
        # Get a word ID for the advanced endpoints that need one
        print(f"Getting word ID for '{TEST_WORD}'...", end="")
        response = requests.get(f"{BASE_URL}/words/{TEST_WORD}", timeout=5)
        if response.status_code == 200:
            word_id = response.json().get('id')
            if word_id:
                print(f" Found ID: {word_id}")
                
                # Test ID-based endpoints
                results.append(test_endpoint(f"{BASE_URL}/words/{word_id}/relations/graph?max_depth=1", "Relations Graph"))
                results.append(test_endpoint(f"{BASE_URL}/words/{word_id}/etymology/tree", "Etymology Tree"))
            else:
                print(f" Failed to get ID from response: {response.json()}")
        else:
            print(f" Failed with status {response.status_code}")
    except Exception as e:
        print(f" ERROR: {str(e)}")
    
    # Word relationship endpoints
    results.append(test_endpoint(f"{BASE_URL}/words/{TEST_WORD}/etymology", "Etymology"))
    results.append(test_endpoint(f"{BASE_URL}/words/{TEST_WORD}/relations", "Relations"))
    results.append(test_endpoint(f"{BASE_URL}/words/{TEST_WORD}/semantic_network", "Semantic Network"))
    results.append(test_endpoint(f"{BASE_URL}/words/{TEST_WORD}/affixations", "Affixations"))
    
    # Summary
    success_count = sum(1 for r in results if r)
    print("\n=============================")
    print(f"RESULTS: {success_count}/{len(results)} endpoints working")
    print("=============================")
    
    return success_count == len(results)

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 