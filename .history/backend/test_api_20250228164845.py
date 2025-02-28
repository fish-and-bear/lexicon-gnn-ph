"""
Simple script to test the API endpoints.
"""

import requests
import json
import sys
from datetime import datetime

def test_api():
    """Test the API endpoints."""
    base_url = "http://127.0.0.1:10000"
    
    # Test root endpoint
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Test words endpoint
    print("\nTesting words endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v2/words")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")

    # Test search endpoint
    print("\nTesting search endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v2/search?q=test")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api() 