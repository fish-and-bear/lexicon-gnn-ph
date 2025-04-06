#!/usr/bin/env python3
"""
Installation script for the search suggestions system.
This script:
1. Runs the database migration
2. Tests the new search suggestions endpoint
3. Provides information on next steps
"""

import os
import sys
import logging
import requests
import time
from migrations.add_search_suggestions import run_migration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api_test(base_url="http://localhost:5000"):
    """Test the search suggestions API."""
    logger.info("Testing search suggestions endpoint...")
    
    test_queries = [
        "tul",         # Common prefix
        "mangga",      # Common word
        "tagalog",     # Language name
        "aso",         # Common animal
        "kumusta"      # Greeting
    ]
    
    endpoint = f"{base_url}/api/search/suggestions"
    
    for query in test_queries:
        logger.info(f"Testing query: '{query}'")
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}?q={query}")
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                suggestions = data.get("suggestions", [])
                logger.info(f"  ✓ Query '{query}' returned {len(suggestions)} suggestions in {elapsed:.3f}s")
                
                if suggestions:
                    # Show a few examples
                    for i, s in enumerate(suggestions[:3]):
                        logger.info(f"    - {s.get('text')} ({s.get('type')})")
                    if len(suggestions) > 3:
                        logger.info(f"    - ... and {len(suggestions) - 3} more")
            else:
                logger.error(f"  ✗ Query '{query}' failed with status code {response.status_code}")
                logger.error(f"  Response: {response.text}")
                
        except Exception as e:
            logger.error(f"  ✗ Error testing query '{query}': {str(e)}")
    
    # Test the tracking endpoint
    logger.info("Testing search selection tracking endpoint...")
    try:
        response = requests.post(
            f"{base_url}/api/search/track-selection",
            json={
                "query": "test query",
                "selected_id": 1,
                "selected_text": "test word"
            }
        )
        
        if response.status_code == 200:
            logger.info("  ✓ Selection tracking endpoint is working")
        else:
            logger.error(f"  ✗ Selection tracking test failed with status code {response.status_code}")
            logger.error(f"  Response: {response.text}")
            
    except Exception as e:
        logger.error(f"  ✗ Error testing selection tracking: {str(e)}")

def main():
    """Install and test the search suggestions system."""
    logger.info("=== Installing Search Suggestions System ===")
    
    # Step 1: Run the database migration
    logger.info("Step 1: Running database migration...")
    success = run_migration()
    
    if not success:
        logger.error("Migration failed! Please check the logs and try again.")
        return 1
    
    logger.info("Database migration completed successfully.")
    
    # Step 2: Check if the API is running
    logger.info("Step 2: Checking if the API is running...")
    try:
        response = requests.get("http://localhost:5000/api/status")
        if response.status_code == 200:
            logger.info("API is running. Testing search suggestions...")
            run_api_test()
        else:
            logger.warning("API seems to be running but status endpoint returned an error.")
            response = input("Would you like to test the endpoints anyway? (y/n): ")
            if response.lower() == 'y':
                run_api_test()
    except requests.ConnectionError:
        logger.warning("API is not running. Cannot test the endpoints.")
        logger.info("To test the endpoints, please start the API server:")
        logger.info("  $ python serve.py")
        logger.info("Then you can test the endpoints manually or run this script again.")
    
    # Step 3: Show next steps
    logger.info("\n=== Installation Complete ===")
    logger.info("The search suggestions system has been installed.")
    logger.info("\nNext steps:")
    logger.info("1. Restart your API server to initialize the background tasks")
    logger.info("2. Use the search suggestions endpoint: GET /api/search/suggestions?q=<query>")
    logger.info("3. Track user selections with: POST /api/search/track-selection")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 