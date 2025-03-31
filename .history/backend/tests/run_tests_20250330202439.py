import unittest
import sys
import os
import logging
from typing import List, Type
from unittest import TestCase

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.setup_test_db import setup_test_environment, cleanup_test_database
from tests.test_api import TestFilipinoDictionaryAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_tests(test_classes: List[Type[TestCase]] = None) -> bool:
    """
    Run the test suite with proper setup and teardown.
    
    Args:
        test_classes: List of test classes to run. If None, runs all tests.
    
    Returns:
        bool: True if all tests passed, False otherwise.
    """
    try:
        # Set up test environment
        logger.info("Setting up test environment...")
        setup_test_environment()
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add test classes to suite
        if test_classes:
            for test_class in test_classes:
                suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
        else:
            # Add all test classes from test_api.py
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFilipinoDictionaryAPI))
        
        # Run tests
        logger.info("Running tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Clean up test environment
        logger.info("Cleaning up test environment...")
        cleanup_test_database()
        
        # Return success status
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        # Try to clean up even if there was an error
        try:
            cleanup_test_database()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        return False

if __name__ == "__main__":
    # Run all tests
    success = run_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 