import unittest
import logging
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List
from unittest.runner import TextTestResult
from unittest.suite import TestSuite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetailedTestResult(TextTestResult):
    """Custom test result class that generates detailed reports."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    def startTestRun(self):
        """Called before any tests are run."""
        super().startTestRun()
        self.start_time = time.time()
    
    def stopTestRun(self):
        """Called after all tests are run."""
        self.end_time = time.time()
        super().stopTestRun()
    
    def addSuccess(self, test):
        """Called when a test succeeds."""
        super().addSuccess(test)
        self.test_results.append({
            "name": test.id(),
            "status": "success",
            "time": test.time
        })
    
    def addError(self, test, err):
        """Called when a test raises an error."""
        super().addError(test, err)
        self.test_results.append({
            "name": test.id(),
            "status": "error",
            "time": test.time,
            "error": str(err[1])
        })
    
    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        self.test_results.append({
            "name": test.id(),
            "status": "failure",
            "time": test.time,
            "error": str(err[1])
        })
    
    def printSummary(self):
        """Print a detailed summary of the test run."""
        super().printSummary()
        
        # Calculate statistics
        total_tests = len(self.test_results)
        successful = sum(1 for r in self.test_results if r["status"] == "success")
        failed = sum(1 for r in self.test_results if r["status"] == "failure")
        errors = sum(1 for r in self.test_results if r["status"] == "error")
        total_time = self.end_time - self.start_time
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "total_time": total_time,
            "test_results": self.test_results
        }
        
        # Save report to file
        report_dir = "test_reports"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(
            report_dir,
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed test report saved to: {report_file}")
        
        # Print summary to console
        logger.info("\nTest Summary:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Total Time: {total_time:.2f}s")
        
        if failed > 0 or errors > 0:
            logger.info("\nFailed Tests:")
            for result in self.test_results:
                if result["status"] in ["failure", "error"]:
                    logger.info(f"- {result['name']}: {result['error']}")

def run_tests():
    """Run all test suites and generate reports."""
    # Discover all test files
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Run tests with detailed reporting
    runner = unittest.TextTestRunner(
        resultclass=DetailedTestResult,
        verbosity=2
    )
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 