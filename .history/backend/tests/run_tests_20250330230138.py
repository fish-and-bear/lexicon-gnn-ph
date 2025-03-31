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

class TimedTestCase(unittest.TestCase):
    """Base test case class that tracks execution time."""
    def setUp(self):
        self._started = time.time()
        super().setUp()
        
    def tearDown(self):
        self._time = time.time() - self._started
        super().tearDown()
        
    @property
    def time(self):
        return getattr(self, '_time', 0)

class DetailedTestResult(TextTestResult):
    """Custom test result class that generates detailed reports."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_time = None
        self.end_time = None
        self.slowest_tests = []
    
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
        test_time = getattr(test, 'time', 0)
        self.test_results.append({
            "name": test.id(),
            "status": "success",
            "time": test_time
        })
        self.slowest_tests.append({
            "test": test.id(),
            "time": test_time
        })
    
    def addError(self, test, err):
        """Called when a test raises an error."""
        super().addError(test, err)
        test_time = getattr(test, 'time', 0)
        self.test_results.append({
            "name": test.id(),
            "status": "error",
            "time": test_time,
            "error": str(err[1])
        })
    
    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        test_time = getattr(test, 'time', 0)
        self.test_results.append({
            "name": test.id(),
            "status": "failure",
            "time": test_time,
            "error": str(err[1])
        })
    
    def generateReport(self):
        """Generate a detailed test report."""
        total_time = self.end_time - self.start_time
        total_tests = len(self.test_results)
        successful = sum(1 for r in self.test_results if r["status"] == "success")
        failed = sum(1 for r in self.test_results if r["status"] == "failure")
        errors = sum(1 for r in self.test_results if r["status"] == "error")
        
        # Sort slowest tests
        self.slowest_tests.sort(key=lambda x: x["time"], reverse=True)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "total_time": total_time,
            "success_rate": (successful / total_tests * 100) if total_tests > 0 else 0,
            "test_results": self.test_results,
            "slowest_tests": self.slowest_tests[:5],  # Top 5 slowest tests
            "failed_tests": [
                r for r in self.test_results if r["status"] == "failure"
            ],
            "error_tests": [
                r for r in self.test_results if r["status"] == "error"
            ]
        }
        
        return report

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
    
    # Generate and save report
    if hasattr(result, 'generateReport'):
        report = result.generateReport()
        
        # Create reports directory if it doesn't exist
        report_dir = os.path.join(start_dir, "test_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save report
        report_file = os.path.join(
            report_dir,
            f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed test report saved to: {report_file}")
    
    # Return True if all tests passed
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 