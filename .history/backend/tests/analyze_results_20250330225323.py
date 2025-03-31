import json
import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from statistics import mean, median, stdev

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResultAnalyzer:
    """Analyzes test results and suggests improvements."""
    
    def __init__(self, report_file: str):
        """Initialize analyzer with a test report file."""
        self.report_file = report_file
        with open(report_file, 'r') as f:
            self.report = json.load(f)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze test performance metrics."""
        test_times = [r["time"] for r in self.report["test_results"]]
        
        return {
            "total_time": self.report["total_time"],
            "average_test_time": mean(test_times),
            "median_test_time": median(test_times),
            "std_dev": stdev(test_times) if len(test_times) > 1 else 0,
            "slowest_tests": sorted(
                self.report["test_results"],
                key=lambda x: x["time"],
                reverse=True
            )[:5]
        }
    
    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze test failures and errors."""
        failures = [r for r in self.report["test_results"] if r["status"] == "failure"]
        errors = [r for r in self.report["test_results"] if r["status"] == "error"]
        
        return {
            "total_failures": len(failures),
            "total_errors": len(errors),
            "failure_details": [
                {
                    "test": f["name"],
                    "error": f["error"]
                }
                for f in failures
            ],
            "error_details": [
                {
                    "test": e["name"],
                    "error": e["error"]
                }
                for e in errors
            ]
        }
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage by endpoint and feature."""
        endpoints = {}
        features = {}
        
        for result in self.report["test_results"]:
            test_name = result["name"]
            
            # Analyze by endpoint
            if "words" in test_name:
                endpoints["words"] = endpoints.get("words", 0) + 1
            elif "search" in test_name:
                endpoints["search"] = endpoints.get("search", 0) + 1
            elif "baybayin" in test_name:
                endpoints["baybayin"] = endpoints.get("baybayin", 0) + 1
            elif "statistics" in test_name:
                endpoints["statistics"] = endpoints.get("statistics", 0) + 1
            
            # Analyze by feature
            if "performance" in test_name:
                features["performance"] = features.get("performance", 0) + 1
            elif "error_handling" in test_name:
                features["error_handling"] = features.get("error_handling", 0) + 1
            elif "security" in test_name:
                features["security"] = features.get("security", 0) + 1
        
        return {
            "endpoints": endpoints,
            "features": features
        }
    
    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Analyze performance
        perf = self.analyze_performance()
        if perf["average_test_time"] > 1.0:  # More than 1 second
            suggestions.append({
                "type": "performance",
                "priority": "high",
                "message": "Test execution is slow. Consider optimizing database queries and caching.",
                "details": {
                    "average_time": perf["average_test_time"],
                    "slowest_tests": [t["name"] for t in perf["slowest_tests"]]
                }
            })
        
        # Analyze failures
        failures = self.analyze_failures()
        if failures["total_failures"] > 0:
            suggestions.append({
                "type": "reliability",
                "priority": "high",
                "message": "Some tests are failing. Review and fix failing tests.",
                "details": {
                    "failing_tests": failures["failure_details"]
                }
            })
        
        # Analyze coverage
        coverage = self.analyze_test_coverage()
        if not coverage["endpoints"].get("baybayin"):
            suggestions.append({
                "type": "coverage",
                "priority": "medium",
                "message": "Missing tests for Baybayin endpoints.",
                "details": {
                    "missing_endpoints": ["baybayin"]
                }
            })
        
        return suggestions
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_tests": self.report["total_tests"],
                "successful": self.report["successful"],
                "failed": self.report["failed"],
                "errors": self.report["errors"]
            },
            "performance_analysis": self.analyze_performance(),
            "failure_analysis": self.analyze_failures(),
            "coverage_analysis": self.analyze_test_coverage(),
            "improvement_suggestions": self.generate_improvement_suggestions()
        }

def analyze_latest_report():
    """Analyze the most recent test report."""
    report_dir = "test_reports"
    if not os.path.exists(report_dir):
        logger.error("No test reports found")
        return
    
    # Get the most recent report file
    report_files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
    if not report_files:
        logger.error("No test reports found")
        return
    
    latest_report = max(
        report_files,
        key=lambda x: os.path.getctime(os.path.join(report_dir, x))
    )
    report_path = os.path.join(report_dir, latest_report)
    
    # Analyze the report
    analyzer = TestResultAnalyzer(report_path)
    analysis = analyzer.generate_report()
    
    # Save analysis report
    analysis_dir = "analysis_reports"
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_file = os.path.join(
        analysis_dir,
        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    logger.info(f"\nAnalysis Report Summary:")
    logger.info(f"Total Tests: {analysis['test_summary']['total_tests']}")
    logger.info(f"Successful: {analysis['test_summary']['successful']}")
    logger.info(f"Failed: {analysis['test_summary']['failed']}")
    logger.info(f"Errors: {analysis['test_summary']['errors']}")
    
    if analysis['improvement_suggestions']:
        logger.info("\nImprovement Suggestions:")
        for suggestion in analysis['improvement_suggestions']:
            logger.info(f"- [{suggestion['priority']}] {suggestion['message']}")
    
    logger.info(f"\nDetailed analysis saved to: {analysis_file}")

if __name__ == "__main__":
    analyze_latest_report() 