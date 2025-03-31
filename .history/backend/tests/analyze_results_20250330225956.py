import json
import os
import logging
import datetime
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics."""
    total_time: float
    average_time: float
    median_time: float
    std_dev: float
    slowest_tests: List[Dict[str, Any]]

@dataclass
class TestCoverage:
    """Container for test coverage information."""
    total_tests: int
    success_rate: float
    endpoint_coverage: Dict[str, int]
    feature_coverage: Dict[str, int]

@dataclass
class ImprovementSuggestion:
    """Container for improvement suggestions."""
    category: str
    priority: str
    description: str
    impact: str
    effort: str

class TestResultAnalyzer:
    """Analyzes test results and generates improvement suggestions."""
    
    def __init__(self, report_file: str):
        """Initialize analyzer with test report file."""
        self.report_file = report_file
        with open(report_file, 'r') as f:
            self.report_data = json.load(f)
    
    def analyze_performance(self) -> PerformanceMetrics:
        """Analyze performance metrics from test results."""
        test_times = [test['time'] for test in self.report_data['slowest_tests']]
        
        if test_times:
            metrics = PerformanceMetrics(
                total_time=self.report_data['total_time'],
                average_time=statistics.mean(test_times),
                median_time=statistics.median(test_times),
                std_dev=statistics.stdev(test_times) if len(test_times) > 1 else 0,
                slowest_tests=self.report_data['slowest_tests']
            )
        else:
            metrics = PerformanceMetrics(
                total_time=self.report_data['total_time'],
                average_time=0,
                median_time=0,
                std_dev=0,
                slowest_tests=[]
            )
        
        return metrics
    
    def analyze_failures(self) -> Dict[str, List[Dict[str, str]]]:
        """Analyze test failures and errors."""
        return {
            'failures': self.report_data['failed_tests'],
            'errors': self.report_data['error_tests']
        }
    
    def analyze_test_coverage(self) -> TestCoverage:
        """Analyze test coverage statistics."""
        # Extract test names and categorize them
        all_tests = [
            test['test'] for test in self.report_data['slowest_tests']
        ]
        
        # Count tests by endpoint
        endpoint_coverage = {
            'word_lookup': sum(1 for t in all_tests if 'word_lookup' in t.lower()),
            'search': sum(1 for t in all_tests if 'search' in t.lower()),
            'baybayin': sum(1 for t in all_tests if 'baybayin' in t.lower()),
            'network': sum(1 for t in all_tests if 'network' in t.lower()),
            'etymology': sum(1 for t in all_tests if 'etymology' in t.lower())
        }
        
        # Count tests by feature
        feature_coverage = {
            'functionality': sum(1 for t in all_tests if 'test_' in t.lower()),
            'performance': sum(1 for t in all_tests if 'performance' in t.lower()),
            'error_handling': sum(1 for t in all_tests if 'error' in t.lower()),
            'security': sum(1 for t in all_tests if 'security' in t.lower()),
            'integration': sum(1 for t in all_tests if 'integration' in t.lower())
        }
        
        return TestCoverage(
            total_tests=self.report_data['total_tests'],
            success_rate=self.report_data['success_rate'],
            endpoint_coverage=endpoint_coverage,
            feature_coverage=feature_coverage
        )
    
    def generate_improvement_suggestions(self) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Analyze performance
        perf_metrics = self.analyze_performance()
        if perf_metrics.average_time > 1.0:
            suggestions.append(ImprovementSuggestion(
                category='Performance',
                priority='High',
                description='Response times are higher than target. Consider implementing caching or optimizing database queries.',
                impact='Improved user experience and reduced server load',
                effort='Medium'
            ))
        
        # Analyze failures
        failures = self.analyze_failures()
        if failures['failures'] or failures['errors']:
            suggestions.append(ImprovementSuggestion(
                category='Reliability',
                priority='High',
                description=f"Found {len(failures['failures'])} failures and {len(failures['errors'])} errors. Address these issues immediately.",
                impact='Improved API reliability and stability',
                effort='High'
            ))
        
        # Analyze coverage
        coverage = self.analyze_test_coverage()
        if coverage.success_rate < 95:
            suggestions.append(ImprovementSuggestion(
                category='Test Coverage',
                priority='Medium',
                description=f'Test success rate ({coverage.success_rate:.1f}%) is below target. Add more test cases.',
                impact='Better code quality and fewer bugs',
                effort='Medium'
            ))
        
        # Check endpoint coverage
        low_coverage_endpoints = [
            endpoint for endpoint, count in coverage.endpoint_coverage.items()
            if count < 2
        ]
        if low_coverage_endpoints:
            suggestions.append(ImprovementSuggestion(
                category='API Coverage',
                priority='Medium',
                description=f'Low test coverage for endpoints: {", ".join(low_coverage_endpoints)}',
                impact='More comprehensive API testing',
                effort='Medium'
            ))
        
        # Check feature coverage
        low_coverage_features = [
            feature for feature, count in coverage.feature_coverage.items()
            if count < 2
        ]
        if low_coverage_features:
            suggestions.append(ImprovementSuggestion(
                category='Feature Coverage',
                priority='Medium',
                description=f'Low test coverage for features: {", ".join(low_coverage_features)}',
                impact='More robust feature testing',
                effort='Medium'
            ))
        
        return suggestions
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        performance = self.analyze_performance()
        failures = self.analyze_failures()
        coverage = self.analyze_test_coverage()
        suggestions = self.generate_improvement_suggestions()
        
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'source_report': os.path.basename(self.report_file),
            'performance_metrics': {
                'total_time': performance.total_time,
                'average_time': performance.average_time,
                'median_time': performance.median_time,
                'std_dev': performance.std_dev,
                'slowest_tests': performance.slowest_tests
            },
            'test_results': {
                'total_tests': coverage.total_tests,
                'success_rate': coverage.success_rate,
                'failures': len(failures['failures']),
                'errors': len(failures['errors'])
            },
            'coverage_analysis': {
                'endpoint_coverage': coverage.endpoint_coverage,
                'feature_coverage': coverage.feature_coverage
            },
            'improvement_suggestions': [
                {
                    'category': s.category,
                    'priority': s.priority,
                    'description': s.description,
                    'impact': s.impact,
                    'effort': s.effort
                }
                for s in suggestions
            ]
        }
        
        return report

def analyze_latest_report():
    """Analyze the most recent test report."""
    # Find the latest report
    report_dir = os.path.join(os.path.dirname(__file__), 'test_reports')
    if not os.path.exists(report_dir):
        logger.error("No test reports directory found")
        return
    
    report_files = [
        f for f in os.listdir(report_dir)
        if f.startswith('test_report_') and f.endswith('.json')
    ]
    
    if not report_files:
        logger.error("No test reports found")
        return
    
    latest_report = max(report_files)
    report_path = os.path.join(report_dir, latest_report)
    
    # Analyze report
    analyzer = TestResultAnalyzer(report_path)
    analysis = analyzer.generate_report()
    
    # Save analysis report
    analysis_dir = os.path.join(os.path.dirname(__file__), 'analysis_reports')
    os.makedirs(analysis_dir, exist_ok=True)
    
    analysis_file = os.path.join(
        analysis_dir,
        f'analysis_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Log summary
    logger.info("\nAnalysis Summary:")
    logger.info(f"Total tests: {analysis['test_results']['total_tests']}")
    logger.info(f"Success rate: {analysis['test_results']['success_rate']:.1f}%")
    logger.info(f"Failures: {analysis['test_results']['failures']}")
    logger.info(f"Errors: {analysis['test_results']['errors']}")
    
    if analysis['improvement_suggestions']:
        logger.info("\nImprovement Suggestions:")
        for suggestion in analysis['improvement_suggestions']:
            logger.info(f"\n{suggestion['category']} ({suggestion['priority']}):")
            logger.info(f"- {suggestion['description']}")
            logger.info(f"- Impact: {suggestion['impact']}")
            logger.info(f"- Effort: {suggestion['effort']}")
    
    logger.info(f"\nDetailed analysis saved to: {analysis_file}")
    
    return analysis

if __name__ == '__main__':
    analyze_latest_report() 