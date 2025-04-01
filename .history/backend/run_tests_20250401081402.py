"""
Test runner script for the Filipino Dictionary API.
Runs tests and generates a comprehensive report.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from test_api import APITester
from typing import Dict, Any, List
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRunner:
    """Class to run tests and generate reports."""
    
    def __init__(self):
        self.results = {
            'word_lookup': [],
            'search': [],
            'relations': [],
            'etymology': [],
            'pronunciation': [],
            'statistics': None
        }
        self.timings = {
            'word_lookup': [],
            'search': [],
            'relations': [],
            'etymology': [],
            'pronunciation': []
        }
        self.test_words = [
            "aklat", "bahay", "tao", "guro", "paaralan",
            "bata", "maganda", "masaya", "kumain", "magluto"
        ]
        self.search_queries = [
            ("a", {}),
            ("bahay", {"mode": "exact"}),
            ("ᜊ", {"mode": "baybayin"}),
            ("ma", {"limit": 5}),
            ("tao", {"language": "tl"})
        ]
    
    def run_tests(self):
        """Run all tests."""
        logger.info("Starting test run")
        start_time = time.time()
        
        tester = APITester()
        try:
            # Test word lookups
            for word in self.test_words:
                test_start = time.time()
                api_data, db_data = tester.test_word_lookup(word)
                self.timings['word_lookup'].append(time.time() - test_start)
                self.results['word_lookup'].append({
                    'word': word,
                    'api_data': api_data,
                    'db_data': db_data,
                    'match': api_data == db_data if api_data and db_data else False
                })
            
            # Test searches
            for query, params in self.search_queries:
                test_start = time.time()
                api_data, db_data = tester.test_search(query, **params)
                self.timings['search'].append(time.time() - test_start)
                self.results['search'].append({
                    'query': query,
                    'params': params,
                    'api_data': api_data,
                    'db_data': db_data,
                    'match': api_data == db_data if api_data and db_data else False
                })
            
            # Test word details
            for word in self.test_words[:5]:  # Test first 5 words for details
                # Relations
                test_start = time.time()
                api_data, db_data = tester.test_word_relations(word)
                self.timings['relations'].append(time.time() - test_start)
                self.results['relations'].append({
                    'word': word,
                    'api_data': api_data,
                    'db_data': db_data,
                    'match': api_data == db_data if api_data and db_data else False
                })
                
                # Etymology
                test_start = time.time()
                api_data, db_data = tester.test_word_etymology(word)
                self.timings['etymology'].append(time.time() - test_start)
                self.results['etymology'].append({
                    'word': word,
                    'api_data': api_data,
                    'db_data': db_data,
                    'match': api_data == db_data if api_data and db_data else False
                })
                
                # Pronunciation
                test_start = time.time()
                api_data, db_data = tester.test_word_pronunciation(word)
                self.timings['pronunciation'].append(time.time() - test_start)
                self.results['pronunciation'].append({
                    'word': word,
                    'api_data': api_data,
                    'db_data': db_data,
                    'match': api_data == db_data if api_data and db_data else False
                })
            
            # Test statistics
            test_start = time.time()
            api_data, db_data = tester.test_statistics()
            self.results['statistics'] = {
                'api_data': api_data,
                'db_data': db_data,
                'match': api_data == db_data if api_data and db_data else False
            }
            
        finally:
            tester.close()
        
        self.total_time = time.time() - start_time
        logger.info(f"Test run completed in {self.total_time:.2f} seconds")
    
    def generate_report(self):
        """Generate test report."""
        report_dir = Path('test_reports')
        report_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"test_report_{timestamp}"
        report_path.mkdir(exist_ok=True)
        
        # Generate summary
        summary = {
            'total_time': self.total_time,
            'total_tests': sum(len(results) for results in self.results.values() if isinstance(results, list)),
            'successful_tests': sum(
                sum(1 for r in results if r['match'])
                for results in self.results.values()
                if isinstance(results, list)
            ),
            'average_timings': {
                endpoint: sum(times) / len(times)
                for endpoint, times in self.timings.items()
                if times
            }
        }
        
        # Save raw results
        with open(report_path / 'raw_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate timing plots
        plt.figure(figsize=(12, 6))
        for endpoint, times in self.timings.items():
            if times:
                sns.boxplot(data=times, label=endpoint)
        plt.title('API Response Times by Endpoint')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(report_path / 'timing_boxplot.png')
        plt.close()
        
        # Generate success rate plot
        success_rates = {
            endpoint: sum(r['match'] for r in results) / len(results) * 100
            for endpoint, results in self.results.items()
            if isinstance(results, list) and results
        }
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()))
        plt.title('API Success Rate by Endpoint')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(report_path / 'success_rates.png')
        plt.close()
        
        # Generate detailed report
        report = []
        report.append("# Filipino Dictionary API Test Report")
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n## Summary")
        report.append(f"- Total time: {summary['total_time']:.2f} seconds")
        report.append(f"- Total tests: {summary['total_tests']}")
        report.append(f"- Successful tests: {summary['successful_tests']}")
        report.append(f"- Success rate: {summary['successful_tests'] / summary['total_tests'] * 100:.2f}%")
        
        report.append("\n## Average Response Times")
        timing_table = []
        for endpoint, avg_time in summary['average_timings'].items():
            timing_table.append([endpoint, f"{avg_time:.3f}s"])
        report.append(tabulate(timing_table, headers=['Endpoint', 'Average Time'], tablefmt='pipe'))
        
        report.append("\n## Success Rates by Endpoint")
        success_table = []
        for endpoint, rate in success_rates.items():
            success_table.append([endpoint, f"{rate:.2f}%"])
        report.append(tabulate(success_table, headers=['Endpoint', 'Success Rate'], tablefmt='pipe'))
        
        report.append("\n## Detailed Results")
        for endpoint, results in self.results.items():
            report.append(f"\n### {endpoint.title()}")
            if isinstance(results, list):
                for result in results:
                    if endpoint == 'word_lookup':
                        report.append(f"\n#### Word: {result['word']}")
                    elif endpoint == 'search':
                        report.append(f"\n#### Query: {result['query']} (Params: {result['params']})")
                    report.append(f"- Match: {'✓' if result['match'] else '✗'}")
            elif endpoint == 'statistics':
                report.append(f"- Match: {'✓' if results['match'] else '✗'}")
        
        # Save report
        with open(report_path / 'report.md', 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report generated at {report_path}")
        return report_path

def main():
    """Main function to run tests and generate report."""
    runner = TestRunner()
    runner.run_tests()
    report_path = runner.generate_report()
    logger.info(f"Test report available at: {report_path}")

if __name__ == "__main__":
    main() 