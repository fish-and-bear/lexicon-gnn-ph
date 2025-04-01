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
import subprocess
import signal
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for the server to become available."""
    start_time = time.time()
    while True:
        try:
            requests.get(f"http://localhost:{port}/api/v2/statistics")
            return True
        except requests.ConnectionError:
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)

def start_api_server() -> subprocess.Popen:
    """Start the API server."""
    logger.info("Starting API server...")
    server_process = subprocess.Popen(
        [sys.executable, "deploy_local.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    if not wait_for_server(10000):
        logger.error("Failed to start API server")
        server_process.terminate()
        raise RuntimeError("API server failed to start")
    
    logger.info("API server started successfully")
    return server_process

def stop_api_server(server_process: subprocess.Popen):
    """Stop the API server."""
    logger.info("Stopping API server...")
    server_process.terminate()
    server_process.wait()
    logger.info("API server stopped")

class TestRunner:
    """Class to run tests and generate reports."""
    
    def __init__(self):
        self.results = {
            'word_lookup': [],
            'search': [],
            'etymology': [],
            'relations': [],
            'affixations': [],
            'pronunciation': []
        }
        self.server_process = None

    def run_tests(self):
        """Run all tests and collect results."""
        try:
            # Start API server
            self.server_process = start_api_server()
            
            # Initialize API tester
            tester = APITester()
            
            # Run tests
            logger.info("Starting test run")
            
            # Test word lookup
            test_words = ['aklat', 'bata', 'ganda']
            for word in test_words:
                logger.info(f"Testing word lookup for: {word}")
                api_data, db_data = tester.test_word_lookup(word)
                self.results['word_lookup'].append({
                    'word': word,
                    'api_data': api_data,
                    'db_data': db_data
                })
            
            # Test search functionality
            search_queries = ['bata', 'ganda', 'mahal']
            for query in search_queries:
                logger.info(f"Testing search for: {query}")
                api_data, db_data = tester.test_search(query)
                self.results['search'].append({
                    'query': query,
                    'api_data': api_data,
                    'db_data': db_data
                })
            
            # Generate report
            self.generate_report()
            
        finally:
            # Stop API server
            if self.server_process:
                stop_api_server(self.server_process)

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
                        report.append(f"\n#### Query: {result['query']}")
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