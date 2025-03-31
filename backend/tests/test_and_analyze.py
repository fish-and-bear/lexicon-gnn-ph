import os
import sys
import logging
import subprocess
import datetime
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd: str) -> bool:
    """Run a shell command and log output."""
    try:
        logger.info(f"Running command: {cmd}")
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get any remaining output
        stdout, stderr = process.communicate()
        if stdout:
            logger.info(stdout.strip())
        if stderr:
            logger.error(stderr.strip())
        
        return process.returncode == 0
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def check_critical_issues(analysis: Dict[str, Any]) -> bool:
    """Check for critical issues in test results."""
    critical_issues = []
    
    # Check test results
    if analysis['test_results']['success_rate'] < 90:
        critical_issues.append(
            f"Low test success rate: {analysis['test_results']['success_rate']:.1f}%"
        )
    
    if analysis['test_results']['failures'] > 0:
        critical_issues.append(
            f"Test failures: {analysis['test_results']['failures']}"
        )
    
    if analysis['test_results']['errors'] > 0:
        critical_issues.append(
            f"Test errors: {analysis['test_results']['errors']}"
        )
    
    # Check performance
    if analysis['performance_metrics']['average_time'] > 2.0:
        critical_issues.append(
            f"High average response time: {analysis['performance_metrics']['average_time']:.2f}s"
        )
    
    # Check for high-priority improvement suggestions
    high_priority_issues = [
        s for s in analysis['improvement_suggestions']
        if s['priority'].lower() == 'high'
    ]
    if high_priority_issues:
        critical_issues.extend([
            f"High priority issue: {issue['description']}"
            for issue in high_priority_issues
        ])
    
    if critical_issues:
        logger.error("Critical issues found:")
        for issue in critical_issues:
            logger.error(f"- {issue}")
        return True
    
    return False

def main():
    """Run tests and analysis."""
    logger.info("Starting test and analysis process...")
    start_time = datetime.datetime.now()
    
    # Run tests
    logger.info("\nRunning tests...")
    if not run_command("python run_tests.py"):
        logger.error("Test execution failed")
        return False
    
    # Run analysis
    logger.info("\nAnalyzing test results...")
    sys.path.append(os.path.dirname(__file__))
    from analyze_results import analyze_latest_report
    
    analysis = analyze_latest_report()
    if not analysis:
        logger.error("Analysis failed")
        return False
    
    # Check for critical issues
    if check_critical_issues(analysis):
        logger.error("Critical issues found in analysis")
        return False
    
    # Calculate total execution time
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    logger.info("\nTest and analysis process completed successfully")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 