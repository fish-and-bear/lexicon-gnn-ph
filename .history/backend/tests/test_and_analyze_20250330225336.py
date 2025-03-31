import os
import sys
import logging
import subprocess
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command and return True if successful."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Run tests and analysis in sequence."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Starting API test and analysis process...")
    
    # Step 1: Run the test suite
    logger.info("\nStep 1: Running test suite...")
    if not run_command("python run_tests.py", cwd=script_dir):
        logger.error("Test suite failed")
        sys.exit(1)
    
    # Step 2: Analyze test results
    logger.info("\nStep 2: Analyzing test results...")
    if not run_command("python analyze_results.py", cwd=script_dir):
        logger.error("Analysis failed")
        sys.exit(1)
    
    # Step 3: Check for any critical issues
    logger.info("\nStep 3: Checking for critical issues...")
    analysis_dir = os.path.join(script_dir, "analysis_reports")
    latest_analysis = max(
        [f for f in os.listdir(analysis_dir) if f.endswith('.json')],
        key=lambda x: os.path.getctime(os.path.join(analysis_dir, x))
    )
    
    with open(os.path.join(analysis_dir, latest_analysis), 'r') as f:
        analysis = eval(f.read())
    
    critical_issues = [
        s for s in analysis['improvement_suggestions']
        if s['priority'] == 'high'
    ]
    
    if critical_issues:
        logger.warning("\nCritical issues found:")
        for issue in critical_issues:
            logger.warning(f"- {issue['message']}")
        sys.exit(1)
    
    logger.info("\nTest and analysis completed successfully!")
    logger.info("No critical issues found.")

if __name__ == "__main__":
    main() 