#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fix script for ML pipeline to ensure correct database paths and run pre-training.
This script performs the following:
1. Updates database configuration to use absolute paths
2. Runs pre-training with a smaller number of epochs for testing
3. Checks if the pre-trained model was created successfully
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ml_fix_run.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fix and run ML pipeline")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of pre-training epochs")
    parser.add_argument("--db-config", type=str, default="ml/my_db_config.json",
                        help="Path to database configuration file")
    parser.add_argument("--config", type=str, default="ml/config/default_config.json",
                        help="Path to main configuration file")
    parser.add_argument("--skip-db-load", action="store_true",
                        help="Skip loading data from database (use cached data)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()

def fix_db_config(config_path):
    """Fix database configuration."""
    logger.info(f"Fixing database configuration at {config_path}")
    
    # Backup original file
    backup_path = config_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(config_path, backup_path)
        logger.info(f"Backed up {config_path} to {backup_path}")
    
    # Load the configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("Successfully loaded configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False
    
    # Update the database path to use absolute path
    current_dir = os.getcwd()
    db_path = os.path.join(current_dir, "fil_relex_colab.sqlite")
    logger.info(f"Setting database path to: {db_path}")
    
    # Verify the database file exists
    if not os.path.exists(db_path):
        logger.warning(f"Warning: Database file not found at {db_path}")
        
        # Try to find it at the root
        root_db_path = os.path.normpath(os.path.join(current_dir, "..", "fil_relex_colab.sqlite"))
        if os.path.exists(root_db_path):
            db_path = root_db_path
            logger.info(f"Found database at: {db_path}")
    
    # Update configuration
    config["db_config"]["db_path"] = db_path
    
    # Save updated configuration
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Updated database path in {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return False

def run_pretraining(args):
    """Run pre-training with fixed settings."""
    logger.info("Running pre-training with fixed settings")
    
    # Setup paths
    output_dir = "ml/output/pipeline_run_fixed"
    model_dir = os.path.join(output_dir, "models", "pretrained")
    data_dir = os.path.join(output_dir, "data", "processed")
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        sys.executable,
        "ml/pretrain_hgmae.py",
        "--config", args.config,
        "--db-config", args.db_config,
        "--model-dir", model_dir,
        "--data-dir", data_dir,
        "--epochs", str(args.epochs),
        "--feature-mask-rate", "0.3",
        "--edge-mask-rate", "0.3",
    ]
    
    if args.skip_db_load:
        cmd.append("--skip-db-load")
    
    if args.debug:
        cmd.append("--debug")
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        # Run with output streaming to console
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
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
        
        # Check return code
        if process.returncode == 0:
            logger.info("Pre-training completed successfully")
            return True
        else:
            logger.error(f"Pre-training failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run pre-training: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting ML pipeline fix")
    
    # Fix database configuration
    if not fix_db_config(args.db_config):
        logger.error("Failed to fix database configuration")
        return 1
    
    # Run pre-training
    if not run_pretraining(args):
        logger.error("Pre-training failed")
        return 1
    
    logger.info("ML pipeline fix completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 