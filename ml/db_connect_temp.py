#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script to fix database connection path in configuration files.
This script updates the database path in the configuration to point to the correct location.
"""

import os
import json
import shutil
import sys

def fix_db_config():
    print("Starting database configuration fix...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Paths to configuration files
    my_db_config_path = "ml/my_db_config.json"
    print(f"Looking for config at: {os.path.abspath(my_db_config_path)}")
    
    # Check if file exists
    if not os.path.exists(my_db_config_path):
        print(f"ERROR: Config file not found: {my_db_config_path}")
        return False
    
    # Backup original file
    backup_path = my_db_config_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy(my_db_config_path, backup_path)
        print(f"Backed up {my_db_config_path} to {backup_path}")
    
    # Load the configuration
    try:
        with open(my_db_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Successfully loaded configuration")
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return False
    
    # Update the database path to use absolute path
    current_dir = os.getcwd()
    db_path = os.path.join(current_dir, "fil_relex_colab.sqlite")
    print(f"Database path to set: {db_path}")
    
    # Verify the database file exists
    if not os.path.exists(db_path):
        print(f"WARNING: Database file not found at {db_path}")
    else:
        print(f"Verified database exists at: {db_path}")
    
    # Update configuration
    try:
        config["db_config"]["db_path"] = db_path
        
        # Save updated configuration
        with open(my_db_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Updated database path in {my_db_config_path}")
    except Exception as e:
        print(f"ERROR: Failed to update configuration: {e}")
        return False
    
    print("Database configuration fixed. You can now run the pipeline again.")
    return True

if __name__ == "__main__":
    success = fix_db_config()
    sys.exit(0 if success else 1)
