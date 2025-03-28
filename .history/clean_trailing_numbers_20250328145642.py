#!/usr/bin/env python3
"""
Script to clean trailing numbers from strings in tagalog-words.json
This script specifically targets strings that end with a number like "KUSLAB1"
"""

import json
import re
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Regular expression to match strings ending with a number
TRAILING_NUMBER_PATTERN = re.compile(r'^(.+?)(\d+)$')

def remove_trailing_numbers(value):
    """
    Recursively process JSON values to remove trailing numbers from strings.
    Returns: The processed value and a count of changes made
    """
    changes_count = 0
    
    # Process strings
    if isinstance(value, str):
        match = TRAILING_NUMBER_PATTERN.match(value)
        if match:
            new_value = match.group(1)  # Get the string without the trailing number
            logger.debug(f"Changed: '{value}' -> '{new_value}'")
            return new_value, 1
        return value, 0
    
    # Process lists
    elif isinstance(value, list):
        new_list = []
        for item in value:
            processed_item, changes = remove_trailing_numbers(item)
            new_list.append(processed_item)
            changes_count += changes
        return new_list, changes_count
    
    # Process dictionaries
    elif isinstance(value, dict):
        new_dict = {}
        for key, item in value.items():
            processed_item, changes = remove_trailing_numbers(item)
            new_dict[key] = processed_item
            changes_count += changes
        return new_dict, changes_count
    
    # Return other types unchanged
    else:
        return value, 0

def main():
    """Main function to process the tagalog-words.json file"""
    # File paths
    input_file = Path('data/tagalog-words.json')
    backup_file = Path(f'data/tagalog-words.json.bak.{datetime.now().strftime("%Y%m%d%H%M%S")}')
    output_file = Path('data/tagalog-words.json.cleaned')
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Create a backup first
    logger.info(f"Creating backup at {backup_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return 1
    
    # Process the file
    logger.info(f"Processing {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Clean trailing numbers
        logger.info("Removing trailing numbers from strings...")
        processed_data, changes_count = remove_trailing_numbers(data)
        
        # Save processed data
        logger.info(f"Writing cleaned data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Made {changes_count} changes")
        
        # Give instructions for replacing the original file
        if changes_count > 0:
            logger.info(f"\nTo replace the original file with the cleaned version, run:")
            logger.info(f"  mv {output_file} {input_file}")
        else:
            logger.info("No changes were needed.")
            os.remove(output_file)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 