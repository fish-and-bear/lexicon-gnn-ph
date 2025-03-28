#!/usr/bin/env python3
"""
Script to clean trailing numbers from strings in tagalog-words.json
This script specifically targets strings that end with a number like "KUSLAB1"
or "sagupà1" in any field, including nested fields like "synonyms" arrays.
It also processes words inside definition texts that end with numbers.
"""

import json
import re
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict
import argparse
import random

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

# Regular expression for handling special cases like hyphenated words with trailing numbers
HYPHENATED_NUMBER_PATTERN = re.compile(r'^(.+?)(\d+)-(\d+)$')

# Fields to exclude from processing (these fields are supposed to contain numbers)
EXCLUDED_FIELDS = {'counter'}

# Words that should not be processed (e.g. chemical names, vitamin codes)
EXCLUDED_PREFIXES = {'bitamína A', 'bitamína B', 'bitamína C', 'bitamína D', 'bitamína E', 'bitamína K'}

# Pattern to match individual words within text (for definitions)
WORD_WITH_TRAILING_NUMBER_PATTERN = re.compile(r'(\b[^\s\d.,;()]+?)(\d+)([.,;)\s]|$)')

# Statistics to track changes by field type
change_stats = defaultdict(int)
# Store examples of changes for each field type (limit to avoid memory issues)
change_examples = defaultdict(list)
MAX_EXAMPLES_PER_FIELD = 5

def clean_word_in_text(text):
    """
    Remove trailing numbers from words within a text string.
    
    Args:
        text: The text to process
        
    Returns:
        Tuple of (processed_text, count_of_changes)
    """
    if not text or not isinstance(text, str):
        return text, 0
    
    # Skip if this is a known excluded prefix
    if any(text.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return text, 0
    
    changes = 0
    result = text
    
    # Handle words embedded in text (like definitions containing references)
    def replace_word(match):
        nonlocal changes
        word = match.group(1)  # The word without the number
        number = match.group(2)  # The trailing number
        ending = match.group(3)  # The character after the number (space, comma, etc.)
        changes += 1
        return word + ending
    
    # Process words within the text
    processed_text = WORD_WITH_TRAILING_NUMBER_PATTERN.sub(replace_word, result)
    
    # Also handle special case for hyphenated words with trailing numbers
    hyphen_match = HYPHENATED_NUMBER_PATTERN.match(result)
    if hyphen_match:
        result = hyphen_match.group(1)  # Get the base word without numbers
        changes += 1
    # Handle regular case for entire string ending with a number
    else:
        match = TRAILING_NUMBER_PATTERN.match(result)
        if match:
            result = match.group(1)  # Get the string without the trailing number
            changes += 1
    
    # If both ways changed the text, use the one with more changes (likely the in-text processing)
    if processed_text != text and result != text:
        return processed_text, changes
    elif processed_text != text:
        return processed_text, changes
    elif result != text:
        return result, changes
    else:
        return text, 0

def remove_trailing_numbers(value, path=""):
    """
    Recursively process JSON values to remove trailing numbers from strings.
    
    Args:
        value: The value to process
        path: Current path in the JSON structure (for logging)
        
    Returns: 
        Tuple of (processed_value, changes_count)
    """
    changes_count = 0
    
    # Process strings
    if isinstance(value, str):
        # Skip if this is a counter field or other excluded field
        field_name = path.split('.')[-1] if '.' in path else path
        if field_name in EXCLUDED_FIELDS:
            return value, 0
        
        # Process for definition fields - more thorough text processing
        if field_name == 'definition':
            new_value, changes = clean_word_in_text(value)
            if changes > 0:
                change_stats[field_name] += changes
                if len(change_examples[field_name]) < MAX_EXAMPLES_PER_FIELD:
                    change_examples[field_name].append((value, new_value, path))
                return new_value, changes
        
        # Skip if this is a known excluded prefix (like vitamin codes)
        if any(value.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            return value, 0
        
        # Handle special case for hyphenated words with trailing numbers (like "bigáy1-2")
        hyphen_match = HYPHENATED_NUMBER_PATTERN.match(value)
        if hyphen_match:
            new_value = hyphen_match.group(1)  # Get the base word without numbers
            
            # Track where this change was found
            change_stats[field_name] += 1
            
            # Store example of this change (limit examples per field)
            if len(change_examples[field_name]) < MAX_EXAMPLES_PER_FIELD:
                change_examples[field_name].append((value, new_value, path))
                
            return new_value, 1
        
        # Handle comma-separated list of words (often in definitions)
        if ',' in value and field_name == 'definition':
            parts = value.split(',')
            new_parts = []
            part_changes = 0
            
            for part in parts:
                cleaned_part, part_change = clean_word_in_text(part.strip())
                new_parts.append(cleaned_part)
                part_changes += part_change
            
            if part_changes > 0:
                new_value = ', '.join(new_parts)
                change_stats[field_name] += part_changes
                
                # Store example of this change
                if len(change_examples[field_name]) < MAX_EXAMPLES_PER_FIELD:
                    change_examples[field_name].append((value, new_value, path))
                
                return new_value, part_changes
            
        # Handle regular case for words ending with numbers
        match = TRAILING_NUMBER_PATTERN.match(value)
        if match:
            new_value = match.group(1)  # Get the string without the trailing number
            
            # Track where this change was found
            change_stats[field_name] += 1
            
            # Store example of this change (limit examples per field)
            if len(change_examples[field_name]) < MAX_EXAMPLES_PER_FIELD:
                change_examples[field_name].append((value, new_value, path))
                
            return new_value, 1
        
        return value, 0
    
    # Process lists
    elif isinstance(value, list):
        new_list = []
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            processed_item, changes = remove_trailing_numbers(item, item_path)
            new_list.append(processed_item)
            changes_count += changes
        return new_list, changes_count
    
    # Process dictionaries
    elif isinstance(value, dict):
        new_dict = {}
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else key
            processed_item, changes = remove_trailing_numbers(item, item_path)
            new_dict[key] = processed_item
            changes_count += changes
        return new_dict, changes_count
    
    # Return other types unchanged
    else:
        return value, 0

def main():
    """Main function to process the tagalog-words.json file"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean trailing numbers from strings in JSON file')
    parser.add_argument('--input', '-i', default='data/tagalog-words.json', 
                        help='Input JSON file (default: data/tagalog-words.json)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file (default: {input}.cleaned)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating a backup of the input file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--apply', '-a', action='store_true',
                        help='Apply changes directly to input file (creates backup)')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run on a small random sample of entries first')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of random entries to use in test mode (default: 100)')
    parser.add_argument('--exclude-fields', type=str, default='counter',
                        help='Comma-separated list of field names to exclude from processing')
    parser.add_argument('--exclude-prefixes', type=str, default='',
                        help='Comma-separated list of word prefixes to exclude from processing')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only calculate statistics without making changes')
    parser.add_argument('--deep-clean', action='store_true',
                        help='Perform a deep clean of text fields, looking for words with trailing numbers')
    
    args = parser.parse_args()
    
    # Update excluded fields from command line
    global EXCLUDED_FIELDS
    if args.exclude_fields:
        EXCLUDED_FIELDS = set(field.strip() for field in args.exclude_fields.split(','))
    
    # Update excluded prefixes from command line
    global EXCLUDED_PREFIXES
    if args.exclude_prefixes:
        EXCLUDED_PREFIXES.update(prefix.strip() for prefix in args.exclude_prefixes.split(','))
    
    # File paths
    input_file = Path(args.input)
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"{input_file}.cleaned")
    
    backup_file = Path(f"{input_file}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    # Create a backup first (unless in test or stats-only mode)
    if not args.no_backup and not args.test and not args.stats_only:
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
        
        # If test mode, take a random sample of entries
        if args.test:
            logger.info(f"TEST MODE: Selecting {args.sample_size} random entries")
            all_keys = list(data.keys())
            sample_size = min(args.sample_size, len(all_keys))
            sample_keys = random.sample(all_keys, sample_size)
            sample_data = {key: data[key] for key in sample_keys}
            data = sample_data
            output_file = Path(f"{input_file}.test_cleaned")
        
        # Clean trailing numbers
        mode = "Deep clean" if args.deep_clean else "Standard clean"
        logger.info(f"{mode} - Removing trailing numbers from strings (excluding fields: {', '.join(EXCLUDED_FIELDS)})")
        processed_data, changes_count = remove_trailing_numbers(data)
        
        # If stats-only mode, just display statistics
        if args.stats_only:
            logger.info(f"STATS ONLY MODE: Found {changes_count} potential changes across {len(change_stats)} fields")
            for field, count in sorted(change_stats.items(), key=lambda x: x[1], reverse=True):
                logger.info(f" - {field}: {count} changes")
                
                # Show examples of changes for this field
                if field in change_examples and not args.quiet:
                    logger.info(f"   Examples:")
                    for old, new, path in change_examples[field][:MAX_EXAMPLES_PER_FIELD]:
                        logger.info(f"   - '{old}' -> '{new}' at {path}")
            return 0
        
        # Save processed data
        logger.info(f"Writing cleaned data to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        # Show detailed statistics unless quiet mode
        if not args.quiet:
            logger.info(f"Made {changes_count} changes across {len(change_stats)} fields:")
            for field, count in sorted(change_stats.items(), key=lambda x: x[1], reverse=True):
                logger.info(f" - {field}: {count} changes")
                
                # Show examples of changes for this field
                if field in change_examples:
                    logger.info(f"   Examples:")
                    for old, new, path in change_examples[field][:MAX_EXAMPLES_PER_FIELD]:
                        logger.info(f"   - '{old}' -> '{new}' at {path}")
        else:
            logger.info(f"Made {changes_count} changes. Use --quiet=False to see details.")
        
        # Special instructions for test mode
        if args.test:
            logger.info("\nTest run completed. To run on the full dataset, run the script without --test")
            return 0
            
        # If apply flag is set and changes were made, replace the original file
        if args.apply and changes_count > 0:
            logger.info(f"Applying changes to {input_file}")
            output_file.replace(input_file)
            logger.info(f"Changes applied. Original file backed up at {backup_file}")
        # Otherwise give instructions for replacing the original file
        elif changes_count > 0:
            logger.info(f"\nTo replace the original file with the cleaned version, run:")
            if sys.platform == 'win32':
                logger.info(f"  Move-Item -Path {output_file} -Destination {input_file} -Force")
            else:
                logger.info(f"  mv {output_file} {input_file}")
        else:
            logger.info("No changes were needed.")
            if output_file.exists():
                os.remove(output_file)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 