#!/usr/bin/env python3

"""
Simple test script to check if display_dictionary_stats function exists in the dictionary_manager module
"""

import os
import sys

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import the function
try:
    # Import the module
    import dictionary_manager
    
    # Check if the function exists
    print(f"Module dictionary_manager imported successfully")
    print(f"Module file path: {dictionary_manager.__file__}")
    
    # Check available attributes
    print("\nAvailable attributes in dictionary_manager module:")
    for attr in dir(dictionary_manager):
        if not attr.startswith('_'):  # Skip private attributes
            print(f"- {attr}")
            
    # Specifically check for the display_dictionary_stats function
    if hasattr(dictionary_manager, 'display_dictionary_stats'):
        print("\nFound display_dictionary_stats function!")
    else:
        print("\nCould not find display_dictionary_stats function.")
        
except ImportError as e:
    print(f"Error importing dictionary_manager: {e}")
except Exception as e:
    print(f"Unexpected error: {e}") 