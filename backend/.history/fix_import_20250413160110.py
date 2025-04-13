import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Now import the modules
import source_standardization
from source_standardization import extract_etymology_components

# Import dictionary_manager after adding the path
import dictionary_manager

print("Imports successful!")
print(f"extract_etymology_components function: {extract_etymology_components}") 