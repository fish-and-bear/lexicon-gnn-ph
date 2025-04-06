"""
Script to fix import statements in model files.
"""

import os
import re

def fix_imports_in_file(file_path):
    """Fix import statements in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace various import patterns
    replacements = [
        (r'from database import db', 'from backend.database import db'),
        (r'from database import cached_query', 'from backend.database import cached_query'),
        (r'from database import db, cached_query', 'from backend.database import db, cached_query'),
        (r'from database import invalidate_cache', 'from backend.database import invalidate_cache'),
        (r'import database', 'import backend.database'),
        (r'from \.database', 'from backend.database'),
        (r'from \.\.database', 'from backend.database'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in {file_path}")

def process_directory(directory):
    """Process all Python files in a directory recursively."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_imports_in_file(file_path)

if __name__ == '__main__':
    # Fix imports in models, utils, and other directories
    process_directory('backend/models')
    process_directory('backend/utils')
    process_directory('backend/gql')
    process_directory('backend')
    
    print("Import statements fixed!") 