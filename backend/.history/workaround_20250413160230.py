#!/usr/bin/env python

# Create a temporary copy of source_standardization.py in the backend folder
import shutil
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Copy source_standardization.py to backend/source_standardization.py
src_file = os.path.join(current_dir, 'source_standardization.py')
dst_file = os.path.join(current_dir, '..', 'backend', 'source_standardization.py')

try:
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print(f"Successfully copied {src_file} to {dst_file}")
except Exception as e:
    print(f"Error copying file: {e}")

print("Now try importing the module again.") 