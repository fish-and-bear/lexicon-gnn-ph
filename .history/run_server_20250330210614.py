"""
Script to run the Filipino Dictionary API server.
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from serve import main

if __name__ == '__main__':
    main() 