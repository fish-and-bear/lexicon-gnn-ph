"""
Helper module to import source_standardization functions correctly
regardless of how the application is run.
"""
import os
import sys
import logging

# Add current directory to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Try to import directly from the module in the same directory
    from source_standardization import extract_etymology_components
except ImportError as e:
    logging.warning(f"Could not import source_standardization directly: {e}")
    
    # Define fallback function
    def extract_etymology_components(etymology_text):
        """Fallback implementation when the module can't be imported."""
        if not etymology_text:
            return []
        
        # Simple implementation to extract words that look like they might be components
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', etymology_text)
        return [word.lower() for word in words if len(word) > 2] 