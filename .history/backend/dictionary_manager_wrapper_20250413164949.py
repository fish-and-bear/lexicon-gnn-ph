"""
Run this script to execute dictionary_manager.py with proper imports.
This is a wrapper that handles the module import issues.
"""
import os
import sys
import logging
import importlib.util

# Add current directory to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# First try to import the module directly
try:
    # Try to import directly from the module in the same directory
    from source_standardization import extract_etymology_components
    print("Successfully imported source_standardization directly")
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

# Monkey patch the import into sys.modules to ensure dictionary_manager can find it
sys.modules['source_standardization'] = type('', (), {
    'extract_etymology_components': extract_etymology_components
})

# Now we can safely import dictionary_manager
try:
    import dictionary_manager
    print("Successfully imported dictionary_manager")
    
    # Execute dictionary_manager's main function if it exists and script is run directly
    if __name__ == "__main__":
        if hasattr(dictionary_manager, 'main') and callable(dictionary_manager.main):
            print("Running dictionary_manager.main()")
            import sys
            dictionary_manager.main(sys.argv[1:] if len(sys.argv) > 1 else [])
        else:
            print("No main function found in dictionary_manager module")
except Exception as e:
    print(f"Error importing or running dictionary_manager: {e}")
    import traceback
    traceback.print_exc() 