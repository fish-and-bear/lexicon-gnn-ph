#!/usr/bin/env python3
"""
This script patches dictionary_manager.py to remove the problematic import
and replace it with an inline implementation.
"""
import re
import os

DICTIONARY_MANAGER_PATH = os.path.join(os.path.dirname(__file__), 'dictionary_manager.py')

# First, define the replacement function content
REPLACEMENT_FUNCTION = """
# Define the extract_etymology_components function directly instead of importing it
def source_extract_etymology_components(etymology_text):
    \"\"\"Extract word components from etymology text.\"\"\"
    if not etymology_text:
        return []
    
    # Skip bracketed language codes like "[ Ing ]" or "[ Esp ]" as they're not actual components
    if re.match(r'^\\s*\\[\\s*(?:Ing|Esp|War|San|Arb|Ch|Jap|Mal|Tsino)\\s*\\]\\s*$', etymology_text, re.IGNORECASE):
        # Return empty list for these cases - they're language indicators, not components
        return []
    
    # Common patterns for component extraction
    patterns = [
        r'from\\s+([^\\s,;.]+)',
        r'derived from\\s+([^\\s,;.]+)',
        r'compound of\\s+([^\\s,;.]+)\\s+and\\s+([^\\s,;.]+)',
        r'combining\\s+([^\\s,;.]+)\\s+with\\s+([^\\s,;.]+)',
        r'root word\\s+([^\\s,;.]+)'
    ]
    
    components = []
    for pattern in patterns:
        matches = re.finditer(pattern, etymology_text, re.IGNORECASE)
        for match in matches:
            components.extend(match.groups())
    
    # Clean and normalize components
    cleaned = []
    for comp in components:
        # Remove punctuation and normalize
        cleaned_comp = re.sub(r'[^\\w\\s-]', '', comp).strip().lower()
        if cleaned_comp and len(cleaned_comp) > 1:  # Ignore single letters
            cleaned.append(cleaned_comp)
    
    # If no components were found using patterns, extract all words as a fallback
    if not cleaned:
        words = re.findall(r'\\b[a-zA-Z]{3,}\\b', etymology_text.lower())
        cleaned = list(set(words))  # Remove duplicates
    
    return cleaned
"""

# Function to patch the file
def patch_dictionary_manager():
    """Patch dictionary_manager.py to fix the import issue."""
    if not os.path.exists(DICTIONARY_MANAGER_PATH):
        print(f"Error: {DICTIONARY_MANAGER_PATH} not found!")
        return False

    with open(DICTIONARY_MANAGER_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the problematic import
    pattern = r'from\s+source_standardization\s+import\s+extract_etymology_components\s+as\s+source_extract_etymology_components'
    if not re.search(pattern, content):
        print("Error: Could not find the import statement to replace.")
        return False

    # Replace the import with our inline function definition
    modified_content = re.sub(pattern, REPLACEMENT_FUNCTION, content)

    # Create a backup of the original file
    backup_path = DICTIONARY_MANAGER_PATH + '.bak'
    print(f"Creating backup at {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Write the modified content
    print(f"Patching {DICTIONARY_MANAGER_PATH}")
    with open(DICTIONARY_MANAGER_PATH, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print("Patch applied successfully.")
    return True

if __name__ == "__main__":
    print("Running dictionary_manager.py patcher...")
    if patch_dictionary_manager():
        print("Patch completed successfully. Try running the application now.")
    else:
        print("Patch failed. Please fix the issue manually.") 