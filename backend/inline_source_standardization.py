"""
Inline implementation of source_standardization module
This module is designed to be imported early in the app initialization
to ensure the source_standardization module is available to all other modules.
"""
import sys
import re

def extract_etymology_components(etymology_text):
    """Extract word components from etymology text."""
    if not etymology_text:
        return []
    
    # Skip bracketed language codes like "[ Ing ]" or "[ Esp ]" as they're not actual components
    if re.match(r'^\s*\[\s*(?:Ing|Esp|War|San|Arb|Ch|Jap|Mal|Tsino)\s*\]\s*$', etymology_text, re.IGNORECASE):
        # Return empty list for these cases - they're language indicators, not components
        return []
    
    # Common patterns for component extraction
    patterns = [
        r'from\s+([^\s,;.]+)',
        r'derived from\s+([^\s,;.]+)',
        r'compound of\s+([^\s,;.]+)\s+and\s+([^\s,;.]+)',
        r'combining\s+([^\s,;.]+)\s+with\s+([^\s,;.]+)',
        r'root word\s+([^\s,;.]+)'
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
        cleaned_comp = re.sub(r'[^\w\s-]', '', comp).strip().lower()
        if cleaned_comp and len(cleaned_comp) > 1:  # Ignore single letters
            cleaned.append(cleaned_comp)
    
    # If no components were found using patterns, extract all words as a fallback
    if not cleaned:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', etymology_text.lower())
        cleaned = list(set(words))  # Remove duplicates
    
    return cleaned

# Install this module as 'source_standardization' in sys.modules
sys.modules['source_standardization'] = sys.modules[__name__] 