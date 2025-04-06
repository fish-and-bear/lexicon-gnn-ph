"""Utility functions for word processing."""

import re
import unicodedata

def normalize_word(word):
    """
    Normalize a word by removing diacritics, converting to lowercase,
    and removing special characters.
    
    Args:
        word (str): The word to normalize
        
    Returns:
        str: The normalized word
    """
    if not word:
        return ""
        
    # Convert to lowercase
    word = word.lower()
    
    # Normalize unicode characters (convert diacritics to base characters)
    word = unicodedata.normalize('NFKD', word)
    word = ''.join([c for c in word if not unicodedata.combining(c)])
    
    # Remove special characters, keep only letters, numbers, and spaces
    word = re.sub(r'[^\w\s]', '', word)
    
    # Replace multiple spaces with a single space
    word = re.sub(r'\s+', ' ', word)
    
    # Strip leading and trailing spaces
    word = word.strip()
    
    return word 