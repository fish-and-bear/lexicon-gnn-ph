"""
Source standardization utilities for handling etymology and source references.
"""

import re
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SourceStandardization:
    """Handles standardization of source references and citations."""
    
    STANDARD_SOURCES = {
        'kwf': 'Komisyon sa Wikang Filipino',
        'upd': 'UP Diksiyonaryong Filipino',
        'diksiyonaryo': 'Diksiyonaryo.ph',
        'leo': 'Leo James English Dictionary',
        'santos': 'Vito Santos Dictionary',
        'panganiban': 'Diksiyonaryong Pilipino',
    }

    @classmethod
    def standardize_sources(cls, sources: str) -> str:
        """Standardize source references to a consistent format."""
        if not sources:
            return ''
        
        source_list = [s.strip().lower() for s in sources.split(',')]
        standardized = []
        
        for source in source_list:
            if source in cls.STANDARD_SOURCES:
                standardized.append(source)
            else:
                # Try to match partial names
                matched = False
                for code, name in cls.STANDARD_SOURCES.items():
                    if source in name.lower():
                        standardized.append(code)
                        matched = True
                        break
                if not matched:
                    standardized.append(source)  # Keep original if no match
        
        return ', '.join(sorted(set(standardized)))

    @classmethod
    def get_display_name(cls, source_code: str) -> str:
        """Get the display name for a source code."""
        return cls.STANDARD_SOURCES.get(source_code.lower(), source_code)

def extract_etymology_components(etymology_text: str) -> List[str]:
    """Extract word components from etymology text."""
    if not etymology_text:
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
    
    return list(set(cleaned))  # Remove duplicates

def extract_meaning(etymology_text: str) -> Tuple[str, Optional[str]]:
    """Extract meaning from etymology text if present."""
    if not etymology_text:
        return '', None
    
    # Common patterns for meaning extraction
    meaning_patterns = [
        r'meaning\s+"([^"]+)"',
        r'meaning\s+\'([^\']+)\'',
        r'lit\.\s+"([^"]+)"',
        r'literally\s+"([^"]+)"'
    ]
    
    for pattern in meaning_patterns:
        match = re.search(pattern, etymology_text, re.IGNORECASE)
        if match:
            meaning = match.group(1).strip()
            # Remove the meaning part from the text
            text = re.sub(pattern, '', etymology_text, flags=re.IGNORECASE).strip()
            return text, meaning
    
    return etymology_text, None 