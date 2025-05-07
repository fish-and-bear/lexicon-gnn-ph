"""
Source standardization utilities for handling etymology and source references.
"""

import re
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SourceStandardization:
    """Handles standardization of source references and citations."""
    
    # Comprehensive configuration for sources
    SOURCE_CONFIG: List[Dict[str, Union[List[str], str]]] = [
        # Based on dictionary_manager.py migrate_data
        {
            "identifiers": ["root_words_with_associated_words_cleaned.json", "root_words_with_associated_words_cleaned", "root words", "tagalog.com"],
            "canonical_id": "tagalog_com_roots",
            "display_name": "tagalog.com"
        },
        {
            "identifiers": ["kwf_dictionary.json", "kwf dictionary", "kwf"],
            "canonical_id": "kwf",
            "display_name": "KWF Diksiyonaryo ng Wikang Filipino"
        },
        {
            "identifiers": ["kaikki.jsonl", "kaikki.org (tagalog)", "kaikki", "kaikki_tagalog"],
            "canonical_id": "kaikki_tl",
            "display_name": "Kaikki.org (Tagalog)"
        },
        {
            "identifiers": ["kaikki-ceb.jsonl", "kaikki.org (cebuano)", "kaikki_cebuano"],
            "canonical_id": "kaikki_ceb",
            "display_name": "Kaikki.org (Cebuano)"
        },
        {
            "identifiers": ["marayum_dictionaries", "project marayum", "marayum-"],
            "canonical_id": "marayum_generic", 
            "display_name": "Project Marayum"
        },
        {
            "identifiers": ["gay-slang.json", "philippine slang and gay dictionary", "gayslang"],
            "canonical_id": "gay_slang_dict",
            "display_name": "Philippine Slang and Gay Dictionary (2023)"
        },
        {
            "identifiers": ["tagalog-words.json", "tagalog words", "diksiyonaryo.ph", "diksiyonaryo"],
            "canonical_id": "diksiyonaryo_ph",
            "display_name": "diksiyonaryo.ph"
        },
        {
            "identifiers": ["calderon_dictionary.json", "calderon diccionario 1915", "calderon"],
            "canonical_id": "calderon_1915",
            "display_name": "Calderon Diccionario (1915)"
        },
        
        # Other known sources (previously in config, kept for broader compatibility)
        {
            "identifiers": ["upd", "up diksiyonaryong filipino", "up_diksiyonaryong_filipino"],
            "canonical_id": "upd",
            "display_name": "UP Diksiyonaryong Filipino"
        },
        {
            "identifiers": ["leo james english dictionary", "leo", "leo_james"],
            "canonical_id": "leo_james_english",
            "display_name": "Leo James English Dictionary"
        },
        {
            "identifiers": ["vito santos dictionary", "santos", "vito_santos"],
            "canonical_id": "vito_santos",
            "display_name": "Vito Santos Dictionary"
        },
        {
            "identifiers": ["diksiyonaryong pilipino", "panganiban", "panganiban_diksiyonaryo"],
            "canonical_id": "panganiban_pilipino",
            "display_name": "Panganiban - Diksiyonaryong Pilipino"
        }
    ]

    # Default entry for unknown sources
    UNKNOWN_SOURCE_CANONICAL_ID = "unknown_source"
    UNKNOWN_SOURCE_DISPLAY_NAME = "Unknown Source"

    @classmethod
    def standardize_sources(cls, raw_source_input: Optional[str]) -> str:
        """Standardize a raw source input string to a canonical source ID.
        The input can be a filename, a short tag, or a descriptive name.
        """
        if not raw_source_input or not isinstance(raw_source_input, str):
            return cls.UNKNOWN_SOURCE_CANONICAL_ID

        normalized_input = raw_source_input.strip().lower()
        
        # ADDED: Check if the input is already a canonical ID
        for config_entry in cls.SOURCE_CONFIG:
            if str(config_entry["canonical_id"]).lower() == normalized_input:
                logger.debug(f"Input '{raw_source_input}' is already a canonical ID. Returning it directly as '{config_entry['canonical_id']}'.")
                return str(config_entry["canonical_id"]) # Return the original case from config
        
        # Attempt to remove common file extensions for better matching against identifiers
        cleaned_input = re.sub(r'\\.(jsonl|json|csv|txt)$', '', normalized_input)
        if not cleaned_input: # If only extension was provided
            cleaned_input = normalized_input # revert to original normalized input

        for config_entry in cls.SOURCE_CONFIG:
            # Ensure identifiers is always treated as a list, even if it's a single string in config
            current_identifiers = config_entry["identifiers"]
            if not isinstance(current_identifiers, list):
                current_identifiers = [str(current_identifiers)] # Convert to list if it's a single string
            
            for identifier_pattern in current_identifiers:
                pattern_lower = identifier_pattern.lower()
                # Exact match on cleaned input OR pattern is substring of cleaned input OR input is substring of pattern (for partials like 'marayum-')
                if cleaned_input == pattern_lower or \
                   pattern_lower in cleaned_input or \
                   (pattern_lower.endswith('-') and cleaned_input.startswith(pattern_lower[:-1])) or \
                   (cleaned_input.endswith('-') and pattern_lower.startswith(cleaned_input[:-1])):
                    return str(config_entry["canonical_id"])
        
        # If no specific match, log and return a generic or unknown ID.
        logger.debug(f"No specific source configuration found for input: '{raw_source_input}'. Generating generic ID from input.")
        # Generate a slug-like ID from the original (cleaned) input if no match
        generic_id = re.sub(r'[^a-z0-9_.-]+', '_', cleaned_input).strip('_')
        
        if not generic_id: # if input was all special chars or became empty
             return cls.UNKNOWN_SOURCE_CANONICAL_ID
        return generic_id

    @classmethod
    def get_display_name(cls, canonical_id: Optional[str]) -> str:
        """Get the display name for a canonical source ID."""
        if not canonical_id or not isinstance(canonical_id, str):
            return cls.UNKNOWN_SOURCE_DISPLAY_NAME

        for config_entry in cls.SOURCE_CONFIG:
            if config_entry["canonical_id"] == canonical_id:
                return str(config_entry["display_name"])
        
        logger.warning(f"No display name found in SOURCE_CONFIG for canonical_id: '{canonical_id}'. Returning a formatted version of the ID or unknown.")
        if canonical_id == cls.UNKNOWN_SOURCE_CANONICAL_ID:
            return cls.UNKNOWN_SOURCE_DISPLAY_NAME
        # Attempt to pretty-print the ID if it's not found in config
        return canonical_id.replace('_', ' ').replace('-', ' ').title()

def extract_etymology_components(etymology_text: str) -> List[str]:
    """Extract word components from etymology text."""
    if not etymology_text:
        return []
    
    # Skip bracketed language codes like "[ Ing ]" or "[ Esp ]" as they're not actual components
    # These indicate language origin but aren't component words themselves
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