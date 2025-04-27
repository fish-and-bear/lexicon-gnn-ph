#!/usr/bin/env python3
"""
text_helpers.py

Utility functions for text processing related to dictionary management.
"""

import re
import unicodedata
import unidecode
import logging
from typing import Optional, Tuple, Union, List, Dict
import dataclasses
import os
import json
import glob
from psycopg2.extras import Json # Ensure Json is imported

from backend.dictionary_manager.enums import BaybayinCharType

# Try to import BeautifulSoup for robust HTML cleaning
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup not found. HTML cleaning will be basic.")

logger = logging.getLogger(__name__)

# Regex for valid Baybayin characters (moved from dictionary_manager)
VALID_BAYBAYIN_REGEX = re.compile(
    r"^[\u1700-\u171F\s\u1735]+$"
)  # Added \u1735 for Philippine Single Punctuation

# Moved from dictionary_manager.py
# Set of language codes, descriptors, and other non-word strings often found in relation fields
NON_WORD_STRINGS = {
    # Language Codes (ISO 639) - Consolidating duplicates from original list
    "tl", "en", "es", "la", "ceb", "hil", "ilo", "bik", "war", "pam", "pag",
    "kpm", "mdh", "mrw", "tsg", "yka", "zh", "ms", "ar", "fr", "de", "ja",
    "ko", "hi", "nan", "nan-hbl", "sa", "phi-pro", "poz-pro", "pt", "ru",
    "it", "nl", "sv", "da", "no", "bcl", "sml", "phi", "poz", "ine-pro",
    "iir-pro", "grc", "fi", "pl", "tr", "vi", "th", "map-pro",
    # Descriptive/Orthographic terms
    "spanish-based orthography", "obsolete", "archaic", "dialectal", "regional",
    "formal", "informal", "slang", "colloquial", "figurative", "literal",
    "standard", "nonstandard", "proscribed", "dated", "rare", "poetic",
    "historical", "uncommon", "inclusive", "abbr.", "fig.", "lit.",
    # Place Names / Regions
    "Batangas", "Marinduque", "Rizal",
    # Placeholders or fragments
    "-", "?", "*", "+", "...",
}

# POS Mapping Constants (Moved from dictionary_manager.py)
POS_MAPPING = {
    # Nouns - Pangngalan ('n')
    "noun": "n", "pangngalan": "n", "name": "n", "n": "n", "pangalan": "n",
    "pangngalang pantangi": "n", "pangngalang pambalana": "n", "salitang ugat": "n",
    "salitang hango": "n", "pulutong": "n", "sangkap": "n", "uri": "n",
    "simbolo": "n", "dinidiinang pantig": "n", "pagbabago": "n", "palatandaan": "n",
    "pahiwatig": "n", "tugma": "n", "kahulugan": "n", "salitang hiram": "n",
    "damdamin": "n", "katumbas na salita": "n", "kabaligtaran": "n", "aspekto": "n",
    "kasarian": "n", "anyo": "n", "kayarian": "n", "gamit": "n", "pagkakaugnay": "n",
    "Pangngalan": "n", "sa sanga ng punongkahoy": "n", "sa anumang bukás na bahagi": "n",
    "sa isang tao": "n", "sa isang sakít na": "n", "sa isang rabáw": "n",
    "sa isang sasakyang-dagat": "n", "sa isang serbisyo, binili, o inutang": "n",
    "sa isang súgat": "n", "kung sa bakal": "n", "sa pananim": "n", "sa isang pook": "n",
    "sa isang materyal": "n", "sa dalawang tao": "n", "sa mga pananim": "n",
    "sa isang likido": "n", "sa isang salita": "n", "sa isang kilos": "n",
    "sa kagamitan": "n", "sa gamot": "n", "sa isang halaga": "n", "sa buhok": "n",
    "sa lupa": "n", "sa damit": "n", "sa isang sawsáwan": "n", "sa sakít": "n",
    "sa pagluluto": "n", "kung sa batok o ilong": "n", "png": "n",
    # Verbs - Pandiwa ('v')
    "verb": "v", "pandiwa": "v", "v": "v", "action": "v", "Pandiwa": "v", "pnd": "v",
    # Adjectives - Pang-uri ('adj')
    "adjective": "adj", "pang-uri": "adj", "adj": "adj", "quality": "adj",
    "uri": "adj", "antas na paghahambing": "adj", "Pang-uri": "adj",
    "pang-uri sa panlagay": "adj", "pang-uri sa panlapi": "adj", "pnr": "adj",
    # Adverbs - Pang-abay ('adv')
    "adverb": "adv", "pang-abay": "adv", "adv": "adv", "manner": "adv", "abay": "adv",
    "Pang-abay": "adv", "pang-abay na pamaraan": "adv", "pang-abay na panlunan": "adv",
    "pang-abay na pamanahon": "adv", "pang-abay na pang-uring": "adv", "pnb": "adv",
    # Pronouns - Panghalip ('pron')
    "pronoun": "pron", "panghalip": "pron", "pron": "pron", "halip": "pron",
    "Panghalip": "pron", "panghalip panao": "pron", "panghalip pamatlig": "pron",
    "panghalip pananong": "pron", "panghalip paari": "pron", "pnh": "pron",
    # Prepositions - Pang-ukol ('prep')
    "preposition": "prep", "pang-ukol": "prep", "prep": "prep", "ukol": "prep",
    "Pang-ukol": "prep", "pnu": "prep",
    # Conjunctions - Pangatnig ('conj')
    "conjunction": "conj", "pangatnig": "conj", "conj": "conj", "katnig": "conj",
    "Pangatnig": "conj", "pnt": "conj",
    # Interjections - Pandamdam ('intj')
    "interjection": "intj", "pandamdam": "intj", "padamdam": "intj", "intj": "intj",
    "damdam": "intj", "Pandamdam": "intj", "Padamdam": "intj", "pdd": "intj",
    # Determiners / Articles - Pantukoy ('det')
    "determiner": "det", "pantukoy": "det", "article": "det", "art": "det",
    "tukoy": "det", "Pantukoy": "det", "panuri": "det", "ptk": "det",
    # Affixes - Panlapi ('affix')
    "affix": "affix", "panlapi": "affix", "aff": "affix", "lapi": "affix",
    "prefix": "affix", "unlapi": "affix", "pref": "affix", "suffix": "affix",
    "hulapi": "affix", "suff": "affix", "infix": "affix", "gitlapi": "affix",
    "inf": "affix", "Panlapi": "affix", "Pangkayarian": "affix", "pnl": "affix",
    # Ligatures - Pang-angkop ('lig')
    "ligature": "lig", "linker": "lig", "pang-angkop": "lig", "Pang-angkop": "lig", "pnk": "lig",
    # Particles - Kataga ('part')
    "particle": "part", "kataga": "part", "part": "part",
    # Numbers - Pamilang ('num')
    "number": "num", "bilang": "num", "num": "num", "pamilang": "num", "karamihan": "num",
    # Expressions / Phrases ('expr')
    "expression": "expr", "pahayag": "expr", "expr": "expr", "phrase": "expr", "parirala": "expr", "phr": "expr",
    # Punctuation ('punc')
    "punctuation": "punc", "bantas": "punc", "punto": "punc",
    # Other Categories / Mappings
    "idm": "idm", "kol": "col", "syn": "unc", "ant": "unc", "eng": "unc", "spa": "unc",
    "tx": "unc", "var": "var", "st": "unc", "auxiliary": "unc", "pantulong": "unc", "aux": "unc",
    "unknown": "unc", "unc": "unc",
    # Daglat
    "id": "idm", "kol": "col",
}
LOWERCASE_POS_MAPPING = {k.lower(): v for k, v in POS_MAPPING.items()}


# Replacing placeholder with the original from dictionary_manager.py
class SourceStandardization:
    @staticmethod
    def standardize_sources(source: str) -> str:
        """Convert source filenames to standardized display names."""
        if not source:
            return "unknown"

        source_mapping = {
            "kaikki-ceb.jsonl": "kaikki.org (Cebuano)",
            "kaikki.jsonl": "kaikki.org (Tagalog)",
            "kwf_dictionary.json": "KWF Diksiyonaryo ng Wikang Filipino",
            "root_words_with_associated_words_cleaned.json": "tagalog.com",
            "tagalog-words.json": "diksiyonaryo.ph",
        }

        # Try direct mapping first
        norm_source = str(source).lower().strip()
        if norm_source in source_mapping:
            return source_mapping[norm_source]

        # Handle cases where only part of the filename is matched
        for key, value in source_mapping.items():
            if key in norm_source:
                return value

        # Special case for Marayum dictionaries
        if "marayum" in norm_source:
            return "Project Marayum"

        # Return the original if no mapping is found
        return source

    @staticmethod
    def get_display_name(source: str) -> str:
        """Get a display-friendly name for a source."""
        return SourceStandardization.standardize_sources(source)



# Moved from dictionary_manager.py
def standardize_source_identifier(source_identifier: str) -> str:
    """
    Standardize a source identifier string using the SourceStandardization class.

    Args:
        source_identifier: Raw source identifier (typically a filename)

    Returns:
        Standardized source name
    """
    # Use the SourceStandardization class defined above
    return SourceStandardization.standardize_sources(source_identifier)



# Helper function to determine the type of a non-word string
def get_non_word_note_type(non_word_string: str) -> str:
    """
    Determines the category ('note_type') of a non-word string.

    Args:
        non_word_string: The string to categorize.

    Returns:
        A string representing the category (e.g., "Language Code", "Usage", "Place Name").
    """
    if not non_word_string or not isinstance(non_word_string, str):
        return "Unknown Non-Word"

    text_lower = non_word_string.lower().strip()

    # Define known categories
    usage_tags = {
        "obsolete",
        "archaic",
        "dialectal",
        "uncommon",
        "inclusive",
        "regional",
        "formal",
        "informal",
        "slang",
        "colloquial",
        "figurative",
        "literal",
        "standard",
        "nonstandard",
        "proscribed",
        "dated",
        "rare",
        "poetic",
        "historical",
        "abbr.",
        "fig.",
        "lit.",  # Added common abbreviations seen
    }
    # Add more known place names encountered if needed
    place_names = {"batangas", "marinduque", "rizal"}
    orthography_terms = {"spanish-based orthography"}

    if text_lower in usage_tags:
        return "Usage"
    elif text_lower in place_names:
        return "Place Name"
    elif text_lower in orthography_terms:
        return "Orthography"
    elif (len(text_lower) == 2 or len(text_lower) == 3) and text_lower.isalpha():
        # Basic check for language codes (2-3 letters)
        # Could also check against a known list of language codes for more accuracy
        return "Language Code"
    # Add more specific checks here if needed (e.g., checking against a list of known descriptors)

    # Default category if none of the above match
    return "Descriptor"


# Moved from dictionary_manager.py
def normalize_lemma(text: str) -> str:
    if not text:
        logger.warning("normalize_lemma received empty or None text")
        return ""
    # Use unidecode for broader transliteration first, then handle specific cases
    base_normalized = unidecode.unidecode(text).lower()
    # Specific rule: Replace 'ñ' with 'n' AFTER unidecode might have handled others
    final_normalized = base_normalized.replace('ñ', 'n')
    return final_normalized

# Moved from dictionary_manager.py
def remove_trailing_numbers(text: str) -> str:
    """
    Remove trailing numbers from a string, unless it's a single letter followed by numbers (e.g., 'd2').
    """
    if not text:
        return ""
    if re.match(r"^[a-zA-Z]\d+$", text):
        return text
    return re.sub(r"\d+$", "", text)

# Moved from dictionary_manager.py
def preserve_acronyms_in_parentheses(text: str) -> str:
    """
    Preserve acronyms in parentheses by replacing them with special markers.
    """
    if not text:
        return text

    def is_acronym(content):
        return (
            re.match(r"^[A-Z0-9\s\-\.]+$", content)
            and len(content) <= 10
            and any(c.isalpha() for c in content)
        )

    pattern = r"\(([^)]+)\)"
    matches = list(re.finditer(pattern, text))
    result = text
    for match in reversed(matches):
        content = match.group(1)
        if is_acronym(content):
            marker = f"ACRONYM_PRESERVE_{content}_PRESERVE_ACRONYM"
            result = result[: match.start()] + marker + result[match.end() :]
    return result

# Moved from dictionary_manager.py
def restore_preserved_acronyms(text: str) -> str:
    """
    Restore preserved acronyms from markers back to their original form with parentheses.
    """
    if not text:
        return text
    pattern = r"ACRONYM_PRESERVE_([^_]+)_PRESERVE_ACRONYM"
    return re.sub(pattern, r"(\1)", text)

# Moved from dictionary_manager.py
def get_standard_code(pos_string: Optional[str]) -> str:
    """
    Convert a part of speech string to a standardized short code (e.g., 'n', 'v', 'adj').
    """
    if not pos_string:
        return "unc"

    pos_key = str(pos_string).lower().strip()
    pos_key = re.sub(r"\([^)]*\)", "", pos_key).strip()
    if not pos_key:
        return "unc"

    if pos_key in LOWERCASE_POS_MAPPING:
        return LOWERCASE_POS_MAPPING[pos_key]

    # Check original mapping keys if needed, though lowercase lookup is primary
    # if pos_string in POS_MAPPING:
    #     return POS_MAPPING[pos_string]

    if pos_key.startswith("sa ") or pos_key.startswith("kung sa "):
        return "n"

    logger.debug(f"Unmapped POS: '{pos_string}' -> mapped to 'unc'")
    return "unc"

# Moved from dictionary_manager.py
def standardize_entry_pos(pos_input: Union[str, list, None]) -> str:
    """
    Standardize part-of-speech from dictionary entries (string or list).
    """
    if not pos_input:
        return "unc"

    pos_to_standardize = None
    if isinstance(pos_input, list):
        for pos_item in pos_input:
            if pos_item and isinstance(pos_item, str) and pos_item.strip():
                temp_code = get_standard_code(pos_item) # Standardize item
                if temp_code != "unc":
                    return temp_code # Return first valid code found
        # If no valid item found in list, or if list was empty
        return "unc"
    elif isinstance(pos_input, str):
        pos_to_standardize = pos_input
    else:
        return "unc"

    return get_standard_code(pos_to_standardize)


def clean_html(raw_html: Optional[str]) -> str:
    """Remove HTML tags from a string, preferably using BeautifulSoup."""
    if not raw_html:
        return ""
    if not isinstance(raw_html, str):
        try:
            raw_html = str(raw_html)
        except Exception:
             logger.warning(f"Could not convert raw_html of type {type(raw_html)} to string. Returning empty.")
             return ""

    # Use BeautifulSoup if available for better parsing
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            # Get text content, join multi-line text, strip extra whitespace
            text = ' '.join(soup.stripped_strings)
            return text
        except Exception as e:
            logger.warning(f"BeautifulSoup failed to parse HTML: {e}. Falling back to regex.")
            # Fallback to regex if BS fails
            clean_text = re.sub("<.*?>", "", raw_html)
    else:
        # Basic regex fallback if BeautifulSoup is not installed
        clean_text = re.sub("<.*?>", "", raw_html)

    # Decode HTML entities like &amp; after tag removal
    try:
        import html
        clean_text = html.unescape(clean_text)
    except ImportError:
        # logger.warning("html module not found, cannot unescape HTML entities.") # Already logged if BS missing
        pass
    except Exception as e:
        logger.warning(f"Error unescaping HTML entities: {e}")

    return clean_text.strip()

# Moved from dictionary_manager.py
@dataclasses.dataclass
class BaybayinChar:
    """Define a Baybayin character with its properties."""

    char: str
    char_type: BaybayinCharType # Depends on BaybayinCharType from enums
    default_sound: str
    possible_sounds: List[str]

    def __post_init__(self):
        if not self.char:
            raise ValueError("Character cannot be empty")
        if not isinstance(self.char_type, BaybayinCharType):
            raise ValueError(f"Invalid character type: {self.char_type}")
        if not self.default_sound and self.char_type not in (
            BaybayinCharType.VIRAMA,
            BaybayinCharType.PUNCTUATION,
        ):
            raise ValueError("Default sound required for non-virama characters")
        code_point = ord(self.char)
        if not (0x1700 <= code_point <= 0x171F) and not (
            0x1735 <= code_point <= 0x1736
        ):
            raise ValueError(
                f"Invalid Baybayin character: {self.char} (U+{code_point:04X})"
            )
        expected_type = BaybayinCharType.get_type(self.char)
        if (
            expected_type != self.char_type
            and expected_type != BaybayinCharType.UNKNOWN
        ):
            raise ValueError(
                f"Character type mismatch for {self.char}: expected {expected_type}, got {self.char_type}"
            )

    def get_sound(self, next_char: Optional["BaybayinChar"] = None) -> str:
        """Get the sound of this character, considering the next character."""
        if self.char_type == BaybayinCharType.CONSONANT and next_char:
            if next_char.char_type == BaybayinCharType.VOWEL_MARK:
                return self.default_sound[:-1] + next_char.default_sound
            elif next_char.char_type == BaybayinCharType.VIRAMA:
                return self.default_sound[:-1]
        return self.default_sound

    def __str__(self) -> str:
        return f"{self.char} ({self.char_type.value}, sounds: {self.default_sound})"

# Moved from dictionary_manager.py
def extract_etymology_components(etymology_text):
    """Extract structured components from etymology text"""
    if not etymology_text:
        return []

    # Here we could implement more structured etymology parsing
    # For now, just returning a simple structure
    return {"original_text": etymology_text, "processed": True}

# Moved from dictionary_manager.py
def extract_meaning(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        meaning = match.group(1)
        clean_text = text.replace(match.group(0), "").strip()
        return clean_text, meaning
    return text, None

# Moved from dictionary_manager.py
def has_diacritics(text: str) -> bool:
    normalized = normalize_lemma(text) # Uses normalize_lemma from this file
    return text != normalized

# Moved from dictionary_manager.py
# Note: Depends on BaybayinChar and BaybayinCharType from enums
#       and normalize_lemma from this file
class BaybayinRomanizer:
    """Handles romanization of Baybayin text."""

    # Assuming BaybayinChar is imported or defined elsewhere (e.g., in models)
    # And BaybayinCharType is imported from .enums
    VOWELS = {
        "ᜀ": BaybayinChar("ᜀ", BaybayinCharType.VOWEL, "a", ["a"]),
        "ᜁ": BaybayinChar("ᜁ", BaybayinCharType.VOWEL, "i", ["i", "e"]),
        "ᜂ": BaybayinChar("ᜂ", BaybayinCharType.VOWEL, "u", ["u", "o"]),
    }
    CONSONANTS = {
        "ᜃ": BaybayinChar("ᜃ", BaybayinCharType.CONSONANT, "ka", ["ka"]),
        "ᜄ": BaybayinChar("ᜄ", BaybayinCharType.CONSONANT, "ga", ["ga"]),
        "ᜅ": BaybayinChar("ᜅ", BaybayinCharType.CONSONANT, "nga", ["nga"]),
        "ᜆ": BaybayinChar("ᜆ", BaybayinCharType.CONSONANT, "ta", ["ta"]),
        "ᜇ": BaybayinChar("ᜇ", BaybayinCharType.CONSONANT, "da", ["da"]),
        "ᜈ": BaybayinChar("ᜈ", BaybayinCharType.CONSONANT, "na", ["na"]),
        "ᜉ": BaybayinChar("ᜉ", BaybayinCharType.CONSONANT, "pa", ["pa"]),
        "ᜊ": BaybayinChar("ᜊ", BaybayinCharType.CONSONANT, "ba", ["ba"]),
        "ᜋ": BaybayinChar("ᜋ", BaybayinCharType.CONSONANT, "ma", ["ma"]),
        "ᜌ": BaybayinChar("ᜌ", BaybayinCharType.CONSONANT, "ya", ["ya"]),
        "ᜎ": BaybayinChar("ᜎ", BaybayinCharType.CONSONANT, "la", ["la"]),
        "ᜏ": BaybayinChar("ᜏ", BaybayinCharType.CONSONANT, "wa", ["wa"]),
        "ᜐ": BaybayinChar("ᜐ", BaybayinCharType.CONSONANT, "sa", ["sa"]),
        "ᜑ": BaybayinChar("ᜑ", BaybayinCharType.CONSONANT, "ha", ["ha"]),
        "ᜍ": BaybayinChar("ᜍ", BaybayinCharType.CONSONANT, "ra", ["ra"]),  # Added ra
    }
    VOWEL_MARKS = {
        "ᜒ": BaybayinChar("ᜒ", BaybayinCharType.VOWEL_MARK, "i", ["i", "e"]),
        "ᜓ": BaybayinChar("ᜓ", BaybayinCharType.VOWEL_MARK, "u", ["u", "o"]),
    }
    VIRAMA = BaybayinChar("᜔", BaybayinCharType.VIRAMA, "", [])
    PUNCTUATION = {
        "᜵": BaybayinChar("᜵", BaybayinCharType.PUNCTUATION, ",", [","]),
        "᜶": BaybayinChar("᜶", BaybayinCharType.PUNCTUATION, ".", ["."]),
    }

    def __init__(self):
        """Initialize the romanizer with a combined character mapping."""
        self.all_chars = {}
        # Combine all character mappings for easy lookup
        for char_map in [
            self.VOWELS,
            self.CONSONANTS,
            self.VOWEL_MARKS,
            {self.VIRAMA.char: self.VIRAMA},
            self.PUNCTUATION,
        ]:
            self.all_chars.update(char_map)

    def is_baybayin(self, text: str) -> bool:
        """Check if a string contains any Baybayin characters."""
        if not text:
            return False
        # Check for characters in the Baybayin Unicode block (U+1700 to U+171F)
        return any(0x1700 <= ord(c) <= 0x171F for c in text)

    def get_char_info(self, char: str) -> Optional[BaybayinChar]:
        """Get character information for a Baybayin character."""
        return self.all_chars.get(char)

    def process_syllable(self, chars: List[str]) -> Tuple[str, int]:
        """
        Process a Baybayin syllable and return its romanized form.

        Args:
            chars: List of characters in the potential syllable

        Returns:
            (romanized_syllable, number_of_characters_consumed)
        """
        if not chars:
            return "", 0

        # Get information about the first character
        first_char = self.get_char_info(chars[0])
        if not first_char:
            # Not a recognized Baybayin character
            return chars[0], 1

        if first_char.char_type == BaybayinCharType.VOWEL:
            # Simple vowel
            return first_char.default_sound, 1

        elif first_char.char_type == BaybayinCharType.CONSONANT:
            # Start with default consonant sound (with 'a' vowel)
            result = first_char.default_sound
            pos = 1

            # Check for vowel marks or virama (vowel killer)
            if pos < len(chars):
                next_char = self.get_char_info(chars[pos])
                if next_char:
                    if next_char.char_type == BaybayinCharType.VOWEL_MARK:
                        # Replace default 'a' vowel with the marked vowel
                        result = result[:-1] + next_char.default_sound
                        pos += 1
                    elif next_char.char_type == BaybayinCharType.VIRAMA:
                        # Remove the default 'a' vowel (final consonant)
                        result = result[:-1]
                        pos += 1

            return result, pos

        elif first_char.char_type == BaybayinCharType.PUNCTUATION:
            # Baybayin punctuation
            return first_char.default_sound, 1

        # For unhandled cases (shouldn't normally happen)
        return "", 1

    def romanize(self, text: str) -> str:
        """
        Convert Baybayin text to its romanized form.

        Args:
            text: The Baybayin text to romanize

        Returns:
            The romanized text, or original text if romanization failed
        """
        if not text:
            return ""

        # Normalize Unicode for consistent character handling
        text = unicodedata.normalize("NFC", text)

        result = []
        i = 0

        while i < len(text):
            # Skip spaces and non-Baybayin characters
            if text[i].isspace() or not self.is_baybayin(text[i]):
                result.append(text[i])
                i += 1
                continue

            # Process a syllable
            try:
                processed_syllable, chars_consumed = self.process_syllable(
                    list(text[i:])
                )

                if processed_syllable:
                    result.append(processed_syllable)
                    i += chars_consumed
                else:
                    # Handle unrecognized characters
                    result.append(text[i])
                    i += 1
            except Exception as e:
                logger.error(f"Error during Baybayin romanization at position {i}: {e}")
                # Skip problematic character
                i += 1

        return "".join(result)

    def validate_text(self, text: str) -> bool:
        """
        Validate that a string contains valid Baybayin text.

        Args:
            text: The text to validate

        Returns:
            True if the text is valid Baybayin, False otherwise
        """
        if not text:
            return False

        # Normalize Unicode
        text = unicodedata.normalize("NFC", text)
        chars = list(text)
        i = 0

        while i < len(chars):
            # Skip spaces
            if chars[i].isspace():
                i += 1
                continue

            # Get character info
            char_info = self.get_char_info(chars[i])

            # Not a valid Baybayin character
            if not char_info:
                if 0x1700 <= ord(chars[i]) <= 0x171F:
                    # It's in the Baybayin Unicode range but not recognized
                    logger.warning(
                        f"Unrecognized Baybayin character at position {i}: {chars[i]} (U+{ord(chars[i]):04X})"
                    )
                return False

            # Vowel mark must follow a consonant
            if char_info.char_type == BaybayinCharType.VOWEL_MARK:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(
                        f"Vowel mark not following a consonant at position {i}"
                    )
                    return False

            # Virama (vowel killer) must follow a consonant
            if char_info.char_type == BaybayinCharType.VIRAMA:
                if (
                    i == 0
                    or not self.get_char_info(chars[i - 1])
                    or self.get_char_info(chars[i - 1]).char_type
                    != BaybayinCharType.CONSONANT
                ):
                    logger.warning(f"Virama not following a consonant at position {i}")
                    return False

            i += 1

        return True

# Moved from dictionary_manager.py
def process_baybayin_text(text: str) -> Tuple[str, Optional[str], bool]:
    if not text:
        return text, None, False
    romanizer = BaybayinRomanizer()
    has_bb = romanizer.is_baybayin(text)
    if not has_bb:
        return text, None, False
    if not romanizer.validate_text(text):
        logger.warning(f"Invalid Baybayin text detected: {text}")
        return text, None, True # Return True for has_baybayin even if invalid structure
    try:
        romanized = romanizer.romanize(text)
        return text, romanized, True
    except ValueError as e:
        logger.error(f"Error romanizing Baybayin text: {str(e)}")
        return text, None, True # Return True for has_baybayin even if romanization fails

# Moved from dictionary_manager.py
def transliterate_to_baybayin(text: str) -> str:
    """
    Transliterate Latin text to Baybayin script.
    Handles all Filipino vowels (a, e, i, o, u) and consonants,
    including final consonants with virama.

    Args:
        text: Latin text to convert to Baybayin

    Returns:
        Baybayin text
    """
    if not text:
        return ""

    # Handle prefix starting with '-' (like '-an')
    if text.startswith("-"):
        # Skip the hyphen and process the rest
        text = text[1:]

    # Normalize text: lowercase and remove diacritical marks
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )

    # Define Baybayin character mappings (reuse from BaybayinRomanizer for consistency)
    romanizer = BaybayinRomanizer() # Create instance to access mappings
    consonants = {sound[:-1]: char for char, bb_char in romanizer.CONSONANTS.items() for sound in bb_char.possible_sounds if sound.endswith('a')} # Derive from romanizer
    vowels = {sound: char for char, bb_char in romanizer.VOWELS.items() for sound in bb_char.possible_sounds} # Derive from romanizer
    vowel_marks = {sound: char for char, bb_char in romanizer.VOWEL_MARKS.items() for sound in bb_char.possible_sounds}
    virama = romanizer.VIRAMA.char
    # Add 'ng' mapping manually if not derived easily
    consonants['ng'] = 'ᜅ' # Assuming ᜅ corresponds to 'nga'

    result = []
    i = 0

    while i < len(text):
        # Check for 'ng' digraph first
        if i + 1 < len(text) and text[i : i + 2] == "ng":
            if i + 2 < len(text) and text[i + 2] in "aeiou":
                # ng + vowel
                vowel_sound = text[i + 2]
                if vowel_sound == "a":
                    result.append(consonants["ng"])
                else:
                    result.append(consonants["ng"] + vowel_marks.get(vowel_sound, ''))
                i += 3
            else:
                # Final 'ng'
                result.append(consonants["ng"] + virama)
                i += 2

        # Handle single consonants
        elif text[i] in consonants:
            consonant = text[i]
            if i + 1 < len(text) and text[i + 1] in "aeiou":
                # Consonant + vowel
                vowel_sound = text[i + 1]
                if vowel_sound == "a":
                    result.append(consonants[consonant])
                else:
                    result.append(consonants[consonant] + vowel_marks.get(vowel_sound, ''))
                i += 2
            else:
                # Final consonant
                result.append(consonants[consonant] + virama)
                i += 1

        # Handle vowels
        elif text[i] in "aeiou":
            result.append(vowels.get(text[i], ''))
            i += 1

        # Skip spaces and other characters
        elif text[i].isspace():
            result.append(" ")
            i += 1
        else:
            # Skip non-convertible characters
            logger.warning(f"Skipping non-convertible character '{text[i]}' in transliteration")
            i += 1

    # Final validation - ensure only valid characters are included
    valid_output = "".join(
        c for c in result if (0x1700 <= ord(c) <= 0x171F) or c.isspace()
    )

    # Verify the output meets database constraints
    if not re.match(r"^[\\u1700-\\u171F\\s]*$", valid_output):
        logger.warning(
            f"Transliterated Baybayin doesn't match required regex pattern: {valid_output}"
        )
        # Additional cleanup to ensure it matches the pattern
        valid_output = re.sub(r"[^\\u1700-\\u171F\\s]", "", valid_output)

    return valid_output

# Moved from dictionary_manager.py
def clean_baybayin_text(text: str) -> str:
    """
    Clean Baybayin text by removing non-Baybayin characters.

    Args:
        text: Text that may contain Baybayin and other characters

    Returns:
        Cleaned text with only valid Baybayin characters and spaces
    """
    if not text:
        return ""

    # Keep only characters in the Baybayin Unicode range (U+1700 to U+171F) and spaces
    cleaned = "".join(c for c in text if (0x1700 <= ord(c) <= 0x171F) or c.isspace())

    # Normalize whitespace and trim
    return re.sub(r"\\s+", " ", cleaned).strip()

# Moved from dictionary_manager.py
def extract_baybayin_text(text: str) -> List[str]:
    """
    Extract Baybayin text segments from a string.

    Args:
        text: Text that may contain Baybayin

    Returns:
        List of Baybayin segments
    """
    if not text:
        return []

    # Split by non-Baybayin characters
    parts = re.split(r"[^\\u1700-\\u171F\\s]+", text)
    results = []

    for part in parts:
        # Clean and normalize
        cleaned_part = clean_baybayin_text(part)

        # Make sure part contains at least one Baybayin character
        if cleaned_part and any(0x1700 <= ord(c) <= 0x171F for c in cleaned_part):
            # Verify it meets database constraints
            if re.match(r"^[\\u1700-\\u171F\\s]*$", cleaned_part):
                results.append(cleaned_part)

    return results

# Moved from dictionary_manager.py
def validate_baybayin_entry(
    baybayin_form: str, romanized_form: Optional[str] = None
) -> bool:
    """
    Validate if a Baybayin form is correct and matches the romanized form if provided.

    Args:
        baybayin_form: The Baybayin text to validate
        romanized_form: Optional romanized form to verify against

    Returns:
        bool: True if valid, False otherwise
    """
    if not baybayin_form:
        return False

    try:
        # Clean and validate the form
        cleaned_form = clean_baybayin_text(baybayin_form)
        if not cleaned_form:
            logger.warning(f"No valid Baybayin characters found in: {baybayin_form}")
            return False

        # Check against the database regex constraint
        if not re.match(r"^[\\u1700-\\u171F\\s]*$", cleaned_form):
            logger.warning(
                f"Baybayin form doesn't match required regex pattern: {cleaned_form}"
            )
            return False

        # Create a romanizer to validate structure
        romanizer = BaybayinRomanizer()
        if not romanizer.validate_text(cleaned_form):
            logger.warning(f"Invalid Baybayin structure in: {cleaned_form}")
            return False

        # If romanized form is provided, check if it matches our romanization
        if romanized_form:
            try:
                generated_rom = romanizer.romanize(cleaned_form)
                # Compare normalized versions to avoid case and diacritic issues
                if normalize_lemma(generated_rom) == normalize_lemma(romanized_form):
                    return True
                else:
                    logger.warning(
                        f"Romanization mismatch: expected '{romanized_form}', got '{generated_rom}'"
                    )
                    # Still return True if structure is valid but romanization doesn't match
                    # This allows for different romanization standards
                    return True
            except Exception as e:
                logger.error(f"Error during romanization validation: {e}")
                return False

        # If no romanized form to check against, return True if the form is valid
        return True

    except Exception as e:
        logger.error(f"Error validating Baybayin entry: {e}")
        return False

# Moved from dictionary_manager.py
def get_romanized_text(text: str) -> str:
    romanizer = BaybayinRomanizer()
    try:
        return romanizer.romanize(text)
    except ValueError:
        return text

# Added functions from dictionary_manager.py

def get_language_mapping():
    """Dynamically build language mapping from dictionary files."""
    # Base ISO 639-3 codes for Philippine languages
    base_language_map = {
        "onhan": "onx",
        "waray": "war",
        "ibanag": "ibg",
        "iranon": "iro",
        "ilocano": "ilo",
        "cebuano": "ceb",
        "hiligaynon": "hil",
        "kinaray-a": "krj",
        "kinaraya": "krj",
        "kinaray": "krj",
        "asi": "asi",
        "bikol": "bik",
        "bikolano": "bik",
        "bicol": "bik",
        "surigaonon": "sgd",
        "aklanon": "akl",
        "masbatenyo": "msb",
        "chavacano": "cbk",
        "tagalog": "tgl",
        "filipino": "tgl", # Use tgl for Filipino as primary Tagalog code
        "pilipino": "tgl",
        "pangasinan": "pag",
        "kapampangan": "pam",
        "manobo": "mbt", # Default to Erumanen Manobo
        "manide": "abd",
        "maguindanaon": "mdh",
        "ivatan": "ivv",
        "itawis": "itv",
        "isneg": "isd",
        "ifugao": "ifk",
        "gaddang": "gad",
        "cuyonon": "cyo",
        "blaan": "bpr",  # Default to Koronadal Blaan
    }

    # Add regional variants
    regional_variants = {
        "bikol-central": "bcl",
        "bikol-albay": "bik",
        "bikol-rinconada": "bto",
        "bikol-partido": "bik",
        "bikol-miraya": "bik",
        "bikol-libon": "bik",
        "bikol-west-albay": "fbl",
        "bikol-southern-catanduanes": "bln",
        "bikol-northern-catanduanes": "bln",
        "bikol-boinen": "bik",
        "blaan-koronadal": "bpr",
        "blaan-sarangani": "bps",
        "cebuano-cotabato": "ceb",
        "hiligaynon-cotabato": "hil",
        "tagalog-marinduque": "tgl",
        "manobo-erumanen": "mbt",
        "isneg-yapayao": "isd",
        "ifugao-tuwali-ihapo": "ifk",
    }

    # Combine base map with variants
    language_map = {**base_language_map, **regional_variants}

    # Try to get additional mappings from dictionary files (optional)
    # Assumes data/marayum_dictionaries exists relative to execution context
    try:
        json_pattern = os.path.join("data", "marayum_dictionaries", "*_processed.json")
        json_files = glob.glob(json_pattern)

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "dictionary_info" not in data:
                        continue

                    dict_info = data["dictionary_info"]
                    base_language = dict_info.get("base_language", "").lower()
                    if not base_language:
                        continue

                    # Extract language code from filename
                    filename = os.path.basename(json_file)
                    if filename.endswith("_processed.json"):
                        filename = filename[:-14]  # Remove '_processed.json'

                    # Split on first hyphen to get language part
                    lang_code = filename.split("-")[0].lower()

                    # Add to mapping if not already present
                    if base_language not in language_map:
                        language_map[base_language] = lang_code

                    # Handle compound names
                    if "-" in base_language:
                        parts = base_language.split("-")
                        for part in parts:
                            if part not in language_map:
                                language_map[part] = lang_code

                    # Add filename-based mapping
                    orig_name = filename.replace("-english", "").lower()
                    if orig_name not in language_map:
                        language_map[orig_name] = lang_code

            except Exception as e:
                logger.warning(
                    "Error processing language mapping from %s: %s", json_file, str(e)
                )
                continue
    except Exception as e:
        logger.error(f"Error building language mapping: {str(e)}")

    return language_map


# Cache the language mapping for performance
LANGUAGE_MAPPING_CACHE = get_language_mapping()


def get_language_code(language: str) -> str:
    """Get standardized ISO 639-3 language code."""
    if not language:
        return ""

    language = language.lower().strip()

    # Use the cached mapping
    if language in LANGUAGE_MAPPING_CACHE:
        return LANGUAGE_MAPPING_CACHE[language]

    # General ISO 639-1 mappings (add more as needed)
    general_codes = {
        "english": "en",
        "spanish": "es",
        "chinese": "zh",
        "japanese": "ja",
        "sanskrit": "sa",
        "malay": "ms",
        "arabic": "ar",
        "latin": "la",
        "french": "fr",
        "german": "de",
        "portuguese": "pt",
    }
    if language in general_codes:
        return general_codes[language]

    # Fallback for unknown languages
    logger.warning(f"Could not map language: '{language}'")
    return "und" # Undetermined language code 


def process_kaikki_lemma(lemma):
    """
    Special processing for Kaikki lemmas that preserves parenthesized variants
    and handles numbers appropriately.

    Args:
        lemma: The original lemma text

    Returns:
        Processed lemma with parenthesized variants intact and appropriate number handling
    """
    if not lemma:
        return lemma

    # QUICK PRE-CHECK - If it's just a number, return immediately
    if lemma.isdigit():
        return lemma

    # SPECIAL HANDLING FOR NUMBER PATTERNS
    # Check for these patterns before any processing:

    # 1. Phrases that end with numbers (space + number)
    if re.search(r"\\s\\d+$", lemma):
        return lemma  # Keep "article on 100" intact

    # 2. Numbers with commas (like "1,000")
    if re.search(r"\\d,\\d", lemma):
        return lemma

    # 3. Hyphenated expressions (like "ika-100")
    if re.search(r"[-–]\\d", lemma):
        return lemma

    # 4. No-space parentheses like "word(variant)"
    if re.search(r"\\w+\\([^)]+\\)", lemma):
        return lemma

    # DON\'T use regex for trailing number removal
    # Instead, manually check if it ends with digits and has letters before them

    has_trailing_digits = False
    last_letter_pos = -1

    # Find position of last letter
    for i in range(len(lemma) - 1, -1, -1):
        if lemma[i].isalpha():
            last_letter_pos = i
            break

    # Check if there are only digits after the last letter
    if last_letter_pos >= 0 and last_letter_pos < len(lemma) - 1:
        remaining = lemma[last_letter_pos + 1 :]
        if remaining.isdigit():
            has_trailing_digits = True

    # Only apply trailing digit removal if we found a clear pattern
    if has_trailing_digits:
        result = lemma[: last_letter_pos + 1]
        logger.info(f"Removed trailing numbers: '{lemma}' → '{result}'")
        return result

    # Otherwise, return the original
    return lemma

def extract_parenthesized_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract text in parentheses from a string.
    If parentheses are embedded (e.g., 'hig(ə)pít', '(si-)ida'), return the original text as note.
    If parentheses are purely annotations (e.g., 'bahay (fig.)'), return the extracted content.

    Args:
        text: The input text to process

    Returns:
        A tuple containing:
        - The cleaned text (parentheses removed and stripped).
        - The original text if parentheses are embedded.
        - The joined content of parentheses if they are only trailing/wrapping notes.
        - None if no parentheses are found.

    Examples:
        "hig(ə)pít" -> ("higpít", "hig(ə)pít")
        "(si-)ida" -> ("ida", "(si-)ida")
        "(q)abu(s)" -> ("abu", "(q)abu(s)")
        "bahay (fig.)" -> ("bahay", "fig.")
        "word (note1) (note2)" -> ("word", "note1, note2")
        "(just a note)" -> ("", "just a note") # Text becomes empty
        "normal" -> ("normal", None)
        "  (note)  " -> ("", "note") # Text becomes empty
    """
    if not text:
        return "", None

    original_text = text
    parenthesized_parts = []
    # Store start/end indices of matches to remove them precisely later
    matches_indices = []

    pattern = r"\\(([^)]+)\\)"
    for match in re.finditer(pattern, text):
        parenthesized_parts.append(match.group(1))
        matches_indices.append((match.start(), match.end()))

    if not parenthesized_parts:
        return text, None  # No parentheses found

    # Build the text *without* the parenthesized parts by replacing them
    # Iterate backwards to avoid index shifting issues
    cleaned_text_intermediate = text
    for start, end in sorted(matches_indices, reverse=True):
        cleaned_text_intermediate = (
            cleaned_text_intermediate[:start] + cleaned_text_intermediate[end:]
        )

    # Check if any non-whitespace characters remain in the cleaned text
    cleaned_text_final = cleaned_text_intermediate.strip()

    if cleaned_text_final:
        # Parentheses were embedded within other text
        return cleaned_text_final, original_text
    else:
        # Only parenthesized content existed, or text was just spaces outside them
        # Return empty string for text, and the joined notes
        return "", ", ".join(parenthesized_parts)


def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    # Simple map, can be expanded
    lang_map = {
        "Esp": "es",
        "Eng": "en",
        "Ch": "zh",
        "Tsino": "zh",
        "Jap": "ja",
        "San": "sa",
        "Sanskrit": "sa",
        "Tag": "tl",
        "Mal": "ms",
        "Arb": "ar",
    }
    # This implementation might be basic, consider regex or more robust parsing
    return [lang_map[lang] for lang in lang_map if lang in etymology]


# --- Helper function to extract script info (from backup) ---
def extract_script_info(
    entry: Dict, script_tag: str, script_name_in_template: str
) -> Tuple[Optional[str], Optional[str]]:
    """Extracts specific script form and explicit romanization if available."""
    script_form = None
    romanized = None

    # Try 'forms' array first
    if "forms" in entry and isinstance(entry["forms"], list):
        for form_data in entry["forms"]:
            if (
                isinstance(form_data, dict)
                and "tags" in form_data
                and script_tag in form_data.get("tags", [])
            ):
                form_text = form_data.get("form", "").strip()
                if form_text:
                    prefixes = ["spelling ", "script ", script_tag.lower() + " "]
                    cleaned_form = form_text
                    for prefix in prefixes:
                        if cleaned_form.lower().startswith(prefix):
                            cleaned_form = cleaned_form[len(prefix) :].strip()
                    # Basic validation - check for non-Latin chars
                    if cleaned_form and any(
                        not ("a" <= char.lower() <= "z") for char in cleaned_form
                    ):
                        script_form = cleaned_form
                        romanized = form_data.get(
                            "romanized"
                        )  # Get explicit romanization
                        return script_form, romanized

    # Fallback: Try 'head_templates' expansion
    if "head_templates" in entry and isinstance(entry["head_templates"], list):
        for template in entry["head_templates"]:
            if isinstance(template, dict) and "expansion" in template:
                expansion = template.get("expansion", "")
                if isinstance(expansion, str):
                    # Regex to find script spelling after specific text
                    # Adjust range for other scripts if needed (e.g., \u1760-\u177F for Badlit)
                    # For now, this targets Baybayin specifically U+1700-U+171F, U+1730-U+173F
                    # A more generic approach might be needed for multiple scripts.
                    pattern = rf"{script_name_in_template} spelling\s+([\u1700-\u171F\u1730-\u173F\s]+)"
                    match = re.search(pattern, expansion, re.IGNORECASE)
                    if match:
                        potential_script = match.group(1).strip()
                        # Basic validation
                        if potential_script and any(
                            not ("a" <= char.lower() <= "z")
                            for char in potential_script
                        ):
                            script_form = potential_script
                            romanized = None  # Romanization unlikely here
                            return script_form, romanized
    return None, None



