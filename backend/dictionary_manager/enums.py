# Extracted from dictionary_manager.py
import enum
from typing import Optional, List
import logging
import dataclasses

# Configure logger for this module if needed, or rely on root logger
# logger = logging.getLogger(__name__)
# If using logger.warning in from_string, ensure logger is defined
logger = logging.getLogger() # Use root logger for simplicity here

class RelationshipCategory(enum.Enum):
    """Categories for organizing relationship types"""

    SEMANTIC = "semantic"  # Meaning-based relationships
    DERIVATIONAL = "derivational"  # Word formation relationships
    VARIANT = "variant"  # Form variations
    TAXONOMIC = "taxonomic"  # Hierarchical relationships
    USAGE = "usage"  # Usage-based relationships
    OTHER = "other"  # Miscellaneous relationships


class RelationshipType(enum.Enum):
    """
    Centralized enum for all relationship types with their properties.

    Each relationship type has the following properties:
    - rel_value: The string value stored in the database
    - category: The category this relationship belongs to
    - bidirectional: Whether this relationship applies in both directions
    - inverse: The inverse relationship type (if bidirectional is False)
    - transitive: Whether this relationship is transitive (A->B and B->C implies A->C)
    - strength: Default strength/confidence for this relationship (0-100)
    """

    # Semantic relationships
    SYNONYM = ("synonym", RelationshipCategory.SEMANTIC, True, None, True, 90)
    ANTONYM = ("antonym", RelationshipCategory.SEMANTIC, True, None, False, 90)
    RELATED = ("related", RelationshipCategory.SEMANTIC, True, None, False, 70)
    SIMILAR = ("similar", RelationshipCategory.SEMANTIC, True, None, False, 60)

    # Translation relationships
    HAS_TRANSLATION = ("has_translation", RelationshipCategory.SEMANTIC, False, "TRANSLATION_OF", False, 98)
    TRANSLATION_OF = ("translation_of", RelationshipCategory.SEMANTIC, False, "HAS_TRANSLATION", False, 98)

    # Hierarchical/taxonomic relationships
    HYPERNYM = ("hypernym", RelationshipCategory.TAXONOMIC, False, "HYPONYM", True, 85)
    HYPONYM = ("hyponym", RelationshipCategory.TAXONOMIC, False, "HYPERNYM", True, 85)
    MERONYM = ("meronym", RelationshipCategory.TAXONOMIC, False, "HOLONYM", False, 80)
    HOLONYM = ("holonym", RelationshipCategory.TAXONOMIC, False, "MERONYM", False, 80)

    # Derivational relationships
    DERIVED_FROM = (
        "derived_from",
        RelationshipCategory.DERIVATIONAL,
        False,
        "ROOT_OF",
        False,
        95,
    )
    ROOT_OF = (
        "root_of",
        RelationshipCategory.DERIVATIONAL,
        False,
        "DERIVED_FROM",
        False,
        95,
    )

    # Variant relationships
    VARIANT = ("variant", RelationshipCategory.VARIANT, True, None, False, 85)
    SPELLING_VARIANT = (
        "spelling_variant",
        RelationshipCategory.VARIANT,
        True,
        None,
        False,
        95,
    )
    REGIONAL_VARIANT = (
        "regional_variant",
        RelationshipCategory.VARIANT,
        True,
        None,
        False,
        90,
    )

    # Usage relationships
    COMPARE_WITH = ("compare_with", RelationshipCategory.USAGE, True, None, False, 50)
    SEE_ALSO = ("see_also", RelationshipCategory.USAGE, True, None, False, 40)

    # Other relationships
    EQUALS = ("equals", RelationshipCategory.OTHER, True, None, True, 100)
    # Added for Kaikki specific etymology templates
    COGNATE_OF = (
        "cognate_of",
        RelationshipCategory.SEMANTIC,
        True,
        None,
        False,
        75,
    )  # Cognates are related semantically
    DOUBLET_OF = (
        "doublet_of",
        RelationshipCategory.DERIVATIONAL,
        True,
        None,
        False,
        80,
    )  # Doublets share derivation

    # --- Using __new__ for proper Enum initialization with tuple values --- 
    def __new__(cls, value: str, category: RelationshipCategory, bidirectional: bool, inverse_name: Optional[str], transitive: bool, strength: int):
        obj = object.__new__(cls)
        obj._value_ = value # The first item in the tuple is the value
        obj.category = category
        obj.bidirectional = bidirectional
        obj.inverse_name = inverse_name # Store the name for later lookup
        obj.transitive = transitive
        obj.strength = strength
        return obj

    @classmethod
    def from_string(cls, relation_str: str) -> "RelationshipType":
        """Convert a string to a RelationshipType enum value"""
        # --- Added check for non-string input --- 
        if not isinstance(relation_str, str):
            # Log a warning or error if unexpected type received
            # logger.warning(f"Expected string for RelationshipType.from_string, got {type(relation_str)}. Attempting conversion.")
            try:
                 relation_str = str(relation_str) # Attempt conversion
            except Exception:
                 logger.error(f"Could not convert input {relation_str} to string for RelationshipType lookup. Returning RELATED.")
                 return cls.RELATED # Fallback if conversion fails

        normalized = relation_str.lower().replace(" ", "_").strip()
        for rel_type in cls:
            # Compare against the value (the first element in the tuple)
            if rel_type.value == normalized:
                return rel_type

        # Handle legacy/alternative names
        legacy_mapping = {
            # Semantic relationships
            "synonym_of": cls.SYNONYM,
            "antonym_of": cls.ANTONYM,
            "related_to": cls.RELATED,
            "kasingkahulugan": cls.SYNONYM,
            "katulad": cls.SYNONYM,
            "kasalungat": cls.ANTONYM,
            "kabaligtaran": cls.ANTONYM,
            "kaugnay": cls.RELATED,
            # Derivational
            "derived": cls.DERIVED_FROM,
            "mula_sa": cls.DERIVED_FROM,
            # Variants
            "alternative_spelling": cls.SPELLING_VARIANT,
            "alternate_form": cls.VARIANT,
            "varyant": cls.VARIANT,
            "variants": cls.VARIANT,
            # Taxonomy
            "uri_ng": cls.HYPONYM,
            # Usage
            "see": cls.SEE_ALSO,
        }

        if normalized in legacy_mapping:
            return legacy_mapping[normalized]

        # Fall back to RELATED for unknown types
        logger.warning(
            f"Unknown relationship type: '{relation_str}' (normalized: '{normalized}'), using RELATED as fallback"
        )
        return cls.RELATED

    # --- Using property for inverse lookup based on inverse_name ---
    @property
    def inverse(self) -> Optional["RelationshipType"] :
        """Return the inverse RelationshipType enum member, or None if not applicable/found."""
        if self.bidirectional:
            return self # Bidirectional means it's its own inverse
        if self.inverse_name:
            try:
                # Look up the enum member by its key (name)
                return RelationshipType[self.inverse_name]
            except KeyError:
                logger.error(f"Inverse name '{self.inverse_name}' defined for '{self.name}' does not match any RelationshipType member key.")
                return None # Or potentially RELATED as a fallback?
        return None # Non-bidirectional without a defined inverse name

    # --- Original __str__ method --- 
    def __str__(self):
        # Return the primary string value (e.g., 'synonym')
        return self.value 

# Moved from dictionary_manager.py
class BaybayinCharType(enum.Enum):
    """Define types of Baybayin characters."""

    CONSONANT = "consonant"
    VOWEL = "vowel"
    VOWEL_MARK = "vowel_mark"
    VIRAMA = "virama"
    PUNCTUATION = "punctuation"
    UNKNOWN = "unknown"

    @classmethod
    def get_type(cls, char: str) -> "BaybayinCharType":
        """Determine the type of a Baybayin character."""
        if not char:
            return cls.UNKNOWN
        code_point = ord(char)
        if 0x1700 <= code_point <= 0x1702:
            return cls.VOWEL
        elif 0x1703 <= code_point <= 0x1711:
            return cls.CONSONANT
        elif code_point in (0x1712, 0x1713):
            return cls.VOWEL_MARK
        elif code_point == 0x1714:
            return cls.VIRAMA
        elif 0x1735 <= code_point <= 0x1736:
            return cls.PUNCTUATION
        return cls.UNKNOWN

# ... Add newline if needed ... 

# Moved from dictionary_manager.py (Original location around line 261)
@dataclasses.dataclass
class WordEntry:
    lemma: str
    normalized_lemma: str
    language_code: str
    root_word_id: Optional[int] = None
    preferred_spelling: Optional[str] = None
    tags: Optional[str] = None # Assuming tags were stored as string originally
    has_baybayin: bool = False
    baybayin_form: Optional[str] = None
    romanized_form: Optional[str] = None
    # Note: Add other fields if they were part of the original WordEntry

# ... Add newline if needed ... 