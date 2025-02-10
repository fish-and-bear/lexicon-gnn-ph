"""Configuration for language classification and standardization."""

from typing import Dict, Any
from backend.default_mappings import (
    LANGUAGE_FAMILIES,
    WRITING_SYSTEMS,
    REGIONAL_MAPPING,
    LANGUAGE_STANDARDIZATION,
    VALID_CODES
)

class LanguageSystemConfig:
    """Configuration loader for language system."""
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load language system configuration."""
        return {
            'families': LANGUAGE_FAMILIES,
            'writing_systems': WRITING_SYSTEMS,
            'regions': REGIONAL_MAPPING,
            'standardization': LANGUAGE_STANDARDIZATION,
            'valid_codes': VALID_CODES
        } 