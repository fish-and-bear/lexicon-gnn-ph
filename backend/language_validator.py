from typing import Dict, List, Set
from language_types import LanguageSystemError, InvalidLanguageCode, InvalidLanguageMapping

class LanguageSystemValidator:
    """Validates language system data consistency."""
    
    def __init__(self, valid_codes: Set[str]):
        self.valid_codes = valid_codes

    def validate_language_code(self, code: str) -> bool:
        """Validate a single language code."""
        if not code or code not in self.valid_codes:
            raise InvalidLanguageCode(f"Invalid language code: {code}")
        return True

    def validate_family_tree(self, tree: Dict) -> bool:
        """Recursively validate language family tree."""
        for key, value in tree.items():
            if isinstance(value, list):
                for lang in value:
                    self.validate_language_code(lang)
            elif isinstance(value, dict):
                self.validate_family_tree(value)
        return True

    def validate_writing_systems(self, systems: Dict) -> bool:
        """Validate writing system configurations."""
        for category in systems.values():
            for script in category.values():
                for lang in script['Languages']:
                    self.validate_language_code(lang)
        return True

    def validate_regions(self, regions: Dict) -> bool:
        """Validate regional mappings."""
        for region in regions.values():
            for languages in region.values():
                for lang in languages:
                    self.validate_language_code(lang)
        return True

    def validate_standardization(self, standardization: Dict) -> bool:
        """Validate standardization mappings."""
        for std_code in standardization.values():
            if std_code not in self.valid_codes:
                raise InvalidLanguageMapping(f"Invalid standardized code: {std_code}")
        return True 