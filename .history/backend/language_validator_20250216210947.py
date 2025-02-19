from typing import Dict, List, Set
from backend.language_types import LanguageSystemError, InvalidLanguageCode, InvalidLanguageMapping

class LanguageSystemValidator:
    """Validates language system configuration and data."""
    
    def __init__(self, valid_codes: Set[str]):
        self.valid_codes = valid_codes

    def validate_language_code(self, code: str) -> None:
        """Validate a language code."""
        if not code or code not in self.valid_codes:
            raise InvalidLanguageCode(f"Invalid language code: {code}")

    def validate_family_tree(self, families: Dict) -> None:
        """Validate language family tree structure."""
        def validate_node(node):
            if isinstance(node, dict):
                for value in node.values():
                    validate_node(value)
            elif isinstance(node, list):
                for lang in node:
                    if lang not in self.valid_codes:
                        raise InvalidLanguageCode(f"Invalid language in family tree: {lang}")

        validate_node(families)

    def validate_writing_systems(self, systems: Dict) -> None:
        """Validate writing system configuration."""
        for category in systems.values():
            for script_info in category.values():
                for lang in script_info['Languages']:
                    if lang not in self.valid_codes:
                        raise InvalidLanguageCode(
                            f"Invalid language in writing systems: {lang}"
                        )

    def validate_regions(self, regions: Dict) -> None:
        """Validate regional language mappings."""
        for region in regions.values():
            for subregion in region.values():
                for lang in subregion:
                    if lang not in self.valid_codes:
                        raise InvalidLanguageCode(
                            f"Invalid language in regions: {lang}"
                        )

    def validate_standardization(self, standardization: Dict) -> None:
        """Validate language code standardization mappings."""
        for standard in standardization.values():
            if standard not in self.valid_codes:
                raise InvalidLanguageCode(
                    f"Invalid standardized language code: {standard}"
                ) 