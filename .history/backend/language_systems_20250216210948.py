"""Language classification and standardization systems for Filipino dictionary."""

import functools
from typing import Dict, List, Optional, Set, Tuple
import re
from backend.language_types import (
    LanguageMetadata, WritingSystemInfo, 
    LanguageSystemError, InvalidLanguageCode, InvalidLanguageMapping
)
from backend.language_config import LanguageSystemConfig
from backend.language_validator import LanguageSystemValidator

class LanguageSystem:
    """Manages language classification, standardization, and metadata."""
    
    def __init__(self):
        # Load configuration
        config = LanguageSystemConfig.load_config()
        self.families = config['families']
        self.regions = config['regions']
        self.writing_systems = config['writing_systems']
        self.standardization = config['standardization']
        self.valid_codes = set(config['valid_codes'])
        
        # Initialize validator
        self.validator = LanguageSystemValidator(self.valid_codes)
        
        # Validate all mappings
        self._validate_mappings()

    def _validate_mappings(self) -> None:
        """Validate consistency of all language mappings."""
        try:
            self.validator.validate_family_tree(self.families)
            self.validator.validate_writing_systems(self.writing_systems)
            self.validator.validate_regions(self.regions)
            self.validator.validate_standardization(self.standardization)
        except LanguageSystemError as e:
            raise InvalidLanguageMapping(f"Invalid language mappings: {str(e)}")

    @functools.lru_cache(maxsize=128)
    def standardize_code(self, code: str) -> str:
        """Standardize a language code to its canonical form."""
        if not code:
            return "-"
        normalized = code.lower().strip()
        return self.standardization.get(normalized, code)

    @functools.lru_cache(maxsize=128)
    def get_family_tree(self, language: str) -> List[str]:
        """Get full language family tree path."""
        self.validator.validate_language_code(language)
        
        def search_tree(tree: Dict, target: str) -> Optional[List[str]]:
            paths = []
            for key, value in tree.items():
                if isinstance(value, list) and target in value:
                    paths.append([key])
                elif isinstance(value, dict):
                    for path in search_tree(value, target) or []:
                        paths.append([key] + path)
            return paths

        paths = search_tree(self.families, language)
        return paths[0] if paths else ["Unclassified"]

    @functools.lru_cache(maxsize=128)
    def get_writing_systems(self, language: str) -> List[WritingSystemInfo]:
        """Get writing system information for a language."""
        self.validator.validate_language_code(language)
        
        systems = []
        for category, script_types in self.writing_systems.items():
            for script, details in script_types.items():
                if language in details['Languages']:
                    systems.append({
                        'script': script,
                        'category': category,
                        'period': details['Period'],
                        'status': details['Status']
                    })
        return systems

    @functools.lru_cache(maxsize=128)
    def get_regions(self, language: str) -> List[str]:
        """Get regions where a language is spoken."""
        self.validator.validate_language_code(language)
        
        regions = []
        for region, subregions in self.regions.items():
            for subregion, languages in subregions.items():
                if language in languages:
                    regions.append(f"{region} ({subregion})")
        return regions

    def get_language_metadata(self, language: str) -> LanguageMetadata:
        """Get comprehensive metadata for a language."""
        self.validator.validate_language_code(language)
        
        return LanguageMetadata(
            code=language,
            name=self.standardize_code(language),
            family=self.get_family_tree(language),
            regions=self.get_regions(language),
            writing_systems=self.get_writing_systems(language)
        )

    def extract_and_remove_language_codes(self, text: str) -> Tuple[List[str], str]:
        """
        Extract language codes from text and return both codes and cleaned text.
        
        Args:
            text: Text containing language codes in brackets/parentheses
            
        Returns:
            Tuple of (list of language codes, cleaned text)
        """
        codes = []
        cleaned_text = text
        
        # Match patterns like [es], (fr), <tl>, etc.
        code_pattern = r'[\[\(\<]([\w\-]+)[\]\)\>]'
        
        for match in re.finditer(code_pattern, text):
            code = match.group(1).strip()
            if code:
                standardized = self.standardize_code(code)
                if standardized in self.valid_codes:
                    codes.append(standardized)
        
        # Remove the code markers from text
        cleaned_text = re.sub(code_pattern, '', cleaned_text).strip()
        
        return codes, cleaned_text 