import json
from pathlib import Path
from typing import Dict, Any
from default_mappings import (
    LANGUAGE_FAMILIES, REGIONAL_MAPPING,
    WRITING_SYSTEMS, LANGUAGE_STANDARDIZATION,
    VALID_CODES
)

class LanguageSystemConfig:
    """Manages configuration loading for language systems."""
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load all language configuration files."""
        config_dir = Path(__file__).parent / 'config'
        
        configs = {
            'families': 'language_families.json',
            'regions': 'regions.json',
            'writing_systems': 'writing_systems.json',
            'standardization': 'standardization.json',
            'valid_codes': 'valid_codes.json'
        }
        
        defaults = {
            'families': LANGUAGE_FAMILIES,
            'regions': REGIONAL_MAPPING,
            'writing_systems': WRITING_SYSTEMS,
            'standardization': LANGUAGE_STANDARDIZATION,
            'valid_codes': list(VALID_CODES)
        }
        
        result = {}
        for key, filename in configs.items():
            path = config_dir / filename
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    result[key] = json.load(f)
            else:
                result[key] = defaults[key]
        
        return result 