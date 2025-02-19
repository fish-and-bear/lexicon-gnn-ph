"""
Default mappings and language utilities for the dictionary system.
Provides standardized language codes, families, regions, and writing systems.
"""

from typing import Dict, List, Set, Optional, Any
from functools import lru_cache
import logging
from prometheus_client import Counter, Histogram
import time
from dataclasses import dataclass
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Metrics
MAPPING_ERRORS = Counter('mapping_errors_total', 'Total mapping errors', ['operation'])
MAPPING_OPERATION_DURATION = Histogram('mapping_operation_duration_seconds', 'Mapping operation duration')
MAPPING_VALIDATIONS = Counter('mapping_validations_total', 'Total mapping validations', ['result'])
MAPPING_CACHE_HITS = Counter('mapping_cache_hits_total', 'Total mapping cache hits')

class MappingError(Exception):
    """Base exception for mapping-related errors."""
    pass

class ValidationError(MappingError):
    """Raised when language validation fails."""
    pass

class LanguageGroup(Enum):
    """Enumeration of language groups."""
    PHILIPPINE = "Philippine"
    FOREIGN = "Foreign"
    HISTORICAL = "Historical"

@dataclass
class LanguageInfo:
    """Information about a language."""
    name: str
    code: str
    group: LanguageGroup
    family: str
    subfamily: Optional[str] = None
    branch: Optional[str] = None
    region: Optional[str] = None
    subregion: Optional[str] = None
    writing_systems: List[str] = None
    alternative_names: List[str] = None
    iso_codes: List[str] = None
    glottocode: Optional[str] = None
    endangered: bool = False
    extinct: bool = False
    revitalization_status: Optional[str] = None

    def __post_init__(self):
        if self.writing_systems is None:
            self.writing_systems = []
        if self.alternative_names is None:
            self.alternative_names = []
        if self.iso_codes is None:
            self.iso_codes = []

class LanguageMappings:
    """Centralized language mappings with validation and caching."""

LANGUAGE_FAMILIES = {
    'Austronesian': {
        'Malayo-Polynesian': {
            'Philippine': {
                'Greater Central Philippine': {
                    'Central Philippine': [
                        'Tagálog', 'Bíkol', 'Sebwáno', 'Hiligaynón', 'Waráy'
                    ],
                    'Mansakan': [
                        'Mandayá', 'Kalagan'
                    ]
                },
                'Northern Luzon': {
                    'Cordilleran': [
                        'Ilokáno', 'Ifugáw', 'Kankanáëy', 'Balangáw', 'Gáddang', 'Isnë́g'
                    ],
                    'Cagayan Valley': [
                        'Ibanág', 'Itáwit', 'Ivatán'
                    ]
                },
                'Central Luzon': [
                    'Kapampángan', 'Sambali'
                ],
                'Manobo': [
                    'Binúkid', 'Agutaynë́n'
                ],
                'Danao': [
                    'Mëranáw', 'Magindanáw'
                ]
            }
        }
    },
    'Indo-European': {
        'Romance': ['Español', 'Portugués', 'French', 'Italian'],
        'Germanic': ['Ingles', 'German', 'Dutch'],
        'Indo-Iranian': ['Sanskrit', 'Hindi'],
        'Hellenic': ['Griego']
    },
    'Sino-Tibetan': {
        'Sinitic': ['Chinese', 'Chino']
    },
    'Afro-Asiatic': {
        'Semitic': ['Arabic', 'Hebrew']
    },
    'Historical': {
        'Classical': ['Latin'],
        'Ancient Philippine': ['Sinaunang Tagalog']
    }
}

REGIONAL_MAPPING = {
    'Luzon': {
        'Metro Manila': ['Tagálog'],
        'Central Luzon': ['Kapampángan', 'Sambali'],
        'Southern Luzon': ['Bíkol'],
        'Cordillera': ['Ifugáw', 'Kankanáëy', 'Balangáw', 'Gáddang', 'Isnë́g'],
        'Ilocos': ['Ilokáno'],
        'Cagayan Valley': ['Ibanág', 'Itáwit', 'Ivatán']
    },
    'Visayas': {
        'Western': ['Hiligaynón', 'Kinaráy-a', 'Aklánon'],
        'Central': ['Sebwáno', 'Waráy'],
        'Eastern': ['Waráy']
    },
    'Mindanao': {
        'Northern': ['Binúkid', 'Mandayá'],
        'Southern': ['Magindanáw', 'Mëranáw'],
        'Western': ['Súg', 'Tausug'],
        'Eastern': ['Mandayá', 'Mansaka']
    }
}

WRITING_SYSTEMS = {
    'Native Scripts': {
        'Baybayin': {
            'Languages': ['Tagálog', 'Sinaunang Tagalog', 'Bíkol', 'Ilokáno'],
            'Period': 'Pre-colonial to Early Spanish Period',
            'Status': 'Historical/Revival'
        },
        'Kulitan': {
            'Languages': ['Kapampángan'],
            'Period': 'Pre-colonial to Early Spanish Period',
            'Status': 'Historical/Limited Use'
        },
        'Tagbanwa': {
            'Languages': ['Tagbanwá'],
            'Period': 'Pre-colonial to Present',
            'Status': 'Limited Use'
        }
    },
    'Adapted Scripts': {
        'Latin': {
            'Languages': ['Tagálog', 'Sebwáno', 'Hiligaynón', 'Waráy', 'Ilokáno'],
            'Period': 'Spanish Period to Present',
            'Status': 'Primary'
        },
        'Arabic': {
            'Languages': ['Mëranáw', 'Magindanáw', 'Tausug'],
            'Period': 'Islamic Period to Present',
            'Status': 'Traditional/Limited'
        }
    }
}

LANGUAGE_STANDARDIZATION = {
    # Historical Tagalog variations
    'ST': 'Sinaunang Tagalog', 'STl': 'Sinaunang Tagalog',
    'old_tagalog': 'Sinaunang Tagalog', 'classical_tagalog': 'Sinaunang Tagalog',
    
    # Philippine languages - major
    'tl': 'Tagálog', 'tag': 'Tagálog', 'Tag': 'Tagálog',
    'bik': 'Bíkol', 'bcl': 'Bíkol', 'Bik': 'Bíkol',
    'war': 'Waráy', 'War': 'Waráy',
    'hil': 'Hiligaynón', 'Hil': 'Hiligaynón', 'Hik': 'Hiligaynón',
    'ceb': 'Sebwáno', 'Seb': 'Sebwáno',
    'klg': 'Kalagan', 'Klg': 'Kalagan',
    'tgb': 'Tagbanwá', 'Tgb': 'Tagbanwá', 'Tbw': 'Tagbanwá',
    
    # Philippine languages - Luzon
    'Ilo': 'Ilokáno', 'Ilk': 'Ilokáno',
    'Ifg': 'Ifugáw', 'Ifu': 'Ifugáw',
    'Isn': 'Isnë́g', 'Isg': 'Isnë́g',
    'Iva': 'Ivatán', 'Itw': 'Itáwit',
    'Kap': 'Kapampángan', 'pam': 'Kapampángan',
    'Gad': 'Gáddang',
    'Kan': 'Kankanáëy', 'Kbn': 'Kankanáëy',
    'Sbl': 'Sambali',
    
    # Philippine languages - Mindanao
    'Mag': 'Magindanáw', 'mdh': 'Magindanáw',
    'Mrw': 'Mëranáw', 'Mrs': 'Mëranáw',
    'Mnd': 'Mandayá', 'Man': 'Mandayá',
    'Bin': 'Binúkid',
    'Agu': 'Agutaynë́n',
    'Tau': 'Tausug', 'Tsg': 'Tausug', 'Súg': 'Tausug',
    
    # Foreign languages
    'en': 'Ingles', 'eng': 'Ingles', 'Ing': 'Ingles',
    'es': 'Español', 'spa': 'Español', 'Esp': 'Español',
    'fr': 'French', 'fra': 'French', 'Fre': 'French',
    'de': 'German', 'deu': 'German', 'Ger': 'German',
    'la': 'Latin', 'lat': 'Latin', 'Lat': 'Latin',
    'el': 'Griego', 'grc': 'Griego', 'Gre': 'Griego',
    'pt': 'Portugués', 'por': 'Portugués', 'Por': 'Portugués',
    'nl': 'Dutch', 'dut': 'Dutch', 'nld': 'Dutch',
    'sa': 'Sanskrit', 'san': 'Sanskrit', 'Skr': 'Sanskrit',
    'hi': 'Hindi', 'hin': 'Hindi', 'Hin': 'Hindi',
    'zh': 'Chinese', 'cmn': 'Chinese', 'Chi': 'Chinese',
    'ar': 'Arabic', 'ara': 'Arabic', 'Ara': 'Arabic',
    'he': 'Hebrew', 'heb': 'Hebrew', 'Heb': 'Hebrew',
    'it': 'Italian', 'ita': 'Italian', 'Ita': 'Italian'
}

VALID_CODES = set((
    # Main Philippine Languages
    'Tagálog', 'Bíkol', 'Sebwáno', 'Hiligaynón', 'Waráy',
    'Kalagan', 'Mandayá', 'Ilokáno', 'Ifugáw', 'Kankanáëy',
    'Balangáw', 'Gáddang', 'Isnë́g', 'Ibanág', 'Itáwit',
    'Ivatán', 'Kapampángan', 'Sambali', 'Binúkid', 'Agutaynë́n',
    'Mëranáw', 'Magindanáw', 'Sinaunang Tagalog', 'Tagbanwá',
    'Tausug', 'Súg', 'Kinaráy-a', 'Aklánon', 'Mansaka',
    
    # Foreign Languages
    'Ingles', 'Español', 'French', 'German', 'Latin',
    'Griego', 'Portugués', 'Dutch', 'Sanskrit', 'Hindi',
    'Chinese', 'Chino', 'Arabic', 'Hebrew', 'Italian',
    
    # Alternative codes - Philippine
    'tl', 'tag', 'Tag', 'bik', 'bcl', 'Bik',
    'ceb', 'Seb', 'hil', 'Hil', 'Hik',
    'war', 'War', 'klg', 'Klg', 'Mnd', 'Man',
    'Ilo', 'Ilk', 'Ifg', 'Ifu', 'Isn', 'Isg',
    'Iva', 'Itw', 'Kap', 'pam', 'Gad',
    'Kan', 'Kbn', 'Sbl', 'Mag', 'mdh',
    'Mrw', 'Mrs', 'Bin', 'Agu', 'Knr',
    'ST', 'STl', 'tgb', 'Tgb', 'Tbw',
    'Tau', 'Tsg', 'Súg', 'Akl', 'Msk',
    
    # Alternative codes - Foreign
    'en', 'eng', 'Ing', 'es', 'spa', 'Esp',
    'fr', 'fra', 'Fre', 'de', 'deu', 'Ger',
    'la', 'lat', 'Lat', 'el', 'grc', 'Gre',
    'pt', 'por', 'Por', 'nl', 'dut', 'nld',
    'sa', 'san', 'Skr', 'hi', 'hin', 'Hin',
    'zh', 'cmn', 'Chi', 'ar', 'ara', 'Ara',
    'he', 'heb', 'Heb', 'it', 'ita', 'Ita'
))

    @classmethod
    @lru_cache(maxsize=128)
    def get_language_info(cls, code: str) -> Optional[LanguageInfo]:
        """Get comprehensive information about a language."""
        start_time = time.time()
        try:
            if not code:
                raise ValidationError("Empty language code provided")

            # Standardize the code first
            std_name = cls.get_standardized_name(code)
            if not std_name:
                MAPPING_VALIDATIONS.labels(result='invalid_code').inc()
                return None

            # Find language family and group
            family = None
            subfamily = None
            branch = None
            for f_name, f_data in cls.LANGUAGE_FAMILIES.items():
                for sf_name, sf_data in f_data.items():
                    if isinstance(sf_data, list):
                        if std_name in sf_data:
                            family = f_name
                            subfamily = sf_name
                            break
                    else:
                        for b_name, b_data in sf_data.items():
                            if isinstance(b_data, list):
                                if std_name in b_data:
                                    family = f_name
                                    subfamily = sf_name
                                    branch = b_name
                                    break
                            else:
                                for sub_b_name, languages in b_data.items():
                                    if std_name in languages:
                                        family = f_name
                                        subfamily = sf_name
                                        branch = b_name
                                        break

            # Find region and subregion
            region = None
            subregion = None
            for r_name, r_data in cls.REGIONAL_MAPPING.items():
                for sr_name, languages in r_data.items():
                    if std_name in languages:
                        region = r_name
                        subregion = sr_name
                        break

            # Find writing systems
            writing_systems = []
            for script_type, scripts in cls.WRITING_SYSTEMS.items():
                for script_name, script_data in scripts.items():
                    if std_name in script_data['Languages']:
                        writing_systems.append({
                            'name': script_name,
                            'type': script_type,
                            'period': script_data['Period'],
                            'status': script_data['Status']
                        })

            # Determine language group
            if family == 'Historical':
                group = LanguageGroup.HISTORICAL
            elif family == 'Austronesian':
                group = LanguageGroup.PHILIPPINE
            else:
                group = LanguageGroup.FOREIGN

            # Get alternative names and ISO codes
            alt_names = []
            iso_codes = []
            for code, name in cls.LANGUAGE_STANDARDIZATION.items():
                if name == std_name:
                    if len(code) == 2 or len(code) == 3:
                        iso_codes.append(code)
                    alt_names.append(code)

            MAPPING_VALIDATIONS.labels(result='valid').inc()
            MAPPING_CACHE_HITS.inc()

            return LanguageInfo(
                name=std_name,
                code=code,
                group=group,
                family=family,
                subfamily=subfamily,
                branch=branch,
                region=region,
                subregion=subregion,
                writing_systems=[ws['name'] for ws in writing_systems],
                alternative_names=alt_names,
                iso_codes=iso_codes,
                endangered=cls._is_endangered(std_name),
                extinct=cls._is_extinct(std_name),
                revitalization_status=cls._get_revitalization_status(std_name)
            )

        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_language_info').inc()
            logger.error(f"Error getting language info for {code}: {str(e)}")
            return None
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    @lru_cache(maxsize=128)
    def get_related_languages(cls, code: str, include_extinct: bool = False) -> List[str]:
        """Get list of related languages based on family and region."""
        start_time = time.time()
        try:
            info = cls.get_language_info(code)
            if not info:
                return []

            related = set()

            # Add languages from same branch/subfamily
            for f_name, f_data in cls.LANGUAGE_FAMILIES.items():
                if f_name == info.family:
                    for sf_name, sf_data in f_data.items():
                        if sf_name == info.subfamily:
                            if isinstance(sf_data, list):
                                related.update(sf_data)
                            else:
                                for b_name, b_data in sf_data.items():
                                    if b_name == info.branch:
                                        if isinstance(b_data, list):
                                            related.update(b_data)
                                        else:
                                            for languages in b_data.values():
                                                related.update(languages)

            # Add languages from same region
            if info.region and info.subregion:
                related.update(cls.REGIONAL_MAPPING[info.region][info.subregion])

            # Remove the language itself and filter extinct languages if requested
            related.discard(info.name)
            if not include_extinct:
                related = {lang for lang in related if not cls._is_extinct(lang)}

            return sorted(related)

        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_related_languages').inc()
            logger.error(f"Error getting related languages for {code}: {str(e)}")
            return []
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def validate_code(cls, code: str) -> bool:
        """Validate if a language code is recognized."""
        start_time = time.time()
        try:
            if not code:
                MAPPING_VALIDATIONS.labels(result='invalid_empty').inc()
                return False
            
            is_valid = code in cls.VALID_CODES or code in cls.LANGUAGE_STANDARDIZATION
            MAPPING_VALIDATIONS.labels(result='valid' if is_valid else 'invalid').inc()
            return is_valid
            
        except Exception as e:
            MAPPING_ERRORS.labels(operation='validate_code').inc()
            logger.error(f"Error validating code {code}: {str(e)}")
            return False
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    def get_all_languages(cls, include_extinct: bool = False) -> List[LanguageInfo]:
        """Get information about all supported languages."""
        start_time = time.time()
        try:
            languages = []
            for code in cls.VALID_CODES:
                info = cls.get_language_info(code)
                if info and (include_extinct or not info.extinct):
                    languages.append(info)
            return sorted(languages, key=lambda x: (x.group.value, x.name))
        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_all_languages').inc()
            logger.error(f"Error getting all languages: {str(e)}")
            return []
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    def get_writing_systems(cls, code: str) -> List[Dict[str, str]]:
        """Get writing systems for a language with their metadata."""
        start_time = time.time()
        try:
            info = cls.get_language_info(code)
            if not info:
                return []

            systems = []
            for script_type, scripts in cls.WRITING_SYSTEMS.items():
                for script_name, script_data in scripts.items():
                    if info.name in script_data['Languages']:
                        systems.append({
                            'name': script_name,
                            'type': script_type,
                            'period': script_data['Period'],
                            'status': script_data['Status']
                        })
            return systems

        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_writing_systems').inc()
            logger.error(f"Error getting writing systems for {code}: {str(e)}")
            return []
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    @lru_cache(maxsize=256)
    def get_standardized_name(cls, code: str) -> Optional[str]:
        """Get standardized language name from code."""
        start_time = time.time()
        try:
            if not code:
                return None
            return cls.LANGUAGE_STANDARDIZATION.get(code)
        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_standardized_name').inc()
            logger.error(f"Error getting standardized name for {code}: {str(e)}")
            return None
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    @lru_cache(maxsize=256)
    def get_codes_for_name(cls, name: str) -> List[str]:
        """Get all codes that map to a standardized name."""
        start_time = time.time()
        try:
            if not name:
                return []
            return [
                code for code, std_name in cls.LANGUAGE_STANDARDIZATION.items()
                if std_name == name
            ]
        except Exception as e:
            MAPPING_ERRORS.labels(operation='get_codes_for_name').inc()
            logger.error(f"Error getting codes for name {name}: {str(e)}")
            return []
        finally:
            MAPPING_OPERATION_DURATION.observe(time.time() - start_time)

    @classmethod
    def _is_endangered(cls, language_name: str) -> bool:
        """Check if a language is endangered."""
        # Add implementation based on UNESCO's Atlas of the World's Languages in Danger
        return False

    @classmethod
    def _is_extinct(cls, language_name: str) -> bool:
        """Check if a language is extinct."""
        return language_name in ['Sinaunang Tagalog']

    @classmethod
    def _get_revitalization_status(cls, language_name: str) -> Optional[str]:
        """Get the revitalization status of a language."""
        # Add implementation based on language revitalization programs
        return None

    @classmethod
    def get_language_stats(cls) -> Dict[str, Any]:
        """Get comprehensive language statistics."""
        stats = {
            "total_languages": len(cls.VALID_CODES),
            "by_group": {
                "philippine": 0,
                "foreign": 0,
                "historical": 0
            },
            "by_family": {},
            "by_region": {},
            "writing_systems": {
                "native": 0,
                "adapted": 0
            },
            "endangered": 0,
            "extinct": 0
        }

        for code in cls.VALID_CODES:
            info = cls.get_language_info(code)
            if info:
                stats["by_group"][info.group.value.lower()] += 1
                
                if info.family:
                    stats["by_family"][info.family] = stats["by_family"].get(info.family, 0) + 1
                
                if info.region:
                    stats["by_region"][info.region] = stats["by_region"].get(info.region, 0) + 1
                
                if info.endangered:
                    stats["endangered"] += 1
                
                if info.extinct:
                    stats["extinct"] += 1

        return stats