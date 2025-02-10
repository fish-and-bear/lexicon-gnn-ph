from typing import Dict, Any, List, Optional
from backend.source_standardization import DictionarySource, SourceStandardization
from backend.language_systems import LanguageSystem
from backend.language_types import ConsolidatedEntry, WordForm, Definition, SourceInfo
import logging
import re

class DictionaryProcessor:
    def __init__(self, language_system: LanguageSystem):
        self.language_system = language_system
        self.logger = logging.getLogger(__name__)
        
    def process_entry(self, word: str, source_file: str, data: Dict) -> Optional[Dict]:
        """Process a dictionary entry from a specific source."""
        try:
            source = SourceStandardization.get_source_enum(source_file)
            if not source:
                self.logger.error(f"Unknown source file: {source_file}")
                return None
                
            processed = {
                'word': word,
                'source': SourceStandardization.get_display_name(source_file),
                'source_enum': source,
                'data': {}
            }
            
            if source == DictionarySource.KAIKKI:
                processed['data'] = self._process_kaikki(data)
            elif source == DictionarySource.KWF:
                processed['data'] = self._process_kwf(data)
            elif source == DictionarySource.ROOT_WORDS:
                processed['data'] = self._process_root_words(data)
            elif source == DictionarySource.TAGALOG_WORDS:
                processed['data'] = self._process_tagalog_words(data)
                
            return processed

        except Exception as e:
            self.logger.error(f"Error processing {word} from {source_file}: {str(e)}")
            return None
            
    def _process_kaikki(self, data: Dict) -> Dict:
        """Process Kaikki dictionary entry."""
        processed = {
            'pos': data.get('pos'),
            'definitions': [],
            'forms': [],
            'metadata': {}
        }
        
        # Process definitions from senses
        if 'senses' in data:
            for sense in data['senses']:
                processed['definitions'].extend(sense.get('glosses', []))
                
        # Process forms including Baybayin
        if 'forms' in data:
            for form in data['forms']:
                processed['forms'].append({
                    'form': form['form'],
                    'tags': form.get('tags', [])
                })
                
        # Process metadata
        if 'etymology_text' in data:
            processed['metadata']['etymology'] = data['etymology_text']
            
        if 'sounds' in data:
            processed['metadata']['pronunciation'] = [
                s['ipa'] for s in data['sounds'] if 'ipa' in s
            ]
            
        return processed

    def _process_kwf(self, data: Dict) -> Dict:
        """Process KWF dictionary entry."""
        processed = {
            'pos': data.get('part_of_speech', []),
            'definitions': [],
            'metadata': {}
        }

            # Process definitions
        if 'definitions' in data:
            for pos_defs in data['definitions'].values():
                for def_entry in pos_defs:
                    processed['definitions'].append({
                        'meaning': def_entry['meaning'],
                        'category': def_entry.get('category'),
                        'usage_notes': def_entry.get('usage_notes')
                    })

            # Process metadata
        if 'metadata' in data:
            processed['metadata'].update(data['metadata'])

        return processed

    def _process_root_words(self, data: Dict) -> Dict:
        """Process Root Words dictionary entry."""
        processed = {
            'type': data.get('type'),
            'definition': data.get('definition'),
            'related_words': {}
        }
        
        if 'process' in data:
            process_type = data['process']
            processed['related_words'][process_type] = data.get('derived_word', '')
            
        return processed

    def _process_tagalog_words(self, data: Dict) -> Dict:
        """Process Tagalog Words dictionary entry."""
        processed = {
            'pos': data.get('part_of_speech'),
            'pronunciation': data.get('pronunciation'),
            'definitions': data.get('definitions', []),
            'etymology': data.get('etymology'),
            'derivative': data.get('derivative')
        }
        return processed

    def process_word_metadata(self, entry: Dict, source: DictionarySource) -> Dict:
        """Process metadata from any dictionary source."""
        metadata = {
            'etymology': None,
            'source_language': None,
            'pronunciation': {},
            'baybayin': None,
            'regional_usage': [],
            'register': None,
            'domain': None
        }

        try:
            if source == DictionarySource.KAIKKI:
                if 'etymology_text' in entry:
                    metadata['etymology'] = entry['etymology_text']
                    # Extract source language from etymology
                    if 'Borrowed from' in entry['etymology_text']:
                        match = re.search(r'Borrowed from (\w+)', entry['etymology_text'])
                        if match:
                            metadata['source_language'] = match.group(1)
            
            if 'sounds' in entry:
                metadata['pronunciation']['ipa'] = [
                    s['ipa'] for s in entry['sounds'] 
                    if 'ipa' in s and 'Standard-Tagalog' in s.get('tags', [])
                ]

            elif source == DictionarySource.KWF:
                if 'metadata' in entry:
                    metadata.update(entry['metadata'])

            elif source == DictionarySource.TAGALOG_WORDS:
                if 'pronunciation' in entry:
                    metadata['pronunciation']['syllables'] = [entry['pronunciation']]
                if 'etymology' in entry:
                    metadata['etymology'] = entry['etymology']

        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}")

        return metadata

    def process_etymology(self, etymology: str, source: DictionarySource) -> Dict:
        """Process etymology information."""
        etymology_data = {
            'etymology_text': etymology,
            'source_language': None,
            'period': None,
            'confidence': 1.0
        }
        
        # Extract source language
        if 'Borrowed from' in etymology:
            match = re.search(r'Borrowed from (\w+)', etymology)
            if match:
                etymology_data['source_language'] = match.group(1)
                etymology_data['confidence'] = 0.9
                
        # Extract period if available
        period_matches = re.findall(r'(\d{1,4}(?:st|nd|rd|th)? century|\d{4}s|\d{4})', etymology)
        if period_matches:
            etymology_data['period'] = period_matches[0]
            
        return etymology_data

    def standardize_pronunciation(self, pron_data: Any) -> Optional[Dict]:
        """Standardize pronunciation data."""
        if not pron_data:
            return None
        
        result = {
            'ipa': [],
            'romanized': None,
            'syllables': [],
            'stress_pattern': None
        }
        
        if isinstance(pron_data, str):
            result['romanized'] = pron_data
            result['syllables'] = self.language_system.get_syllables(pron_data)
        elif isinstance(pron_data, list):
            for p in pron_data:
                if 'ipa' in str(p).lower():
                    result['ipa'].append(p)
                else:
                    result['syllables'].append(p)
            if result['syllables']:
                result['romanized'] = ''.join(result['syllables'])
                
        return result

    def validate_relation_type(self, rel_type: str) -> bool:
        """Validate relation type."""
        valid_types = {
            'synonym', 'antonym', 'hypernym', 'hyponym', 
            'derived_from', 'variant_of', 'root_word',
            'compound_of', 'see_also'
        }
        return rel_type.lower() in valid_types