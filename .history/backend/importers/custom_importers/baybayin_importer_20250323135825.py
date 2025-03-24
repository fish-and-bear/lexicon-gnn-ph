"""
Custom importer for Baybayin text files.
"""

import re
import json
from typing import Dict, List, Any, Optional
import sys
import os

# Add the parent directory to sys.path so we can import from data_manager
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data_manager import BaseImporter

class BaybayinImporter(BaseImporter):
    """Importer for specially formatted Baybayin text files."""
    
    def extract(self) -> bool:
        """Extract data from a Baybayin text file."""
        try:
            encoding = self.options.get('encoding', 'utf-8')
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Store the raw content
            self.raw_data = content
            return True
        except Exception as e:
            print(f"Error extracting data from Baybayin file: {e}")
            return False
    
    def transform(self) -> bool:
        """Transform Baybayin data into standard format."""
        try:
            if not hasattr(self, 'raw_data'):
                print("Data has not been extracted yet")
                return False
            
            # Expected format:
            # word (baybayin: ᜊᜌ᜔ᜊᜌᜒᜈ᜔) - definition
            # or
            # word [baybayin: ᜊᜌ᜔ᜊᜌᜒᜈ᜔] - definition
            
            pattern = r'([^\(\)\[\]]+)[\(\[](baybayin:?\s*([^\)\]]+))[\)\]]\s*-\s*(.+)'
            matches = re.findall(pattern, self.raw_data, re.MULTILINE)
            
            transformed_data = []
            
            for match in matches:
                word = match[0].strip()
                baybayin = match[2].strip()
                definition = match[3].strip()
                
                word_data = {
                    'lemma': word,
                    'language_code': 'tl',  # Assume Tagalog
                    'has_baybayin': True,
                    'baybayin_form': baybayin,
                    'definitions': [
                        {
                            'definition_text': definition,
                            'sources': self.options.get('source', 'Baybayin Import')
                        }
                    ]
                }
                
                transformed_data.append(word_data)
            
            # Look for another format:
            # word - baybayin: ᜊᜌ᜔ᜊᜌᜒᜈ᜔ - definition
            pattern2 = r'([^\-]+)\s*-\s*baybayin:?\s*([^\-]+)\s*-\s*(.+)'
            matches2 = re.findall(pattern2, self.raw_data, re.MULTILINE)
            
            for match in matches2:
                word = match[0].strip()
                baybayin = match[1].strip()
                definition = match[2].strip()
                
                word_data = {
                    'lemma': word,
                    'language_code': 'tl',  # Assume Tagalog
                    'has_baybayin': True,
                    'baybayin_form': baybayin,
                    'definitions': [
                        {
                            'definition_text': definition,
                            'sources': self.options.get('source', 'Baybayin Import')
                        }
                    ]
                }
                
                transformed_data.append(word_data)
            
            # Store the transformed data
            self.data = transformed_data
            return True
        except Exception as e:
            print(f"Error transforming Baybayin data: {e}")
            import traceback
            traceback.print_exc()
            return False

# Export the importer as 'Importer' to match the expected interface
Importer = BaybayinImporter 