"""
Data Manager for Filipino Dictionary Application

This module provides a flexible ETL (Extract, Transform, Load) pipeline
for importing data from various formats into the database.
"""

import os
import json
import csv
import logging
import importlib
import traceback
from typing import Dict, List, Any, Optional, Union, Type
import pandas as pd
from sqlalchemy.orm import Session
from models import db, Word, Definition, PartOfSpeech, Etymology, Relation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseImporter:
    """Base class for all data importers."""
    
    def __init__(self, file_path: str, options: Dict[str, Any] = None):
        self.file_path = file_path
        self.options = options or {}
        self.data = None
        
    def extract(self) -> bool:
        """Extract data from the source file.
        
        Returns:
            bool: True if extraction was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    def transform(self) -> bool:
        """Transform the extracted data into a standard format.
        
        Returns:
            bool: True if transformation was successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement transform()")
    
    def get_transformed_data(self) -> List[Dict[str, Any]]:
        """Get the transformed data in a standard format.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing transformed data
        """
        if self.data is None:
            raise ValueError("Data has not been extracted and transformed yet")
        return self.data


class CSVImporter(BaseImporter):
    """Importer for CSV files."""
    
    def extract(self) -> bool:
        """Extract data from CSV file."""
        try:
            encoding = self.options.get('encoding', 'utf-8')
            delimiter = self.options.get('delimiter', ',')
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                raw_data = list(reader)
            
            self.raw_data = raw_data
            return True
        except Exception as e:
            logger.error(f"Error extracting data from CSV file: {e}")
            traceback.print_exc()
            return False
    
    def transform(self) -> bool:
        """Transform CSV data into standard format."""
        try:
            if not hasattr(self, 'raw_data'):
                logger.error("Data has not been extracted yet")
                return False
            
            # Define field mappings from CSV to our data model
            field_mappings = self.options.get('field_mappings', {
                'lemma': 'word',
                'language_code': 'language',
                'definitions': 'definition',
                'part_of_speech': 'pos'
            })
            
            # Transform the data
            transformed_data = []
            
            for row in self.raw_data:
                word_data = {
                    'lemma': row.get(field_mappings.get('lemma', 'word'), ''),
                    'language_code': row.get(field_mappings.get('language_code', 'language'), 'tl'),
                    'definitions': []
                }
                
                # Handle definitions
                def_field = field_mappings.get('definitions', 'definition')
                pos_field = field_mappings.get('part_of_speech', 'pos')
                
                if def_field in row and row[def_field]:
                    definition = {
                        'definition_text': row[def_field],
                        'original_pos': row.get(pos_field, ''),
                        'sources': self.options.get('source', 'CSV Import')
                    }
                    word_data['definitions'].append(definition)
                
                transformed_data.append(word_data)
            
            self.data = transformed_data
            return True
        except Exception as e:
            logger.error(f"Error transforming CSV data: {e}")
            traceback.print_exc()
            return False


class JSONImporter(BaseImporter):
    """Importer for JSON files."""
    
    def extract(self) -> bool:
        """Extract data from JSON file."""
        try:
            encoding = self.options.get('encoding', 'utf-8')
            
            with open(self.file_path, 'r', encoding=encoding) as f:
                self.raw_data = json.load(f)
            
            return True
        except Exception as e:
            logger.error(f"Error extracting data from JSON file: {e}")
            traceback.print_exc()
            return False
    
    def transform(self) -> bool:
        """Transform JSON data into standard format."""
        try:
            if not hasattr(self, 'raw_data'):
                logger.error("Data has not been extracted yet")
                return False
            
            # The transformation depends on the structure of the JSON
            structure_type = self.options.get('structure', 'array')
            
            if structure_type == 'array':
                # JSON is an array of word objects
                self.data = self._transform_array()
            elif structure_type == 'object':
                # JSON is an object with words as keys
                self.data = self._transform_object()
            else:
                logger.error(f"Unknown JSON structure type: {structure_type}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error transforming JSON data: {e}")
            traceback.print_exc()
            return False
    
    def _transform_array(self) -> List[Dict[str, Any]]:
        """Transform JSON data in array format."""
        transformed_data = []
        
        for item in self.raw_data:
            word_data = {
                'lemma': item.get('word', ''),
                'language_code': item.get('language', 'tl'),
                'definitions': []
            }
            
            # Optional fields
            if 'baybayin' in item:
                word_data['has_baybayin'] = True
                word_data['baybayin_form'] = item['baybayin']
            
            # Handle definitions
            defs = item.get('definitions', [])
            if isinstance(defs, list):
                for d in defs:
                    if isinstance(d, str):
                        definition = {
                            'definition_text': d,
                            'sources': self.options.get('source', 'JSON Import')
                        }
                    else:
                        definition = {
                            'definition_text': d.get('text', ''),
                            'original_pos': d.get('pos', ''),
                            'examples': json.dumps(d.get('examples', [])),
                            'sources': d.get('source', self.options.get('source', 'JSON Import'))
                        }
                    word_data['definitions'].append(definition)
            
            transformed_data.append(word_data)
        
        return transformed_data
    
    def _transform_object(self) -> List[Dict[str, Any]]:
        """Transform JSON data in object format."""
        transformed_data = []
        
        for lemma, details in self.raw_data.items():
            word_data = {
                'lemma': lemma,
                'language_code': details.get('language', 'tl'),
                'definitions': []
            }
            
            # Optional fields
            if 'baybayin' in details:
                word_data['has_baybayin'] = True
                word_data['baybayin_form'] = details['baybayin']
            
            # Handle definitions
            defs = details.get('definitions', [])
            if isinstance(defs, list):
                for d in defs:
                    if isinstance(d, str):
                        definition = {
                            'definition_text': d,
                            'sources': self.options.get('source', 'JSON Import')
                        }
                    else:
                        definition = {
                            'definition_text': d.get('text', ''),
                            'original_pos': d.get('pos', ''),
                            'examples': json.dumps(d.get('examples', [])),
                            'sources': d.get('source', self.options.get('source', 'JSON Import'))
                        }
                    word_data['definitions'].append(definition)
            
            transformed_data.append(word_data)
        
        return transformed_data


class ExcelImporter(BaseImporter):
    """Importer for Excel files."""
    
    def extract(self) -> bool:
        """Extract data from Excel file."""
        try:
            sheet_name = self.options.get('sheet_name', 0)
            
            df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            self.raw_data = df.to_dict('records')
            
            return True
        except Exception as e:
            logger.error(f"Error extracting data from Excel file: {e}")
            traceback.print_exc()
            return False
    
    def transform(self) -> bool:
        """Transform Excel data into standard format."""
        try:
            if not hasattr(self, 'raw_data'):
                logger.error("Data has not been extracted yet")
                return False
            
            # Define field mappings from Excel to our data model
            field_mappings = self.options.get('field_mappings', {
                'lemma': 'Word',
                'language_code': 'Language',
                'definitions': 'Definition',
                'part_of_speech': 'POS'
            })
            
            # Transform the data
            transformed_data = []
            
            for row in self.raw_data:
                word_data = {
                    'lemma': row.get(field_mappings.get('lemma', 'Word'), ''),
                    'language_code': row.get(field_mappings.get('language_code', 'Language'), 'tl'),
                    'definitions': []
                }
                
                # Handle definitions
                def_field = field_mappings.get('definitions', 'Definition')
                pos_field = field_mappings.get('part_of_speech', 'POS')
                
                if def_field in row and row[def_field]:
                    definition = {
                        'definition_text': str(row[def_field]),
                        'original_pos': str(row.get(pos_field, '')),
                        'sources': self.options.get('source', 'Excel Import')
                    }
                    word_data['definitions'].append(definition)
                
                transformed_data.append(word_data)
            
            self.data = transformed_data
            return True
        except Exception as e:
            logger.error(f"Error transforming Excel data: {e}")
            traceback.print_exc()
            return False


def get_importer_for_file(file_path: str, options: Dict[str, Any] = None) -> Optional[BaseImporter]:
    """Get the appropriate importer for a file based on its extension.
    
    Args:
        file_path: Path to the file to import
        options: Dictionary of options for the importer
        
    Returns:
        BaseImporter: An instance of the appropriate importer
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    importers = {
        '.csv': CSVImporter,
        '.json': JSONImporter,
        '.xls': ExcelImporter,
        '.xlsx': ExcelImporter
    }
    
    if ext in importers:
        return importers[ext](file_path, options)
    
    # Check if there's a custom importer registered for this extension
    try:
        custom_importer_module = f"importers.custom_importers.{ext[1:]}_importer"
        custom_importer = importlib.import_module(custom_importer_module)
        return custom_importer.Importer(file_path, options)
    except (ImportError, AttributeError):
        logger.error(f"No importer available for file extension: {ext}")
        return None


def load_to_database(data: List[Dict[str, Any]], session: Session) -> int:
    """Load transformed data into the database.
    
    Args:
        data: List of dictionaries containing word data
        session: SQLAlchemy session
        
    Returns:
        int: Number of words added to the database
    """
    added_count = 0
    
    for word_data in data:
        try:
            # Check if word already exists
            existing_word = session.query(Word).filter_by(
                normalized_lemma=word_data['lemma'].lower(),
                language_code=word_data['language_code']
            ).first()
            
            if existing_word:
                # Update existing word
                for key, value in word_data.items():
                    if key != 'definitions' and hasattr(existing_word, key):
                        setattr(existing_word, key, value)
                
                word = existing_word
            else:
                # Create new word
                word = Word(
                    lemma=word_data['lemma'],
                    normalized_lemma=word_data['lemma'].lower(),
                    language_code=word_data['language_code'],
                    has_baybayin=word_data.get('has_baybayin', False),
                    baybayin_form=word_data.get('baybayin_form', None)
                )
                session.add(word)
                added_count += 1
            
            # Add definitions
            for def_data in word_data.get('definitions', []):
                # Check if definition already exists
                existing_def = None
                if word.id:  # Only check if word has been saved
                    existing_def = session.query(Definition).filter_by(
                        word_id=word.id,
                        definition_text=def_data['definition_text']
                    ).first()
                
                if existing_def:
                    # Update existing definition
                    for key, value in def_data.items():
                        if hasattr(existing_def, key):
                            setattr(existing_def, key, value)
                else:
                    # Create new definition
                    definition = Definition(
                        definition_text=def_data['definition_text'],
                        original_pos=def_data.get('original_pos', ''),
                        examples=def_data.get('examples', ''),
                        sources=def_data.get('sources', 'Data Import')
                    )
                    word.definitions.append(definition)
            
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding word {word_data.get('lemma', 'unknown')}: {e}")
            traceback.print_exc()
    
    return added_count


class DataManager:
    """Manager for importing data from various sources into the database."""
    
    @staticmethod
    def import_file(file_path: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Import data from a file into the database.
        
        Args:
            file_path: Path to the file to import
            options: Dictionary of options for the importer
            
        Returns:
            Dict: Results of the import operation
        """
        results = {
            'success': False,
            'file': file_path,
            'records_processed': 0,
            'records_added': 0,
            'errors': []
        }
        
        # Get the appropriate importer
        importer = get_importer_for_file(file_path, options)
        if importer is None:
            results['errors'].append(f"No importer available for {file_path}")
            return results
        
        # Extract and transform the data
        if not importer.extract():
            results['errors'].append("Failed to extract data from file")
            return results
        
        if not importer.transform():
            results['errors'].append("Failed to transform data")
            return results
        
        # Get the transformed data
        try:
            data = importer.get_transformed_data()
            results['records_processed'] = len(data)
        except Exception as e:
            results['errors'].append(f"Error getting transformed data: {str(e)}")
            return results
        
        # Load the data into the database
        try:
            # Create a new session
            session = db.session
            
            # Load the data
            results['records_added'] = load_to_database(data, session)
            results['success'] = True
        except Exception as e:
            results['errors'].append(f"Error loading data into database: {str(e)}")
            traceback.print_exc()
        
        return results
    
    @staticmethod
    def import_directory(directory: str, options: Dict[str, Any] = None, recursive: bool = False) -> List[Dict[str, Any]]:
        """Import all files from a directory.
        
        Args:
            directory: Path to the directory containing files to import
            options: Dictionary of options for the importers
            recursive: Whether to recursively search for files in subdirectories
            
        Returns:
            List[Dict]: Results for each import operation
        """
        results = []
        
        # Get all files in the directory
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_results = DataManager.import_file(file_path, options)
                    results.append(file_results)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    file_results = DataManager.import_file(file_path, options)
                    results.append(file_results)
        
        return results


def import_data(source: str, options: Dict[str, Any] = None, recursive: bool = False) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Import data from a file or directory.
    
    Args:
        source: Path to the file or directory to import
        options: Dictionary of options for the importer(s)
        recursive: Whether to recursively search for files in subdirectories
        
    Returns:
        Dict or List[Dict]: Results of the import operation(s)
    """
    if os.path.isfile(source):
        return DataManager.import_file(source, options)
    elif os.path.isdir(source):
        return DataManager.import_directory(source, options, recursive)
    else:
        return {'success': False, 'errors': [f"Source not found: {source}"]}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_manager.py <file_or_directory> [--recursive]")
        sys.exit(1)
    
    source = sys.argv[1]
    recursive = "--recursive" in sys.argv
    
    results = import_data(source, recursive=recursive)
    
    if isinstance(results, list):
        total_processed = sum(r.get('records_processed', 0) for r in results)
        total_added = sum(r.get('records_added', 0) for r in results)
        print(f"Processed {total_processed} records, added {total_added} new records")
        
        for result in results:
            if not result['success']:
                print(f"Error importing {result['file']}: {', '.join(result['errors'])}")
    else:
        if results['success']:
            print(f"Successfully imported {results['records_added']} new records from {results['records_processed']} processed records")
        else:
            print(f"Error importing {results['file']}: {', '.join(results['errors'])}") 