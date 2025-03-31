#!/usr/bin/env python
"""
Create test data files for the dictionary manager.

This script generates minimal test data files for each of the supported
file formats to test the dictionary manager without requiring full data files.
"""

import os
import json
import argparse
from typing import Dict, Any, List

def ensure_data_dir(data_dir: str):
    """Create the data directory if it doesn't exist."""
    os.makedirs(data_dir, exist_ok=True)

def create_tagalog_words_file(data_dir: str):
    """Create a minimal tagalog-words.json file."""
    file_path = os.path.join(data_dir, "tagalog-words.json")
    
    # Create sample data
    data = {
        "abaniko": {
            "part_of_speech": "noun",
            "definitions": ["a fan"],
            "etymology": "from Spanish abanico",
            "pronunciation": "a-ba-ni-ko"
        },
        "aklat": {
            "part_of_speech": "noun",
            "definitions": ["a book", "written or printed work"],
            "pronunciation": "ak-lat"
        },
        "bahay": {
            "part_of_speech": "noun",
            "definitions": ["a house or home; a dwelling place", "a structure for habitation"],
            "derivative": "mag-bahay, bahay-bahayan",
            "pronunciation": "ba-hay"
        }
    }
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {file_path}")

def create_root_words_file(data_dir: str):
    """Create a minimal root-words-cleaned.json file."""
    file_path = os.path.join(data_dir, "root-words-cleaned.json")
    
    # Create sample data
    data = [
        {
            "root_word": "aklat",
            "language_code": "tl",
            "pos": "n",
            "definition": "book",
            "associated_words": ["aklatan", "mag-aklat", "pag-aaklat"]
        },
        {
            "root_word": "bahay",
            "language_code": "tl",
            "pos": "n",
            "definition": "house",
            "associated_words": ["bahayan", "kabahayan", "pambahay"]
        },
        {
            "root_word": "turo",
            "language_code": "tl",
            "pos": "v",
            "definition": "to teach; to point",
            "associated_words": ["magturo", "pagtuturo", "ituro", "magtuturo"]
        }
    ]
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {file_path}")

def create_kwf_dictionary_file(data_dir: str):
    """Create a minimal kwf-dictionary.json file."""
    file_path = os.path.join(data_dir, "kwf-dictionary.json")
    
    # Create sample data
    data = {
        "aklat": {
            "pos": {
                "n": {
                    "definitions": [
                        {
                            "meaning": "anumang limbag na yari sa magkakabindeng mga pahina",
                            "example_sets": [
                                {
                                    "examples": [
                                        {"text": "Nabasa mo ba ang aklat?"}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "bahay": {
            "pos": {
                "n": {
                    "definitions": [
                        {
                            "meaning": "tirahan o tahanan ng tao",
                            "example_sets": [
                                {
                                    "examples": [
                                        {"text": "Malinis ang bahay nila."}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }
        }
    }
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {file_path}")

def create_kaikki_file(data_dir: str):
    """Create a minimal kaikki-tl.jsonl file."""
    file_path = os.path.join(data_dir, "kaikki-tl.jsonl")
    
    # Create sample data
    entries = [
        {
            "word": "aso",
            "pos": "n",
            "senses": [
                {
                    "glosses": ["dog", "canine"],
                    "examples": [
                        {"text": "Ang aso ay kumakahol."}
                    ]
                }
            ]
        },
        {
            "word": "pusa",
            "pos": "n",
            "senses": [
                {
                    "glosses": ["cat", "feline"],
                    "examples": [
                        {"text": "Ang pusa ay umiiyak."}
                    ]
                }
            ]
        },
        {
            "word": "mahal",
            "pos": "adj",
            "senses": [
                {
                    "glosses": ["expensive", "costly"],
                    "examples": [
                        {"text": "Mahal ang bilihin."}
                    ]
                },
                {
                    "glosses": ["dear", "beloved"],
                    "examples": [
                        {"text": "Mahal kita."}
                    ]
                }
            ],
            "relations": {
                "synonyms": ["minamahal"],
                "antonyms": ["mura"]
            }
        }
    ]
    
    # Write to file (JSONL format - one JSON object per line)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created {file_path}")

def main():
    """Main function to create test data files."""
    parser = argparse.ArgumentParser(description="Create test data files for the dictionary manager.")
    parser.add_argument('--data-dir', default='data', help='Data directory (default: data)')
    parser.add_argument('--all', action='store_true', help='Create all test data files')
    parser.add_argument('--tagalog-words', action='store_true', help='Create tagalog-words.json')
    parser.add_argument('--root-words', action='store_true', help='Create root-words-cleaned.json')
    parser.add_argument('--kwf', action='store_true', help='Create kwf-dictionary.json')
    parser.add_argument('--kaikki', action='store_true', help='Create kaikki-tl.jsonl')
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    ensure_data_dir(args.data_dir)
    
    # Create requested files
    if args.all or args.tagalog_words:
        create_tagalog_words_file(args.data_dir)
    
    if args.all or args.root_words:
        create_root_words_file(args.data_dir)
    
    if args.all or args.kwf:
        create_kwf_dictionary_file(args.data_dir)
    
    if args.all or args.kaikki:
        create_kaikki_file(args.data_dir)
    
    print(f"Test data files created in {os.path.abspath(args.data_dir)}")

if __name__ == "__main__":
    main() 