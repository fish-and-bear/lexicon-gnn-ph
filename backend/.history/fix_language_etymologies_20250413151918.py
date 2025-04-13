#!/usr/bin/env python3
"""
Fix words in the database that were incorrectly set as root words 
just because they have simple language etymologies in bracket format.

This script:
1. Identifies words with simple language code etymologies (like "[ Ing ]", "[ Esp ]")
2. Sets a proper borrowed_from relation for these words
3. Updates them to not be treated as root words where appropriate
"""

import psycopg2
import re
import os
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import json
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_language_etymologies.log')
    ]
)
logger = logging.getLogger('fix_etymology')

# Dictionary of language codes to full language names
LANGUAGE_MAP = {
    "Ing": {"code": "en", "name": "English"},
    "Esp": {"code": "es", "name": "Spanish"},
    "War": {"code": "war", "name": "Waray"},
    "Bik": {"code": "bik", "name": "Bikol"},
    "Hil": {"code": "hil", "name": "Hiligaynon"},
    "Ilk": {"code": "ilo", "name": "Ilokano"},
    "Kap": {"code": "pam", "name": "Kapampangan"},
    "Mag": {"code": "mdh", "name": "Maguindanao"},
    "Mrw": {"code": "mrw", "name": "Maranaw"},
    "Pan": {"code": "pag", "name": "Pangasinan"},
    "Tag": {"code": "tl", "name": "Tagalog"},
    "San": {"code": "sa", "name": "Sanskrit"},
    "Arb": {"code": "ar", "name": "Arabic"},
    "Ch": {"code": "zh", "name": "Chinese"},
    "Jap": {"code": "ja", "name": "Japanese"},
    "Mal": {"code": "ms", "name": "Malay"},
    "Tsino": {"code": "zh", "name": "Chinese"},
}

def connect_to_db(db_url: Optional[str] = None) -> Tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """Connect to the database using connection URL or environment variables."""
    if not db_url:
        db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        # Construct from individual env vars
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = os.environ.get('DB_PORT', '5432')
        db_name = os.environ.get('DB_NAME', 'fil_dict_db')
        db_user = os.environ.get('DB_USER', 'postgres')
        db_password = os.environ.get('DB_PASSWORD', 'postgres')
        
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    logger.info(f"Connecting to database...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    logger.info("Connected successfully")
    return conn, cur

def identify_language_etymology_words(cur) -> List[Dict]:
    """
    Find words with simple language etymology patterns like "[ Ing ]" or "[ Esp ]"
    that are incorrectly being treated as root words.
    """
    # Find words with bracketed language code etymologies
    language_codes = "|".join(LANGUAGE_MAP.keys())
    pattern = f"\\[ *({language_codes}) *\\]"
    
    query = """
    SELECT w.id, w.lemma, w.language_code, e.etymology_text, e.id as etymology_id
    FROM words w
    JOIN etymologies e ON w.id = e.word_id
    WHERE e.etymology_text ~ %s
    AND w.root_word_id IS NULL
    ORDER BY w.lemma
    """
    
    cur.execute(query, (pattern,))
    rows = cur.fetchall()
    
    result = []
    for row in rows:
        word_id, lemma, lang_code, ety_text, ety_id = row
        
        # Extract language code from etymology
        match = re.search(r'\[ *([A-Za-z]+) *\]', ety_text)
        if match:
            source_lang_code = match.group(1)
            
            # Skip if lemma is actually a language code (like "es")
            if source_lang_code.lower() == lemma.lower():
                logger.info(f"Skipping language code word: {lemma}")
                continue
                
            result.append({
                'word_id': word_id,
                'lemma': lemma,
                'language_code': lang_code,
                'etymology_text': ety_text,
                'etymology_id': ety_id,
                'source_language': source_lang_code
            })
    
    logger.info(f"Found {len(result)} words with language etymology patterns")
    return result

def fix_etymology_relations(cur, words_to_fix: List[Dict]) -> int:
    """
    Update the database to correctly classify words with language etymologies.
    
    This will:
    1. Add proper BORROWED_FROM relations
    2. Update the word's etymology classification
    """
    count_fixed = 0
    
    for word in words_to_fix:
        word_id = word['word_id']
        source_lang_code = word['source_language']
        
        # Skip if language not in our map
        if source_lang_code not in LANGUAGE_MAP:
            logger.warning(f"Unknown language code: {source_lang_code} for word {word['lemma']}")
            continue
            
        # Get language information
        language_info = LANGUAGE_MAP[source_lang_code]
        
        # Create borrowed_from relation in word_metadata
        try:
            # First get current metadata
            cur.execute(
                "SELECT word_metadata FROM words WHERE id = %s",
                (word_id,)
            )
            metadata_row = cur.fetchone()
            
            if metadata_row and metadata_row[0]:
                metadata = metadata_row[0]
            else:
                metadata = {}
                
            # Update metadata with etymology information
            if 'etymology' not in metadata:
                metadata['etymology'] = {}
                
            metadata['etymology']['borrowed_from'] = {
                'language_code': language_info['code'],
                'language_name': language_info['name'],
                'is_loanword': True
            }
            
            # Update the word record
            cur.execute(
                """
                UPDATE words 
                SET word_metadata = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (json.dumps(metadata), word_id)
            )
            
            # Also update etymology record with proper language codes
            cur.execute(
                """
                UPDATE etymologies
                SET language_codes = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (language_info['code'], word['etymology_id'])
            )
            
            count_fixed += 1
            logger.info(f"Fixed word: {word['lemma']} - added borrowed_from relation to {language_info['name']}")
            
        except Exception as e:
            logger.error(f"Error updating word {word['lemma']} (ID: {word_id}): {str(e)}")
    
    return count_fixed

def main():
    parser = argparse.ArgumentParser(description='Fix words with bracketed language etymologies in the database')
    parser.add_argument('--db-url', help='Database connection URL')
    parser.add_argument('--dry-run', action='store_true', help='Identify issues but don\'t fix them')
    args = parser.parse_args()
    
    try:
        conn, cur = connect_to_db(args.db_url)
        
        # Start transaction
        conn.autocommit = False
        
        # Identify words to fix
        words_to_fix = identify_language_etymology_words(cur)
        
        if args.dry_run:
            logger.info(f"DRY RUN: Found {len(words_to_fix)} words that need fixing")
            for word in words_to_fix[:10]:  # Show first 10
                logger.info(f"  {word['lemma']} - Etymology: {word['etymology_text']}")
            if len(words_to_fix) > 10:
                logger.info(f"  ... and {len(words_to_fix) - 10} more")
        else:
            # Fix the issues
            count_fixed = fix_etymology_relations(cur, words_to_fix)
            logger.info(f"Fixed {count_fixed} out of {len(words_to_fix)} words")
            
            # Commit the changes
            conn.commit()
            logger.info("Changes committed to database")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            logger.info("Changes rolled back due to error")
        return 1
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 