"""
Debug script to test direct database lookup of words.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from dictionary_manager import normalize_lemma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def lookup_word(word: str):
    """Look up a word directly in the database."""
    # Get database connection URL
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
    
    # Create engine
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        # Normalize the input word
        normalized = normalize_lemma(word)
        logger.info(f"Looking up word: {word} (normalized: {normalized})")
        
        # Execute direct SQL query for diagnostics
        sql_result = session.execute(text(
            "SELECT id, lemma, normalized_lemma, language_code FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized}).fetchall()
        
        if not sql_result:
            logger.info(f"No results found for: {normalized}")
            return None
        
        logger.info(f"SQL query found {len(sql_result)} results")
        for row in sql_result:
            logger.info(f"  ID: {row.id}, Lemma: {row.lemma}, Language: {row.language_code}")
        
        # Fetch related data
        word_id = sql_result[0].id
        definitions = session.execute(text(
            "SELECT id, definition_text FROM definitions WHERE word_id = :word_id"
        ), {"word_id": word_id}).fetchall()
        
        logger.info(f"Found {len(definitions)} definitions")
        for def_row in definitions:
            logger.info(f"  Definition ID: {def_row.id}, Text: {def_row.definition_text[:50]}...")
        
        return {
            "word_info": dict(sql_result[0]),
            "definitions": [dict(d) for d in definitions]
        }
        
    except Exception as e:
        logger.error(f"Error looking up word: {str(e)}")
        return None
    finally:
        session.close()

if __name__ == "__main__":
    # Get word from command line argument or use default
    word = sys.argv[1] if len(sys.argv) > 1 else "aso"
    result = lookup_word(word)
    
    if result:
        logger.info("Lookup successful!")
        logger.info(f"Word details: {result}")
    else:
        logger.error("Word lookup failed") 