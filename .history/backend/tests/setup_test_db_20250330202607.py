import os
import sys
import logging
from typing import Dict, Any, List
from datetime import datetime, UTC

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    db, Word, Definition, Etymology, Relation,
    PartOfSpeech, Pronunciation, Affixation
)
from config import TestConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_database():
    """Create test database and tables."""
    try:
        # Create all tables
        db.create_all()
        logger.info("Test database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating test database tables: {e}")
        raise

def create_test_data():
    """Create test data in the database."""
    try:
        # Create parts of speech
        pos_data = [
            {"code": "n", "name_en": "Noun", "name_tl": "Pangngalan"},
            {"code": "v", "name_en": "Verb", "name_tl": "Pandiwa"},
            {"code": "adj", "name_en": "Adjective", "name_tl": "Pang-uri"},
            {"code": "adv", "name_en": "Adverb", "name_tl": "Pang-abay"}
        ]
        
        pos_dict = {}
        for pos in pos_data:
            part_of_speech = PartOfSpeech(**pos)
            db.session.add(part_of_speech)
            pos_dict[pos["code"]] = part_of_speech
        
        # Create test words
        for word_data in TestConfig.TEST_WORDS.values():
            # Create word
            word = Word(
                lemma=word_data["lemma"],
                normalized_lemma=word_data["normalized_lemma"],
                language_code=word_data["language_code"],
                has_baybayin=word_data["has_baybayin"],
                baybayin_form=word_data.get("baybayin_form"),
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            db.session.add(word)
            
            # Create definitions
            for def_data in word_data.get("definitions", []):
                definition = Definition(
                    word=word,
                    definition_text=def_data["definition_text"],
                    standardized_pos=pos_dict.get(def_data["part_of_speech"]),
                    examples=def_data.get("examples", []),
                    sources="test_source",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC)
                )
                db.session.add(definition)
            
            # Create etymologies
            for etym_data in word_data.get("etymologies", []):
                etymology = Etymology(
                    word=word,
                    etymology_text=etym_data["etymology_text"],
                    language_codes=etym_data["language_codes"],
                    confidence_score=etym_data.get("confidence_score", 0.8),
                    sources="test_source",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC)
                )
                db.session.add(etymology)
        
        # Create some test relationships
        words = Word.query.all()
        if len(words) >= 2:
            relation = Relation(
                from_word=words[0],
                to_word=words[1],
                relation_type="synonym",
                confidence_score=0.8,
                sources="test_source",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            db.session.add(relation)
        
        # Commit all changes
        db.session.commit()
        logger.info("Test data created successfully")
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating test data: {e}")
        raise

def cleanup_test_database():
    """Clean up test database."""
    try:
        # Drop all tables
        db.drop_all()
        logger.info("Test database cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up test database: {e}")
        raise

def setup_test_environment():
    """Set up the complete test environment."""
    try:
        # Create test database and tables
        create_test_database()
        
        # Create test data
        create_test_data()
        
        logger.info("Test environment set up successfully")
    except Exception as e:
        logger.error(f"Error setting up test environment: {e}")
        cleanup_test_database()
        raise

if __name__ == "__main__":
    # Set up test environment
    setup_test_environment() 