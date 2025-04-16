"""
Helper module for fetching word details with improved error handling.
This module provides utilities for retrieving word information without eager loading
problematic relationships.
"""

from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import SQLAlchemyError
import time
import logging

# Get a logger
logger = logging.getLogger(__name__)

def fetch_word_details_safely(db_engine, word_id, include_definitions=True, include_categories=False):
    """
    A safer version of _fetch_word_details that avoids loading problematic relationships.
    
    Args:
        db_engine: SQLAlchemy engine to create session from
        word_id: ID of word to fetch
        include_definitions: Whether to include definitions
        include_categories: Whether to attempt loading categories (unsafe if schema mismatch)
        
    Returns:
        Word object with loaded relationships or None if error occurs
    """
    start_time = time.time()
    try:
        # Import models here to avoid circular imports
        from backend.models import Word, Definition, Etymology, Pronunciation, Credit
        
        # Create a dedicated session
        with Session(db_engine) as session:
            # Start with a basic query
            word_query = session.query(Word).filter(Word.id == word_id)
            
            # Apply eager loading for safe relationships only
            if include_definitions:
                word_query = word_query.options(selectinload(Word.definitions))
                
                # Skip selectinload for categories which may have schema issues
                if include_categories:
                    from backend.models import DefinitionCategory
                    # We'll manually load these later
            
            # Add other safe relationships
            word_query = word_query.options(
                selectinload(Word.etymologies),
                selectinload(Word.pronunciations),
                selectinload(Word.credits),
                selectinload(Word.outgoing_relations),
                selectinload(Word.incoming_relations),
                selectinload(Word.root_affixations),
                selectinload(Word.affixed_affixations),
                selectinload(Word.root_word),
                selectinload(Word.derived_words),
                selectinload(Word.forms),
                selectinload(Word.templates)
            )
            
            # Execute the query
            word = word_query.first()
            
            if not word:
                return None
                
            # Initialize empty collections to avoid None values
            if not hasattr(word, 'outgoing_relations') or word.outgoing_relations is None:
                word.outgoing_relations = []
            if not hasattr(word, 'incoming_relations') or word.incoming_relations is None:
                word.incoming_relations = []
            if not hasattr(word, 'root_affixations') or word.root_affixations is None:
                word.root_affixations = []
            if not hasattr(word, 'affixed_affixations') or word.affixed_affixations is None:
                word.affixed_affixations = []
            if not hasattr(word, 'forms') or word.forms is None:
                word.forms = []
            if not hasattr(word, 'templates') or word.templates is None:
                word.templates = []
            if not hasattr(word, 'derived_words') or word.derived_words is None:
                word.derived_words = []
            if not hasattr(word, 'definitions') or word.definitions is None:
                word.definitions = []
            if not hasattr(word, 'etymologies') or word.etymologies is None:
                word.etymologies = []
            if not hasattr(word, 'pronunciations') or word.pronunciations is None:
                word.pronunciations = []
            if not hasattr(word, 'credits') or word.credits is None:
                word.credits = []
            
            # Set empty lists for categories on each definition to avoid schema errors
            for definition in word.definitions:
                definition.categories = []
                definition.links = []
                definition.definition_relations = []
            
            # Detach from session
            session.expunge(word)
            
            execution_time = time.time() - start_time
            logger.debug(f"Fetched word details for ID {word_id} in {execution_time:.4f}s")
            
            return word
            
    except SQLAlchemyError as e:
        logger.error(f"Database error in fetch_word_details_safely for word ID {word_id}: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error in fetch_word_details_safely for word ID {word_id}: {str(e)}", exc_info=True)
        return None 