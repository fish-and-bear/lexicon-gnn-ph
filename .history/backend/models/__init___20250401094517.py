"""
Models package initialization.
"""

import logging
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy with no app context yet
db = SQLAlchemy()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_models():
    """Initialize models with retry logic."""
    try:
        # Import models after db initialization to avoid circular imports
        from .word import Word
        from .definition import Definition
        from .etymology import Etymology
        from .relation import Relation
        from .affixation import Affixation
        from .parts_of_speech import PartOfSpeech
        from .pronunciation import Pronunciation
        from .credit import Credit
        
        logger.info("Models initialized successfully")
        return {
            'Word': Word,
            'Definition': Definition,
            'Etymology': Etymology,
            'Relation': Relation,
            'Affixation': Affixation,
            'PartOfSpeech': PartOfSpeech,
            'Pronunciation': Pronunciation,
            'Credit': Credit
        }
    except ImportError as e:
        logger.error(f"Failed to import models: {str(e)}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error during model initialization: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model initialization: {str(e)}")
        raise

# Initialize models
models = init_models()

# Export models
globals().update(models)

__all__ = ['db'] + list(models.keys()) 