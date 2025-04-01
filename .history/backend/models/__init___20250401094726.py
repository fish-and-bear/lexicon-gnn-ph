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

def init_app(app):
    """Initialize the models with the Flask app."""
    db.init_app(app)
    
    # Import models here to avoid circular imports
    from .parts_of_speech import PartOfSpeech
    from .word import Word
    from .definition import Definition
    from .etymology import Etymology
    from .relation import Relation
    from .affixation import Affixation
    from .pronunciation import Pronunciation
    from .credit import Credit
    from .definition_relation import DefinitionRelation
    
    # Create all tables
    with app.app_context():
        db.create_all()
        logger.info("Database tables created successfully")

__all__ = [
    'db',
    'init_app'
] 