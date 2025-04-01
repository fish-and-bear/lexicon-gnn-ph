"""
Models package initialization.
"""

from flask_sqlalchemy import SQLAlchemy
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy with no app context yet
db = SQLAlchemy()

def init_app(app):
    """Initialize the models with the Flask app."""
    from .base import init_app as init_base
    init_base(app)

# Import models after db initialization
from .word import Word
from .definition import Definition
from .etymology import Etymology
from .relation import Relation
from .affixation import Affixation
from .parts_of_speech import PartOfSpeech
from .pronunciation import Pronunciation
from .credit import Credit

__all__ = [
    'db',
    'Word',
    'Definition',
    'Etymology',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'Credit',
    'init_app',
] 