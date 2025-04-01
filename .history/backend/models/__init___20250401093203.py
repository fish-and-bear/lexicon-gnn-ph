"""
Models package initialization.
"""

from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy with no app context yet
db = SQLAlchemy()

# Import models after db initialization to avoid circular imports
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
    'Credit'
] 