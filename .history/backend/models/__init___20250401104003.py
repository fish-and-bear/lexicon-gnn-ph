"""
Models initialization and exports.
"""

from database import db
from .base_model import BaseModel
from .word import Word
from .definition import Definition
from .etymology import Etymology
from .relation import Relation
from .affixation import Affixation
from .part_of_speech import PartOfSpeech
from .pronunciation import Pronunciation
from .credit import Credit
from .definition_relation import DefinitionRelation
from .definition_category import DefinitionCategory
from .word_form import WordForm
from .word_template import WordTemplate

__all__ = [
    'db',
    'BaseModel',
    'Word',
    'Definition',
    'Etymology',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'Credit',
    'DefinitionRelation',
    'DefinitionCategory',
    'WordForm',
    'WordTemplate'
] 