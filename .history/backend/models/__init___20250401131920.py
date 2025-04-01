"""
Models package initialization.
"""

from ..database import db

from .word import Word
from .definition import Definition
from .etymology import Etymology
from .relation import Relation
from .affixation import Affixation
from .part_of_speech import PartOfSpeech
from .pronunciation import Pronunciation
from .definition_relation import DefinitionRelation
from .credit import Credit
from .word_form import WordForm
from .word_template import WordTemplate

__all__ = [
    'db',
    'Word',
    'Definition',
    'Etymology',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'DefinitionRelation',
    'Credit',
    'WordForm',
    'WordTemplate'
] 