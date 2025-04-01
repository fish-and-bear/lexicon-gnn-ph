"""
Models package initialization.
"""

from database import db
from .word import Word
from .definition import Definition
from .definition_relation import DefinitionRelation
from .etymology import Etymology
from .language import Language
from .relation import Relation
from .affixation import Affixation
from .part_of_speech import PartOfSpeech
from .pronunciation import Pronunciation
from .credit import Credit
from .word_form import WordForm
from .word_template import WordTemplate

__all__ = [
    'Word',
    'Definition',
    'DefinitionRelation',
    'Etymology',
    'Language',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'Credit',
    'WordForm',
    'WordTemplate'
] 