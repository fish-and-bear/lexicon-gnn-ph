"""
Models initialization and exports.
"""

from .base import init_app
from .word import Word
from .definition import Definition
from .etymology import Etymology
from .relation import Relation
from .affixation import Affixation
from .part_of_speech import PartOfSpeech
from .pronunciation import Pronunciation
from .credit import Credit
from .definition_relation import DefinitionRelation

__all__ = [
    'init_app',
    'Word',
    'Definition',
    'Etymology',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'Credit',
    'DefinitionRelation'
] 