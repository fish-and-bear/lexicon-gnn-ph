"""
Models package initialization.
"""

from backend.database import db
from backend.models.word import Word
from backend.models.definition import Definition
from backend.models.definition_relation import DefinitionRelation
from backend.models.etymology import Etymology
from backend.models.language import Language
from backend.models.relation import Relation
from backend.models.affixation import Affixation
from backend.models.part_of_speech import PartOfSpeech
from backend.models.pronunciation import Pronunciation
from backend.models.credit import Credit
from backend.models.word_form import WordForm
from backend.models.word_template import WordTemplate

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