"""
Models package initialization for Filipino Dictionary.

This package contains all SQLAlchemy ORM models for the Filipino Dictionary application.
Models support caching, efficient queries, and structured API responses.
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
from .parts_of_speech import PartOfSpeechCode
from .pronunciation import Pronunciation
from .credit import Credit
from .word_form import WordForm
from .word_template import WordTemplate
from .definition_category import DefinitionCategory
from .definition_link import DefinitionLink
from .index_manager import IndexManager

# Export all models
__all__ = [
    # Core models
    'Word',
    'Definition',
    'Etymology',
    'Pronunciation',
    'Credit',
    
    # Relational models
    'Relation',
    'Affixation',
    'DefinitionRelation',
    'DefinitionLink',
    
    # Reference models
    'Language',
    'PartOfSpeech',
    'PartOfSpeechCode',
    'DefinitionCategory',
    
    # Utility models
    'WordForm',
    'WordTemplate',
    'IndexManager'
]

# Version information
__version__ = '1.2.0'

def get_model_class(model_name):
    """Get model class by name."""
    models = {
        'word': Word,
        'definition': Definition,
        'etymology': Etymology,
        'pronunciation': Pronunciation,
        'relation': Relation,
        'affixation': Affixation,
        'language': Language,
        'part_of_speech': PartOfSpeech,
        'credit': Credit,
        'word_form': WordForm,
        'word_template': WordTemplate,
        'definition_category': DefinitionCategory,
        'definition_relation': DefinitionRelation,
        'definition_link': DefinitionLink,
        'index_manager': IndexManager
    }
    return models.get(model_name.lower()) 