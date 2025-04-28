"""
Models package for Filipino Dictionary API.
This package contains all SQLAlchemy models for the dictionary database.
"""

from backend.database import db

# Import all models - handle possible import errors gracefully
# Ensure models are imported in an order that respects foreign key dependencies
# Import reference tables first (like PartOfSpeech)
try:
    # Import PartOfSpeech first (it has no dependencies)
    from .part_of_speech import PartOfSpeech
    
    # Then import core models
    from .word import Word
    from .definition import Definition
    from .etymology import Etymology
    from .relation import Relation
    from .affixation import Affixation
    from .pronunciation import Pronunciation
    from .credit import Credit
    from .word_form import WordForm
    from .word_template import WordTemplate 
    
    # Then import models linking definitions/words to others
    from .definition_category import DefinitionCategory
    from .definition_link import DefinitionLink
    from .definition_relation import DefinitionRelation
    from .definition_example import DefinitionExample
except ImportError as e:
    import logging
    logging.error(f"Error importing models: {e}")
    # Define minimal stub versions of critical models
    class Word: pass
    class Definition: pass
    class Etymology: pass
    class Relation: pass
    class Affixation: pass
    class PartOfSpeech: pass # Keep stub definition here too
    class Pronunciation: pass
    class Credit: pass
    class WordForm: pass
    class WordTemplate: pass
    class DefinitionCategory: pass
    class DefinitionLink: pass
    class DefinitionRelation: pass
    class DefinitionExample: pass

# Export all models
__all__ = [
    # Reference models
    'PartOfSpeech', # List POS first
    
    # Core models
    'Word',
    'Definition',
    'Etymology',
    'Pronunciation',
    'Credit',
    'DefinitionExample',
    
    # Relational models
    'Relation',
    'Affixation',
    'DefinitionRelation',
    'DefinitionLink',
    
    'DefinitionCategory',
    
    # Utility models
    'WordForm',
    'WordTemplate'
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
        'part_of_speech': PartOfSpeech,
        'credit': Credit,
        'word_form': WordForm,
        'word_template': WordTemplate,
        'definition_category': DefinitionCategory,
        'definition_relation': DefinitionRelation,
        'definition_link': DefinitionLink,
        'definition_example': DefinitionExample
    }
    return models.get(model_name.lower()) 