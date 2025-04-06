"""
Definition relation model definition.
"""

from backend.database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any, Optional
import json
from sqlalchemy.dialects.postgresql import JSONB

class DefinitionRelation(BaseModel, BasicColumnsMixin):
    """Model for relationships between definitions and words."""
    __tablename__ = 'definition_relations'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(64), nullable=False, index=True)
    sources = db.Column(db.Text, nullable=True)
    # relation_metadata column completely removed as it doesn't exist in the database
    
    # Valid relation types
    VALID_RELATION_TYPES = {
        'related_word': 'Definition semantically related to word',
        'synonym': 'Definition provides synonym',
        'antonym': 'Definition provides antonym',
        'derived_from': 'Definition derived from word',
        'example_of': 'Definition provides example of word usage',
        'see_also': 'Definition recommends also seeing word',
        'compare_with': 'Definition suggests comparing with word',
        'variant_of': 'Definition describes a variant of word',
        'derived_term': 'Definition is derived term of word',
        'hypernym': 'Definition is hypernym of word',
        'hyponym': 'Definition is hyponym of word'
    }
    
    # Use __getattr__ to handle missing attributes
    def __getattr__(self, name):
        """Handle missing attributes, specifically relation_metadata."""
        if name == 'relation_metadata':
            # Return empty dict or parse from sources if available
            if hasattr(self, 'sources') and self.sources and self.sources.startswith('{'):
                try:
                    return json.loads(self.sources)
                except:
                    pass
            return {}
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # Optimized relationships with proper cascade rules
    definition = db.relationship('Definition', back_populates='definition_relations', lazy='selectin', overlaps="related_words,word.definition_relations")
    related_word = db.relationship('Word', back_populates='definition_relations', lazy='selectin', overlaps="related_definitions,definition.definition_relations")
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique'),
        db.Index('idx_def_relations_def', 'definition_id'),
        db.Index('idx_def_relations_word', 'word_id'),
        db.Index('idx_def_relations_type', 'relation_type'),
        # Commented out since this column might not exist
        # db.Index('idx_def_relations_metadata', 'relation_metadata', postgresql_using='gin')
    )
    
    @validates('relation_type')
    def validate_relation_type(self, key: str, value: str) -> str:
        """Validate relation type."""
        if not value:
            raise ValueError("Relation type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Relation type must be a string")
        value = value.strip().lower()
        # Uncomment to enforce valid relation types
        # if value not in self.VALID_RELATION_TYPES:
        #     raise ValueError(f"Invalid relation type. Must be one of: {', '.join(self.VALID_RELATION_TYPES.keys())}")
        return value
    
    @validates('sources')
    def validate_sources(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate sources."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Sources must be a string")
        value = value.strip()
        if not value:
            return None
        return value
    
    @validates('relation_metadata')
    def validate_relation_metadata(self, key: str, value: Any) -> Dict:
        """Validate metadata JSON."""
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for metadata")
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a dictionary")
        return value
    
    # Add property for compatibility
    @property
    def relation_data(self):
        """Provide compatibility for relation_metadata when the column doesn't exist."""
        if hasattr(self, 'relation_metadata'):
            return self.relation_metadata
        # Fallback to empty dict or parse from sources
        if self.sources and self.sources.startswith('{'):
            try:
                return json.loads(self.sources)
            except:
                pass
        return {}
    
    @relation_data.setter
    def relation_data(self, value):
        """Set relation data appropriately based on what's available."""
        if hasattr(self, 'relation_metadata'):
            self.relation_metadata = value
        else:
            # Store in instance variable for this session
            self._relation_data = value
            # Optionally serialize to sources as a fallback
            if value and isinstance(value, dict):
                self.sources = json.dumps(value)
    
    def to_dict(self):
        """Convert relation to dictionary."""
        # Get metadata from relation_metadata if available, otherwise use relation_data
        metadata = self.relation_metadata if hasattr(self, 'relation_metadata') else self.relation_data
        
        result = {
            'id': self.id,
            'definition_id': self.definition_id,
            'word_id': self.word_id,
            'relation_type': self.relation_type,
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add related definition and word if loaded
        if hasattr(self, 'definition') and self.definition:
            result['definition'] = {
                'id': self.definition.id,
                'definition_text': self.definition.definition_text[:100] + '...' if len(self.definition.definition_text) > 100 else self.definition.definition_text
            }
            
        if hasattr(self, 'related_word') and self.related_word:
            result['related_word'] = {
                'id': self.related_word.id,
                'lemma': self.related_word.lemma,
                'language_code': self.related_word.language_code
            }
            
        return result
    
    def __repr__(self) -> str:
        return f'<DefinitionRelation {self.id}: Def {self.definition_id} -> Word {self.word_id} ({self.relation_type})>' 