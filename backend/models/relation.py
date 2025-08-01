"""
Relation model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates, relationship
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
import json
from enum import Enum
from sqlalchemy.dialects.postgresql import JSONB
from typing import Dict, Any, Optional
import logging

# Add a logger
logger = logging.getLogger(__name__)

class RelationType(Enum):
    """Enumeration of valid relation types."""
    SYNONYM = 'synonym'
    ANTONYM = 'antonym'
    HYPERNYM = 'hypernym'
    HYPONYM = 'hyponym'
    MERONYM = 'meronym'
    HOLONYM = 'holonym'
    DERIVED = 'derived_from'
    ROOT = 'root_of'
    VARIANT = 'variant'
    RELATED = 'related'
    SPELLING_VARIANT = 'spelling_variant'
    REGIONAL_VARIANT = 'regional_variant'
    COMPARE_WITH = 'compare_with'
    SEE_ALSO = 'see_also'
    EQUALS = 'equals'
    
    @classmethod
    def get_inverse(cls, relation_type: str) -> str:
        """Get the inverse relation type."""
        inverse_map = {
            cls.HYPERNYM.value: cls.HYPONYM.value,
            cls.HYPONYM.value: cls.HYPERNYM.value,
            cls.MERONYM.value: cls.HOLONYM.value,
            cls.HOLONYM.value: cls.MERONYM.value,
            cls.DERIVED.value: cls.ROOT.value,
            cls.ROOT.value: cls.DERIVED.value
        }
        return inverse_map.get(relation_type, relation_type)

class Relation(BaseModel, BasicColumnsMixin):
    """Model for relationships between words (synonyms, antonyms, etc.)."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    from_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    to_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(64), nullable=False, index=True)
    sources = db.Column(db.Text)
    # relation_metadata = db.Column(JSONB, default=lambda: {})  # This column doesn't exist in the database
    
    # Relationships
    source_word = db.relationship('Word', foreign_keys=[from_word_id], back_populates='outgoing_relations', lazy='selectin')
    target_word = db.relationship('Word', foreign_keys=[to_word_id], back_populates='incoming_relations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
        db.Index('idx_relations_from', 'from_word_id'),
        db.Index('idx_relations_to', 'to_word_id'),
        db.Index('idx_relations_type', 'relation_type'),
        # Remove these indexes as they reference a column that doesn't exist
        # db.Index('idx_relations_metadata', 'relation_metadata', postgresql_using='gin'),
        # db.Index('idx_relations_metadata_strength', db.text("(relation_metadata->>'strength')"))
    )
    
    # Add instance variable to store relation_metadata for the current session
    _relation_metadata = {}
    
    @property
    def relation_metadata(self):
        """Compatibility property for the missing column."""
        # Return instance variable if set
        if hasattr(self, '_relation_metadata'):
            return self._relation_metadata
        
        # Try to parse from sources
        if self.sources and self.sources.startswith('{'):
            try:
                return json.loads(self.sources)
            except:
                pass
        return {}
    
    @relation_metadata.setter
    def relation_metadata(self, value):
        """Store relation_metadata in instance variable."""
        if not isinstance(value, dict) and value is not None:
            value = {}
        self._relation_metadata = value
        
        # Optionally serialize to sources as a fallback
        if value and isinstance(value, dict):
            self.sources = json.dumps(value)
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if not value:
            raise ValueError("Relation type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Relation type must be a string")
        value = value.strip().lower()
        if len(value) > 64:
            raise ValueError("Relation type cannot exceed 64 characters")
        # Try to validate against RelationType enum but don't enforce it strictly
        # as there might be custom relation types
        try:
            RelationType(value)
        except ValueError:
            # Log but don't raise error for non-standard relation types
            pass
        return value
    
    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources field."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Sources must be a string")
        value = value.strip()
        if not value:
            return None
        return value
    
    def __repr__(self) -> str:
        return f'<Relation {self.id}: {self.source_word.lemma if hasattr(self.source_word, "lemma") else "?"} -> {self.target_word.lemma if hasattr(self.target_word, "lemma") else "?"} ({self.relation_type})>'
    
    def to_dict(self):
        """Convert relation to dictionary."""
        result = {
            'id': self.id,
            'relation_type': self.relation_type,
            'from_word_id': self.from_word_id,
            'to_word_id': self.to_word_id,
            'sources': self.sources,
            'relation_data': self.relation_data or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add source and target word if they're loaded
        if hasattr(self, 'source_word') and self.source_word:
            result['source_word'] = {
                'id': self.source_word.id,
                'lemma': self.source_word.lemma,
                'language_code': self.source_word.language_code,
                'has_baybayin': getattr(self.source_word, 'has_baybayin', False),
                'baybayin_form': getattr(self.source_word, 'baybayin_form', None)
            }
            
        if hasattr(self, 'target_word') and self.target_word:
            result['target_word'] = {
                'id': self.target_word.id,
                'lemma': self.target_word.lemma,
                'language_code': self.target_word.language_code,
                'has_baybayin': getattr(self.target_word, 'has_baybayin', False),
                'baybayin_form': getattr(self.target_word, 'baybayin_form', None)
            }
            
        return result
    
    @classmethod
    def get_by_words(cls, from_word_id: int, to_word_id: int) -> list:
        """Get all relations between two words."""
        return cls.query.filter_by(
            from_word_id=from_word_id,
            to_word_id=to_word_id
        ).all()
    
    @classmethod
    def get_by_type(cls, relation_type: str) -> list:
        """Get all relations of a specific type."""
        return cls.query.filter_by(
            relation_type=relation_type.strip().lower()
        ).all()
    
    def get_inverse_type(self) -> str:
        """Get the inverse relation type."""
        return RelationType.get_inverse(self.relation_type)
    
    def create_inverse(self) -> 'Relation':
        """Create an inverse relation."""
        return Relation(
            from_word_id=self.to_word_id,
            to_word_id=self.from_word_id,
            relation_type=self.get_inverse_type(),
            sources=self.sources
        )
    
    # Add property for relation_data compatibility
    @property
    def relation_data(self):
        """Provide compatibility for relation_data."""
        return self.relation_metadata
        
    @relation_data.setter
    def relation_data(self, value):
        """Set relation data by updating relation_metadata."""
        self.relation_metadata = value
    
    # Fix language code validation issues
    def process_relation(self):
        """Process relation data safely with fallbacks for missing information."""
        try:
            if hasattr(self, 'source_word') and self.source_word:
                if not self.source_word.language_code or not isinstance(self.source_word.language_code, str):
                    # Set a default language code if missing or invalid
                    logger.warning(f"Missing language code for source_word {self.from_word_id}, using 'tl' as default")
                    self.source_word.language_code = 'tl'
            
            if hasattr(self, 'target_word') and self.target_word:
                if not self.target_word.language_code or not isinstance(self.target_word.language_code, str):
                    # Set a default language code if missing or invalid
                    logger.warning(f"Missing language code for target_word {self.to_word_id}, using 'tl' as default")
                    self.target_word.language_code = 'tl'
        except Exception as ex:
            logger.warning(f"Error during relation processing: {ex}") 