"""
Affixation model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from typing import Dict, Any, Optional
from enum import Enum
import json

class AffixType(Enum):
    PREFIX = 'prefix'
    SUFFIX = 'suffix'
    INFIX = 'infix'
    CIRCUMFIX = 'circumfix'

# Create a custom mixin with only the columns we need
class BasicColumnsMixin:
    """Mixin to add only essential columns to models."""
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class Affixation(BaseModel, BasicColumnsMixin):
    """Model for word affixations (root/affixed word relationships)."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affix_type = db.Column(db.String(50), nullable=False) # E.g., prefix, suffix, infix
    sources = db.Column(db.Text, nullable=True) # Changed to nullable
    
    # Relationships
    root_word = db.relationship('Word', foreign_keys=[root_word_id], back_populates='root_affixations', lazy='selectin')
    affixed_word = db.relationship('Word', foreign_keys=[affixed_word_id], back_populates='affixed_affixations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
        # Added missing standard indexes
        db.Index('idx_affixations_root', 'root_word_id'),
        db.Index('idx_affixations_affixed', 'affixed_word_id'),
        db.Index('idx_affixations_type', 'affix_type')
    )
    
    def __init__(self, **kwargs):
        metadata = kwargs.pop('metadata', None)
        if metadata is not None:
            # Store metadata in sources if needed
            if 'sources' not in kwargs and metadata:
                kwargs['sources'] = json.dumps(metadata)
        super().__init__(**kwargs)
    
    @validates('affix_type')
    def validate_affix_type(self, key: str, value: str) -> str:
        """Validate affix type."""
        if value not in AffixType.__members__:
            raise ValueError(f"Invalid affix type: {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert affixation to dictionary."""
        result = {
            'id': self.id,
            'root_word_id': self.root_word_id,
            'affixed_word_id': self.affixed_word_id,
            'affix_type': self.affix_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Handle sources field
        if self.sources:
            # Try to parse as JSON metadata first
            try:
                metadata = json.loads(self.sources)
                if isinstance(metadata, dict):
                    result['metadata'] = metadata
                    # Don't include sources twice
                else:
                    result['sources'] = self.sources.split(', ')
            except (json.JSONDecodeError, TypeError):
                # Not JSON, treat as normal sources list
                result['sources'] = self.sources.split(', ')
        else:
            result['sources'] = []
            result['metadata'] = {}
            
        return result
    
    def __repr__(self) -> str:
        return f'<Affixation {self.id}: {self.root_word.lemma} + {self.affixed_word.lemma} ({self.affix_type})>' 