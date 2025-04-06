"""
Affixation model definition.
"""

from database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from typing import Dict, Any, Optional
from enum import Enum

class AffixType(Enum):
    PREFIX = 'prefix'
    SUFFIX = 'suffix'
    INFIX = 'infix'
    CIRCUMFIX = 'circumfix'

class Affixation(BaseModel, StandardColumnsMixin):
    """Model for word affixations (root/affixed word relationships)."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affix_type = db.Column(db.String(50), nullable=False) # E.g., prefix, suffix, infix
    sources = db.Column(db.Text, nullable=True) # Changed to nullable
    metadata = db.Column(db.JSON, default=lambda: {}) # Added metadata field
    
    # Relationships
    root_word = db.relationship('Word', foreign_keys=[root_word_id], back_populates='root_affixations', lazy='selectin')
    affixed_word = db.relationship('Word', foreign_keys=[affixed_word_id], back_populates='affixed_affixations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
        # Added missing standard indexes
        db.Index('idx_affixations_root', 'root_word_id'),
        db.Index('idx_affixations_affixed', 'affixed_word_id'),
        db.Index('idx_affixations_type', 'affix_type'),
        db.Index('idx_affixations_metadata', 'metadata', postgresql_using='gin') # Index for metadata
    )
    
    @validates('affix_type')
    def validate_affix_type(self, key: str, value: str) -> str:
        """Validate affix type."""
        if value not in AffixType.__members__:
            raise ValueError(f"Invalid affix type: {value}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert affixation to dictionary."""
        return {
            'id': self.id,
            'root_word_id': self.root_word_id,
            'affixed_word_id': self.affixed_word_id,
            'affix_type': self.affix_type,
            'sources': self.sources.split(', ') if self.sources else [], # Adapt source splitting
            'metadata': self.metadata or {}, # Include metadata
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self) -> str:
        return f'<Affixation {self.id}: {self.root_word.lemma} + {self.affixed_word.lemma} ({self.affix_type})>' 