"""
Affixation model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any, Optional
from enum import Enum
import json
from sqlalchemy.dialects.postgresql import JSONB

class AffixType(Enum):
    """Enumeration of valid affix types."""
    PREFIX = 'prefix'
    SUFFIX = 'suffix'
    INFIX = 'infix'
    CIRCUMFIX = 'circumfix'
    REDUPLICATION = 'reduplication'
    COMPOUND = 'compound'
    CLIPPING = 'clipping'
    BLENDING = 'blending'
    CONVERSION = 'conversion'
    DERIVATION = 'derivation'

class Affixation(BaseModel, BasicColumnsMixin):
    """Model for word affixations (root/affixed word relationships)."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affix_type = db.Column(db.String(64), nullable=False)
    sources = db.Column(db.Text, nullable=True)
    
    # Relationships
    root_word = db.relationship('Word', foreign_keys=[root_word_id], back_populates='root_affixations', lazy='selectin')
    affixed_word = db.relationship('Word', foreign_keys=[affixed_word_id], back_populates='affixed_affixations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
        db.Index('idx_affixations_root', 'root_word_id'),
        db.Index('idx_affixations_affixed', 'affixed_word_id'),
        db.Index('idx_affixations_type', 'affix_type'),
    )
    
    @validates('affix_type')
    def validate_affix_type(self, key: str, value: str) -> str:
        """Validate affix type."""
        if not value:
            raise ValueError("Affix type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Affix type must be a string")
        value = value.strip().lower()
        
        # Try to validate against AffixType enum but don't enforce it strictly
        try:
            AffixType(value)
        except ValueError:
            # Log but don't raise error for non-standard affix types
            pass
            
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
    
    @property
    def affixation_metadata(self):
        """Provide compatibility for the affixation_metadata column."""
        # Fallback to empty dict or parse from sources
        if self.sources and self.sources.startswith('{'):
            try:
                return json.loads(self.sources)
            except:
                pass
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert affixation to dictionary."""
        result = {
            'id': self.id,
            'root_word_id': self.root_word_id,
            'affixed_word_id': self.affixed_word_id,
            'affix_type': self.affix_type,
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': self.affixation_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add related root and affixed words if they're loaded
        if hasattr(self, 'root_word') and self.root_word:
            result['root_word'] = {
                'id': self.root_word.id,
                'lemma': self.root_word.lemma,
                'language_code': getattr(self.root_word, 'language_code', 'tl')
            }
            
        if hasattr(self, 'affixed_word') and self.affixed_word:
            result['affixed_word'] = {
                'id': self.affixed_word.id,
                'lemma': self.affixed_word.lemma,
                'language_code': getattr(self.affixed_word, 'language_code', 'tl')
            }
            
        return result
    
    def __repr__(self) -> str:
        root_lemma = self.root_word.lemma if hasattr(self.root_word, 'lemma') else "?"
        affixed_lemma = self.affixed_word.lemma if hasattr(self.affixed_word, 'lemma') else "?"
        return f'<Affixation {self.id}: {root_lemma} + {affixed_lemma} ({self.affix_type})>' 