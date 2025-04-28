"""
Model for word forms.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, JSON, Boolean, UniqueConstraint, Index, Text
from sqlalchemy.orm import relationship
from backend.database import db
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any
from sqlalchemy.dialects.postgresql import JSONB

class WordForm(BaseModel, BasicColumnsMixin):
    """Model for word forms (inflections, conjugations)."""
    __tablename__ = 'word_forms'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    form = Column(Text, nullable=False)
    tags = Column(JSONB, default=lambda: {})
    is_canonical = Column(Boolean, default=False, nullable=False)
    is_primary = Column(Boolean, default=False, nullable=False)
    sources = Column(Text, nullable=True)
    
    # Relationships
    word = relationship('Word', back_populates='forms', lazy='selectin')
    
    # Added table args based on schema
    __table_args__ = (
        UniqueConstraint('word_id', 'form'),
        Index('idx_word_forms_word', 'word_id'),
        Index('idx_word_forms_form', 'form')
    )
    
    def __repr__(self):
        return f"<WordForm {self.form} ({self.word_id})>"
    
    def to_dict(self):
        """Convert word form to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'form': self.form,
            'tags': self.tags or {},
            'is_canonical': self.is_canonical,
            'is_primary': self.is_primary,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 