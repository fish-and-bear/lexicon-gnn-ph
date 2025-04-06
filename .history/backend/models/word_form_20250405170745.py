"""
Model for word forms.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from typing import Dict, Any, Optional
from sqlalchemy.dialects.postgresql import JSONB

class WordForm(BaseModel, StandardColumnsMixin):
    """Model for word forms (inflections, conjugations)."""
    __tablename__ = 'word_forms'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    form = Column(String(255), nullable=False)
    tags = Column(JSONB, default=lambda: {})
    is_canonical = Column(Boolean, default=False, nullable=False)
    is_primary = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', back_populates='forms', lazy='selectin')
    
    # Added table args based on schema
    __table_args__ = (
        UniqueConstraint('word_id', 'form', name='word_forms_unique'),
        Index('idx_word_forms_word', 'word_id'),
        Index('idx_word_forms_tags', 'tags', postgresql_using='gin'),
        Index('idx_word_forms_form_trgm', 'form', postgresql_using='gin', postgresql_ops={'form': 'gin_trgm_ops'})
    )
    
    @validates('form')
    def validate_form(self, key: str, value: str) -> str:
        """Validate form text."""
        return value
    
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 