"""
Model for word templates.
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

class WordTemplate(BaseModel, StandardColumnsMixin):
    """Model for word templates (e.g., for conjugation patterns)."""
    __tablename__ = 'word_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    template_name = db.Column(db.Text, nullable=False)
    args = db.Column(JSONB, default=lambda: {})
    expansion = db.Column(db.Text)
    
    # Optimized relationship with Word
    word = db.relationship('Word', back_populates='templates', lazy='selectin')
    
    # Added table args based on schema
    __table_args__ = (
        db.UniqueConstraint('word_id', 'template_name', name='word_templates_unique'),
        db.Index('idx_word_templates_word', 'word_id'),
        db.Index('idx_word_templates_name', 'template_name'),
        db.Index('idx_word_templates_args', 'args', postgresql_using='gin')
    )
    
    @validates('expansion')
    def validate_text_field(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate text fields."""
        if value is None:
            raise ValueError(f"{key} cannot be None")
        return value
    
    def __repr__(self) -> str:
        return f'<WordTemplate {self.id}: {self.template_name} for word {self.word_id}>'
    
    def to_dict(self):
        """Convert word template to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'template_name': self.template_name,
            'args': self.args or {},
            'expansion': self.expansion,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 