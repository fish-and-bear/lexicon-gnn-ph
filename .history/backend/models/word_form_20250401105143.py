"""
Model for word forms.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import db

class WordForm(db.Model):
    """Model for word forms."""
    __tablename__ = 'word_forms'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    form = Column(String(255), nullable=False)
    form_type = Column(String(64))
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', backref='forms')
    
    def __repr__(self):
        return f"<WordForm {self.form} ({self.form_type})>" 