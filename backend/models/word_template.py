"""
Model for word templates.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import db

class WordTemplate(db.Model):
    """Model for word templates."""
    __tablename__ = 'word_templates'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    template = Column(String(255), nullable=False)
    template_type = Column(String(64))
    pattern = Column(String(255))
    template_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', backref='templates')
    
    def __repr__(self):
        return f"<WordTemplate {self.template} ({self.template_type})>"
    
    def to_dict(self):
        """Convert word template to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'template': self.template,
            'template_type': self.template_type,
            'pattern': self.pattern,
            'template_metadata': self.template_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 