"""
Etymology model definition.
"""

from database import db
from datetime import datetime
import json

class Etymology(db.Model):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = db.Column(db.Text, nullable=False)
    normalized_components = db.Column(db.Text)
    etymology_structure = db.Column(db.Text)
    language_codes = db.Column(db.JSON)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Etymology {self.id}: {self.etymology_text[:50]}...>'
    
    def to_dict(self):
        """Convert etymology to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'normalized_components': self.normalized_components,
            'etymology_structure': self.etymology_structure,
            'language_codes': self.language_codes,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 