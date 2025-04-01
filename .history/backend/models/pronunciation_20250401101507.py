"""
Pronunciation model definition.
"""

from database import db
from datetime import datetime
import json

class Pronunciation(db.Model):
    """Model for word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    type = db.Column(db.String(50), nullable=False)  # e.g., 'ipa', 'phonetic', 'audio'
    value = db.Column(db.String(255), nullable=False)
    tags = db.Column(db.JSON)
    pronunciation_metadata = db.Column(db.JSON)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Pronunciation {self.id}: {self.type}>'
    
    def to_dict(self):
        """Convert pronunciation to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'type': self.type,
            'value': self.value,
            'tags': self.tags,
            'pronunciation_metadata': self.pronunciation_metadata,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 