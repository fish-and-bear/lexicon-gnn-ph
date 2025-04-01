"""
Definition model definition.
"""

from database import db
from datetime import datetime
import json

class Definition(db.Model):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.String(50), index=True)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id', ondelete='SET NULL'), index=True)
    examples = db.Column(db.JSON)
    usage_notes = db.Column(db.Text)
    tags = db.Column(db.JSON)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    word = db.relationship('Word', backref=db.backref('definitions', lazy='dynamic', cascade='all, delete-orphan'))
    related_words = db.relationship('Word', secondary='definition_relations', backref=db.backref('related_definitions', lazy='dynamic'))
    
    def __repr__(self):
        return f'<Definition {self.id}: {self.definition_text[:50]}...>'
    
    def to_dict(self):
        """Convert definition to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'definition_text': self.definition_text,
            'original_pos': self.original_pos,
            'standardized_pos_id': self.standardized_pos_id,
            'examples': self.examples,
            'usage_notes': self.usage_notes,
            'tags': self.tags,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 