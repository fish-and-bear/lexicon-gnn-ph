"""
Definition model for word definitions.
"""

from database import db
from datetime import datetime
import json
from .base_model import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, UniqueConstraint

class Definition(BaseModel):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.String, nullable=False)
    original_pos = db.Column(db.String(64))
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'), index=True)
    examples = db.Column(db.String)  # JSON stored as text
    usage_notes = db.Column(db.String)
    tags = db.Column(db.JSON)  # JSON stored as text
    sources = db.Column(db.String, nullable=False)
    word_metadata = db.Column(db.JSON)  # Renamed from metadata to avoid conflict
    
    # Relationships
    word = db.relationship('Word', back_populates='definitions')
    standardized_pos = db.relationship('PartOfSpeech', back_populates='definitions')
    related_words = db.relationship('Word', 
                                  secondary='definition_relations',
                                  primaryjoin='Definition.id == DefinitionRelation.definition_id',
                                  secondaryjoin='DefinitionRelation.word_id == Word.id',
                                  backref=db.backref('related_definitions', lazy='dynamic'))
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
    )
    
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
            'examples': json.loads(self.examples) if self.examples else None,
            'usage_notes': self.usage_notes,
            'tags': json.loads(self.tags) if self.tags else None,
            'sources': self.sources,
            'metadata': self.word_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 