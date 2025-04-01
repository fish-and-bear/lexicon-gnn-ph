"""
Definition model definition.
"""

from database import db
from datetime import datetime
import json
from .base_model import BaseModel

class Definition(BaseModel):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.Text)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'), index=True)
    examples = db.Column(db.Text)  # JSON stored as text
    usage_notes = db.Column(db.Text)
    tags = db.Column(db.Text)  # JSON stored as text
    sources = db.Column(db.Text, nullable=False)
    metadata = db.Column(db.JSON)
    
    # Relationships
    word = db.relationship('Word', backref=db.backref('definitions', lazy='dynamic', cascade='all, delete-orphan'))
    standardized_pos = db.relationship('PartOfSpeech', backref=db.backref('definitions', lazy='dynamic'))
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
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 