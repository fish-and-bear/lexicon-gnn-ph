"""
Definition model for word definitions.
"""

from database import db
from datetime import datetime
import json
from .base_model import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, UniqueConstraint, Index

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
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Optimized relationships with proper cascade rules
    word = db.relationship('Word', 
                         back_populates='definitions', 
                         lazy='joined')
    standardized_pos = db.relationship('PartOfSpeech', 
                                     back_populates='definitions', 
                                     lazy='joined')
    
    # Definition relation relationships with proper cascade rules
    definition_relations = db.relationship('DefinitionRelation', 
                                         back_populates='definition', 
                                         lazy='dynamic', 
                                         cascade='all, delete-orphan',
                                         overlaps="related_words")
    
    # Many-to-many relationship with Word through DefinitionRelation
    related_words = db.relationship('Word', 
                                  secondary='definition_relations',
                                  primaryjoin='Definition.id == DefinitionRelation.definition_id',
                                  secondaryjoin='DefinitionRelation.word_id == Word.id',
                                  overlaps="definition_relations,word.definition_relations,word.related_definitions",
                                  lazy='dynamic',
                                  viewonly=True)
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word_pos', 'word_id', 'standardized_pos_id'),
        db.Index('idx_definitions_text', 'definition_text', postgresql_using='gin', postgresql_ops={'definition_text': 'gin_trgm_ops'})
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 