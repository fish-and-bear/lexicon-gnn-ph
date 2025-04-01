"""
Definition relation model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

class DefinitionRelation(BaseModel):
    """Model for definition relationships."""
    __tablename__ = 'definition_relations'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(64), nullable=False)
    sources = db.Column(db.Text, nullable=False)
    
    # Optimized relationships with proper cascade rules
    definition = db.relationship('Definition', 
                               back_populates='definition_relations', 
                               lazy='joined',
                               overlaps="related_words,definition.related_words")
    related_word = db.relationship('Word', 
                                 back_populates='definition_relations', 
                                 lazy='joined',
                                 overlaps="related_definitions,word.related_definitions")
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique'),
        db.Index('idx_definition_relations_type', 'relation_type'),
        db.Index('idx_definition_relations_word', 'word_id', 'relation_type')
    )
    
    def __repr__(self):
        return f'<DefinitionRelation {self.id}: {self.relation_type}>'
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'word_id': self.word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None
        } 