"""
Definition relation model definition.
"""

from database import db
from datetime import datetime

class DefinitionRelation(db.Model):
    """Model for definition relationships."""
    __tablename__ = 'definition_relations'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False)
    related_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = db.Column(db.String(50), nullable=False)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    definition = db.relationship('Definition', backref=db.backref('definition_relations', lazy='dynamic', cascade='all, delete-orphan'))
    related_word = db.relationship('Word', backref=db.backref('definition_relations', lazy='dynamic', cascade='all, delete-orphan'))
    
    def __repr__(self):
        return f'<DefinitionRelation {self.id}: {self.relation_type}>'
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'related_word_id': self.related_word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 