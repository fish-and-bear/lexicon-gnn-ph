"""
Relation model definition.
"""

from database import db
from datetime import datetime
import json
from .base_model import BaseModel

class Relation(BaseModel):
    """Model for word relationships."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    from_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    to_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(50), nullable=False, index=True)
    sources = db.Column(db.String(255))
    relation_metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Relation {self.id}: {self.relation_type}>'
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if not value.strip():
            raise ValueError(f"{key} cannot be empty")
        return value.strip()
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'from_word_id': self.from_word_id,
            'to_word_id': self.to_word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'relation_metadata': self.relation_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    # Relationships
    source_word = db.relationship('Word', 
                                foreign_keys=[from_word_id], 
                                backref=db.backref('outgoing_relations', 
                                                  lazy='dynamic', 
                                                  cascade='all, delete-orphan'))
    target_word = db.relationship('Word', 
                                foreign_keys=[to_word_id], 
                                backref=db.backref('incoming_relations', 
                                                  lazy='dynamic', 
                                                  cascade='all, delete-orphan')) 