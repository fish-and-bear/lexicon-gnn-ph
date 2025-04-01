"""
Relation model definition.
"""

from backend.database import db
from datetime import datetime
from backend.models.base_model import BaseModel
import json

class Relation(BaseModel):
    """Model for word relationships."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    from_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    to_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(64), nullable=False)
    sources = db.Column(db.Text)
    metadata = db.Column(db.JSON, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_word = db.relationship('Word', 
                                foreign_keys=[from_word_id],
                                back_populates='outgoing_relations')
    target_word = db.relationship('Word', 
                                foreign_keys=[to_word_id],
                                back_populates='incoming_relations')
    
    __table_args__ = (
        db.UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
        db.Index('idx_relations_type', 'relation_type'),
        db.Index('idx_relations_metadata', metadata, postgresql_using='gin'),
    )
    
    @property
    def bidirectional(self):
        """Check if relation is bidirectional."""
        if not self.metadata or not isinstance(self.metadata, dict):
            return False
        return bool(self.metadata.get('bidirectional', False))
    
    @property
    def confidence_score(self):
        """Get confidence score for relation."""
        if not self.metadata or not isinstance(self.metadata, dict):
            return 0.0
        return float(self.metadata.get('confidence', 0.0))
    
    def __repr__(self):
        return f'<Relation {self.id}: {self.from_word_id}->{self.to_word_id} ({self.relation_type})>'
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'from_word_id': self.from_word_id,
            'to_word_id': self.to_word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'metadata': self.metadata if isinstance(self.metadata, dict) else json.loads(self.metadata or '{}'),
            'bidirectional': self.bidirectional,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
