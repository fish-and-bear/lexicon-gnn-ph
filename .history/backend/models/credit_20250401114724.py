"""
Credit model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

class Credit(BaseModel):
    """Model for word credits."""
    __tablename__ = 'credits'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    credit = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)
    
    # Relationships
    word = db.relationship('Word', back_populates='credits')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'credit', name='credits_unique'),
    )
    
    def __repr__(self):
        return f'<Credit {self.id}: {self.credit[:50]}...>'
    
    def to_dict(self):
        """Convert credit to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'credit': self.credit,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 