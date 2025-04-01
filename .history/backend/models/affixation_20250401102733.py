"""
Affixation model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

class Affixation(BaseModel):
    """Model for word affixations."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    affix_type = db.Column(db.String(64), nullable=False)
    sources = db.Column(db.Text, nullable=False)
    
    # Relationships
    root_word = db.relationship('Word', 
                              foreign_keys=[root_word_id],
                              backref=db.backref('root_affixations', lazy='dynamic', cascade='all, delete-orphan'))
    affixed_word = db.relationship('Word',
                                 foreign_keys=[affixed_word_id],
                                 backref=db.backref('affixed_affixations', lazy='dynamic', cascade='all, delete-orphan'))
    
    __table_args__ = (
        db.UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
    )
    
    def __repr__(self):
        return f'<Affixation {self.id}: {self.affix_type}>'
    
    def to_dict(self):
        """Convert affixation to dictionary."""
        return {
            'id': self.id,
            'root_word_id': self.root_word_id,
            'affixed_word_id': self.affixed_word_id,
            'affix_type': self.affix_type,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None
        } 