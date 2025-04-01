"""
Affixation model definition.
"""

from database import db
from datetime import datetime

class Affixation(db.Model):
    """Model for word affixations."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affix_type = db.Column(db.String(50), nullable=False)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 