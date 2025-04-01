"""
Credit model definition.
"""

from database import db
from datetime import datetime

class Credit(db.Model):
    """Model for word credits."""
    __tablename__ = 'credits'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    credit = db.Column(db.String(255), nullable=False)
    sources = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
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