"""
Etymology model definition.
"""

from backend.database import db
from datetime import datetime
from backend.models.base_model import BaseModel

class Etymology(BaseModel):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    etymology_text = db.Column(db.Text, nullable=False)
    normalized_components = db.Column(db.Text)
    etymology_structure = db.Column(db.Text)
    language_codes = db.Column(db.Text)
    sources = db.Column(db.Text, nullable=False)
    
    # Relationships
    word = db.relationship('Word', back_populates='etymologies')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq'),
        # Add GIN index for language_codes full-text search
        db.Index('idx_etymologies_langs', db.text("to_tsvector('simple', language_codes)"), postgresql_using='gin')
    )
    
    def __repr__(self):
        return f'<Etymology {self.id}: {self.etymology_text[:50]}...>'
    
    def to_dict(self):
        """Convert etymology to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'normalized_components': self.normalized_components,
            'etymology_structure': self.etymology_structure,
            'language_codes': self.language_codes,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 