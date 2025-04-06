"""
Credit model definition.
"""

from backend.database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from typing import Dict, Any, Optional

class Credit(BaseModel, StandardColumnsMixin):
    """Model for word credits (contributors, sources)."""
    __tablename__ = 'credits'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    credit = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text, nullable=True)
    data_metadata = db.Column(db.JSON, default=lambda: {})
    
    # Optimized relationship with Word
    word = db.relationship('Word', back_populates='credits', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'credit', name='credits_unique'),
        db.Index('idx_credits_word', 'word_id'),
        db.Index('idx_credits_metadata', 'data_metadata', postgresql_using='gin')
    )
    
    @validates('sources')
    def validate_text_field(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate text fields."""
        if value:
            return value
        else:
            raise ValueError(f"{key} cannot be empty")
    
    def to_dict(self):
        """Convert credit to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'credit': self.credit,
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': self.data_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self) -> str:
        return f'<Credit {self.id} for word {self.word_id}: {self.credit[:50]}...>' 