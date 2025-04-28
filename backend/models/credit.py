"""
Credit model definition.
"""

from backend.database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any, Optional

class Credit(BaseModel, BasicColumnsMixin):
    """Model for word credits (contributors, sources)."""
    __tablename__ = 'credits'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    credit = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text, nullable=True)
    
    # Optimized relationship with Word
    word = db.relationship('Word', back_populates='credits', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'credit', name='credits_unique'),
        db.Index('idx_credits_word', 'word_id'),
        db.Index('idx_credits_text', 'credit', postgresql_using='gin', postgresql_ops={'credit': 'gin_trgm_ops'})
    )
    
    @validates('credit')
    def validate_credit(self, key: str, value: str) -> str:
        """Validate credit text."""
        if not value:
            raise ValueError("Credit cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Credit must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Credit cannot be empty after stripping")
        self._is_modified = True
        return value
    
    @validates('sources')
    def validate_sources(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate sources field."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Sources must be a string")
        value = value.strip()
        if not value:
            return None
        self._is_modified = True
        return value
    
    def to_dict(self):
        """Convert credit to dictionary."""
        result = {
            'id': self.id,
            'word_id': self.word_id,
            'credit': self.credit,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add word if it's loaded
        if hasattr(self, 'word') and self.word:
            result['word'] = {
                'id': self.word.id,
                'lemma': self.word.lemma,
                'language_code': getattr(self.word, 'language_code', 'tl')
            }
            
        return result
    
    def __repr__(self) -> str:
        return f'<Credit {self.id} for word {self.word_id}: {self.credit[:50]}...>' 