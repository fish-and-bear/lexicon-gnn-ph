"""
Pronunciation model definition.
"""

from database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
import json

class Pronunciation(BaseModel):
    """Model for word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    type = db.Column(db.String(20), nullable=False, default='ipa', index=True)
    value = db.Column(db.Text, nullable=False, index=True)
    tags = db.Column(db.JSON)  # JSONB in PostgreSQL
    pronunciation_metadata = db.Column(db.JSON)  # JSONB in PostgreSQL
    sources = db.Column(db.Text)
    
    # Relationships
    word = db.relationship('Word', backref=db.backref('pronunciations', lazy='dynamic', cascade='all, delete-orphan'))
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'type', 'value', name='pronunciations_unique'),
    )
    
    VALID_TYPES = {
        'ipa': 'International Phonetic Alphabet',
        'x-sampa': 'Extended SAMPA',
        'pinyin': 'Hanyu Pinyin',
        'jyutping': 'Jyutping',
        'romaji': 'Romaji',
        'audio': 'Audio file reference'
    }
    
    @validates('type')
    def validate_type(self, key, value):
        """Validate pronunciation type."""
        if not value:
            raise ValueError("Pronunciation type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Pronunciation type must be a string")
        value = value.strip().lower()
        if len(value) > 20:
            raise ValueError("Pronunciation type cannot exceed 20 characters")
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid pronunciation type. Must be one of: {', '.join(self.VALID_TYPES.keys())}")
        return value
    
    @validates('value')
    def validate_value(self, key, value):
        """Validate pronunciation value."""
        if not value:
            raise ValueError("Pronunciation value cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Pronunciation value must be a string")
        value = value.strip()
        
        # Validate based on type
        if self.type == 'ipa':
            # Basic IPA validation - could be more comprehensive
            valid_chars = set('ˈˌːəɪʊeɔæaɒʌɜɛɨʉɯɪʏʊøɘɵɤəɚɛœɜɞʌɔɑɒæɐɪ̯ʏ̯ʊ̯e̯ø̯ə̯ɚ̯ɛ̯œ̯ɜ̯ɞ̯ʌ̯ɔ̯ɑ̯ɒ̯æ̯ɐ̯ˈˌ./')
            if not all(c in valid_chars for c in value if c.isalpha()):
                raise ValueError("Invalid IPA characters")
        elif self.type == 'x-sampa':
            # Basic X-SAMPA validation
            valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"\'@{}/\\[]()=+-_<>!?')
            if not all(c in valid_chars for c in value):
                raise ValueError("Invalid X-SAMPA characters")
        elif self.type == 'audio':
            # Audio file reference validation
            if not value.endswith(('.mp3', '.wav', '.ogg')):
                raise ValueError("Audio file must be mp3, wav, or ogg format")
        
        return value
    
    @validates('tags', 'pronunciation_metadata')
    def validate_json(self, key, value):
        """Validate JSON fields."""
        if value is not None:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"{key} must be valid JSON")
            elif not isinstance(value, (dict, list)):
                raise ValueError(f"{key} must be a dict or list")
        return value
    
    def __repr__(self):
        return f'<Pronunciation {self.id}: {self.type}={self.value}>'
    
    def to_dict(self):
        """Convert pronunciation to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'type': self.type,
            'value': self.value,
            'tags': self.tags,
            'pronunciation_metadata': self.pronunciation_metadata,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_value(cls, value: str, type: str = 'ipa') -> list:
        """Get pronunciations by value and type."""
        return cls.query.filter_by(value=value.strip(), type=type.strip().lower()).all()
    
    @classmethod
    def get_by_word(cls, word_id: int) -> list:
        """Get all pronunciations for a word."""
        return cls.query.filter_by(word_id=word_id).order_by(cls.type).all() 