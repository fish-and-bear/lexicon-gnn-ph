"""
Part of speech model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from sqlalchemy import (
    Column, Integer, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from .base import Base, TimestampMixin # Assuming TimestampMixin provides created_at, updated_at

class PartOfSpeech(Base, TimestampMixin):
    """Model for parts of speech."""
    __tablename__ = 'parts_of_speech'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(32), unique=True, nullable=False)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64), nullable=False)
    description = Column(Text)
    
    # Relationships
    definitions = relationship("Definition", back_populates="standardized_pos")
    
    __table_args__ = (
        Index('idx_parts_of_speech_code', 'code'),
        Index('idx_parts_of_speech_name', 'name_en', 'name_tl'),
        UniqueConstraint('code', name='parts_of_speech_code_uniq'),
        {'schema': 'public'}
    )
    
    @validates('code')
    def validate_code(self, key, value):
        """Validate the POS code."""
        if not value:
            raise ValueError("POS code cannot be empty")
        if not isinstance(value, str):
            raise ValueError("POS code must be a string")
        value = value.strip().lower()
        if len(value) > 32:
            raise ValueError("POS code cannot exceed 32 characters")
        if not value.isascii():
            raise ValueError("POS code must contain only ASCII characters")
        return value
    
    @validates('name_en', 'name_tl')
    def validate_name(self, key, value):
        """Validate the POS names."""
        if not value:
            raise ValueError(f"{key} cannot be empty")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        value = value.strip()
        if len(value) > 64:
            raise ValueError(f"{key} cannot exceed 64 characters")
        return value
    
    def __repr__(self):
        return f'<PartOfSpeech {self.code}: {self.name_en}>'
    
    def to_dict(self):
        """Convert part of speech to dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'name_en': self.name_en,
            'name_tl': self.name_tl,
            'description': self.description
        }
    
    @classmethod
    def get_by_code(cls, code: str) -> 'PartOfSpeech':
        """Get a part of speech by its code."""
        return cls.query.filter_by(code=code.strip().lower()).first()
    
    @classmethod
    def get_standard_codes(cls) -> list:
        """Get list of standard POS codes."""
        return [
            'n',    # Noun
            'v',    # Verb
            'adj',  # Adjective
            'adv',  # Adverb
            'pron', # Pronoun
            'prep', # Preposition
            'conj', # Conjunction
            'intj', # Interjection
            'det',  # Determiner
            'art',  # Article
            'num',  # Number
            'part', # Particle
            'aux',  # Auxiliary
            'mod',  # Modal
            'affix' # Affix
        ] 