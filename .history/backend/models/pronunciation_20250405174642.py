"""
Pronunciation model definition with optimized caching and performance.
"""

from backend.database import db, cached_query
from datetime import datetime
from sqlalchemy.orm import validates
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func, and_, or_, Index, text
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
import json
from typing import List, Dict, Any, Optional, Set, Union
from sqlalchemy.dialects.postgresql import JSONB
import re

class Pronunciation(BaseModel, StandardColumnsMixin):
    """Model for word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    type = db.Column(db.String(50), nullable=False)  # E.g., 'IPA', 'syllabification', 'rhyme'
    value = db.Column(db.Text, nullable=False)
    tags = db.Column(JSONB, default=lambda: {})  # Changed to JSONB
    is_primary = db.Column(db.Boolean, default=False)
    source_info = db.Column(db.JSON, default=lambda: {})
    data_metadata = db.Column(db.JSON, default=lambda: {})  # Renamed from pronunciation_metadata
    
    # Maintain compatibility with old code via hybrid property
    @hybrid_property
    def pronunciation_metadata(self) -> Dict[str, Any]:
        """Provide access to the metadata column through a property to avoid SQLAlchemy conflicts."""
        return self.data_metadata or {}
    
    @pronunciation_metadata.setter
    def pronunciation_metadata(self, value: Dict[str, Any]) -> None:
        """Set metadata via the property."""
        self.data_metadata = value
        self._is_modified = True
    
    # Optimized relationship with Word
    word = db.relationship('Word', back_populates='pronunciations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'type', 'value', name='uq_pronunciation_word_type_value'),
        # Added missing standard indexes
        db.Index('idx_pronunciations_word', 'word_id'),
        db.Index('idx_pronunciations_type', 'type'),
        db.Index('idx_pronunciations_value_trgm', 'value', postgresql_using='gin', postgresql_ops={'value': 'gin_trgm_ops'}),
        db.Index('idx_pronunciations_tags', 'tags', postgresql_using='gin'),
        db.Index('idx_pronunciations_metadata', 'data_metadata', postgresql_using='gin')
    )
    
    VALID_TYPES = {
        'ipa': 'International Phonetic Alphabet',
        'x-sampa': 'Extended SAMPA',
        'pinyin': 'Hanyu Pinyin',
        'jyutping': 'Jyutping',
        'romaji': 'Romaji',
        'audio': 'Audio file reference',
        'respelling': 'Respelling',
        'phonemic': 'Phonemic transcription'
    }
    
    IPA_CHARS = set('ˈˌːəɪʊeɔæaɒʌɜɛɨʉɯɪʏʊøɘɵɤəɚɛœɜɞʌɔɑɒæɐɪ̯ʏ̯ʊ̯e̯ø̯ə̯ɚ̯ɛ̯œ̯ɜ̯ɞ̯ʌ̯ɔ̯ɑ̯ɒ̯æ̯ɐ̯ˈˌ./')
    X_SAMPA_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"\'@{}/\\[]()=+-_<>!?')
    
    @validates('type')
    def validate_type(self, key: str, value: str) -> str:
        """Validate pronunciation type."""
        if not value:
            raise ValueError("Pronunciation type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Pronunciation type must be a string")
        value = value.strip().lower()
        if len(value) > 50:
            raise ValueError("Pronunciation type cannot exceed 50 characters")
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid pronunciation type. Must be one of: {', '.join(self.VALID_TYPES.keys())}")
        self._is_modified = True
        return value
    
    @validates('value')
    def validate_value(self, key: str, value: str) -> str:
        """Validate pronunciation value."""
        if not value:
            raise ValueError("Pronunciation value cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Pronunciation value must be a string")
        value = value.strip()
        
        # Validate based on type
        if hasattr(self, 'type') and self.type == 'ipa':
            # Basic IPA validation - could be more comprehensive
            if not all(c in self.IPA_CHARS for c in value if c.isalpha()):
                raise ValueError("Invalid IPA characters in pronunciation value")
        elif hasattr(self, 'type') and self.type == 'x-sampa':
            # Basic X-SAMPA validation
            if not all(c in self.X_SAMPA_CHARS for c in value):
                raise ValueError("Invalid X-SAMPA characters in pronunciation value")
        elif hasattr(self, 'type') and self.type == 'audio':
            # Audio file reference validation
            if not value.endswith(('.mp3', '.wav', '.ogg')):
                raise ValueError("Audio file must be mp3, wav, or ogg format")
        
        self._is_modified = True
        return value
    
    @validates('data_metadata')
    def validate_json_field(self, key: str, value: Any) -> Dict:
        """Validate JSON fields."""
        if value is not None:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"{key} must be valid JSON")
            elif not isinstance(value, (dict, list)):
                raise ValueError(f"{key} must be a dict or list")
            self._is_modified = True
        return value
    
    def __repr__(self) -> str:
        return f'<Pronunciation {self.id}: {self.type} for word {self.word_id}>'
    
    def invalidate_cache(self) -> None:
        """Invalidate pronunciation cache."""
        # Define pattern for cache invalidation based on this model
        pattern = f"*pronunciation*{self.id}*"
        from backend.database import invalidate_cache
        invalidate_cache(pattern)
        
        # Also invalidate word-related caches
        pattern = f"*word*{self.word_id}*"
        invalidate_cache(pattern)
    
    @cached_query(timeout=3600, key_prefix="pronunciation_dict")
    def to_dict(self) -> Dict[str, Any]:
        """Convert pronunciation to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'type': self.type,
            'value': self.value,
            'tags': self.tags or {},
            'is_primary': self.is_primary,
            'source_info': self.source_info or {},
            'metadata': self.data_metadata or {},  # Use renamed column
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_by_value")
    def get_by_value(cls, value: str, type: str = 'ipa') -> List['Pronunciation']:
        """Get pronunciations by value and type."""
        return cls.query.filter(
            and_(
                func.lower(cls.value) == value.strip().lower(),
                func.lower(cls.type) == type.strip().lower()
            )
        ).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_by_word")
    def get_by_word(cls, word_id: int) -> List['Pronunciation']:
        """Get all pronunciations for a word."""
        return cls.query.filter_by(word_id=word_id).order_by(cls.type).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_similar")
    def find_similar(cls, value: str, type: str = 'ipa', min_similarity: float = 0.6) -> List['Pronunciation']:
        """Find similar pronunciations using trigram similarity."""
        return cls.query.filter(
            and_(
                func.similarity(cls.value, value) > min_similarity,
                cls.type == type
            )
        ).order_by(
            func.similarity(cls.value, value).desc()
        ).limit(10).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_by_tag")
    def find_by_tag(cls, tag: str) -> List['Pronunciation']:
        """Find pronunciations by tag."""
        return cls.query.filter(
            text("tags ? :tag")
        ).params(tag=tag).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_stats")
    def get_stats(cls, language_code: Optional[str] = None) -> Dict[str, Any]:
        """Get pronunciation statistics."""
        from .word import Word
        
        # Base query joining with words table
        query = db.session.query(
            cls.type,
            func.count(cls.id).label('count')
        ).join(Word, cls.word_id == Word.id)
        
        # Filter by language code if provided
        if language_code:
            query = query.filter(Word.language_code == language_code)
        
        # Group by type and sort by count
        result = query.group_by(cls.type).order_by(func.count(cls.id).desc()).all()
        
        # Format as dictionary
        return {
            'types': {r[0]: r[1] for r in result},
            'total': sum(r[1] for r in result),
            'language': language_code or 'all'
        } 