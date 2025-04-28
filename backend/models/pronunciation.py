"""
Pronunciation model definition with optimized caching and performance.
"""

from backend.database import db, cached_query
from datetime import datetime
from sqlalchemy.orm import validates
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import func, and_, or_, Index, text
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
import json
from typing import List, Dict, Any, Optional, Set, Union
from sqlalchemy.dialects.postgresql import JSONB
import re
import logging

logger = logging.getLogger(__name__)

class Pronunciation(BaseModel, BasicColumnsMixin):
    """Model for word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    type = db.Column(db.String(50), nullable=False)  # E.g., 'IPA', 'syllabification', 'rhyme'
    value = db.Column(db.Text, nullable=False)
    pronunciation_metadata = db.Column(JSONB, default=lambda: {})
    # sources = db.Column(db.Text, nullable=True)  # This column doesn't exist in the database schema
    
    # Optimized relationship with Word
    word = db.relationship('Word', back_populates='pronunciations', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'type', 'value', name='uq_pronunciation_word_type_value'),
        # Added missing standard indexes
        db.Index('idx_pronunciations_word', 'word_id'),
        db.Index('idx_pronunciations_type', 'type'),
        db.Index('idx_pronunciations_value_trgm', 'value', postgresql_using='gin', postgresql_ops={'value': 'gin_trgm_ops'}),
    )
    
    VALID_TYPES = {
        'ipa': 'International Phonetic Alphabet',
        'x-sampa': 'Extended SAMPA',
        'pinyin': 'Hanyu Pinyin',
        'jyutping': 'Jyutping',
        'romaji': 'Romaji',
        'audio': 'Audio file reference',
        'respelling': 'Respelling',
        'phonemic': 'Phonemic transcription',
        'rhyme': 'Rhyme'
    }
    
    # Extended IPA character set
    IPA_CHARS = set('ˈˌːəɪʊeɔæaɒʌɜɛɨʉɯɪʏʊøɘɵɤəɚɛœɜɞʌɔɑɒæɐɪ̯ʏ̯ʊ̯e̯ø̯ə̯ɚ̯ɛ̯œ̯ɜ̯ɞ̯ʌ̯ɔ̯ɑ̯ɒ̯æ̯ɐ̯ˈˌ./')
    X_SAMPA_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"\'@{}/\\[]()=+-_<>!?')
    
    # Add more IPA characters for better compatibility
    EXTENDED_IPA_CHARS = set('ˈˌːəɪʊeɔæaɒʌɜɛɨʉɯɪʏʊøɘɵɤəɚɛœɜɞʌɔɑɒæɐɪ̯ʏ̯ʊ̯e̯ø̯ə̯ɚ̯ɛ̯œ̯ɜ̯ɞ̯ʌ̯ɔ̯ɑ̯ɒ̯æ̯ɐ̯ˈˌ./' + 
                          'pbtdkɡqɢʔszʃʒxɣχʁħʕhfvθðmnŋɴɲɳrlɬɮʎʟjwɥɰiɯuɪʏʊeøɘɵɤəɚɛœɜɞʌɔɑɒæɐãẽĩõũĩńɲ̃ɲ̃ãẽĩõũẽ' +
                          'ʰʱʲʷˠˤ̟̠̘̙̰̤̥̬̹̜̩̯̪̺̻̼̝̞̃̈̽̚ᵊ̋́̄̀̏̂̌ꜜꜛ↓↑→↗↘')
    
    # Store tags in memory
    _tags_dict = {}
    _sources_dict = {}  # Add a dictionary to store sources in memory
    
    # Define tags as a property to handle it without a database column
    @property
    def tags(self):
        """Get tags from memory dictionary."""
        if not hasattr(self, '_tags'):
            # Initialize from class dictionary if available
            if hasattr(self, 'id') and self.id in self.__class__._tags_dict:
                self._tags = self.__class__._tags_dict[self.id]
            else:
                self._tags = {}
        return self._tags
    
    @tags.setter
    def tags(self, value):
        """Store tags in memory dictionary."""
        if value is None:
            value = {}
        elif isinstance(value, str):
            try:
                value = json.loads(value) if value else {}
            except:
                value = {}
        
        if not isinstance(value, dict):
            value = {}
            
        self._tags = value
        # Store in class dictionary for persistence
        if hasattr(self, 'id') and self.id:
            self.__class__._tags_dict[self.id] = value
    
    # Define sources as a property to handle missing sources column
    @property
    def sources(self):
        """Get sources from memory dictionary."""
        if not hasattr(self, '_sources'):
            # Initialize from class dictionary if available
            if hasattr(self, 'id') and self.id in self.__class__._sources_dict:
                self._sources = self.__class__._sources_dict[self.id]
            else:
                # Try to get from pronunciation_metadata
                if hasattr(self, 'pronunciation_metadata') and self.pronunciation_metadata:
                    if isinstance(self.pronunciation_metadata, dict) and 'sources' in self.pronunciation_metadata:
                        self._sources = self.pronunciation_metadata['sources']
                    else:
                        self._sources = None
                else:
                    self._sources = None
        return self._sources
    
    @sources.setter
    def sources(self, value):
        """Store sources in memory dictionary."""
        self._sources = value
        # Store in class dictionary for persistence
        if hasattr(self, 'id') and self.id:
            self.__class__._sources_dict[self.id] = value
            # Also store in pronunciation_metadata for persistence
            if hasattr(self, 'pronunciation_metadata'):
                if self.pronunciation_metadata is None:
                    self.pronunciation_metadata = {}
                if isinstance(self.pronunciation_metadata, dict):
                    self.pronunciation_metadata['sources'] = value
    
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
        
        # More permissive type validation - warning instead of error
        if value not in self.VALID_TYPES:
            logger.warning(f"Non-standard pronunciation type: {value}. Expected one of: {', '.join(self.VALID_TYPES.keys())}")
        
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
        
        # Validate based on type - more permissive validation with warnings instead of errors
        if hasattr(self, 'type') and self.type == 'ipa':
            # Relaxed IPA validation - allow more characters and handle problematic cases
            # Most IPA validation failures are due to the character set being too restrictive
            invalid_chars = []
            
            for c in value:
                # Skip non-alphabetic characters like spaces, numbers, etc.
                if not c.isalpha():
                    continue
                    
                # Skip ASCII chars as they could be respellings or annotations
                if c.isascii():
                    continue
                    
                # Check if it's in our extended set
                if c not in self.EXTENDED_IPA_CHARS:
                    invalid_chars.append(c)
            
            if invalid_chars:
                logger.warning(f"Potentially invalid IPA characters in pronunciation: {''.join(invalid_chars)} (value: {value})")
                # We don't raise error so data can still be stored
        
        elif hasattr(self, 'type') and self.type == 'x-sampa':
            # Basic X-SAMPA validation - also relaxed to avoid breaking data load
            invalid_chars = [c for c in value if c not in self.X_SAMPA_CHARS and not c.isspace()]
            if invalid_chars and len(invalid_chars) > len(value) * 0.2:  # Allow up to 20% "invalid" chars
                logger.warning(f"Potentially invalid X-SAMPA characters detected: {''.join(invalid_chars)} (value: {value})")
        
        elif hasattr(self, 'type') and self.type == 'audio':
            # Audio file reference validation - also relaxed
            if not (value.endswith(('.mp3', '.wav', '.ogg', '.m4a')) or '://' in value or 'sound' in value.lower()):
                logger.warning(f"Audio file format not recognized: {value}")
        
        self._is_modified = True
        return value
    
    @validates('pronunciation_metadata')
    def validate_json_field(self, key: str, value: Any) -> Dict:
        """Validate JSON fields."""
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON format for {key}")
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be a dictionary")
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
        result = {
            'id': self.id,
            'word_id': self.word_id,
            'type': self.type,
            'value': self.value,
            'tags': self.tags or {},
            'pronunciation_metadata': self.pronunciation_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add word if it's loaded
        if hasattr(self, 'word') and self.word:
            result['word'] = {
                'id': self.word.id,
                'lemma': self.word.lemma,
                'language_code': getattr(self.word, 'language_code', 'tl')
            }
        
        return result
    
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