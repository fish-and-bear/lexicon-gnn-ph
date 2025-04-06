"""
Etymology model definition with enhanced functionality and performance.
"""

from backend.database import db, cached_query
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
from sqlalchemy import func, and_, or_, text
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
import json

class Etymology(BaseModel, StandardColumnsMixin):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    etymology_text = db.Column(db.Text, nullable=False)
    normalized_components = db.Column(db.Text)
    etymology_structure = db.Column(db.Text)
    language_codes = db.Column(db.Text)
    source_language = db.Column(db.String(100))
    sources = db.Column(db.Text, nullable=True)
    data_metadata = db.Column(db.JSON, default=lambda: {})
    
    # Relationships
    word = db.relationship('Word', back_populates='etymologies', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'etymology_text', name='etymologies_unique'),
        db.Index('idx_etymologies_word', 'word_id'),
        # Add GIN index for language_codes full-text search
        db.Index('idx_etymologies_langs', db.text("to_tsvector('simple', language_codes)"), postgresql_using='gin'),
        # Add trigram index for etymology_text
        db.Index('idx_etymologies_text_trgm', 'etymology_text', postgresql_using='gin', 
                postgresql_ops={'etymology_text': 'gin_trgm_ops'}),
        # Add index for normalized components
        db.Index('idx_etymologies_components', 'normalized_components', postgresql_using='gin',
                postgresql_ops={'normalized_components': 'gin_trgm_ops'}),
        db.Index('idx_etymologies_metadata', 'data_metadata', postgresql_using='gin')
    )
    
    @validates('sources')
    def validate_text_field(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate text fields."""
        if not value:
            raise ValueError("Text field cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Text field must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Text field cannot be empty after stripping")
        self._is_modified = True
        return value

    @validates('language_codes')
    def validate_language_codes(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate language codes."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Language codes must be a string")
            # Format as comma-separated list of lowercase language codes
            codes = [code.strip().lower() for code in value.split(',') if code.strip()]
            value = ','.join(codes)
            self._is_modified = True
        return value
    
    def __repr__(self) -> str:
        return f'<Etymology {self.id} for word {self.word_id}: {self.etymology_text[:50]}...>'
    
    def invalidate_cache(self) -> None:
        """Invalidate etymology cache."""
        # Define pattern for cache invalidation based on this model
        pattern = f"*etymology*{self.id}*"
        from backend.database import invalidate_cache
        invalidate_cache(pattern)
        
        # Also invalidate word-related caches
        pattern = f"*word*{self.word_id}*"
        invalidate_cache(pattern)
    
    @cached_query(timeout=3600, key_prefix="etymology_dict")
    def to_dict(self) -> Dict[str, Any]:
        """Convert etymology to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'normalized_components': self.normalized_components,
            'etymology_structure': self.etymology_structure,
            'language_codes': self.language_codes.split(',') if self.language_codes else [],
            'source_language': self.source_language,
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': self.data_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="etymology_word")
    def get_by_word(cls, word_id: int) -> List['Etymology']:
        """Get all etymologies for a word."""
        return cls.query.filter_by(word_id=word_id).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="etymology_language")
    def get_by_language(cls, language_code: str) -> List['Etymology']:
        """Get all etymologies containing a specific language."""
        return cls.query.filter(
            cls.language_codes.ilike(f'%{language_code.lower()}%')
        ).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="etymology_similar")
    def find_similar(cls, text: str, min_similarity: float = 0.3) -> List['Etymology']:
        """Find etymologies with similar text."""
        return cls.query.filter(
            func.similarity(cls.etymology_text, text) > min_similarity
        ).order_by(
            func.similarity(cls.etymology_text, text).desc()
        ).limit(10).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="etymology_search")
    def search(cls, query: str) -> List['Etymology']:
        """Search etymologies by text."""
        return cls.query.filter(
            or_(
                cls.etymology_text.ilike(f'%{query}%'),
                cls.normalized_components.ilike(f'%{query}%')
            )
        ).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="etymology_stats")
    def get_language_stats(cls) -> Dict[str, int]:
        """Get statistics on languages in etymologies."""
        result = db.session.query(
            text("unnest(string_to_array(language_codes, ',')) as lang"),
            func.count('*').label('count')
        ).filter(
            cls.language_codes.isnot(None)
        ).group_by(
            text("unnest(string_to_array(language_codes, ','))")
        ).order_by(
            func.count('*').desc()
        ).all()
        
        return {r[0]: r[1] for r in result}
    
    @classmethod
    def parse_etymology(cls, text: str) -> Dict[str, Any]:
        """
        Parse etymology text into structured data.
        
        This is a helper method to extract language codes and structure from raw etymology text.
        Returns a dictionary with parsed components.
        """
        # Extract language codes using basic pattern matching
        import re
        
        # Find language codes in text (assumed to be in parentheses or brackets)
        language_match = re.findall(r'\(([a-z]{2,5})\)|\[([a-z]{2,5})\]', text.lower())
        language_codes = []
        
        for match in language_match:
            # Each match is a tuple with one non-empty group
            code = next((code for code in match if code), None)
            if code:
                language_codes.append(code)
        
        # Extract likely word components
        components = re.findall(r'(?:from|via|cf\.)?\s+([\'"][^\'",;]+[\'"]|[^\s,;]+)', text)
        components = [c.strip('\'".,;()[]') for c in components if len(c) > 2]
        
        return {
            'language_codes': language_codes,
            'components': components,
            'structure': {
                'has_from': 'from' in text.lower(),
                'has_via': 'via' in text.lower(),
                'has_cognate': 'cognate' in text.lower() or 'cf.' in text.lower(),
                'has_compound': 'compound' in text.lower() or '+' in text
            }
        } 