"""
Etymology model definition with enhanced functionality and performance.
"""

from backend.database import db, cached_query
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Set
from sqlalchemy import func, and_, or_, text
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
import json
from sqlalchemy.dialects.postgresql import JSONB

class Etymology(BaseModel, BasicColumnsMixin):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    etymology_text = db.Column(db.Text, nullable=False)
    normalized_components = db.Column(db.Text)
    etymology_structure = db.Column(db.Text)
    language_codes = db.Column(db.Text)
    sources = db.Column(db.Text, nullable=True)
    
    # Relationships
    word = db.relationship('Word', back_populates='etymologies', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq'),
        db.Index('idx_etymologies_word', 'word_id'),
        db.Index('idx_etymologies_langs', db.text("to_tsvector('simple', language_codes)"), postgresql_using='gin'),
        db.Index('idx_etymologies_text_trgm', 'etymology_text', postgresql_using='gin', 
                postgresql_ops={'etymology_text': 'gin_trgm_ops'}),
        db.Index('idx_etymologies_components', 'normalized_components', postgresql_using='gin',
                postgresql_ops={'normalized_components': 'gin_trgm_ops'}),
        # Only add this index if the column exists
        # db.Index('idx_etymologies_metadata', 'etymology_metadata', postgresql_using='gin')
    )
    
    def __init__(self, **kwargs):
        metadata = kwargs.pop('metadata', None)
        if metadata is not None:
            # Store metadata in sources if needed
            if 'sources' not in kwargs and metadata:
                kwargs['sources'] = json.dumps(metadata)
        super().__init__(**kwargs)
    
    @validates('etymology_text')
    def validate_etymology_text(self, key: str, value: str) -> str:
        """Validate etymology text."""
        if not value:
            raise ValueError("Etymology text cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Etymology text must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Etymology text cannot be empty after stripping")
        self._is_modified = True
        return value
    
    @validates('normalized_components', 'etymology_structure')
    def validate_optional_text(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate optional text fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            value = value.strip()
            if not value:
                return None
            self._is_modified = True
        return value
    
    @validates('sources')
    def validate_sources(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate sources."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Sources must be a string")
        value = value.strip()
        if not value:
            return None
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
    
    @validates('etymology_metadata')
    def validate_etymology_metadata(self, key: str, value: Any) -> Dict:
        """Validate metadata JSON."""
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for metadata")
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a dictionary")
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
        result = {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'normalized_components': self.normalized_components,
            'etymology_structure': self.etymology_structure,
            'language_codes': self.language_codes,
            'sources': self.sources,
            'metadata': self.etymology_metadata or {},
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
            
        # Parse components if available
        if self.normalized_components:
            try:
                components = json.loads(self.normalized_components)
                if isinstance(components, list):
                    result['parsed_components'] = components
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON, leave as is
                pass
            
        return result
    
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
    
    # Add property to provide compatibility for the missing column
    @property
    def etymology_data(self):
        """Provide compatibility for the etymology_metadata column."""
        if hasattr(self, 'etymology_metadata'):
            return self.etymology_metadata
        # Fallback to empty dict or parse from sources
        if self.sources and self.sources.startswith('{'):
            try:
                return json.loads(self.sources)
            except:
                pass
        return {} 