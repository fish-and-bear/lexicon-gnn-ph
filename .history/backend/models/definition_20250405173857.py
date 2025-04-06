"""
Definition model for word definitions with improved caching and performance.
"""

from backend.database import db, cached_query
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Set, Union
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, UniqueConstraint, Index, func, and_, or_, text
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from sqlalchemy.dialects.postgresql import JSONB

class Definition(BaseModel, StandardColumnsMixin):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    text = db.Column(db.Text, nullable=False)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id', ondelete='SET NULL'), index=True)
    usage_notes = db.Column(db.Text)
    examples = db.Column(db.JSON, default=lambda: [])
    sources = db.Column(db.Text, nullable=True)
    data_metadata = db.Column(JSONB, default=lambda: {})
    
    # Optimized relationships with proper cascade rules
    word = db.relationship('Word', 
                         back_populates='definitions', 
                         lazy='selectin')
    standardized_pos = db.relationship('PartOfSpeech', 
                                     back_populates='definitions', 
                                     lazy='joined')
    
    # Definition relation relationships with proper cascade rules
    definition_relations = db.relationship('DefinitionRelation', 
                                         back_populates='definition', 
                                         lazy='selectin',
                                         cascade='all, delete-orphan',
                                         overlaps="related_words,word.definition_relations")
    
    # Add relationship for DefinitionCategory
    categories = db.relationship('DefinitionCategory', 
                               back_populates='definition', 
                               lazy='selectin',
                               cascade='all, delete-orphan')
    
    # Many-to-many relationship with Word through DefinitionRelation
    related_words = db.relationship('Word', 
                                  secondary='definition_relations',
                                  primaryjoin='Definition.id == DefinitionRelation.definition_id',
                                  secondaryjoin='DefinitionRelation.word_id == Word.id',
                                  overlaps="definition_relations,word.definition_relations,word.related_definitions",
                                  lazy='selectin',  # Changed from dynamic to selectin
                                  viewonly=True)
    
    # Add relationship for DefinitionLink
    links = db.relationship('DefinitionLink', 
                          back_populates='definition', 
                          lazy='selectin',
                          cascade='all, delete-orphan')
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word', 'word_id'),
        db.Index('idx_definitions_pos', 'standardized_pos_id'),
        db.Index('idx_definitions_text_trgm', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}),
        db.Index('idx_definitions_examples', 'examples', postgresql_using='gin')
    )
    
    @validates('text', 'usage_notes', 'sources')
    def validate_text_field(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate text fields."""
        if not value:
            raise ValueError("Text cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Text must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Text cannot be empty after stripping")
        self._is_modified = True
        return value
    
    @validates('examples')
    def validate_json_field(self, key: str, value: Any) -> Any:
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
        pos_name = self.standardized_pos.name_en if self.standardized_pos else "None"
        return f'<Definition {self.id}: {self.text[:50]}...> ({pos_name})'
    
    def invalidate_cache(self) -> None:
        """Invalidate definition-related caches."""
        super().invalidate_cache() # Call base model invalidation
        patterns = [
            f"*definition*{self.id}*",
            f"*word_definitions*{self.word_id}*"
        ]
        # Invalidate related word caches if applicable
        if self.word:
            self.word.invalidate_cache()
        
        from backend.database import invalidate_cache
        for pattern in patterns:
            invalidate_cache(pattern)
        
        # Also invalidate word-related caches
        pattern = f"*word*{self.word_id}*"
        invalidate_cache(pattern)
    
    @cached_query(timeout=3600, key_prefix="definition_dict")
    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to dictionary."""
        result = {
            'id': self.id,
            'word_id': self.word_id,
            'text': self.text,
            'part_of_speech': self.standardized_pos.to_dict() if self.standardized_pos else None,
            'usage_notes': self.usage_notes,
            'examples': self.examples or [],
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': self.data_metadata or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add related categories
        if hasattr(self, 'categories') and self.categories:
            result['categories'] = [c.to_dict() for c in self.categories]
            
        # Add related links
        if hasattr(self, 'links') and self.links:
            result['links'] = [l.to_dict() for l in self.links]
        
        return result
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="definition_word")
    def get_by_word_id(cls, word_id: int) -> List['Definition']:
        """Get all definitions for a word."""
        return cls.query.filter_by(word_id=word_id).order_by(cls.standardized_pos_id).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="definition_search")
    def search(cls, query: str) -> List['Definition']:
        """Search definitions by text."""
        return cls.query.filter(
            cls.text.ilike(f'%{query}%')
        ).order_by(cls.popularity_score.desc()).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="definition_pos")
    def get_by_pos(cls, pos_id: int) -> List['Definition']:
        """Get definitions by part of speech."""
        return cls.query.filter_by(standardized_pos_id=pos_id).all()
    
    def increment_popularity(self, amount: float = 0.1) -> None:
        """Increment the popularity score of this definition."""
        self.popularity_score += amount
        if self.popularity_score > 100.0:
            self.popularity_score = 100.0
        self._is_modified = True
        self.save()
    
    def has_examples(self) -> bool:
        """Check if definition has examples."""
        return bool(self.examples and len(self.examples) > 0) 