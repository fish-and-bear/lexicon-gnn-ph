"""
Definition model for word definitions with improved caching and performance.
"""

from database import db, cached_query
from datetime import datetime
import json
from typing import Dict, Any, List, Optional, Set, Union
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON, UniqueConstraint, Index, func, and_, or_, text
from sqlalchemy.orm import validates
from .base_model import BaseModel

class Definition(BaseModel):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.String, nullable=False)
    original_pos = db.Column(db.String(64))
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'), index=True)
    examples = db.Column(db.JSON)  # Changed from String to JSON for better handling
    usage_notes = db.Column(db.String)
    tags = db.Column(db.JSON)  # Already JSON type 
    sources = db.Column(db.String, nullable=False)
    popularity_score = db.Column(db.Float, default=0.0)  # New field to track definition popularity 
    
    # Optimized relationships with proper cascade rules
    word = db.relationship('Word', 
                         back_populates='definitions', 
                         lazy='joined')
    standardized_pos = db.relationship('PartOfSpeech', 
                                     back_populates='definitions', 
                                     lazy='joined')
    
    # Definition relation relationships with proper cascade rules
    definition_relations = db.relationship('DefinitionRelation', 
                                         back_populates='definition', 
                                         lazy='selectin',  # Changed from dynamic to selectin for better performance 
                                         cascade='all, delete-orphan',
                                         overlaps="related_words")
    
    # Many-to-many relationship with Word through DefinitionRelation
    related_words = db.relationship('Word', 
                                  secondary='definition_relations',
                                  primaryjoin='Definition.id == DefinitionRelation.definition_id',
                                  secondaryjoin='DefinitionRelation.word_id == Word.id',
                                  overlaps="definition_relations,word.definition_relations,word.related_definitions",
                                  lazy='selectin',  # Changed from dynamic to selectin
                                  viewonly=True)
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word_pos', 'word_id', 'standardized_pos_id'),
        db.Index('idx_definitions_text', 'definition_text', postgresql_using='gin', postgresql_ops={'definition_text': 'gin_trgm_ops'}),
        db.Index('idx_definitions_popularity', 'popularity_score', postgresql_using='btree')  # New index for popularity
    )
    
    @validates('definition_text')
    def validate_definition_text(self, key: str, value: str) -> str:
        """Validate definition text."""
        if not value:
            raise ValueError("Definition text cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Definition text must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Definition text cannot be empty after stripping")
        self._is_modified = True
        return value
    
    @validates('examples', 'tags')
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
        return f'<Definition {self.id}: {self.definition_text[:50]}...>'
    
    def invalidate_cache(self) -> None:
        """Invalidate definition cache."""
        # Define pattern for cache invalidation based on this model
        pattern = f"*definition*{self.id}*"
        from database import invalidate_cache
        invalidate_cache(pattern)
        
        # Also invalidate word-related caches
        pattern = f"*word*{self.word_id}*"
        invalidate_cache(pattern)
    
    @cached_query(timeout=3600, key_prefix="definition_dict")
    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'definition_text': self.definition_text,
            'original_pos': self.original_pos,
            'standardized_pos_id': self.standardized_pos_id,
            'examples': self.examples if self.examples else [],  # Changed from json.loads to direct access
            'usage_notes': self.usage_notes,
            'tags': self.tags if self.tags else [],  # Changed from json.loads to direct access
            'sources': self.sources,
            'popularity_score': self.popularity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'pos': self.standardized_pos.to_dict() if self.standardized_pos else None
        }
    
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
            cls.definition_text.ilike(f'%{query}%')
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