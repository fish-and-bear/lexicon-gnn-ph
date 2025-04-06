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
from .mixins.basic_columns import BasicColumnsMixin
from sqlalchemy.dialects.postgresql import JSONB

class Definition(BaseModel, BasicColumnsMixin):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.Text)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id', ondelete='SET NULL'), index=True)
    usage_notes = db.Column(db.Text)
    examples = db.Column(db.JSON, default=lambda: [])
    tags = db.Column(db.Text)
    sources = db.Column(db.Text, nullable=True)
    definition_metadata = db.Column(db.JSON, default=lambda: {})
    popularity_score = db.Column(db.Float, default=0.0)
    
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
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word', 'word_id'),
        db.Index('idx_definitions_pos', 'standardized_pos_id'),
        db.Index('idx_definitions_text_trgm', 'definition_text', postgresql_using='gin', postgresql_ops={'definition_text': 'gin_trgm_ops'}),
        db.Index('idx_definitions_examples', 'examples', postgresql_using='gin'),
        db.Index('idx_definitions_tags', 'tags', postgresql_using='gin', postgresql_ops={'tags': 'gin_trgm_ops'}),
        db.Index('idx_definitions_popularity', 'popularity_score')
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
    
    @validates('original_pos')
    def validate_original_pos(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate original part of speech."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Original part of speech must be a string")
            value = value.strip()
            if not value:
                return None
        self._is_modified = True
        return value
    
    @validates('usage_notes', 'sources', 'tags')
    def validate_other_text_fields(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate other text fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            value = value.strip()
        self._is_modified = True
        return value
    
    @validates('examples', 'definition_metadata')
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
    
    @validates('popularity_score')
    def validate_popularity_score(self, key: str, value: Optional[float]) -> float:
        """Validate popularity score."""
        if value is None:
            return 0.0
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError("Popularity score must be a number")
        if value < 0:
            value = 0.0
        elif value > 100.0:
            value = 100.0
        self._is_modified = True
        return value
    
    def __repr__(self) -> str:
        pos_name = self.standardized_pos.name_en if self.standardized_pos else "None"
        return f'<Definition {self.id}: {self.definition_text[:50]}...> ({pos_name})'
    
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
            'definition_text': self.definition_text,
            'original_pos': self.original_pos,
            'standardized_pos_id': self.standardized_pos_id,
            'part_of_speech': self.standardized_pos.to_dict() if self.standardized_pos else None,
            'usage_notes': self.usage_notes,
            'examples': self.examples or [],
            'tags': self.tags.split(',') if self.tags else [],
            'sources': self.sources.split(', ') if self.sources else [],
            'metadata': self.definition_metadata or {},
            'popularity_score': self.popularity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add related categories
        if hasattr(self, 'categories') and self.categories:
            result['categories'] = [c.to_dict() for c in self.categories]
            
        # Add related links
        if hasattr(self, 'links') and self.links:
            result['links'] = [l.to_dict() for l in self.links]
            
        # Add related definition relations
        if hasattr(self, 'definition_relations') and self.definition_relations:
            result['definition_relations'] = [r.to_dict() for r in self.definition_relations]
            
        # Add related words
        if hasattr(self, 'related_words') and self.related_words:
            result['related_words'] = [{'id': w.id, 'lemma': w.lemma, 'language_code': w.language_code} for w in self.related_words]
        
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