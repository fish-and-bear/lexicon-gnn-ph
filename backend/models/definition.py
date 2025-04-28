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
from flask import current_app
import logging

logger = logging.getLogger(__name__)

class Definition(BaseModel, BasicColumnsMixin):
    """Model for word definitions."""
    __tablename__ = 'definitions'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.Text)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id', ondelete='SET NULL'), index=True)
    usage_notes = db.Column(db.Text)
    tags = db.Column(db.Text)
    sources = db.Column(db.Text, nullable=True)
    
    # For backward compatibility, not expecting this column in the actual DB
    try:
        definition_metadata = Column(JSONB, nullable=True, comment="Flexible JSON field for additional, unstructured metadata specific to this definition.")
    except Exception as e:
        logger.warning(f"Could not define definition_metadata column: {e}. Using property instead.")
        # We'll define a property to handle this below
    
    # Optimized relationships with proper cascade rules
    word = db.relationship('Word', 
                         back_populates='definitions', 
                         lazy='selectin')
    
    # Explicitly reference the PartOfSpeech model
    from .part_of_speech import PartOfSpeech
    standardized_pos = db.relationship('PartOfSpeech', 
                                     back_populates='definitions', 
                                     lazy='joined',
                                     foreign_keys=[standardized_pos_id])
    
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
                          cascade='all, delete-orphan',
                          query_class=db.Query,
                          # Explicitly specify which columns to load to avoid missing columns
                          primaryjoin="Definition.id == DefinitionLink.definition_id",
                          foreign_keys="DefinitionLink.definition_id")
    
    # One-to-many relationship to DefinitionExample (NEW)
    examples = db.relationship('DefinitionExample', 
                             back_populates='definition',
                             lazy='selectin',
                             cascade='all, delete-orphan',
                             order_by='DefinitionExample.id',
                             collection_class=list)  # Explicitly specify collection class
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word', 'word_id'),
        db.Index('idx_definitions_pos', 'standardized_pos_id'),
        db.Index('idx_definitions_text_trgm', 'definition_text', postgresql_using='gin', postgresql_ops={'definition_text': 'gin_trgm_ops'}),
        db.Index('idx_definitions_tags', 'tags', postgresql_using='gin', postgresql_ops={'tags': 'gin_trgm_ops'}),
    )
    
    # Dictionary to store metadata when the column doesn't exist
    _metadata_dict = {}
    
    def __init__(self, **kwargs):
        """Initialize definition with default values."""
        super().__init__(**kwargs)
        
        # Initialize examples to an empty list if None to prevent collection errors
        if not hasattr(self, 'examples') or self.examples is None:
            self.examples = []
        
        # Initialize metadata dict if needed
        if not hasattr(self, '_metadata_cache_key'):
            self._metadata_cache_key = f"def_meta_{self.id}" if hasattr(self, 'id') and self.id else None
            
        # Handle definition_metadata based on whether the column exists
        try:
            # If the column exists in DB, this will work
            if not hasattr(self, 'definition_metadata') or self.definition_metadata is None:
                self.definition_metadata = {}
        except Exception:
            # If accessing the column fails, use our property implementation
            self._definition_metadata = {}
    
    # Property to handle definition_metadata when column doesn't exist
    @property
    def _definition_metadata(self):
        """Get definition metadata from cache or default."""
        if hasattr(self, 'id') and self.id:
            # Try to get from class cache
            if not hasattr(self.__class__, '_metadata_dict'):
                self.__class__._metadata_dict = {}
            
            cache_key = f"def_meta_{self.id}"
            if cache_key in self.__class__._metadata_dict:
                return self.__class__._metadata_dict[cache_key]
        return {}
        
    @_definition_metadata.setter
    def _definition_metadata(self, value):
        """Store definition metadata in class cache."""
        if hasattr(self, 'id') and self.id:
            cache_key = f"def_meta_{self.id}"
            if not hasattr(self.__class__, '_metadata_dict'):
                self.__class__._metadata_dict = {}
            self.__class__._metadata_dict[cache_key] = value if value is not None else {}
    
    # Access definition_metadata through property if column doesn't exist
    def __getattr__(self, name):
        if name == 'definition_metadata':
            try:
                # First try to access as a real column
                return object.__getattribute__(self, name)
            except (AttributeError, Exception):
                # If that fails, use our property implementation
                return self._definition_metadata
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
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
    
    @validates('usage_notes', 'sources')
    def validate_other_text_fields(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate other text fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            value = value.strip()
        self._is_modified = True
        return value
    
    @validates('tags')
    def validate_tags(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate tags field."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Tags must be a string")
            value = value.strip()
        self._is_modified = True
        return value
    
    @validates('definition_metadata')
    def validate_metadata(self, key: str, value: Any) -> Any:
        """Validate metadata field."""
        if value is not None:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"{key} must be valid JSON")
            # Ensure it's a dictionary after potential loading
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a dict or valid JSON string representing a dict")
            self._is_modified = True
            
            try:
                # Try to set as a real column
                return value
            except Exception:
                # If that fails, use our property
                self._definition_metadata = value
                return value
        
        # Default to empty dict
        try:
            return {}
        except Exception:
            self._definition_metadata = {}
            return {}
    
    @property
    def popularity_score(self):
        """Return the popularity score from metadata or default."""
        # Try to get from metadata first
        try:
            meta = self.definition_metadata
            if meta and 'popularity_score' in meta:
                try:
                    return float(meta['popularity_score'])
                except (ValueError, TypeError):
                    return 0.0
        except Exception:
            # If accessing definition_metadata fails, try our property
            meta = self._definition_metadata
            if meta and 'popularity_score' in meta:
                try:
                    return float(meta['popularity_score'])
                except (ValueError, TypeError):
                    return 0.0
        
        # Otherwise return default
        return 0.0

    @popularity_score.setter
    def popularity_score(self, value):
        """Set the popularity score in metadata."""
        # Make sure metadata exists and is a dict
        try:
            # Try to access as a real column
            if not hasattr(self, 'definition_metadata') or not isinstance(self.definition_metadata, dict):
                self.definition_metadata = {}
            
            try:
                self.definition_metadata['popularity_score'] = float(value)
            except (ValueError, TypeError):
                self.definition_metadata['popularity_score'] = 0.0
        except Exception:
            # If that fails, use our property implementation
            meta = self._definition_metadata or {}
            try:
                meta['popularity_score'] = float(value)
            except (ValueError, TypeError):
                meta['popularity_score'] = 0.0
            self._definition_metadata = meta
    
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
        # Use the hybrid property directly
        popularity = self.popularity_score

        # Get metadata safely
        try:
            metadata = self.definition_metadata or {}
        except Exception:
            metadata = self._definition_metadata or {}

        result = {
            'id': self.id,
            'word_id': self.word_id,
            'definition_text': self.definition_text,
            'original_pos': self.original_pos,
            'standardized_pos_id': self.standardized_pos_id,
            'part_of_speech': self.standardized_pos.to_dict() if self.standardized_pos else None,
            'usage_notes': self.usage_notes,
            'examples': [ex.to_dict() for ex in self.examples] if hasattr(self, 'examples') and self.examples else [],
            'tags': self.tags,
            'sources': self.sources,
            'definition_metadata': metadata,
            'popularity_score': popularity,
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
        ).order_by(cls.created_at.desc()).all()
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="definition_pos")
    def get_by_pos(cls, pos_id: int) -> List['Definition']:
        """Get definitions by part of speech."""
        return cls.query.filter_by(standardized_pos_id=pos_id).all()
    
    def increment_popularity(self, amount: float = 0.1) -> None:
        """Increment the popularity score of this definition."""
        # Get current score from the hybrid property
        current_score = self.popularity_score
        
        # Calculate new score
        new_score = current_score + amount
        if new_score > 100.0:
            new_score = 100.0
        
        # Set through the hybrid property
        self.popularity_score = new_score
        
        self._is_modified = True
        self.save()
    
    def has_examples(self) -> bool:
        """Check if definition has examples."""
        return bool(self.examples and len(self.examples) > 0)
    
    @classmethod
    def get_most_popular_definitions(cls, limit: int = 10) -> List['Definition']:
        """Get most popular definitions."""
        # Use a safer approach by ordering by created_at instead
        return cls.query.order_by(cls.created_at.desc()).limit(limit).all()

    @property
    def definition_metadata(self):
        # Fallback for missing column in database
        if hasattr(self, '_definition_metadata'):
            return self._definition_metadata
        return {}
    
    @definition_metadata.setter
    def definition_metadata(self, value):
        self._definition_metadata = value or {} 

    # Add a dedicated setter for examples to ensure it's always a list
    @property
    def all_examples(self):
        """Ensure examples always returns a list."""
        if not hasattr(self, 'examples') or self.examples is None:
            self.examples = []
        return self.examples 