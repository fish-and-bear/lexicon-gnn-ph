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
    # Check if these columns exist - if errors occur, they'll be handled via properties
    try:
        definition_metadata = db.Column(db.JSON, default=lambda: {})
    except:
        # If the column doesn't exist in the database, we'll handle it in code
        pass
    
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
                          cascade='all, delete-orphan',
                          query_class=db.Query,
                          # Explicitly specify which columns to load to avoid missing columns
                          primaryjoin="Definition.id == DefinitionLink.definition_id",
                          foreign_keys="DefinitionLink.definition_id")
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        db.Index('idx_definitions_word', 'word_id'),
        db.Index('idx_definitions_pos', 'standardized_pos_id'),
        db.Index('idx_definitions_text_trgm', 'definition_text', postgresql_using='gin', postgresql_ops={'definition_text': 'gin_trgm_ops'}),
        db.Index('idx_definitions_examples', 'examples', postgresql_using='gin'),
        db.Index('idx_definitions_tags', 'tags', postgresql_using='gin', postgresql_ops={'tags': 'gin_trgm_ops'}),
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
    
    @validates('examples')
    def validate_examples(self, key: str, value: Any) -> Any:
        """Validate examples field."""
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
        
    # Separate validator for definition_metadata to avoid errors if column doesn't exist
    @validates('definition_metadata')
    def validate_metadata(self, key: str, value: Any) -> Any:
        """Validate definition_metadata field if it exists."""
        # Check if the attribute exists before validating
        if hasattr(self, 'definition_metadata'):
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
        return {}
    
    @property
    def popularity_score(self):
        """Return the popularity score from metadata or default."""
        # Try to get from metadata first
        if hasattr(self, 'definition_metadata') and self.definition_metadata and 'popularity_score' in self.definition_metadata:
            try:
                return float(self.definition_metadata['popularity_score'])
            except (ValueError, TypeError):
                return 0.0
        # Otherwise return default
        return 0.0

    @popularity_score.setter
    def popularity_score(self, value):
        """Set the popularity score in metadata."""
        # Make sure definition_metadata exists
        if not hasattr(self, 'definition_metadata') or not self.definition_metadata:
            self.definition_metadata = {}
        
        try:
            self.definition_metadata['popularity_score'] = float(value)
        except (ValueError, TypeError):
            self.definition_metadata['popularity_score'] = 0.0
    
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
            'metadata': getattr(self, 'definition_metadata', {}) or {},
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
    
    # Add property for definition_metadata compatibility
    @property
    def definition_metadata(self):
        """Provide compatibility for definition_metadata when the column doesn't exist."""
        # This property is used when the column doesn't exist in the database
        # If the actual column exists, SQLAlchemy will use that instead
        if hasattr(self, '_definition_metadata'):
            return self._definition_metadata
        # Fallback to empty dict or parse from sources
        if self.sources and self.sources.startswith('{'):
            try:
                return json.loads(self.sources)
            except:
                pass
        return {}
        
    @definition_metadata.setter
    def definition_metadata(self, value):
        """Set definition metadata appropriately based on what's available."""
        # This setter is only used when the column doesn't exist
        # If the column exists, SQLAlchemy will use that directly
        self._definition_metadata = value
        # Optionally serialize to sources as a fallback
        if value and isinstance(value, dict):
            if not self.sources:
                self.sources = json.dumps(value)
    
    @classmethod
    def get_most_popular_definitions(cls, limit: int = 10) -> List['Definition']:
        """Get most popular definitions."""
        # Use a safer approach by ordering by created_at instead
        return cls.query.order_by(cls.created_at.desc()).limit(limit).all() 