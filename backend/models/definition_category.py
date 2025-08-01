"""
Definition category model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any, List, Optional
from sqlalchemy.dialects.postgresql import JSONB
import json

class DefinitionCategory(BaseModel, BasicColumnsMixin):
    """Model for definition categories."""
    __tablename__ = 'definition_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    category_name = db.Column(db.Text, nullable=False)
    category_kind = db.Column(db.Text, nullable=True)
    parents = db.Column(JSONB, default=lambda: [])
    sources = db.Column(db.Text, nullable=True)
    category_metadata = db.Column(JSONB, default=lambda: {})
    
    # Relationships
    definition = db.relationship('Definition', back_populates='categories', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'category_name', name='definition_categories_unique'),
        db.Index('idx_def_categories_def', 'definition_id'),
        db.Index('idx_def_categories_name', 'category_name'),
        db.Index('idx_def_categories_kind', 'category_kind'),
        db.Index('idx_def_categories_parents', 'parents', postgresql_using='gin')
    )
    
    VALID_KINDS = {
        'semantic': 'Semantic field or domain',
        'usage': 'Usage context or register',
        'dialect': 'Dialectal or regional category',
        'grammar': 'Grammatical category',
        'topic': 'Topic or subject area',
        'topical': 'Topic or subject area',
        'register': 'Language register',
        'style': 'Style or formality level',
        'etymology': 'Etymology-based category',
        'custom': 'Custom category',
        'lifeform': 'Biological organism classification',
        'other': 'Other category type'
        # All values should be considered valid
    }
    
    @validates('category_name')
    def validate_category_name(self, key: str, value: str) -> str:
        """Validate category name."""
        if not value:
            raise ValueError("Category name cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Category name must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Category name cannot be empty after stripping")
        return value
    
    @validates('category_kind')
    def validate_category_kind(self, key, value):
        """Validate category kind."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Category kind must be a string")
        value = value.strip().lower()
        # Allow any value, removing the validation against VALID_KINDS
        # if value and value not in self.VALID_KINDS:
        #     raise ValueError(f"Invalid category kind. Must be one of: {', '.join(self.VALID_KINDS.keys())}")
        return value
    
    @validates('parents')
    def validate_parents(self, key, value):
        """Validate parents JSON."""
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for parents")
        if not isinstance(value, list):
            raise ValueError("Parents must be a list")
        return value
    
    def __repr__(self) -> str:
        return f'<DefinitionCategory {self.id}: {self.category_name} ({self.category_kind}) for def {self.definition_id}>'
    
    def to_dict(self):
        """Convert definition category to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'category_name': self.category_name,
            'category_kind': self.category_kind,
            'parents': self.parents or [],
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_name(cls, category_name: str) -> list:
        """Get categories by name."""
        return cls.query.filter_by(category_name=category_name.strip()).all()
    
    @classmethod
    def get_by_kind(cls, category_kind: str) -> list:
        """Get categories by kind."""
        return cls.query.filter_by(category_kind=category_kind.strip().lower()).all()
    
    @classmethod
    def get_by_definition(cls, definition_id: int) -> list:
        """Get all categories for a definition."""
        return cls.query.filter_by(definition_id=definition_id).all()
    
    def get_child_categories(self) -> list:
        """Get all categories that have this category as a parent."""
        return DefinitionCategory.query.filter(
            DefinitionCategory.parents.contains([self.category_name])
        ).all()
    
    def get_full_hierarchy(self) -> dict:
        """Get the full category hierarchy."""
        hierarchy = {
            'name': self.category_name,
            'kind': self.category_kind,
            'parents': self.parents or [],
            'children': []
        }
        
        # Add child categories recursively
        for child in self.get_child_categories():
            hierarchy['children'].append(child.get_full_hierarchy())
        
        return hierarchy 