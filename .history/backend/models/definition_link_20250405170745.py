"""
Definition link model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from typing import Dict, Any, Optional
import re

class DefinitionLink(BaseModel, StandardColumnsMixin):
    """Model for definition links."""
    __tablename__ = 'definition_links'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    link_text = db.Column(db.Text, nullable=False)
    link_target = db.Column(db.Text, nullable=False)
    is_wikipedia = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    definition = db.relationship('Definition', back_populates='links', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'link_text', 'link_target', name='definition_links_unique'),
        db.Index('idx_def_links_def', 'definition_id'),
        db.Index('idx_def_links_text_trgm', 'link_text', postgresql_using='gin', postgresql_ops={'link_text': 'gin_trgm_ops'}),
        db.Index('idx_def_links_target', 'link_target')
    )
    
    @validates('link_text')
    def validate_link_text(self, key, value):
        """Validate link text."""
        if not value:
            raise ValueError("Link text cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Link text must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Link text cannot be empty after stripping")
        return value
    
    @validates('link_target')
    def validate_link_target(self, key, value):
        """Validate link target."""
        if not value:
            raise ValueError("Link target cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Link target must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Link target cannot be empty after stripping")
        
        # Validate URL format if it's a URL
        if value.startswith(('http://', 'https://')):
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            if not url_pattern.match(value):
                raise ValueError("Invalid URL format")
        
        return value
    
    @validates('is_wikipedia')
    def validate_is_wikipedia(self, key, value):
        """Validate is_wikipedia flag."""
        if value is None:
            return False
        if not isinstance(value, bool):
            raise ValueError("is_wikipedia must be a boolean")
        if value and not self.link_target.startswith(('https://wikipedia.org/', 'https://www.wikipedia.org/')):
            raise ValueError("Wikipedia links must point to wikipedia.org")
        return value
    
    def __repr__(self):
        return f'<DefinitionLink {self.id}: {self.link_text} -> {self.link_target} for def {self.definition_id}>'
    
    def to_dict(self):
        """Convert definition link to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'link_text': self.link_text,
            'link_target': self.link_target,
            'is_wikipedia': self.is_wikipedia,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_definition(cls, definition_id: int) -> list:
        """Get all links for a definition."""
        return cls.query.filter_by(definition_id=definition_id).all()
    
    @classmethod
    def get_wikipedia_links(cls) -> list:
        """Get all Wikipedia links."""
        return cls.query.filter_by(is_wikipedia=True).all()
    
    def validate_link(self) -> bool:
        """Validate if the link is accessible."""
        if self.link_target.startswith(('http://', 'https://')):
            try:
                import requests
                response = requests.head(self.link_target, timeout=5)
                return response.status_code == 200
            except:
                return False
        return True 