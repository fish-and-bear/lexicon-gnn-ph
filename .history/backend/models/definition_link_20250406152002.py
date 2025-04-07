"""
Definition link model definition.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.basic_columns import BasicColumnsMixin
from typing import Dict, Any, Optional
import re
import json

class DefinitionLink(BaseModel, BasicColumnsMixin):
    """Model for definition links."""
    __tablename__ = 'definition_links'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    link_text = db.Column(db.Text, nullable=False)
    # Remove columns that don't exist in the database
    # target_url = db.Column(db.Text, nullable=False)
    # display_text = db.Column(db.Text)
    # is_external = db.Column(db.Boolean, default=False)
    # tags column is likely a text field, not JSON
    tags = db.Column(db.Text, nullable=True)
    link_metadata = db.Column(db.JSON, default=lambda: {})
    
    # Add property for tags to handle JSON conversion
    @property
    def tags_dict(self):
        """Get tags as a dictionary."""
        if not self.tags:
            return {}
        if isinstance(self.tags, dict):
            return self.tags
        try:
            return json.loads(self.tags)
        except:
            return {}
            
    @tags_dict.setter
    def tags_dict(self, value):
        """Set tags from a dictionary."""
        if isinstance(value, dict):
            self.tags = json.dumps(value)
        else:
            self.tags = str(value)
    
    # Properties for backward compatibility
    @property
    def target_url(self):
        """Compatibility property for missing column."""
        # Try to get from link_metadata
        if hasattr(self, 'link_metadata') and self.link_metadata and 'url' in self.link_metadata:
            return self.link_metadata['url']
        # Try to get from tags
        tags_data = self.tags_dict
        if tags_data and 'url' in tags_data:
            return tags_data['url']
        # Return a default
        return f"#{self.link_text}" # Fallback to a fragment identifier
    
    @target_url.setter
    def target_url(self, value):
        """Store target_url in link_metadata since column doesn't exist."""
        if not self.link_metadata:
            self.link_metadata = {}
        self.link_metadata['url'] = value
    
    @property
    def display_text(self):
        """Compatibility property for missing column."""
        return self.link_metadata.get('display_text', '') if hasattr(self, 'link_metadata') and self.link_metadata else ''
        
    @property
    def is_external(self):
        """Compatibility property for missing column."""
        return self.link_metadata.get('is_external', False) if hasattr(self, 'link_metadata') and self.link_metadata else False
    
    # Property for backward compatibility
    @property
    def link_type(self):
        """Provide compatibility with old code that expects link_type."""
        return self.link_text
    
    @link_type.setter
    def link_type(self, value):
        """Set link_text when link_type is used."""
        self.link_text = value
    
    # Relationships
    definition = db.relationship('Definition', back_populates='links', lazy='selectin')
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'link_text', name='definition_links_unique'),
        db.Index('idx_def_links_def', 'definition_id'),
        db.Index('idx_def_links_type_trgm', 'link_text', postgresql_using='gin', postgresql_ops={'link_text': 'gin_trgm_ops'})
        # Removed index for target_url since column doesn't exist
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
    
    @validates('target_url')
    def validate_target_url(self, key, value):
        """Validate target URL."""
        if not value:
            raise ValueError("Target URL cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Target URL must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Target URL cannot be empty after stripping")
        
        # Validate URL format if it's an external URL
        if self.is_external and value.startswith(('http://', 'https://')):
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
    
    @validates('display_text')
    def validate_display_text(self, key, value):
        """Validate display text."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Display text must be a string")
        value = value.strip()
        if not value:
            return None
        return value
    
    @validates('is_external')
    def validate_is_external(self, key, value):
        """Validate is_external flag."""
        if value is None:
            return False
        if not isinstance(value, bool):
            raise ValueError("is_external must be a boolean")
        return value
    
    @validates('tags')
    def validate_tags(self, key, value):
        """Validate tags."""
        if value is None:
            return None
        if isinstance(value, dict):
            try:
                return json.dumps(value)
            except:
                return None
        if not isinstance(value, str):
            return str(value)
        return value
    
    @validates('link_metadata')
    def validate_link_metadata(self, key, value):
        """Validate link metadata JSON."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Link metadata must be a dictionary")
        return value
    
    def __repr__(self):
        return f'<DefinitionLink {self.id}: {self.link_text} -> {self.target_url} for def {self.definition_id}>'
    
    def to_dict(self):
        """Convert definition link to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'link_text': self.link_text,
            'target_url': self.target_url,
            'display_text': self.display_text,
            'is_external': self.is_external,
            'tags': self.tags,
            'link_metadata': self.link_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_definition(cls, definition_id: int) -> list:
        """Get all links for a definition."""
        return cls.query.filter_by(definition_id=definition_id).all()
    
    @classmethod
    def get_external_links(cls) -> list:
        """Get all external links."""
        return cls.query.filter_by(is_external=True).all()
    
    def validate_link(self) -> bool:
        """Validate if the link is accessible."""
        if self.is_external and self.target_url.startswith(('http://', 'https://')):
            try:
                import requests
                response = requests.head(self.target_url, timeout=5)
                return response.status_code == 200
            except:
                return False
        return True 