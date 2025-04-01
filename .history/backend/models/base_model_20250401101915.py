"""
Base model with common validation methods.
"""

from database import db
from datetime import datetime
from sqlalchemy.orm import validates
import json
import re

class BaseModel(db.Model):
    """Base model with common validation methods."""
    __abstract__ = True
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @validates('created_at', 'updated_at')
    def validate_datetime(self, key, value):
        """Validate datetime fields."""
        if value is not None and not isinstance(value, datetime):
            raise ValueError(f"{key} must be a datetime object")
        return value
    
    @validates('tags', 'examples', 'pronunciation_data', 'source_info', 'word_metadata', 'hyphenation')
    def validate_json(self, key, value):
        """Validate JSON fields."""
        if value is not None:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"{key} must be valid JSON")
            elif not isinstance(value, (dict, list)):
                raise ValueError(f"{key} must be a dict or list")
        return value
    
    @validates('sources', 'credit', 'etymology_text', 'definition_text')
    def validate_string(self, key, value):
        """Validate string fields."""
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value
    
    @validates('language_code')
    def validate_language_code(self, key, value):
        """Validate language code."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if not re.match(r'^[a-z]{2,3}$', value):
                raise ValueError(f"{key} must be a valid ISO 639-1 or 639-2 language code")
        return value
    
    @validates('data_hash')
    def validate_hash(self, key, value):
        """Validate hash fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if not re.match(r'^[a-f0-9]{64}$', value):
                raise ValueError(f"{key} must be a valid SHA-256 hash")
        return value
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if not column.name.startswith('_')
        }
    
    def __repr__(self):
        """String representation of the model."""
        return f'<{self.__class__.__name__} {self.id}>' 