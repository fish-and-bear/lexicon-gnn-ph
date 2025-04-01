"""
Base model with common validation methods.
"""

from database import db
from datetime import datetime
from sqlalchemy.orm import validates
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect
import json
import re
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseModel(db.Model):
    """Base model with common validation methods."""
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name automatically."""
        return cls.__name__.lower() + 's'
    
    @validates('created_at', 'updated_at')
    def validate_datetime(self, key: str, value: Any) -> datetime:
        """Validate datetime fields."""
        if value is not None and not isinstance(value, datetime):
            raise ValueError(f"{key} must be a datetime object")
        return value
    
    @validates('tags', 'args', 'parents', 'metadata', 'pronunciation_data', 'word_metadata', 'hyphenation')
    def validate_json(self, key: str, value: Any) -> Dict:
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
    
    @validates('sources', 'credit', 'etymology_text', 'definition_text', 'form', 'template_name')
    def validate_text(self, key: str, value: str) -> str:
        """Validate text fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            value = value.strip()
            if not value and self.__table__.columns[key].nullable is False:
                raise ValueError(f"{key} cannot be empty")
        return value
    
    @validates('language_code')
    def validate_language_code(self, key: str, value: str) -> str:
        """Validate language code."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if not re.match(r'^[a-z]{2,3}$', value):
                raise ValueError(f"{key} must be a valid ISO 639-1 or 639-2 language code")
        return value
    
    @validates('data_hash')
    def validate_hash(self, key: str, value: str) -> str:
        """Validate hash fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if not re.match(r'^[a-f0-9]{64}$', value):
                raise ValueError(f"{key} must be a valid SHA-256 hash")
        return value
    
    def validate_foreign_keys(self) -> None:
        """Validate all foreign key relationships."""
        for rel in inspect(self.__class__).relationships:
            if rel.direction.name != 'MANYTOONE':
                continue
            
            fk_value = getattr(self, rel.key + '_id', None)
            if fk_value is not None:
                target_cls = rel.mapper.class_
                exists = db.session.query(
                    db.session.query(target_cls).filter_by(id=fk_value).exists()
                ).scalar()
                if not exists:
                    raise ValueError(f"Invalid foreign key: {rel.key}_id={fk_value} does not exist in {target_cls.__name__}")
    
    def validate_unique_constraints(self) -> None:
        """Validate all unique constraints."""
        for constraint in self.__table__.constraints:
            if not hasattr(constraint, 'columns'):
                continue
                
            # Build filter conditions
            conditions = []
            for column in constraint.columns:
                value = getattr(self, column.name)
                if value is not None:
                    conditions.append(column == value)
            
            # Skip if any required values are None
            if len(conditions) != len(constraint.columns):
                continue
            
            # Check if any other record violates the constraint
            exists = db.session.query(
                db.session.query(self.__class__).filter(
                    *conditions,
                    self.__class__.id != self.id
                ).exists()
            ).scalar()
            
            if exists:
                raise ValueError(f"Unique constraint violation: {[c.name for c in constraint.columns]}")
    
    def validate(self) -> None:
        """Validate the model before save."""
        self.validate_foreign_keys()
        self.validate_unique_constraints()
    
    def before_save(self) -> None:
        """Hook to run before saving."""
        self.validate()
    
    def after_save(self) -> None:
        """Hook to run after saving."""
        pass
    
    def save(self) -> None:
        """Save the model with validation."""
        try:
            self.before_save()
            db.session.add(self)
            db.session.commit()
            self.after_save()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to save {self.__class__.__name__}: {str(e)}")
            raise
    
    def delete(self) -> None:
        """Delete the model."""
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to delete {self.__class__.__name__}: {str(e)}")
            raise
    
    @classmethod
    def get_by_id(cls, id: int) -> Optional['BaseModel']:
        """Get a record by ID."""
        return cls.query.get(id)
    
    @classmethod
    def get_or_404(cls, id: int) -> 'BaseModel':
        """Get a record by ID or raise 404."""
        from flask import abort
        obj = cls.get_by_id(id)
        if obj is None:
            abort(404, description=f"{cls.__name__} {id} not found")
        return obj
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
            if not column.name.startswith('_')
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f'<{self.__class__.__name__} {self.id}>' 