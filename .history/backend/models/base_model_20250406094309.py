"""
Base model with common validation methods and improved caching.
"""

from backend.database import db, cached_query
from datetime import datetime
from sqlalchemy.orm import validates, attributes
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.inspection import inspect
from sqlalchemy.types import TypeDecorator, JSON
import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import logging
import structlog
from functools import wraps

logger = structlog.get_logger(__name__)

class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            try:
                value = json.dumps(value)
            except (TypeError, ValueError) as e:
                logger.error(f"JSON serialization error: {e}", value=str(value)[:100])
                raise ValueError(f"Could not serialize JSON: {e}")
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"JSON deserialization error: {e}", value=str(value)[:100])
                return {}  # Return empty dict instead of failing
        return value

class BaseModel(db.Model):
    """Base model with common validation methods."""
    __abstract__ = True
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Track if model has been modified
    _is_modified = False
    
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name automatically."""
        return cls.__name__.lower()
    
    @validates('created_at', 'updated_at')
    def validate_datetime(self, key: str, value: Any) -> datetime:
        """Validate datetime fields."""
        if value is not None and not isinstance(value, datetime):
            raise ValueError(f"{key} must be a datetime object")
        return value
    
    @validates('args', 'pronunciation_data', 'word_metadata', 'hyphenation')
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
    
    @validates('credit', 'etymology_text', 'form', 'template_name')
    def validate_text(self, key: str, value: str) -> str:
        """Validate text fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            # Remove any null bytes that could cause database issues
            value = value.replace('\x00', '')
            # Track modification
            self._is_modified = True
        return value
    
    @validates('language_code')
    def validate_language_code(self, key: str, value: str) -> str:
        """Validate language code."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            value = value.strip().lower()
            if not re.match(r'^[a-z]{2,5}$', value):
                raise ValueError(f"{key} must be a valid language code (2-5 lowercase letters)")
            # Track modification  
            self._is_modified = True
        return value
    
    @validates('data_hash')
    def validate_hash(self, key: str, value: str) -> str:
        """Validate hash fields."""
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if not re.match(r'^[a-f0-9]{32}$', value):
                raise ValueError(f"{key} must be a valid MD5 hash")
        return value
    
    def validate_foreign_keys(self) -> None:
        """Validate foreign key relationships."""
        for relationship in inspect(self.__class__).relationships:
            if relationship.backref is None:
                related_obj = getattr(self, relationship.key)
                if related_obj is not None:
                    self.validate_relationship(relationship.key, related_obj)
    
    def validate_unique_constraints(self) -> None:
        """Validate unique constraints."""
        for constraint in inspect(self.__class__).table.constraints:
            if isinstance(constraint, db.UniqueConstraint):
                values = [getattr(self, col.name) for col in constraint.columns]
                if all(v is not None for v in values):
                    existing = self.__class__.query.filter_by(**{
                        col.name: value for col, value in zip(constraint.columns, values)
                    }).first()
                    if existing and existing.id != self.id:
                        raise ValueError(f"Unique constraint violation: {constraint.name}")
    
    def validate(self) -> None:
        """Validate model data."""
        self.validate_foreign_keys()
        self.validate_unique_constraints()
    
    def before_save(self) -> None:
        """Hook called before saving."""
        self.validate()
    
    def after_save(self) -> None:
        """Hook called after saving."""
        # Invalidate cache if model was modified
        if self._is_modified:
            self.invalidate_cache()
        self._is_modified = False
    
    def save(self) -> None:
        """Save the model to the database."""
        try:
            self.before_save()
            db.session.add(self)
            db.session.commit()
            self.after_save()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error saving {self.__class__.__name__}", error=str(e), id=getattr(self, 'id', None))
            raise
    
    def delete(self) -> None:
        """Delete the model from the database."""
        try:
            self.invalidate_cache()
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting {self.__class__.__name__}", error=str(e), id=getattr(self, 'id', None))
            raise
    
    def invalidate_cache(self) -> None:
        """Invalidate model cache."""
        # Implement cache invalidation logic specific to each model
        pass
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="model_by_id")
    def get_by_id(cls, id: int) -> Optional['BaseModel']:
        """Get model by ID with caching."""
        return cls.query.get(id)
    
    @classmethod
    def get_or_404(cls, id: int) -> 'BaseModel':
        """Get model by ID or raise 404."""
        model = cls.get_by_id(id)
        if model is None:
            raise ValueError(f"{cls.__name__} with id {id} not found")
        return model
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary."""
        result = {}
        for column in inspect(self.__class__).columns:
            value = getattr(self, column.name)
            # Handle datetime serialization
            if isinstance(value, datetime):
                value = value.isoformat()
            result[column.name] = value
        return result
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__} {self.id}>"
    
    @cached_query(timeout=1800, key_prefix="related_objects")
    def get_related(self, relationship_name: str, filters: Optional[Dict] = None) -> List['BaseModel']:
        """Get related objects with optional filters."""
        if not hasattr(self, relationship_name):
            raise ValueError(f"No relationship named {relationship_name}")
        
        relationship = getattr(self.__class__, relationship_name)
        query = relationship.property.mapper.class_.query
        
        if filters:
            query = query.filter_by(**filters)
        
        return query.filter(relationship.any()).all()
    
    def add_related(self, relationship_name: str, related_obj: 'BaseModel') -> None:
        """Add a related object."""
        if not hasattr(self, relationship_name):
            raise ValueError(f"No relationship named {relationship_name}")
        
        self.validate_relationship(relationship_name, related_obj)
        
        try:
            relationship = getattr(self, relationship_name)
            if relationship is None:
                relationship = []
            relationship.append(related_obj)
            self._is_modified = True
            db.session.commit()
            self.after_save()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error adding related {relationship_name}", error=str(e), id=self.id)
            raise
    
    def remove_related(self, relationship_name: str, related_obj: Optional['BaseModel'] = None) -> None:
        """Remove a related object."""
        if not hasattr(self, relationship_name):
            raise ValueError(f"No relationship named {relationship_name}")
        
        try:
            relationship = getattr(self, relationship_name)
            if relationship is None:
                return
            
            if related_obj is None:
                relationship.clear()
            else:
                relationship.remove(related_obj)
            
            self._is_modified = True
            db.session.commit()
            self.after_save()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error removing related {relationship_name}", error=str(e), id=self.id)
            raise
    
    def validate_relationship(self, relationship_name: str, related_obj: 'BaseModel') -> None:
        """Validate a relationship."""
        if not hasattr(self, relationship_name):
            raise ValueError(f"No relationship named {relationship_name}")
        
        relationship = getattr(self.__class__, relationship_name)
        if not isinstance(related_obj, relationship.property.mapper.class_):
            raise ValueError(
                f"Invalid type for {relationship_name}. "
                f"Expected {relationship.property.mapper.class_.__name__}, "
                f"got {related_obj.__class__.__name__}"
            )
    
    @cached_query(timeout=1800, key_prefix="relationship_graph")
    def get_relationship_graph(self, max_depth: int = 2) -> Dict:
        """Get a graph of related objects."""
        def build_graph(obj: 'BaseModel', depth: int, visited: Set[int]) -> Dict:
            if depth <= 0 or obj.id in visited:
                return {}
            
            visited.add(obj.id)
            result = {'id': obj.id, 'type': obj.__class__.__name__}
            
            for relationship in inspect(obj.__class__).relationships:
                if relationship.backref is None:
                    related = getattr(obj, relationship.key)
                    if related is not None:
                        if hasattr(related, '__iter__') and not isinstance(related, (str, bytes, dict)):
                            # Handle collections
                            result[relationship.key] = [
                                build_graph(item, depth - 1, visited.copy()) 
                                for item in related
                                if item.id not in visited
                            ]
                        else:
                            # Handle single objects
                            result[relationship.key] = build_graph(related, depth - 1, visited.copy())
            
            return result
        
        return build_graph(self, max_depth, set()) 