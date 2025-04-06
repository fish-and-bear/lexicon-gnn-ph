"""
Standard columns mixin for models.
Provides common columns used across multiple models.
"""

from backend.database import db
from datetime import datetime
from sqlalchemy.ext.declarative import declared_attr

class StandardColumnsMixin:
    """Mixin to add standard columns to models."""
    
    @declared_attr
    def created_at(cls):
        return db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    @declared_attr
    def updated_at(cls):
        return db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    @declared_attr
    def sources(cls):
        return db.Column(db.Text, nullable=True)
    
    @declared_attr
    def metadata(cls):
        return db.Column(db.JSON, default=lambda: {}, nullable=True)
    
    @declared_attr
    def verified(cls):
        return db.Column(db.Boolean, default=False, nullable=False)
    
    @declared_attr
    def verification_date(cls):
        return db.Column(db.DateTime, nullable=True)
    
    @declared_attr
    def verification_user_id(cls):
        return db.Column(db.Integer, nullable=True) 