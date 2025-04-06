"""
Basic columns mixin for models.
Provides minimal common columns used across multiple models without verification fields.
"""

from backend.database import db
from datetime import datetime

class BasicColumnsMixin:
    """Mixin to add only essential columns to models."""
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False) 