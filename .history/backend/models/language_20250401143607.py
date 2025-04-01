"""
Language model definition.
"""

from backend.database import db
from datetime import datetime
from backend.models.base_model import BaseModel

class Language(BaseModel):
    """Model for language metadata."""
    __tablename__ = 'languages'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(16), nullable=False, unique=True)
    name = db.Column(db.String(64), nullable=False)
    native_name = db.Column(db.String(64))
    family = db.Column(db.String(64))
    iso_639_1 = db.Column(db.String(2))
    iso_639_2 = db.Column(db.String(3))
    iso_639_3 = db.Column(db.String(3))
    notes = db.Column(db.Text)
    description = db.Column(db.Text)
    region = db.Column(db.String(64))
    writing_system = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Language {self.code}: {self.name}>'
    
    def to_dict(self):
        """Convert language to dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'name': self.name,
            'native_name': self.native_name,
            'family': self.family,
            'iso_639_1': self.iso_639_1,
            'iso_639_2': self.iso_639_2,
            'iso_639_3': self.iso_639_3,
            'notes': self.notes,
            'description': self.description,
            'region': self.region,
            'writing_system': self.writing_system,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 