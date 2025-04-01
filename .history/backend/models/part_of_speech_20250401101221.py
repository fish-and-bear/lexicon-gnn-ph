"""
Part of speech model definition.
"""

from database import db
from datetime import datetime

class PartOfSpeech(db.Model):
    """Model for parts of speech."""
    __tablename__ = 'parts_of_speech'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False)
    name_en = db.Column(db.String(50), nullable=False)
    name_tl = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    definitions = db.relationship('Definition', backref='standardized_pos', lazy='dynamic')
    
    def __repr__(self):
        return f'<PartOfSpeech {self.code}: {self.name_en}>' 