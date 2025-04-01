"""
Pronunciation model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Pronunciation(db.Model):
    """Pronunciation model representing word pronunciations."""
    __tablename__ = 'pronunciations'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    value = db.Column(db.String(255), nullable=False)
    tags = db.Column(db.JSON)
    pronunciation_metadata = db.Column(db.JSON)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 