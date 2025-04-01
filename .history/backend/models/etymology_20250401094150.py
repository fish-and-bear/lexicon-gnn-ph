"""
Etymology model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Etymology(db.Model):
    """Etymology model representing word etymologies."""
    __tablename__ = 'etymologies'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    etymology_text = db.Column(db.Text)
    normalized_components = db.Column(db.Text)
    etymology_structure = db.Column(db.Text)
    language_codes = db.Column(db.Text)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 