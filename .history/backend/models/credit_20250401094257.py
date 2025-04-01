"""
Credit model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Credit(db.Model):
    """Credit model representing word credits."""
    __tablename__ = 'credits'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    credit = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 