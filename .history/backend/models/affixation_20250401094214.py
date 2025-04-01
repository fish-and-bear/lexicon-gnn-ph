"""
Affixation model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Affixation(db.Model):
    """Affixation model representing word affixations."""
    __tablename__ = 'affixations'

    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    affix_type = db.Column(db.String(50), nullable=False)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow) 