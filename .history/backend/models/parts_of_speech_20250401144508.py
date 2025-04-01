"""
Part of Speech model for the Filipino Dictionary API.
"""

from . import db

class PartOfSpeech(db.Model):
    """Part of Speech model representing standardized parts of speech."""
    __tablename__ = 'parts_of_speech'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False)
    name_en = db.Column(db.String(50), nullable=False)
    name_tl = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text) 