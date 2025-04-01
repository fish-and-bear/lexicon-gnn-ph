"""
Definition model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Definition(db.Model):
    """Definition model representing word definitions."""
    __tablename__ = 'definitions'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.String(50))
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id', ondelete='SET NULL'))
    examples = db.Column(db.JSON)
    usage_notes = db.Column(db.Text)
    tags = db.Column(db.JSON)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    standardized_pos = db.relationship('PartOfSpeech', backref=db.backref('definitions', lazy='dynamic'))
    # Use string reference to avoid circular import
    definition_relations = db.relationship('DefinitionRelation', backref=db.backref('definition', lazy='joined'), lazy='dynamic', cascade='all, delete-orphan') 