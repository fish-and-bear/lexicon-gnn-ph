"""
Definition model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Definition(db.Model):
    """Definition model representing word definitions."""
    __tablename__ = 'definitions'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    definition_text = db.Column(db.Text, nullable=False)
    original_pos = db.Column(db.String(50))
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'))
    examples = db.Column(db.JSON)
    usage_notes = db.Column(db.Text)
    tags = db.Column(db.JSON)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    standardized_pos = db.relationship('PartOfSpeech', backref='definitions')
    definition_relations = db.relationship('DefinitionRelation', backref='definition', lazy='dynamic') 