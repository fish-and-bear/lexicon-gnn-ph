"""
Definition Relation model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class DefinitionRelation(db.Model):
    """Definition Relation model representing relationships between definitions and words."""
    __tablename__ = 'definition_relations'

    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id'), nullable=False)
    related_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    relation_type = db.Column(db.String(50), nullable=False)
    sources = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    related_word = db.relationship('Word', backref='definition_relations') 