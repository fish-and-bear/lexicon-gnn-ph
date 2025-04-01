"""
Relation model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime

class Relation(db.Model):
    """Relation model representing word relationships."""
    __tablename__ = 'relations'

    id = db.Column(db.Integer, primary_key=True)
    from_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = db.Column(db.String(50), nullable=False)
    sources = db.Column(db.Text)
    relation_metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source_word = db.relationship('Word', foreign_keys=[from_word_id], backref=db.backref('outgoing_relations', lazy='dynamic', cascade='all, delete-orphan'))
    target_word = db.relationship('Word', foreign_keys=[to_word_id], backref=db.backref('incoming_relations', lazy='dynamic', cascade='all, delete-orphan')) 