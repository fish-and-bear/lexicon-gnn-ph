"""
Word model for the Filipino Dictionary API.
"""

from . import db
from datetime import datetime
import json

class Word(db.Model):
    """Word model representing dictionary entries."""
    __tablename__ = 'words'

    id = db.Column(db.Integer, primary_key=True)
    lemma = db.Column(db.String(255), nullable=False)
    normalized_lemma = db.Column(db.String(255), nullable=False, index=True)
    language_code = db.Column(db.String(10), nullable=False, default='tl', index=True)
    has_baybayin = db.Column(db.Boolean, default=False)
    baybayin_form = db.Column(db.String(255))
    romanized_form = db.Column(db.String(255))
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='SET NULL'), index=True)
    preferred_spelling = db.Column(db.String(255))
    tags = db.Column(db.JSON)
    idioms = db.Column(db.JSON)
    pronunciation_data = db.Column(db.JSON)
    source_info = db.Column(db.JSON)
    word_metadata = db.Column(db.JSON)
    data_hash = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    search_text = db.Column(db.Text)
    badlit_form = db.Column(db.String(255))
    hyphenation = db.Column(db.JSON)
    is_proper_noun = db.Column(db.Boolean, default=False)
    is_abbreviation = db.Column(db.Boolean, default=False)
    is_initialism = db.Column(db.Boolean, default=False)
    quality_score = db.Column(db.Float, default=0.0)

    # Relationships with cascade rules
    definitions = db.relationship('Definition', backref='word', lazy='dynamic', cascade='all, delete-orphan')
    etymologies = db.relationship('Etymology', backref='word', lazy='dynamic', cascade='all, delete-orphan')
    pronunciations = db.relationship('Pronunciation', backref='word', lazy='dynamic', cascade='all, delete-orphan')
    credits = db.relationship('Credit', backref='word', lazy='dynamic', cascade='all, delete-orphan')
    root_word = db.relationship('Word', remote_side=[id], backref='derived_words', cascade='save-update')
    outgoing_relations = db.relationship('Relation', foreign_keys='Relation.from_word_id', backref='source_word', lazy='dynamic', cascade='all, delete-orphan')
    incoming_relations = db.relationship('Relation', foreign_keys='Relation.to_word_id', backref='target_word', lazy='dynamic', cascade='all, delete-orphan')
    root_affixations = db.relationship('Affixation', foreign_keys='Affixation.root_word_id', backref='root_word', lazy='dynamic', cascade='all, delete-orphan')
    affixed_affixations = db.relationship('Affixation', foreign_keys='Affixation.affixed_word_id', backref='affixed_word', lazy='dynamic', cascade='all, delete-orphan')

    @property
    def is_root(self):
        """Check if this is a root word."""
        return self.root_word_id is None

    def __repr__(self):
        return f'<Word {self.id}: {self.lemma}>'

    def to_dict(self):
        """Convert word to dictionary."""
        return {
            'id': self.id,
            'lemma': self.lemma,
            'normalized_lemma': self.normalized_lemma,
            'language_code': self.language_code,
            'has_baybayin': self.has_baybayin,
            'baybayin_form': self.baybayin_form,
            'romanized_form': self.romanized_form,
            'root_word_id': self.root_word_id,
            'preferred_spelling': self.preferred_spelling,
            'tags': self.tags,
            'idioms': self.idioms,
            'pronunciation_data': self.pronunciation_data,
            'source_info': self.source_info,
            'word_metadata': self.word_metadata,
            'data_hash': self.data_hash,
            'search_text': self.search_text,
            'badlit_form': self.badlit_form,
            'hyphenation': self.hyphenation,
            'is_proper_noun': self.is_proper_noun,
            'is_abbreviation': self.is_abbreviation,
            'is_initialism': self.is_initialism,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 