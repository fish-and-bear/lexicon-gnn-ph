"""
Word model definition.
"""

from database import db
from datetime import datetime
import json
from .base_model import BaseModel

class Word(BaseModel):
    """Model for dictionary words."""
    __tablename__ = 'words'
    
    id = db.Column(db.Integer, primary_key=True)
    lemma = db.Column(db.String(255), nullable=False)
    normalized_lemma = db.Column(db.String(255), nullable=False, index=True)
    language_code = db.Column(db.String(16), nullable=False, default='tl', index=True)
    has_baybayin = db.Column(db.Boolean, default=False)
    baybayin_form = db.Column(db.String(255))
    romanized_form = db.Column(db.String(255))
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='SET NULL'), index=True)
    preferred_spelling = db.Column(db.String(255))
    tags = db.Column(db.Text)  # Changed from JSON to Text to match schema
    idioms = db.Column(db.JSON)
    pronunciation_data = db.Column(db.JSON)
    source_info = db.Column(db.Text)  # Changed from JSON to Text to match schema
    word_metadata = db.Column(db.JSON)
    data_hash = db.Column(db.Text)  # Changed from String to Text to match schema
    search_text = db.Column(db.Text)
    badlit_form = db.Column(db.Text)
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
    
    __table_args__ = (
        db.UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
        db.CheckConstraint(
            "(has_baybayin = false AND baybayin_form IS NULL) OR (has_baybayin = true AND baybayin_form IS NOT NULL)",
            name='baybayin_form_check'
        ),
        db.CheckConstraint(
            "baybayin_form ~ '^[\u1700-\u171F\s]*$' OR baybayin_form IS NULL",
            name='baybayin_form_regex'
        )
    )
    
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
            'tags': json.loads(self.tags) if self.tags else None,
            'idioms': self.idioms,
            'pronunciation_data': self.pronunciation_data,
            'source_info': json.loads(self.source_info) if self.source_info else None,
            'word_metadata': self.word_metadata,
            'data_hash': self.data_hash,
            'search_text': self.search_text,
            'badlit_form': self.badlit_form,
            'hyphenation': self.hyphenation,
            'is_proper_noun': self.is_proper_noun,
            'is_abbreviation': self.is_abbreviation,
            'is_initialism': self.is_initialism,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 