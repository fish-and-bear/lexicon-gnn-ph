"""
Word form model definition.
"""

from database import db
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
import json

class WordForm(BaseModel):
    """Model for word forms."""
    __tablename__ = 'word_forms'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'))
    form = db.Column(db.Text, nullable=False)
    is_canonical = db.Column(db.Boolean, default=False)
    is_primary = db.Column(db.Boolean, default=False)
    tags = db.Column(db.JSON)  # JSONB in PostgreSQL
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    word = db.relationship('Word', backref=db.backref('forms', lazy='dynamic', cascade='all, delete-orphan'))
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'form', name='word_forms_word_id_form_key'),
    )
    
    VALID_TAGS = {
        'archaic': 'Historical or obsolete form',
        'dialectal': 'Regional or dialectal variant',
        'informal': 'Informal or colloquial form',
        'formal': 'Formal or literary form',
        'misspelling': 'Common misspelling',
        'alternative': 'Alternative spelling',
        'plural': 'Plural form',
        'singular': 'Singular form',
        'conjugation': 'Verb conjugation',
        'declension': 'Noun or adjective declension'
    }
    
    @validates('form')
    def validate_form(self, key, value):
        """Validate word form."""
        if not value:
            raise ValueError("Word form cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Word form must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Word form cannot be empty after stripping")
        return value
    
    @validates('tags')
    def validate_tags(self, key, value):
        """Validate tags."""
        if value is None:
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Tags must be valid JSON")
        if not isinstance(value, list):
            raise ValueError("Tags must be a list")
        
        # Validate each tag
        for tag in value:
            if not isinstance(tag, str):
                raise ValueError("Each tag must be a string")
            if tag not in self.VALID_TAGS:
                raise ValueError(f"Invalid tag: {tag}. Must be one of: {', '.join(self.VALID_TAGS.keys())}")
        
        return value
    
    def __repr__(self):
        return f'<WordForm {self.id}: {self.form}>'
    
    def to_dict(self):
        """Convert word form to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'form': self.form,
            'is_canonical': self.is_canonical,
            'is_primary': self.is_primary,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_form(cls, form: str) -> list:
        """Get word forms by form text."""
        return cls.query.filter_by(form=form.strip()).all()
    
    @classmethod
    def get_by_word(cls, word_id: int) -> list:
        """Get all forms for a word."""
        return cls.query.filter_by(word_id=word_id).all()
    
    @classmethod
    def get_canonical_forms(cls) -> list:
        """Get all canonical forms."""
        return cls.query.filter_by(is_canonical=True).all()
    
    def validate_form_uniqueness(self) -> bool:
        """Check if this form is unique for the word."""
        existing = WordForm.query.filter(
            WordForm.word_id == self.word_id,
            WordForm.form == self.form,
            WordForm.id != self.id
        ).first()
        return existing is None 