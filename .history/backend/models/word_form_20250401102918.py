"""
Word form model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

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