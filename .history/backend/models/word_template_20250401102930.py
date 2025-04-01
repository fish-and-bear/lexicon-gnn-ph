"""
Word template model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

class WordTemplate(BaseModel):
    """Model for word templates."""
    __tablename__ = 'word_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'))
    template_name = db.Column(db.Text, nullable=False)
    args = db.Column(db.JSON)  # JSONB in PostgreSQL
    expansion = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    word = db.relationship('Word', backref=db.backref('templates', lazy='dynamic', cascade='all, delete-orphan'))
    
    __table_args__ = (
        db.UniqueConstraint('word_id', 'template_name', name='word_templates_word_id_template_name_key'),
    )
    
    def __repr__(self):
        return f'<WordTemplate {self.id}: {self.template_name}>'
    
    def to_dict(self):
        """Convert word template to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'template_name': self.template_name,
            'args': self.args,
            'expansion': self.expansion,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 