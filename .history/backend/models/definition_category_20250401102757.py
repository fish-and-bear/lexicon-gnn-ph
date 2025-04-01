"""
Definition category model definition.
"""

from database import db
from datetime import datetime
from .base_model import BaseModel

class DefinitionCategory(BaseModel):
    """Model for definition categories."""
    __tablename__ = 'definition_categories'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'))
    category_name = db.Column(db.Text, nullable=False)
    category_kind = db.Column(db.Text)
    parents = db.Column(db.JSON)  # JSONB in PostgreSQL
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    definition = db.relationship('Definition', backref=db.backref('categories', lazy='dynamic', cascade='all, delete-orphan'))
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'category_name', name='definition_categories_definition_id_category_name_key'),
    )
    
    def __repr__(self):
        return f'<DefinitionCategory {self.id}: {self.category_name}>'
    
    def to_dict(self):
        """Convert category to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'category_name': self.category_name,
            'category_kind': self.category_kind,
            'parents': self.parents,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 