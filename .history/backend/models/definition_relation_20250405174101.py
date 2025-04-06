"""
Definition relation model definition.
"""

from backend.database import db
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.standard_columns import StandardColumnsMixin
from typing import Dict, Any, Optional

class DefinitionRelation(BaseModel, StandardColumnsMixin):
    """Model for relationships between definitions and words."""
    __tablename__ = 'definition_relations'
    
    id = db.Column(db.Integer, primary_key=True)
    definition_id = db.Column(db.Integer, db.ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(50), nullable=False, index=True) # E.g., \'related_word\', \'antonym_definition\'
    # sources = db.Column(db.Text, nullable=False) # Changed based on schema
    sources = db.Column(db.Text, nullable=True) # Changed to nullable
    data_metadata = db.Column(db.JSON, default=lambda: {}) # Added metadata field
    
    # Optimized relationships with proper cascade rules
    definition = db.relationship('Definition', back_populates='definition_relations', lazy='selectin', overlaps="related_words,word.definition_relations")
    related_word = db.relationship('Word', back_populates='definition_relations', lazy='selectin', overlaps="related_definitions,definition.definition_relations")
    
    __table_args__ = (
        db.UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique'),
        db.Index('idx_def_relations_def', 'definition_id'), # Added missing index
        db.Index('idx_def_relations_word', 'word_id'),
        db.Index('idx_def_relations_type', 'relation_type'),
        db.Index('idx_def_relations_metadata', 'data_metadata', postgresql_using='gin') # Index for metadata
    )
    
    @validates('relation_type', 'sources')
    def validate_text_field(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate text fields."""
        if key == 'relation_type':
            if value not in ['related_word', 'antonym_definition']:
                raise ValueError("Invalid relation_type")
        return value
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'word_id': self.word_id,
            'relation_type': self.relation_type,
            'sources': self.sources.split(', ') if self.sources else [], # Adapt source splitting
            'metadata': self.data_metadata or {}, # Include metadata
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self) -> str:
        return f'<DefinitionRelation {self.id}: Def {self.definition_id} -> Word {self.word_id} ({self.relation_type})>' 