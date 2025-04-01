"""
Relation model definition.
"""

from database import db
from datetime import datetime
from sqlalchemy.orm import validates, synonym
from sqlalchemy.ext.hybrid import hybrid_property
from .base_model import BaseModel
import json
from enum import Enum

class RelationType(Enum):
    """Enumeration of valid relation types."""
    SYNONYM = 'synonym'
    ANTONYM = 'antonym'
    HYPERNYM = 'hypernym'
    HYPONYM = 'hyponym'
    MERONYM = 'meronym'
    HOLONYM = 'holonym'
    DERIVED = 'derived'
    ROOT = 'root'
    VARIANT = 'variant'
    RELATED = 'related'

class Relation(BaseModel):
    """Model for word relationships."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    from_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    to_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False, index=True)
    relation_type = db.Column(db.String(64), nullable=False, index=True)
    sources = db.Column(db.Text)
    _metadata = db.Column('metadata', db.JSON, default={})  # JSONB in PostgreSQL, using private name
    
    # Create a synonym that maps to the metadata column
    relation_metadata = synonym('_metadata', descriptor=property(
        lambda self: self._metadata,
        lambda self, value: setattr(self, '_metadata', value)
    ))
    
    # Relationships
    source_word = db.relationship('Word', 
                                foreign_keys=[from_word_id],
                                back_populates='outgoing_relations')
    target_word = db.relationship('Word',
                                foreign_keys=[to_word_id],
                                back_populates='incoming_relations')
    
    __table_args__ = (
        db.UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
        db.Index('idx_relations_metadata', 'metadata', postgresql_using='gin'),
        db.Index('idx_relations_metadata_strength', db.text("(metadata->>'strength')")),
    )
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if not value:
            raise ValueError("Relation type cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Relation type must be a string")
        value = value.strip().lower()
        if len(value) > 64:
            raise ValueError("Relation type cannot exceed 64 characters")
        try:
            RelationType(value)
        except ValueError:
            raise ValueError(f"Invalid relation type. Must be one of: {', '.join(t.value for t in RelationType)}")
        return value
    
    @validates('_metadata')
    def validate_metadata(self, key, value):
        """Validate metadata."""
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Metadata must be valid JSON")
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a dictionary")
        
        # Validate strength if present
        if 'strength' in value:
            try:
                strength = float(value['strength'])
                if not 0 <= strength <= 1:
                    raise ValueError("Strength must be between 0 and 1")
                value['strength'] = strength
            except (TypeError, ValueError):
                raise ValueError("Strength must be a number between 0 and 1")
        
        return value
    
    def __repr__(self):
        return f'<Relation {self.id}: {self.from_word_id}-{self.relation_type}->{self.to_word_id}>'
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'from_word_id': self.from_word_id,
            'to_word_id': self.to_word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'metadata': self.relation_metadata,  # Keep the API response using 'metadata'
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_by_words(cls, from_word_id: int, to_word_id: int) -> list:
        """Get all relations between two words."""
        return cls.query.filter_by(
            from_word_id=from_word_id,
            to_word_id=to_word_id
        ).all()
    
    @classmethod
    def get_by_type(cls, relation_type: str) -> list:
        """Get all relations of a specific type."""
        return cls.query.filter_by(
            relation_type=relation_type.strip().lower()
        ).all()
    
    def get_inverse_type(self) -> str:
        """Get the inverse relation type."""
        inverse_map = {
            RelationType.HYPERNYM.value: RelationType.HYPONYM.value,
            RelationType.HYPONYM.value: RelationType.HYPERNYM.value,
            RelationType.MERONYM.value: RelationType.HOLONYM.value,
            RelationType.HOLONYM.value: RelationType.MERONYM.value,
            RelationType.DERIVED.value: RelationType.ROOT.value,
            RelationType.ROOT.value: RelationType.DERIVED.value
        }
        return inverse_map.get(self.relation_type, self.relation_type)
    
    def create_inverse(self) -> 'Relation':
        """Create an inverse relation."""
        return Relation(
            from_word_id=self.to_word_id,
            to_word_id=self.from_word_id,
            relation_type=self.get_inverse_type(),
            sources=self.sources,
            relation_metadata=self.relation_metadata
        ) 