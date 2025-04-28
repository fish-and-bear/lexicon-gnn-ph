from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship, validates
from backend.database import db
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class DefinitionExample(db.Model):
    """Model for definition examples."""
    __tablename__ = 'definition_examples'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Foreign key to definitions table
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Core example data
    example_text = Column(Text, nullable=False)
    translation = Column(Text)
    example_type = Column(String(50), default='example')
    reference = Column(Text)
    
    # This column doesn't exist in the DB schema
    # example_metadata = Column(db.JSON)
    
    # Provenance and Timestamps
    sources = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships (back reference to the definition)
    definition = relationship("Definition", back_populates="examples", lazy="selectin")
    
    # Storage for example_metadata
    _metadata_dict = {}
    
    def __init__(self, **kwargs):
        """Initialize with default values."""
        super().__init__(**kwargs)
        self._example_metadata_value = {}
    
    @property
    def example_metadata(self):
        """Property for missing example_metadata column."""
        if hasattr(self, '_example_metadata_value'):
            return self._example_metadata_value
        
        # Try to get from class dictionary if available
        if hasattr(self, 'id') and self.id in self.__class__._metadata_dict:
            return self.__class__._metadata_dict[self.id]
            
        return {}
    
    @example_metadata.setter
    def example_metadata(self, value):
        """Store metadata in instance variable."""
        if value is None:
            value = {}
        elif isinstance(value, str):
            try:
                value = json.loads(value)
            except:
                value = {}
                
        self._example_metadata_value = value
        
        # Store in class dictionary for persistence
        if hasattr(self, 'id') and self.id:
            self.__class__._metadata_dict[self.id] = value
    
    # Use a different name to avoid collision with SQLAlchemy's metadata
    @property
    def example_data(self):
        """Alias for example_metadata."""
        return self.example_metadata
    
    @example_data.setter
    def example_data(self, value):
        """Alias setter for example_metadata."""
        self.example_metadata = value
    
    @validates('example_text')
    def validate_text(self, key, value):
        """Validate example text."""
        if not value or not isinstance(value, str):
            raise ValueError("Example text must be a non-empty string")
        return value.strip()
    
    def __repr__(self):
        """String representation."""
        return f"<Example {self.id}: {self.example_text[:30]}...>"
    
    def to_dict(self):
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'definition_id': self.definition_id,
            'example_text': self.example_text,
            'translation': self.translation,
            'reference': self.reference,
            'example_type': self.example_type,
            'example_metadata': self.example_metadata,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Extract romanization from metadata if available
        if self.example_metadata and 'romanization' in self.example_metadata:
            result['romanization'] = self.example_metadata['romanization']
            
        return result 