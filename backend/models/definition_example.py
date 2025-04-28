from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base_model import BaseModel
import datetime
from sqlalchemy.dialects.postgresql import JSONB

class DefinitionExample(BaseModel):
    __tablename__ = 'definition_examples'

    # Primary key
    id = Column(Integer, primary_key=True)

    # Foreign key to definitions table
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False, index=True)

    # Core example data
    example_text = Column(Text, nullable=False)
    translation = Column(Text, nullable=True)
    example_type = Column(String(50), nullable=True) # e.g., 'quotation', 'proverb', 'usage'
    reference = Column(Text, nullable=True) # Source/reference for the example

    # Additional structured data - Renamed column and type corrected
    example_metadata = Column(JSONB, nullable=True)

    # Provenance and Timestamps
    sources = Column(Text, nullable=True) # Comma-separated list of source identifiers
    # created_at and updated_at likely handled by BaseModel

    # Relationships (back reference to the definition)
    definition = relationship("Definition", back_populates="examples")

    # Constraints
    __table_args__ = (
        UniqueConstraint('definition_id', 'example_text', name='definition_examples_unique'),
        # Add other constraints if needed
    )

    def __repr__(self):
        return f"<DefinitionExample(id={self.id}, definition_id={self.definition_id}, text='{self.example_text[:30]}...')>" 