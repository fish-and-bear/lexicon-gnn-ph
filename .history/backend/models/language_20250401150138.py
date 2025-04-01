"""
Language model for the Filipino Dictionary.
"""

from database import db
from .base_model import BaseModel

class Language(BaseModel):
    """Language model representing different languages referenced in the dictionary."""
    
    __tablename__ = 'languages'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Language code (ISO 639-3 preferred)
    code = db.Column(db.String(10), unique=True, nullable=False, index=True)
    
    # Language names
    name_en = db.Column(db.String(50), nullable=False)
    name_tl = db.Column(db.String(50))
    
    # Optional regional information
    region = db.Column(db.String(50))
    
    # Language family
    family = db.Column(db.String(50))
    
    # Language status (e.g., "living", "extinct", "historical")
    status = db.Column(db.String(20))
    
    # Define the relationship with Word table explicitly
    # instead of using the backref in Word model
    words = db.relationship('Word', 
                           primaryjoin="Language.code == foreign(Word.language_code)",
                           viewonly=True,
                           lazy='dynamic')
    
    def __repr__(self):
        return f"<Language {self.code} ({self.name_en})>" 