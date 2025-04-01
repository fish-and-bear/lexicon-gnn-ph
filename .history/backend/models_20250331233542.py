"""
Filipino Dictionary Database Models - aligned with the migrated schema from dictionary_manager.py
"""

from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Boolean, DateTime, 
    func, Index, UniqueConstraint, DDL, event, text, Float, JSON
)
from sqlalchemy.orm import relationship, validates, backref
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
from unidecode import unidecode
from flask import current_app

# Set up logging
logger = logging.getLogger(__name__)

# Set up SQLAlchemy
db = SQLAlchemy()

# Helper function to determine if we're in testing mode
def is_testing_db(connection):
    """Check if we're using a testing database (SQLite)."""
    return connection.engine.url.drivername == 'sqlite'

# Helper function to normalize text
def normalize_lemma(text: str) -> str:
    """Normalize lemma for consistent comparison - matches dictionary_manager.py."""
    if not text:
        logger.warning("normalize_lemma received empty or None text")
        return ""
    return unidecode(text).lower()

# Helper function to get romanized text from Baybayin
def get_romanized_text(text: str) -> str:
    """Convert Baybayin text to romanized form."""
    if not text:
        return text
    if not any(0x1700 <= ord(c) <= 0x171F for c in text):
        return text
    
    # Baybayin character mapping based on dictionary_manager.py's BaybayinRomanizer
    mapping = {
        'ᜀ': 'a', 'ᜁ': 'i', 'ᜂ': 'u', 'ᜃ': 'ka', 'ᜄ': 'ga', 
        'ᜅ': 'nga', 'ᜆ': 'ta', 'ᜇ': 'da', 'ᜈ': 'na',
        'ᜉ': 'pa', 'ᜊ': 'ba', 'ᜋ': 'ma', 'ᜌ': 'ya', 
        'ᜎ': 'la', 'ᜏ': 'wa', 'ᜐ': 'sa', 'ᜑ': 'ha', 'ᜍ': 'ra',
        'ᜒ': 'i', 'ᜓ': 'u', '᜔': '', '᜵': ',', '᜶': '.',
        # Virama (vowel killer) combinations
        'ᜃ᜔': 'k', 'ᜄ᜔': 'g', 'ᜅ᜔': 'ng',
        'ᜆ᜔': 't', 'ᜇ᜔': 'd', 'ᜈ᜔': 'n',
        'ᜉ᜔': 'p', 'ᜊ᜔': 'b', 'ᜋ᜔': 'm',
        'ᜌ᜔': 'y', 'ᜎ᜔': 'l', 'ᜏ᜔': 'w',
        'ᜐ᜔': 's', 'ᜑ᜔': 'h', 'ᜍ᜔': 'r'
    }
    
    # Process each syllable based on dictionary_manager's approach
    result = []
    i = 0
    while i < len(text):
        # Check for two-character combinations first
        if i + 1 < len(text) and text[i:i+2] in mapping:
            result.append(mapping[text[i:i+2]])
            i += 2
        elif text[i] in mapping:
            result.append(mapping[text[i]])
            i += 1
        else:
            # Keep other characters (spaces, etc.) as is
            result.append(text[i])
            i += 1
    
    return ''.join(result)

def extract_etymology_components(etymology_text):
    """Extract structured components from etymology text."""
    if not etymology_text:
        return None
    
    # Basic structure following the original implementation
    return {
        "original_text": etymology_text,
        "processed": True
    }

def extract_language_codes(etymology: str) -> list:
    """Extract ISO 639-1 language codes from etymology string."""
    if not etymology:
        return []
        
    # Language code mapping based on dictionary_manager.py
    lang_map = {
        "Esp": "es", "Eng": "en", "Ch": "zh", "Tsino": "zh", "Jap": "ja",
        "San": "sa", "Sanskrit": "sa", "Tag": "tl", "Mal": "ms", "Arb": "ar"
    }
    return [lang_map[lang] for lang in lang_map if lang in etymology]

# Word model
class Word(db.Model):
    """Model for words - matches the 'words' table from dictionary_manager.py."""
    __tablename__ = 'words'
    
    id = Column(Integer, primary_key=True)
    lemma = Column(String(255), nullable=False)
    normalized_lemma = Column(String(255), nullable=False)
    has_baybayin = Column(Boolean, default=False)
    baybayin_form = Column(String(255))
    romanized_form = Column(String(255))
    language_code = Column(String(16), nullable=False)
    root_word_id = Column(Integer, ForeignKey('words.id'))
    preferred_spelling = Column(String(255))
    tags = Column(Text)
    idioms = Column(JSONB, default=lambda: [])
    pronunciation_data = Column(JSONB)
    source_info = Column(JSONB, default=lambda: {})
    word_metadata = Column(JSONB, default=lambda: {})
    data_hash = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    search_text = Column(TSVECTOR)
    badlit_form = Column(Text)
    hyphenation = Column(JSONB)
    is_proper_noun = Column(Boolean, default=False)
    is_abbreviation = Column(Boolean, default=False)
    is_initialism = Column(Boolean, default=False)
    
    # Relationships
    definitions = relationship('Definition', back_populates='word', cascade="all, delete-orphan")
    etymologies = relationship('Etymology', back_populates='word', cascade="all, delete-orphan")
    pronunciations = relationship('Pronunciation', back_populates='word', cascade="all, delete-orphan")
    credits = relationship('Credit', back_populates='word', cascade="all, delete-orphan")
    
    # Self-referential relationship for root words
    root_word = relationship('Word', remote_side=[id], backref=backref('derived_words', lazy='dynamic'))
    
    # Baybayin constraint from the database schema
    __table_args__ = (
        UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
        Index('idx_words_lemma', 'lemma'),
        Index('idx_words_normalized', 'normalized_lemma'),
        Index('idx_words_baybayin', 'baybayin_form'),
        Index('idx_words_language', 'language_code'),
        Index('idx_words_root', 'root_word_id'),
    )
    
    @validates('lemma')
    def validate_lemma(self, key, value):
        """Validate lemma."""
        if not value or not isinstance(value, str):
            raise ValueError("Lemma must be a non-empty string")
        return value.strip()
    
    @validates('normalized_lemma')
    def validate_normalized_lemma(self, key, value):
        """Validate normalized lemma."""
        if not value or not isinstance(value, str):
            raise ValueError("Normalized lemma must be a non-empty string")
        return value.strip()
    
    @validates('language_code')
    def validate_language_code(self, key, value):
        """Validate language code."""
        if not value or not isinstance(value, str):
            raise ValueError("Language code must be a non-empty string")
        if len(value) > 16:
            raise ValueError("Language code must be 16 characters or less")
        return value.strip()
    
    @validates('baybayin_form')
    def validate_baybayin_form(self, key, value):
        """Validate Baybayin form."""
        if value is not None:
            if not re.match(r'^[\u1700-\u171F\s]*$', value):
                raise ValueError("Baybayin form contains invalid characters")
        return value
    
    @hybrid_property
    def is_root(self):
        """Check if this word is a root word (has no root_word_id)."""
        return self.root_word_id is None
    
    def get_tags_list(self):
        """Get tags as a list."""
        if not self.tags:
            return []
            
        if isinstance(self.tags, list):
            return self.tags
            
        if isinstance(self.tags, str):
            try:
                # Try to parse as JSON first
                return json.loads(self.tags)
            except json.JSONDecodeError:
                # If not valid JSON, treat as comma-separated list
                return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
                
        return []
    
    def to_dict(self, include_related=True):
        """Convert the model to a dictionary."""
        result = {
            'id': self.id,
            'lemma': self.lemma,
            'normalized_lemma': self.normalized_lemma,
            'language_code': self.language_code,
            'has_baybayin': self.has_baybayin,
            'baybayin_form': self.baybayin_form,
            'romanized_form': self.romanized_form,
            'root_word_id': self.root_word_id,
            'preferred_spelling': self.preferred_spelling,
            'tags': self.get_tags_list(),
            'source_info': self.source_info,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'badlit_form': self.badlit_form,
            'hyphenation': self.hyphenation,
            'is_proper_noun': self.is_proper_noun,
            'is_abbreviation': self.is_abbreviation,
            'is_initialism': self.is_initialism,
        }
        
        if include_related:
            if self.definitions:
                result['definitions'] = [d.to_dict() for d in self.definitions]
            
            if self.etymologies:
                result['etymologies'] = [e.to_dict() for e in self.etymologies]
                
            if self.pronunciations:
                result['pronunciations'] = [p.to_dict() for p in self.pronunciations]
        
        return result
    
    def __repr__(self):
        return f"<Word {self.lemma} ({self.language_code})>"


class Definition(db.Model):
    """Model for word definitions - matches the 'definitions' table from dictionary_manager.py."""
    __tablename__ = 'definitions'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    definition_text = Column(Text, nullable=False)
    original_pos = Column(Text)
    standardized_pos_id = Column(Integer, ForeignKey('parts_of_speech.id'))
    examples = Column(Text)
    usage_notes = Column(Text)
    tags = Column(Text)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', back_populates='definitions')
    standardized_pos = relationship('PartOfSpeech', backref='definitions')
    
    __table_args__ = (
        UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique'),
        Index('idx_definitions_word', 'word_id'),
        Index('idx_definitions_pos', 'standardized_pos_id'),
    )
    
    def get_examples_list(self):
        """Get examples as a list."""
        if not self.examples:
            return []
            
        try:
            return json.loads(self.examples) if isinstance(self.examples, str) else self.examples
        except json.JSONDecodeError:
            return [self.examples]
    
    def get_tags_list(self):
        """Get tags as a list."""
        if not self.tags:
            return []
            
        try:
            return json.loads(self.tags) if isinstance(self.tags, str) else self.tags
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def to_dict(self):
        """Convert definition to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'definition_text': self.definition_text,
            'original_pos': self.original_pos,
            'standardized_pos_id': self.standardized_pos_id,
            'standardized_pos': self.standardized_pos.to_dict() if self.standardized_pos else None,
            'examples': self.get_examples_list(),
            'usage_notes': self.usage_notes,
            'tags': self.get_tags_list(),
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Definition {self.id} for word_id {self.word_id}>"


class Etymology(db.Model):
    """Model for word etymologies - matches the 'etymologies' table from dictionary_manager.py."""
    __tablename__ = 'etymologies'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = Column(Text, nullable=False)
    normalized_components = Column(Text)
    etymology_structure = Column(Text)
    language_codes = Column(Text)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', back_populates='etymologies')
    
    __table_args__ = (
        UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq'),
        Index('idx_etymologies_word', 'word_id'),
    )
    
    def get_language_codes_list(self):
        """Get language codes as a list."""
        if not self.language_codes:
            return []
            
        try:
            return json.loads(self.language_codes) if isinstance(self.language_codes, str) else self.language_codes
        except json.JSONDecodeError:
            # If it's a comma-separated string
            return [code.strip() for code in self.language_codes.split(',') if code.strip()]
    
    def to_dict(self):
        """Convert etymology to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'normalized_components': self.normalized_components,
            'etymology_structure': self.etymology_structure,
            'language_codes': self.get_language_codes_list(),
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Etymology {self.id} for word_id {self.word_id}>"


class Relation(db.Model):
    """Model for word relationships - matches the 'relations' table from dictionary_manager.py."""
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True)
    from_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text)
    metadata = Column(JSONB, default=lambda: {})
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships - these match how they would be created in SQLAlchemy
    source_word = relationship('Word', foreign_keys=[from_word_id], 
                              backref=backref('outgoing_relations', cascade='all, delete-orphan'))
    target_word = relationship('Word', foreign_keys=[to_word_id], 
                              backref=backref('incoming_relations', cascade='all, delete-orphan'))
    
    __table_args__ = (
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
        Index('idx_relations_from', 'from_word_id'),
        Index('idx_relations_to', 'to_word_id'),
        Index('idx_relations_type', 'relation_type'),
    )
    
    def to_dict(self):
        """Convert relation to dictionary."""
        return {
            'id': self.id,
            'from_word_id': self.from_word_id,
            'to_word_id': self.to_word_id,
            'source_word': self.source_word.lemma if self.source_word else None,
            'target_word': self.target_word.lemma if self.target_word else None,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Relation {self.id} from {self.from_word_id} to {self.to_word_id} ({self.relation_type})>"


class DefinitionRelation(db.Model):
    """Model for definition relationships - matches the 'definition_relations' table from dictionary_manager.py."""
    __tablename__ = 'definition_relations'
    
    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    definition = relationship('Definition')
    related_word = relationship('Word')
    
    __table_args__ = (
        UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique'),
        Index('idx_def_relations_def', 'definition_id'),
        Index('idx_def_relations_word', 'word_id'),
    )
    
    def to_dict(self):
        """Convert definition relation to dictionary."""
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'word_id': self.word_id,
            'relation_type': self.relation_type,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<DefinitionRelation {self.id} from def {self.definition_id} to word {self.word_id}>"


class Affixation(db.Model):
    """Model for word affixation relationships - matches the 'affixations' table from dictionary_manager.py."""
    __tablename__ = 'affixations'
    
    id = Column(Integer, primary_key=True)
    root_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affixed_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affix_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    root_word = relationship('Word', foreign_keys=[root_word_id], 
                            backref=backref('root_affixations', cascade='all, delete-orphan'))
    affixed_word = relationship('Word', foreign_keys=[affixed_word_id], 
                              backref=backref('affixed_affixations', cascade='all, delete-orphan'))
    
    __table_args__ = (
        UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
        Index('idx_affixations_root', 'root_word_id'),
        Index('idx_affixations_affixed', 'affixed_word_id'),
    )
    
    def to_dict(self):
        """Convert affixation to dictionary."""
        return {
            'id': self.id,
            'root_word_id': self.root_word_id,
            'affixed_word_id': self.affixed_word_id,
            'root_word': self.root_word.lemma if self.root_word else None,
            'affixed_word': self.affixed_word.lemma if self.affixed_word else None,
            'affix_type': self.affix_type,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<Affixation {self.id} from {self.root_word_id} to {self.affixed_word_id}>"


class PartOfSpeech(db.Model):
    """Model for parts of speech - matches the 'parts_of_speech' table from dictionary_manager.py."""
    __tablename__ = 'parts_of_speech'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False, unique=True)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64), nullable=False)
    description = Column(Text)
    
    __table_args__ = (
        UniqueConstraint('code', name='parts_of_speech_code_uniq'),
        Index('idx_parts_of_speech_code', 'code'),
        Index('idx_parts_of_speech_name', 'name_en', 'name_tl'),
    )
    
    def to_dict(self):
        """Convert part of speech to dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'name_en': self.name_en,
            'name_tl': self.name_tl,
            'description': self.description,
        }
    
    def __repr__(self):
        return f"<PartOfSpeech {self.code} ({self.name_en})>"


class Pronunciation(db.Model):
    """Model for word pronunciations - matches the 'pronunciations' table from dictionary_manager.py."""
    __tablename__ = 'pronunciations'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(20), nullable=False, default='ipa')
    value = Column(Text, nullable=False)
    tags = Column(JSONB)
    metadata = Column(JSONB)
    sources = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', back_populates='pronunciations')
    
    __table_args__ = (
        UniqueConstraint('word_id', 'type', 'value', name='pronunciations_unique'),
        Index('idx_pronunciations_word', 'word_id'),
        Index('idx_pronunciations_type', 'type'),
        Index('idx_pronunciations_value', 'value'),
    )
    
    def to_dict(self):
        """Convert pronunciation to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'type': self.type,
            'value': self.value,
            'tags': self.tags,
            'metadata': self.metadata,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Pronunciation {self.id} for word_id {self.word_id} ({self.type})>"


class Credit(db.Model):
    """Model for word credits - matches the 'credits' table from dictionary_manager.py."""
    __tablename__ = 'credits'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    credit = Column(Text, nullable=False)
    sources = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    word = relationship('Word', back_populates='credits')
    
    __table_args__ = (
        UniqueConstraint('word_id', 'credit', name='credits_unique'),
        Index('idx_credits_word', 'word_id'),
    )
    
    def to_dict(self):
        """Convert credit to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'credit': self.credit,
            'sources': self.sources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Credit {self.id} for word_id {self.word_id}>"


# Helper functions for initialization
def initialize_parts_of_speech(db_session):
    """Initialize standard parts of speech based on the dictionary_manager.py schema."""
    standard_pos = [
        {'code': 'n', 'name_en': 'Noun', 'name_tl': 'Pangngalan', 
         'description': 'Word that refers to a person, place, thing, or idea'},
        {'code': 'v', 'name_en': 'Verb', 'name_tl': 'Pandiwa', 
         'description': 'Word that expresses action or state of being'},
        {'code': 'adj', 'name_en': 'Adjective', 'name_tl': 'Pang-uri', 
         'description': 'Word that describes or modifies a noun'},
        {'code': 'adv', 'name_en': 'Adverb', 'name_tl': 'Pang-abay', 
         'description': 'Word that modifies verbs, adjectives, or other adverbs'},
        {'code': 'pron', 'name_en': 'Pronoun', 'name_tl': 'Panghalip', 
         'description': 'Word that substitutes for a noun'},
        {'code': 'prep', 'name_en': 'Preposition', 'name_tl': 'Pang-ukol', 
         'description': 'Word that shows relationship between words'},
        {'code': 'conj', 'name_en': 'Conjunction', 'name_tl': 'Pangatnig', 
         'description': 'Word that connects words, phrases, or clauses'},
        {'code': 'intj', 'name_en': 'Interjection', 'name_tl': 'Pandamdam', 
         'description': 'Word expressing emotion'},
        {'code': 'det', 'name_en': 'Determiner', 'name_tl': 'Pantukoy', 
         'description': 'Word that modifies nouns'},
        {'code': 'affix', 'name_en': 'Affix', 'name_tl': 'Panlapi', 
         'description': 'Word element attached to base or root'},
        {'code': 'idm', 'name_en': 'Idiom', 'name_tl': 'Idyoma', 
         'description': 'Fixed expression with non-literal meaning'},
        {'code': 'col', 'name_en': 'Colloquial', 'name_tl': 'Kolokyal', 
         'description': 'Informal or conversational usage'},
        {'code': 'syn', 'name_en': 'Synonym', 'name_tl': 'Singkahulugan', 
         'description': 'Word with similar meaning'},
        {'code': 'ant', 'name_en': 'Antonym', 'name_tl': 'Di-kasingkahulugan', 
         'description': 'Word with opposite meaning'},
        {'code': 'eng', 'name_en': 'English', 'name_tl': 'Ingles', 
         'description': 'English loanword or translation'},
        {'code': 'spa', 'name_en': 'Spanish', 'name_tl': 'Espanyol', 
         'description': 'Spanish loanword or origin'},
        {'code': 'tx', 'name_en': 'Texting', 'name_tl': 'Texting', 
         'description': 'Text messaging form'},
        {'code': 'var', 'name_en': 'Variant', 'name_tl': 'Varyant', 
         'description': 'Alternative form or spelling'},
        {'code': 'unc', 'name_en': 'Uncategorized', 'name_tl': 'Hindi Tiyak', 
         'description': 'Part of speech not yet determined'}
    ]
    
    existing_codes = {pos.code for pos in db_session.query(PartOfSpeech).all()}
    
    for pos_data in standard_pos:
        if pos_data['code'] not in existing_codes:
            pos = PartOfSpeech(**pos_data)
            db_session.add(pos)
    
    db_session.commit()


# Search functions
def search_words(query, language_code=None, limit=20, offset=0):
    """Search for words in the dictionary."""
    if not query:
        return []
        
    # PostgreSQL specific search if available
    if not is_testing_db(db.get_engine()):
        search_query = Word.query.filter(
            Word.search_text.op('@@')(func.to_tsquery('simple', query.replace(' ', ' & ')))
        ).order_by(
            func.ts_rank_cd(Word.search_text, func.to_tsquery('simple', query.replace(' ', ' & '))).desc()
        )
    else:
        # Fallback for SQLite or testing
        normalized_query = normalize_lemma(query)
        search_query = Word.query.filter(
            db.or_(
                Word.lemma.ilike(f'%{query}%'),
                Word.normalized_lemma.ilike(f'%{normalized_query}%')
            )
        )
    
    # Apply language filter if provided
    if language_code:
        search_query = search_query.filter(Word.language_code == language_code)
    
    return search_query.limit(limit).offset(offset).all()


def search_baybayin(query, limit=20, offset=0):
    """Search specifically for Baybayin words."""
    if not query:
        return []
        
    search_query = Word.query.filter(
        Word.has_baybayin == True
    )
    
    # Add Baybayin specific search if query contains Baybayin characters
    if any(0x1700 <= ord(c) <= 0x171F for c in query):
        search_query = search_query.filter(
            Word.baybayin_form.ilike(f'%{query}%')
        )
    else:
        # If no Baybayin characters, search by lemma or romanized form
        search_query = search_query.filter(
            db.or_(
                Word.lemma.ilike(f'%{query}%'),
                Word.romanized_form.ilike(f'%{query}%')
            )
        )
    
    return search_query.limit(limit).offset(offset).all()


def get_word_by_id(word_id):
    """Get a word by its ID."""
    return Word.query.get(word_id)


def get_word_by_lemma(lemma, language_code='tl'):
    """Get a word by its lemma and language code."""
    return Word.query.filter_by(
        normalized_lemma=normalize_lemma(lemma),
        language_code=language_code
    ).first()


def get_random_words(limit=10, language_code=None):
    """Get random words from the dictionary."""
    query = Word.query
    
    if language_code:
        query = query.filter_by(language_code=language_code)
    
    # PostgreSQL specific random selection if available
    if not is_testing_db(db.get_engine()):
        query = query.order_by(func.random())
    else:
        # Fallback for SQLite - less efficient but works
        # Get total count
        total = query.count()
        if total <= limit:
            return query.all()
            
        # Generate random IDs
        import random
        random_ids = random.sample(range(1, total + 1), limit)
        query = query.filter(Word.id.in_(random_ids))
    
    return query.limit(limit).all()


# Initialize the models
def init_app(app):
    """Initialize the models with the Flask app."""
    db.init_app(app)
    
    # Create all tables
    with app.app_context():
        db.create_all()
        
        # Initialize standard parts of speech
        initialize_parts_of_speech(db.session)
        
        # Set up PostgreSQL-specific features if using PostgreSQL
        if not is_testing_db(db.engine.connect()):
            try:
                # Create extensions
                db.session.execute(text("""
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;
                    CREATE EXTENSION IF NOT EXISTS unaccent;
                    CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                """))
                
                # Create update timestamp function
                db.session.execute(text("""
                    CREATE OR REPLACE FUNCTION update_timestamp()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """))
                
                # Create triggers for automatic timestamp updates
                table_names = ['words', 'definitions', 'etymologies', 'pronunciations', 'credits', 'relations']
                for table in table_names:
                    db.session.execute(text(f"""
                        DROP TRIGGER IF EXISTS update_{table}_timestamp ON {table};
                        CREATE TRIGGER update_{table}_timestamp
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_timestamp();
                    """))
                
                # Create text search indexes
                db.session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
                    CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
                    CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = TRUE;
                    CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
                    CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
                    CREATE INDEX IF NOT EXISTS idx_definitions_pos ON definitions(standardized_pos_id);
                    CREATE INDEX IF NOT EXISTS idx_definitions_text ON definitions USING gin(to_tsvector('english', definition_text));
                    
                    CREATE INDEX IF NOT EXISTS idx_etymologies_word ON etymologies(word_id);
                    CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));
                    
                    CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
                    CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
                    CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
                    CREATE INDEX IF NOT EXISTS idx_relations_metadata ON relations USING GIN(metadata);
                    
                    CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
                    CREATE INDEX IF NOT EXISTS idx_pronunciations_type ON pronunciations(type);
                    CREATE INDEX IF NOT EXISTS idx_pronunciations_value ON pronunciations(value);
                    
                    CREATE INDEX IF NOT EXISTS idx_credits_word ON credits(word_id);
                """))
                
                # Add search vector update trigger for full text search
                try:
                    db.session.execute(text("""
                        DROP TRIGGER IF EXISTS tsvectorupdate ON words;
                        CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
                        ON words FOR EACH ROW EXECUTE FUNCTION
                        tsvector_update_trigger(search_text, 'pg_catalog.simple', lemma, normalized_lemma, baybayin_form, romanized_form);
                    """))
                except:
                    app.logger.warning("Could not set up tsvector_update_trigger. Full-text search may require manual updates.")
                
                db.session.commit()
                app.logger.info("PostgreSQL extensions and indexes set up successfully")
            except Exception as e:
                app.logger.warning(f"Could not set up some PostgreSQL features: {str(e)}")
                db.session.rollback()
        
        app.logger.info("Database models initialized successfully")