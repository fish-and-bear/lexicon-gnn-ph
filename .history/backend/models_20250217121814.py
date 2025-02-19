import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, JSON, ARRAY, Boolean, DateTime, func, Index, UniqueConstraint, DDL, event
from sqlalchemy.orm import relationship, sessionmaker, validates, declarative_base, scoped_session
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from dotenv import load_dotenv
import re
from backend.language_utils import LanguageSystem
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import expression
from typing import List, Optional, Dict, Any
import json
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import column_property
from sqlalchemy.sql import select
import hashlib
from datetime import timedelta

load_dotenv()

# Create SQLAlchemy instance
db = SQLAlchemy()

# Helper function to avoid circular import
def get_romanized_text(text: str) -> str:
    """Simple romanization for Baybayin text."""
    if not text:
        return text
    if not any(0x1700 <= ord(c) <= 0x171F for c in text):
        return text
    # Basic romanization mapping
    mapping = {
        'ᜀ': 'a', 'ᜁ': 'i', 'ᜂ': 'u',
        'ᜃ': 'ka', 'ᜄ': 'ga', 'ᜅ': 'nga',
        'ᜆ': 'ta', 'ᜇ': 'da', 'ᜈ': 'na',
        'ᜉ': 'pa', 'ᜊ': 'ba', 'ᜋ': 'ma',
        'ᜌ': 'ya', 'ᜎ': 'la', 'ᜏ': 'wa',
        'ᜐ': 'sa', 'ᜑ': 'ha',
        'ᜒ': 'i', 'ᜓ': 'u', '᜔': '',
        '᜵': ',', '᜶': '.'
    }
    result = []
    i = 0
    while i < len(text):
        if text[i] in mapping:
            result.append(mapping[text[i]])
        else:
            result.append(text[i])
        i += 1
    return ''.join(result)

# Create text search indexes
word_search_ddl = DDL(
    "CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text)"
)

etymology_search_ddl = DDL(
    "CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes))"
)

# Create timestamp triggers
timestamp_trigger = DDL("""
    CREATE OR REPLACE FUNCTION update_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
""")

# Association Tables
word_language_table = Table(
    'word_languages', db.Model.metadata,
    Column('word_id', Integer, ForeignKey('words.id'), primary_key=True),
    Column('language_id', Integer, ForeignKey('languages.id'), primary_key=True)
)

# Add base mixin class
class BaseMixin:
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    @classmethod
    def get_by_id(cls, id):
        return db.session.query(cls).get(id)

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

# Enhance Word model
class Word(BaseMixin, db.Model):
    __tablename__ = 'words'
    
    lemma = Column(String(255), nullable=False)
    normalized_lemma = Column(String(255), nullable=False)
    language_code = Column(String(16), nullable=False)
    root_word_id = Column(Integer, ForeignKey('words.id'))
    preferred_spelling = Column(String(255))
    tags = Column(Text)
    has_baybayin = Column(Boolean, default=False)
    baybayin_form = Column(String(255))
    romanized_form = Column(String(255))
    pronunciation_data = Column(JSONB)
    source_info = Column(JSONB, default=dict)
    data_hash = Column(Text)
    search_text = Column(TSVECTOR)
    idioms = Column(JSONB, default=list)
    view_count = Column(Integer, default=0)
    last_viewed_at = Column(DateTime(timezone=True))
    is_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    data_quality_score = Column(Integer, default=0)
    search_vector = Column(TSVECTOR)
    complexity_score = Column(Integer, default=0)
    usage_frequency = Column(Integer, default=0)
    last_lookup_at = Column(DateTime(timezone=True))

    # Relationships
    definitions = relationship("Definition", back_populates="word", cascade="all, delete-orphan")
    etymologies = relationship("Etymology", back_populates="word", cascade="all, delete-orphan")
    relations_from = relationship("Relation", foreign_keys="[Relation.from_word_id]", back_populates="from_word", cascade="all, delete-orphan")
    relations_to = relationship("Relation", foreign_keys="[Relation.to_word_id]", back_populates="to_word", cascade="all, delete-orphan")
    affixations_as_root = relationship("Affixation", foreign_keys="[Affixation.root_word_id]", back_populates="root_word", cascade="all, delete-orphan")
    affixations_as_affixed = relationship("Affixation", foreign_keys="[Affixation.affixed_word_id]", back_populates="affixed_word", cascade="all, delete-orphan")
    definition_relations = relationship("DefinitionRelation", back_populates="word", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_words_normalized_lang', 'normalized_lemma', 'language_code'),
        Index('idx_words_baybayin', 'baybayin_form'),
        Index('idx_words_search_text', 'search_text', postgresql_using='gin'),
        Index('idx_words_trgm', 'normalized_lemma', postgresql_using='gin', postgresql_ops={'normalized_lemma': 'gin_trgm_ops'})
    )

    @hybrid_property
    def is_root_word(self):
        """Check if this is a root word (no derivations)."""
        return self.root_word_id is None
    
    @hybrid_property
    def has_complete_data(self):
        """Check if word has complete data (definitions, etymology, etc)."""
        return bool(self.definitions and (self.etymologies or self.relations_from))
    
    @validates('lemma', 'normalized_lemma')
    def validate_lemma(self, key, value):
        if not value or not isinstance(value, str):
            raise ValueError(f"{key} must be a non-empty string")
        if len(value) > 255:
            raise ValueError(f"{key} must be less than 255 characters")
        return value

    def calculate_complexity_score(self) -> int:
        """Calculate word complexity based on various factors."""
        score = 0
        # Length complexity
        score += len(self.lemma) // 2
        # Relation complexity
        score += len(self.relations_from) + len(self.relations_to)
        # Etymology complexity
        score += len(self.etymologies) * 2
        # Definition complexity
        score += len(self.definitions)
        # Affixation complexity
        score += len(self.affixations_as_root) + len(self.affixations_as_affixed)
        return min(score, 100)  # Cap at 100

    def update_search_vector(self):
        """Update the search vector with all searchable content."""
        search_parts = [
            self.lemma,
            self.normalized_lemma,
            self.preferred_spelling or '',
            ' '.join(d.definition_text for d in self.definitions),
            ' '.join(e.etymology_text for e in self.etymologies),
            self.baybayin_form or '',
            self.romanized_form or ''
        ]
        self.search_vector = func.to_tsvector('simple', ' '.join(search_parts))

    def to_dict(self, include_relations: bool = True) -> Dict[str, Any]:
        """Convert word to dictionary with configurable depth."""
        result = {
            "id": self.id,
            "lemma": self.lemma,
            "normalized_lemma": self.normalized_lemma,
            "language_code": self.language_code,
            "preferred_spelling": self.preferred_spelling,
            "tags": self.tags.split(", ") if self.tags else [],
            "has_baybayin": self.has_baybayin,
            "baybayin_form": self.baybayin_form,
            "romanized_form": self.romanized_form,
            "pronunciation_data": self.pronunciation_data,
            "complexity_score": self.complexity_score,
            "usage_frequency": self.usage_frequency,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_lookup_at": self.last_lookup_at.isoformat() if self.last_lookup_at else None,
            "view_count": self.view_count,
            "last_viewed_at": self.last_viewed_at.isoformat() if self.last_viewed_at else None,
            "is_verified": self.is_verified,
            "verification_notes": self.verification_notes,
            "data_quality_score": self.data_quality_score
        }
        
        if include_relations:
            result.update({
                "definitions": [d.to_dict() for d in self.definitions],
                "etymologies": [e.to_dict() for e in self.etymologies],
                "relations": {
                    "from": [r.to_dict() for r in self.relations_from],
                    "to": [r.to_dict() for r in self.relations_to]
                },
                "affixations": {
                    "as_root": [a.to_dict() for a in self.affixations_as_root],
                    "as_affixed": [a.to_dict() for a in self.affixations_as_affixed]
                }
            })
        
        return result

    def validate_baybayin(self):
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")
        if self.baybayin_form and not re.match(r'^[\u1700-\u171F\s]*$', self.baybayin_form):
            raise ValueError("Invalid Baybayin characters")
        if self.has_baybayin and not self.romanized_form:
            self.romanized_form = get_romanized_text(self.baybayin_form)

    def process_baybayin_data(self):
        if self.baybayin_form:
            romanized = get_romanized_text(self.baybayin_form)
            if romanized != self.romanized_form:
                self.romanized_form = romanized

    def get_language_metadata(self):
        lsys = LanguageSystem()
        return lsys.get_language_metadata(self.language_code)

    @validates('language_code')
    def validate_language_code(self, key, code):
        lsys = LanguageSystem()
        if code not in lsys.valid_codes:
            raise ValueError(f"Invalid language code: {code}")
        return code

    @hybrid_property
    def age_days(self):
        """Get the age of the word entry in days."""
        if self.created_at:
            return (datetime.now(UTC) - self.created_at).days
        return 0

    @hybrid_property
    def is_popular(self):
        """Check if word is popular based on view count."""
        return self.view_count > 1000

    @hybrid_property
    def is_recently_updated(self):
        """Check if word was updated in the last week."""
        if self.updated_at:
            return self.updated_at > datetime.now(UTC) - timedelta(days=7)
        return False

    def increment_view_count(self):
        """Increment view count and update last viewed timestamp."""
        self.view_count += 1
        self.last_viewed_at = datetime.now(UTC)
        
    def calculate_data_quality_score(self) -> int:
        """Calculate data quality score based on completeness and verification."""
        score = 0
        
        # Basic data completeness
        if self.lemma and self.normalized_lemma:
            score += 10
        if self.definitions:
            score += len(self.definitions) * 5
        if self.etymologies:
            score += len(self.etymologies) * 5
            
        # Source quality
        if self.source_info:
            reliable_sources = ['kwf', 'upd', 'diksiyonaryo']
            score += sum(5 for source in self.source_info if source in reliable_sources)
            
        # Verification status
        if self.is_verified:
            score += 20
            
        # Relations completeness
        if self.relations_from or self.relations_to:
            score += 10
            
        # Baybayin completeness
        if self.has_baybayin and self.baybayin_form and self.romanized_form:
            score += 15
            
        return min(score, 100)  # Cap at 100

    def generate_data_hash(self) -> str:
        """Generate a hash of the word's data for change detection."""
        data_string = f"{self.lemma}{self.normalized_lemma}{self.language_code}"
        data_string += str(sorted([d.definition_text for d in self.definitions]))
        data_string += str(sorted([e.etymology_text for e in self.etymologies]))
        return hashlib.sha256(data_string.encode()).hexdigest()

class Definition(BaseMixin, db.Model):
    __tablename__ = 'definitions'
    
    definition_text = Column(Text, nullable=False)
    original_pos = Column(Text)
    standardized_pos_id = Column(Integer, ForeignKey('parts_of_speech.id'))
    examples = Column(Text)
    usage_notes = Column(Text)
    sources = Column(Text, nullable=False)
    confidence_score = Column(Integer, default=0)
    is_verified = Column(Boolean, default=False)
    verified_by = Column(String(255))
    verified_at = Column(DateTime(timezone=True))

    # Relationships
    word = relationship("Word", back_populates="definitions")
    part_of_speech = relationship("PartOfSpeech")
    definition_relations = relationship("DefinitionRelation", back_populates="definition", cascade="all, delete-orphan")

    @validates('sources')
    def validate_sources(self, key, sources):
        from source_standardization import SourceStandardization
        return SourceStandardization.standardize_sources(sources)

    def get_standardized_sources(self):
        from source_standardization import SourceStandardization
        return [SourceStandardization.get_display_name(s.strip()) 
                for s in self.sources.split(',') if s.strip()]

    @validates('definition_text')
    def validate_definition_text(self, key, value):
        if not value or not isinstance(value, str):
            raise ValueError("Definition text must be a non-empty string")
        return value.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.definition_text,
            "original_pos": self.original_pos,
            "part_of_speech": self.part_of_speech.to_dict() if self.part_of_speech else None,
            "examples": self.examples.split("\n") if self.examples else [],
            "usage_notes": self.usage_notes.split("; ") if self.usage_notes else [],
            "sources": self.get_standardized_sources(),
            "relations": [r.to_dict() for r in self.definition_relations],
            "confidence_score": self.confidence_score,
            "is_verified": self.is_verified,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None
        }

    def calculate_confidence_score(self) -> int:
        """Calculate confidence score based on various factors."""
        score = 0
        
        # Source reliability
        reliable_sources = ['kwf', 'upd', 'diksiyonaryo']
        if any(source in self.sources for source in reliable_sources):
            score += 30
            
        # Verification status
        if self.is_verified:
            score += 30
            
        # Content completeness
        if self.examples:
            score += 20
        if self.usage_notes:
            score += 10
        if self.part_of_speech:
            score += 10
            
        return min(score, 100)

class Etymology(BaseMixin, db.Model):
    __tablename__ = 'etymologies'
    
    etymology_text = Column(Text, nullable=False)
    normalized_components = Column(Text)
    language_codes = Column(Text)
    sources = Column(Text, nullable=False)
    confidence_level = Column(String(20), default='medium')  # low, medium, high
    verification_status = Column(String(20), default='unverified')
    verification_notes = Column(Text)

    # Relationships
    word = relationship("Word", back_populates="etymologies")

    CONFIDENCE_LEVELS = ['low', 'medium', 'high']
    VERIFICATION_STATUSES = ['unverified', 'pending', 'verified', 'rejected']
    
    @validates('etymology_text')
    def validate_etymology_text(self, key, value):
        if not value or not isinstance(value, str):
            raise ValueError("Etymology text must be a non-empty string")
        return value.strip()

    def extract_components(self):
        """Extract and normalize etymology components."""
        components = extract_etymology_components(self.etymology_text)
        self.normalized_components = ", ".join(components)

    def extract_meaning(self):
        """Extract meaning from etymology text."""
        text, meaning = extract_meaning(self.etymology_text)
        return meaning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.etymology_text,
            "components": self.normalized_components.split(", ") if self.normalized_components else [],
            "languages": self.language_codes.split(", ") if self.language_codes else [],
            "sources": self.sources.split(", ") if self.sources else [],
            "confidence_level": self.confidence_level,
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes
        }

    @validates('confidence_level')
    def validate_confidence_level(self, key, value):
        if value not in self.CONFIDENCE_LEVELS:
            raise ValueError(f"Invalid confidence level. Must be one of: {', '.join(self.CONFIDENCE_LEVELS)}")
        return value
    
    @validates('verification_status')
    def validate_verification_status(self, key, value):
        if value not in self.VERIFICATION_STATUSES:
            raise ValueError(f"Invalid verification status. Must be one of: {', '.join(self.VERIFICATION_STATUSES)}")
        return value

class Relation(db.Model):
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True)
    from_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)

    # Relationships
    from_word = relationship("Word", foreign_keys=[from_word_id], back_populates="relations_from")
    to_word = relationship("Word", foreign_keys=[to_word_id], back_populates="relations_to")

    __table_args__ = (
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
    )

    VALID_TYPES = [
        'synonym', 'antonym', 'variant', 'derived_from', 
        'component_of', 'related', 'cognate'
    ]
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid relation type. Must be one of: {', '.join(self.VALID_TYPES)}")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.relation_type,
            "from_word": self.from_word.lemma,
            "to_word": self.to_word.lemma,
            "sources": self.sources.split(", ") if self.sources else [],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class DefinitionRelation(db.Model):
    __tablename__ = 'definition_relations'
    
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)

    # Relationships
    definition = relationship("Definition", back_populates="definition_relations")
    word = relationship("Word", back_populates="definition_relations")

class Affixation(db.Model):
    __tablename__ = 'affixations'
    
    root_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affixed_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affix_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)

    # Relationships
    root_word = relationship("Word", foreign_keys=[root_word_id], back_populates="affixations_as_root")
    affixed_word = relationship("Word", foreign_keys=[affixed_word_id], back_populates="affixations_as_affixed")

    VALID_TYPES = [
        'prefix', 'infix', 'suffix', 'circumfix', 
        'reduplication', 'compound'
    ]
    
    @validates('affix_type')
    def validate_affix_type(self, key, value):
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid affix type. Must be one of: {', '.join(self.VALID_TYPES)}")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.affix_type,
            "root_word": self.root_word.lemma,
            "affixed_word": self.affixed_word.lemma,
            "sources": self.sources.split(", ") if self.sources else [],
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class PartOfSpeech(db.Model):
    __tablename__ = 'parts_of_speech'

    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False, unique=True)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64), nullable=False)
    description = Column(Text)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "name_en": self.name_en,
            "name_tl": self.name_tl,
            "description": self.description
        }

class Language(db.Model):
    __tablename__ = 'languages'

    id = Column(Integer, primary_key=True)
    code = Column(String(16), unique=True, nullable=False)
    name = Column(String(64))
    words = relationship("Word", secondary=word_language_table)

# Register DDL event listeners
event.listen(
    Word.__table__,
    'after_create',
    word_search_ddl.execute_if(dialect='postgresql')
)

event.listen(
    Etymology.__table__,
    'after_create',
    etymology_search_ddl.execute_if(dialect='postgresql')
)

# Create timestamp triggers for each table with updated_at
for table in [Word.__table__, Definition.__table__, Etymology.__table__, Relation.__table__, DefinitionRelation.__table__, Affixation.__table__]:
    event.listen(
        table,
        'after_create',
        DDL(f"""
            CREATE TRIGGER update_{table.name}_timestamp
            BEFORE UPDATE ON {table.name}
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
        """).execute_if(dialect='postgresql')
    )

# Create timestamp function
event.listen(
    db.Model.metadata,
    'after_create',
    timestamp_trigger.execute_if(dialect='postgresql')
)

# Add database event listeners for automatic updates
@event.listens_for(Word, 'before_insert')
@event.listens_for(Word, 'before_update')
def update_word_metadata(mapper, connection, target):
    """Update word metadata before save."""
    target.complexity_score = target.calculate_complexity_score()
    target.update_search_vector()

@event.listens_for(Word, 'after_update')
def log_word_changes(mapper, connection, target):
    """Log significant word changes."""
    logger.info(f"Word updated: {target.lemma} (ID: {target.id})")

# Add database cleanup triggers
cleanup_trigger = DDL("""
    CREATE OR REPLACE FUNCTION cleanup_orphaned_records()
    RETURNS TRIGGER AS $$
    BEGIN
        DELETE FROM definitions WHERE word_id NOT IN (SELECT id FROM words);
        DELETE FROM etymologies WHERE word_id NOT IN (SELECT id FROM words);
        DELETE FROM relations WHERE from_word_id NOT IN (SELECT id FROM words) 
            OR to_word_id NOT IN (SELECT id FROM words);
        DELETE FROM affixations WHERE root_word_id NOT IN (SELECT id FROM words) 
            OR affixed_word_id NOT IN (SELECT id FROM words);
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql;

    CREATE TRIGGER cleanup_orphaned_records_trigger
    AFTER DELETE ON words
    FOR EACH STATEMENT
    EXECUTE FUNCTION cleanup_orphaned_records();
""")

event.listen(
    Word.__table__,
    'after_create',
    cleanup_trigger.execute_if(dialect='postgresql')
)

# Add database functions
@event.listens_for(Word, 'before_update')
def update_word_version(mapper, connection, target):
    """Update version when word is modified."""
    target.increment_version()
    target.data_quality_score = target.calculate_data_quality_score()
    target.data_hash = target.generate_data_hash()

# Add database performance optimizations
word_stats = Table('word_statistics', db.Model.metadata,
    Column('total_words', Integer),
    Column('total_definitions', Integer),
    Column('total_etymologies', Integer),
    Column('last_updated', DateTime(timezone=True))
)

def update_word_statistics():
    """Update word statistics materialized view."""
    stats = {
        'total_words': db.session.query(func.count(Word.id)).scalar(),
        'total_definitions': db.session.query(func.count(Definition.id)).scalar(),
        'total_etymologies': db.session.query(func.count(Etymology.id)).scalar(),
        'last_updated': datetime.now(UTC)
    }
    
    db.session.execute(
        word_stats.delete()
    )
    db.session.execute(
        word_stats.insert().values(**stats)
    )
    db.session.commit()

# Schedule statistics update
def schedule_stats_update():
    """Schedule periodic statistics update."""
    update_word_statistics()
    threading.Timer(3600, schedule_stats_update).start()  # Update every hour