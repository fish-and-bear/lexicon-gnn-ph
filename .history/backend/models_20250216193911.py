import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, JSON, ARRAY, Boolean, DateTime, func, Index, UniqueConstraint, DDL, event
from sqlalchemy.orm import relationship, sessionmaker, validates, declarative_base, scoped_session
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from dotenv import load_dotenv
import re
from backend.language_systems import LanguageSystem
from backend.dictionary_manager import get_romanized_text
from flask_sqlalchemy import SQLAlchemy

load_dotenv()

# Create SQLAlchemy instance
db = SQLAlchemy()

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

class Word(db.Model):
    __tablename__ = 'words'
    
    id = Column(Integer, primary_key=True)
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
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

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

class Definition(db.Model):
    __tablename__ = 'definitions'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    definition_text = Column(Text, nullable=False)
    original_pos = Column(Text)
    standardized_pos_id = Column(Integer, ForeignKey('parts_of_speech.id'))
    examples = Column(Text)
    usage_notes = Column(Text)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

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

class Etymology(db.Model):
    __tablename__ = 'etymologies'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = Column(Text, nullable=False)
    normalized_components = Column(Text)
    language_codes = Column(Text)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    word = relationship("Word", back_populates="etymologies")

    __table_args__ = (
        Index('idx_etymologies_word', 'word_id'),
        UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq'),
    )

    def extract_components(self):
        """Extract and normalize etymology components."""
        from dictionary_manager import extract_etymology_components
        components = extract_etymology_components(self.etymology_text)
        self.normalized_components = ", ".join(components)

    def extract_meaning(self):
        """Extract meaning from etymology text."""
        from dictionary_manager import extract_meaning
        text, meaning = extract_meaning(self.etymology_text)
        return meaning

class Relation(db.Model):
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True)
    from_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    from_word = relationship("Word", foreign_keys=[from_word_id], back_populates="relations_from")
    to_word = relationship("Word", foreign_keys=[to_word_id], back_populates="relations_to")

    __table_args__ = (
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
    )

class DefinitionRelation(db.Model):
    __tablename__ = 'definition_relations'
    
    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    definition = relationship("Definition", back_populates="definition_relations")
    word = relationship("Word", back_populates="definition_relations")

class Affixation(db.Model):
    __tablename__ = 'affixations'
    
    id = Column(Integer, primary_key=True)
    root_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affixed_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affix_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    root_word = relationship("Word", foreign_keys=[root_word_id], back_populates="affixations_as_root")
    affixed_word = relationship("Word", foreign_keys=[affixed_word_id], back_populates="affixations_as_affixed")

class PartOfSpeech(db.Model):
    __tablename__ = 'parts_of_speech'

    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False, unique=True)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64), nullable=False)
    description = Column(Text)

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