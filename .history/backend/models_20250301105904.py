"""
Filipino Dictionary Database Models
"""

from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Boolean, DateTime, 
    func, Index, UniqueConstraint, DDL, event, text
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import re
import json
import hashlib
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional

# Set up SQLAlchemy
db = SQLAlchemy()

# Helper function for Baybayin romanization
def get_romanized_text(text: str) -> str:
    """Convert Baybayin text to romanized form."""
    if not text:
        return text
    if not any(0x1700 <= ord(c) <= 0x171F for c in text):
        return text
    
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
    for char in text:
        result.append(mapping.get(char, char))
    
    return ''.join(result)

# Create text search indexes DDL
word_search_ddl = DDL(
    "CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text)"
)

etymology_search_ddl = DDL(
    "CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes))"
)

# Timestamp trigger DDL
timestamp_trigger = DDL("""
    CREATE OR REPLACE FUNCTION update_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
""")

# Word model
class Word(db.Model):
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
    idioms = Column(JSONB, default=list)
    pronunciation_data = Column(JSONB)
    source_info = Column(JSONB, default=dict)
    data_hash = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    search_text = Column(TSVECTOR)

    # Relationships
    definitions = relationship("Definition", back_populates="word", cascade="all, delete-orphan")
    etymologies = relationship("Etymology", back_populates="word", cascade="all, delete-orphan")
    relations_from = relationship("Relation", foreign_keys="[Relation.from_word_id]", back_populates="from_word", cascade="all, delete-orphan")
    relations_to = relationship("Relation", foreign_keys="[Relation.to_word_id]", back_populates="to_word", cascade="all, delete-orphan")
    affixations_as_root = relationship("Affixation", foreign_keys="[Affixation.root_word_id]", back_populates="root_word", cascade="all, delete-orphan")
    affixations_as_affixed = relationship("Affixation", foreign_keys="[Affixation.affixed_word_id]", back_populates="affixed_word", cascade="all, delete-orphan")
    root_word = relationship("Word", remote_side=[id], backref="derived_words")

    __table_args__ = (
        Index('idx_words_baybayin', 'baybayin_form'),
        Index('idx_words_language_code', 'language_code'),
        Index('idx_words_lemma', 'lemma'),
        Index('idx_words_normalized_lemma', 'normalized_lemma'),
        Index('idx_words_romanized', 'romanized_form'),
        Index('idx_words_root', 'root_word_id'),
        Index('idx_words_search_text', 'search_text', postgresql_using='gin'),
        UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
    )

    @hybrid_property
    def is_root_word(self) -> bool:
        """Check if this is a root word (no parent)."""
        return self.root_word_id is None
    
    @hybrid_property
    def has_complete_data(self) -> bool:
        """Check if word has complete data (definitions, etymology, etc.)."""
        return bool(self.definitions and (self.etymologies or self.relations_from))
    
    @validates('lemma', 'normalized_lemma')
    def validate_lemma(self, key, value):
        """Validate lemma and normalized_lemma field values."""
        if not value or not isinstance(value, str):
            raise ValueError(f"{key} must be a non-empty string")
        if len(value) > 255:
            raise ValueError(f"{key} must be less than 255 characters")
        return value

    @validates('language_code')
    def validate_language_code(self, key, code):
        """Validate language code."""
        valid_codes = ['tl', 'ceb']
        if code not in valid_codes:
            raise ValueError(f"Invalid language code: {code}. Must be one of: {', '.join(valid_codes)}")
        return code
    
    def validate_baybayin(self):
        """Validate baybayin form and generate romanized form if needed."""
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")
        
        if self.baybayin_form and not re.match(r'^[\u1700-\u171F\s]*$', self.baybayin_form):
            raise ValueError("Invalid Baybayin characters")
            
        if self.has_baybayin and not self.romanized_form:
            self.romanized_form = get_romanized_text(self.baybayin_form)

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
        self.search_text = func.to_tsvector('simple', ' '.join(search_parts))
        
    def calculate_data_quality_score(self) -> int:
        """Calculate data quality score based on completeness."""
        score = 0
        
        # Basic data completeness
        if self.lemma and self.normalized_lemma:
            score += 10
        if self.definitions:
            score += min(len(self.definitions) * 5, 25)
        if self.etymologies:
            score += min(len(self.etymologies) * 5, 20)
            
        # Source quality
        if self.source_info:
            reliable_sources = ['kwf', 'upd', 'diksiyonaryo']
            source_json = self.source_info
            if isinstance(source_json, str):
                try:
                    source_json = json.loads(source_json)
                except:
                    source_json = {}
            score += sum(5 for source in source_json if source in reliable_sources)
            
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

    def get_tags_list(self) -> List[str]:
        """Get tags as a list."""
        return [tag.strip() for tag in self.tags.split(",")] if self.tags else []
    
    def get_idioms_list(self) -> List[Dict[str, Any]]:
        """Get idioms as a list of dictionaries."""
        if not self.idioms or self.idioms == '[]':
            return []
        
        if isinstance(self.idioms, str):
            try:
                return json.loads(self.idioms)
            except json.JSONDecodeError:
                return []
        return self.idioms
    
    def get_related_words(self, relation_type=None) -> List[Dict[str, Any]]:
        """Get related words of a specific type."""
        result = []
        
        for rel in self.relations_from:
            if relation_type and rel.relation_type.lower() != relation_type.lower():
                continue
                
            result.append({
                "id": rel.to_word.id,
                "word": rel.to_word.lemma,
                "normalized_lemma": rel.to_word.normalized_lemma,
                "language_code": rel.to_word.language_code,
                "relation_type": rel.relation_type,
                "has_baybayin": rel.to_word.has_baybayin,
                "baybayin_form": rel.to_word.baybayin_form if rel.to_word.has_baybayin else None
            })
            
        return result
    
    def get_etymology_summary(self) -> str:
        """Get a summary of the word's etymology."""
        if not self.etymologies:
            return "Unknown etymology"
            
        return "; ".join([etym.etymology_text for etym in self.etymologies])
    
    def to_dict(self, include_definitions=True, include_relations=True, 
                include_etymology=True, include_metadata=True) -> Dict[str, Any]:
        """Convert word to dictionary with configurable depth."""
        result = {
            "id": self.id,
            "lemma": self.lemma,
            "normalized_lemma": self.normalized_lemma,
            "language_code": self.language_code,
            "has_baybayin": self.has_baybayin,
            "baybayin_form": self.baybayin_form if self.has_baybayin else None,
            "romanized_form": self.romanized_form if self.has_baybayin else None,
            "preferred_spelling": self.preferred_spelling,
            "tags": self.get_tags_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_root_word": self.is_root_word
        }
        
        if include_metadata:
            # Add additional metadata
            idioms_data = self.get_idioms_list()
            result.update({
                "idioms": idioms_data,
                "pronunciation_data": self.pronunciation_data,
                "source_info": self.source_info,
                "data_quality_score": self.calculate_data_quality_score(),
                "data_hash": self.data_hash,
                "root_word_id": self.root_word_id
            })
        
        if include_definitions and self.definitions:
            result["definitions"] = [d.to_dict() for d in self.definitions]
            
        if include_etymology and self.etymologies:
            result["etymologies"] = [e.to_dict() for e in self.etymologies]
            
        if include_relations:
            # Organize relations
            relations = {
                "synonyms": self.get_related_words("synonym"),
                "antonyms": self.get_related_words("antonym"),
                "variants": self.get_related_words("variant"),
                "related": self.get_related_words("related"),
                "root": None,
                "derived": [],
                "affixations": {
                    "as_root": [],
                    "as_affixed": []
                }
            }
            
            # Add root word relation
            for rel in self.relations_from:
                if rel.relation_type.lower() == "derived_from":
                    relations["root"] = {
                        "id": rel.to_word.id,
                        "word": rel.to_word.lemma,
                        "normalized_lemma": rel.to_word.normalized_lemma,
                        "language_code": rel.to_word.language_code
                    }
                    break
                    
            # Add derived words
            for rel in self.relations_to:
                if rel.relation_type.lower() == "derived_from":
                    relations["derived"].append({
                        "id": rel.from_word.id,
                        "word": rel.from_word.lemma,
                        "normalized_lemma": rel.from_word.normalized_lemma,
                        "language_code": rel.from_word.language_code
                    })
            
            # Add affixations
            if hasattr(self, 'affixations_as_root'):
                for aff in self.affixations_as_root:
                    relations["affixations"]["as_root"].append({
                        "id": aff.id,
                        "affixed_word": aff.affixed_word.lemma,
                        "normalized_word": aff.affixed_word.normalized_lemma,
                        "type": aff.affix_type,
                        "sources": aff.sources.split(", ") if aff.sources else []
                    })
                    
            if hasattr(self, 'affixations_as_affixed'):
                for aff in self.affixations_as_affixed:
                    relations["affixations"]["as_affixed"].append({
                        "id": aff.id,
                        "root_word": aff.root_word.lemma,
                        "normalized_word": aff.root_word.normalized_lemma,
                        "type": aff.affix_type,
                        "sources": aff.sources.split(", ") if aff.sources else []
                    })
                    
            result["relations"] = relations
            
        return result


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
    standardized_pos = relationship("PartOfSpeech", back_populates="definitions")
    definition_relations = relationship("DefinitionRelation", back_populates="definition", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_definitions_pos', 'standardized_pos_id'),
        Index('idx_definitions_word_id', 'word_id'),
        UniqueConstraint('word_id', 'definition_text', 'standardized_pos_id', name='definitions_unique')
    )

    @validates('definition_text')
    def validate_definition_text(self, key, value):
        """Validate definition text."""
        if not value or not isinstance(value, str):
            raise ValueError("Definition text must be a non-empty string")
        return value.strip()

    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources."""
        if not value or not isinstance(value, str):
            raise ValueError("Sources must be a non-empty string")
        return value.strip()
    
    def get_examples_list(self) -> List[str]:
        """Get examples as a list of strings."""
        if not self.examples:
            return []
            
        try:
            examples = json.loads(self.examples)
            if isinstance(examples, list):
                return examples
            return [str(examples)]
        except json.JSONDecodeError:
            return [line.strip() for line in self.examples.split('\n') if line.strip()]
    
    def get_usage_notes_list(self) -> List[str]:
        """Get usage notes as a list of strings."""
        if not self.usage_notes:
            return []
            
        try:
            notes = json.loads(self.usage_notes)
            if isinstance(notes, list):
                return notes
            return [str(notes)]
        except json.JSONDecodeError:
            return [line.strip() for line in self.usage_notes.split('\n') if line.strip()]
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to dictionary."""
        result = {
            "id": self.id,
            "definition_text": self.definition_text,
            "original_pos": self.original_pos,
            "examples": self.get_examples_list(),
            "usage_notes": self.get_usage_notes_list(),
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add part of speech details if available
        if self.standardized_pos:
            result["part_of_speech"] = {
                "id": self.standardized_pos.id,
                "code": self.standardized_pos.code,
                "name_en": self.standardized_pos.name_en,
                "name_tl": self.standardized_pos.name_tl,
                "description": self.standardized_pos.description
            }
        
        # Add definition relations if available
        if hasattr(self, 'definition_relations') and self.definition_relations:
            result["related_words"] = [
                {
                    "word": rel.word.lemma,
                    "type": rel.relation_type,
                    "sources": rel.sources.split(", ") if rel.sources else []
                }
                for rel in self.definition_relations
            ]
        
        return result


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
        Index('idx_etymologies_word_id', 'word_id'),
        UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq')
    )
    
    @validates('etymology_text')
    def validate_etymology_text(self, key, value):
        """Validate etymology text."""
        if not value or not isinstance(value, str):
            raise ValueError("Etymology text must be a non-empty string")
        return value.strip()

    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources."""
        if not value or not isinstance(value, str):
            raise ValueError("Sources must be a non-empty string")
        return value.strip()
    
    def get_components_list(self) -> List[str]:
        """Get normalized components as a list of strings."""
        if not self.normalized_components:
            return []
            
        try:
            components = json.loads(self.normalized_components)
            if isinstance(components, list):
                return components
            return []
        except json.JSONDecodeError:
            if ';' in self.normalized_components:
                return [comp.strip() for comp in self.normalized_components.split(';') if comp.strip()]
            elif ',' in self.normalized_components:
                return [comp.strip() for comp in self.normalized_components.split(',') if comp.strip()]
            return [self.normalized_components] if self.normalized_components.strip() else []
    
    def get_language_codes_list(self) -> List[str]:
        """Get language codes as a list of strings."""
        return [lang.strip() for lang in self.language_codes.split(",")] if self.language_codes else []
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert etymology to dictionary."""
        return {
            "id": self.id,
            "etymology_text": self.etymology_text,
            "components": self.get_components_list(),
            "language_codes": self.get_language_codes_list(),
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


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
        Index('idx_relations_from_word_id', 'from_word_id'),
        Index('idx_relations_to_word_id', 'to_word_id'),
        Index('idx_relations_type', 'relation_type'),
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique')
    )

    VALID_TYPES = [
        'synonym', 'antonym', 'variant', 'derived_from', 
        'component_of', 'related', 'cognate', 'root_of'
    ]
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid relation type. Must be one of: {', '.join(self.VALID_TYPES)}")
        return value

    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources."""
        if not value or not isinstance(value, str):
            raise ValueError("Sources must be a non-empty string")
        return value.strip()
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary."""
        return {
            "id": self.id,
            "from_word": {
                "id": self.from_word.id,
                "lemma": self.from_word.lemma,
                "normalized_lemma": self.from_word.normalized_lemma,
                "language_code": self.from_word.language_code
            },
            "to_word": {
                "id": self.to_word.id,
                "lemma": self.to_word.lemma,
                "normalized_lemma": self.to_word.normalized_lemma,
                "language_code": self.to_word.language_code
            },
            "relation_type": self.relation_type,
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


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
    word = relationship("Word")

    __table_args__ = (
        Index('idx_definition_relations_definition_id', 'definition_id'),
        Index('idx_definition_relations_word_id', 'word_id'),
        UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique')
    )
    
    VALID_TYPES = [
        'synonym', 'antonym', 'variant', 'example', 
        'see_also', 'related', 'usage'
    ]

    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid relation type. Must be one of: {', '.join(self.VALID_TYPES)}")
        return value

    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources."""
        if not value or not isinstance(value, str):
            raise ValueError("Sources must be a non-empty string")
        return value.strip()
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert definition relation to dictionary."""
        return {
            "id": self.id,
            "definition_id": self.definition_id,
            "word": {
                "id": self.word.id,
                "lemma": self.word.lemma,
                "normalized_lemma": self.word.normalized_lemma,
                "language_code": self.word.language_code
            },
            "relation_type": self.relation_type,
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


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
    
    __table_args__ = (
        Index('idx_affixations_affixed_word_id', 'affixed_word_id'),
        Index('idx_affixations_root_word_id', 'root_word_id'),
        UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique')
    )

    VALID_TYPES = [
        'prefix', 'infix', 'suffix', 'circumfix', 
        'reduplication', 'compound'
    ]
    
    @validates('affix_type')
    def validate_affix_type(self, key, value):
        """Validate affix type."""
        if value not in self.VALID_TYPES:
            raise ValueError(f"Invalid affix type. Must be one of: {', '.join(self.VALID_TYPES)}")
        return value

    @validates('sources')
    def validate_sources(self, key, value):
        """Validate sources."""
        if not value or not isinstance(value, str):
            raise ValueError("Sources must be a non-empty string")
        return value.strip()
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert affixation to dictionary."""
        return {
            "id": self.id,
            "root_word": {
                "id": self.root_word.id,
                "lemma": self.root_word.lemma,
                "normalized_lemma": self.root_word.normalized_lemma,
                "language_code": self.root_word.language_code
            },
            "affixed_word": {
                "id": self.affixed_word.id,
                "lemma": self.affixed_word.lemma,
                "normalized_lemma": self.affixed_word.normalized_lemma,
                "language_code": self.affixed_word.language_code
            },
            "affix_type": self.affix_type,
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PartOfSpeech(db.Model):
    __tablename__ = 'parts_of_speech'

    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False, unique=True)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64), nullable=False)
    description = Column(Text)

    # Relationships
    definitions = relationship("Definition", back_populates="standardized_pos")
    
    __table_args__ = (
        Index('idx_parts_of_speech_code', 'code'),
        Index('idx_parts_of_speech_name', 'name_en', 'name_tl'),
        UniqueConstraint('code', name='parts_of_speech_code_uniq')
    )
    
    @validates('code')
    def validate_code(self, key, value):
        """Validate POS code."""
        if not value or not isinstance(value, str):
            raise ValueError("Code must be a non-empty string")
        if len(value) > 32:
            raise ValueError("Code must be less than 32 characters")
        return value
    
    @validates('name_en', 'name_tl')
    def validate_name(self, key, value):
        """Validate name fields."""
        if not value or not isinstance(value, str):
            raise ValueError(f"{key} must be a non-empty string")
        if len(value) > 64:
            raise ValueError(f"{key} must be less than 64 characters")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert part of speech to dictionary."""
        return {
            "id": self.id,
            "code": self.code,
            "name_en": self.name_en,
            "name_tl": self.name_tl,
            "description": self.description
        }


# Register DDL event listeners
@event.listens_for(
    Word.__table__,
    'after_create',
    word_search_ddl.execute_if(dialect='postgresql')
)

@event.listens_for(
    Etymology.__table__,
    'after_create',
    etymology_search_ddl.execute_if(dialect='postgresql')
)

# Create timestamp triggers for each table with updated_at
def create_timestamp_trigger(target, connection, **kw):
    connection.execute(text(f"""
        CREATE TRIGGER update_{target.name}_timestamp
        BEFORE UPDATE ON {target.name}
        FOR EACH ROW
        EXECUTE FUNCTION update_timestamp();
    """))

@event.listens_for(Word.__table__, 'after_create')
def create_word_timestamp_trigger(target, connection, **kw):
    create_timestamp_trigger(target, connection)

@event.listens_for(Definition.__table__, 'after_create')
def create_definition_timestamp_trigger(target, connection, **kw):
    create_timestamp_trigger(target, connection)

@event.listens_for(Etymology.__table__, 'after_create')
def create_etymology_timestamp_trigger(target, connection, **kw):
    create_timestamp_trigger(target, connection)

# Create timestamp function
@event.listens_for(
    db.Model.metadata,
    'after_create'
)
def create_timestamp_function(target, connection, **kw):
    connection.execute(timestamp_trigger)

# Word event listeners for data validation and updates
@event.listens_for(Word, 'before_insert')
@event.listens_for(Word, 'before_update')
def validate_word_data(mapper, connection, target):
    """Validate word data before save."""
    target.validate_baybayin()
    target.update_search_vector()
    target.data_hash = target.generate_data_hash()

# Create cleanup trigger to handle orphaned records
cleanup_trigger = DDL("""
    CREATE OR REPLACE FUNCTION cleanup_orphaned_records()
    RETURNS TRIGGER AS $
    BEGIN
        DELETE FROM definitions WHERE word_id NOT IN (SELECT id FROM words);
        DELETE FROM etymologies WHERE word_id NOT IN (SELECT id FROM words);
        DELETE FROM relations WHERE from_word_id NOT IN (SELECT id FROM words) 
            OR to_word_id NOT IN (SELECT id FROM words);
        DELETE FROM affixations WHERE root_word_id NOT IN (SELECT id FROM words) 
            OR affixed_word_id NOT IN (SELECT id FROM words);
        DELETE FROM definition_relations WHERE definition_id NOT IN (SELECT id FROM definitions)
            OR word_id NOT IN (SELECT id FROM words);
        RETURN NULL;
    END;
    $ LANGUAGE plpgsql;

    CREATE TRIGGER cleanup_orphaned_records_trigger
    AFTER DELETE ON words
    FOR EACH STATEMENT
    EXECUTE FUNCTION cleanup_orphaned_records();
""")

@event.listens_for(
    Word.__table__,
    'after_create'
)
def create_cleanup_trigger(target, connection, **kw):
    connection.execute(cleanup_trigger)

# Create additional indexes for better query performance
@event.listens_for(Word.__table__, 'after_create')
def create_word_indexes(target, connection, **kw):
    """Create additional indexes for better query performance."""
    connection.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_word_normalized_lemma ON words (normalized_lemma);
        CREATE INDEX IF NOT EXISTS idx_word_language_code ON words (language_code);
        CREATE INDEX IF NOT EXISTS idx_word_has_baybayin ON words (has_baybayin);
        CREATE INDEX IF NOT EXISTS idx_word_search_text ON words USING gin(search_text);
        CREATE INDEX IF NOT EXISTS idx_word_relations ON relations (from_word_id, to_word_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_word_affixations ON affixations (root_word_id, affixed_word_id, affix_type);
        CREATE INDEX IF NOT EXISTS idx_word_etymologies ON etymologies (word_id);
        CREATE INDEX IF NOT EXISTS idx_word_definitions ON definitions (word_id, standardized_pos_id);
        CREATE INDEX IF NOT EXISTS idx_definition_relations ON definition_relations (definition_id, word_id);
    """))