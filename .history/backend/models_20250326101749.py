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
import logging

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
    root_word_id = Column(Integer, ForeignKey('words.id'), nullable=True)
    preferred_spelling = Column(String(255))
    tags = Column(Text)
    idioms = Column(JSONB, default=list)
    pronunciation_data = Column(JSONB)
    source_info = Column(JSONB, default=dict)
    data_hash = Column(Text)
    search_text = Column(TSVECTOR)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    definitions = relationship("Definition", back_populates="word", cascade="all, delete-orphan")
    etymologies = relationship("Etymology", back_populates="word", cascade="all, delete-orphan")
    relations_from = relationship("Relation", foreign_keys="[Relation.from_word_id]", back_populates="from_word", cascade="all, delete-orphan")
    relations_to = relationship("Relation", foreign_keys="[Relation.to_word_id]", back_populates="to_word", cascade="all, delete-orphan")
    affixations_as_root = relationship("Affixation", foreign_keys="[Affixation.root_word_id]", back_populates="root_word", cascade="all, delete-orphan")
    affixations_as_affixed = relationship("Affixation", foreign_keys="[Affixation.affixed_word_id]", back_populates="affixed_word", cascade="all, delete-orphan")
    root_word = relationship("Word", remote_side=[id], backref="derived_words")

    __table_args__ = (
        UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
        Index('idx_words_lemma', 'lemma'),
        Index('idx_words_normalized', 'normalized_lemma'),
        Index('idx_words_baybayin', 'baybayin_form', postgresql_where=(has_baybayin == True)),
        Index('idx_words_romanized', 'romanized_form'),
        Index('idx_words_language', 'language_code'),
        Index('idx_words_root', 'root_word_id'),
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
        
        if self.baybayin_form and not re.match(r'^[\u1700-\u171F\\\\s]*$', self.baybayin_form):
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
                "hyponyms": self.get_related_words("hypernym_of"),
                "hypernyms": self.get_related_words("hyponym_of"),
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
    tags = Column(Text)
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
        Index('idx_definitions_tags', 'tags'),
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
    
    def get_tags_list(self) -> List[str]:
        """Get tags as a list of strings."""
        if not self.tags:
            return []
            
        try:
            tags = json.loads(self.tags)
            if isinstance(tags, list):
                return tags
            return [str(tags)]
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
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
            "tags": self.get_tags_list(),
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
    etymology_structure = Column(Text)
    language_codes = Column(Text)
    sources = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    word = relationship("Word", back_populates="etymologies")

    __table_args__ = (
        UniqueConstraint('word_id', 'etymology_text', name='etymologies_wordid_etymtext_uniq'),
        Index('idx_etymologies_word', 'word_id'),
        Index('idx_etymologies_structure', 'etymology_structure')
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
    
    def get_etymology_structure(self) -> Optional[Dict]:
        """Get the structured etymology data if available."""
        if not self.etymology_structure:
            return None
            
        try:
            return json.loads(self.etymology_structure)
        except json.JSONDecodeError:
            return None
    
    def get_language_codes_list(self) -> List[str]:
        """Get language codes as a list of strings."""
        return [lang.strip() for lang in self.language_codes.split(",")] if self.language_codes else []
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list of strings."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert etymology to dictionary."""
        result = {
            "id": self.id,
            "etymology_text": self.etymology_text,
            "components": self.get_components_list(),
            "language_codes": self.get_language_codes_list(),
            "sources": self.get_sources_list(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Add etymology structure if available
        etymology_structure = self.get_etymology_structure()
        if etymology_structure:
            result["etymology_structure"] = etymology_structure
            
        return result


class Relation(db.Model):
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True)
    from_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(64), nullable=False)
    sources = Column(Text, nullable=False)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    from_word = relationship("Word", foreign_keys=[from_word_id], back_populates="relations_from")
    to_word = relationship("Word", foreign_keys=[to_word_id], back_populates="relations_to")

    __table_args__ = (
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique'),
        Index('idx_relations_from', 'from_word_id'),
        Index('idx_relations_to', 'to_word_id'),
        Index('idx_relations_type', 'relation_type'),
        Index('idx_relations_metadata', 'metadata', postgresql_using='gin')
    )

    VALID_TYPES = [
        # Basic semantic relationships
        'synonym', 'antonym', 'variant', 'spelling_variant',
        
        # Hierarchical relationships
        'hypernym_of', 'hyponym_of', 'meronym_of', 'holonym_of',
        
        # Derivational relationships
        'derived_from', 'root_of', 'derived', 'base_of',
        
        # Etymology relationships
        'borrowed_from', 'loaned_to', 'cognate', 'descendant_of', 'ancestor_of',
        
        # Structural relationships
        'component_of', 'abbreviation_of', 'has_abbreviation', 'initialism_of', 'has_initialism',
        
        # General relationships
        'related'
    ]
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        if value not in self.VALID_TYPES:
            logger = logging.getLogger(__name__)
            logger.warning(f"Non-standard relation type encountered: '{value}'. " +
                          f"Standard types are: {', '.join(self.VALID_TYPES)}")
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
        UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique'),
        Index('idx_def_relations_def', 'definition_id'),
        Index('idx_def_relations_word', 'word_id')
    )

    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        standard_types = ['synonym', 'antonym', 'variant', 'example', 'see_also', 'usage']
        if value not in standard_types:
            logger = logging.getLogger(__name__)
            logger.warning(f"Non-standard definition relation type encountered: '{value}'. " +
                          f"Standard types are: {', '.join(standard_types)}")
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
        UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique'),
        Index('idx_affixations_root', 'root_word_id'),
        Index('idx_affixations_affixed', 'affixed_word_id')
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

# Define timestamp functions with error handling
def create_timestamp_function(target, connection, **kw):
    """Create the timestamp update function."""
    try:
        connection.execute(timestamp_trigger)
        print(f"Created timestamp update function successfully")
    except Exception as e:
        print(f"WARNING: Error creating timestamp function: {e}")
        # Try alternate syntax for different PostgreSQL versions
        try:
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION update_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))
            print("Created timestamp function with alternate syntax")
        except Exception as alt_e:
            print(f"FAILED to create timestamp function with alternate syntax: {alt_e}")
            print("Timestamp updates may not work correctly")

def create_timestamp_trigger(target, connection, **kw):
    """Create update trigger for timestamps."""
    try:
        # First drop the trigger if it exists to avoid conflicts
        try:
            connection.execute(text(f"""
                DROP TRIGGER IF EXISTS update_{target.name}_timestamp ON {target.name};
            """))
        except Exception:
            pass  # Ignore errors on drop
            
        # Create the trigger with proper error handling
        connection.execute(text(f"""
            CREATE TRIGGER update_{target.name}_timestamp
            BEFORE UPDATE ON {target.name}
                FOR EACH ROW
                EXECUTE FUNCTION update_timestamp();
        """))
        print(f"Created timestamp trigger for {target.name} table")
    except Exception as e:
        print(f"WARNING: Error creating timestamp trigger for {target.name}: {e}")
        # Try alternate syntax for different PostgreSQL versions
        try:
            connection.execute(text(f"""
                CREATE TRIGGER update_{target.name}_timestamp
                BEFORE UPDATE ON {target.name}
                    FOR EACH ROW
                    EXECUTE PROCEDURE update_timestamp();
            """))
            print(f"Created timestamp trigger for {target.name} with alternate syntax")
        except Exception as alt_e:
            print(f"FAILED to create timestamp trigger for {target.name} with alternate syntax: {alt_e}")
            print(f"Timestamps for {target.name} will not update automatically")

# Register timestamp function and triggers
event.listen(db.Model.metadata, 'after_create', create_timestamp_function)
event.listen(Word.__table__, 'after_create', create_timestamp_trigger)
event.listen(Definition.__table__, 'after_create', create_timestamp_trigger)
event.listen(Etymology.__table__, 'after_create', create_timestamp_trigger)

# Word event listeners for data validation
def validate_word_data(mapper, connection, target):
    """Validate word data before save."""
    target.validate_baybayin()
    target.update_search_vector()
    target.data_hash = target.generate_data_hash()

event.listen(Word, 'before_insert', validate_word_data)
event.listen(Word, 'before_update', validate_word_data)

# Define cleanup trigger for PostgreSQL
cleanup_trigger = DDL("""
CREATE OR REPLACE FUNCTION cleanup_word_data() RETURNS TRIGGER AS $$
BEGIN
    -- Remove any orphaned relations
    DELETE FROM relations WHERE 
        from_word_id NOT IN (SELECT id FROM words) OR 
        to_word_id NOT IN (SELECT id FROM words);
    
    -- Remove any orphaned definition relations
    DELETE FROM definition_relations WHERE 
        word_id NOT IN (SELECT id FROM words) OR
        definition_id NOT IN (SELECT id FROM definitions);
    
    -- Remove any orphaned affixations
    DELETE FROM affixations WHERE 
        root_word_id NOT IN (SELECT id FROM words) OR 
        affixed_word_id NOT IN (SELECT id FROM words);
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER cleanup_word_data_trigger
AFTER DELETE ON words
FOR EACH STATEMENT
EXECUTE FUNCTION cleanup_word_data();
""")

# Create cleanup trigger
event.listen(
    Word.__table__,
    'after_create',
    cleanup_trigger.execute_if(dialect='postgresql')
)

# Create additional indexes
def create_word_indexes(target, connection, **kw):
    """Create additional indexes for better query performance."""
    # Define indexes to create
    indexes = [
        {"name": "idx_word_normalized_lemma", "table": "words", "columns": "(normalized_lemma)", "type": ""},
        {"name": "idx_word_language_code", "table": "words", "columns": "(language_code)", "type": ""},
        {"name": "idx_word_has_baybayin", "table": "words", "columns": "(has_baybayin)", "type": ""},
        {"name": "idx_word_search_text", "table": "words", "columns": "(search_text)", "type": "USING gin"},
        {"name": "idx_word_relations", "table": "relations", "columns": "(from_word_id, to_word_id, relation_type)", "type": ""},
        {"name": "idx_word_affixations", "table": "affixations", "columns": "(root_word_id, affixed_word_id, affix_type)", "type": ""},
        {"name": "idx_word_etymologies", "table": "etymologies", "columns": "(word_id)", "type": ""},
        {"name": "idx_word_definitions", "table": "definitions", "columns": "(word_id, standardized_pos_id)", "type": ""},
        {"name": "idx_definition_relations", "table": "definition_relations", "columns": "(definition_id, word_id)", "type": ""}
    ]
    
    # Try to create each index separately with error handling
    for idx in indexes:
        try:
            sql = f"CREATE INDEX IF NOT EXISTS {idx['name']} ON {idx['table']} {idx['type']} {idx['columns']}"
            connection.execute(text(sql))
            print(f"Created or verified index: {idx['name']}")
        except Exception as e:
            print(f"WARNING: Failed to create index {idx['name']}: {e}")
            # Try a fallback approach for the search_text GIN index which is most critical
            if idx['name'] == 'idx_word_search_text':
                try:
                    # Try with a more compatible syntax
                    connection.execute(text("""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM pg_indexes 
                                WHERE indexname = 'idx_word_search_text'
                            ) THEN
                                CREATE INDEX idx_word_search_text ON words USING gin(search_text);
                            END IF;
                        EXCEPTION WHEN OTHERS THEN
                            -- Log error and continue
                            RAISE NOTICE 'Error creating search_text index: %', SQLERRM;
                        END $$;
                    """))
                    print("Created search text index with alternate method")
                except Exception as alt_e:
                    print(f"CRITICAL: Could not create search text index: {alt_e}")
                    print("Full-text search functionality will be limited")

event.listen(Word.__table__, 'after_create', create_word_indexes)

# Add event listeners to update search_text when word changes
@event.listens_for(Word, 'before_insert')
@event.listens_for(Word, 'before_update')
def update_word_search_text(mapper, connection, target):
    """Update the search_text column before insert or update."""
    if hasattr(target, 'lemma') and target.lemma:
        search_parts = [
            target.lemma,
            target.normalized_lemma,
            target.baybayin_form if target.baybayin_form else '',
            target.romanized_form if target.romanized_form else ''
        ]
        search_text = ' '.join(p for p in search_parts if p)
        target.search_text = func.to_tsvector('simple', search_text)

# Add event listeners for schema compatibility
@event.listens_for(db.metadata, 'after_create')
def create_constraints(target, engine_or_conn, **kw):
    """Create any necessary constraints for database compatibility."""
    try:
        # Create SQL statements with proper escape sequences
        baybayin_check_sql = """
            ALTER TABLE words DROP CONSTRAINT IF EXISTS baybayin_form_check;
            ALTER TABLE words ADD CONSTRAINT baybayin_form_check CHECK (
                (has_baybayin = FALSE AND baybayin_form IS NULL) OR 
                (has_baybayin = TRUE AND baybayin_form IS NOT NULL)
            );
        """
        
        baybayin_regex_sql = """
            -- First clean up any invalid data
            UPDATE words 
            SET baybayin_form = NULL, has_baybayin = FALSE
            WHERE has_baybayin = TRUE AND (baybayin_form IS NULL OR baybayin_form !~ '^[\u1700-\u171F\\\\s]*$');
            
            -- Then apply the constraint
            ALTER TABLE words DROP CONSTRAINT IF EXISTS baybayin_form_regex;
            ALTER TABLE words ADD CONSTRAINT baybayin_form_regex CHECK (
                baybayin_form ~ '^[\u1700-\u171F\\\\s]*$' OR baybayin_form IS NULL
            );
        """
        
        search_text_sql = """
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'words' AND column_name = 'search_text'
                ) THEN
                    ALTER TABLE words ADD COLUMN search_text TSVECTOR;
                    CREATE INDEX idx_words_search ON words USING gin(search_text);
                END IF;
            END $$;
        """
        
        # Handle multiple SQLAlchemy versions and connection types
        try:
            # Handle being passed either an engine or connection
            if hasattr(engine_or_conn, 'connect'):
                # It's an engine - SQLAlchemy 2.0+ style
                try:
                    with engine_or_conn.connect() as conn:
                        conn.execute(text(baybayin_check_sql))
                        conn.execute(text(baybayin_regex_sql))
                        conn.execute(text(search_text_sql))
                        
                        # Explicitly commit in SQLAlchemy 2.0+
                        try:
                            conn.commit()
                        except Exception:
                            # Some versions might auto-commit or not have this method
                            pass
                except Exception as e:
                    print(f"Error executing constraints with SQLAlchemy 2.0 API: {e}")
                    # Fallback to legacy approach
                    with engine_or_conn.begin() as conn:
                        conn.execute(text(baybayin_check_sql))
                        conn.execute(text(baybayin_regex_sql))
                        conn.execute(text(search_text_sql))
            else:
                # Assume it's a connection (SQLAlchemy 1.x style)
                engine_or_conn.execute(text(baybayin_check_sql))
                engine_or_conn.execute(text(baybayin_regex_sql))
                engine_or_conn.execute(text(search_text_sql))
                
                # Try to commit if applicable
                try:
                    if hasattr(engine_or_conn, 'commit'):
                        engine_or_conn.commit()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error during constraint execution: {e}")
            # Last resort attempt
            try:
                if hasattr(engine_or_conn, 'execute'):
                    engine_or_conn.execute(text("""
                        -- Simple constraint version that should work in most PostgreSQL setups
                        ALTER TABLE IF EXISTS words DROP CONSTRAINT IF EXISTS baybayin_form_check;
                        ALTER TABLE IF EXISTS words DROP CONSTRAINT IF EXISTS baybayin_form_regex;
                    """))
            except Exception:
                pass
                        
    except Exception as e:
        print(f"Error applying constraints: {e}")
        print("Application will continue, but some database constraints may not be applied correctly.")
        # Continue anyway to avoid blocking application startup

# Define constraints to be applied after table creation (moved from table_args)
word_baybayin_check = DDL("""
    ALTER TABLE words ADD CONSTRAINT IF NOT EXISTS baybayin_form_check CHECK (
        (has_baybayin = FALSE AND baybayin_form IS NULL) OR 
        (has_baybayin = TRUE AND baybayin_form IS NOT NULL)
    )
""")

word_baybayin_regex = DDL("""
    -- First clean up any invalid data
    UPDATE words 
    SET baybayin_form = NULL, has_baybayin = FALSE
    WHERE has_baybayin = TRUE AND (baybayin_form IS NULL OR baybayin_form !~ '^[\u1700-\u171F\\\\s]*$');
    
    -- Then apply the constraint
    ALTER TABLE words ADD CONSTRAINT IF NOT EXISTS baybayin_form_regex CHECK (
        baybayin_form ~ '^[\u1700-\u171F\\\\s]*$' OR baybayin_form IS NULL
    )
""")

# Register events to apply constraints after table creation
event.listen(Word.__table__, 'after_create', word_baybayin_check)
event.listen(Word.__table__, 'after_create', word_baybayin_regex)

# Define index to be applied after table creation
etymology_langs_index = DDL("CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes))")

# Register event to apply index after table creation
event.listen(Etymology.__table__, 'after_create', etymology_langs_index)

# Initialize models
def init_app(app):
    """Initialize the models with the application."""
    # Set up database connection with robust error handling
    try:
        db.init_app(app)
        print("SQLAlchemy successfully initialized with Flask application")
    except Exception as e:
        print(f"WARNING: Error initializing SQLAlchemy with Flask app: {e}")
        print("Attempting to continue anyway...")
    
    try:
        with app.app_context():
            # Try to create all tables if they don't exist
            try:
                print("Creating database tables if they don't exist...")
                db.create_all()
                print("Database tables successfully created or verified")
            except Exception as table_error:
                print(f"WARNING: Error creating database tables: {table_error}")
                print("Some tables may not have been created correctly")
                
                # Try connecting to the database directly to verify basic connectivity
                try:
                    engine = db.engine
                    with engine.connect() as conn:
                        result = conn.execute(text("SELECT 1"))
                        if result:
                            print("Database connection is working, but table creation failed")
                        conn.commit()  # Ensure any transactions are properly closed
                except Exception as conn_error:
                    print(f"ERROR: Could not connect to database: {conn_error}")
                    print("Database may be unavailable or configuration incorrect")
                    
            # Apply schema constraints with error handling
            try:
                print("Applying database schema constraints...")
                create_constraints(None, db.engine)
                print("Database constraints successfully applied")
            except Exception as e:
                print(f"WARNING: Error during schema constraints application: {e}")
                print("Application will continue, but some database validations will be performed in application code instead")
                
    except Exception as e:
        print(f"WARNING: Database initialization error: {e}")
        print("Application will start with limited database functionality")
        
    # Register validation event listeners with error handling
    try:
        # Re-register event listeners for Word validation to ensure they're active
        event.listen(Word, 'before_insert', validate_word_data)
        event.listen(Word, 'before_update', validate_word_data)
        print("Model validation event listeners registered")
    except Exception as e:
        print(f"WARNING: Error registering validation event listeners: {e}")
        print("Data validation will be limited")