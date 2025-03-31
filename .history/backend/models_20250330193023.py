"""
Filipino Dictionary Database Models with enhanced functionality and data validation.
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
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional
import logging
from unidecode import unidecode

# Set up logging
logger = logging.getLogger(__name__)

# Set up SQLAlchemy
db = SQLAlchemy()

# Helper functions
def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison."""
    if not text:
        return ""
    # Convert to lowercase and remove diacritics
    normalized = unidecode(text.lower())
    # Remove non-word characters except hyphens
    normalized = re.sub(r'[^\w\s\-]', '', normalized)
    return normalized.strip()

def get_romanized_text(text: str) -> str:
    """Convert Baybayin text to romanized form with enhanced accuracy."""
    if not text:
        return text
    if not any(0x1700 <= ord(c) <= 0x171F for c in text):
        return text
    
    # Enhanced mapping with more accurate representations
    mapping = {
        'ᜀ': 'a', 'ᜁ': 'i', 'ᜂ': 'u', 'ᜃ': 'ka', 'ᜄ': 'ga', 
        'ᜅ': 'nga', 'ᜆ': 'ta', 'ᜇ': 'da', 'ᜈ': 'na',
        'ᜉ': 'pa', 'ᜊ': 'ba', 'ᜋ': 'ma', 'ᜌ': 'ya', 
        'ᜎ': 'la', 'ᜏ': 'wa', 'ᜐ': 'sa', 'ᜑ': 'ha',
        'ᜒ': 'i', 'ᜓ': 'u', '᜔': '', '᜵': ',', '᜶': '.',
        # Additional combinations
        'ᜃ᜔': 'k', 'ᜄ᜔': 'g', 'ᜅ᜔': 'ng',
        'ᜆ᜔': 't', 'ᜇ᜔': 'd', 'ᜈ᜔': 'n',
        'ᜉ᜔': 'p', 'ᜊ᜔': 'b', 'ᜋ᜔': 'm',
        'ᜌ᜔': 'y', 'ᜎ᜔': 'l', 'ᜏ᜔': 'w',
        'ᜐ᜔': 's', 'ᜑ᜔': 'h'
    }
    
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
            result.append(text[i])
            i += 1
    
    return ''.join(result)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)
    
    # Calculate Levenshtein distance
    len1, len2 = len(text1), len(text2)
    if len1 == 0:
        return 0.0
    if len2 == 0:
        return 0.0
    
    # Initialize matrix
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    
    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if text1[i-1] == text2[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,    # deletion
                    matrix[i][j-1] + 1,    # insertion
                    matrix[i-1][j-1] + 1   # substitution
                )
    
    # Calculate similarity score
    max_len = max(len1, len2)
    distance = matrix[len1][len2]
    similarity = 1 - (distance / max_len)
    
    return round(similarity, 3)

# Create text search indexes DDL
word_search_ddl = DDL("""
    CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
    CREATE INDEX IF NOT EXISTS idx_words_normalized ON words USING gin(to_tsvector('simple', normalized_lemma));
    CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words USING gin(to_tsvector('simple', baybayin_form)) 
    WHERE has_baybayin = true;
""")

etymology_search_ddl = DDL("""
    CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));
    CREATE INDEX IF NOT EXISTS idx_etymologies_components ON etymologies USING gin(to_tsvector('simple', normalized_components));
""")

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
    """Enhanced word model with additional fields and functionality."""
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
    metadata = Column(JSONB, default=dict)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())

    # Relationships
    definitions = relationship(
        "Definition",
        back_populates="word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    etymologies = relationship(
        "Etymology",
        back_populates="word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    relations_from = relationship(
        "Relation",
        foreign_keys="[Relation.from_word_id]",
        back_populates="from_word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    relations_to = relationship(
        "Relation",
        foreign_keys="[Relation.to_word_id]",
        back_populates="to_word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    affixations_as_root = relationship(
        "Affixation",
        foreign_keys="[Affixation.root_word_id]",
        back_populates="root_word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    affixations_as_affixed = relationship(
        "Affixation",
        foreign_keys="[Affixation.affixed_word_id]",
        back_populates="affixed_word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    pronunciations = relationship(
        "Pronunciation",
        back_populates="word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    root_word = relationship(
        "Word",
        remote_side=[id],
        backref=backref("derived_words", lazy="joined"),
        lazy="joined"
    )

    __table_args__ = (
        Index('idx_words_lemma', 'lemma'),
        Index('idx_words_normalized', 'normalized_lemma'),
        Index('idx_words_language', 'language_code'),
        Index('idx_words_baybayin', 'baybayin_form', postgresql_where=text('has_baybayin = true')),
        Index('idx_words_verification', 'verification_status'),
        Index('idx_words_created', 'created_at'),
        Index('idx_words_search', 'search_text', postgresql_using='gin'),
        UniqueConstraint('normalized_lemma', 'language_code', name='words_normalized_lang_unique')
    )

    @hybrid_property
    def is_root_word(self) -> bool:
        """Check if this is a root word (no parent)."""
        return self.root_word_id is None
    
    @hybrid_property
    def has_complete_data(self) -> bool:
        """Check if word has complete data."""
        return bool(
            self.definitions and
            (self.etymologies or self.relations_from) and
            self.source_info
        )
    
    @hybrid_property
    def is_verified(self) -> bool:
        """Check if word is verified."""
        return self.verification_status == 'verified'
    
    @validates('lemma', 'normalized_lemma')
    def validate_lemma(self, key, value):
        """Validate lemma fields."""
        if not value or not isinstance(value, str):
            raise ValueError(f"{key} must be a non-empty string")
        if len(value) > 255:
            raise ValueError(f"{key} must be less than 255 characters")
        return value.strip()
    
    @validates('language_code')
    def validate_language_code(self, key, code):
        """Validate language code."""
        valid_codes = ['tl', 'ceb', 'chv', 'en', 'es', 'spl', 'psp']
        if code not in valid_codes:
            raise ValueError(f"Invalid language code: {code}")
        return code
    
    @validates('verification_status')
    def validate_verification_status(self, key, status):
        """Validate verification status."""
        valid_statuses = ['unverified', 'verified', 'needs_review', 'disputed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid verification status: {status}")
        return status
    
    def validate_baybayin(self):
        """Validate Baybayin form and generate romanized form."""
        if self.has_baybayin and not self.baybayin_form:
            raise ValueError("Baybayin form required when has_baybayin is True")
        
        if self.baybayin_form:
            if not re.match(r'^[\u1700-\u171F\s]*$', self.baybayin_form):
                raise ValueError("Invalid Baybayin characters")
            
            if not self.romanized_form:
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
            self.romanized_form or '',
            ' '.join(self.get_tags_list()),
            ' '.join(tag for d in self.definitions for tag in d.get_tags_list())
        ]
        self.search_text = func.to_tsvector('simple', ' '.join(filter(None, search_parts)))
    
    def calculate_data_quality_score(self) -> int:
        """Calculate comprehensive data quality score."""
        score = 0
        
        # Basic data completeness (30 points)
        if self.lemma and self.normalized_lemma:
            score += 15
        if self.language_code:
            score += 5
        if self.has_baybayin and self.baybayin_form and self.romanized_form:
            score += 5
        if self.pronunciation_data:
            score += 5
            
        # Definitions quality (25 points)
        if self.definitions:
            def_score = min(len(self.definitions) * 5, 15)
            def_score += sum(2 for d in self.definitions if d.examples)
            def_score += sum(2 for d in self.definitions if d.usage_notes)
            def_score += sum(1 for d in self.definitions if d.tags)
            score += min(def_score, 25)
            
        # Etymology quality (20 points)
        if self.etymologies:
            etym_score = min(len(self.etymologies) * 5, 10)
            etym_score += sum(2 for e in self.etymologies if e.normalized_components)
            etym_score += sum(2 for e in self.etymologies if e.language_codes)
            etym_score += sum(1 for e in self.etymologies if e.etymology_structure)
            score += min(etym_score, 20)
            
        # Relationships quality (15 points)
        rel_score = 0
        if self.relations_from or self.relations_to:
            rel_count = len(self.relations_from) + len(self.relations_to)
            rel_score += min(rel_count * 2, 10)
        if self.affixations_as_root or self.affixations_as_affixed:
            aff_count = len(self.affixations_as_root) + len(self.affixations_as_affixed)
            rel_score += min(aff_count * 1, 5)
        score += rel_score
        
        # Additional features (10 points)
        if self.idioms and self.idioms != '[]':
            score += 3
        if self.source_info and self.source_info != '{}':
            score += 3
        if self.tags:
            score += 2
        if self.is_verified:
            score += 2
            
        return min(score, 100)
    
    def calculate_similarity_score(self, other_word: 'Word') -> float:
        """Calculate similarity score with another word."""
        if not other_word:
            return 0.0
            
        # Calculate base similarity from lemmas
        base_similarity = calculate_text_similarity(self.lemma, other_word.lemma)
        
        # Add points for shared features
        feature_score = 0.0
        
        # Check language
        if self.language_code == other_word.language_code:
            feature_score += 0.1
            
        # Check Baybayin
        if self.has_baybayin and other_word.has_baybayin:
            baybayin_similarity = calculate_text_similarity(
                self.baybayin_form,
                other_word.baybayin_form
            )
            feature_score += baybayin_similarity * 0.1
            
        # Check etymologies
        if self.etymologies and other_word.etymologies:
            shared_languages = set(
                lang for e in self.etymologies for lang in e.get_language_codes_list()
            ).intersection(
                lang for e in other_word.etymologies for lang in e.get_language_codes_list()
            )
            if shared_languages:
                feature_score += len(shared_languages) * 0.05
                
        # Check definitions
        if self.definitions and other_word.definitions:
            def_similarity = max(
                calculate_text_similarity(d1.definition_text, d2.definition_text)
                for d1 in self.definitions
                for d2 in other_word.definitions
            )
            feature_score += def_similarity * 0.2
            
        # Calculate final score
        final_score = base_similarity * 0.6 + feature_score
        return round(min(final_score, 1.0), 3)
    
    def generate_data_hash(self) -> str:
        """Generate a hash of the word's data for change detection."""
        data_parts = [
            self.lemma,
            self.normalized_lemma,
            self.language_code,
            self.baybayin_form or '',
            self.romanized_form or '',
            str(sorted([d.definition_text for d in self.definitions])),
            str(sorted([e.etymology_text for e in self.etymologies])),
            str(sorted([r.relation_type for r in self.relations_from])),
            str(sorted([r.relation_type for r in self.relations_to])),
            str(self.source_info or {}),
            str(self.metadata or {})
        ]
        data_string = '|'.join(data_parts)
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def get_tags_list(self) -> List[str]:
        """Get tags as a list."""
        if not self.tags:
            return []
        try:
            return json.loads(self.tags) if isinstance(self.tags, str) else self.tags
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def get_idioms_list(self) -> List[Dict[str, Any]]:
        """Get idioms as a list of dictionaries."""
        if not self.idioms or self.idioms == '[]':
            return []
        
        try:
            idioms = json.loads(self.idioms) if isinstance(self.idioms, str) else self.idioms
            return [
                {
                    'idiom': idiom.get('idiom', '') or idiom.get('text', ''),
                    'meaning': idiom.get('meaning', ''),
                    'examples': idiom.get('examples', []),
                    'notes': idiom.get('notes'),
                    'tags': idiom.get('tags', []),
                    'source': idiom.get('source')
                }
                for idiom in idioms
                if isinstance(idiom, dict) and (idiom.get('idiom') or idiom.get('text'))
            ]
        except json.JSONDecodeError:
            return []
    
    def get_related_words(self, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get related words filtered by relation type."""
        related = []
        
        # Get outgoing relations
        for rel in self.relations_from:
            if not relation_type or rel.relation_type == relation_type:
                related.append({
                    'word': rel.to_word.lemma,
                    'normalized_lemma': rel.to_word.normalized_lemma,
                    'language_code': rel.to_word.language_code,
                    'type': rel.relation_type,
                    'direction': 'outgoing',
                    'metadata': rel.metadata
                })
        
        # Get incoming relations
        for rel in self.relations_to:
            if not relation_type or rel.relation_type == relation_type:
                related.append({
                    'word': rel.from_word.lemma,
                    'normalized_lemma': rel.from_word.normalized_lemma,
                    'language_code': rel.from_word.language_code,
                    'type': rel.relation_type,
                    'direction': 'incoming',
                    'metadata': rel.metadata
                })
        
        return related
    
    def get_affixations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get affixations grouped by type."""
        result = {
            'as_root': [],
            'as_affixed': []
        }
        
        # Get affixations where this word is the root
        for aff in self.affixations_as_root:
            result['as_root'].append({
                'word': aff.affixed_word.lemma,
                'normalized_lemma': aff.affixed_word.normalized_lemma,
                'language_code': aff.affixed_word.language_code,
                'type': aff.affix_type,
                'value': aff.affix_value,
                'position': aff.position,
                'metadata': aff.metadata
            })
        
        # Get affixations where this word is the affixed form
        for aff in self.affixations_as_affixed:
            result['as_affixed'].append({
                'word': aff.root_word.lemma,
                'normalized_lemma': aff.root_word.normalized_lemma,
                'language_code': aff.root_word.language_code,
                'type': aff.affix_type,
                'value': aff.affix_value,
                'position': aff.position,
                'metadata': aff.metadata
            })
        
        return result
    
    def get_pronunciation_data(self) -> Dict[str, Any]:
        """Get pronunciation data in structured format."""
        if not self.pronunciation_data:
            return {}
        
        try:
            data = json.loads(self.pronunciation_data) if isinstance(self.pronunciation_data, str) else self.pronunciation_data
            return {
                'ipa': data.get('ipa'),
                'respelling': data.get('respelling'),
                'audio': data.get('audio'),
                'phonemic': data.get('phonemic'),
                'syllables': data.get('syllables'),
                'stress': data.get('stress'),
                'variants': data.get('variants', [])
            }
        except json.JSONDecodeError:
            return {}
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get source information in structured format."""
        if not self.source_info:
            return {}
        
        try:
            info = json.loads(self.source_info) if isinstance(self.source_info, str) else self.source_info
            return {
                'primary': info.get('primary'),
                'secondary': info.get('secondary', []),
                'contributors': info.get('contributors', []),
                'last_updated': info.get('last_updated'),
                'notes': info.get('notes')
            }
        except json.JSONDecodeError:
            return {}
    
    def to_dict(self, include_definitions: bool = True, include_etymology: bool = True,
                include_relations: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert word to dictionary with configurable detail levels."""
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
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "quality_score": self.calculate_data_quality_score()
        }
        
        if include_metadata:
            result.update({
                "idioms": self.get_idioms_list(),
                "pronunciation": self.get_pronunciation_data(),
                "source_info": self.get_source_info(),
                "metadata": self.metadata,
                "data_hash": self.data_hash
            })
        
        if include_definitions and self.definitions:
            result["definitions"] = [d.to_dict() for d in self.definitions]
        
        if include_etymology and self.etymologies:
            result["etymologies"] = [e.to_dict() for e in self.etymologies]
        
        if include_relations:
            result["relations"] = {
                "synonyms": self.get_related_words("synonym"),
                "antonyms": self.get_related_words("antonym"),
                "variants": self.get_related_words("variant"),
                "related": self.get_related_words("related"),
                "affixations": self.get_affixations()
            }
        
        return result


class Definition(db.Model):
    """Enhanced definition model with additional fields and functionality."""
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
    metadata = Column(JSONB, default=dict)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    word = relationship("Word", back_populates="definitions", lazy="joined")
    standardized_pos = relationship("PartOfSpeech", back_populates="definitions", lazy="joined")
    definition_relations = relationship(
        "DefinitionRelation",
        back_populates="definition",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    
    __table_args__ = (
        Index('idx_definitions_pos', 'standardized_pos_id'),
        Index('idx_definitions_word_id', 'word_id'),
        Index('idx_definitions_tags', 'tags'),
        Index('idx_definitions_verification', 'verification_status'),
        Index('idx_definitions_created', 'created_at'),
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
    
    @validates('verification_status')
    def validate_verification_status(self, key, status):
        """Validate verification status."""
        valid_statuses = ['unverified', 'verified', 'needs_review', 'disputed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid verification status: {status}")
        return status
    
    def get_examples_list(self) -> List[Dict[str, Any]]:
        """Get examples as a list of dictionaries."""
        if not self.examples:
            return []
        
        try:
            examples = json.loads(self.examples)
            if isinstance(examples, list):
                return [
                    {
                        'text': ex.get('text', ex) if isinstance(ex, dict) else str(ex),
                        'translation': ex.get('translation') if isinstance(ex, dict) else None,
                        'notes': ex.get('notes') if isinstance(ex, dict) else None,
                        'source': ex.get('source') if isinstance(ex, dict) else None,
                        'tags': ex.get('tags', []) if isinstance(ex, dict) else []
                    }
                    for ex in examples
                ]
            return [{'text': str(examples)}]
        except json.JSONDecodeError:
            return [{'text': line.strip()} for line in self.examples.split('\n') if line.strip()]
    
    def get_usage_notes_list(self) -> List[Dict[str, Any]]:
        """Get usage notes as a list of dictionaries."""
        if not self.usage_notes:
            return []
        
        try:
            notes = json.loads(self.usage_notes)
            if isinstance(notes, list):
                return [
                    {
                        'text': note.get('text', note) if isinstance(note, dict) else str(note),
                        'type': note.get('type') if isinstance(note, dict) else None,
                        'source': note.get('source') if isinstance(note, dict) else None
                    }
                    for note in notes
                ]
            return [{'text': str(notes)}]
        except json.JSONDecodeError:
            return [{'text': line.strip()} for line in self.usage_notes.split('\n') if line.strip()]
    
    def get_tags_list(self) -> List[str]:
        """Get tags as a list."""
        if not self.tags:
            return []
        try:
            return json.loads(self.tags) if isinstance(self.tags, str) else self.tags
        except json.JSONDecodeError:
            return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        return [source.strip() for source in self.sources.split(",")] if self.sources else []
    
    def calculate_quality_score(self) -> int:
        """Calculate quality score for the definition."""
        score = 0
        
        # Basic completeness (40 points)
        if self.definition_text:
            score += 20
        if self.standardized_pos:
            score += 10
        if self.sources:
            score += 10
            
        # Examples (20 points)
        examples = self.get_examples_list()
        if examples:
            example_score = min(len(examples) * 5, 15)
            if any(ex.get('translation') for ex in examples):
                example_score += 5
            score += example_score
            
        # Usage notes (15 points)
        usage_notes = self.get_usage_notes_list()
        if usage_notes:
            score += min(len(usage_notes) * 5, 15)
            
        # Related words (15 points)
        if self.definition_relations:
            score += min(len(self.definition_relations) * 3, 15)
            
        # Additional features (10 points)
        if self.tags:
            score += 5
        if self.verification_status == 'verified':
            score += 5
            
        return min(score, 100)
    
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
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "quality_score": self.calculate_quality_score()
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
        if self.definition_relations:
            result["related_words"] = [
                {
                    "word": rel.word.lemma,
                    "type": rel.relation_type,
                    "sources": rel.get_sources_list(),
                    "metadata": rel.metadata
                }
                for rel in self.definition_relations
            ]
        
        # Add metadata if available
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


class Etymology(db.Model):
    """Etymology model with enhanced functionality."""
    __tablename__ = 'etymologies'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = Column(Text, nullable=False)
    language_codes = Column(Text)
    normalized_components = Column(Text)
    etymology_structure = Column(JSONB)
    confidence_score = Column(Float)
    sources = Column(Text)
    metadata = Column(JSONB, default=dict)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    word = relationship("Word", back_populates="etymologies", lazy="joined")
    
    __table_args__ = (
        Index('idx_etymologies_word_id', 'word_id'),
        Index('idx_etymologies_language_codes', 'language_codes'),
        Index('idx_etymologies_verification', 'verification_status'),
        Index('idx_etymologies_created', 'created_at')
    )
    
    @validates('etymology_text')
    def validate_etymology_text(self, key, value):
        """Validate etymology text."""
        if not value or not isinstance(value, str):
            raise ValueError("Etymology text must be a non-empty string")
        return value.strip()
    
    def get_language_codes_list(self) -> List[str]:
        """Get language codes as a list."""
        if not self.language_codes:
            return []
        return [code.strip() for code in self.language_codes.split(',') if code.strip()]
    
    def get_components_list(self) -> List[Dict[str, Any]]:
        """Get etymology components as a list."""
        if not self.normalized_components:
            return []
        try:
            components = json.loads(self.normalized_components)
            if isinstance(components, list):
                return [
                    {
                        'text': comp.get('text', comp) if isinstance(comp, dict) else str(comp),
                        'language': comp.get('language') if isinstance(comp, dict) else None,
                        'type': comp.get('type') if isinstance(comp, dict) else None,
                        'confidence': comp.get('confidence') if isinstance(comp, dict) else None
                    }
                    for comp in components
                ]
            return [{'text': str(components)}]
        except json.JSONDecodeError:
            return [{'text': comp.strip()} for comp in self.normalized_components.split(';') if comp.strip()]
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        if not self.sources:
            return []
        return [source.strip() for source in self.sources.split(',') if source.strip()]
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert etymology to dictionary."""
        result = {
            "id": self.id,
            "etymology_text": self.etymology_text,
            "language_codes": self.get_language_codes_list(),
            "components": self.get_components_list(),
            "confidence_score": self.confidence_score,
            "sources": self.get_sources_list(),
            "verification_status": self.verification_status
        }
        
        if include_metadata:
            result.update({
                "etymology_structure": self.etymology_structure,
                "metadata": self.metadata,
                "verification_notes": self.verification_notes,
                "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


class Relation(db.Model):
    """Word relationship model with enhanced functionality."""
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True)
    from_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    to_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(32), nullable=False)
    bidirectional = Column(Boolean, default=False)
    confidence_score = Column(Float)
    sources = Column(Text)
    metadata = Column(JSONB, default=dict)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    from_word = relationship("Word", foreign_keys=[from_word_id], back_populates="relations_from", lazy="joined")
    to_word = relationship("Word", foreign_keys=[to_word_id], back_populates="relations_to", lazy="joined")
    
    __table_args__ = (
        Index('idx_relations_from_word', 'from_word_id'),
        Index('idx_relations_to_word', 'to_word_id'),
        Index('idx_relations_type', 'relation_type'),
        Index('idx_relations_verification', 'verification_status'),
        UniqueConstraint('from_word_id', 'to_word_id', 'relation_type', name='relations_unique')
    )
    
    @validates('relation_type')
    def validate_relation_type(self, key, value):
        """Validate relation type."""
        valid_types = [
            'synonym', 'antonym', 'related', 'similar',
            'hypernym', 'hyponym', 'meronym', 'holonym',
            'derived_from', 'root_of', 'variant',
            'spelling_variant', 'regional_variant',
            'compare_with', 'see_also', 'equals'
        ]
        if value not in valid_types:
            raise ValueError(f"Invalid relation type: {value}")
        return value
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        if not self.sources:
            return []
        return [source.strip() for source in self.sources.split(',') if source.strip()]
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert relation to dictionary."""
        result = {
            "id": self.id,
            "from_word": {
                "id": self.from_word.id,
                "lemma": self.from_word.lemma,
                "language_code": self.from_word.language_code
            },
            "to_word": {
                "id": self.to_word.id,
                "lemma": self.to_word.lemma,
                "language_code": self.to_word.language_code
            },
            "relation_type": self.relation_type,
            "bidirectional": self.bidirectional,
            "confidence_score": self.confidence_score,
            "sources": self.get_sources_list()
        }
        
        if include_metadata:
            result.update({
                "metadata": self.metadata,
                "verification_status": self.verification_status,
                "verification_notes": self.verification_notes,
                "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


class DefinitionRelation(db.Model):
    """Definition relationship model."""
    __tablename__ = 'definition_relations'
    
    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id', ondelete='CASCADE'), nullable=False)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    relation_type = Column(String(32), nullable=False)
    metadata = Column(JSONB, default=dict)
    sources = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    definition = relationship("Definition", back_populates="definition_relations", lazy="joined")
    word = relationship("Word", lazy="joined")
    
    __table_args__ = (
        Index('idx_definition_relations_definition', 'definition_id'),
        Index('idx_definition_relations_word', 'word_id'),
        Index('idx_definition_relations_type', 'relation_type'),
        UniqueConstraint('definition_id', 'word_id', 'relation_type', name='definition_relations_unique')
    )
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        if not self.sources:
            return []
        return [source.strip() for source in self.sources.split(',') if source.strip()]


class Affixation(db.Model):
    """Affixation model for tracking word formation."""
    __tablename__ = 'affixations'
    
    id = Column(Integer, primary_key=True)
    root_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affixed_word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    affix_type = Column(String(32), nullable=False)
    affix_value = Column(String(64))
    position = Column(String(32))
    sources = Column(Text)
    metadata = Column(JSONB, default=dict)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    root_word = relationship("Word", foreign_keys=[root_word_id], back_populates="affixations_as_root", lazy="joined")
    affixed_word = relationship("Word", foreign_keys=[affixed_word_id], back_populates="affixations_as_affixed", lazy="joined")
    
    __table_args__ = (
        Index('idx_affixations_root_word', 'root_word_id'),
        Index('idx_affixations_affixed_word', 'affixed_word_id'),
        Index('idx_affixations_type', 'affix_type'),
        Index('idx_affixations_verification', 'verification_status'),
        UniqueConstraint('root_word_id', 'affixed_word_id', 'affix_type', name='affixations_unique')
    )
    
    @validates('affix_type')
    def validate_affix_type(self, key, value):
        """Validate affix type."""
        valid_types = ['prefix', 'infix', 'suffix', 'circumfix', 'reduplication', 'compound']
        if value not in valid_types:
            raise ValueError(f"Invalid affix type: {value}")
        return value
    
    @validates('position')
    def validate_position(self, key, value):
        """Validate affix position."""
        if value:
            valid_positions = ['initial', 'medial', 'final', 'both']
            if value not in valid_positions:
                raise ValueError(f"Invalid position: {value}")
        return value
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        if not self.sources:
            return []
        return [source.strip() for source in self.sources.split(',') if source.strip()]
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert affixation to dictionary."""
        result = {
            "id": self.id,
            "root_word": {
                "id": self.root_word.id,
                "lemma": self.root_word.lemma,
                "language_code": self.root_word.language_code
            },
            "affixed_word": {
                "id": self.affixed_word.id,
                "lemma": self.affixed_word.lemma,
                "language_code": self.affixed_word.language_code
            },
            "affix_type": self.affix_type,
            "affix_value": self.affix_value,
            "position": self.position,
            "sources": self.get_sources_list()
        }
        
        if include_metadata:
            result.update({
                "metadata": self.metadata,
                "verification_status": self.verification_status,
                "verification_notes": self.verification_notes,
                "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


class PartOfSpeech(db.Model):
    """Part of speech model."""
    __tablename__ = 'parts_of_speech'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(16), nullable=False, unique=True)
    name_en = Column(String(64), nullable=False)
    name_tl = Column(String(64))
    description = Column(Text)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    definitions = relationship("Definition", back_populates="standardized_pos", lazy="dynamic")
    
    __table_args__ = (
        Index('idx_parts_of_speech_code', 'code'),
    )
    
    @validates('code')
    def validate_code(self, key, value):
        """Validate POS code."""
        valid_codes = ['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix']
        if value not in valid_codes:
            raise ValueError(f"Invalid POS code: {value}")
        return value
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert part of speech to dictionary."""
        result = {
            "id": self.id,
            "code": self.code,
            "name_en": self.name_en,
            "name_tl": self.name_tl,
            "description": self.description
        }
        
        if include_metadata:
            result.update({
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


class Pronunciation(db.Model):
    """Pronunciation model for storing word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(32), nullable=False)
    value = Column(Text, nullable=False)
    sources = Column(Text)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    
    # Relationships
    word = relationship("Word", back_populates="pronunciations", lazy="joined")
    
    __table_args__ = (
        Index('idx_pronunciations_word_id', 'word_id'),
        Index('idx_pronunciations_type', 'type'),
        UniqueConstraint('word_id', 'type', name='pronunciations_unique')
    )
    
    @validates('type')
    def validate_type(self, key, value):
        """Validate pronunciation type."""
        valid_types = ['ipa', 'respelling', 'audio', 'phonemic']
        if value not in valid_types:
            raise ValueError(f"Invalid pronunciation type: {value}")
        return value
    
    def get_sources_list(self) -> List[str]:
        """Get sources as a list."""
        if not self.sources:
            return []
        return [source.strip() for source in self.sources.split(',') if source.strip()]
    
    def to_dict(self, include_metadata: bool = False) -> Dict[str, Any]:
        """Convert pronunciation to dictionary."""
        result = {
            "id": self.id,
            "type": self.type,
            "value": self.value,
            "sources": self.get_sources_list()
        }
        
        if include_metadata:
            result.update({
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


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
        {"name": "idx_definition_relations", "table": "definition_relations", "columns": "(definition_id, word_id)", "type": ""},
        {"name": "idx_word_verification", "table": "words", "columns": "(verification_status)", "type": ""},
        {"name": "idx_etymology_verification", "table": "etymologies", "columns": "(verification_status)", "type": ""},
        {"name": "idx_relation_verification", "table": "relations", "columns": "(verification_status)", "type": ""},
        {"name": "idx_affixation_verification", "table": "affixations", "columns": "(verification_status)", "type": ""},
        {"name": "idx_definition_verification", "table": "definitions", "columns": "(verification_status)", "type": ""}
    ]
    
    # Try to create each index separately with error handling
    for idx in indexes:
        try:
            sql = f"CREATE INDEX IF NOT EXISTS {idx['name']} ON {idx['table']} {idx['type']} {idx['columns']}"
            connection.execute(text(sql))
            print(f"Created or verified index: {idx['name']}")
        except Exception as e:
            print(f"Failed to create index {idx['name']}: {e}")
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