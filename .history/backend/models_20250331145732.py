"""
Filipino Dictionary Database Models with enhanced functionality and data validation.
"""

from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Boolean, DateTime, 
    func, Index, UniqueConstraint, DDL, event, text, Float, JSON, cast
)
from sqlalchemy.orm import relationship, validates, backref
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, TSQUERY
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import re
import json
import hashlib
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Any, Optional
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

# Helper function to get the appropriate JSON type
def get_json_type():
    """Get the appropriate JSON type based on the database."""
    return JSON

# Helper function to get the appropriate index DDL
def get_index_ddl(connection):
    """Get the appropriate index DDL based on the database."""
    if is_testing_db(connection):
        return [
            DDL("CREATE INDEX IF NOT EXISTS idx_words_search ON words(search_text)"),
            DDL("CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma)"),
            DDL("CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = 1"),
            DDL("CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies(language_codes)"),
            DDL("CREATE INDEX IF NOT EXISTS idx_etymologies_components ON etymologies(normalized_components)"),
            DDL("""
                CREATE TRIGGER IF NOT EXISTS update_timestamp
                AFTER UPDATE ON words
                FOR EACH ROW
                BEGIN
                    UPDATE words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
            """)
        ]
    else:
        return [DDL("""
            CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
            CREATE INDEX IF NOT EXISTS idx_words_normalized ON words USING gin(to_tsvector('simple', normalized_lemma));
            CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words USING gin(to_tsvector('simple', baybayin_form)) 
            WHERE has_baybayin = true;
            
            CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));
            CREATE INDEX IF NOT EXISTS idx_etymologies_components ON etymologies USING gin(to_tsvector('simple', normalized_components));
            
            CREATE OR REPLACE FUNCTION update_timestamp()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)]

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

# Word model
class Word(db.Model):
    """Enhanced word model with comprehensive data fields."""
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
    idioms = Column(get_json_type(), default=list)
    pronunciation_data = Column(get_json_type())
    source_info = Column(get_json_type(), default=dict)
    word_metadata = Column(get_json_type(), default=dict)
    data_hash = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), server_default=func.current_timestamp())
    search_text = Column(Text)  # Use Text instead of TSVECTOR for SQLite
    badlit_form = Column(String(255))
    hyphenation = Column(String(255))
    is_proper_noun = Column(Boolean, default=False)
    is_abbreviation = Column(Boolean, default=False)
    is_initialism = Column(Boolean, default=False)
    verification_status = Column(String(32), default='unverified')
    verification_notes = Column(Text)
    last_verified_at = Column(DateTime(timezone=True))

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
        foreign_keys="[Relation.source_word_id]",
        back_populates="source_word",
        cascade="all, delete-orphan",
        lazy="joined"
    )
    relations_to = relationship(
        "Relation",
        foreign_keys="[Relation.target_word_id]",
        back_populates="target_word",
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

    @property
    def is_root(self) -> bool:
        """Check if word is a root word (has no root_word_id)."""
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
    
    @validates('language_code')
    def validate_language_code(self, key, code):
        """Validate language code."""
        if not code or not isinstance(code, str):
            raise ValueError("Language code must be a non-empty string")
        if len(code) > 16:
            raise ValueError("Language code must be less than 16 characters")
        # Only allow lowercase letters and hyphens
        if not re.match(r'^[a-z-]+$', code):
            raise ValueError("Language code must contain only lowercase letters and hyphens")
        return code.strip()

    @validates('lemma', 'normalized_lemma')
    def validate_lemma(self, key, value):
        """Validate lemma fields."""
        if not value or not isinstance(value, str):
            raise ValueError(f"{key} must be a non-empty string")
        if len(value) > 255:
            raise ValueError(f"{key} must be less than 255 characters")
        # Remove trailing numbers from lemma
        if key == 'lemma':
            value = re.sub(r'\d+$', '', value)
        return value.strip()
    
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
        # Remove trailing numbers from any text before adding to search vector
        clean_lemma = re.sub(r'\d+$', '', self.lemma)
        clean_normalized = re.sub(r'\d+$', '', self.normalized_lemma)
        
        search_parts = [
            clean_lemma,
            clean_normalized,
            self.preferred_spelling or '',
            ' '.join(d.definition_text for d in self.definitions),
            ' '.join(e.etymology_text for e in self.etymologies),
            self.baybayin_form or '',
            self.romanized_form or '',
            ' '.join(self.get_tags_list()),
            ' '.join(tag for d in self.definitions for tag in d.get_tags_list()),
            self.word_metadata.get('geographic_region', ''),
            self.word_metadata.get('time_period', ''),
            self.word_metadata.get('cultural_notes', ''),
            ' '.join(self.word_metadata.get('grammatical_categories', [])),
            ' '.join(self.word_metadata.get('semantic_domains', []))
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
            str(self.word_metadata or {})
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
                    'word': rel.target_word.lemma,
                    'normalized_lemma': rel.target_word.normalized_lemma,
                    'language_code': rel.target_word.language_code,
                    'type': rel.relation_type,
                    'direction': 'outgoing',
                    'metadata': rel.relation_metadata
                })
        
        # Get incoming relations
        for rel in self.relations_to:
            if not relation_type or rel.relation_type == relation_type:
                related.append({
                    'word': rel.source_word.lemma,
                    'normalized_lemma': rel.source_word.normalized_lemma,
                    'language_code': rel.source_word.language_code,
                    'type': rel.relation_type,
                    'direction': 'incoming',
                    'metadata': rel.relation_metadata
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
                'metadata': aff.meta_info
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
                'metadata': aff.meta_info
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
            "baybayin_form": self.baybayin_form,
            "romanized_form": self.romanized_form,
            "is_root": self.is_root,
            "verification_status": self.verification_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "data_completeness": self._calculate_completeness()
        }
        
        if include_definitions and self.definitions:
            result['definitions'] = [
                {
                    'id': d.id,
                    'definition_text': d.definition_text,
                    'part_of_speech': d.standardized_pos.to_dict() if d.standardized_pos else None,
                    'original_pos': d.original_pos,
                    'quality_score': d.quality_score,
                    'verification_status': d.verification_status,
                    'verification_notes': d.verification_notes,
                    'last_verified_at': d.last_verified_at.isoformat() if d.last_verified_at else None,
                    'sources': d.sources,
                    'tags': d.tags,
                    'usage_notes': d.usage_notes,
                    'meta_info': d.meta_info,
                    'created_at': d.created_at.isoformat() if d.created_at else None,
                    'updated_at': d.updated_at.isoformat() if d.updated_at else None,
                    'examples': [
                        {
                            'text': e.text,
                            'translation': e.translation,
                            'notes': e.notes,
                            'source': e.source,
                            'tags': e.tags
                        } for e in d.examples
                    ] if d.examples else []
                } for d in self.definitions
            ]

        if include_etymology and self.etymologies:
            result['etymologies'] = [
                {
                    'id': e.id,
                    'etymology_text': e.etymology_text,
                    'components': e.components,
                    'language_codes': e.language_codes,
                    'confidence_score': e.confidence_score,
                    'verification_status': e.verification_status,
                    'sources': e.sources
                } for e in self.etymologies
            ]

        if include_relations:
            result['relations'] = {
                'synonyms': [
                    {
                        'word': r.target_word.lemma,
                        'normalized_lemma': r.target_word.normalized_lemma,
                        'language_code': r.target_word.language_code,
                        'type': 'synonym',
                        'direction': 'outgoing',
                        'metadata': r.relation_metadata or {}
                    } for r in self.relations_from if r.relation_type == 'synonym'
                ],
                'antonyms': [
                    {
                        'word': r.target_word.lemma,
                        'normalized_lemma': r.target_word.normalized_lemma,
                        'language_code': r.target_word.language_code,
                        'type': 'antonym',
                        'direction': 'outgoing',
                        'metadata': r.relation_metadata or {}
                    } for r in self.relations_from if r.relation_type == 'antonym'
                ],
                'related': [
                    {
                        'word': r.target_word.lemma,
                        'normalized_lemma': r.target_word.normalized_lemma,
                        'language_code': r.target_word.language_code,
                        'type': 'related',
                        'direction': 'outgoing',
                        'metadata': r.relation_metadata or {}
                    } for r in self.relations_from if r.relation_type == 'related'
                ],
                'variants': [
                    {
                        'word': r.target_word.lemma,
                        'normalized_lemma': r.target_word.normalized_lemma,
                        'language_code': r.target_word.language_code,
                        'type': 'variant',
                        'direction': 'outgoing',
                        'metadata': r.relation_metadata or {}
                    } for r in self.relations_from if r.relation_type == 'variant'
                ],
                'affixations': {
                    'as_root': [
                        {
                            'word': r.target_word.lemma,
                            'normalized_lemma': r.target_word.normalized_lemma,
                            'language_code': r.target_word.language_code,
                            'type': 'affixation',
                            'direction': 'outgoing',
                            'metadata': r.relation_metadata or {}
                        } for r in self.relations_from if r.relation_type == 'affixation'
                    ],
                    'as_affixed': [
                        {
                            'word': r.source_word.lemma,
                            'normalized_lemma': r.source_word.normalized_lemma,
                            'language_code': r.source_word.language_code,
                            'type': 'affixation',
                            'direction': 'incoming',
                            'metadata': r.relation_metadata or {}
                        } for r in self.relations_to if r.relation_type == 'affixation'
                    ]
                }
            }

        if include_metadata:
            result['word_metadata'] = self.word_metadata or {}
            result['source_info'] = self.source_info or {}
            result['pronunciation_data'] = self.pronunciation_data or {}
            result['idioms'] = self.idioms or []

        return result

    def _calculate_completeness(self):
        """Calculate the completeness score of the word's data."""
        score = 0
        total = 0

        # Basic information
        fields = [
            self.lemma,
            self.language_code,
            self.verification_status
        ]
        score += sum(1 for f in fields if f is not None)
        total += len(fields)

        # Definitions
        if self.definitions:
            score += 1
            for d in self.definitions:
                if d.examples:
                    score += 0.5
                if d.usage_notes:
                    score += 0.5
        total += 2

        # Etymology
        if self.etymologies:
            score += 1
        total += 1

        # Relations
        if any([
            self.relations_from,
            self.relations_to
        ]):
            score += 1
        total += 1

        # Additional data
        if self.has_baybayin and self.baybayin_form:
            score += 1
        if self.pronunciation_data:
            score += 1
        if self.word_metadata:
            score += 1
        total += 5

        return round(score / total, 2) if total > 0 else 0.0

    @classmethod
    def search(cls, query_text, language_code=None):
        """Search for words using full text search."""
        search_query = func.to_tsquery('simple', query_text)
        rank_func = func.ts_rank(
            cast(cls.search_text, TSVECTOR),
            cast(search_query, TSQUERY)
        )
        
        base_query = cls.query
        if language_code:
            base_query = base_query.filter(cls.language_code == language_code)
            
        return base_query.filter(
            db.or_(
                cls.normalized_lemma.ilike(f'%{query_text}%'),
                cast(cls.search_text, TSVECTOR).op('@@')(cast(search_query, TSQUERY))
            )
        ).order_by(rank_func.desc(), func.similarity(cls.lemma, query_text).desc())


class Definition(db.Model):
    """Model for word definitions."""
    __tablename__ = 'definitions'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    definition_text = db.Column(db.Text, nullable=False)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'))
    original_pos = db.Column(db.String(50))
    quality_score = db.Column(db.Float, default=0.0)
    verification_status = db.Column(db.String(20))
    verification_notes = db.Column(db.Text)
    last_verified_at = db.Column(db.DateTime(timezone=True))
    sources = db.Column(db.JSON)
    tags = db.Column(db.JSON)
    usage_notes = db.Column(db.JSON)
    meta_info = db.Column(db.JSON)
    examples = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    # Relationships
    word = db.relationship('Word', back_populates='definitions')
    standardized_pos = db.relationship('PartOfSpeech', back_populates='definitions')


class Etymology(db.Model):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = db.Column(db.Text, nullable=False)
    components = db.Column(db.JSON)
    language_codes = db.Column(db.JSON)
    confidence_score = db.Column(db.Float)
    verification_status = db.Column(db.String(20))
    sources = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    # Relationships
    word = db.relationship('Word', back_populates='etymologies')

    def get_language_codes_list(self):
        """Get list of language codes from etymology data."""
        if not self.language_codes:
            return []
        languages = set()
        if isinstance(self.language_codes, list):
            languages.update(self.language_codes)
        elif isinstance(self.language_codes, dict):
            if 'language' in self.language_codes:
                languages.add(self.language_codes['language'])
            if 'cognates' in self.language_codes:
                for cognate in self.language_codes['cognates']:
                    if isinstance(cognate, dict) and 'language' in cognate:
                        languages.add(cognate['language'])
        return list(languages)


class Relation(db.Model):
    """Model for word relations."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    source_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    target_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    relation_metadata = db.Column(db.JSON)
    confidence_score = db.Column(db.Float, default=1.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    source_word = db.relationship('Word', foreign_keys=[source_word_id], backref='source_relations')
    target_word = db.relationship('Word', foreign_keys=[target_word_id], backref='target_relations')
    
    def to_dict(self):
        return {
            'id': self.id,
            'source_word': self.source_word.to_dict(include_definitions=False),
            'target_word': self.target_word.to_dict(include_definitions=False),
            'type': self.type,
            'metadata': self.relation_metadata,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


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
    meta_info = Column(get_json_type(), default=dict)
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
                "metadata": self.meta_info,
                "verification_status": self.verification_status,
                "verification_notes": self.verification_notes,
                "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


class PartOfSpeech(db.Model):
    """Model for parts of speech."""
    __tablename__ = 'parts_of_speech'

    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), unique=True, nullable=False)
    name_en = db.Column(db.String(50), nullable=False)
    name_tl = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    meta_info = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    # Relationships
    definitions = db.relationship('Definition', back_populates='standardized_pos', lazy='dynamic')

    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'name_en': self.name_en,
            'name_tl': self.name_tl,
            'description': self.description,
            'meta_info': self.meta_info or {}
        }


class Pronunciation(db.Model):
    """Pronunciation model for storing word pronunciations."""
    __tablename__ = 'pronunciations'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    type = Column(String(32), nullable=False)
    value = Column(Text, nullable=False)
    sources = Column(Text)
    meta_info = Column(get_json_type(), default=dict)
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
                "metadata": self.meta_info,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None
            })
            
        return result


def create_timestamp_function(target, connection, **kw):
    """Create the timestamp update function."""
    try:
        if is_testing_db(connection):
            # SQLite doesn't need a separate function
            pass
        else:
            # PostgreSQL needs a function
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION update_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))
            print(f"Created timestamp update function successfully")
    except Exception as e:
        print(f"WARNING: Error creating timestamp function: {e}")
        print("Timestamp updates may not work correctly")

def create_timestamp_trigger(target, connection, **kw):
    """Create the timestamp update trigger for a table."""
    try:
        if is_testing_db(connection):
            # SQLite trigger
            connection.execute(text(f"""
                CREATE TRIGGER IF NOT EXISTS update_{target.name}_timestamp
                AFTER UPDATE ON {target.name}
                FOR EACH ROW
                BEGIN
                    UPDATE {target.name} SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END;
            """))
        else:
            # PostgreSQL trigger
            connection.execute(text(f"""
                DROP TRIGGER IF EXISTS update_{target.name}_timestamp ON {target.name};
                CREATE TRIGGER update_{target.name}_timestamp
                BEFORE UPDATE ON {target.name}
                FOR EACH ROW
                EXECUTE FUNCTION update_timestamp();
            """))
            print(f"Created timestamp trigger for {target.name} successfully")
    except Exception as e:
        print(f"WARNING: Error creating timestamp trigger for {target.name}: {e}")
        print(f"Timestamps for {target.name} will not update automatically")

# Word event listeners for data validation
def validate_word_data(mapper, connection, target):
    """Validate word data before insert or update."""
    target.validate_baybayin()
    target.update_search_vector()
    target.data_hash = target.generate_data_hash()

# Initialize models
def init_app(app):
    """Initialize the database."""
    db.init_app(app)

    with app.app_context():
        # Create extensions
        try:
            db.session.execute(text("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
            """))
            db.session.commit()
        except Exception as e:
            app.logger.warning(f"Some optional extensions could not be installed: {str(e)}")
            db.session.rollback()

        # Create custom collations
        try:
            db.session.execute(text("""
                CREATE COLLATION IF NOT EXISTS case_insensitive (
                    provider = icu,
                    locale = 'und-u-ks-level2',
                    deterministic = false
                );
            """))
            db.session.commit()
        except Exception as e:
            app.logger.warning(f"Custom collations could not be created: {str(e)}")
            db.session.rollback()

        # Create text search configuration
        try:
            db.session.execute(text("""
                CREATE TEXT SEARCH CONFIGURATION simple_tagalog (COPY = simple);
                ALTER TEXT SEARCH CONFIGURATION simple_tagalog
                    ALTER MAPPING FOR asciiword, asciihword, hword_asciipart,
                                    word, hword, hword_part
                    WITH unaccent, simple;
            """))
            db.session.commit()
        except Exception as e:
            app.logger.warning(f"Text search configuration could not be created: {str(e)}")
            db.session.rollback()

        app.logger.info("Database extensions and language support setup completed")

        # Create all tables
        db.create_all()

        # Register event listeners
        event.listen(Word, 'after_insert', Word.update_search_vector)
        event.listen(Word, 'after_update', Word.update_search_vector)

        app.logger.info("Database initialization completed successfully")