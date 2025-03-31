"""
Filipino Dictionary Database Models with enhanced functionality and data validation.
"""

from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Boolean, DateTime, 
    func, Index, UniqueConstraint, DDL, event, text, Float, JSON, cast
)
from sqlalchemy.orm import relationship, validates, backref
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import re
import json
import hashlib
from datetime import datetime, timedelta, UTC
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
    """Model for words."""
    __tablename__ = 'words'
    
    id = db.Column(db.Integer, primary_key=True)
    lemma = db.Column(db.String(100), nullable=False)
    normalized_lemma = db.Column(db.String(100), nullable=False)
    language_code = db.Column(db.String(10), nullable=False)
    has_baybayin = db.Column(db.Boolean, default=False)
    baybayin_form = db.Column(db.String(100))
    romanized_form = db.Column(db.String(100))
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id'))
    preferred_spelling = db.Column(db.String(100))
    tags = db.Column(db.JSON)
    idioms = db.Column(db.JSON)
    pronunciation_data = db.Column(db.JSON)
    source_info = db.Column(db.JSON)
    word_metadata = db.Column(db.JSON)
    data_hash = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    search_text = db.Column(db.Text)
    badlit_form = db.Column(db.String(100))
    hyphenation = db.Column(db.String(100))
    is_proper_noun = db.Column(db.Boolean, default=False)
    is_abbreviation = db.Column(db.Boolean, default=False)
    is_initialism = db.Column(db.Boolean, default=False)
    verification_status = db.Column(db.String(20), default='unverified')
    verification_notes = db.Column(db.Text)
    last_verified_at = db.Column(db.DateTime)
    
    # Relationships
    definitions = db.relationship('Definition', back_populates='word', lazy='joined')
    etymologies = db.relationship('Etymology', back_populates='word', lazy='joined')
    root_word = db.relationship('Word', remote_side=[id], backref=db.backref('derived_words', lazy='joined'))
    
    # Relations are now handled through the Relation model's backrefs
    # outgoing_relations and incoming_relations
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            'id': self.id,
            'lemma': self.lemma,
            'normalized_lemma': self.normalized_lemma,
            'language_code': self.language_code,
            'has_baybayin': self.has_baybayin,
            'baybayin_form': self.baybayin_form,
            'romanized_form': self.romanized_form,
            'root_word_id': self.root_word_id,
            'preferred_spelling': self.preferred_spelling,
            'tags': self.tags or [],
            'idioms': self.idioms or [],
            'pronunciation_data': self.pronunciation_data or {},
            'source_info': self.source_info or {},
            'word_metadata': self.word_metadata or {},
            'data_hash': self.data_hash,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'badlit_form': self.badlit_form,
            'hyphenation': self.hyphenation,
            'is_proper_noun': self.is_proper_noun,
            'is_abbreviation': self.is_abbreviation,
            'is_initialism': self.is_initialism,
            'verification_status': self.verification_status,
            'verification_notes': self.verification_notes,
            'last_verified_at': self.last_verified_at.isoformat() if self.last_verified_at else None,
            'definitions': [d.to_dict() for d in self.definitions],
            'etymologies': [e.to_dict() for e in self.etymologies],
            'outgoing_relations': [r.to_dict() for r in self.outgoing_relations],
            'incoming_relations': [r.to_dict() for r in self.incoming_relations]
        }
    
    def __json__(self):
        """JSON serialization method."""
        return self.to_dict()

    @property
    def is_root(self) -> bool:
        """Check if word is a root word (has no root_word_id)."""
        return self.root_word_id is None
    
    @hybrid_property
    def has_complete_data(self) -> bool:
        """Check if word has complete data."""
        return bool(
            self.definitions and
            (self.etymologies or self.source_relations) and
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
    
    @staticmethod
    def update_search_vector(mapper, connection, target):
        """Update the search vector with all searchable content."""
        # Remove trailing numbers from any text before adding to search vector
        clean_lemma = re.sub(r'\d+$', '', target.lemma)
        clean_normalized = re.sub(r'\d+$', '', target.normalized_lemma)
        
        search_parts = [
            clean_lemma,
            clean_normalized,
            target.preferred_spelling or '',
            ' '.join(d.definition_text for d in target.definitions),
            ' '.join(e.etymology_text for e in target.etymologies),
            target.baybayin_form or '',
            target.romanized_form or '',
            ' '.join(target.get_tags_list()),
            ' '.join(tag for d in target.definitions for tag in d.get_tags_list()),
            target.word_metadata.get('geographic_region', ''),
            target.word_metadata.get('time_period', ''),
            target.word_metadata.get('cultural_notes', ''),
            ' '.join(target.word_metadata.get('grammatical_categories', [])),
            ' '.join(target.word_metadata.get('semantic_domains', []))
        ]
        target.search_text = func.to_tsvector('simple', ' '.join(filter(None, search_parts)))
    
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
        if self.source_relations or self.target_relations:
            rel_count = len(self.source_relations) + len(self.target_relations)
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
            str(sorted([r.relation_type for r in self.source_relations])),
            str(sorted([r.relation_type for r in self.target_relations])),
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
        for rel in self.source_relations:
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
        for rel in self.target_relations:
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
            self.source_relations,
            self.target_relations
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
    def search(cls, query, language_code=None, limit=10, offset=0):
        """Search for words using full-text search."""
        if is_testing_db():
            # For SQLite, use a simpler search approach
            search_query = cls.query.filter(cls.normalized_lemma.ilike(f'%{query}%'))
        else:
            # For PostgreSQL, use full-text search
            search_query = cls.query.filter(
                cls.search_text.op('@@')(func.plainto_tsquery('simple', query))
            ).order_by(
                func.ts_rank(cls.search_text, func.plainto_tsquery('simple', query)).desc()
            )

        if language_code:
            search_query = search_query.filter(cls.language_code == language_code)

        return search_query.offset(offset).limit(limit).all()


class Definition(db.Model):
    """Model for word definitions."""
    __tablename__ = 'definitions'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    definition_text = db.Column(db.Text, nullable=False)
    standardized_pos_id = db.Column(db.Integer, db.ForeignKey('parts_of_speech.id'), nullable=False)
    examples = db.Column(db.JSON)
    usage_notes = db.Column(db.Text)
    meta_info = db.Column(db.JSON)
    sources = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    word = db.relationship('Word', back_populates='definitions', lazy='joined')
    standardized_pos = db.relationship('PartOfSpeech', back_populates='definitions', lazy='joined')
    
    def get_tags_list(self) -> List[str]:
        """Get tags from meta_info as a list."""
        if not self.meta_info or 'tags' not in self.meta_info:
            return []
        tags = self.meta_info.get('tags', [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        return tags if isinstance(tags, list) else []
    
    def to_dict(self):
        """Convert definition to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'definition_text': self.definition_text,
            'standardized_pos': self.standardized_pos.to_dict() if self.standardized_pos else None,
            'examples': self.examples or [],
            'usage_notes': self.usage_notes,
            'meta_info': self.meta_info or {},
            'sources': self.sources or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __json__(self):
        """Make the model JSON serializable."""
        return self.to_dict()


class Etymology(db.Model):
    """Model for word etymologies."""
    __tablename__ = 'etymologies'

    id = db.Column(db.Integer, primary_key=True)
    word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='CASCADE'), nullable=False)
    etymology_text = db.Column(db.Text, nullable=False)
    components = db.Column(db.JSON)
    language_codes = db.Column(db.JSON)
    structure = db.Column(db.JSON)
    confidence_score = db.Column(db.Float, default=0.0)
    sources = db.Column(db.JSON)
    related_words = db.Column(db.JSON)
    meta_info = db.Column(db.JSON)
    verification_status = db.Column(db.String(20))
    verification_notes = db.Column(db.Text)
    last_verified_at = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    # Relationships
    word = db.relationship('Word', back_populates='etymologies')

    def get_components_list(self) -> List[str]:
        """Get components as a list."""
        if not self.components:
            return []
        try:
            return json.loads(self.components) if isinstance(self.components, str) else self.components
        except json.JSONDecodeError:
            return [comp.strip() for comp in self.components.split(',') if comp.strip()]

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

    def to_dict(self):
        """Convert etymology to dictionary."""
        return {
            'id': self.id,
            'word_id': self.word_id,
            'etymology_text': self.etymology_text,
            'components': self.components or [],
            'language_codes': self.language_codes or [],
            'structure': self.structure or {},
            'confidence_score': self.confidence_score,
            'sources': self.sources or [],
            'related_words': self.related_words or [],
            'meta_info': self.meta_info or {},
            'verification_status': self.verification_status,
            'verification_notes': self.verification_notes,
            'last_verified_at': self.last_verified_at.isoformat() if self.last_verified_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __json__(self):
        """Make the model JSON serializable."""
        return self.to_dict()


class Relation(db.Model):
    """Model for word relationships."""
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    source_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    target_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    relation_type = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, default=1.0)
    relation_metadata = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_word = db.relationship('Word', foreign_keys=[source_word_id], backref=db.backref('outgoing_relations', lazy='joined'))
    target_word = db.relationship('Word', foreign_keys=[target_word_id], backref=db.backref('incoming_relations', lazy='joined'))
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            'id': self.id,
            'source_word_id': self.source_word_id,
            'target_word_id': self.target_word_id,
            'type': self.relation_type,
            'confidence_score': self.confidence_score,
            'relation_metadata': self.relation_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __json__(self):
        """JSON serialization method."""
        return self.to_dict()


class Affixation(db.Model):
    """Affixation model for tracking word formation."""
    __tablename__ = 'affixations'
    
    id = db.Column(db.Integer, primary_key=True)
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    affixed_word_id = db.Column(db.Integer, db.ForeignKey('words.id'), nullable=False)
    affix = db.Column(db.String(50), nullable=False)
    affix_type = db.Column(db.String(50), nullable=False)  # prefix, infix, suffix
    derived_form = db.Column(db.String(255), nullable=False)
    meaning = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    root_word = db.relationship('Word', foreign_keys=[root_word_id], back_populates='affixations_as_root')
    affixed_word = db.relationship('Word', foreign_keys=[affixed_word_id], back_populates='affixations_as_affixed')
    
    def to_dict(self):
        return {
            'id': self.id,
            'root_word': self.root_word.to_dict(include_definitions=False),
            'affixed_word': self.affixed_word.to_dict(include_definitions=False),
            'affix': self.affix,
            'affix_type': self.affix_type,
            'derived_form': self.derived_form,
            'meaning': self.meaning,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PartOfSpeech(db.Model):
    """Model for parts of speech."""
    __tablename__ = 'parts_of_speech'
    
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), nullable=False)
    name_en = db.Column(db.String(50), nullable=False)
    name_tl = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    definitions = db.relationship('Definition', back_populates='standardized_pos', lazy='joined')
    
    def to_dict(self):
        """Convert the model to a dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'name_en': self.name_en,
            'name_tl': self.name_tl,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __json__(self):
        """JSON serialization method."""
        return self.to_dict()


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
    """Initialize the database with the Flask app."""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Only set up PostgreSQL-specific features if we're using PostgreSQL
        if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgresql'):
            try:
                # Create extensions
                db.session.execute(text("""
                    CREATE EXTENSION IF NOT EXISTS pg_trgm;
                    CREATE EXTENSION IF NOT EXISTS unaccent;
                    CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                    CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
                    CREATE EXTENSION IF NOT EXISTS btree_gin;
                    CREATE EXTENSION IF NOT EXISTS dict_xsyn;
                """))
                
                # Create custom collation
                try:
                    db.session.execute(text("""
                        CREATE COLLATION IF NOT EXISTS tagalog_ci (
                            provider = icu,
                            locale = 'tl'
                        );
                    """))
                except Exception as e:
                    app.logger.warning(f"Custom collations could not be created: {str(e)}")
                
                # Create text search configuration
                try:
                    db.session.execute(text("""
                        DROP TEXT SEARCH CONFIGURATION IF EXISTS simple_tagalog;
                        CREATE TEXT SEARCH CONFIGURATION simple_tagalog (COPY = simple);
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR word WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR asciiword WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR numword WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR asciihword WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR hword WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR numhword WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR hword_part WITH unaccent, simple;
                        ALTER TEXT SEARCH CONFIGURATION simple_tagalog ALTER MAPPING FOR hword_numpart WITH unaccent, simple;
                    """))
                except Exception as e:
                    app.logger.warning(f"Text search configuration could not be created: {str(e)}")
                
                # Create functions
                db.session.execute(text("""
                    CREATE OR REPLACE FUNCTION ts_rank_cd(text, tsquery)
                    RETURNS float4 AS $$
                    BEGIN
                        RETURN ts_rank_cd(to_tsvector('simple_tagalog', $1), $2);
                    END;
                    $$ LANGUAGE plpgsql;
                    
                    CREATE OR REPLACE FUNCTION similarity(text, text)
                    RETURNS float4 AS $$
                    BEGIN
                        RETURN similarity($1, $2);
                    END;
                    $$ LANGUAGE plpgsql;
                """))
                
                db.session.commit()
                app.logger.info("Database extensions and language support setup completed")
            except Exception as e:
                app.logger.warning(f"Some optional extensions could not be installed: {str(e)}")
        
        # Register event listeners
        event.listen(Word, 'after_insert', Word.update_search_vector)
        event.listen(Word, 'after_update', Word.update_search_vector)
        
        app.logger.info("Database initialization completed successfully")