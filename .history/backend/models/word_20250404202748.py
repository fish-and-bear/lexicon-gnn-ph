"""
Word model definition.
"""

from database import db, cached_query
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.text_search import TextSearchMixin
from .mixins.gin_index import GINIndexMixin
from .mixins.trigram_search import TrigramSearchMixin
import json
from typing import List, Tuple, Dict, Any, Optional
from sqlalchemy.sql import func
import re
from sqlalchemy.orm import column_property
from sqlalchemy.ext.hybrid import hybrid_property

class Word(BaseModel, TextSearchMixin, GINIndexMixin, TrigramSearchMixin):
    """Model for dictionary words."""
    __tablename__ = 'words'
    
    id = db.Column(db.Integer, primary_key=True)
    lemma = db.Column(db.String(255), nullable=False)
    normalized_lemma = db.Column(db.String(255), nullable=False, index=True)
    language_code = db.Column(db.String(16), nullable=False, default='tl', index=True)
    has_baybayin = db.Column(db.Boolean, default=False)
    baybayin_form = db.Column(db.String(255))
    romanized_form = db.Column(db.String(255))
    root_word_id = db.Column(db.Integer, db.ForeignKey('words.id', ondelete='SET NULL'), index=True)
    preferred_spelling = db.Column(db.String(255))
    tags = db.Column(db.Text)
    idioms = db.Column(db.JSON)
    pronunciation_data = db.Column(db.JSON)
    source_info = db.Column(db.JSON)
    word_metadata = db.Column(db.JSON)
    data_hash = db.Column(db.Text)
    search_text = db.Column(db.Text)
    badlit_form = db.Column(db.Text)
    hyphenation = db.Column(db.JSON)
    is_proper_noun = db.Column(db.Boolean, default=False)
    is_abbreviation = db.Column(db.Boolean, default=False)
    is_initialism = db.Column(db.Boolean, default=False)
    is_root = db.Column(db.Boolean, default=True)
    completeness_score = db.Column(db.Float, default=0.0)
    
    # Define relationship with Language - reference by code not foreign key
    language = db.relationship('Language', 
                            foreign_keys=[language_code],
                            primaryjoin="Word.language_code == Language.code",
                            viewonly=True,
                            uselist=False)
    
    # Optimized relationships with proper cascade rules and query strategies
    definitions = db.relationship('Definition', 
                                back_populates='word', 
                                lazy='selectin',
                                cascade='all, delete-orphan',
                                order_by='Definition.standardized_pos_id')
    etymologies = db.relationship('Etymology', 
                                back_populates='word', 
                                lazy='selectin',
                                cascade='all, delete-orphan')
    pronunciations = db.relationship('Pronunciation', 
                                   back_populates='word', 
                                   lazy='selectin',
                                   cascade='all, delete-orphan')
    credits = db.relationship('Credit', 
                            back_populates='word', 
                            lazy='selectin',
                            cascade='all, delete-orphan')
    
    # Self-referential relationships with proper cascade rules
    root_word = db.relationship('Word', 
                              remote_side=[id], 
                              back_populates='derived_words', 
                              lazy='selectin',
                              cascade='save-update')
    derived_words = db.relationship('Word', 
                                  back_populates='root_word', 
                                  lazy='selectin',
                                  cascade='save-update')
    
    # Relation relationships with proper cascade rules
    outgoing_relations = db.relationship('Relation', 
                                       foreign_keys='Relation.from_word_id', 
                                       back_populates='source_word', 
                                       lazy='selectin',
                                       cascade='all, delete-orphan')
    incoming_relations = db.relationship('Relation', 
                                       foreign_keys='Relation.to_word_id', 
                                       back_populates='target_word', 
                                       lazy='selectin',
                                       cascade='all, delete-orphan')
    
    # Affixation relationships with proper cascade rules
    root_affixations = db.relationship('Affixation', 
                                     foreign_keys='Affixation.root_word_id', 
                                     back_populates='root_word', 
                                     lazy='selectin',
                                     cascade='all, delete-orphan')
    affixed_affixations = db.relationship('Affixation', 
                                        foreign_keys='Affixation.affixed_word_id', 
                                        back_populates='affixed_word', 
                                        lazy='selectin',
                                        cascade='all, delete-orphan')
    
    # Definition relation relationships with proper cascade rules
    definition_relations = db.relationship('DefinitionRelation', 
                                         back_populates='related_word', 
                                         lazy='selectin',
                                         cascade='all, delete-orphan',
                                         overlaps="related_definitions,definition.related_words")
    related_definitions = db.relationship('Definition',
                                        secondary='definition_relations',
                                        primaryjoin='Word.id == DefinitionRelation.word_id',
                                        secondaryjoin='DefinitionRelation.definition_id == Definition.id',
                                        overlaps="definition_relations,definition.definition_relations",
                                        lazy='selectin',
                                        viewonly=True)
    
    __table_args__ = (
        db.UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
        db.CheckConstraint(
            "(has_baybayin = false AND baybayin_form IS NULL) OR (has_baybayin = true AND baybayin_form IS NOT NULL)",
            name='baybayin_form_check'
        ),
        db.CheckConstraint(
            "baybayin_form ~ '^[\u1700-\u171F[:space:]]*$' OR baybayin_form IS NULL",
            name='baybayin_form_regex'
        ),
        db.Index('idx_words_is_root', 'is_root'),
        db.Index('idx_words_completeness', 'completeness_score')
    )
    
    # Optimized GIN indexes for Filipino language features
    __gin_indexes__ = [
        {'field': 'search_text', 'config': 'filipino'},
        {'field': 'tags', 'opclass': 'gin_trgm_ops'},
        {'field': 'idioms', 'opclass': 'jsonb_path_ops'},
        {'field': 'pronunciation_data', 'opclass': 'jsonb_path_ops'},
        {'field': 'word_metadata', 'opclass': 'jsonb_path_ops'},
        {'field': 'source_info', 'opclass': 'jsonb_path_ops'},
        {
            'name': 'words_lemma_trgm_idx',
            'fields': ['lemma'],
            'type': 'trgm'
        },
        {
            'name': 'words_normalized_lemma_trgm_idx',
            'fields': ['normalized_lemma'],
            'type': 'trgm'
        },
        {
            'name': 'words_baybayin_trgm_idx',
            'fields': ['baybayin_form'],
            'type': 'trgm'
        }
    ]
    
    # Optimized fields for tsvector with Filipino language support
    __ts_vector_fields__ = ['lemma', 'normalized_lemma', 'baybayin_form', 'romanized_form']
    
    # Baybayin pattern (using raw string to avoid escape sequence issues)
    BAYBAYIN_PATTERN = r'[\u1700-\u171F]'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_root = self.root_word_id is None
        self.calculate_completeness_score()
    
    @validates('lemma')
    def validate_lemma(self, key, value):
        """Validate lemma."""
        if not value:
            raise ValueError("Lemma cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Lemma must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Lemma cannot be empty after stripping")
        if len(value) > 255:
            raise ValueError("Lemma cannot exceed 255 characters")
        return value
    
    @validates('normalized_lemma')
    def validate_normalized_lemma(self, key, value):
        """Validate normalized lemma."""
        if not value:
            raise ValueError("Normalized lemma cannot be empty")
        if not isinstance(value, str):
            raise ValueError("Normalized lemma must be a string")
        value = value.strip().lower()
        if not value:
            raise ValueError("Normalized lemma cannot be empty after stripping")
        if len(value) > 255:
            raise ValueError("Normalized lemma cannot exceed 255 characters")
        return value
    
    @validates('language_code')
    def validate_language_code(self, key, value):
        """Validate language code."""
        if not value:
            return 'tl'  # Default to Tagalog
        if not isinstance(value, str):
            raise ValueError("Language code must be a string")
        value = value.strip().lower()
        if not re.match(r'^[a-z]{2,3}$', value):
            raise ValueError("Language code must be a valid ISO 639-1 or 639-2 code")
        return value
    
    @validates('baybayin_form')
    def validate_baybayin_form(self, key, value):
        """Validate baybayin form."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Baybayin form must be a string")
        value = value.strip()
        if not value:
            return None
        if not re.match(r'^[\u1700-\u171F\s]*$', value):
            raise ValueError("Baybayin form must contain only valid Baybayin characters")
        return value
    
    @validates('root_word_id')
    def validate_root_word_id(self, key, value):
        """Update is_root when root_word_id changes."""
        self.is_root = (value is None)
        return value
    
    @property
    def get_definitions_count(self) -> int:
        """Get count of definitions."""
        return len(self.definitions)
    
    @property 
    def has_etymology(self) -> bool:
        """Check if word has etymology information."""
        return len(self.etymologies) > 0
    
    @property
    def has_pronunciation(self) -> bool:
        """Check if word has pronunciation data."""
        return len(self.pronunciations) > 0
    
    def calculate_completeness_score(self) -> float:
        """Calculate data completeness score (0.0-1.0)."""
        score = 0.0
        total_points = 10.0  # Total possible points
        
        # Basic information (2 points)
        if self.lemma and self.language_code:
            score += 1.0
        if self.normalized_lemma:
            score += 0.5
        if self.has_baybayin and self.baybayin_form:
            score += 0.5
            
        # Definitions (3 points)
        if hasattr(self, 'definitions') and self.definitions:
            definition_count = len(self.definitions)
            if definition_count > 0:
                score += min(1.5, definition_count * 0.5)  # Up to 1.5 points based on definition count
                
            # Check for examples in definitions
            has_examples = any(d.examples for d in self.definitions if hasattr(d, 'examples') and d.examples)
            if has_examples:
                score += 0.75
                
            # Check for usage notes
            has_usage = any(d.usage_notes for d in self.definitions if hasattr(d, 'usage_notes') and d.usage_notes)
            if has_usage:
                score += 0.75
        
        # Etymology (2 points)
        if hasattr(self, 'etymologies') and self.etymologies:
            score += min(2.0, len(self.etymologies) * 1.0)
            
        # Pronunciation (1 point)
        if hasattr(self, 'pronunciations') and self.pronunciations:
            score += min(1.0, len(self.pronunciations) * 0.5)
            
        # Relations (2 points)
        relation_count = 0
        if hasattr(self, 'outgoing_relations'):
            relation_count += len(self.outgoing_relations)
        if hasattr(self, 'incoming_relations'):
            relation_count += len(self.incoming_relations)
        score += min(2.0, relation_count * 0.25)
        
        # Calculate final score as percentage
        self.completeness_score = round(score / total_points, 2)
        return self.completeness_score
    
    def __repr__(self):
        return f'<Word {self.id}: {self.lemma}>'
    
    @cached_query(timeout=3600, key_prefix="word_dict")
    def to_dict(self, include_related: bool = False) -> Dict[str, Any]:
        """Convert word to dictionary with optional related data."""
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
            'tags': self.tags.split(',') if self.tags else [],
            'idioms': self.idioms,
            'source_info': self.source_info,
            'word_metadata': self.word_metadata,
            'badlit_form': self.badlit_form,
            'hyphenation': self.hyphenation,
            'is_proper_noun': self.is_proper_noun,
            'is_abbreviation': self.is_abbreviation,
            'is_initialism': self.is_initialism,
            'is_root': self.is_root,
            'completeness_score': self.completeness_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'data_completeness': {
                'has_definitions': self.get_definitions_count > 0,
                'has_etymology': self.has_etymology,
                'has_pronunciation': self.has_pronunciation,
                'definitions_count': self.get_definitions_count
            }
        }
        
        # Include related data if requested
        if include_related:
            self._add_related_data_to_dict(result)
            
        return result
    
    def _add_related_data_to_dict(self, result_dict: Dict[str, Any]) -> None:
        """Add related data to dictionary representation."""
        # Add definitions
        if hasattr(self, 'definitions'):
            result_dict['definitions'] = [d.to_dict() for d in self.definitions]
            
        # Add etymologies
        if hasattr(self, 'etymologies'):
            result_dict['etymologies'] = [e.to_dict() for e in self.etymologies]
            
        # Add pronunciations
        if hasattr(self, 'pronunciations'):
            result_dict['pronunciations'] = [p.to_dict() for p in self.pronunciations]
            
        # Add relations (limited to avoid circular references)
        if hasattr(self, 'outgoing_relations'):
            result_dict['outgoing_relations'] = [
                {
                    'id': r.id,
                    'relation_type': r.relation_type,
                    'target_word': {
                        'id': r.target_word.id,
                        'lemma': r.target_word.lemma,
                        'language_code': r.target_word.language_code
                    },
                    'metadata': r.metadata
                }
                for r in self.outgoing_relations
            ]
            
        if hasattr(self, 'incoming_relations'):
            result_dict['incoming_relations'] = [
                {
                    'id': r.id,
                    'relation_type': r.relation_type,
                    'source_word': {
                        'id': r.source_word.id,
                        'lemma': r.source_word.lemma,
                        'language_code': r.source_word.language_code
                    },
                    'metadata': r.metadata
                }
                for r in self.incoming_relations
            ]
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="similar_words")
    def search_similar_words(cls, search_term: str, min_similarity: float = 0.3) -> List[Tuple['Word', float]]:
        """Search for similar words using multiple methods."""
        return cls.search_combined(search_term, 'lemma', min_similarity)
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="pronunciation_search")
    def search_by_pronunciation(cls, pronunciation: str) -> List['Word']:
        """Search by pronunciation using phonetic algorithms."""
        results = set()
        results.update(cls.search_by_metaphone(pronunciation, 'lemma'))
        results.update(cls.search_by_soundex(pronunciation, 'lemma'))
        results.update(cls.search_by_dmetaphone(pronunciation, 'lemma'))
        return list(results)
    
    @classmethod
    @cached_query(timeout=3600, key_prefix="baybayin_search")
    def search_by_baybayin(cls, baybayin_text: str) -> List['Word']:
        """Search by Baybayin text."""
        return cls.query.filter(
            cls.has_baybayin == True,
            cls.baybayin_form.ilike(f'%{baybayin_text}%')
        ).all()
    
    def get_all_forms(self) -> List[str]:
        """Get all forms of the word."""
        forms = [self.lemma]
        if self.baybayin_form:
            forms.append(self.baybayin_form)
        if self.romanized_form:
            forms.append(self.romanized_form)
        if self.badlit_form:
            forms.append(self.badlit_form)
        if hasattr(self, 'forms'):
            forms.extend(form.form for form in self.forms)
        return list(set(forms))
    
    @classmethod
    @cached_query(timeout=1800, key_prefix="similar_words")
    def find_similar_words(cls, word: str, min_similarity: float = 0.3) -> List[Tuple['Word', float]]:
        """Find similar words using combined similarity measures."""
        return cls.search_combined_similarity('lemma', word, {
            'trigram': 0.4,
            'word': 0.4,
            'strict_word': 0.2
        })
    
    @classmethod
    @cached_query(timeout=1800, key_prefix="similar_normalized")
    def find_similar_normalized(cls, word: str, min_similarity: float = 0.3) -> List[Tuple['Word', float]]:
        """Find similar words using normalized form."""
        return cls.search_combined_similarity('normalized_lemma', word.lower(), {
            'trigram': 0.4,
            'word': 0.4,
            'strict_word': 0.2
        })
    
    @classmethod
    @cached_query(timeout=1800, key_prefix="baybayin_similarity")
    def find_by_baybayin(cls, baybayin_text: str, min_similarity: float = 0.3) -> List[Tuple['Word', float]]:
        """Find words by baybayin text using trigram similarity."""
        return cls.search_trigram_similarity('baybayin_form', baybayin_text, min_similarity)
    
    @classmethod
    def get_random_word(cls, language_code: Optional[str] = None, min_completeness: float = 0.0) -> Optional['Word']:
        """Get a random word, optionally filtered by language and completeness."""
        query = cls.query
        
        # Filter by language if specified
        if language_code:
            query = query.filter(cls.language_code == language_code)
            
        # Filter by minimum completeness score if specified
        if min_completeness > 0:
            query = query.filter(cls.completeness_score >= min_completeness)
            
        # Get a random row using SQL random() function
        return query.order_by(func.random()).first()
    
    @classmethod
    def get_words_by_completeness(cls, min_score: float = 0.5, 
                                limit: int = 100, 
                                language_code: Optional[str] = None) -> List['Word']:
        """Get words with minimum completeness score."""
        query = cls.query.filter(cls.completeness_score >= min_score)
        
        if language_code:
            query = query.filter(cls.language_code == language_code)
            
        return query.order_by(cls.completeness_score.desc()).limit(limit).all() 