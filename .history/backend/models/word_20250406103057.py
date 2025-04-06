"""
Word model definition with optimized performance, caching, and API response formatting.
"""

from backend.database import db, cached_query
from datetime import datetime
from sqlalchemy.orm import validates
from .base_model import BaseModel
from .mixins.text_search import TextSearchMixin
from .mixins.gin_index import GINIndexMixin
from .mixins.trigram_search import TrigramSearchMixin
import json
from typing import List, Tuple, Dict, Any, Optional, Set, Union
from sqlalchemy.sql import func, and_, or_, text
from sqlalchemy.dialects.postgresql import TSVECTOR
import re
from sqlalchemy.orm import column_property, lazyload, joinedload, selectinload
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
    idioms = db.Column(db.JSON, default=lambda: [])
    word_metadata = db.Column(db.JSON, default=lambda: {})
    data_hash = db.Column(db.Text)
    search_text = db.Column(TSVECTOR)
    badlit_form = db.Column(db.Text)
    hyphenation = db.Column(db.JSON, default=lambda: {})
    is_proper_noun = db.Column(db.Boolean, default=False)
    is_abbreviation = db.Column(db.Boolean, default=False)
    is_initialism = db.Column(db.Boolean, default=False)
    source_info = db.Column(db.JSON, default=lambda: {})
    pronunciation_data = db.Column(db.JSON, default=lambda: {})
    
    @hybrid_property
    def completeness_score(self) -> float:
        """Calculate the completeness score based on available data."""
        score = 0.0
        
        # Base score for having definitions
        if hasattr(self, 'definitions') and self.definitions:
            score += 0.3
            
            # Additional score for having multiple definitions
            if len(self.definitions) > 1:
                score += 0.05
                
            # Additional score for having examples
            if any(d.examples for d in self.definitions if d.examples):
                score += 0.05
        
        # Score for having etymology
        if hasattr(self, 'etymologies') and self.etymologies:
            score += 0.15
            
        # Score for having pronunciation
        if hasattr(self, 'pronunciations') and self.pronunciations:
            score += 0.1
            
        # Score for having Baybayin form
        if self.has_baybayin and self.baybayin_form:
            score += 0.1
            
        # Score for having relations
        if (hasattr(self, 'outgoing_relations') and self.outgoing_relations) or \
           (hasattr(self, 'incoming_relations') and self.incoming_relations):
            score += 0.15
            
        # Score for having affixations
        if (hasattr(self, 'root_affixations') and self.root_affixations) or \
           (hasattr(self, 'affixed_affixations') and self.affixed_affixations):
            score += 0.1
            
        return min(1.0, score)  # Cap at 1.0
    
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
    
    # Add relationship for WordForm
    forms = db.relationship('WordForm', 
                            back_populates='word', 
                            lazy='selectin',
                            cascade='all, delete-orphan')
    
    # Add relationship for WordTemplate
    templates = db.relationship('WordTemplate', 
                                back_populates='word', 
                                lazy='selectin',
                                cascade='all, delete-orphan')
    
    __table_args__ = (
        db.UniqueConstraint('normalized_lemma', 'language_code', name='words_lang_lemma_uniq'),
        db.CheckConstraint(
            "(has_baybayin = false AND baybayin_form IS NULL) OR (has_baybayin = true AND baybayin_form IS NOT NULL)",
            name='baybayin_form_check'
        ),
        db.CheckConstraint(
            "baybayin_form ~ '^[\\u1700-\\u171F[:space:]]*$' OR baybayin_form IS NULL",
            name='baybayin_form_regex'
        ),
        db.Index('idx_words_is_properties', 'is_proper_noun', 'is_abbreviation', 'is_initialism'),
        db.Index('idx_words_lemma', 'lemma'),
        db.Index('idx_words_normalized', 'normalized_lemma'),
        db.Index('idx_words_baybayin', 'baybayin_form'),
        db.Index('idx_words_romanized', 'romanized_form'),
        db.Index('idx_words_language', 'language_code'),
        db.Index('idx_words_root', 'root_word_id'),
    )
    
    # Optimized GIN indexes for Filipino language features
    __gin_indexes__ = [
        {'field': 'search_text', 'type': 'tsvector'},
        {'field': 'tags', 'opclass': 'gin_trgm_ops', 'type': 'gin'},
        {'field': 'idioms', 'opclass': 'jsonb_path_ops', 'type': 'gin'},
        {'field': 'word_metadata', 'opclass': 'jsonb_path_ops', 'type': 'gin'},
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
    BAYBAYIN_PATTERN = r'[\\u1700-\\u171F]'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.root_word_id is None:
            self.is_root = True
        else:
            self.is_root = False
    
    @validates('lemma')
    def validate_lemma(self, key: str, value: str) -> str:
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
        self._is_modified = True
        return value
    
    @validates('normalized_lemma')
    def validate_normalized_lemma(self, key: str, value: str) -> str:
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
        self._is_modified = True
        return value
    
    @validates('language_code')
    def validate_language_code(self, key: str, value: str) -> str:
        """Validate language code."""
        if not value:
            return 'tl'  # Default to Tagalog
        if not isinstance(value, str):
            raise ValueError("Language code must be a string")
        value = value.strip().lower()
        if not re.match(r'^[a-z]{2,3}$', value):
            raise ValueError("Language code must be a valid ISO 639-1 or 639-2 code")
        self._is_modified = True
        return value
    
    @validates('baybayin_form')
    def validate_baybayin_form(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate baybayin form."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Baybayin form must be a string")
        value = value.strip()
        if not value:
            return None
        if not re.match(r'^[\\u1700-\\u171F\\s]*$', value):
            raise ValueError("Baybayin form must contain only valid Baybayin characters")
        self._is_modified = True
        return value
    
    @validates('root_word_id')
    def validate_root_word_id(self, key: str, value: Optional[int]) -> Optional[int]:
        """Update is_root when root_word_id changes."""
        self._is_modified = True
        return value
    
    @hybrid_property
    def is_root(self):
        return self.root_word_id is None
    
    @property
    def get_definitions_count(self) -> int:
        """Get count of definitions."""
        return len(self.definitions) if hasattr(self, 'definitions') else 0
    
    @property 
    def has_etymology(self) -> bool:
        """Check if word has etymology information."""
        return bool(hasattr(self, 'etymologies') and self.etymologies)
    
    @property
    def has_pronunciation(self) -> bool:
        """Check if word has pronunciation data."""
        return bool(hasattr(self, 'pronunciations') and self.pronunciations)
    
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
    
    def invalidate_cache(self) -> None:
        """Invalidate word-related caches."""
        # Define patterns for cache invalidation based on this model
        patterns = [
            f"*word*{self.id}*",
            f"*word_dict*{self.id}*",
            f"*similar_words*{self.normalized_lemma[:10]}*",
            f"*baybayin_similarity*{self.baybayin_form[:10] if self.baybayin_form else ''}*"
        ]
        
        from backend.database import invalidate_cache
        for pattern in patterns:
            invalidate_cache(pattern)
    
    def __repr__(self) -> str:
        return f'<Word {self.id}: {self.lemma}>'
    
    @cached_query(timeout=3600, key_prefix="word_dict")
    def to_dict(self, include_related: bool = False, include_ids_only: bool = False) -> Dict[str, Any]:
        """
        Convert word to dictionary with optional related data.
        
        Args:
            include_related: Include all related data (definitions, etymologies, etc.)
            include_ids_only: If True, only include IDs for related data to reduce payload size
        """
        # Base dictionary with essential fields
        result = {
            'id': self.id,
            'lemma': self.lemma,
            'normalized_lemma': self.normalized_lemma,
            'language_code': self.language_code,
            'has_baybayin': self.has_baybayin,
            'baybayin_form': self.baybayin_form,
            'romanized_form': self.romanized_form,
            'root_word_id': self.root_word_id,
            'is_root': self.is_root,
            'completeness_score': self.completeness_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        
        # Add optional detailed fields
        if not include_ids_only:
            result.update({
                'preferred_spelling': self.preferred_spelling,
                'tags': self.tags.split(',') if self.tags else [],
                'idioms': self.idioms or {},
                'word_metadata': self.word_metadata or {},
                'badlit_form': self.badlit_form,
                'hyphenation': self.hyphenation or {},
                'is_proper_noun': self.is_proper_noun,
                'is_abbreviation': self.is_abbreviation,
                'is_initialism': self.is_initialism,
                'pronunciation_data': self.pronunciation_data or {},
                'source_info': self.source_info or {},
                'data_completeness': {
                    'has_definitions': self.get_definitions_count > 0,
                    'has_etymology': self.has_etymology,
                    'has_pronunciation': self.has_pronunciation,
                    'definitions_count': self.get_definitions_count
                }
            })
        
        # Include related data if requested
        if include_related:
            if include_ids_only:
                self._add_related_ids_to_dict(result)
            else:
                self._add_related_data_to_dict(result)
            
        return result
    
    def _add_related_data_to_dict(self, result_dict: Dict[str, Any]) -> None:
        """Add complete related data to dictionary representation."""
        # Add definitions
        if hasattr(self, 'definitions'):
            result_dict['definitions'] = [d.to_dict() for d in self.definitions]
            
        # Add etymologies
        if hasattr(self, 'etymologies'):
            result_dict['etymologies'] = [e.to_dict() for e in self.etymologies]
            
        # Add pronunciations
        if hasattr(self, 'pronunciations'):
            result_dict['pronunciations'] = [p.to_dict() for p in self.pronunciations]
            
        # Add credits
        if hasattr(self, 'credits'):
            result_dict['credits'] = [c.to_dict() for c in self.credits]
            
        # Add relations
        if hasattr(self, 'outgoing_relations'):
            result_dict['outgoing_relations'] = [
                {
                    'id': r.id,
                    'relation_type': r.relation_type,
                    'target_word': {
                        'id': r.target_word.id,
                        'lemma': r.target_word.lemma,
                        'language_code': r.target_word.language_code,
                        'has_baybayin': r.target_word.has_baybayin,
                        'baybayin_form': r.target_word.baybayin_form
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
                        'language_code': r.source_word.language_code,
                        'has_baybayin': r.source_word.has_baybayin,
                        'baybayin_form': r.source_word.baybayin_form
                    },
                    'metadata': r.metadata
                }
                for r in self.incoming_relations
            ]
            
        # Add affixations
        if hasattr(self, 'root_affixations'):
            result_dict['root_affixations'] = [
                {
                    'id': a.id,
                    'affix_type': a.affix_type,
                    'affixed_word': {
                        'id': a.affixed_word.id,
                        'lemma': a.affixed_word.lemma,
                        'language_code': a.affixed_word.language_code
                    }
                }
                for a in self.root_affixations
            ]
            
        if hasattr(self, 'affixed_affixations'):
            result_dict['affixed_affixations'] = [
                {
                    'id': a.id,
                    'affix_type': a.affix_type,
                    'root_word': {
                        'id': a.root_word.id,
                        'lemma': a.root_word.lemma,
                        'language_code': a.root_word.language_code
                    }
                }
                for a in self.affixed_affixations
            ]
            
        # Add root word
        if self.root_word_id and hasattr(self, 'root_word') and self.root_word:
            result_dict['root_word'] = {
                'id': self.root_word.id,
                'lemma': self.root_word.lemma,
                'language_code': self.root_word.language_code,
                'has_baybayin': self.root_word.has_baybayin,
                'baybayin_form': self.root_word.baybayin_form
            }
            
        # Add derived words
        if hasattr(self, 'derived_words') and self.derived_words:
            result_dict['derived_words'] = [
                {
                    'id': d.id,
                    'lemma': d.lemma,
                    'language_code': d.language_code,
                    'has_baybayin': d.has_baybayin,
                    'baybayin_form': d.baybayin_form
                }
                for d in self.derived_words
            ]
            
        # Add forms
        if hasattr(self, 'forms') and self.forms:
            result_dict['forms'] = [f.to_dict() for f in self.forms]
            
        # Add templates
        if hasattr(self, 'templates') and self.templates:
            result_dict['templates'] = [t.to_dict() for t in self.templates]
    
    def _add_related_ids_to_dict(self, result_dict: Dict[str, Any]) -> None:
        """Add only related IDs to dictionary representation for lighter payload."""
        # Add definition IDs
        if hasattr(self, 'definitions'):
            result_dict['definition_ids'] = [d.id for d in self.definitions]
            
        # Add etymology IDs
        if hasattr(self, 'etymologies'):
            result_dict['etymology_ids'] = [e.id for e in self.etymologies]
            
        # Add pronunciation IDs
        if hasattr(self, 'pronunciations'):
            result_dict['pronunciation_ids'] = [p.id for p in self.pronunciations]
            
        # Add relation IDs
        if hasattr(self, 'outgoing_relations'):
            result_dict['outgoing_relation_ids'] = [
                {'id': r.id, 'relation_type': r.relation_type, 'target_id': r.to_word_id}
                for r in self.outgoing_relations
            ]
            
        if hasattr(self, 'incoming_relations'):
            result_dict['incoming_relation_ids'] = [
                {'id': r.id, 'relation_type': r.relation_type, 'source_id': r.from_word_id}
                for r in self.incoming_relations
            ]
            
        # Add affixation IDs
        if hasattr(self, 'root_affixations'):
            result_dict['root_affixation_ids'] = [
                {'id': a.id, 'affix_type': a.affix_type, 'affixed_id': a.affixed_word_id}
                for a in self.root_affixations
            ]
            
        if hasattr(self, 'affixed_affixations'):
            result_dict['affixed_affixation_ids'] = [
                {'id': a.id, 'affix_type': a.affix_type, 'root_id': a.root_word_id}
                for a in self.affixed_affixations
            ]
            
        # Add form IDs
        if hasattr(self, 'forms'):
            result_dict['form_ids'] = [f.id for f in self.forms]
            
        # Add template IDs
        if hasattr(self, 'templates'):
            result_dict['template_ids'] = [t.id for t in self.templates]
    
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
    
    @classmethod
    @cached_query(timeout=1800, key_prefix="word_verification")
    def get_verified_words(cls, language_code: Optional[str] = None, limit: int = 100) -> List['Word']:
        """Get verified words, optionally filtered by language."""
        query = cls.query.filter(cls.verified == True)
        
        if language_code:
            query = query.filter(cls.language_code == language_code)
            
        return query.order_by(cls.updated_at.desc()).limit(limit).all()
    
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