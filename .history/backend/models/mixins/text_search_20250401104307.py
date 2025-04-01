"""
Text search mixin for models.
"""

from sqlalchemy import func, text
from typing import List, Tuple

class TextSearchMixin:
    """Mixin to add text search capabilities to models."""
    
    @classmethod
    def search_by_similarity(cls, search_term: str, field_name: str, min_similarity: float = 0.3) -> List[Tuple['TextSearchMixin', float]]:
        """Search by text similarity using trigrams."""
        return cls.query.with_entities(
            cls,
            func.similarity(getattr(cls, field_name), search_term).label('similarity')
        ).filter(
            func.similarity(getattr(cls, field_name), search_term) >= min_similarity
        ).order_by(
            func.similarity(getattr(cls, field_name), search_term).desc()
        ).all()
    
    @classmethod
    def search_by_levenshtein(cls, search_term: str, field_name: str, max_distance: int = 3) -> List[Tuple['TextSearchMixin', int]]:
        """Search by Levenshtein distance."""
        return cls.query.with_entities(
            cls,
            func.levenshtein(getattr(cls, field_name), search_term).label('distance')
        ).filter(
            func.levenshtein(getattr(cls, field_name), search_term) <= max_distance
        ).order_by(
            func.levenshtein(getattr(cls, field_name), search_term)
        ).all()
    
    @classmethod
    def search_by_metaphone(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search by Metaphone phonetic encoding."""
        return cls.query.filter(
            func.metaphone(getattr(cls, field_name), 10) == func.metaphone(search_term, 10)
        ).all()
    
    @classmethod
    def search_by_soundex(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search by Soundex phonetic encoding."""
        return cls.query.filter(
            func.soundex(getattr(cls, field_name)) == func.soundex(search_term)
        ).all()
    
    @classmethod
    def search_unaccented(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search ignoring accents."""
        return cls.query.filter(
            func.unaccent(getattr(cls, field_name)).ilike(func.unaccent(f'%{search_term}%'))
        ).all()
    
    @classmethod
    def search_by_word_similarity(cls, search_term: str, field_name: str, min_similarity: float = 0.3) -> List[Tuple['TextSearchMixin', float]]:
        """Search by word similarity."""
        return cls.query.with_entities(
            cls,
            func.word_similarity(getattr(cls, field_name), search_term).label('similarity')
        ).filter(
            func.word_similarity(getattr(cls, field_name), search_term) >= min_similarity
        ).order_by(
            func.word_similarity(getattr(cls, field_name), search_term).desc()
        ).all()
    
    @classmethod
    def search_by_strict_word_similarity(cls, search_term: str, field_name: str, min_similarity: float = 0.3) -> List[Tuple['TextSearchMixin', float]]:
        """Search by strict word similarity."""
        return cls.query.with_entities(
            cls,
            func.strict_word_similarity(getattr(cls, field_name), search_term).label('similarity')
        ).filter(
            func.strict_word_similarity(getattr(cls, field_name), search_term) >= min_similarity
        ).order_by(
            func.strict_word_similarity(getattr(cls, field_name), search_term).desc()
        ).all()
    
    @classmethod
    def search_by_daitch_mokotoff(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search by Daitch-Mokotoff Soundex."""
        return cls.query.filter(
            func.daitch_mokotoff(getattr(cls, field_name)).overlap(
                func.daitch_mokotoff(search_term)
            )
        ).all()
    
    @classmethod
    def search_by_dmetaphone(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search by Double Metaphone."""
        return cls.query.filter(
            func.dmetaphone(getattr(cls, field_name)) == func.dmetaphone(search_term)
        ).all()
    
    @classmethod
    def search_by_dmetaphone_alt(cls, search_term: str, field_name: str) -> List['TextSearchMixin']:
        """Search by Double Metaphone alternate encoding."""
        return cls.query.filter(
            func.dmetaphone_alt(getattr(cls, field_name)) == func.dmetaphone_alt(search_term)
        ).all()
    
    @classmethod
    def search_combined(cls, search_term: str, field_name: str, min_similarity: float = 0.3) -> List[Tuple['TextSearchMixin', float]]:
        """Combined search using multiple methods."""
        # Start with trigram similarity
        results = set(r[0].id for r in cls.search_by_similarity(search_term, field_name, min_similarity))
        
        # Add Levenshtein matches
        results.update(r[0].id for r in cls.search_by_levenshtein(search_term, field_name))
        
        # Add phonetic matches
        results.update(r.id for r in cls.search_by_metaphone(search_term, field_name))
        results.update(r.id for r in cls.search_by_soundex(search_term, field_name))
        
        # Add unaccented matches
        results.update(r.id for r in cls.search_unaccented(search_term, field_name))
        
        # Get full records and calculate final similarity
        return cls.query.with_entities(
            cls,
            (func.similarity(getattr(cls, field_name), search_term) +
             func.word_similarity(getattr(cls, field_name), search_term)) / 2.0
        ).filter(
            cls.id.in_(results)
        ).order_by(
            ((func.similarity(getattr(cls, field_name), search_term) +
              func.word_similarity(getattr(cls, field_name), search_term)) / 2.0).desc()
        ).all() 