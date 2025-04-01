"""
Trigram search mixin for models.
"""

from sqlalchemy import func, text, Float
from typing import List, Tuple, Optional
from sqlalchemy.sql.expression import cast

class TrigramSearchMixin:
    """Mixin to add trigram-based search capabilities to models."""
    
    @classmethod
    def set_similarity_threshold(cls, threshold: float = 0.3) -> None:
        """Set the similarity threshold for trigram searches."""
        from database import db
        db.session.execute(text(f"SELECT set_limit({threshold})"))
        db.session.commit()
    
    @classmethod
    def get_similarity_threshold(cls) -> float:
        """Get the current similarity threshold."""
        from database import db
        result = db.session.execute(text("SELECT show_limit()")).scalar()
        return float(result)
    
    @classmethod
    def get_trigrams(cls, text_value: str) -> List[str]:
        """Get trigrams for a text value."""
        from database import db
        result = db.session.execute(
            text("SELECT show_trgm(:text)"),
            {'text': text_value}
        ).scalar()
        return result
    
    @classmethod
    def search_trigram_similarity(cls, field_name: str, search_term: str, min_similarity: Optional[float] = None) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using trigram similarity."""
        query = cls.query.with_entities(
            cls,
            func.similarity(getattr(cls, field_name), search_term).label('similarity')
        )
        
        if min_similarity is not None:
            query = query.filter(func.similarity(getattr(cls, field_name), search_term) >= min_similarity)
        
        return query.order_by(func.similarity(getattr(cls, field_name), search_term).desc()).all()
    
    @classmethod
    def search_word_similarity(cls, field_name: str, search_term: str, min_similarity: Optional[float] = None) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using word similarity."""
        query = cls.query.with_entities(
            cls,
            func.word_similarity(getattr(cls, field_name), search_term).label('similarity')
        )
        
        if min_similarity is not None:
            query = query.filter(func.word_similarity(getattr(cls, field_name), search_term) >= min_similarity)
        
        return query.order_by(func.word_similarity(getattr(cls, field_name), search_term).desc()).all()
    
    @classmethod
    def search_strict_word_similarity(cls, field_name: str, search_term: str, min_similarity: Optional[float] = None) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using strict word similarity."""
        query = cls.query.with_entities(
            cls,
            func.strict_word_similarity(getattr(cls, field_name), search_term).label('similarity')
        )
        
        if min_similarity is not None:
            query = query.filter(func.strict_word_similarity(getattr(cls, field_name), search_term) >= min_similarity)
        
        return query.order_by(func.strict_word_similarity(getattr(cls, field_name), search_term).desc()).all()
    
    @classmethod
    def search_similarity_distance(cls, field_name: str, search_term: str, max_distance: float = 0.7) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using similarity distance."""
        query = cls.query.with_entities(
            cls,
            func.similarity_dist(getattr(cls, field_name), search_term).label('distance')
        )
        
        if max_distance is not None:
            query = query.filter(func.similarity_dist(getattr(cls, field_name), search_term) <= max_distance)
        
        return query.order_by(func.similarity_dist(getattr(cls, field_name), search_term)).all()
    
    @classmethod
    def search_word_similarity_distance(cls, field_name: str, search_term: str, max_distance: float = 0.7) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using word similarity distance."""
        query = cls.query.with_entities(
            cls,
            func.word_similarity_dist(getattr(cls, field_name), search_term).label('distance')
        )
        
        if max_distance is not None:
            query = query.filter(func.word_similarity_dist(getattr(cls, field_name), search_term) <= max_distance)
        
        return query.order_by(func.word_similarity_dist(getattr(cls, field_name), search_term)).all()
    
    @classmethod
    def search_strict_word_similarity_distance(cls, field_name: str, search_term: str, max_distance: float = 0.7) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using strict word similarity distance."""
        query = cls.query.with_entities(
            cls,
            func.strict_word_similarity_dist(getattr(cls, field_name), search_term).label('distance')
        )
        
        if max_distance is not None:
            query = query.filter(func.strict_word_similarity_dist(getattr(cls, field_name), search_term) <= max_distance)
        
        return query.order_by(func.strict_word_similarity_dist(getattr(cls, field_name), search_term)).all()
    
    @classmethod
    def search_combined_similarity(cls, field_name: str, search_term: str, weights: Optional[dict] = None) -> List[Tuple['TrigramSearchMixin', float]]:
        """Search using a weighted combination of similarity measures."""
        if weights is None:
            weights = {
                'trigram': 0.4,
                'word': 0.3,
                'strict_word': 0.3
            }
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Build weighted similarity expression
        similarity_expr = (
            weights.get('trigram', 0) * func.similarity(getattr(cls, field_name), search_term) +
            weights.get('word', 0) * func.word_similarity(getattr(cls, field_name), search_term) +
            weights.get('strict_word', 0) * func.strict_word_similarity(getattr(cls, field_name), search_term)
        ).label('similarity')
        
        return cls.query.with_entities(
            cls,
            similarity_expr
        ).order_by(similarity_expr.desc()).all()
    
    @classmethod
    def find_similar_pairs(cls, field_name: str, min_similarity: float = 0.3) -> List[Tuple['TrigramSearchMixin', 'TrigramSearchMixin', float]]:
        """Find pairs of records with similar field values."""
        t1 = cls.__table__.alias('t1')
        t2 = cls.__table__.alias('t2')
        
        query = cls.query.with_entities(
            cls,
            cls,
            func.similarity(
                getattr(t1.c, field_name),
                getattr(t2.c, field_name)
            ).label('similarity')
        ).filter(
            t1.c.id < t2.c.id,
            func.similarity(
                getattr(t1.c, field_name),
                getattr(t2.c, field_name)
            ) >= min_similarity
        ).order_by(
            func.similarity(
                getattr(t1.c, field_name),
                getattr(t2.c, field_name)
            ).desc()
        )
        
        return query.all() 