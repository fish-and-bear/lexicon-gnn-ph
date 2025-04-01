"""
Mixin for trigram search functionality.
"""

from sqlalchemy import func, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from backend.database import db

class TrigramSearchMixin:
    """Mixin for trigram search functionality."""
    
    @classmethod
    def __declare_last__(cls):
        """Create trigram index after table creation."""
        if hasattr(cls, '__table__'):
            Index(
                f'idx_{cls.__tablename__}_trigram_search',
                cls.search_text,
                postgresql_using='gin',
                postgresql_ops={'search_text': 'gin_trgm_ops'}
            )
    
    def calculate_similarity_score(self, query: str) -> float:
        """Calculate similarity score between word and query."""
        return db.session.scalar(
            func.similarity(self.search_text, query)
        ) or 0.0