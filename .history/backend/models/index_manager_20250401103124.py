"""
Index manager for database optimization.
"""

from sqlalchemy import event, DDL
from database import db
import logging

logger = logging.getLogger(__name__)

class IndexManager:
    """Manage database indexes for optimization."""
    
    @staticmethod
    def create_text_search_index(table_name: str, column_name: str, language: str = 'simple'):
        """Create a GIN index for text search."""
        index_name = f'idx_{table_name}_{column_name}_tsv'
        return DDL(f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {table_name} 
            USING gin(to_tsvector('{language}', {column_name}));
        """)
    
    @staticmethod
    def create_trigram_index(table_name: str, column_name: str):
        """Create a GIN index for trigram search."""
        index_name = f'idx_{table_name}_{column_name}_trgm'
        return DDL(f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {table_name} 
            USING gin({column_name} gin_trgm_ops);
        """)
    
    @staticmethod
    def create_btree_index(table_name: str, column_name: str, unique: bool = False):
        """Create a B-tree index."""
        index_name = f'idx_{table_name}_{column_name}'
        unique_str = 'UNIQUE' if unique else ''
        return DDL(f"""
            CREATE {unique_str} INDEX IF NOT EXISTS {index_name} 
            ON {table_name} 
            USING btree({column_name});
        """)
    
    @staticmethod
    def create_partial_index(table_name: str, column_name: str, condition: str):
        """Create a partial index."""
        index_name = f'idx_{table_name}_{column_name}_partial'
        return DDL(f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {table_name}({column_name}) 
            WHERE {condition};
        """)
    
    @staticmethod
    def create_composite_index(table_name: str, columns: list, unique: bool = False):
        """Create a composite index."""
        column_str = '_'.join(columns)
        index_name = f'idx_{table_name}_{column_str}'
        unique_str = 'UNIQUE' if unique else ''
        return DDL(f"""
            CREATE {unique_str} INDEX IF NOT EXISTS {index_name} 
            ON {table_name} 
            USING btree({', '.join(columns)});
        """)

# Register indexes for specific models
def setup_indexes():
    """Set up all database indexes."""
    try:
        # Word indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_text_search_index('words', 'lemma')
        )
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_trigram_index('words', 'normalized_lemma')
        )
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_partial_index('words', 'baybayin_form', 'has_baybayin = true')
        )
        
        # Definition indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_text_search_index('definitions', 'definition_text', 'english')
        )
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_composite_index('definitions', ['word_id', 'standardized_pos_id'])
        )
        
        # Etymology indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_text_search_index('etymologies', 'language_codes')
        )
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_text_search_index('etymologies', 'etymology_text')
        )
        
        # Relation indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_composite_index('relations', ['from_word_id', 'to_word_id', 'relation_type'], unique=True)
        )
        
        # Word form indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_trigram_index('word_forms', 'form')
        )
        
        # Definition category indexes
        event.listen(
            db.metadata,
            'after_create',
            IndexManager.create_composite_index('definition_categories', ['definition_id', 'category_kind'])
        )
        
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Failed to create database indexes: {str(e)}")
        raise 