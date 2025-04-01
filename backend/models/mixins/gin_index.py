"""
GIN index mixin for models.
"""

from sqlalchemy import Index, text
from sqlalchemy.ext.declarative import declared_attr
from typing import List, Dict, Any

class GINIndexMixin:
    """Mixin to add GIN indexing capabilities to models."""
    
    @declared_attr
    def __gin_indexes__(cls) -> List[Dict[str, Any]]:
        """Define GIN indexes for the model."""
        return []
    
    @classmethod
    def create_gin_indexes(cls):
        """Create GIN indexes for the model."""
        for index_def in cls.__gin_indexes__:
            field_name = index_def['field']
            index_type = index_def.get('type', 'gin')
            opclass = index_def.get('opclass', None)
            
            index_name = f'idx_{cls.__tablename__}_{field_name}_gin'
            
            if opclass:
                Index(
                    index_name,
                    text(f"{field_name} {opclass}"),
                    postgresql_using=index_type
                )
            else:
                Index(
                    index_name,
                    text(field_name),
                    postgresql_using=index_type
                )
    
    @classmethod
    def create_tsvector_trigger(cls, field_name: str, config: str = 'simple'):
        """Create a tsvector update trigger for full text search."""
        trigger_name = f'tsvector_update_{cls.__tablename__}_{field_name}'
        trigger_function = f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT OR UPDATE ON {cls.__tablename__}
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger(
                {field_name},
                'pg_catalog.{config}',
                {', '.join(cls.__ts_vector_fields__)}
            );
        """
        return text(trigger_function)
    
    @classmethod
    def create_jsonb_path_ops_index(cls, field_name: str):
        """Create a GIN index with jsonb_path_ops for JSONB fields."""
        index_name = f'idx_{cls.__tablename__}_{field_name}_jsonb_path'
        return Index(
            index_name,
            text(f"{field_name} jsonb_path_ops"),
            postgresql_using='gin'
        )
    
    @classmethod
    def create_trgm_index(cls, field_name: str):
        """Create a GIN index with gin_trgm_ops for text fields."""
        index_name = f'idx_{cls.__tablename__}_{field_name}_trgm'
        return Index(
            index_name,
            text(f"{field_name} gin_trgm_ops"),
            postgresql_using='gin'
        )
    
    @classmethod
    def create_btree_gin_index(cls, field_name: str):
        """Create a GIN index with btree_gin extension."""
        index_name = f'idx_{cls.__tablename__}_{field_name}_btree_gin'
        return Index(
            index_name,
            text(field_name),
            postgresql_using='gin'
        )
    
    @classmethod
    def create_array_index(cls, field_name: str):
        """Create a GIN index for array fields."""
        index_name = f'idx_{cls.__tablename__}_{field_name}_array'
        return Index(
            index_name,
            text(field_name),
            postgresql_using='gin'
        )
    
    @classmethod
    def create_composite_gin_index(cls, fields: List[str], opclass: str = None):
        """Create a composite GIN index."""
        field_str = '_'.join(fields)
        index_name = f'idx_{cls.__tablename__}_{field_str}_gin'
        
        if opclass:
            field_expr = ', '.join(f"{f} {opclass}" for f in fields)
        else:
            field_expr = ', '.join(fields)
        
        return Index(
            index_name,
            text(field_expr),
            postgresql_using='gin'
        )
    
    @classmethod
    def create_partial_gin_index(cls, field_name: str, condition: str, opclass: str = None):
        """Create a partial GIN index."""
        index_name = f'idx_{cls.__tablename__}_{field_name}_partial_gin'
        
        if opclass:
            field_expr = f"{field_name} {opclass}"
        else:
            field_expr = field_name
        
        return Index(
            index_name,
            text(field_expr),
            postgresql_using='gin',
            postgresql_where=text(condition)
        ) 