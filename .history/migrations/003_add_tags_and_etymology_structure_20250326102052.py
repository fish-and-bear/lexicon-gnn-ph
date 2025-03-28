#!/usr/bin/env python
"""
Migration script to add the 'tags' column to the definitions table
and 'etymology_structure' column to the etymologies table.
Also adds a metadata JSONB column to the relations table.
"""
import logging
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_migration():
    """Run the migration to add columns for tags and etymology structure."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Add 'tags' column to definitions table if it doesn't exist
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'definitions' AND column_name = 'tags'
            ) THEN
                ALTER TABLE definitions ADD COLUMN tags JSONB;
                CREATE INDEX IF NOT EXISTS idx_definitions_tags ON definitions USING GIN (tags);
            END IF;
        END
        $$;
        """)
        logger.info("Added 'tags' column to definitions table (if it didn't exist)")
        
        # Add 'etymology_structure' column to etymologies table if it doesn't exist
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'etymologies' AND column_name = 'etymology_structure'
            ) THEN
                ALTER TABLE etymologies ADD COLUMN etymology_structure JSONB;
                CREATE INDEX IF NOT EXISTS idx_etymologies_etymology_structure ON etymologies USING GIN (etymology_structure);
            END IF;
        END
        $$;
        """)
        logger.info("Added 'etymology_structure' column to etymologies table (if it didn't exist)")
        
        # Add 'metadata' column to relations table if it doesn't exist
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'relations' AND column_name = 'metadata'
            ) THEN
                ALTER TABLE relations ADD COLUMN metadata JSONB;
                CREATE INDEX IF NOT EXISTS idx_relations_metadata ON relations USING GIN (metadata);
            END IF;
        END
        $$;
        """)
        logger.info("Added 'metadata' column to relations table (if it didn't exist)")
        
        # Create jsonb_merge function for combining metadata entries
        cur.execute("""
        CREATE OR REPLACE FUNCTION jsonb_merge(a jsonb, b jsonb)
        RETURNS jsonb AS $$
        BEGIN
            RETURN (
                SELECT jsonb_object_agg(
                    COALESCE(ka, kb),
                    CASE
                        WHEN (va ? 'type') AND (va->>'type' = 'array') AND (vb ? 'type') AND (vb->>'type' = 'array')
                        THEN jsonb_build_object('type', 'array', 'value', 
                            (SELECT jsonb_agg(value) FROM (
                                SELECT DISTINCT value
                                FROM jsonb_array_elements(va->'value') 
                                UNION
                                SELECT DISTINCT value
                                FROM jsonb_array_elements(vb->'value')
                            ) arr)
                        )
                        WHEN (va::text <> '{}' AND vb::text <> '{}') AND 
                             (jsonb_typeof(va) = 'object' AND jsonb_typeof(vb) = 'object')
                        THEN jsonb_merge(va, vb)
                        ELSE COALESCE(vb, va)
                    END
                )
                FROM jsonb_each(a) a_fields(ka, va)
                FULL JOIN jsonb_each(b) b_fields(kb, vb) ON ka = kb
            );
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
        """)
        logger.info("Created or replaced jsonb_merge function")
        
        conn.commit()
        logger.info("Migration completed successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    run_migration() 