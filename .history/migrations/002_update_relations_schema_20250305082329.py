#!/usr/bin/env python3
"""
Migration 002: Update relations schema with confidence scores and metadata.
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add parent directory to path to import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dictionary_manager import get_connection, logger

def run_migration():
    """Run the migration."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Add new columns to relations table
        logger.info("Adding new columns to relations table...")
        cur.execute("""
            ALTER TABLE relations
            ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.5,
            ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS context TEXT,
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;
        """)
        
        # Create index on confidence for filtering
        logger.info("Creating index on confidence...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_confidence 
            ON relations (confidence DESC);
        """)
        
        # Create index on metadata for JSON querying
        logger.info("Creating index on metadata...")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_metadata 
            ON relations USING GIN (metadata);
        """)
        
        # Add trigger to update updated_at
        logger.info("Adding update trigger...")
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_relations_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            DROP TRIGGER IF EXISTS update_relations_updated_at_trigger ON relations;
            
            CREATE TRIGGER update_relations_updated_at_trigger
                BEFORE UPDATE ON relations
                FOR EACH ROW
                EXECUTE FUNCTION update_relations_updated_at();
        """)
        
        # Create view for relationship analysis
        logger.info("Creating relationship analysis view...")
        cur.execute("""
            CREATE OR REPLACE VIEW relationship_analysis AS
            SELECT 
                r.id,
                w1.lemma as from_word,
                w2.lemma as to_word,
                r.relation_type,
                r.confidence,
                r.context,
                r.metadata,
                r.created_at,
                r.updated_at
            FROM relations r
            JOIN words w1 ON r.from_word_id = w1.id
            JOIN words w2 ON r.to_word_id = w2.id
            WHERE r.confidence >= 0.3
            ORDER BY r.confidence DESC;
        """)
        
        # Create function to get related words with confidence
        logger.info("Creating get_related_words function...")
        cur.execute("""
            CREATE OR REPLACE FUNCTION get_related_words(
                p_word_id INTEGER,
                p_min_confidence FLOAT DEFAULT 0.3,
                p_relation_type TEXT DEFAULT NULL
            )
            RETURNS TABLE (
                related_word TEXT,
                relation_type TEXT,
                confidence FLOAT,
                context TEXT,
                metadata JSONB
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    w2.lemma,
                    r.relation_type,
                    r.confidence,
                    r.context,
                    r.metadata
                FROM relations r
                JOIN words w2 ON r.to_word_id = w2.id
                WHERE r.from_word_id = p_word_id
                    AND r.confidence >= p_min_confidence
                    AND (p_relation_type IS NULL OR r.relation_type = p_relation_type)
                ORDER BY r.confidence DESC;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        # Create function to find word clusters
        logger.info("Creating find_word_clusters function...")
        cur.execute("""
            CREATE OR REPLACE FUNCTION find_word_clusters(
                p_min_confidence FLOAT DEFAULT 0.5,
                p_max_distance INTEGER DEFAULT 2
            )
            RETURNS TABLE (
                cluster_id INTEGER,
                words TEXT[],
                avg_confidence FLOAT
            ) AS $$
            WITH RECURSIVE
            word_graph AS (
                SELECT 
                    from_word_id as word_id,
                    ARRAY[from_word_id] as path,
                    0 as distance,
                    confidence
                FROM relations
                WHERE confidence >= p_min_confidence
                
                UNION
                
                SELECT 
                    r.to_word_id,
                    wg.path || r.to_word_id,
                    wg.distance + 1,
                    (wg.confidence + r.confidence) / 2
                FROM relations r
                JOIN word_graph wg ON r.from_word_id = wg.word_id
                WHERE r.confidence >= p_min_confidence
                    AND wg.distance < p_max_distance
                    AND NOT r.to_word_id = ANY(wg.path)
            ),
            clusters AS (
                SELECT 
                    DENSE_RANK() OVER (ORDER BY path[1]) as cluster_id,
                    ARRAY_AGG(DISTINCT w.lemma) as words,
                    AVG(confidence) as avg_confidence
                FROM word_graph wg
                JOIN words w ON w.id = ANY(wg.path)
                GROUP BY path[1]
                HAVING COUNT(DISTINCT wg.word_id) > 1
            )
            SELECT * FROM clusters
            ORDER BY avg_confidence DESC;
            $$ LANGUAGE SQL;
        """)
        
        # Create function to analyze relationship paths
        logger.info("Creating analyze_relationship_path function...")
        cur.execute("""
            CREATE OR REPLACE FUNCTION analyze_relationship_path(
                p_word1_id INTEGER,
                p_word2_id INTEGER,
                p_max_distance INTEGER DEFAULT 3
            )
            RETURNS TABLE (
                path TEXT[],
                total_confidence FLOAT,
                relationship_types TEXT[]
            ) AS $$
            WITH RECURSIVE
            paths AS (
                SELECT 
                    ARRAY[w1.lemma] as path,
                    ARRAY[]::TEXT[] as rel_types,
                    from_word_id,
                    0 as distance,
                    1.0 as confidence
                FROM relations r
                JOIN words w1 ON r.from_word_id = w1.id
                WHERE from_word_id = p_word1_id
                
                UNION ALL
                
                SELECT 
                    p.path || w2.lemma,
                    p.rel_types || r.relation_type,
                    r.to_word_id,
                    p.distance + 1,
                    p.confidence * r.confidence
                FROM relations r
                JOIN paths p ON r.from_word_id = p.from_word_id
                JOIN words w2 ON r.to_word_id = w2.id
                WHERE r.to_word_id != ALL(
                    SELECT id FROM unnest(p.path) WITH ORDINALITY AS t(word, idx)
                    JOIN words w ON w.lemma = t.word
                )
                AND p.distance < p_max_distance
            )
            SELECT 
                path,
                confidence as total_confidence,
                rel_types as relationship_types
            FROM paths
            WHERE from_word_id = p_word2_id
            ORDER BY confidence DESC;
            $$ LANGUAGE SQL;
        """)
        
        conn.commit()
        logger.info("Migration completed successfully")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_migration() 