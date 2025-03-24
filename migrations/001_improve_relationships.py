#!/usr/bin/env python3
"""
Migration 001: Improve relationship handling and data organization.
"""

import sys
import os
import json
from typing import Dict, List, Set, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path to import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.dictionary_manager import (
    get_connection, logger, normalize_lemma, 
    RelationshipAnalyzer, get_or_create_word_id,
    insert_relation, insert_etymology, insert_definition
)

class RelationshipMigration:
    """Handles migration of improved relationship data."""
    
    def __init__(self, conn):
        self.conn = conn
        self.analyzer = RelationshipAnalyzer()
        self.processed_words = set()
        self.relationship_cache = {}
        self.etymology_cache = {}
        
    def migrate(self):
        """Execute the migration."""
        try:
            # First pass: Extract and cache relationships
            self._extract_relationships()
            
            # Second pass: Process and store relationships
            self._process_relationships()
            
            # Third pass: Validate and enhance
            self._enhance_relationships()
            
            # Final pass: Cleanup and optimize
            self._cleanup()
            
            self.conn.commit()
            logger.info("Relationship migration completed successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Migration failed: {str(e)}")
            raise

    def _extract_relationships(self):
        """First pass: Extract relationships from all sources."""
        with self.conn.cursor() as cur:
            # Process kaikki.jsonl entries
            self._process_kaikki_entries(cur)
            
            # Process root words
            self._process_root_words(cur)
            
            # Process KWF dictionary
            self._process_kwf_entries(cur)

    def _process_kaikki_entries(self, cur):
        """Process entries from kaikki.jsonl for relationships."""
        logger.info("Processing Kaikki entries...")
        
        try:
            with open('data/kaikki.jsonl', 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Extracting Kaikki relationships"):
                    try:
                        entry = json.loads(line)
                        word = entry.get('word')
                        if not word:
                            continue
                            
                        # Extract etymology relationships
                        if 'etymology_templates' in entry:
                            for template in entry['etymology_templates']:
                                rel = self._process_etymology_template(template)
                                if rel:
                                    self._cache_relationship(word, rel)
                        
                        # Extract derived forms
                        if 'derived' in entry:
                            for derived in entry['derived']:
                                if isinstance(derived, dict):
                                    derived_word = derived.get('word')
                                else:
                                    derived_word = derived
                                if derived_word:
                                    self._cache_relationship(word, {
                                        'type': 'derives',
                                        'word': derived_word,
                                        'confidence': 0.9,
                                        'source': 'kaikki.jsonl'
                                    })
                        
                        # Extract semantic relationships from senses
                        if 'senses' in entry:
                            for sense in entry['senses']:
                                # Process links
                                for link_type, link_word in sense.get('links', []):
                                    self._cache_relationship(word, {
                                        'type': self._map_link_type(link_type),
                                        'word': link_word,
                                        'confidence': 0.8,
                                        'source': 'kaikki.jsonl',
                                        'context': str(sense.get('glosses', []))
                                    })
                                
                                # Process examples
                                for example in sense.get('examples', []):
                                    if isinstance(example, dict):
                                        text = example.get('text', '')
                                        translation = example.get('english', '')
                                        self._process_example_relationships(word, text, translation)
                                    
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing Kaikki entry: {str(e)}")
                        continue
                        
        except FileNotFoundError:
            logger.warning("kaikki.jsonl not found, skipping...")
            return

    def _process_root_words(self, cur):
        """Process root words relationships."""
        logger.info("Processing root words relationships...")
        
        try:
            with open('data/root_words_with_associated_words_cleaned.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for root, entries in tqdm(data.items(), desc="Processing root words"):
                    for word, info in entries.items():
                        # Cache root relationship
                        if word != root:
                            self._cache_relationship(word, {
                                'type': 'root',
                                'word': root,
                                'confidence': 1.0,
                                'source': 'root_words_with_associated_words_cleaned.json'
                            })
                        
                        # Extract relationships from definition
                        definition = info.get('definition', '')
                        if definition:
                            self._process_definition_relationships(word, definition)
                            
        except FileNotFoundError:
            logger.warning("root_words_with_associated_words_cleaned.json not found, skipping...")
            return

    def _process_kwf_entries(self, cur):
        """Process KWF dictionary entries for relationships."""
        logger.info("Processing KWF dictionary relationships...")
        
        try:
            with open('data/kwf_dictionary.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for entry in tqdm(data, desc="Processing KWF entries"):
                    word = entry.get('word')
                    if not word:
                        continue
                    
                    # Process synonyms
                    for pos_defs in entry.get('definitions', {}).values():
                        for def_entry in pos_defs:
                            for synonym in def_entry.get('synonyms', []):
                                if synonym and synonym.strip():
                                    self._cache_relationship(word, {
                                        'type': 'synonym',
                                        'word': synonym,
                                        'confidence': 0.9,
                                        'source': 'kwf_dictionary.json',
                                        'context': def_entry.get('meaning', '')
                                    })
                    
                    # Process related terms
                    for related in entry.get('related', {}).get('related_terms', []):
                        term = related.get('term', '').strip()
                        if term:
                            self._cache_relationship(word, {
                                'type': 'related',
                                'word': term,
                                'confidence': 0.8,
                                'source': 'kwf_dictionary.json'
                            })
                            
        except FileNotFoundError:
            logger.warning("kwf_dictionary.json not found, skipping...")
            return

    def _process_relationships(self):
        """Second pass: Process and store extracted relationships."""
        logger.info("Processing extracted relationships...")
        
        with self.conn.cursor() as cur:
            for word, relationships in tqdm(self.relationship_cache.items(), desc="Storing relationships"):
                try:
                    # Get or create word ID
                    word_id = get_or_create_word_id(cur, word, language_code='tl')
                    
                    # Process each relationship
                    for rel in relationships:
                        try:
                            related_word = rel['word']
                            # Get or create related word ID
                            related_id = get_or_create_word_id(
                                cur,
                                related_word,
                                language_code=rel.get('language', 'tl')
                            )
                            
                            # Store relationship
                            insert_relation(
                                cur,
                                word_id,
                                related_id,
                                rel['type'],
                                rel['source']
                            )
                            
                            # Store bidirectional relationship if needed
                            if self.analyzer.relationship_types[rel['type']]['bidirectional']:
                                insert_relation(
                                    cur,
                                    related_id,
                                    word_id,
                                    rel['type'],
                                    rel['source']
                                )
                                
                        except Exception as e:
                            logger.error(f"Error processing relationship {rel}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing word {word}: {str(e)}")
                    continue

    def _enhance_relationships(self):
        """Third pass: Validate and enhance relationships."""
        logger.info("Enhancing relationships...")
        
        with self.conn.cursor() as cur:
            # Get all relationships
            cur.execute("""
                SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type,
                       w1.lemma as from_word, w2.lemma as to_word,
                       r.confidence, r.metadata
                FROM relations r
                JOIN words w1 ON r.from_word_id = w1.id
                JOIN words w2 ON r.to_word_id = w2.id
                ORDER BY r.confidence DESC
            """)
            
            relationships = cur.fetchall()
            
            for rel in tqdm(relationships, desc="Enhancing relationships"):
                rel_id, from_id, to_id, rel_type, from_word, to_word, confidence, metadata = rel
                
                try:
                    # Calculate new confidence based on multiple factors
                    new_confidence = self._calculate_relationship_confidence(
                        rel_type, from_word, to_word, confidence
                    )
                    
                    # Update relationship if confidence changed
                    if new_confidence != confidence:
                        cur.execute("""
                            UPDATE relations
                            SET confidence = %s,
                                metadata = metadata || '{"enhanced": true}'::jsonb
                            WHERE id = %s
                        """, (new_confidence, rel_id))
                        
                except Exception as e:
                    logger.error(f"Error enhancing relationship {rel_id}: {str(e)}")
                    continue

    def _cleanup(self):
        """Final pass: Cleanup and optimize."""
        logger.info("Cleaning up relationships...")
        
        with self.conn.cursor() as cur:
            # Remove duplicate relationships
            cur.execute("""
                WITH RankedRelations AS (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY from_word_id, to_word_id, relation_type
                               ORDER BY confidence DESC
                           ) as rn
                    FROM relations
                )
                DELETE FROM relations
                WHERE id IN (
                    SELECT id FROM RankedRelations WHERE rn > 1
                )
            """)
            
            # Remove low confidence relationships
            cur.execute("""
                DELETE FROM relations
                WHERE confidence < 0.3
            """)
            
            # Update search vectors
            cur.execute("""
                UPDATE words
                SET search_text = to_tsvector('english',
                    COALESCE(lemma, '') || ' ' ||
                    COALESCE(normalized_lemma, '') || ' ' ||
                    COALESCE(
                        (
                            SELECT string_agg(w2.lemma, ' ')
                            FROM relations r
                            JOIN words w2 ON r.to_word_id = w2.id
                            WHERE r.from_word_id = words.id
                        ),
                        ''
                    )
                )
            """)

    def _cache_relationship(self, word: str, relationship: Dict):
        """Cache a relationship for later processing."""
        if not word or not relationship or 'type' not in relationship:
            return
            
        if word not in self.relationship_cache:
            self.relationship_cache[word] = []
            
        self.relationship_cache[word].append(relationship)

    def _process_etymology_template(self, template: Dict) -> Optional[Dict]:
        """Process an etymology template into a relationship."""
        template_type = template.get('name')
        args = template.get('args', {})
        
        if template_type == 'inh':  # Inherited
            return {
                'type': 'inherited',
                'from_lang': args.get('2'),
                'to_lang': args.get('1'),
                'word': args.get('3'),
                'confidence': 1.0,
                'source': 'kaikki.jsonl',
                'metadata': {
                    'stage': args.get('4'),
                    'meaning': args.get('5')
                }
            }
        elif template_type == 'bor':  # Borrowed
            return {
                'type': 'borrowed',
                'from_lang': args.get('2'),
                'to_lang': args.get('1'),
                'word': args.get('3'),
                'confidence': 0.9,
                'source': 'kaikki.jsonl'
            }
        elif template_type == 'cog':  # Cognate
            return {
                'type': 'cognate',
                'lang': args.get('1'),
                'word': args.get('2'),
                'confidence': 0.8,
                'source': 'kaikki.jsonl'
            }
        
        return None

    def _map_link_type(self, link_type: str) -> str:
        """Map link types to standard relationship types."""
        mapping = {
            'synonym': 'synonym',
            'antonym': 'antonym',
            'hypernym': 'hypernym',
            'hyponym': 'hyponym',
            'holonym': 'holonym',
            'meronym': 'meronym',
            'derived': 'derived',
            'related': 'related'
        }
        return mapping.get(link_type.lower(), 'related')

    def _process_example_relationships(self, word: str, text: str, translation: str):
        """Process relationships from example sentences."""
        # Extract words from example
        words = set(w.strip().lower() for w in text.split() if len(w.strip()) > 2)
        for w in words:
            if w != word.lower():
                self._cache_relationship(word, {
                    'type': 'occurs_with',
                    'word': w,
                    'confidence': 0.5,
                    'source': 'kaikki.jsonl',
                    'context': text
                })
        
        # Process translation
        if translation:
            eng_words = set(w.strip().lower() for w in translation.split() if len(w.strip()) > 2)
            for w in eng_words:
                self._cache_relationship(word, {
                    'type': 'translates_to',
                    'word': w,
                    'language': 'en',
                    'confidence': 0.6,
                    'source': 'kaikki.jsonl',
                    'context': translation
                })

    def _process_definition_relationships(self, word: str, definition: str):
        """Extract relationships from definition text."""
        for pattern, rel_type in self.analyzer.semantic_patterns.items():
            matches = re.finditer(pattern, definition, re.IGNORECASE)
            for match in matches:
                related_word = match.group(1).strip()
                if related_word and related_word.lower() != word.lower():
                    self._cache_relationship(word, {
                        'type': rel_type,
                        'word': related_word,
                        'confidence': 0.7,
                        'source': 'root_words_with_associated_words_cleaned.json',
                        'context': definition
                    })

    def _calculate_relationship_confidence(
        self, rel_type: str, from_word: str, to_word: str, base_confidence: float
    ) -> float:
        """Calculate relationship confidence score."""
        # Get base weight for relationship type
        type_weight = self.analyzer.relationship_types[rel_type]['weight']
        
        # Calculate word similarity
        similarity = self._calculate_word_similarity(from_word, to_word)
        
        # Calculate final confidence
        confidence = base_confidence * type_weight * (1 + similarity) / 2
        
        return min(1.0, confidence)

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        # Normalize words
        w1 = normalize_lemma(word1)
        w2 = normalize_lemma(word2)
        
        # Calculate Levenshtein ratio
        from difflib import SequenceMatcher
        return SequenceMatcher(None, w1, w2).ratio()

def run_migration():
    """Run the migration."""
    conn = None
    try:
        conn = get_connection()
        migration = RelationshipMigration(conn)
        migration.migrate()
        logger.info("Migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_migration() 