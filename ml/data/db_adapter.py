"""
Database adapter for fetching lexical data from PostgreSQL.

This module provides functionality to load data from the Filipino lexical database
into pandas DataFrames for processing by the ML pipeline.
"""

import logging
import pandas as pd
import psycopg2
import psycopg2.extras
import sqlite3  # Added
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

class DatabaseAdapter:
    """
    Adapter for connecting to and querying the Filipino lexical database.
    Supports both PostgreSQL and SQLite.
    """
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize database connection based on configuration.
        
        Args:
            db_config: Dictionary containing database configuration.
                       Expected keys for postgres: host, port, dbname, user, password, [application_name]
                       Expected keys for sqlite: db_path
                       Optional key: db_type ('postgres' or 'sqlite', defaults to 'postgres')
        """
        self.connection_params = db_config
        self.db_type = self.connection_params.get('db_type', 'postgres').lower()
        
        # Validate required params
        if self.db_type == 'sqlite':
            if 'db_path' not in self.connection_params:
                raise ValueError("Missing 'db_path' in db_config for SQLite connection")
        elif self.db_type == 'postgres':
            required_pg_keys = ['host', 'port', 'dbname', 'user', 'password']
            if not all(key in self.connection_params for key in required_pg_keys):
                missing = [key for key in required_pg_keys if key not in self.connection_params]
                raise ValueError(f"Missing keys in db_config for PostgreSQL connection: {missing}")
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

        self.conn = None
        self._connect()
        
    def _connect(self):
        """Establish database connection based on db_type."""
        try:
            if self.db_type == 'sqlite':
                db_path = self.connection_params['db_path']
                logger.info(f"Connecting to SQLite database at {db_path}")
                self.conn = sqlite3.connect(db_path)
                self.conn.row_factory = sqlite3.Row  # Return rows that behave like dicts
                logger.info("SQLite connection established")
            
            elif self.db_type == 'postgres':
                pg_params = {
                    k: v for k, v in self.connection_params.items() 
                    if k in ['host', 'port', 'dbname', 'user', 'password', 'application_name']
                }
                # Set default application_name if not provided
                pg_params.setdefault('application_name', 'FilRelex-ML') 
                
                logger.info(f"Connecting to PostgreSQL database {pg_params['dbname']} on {pg_params['host']}")
                self.conn = psycopg2.connect(**pg_params)
                logger.info("PostgreSQL connection established")

        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Failed to connect to {self.db_type} database: {e}")
            self.conn = None # Ensure conn is None on failure
            raise
    
    def _ensure_connection(self):
        """Ensure connection is active, reconnect if needed."""
        # For SQLite, connect() creates the file if it doesn't exist, so checking self.conn is enough.
        # For psycopg2, conn.closed tells us if it's disconnected.
        connection_active = False
        if self.conn:
            if self.db_type == 'postgres':
                try:
                    # Check if the connection is closed
                    connection_active = not self.conn.closed
                    # Optionally, ping the server for a more robust check
                    # with self.conn.cursor() as cur:
                    #     cur.execute("SELECT 1")
                    # connection_active = True
                except psycopg2.Error:
                    connection_active = False 
            elif self.db_type == 'sqlite':
                 # Simple check: if self.conn exists, assume it's usable unless an operation fails
                 # sqlite3 doesn't have a built-in conn.closed or conn.ping()
                 connection_active = True 

        if not connection_active:
            logger.warning(f"{self.db_type} database connection not active or closed, reconnecting...")
            self._connect()
        
        if not self.conn: # If reconnect failed
             raise ConnectionError(f"Failed to establish or re-establish connection to {self.db_type} database.")

    def _execute_query(self, query: str, params: Optional[Union[List, Tuple, Dict]] = None) -> List[Dict]:
        """Execute a query and return results as a list of dictionaries."""
        self._ensure_connection()
        results = []
        try:
            with self.conn.cursor() as cur: # Get a standard cursor first
                if self.db_type == 'postgres':
                     # Use RealDictCursor specifically for PostgreSQL if needed, or process rows later
                     # Recreating the cursor with a different factory:
                     with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as dict_cur:
                        dict_cur.execute(query, params)
                        results = dict_cur.fetchall()
                elif self.db_type == 'sqlite':
                    cur.execute(query, params or []) # params must be a list/tuple for sqlite
                    # Fetchall returns list of sqlite3.Row objects due to row_factory
                    # Convert sqlite3.Row to plain dict
                    results = [dict(row) for row in cur.fetchall()]

            return results
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Error executing query for {self.db_type}: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    def get_lemmas_df(self, target_languages: Optional[List[str]] = None) -> pd.DataFrame:
        """Get words/lemmas from the database."""
        query = """
            SELECT 
                id, lemma, normalized_lemma, language_code, root_word_id, 
                is_proper_noun, is_abbreviation, is_initialism, 
                has_baybayin, baybayin_form, romanized_form, badlit_form,
                tags, word_metadata, hyphenation, pronunciation_data
            FROM words
        """
        
        params: Union[List, Tuple] = []
        if target_languages:
            if self.db_type == 'sqlite':
                placeholders = ', '.join('?' * len(target_languages))
                query += f" WHERE language_code IN ({placeholders})"
                params = target_languages # Use list directly for sqlite
            elif self.db_type == 'postgres':
                query += " WHERE language_code IN %s"
                params = (tuple(target_languages),) # Use tuple within a tuple for psycopg2 IN %s

        results = self._execute_query(query, params)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} words from {self.db_type} database")
        return df
    
    def get_relations_df(self) -> pd.DataFrame:
        """Get word relations from the database."""
        query = """
            SELECT 
                id, from_word_id, to_word_id, relation_type
                -- is_automatic, confidence_score, metadata -- Columns may not exist in SQLite export
            FROM relations
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} relations from {self.db_type} database")
        return df
    
    def get_definitions_df(self) -> pd.DataFrame:
        """Get word definitions from the database."""
        query = """
            SELECT 
                id, word_id, definition_text, standardized_pos_id, original_pos,
                sources, examples, usage_notes, tags, definition_metadata
            FROM definitions
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} definitions from {self.db_type} database")
        return df
    
    def get_etymologies_df(self) -> pd.DataFrame:
        """Get word etymologies from the database."""
        query = """
            SELECT 
                id, word_id, etymology_text, language_codes, normalized_components,
                etymology_structure, sources
                 -- metadata, is_validated -- Columns may not exist in SQLite export
            FROM etymologies
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} etymologies from {self.db_type} database")
        return df
    
    def get_pronunciations_df(self) -> pd.DataFrame:
        """Get word pronunciations from the database."""
        query = """
            SELECT 
                id, word_id, type, value, tags, pronunciation_metadata
                -- dialect, notes, audio_url -- Columns may not exist in SQLite export
            FROM pronunciations
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} pronunciations from {self.db_type} database")
        return df
    
    def get_word_forms_df(self) -> pd.DataFrame:
        """Get word forms from the database."""
        query = """
            SELECT 
                id, word_id, form, is_canonical, is_primary, 
                tags, sources
                -- standardized_pos_id, inflection_data -- Columns may not exist in SQLite export
            FROM word_forms
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} word forms from {self.db_type} database")
        return df

    def get_affixations_df(self) -> pd.DataFrame:
        """Get affixation data from the database."""
        query = """
            SELECT 
                id, root_word_id, affixed_word_id, affix_type, sources
            FROM affixations
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} affixations from {self.db_type} database")
        return df

    def get_pos_df(self) -> pd.DataFrame:
        """Get parts of speech definitions."""
        query = """
            SELECT 
                id, code, name_en, name_tl, description
            FROM parts_of_speech
        """
        results = self._execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"Fetched {len(df)} parts of speech from {self.db_type} database")
        return df

    # --- Methods below might need more significant adjustments for SQLite vs PG syntax/functions ---

    def get_relation_types(self) -> List[str]:
        """Get distinct relation types present in the relations table."""
        query = "SELECT DISTINCT relation_type FROM relations ORDER BY relation_type;"
        try:
            results = self._execute_query(query)
            relation_types = [row['relation_type'] for row in results]
            logger.info(f"Found {len(relation_types)} distinct relation types")
            return relation_types
        except Exception as e:
            logger.error(f"Error fetching relation types: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics about the graph data in the database."""
        logger.info("Calculating graph statistics...")
        stats = {}
        table_counts = [
            ('words', 'num_words'), 
            ('definitions', 'num_definitions'), 
            ('relations', 'num_relations'), 
            ('etymologies', 'num_etymologies')
        ]
        
        for table, key in table_counts:
            try:
                # Query adjustments might be needed if table/column names differ, but COUNT(*) is standard
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = self._execute_query(count_query)
                stats[key] = result[0]['count'] if result else 0
            except Exception as e:
                logger.error(f"Error getting count for table {table}: {e}")
                stats[key] = -1 # Indicate error

        try:
            # This query uses string aggregation which differs between PG and SQLite
            if self.db_type == 'postgres':
                rel_type_query = """SELECT relation_type, COUNT(*) as count 
                                    FROM relations GROUP BY relation_type ORDER BY count DESC;"""
            elif self.db_type == 'sqlite':
                # GROUP_CONCAT is the SQLite equivalent of STRING_AGG (though order isn't guaranteed by default)
                 rel_type_query = """SELECT relation_type, COUNT(*) as count 
                                    FROM relations GROUP BY relation_type ORDER BY count DESC;"""
            else:
                 raise NotImplementedError(f"Relation type stats not implemented for {self.db_type}")

            results = self._execute_query(rel_type_query)
            stats['relation_type_counts'] = {row['relation_type']: row['count'] for row in results}
        except Exception as e:
             logger.error(f"Error getting relation type counts: {e}")
             stats['relation_type_counts'] = {}
        
        logger.info(f"Graph statistics: {stats}")
        return stats

    def execute_custom_query(self, query: str, params: Optional[Union[List, Tuple, Dict]] = None) -> List[Dict]:
        """Execute an arbitrary query provided by the user."""
        logger.info(f"Executing custom query: {query[:100]}... Params: {params}")
        # This directly uses _execute_query, which handles the db_type difference
        return self._execute_query(query, params)

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                logger.info(f"{self.db_type} database connection closed")
                self.conn = None
            except (sqlite3.Error, psycopg2.Error) as e:
                logger.error(f"Error closing {self.db_type} connection: {e}")

    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        self.close() 