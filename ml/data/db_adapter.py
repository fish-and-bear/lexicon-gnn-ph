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
import os  # Added for path operations
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
        self.logger = logger
        self.connection_params = db_config
        self.db_type = self.connection_params.get('db_type', 'postgres').lower()
        
        # Default table names, can be overridden if db_config provides them
        default_table_names = {
            "lemmas": "words",
            "relations": "relations",
            "definitions": "definitions",
            "etymologies": "etymologies",
            "pronunciations": "pronunciations",
            "word_forms": "word_forms",
            "affixations": "affixations",
            "pos": "parts_of_speech" # Or 'pos_tags' as another common default
        }
        # Allow db_config to override table names if a 'table_names' key is present
        self.table_names = self.connection_params.get('table_names', default_table_names)
        
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
                
                # Check if the path exists and is absolute
                if not os.path.isabs(db_path):
                    # Try various relative paths
                    possible_paths = [
                        db_path,  # Current working directory
                        os.path.join(os.getcwd(), db_path),  # Explicit current working directory
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', db_path),  # Project root
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db_path),  # Direct from ml directory
                        os.path.normpath(os.path.join(os.getcwd(), '..', db_path))  # One level up
                    ]
                    
                    # Try the main project root location as well (absolute path)
                    main_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..'))
                    possible_paths.append(os.path.join(main_root, db_path))
                    
                    # Find the first path that exists
                    for path in possible_paths:
                        if os.path.exists(path):
                            db_path = path
                            logger.info(f"Found database file at: {db_path}")
                            break
                
                # Log paths we're checking
                logger.info(f"Connecting to SQLite database at {db_path}")
                logger.info(f"Absolute path: {os.path.abspath(db_path)}")
                
                # Check if file exists
                if not os.path.exists(db_path):
                    logger.error(f"SQLite database file not found: {db_path}")
                    logger.error(f"Current working directory: {os.getcwd()}")
                    raise FileNotFoundError(f"SQLite database file not found: {db_path}")
                
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
        self._ensure_connection()
        results = []
        cur = None  # Initialize cursor for SQLite
        try:
            if self.db_type == 'postgres':
                 # For PostgreSQL, use RealDictCursor within its own context
                with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as dict_cur:
                    dict_cur.execute(query, params)
                    results = dict_cur.fetchall()
            elif self.db_type == 'sqlite':
                cur = self.conn.cursor()  # Get a standard cursor
                cur.execute(query, params or [])  # params must be a list/tuple for sqlite
                # Fetchall returns list of sqlite3.Row objects due to row_factory
                # Convert sqlite3.Row to plain dict
                results = [dict(row) for row in cur.fetchall()]
            
            # No explicit commit/rollback here as these are SELECT queries primarily.
            # Connection-level context management (if used) or explicit commit in DML methods
            # would handle transactions.

            return results
        except (sqlite3.Error, psycopg2.Error) as e:
            logger.error(f"Error executing query for {self.db_type}: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            # Rollback is typically handled at a higher transaction level or by connection context manager
            raise
        finally:
            if self.db_type == 'sqlite' and cur:
                try:
                    cur.close()  # Explicitly close SQLite cursor
                except Exception as e_close: # pylint: disable=broad-except
                    logger.warning(f"Could not close SQLite cursor: {e_close}")

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
        """Get affixation data."""
        query = """
            SELECT id, word_id, affix_type, affix_form, description
            FROM affixations
        """
        try:
            results = self._execute_query(query)
            df = pd.DataFrame(results)
            self.logger.info(f"Fetched {len(df)} affixations from {self.db_type} database")
            return df
        except sqlite3.OperationalError as e:
            self.logger.warning(f"Could not fetch affixations from {self.table_names.get('affixations', 'affixations')} table due to OperationalError (e.g., missing column like 'word_id'): {e}. Returning empty DataFrame.")
            return pd.DataFrame()
        except Exception as e: # Catch other potential errors during fetch
            self.logger.error(f"An unexpected error occurred while fetching affixations: {e}")
            return pd.DataFrame()

    def get_pos_df(self) -> Optional[pd.DataFrame]:
        """Get parts of speech definitions from the database."""
        # Attempt to restore a functional get_pos_df method.
        # Common table names for POS tags could be 'parts_of_speech' or 'pos_tags'.
        # Common columns are 'id', 'code', 'name', 'description'.
        # The user might need to adjust table_name or column names if this guess is incorrect.
        table_name = self.table_names.get('pos', 'parts_of_speech') # Default to 'parts_of_speech'
        query = f"""
            SELECT 
                id, 
                code, 
                name,  -- Assuming a general 'name' column
                name_en, -- Or specific language names if available
                name_tl, 
                description
            FROM {table_name}
        """
        # Fallback if 'name' doesn't exist, try only specific ones
        fallback_query = f"""
            SELECT 
                id, 
                code, 
                name_en, 
                name_tl, 
                description
            FROM {table_name}
        """

        try:
            self.logger.info(f"Fetching POS data from table: {table_name}")
            # Try the query with a general 'name' column first
            try:
                results = self._execute_query(query)
            except Exception as e_general_name:
                self.logger.warning(f"Query with general 'name' column failed for POS: {e_general_name}. Trying fallback query.")
                results = self._execute_query(fallback_query)
            
            df = pd.DataFrame(results)
            self.logger.info(f"Successfully fetched POS data: {df.shape[0]} rows from {table_name}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching POS data from {table_name}: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    def get_all_dataframes(self, languages_to_include: Optional[List[str]] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Fetches all relevant dataframes from the database.

        Args:
            languages_to_include: Optional list of language codes (e.g., ['tl', 'en']) 
                                  to filter lemmas. If None, lemmas for all languages are fetched.

        Returns:
            Dict[str, Optional[pd.DataFrame]]: A dictionary where keys are dataframe names
                                                (e.g., 'lemmas_df', 'relations_df') and
                                                values are the corresponding pandas DataFrames.
                                                Returns an empty DataFrame for a key if fetching fails.
        """
        self.logger.info(f"Fetching all dataframes... Languages for lemmas_df: {languages_to_include if languages_to_include else 'all'}")
        data_frames: Dict[str, Optional[pd.DataFrame]] = {} # Ensure type hint for empty dict
        
        fetch_map = {
            "lemmas_df": self.get_lemmas_df,
            "definitions_df": self.get_definitions_df,
            "etymologies_df": self.get_etymologies_df,
            "relations_df": self.get_relations_df,
            "pos_df": self.get_pos_df,
            "pronunciations_df": self.get_pronunciations_df,
            "word_forms_df": self.get_word_forms_df,
            "affixations_df": self.get_affixations_df
        }

        for name, fetch_func in fetch_map.items():
            try:
                if hasattr(self, fetch_func.__name__):
                    if name == "lemmas_df":
                        df_val = fetch_func(target_languages=languages_to_include)
                    else:
                        df_val = fetch_func()
                    
                    data_frames[name] = df_val
                    if df_val is not None:
                        self.logger.info(f"Successfully fetched '{name}': {df_val.shape[0]} rows, {df_val.shape[1]} cols")
                    else:
                        self.logger.warning(f"Fetching '{name}' resulted in None. Storing as empty DataFrame.")
                        data_frames[name] = pd.DataFrame() 
                else:
                    self.logger.warning(f"Method {fetch_func.__name__} not found in DatabaseAdapter. Skipping for '{name}'.")
                    data_frames[name] = pd.DataFrame()
            except Exception as e_fetch:
                self.logger.error(f"Error fetching dataframe '{name}': {e_fetch}", exc_info=True)
                data_frames[name] = pd.DataFrame()

        self.logger.info("Finished fetching all dataframes attempt.")
        return data_frames

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
        """Execute a custom SQL query and return results."""
        return self._execute_query(query, params)

    def close(self):
        """Close database connection."""
        # Check if 'conn' attribute exists and if it's not None
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
                # Use getattr for db_type as well, in case __init__ failed very early
                db_type_str = getattr(self, 'db_type', 'unknown type')
                logger.info(f"{db_type_str} database connection closed.")
                self.conn = None # Set to None after closing
            except (sqlite3.Error, psycopg2.Error) as e:
                db_type_str = getattr(self, 'db_type', 'unknown type')
                logger.error(f"Error closing {db_type_str} database connection: {e}")
        else:
            # Log slightly differently if 'conn' attribute didn't even exist vs. it was None
            if not hasattr(self, 'conn'):
                logger.info("DatabaseAdapter object did not have a 'conn' attribute to close (possibly due to an early initialization error).")
            else: # self.conn was None or didn't exist but hasattr was false (should be caught by first check)
                db_type_str = getattr(self, 'db_type', 'unknown type')
                logger.info(f"No active {db_type_str} database connection to close (conn was None or attribute missing).")

    def __del__(self):
        # This will now call the more robust close method
        self.close() 