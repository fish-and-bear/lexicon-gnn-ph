"""
Base models initialization and utilities.
"""

import logging
import threading
from sqlalchemy import text
from flask import Flask, current_app
from typing import Optional, Set, Dict, Any
from datetime import datetime
import time
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Global state
_initialized_features: Set[str] = set()
_state_lock = threading.RLock()  # Reentrant lock for all state changes

def is_feature_initialized(feature: str) -> bool:
    """Check if a feature has been initialized."""
    with _state_lock:
        return feature in _initialized_features

def mark_feature_initialized(feature: str) -> None:
    """Mark a feature as initialized."""
    with _state_lock:
        _initialized_features.add(feature)

@contextmanager
def feature_lock(feature: str):
    """Lock for feature initialization."""
    with _state_lock:
        if is_feature_initialized(feature):
            yield False  # Feature already initialized
            return
        yield True  # Feature needs initialization

def check_extensions(session) -> bool:
    """Check if required extensions are installed."""
    try:
        result = session.execute(text("""
            SELECT extname FROM pg_extension 
            WHERE extname IN ('pg_trgm', 'unaccent', 'fuzzystrmatch');
        """))
        installed = {row[0] for row in result}
        required = {'pg_trgm', 'unaccent', 'fuzzystrmatch'}
        missing = required - installed
        
        if missing:
            logger.warning(f"Missing extensions: {missing}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check extensions: {str(e)}")
        return False

def setup_extensions(session):
    """Set up PostgreSQL extensions."""
    with feature_lock('extensions') as needs_init:
        if not needs_init:
            logger.info("Extensions already initialized")
            return
            
        try:
            # Create extensions
            session.execute(text("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
            """))
            session.commit()
            mark_feature_initialized('extensions')
            logger.info("PostgreSQL extensions created successfully")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL extensions: {str(e)}")
            session.rollback()
            raise

def check_tables_exist(session) -> bool:
    """Check if required tables exist."""
    try:
        result = session.execute(text("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' AND 
            tablename IN ('words', 'definitions', 'etymologies', 'pronunciations', 'credits', 'relations', 'parts_of_speech');
        """))
        existing = {row[0] for row in result}
        required = {'words', 'definitions', 'etymologies', 'pronunciations', 'credits', 'relations', 'parts_of_speech'}
        missing = required - existing
        
        if missing:
            logger.warning(f"Missing tables: {missing}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check tables: {str(e)}")
        return False

def setup_timestamp_trigger(session):
    """Set up automatic timestamp updates."""
    with feature_lock('timestamp_trigger') as needs_init:
        if not needs_init:
            logger.info("Timestamp triggers already initialized")
            return
            
        if not check_tables_exist(session):
            logger.warning("Tables not ready for timestamp triggers")
            return
            
        try:
            # Create update timestamp function
            session.execute(text("""
                CREATE OR REPLACE FUNCTION update_timestamp()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
            """))
            
            # Create triggers for automatic timestamp updates
            table_names = ['words', 'definitions', 'etymologies', 'pronunciations', 'credits', 'relations']
            for table in table_names:
                session.execute(text(f"""
                    DROP TRIGGER IF EXISTS update_{table}_timestamp ON {table};
                    CREATE TRIGGER update_{table}_timestamp
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_timestamp();
                """))
            
            session.commit()
            mark_feature_initialized('timestamp_trigger')
            logger.info("Timestamp triggers created successfully")
        except Exception as e:
            logger.error(f"Failed to create timestamp triggers: {str(e)}")
            session.rollback()
            raise

def setup_text_search(session):
    """Set up text search configuration."""
    with feature_lock('text_search') as needs_init:
        if not needs_init:
            logger.info("Text search already initialized")
            return
            
        try:
            session.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_ts_config WHERE cfgname = 'filipino'
                    ) THEN
                        CREATE TEXT SEARCH CONFIGURATION filipino (COPY = simple);
                        ALTER TEXT SEARCH CONFIGURATION filipino
                            ALTER MAPPING FOR asciiword, word, numword, asciihword, hword, numhword
                            WITH unaccent, simple;
                    END IF;
                END;
                $$;
            """))
            session.commit()
            mark_feature_initialized('text_search')
            logger.info("Text search configuration created successfully")
        except Exception as e:
            logger.error(f"Failed to create text search configuration: {str(e)}")
            session.rollback()
            raise

def setup_indexes(session):
    """Set up database indexes."""
    with feature_lock('indexes') as needs_init:
        if not needs_init:
            logger.info("Indexes already initialized")
            return
            
        if not check_tables_exist(session):
            logger.warning("Tables not ready for indexes")
            return
            
        try:
            # Basic indexes
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
                CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = TRUE;
                CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
                CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);
                
                CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
                CREATE INDEX IF NOT EXISTS idx_definitions_pos ON definitions(standardized_pos_id);
                
                CREATE INDEX IF NOT EXISTS idx_etymologies_word ON etymologies(word_id);
                
                CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
                CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
                CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
                
                CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
                CREATE INDEX IF NOT EXISTS idx_pronunciations_type ON pronunciations(type);
                CREATE INDEX IF NOT EXISTS idx_pronunciations_value ON pronunciations(value);
                
                CREATE INDEX IF NOT EXISTS idx_credits_word ON credits(word_id);
            """))
            session.commit()

            # Full text search indexes - only if the columns exist
            try:
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
                    CREATE INDEX IF NOT EXISTS idx_definitions_text ON definitions USING gin(to_tsvector('english', definition_text));
                    CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));
                """))
                session.commit()
            except Exception as e:
                logger.warning(f"Some full text search indexes could not be created: {str(e)}")
                session.rollback()

            mark_feature_initialized('indexes')
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create database indexes: {str(e)}")
            session.rollback()
            raise

def setup_search_trigger(session):
    """Set up search vector update trigger."""
    with feature_lock('search_trigger') as needs_init:
        if not needs_init:
            logger.info("Search trigger already initialized")
            return
            
        if not check_tables_exist(session):
            logger.warning("Tables not ready for search trigger")
            return
            
        try:
            session.execute(text("""
                DROP TRIGGER IF EXISTS tsvectorupdate ON words;
                CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
                ON words FOR EACH ROW EXECUTE FUNCTION
                tsvector_update_trigger(search_text, 'pg_catalog.simple', lemma, normalized_lemma, baybayin_form, romanized_form);
            """))
            session.commit()
            mark_feature_initialized('search_trigger')
            logger.info("Search vector trigger created successfully")
        except Exception as e:
            logger.error(f"Failed to create search vector trigger: {str(e)}")
            session.rollback()
            raise

def setup_parts_of_speech(session):
    """Set up standard parts of speech."""
    with feature_lock('parts_of_speech') as needs_init:
        if not needs_init:
            logger.info("Parts of speech already initialized")
            return
            
        if not check_tables_exist(session):
            logger.warning("Tables not ready for parts of speech")
            return
            
        from models import PartOfSpeech
        
        standard_pos = [
            {'code': 'n', 'name_en': 'Noun', 'name_tl': 'Pangngalan'},
            {'code': 'v', 'name_en': 'Verb', 'name_tl': 'Pandiwa'},
            {'code': 'adj', 'name_en': 'Adjective', 'name_tl': 'Pang-uri'},
            {'code': 'adv', 'name_en': 'Adverb', 'name_tl': 'Pang-abay'},
            {'code': 'pron', 'name_en': 'Pronoun', 'name_tl': 'Panghalip'},
            {'code': 'prep', 'name_en': 'Preposition', 'name_tl': 'Pang-ukol'},
            {'code': 'conj', 'name_en': 'Conjunction', 'name_tl': 'Pangatnig'},
            {'code': 'intj', 'name_en': 'Interjection', 'name_tl': 'Pandamdam'},
            {'code': 'det', 'name_en': 'Determiner', 'name_tl': 'Pantukoy'},
            {'code': 'affix', 'name_en': 'Affix', 'name_tl': 'Panlapi'}
        ]
        
        try:
            # Get existing codes
            existing = {pos.code for pos in session.query(PartOfSpeech).all()}
            
            # Add missing parts of speech
            for pos_data in standard_pos:
                if pos_data['code'] not in existing:
                    pos = PartOfSpeech(**pos_data)
                    session.add(pos)
            
            session.commit()
            mark_feature_initialized('parts_of_speech')
            logger.info("Standard parts of speech initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize parts of speech: {str(e)}")
            session.rollback()
            raise

def init_app(app: Flask) -> None:
    """Initialize the models with the Flask app."""
    from models import db
    
    try:
        with app.app_context():
            # Create all tables first
            db.create_all()
            db.session.commit()
            logger.info("Database tables created successfully")
            
            # Set up database features in order
            setup_extensions(db.session)
            setup_text_search(db.session)
            
            # Wait for tables to be ready
            retries = 3
            while retries > 0 and not check_tables_exist(db.session):
                logger.warning(f"Waiting for tables to be ready, {retries} retries left")
                time.sleep(1)
                retries -= 1
            
            if not check_tables_exist(db.session):
                raise Exception("Tables not ready after waiting")
            
            # Set up remaining features
            setup_timestamp_trigger(db.session)
            setup_indexes(db.session)
            setup_search_trigger(db.session)
            setup_parts_of_speech(db.session)
            
            # Final commit
            db.session.commit()
            logger.info("Database models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database models: {str(e)}")
        db.session.rollback()
        raise 