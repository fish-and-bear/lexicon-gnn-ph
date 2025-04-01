"""
Base models initialization and utilities.
"""

import logging
from sqlalchemy import text
from flask import Flask
from typing import Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def setup_extensions(session):
    """Set up PostgreSQL extensions."""
    try:
        # Create extensions
        session.execute(text("""
            CREATE EXTENSION IF NOT EXISTS pg_trgm;
            CREATE EXTENSION IF NOT EXISTS unaccent;
            CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
        """))
        logger.info("PostgreSQL extensions created successfully")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL extensions: {str(e)}")
        session.rollback()
        raise

def setup_timestamp_trigger(session):
    """Set up automatic timestamp updates."""
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
        
        logger.info("Timestamp triggers created successfully")
    except Exception as e:
        logger.error(f"Failed to create timestamp triggers: {str(e)}")
        session.rollback()
        raise

def setup_text_search(session):
    """Set up text search configuration."""
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
        logger.info("Text search configuration created successfully")
    except Exception as e:
        logger.error(f"Failed to create text search configuration: {str(e)}")
        session.rollback()
        raise

def setup_indexes(session):
    """Set up database indexes."""
    try:
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_words_search ON words USING gin(search_text);
            CREATE INDEX IF NOT EXISTS idx_words_normalized ON words(normalized_lemma);
            CREATE INDEX IF NOT EXISTS idx_words_baybayin ON words(baybayin_form) WHERE has_baybayin = TRUE;
            CREATE INDEX IF NOT EXISTS idx_words_language ON words(language_code);
            CREATE INDEX IF NOT EXISTS idx_words_root ON words(root_word_id);
            
            CREATE INDEX IF NOT EXISTS idx_definitions_word ON definitions(word_id);
            CREATE INDEX IF NOT EXISTS idx_definitions_pos ON definitions(standardized_pos_id);
            CREATE INDEX IF NOT EXISTS idx_definitions_text ON definitions USING gin(to_tsvector('english', definition_text));
            
            CREATE INDEX IF NOT EXISTS idx_etymologies_word ON etymologies(word_id);
            CREATE INDEX IF NOT EXISTS idx_etymologies_langs ON etymologies USING gin(to_tsvector('simple', language_codes));
            
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_word_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_word_id);
            CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_relations_metadata ON relations USING GIN(relation_metadata);
            
            CREATE INDEX IF NOT EXISTS idx_pronunciations_word ON pronunciations(word_id);
            CREATE INDEX IF NOT EXISTS idx_pronunciations_type ON pronunciations(type);
            CREATE INDEX IF NOT EXISTS idx_pronunciations_value ON pronunciations(value);
            
            CREATE INDEX IF NOT EXISTS idx_credits_word ON credits(word_id);
        """))
        logger.info("Database indexes created successfully")
    except Exception as e:
        logger.error(f"Failed to create database indexes: {str(e)}")
        session.rollback()
        raise

def setup_search_trigger(session):
    """Set up search vector update trigger."""
    try:
        session.execute(text("""
            DROP TRIGGER IF EXISTS tsvectorupdate ON words;
            CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
            ON words FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger(search_text, 'pg_catalog.simple', lemma, normalized_lemma, baybayin_form, romanized_form);
        """))
        logger.info("Search vector trigger created successfully")
    except Exception as e:
        logger.error(f"Failed to create search vector trigger: {str(e)}")
        session.rollback()
        raise

def setup_parts_of_speech(session):
    """Set up standard parts of speech."""
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
            # Create all tables
            db.create_all()
            
            # Set up database features
            setup_extensions(db.session)
            setup_timestamp_trigger(db.session)
            setup_text_search(db.session)
            setup_indexes(db.session)
            setup_search_trigger(db.session)
            setup_parts_of_speech(db.session)
            
            # Commit all changes
            db.session.commit()
            logger.info("Database models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database models: {str(e)}")
        raise 