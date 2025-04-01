"""
Filipino Dictionary Database Models
"""

from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, Boolean, DateTime, 
    func, Index, UniqueConstraint, DDL, event, text, Float, JSON
)
from sqlalchemy.orm import relationship, validates, backref
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
from unidecode import unidecode
from flask import current_app

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy with no app context yet
db = SQLAlchemy()

# Helper function to determine if we're in testing mode
def is_testing_db(connection):
    """Check if we're using a testing database (SQLite)."""
    return connection.engine.url.drivername == 'sqlite'

# Helper function to normalize text
def normalize_lemma(text: str) -> str:
    """Normalize lemma for consistent comparison."""
    if not text:
        logger.warning("normalize_lemma received empty or None text")
        return ""
    return unidecode(text).lower()

def initialize_parts_of_speech(session):
    """Initialize standard parts of speech."""
    from models.parts_of_speech import PartOfSpeech  # Local import to avoid circular dependency
    
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
    
    existing_codes = {pos.code for pos in session.query(PartOfSpeech).all()}
    
    for pos_data in standard_pos:
        if pos_data['code'] not in existing_codes:
            pos = PartOfSpeech(**pos_data)
            session.add(pos)
    
    try:
        session.commit()
    except Exception as e:
        logger.error(f"Failed to initialize parts of speech: {str(e)}")
        session.rollback()
        raise

def setup_postgresql_features(app):
    """Set up PostgreSQL-specific features."""
    try:
        with app.app_context():
            # Create extensions
            db.session.execute(text("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
            """))
            
            # Create update timestamp function
            db.session.execute(text("""
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
                db.session.execute(text(f"""
                    DROP TRIGGER IF EXISTS update_{table}_timestamp ON {table};
                    CREATE TRIGGER update_{table}_timestamp
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_timestamp();
                """))
            
            # Create text search configuration
            db.session.execute(text("""
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
            
            # Create indexes
            db.session.execute(text("""
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
            
            # Add search vector update trigger
            db.session.execute(text("""
                DROP TRIGGER IF EXISTS tsvectorupdate ON words;
                CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
                ON words FOR EACH ROW EXECUTE FUNCTION
                tsvector_update_trigger(search_text, 'pg_catalog.simple', lemma, normalized_lemma, baybayin_form, romanized_form);
            """))
            
            db.session.commit()
            logger.info("PostgreSQL features set up successfully")
    except Exception as e:
        logger.error(f"Failed to set up PostgreSQL features: {str(e)}")
        db.session.rollback()
        raise

def init_app(app):
    """Initialize the models with the Flask app."""
    try:
        # Initialize Flask-SQLAlchemy
        db.init_app(app)
        
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Initialize standard data
            initialize_parts_of_speech(db.session)
            
            # Set up PostgreSQL features if not using SQLite
            if not is_testing_db(db.engine):
                setup_postgresql_features(app)
                
        logger.info("Database models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database models: {str(e)}")
        raise

# Import models after db initialization to avoid circular imports
from .models.word import Word
from .models.definition import Definition
from .models.etymology import Etymology
from .models.relation import Relation
from .models.affixation import Affixation
from .models.parts_of_speech import PartOfSpeech
from .models.pronunciation import Pronunciation
from .models.credit import Credit

__all__ = [
    'db',
    'Word',
    'Definition',
    'Etymology',
    'Relation',
    'Affixation',
    'PartOfSpeech',
    'Pronunciation',
    'Credit',
    'init_app',
]