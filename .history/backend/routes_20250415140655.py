"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, request, jsonify, send_file, abort, current_app, g, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast, not_, case, exists, extract
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload, raiseload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
import logging
from sqlalchemy.exc import SQLAlchemyError
import time
import random
import json
import csv
import os
import io
import re
import zipfile
import math
import pickle
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Histogram, REGISTRY
# Comment out problematic import
# from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict

from backend.models import (
    Word, Definition, Etymology, Pronunciation, Relation, DefinitionCategory,
    DefinitionLink, DefinitionRelation, Affixation, WordForm, WordTemplate,
    PartOfSpeech, Credit
)
from backend.database import db, cached_query, get_cache_client
from backend.dictionary_manager import (
    normalize_lemma, extract_etymology_components, extract_language_codes,
    RelationshipType, RelationshipCategory, BaybayinRomanizer
)
from backend.utils.word_processing import normalize_word
from backend.utils.rate_limiting import limiter
from backend.utils.ip import get_remote_address
from backend.schemas import (
    WordSchema as SchemasWordSchema, 
    DefinitionSchema as SchemasDefinitionSchema,
    EtymologySchema as SchemasEtymologySchema,
    PronunciationSchema as SchemasPronunciationSchema,
    RelationSchema as SchemasRelationSchema,
    AffixationSchema as SchemasAffixationSchema,
    DefinitionCategorySchema as SchemasDefinitionCategorySchema,
    DefinitionLinkSchema as SchemasDefinitionLinkSchema,
    PartOfSpeechSchema as SchemasPartOfSpeechSchema,
    DefinitionRelationSchema as SchemasDefinitionRelationSchema,
    CreditSchema as SchemasCreditSchema,
    WordFormSchema as SchemasWordFormSchema,
    WordTemplateSchema as SchemasWordTemplateSchema,
    BaseSchema as SchemasBaseSchema
)

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__, url_prefix='/api/v2')

# Get a cache client - initialize it once
try:
    cache_client = get_cache_client()
except Exception as e:
    logger.error(f"Failed to initialize cache client: {e}")
    cache_client = None

# Define request latency histogram
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Flask Request Latency',
    ['endpoint']
)

# Define request counter
REQUEST_COUNT = Counter(
    'request_count', 
    'Flask Request Count',
    ['method', 'endpoint', 'status']
)

# Unregister existing metrics to avoid duplication
for collector in list(REGISTRY._collector_to_names.keys()):
    # Check if the collector is a metric by checking for _type attribute
    if hasattr(collector, '_type'):
        try:
            REGISTRY.unregister(collector)
        except Exception as e:
            logger.error(f"Error unregistering metric: {e}")
            pass

# Test endpoint - quick connection test without hitting the database
@bp.route('/test', methods=['GET'])
def test_api():
    """Simple test endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'API is running',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

# Health check endpoint
# @bp.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint."""
#     try:
#         # Check database connection
#         db.session.execute(text('SELECT 1'))
#         return jsonify({
#             'status': 'healthy',
#             'database': 'connected',
#             'timestamp': datetime.now(timezone.utc).isoformat()
#         })
#     except Exception as e:
#         logger.error('Health check failed', error=str(e))
#         return jsonify({
#             'status': 'unhealthy',
#             'error': str(e),
#             'timestamp': datetime.now(timezone.utc).isoformat()
#         }), 500

def is_testing_db(engine):
    """Check if we're using a test database."""
    return engine.url.database.endswith('_test')

# Metrics
# First, try to unregister existing metrics if they exist
for collector in list(REGISTRY._collector_to_names.keys()):
    if hasattr(collector, '_type'):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

# Then register new metrics
API_REQUESTS = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
API_ERRORS = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])
REQUEST_COUNT = Counter('request_count', 'Total request count')

# Schema definitions
class BaseSchema(Schema):
    """Base schema with common metadata fields."""
    id = fields.Int(dump_only=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    sources = fields.String()

class PronunciationType(Schema):
    """Schema for pronunciation data."""
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic', 'x-sampa', 'pinyin', 'jyutping', 'romaji']))
    value = fields.Str(required=True)
    tags = fields.Dict()
    pronunciation_metadata = fields.Dict()  # Renamed from metadata to match DB/model
    sources = fields.String()
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class EtymologySchema(BaseSchema):
    """Schema for etymology data."""
    etymology_text = fields.Str(required=True)
    normalized_components = fields.String()
    etymology_structure = fields.String()
    language_codes = fields.String()
    sources = fields.String()  # Add sources field
    etymology_data = fields.Dict()  # Use etymology_data property instead of etymology_metadata
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True)
    relation_data = fields.Dict()
    source_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    target_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class AffixationSchema(BaseSchema):
    """Schema for word affixation data."""
    affix_type = fields.Str(required=True)
    sources = fields.String() # Added sources
    # Adjusted nesting to match common relationship patterns
    root_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))
    affixed_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class DefinitionSchema(BaseSchema):
    """Schema for word definitions."""
    id = fields.Int(dump_only=True)
    definition_text = fields.Str(required=True)
    word_id = fields.Int()
    standardized_pos_id = fields.Int()
    usage_notes = fields.String()
    examples = fields.List(fields.String(), dump_default=[])
    sources = fields.String()
    definition_metadata = fields.Dict()  # Add this field to match the property in the model
    
    # Relationships
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    standardized_pos = fields.Nested('PartOfSpeechSchema', dump_default=None)
    categories = fields.List(fields.Nested('DefinitionCategorySchema', exclude=('definition',)))
    # Skip links due to column mismatch
    # links = fields.List(fields.Nested('DefinitionLinkSchema', exclude=('definition',)))
    
    # Related definitions/words relationships
    definition_relations = fields.List(fields.Nested('DefinitionRelationSchema', exclude=('definition',)))
    related_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))

class DefinitionCategorySchema(BaseSchema):
    """Schema for definition categories."""
    id = fields.Int(dump_only=True)
    definition_id = fields.Int(required=True)
    category_name = fields.Str(required=True)
    description = fields.Str(allow_none=True)  # Make it optional to match database
    category_kind = fields.Str(allow_none=True)  # Add this missing field
    tags = fields.Dict()
    category_metadata = fields.Dict()
    parents = fields.List(fields.Str(), dump_default=[])  # Add this field to match model
    definition = fields.Nested('DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

class DefinitionLinkSchema(BaseSchema):
    """Schema for definition links."""
    id = fields.Int(dump_only=True)
    definition_id = fields.Int(required=True)
    link_text = fields.Str(required=True)  # Changed from link_type to link_text
    target_url = fields.Str(required=True)
    display_text = fields.Str()
    is_external = fields.Bool(dump_default=False)
    tags = fields.Dict()
    link_metadata = fields.Dict()
    definition = fields.Nested('DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Int(dump_only=True)
    code = fields.Str(required=True)
    name_en = fields.Str(required=True)
    name_tl = fields.Str(required=True)
    description = fields.String()
    # Represent derived_words relationship correctly
    derived_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))

    # --- Add missing relationships and columns ---
    forms = fields.List(fields.Nested('WordFormSchema', exclude=("word",)))
    templates = fields.List(fields.Nested('WordTemplateSchema', exclude=("word",)))
    # language = fields.Nested('LanguageSchema') # Add nested language info
    # Add computed/hybrid properties
    completeness_score = fields.Float(dump_only=True) # Use the hybrid property
    # ------------------------------------------

class CreditSchema(BaseSchema):
    """Schema for word credits."""
    credit = fields.Str(required=True)
    # Removed word nesting
    # word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

# --- Add Schemas for missing models ---
class WordFormSchema(BaseSchema):
    """Schema for word forms (inflections, conjugations)."""
    form = fields.Str(required=True)
    tags = fields.Dict() # Assuming JSONB tags map to a dict
    is_canonical = fields.Bool(dump_default=False)
    is_primary = fields.Bool(dump_default=False)
    word = fields.Nested('WordSchema', only=('id', 'lemma')) # Nested word info

class WordTemplateSchema(BaseSchema):
    """Schema for word templates."""
    template_name = fields.Str(required=True)
    args = fields.Dict() # Assuming JSONB args map to a dict
    expansion = fields.Str()
    word = fields.Nested('WordSchema', only=('id', 'lemma')) # Nested word info

class WordSchema(BaseSchema):
    """Schema for word entries."""
    id = fields.Int(dump_only=True)
    lemma = fields.Str(required=True)
    normalized_lemma = fields.Str()
    language_code = fields.Str(required=True)
    has_baybayin = fields.Bool()
    baybayin_form = fields.Str()
    romanized_form = fields.Str()
    root_word_id = fields.Int()
    preferred_spelling = fields.Str()
    tags = fields.String()
    idioms = fields.Dict()
    source_info = fields.Dict()
    word_metadata = fields.Dict()  # Added to match model
    data_hash = fields.String()
    search_text = fields.String()
    badlit_form = fields.String()
    hyphenation = fields.Dict()
    is_proper_noun = fields.Bool()
    is_abbreviation = fields.Bool()
    is_initialism = fields.Bool()
    is_root = fields.Bool()
    completeness_score = fields.Float(dump_only=True)
    
    # Relationships - ensure names match model relationship names
    # Use selectinload in helper, so schema just defines nesting
    definitions = fields.List(fields.Nested(DefinitionSchema, exclude=("word",)))
    etymologies = fields.List(fields.Nested(EtymologySchema, exclude=("word",)))
    pronunciations = fields.List(fields.Nested(PronunciationType, exclude=("word",)))
    credits = fields.List(fields.Nested(CreditSchema, exclude=("word",)))
    # Adjust nesting based on expected structure from helper function
    outgoing_relations = fields.List(fields.Nested(RelationSchema, exclude=("source_word",)))
    incoming_relations = fields.List(fields.Nested(RelationSchema, exclude=("target_word",)))
    root_affixations = fields.List(fields.Nested(AffixationSchema, exclude=("root_word",))) # Affixes where this word is the root
    affixed_affixations = fields.List(fields.Nested(AffixationSchema, exclude=("affixed_word",))) # Affixes where this word is the result

    # Represent root_word relationship correctly
    root_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    # Represent derived_words relationship correctly
    derived_words = fields.List(fields.Nested('WordSchema', only=('id', 'lemma', 'language_code')))

    # Add forms and templates relationships
    forms = fields.List(fields.Nested(WordFormSchema, exclude=("word",)))
    templates = fields.List(fields.Nested(WordTemplateSchema, exclude=("word",)))
    
    # Add definition relation relationships
    definition_relations = fields.List(fields.Nested('DefinitionRelationSchema', exclude=("related_word",)))
    related_definitions = fields.List(fields.Nested(DefinitionSchema, exclude=("word", "related_words")))

# ------------------------------------

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'prefix', 'suffix']),
                     dump_default='all', load_default='all')
    limit = fields.Int(dump_default=50, load_default=50)
    offset = fields.Int(dump_default=0, load_default=0)
    
    # Sorting options
    sort = fields.Str(dump_default='relevance', load_default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), 
                      dump_default='desc', load_default='desc')
    
    include_full = fields.Bool(dump_default=False, load_default=False)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_pronunciations = fields.Bool(dump_default=True, load_default=True)
    include_etymologies = fields.Bool(dump_default=True, load_default=True)
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_forms = fields.Bool(dump_default=True, load_default=True)
    include_templates = fields.Bool(dump_default=True, load_default=True)
    include_metadata = fields.Bool(dump_default=True, load_default=True)
    
    # Relation expansion options
    include_related_words = fields.Bool(dump_default=False, load_default=False)
    include_definition_relations = fields.Bool(dump_default=False, load_default=False)
    
    # Filter parameters
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    exclude_baybayin = fields.Bool(dump_default=False, load_default=False)
    has_forms = fields.Bool(dump_default=None, load_default=None)
    has_templates = fields.Bool(dump_default=None, load_default=None)
    
    # Advanced filtering
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    tag = fields.Str(dump_default=None, load_default=None)
    min_completeness = fields.Float(dump_default=None, load_default=None)

class StatisticsSchema(Schema):
    """Schema for dictionary statistics."""
    total_words = fields.Int()
    total_definitions = fields.Int()
    total_etymologies = fields.Int()
    total_relations = fields.Int()
    total_affixations = fields.Int()
    words_with_examples = fields.Int()
    words_with_etymology = fields.Int()
    words_with_relations = fields.Int()
    words_with_baybayin = fields.Int()
    words_by_language = fields.Dict(keys=fields.Str(), values=fields.Int())
    words_by_pos = fields.Dict(keys=fields.Str(), values=fields.Int())
    verification_stats = fields.Dict(keys=fields.Str(), values=fields.Int())
    quality_distribution = fields.Dict(keys=fields.Str(), values=fields.Int())
    update_frequency = fields.Dict(keys=fields.Str(), values=fields.Int())

# Error handlers
@bp.errorhandler(404)
def not_found_error(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='not_found').inc()
    return jsonify({"error": "Resource not found"}), 404

@bp.errorhandler(500)
def internal_error(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='server_error').inc()
    logger.error(f"Internal error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

@bp.errorhandler(SQLAlchemyError)
def database_error(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='database_error').inc()
    logger.error(f"Database error: {str(error)}", exc_info=True)
    return jsonify({"error": "Database error", "details": str(error)}), 500

@bp.errorhandler(Exception)
def handle_exception(error):
    API_ERRORS.labels(endpoint=request.endpoint or '', error_type='unhandled_exception').inc()
    logger.error(f"Unhandled exception: {str(error)}", exc_info=True)
    return jsonify({
        "error": "Unexpected error",
        "message": str(error),
        "type": error.__class__.__name__
    }), 500

# --- Reorganized route order: ID route first, then path route ---
@bp.route("/words/id/<int:word_id>", methods=["GET"])
def get_word_by_id(word_id: int):
    """
    Get a word by its ID.
    ---
    parameters:
      - name: word_id
        in: path
        type: integer
        required: true
        description: ID of the word to retrieve
    responses:
      200:
        description: Word found
      404:
        description: Word not found
    """
    try:
        API_REQUESTS.labels(endpoint="get_word_by_id", method="GET").inc()
        
        # First, check if the word exists
        from sqlalchemy.orm import Session
        
        with Session(db.engine) as session:
            # Use a direct SQL query for better performance
            check_sql = "SELECT 1 FROM words WHERE id = :id"
            exists = session.execute(text(check_sql), {"id": word_id}).fetchone()
            
            if not exists:
                return jsonify({
                    "error": f"Word with ID {word_id} not found",
                    "details": "The requested word ID does not exist in the database."
                }), 404
        
        # Fetch word details with full relationships using a safe method
        # instead of the standard _fetch_word_details to ensure proper transaction isolation
        from backend._fetch_word_details_helper import fetch_word_details_safely
        word_details = fetch_word_details_safely(
            db.engine,
            word_id,
            include_definitions=True,
            include_categories=False  # Skip loading categories directly due to schema issues
        )
        
        if not word_details:
            return jsonify({
                "error": f"Failed to retrieve details for word ID {word_id}",
                "details": "The word exists but details could not be retrieved."
            }), 500
        
        # Serialize the result
        schema = SchemasWordSchema()
        result = schema.dump(word_details)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_word_by_id for ID {word_id}: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_word_by_id", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {type(e).__name__}",
            "message": str(e)
        }), 500

@bp.route("/words/<path:word>", methods=["GET"])
def get_word(word: str):
    """
    Get a word by its lemma.
    ---
    parameters:
      - name: word
        in: path
        type: string
        required: true
        description: The word to retrieve
      - name: language
        in: query
        type: string
        description: Optional language code to filter by
    responses:
      200:
        description: Word found
      404:
        description: Word not found
    """
    try:
        API_REQUESTS.labels(endpoint="get_word", method="GET").inc()
        
        # Sanitize input
        if not word:
            return jsonify({"error": "Word parameter is required"}), 400
            
        # Remove starting/trailing spaces and normalize
        word = word.strip()
        
        # Get optional language parameter
        language = request.args.get('language')
        
        # First check if the word exists
        from sqlalchemy.orm import Session
        
        with Session(db.engine) as session:
            # Try to find the word ID by lemma (and language if provided)
            sql_check = "SELECT id FROM words WHERE LOWER(lemma) = LOWER(:word)"
            params = {"word": word}
            
            if language:
                sql_check += " AND language_code = :language"
                params["language"] = language
                
            word_id = session.execute(text(sql_check), params).fetchone()
            
            if not word_id:
                # Try normalized_lemma instead
                sql_find = "SELECT id FROM words WHERE LOWER(normalized_lemma) = LOWER(:word)"
                params_find = {"word": word}
                
                if language:
                    sql_find += " AND language_code = :language"
                    params_find["language"] = language
                    
                word_result_find = session.execute(text(sql_find), params_find).fetchone()
                
                if not word_result_find:
                    return jsonify({
                        "error": f"Word '{word}' not found",
                        "details": f"No word matching '{word}' exists in the dictionary."
                    }), 404
                    
                word_id = word_result_find[0]
            else:
                word_id = word_id[0]
        
        # Fetch word details with safer transaction handling
        from backend._fetch_word_details_helper import fetch_word_details_safely
        word_details = fetch_word_details_safely(
            db.engine,
            word_id,
            include_definitions=True,
            include_categories=False  # Skip loading categories directly due to schema issues
        )
        
        if not word_details:
            return jsonify({
                "error": f"Failed to retrieve details for word '{word}'",
                "details": "The word exists but details could not be retrieved."
            }), 500
        
        # Serialize the result
        schema = SchemasWordSchema()
        result = schema.dump(word_details)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_word for word '{word}': {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_word", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {type(e).__name__}",
            "message": str(e)
        }), 500

def _fetch_word_details(word_id, include_definitions, include_etymologies, include_pronunciations, include_credits, include_relations, include_affixations, include_root, include_derived, include_forms, include_templates, include_definition_relations):
    """Helper function to fetch detailed word information with relationships."""
    start_time = time.time()
    try:
        # Create a brand new session for this operation
        from sqlalchemy.orm import Session
        
        with Session(db.engine) as session:
            # Start with a basic word query
            word_query = session.query(Word).filter(Word.id == word_id)
            
            # Apply eager loading based on the include parameters
            if include_definitions:
                word_query = word_query.options(selectinload(Word.definitions))
                
                # Skip loading categories to avoid schema errors
                # if include_definition_relations:
                #     word_query = word_query.options(
                #         selectinload(Word.definitions).selectinload(Definition.definition_relations)
                #     )
            
            if include_etymologies:
                word_query = word_query.options(selectinload(Word.etymologies))
                
            if include_pronunciations:
                word_query = word_query.options(selectinload(Word.pronunciations))
                
            if include_credits:
                word_query = word_query.options(selectinload(Word.credits))
            
            if include_relations:
                word_query = word_query.options(
                    selectinload(Word.outgoing_relations),
                    selectinload(Word.incoming_relations)
                )
                
            if include_affixations:
                word_query = word_query.options(
                    selectinload(Word.root_affixations),
                    selectinload(Word.affixed_affixations)
                )
                
            if include_root and include_relations:
                word_query = word_query.options(selectinload(Word.root_word))
                
            if include_derived and include_relations:
                word_query = word_query.options(selectinload(Word.derived_words))
                
            if include_forms:
                word_query = word_query.options(selectinload(Word.forms))
                
            if include_templates:
                word_query = word_query.options(selectinload(Word.templates))
                
            # Execute the query and get the result
            word = word_query.first()
            
            if not word:
                logger.warning(f"Word with ID {word_id} not found")
                return None
                
            # Initialize empty collections to avoid None values
            if not hasattr(word, 'outgoing_relations') or word.outgoing_relations is None:
                word.outgoing_relations = []
            if not hasattr(word, 'incoming_relations') or word.incoming_relations is None:
                word.incoming_relations = []
            if not hasattr(word, 'root_affixations') or word.root_affixations is None:
                word.root_affixations = []
            if not hasattr(word, 'affixed_affixations') or word.affixed_affixations is None:
                word.affixed_affixations = []
            if not hasattr(word, 'forms') or word.forms is None:
                word.forms = []
            if not hasattr(word, 'templates') or word.templates is None:
                word.templates = []
            if not hasattr(word, 'derived_words') or word.derived_words is None:
                word.derived_words = []
            
            # Set empty lists for categories on each definition to avoid schema errors
            if include_definitions and word.definitions:
                for definition in word.definitions:
                    definition.categories = []
                    definition.links = []
                    if hasattr(definition, 'definition_relations'):
                        definition.definition_relations = []
            
            # Detach the word from the session to avoid issues when returning
            session.expunge(word)
            
            # Log completion
            execution_time = time.time() - start_time
            logger.debug(f"Fetched word details for ID {word_id} in {execution_time:.4f}s")
            
            return word
            
    except SQLAlchemyError as e:
        logger.error(f"Database error in _fetch_word_details for word ID {word_id}: {str(e)}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error in _fetch_word_details for word ID {word_id}: {str(e)}", exc_info=True)
        return None
