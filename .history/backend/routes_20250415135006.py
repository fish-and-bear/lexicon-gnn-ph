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
from backend.schemas import WordSchema # Check if this covers needed schemas

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
    """Get detailed information about a specific word by ID, including all relations."""
    logger.info(f"Fetching word details by ID: {word_id}")
    
    # Validate that this is actually a numeric ID
    if not isinstance(word_id, int):
        logger.warning(f"Invalid word ID format: {word_id}")
        return jsonify({"error": f"Invalid word ID format. ID must be numeric."}), 400
        
    start_time = time.time()
    
    try:
        # First check if the word ID exists
        check_sql = "SELECT id FROM words WHERE id = :id"
        exists = db.session.execute(text(check_sql), {"id": word_id}).fetchone()
        
        if not exists:
            logger.warning(f"Word with ID {word_id} not found in database")
            return jsonify({"error": f"Word with ID {word_id} not found"}), 404
            
        # Directly use the word_id to fetch details
        word_details = _fetch_word_details(
            word_id,
            include_definitions=True,
            include_etymologies=True,
            include_pronunciations=True,
            include_credits=True,
            include_relations=True,
            include_affixations=True,
            include_root=True,
            include_derived=True,
            include_forms=True,
            include_templates=True,
            include_definition_relations=True
        )

        if not word_details:
            logger.error(f"Failed to fetch details for word ID {word_id}")
            return jsonify({"error": f"Failed to retrieve details for word ID {word_id}"}), 404

        # Serialize the result
        from backend.schemas import WordSchema as SchemasWordSchema
        schema = SchemasWordSchema()
        result = schema.dump(word_details)
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully fetched details for word ID {word_id} in {execution_time:.4f}s")
        REQUEST_LATENCY.labels(endpoint="get_word_by_id").observe(execution_time)

        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving word with ID {word_id}", exc_info=True)
        API_ERRORS.labels(endpoint="get_word_by_id", error_type="database_error").inc()
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"Error retrieving word with ID {word_id}", exc_info=True)
        API_ERRORS.labels(endpoint="get_word_by_id", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>", methods=["GET"])
def get_word(word: str):
    """Get FULL detailed information about a specific word, including all relations."""
    try:
        API_REQUESTS.labels(endpoint="get_word", method="GET").inc()
        start_time = time.time()

        # Check if someone is trying to access id/something incorrectly
        if word.startswith('id/'):
            logger.warning(f"Invalid access to '/words/id/' path: {word}")
            return jsonify({"error": "Invalid URL format. For ID lookup, use /words/id/{numeric_id} (must be a number)"}), 400

        word_id = None
        normalized = normalize_word(word)
        
        # --- 1. Find Word ID (handle integer ID or lemma) ---
        try:
            word_id = int(word)
            # Verify ID exists
            sql_check = "SELECT id FROM words WHERE id = :id"
            if not db.session.execute(text(sql_check), {"id": word_id}).fetchone():
                logger.warning(f"Word ID {word_id} provided but not found.")
                return jsonify({"error": f"Word with ID {word_id} not found"}), 404
        except ValueError:
            # Lookup by lemma/normalized_lemma
            sql_find = """
            SELECT id FROM words
            WHERE LOWER(lemma) = LOWER(:word) OR LOWER(normalized_lemma) = LOWER(:normalized)
            ORDER BY (CASE WHEN LOWER(lemma) = LOWER(:word) THEN 0 ELSE 1 END) -- Prioritize exact lemma match
            LIMIT 1
            """
            params_find = {"word": word, "normalized": normalized}
            word_result_find = db.session.execute(text(sql_find), params_find).fetchone()
            if not word_result_find:
                logger.warning(f"Word '{word}' (normalized: '{normalized}') not found.")
                return jsonify({"error": f"Word '{word}' not found"}), 404
            word_id = word_result_find.id

        if not word_id:
             logger.error(f"Failed to resolve word ID for input: '{word}'")
             return jsonify({"error": "Could not resolve word ID"}), 500
             
        logger.debug(f"Fetching full details for word ID: {word_id} ('{word}')")

        # --- 2. Fetch Full Details using Helper Function ---
        word_details = _fetch_word_details(
                word_id,
            include_definitions=True,
            include_etymologies=True,
            include_pronunciations=True,
            include_credits=True,
            include_relations=True, # Ensure relations are included
            include_affixations=True,
            include_root=True,
            include_derived=True,
            include_forms=True,
            include_templates=True,
            include_definition_relations=True
        )

        if not word_details:
            # This case should be rare if ID was validated, but handle defensively
            logger.error(f"Found word ID {word_id} but failed to fetch full details via helper.")
            return jsonify({"error": f"Failed to retrieve details for word ID {word_id}"}), 500

        # --- 3. Serialize the Result ---
        from backend.schemas import WordSchema as SchemasWordSchema
        schema = SchemasWordSchema()
        result = schema.dump(word_details)
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully fetched full details for '{word}' (ID: {word_id}) in {execution_time:.4f}s")
        REQUEST_LATENCY.labels(endpoint="get_word").observe(execution_time)

        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving word '{word}'", exc_info=True)
        API_ERRORS.labels(endpoint="get_word", error_type="database_error").inc()
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"Error retrieving word '{word}'", exc_info=True)
        API_ERRORS.labels(endpoint="get_word", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/search", methods=["GET"])
@cached_query(timeout=300)  # Increase cache timeout to 5 minutes for search results
@limiter.limit("20 per minute", key_func=get_remote_address) # Apply rate limit
def search():
    """
    Search the dictionary.
    ---
    parameters:
      - name: q
        in: query
        type: string
        required: true
        description: Query string to search for
      - name: mode
        in: query
        type: string
        default: "all"
        enum: ["all", "exact", "prefix", "suffix"]
        description: Search mode
      - name: limit
        in: query
        type: integer
        default: 50
        description: Maximum number of results to return
      - name: offset
        in: query
        type: integer
        default: 0
        description: Number of results to skip
      - name: include_full
        in: query
        type: boolean
        default: false
        description: Include full word data in the results
    responses:
      200:
        description: Search results
    """
    try:
        # Get and validate parameters
        schema = SearchQuerySchema()
        params = schema.load(request.args)
        
        query = params.get('q')
        mode = params.get('mode', 'all')
        limit = int(params.get('limit', 50))
        offset = int(params.get('offset', 0))
        include_full = params.get('include_full', False)
        
        # Sanitize inputs
        if isinstance(query, str):
            query = query.strip()
        
        if not query:
            return jsonify({
                "error": "Query parameter 'q' is required",
                "details": "Please provide a search query"
            }), 400
            
        # Create a dedicated session for search operations
        from sqlalchemy.orm import Session
        with Session(db.engine) as session:
            # Normalize query for more accurate matching
            normalized_query = normalize_word(query)
            
            # Build search conditions based on mode
            conditions = []
            search_params = {}
            
            # Build mode-specific search conditions
            if mode == 'exact':
                conditions.append("(LOWER(lemma) = LOWER(:query) OR LOWER(normalized_lemma) = LOWER(:normalized_query))")
                search_params['query'] = query
                search_params['normalized_query'] = normalized_query
            elif mode == 'prefix':
                conditions.append("(LOWER(lemma) LIKE LOWER(:prefix) OR LOWER(normalized_lemma) LIKE LOWER(:normalized_prefix))")
                search_params['prefix'] = f"{query}%"
                search_params['normalized_prefix'] = f"{normalized_query}%"
            elif mode == 'suffix':
                conditions.append("(LOWER(lemma) LIKE LOWER(:suffix) OR LOWER(normalized_lemma) LIKE LOWER(:normalized_suffix))")
                search_params['suffix'] = f"%{query}"
                search_params['normalized_suffix'] = f"%{normalized_query}"
            else:  # Default to 'all' mode (approximate match)
                conditions.append("""(
                    LOWER(lemma) LIKE LOWER(:wildcard) 
                    OR LOWER(normalized_lemma) LIKE LOWER(:normalized_wildcard)
                    OR search_text IS NOT NULL AND search_text @@ plainto_tsquery('simple', :tsquery)
                )""")
                search_params['wildcard'] = f"%{query}%"
                search_params['normalized_wildcard'] = f"%{normalized_query}%"
                search_params['tsquery'] = query
            
            # Add language filter if specified
            language = params.get('language')
            if language:
                conditions.append("language_code = :language")
                search_params['language'] = language
            
            # Add part of speech filter if specified
            pos = params.get('pos')
            if pos:
                conditions.append("""id IN (
                    SELECT word_id FROM definitions 
                    WHERE standardized_pos_id IN (
                        SELECT id FROM parts_of_speech WHERE code = :pos
                    )
                )""")
                search_params['pos'] = pos
            
            # Add baybayin filter if specified
            has_baybayin = params.get('has_baybayin')
            if has_baybayin is not None:
                conditions.append("has_baybayin = :has_baybayin")
                search_params['has_baybayin'] = has_baybayin
                
            exclude_baybayin = params.get('exclude_baybayin', False)
            if exclude_baybayin:
                conditions.append("has_baybayin = FALSE")
                
            # Add tag filter if specified
            tag = params.get('tag')
            if tag:
                conditions.append("tags LIKE :tag")
                search_params['tag'] = f"%{tag}%"
            
            # Add etymology filter if specified
            has_etymology = params.get('has_etymology')
            if has_etymology is not None:
                if has_etymology:
                    conditions.append("id IN (SELECT word_id FROM etymologies)")
                else:
                    conditions.append("id NOT IN (SELECT word_id FROM etymologies)")
            
            # Add pronunciation filter if specified
            has_pronunciation = params.get('has_pronunciation')
            if has_pronunciation is not None:
                if has_pronunciation:
                    conditions.append("id IN (SELECT word_id FROM pronunciations)")
                else:
                    conditions.append("id NOT IN (SELECT word_id FROM pronunciations)")
            
            # Add forms filter if specified
            has_forms = params.get('has_forms')
            if has_forms is not None:
                if has_forms:
                    conditions.append("id IN (SELECT word_id FROM word_forms)")
                else:
                    conditions.append("id NOT IN (SELECT word_id FROM word_forms)")
            
            # Add templates filter if specified
            has_templates = params.get('has_templates')
            if has_templates is not None:
                if has_templates:
                    conditions.append("id IN (SELECT word_id FROM word_templates)")
                else:
                    conditions.append("id NOT IN (SELECT word_id FROM word_templates)")
            
            # Add minimum completeness score filter if specified
            min_completeness = params.get('min_completeness')
            if min_completeness is not None:
                conditions.append("(SELECT COUNT(*) FROM definitions WHERE word_id = words.id) > 0")
                
            # Finalize WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Determine sort method
            sort_method = params.get('sort', 'relevance')
            sort_dir = params.get('order', 'desc').upper()
            
            if sort_method == 'alphabetical':
                order_clause = f"lemma {sort_dir}"
            elif sort_method == 'created':
                order_clause = f"created_at {sort_dir}"
            elif sort_method == 'updated':
                order_clause = f"updated_at {sort_dir}"
            elif sort_method == 'quality':
                order_clause = f"(SELECT COUNT(*) FROM definitions WHERE word_id = words.id) {sort_dir}"
            else:  # Default to relevance-based sorting
                # For relevance, use a scoring system that prioritizes exact matches
                if mode == 'exact':
                    order_clause = f"(CASE WHEN LOWER(lemma) = LOWER(:query) THEN 10 WHEN LOWER(normalized_lemma) = LOWER(:normalized_query) THEN 5 ELSE 0 END) {sort_dir}"
                elif mode in ['prefix', 'suffix']:
                    order_clause = f"LENGTH(lemma) {('ASC' if sort_dir == 'DESC' else 'DESC')}"  # Shorter words first for prefix/suffix search
                else:  # 'all' mode
                    order_clause = f"""(
                        CASE 
                            WHEN LOWER(lemma) = LOWER(:query) THEN 10
                            WHEN LOWER(normalized_lemma) = LOWER(:normalized_query) THEN 8
                            WHEN LOWER(lemma) LIKE LOWER(:query || '%') THEN 6
                            WHEN LOWER(normalized_lemma) LIKE LOWER(:normalized_query || '%') THEN 5
                            WHEN LOWER(lemma) LIKE LOWER('%' || :query) THEN 4
                            WHEN LOWER(normalized_lemma) LIKE LOWER('%' || :normalized_query) THEN 3
                            WHEN LOWER(lemma) LIKE LOWER('%' || :query || '%') THEN 2
                            WHEN LOWER(normalized_lemma) LIKE LOWER('%' || :normalized_query || '%') THEN 1
                            ELSE 0
                        END
                    ) {sort_dir}"""
                    search_params['query'] = query
                    search_params['normalized_query'] = normalized_query
            
            # Build the complete search query
            count_sql = f"""
            SELECT COUNT(*) AS total
            FROM words
            WHERE {where_clause}
            """
            
            search_sql = f"""
            SELECT 
                id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form
            FROM words
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT :limit OFFSET :offset
            """
            
            # Add limit and offset parameters
            search_params['limit'] = limit
            search_params['offset'] = offset
            
            # Execute count query
            count_result = session.execute(text(count_sql), search_params).fetchone()
            total_count = count_result.total if count_result else 0
            
            # Execute search query
            search_results = list(session.execute(text(search_sql), search_params))
            
            # Format results
            results = []
            word_ids = [r.id for r in search_results]
            
            # If word IDs found, gather definitions (batch query)
            definitions_by_word = {}
            if word_ids:
                # Query for word definitions
                def_sql = """
                SELECT d.word_id, d.id, d.definition_text, pos.code as pos_code
                FROM definitions d
                LEFT JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
                WHERE d.word_id = ANY(:word_ids)
                """
                
                def_results = session.execute(text(def_sql), {"word_ids": word_ids})
                
                # Organize definitions by word ID
                for def_row in def_results:
                    if def_row.word_id not in definitions_by_word:
                        definitions_by_word[def_row.word_id] = []
                    
                    definitions_by_word[def_row.word_id].append({
                        "id": def_row.id,
                        "definition_text": def_row.definition_text,
                        "part_of_speech": def_row.pos_code
                    })
            
            # Format word results
            for row in search_results:
                word_result = {
                    "id": row.id,
                    "lemma": row.lemma,
                    "normalized_lemma": row.normalized_lemma,
                    "language_code": row.language_code,
                    "has_baybayin": row.has_baybayin,
                    "baybayin_form": row.baybayin_form,
                    "definitions": definitions_by_word.get(row.id, [])
                }
                
                results.append(word_result)
            
            # If include_full is requested, add complete word data for first result
            if include_full and results:
                # In case include_full is enabled, fetch full details for the first result
                first_id = results[0]["id"]
                
                # Create a new session for the additional query
                with Session(db.engine) as detail_session:
                    # Fetch full word data 
                    full_word = _fetch_word_details(
                        first_id,
                        include_definitions=True,
                        include_etymologies=True,
                        include_pronunciations=True,
                        include_credits=True,
                        include_relations=params.get('include_relations', True),
                        include_affixations=True,
                        include_root=True,
                        include_derived=True,
                        include_forms=params.get('include_forms', True),
                        include_templates=params.get('include_templates', True),
                        include_definition_relations=params.get('include_definition_relations', False)
                    )
                    
                    if full_word:
                        # Serialize using schema and add to response
                        from backend.schemas import WordSchema as SchemasWordSchema
                        schema = SchemasWordSchema()
                        full_data = schema.dump(full_word)
                        results[0]["full_data"] = full_data
            
            # Return formatted response
            return jsonify({
                "words": results,
                "total": total_count,
                "page": offset // limit + 1 if limit > 0 else 1,
                "per_page": limit,
                "query": query,
                "mode": mode
            })
        
    except ValidationError as ve:
        # Handle validation errors
        logger.warning(f"Search validation error: {str(ve)}")
        return jsonify({"error": "Invalid search parameters", "details": str(ve)}), 400
    except Exception as e:
        # Log and handle unexpected errors
        logger.error(f"Search error: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="search", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {type(e).__name__}", "message": str(e)}), 500

@bp.route("/random", methods=["GET"])
def get_random_word():
    """Get a random word from the dictionary, with optional filters."""
    try:
        API_REQUESTS.labels(endpoint="get_random_word", method="GET").inc()
        
        # Parse and validate filters
        language = request.args.get('language', 'tl')
        pos_code = request.args.get('pos')
        has_etymology = request.args.get('has_etymology', 'false').lower() == 'true'
        has_definitions = request.args.get('has_definitions', 'true').lower() == 'true'
        has_baybayin = request.args.get('has_baybayin', 'false').lower() == 'true'
        min_definitions = int(request.args.get('min_definitions', '0'))
        
        # Build the SQL query with filters
        query = build_random_word_query(
            language, pos_code, has_etymology, has_definitions, has_baybayin, min_definitions
        )
        
        # Execute the query to get a random word
        params = {
            'language': language,
            'pos_code': pos_code,
            'min_defs': min_definitions
        }
        
        # Create a new session for this operation to prevent transaction conflicts
        from sqlalchemy.orm import Session
        with Session(db.engine) as session:
            word_result = session.execute(text(query), params).fetchone()
            
            if not word_result:
                return jsonify({
                    "error": "No words found matching the criteria",
                    "filters": {
                        "language": language,
                        "pos_code": pos_code,
                        "has_etymology": has_etymology,
                        "has_definitions": has_definitions,
                        "has_baybayin": has_baybayin,
                        "min_definitions": min_definitions
                    }
                }), 404
                
            # Convert the raw result to a Word object
            word = Word()
            word.id = word_result.id
            word.lemma = word_result.lemma
            word.normalized_lemma = word_result.normalized_lemma
            word.language_code = word_result.language_code
            word.has_baybayin = word_result.has_baybayin
            word.baybayin_form = word_result.baybayin_form if word_result.has_baybayin else None
            
            # Commit the session and detach the word object
            session.expunge(word)
        
        # Use a separate method for fetching detailed data
        def fetch_word_details_safely(word_id):
            # Create a new session to avoid transaction conflicts
            with Session(db.engine) as new_session:
                try:
                    # Basic query to get word data
                    sql_word = """
                    SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
                          root_word_id, preferred_spelling, tags, source_info, word_metadata
                    FROM words
                    WHERE id = :id
                    """
                    result = new_session.execute(text(sql_word), {"id": word_id}).fetchone()
                    if not result:
                        return None
                        
                    # Manually create a Word object
                    full_word = Word()
                    full_word.id = result.id
                    full_word.lemma = result.lemma
                    full_word.normalized_lemma = result.normalized_lemma
                    full_word.language_code = result.language_code
                    full_word.has_baybayin = result.has_baybayin
                    full_word.baybayin_form = result.baybayin_form
                    full_word.root_word_id = result.root_word_id
                    
                    # Initialize empty collections
                    full_word.definitions = []
                    full_word.etymologies = []
                    full_word.pronunciations = []
                    full_word.credits = []
                    
                    # Add definitions
                    defs_sql = "SELECT id, definition_text, original_pos, standardized_pos_id FROM definitions WHERE word_id = :word_id"
                    for def_row in new_session.execute(text(defs_sql), {"word_id": word_id}):
                        definition = Definition()
                        definition.id = def_row.id
                        definition.definition_text = def_row.definition_text
                        definition.original_pos = def_row.original_pos
                        definition.standardized_pos_id = def_row.standardized_pos_id
                        full_word.definitions.append(definition)
                    
                    # Add etymologies
                    etym_sql = "SELECT id, etymology_text FROM etymologies WHERE word_id = :word_id"
                    for etym_row in new_session.execute(text(etym_sql), {"word_id": word_id}):
                        etymology = Etymology()
                        etymology.id = etym_row.id
                        etymology.etymology_text = etym_row.etymology_text
                        full_word.etymologies.append(etymology)
                    
                    # Add pronunciations
                    pron_sql = "SELECT id, type, value FROM pronunciations WHERE word_id = :word_id"
                    for pron_row in new_session.execute(text(pron_sql), {"word_id": word_id}):
                        pronunciation = Pronunciation()
                        pronunciation.id = pron_row.id
                        pronunciation.type = pron_row.type
                        pronunciation.value = pron_row.value
                        full_word.pronunciations.append(pronunciation)
                    
                    # Add root word reference if needed
                    if full_word.root_word_id:
                        root_sql = "SELECT id, lemma, language_code FROM words WHERE id = :root_id"
                        root_row = new_session.execute(text(root_sql), {"root_id": full_word.root_word_id}).fetchone()
                        if root_row:
                            root_word = Word()
                            root_word.id = root_row.id
                            root_word.lemma = root_row.lemma
                            root_word.language_code = root_row.language_code
                            full_word.root_word = root_word
                    
                    # Detach the word from the session
                    new_session.expunge_all()
                    return full_word
                    
                except Exception as e:
                    logger.error(f"Error in fetch_word_details_safely: {str(e)}", exc_info=True)
                    return None
        
        # Get word details using the safe function
        full_word = fetch_word_details_safely(word.id)
        
        if not full_word:
            return jsonify({"error": f"Failed to retrieve details for word ID {word.id}"}), 500
        
        # Serialize the word data
        from backend.schemas import WordSchema as SchemasWordSchema
        schema = SchemasWordSchema()
        result = schema.dump(full_word)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in get_random_word: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_random_word", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {type(e).__name__}",
            "message": str(e)
        }), 500

# Helper function to build the random word query
def build_random_word_query(language, pos_code, has_etymology, has_definitions, has_baybayin, min_definitions):
    """Build the SQL query for random word selection."""
    conditions = []
    
    # Build base query for random selection
    sql_query = """
    WITH filtered_words AS (
        SELECT w.id 
        FROM words w
    """
    
    # Add joins based on filters
    if pos_code:
        sql_query += """
        JOIN definitions d ON d.word_id = w.id
        JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
        """
        conditions.append("pos.code = :pos_code")
        
    if has_definitions or min_definitions > 0:
        if "JOIN definitions d" not in sql_query:
            sql_query += " JOIN definitions d ON d.word_id = w.id"
            
    if has_etymology:
        sql_query += " JOIN etymologies e ON e.word_id = w.id"
        
    # Add WHERE conditions
    if language:
        conditions.append("w.language_code = :language")
        
    if has_baybayin:
        # Add stricter condition for baybayin to avoid validation issues
        conditions.append("w.has_baybayin = TRUE AND w.baybayin_form IS NOT NULL AND TRIM(w.baybayin_form) != ''")
    else:
        # When not requesting Baybayin specifically, prefer words without Baybayin to avoid issues
        conditions.append("(w.has_baybayin = FALSE OR w.baybayin_form IS NULL)")
        
    if min_definitions > 0:
        conditions.append("(SELECT COUNT(*) FROM definitions WHERE word_id = w.id) >= :min_defs")
    
    # Ensure lemma is valid
    conditions.append("w.lemma IS NOT NULL AND TRIM(w.lemma) != ''")
        
    # Add conditions to query
    if conditions:
        sql_query += " WHERE " + " AND ".join(conditions)
        
    # Add GROUP BY if needed and final random selection
    sql_query += """
        GROUP BY w.id
    )
    SELECT w.*
    FROM words w
    WHERE w.id IN (SELECT id FROM filtered_words)
    ORDER BY RANDOM()
    LIMIT 1
    """
    
    return sql_query

@bp.route("/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get affixations for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_affixations", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for affixations lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in affixations lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with affixations, root, and derived using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=True, # Only include affixations
            include_root=True,       # Include root word
            include_derived=True,     # Include derived words
            include_forms=False,
            include_templates=False,
            include_definition_relations=False
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for affixations.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize affixations using AffixationSchema
        from backend.schemas import AffixationSchema, WordSchema
        affixation_schema = AffixationSchema(many=True)
        root_affix = affixation_schema.dump(word_entry.root_affixations)
        affixed_affix = affixation_schema.dump(word_entry.affixed_affixations)

        # Serialize root word and derived words using a limited WordSchema
        simple_word_schema = WordSchema(many=True, only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
        derived_words_dump = simple_word_schema.dump(word_entry.derived_words)

        root_word_dump = None
        if word_entry.root_word:
            # Use single instance schema for root word
            root_word_dump = WordSchema(only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form')).dump(word_entry.root_word)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "is_root": not word_entry.root_word_id, # Check if it's a root word itself
            "root_affixations": root_affix, # List of words derived from this word via affixation
            "affixed_affixations": affixed_affix, # List showing how this word was formed via affixation (if applicable)
            "root_word": root_word_dump, # The root word this word is derived from (if applicable)
            "derived_words": derived_words_dump, # List of words directly derived from this word (non-affixation based, e.g., via root_word_id)
            "total_affixations": len(root_affix) + len(affixed_affix)
        }

        logger.debug(f"Returning affixations for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving affixations for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving affixations for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_affixations", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get pronunciation data for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_pronunciation", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for pronunciation lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in pronunciation lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only pronunciations using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=True, # Only include pronunciations
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=False,
            include_templates=False,
            include_definition_relations=False
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for pronunciation.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize pronunciations using PronunciationType schema
        pronunciation_schema = PronunciationType(many=True)
        pronunciations_dump = pronunciation_schema.dump(word_entry.pronunciations)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "pronunciations": pronunciations_dump,
            "has_audio": any(p['type'] == 'audio' for p in pronunciations_dump), # Check dumped data
            "has_ipa": any(p['type'] == 'ipa' for p in pronunciations_dump),     # Check dumped data
            "total_pronunciations": len(pronunciations_dump)
        }

        logger.debug(f"Returning pronunciations for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving pronunciation for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving pronunciation for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_pronunciation", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get etymology data for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_etymology", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for etymology lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in etymology lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only etymologies using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=True, # Only include etymologies
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=False,
            include_templates=False,
            include_definition_relations=False
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for etymology.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize etymologies using EtymologySchema
        etymology_schema = EtymologySchema(many=True)
        etymologies_dump = etymology_schema.dump(word_entry.etymologies)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "etymologies": etymologies_dump,
            "total_etymologies": len(etymologies_dump)
        }

        # Note: Manual extraction of components/languages removed.
        # This logic can be handled by the schema or client if needed.

        logger.debug(f"Returning etymologies for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving etymology for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving etymology for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_etymology", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/semantic_network", methods=["GET"])
@cached_query(timeout=900)  # Cache for 15 minutes as this is resource-intensive
def get_semantic_network(word: str):
    """Generate a semantic network for a word."""
    try:
        depth = min(int(request.args.get("depth", 2)), 4)  # Default 2, max 4
        breadth = min(int(request.args.get("breadth", 10)), 50)  # Default 10, max 50
        relation_types = request.args.get("relation_types", None)

        # Start timing
        start_time = time.time()
        
        # If relation_types is provided, split by comma and filter
        allowed_types = None
        if relation_types:
            allowed_types = [rt.strip() for rt in relation_types.split(",")]
        
        # Get the word ID
        word_id = None
        normalized = normalize_word(word)

        # Try to parse as integer ID first
        try:
            word_id = int(word)
            # Verify the ID exists
            sql = "SELECT id, lemma, normalized_lemma, language_code FROM words WHERE id = :id"
            word_result = db.session.execute(text(sql), {"id": word_id}).fetchone()
            
            if not word_result:
                return jsonify({
                    "error": f"Word with ID {word_id} not found",
                    "nodes": [],
                    "links": [],
                    "metadata": {
                        "root_word": None,
                        "normalized_lemma": None,
                        "language_code": None,
                        "depth": depth,
                        "total_nodes": 0,
                        "total_edges": 0
                    }
                }), 404
                
            word_data = {
                "id": word_result.id,
                "lemma": word_result.lemma,
                "normalized_lemma": word_result.normalized_lemma,
                "language_code": word_result.language_code
            }
        except ValueError:
            # Use direct SQL to lookup the ID by word
            sql = """
            SELECT id, lemma, normalized_lemma, language_code FROM words
            WHERE LOWER(lemma) = LOWER(:word) 
            OR LOWER(normalized_lemma) = LOWER(:normalized)
            OR lemma = :word
            OR normalized_lemma = :normalized
            LIMIT 1
            """
            params = {"word": word, "normalized": normalized}
            try:
                word_result = db.session.execute(text(sql), params).fetchone()
            except SQLAlchemyError as e:
                logger.error(f"Database error looking up word '{word}': {str(e)}")
                return jsonify({
                    "error": "Database error occurred",
                    "nodes": [],
                    "links": [],
                    "metadata": {
                        "root_word": word,
                        "normalized_lemma": normalized,
                        "language_code": None,
                        "depth": depth,
                        "total_nodes": 0,
                        "total_edges": 0,
                        "error_details": str(e)
                    }
                }), 500
            
            if not word_result:
                return jsonify({
                    "error": f"Word '{word}' not found",
                    "nodes": [],
                    "links": [],
                    "metadata": {
                        "root_word": word,
                        "normalized_lemma": normalized,
                        "language_code": None,
                        "depth": depth,
                        "total_nodes": 0,
                        "total_edges": 0
                    }
                }), 404
            
            word_id = word_result.id
            word_data = {
                "id": word_result.id,
                "lemma": word_result.lemma,
                "normalized_lemma": word_result.normalized_lemma,
                "language_code": word_result.language_code
            }
        
        # Log the start of the operation
        logger.info(f"Generating semantic network for word '{word}' (depth: {depth}, breadth: {breadth})")

        # Initialize nodes and edges sets to avoid duplicates
        nodes = {}
        edges = {}

        # Add the main word as the first node
        nodes[word_id] = {
            "id": str(word_id),
            "label": word_data["lemma"],
            "word": word_data["lemma"],
            "language": word_data["language_code"],
            "type": "main",
            "depth": 0,
            "main": True,
            "has_baybayin": False,  # Will be updated if available
            "baybayin_form": None,
            "normalized_lemma": word_data["normalized_lemma"]
        }

        # Function to process relations recursively
        def process_relations(current_id, current_depth):
            if current_depth >= depth:
                return
                
            # Get outgoing relations
            outgoing_sql = """
                SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type,
                       w.id as target_id, w.lemma as target_lemma, 
                       w.normalized_lemma as target_normalized_lemma,
                       w.language_code as target_language_code, 
                       w.has_baybayin as target_has_baybayin,
                       w.baybayin_form as target_baybayin_form
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :word_id
                ORDER BY 
                    CASE 
                        WHEN r.relation_type IN ('synonym', 'antonym') THEN 1
                        WHEN r.relation_type IN ('hypernym', 'hyponym') THEN 2
                        ELSE 3
                    END,
                    w.lemma
                LIMIT :breadth
            """
            outgoing = db.session.execute(text(outgoing_sql), 
                                        {"word_id": current_id, "breadth": breadth}).fetchall()

            # Get incoming relations
            incoming_sql = """
                SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type,
                       w.id as source_id, w.lemma as source_lemma,
                       w.normalized_lemma as source_normalized_lemma,
                       w.language_code as source_language_code,
                       w.has_baybayin as source_has_baybayin,
                       w.baybayin_form as source_baybayin_form
                FROM relations r
                JOIN words w ON r.from_word_id = w.id
                WHERE r.to_word_id = :word_id
                ORDER BY 
                    CASE 
                        WHEN r.relation_type IN ('synonym', 'antonym') THEN 1
                        WHEN r.relation_type IN ('hypernym', 'hyponym') THEN 2
                        ELSE 3
                    END,
                    w.lemma
                LIMIT :breadth
            """
            incoming = db.session.execute(text(incoming_sql), 
                                        {"word_id": current_id, "breadth": breadth}).fetchall()
            
            # Process outgoing relations
            for rel in outgoing:
                if allowed_types and rel.relation_type not in allowed_types:
                    continue
                    
                target_id = rel.target_id
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": str(target_id),
                        "label": rel.target_lemma,
                        "word": rel.target_lemma,
                        "language": rel.target_language_code,
                        "type": rel.relation_type,
                        "depth": current_depth + 1,
                        "main": False,
                        "has_baybayin": rel.target_has_baybayin,
                        "baybayin_form": rel.target_baybayin_form,
                        "normalized_lemma": rel.target_normalized_lemma
                    }

                edge_id = f"{current_id}-{target_id}-{rel.relation_type}"
                if edge_id not in edges:
                    edges[edge_id] = {
                    "id": edge_id,
                        "source": str(current_id),
                        "target": str(target_id),
                        "type": rel.relation_type,
                        "directed": rel.relation_type not in ["synonym", "antonym", "related"]
                    }

                # Recursively process the target word's relations
                if current_depth + 1 < depth:
                    process_relations(target_id, current_depth + 1)
            
            # Process incoming relations
            for rel in incoming:
                if allowed_types and rel.relation_type not in allowed_types:
                    continue
                    
                source_id = rel.source_id
                if source_id not in nodes:
                    nodes[source_id] = {
                        "id": str(source_id),
                        "label": rel.source_lemma,
                        "word": rel.source_lemma,
                        "language": rel.source_language_code,
                        "type": rel.relation_type,
                        "depth": current_depth + 1,
                        "main": False,
                        "has_baybayin": rel.source_has_baybayin,
                        "baybayin_form": rel.source_baybayin_form,
                        "normalized_lemma": rel.source_normalized_lemma
                    }

                edge_id = f"{source_id}-{current_id}-{rel.relation_type}"
                if edge_id not in edges:
                    edges[edge_id] = {
                    "id": edge_id,
                        "source": str(source_id),
                        "target": str(current_id),
                        "type": rel.relation_type,
                        "directed": rel.relation_type not in ["synonym", "antonym", "related"]
                    }

                # Recursively process the source word's relations
                if current_depth + 1 < depth:
                    process_relations(source_id, current_depth + 1)

        # Start processing relations from the main word
        process_relations(word_id, 0)

        # Convert nodes and edges to lists
        nodes_list = list(nodes.values())
        edges_list = list(edges.values())

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare response
        result = {
            "nodes": nodes_list,
            "links": edges_list,
            "metadata": {
                "root_word": word_data["lemma"],
                "normalized_lemma": word_data["normalized_lemma"],
                "language_code": word_data["language_code"],
                "depth": depth,
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
                "execution_time": execution_time
            }
        }
        
        logger.info(f"Generated semantic network for '{word}' with {len(nodes_list)} nodes and {len(edges_list)} edges in {execution_time:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating semantic network for word '{word}'", exc_info=True)
        return jsonify({
            "error": str(e),
            "nodes": [],
            "links": [],
            "metadata": {
                "root_word": word,
                "normalized_lemma": normalized if 'normalized' in locals() else None,
                "language_code": None,
                "depth": depth if 'depth' in locals() else None,
                "total_nodes": 0,
                "total_edges": 0,
                "error_details": str(e)
            }
        }), 500

@bp.route("/words/<int:word_id>/etymology/tree", methods=["GET"])
def get_etymology_tree(word_id: int):
    """Get etymology tree for a word."""
    try:
        # Validate parameters
        depth = min(int(request.args.get("depth", 2)), 3)  # Max depth 3 to prevent excessive recursion
        
        # Use direct SQL query instead of ORM to avoid issues with missing columns
        sql = "SELECT id, lemma, language_code FROM words WHERE id = :id"
        word_result = db.session.execute(text(sql), {"id": word_id}).fetchone()
        
        if not word_result:
            return jsonify({"error": f"Word with ID {word_id} not found"}), 404
            
        # Create a minimal Word object manually to avoid ORM loading issues
        class MinimalWord:
            def __init__(self, id, lemma, language_code):
                self.id = id
                self.lemma = lemma
                self.language_code = language_code
                
        word = MinimalWord(word_result.id, word_result.lemma, word_result.language_code)
            
        # Helper functions for building etymology tree
        def get_word_by_lemma(lemma, language=None):
            # Use direct SQL instead of ORM
            if language:
                sql = "SELECT id, lemma, language_code FROM words WHERE lemma = :lemma AND language_code = :language LIMIT 1"
                result = db.session.execute(text(sql), {"lemma": lemma, "language": language}).fetchone()
            else:
                sql = "SELECT id, lemma, language_code FROM words WHERE lemma = :lemma LIMIT 1"
                result = db.session.execute(text(sql), {"lemma": lemma}).fetchone()
                
            if result:
                return MinimalWord(result.id, result.lemma, result.language_code)
            return None
        
        def get_etymologies(word_id):
            # Use direct SQL instead of ORM
            sql = "SELECT id, etymology_text, normalized_components FROM etymologies WHERE word_id = :word_id"
            results = db.session.execute(text(sql), {"word_id": word_id}).fetchall()
            return results
        
        def build_etymology_tree(word, depth=0, max_depth=2, visited=None):
            # Limit max depth to prevent hanging
            if visited is None:
                visited = set()
                
            if depth > max_depth or word.id in visited:
                return None
                
            visited.add(word.id)
            
            # Get word details
            word_node = {
                "id": word.id,
                "lemma": word.lemma,
                "language_code": word.language_code,
                "etymologies": []
            }
            
            # Get etymologies
            etymologies = get_etymologies(word.id)
            
            # Process each etymology
            for etym in etymologies:
                etymology_node = {
                    "id": etym.id,
                    "text": etym.etymology_text,
                    "components": []
                }
                
                # Extract components and try to link to actual words
                if etym.normalized_components:
                    try:
                        components = json.loads(etym.normalized_components)
                    except:
                        components = []
                    
                    for comp in components:
                        # Try to find matching word
                        language = comp.get("language")
                        component_word = get_word_by_lemma(comp.get("term"), language)
                        
                        component_node = {
                            "term": comp.get("term"),
                            "language": comp.get("language"),
                            "meaning": comp.get("meaning"),
                            "type": comp.get("type"),
                            "matched": bool(component_word)
                        }
                        
                        # If we found a matching word and haven't reached max depth,
                        # recursively build its etymology tree
                        if component_word and depth < max_depth:
                            component_node["word"] = build_etymology_tree(
                                component_word, 
                                depth + 1, 
                                max_depth, 
                                visited
                            )
                            
                        etymology_node["components"].append(component_node)
                        
                word_node["etymologies"].append(etymology_node)
                
            return word_node
        
        # Build the etymology tree
        etymology_tree = build_etymology_tree(word, 0, depth)
        
        return jsonify(etymology_tree)
    except Exception as e:
        logger.error(f"Error generating etymology tree for word ID {word_id}", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/relationships/types", methods=["GET"])
def get_relationship_types():
    """Get all relationship types."""
    try:
        # Query distinct relationship types
        relation_types = db.session.query(distinct(Relation.relation_type)).all()
        relation_types = [r[0] for r in relation_types]
        
        # Create structured response with categories
        result = {
            "types": relation_types,
            "categories": {
                "semantic": [],
                "derivational": [],
                "taxonomic": [],
                "variant": [],
                "usage": [],
                "other": []
            },
            "descriptions": {}
        }
        
        # Manual categorization based on types
        category_mapping = {
            "synonym": "semantic",
            "antonym": "semantic",
            "related": "semantic",
            "similar": "semantic",
            "hypernym": "taxonomic",
            "hyponym": "taxonomic",
            "meronym": "taxonomic",
            "holonym": "taxonomic",
            "derived": "derivational",
            "root": "derivational",
            "affix": "derivational",
            "variant": "variant",
            "spelling": "variant",
            "abbreviation": "variant",
            "usage": "usage"
        }
        
        # Categorize each type
        for type_name in relation_types:
            category = category_mapping.get(type_name, "other")
            result["categories"][category].append(type_name)
            
            # Add descriptions for common types
            if type_name == "synonym":
                result["descriptions"][type_name] = "Words with the same or similar meaning"
            elif type_name == "antonym":
                result["descriptions"][type_name] = "Words with the opposite meaning"
            elif type_name == "hypernym":
                result["descriptions"][type_name] = "A word with a broader meaning that includes the meaning of the related word"
            elif type_name == "hyponym":
                result["descriptions"][type_name] = "A word with a more specific meaning than the related word"
            elif type_name == "derived":
                result["descriptions"][type_name] = "Word derived from the related word"
            elif type_name == "root":
                result["descriptions"][type_name] = "Root word of the related word"
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error retrieving relationship types: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Add schema for DefinitionRelation
class DefinitionRelationSchema(BaseSchema):
    """Schema for definition relation data."""
    relation_type = fields.Str(required=True)
    definition_id = fields.Int(required=True) 
    word_id = fields.Int(required=True)
    relation_data = fields.Dict()  # Use relation_data to match our property
    definition = fields.Nested(DefinitionSchema, only=('id', 'definition_text'))
    related_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

@bp.route("/words/<path:word>/definition_relations", methods=["GET"])
def get_word_definition_relations(word):
    """Get definition relations for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_definition_relations", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for definition relations lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in definition relations lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only definition relations using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=True,  # Include definitions as they're related
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=False,
            include_templates=False,
            include_definition_relations=True  # Only include definition relations
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for definition relations.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize definition relations using DefinitionRelationSchema
        definition_relation_schema = DefinitionRelationSchema(many=True)
        definition_relations_dump = definition_relation_schema.dump(word_entry.definition_relations)
        
        # Get related definitions
        related_definitions = []
        if hasattr(word_entry, 'related_definitions') and word_entry.related_definitions:
            definition_schema = DefinitionSchema(many=True, exclude=("word", "related_words"))
            related_definitions = definition_schema.dump(word_entry.related_definitions)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "definition_relations": definition_relations_dump,
            "related_definitions": related_definitions,
            "total_definition_relations": len(definition_relations_dump)
        }

        logger.debug(f"Returning definition relations for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving definition relations for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving definition relations for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_definition_relations", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/forms", methods=["GET"])
def get_word_forms(word):
    """Get word forms for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_forms", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for forms lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in forms lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only forms using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=True,  # Only include forms
            include_templates=False,
            include_definition_relations=False
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for forms.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize forms using WordFormSchema
        form_schema = WordFormSchema(many=True, exclude=("word",))
        forms_dump = form_schema.dump(word_entry.forms)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "forms": forms_dump,
            "total_forms": len(forms_dump),
            "canonical_forms": [f for f in forms_dump if f.get('is_canonical', False)],
            "primary_forms": [f for f in forms_dump if f.get('is_primary', False)]
        }

        logger.debug(f"Returning forms for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving forms for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving forms for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_forms", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/templates", methods=["GET"])
def get_word_templates(word):
    """Get word templates for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_templates", method="GET").inc()

        word_id = None
        # Find the word entry ID (handle ID or lemma)
        try:
            word_id = int(word)
        except ValueError:
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                    func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
            else:
                logger.warning(f"Word '{word}' not found for templates lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
            logger.error(f"Failed to determine word ID for '{word}' in templates lookup.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only templates using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=False,
            include_templates=True,  # Only include templates
            include_definition_relations=False
        )

        if not word_entry:
            logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for templates.")
            return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize templates using WordTemplateSchema
        template_schema = WordTemplateSchema(many=True, exclude=("word",))
        templates_dump = template_schema.dump(word_entry.templates)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "templates": templates_dump,
            "total_templates": len(templates_dump),
        }

        logger.debug(f"Returning templates for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving templates for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving templates for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_templates", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

# Import serialization and file handling
import json
import csv
import io
import zipfile
import datetime
import uuid
from werkzeug.utils import secure_filename
from flask import request, jsonify, send_file, make_response
from sqlalchemy import or_, and_, not_, func
import os
from datetime import datetime, timezone
from marshmallow import Schema, fields, validate, ValidationError

# Add schema for export/import filters
class ExportFilterSchema(Schema):
    """Schema for export filter parameters."""
    language_code = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    created_after = fields.DateTime(dump_default=None, load_default=None)
    created_before = fields.DateTime(dump_default=None, load_default=None)
    updated_after = fields.DateTime(dump_default=None, load_default=None)
    updated_before = fields.DateTime(dump_default=None, load_default=None)
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_etymologies = fields.Bool(dump_default=True, load_default=True)
    include_pronunciations = fields.Bool(dump_default=True, load_default=True)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_credits = fields.Bool(dump_default=True, load_default=True)
    include_forms = fields.Bool(dump_default=True, load_default=True)
    include_templates = fields.Bool(dump_default=True, load_default=True)
    include_definition_relations = fields.Bool(dump_default=True, load_default=True)
    limit = fields.Int(validate=validate.Range(min=1, max=10000), dump_default=5000, load_default=5000)
    offset = fields.Int(validate=validate.Range(min=0), dump_default=0, load_default=0)
    format = fields.Str(validate=validate.OneOf(['json', 'csv', 'zip']), dump_default='json', load_default='json')

# Add export endpoint
@bp.route("/export", methods=["GET"])
@limiter.limit("5 per hour", key_func=get_remote_address)  # Rate limit exports
def export_dictionary():
    """
    Export dictionary data based on filters.
    Supports JSON, CSV, and ZIP formats.
    """
    try:
        # Track metrics
        API_REQUESTS.labels(endpoint="export_dictionary", method="GET").inc()
        start_time = time.time()

        # Parse and validate query parameters
        export_schema = ExportFilterSchema()
        try:
            filter_args = export_schema.load(request.args)
        except ValidationError as err:
            logger.warning("Export filter validation failed", errors=err.messages)
            return jsonify({"error": "Invalid export parameters", "details": err.messages}), 400

        # Check authorization (implement proper auth middleware)
        # This is a placeholder - implement actual auth
        if not request.headers.get('X-Api-Key') and 'Authorization' not in request.headers:
            return jsonify({
                "error": "Authorization required",
                "message": "You must provide an API key to export dictionary data"
            }), 401

        # Base query for words with optimized loading
        query = Word.query

        # Apply filters
        if filter_args.get('language_code'):
            query = query.filter(Word.language_code == filter_args['language_code'])
        
        if filter_args.get('has_etymology'):
            query = query.join(Word.etymologies).group_by(Word.id).having(func.count(Etymology.id) > 0)
            
        if filter_args.get('has_pronunciation'):
            query = query.join(Word.pronunciations).group_by(Word.id).having(func.count(Pronunciation.id) > 0)
            
        if filter_args.get('has_baybayin'):
            query = query.filter(Word.has_baybayin == True, Word.baybayin_form.isnot(None))
            
        # Don't filter by completeness_score in SQL as it's a hybrid property
        min_completeness = filter_args.get('min_completeness')
            
        if filter_args.get('created_after'):
            query = query.filter(Word.created_at >= filter_args['created_after'])
            
        if filter_args.get('created_before'):
            query = query.filter(Word.created_at <= filter_args['created_before'])
            
        if filter_args.get('updated_after'):
            query = query.filter(Word.updated_at >= filter_args['updated_after'])
            
        if filter_args.get('updated_before'):
            query = query.filter(Word.updated_at <= filter_args['updated_before'])
            
        if filter_args.get('pos'):
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(PartOfSpeech.code == filter_args['pos'])

        # Get count before pagination for metadata
        total_count = query.count()
        
        # Apply pagination (but don't do that if we need to filter by completeness)
        apply_pagination = not min_completeness
        if apply_pagination:
            query = query.limit(filter_args['limit']).offset(filter_args['offset'])
        
        # Configure eager loading options based on parameters
        load_options = []
        
        if filter_args.get('include_definitions', True):
            load_options.append(selectinload(Word.definitions).selectinload(Definition.standardized_pos))
            load_options.append(selectinload(Word.definitions).selectinload(Definition.links))
            load_options.append(selectinload(Word.definitions).selectinload(Definition.categories))
        
        if filter_args.get('include_etymologies', True):
            load_options.append(selectinload(Word.etymologies))
            
        if filter_args.get('include_pronunciations', True):
            load_options.append(selectinload(Word.pronunciations))
            
        if filter_args.get('include_credits', True):
            load_options.append(selectinload(Word.credits))
            
        if filter_args.get('include_relations', True):
            load_options.append(selectinload(Word.outgoing_relations).selectinload(Relation.target_word))
            load_options.append(selectinload(Word.incoming_relations).selectinload(Relation.source_word))
            
        if filter_args.get('include_forms', True):
            load_options.append(selectinload(Word.forms))
            
        if filter_args.get('include_templates', True):
            load_options.append(selectinload(Word.templates))
            
        if filter_args.get('include_definition_relations', True):
            load_options.append(selectinload(Word.definition_relations))
            load_options.append(selectinload(Word.related_definitions))
            
        if load_options:
            query = query.options(*load_options)
            
        # Execute query
        words = query.all()
        
        # Apply completeness filtering after retrieving from database
        if min_completeness is not None:
            original_count = len(words)
            words = [w for w in words if getattr(w, 'completeness_score', 0) >= min_completeness]
            # If we didn't apply pagination earlier, do it now on the filtered results
            if not apply_pagination and len(words) > filter_args['limit']:
                offset = min(filter_args['offset'], len(words))
                limit = min(filter_args['limit'], len(words) - offset)
                words = words[offset:offset + limit]
                
            logger.debug(f"Filtered words by completeness: {original_count} -> {len(words)}")
        
        # Prepare metadata
        export_metadata = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "total_records": total_count,
            "exported_records": len(words),
            "filters_applied": {k: v for k, v in filter_args.items() if k != 'format' and v is not None},
            "format": filter_args.get('format', 'json')
        }
        
        # Use WordSchema for serialization
        schema = WordSchema(many=True)
        word_data = schema.dump(words)
        
        # Prepare export based on format
        if filter_args.get('format') == 'csv':
            # CSV export (simplified - just words and basic info)
            output = io.StringIO()
            fieldnames = ['id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form', 
                         'completeness_score', 'created_at', 'updated_at']
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for word in word_data:
                # Extract only CSV fields
                row = {field: word.get(field) for field in fieldnames if field in word}
                writer.writerow(row)
                
            # Set up response
            output.seek(0)
            
            # Create response with CSV file
            response = make_response(output.getvalue())
            response.headers['Content-Type'] = 'text/csv'
            response.headers['Content-Disposition'] = f'attachment; filename=dictionary_export_{datetime.now().strftime("%Y%m%d")}.csv'
            
            # Record request latency
            request_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="export_dictionary").observe(request_time)
            
            return response
            
        elif filter_args.get('format') == 'zip':
            # ZIP export (full data in JSON)
            memory_file = io.BytesIO()
            
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add metadata file
                zf.writestr('metadata.json', json.dumps(export_metadata, indent=2))
                
                # Add data file with all words
                zf.writestr('words.json', json.dumps(word_data, indent=2))
                
                # Add separate files for each word for easier access
                for word in word_data:
                    lemma = secure_filename(word['lemma'])
                    filename = f"words/{word['id']}_{lemma}.json"
                    zf.writestr(filename, json.dumps(word, indent=2))
            
            # Reset file pointer
            memory_file.seek(0)
            
            # Record request latency
            request_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="export_dictionary").observe(request_time)
            
            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'dictionary_export_{datetime.now().strftime("%Y%m%d")}.zip'
            )
        
        else:  # Default to JSON
            # JSON export (full data)
            export_data = {
                "metadata": export_metadata,
                "words": word_data
            }
            
            # Record request latency
            request_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="export_dictionary").observe(request_time)
            
            return jsonify(export_data)
            
    except SQLAlchemyError as e:
        logger.error(f"Database error during export: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="export_dictionary", error_type="database_error").inc()
        return jsonify({"error": "Database error during export", "details": str(e)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error during export: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="export_dictionary", error_type="unexpected_error").inc()
        return jsonify({"error": "Unexpected error during export", "details": str(e)}), 500

# Add import endpoint  
@bp.route("/import", methods=["POST"])
@limiter.limit("10 per hour", key_func=get_remote_address)  # Rate limit imports
def import_dictionary():
    """
    Import dictionary data from a JSON file.
    Validates the data, handles duplicates, and reports results.
    """
    try:
        # Track metrics
        API_REQUESTS.labels(endpoint="import_dictionary", method="POST").inc()
        start_time = time.time()
        
        # Check authorization (implement proper auth middleware)
        # This is a placeholder - implement actual auth
        if not request.headers.get('X-Api-Key') and 'Authorization' not in request.headers:
            return jsonify({
                "error": "Authorization required",
                "message": "You must provide an API key to import dictionary data"
            }), 401
            
        # Check content type
        if 'multipart/form-data' not in request.content_type and 'application/json' not in request.content_type:
            return jsonify({
                "error": "Invalid content type",
                "message": "Request must be multipart/form-data with a file or application/json"
            }), 415
            
        # Get import options
        update_existing = request.form.get('update_existing', 'false').lower() == 'true'
        skip_validation = request.form.get('skip_validation', 'false').lower() == 'true'
        batch_size = min(int(request.form.get('batch_size', 100)), 1000)  # Default 100, max 1000
        import_relations = request.form.get('import_relations', 'true').lower() == 'true'
        dry_run = request.form.get('dry_run', 'false').lower() == 'true'
        
        # Process input data
        import_data = None
        
        if 'multipart/form-data' in request.content_type:
            # Handle file upload
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
                
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
                
            if not file.filename.endswith(('.json', '.zip')):
                return jsonify({"error": "Only JSON and ZIP files are supported"}), 400
                
            # Process based on file type
            if file.filename.endswith('.zip'):
                # Extract JSON from ZIP
                with zipfile.ZipFile(file) as zip_file:
                    # Look for words.json or similar in the ZIP
                    json_files = [f for f in zip_file.namelist() if f.endswith('.json') and ('words' in f or 'dictionary' in f)]
                    
                    if not json_files:
                        return jsonify({"error": "No dictionary data found in ZIP file"}), 400
                        
                    # Use the first matching file
                    with zip_file.open(json_files[0]) as json_file:
                        import_data = json.load(json_file)
            else:
                # Direct JSON file
                import_data = json.load(file)
        else:
            # Handle direct JSON input
            import_data = request.json
            
        if not import_data:
            return jsonify({"error": "No valid import data provided"}), 400
            
        # Extract words from different possible formats
        words_to_import = []
        
        if isinstance(import_data, list):
            # Direct array of words
            words_to_import = import_data
        elif isinstance(import_data, dict):
            # Dictionary with metadata and words
            if 'words' in import_data and isinstance(import_data['words'], list):
                words_to_import = import_data['words']
            elif 'data' in import_data and isinstance(import_data['data'], list):
                words_to_import = import_data['data']
            else:
                # Single word entry
                words_to_import = [import_data]
                
        if not words_to_import:
            return jsonify({"error": "No words found in import data"}), 400
            
        logger.info(f"Starting import of {len(words_to_import)} words, update_existing={update_existing}, dry_run={dry_run}")
        
        # Validate import data unless skipped
        if not skip_validation:
            # Basic schema validation
            schema = WordSchema(many=True)
            try:
                # Validate but don't load into Python objects yet
                validation_errors = schema.validate(words_to_import)
                if validation_errors:
                    return jsonify({
                        "error": "Import data validation failed",
                        "details": validation_errors
                    }), 400
            except Exception as e:
                return jsonify({
                    "error": "Error validating import data",
                    "details": str(e)
                }), 400
                
        # Initialize tracking variables
        results = {
            "total": len(words_to_import),
            "created": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process in batches to avoid memory issues
        for i in range(0, len(words_to_import), batch_size):
            batch = words_to_import[i:i+batch_size]
            
            # Start a new transaction for each batch
            try:
                if not dry_run:
                    # Process this batch
                    for word_data in batch:
                        try:
                            # Check if word exists (by ID or lemma)
                            existing_word = None
                            if 'id' in word_data and word_data['id']:
                                existing_word = Word.query.get(word_data['id'])
                                
                            if not existing_word and 'lemma' in word_data and word_data['lemma']:
                                # Try to find by normalized lemma
                                normalized = normalize_lemma(word_data['lemma'])
                                existing_word = Word.query.filter(
                                    or_(
                                        Word.normalized_lemma == normalized,
                                        Word.lemma == word_data['lemma']
                                    )
                                ).first()
                                
                            if existing_word:
                                if update_existing:
                                    # Update existing word with new data
                                    for key, value in word_data.items():
                                        # Skip relationships for direct update
                                        if key not in ['definitions', 'etymologies', 'pronunciations', 
                                                      'credits', 'relations', 'forms', 'templates',
                                                      'outgoing_relations', 'incoming_relations',
                                                      'definition_relations']:
                                            setattr(existing_word, key, value)
                                            
                                    # Handle relationship updates selectively if needed
                                    # This would require more complex logic to update
                                    # related objects without creating duplicates
                                    
                                    # Mark as updated
                                    existing_word.updated_at = datetime.now(timezone.utc)
                                    db.session.add(existing_word)
                                    results["updated"] += 1
                                else:
                                    # Skip this word
                                    results["skipped"] += 1
                            else:
                                # Create new word
                                # Strip relationships for initial creation
                                new_word_data = {k: v for k, v in word_data.items() 
                                              if k not in ['definitions', 'etymologies', 'pronunciations', 
                                                          'credits', 'relations', 'forms', 'templates',
                                                          'outgoing_relations', 'incoming_relations',
                                                          'definition_relations']}
                                
                                # Generate a normalized lemma if not provided
                                if 'normalized_lemma' not in new_word_data and 'lemma' in new_word_data:
                                    new_word_data['normalized_lemma'] = normalize_lemma(new_word_data['lemma'])
                                    
                                # Create the new word
                                new_word = Word(**new_word_data)
                                db.session.add(new_word)
                                db.session.flush()  # Get the ID without committing
                                
                                # Now handle relationships - just a stub, would need expansion
                                # for each relationship type
                                
                                results["created"] += 1
                                
                        except Exception as word_error:
                            # Track individual word errors
                            results["failed"] += 1
                            results["errors"].append({
                                "lemma": word_data.get('lemma', 'Unknown'),
                                "error": str(word_error)
                            })
                            logger.error(f"Error importing word {word_data.get('lemma', 'Unknown')}: {str(word_error)}")
                            # Continue with next word, don't break the batch
                    
                    # Commit the batch
                    db.session.commit()
                else:
                    # Dry run - just count
                    for word_data in batch:
                        # Check if word exists
                        existing_word = None
                        if 'id' in word_data and word_data['id']:
                            existing_word = Word.query.get(word_data['id'])
                            
                        if not existing_word and 'lemma' in word_data and word_data['lemma']:
                            normalized = normalize_lemma(word_data['lemma'])
                            existing_word = Word.query.filter(
                                or_(
                                    Word.normalized_lemma == normalized,
                                    Word.lemma == word_data['lemma']
                                )
                            ).first()
                            
                        if existing_word:
                            if update_existing:
                                results["updated"] += 1
                            else:
                                results["skipped"] += 1
                        else:
                            results["created"] += 1
                    
            except SQLAlchemyError as batch_error:
                # Roll back on batch error
                db.session.rollback()
                logger.error(f"Database error processing batch {i//batch_size + 1}: {str(batch_error)}")
                results["errors"].append({
                    "batch": i//batch_size + 1,
                    "error": str(batch_error)
                })
                
            # Log progress
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(words_to_import)-1)//batch_size + 1}")
            
        # Add import summary to results
        results["dry_run"] = dry_run
        results["import_time"] = time.time() - start_time
        
        # Record request latency
        REQUEST_LATENCY.labels(endpoint="import_dictionary").observe(results["import_time"])
        
        return jsonify({
            "status": "success" if results["failed"] == 0 else "partial_success",
            "message": "Import completed" if not dry_run else "Dry run completed",
            "results": results
        })
        
    except SQLAlchemyError as e:
        logger.error(f"Database error during import: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="import_dictionary", error_type="database_error").inc()
        return jsonify({"error": "Database error during import", "details": str(e)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error during import: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="import_dictionary", error_type="unexpected_error").inc()
        return jsonify({"error": "Unexpected error during import", "details": str(e)}), 500

class QualityAssessmentFilterSchema(Schema):
    """Schema for quality assessment filter parameters."""
    language_code = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    max_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    created_after = fields.DateTime(dump_default=None, load_default=None)
    created_before = fields.DateTime(dump_default=None, load_default=None)
    updated_after = fields.DateTime(dump_default=None, load_default=None)
    updated_before = fields.DateTime(dump_default=None, load_default=None)
    include_issues = fields.Bool(dump_default=True, load_default=True)
    issue_severity = fields.Str(validate=validate.OneOf(['all', 'critical', 'warning', 'info']), dump_default='all', load_default='all')
    max_results = fields.Int(validate=validate.Range(min=1, max=1000), dump_default=100, load_default=100)

@bp.route("/quality_assessment", methods=["GET"])
@cached_query(timeout=1800)  # Cache for 30 minutes as this is a resource-intensive operation
@limiter.limit("5 per hour", key_func=get_remote_address)  # Rate limit to prevent abuse
def quality_assessment():
    """
    Analyze dictionary data quality, completeness, and identify issues.
    Returns detailed statistics and potential improvement areas.
    """
    try:
        # Track metrics
        API_REQUESTS.labels(endpoint="quality_assessment", method="GET").inc()
        start_time = time.time()
        
        # Validate and parse filter parameters
        filter_schema = QualityAssessmentFilterSchema()
        try:
            filter_args = filter_schema.load(request.args)
        except ValidationError as err:
            logger.warning("Quality assessment filter validation failed", errors=err.messages)
            return jsonify({"error": "Invalid filter parameters", "details": err.messages}), 400
        
        # Build base query
        query = db.session.query(Word)
        
        # Apply filters
        if filter_args.get('language_code'):
            query = query.filter(Word.language_code == filter_args['language_code'])
            
        if filter_args.get('pos'):
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == filter_args['pos']
            ).group_by(Word.id)
            
        if filter_args.get('min_completeness') is not None:
            query = query.filter(Word.completeness_score >= filter_args['min_completeness'])
            
        if filter_args.get('max_completeness') is not None:
            query = query.filter(Word.completeness_score <= filter_args['max_completeness'])
            
        if filter_args.get('created_after'):
            query = query.filter(Word.created_at >= filter_args['created_after'])
            
        if filter_args.get('created_before'):
            query = query.filter(Word.created_at <= filter_args['created_before'])
            
        if filter_args.get('updated_after'):
            query = query.filter(Word.updated_at >= filter_args['updated_after'])
            
        if filter_args.get('updated_before'):
            query = query.filter(Word.updated_at <= filter_args['updated_before'])
        
        # Get total count
        total_count = query.count()
        
        # Initialize results structure with empty data
        results = {
            "total_entries": total_count,
            "filters_applied": {k: v for k, v in filter_args.items() if v is not None},
            "verification_status": {
                "verified": 0,
                "unverified": 0,
                "percent_verified": 0.0
            },
            "completeness": {
                "average_score": 0.0,
                "distribution": {
                    "excellent": 0,
                    "good": 0,
                    "fair": 0,
                    "poor": 0,
                    "incomplete": 0
                },
                "percent_by_category": {
                    "excellent": 0.0,
                    "good": 0.0,
                    "fair": 0.0,
                    "poor": 0.0,
                    "incomplete": 0.0
                }
            },
            "language_distribution": {},
            "pos_distribution": {},
            "relations": {
                "with_relations": 0,
                "without_relations": 0,
                "average_relations_per_word": 0.0
            },
            "timestamps": {
                "created": {
                    "oldest": None,
                    "newest": None,
                    "distribution_by_year": {}
                },
                "updated": {
                    "oldest": None,
                    "newest": None,
                    "distribution_by_month": {}
                }
            },
            "components": {
                "with_definitions": 0,
                "without_definitions": 0,
                "with_etymology": 0,
                "without_etymology": 0,
                "with_pronunciations": 0,
                "without_pronunciations": 0,
                "with_baybayin": 0,
                "without_baybayin": 0
            },
            "issues": [],
            "issue_counts": {
                "total": 0,
                "by_severity": {
                    "critical": 0,
                    "warning": 0,
                    "info": 0
                }
            },
            "assessment_date": datetime.now(timezone.utc).isoformat(),
            "execution_time_ms": 0
        }
        
        # If no entries found, return early with empty results
        if total_count == 0:
            return jsonify(results)
        
        # --- Aggregation queries for statistics ---
        
        # Verification status - use has_etymology as a proxy for verification
        verification_counts = db.session.query(
            exists().where(Etymology.word_id == Word.id).label('has_etymology'),
            func.count(Word.id)
        ).outerjoin(Etymology).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).group_by('has_etymology').all()
        
        verified_count = 0
        for has_etymology, count in verification_counts:
            if has_etymology:
                verified_count = count
                results["verification_status"]["verified"] = count
            else:
                results["verification_status"]["unverified"] = count
                
        results["verification_status"]["percent_verified"] = round((verified_count / total_count) * 100, 1) if total_count > 0 else 0.0
        
        # Completeness statistics
        # Since completeness_score is a hybrid property, we need to get actual Word objects
        # and compute the average in Python rather than SQL
        words_sample = query.limit(1000).all()  # Limit to avoid performance issues
        
        if words_sample:
            # Calculate average completeness score
            total_score = sum(getattr(word, 'completeness_score', 0) for word in words_sample)
            avg_score = total_score / len(words_sample) if words_sample else 0
            results["completeness"]["average_score"] = round(float(avg_score), 2)
            
            # Calculate completeness distribution
            distribution = {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0,
                "incomplete": 0
            }
            
            for word in words_sample:
                score = getattr(word, 'completeness_score', 0)
                if score >= 0.9:
                    distribution["excellent"] += 1
                elif score >= 0.7:
                    distribution["good"] += 1
                elif score >= 0.5:
                    distribution["fair"] += 1
                elif score >= 0.3:
                    distribution["poor"] += 1
                else:
                    distribution["incomplete"] += 1
            
            # Update the distribution counts
            results["completeness"]["distribution"] = distribution
            
            # Calculate percentages
            for category, count in distribution.items():
                results["completeness"]["percent_by_category"][category] = round((count / len(words_sample)) * 100, 1)
        else:
            # If no words were sampled, leave default values
            results["completeness"]["average_score"] = 0.0
        
        # Language distribution
        language_distribution = db.session.query(
            Word.language_code, func.count(Word.id)
        ).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).group_by(Word.language_code).all()
        
        for language, count in language_distribution:
            results["language_distribution"][language] = count
        
        # Part of speech distribution
        pos_distribution = db.session.query(
            PartOfSpeech.code, func.count(distinct(Word.id))
        ).join(
            Definition, Definition.standardized_pos_id == PartOfSpeech.id
        ).join(
            Word, Word.id == Definition.word_id
        ).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).group_by(PartOfSpeech.code).all()
        
        for pos, count in pos_distribution:
            results["pos_distribution"][pos] = count
        
        # Relations statistics
        words_with_relations = db.session.query(func.count(distinct(Word.id))).filter(
            and_(
                Word.id.in_(query.with_entities(Word.id)),
                or_(
                    exists().where(Relation.from_word_id == Word.id),
                    exists().where(Relation.to_word_id == Word.id)
                )
            )
        ).scalar() or 0
        
        results["relations"]["with_relations"] = words_with_relations
        results["relations"]["without_relations"] = total_count - words_with_relations
        
        # Average relations per word
        total_relations = db.session.query(func.count(Relation.id)).filter(
            or_(
                Relation.from_word_id.in_(query.with_entities(Word.id)),
                Relation.to_word_id.in_(query.with_entities(Word.id))
            )
        ).scalar() or 0
        
        results["relations"]["average_relations_per_word"] = round(total_relations / total_count, 2) if total_count > 0 else 0.0
        
        # Timestamp statistics
        oldest_created = db.session.query(func.min(Word.created_at)).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).scalar()
        
        newest_created = db.session.query(func.max(Word.created_at)).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).scalar()
        
        results["timestamps"]["created"]["oldest"] = oldest_created.isoformat() if oldest_created else None
        results["timestamps"]["created"]["newest"] = newest_created.isoformat() if newest_created else None
        
        # Year distribution
        if oldest_created and newest_created:
            years = [dt.year for dt in [oldest_created, newest_created]]
            min_year, max_year = min(years), max(years)
            
            year_distribution = db.session.query(
                extract('year', Word.created_at).label('year'), func.count(Word.id)
            ).filter(
                Word.id.in_(query.with_entities(Word.id))
            ).group_by('year').all()
            
            for year, count in year_distribution:
                results["timestamps"]["created"]["distribution_by_year"][str(int(year))] = count
        
        # Components statistics
        components_stats = {}
        
        # Words with definitions
        words_with_definitions = db.session.query(func.count(distinct(Word.id))).filter(
            and_(
                Word.id.in_(query.with_entities(Word.id)),
                exists().where(Definition.word_id == Word.id)
            )
        ).scalar() or 0
        
        results["components"]["with_definitions"] = words_with_definitions
        results["components"]["without_definitions"] = total_count - words_with_definitions
        
        # Words with etymology
        words_with_etymology = db.session.query(func.count(distinct(Word.id))).filter(
            and_(
                Word.id.in_(query.with_entities(Word.id)),
                exists().where(Etymology.word_id == Word.id)
            )
        ).scalar() or 0
        
        results["components"]["with_etymology"] = words_with_etymology
        results["components"]["without_etymology"] = total_count - words_with_etymology
        
        # Words with pronunciations
        words_with_pronunciations = db.session.query(func.count(distinct(Word.id))).filter(
            and_(
                Word.id.in_(query.with_entities(Word.id)),
                exists().where(Pronunciation.word_id == Word.id)
            )
        ).scalar() or 0
        
        results["components"]["with_pronunciations"] = words_with_pronunciations
        results["components"]["without_pronunciations"] = total_count - words_with_pronunciations
        
        # Words with baybayin
        words_with_baybayin = db.session.query(func.count(distinct(Word.id))).filter(
            and_(
                Word.id.in_(query.with_entities(Word.id)),
                Word.has_baybayin == True,
                Word.baybayin_form.isnot(None)
            )
        ).scalar() or 0
        
        results["components"]["with_baybayin"] = words_with_baybayin
        results["components"]["without_baybayin"] = total_count - words_with_baybayin
        
        # --- Identify Issues ---
        if filter_args.get('include_issues', True):
            issues = []
            
            # Words without definitions
            if results["components"]["without_definitions"] > 0:
                issue_severity = "critical" if results["components"]["without_definitions"] / total_count > 0.1 else "warning"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "missing_definitions",
                        "severity": issue_severity,
                        "count": results["components"]["without_definitions"],
                        "message": f"{results['components']['without_definitions']} words have no definitions",
                        "suggestions": ["Review words without definitions", "Add missing definitions"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Words without etymology
            if results["components"]["without_etymology"] > 0:
                issue_severity = "warning"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "missing_etymology",
                        "severity": issue_severity,
                        "count": results["components"]["without_etymology"],
                        "message": f"{results['components']['without_etymology']} words have no etymology information",
                        "suggestions": ["Add etymology information to key words", "Focus on high-frequency words first"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Words without relations
            if results["relations"]["without_relations"] > 0:
                issue_severity = "info"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "missing_relations",
                        "severity": issue_severity,
                        "count": results["relations"]["without_relations"],
                        "message": f"{results['relations']['without_relations']} words have no relationships to other words",
                        "suggestions": ["Add synonyms and antonyms", "Link related words"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Low completeness scores
            low_completeness = results["completeness"]["distribution"]["poor"] + results["completeness"]["distribution"]["incomplete"]
            if low_completeness > 0:
                issue_severity = "warning" if low_completeness / total_count > 0.3 else "info"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "low_completeness",
                        "severity": issue_severity,
                        "count": low_completeness,
                        "message": f"{low_completeness} words have low completeness scores",
                        "suggestions": ["Focus on improving low completeness words", "Add missing components systematically"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Unverified words
            if results["verification_status"]["unverified"] > 0:
                issue_severity = "warning" if results["verification_status"]["unverified"] / total_count > 0.5 else "info"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "unverified_words",
                        "severity": issue_severity,
                        "count": results["verification_status"]["unverified"],
                        "message": f"{results['verification_status']['unverified']} words are not verified",
                        "suggestions": ["Implement verification process", "Prioritize high-frequency words for verification"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Words with no pronunciations
            if results["components"]["without_pronunciations"] > 0:
                issue_severity = "info"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "missing_pronunciations",
                        "severity": issue_severity,
                        "count": results["components"]["without_pronunciations"],
                        "message": f"{results['components']['without_pronunciations']} words have no pronunciation data",
                        "suggestions": ["Add IPA transcriptions", "Record audio pronunciations"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Get detailed issue examples if requested (limit results for performance)
            max_results = min(filter_args.get('max_results', 100), 1000)
            
            # Find problematic words with low completeness scores or missing components
            problematic_words = query.with_entities(
                Word.id, Word.lemma, Word.language_code, Word.completeness_score
            ).filter(
                or_(
                    Word.completeness_score < 0.3,
                    ~exists().where(Definition.word_id == Word.id)
                )
            ).order_by(Word.completeness_score).limit(max_results).all()
            
            if problematic_words:
                issue_severity = "critical"
                if filter_args.get('issue_severity') == 'all' or filter_args.get('issue_severity') == issue_severity:
                    issues.append({
                        "issue_type": "problematic_words",
                        "severity": issue_severity,
                        "count": len(problematic_words),
                        "message": f"Found {len(problematic_words)} problematic words with serious quality issues",
                        "examples": [
                            {
                                "id": w.id,
                                "lemma": w.lemma,
                                "language_code": w.language_code,
                                "completeness_score": w.completeness_score
                            } for w in problematic_words[:10]  # Limit examples to 10
                        ],
                        "suggestions": ["Focus on these words first", "Review automatically for potential errors"]
                    })
                    results["issue_counts"]["by_severity"][issue_severity] += 1
            
            # Add issues to results
            results["issues"] = sorted(issues, key=lambda x: {
                "critical": 0,
                "warning": 1,
                "info": 2
            }.get(x["severity"], 3))
            
            # Update total issue count
            results["issue_counts"]["total"] = sum(results["issue_counts"]["by_severity"].values())
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results["execution_time_ms"] = round(execution_time * 1000, 2)
        
        # Record request latency
        REQUEST_LATENCY.labels(endpoint="quality_assessment").observe(execution_time)
        
        return jsonify(results)
        
    except SQLAlchemyError as e:
        logger.error(f"Database error during quality assessment: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="quality_assessment", error_type="database_error").inc()
        return jsonify({"error": "Database error during quality assessment", "details": str(e)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error during quality assessment: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="quality_assessment", error_type="unexpected_error").inc()
        return jsonify({"error": "Unexpected error during quality assessment", "details": str(e)}), 500

@bp.route("/bulk_operations", methods=["POST"])
@limiter.limit("20 per hour", key_func=get_remote_address)  # Rate limit bulk operations
def bulk_operations():
    """
    Perform bulk operations on words such as verification, tagging, or categorization.
    This endpoint is designed for dictionary administrators to manage multiple words efficiently.
    """
    try:
        # Track metrics
        API_REQUESTS.labels(endpoint="bulk_operations", method="POST").inc()
        start_time = time.time()
        
        # Check authorization (implement proper auth middleware)
        # This is a placeholder - implement actual auth
        if not request.headers.get('X-Api-Key') and 'Authorization' not in request.headers:
            return jsonify({
                "error": "Authorization required",
                "message": "You must provide an API key to perform bulk operations"
            }), 401
            
        # Parse JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Required fields
        if 'operation' not in data:
            return jsonify({"error": "Operation field is required"}), 400
            
        if 'word_ids' not in data and 'filters' not in data:
            return jsonify({"error": "Either word_ids or filters must be provided"}), 400
            
        # Get the list of word IDs to operate on
        word_ids = []
        
        # If word_ids directly provided
        if 'word_ids' in data and isinstance(data['word_ids'], list):
            word_ids = data['word_ids']
            
        # If filters provided, build query to get word IDs
        elif 'filters' in data and isinstance(data['filters'], dict):
            filters = data['filters']
            query = Word.query.with_entities(Word.id)
            
            # Apply filters
            if 'language_code' in filters:
                query = query.filter(Word.language_code == filters['language_code'])
                
            if 'min_completeness' in filters:
                query = query.filter(Word.completeness_score >= filters['min_completeness'])
                
            if 'max_completeness' in filters:
                query = query.filter(Word.completeness_score <= filters['max_completeness'])
                
            if 'has_baybayin' in filters:
                query = query.filter(Word.has_baybayin == filters['has_baybayin'])
                
            if 'pos' in filters:
                query = query.join(Word.definitions).join(Definition.standardized_pos).filter(
                    PartOfSpeech.code == filters['pos']
                )
                
            if 'has_etymology' in filters:
                query = query.join(Word.etymologies).group_by(Word.id)
                
            if 'has_pronunciation' in filters:
                query = query.join(Word.pronunciations).group_by(Word.id)
                
            # Get word IDs from query
            word_ids = [row[0] for row in query.all()]
            
        # Validate we have word IDs to operate on
        if not word_ids:
            return jsonify({
                "error": "No words match the provided criteria",
                "word_count": 0
            }), 404
            
        # Limit the number of words that can be operated on at once
        max_words = 1000
        if len(word_ids) > max_words:
            return jsonify({
                "error": f"Too many words selected. Maximum is {max_words}, you selected {len(word_ids)}",
                "word_count": len(word_ids)
            }), 400
            
        # Execute the requested operation
        operation = data['operation']
        results = {
            "operation": operation,
            "word_count": len(word_ids),
            "success_count": 0,
            "failure_count": 0,
            "errors": []
        }
        
        # Check for dry run mode
        dry_run = data.get('dry_run', False)
        
        if not dry_run:
            # Begin transaction
            try:
                # Different operations available
                if operation == 'verify':
                    # Verify words - mark them as verified
                    verify_status = data.get('verify_status', True)
                    verification_notes = data.get('verification_notes', '')
                    
                    for word_id in word_ids:
                        try:
                            word = Word.query.get(word_id)
                            if word:
                                word.verified = verify_status
                                word.verification_date = datetime.now(timezone.utc) if verify_status else None
                                word.verification_notes = verification_notes
                                db.session.add(word)
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} not found")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error processing word ID {word_id}: {str(e)}")
                
                elif operation == 'add_tags':
                    # Add tags to words
                    if 'tags' not in data or not isinstance(data['tags'], dict):
                        return jsonify({"error": "Tags field is required and must be a dictionary"}), 400
                        
                    tags = data['tags']
                    
                    for word_id in word_ids:
                        try:
                            word = Word.query.get(word_id)
                            if word:
                                # If word has no existing tags, initialize as empty dict
                                current_tags = {}
                                if word.tags:
                                    try:
                                        if isinstance(word.tags, str):
                                            current_tags = json.loads(word.tags)
                                        elif isinstance(word.tags, dict):
                                            current_tags = word.tags
                                    except:
                                        current_tags = {}
                                
                                # Update tags
                                current_tags.update(tags)
                                word.tags = current_tags
                                db.session.add(word)
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} not found")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error processing word ID {word_id}: {str(e)}")
                
                elif operation == 'remove_tags':
                    # Remove tags from words
                    if 'tag_keys' not in data or not isinstance(data['tag_keys'], list):
                        return jsonify({"error": "tag_keys field is required and must be a list"}), 400
                        
                    tag_keys = data['tag_keys']
                    
                    for word_id in word_ids:
                        try:
                            word = Word.query.get(word_id)
                            if word and word.tags:
                                current_tags = {}
                                try:
                                    if isinstance(word.tags, str):
                                        current_tags = json.loads(word.tags)
                                    elif isinstance(word.tags, dict):
                                        current_tags = word.tags
                                except:
                                    current_tags = {}
                                
                                # Remove specified tags
                                for key in tag_keys:
                                    if key in current_tags:
                                        del current_tags[key]
                                
                                word.tags = current_tags
                                db.session.add(word)
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} not found or has no tags")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error processing word ID {word_id}: {str(e)}")
                
                elif operation == 'update_metadata':
                    # Update metadata fields
                    if 'metadata' not in data or not isinstance(data['metadata'], dict):
                        return jsonify({"error": "metadata field is required and must be a dictionary"}), 400
                        
                    metadata = data['metadata']
                    
                    for word_id in word_ids:
                        try:
                            word = Word.query.get(word_id)
                            if word:
                                # If word has no existing metadata, initialize as empty dict
                                current_metadata = {}
                                if hasattr(word, 'word_metadata') and word.word_metadata:
                                    try:
                                        if isinstance(word.word_metadata, str):
                                            current_metadata = json.loads(word.word_metadata)
                                        elif isinstance(word.word_metadata, dict):
                                            current_metadata = word.word_metadata
                                    except:
                                        current_metadata = {}
                                
                                # Update metadata
                                current_metadata.update(metadata)
                                word.word_metadata = current_metadata
                                db.session.add(word)
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} not found")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error processing word ID {word_id}: {str(e)}")
                
                elif operation == 'add_category':
                    # Add category to definitions
                    if 'category_data' not in data or not isinstance(data['category_data'], dict):
                        return jsonify({"error": "category_data field is required and must be a dictionary"}), 400
                        
                    category_data = data['category_data']
                    category_name = category_data.get('category_name')
                    if not category_name:
                        return jsonify({"error": "category_name is required in category_data"}), 400
                    
                    # Process words
                    for word_id in word_ids:
                        try:
                            # Get the word's definitions
                            definitions = Definition.query.filter(Definition.word_id == word_id).all()
                            
                            if definitions:
                                for definition in definitions:
                                    # Check if category already exists
                                    existing_category = DefinitionCategory.query.filter(
                                        DefinitionCategory.definition_id == definition.id,
                                        DefinitionCategory.category_name == category_name
                                    ).first()
                                    
                                    if not existing_category:
                                        # Create new category
                                        new_category = DefinitionCategory(
                                            definition_id=definition.id,
                                            category_name=category_name,
                                            description=category_data.get('description', ''),
                                            tags=category_data.get('tags', {}),
                                            category_metadata=category_data.get('category_metadata', {})
                                        )
                                        db.session.add(new_category)
                                
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} has no definitions")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error processing word ID {word_id}: {str(e)}")
                
                elif operation == 'delete':
                    # Delete words (use with caution!)
                    if not data.get('confirm_delete', False):
                        return jsonify({
                            "error": "Delete operation requires confirmation",
                            "message": "Set confirm_delete to true to proceed with deletion"
                        }), 400
                    
                    for word_id in word_ids:
                        try:
                            word = Word.query.get(word_id)
                            if word:
                                db.session.delete(word)
                                results['success_count'] += 1
                            else:
                                results['failure_count'] += 1
                                results['errors'].append(f"Word with ID {word_id} not found")
                        except Exception as e:
                            results['failure_count'] += 1
                            results['errors'].append(f"Error deleting word ID {word_id}: {str(e)}")
                
                else:
                    return jsonify({
                        "error": "Unknown operation",
                        "message": f"Operation '{operation}' is not supported"
                    }), 400
                
                # Commit transaction if not in dry run mode
                db.session.commit()
                
            except SQLAlchemyError as e:
                # Roll back on error
                db.session.rollback()
                logger.error(f"Database error during bulk operation: {str(e)}")
                return jsonify({
                    "error": "Database error during bulk operation",
                    "details": str(e)
                }), 500
        else:
            # Dry run mode - just report what would be done
            results["word_ids"] = word_ids[:100]  # Limit to first 100 for response size
            results["dry_run"] = True
            if len(word_ids) > 100:
                results["note"] = "Only showing first 100 word IDs in response"
        
        # Record request latency
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="bulk_operations").observe(request_time)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Unexpected error during bulk operation: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="bulk_operations", error_type="unexpected_error").inc()
        return jsonify({"error": "Unexpected error during bulk operation", "details": str(e)}), 500

@bp.route("/parts_of_speech", methods=["GET"])
def get_parts_of_speech():
    """Get all parts of speech."""
    try:
        # Use direct SQL to get parts of speech
        sql = """
        SELECT id, code, name_en, name_tl, description
        FROM parts_of_speech
        ORDER BY id
        """
        
        results = db.session.execute(text(sql)).fetchall()
        
        # Format the response
        formatted_pos = []
        for pos in results:
            formatted_pos.append({
                "id": pos.id,
                "code": pos.code,
                "name": pos.name_en,  # Frontend may expect 'name' instead of 'name_en'
                "name_en": pos.name_en,
                "name_tl": pos.name_tl,
                "description": pos.description or "",
                "count": 0  # Frontend might expect a count field
            })
        
        return jsonify(formatted_pos)
    except Exception as e:
        logger.error(f"Error retrieving parts of speech: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics summary."""
    try:
        # Use simple SQL queries to get statistics
        stats_queries = {
            "total_words": "SELECT COUNT(*) FROM words",
            "total_definitions": "SELECT COUNT(*) FROM definitions",
            "total_etymologies": "SELECT COUNT(*) FROM etymologies",
            "total_relations": "SELECT COUNT(*) FROM relations",
            "words_with_baybayin": "SELECT COUNT(*) FROM words WHERE has_baybayin = true",
            "languages": "SELECT language_code, COUNT(*) FROM words GROUP BY language_code",
            "pos": "SELECT standardized_pos_id, COUNT(*) FROM definitions GROUP BY standardized_pos_id"
        }
        
        # Execute each query and build the result
        result = {
            "timestamp": datetime.now().isoformat()
        }
        
        # Get total counts
        for key, query in stats_queries.items():
            if key not in ["languages", "pos"]:
                count = db.session.execute(text(query)).scalar() or 0
                result[key] = count
        
        # Get languages distribution
        languages = {}
        lang_results = db.session.execute(text(stats_queries["languages"])).fetchall()
        for lang in lang_results:
            if lang[0]:  # Ensure language code is not None
                languages[lang[0]] = lang[1]
        result["languages"] = languages
        
        # Get parts of speech distribution
        pos_counts = {}
        pos_results = db.session.execute(text(stats_queries["pos"])).fetchall()
        
        # Get POS codes by ID
        pos_codes = {}
        pos_query = "SELECT id, code FROM parts_of_speech"
        pos_data = db.session.execute(text(pos_query)).fetchall()
        for p in pos_data:
            pos_codes[p.id] = p.code
            
        # Map POS IDs to codes
        for pos in pos_results:
            pos_id = pos[0]
            count = pos[1]
            if pos_id in pos_codes:
                pos_counts[pos_codes[pos_id]] = count
        
        result["parts_of_speech"] = pos_counts
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

# Helper function to fetch word details with specific relationships
def _fetch_word_details(word_id,
                        include_definitions=True,
                        include_etymologies=True,
                        include_pronunciations=True,
                        include_credits=True,
                        include_relations=True,
                        include_affixations=True,
                        include_root=True,
                        include_derived=True,
                        include_forms=True,
                        include_templates=True,
                        include_definition_relations=False):
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
                
                # If we need definition relations, load those as well
                if include_definition_relations:
                    word_query = word_query.options(
                        selectinload(Word.definitions).selectinload(Definition.definition_relations)
                    )
            
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
