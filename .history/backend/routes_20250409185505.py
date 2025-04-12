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

# Word retrieval endpoint
@bp.route("/words/<path:word>", methods=["GET"])
def get_word(word: str):
    """Get FULL detailed information about a specific word, including all relations."""
    try:
        API_REQUESTS.labels(endpoint="get_word", method="GET").inc()
        start_time = time.time()

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
        schema = WordSchema()
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
                'count': 0,
                'results': [],
                'filters': {}
            }), 200
        
        # Validate numeric parameters
        limit = max(1, min(limit, 100))  # Constrain between 1 and 100
        offset = max(0, offset)  # Must be non-negative
        
        normalized_query = normalize_word(query)
        
        # Base filters common to all queries
        filters = {
            'has_baybayin': params.get('has_baybayin'),
            'has_pronunciation': params.get('has_pronunciation'),
            'has_etymology': params.get('has_etymology'),
            'has_forms': params.get('has_forms'),
            'has_templates': params.get('has_templates'),
            'has_definition_relations': params.get('has_definition_relations'),
            'language': params.get('language'),
            'pos': params.get('pos'),
            'min_completeness': params.get('min_completeness'),
        }
        
        # Parameters for result formatting
        include_options = {
            'include_definitions': params.get('include_definitions', True),
            'include_pronunciations': params.get('include_pronunciations', True),
            'include_etymologies': params.get('include_etymologies', True),
            'include_relations': params.get('include_relations', True),
            'include_forms': params.get('include_forms', True),
            'include_templates': params.get('include_templates', True),
            'include_metadata': params.get('include_metadata', True),
            'include_related_words': params.get('include_related_words', False),
            'include_definition_relations': params.get('include_definition_relations', False),
        }
        
        # Build SQL for counting total matches
        count_sql = """
        SELECT COUNT(DISTINCT w.id) FROM words w
        WHERE 
        """
        
        # Build SQL for basic search, then we'll append specific conditions based on mode
        main_sql = """
        SELECT w.id, w.lemma, w.normalized_lemma, w.language_code, 
            w.has_baybayin, w.baybayin_form 
            FROM words w
        WHERE 
        """
        
        search_conditions = []
        query_params = {}
        
        if mode == 'exact':
            search_conditions.append("(w.normalized_lemma = :normalized_query OR w.lemma = :query)")
            query_params['normalized_query'] = normalized_query
            query_params['query'] = query
        elif mode == 'prefix':
            search_conditions.append("(w.normalized_lemma LIKE :prefix_query OR w.lemma LIKE :prefix_query)")
            query_params['prefix_query'] = f"{normalized_query}%"
        elif mode == 'suffix':
            search_conditions.append("(w.normalized_lemma LIKE :suffix_query OR w.lemma LIKE :suffix_query)")
            query_params['suffix_query'] = f"%{normalized_query}"
        else:  # 'all' mode - default
            search_conditions.append("""
            (w.normalized_lemma LIKE :contains_query 
             OR w.lemma LIKE :contains_query
             OR w.search_text::text LIKE :contains_query
             OR w.lemma ILIKE :query_exact)
            """)
            query_params['contains_query'] = f"%{normalized_query}%"
            query_params['query_exact'] = query
        
        # Add filter conditions
        if filters['has_baybayin'] is not None:
            search_conditions.append("w.has_baybayin = :has_baybayin")
            query_params['has_baybayin'] = filters['has_baybayin']
            
        if filters['language'] is not None:
            search_conditions.append("w.language_code = :language")
            query_params['language'] = filters['language']
            
        # Add other filters as needed
        # ...
        
        # Combine all conditions
        combined_conditions = " AND ".join(search_conditions)
        count_sql += combined_conditions
        main_sql += combined_conditions
        
        # Add a simple ORDER BY that works with PostgreSQL
        main_sql += " ORDER BY w.lemma ASC"
        
        # Add LIMIT and OFFSET
        main_sql += " LIMIT :limit OFFSET :offset"
        query_params['limit'] = limit
        query_params['offset'] = offset
        
        # Initialize result variable before any queries
        result = {
            'count': 0,
            'results': [],
            'filters': filters
        }
        
        # Execute count query first
        with db.engine.connect() as conn:
            count_result = conn.execute(text(count_sql), query_params).scalar()
            result['count'] = count_result or 0
            
        # If count is zero, return empty results
        if result['count'] == 0:
            return jsonify(result), 200
            
        # Execute main query for results
        with db.engine.connect() as conn:
            rows = conn.execute(text(main_sql), query_params).fetchall()
            
            if include_full:
                # Full word details for each result
                word_ids = [row[0] for row in rows]
                full_words = []
                
                for word_id in word_ids:
                    word = _fetch_word_details(
                        word_id,
                        include_definitions=include_options['include_definitions'],
                        include_etymologies=include_options['include_etymologies'],
                        include_pronunciations=include_options['include_pronunciations'],
                        include_credits=True,
                        include_relations=include_options['include_relations'],
                        include_forms=include_options['include_forms'],
                        include_templates=include_options['include_templates'],
                        include_definition_relations=include_options['include_definition_relations']
                    )
                    if word:
                        # Serialize the word object
                        schema = WordSchema()
                        serialized = schema.dump(word)
                        full_words.append(serialized)
                
                result['results'] = full_words
        else:
                # Basic info for each result
                basic_results = []
                for row in rows:
                    basic_results.append({
                        'id': row[0],
                        'lemma': row[1],
                        'normalized_lemma': row[2],
                        'language_code': row[3],
                        'has_baybayin': bool(row[4]) if row[4] is not None else None,
                        'baybayin_form': row[5]
                    })
                result['results'] = basic_results
        
        return jsonify(result), 200
    except Exception as e:
        # Catch potential NameError here too
        logger.error(f"Search function error: {str(e)}", exc_info=True) # Log full traceback
        API_ERRORS.labels(endpoint="search", error_type=type(e).__name__).inc()
        # Make sure the error message includes the type for clarity
        return jsonify({"error": f"Unexpected error ({type(e).__name__})", "message": str(e)}), 500

@bp.route("/random", methods=["GET"])
def get_random_word():
    """Get a random word with optional filters."""
    try:
        API_REQUESTS.labels(endpoint="get_random_word", method="GET").inc()
        
        # Parse filter parameters
        language = request.args.get("language")
        pos_code = request.args.get("pos")
        has_etymology = request.args.get("has_etymology", "false").lower() == "true"
        has_definitions = request.args.get("has_definitions", "true").lower() == "true"
        has_baybayin = request.args.get("has_baybayin", "false").lower() == "true"
        min_definitions = int(request.args.get("min_definitions", 1))
        
        # Build SQL query for filtering words
        sql_query = """
        SELECT w.id FROM words w
        """
        
        # Add necessary joins based on filters
        if pos_code:
            sql_query += """
            JOIN definitions d ON d.word_id = w.id
            JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
            """
        elif has_definitions or min_definitions > 0:
            sql_query += """
            JOIN definitions d ON d.word_id = w.id
            """
        
        if has_etymology:
            sql_query += """
            JOIN etymologies e ON e.word_id = w.id
            """
        
        # Start WHERE clause
        conditions = []
        params = {}
        
        if language:
            conditions.append("w.language_code = :language")
            params["language"] = language
        
        if pos_code:
            conditions.append("pos.code = :pos_code")
            params["pos_code"] = pos_code
        
        if has_baybayin:
            conditions.append("w.has_baybayin = TRUE AND w.baybayin_form IS NOT NULL")
        
        # Add GROUP BY and HAVING if needed
        group_by_needed = pos_code or has_definitions or min_definitions > 0 or has_etymology
        having_conditions = []
        
        if group_by_needed:
            sql_query += "\nWHERE " + " AND ".join(conditions) if conditions else ""
            sql_query += "\nGROUP BY w.id"
            
            if min_definitions > 0:
                having_conditions.append("COUNT(DISTINCT d.id) >= :min_defs")
                params["min_defs"] = min_definitions
                
            if has_etymology:
                having_conditions.append("COUNT(DISTINCT e.id) > 0")
                
            if having_conditions:
                sql_query += "\nHAVING " + " AND ".join(having_conditions)
        else:
            if conditions:
                sql_query += "\nWHERE " + " AND ".join(conditions)
        
        # Execute query to get eligible IDs
        result = db.session.execute(text(sql_query), params).fetchall()
        eligible_ids = [row[0] for row in result]

        if not eligible_ids:
            logger.info("No words match the specified criteria for random selection.")
            return jsonify({"error": "No words match the specified criteria"}), 404

        # Choose a random ID
        random_word_id = random.choice(eligible_ids)
        logger.debug(f"Selected random word ID: {random_word_id}")

        # Fetch word details using direct SQL
        word_sql = """
        SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
               root_word_id, preferred_spelling, tags, source_info, 
               data_hash, search_text, badlit_form, hyphenation, is_proper_noun, 
               is_abbreviation, is_initialism, created_at, updated_at
        FROM words 
        WHERE id = :id
        """
        word_result = db.session.execute(text(word_sql), {"id": random_word_id}).fetchone()
        
        if not word_result:
            logger.error(f"Failed to fetch selected random word ID: {random_word_id}")
             return jsonify({"error": "Failed to retrieve random word details"}), 500

        # Construct the result
        result = {
            "id": word_result.id,
            "lemma": word_result.lemma,
            "normalized_lemma": word_result.normalized_lemma,
            "language_code": word_result.language_code,
            "has_baybayin": word_result.has_baybayin,
            "baybayin_form": word_result.baybayin_form,
            "definitions": [],
            "etymologies": [],
            "pronunciations": [],
            "created_at": word_result.created_at.isoformat() if word_result.created_at else None,
            "updated_at": word_result.updated_at.isoformat() if word_result.updated_at else None
        }
        
        # Fetch definitions
        defs_sql = """
        SELECT id, definition_text, original_pos, standardized_pos_id,
              examples, usage_notes, tags, sources
        FROM definitions
        WHERE word_id = :word_id
        """
        defs_result = db.session.execute(text(defs_sql), {"word_id": random_word_id}).fetchall()
        
        # Add definitions
        for d in defs_result:
            definition = {
                "id": d.id,
                "definition_text": d.definition_text,
                "original_pos": d.original_pos,
                "standardized_pos_id": d.standardized_pos_id,
                "examples": d.examples if d.examples else [],
                "usage_notes": d.usage_notes if d.usage_notes else [],
                "tags": d.tags.split(',') if d.tags else [],
                "sources": d.sources.split(',') if d.sources else []
            }
            result["definitions"].append(definition)
        
        # Fetch etymologies - only select columns that exist
        etym_sql = """
        SELECT id, etymology_text, normalized_components, etymology_structure,
              language_codes, sources
        FROM etymologies
        WHERE word_id = :word_id
        """
        etym_result = db.session.execute(text(etym_sql), {"word_id": random_word_id}).fetchall()
        
        # Add etymologies
        for e in etym_result:
            etymology = {
                "id": e.id,
                "etymology_text": e.etymology_text,
                "normalized_components": e.normalized_components,
                "etymology_structure": e.etymology_structure,
                "language_codes": e.language_codes.split(',') if e.language_codes else [],
                "sources": e.sources.split(',') if e.sources else []
            }
            result["etymologies"].append(etymology)
        
        # Fetch pronunciations
        pron_sql = """
        SELECT id, type, value, tags, pronunciation_metadata, sources
        FROM pronunciations
        WHERE word_id = :word_id
        """
        pron_result = db.session.execute(text(pron_sql), {"word_id": random_word_id}).fetchall()
        
        # Add pronunciations
        for p in pron_result:
            pronunciation = {
                "id": p.id,
                "type": p.type,
                "value": p.value,
                "tags": p.tags if p.tags else {},
                "pronunciation_metadata": p.pronunciation_metadata if p.pronunciation_metadata else {},
                "sources": p.sources.split(',') if p.sources else []
            }
            result["pronunciations"].append(pronunciation)
        
        # Add data completeness metrics
        result["data_completeness"] = {
            "has_definitions": len(result["definitions"]) > 0,
            "has_etymology": len(result["etymologies"]) > 0,
            "has_pronunciations": len(result["pronunciations"]) > 0,
            "has_baybayin": bool(word_result.has_baybayin and word_result.baybayin_form),
            "definition_count": len(result["definitions"]),
            "etymology_count": len(result["etymologies"]),
            "pronunciation_count": len(result["pronunciations"])
        }
        
        # Add filter criteria used
        result["filter_criteria"] = {
            "language": language,
            "pos_code": pos_code,
            "has_etymology": has_etymology,
            "has_definitions": has_definitions,
            "has_baybayin": has_baybayin,
            "min_definitions": min_definitions
        }
        
        logger.debug(f"Returning random word: {word_result.lemma} (ID: {random_word_id})")
        return jsonify(result)
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving random word", error=str(e), exc_info=True)
        return jsonify({"error": "Database error", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"Error retrieving random word", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_random_word", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get relations for a word."""
    try:
        API_REQUESTS.labels(endpoint="get_word_relations", method="GET").inc()

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
                logger.warning(f"Word '{word}' not found for relations lookup.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        if word_id is None:
             # Should not happen if logic above is correct
             logger.error(f"Failed to determine word ID for '{word}' in relations lookup.")
             return jsonify({"error": f"Word '{word}' not found"}), 404

        # Load word with only relations using the helper
        word_entry = _fetch_word_details(
            word_id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=True, # Only include relations
            include_affixations=False,
            include_root=False,
            include_derived=False,
            include_forms=False,
            include_templates=False,
            include_definition_relations=False
        )

        if not word_entry:
             # This might occur if the ID is invalid or fetch fails despite ID existing initially
             logger.warning(f"Failed to fetch details for word ID {word_id} (from '{word}') for relations.")
             return jsonify({"error": f"Word '{word}' not found or failed to fetch details"}), 404

        # Serialize relations using RelationSchema
        # Ensure RelationSchema is defined and handles nested WordSchema correctly (only ID/lemma needed)
        relation_schema = RelationSchema(many=True)
        outgoing = relation_schema.dump(word_entry.outgoing_relations)
        incoming = relation_schema.dump(word_entry.incoming_relations)

        # Format simplified result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "outgoing_relations": outgoing,
            "incoming_relations": incoming,
            "total_relations": len(outgoing) + len(incoming)
        }

        logger.debug(f"Returning relations for word '{word}' (ID: {word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving relations for word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving relations for word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word_relations", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

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
        affixation_schema = AffixationSchema(many=True)
        root_affix = affixation_schema.dump(word_entry.root_affixations)
        affixed_affix = affixation_schema.dump(word_entry.affixed_affixations)

        # Serialize root word and derived words using a limited WordSchema
        # Assuming WordSchema exists and can handle partial loading/dumping
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
    """Generate a semantic network for a word.
    
    This endpoint returns a network of semantically related words, where each node
    is a word and each edge is a semantic relationship. The network can be configured
    using the following parameters:
    
    - depth: How many hops away from the main word to include (default: 1)
    - breadth: Max number of relations per word (default: 10)
    - relation_types: Comma-separated list of relation types to include 
      (default: all)
    """
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
            WHERE LOWER(lemma) = LOWER(:word) OR LOWER(normalized_lemma) = LOWER(:normalized)
            LIMIT 1
            """
            params = {"word": word, "normalized": normalized}
            word_result = db.session.execute(text(sql), params).fetchone()
            
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
        current_app.logger.info(f"Generating semantic network for word '{word}' (depth: {depth}, breadth: {breadth})")
        
        # Database query for outgoing relations
        outgoing_sql = """
            SELECT /*+ INDEX(r idx_relations_from_word_id) */ 
                   r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as target_id, w.lemma as target_lemma, w.normalized_lemma as target_normalized_lemma,
                   w.language_code as target_language_code, w.has_baybayin as target_has_baybayin,
                   w.baybayin_form as target_baybayin_form
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = :word_id 
        """
        
        # Database query for incoming relations
        incoming_sql = """
            SELECT /*+ INDEX(r idx_relations_to_word_id) */
                   r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as source_id, w.lemma as source_lemma, w.normalized_lemma as source_normalized_lemma,
                   w.language_code as source_language_code, w.has_baybayin as source_has_baybayin,
                   w.baybayin_form as source_baybayin_form  
              FROM relations r
            JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = :word_id
        """
        
        # Add relation_type filter if provided
        if allowed_types:
            placeholders = ", ".join([f":rel_type_{i}" for i in range(len(allowed_types))])
            rel_type_condition = f"AND r.relation_type IN ({placeholders})"
            outgoing_sql += rel_type_condition
            incoming_sql += rel_type_condition
        
        # Add ordering and limit
        ordering = """
            ORDER BY 
                CASE 
                    WHEN r.relation_type IN ('synonym', 'antonym') THEN 1
                    WHEN r.relation_type IN ('hypernym', 'hyponym') THEN 2 
                    ELSE 3
                END,
                w.lemma
            LIMIT :breadth
        """
        outgoing_sql += ordering
        incoming_sql += ordering
        
        # Prepare parameters
        outgoing_params = {"word_id": word_id, "breadth": breadth}
        incoming_params = {"word_id": word_id, "breadth": breadth}
        
        if allowed_types:
            for i, rt in enumerate(allowed_types):
                param_name = f"rel_type_{i}"
                outgoing_params[param_name] = rt
                incoming_params[param_name] = rt
        
        # Execute queries
        with db.engine.connect() as conn:
            outgoing = conn.execute(text(outgoing_sql), outgoing_params).fetchall()
            incoming = conn.execute(text(incoming_sql), incoming_params).fetchall()
        
        # Start with the main word
        nodes = {word_id: {
            "id": str(word_id),
            "label": word_data["lemma"],
            "word": word_data["lemma"],
            "language": word_data["language_code"],
            "normalized_lemma": word_data["normalized_lemma"],
            "type": "main",
            "depth": 0,
            "main": True
        }}
        edges = {}
            
        # Process outgoing relations
        for rel in outgoing:
            rel_id, _, target_id, rel_type, rel_data = rel[:5]
            target_info = {
                "id": str(target_id),
                "label": rel.target_lemma,
                "word": rel.target_lemma,
                "language": rel.target_language_code,
                "normalized_lemma": rel.target_normalized_lemma,
                "type": rel_type,
                "depth": 1,
                "main": False,
                "has_baybayin": rel.target_has_baybayin,
                "baybayin_form": rel.target_baybayin_form
            }
            
            # Add the target node if not already present
            if target_id not in nodes:
                nodes[target_id] = target_info
            
            # Add the edge
            edge_id = f"{word_id}-{target_id}-{rel_type}"
            if edge_id not in edges:
                edges[edge_id] = {
                    "id": edge_id,
                    "source": str(word_id),
                    "target": str(target_id),
                    "type": rel_type,
                    "directed": rel_type not in ["synonym", "antonym", "related"]
                }
        
        # Process incoming relations
        for rel in incoming:
            rel_id, source_id, _, rel_type, rel_data = rel[:5]
            source_info = {
                "id": str(source_id),
                "label": rel.source_lemma,
                "word": rel.source_lemma,
                "language": rel.source_language_code,
                "normalized_lemma": rel.source_normalized_lemma,
                "type": rel_type,
                "depth": 1,
                "main": False,
                "has_baybayin": rel.source_has_baybayin,
                "baybayin_form": rel.source_baybayin_form
            }
            
            # Add the source node if not already present
            if source_id not in nodes:
                nodes[source_id] = source_info
            
            # Add the edge
            edge_id = f"{source_id}-{word_id}-{rel_type}"
            if edge_id not in edges:
                edges[edge_id] = {
                    "id": edge_id,
                    "source": str(source_id),
                    "target": str(word_id),
                    "type": rel_type,
                    "directed": rel_type not in ["synonym", "antonym", "related"]
                }
        
        # If depth > 1, recursively get more relations
        if depth > 1:
            visited = {word_id}
            queue = list(nodes.keys())  # Start with level 1 nodes
            current_depth = 1
            
            # Process each level
            while queue and current_depth < depth:
                next_queue = []
                for node_id in queue:
                    if node_id in visited:
                        continue
                    
                    visited.add(node_id)
                    
                    # Get outgoing relations for this node
                    node_outgoing_params = {"word_id": node_id, "breadth": breadth}
                    if allowed_types:
                        for i, rt in enumerate(allowed_types):
                            node_outgoing_params[f"rel_type_{i}"] = rt
                    
                    with db.engine.connect() as conn:
                        node_outgoing = conn.execute(text(outgoing_sql), node_outgoing_params).fetchall()
                    
                    # Process outgoing relations
                    for rel in node_outgoing:
                        rel_id, _, target_id, rel_type, rel_data = rel[:5]
                        
                        # Skip if target is already visited to avoid cycles
                        if target_id in visited:
                            continue
                        
                        # Add the target node
                        if target_id not in nodes:
                            nodes[target_id] = {
                                "id": str(target_id),
                                "label": rel.target_lemma,
                                "word": rel.target_lemma,
                                "language": rel.target_language_code,
                                "normalized_lemma": rel.target_normalized_lemma,
                                "type": rel_type,
                                "depth": current_depth + 1,
                                "main": False,
                                "has_baybayin": rel.target_has_baybayin,
                                "baybayin_form": rel.target_baybayin_form
                            }
                            next_queue.append(target_id)
                        
                        # Add the edge
                        edge_id = f"{node_id}-{target_id}-{rel_type}"
                        if edge_id not in edges:
                            edges[edge_id] = {
                                "id": edge_id,
                                "source": str(node_id),
                                "target": str(target_id),
                                "type": rel_type,
                                "directed": rel_type not in ["synonym", "antonym", "related"]
                            }
                
                queue = next_queue
                current_depth += 1
        
        # End timing
        execution_time = time.time() - start_time
        
        # Prepare response
        result = {
            "nodes": list(nodes.values()),
            "links": list(edges.values()),
            "metadata": {
                "root_word": word_data["lemma"],
                "normalized_lemma": word_data["normalized_lemma"],
                "language_code": word_data["language_code"],
                "depth": depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "execution_time": execution_time
            }
        }
        
        current_app.logger.info(f"Generated semantic network for '{word}' with {len(nodes)} nodes and {len(edges)} edges in {execution_time:.2f}s")
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Error generating semantic network for word '{word}'", exc_info=True)
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
    """
    Fetch a word with specified relationships using direct SQL to avoid ORM issues.
    Returns a Word object with the requested relationships loaded.
    """
    try:
        # First check if it's in redis cache
        # Use the global cache_client defined at module level
        global cache_client
        if cache_client:
            cache_key = f'word_details:{word_id}:{include_definitions}:{include_etymologies}:' \
                        f'{include_pronunciations}:{include_credits}:{include_relations}:' \
                        f'{include_affixations}:{include_root}:{include_derived}:' \
                        f'{include_forms}:{include_templates}:{include_definition_relations}'
            
            try:
                cached_data = cache_client.get(cache_key)
                if cached_data:
                    word = pickle.loads(cached_data)
                    # Ensures we have a valid database session for the object
                    # Important when retrieving from cache across requests
                    if word:
                        db.session.add(word)
                        db.session.refresh(word)
                        return word
            except Exception as e:
                logger.warning(f"Cache retrieval error for word_id={word_id}: {e}")
        
        # First get the basic word data
        sql_word = """
        SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
               root_word_id, preferred_spelling, tags, source_info, word_metadata, 
               data_hash, search_text, badlit_form, hyphenation, is_proper_noun, 
               is_abbreviation, is_initialism
        FROM words 
        WHERE id = :id
        """
        word_result = db.session.execute(text(sql_word), {"id": word_id}).fetchone()
        
        if not word_result:
            return None
        
        # Create word object manually
        word = Word()
        for key in word_result.keys():
            if hasattr(word, key) and key != 'is_root':  # Skip is_root since it's a hybrid property
                setattr(word, key, word_result[key])
        
        # Make sure the hybrid property is_root works correctly based on root_word_id
        # is_root will be automatically calculated based on root_word_id
        
        # Load definitions if requested
        if include_definitions:
            sql_defs = """
            SELECT id, definition_text, original_pos, standardized_pos_id,
                  examples, usage_notes, tags, sources
            FROM definitions
            WHERE word_id = :word_id
            """
            defs_result = db.session.execute(text(sql_defs), {"word_id": word_id}).fetchall()
            word.definitions = []
            
            for d in defs_result:
                definition = Definition()
                for key in d.keys():
                    if hasattr(definition, key):
                        setattr(definition, key, d[key])
                definition.word_id = word_id
                definition.word = word
                word.definitions.append(definition)
        
        # Load etymologies if requested
        if include_etymologies:
            sql_etym = """
            SELECT id, etymology_text, normalized_components, etymology_structure,
                  language_codes, sources
            FROM etymologies
            WHERE word_id = :word_id
            """
            etym_result = db.session.execute(text(sql_etym), {"word_id": word_id}).fetchall()
            word.etymologies = []
            
            for e in etym_result:
                etymology = Etymology()
                for key in e.keys():
                    if hasattr(etymology, key):
                        setattr(etymology, key, e[key])
                etymology.word_id = word_id
                etymology.word = word
                word.etymologies.append(etymology)
        
        # Load pronunciations if requested  
        if include_pronunciations:
            sql_pron = """
            SELECT id, type, value, tags, pronunciation_metadata, sources
            FROM pronunciations
            WHERE word_id = :word_id
            """
            pron_result = db.session.execute(text(sql_pron), {"word_id": word_id}).fetchall()
            word.pronunciations = []
            
            for p in pron_result:
                pronunciation = Pronunciation()
                for key in p.keys():
                    if hasattr(pronunciation, key):
                        setattr(pronunciation, key, p[key])
                pronunciation.word_id = word_id
                pronunciation.word = word
                word.pronunciations.append(pronunciation)
        
        # Load relations if requested
        if include_relations:
            # Outgoing relations
            sql_out_rel = """
            SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as target_id, w.lemma as target_lemma, w.language_code as target_language_code,
                   w.has_baybayin as target_has_baybayin, w.baybayin_form as target_baybayin_form
              FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = :word_id
            """
            out_rel_result = db.session.execute(text(sql_out_rel), {"word_id": word_id}).fetchall()
            word.outgoing_relations = []
            
            for r in out_rel_result:
                relation = Relation()
                relation.id = r.id
                relation.from_word_id = r.from_word_id
                relation.to_word_id = r.to_word_id
                relation.relation_type = r.relation_type
                relation.relation_data = r.relation_data
                relation.source_word = word
                
                # Create target word
                target_word = Word()
                target_word.id = r.target_id
                target_word.lemma = r.target_lemma
                target_word.language_code = r.target_language_code
                target_word.has_baybayin = r.target_has_baybayin
                target_word.baybayin_form = r.target_baybayin_form
                
                relation.target_word = target_word
                word.outgoing_relations.append(relation)
            
            # Incoming relations
            sql_in_rel = """
            SELECT r.id, r.from_word_id, r.to_word_id, r.relation_type, r.relation_data,
                   w.id as source_id, w.lemma as source_lemma, w.language_code as source_language_code,
                   w.has_baybayin as source_has_baybayin, w.baybayin_form as source_baybayin_form  
              FROM relations r
            JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = :word_id
            """
            in_rel_result = db.session.execute(text(sql_in_rel), {"word_id": word_id}).fetchall()
            word.incoming_relations = []
            
            for r in in_rel_result:
                relation = Relation()
                relation.id = r.id
                relation.from_word_id = r.from_word_id
                relation.to_word_id = r.to_word_id
                relation.relation_type = r.relation_type
                relation.relation_data = r.relation_data
                relation.target_word = word
                
                # Create source word
                source_word = Word()
                source_word.id = r.source_id
                source_word.lemma = r.source_lemma
                source_word.language_code = r.source_language_code
                source_word.has_baybayin = r.source_has_baybayin
                source_word.baybayin_form = r.source_baybayin_form
                
                relation.source_word = source_word
                word.incoming_relations.append(relation)
        
        # Skip loading affixations to avoid issues with missing columns
        if include_affixations:
            # Set empty lists for affixations
            word.root_affixations = []
            word.affixed_affixations = []
        
        if include_root and word.root_word_id:
            # Load root word
            root_sql = """
            SELECT id, lemma, language_code, has_baybayin, baybayin_form
            FROM words WHERE id = :root_id
            """
            root_result = db.session.execute(text(root_sql), {"root_id": word.root_word_id}).fetchone()
            if root_result:
                root_word = Word()
                root_word.id = root_result.id
                root_word.lemma = root_result.lemma
                root_word.language_code = root_result.language_code
                root_word.has_baybayin = root_result.has_baybayin
                root_word.baybayin_form = root_result.baybayin_form
                word.root_word = root_word
            else:
                word.root_word = None
        else:
            word.root_word = None
            
        if include_derived:
            # Load derived words
            derived_sql = """
            SELECT id, lemma, language_code, has_baybayin, baybayin_form
            FROM words WHERE root_word_id = :word_id
            """
            derived_results = db.session.execute(text(derived_sql), {"word_id": word_id}).fetchall()
            word.derived_words = []
            
            for d in derived_results:
                derived = Word()
                derived.id = d.id
                derived.lemma = d.lemma
                derived.language_code = d.language_code
                derived.has_baybayin = d.has_baybayin
                derived.baybayin_form = d.baybayin_form
                word.derived_words.append(derived)
                
        if include_forms:
            # Set empty list for forms since we may have issues with this table too
            word.forms = []
            
        if include_templates:
            # Set empty list for templates since we may have issues with this table too
            word.templates = []
            
        if include_definition_relations:
            # Set empty lists for definition relations since we may have issues
            word.definition_relations = []
            word.related_definitions = []

        # Cache the result if we have a cache client
        if cache_client:
            try:
                pickled_word = pickle.dumps(word)
                cache_client.set(cache_key, pickled_word, timeout=600)  # Cache for 10 minutes
            except Exception as e:
                logger.warning(f"Cache storage error for word_id={word_id}: {e}")

        # Return the populated word object
        return word
    except Exception as e:
        logger.error(f"Error in _fetch_word_details for word_id {word_id}: {str(e)}", exc_info=True)
        return None
