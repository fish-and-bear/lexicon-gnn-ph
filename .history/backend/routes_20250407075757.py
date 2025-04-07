"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, request, jsonify, send_file, abort, current_app, g, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast, not_, case, exists, extract
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload, raiseload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
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
from backend.utils.db_helpers import _fetch_word_details # Make sure helper is imported

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
    """Get detailed information about a specific word."""
    try:
        # Try to parse as integer ID first
        word_id = None
        try:
            word_id = int(word)
            logger.debug(f"Looking up word by ID: {word_id}")
        except ValueError:
            # Use direct SQL to lookup the ID by word
            normalized = normalize_word(word)
            sql = """
            SELECT id FROM words
            WHERE LOWER(lemma) = LOWER(:word) OR LOWER(normalized_lemma) = LOWER(:normalized)
            LIMIT 1
            """
            params = {"word": word, "normalized": normalized}
            result = db.session.execute(text(sql), params).fetchone()
            
            if result:
                word_id = result[0]
                logger.debug(f"Found word ID {word_id} for lemma '{word}'")
            else:
                logger.warning(f"Word '{word}' not found by lemma")
                return jsonify({"error": f"Word '{word}' not found"}), 404
                
        # Now get word details directly with SQL
        if word_id:
            sql_word = """
            SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form 
            FROM words WHERE id = :id
            """
            word_result = db.session.execute(text(sql_word), {"id": word_id}).fetchone()
            
            if not word_result:
                return jsonify({"error": f"Word with ID {word_id} not found"}), 404
                
            # Get definitions - removed definition_metadata column
            sql_defs = """
            SELECT id, definition_text, original_pos, standardized_pos_id,
                  examples, usage_notes, tags, sources
            FROM definitions
            WHERE word_id = :word_id
            """
            defs_result = db.session.execute(text(sql_defs), {"word_id": word_id}).fetchall()
            
            # Build response
            result = {
                "id": word_result.id,
                "lemma": word_result.lemma,
                "normalized_lemma": word_result.normalized_lemma,
                "language_code": word_result.language_code,
                "has_baybayin": word_result.has_baybayin,
                "baybayin_form": word_result.baybayin_form,
                "definitions": []
            }
            
            # Add definitions if any
            for d in defs_result:
                definition = {
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "original_pos": d.original_pos,
                    "standardized_pos_id": d.standardized_pos_id,
                    "examples": d.examples if d.examples else [],
                    "usage_notes": d.usage_notes if d.usage_notes else [],
                    "tags": d.tags.split(',') if d.tags else [],
                    "sources": d.sources.split(',') if d.sources else [],
                    # Leave metadata as empty object since the column doesn't exist
                    "metadata": {}
                }
                result["definitions"].append(definition)
            
            return jsonify(result)
        else:
            return jsonify({"error": f"Word '{word}' not found"}), 404
            
    except Exception as e:
        logger.error(f"Error retrieving word '{word}'", exc_info=True)
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

# Helper function to create a minimal node representation for the graph
def _create_graph_node(word_id: int, lemma: str, language_code: Optional[str], is_main: bool = False, depth: int = 0) -> Dict[str, Any]:
    return {
        "id": str(word_id),
        "label": lemma,
        "language": language_code,
        "main": is_main,
        "depth": depth
    }

# Helper function to create a graph edge
def _create_graph_edge(source_id: int, target_id: int, rel_type: str, rel_id: Optional[int] = None) -> Dict[str, Any]:
    edge_id_str = f"{source_id}-{target_id}-{rel_type}"
    if rel_id:
        edge_id_str = f"{edge_id_str}-{rel_id}" # Make more unique if needed
        
    return {
        "id": edge_id_str,
        "source": str(source_id),
        "target": str(target_id),
        "type": rel_type,
        "directed": rel_type not in ["synonym", "antonym", "related"] # Example non-directed types
    }

@bp.route("/words/<path:word>/semantic_network", methods=["GET"])
@cached_query(timeout=900)  # Cache for 15 minutes
def get_word_network(word: str):
    """Generate a semantic network AND full details for the central word."""
    try:
        depth = min(int(request.args.get("depth", 3)), 5) # Default depth 3, max 5
        breadth = min(int(request.args.get("breadth", 15)), 50) # Default breadth 15, max 50
        relation_types_str = request.args.get("relation_types", None)

        start_time = time.time()
        
        allowed_types = None
        if relation_types_str:
            allowed_types = [rt.strip().lower() for rt in relation_types_str.split(",") if rt.strip()]
            logger.debug(f"Filtering relations by types: {allowed_types}")

        word_id = None
        main_word_data = None
        normalized = normalize_word(word)

        # --- 1. Find the Main Word ID ---
        try:
            word_id = int(word)
            # Verify ID exists (minimal check)
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

        if not word_id: # Should not happen if logic above is correct
             logger.error(f"Failed to resolve word ID for input: '{word}'")
             return jsonify({"error": "Could not resolve word ID"}), 500

        logger.info(f"Generating network for word ID: {word_id} ('{word}') | Depth: {depth}, Breadth: {breadth}")

        # --- 2. Fetch FULL Details for the Main Word ---
        # Use the helper function to get comprehensive details
        main_word_details_obj = _fetch_word_details(
            word_id,
            include_definitions=True,
            include_etymologies=True,
            include_pronunciations=True,
            include_credits=True,
            include_relations=True, # Fetch relations for the main word as well
            include_affixations=True,
            include_root=True,
            include_derived=True,
            include_forms=True,
            include_templates=True,
            include_definition_relations=True
        )

        if not main_word_details_obj:
            logger.error(f"Found word ID {word_id} but failed to fetch its full details.")
            # Fallback: get minimal data if full fetch fails unexpectedly
            sql_minimal = "SELECT id, lemma, language_code FROM words WHERE id = :id"
            minimal_res = db.session.execute(text(sql_minimal), {"id": word_id}).fetchone()
            if minimal_res:
                 main_word_data = {
                     "id": minimal_res.id,
                     "lemma": minimal_res.lemma,
                     "language_code": minimal_res.language_code
                 }
            else: # Should absolutely not happen if ID was found earlier
                 return jsonify({"error": "Failed to retrieve main word data"}), 500
                 
            # Serialize the minimal data if full fetch failed
            main_word_serialized = WordSchema(exclude=("definitions", "etymologies", "pronunciations", "credits", "outgoing_relations", "incoming_relations", "root_affixations", "affixed_affixations", "root_word", "derived_words", "forms", "templates", "definition_relations", "related_definitions")).dump(main_word_data)

        else:
            # Serialize the full details using the comprehensive WordSchema
            main_word_schema = WordSchema()
            main_word_serialized = main_word_schema.dump(main_word_details_obj)
            # Extract basic data for graph node creation
            main_word_data = {
                 "id": main_word_details_obj.id,
                 "lemma": main_word_details_obj.lemma,
                 "language_code": main_word_details_obj.language_code
            }


        # --- 3. Build the Network Graph (Nodes and Links) ---
        nodes: Dict[int, Dict[str, Any]] = {} # Store nodes by ID to avoid duplicates
        links: Dict[str, Dict[str, Any]] = {} # Store links by unique ID (e.g., "source-target-type")
        queue: List[Tuple[int, int]] = [(word_id, 0)] # (word_id, current_depth)
        visited_ids: Set[int] = {word_id} # Track visited nodes to prevent cycles/redundancy

        # Add the main word node first
        nodes[word_id] = _create_graph_node(main_word_data['id'], main_word_data['lemma'], main_word_data['language_code'], is_main=True, depth=0)

        # --- Optimized Relation Fetching using CTE ---
        # Build the recursive query if depth > 0
        network_nodes = []
        network_edges = []

        if depth > 0:
            cte_sql = """
            WITH RECURSIVE word_network(source_id, target_id, relation_type, relation_id, depth, path_ids) AS (
              -- Anchor member: Direct relations of the starting word
              SELECT
                r.from_word_id AS source_id,
                r.to_word_id AS target_id,
                r.relation_type,
                r.id AS relation_id,
                1 AS depth,
                ARRAY[r.from_word_id, r.to_word_id] AS path_ids
              FROM relations r
              WHERE r.from_word_id = :start_word_id
                {relation_type_filter_sql}

              UNION ALL

              SELECT
                r.from_word_id AS source_id,
                r.to_word_id AS target_id,
                r.relation_type,
                r.id AS relation_id,
                wn.depth + 1 AS depth,
                wn.path_ids || r.to_word_id AS path_ids
              FROM relations r
              JOIN word_network wn ON r.from_word_id = wn.target_id
              WHERE wn.depth < :max_depth
                AND r.to_word_id != ALL(wn.path_ids) -- Prevent cycles using path
                {relation_type_filter_sql}
            )
            -- Select distinct nodes and edges, limiting breadth implicitly by recursion limits
            SELECT DISTINCT
              source_id,
              target_id,
              relation_type,
              relation_id,
              depth
            FROM word_network
            -- Add limit here if needed, though CTE depth limit is primary control
            -- LIMIT 500 -- Example overall limit for safety
            """

            relation_filter_clause = ""
            cte_params = {"start_word_id": word_id, "max_depth": depth}

            if allowed_types:
                placeholders = ", ".join([f":rel_type_{i}" for i in range(len(allowed_types))])
                relation_filter_clause = f"AND r.relation_type IN ({placeholders})"
                for i, rt in enumerate(allowed_types):
                    cte_params[f"rel_type_{i}"] = rt

            # Inject the filter clause into the SQL template
            cte_sql = cte_sql.format(relation_type_filter_sql=relation_filter_clause)

            logger.debug(f"Executing CTE query for network with params: {cte_params}")
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
