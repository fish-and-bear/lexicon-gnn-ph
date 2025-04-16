"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, request, jsonify, send_file, abort, current_app, g, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast, not_, case, exists, extract
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload, raiseload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE, validates_schema, post_load
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
from prometheus_client import REGISTRY, Counter, Histogram
# Comment out problematic import
# from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict
import unicodedata

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



def normalize_query(text):
    """Normalize text for search by lowercasing and removing diacritics."""
    if not text:
        return ""
    # Convert to lowercase
    text =text.lower()
    # Remove diacritics/accents
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                  if not unicodedata.combining(c))
    return text

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

# Import metrics from the metrics module
from backend.metrics import API_REQUESTS, API_ERRORS, REQUEST_LATENCY, REQUEST_COUNT

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
    word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'))

class EtymologySchema(BaseSchema):
    """Schema for etymology data."""
    etymology_text = fields.Str(required=True)
    normalized_components = fields.String()
    etymology_structure = fields.String()
    language_codes = fields.String()
    sources = fields.String()  # Add sources field
    etymology_data = fields.Dict()  # Use etymology_data property instead of etymology_metadata
    word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'))

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True)
    relation_data = fields.Dict()
    source_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    target_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class AffixationSchema(BaseSchema):
    """Schema for word affixation data."""
    affix_type = fields.Str(required=True)
    sources = fields.String() # Added sources
    # Adjusted nesting to match common relationship patterns
    root_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'))
    affixed_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'))

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
    word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    # Use fully qualified name to avoid RegistryError
    standardized_pos = fields.Nested('backend.routes.PartOfSpeechSchema', dump_default=None)
    categories = fields.List(fields.Nested('backend.routes.DefinitionCategorySchema', exclude=('definition',)))
    # Skip links due to column mismatch
    # links = fields.List(fields.Nested('backend.routes.DefinitionLinkSchema', exclude=('definition',)))
    
    # Related definitions/words relationships
    definition_relations = fields.List(fields.Nested('backend.routes.DefinitionRelationSchema', exclude=('definition',)))
    related_words = fields.List(fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code')))

class DefinitionCategorySchema(BaseSchema):
    """Schema for definition categories."""
    id = fields.Int(dump_only=True)
    definition_id = fields.Int(required=True)
    category_name = fields.Str(required=True)
    category_kind = fields.Str(allow_none=True)  # Add this missing field
    tags = fields.Dict()
    category_metadata = fields.Dict()
    parents = fields.List(fields.Str(), dump_default=[])  # Add this field to match model
    definition = fields.Nested('backend.routes.DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

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
    definition = fields.Nested('backend.routes.DefinitionSchema', only=('id', 'definition_text'), dump_default=None)

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Int(dump_only=True)
    code = fields.Str(required=True)
    name_en = fields.Str(required=True)
    name_tl = fields.Str(required=True)
    description = fields.String()
    # Represent derived_words relationship correctly
    derived_words = fields.List(fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code')))

    # --- Add missing relationships and columns ---
    forms = fields.List(fields.Nested('backend.routes.WordFormSchema', exclude=("word",)))
    templates = fields.List(fields.Nested('backend.routes.WordTemplateSchema', exclude=("word",)))
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
    word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma')) # Nested word info

class WordTemplateSchema(BaseSchema):
    """Schema for word templates."""
    template_name = fields.Str(required=True)
    args = fields.Dict() # Assuming JSONB args map to a dict
    expansion = fields.Str()
    word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma')) # Nested word info

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
    idioms = fields.Raw(allow_none=True)  # Changed from Dict to Raw
    source_info = fields.Raw(allow_none=True)  # Changed from Dict to Raw
    word_metadata = fields.Raw(allow_none=True)  # Changed from Dict to Raw
    pronunciation_data = fields.Raw(allow_none=True)  # Added to match DB column
    data_hash = fields.String()
    search_text = fields.String()
    badlit_form = fields.String()
    hyphenation = fields.Raw(allow_none=True)  # Changed from Dict to Raw
    is_proper_noun = fields.Bool()
    is_abbreviation = fields.Bool()
    is_initialism = fields.Bool()
    is_root = fields.Bool()
    completeness_score = fields.Float(dump_only=True)
    
    # Relationships - ensure names match model relationship names
    # Use selectinload in helper, so schema just defines nesting
    definitions = fields.List(fields.Nested('backend.routes.DefinitionSchema', exclude=("word",)))
    etymologies = fields.List(fields.Nested('backend.routes.EtymologySchema', exclude=("word",)))
    pronunciations = fields.List(fields.Nested('backend.routes.PronunciationType', exclude=("word",)))
    credits = fields.List(fields.Nested('backend.routes.CreditSchema', exclude=("word",)))
    # Adjust nesting based on expected structure from helper function
    outgoing_relations = fields.List(fields.Nested('backend.routes.RelationSchema', exclude=("source_word",)))
    incoming_relations = fields.List(fields.Nested('backend.routes.RelationSchema', exclude=("target_word",)))
    root_affixations = fields.List(fields.Nested('backend.routes.AffixationSchema', exclude=("root_word",))) # Affixes where this word is the root
    affixed_affixations = fields.List(fields.Nested('backend.routes.AffixationSchema', exclude=("affixed_word",))) # Affixes where this word is the result

    # Represent root_word relationship correctly
    root_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
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

class SearchFilterSchema(Schema):
    """Schema for validating filter query parameters."""
    # Basic filters
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(dump_default=None, load_default=None)
    
    # Feature filters
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_forms = fields.Bool(dump_default=None, load_default=None)
    
    # Date range filters
    date_added_from = fields.DateTime(dump_default=None, load_default=None)
    date_added_to = fields.DateTime(dump_default=None, load_default=None)
    date_modified_from = fields.DateTime(dump_default=None, load_default=None)
    date_modified_to = fields.DateTime(dump_default=None, load_default=None)
    
    # Definition and relation count filters
    min_definition_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    max_definition_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    min_relation_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    max_relation_count = fields.Int(validate=validate.Range(min=0), dump_default=None, load_default=None)
    
    # Completeness score range
    min_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    max_completeness = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    
    # Specific tags and categories
    tags = fields.List(fields.Str(), dump_default=None, load_default=None)
    categories = fields.List(fields.Str(), dump_default=None, load_default=None)

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
        schema = WordSchema()
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
    Search the dictionary with advanced filtering options.
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
        enum: ["all", "exact", "prefix", "suffix", "fuzzy"]  # Added fuzzy mode
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
      - name: fuzzy_threshold
        in: query
        type: number
        default: 0.3
        description: Threshold for fuzzy matching (0.0-1.0, higher is more strict)
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
        
        # New parameters
        fuzzy_threshold = float(params.get('fuzzy_threshold', 0.3))
        sort_field = params.get('sort', 'relevance')
        sort_order = params.get('order', 'desc').upper()
        
        # Enhanced filters
        filters = {
            # Existing filters
            'has_baybayin': params.get('has_baybayin'),
            'has_pronunciation': params.get('has_pronunciation'),
            'has_etymology': params.get('has_etymology'),
            'has_forms': params.get('has_forms'),
            'has_templates': params.get('has_templates'),
            'has_definition_relations': params.get('has_definition_relations'),
            'language': params.get('language'),
            'pos': params.get('pos'),
            'min_completeness': params.get('min_completeness'),
            
            # New advanced filters
            'has_examples': params.get('has_examples'),
            'is_root': params.get('is_root'),
            'is_proper_noun': params.get('is_proper_noun'), 
            'is_abbreviation': params.get('is_abbreviation'),
            'is_initialism': params.get('is_initialism'),
            'min_definitions': params.get('min_definitions'),
            'max_definitions': params.get('max_definitions'),
            'created_after': params.get('created_after'),
            'created_before': params.get('created_before'),
            'updated_after': params.get('updated_after'),
            'updated_before': params.get('updated_before'),
            'relation_type': params.get('relation_type'),
            'etymology_source': params.get('etymology_source'),
            'definition_contains': params.get('definition_contains'),
            'tag_contains': params.get('tag_contains')
        }
        
        # Parameters for result formatting - same as before
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
        
        # Sanitize inputs
        if isinstance(query, str):
            query = query.strip()
        
        if not query and not any(v is not None for k, v in filters.items()):
            return jsonify({
                'count': 0,
                'results': [],
                'filters': {}
            }), 200
        
        # Validate numeric parameters
        limit = max(1, min(limit, 100))  # Constrain between 1 and 100
        offset = max(0, offset)  # Must be non-negative
        
        normalized_query = normalize_word(query) if query else ""
        
        # Build base SQL queries
        main_sql_base = """
        SELECT w.id, w.lemma, w.normalized_lemma, w.language_code, 
               w.has_baybayin, w.baybayin_form
        FROM words w
        """
        
        count_sql_base = """
        SELECT COUNT(DISTINCT w.id) FROM words w
        """
        
        # Build JOINs and conditions
        join_clauses = []
        where_conditions = []
        having_conditions = []
        query_params = {}
        
        # Search condition based on mode
        if query:
            if mode == 'exact':
                where_conditions.append("(w.normalized_lemma = :normalized_query OR w.lemma = :query)")
                query_params['normalized_query'] = normalized_query
                query_params['query'] = query
            elif mode == 'prefix':
                where_conditions.append("(w.normalized_lemma LIKE :prefix_query OR w.lemma LIKE :prefix_query)")
                query_params['prefix_query'] = f"{normalized_query}%"
            elif mode == 'suffix':
                where_conditions.append("(w.normalized_lemma LIKE :suffix_query OR w.lemma LIKE :suffix_query)")
                query_params['suffix_query'] = f"%{normalized_query}"
            elif mode == 'fuzzy':
                # Use trigram similarity for fuzzy matching
                where_conditions.append("""
                    (SIMILARITY(w.lemma, :query) > :threshold 
                     OR SIMILARITY(w.normalized_lemma, :query) > :threshold)
                """)
                query_params['query'] = query
                query_params['threshold'] = fuzzy_threshold
            else:  # 'all' mode - default
                where_conditions.append("""
                (w.normalized_lemma LIKE :contains_query 
                 OR w.lemma LIKE :contains_query
                 OR w.search_text::text LIKE :contains_query
                 OR w.lemma ILIKE :query_exact)
                """)
                query_params['contains_query'] = f"%{normalized_query}%"
                query_params['query_exact'] = query
        
        # Process filters
        if filters['has_baybayin'] is not None:
            where_conditions.append("w.has_baybayin = :has_baybayin")
            query_params['has_baybayin'] = filters['has_baybayin']
        
        # Handle exclude_baybayin parameter
        exclude_baybayin = params.get('exclude_baybayin', False)
        if exclude_baybayin:
            where_conditions.append("(w.has_baybayin = FALSE OR w.has_baybayin IS NULL)")
            
        if filters['language'] is not None:
            # Support multiple languages with comma separation
            languages = filters['language'].split(',')
            if len(languages) == 1:
                where_conditions.append("w.language_code = :language")
                query_params['language'] = filters['language']
            else:
                lang_conditions = []
                for i, lang in enumerate(languages):
                    lang_param = f"language_{i}"
                    lang_conditions.append(f"w.language_code = :{lang_param}")
                    query_params[lang_param] = lang.strip()
                where_conditions.append(f"({' OR '.join(lang_conditions)})")
        
        # Advanced boolean filters
        if filters['is_root'] is not None:
            if filters['is_root']:
                where_conditions.append("w.root_word_id IS NULL")
            else:
                where_conditions.append("w.root_word_id IS NOT NULL")
                
        if filters['is_proper_noun'] is not None:
            where_conditions.append("w.is_proper_noun = :is_proper_noun")
            query_params['is_proper_noun'] = filters['is_proper_noun']
            
        if filters['is_abbreviation'] is not None:
            where_conditions.append("w.is_abbreviation = :is_abbreviation")
            query_params['is_abbreviation'] = filters['is_abbreviation']
            
        if filters['is_initialism'] is not None:
            where_conditions.append("w.is_initialism = :is_initialism")
            query_params['is_initialism'] = filters['is_initialism']
        
        # Date range filters
        if filters['created_after'] is not None:
            where_conditions.append("w.created_at >= :created_after")
            query_params['created_after'] = filters['created_after']
            
        if filters['created_before'] is not None:
            where_conditions.append("w.created_at <= :created_before")
            query_params['created_before'] = filters['created_before']
            
        if filters['updated_after'] is not None:
            where_conditions.append("w.updated_at >= :updated_after")
            query_params['updated_after'] = filters['updated_after']
            
        if filters['updated_before'] is not None:
            where_conditions.append("w.updated_at <= :updated_before")
            query_params['updated_before'] = filters['updated_before']
        
        # Relationship-requiring filters
        if filters['has_pronunciation'] is not None:
            join_clauses.append("LEFT JOIN pronunciations p ON p.word_id = w.id")
            having_conditions.append("COUNT(DISTINCT p.id) > 0" if filters['has_pronunciation'] else "COUNT(DISTINCT p.id) = 0")
            
        if filters['has_etymology'] is not None:
            join_clauses.append("LEFT JOIN etymologies e ON e.word_id = w.id")
            having_conditions.append("COUNT(DISTINCT e.id) > 0" if filters['has_etymology'] else "COUNT(DISTINCT e.id) = 0")
            
        if filters['has_forms'] is not None:
            join_clauses.append("LEFT JOIN word_forms wf ON wf.word_id = w.id")
            having_conditions.append("COUNT(DISTINCT wf.id) > 0" if filters['has_forms'] else "COUNT(DISTINCT wf.id) = 0")
            
        if filters['has_templates'] is not None:
            join_clauses.append("LEFT JOIN word_templates wt ON wt.word_id = w.id")
            having_conditions.append("COUNT(DISTINCT wt.id) > 0" if filters['has_templates'] else "COUNT(DISTINCT wt.id) = 0")
        
        # POS filter with multiple options support
        if filters['pos'] is not None:
            pos_codes = filters['pos'].split(',')
            if len(pos_codes) == 1:
                join_clauses.append("JOIN definitions d_pos ON d_pos.word_id = w.id")
                join_clauses.append("JOIN parts_of_speech pos ON d_pos.standardized_pos_id = pos.id")
                where_conditions.append("pos.code = :pos_code")
                query_params['pos_code'] = filters['pos']
            else:
                join_clauses.append("JOIN definitions d_pos ON d_pos.word_id = w.id")
                join_clauses.append("JOIN parts_of_speech pos ON d_pos.standardized_pos_id = pos.id")
                pos_conditions = []
                for i, pos_code in enumerate(pos_codes):
                    pos_param = f"pos_code_{i}"
                    pos_conditions.append(f"pos.code = :{pos_param}")
                    query_params[pos_param] = pos_code.strip()
                where_conditions.append(f"({' OR '.join(pos_conditions)})")
        
        # Definition count filters
        if filters['min_definitions'] is not None or filters['max_definitions'] is not None:
            join_clauses.append("LEFT JOIN definitions d_count ON d_count.word_id = w.id")
            
            if filters['min_definitions'] is not None:
                having_conditions.append("COUNT(DISTINCT d_count.id) >= :min_definitions")
                query_params['min_definitions'] = int(filters['min_definitions'])
                
            if filters['max_definitions'] is not None:
                having_conditions.append("COUNT(DISTINCT d_count.id) <= :max_definitions")
                query_params['max_definitions'] = int(filters['max_definitions'])
        
        # Definition content filter
        if filters['definition_contains'] is not None:
            join_clauses.append("JOIN definitions d_content ON d_content.word_id = w.id")
            where_conditions.append("d_content.definition_text ILIKE :definition_contains")
            query_params['definition_contains'] = f"%{filters['definition_contains']}%"
        
        # Examples filter
        if filters['has_examples'] is not None:
            join_clauses.append("JOIN definitions d_examples ON d_examples.word_id = w.id")
            if filters['has_examples']:
                where_conditions.append("(d_examples.examples IS NOT NULL AND d_examples.examples != '[]')")
            else:
                where_conditions.append("(d_examples.examples IS NULL OR d_examples.examples = '[]')")
        
        # Relation type filter
        if filters['relation_type'] is not None:
            relation_types = filters['relation_type'].split(',')
            if len(relation_types) == 1:
                join_clauses.append("JOIN relations r ON (r.from_word_id = w.id OR r.to_word_id = w.id)")
                where_conditions.append("r.relation_type = :relation_type")
                query_params['relation_type'] = filters['relation_type']
            else:
                join_clauses.append("JOIN relations r ON (r.from_word_id = w.id OR r.to_word_id = w.id)")
                rel_conditions = []
                for i, rel_type in enumerate(relation_types):
                    rel_param = f"relation_type_{i}"
                    rel_conditions.append(f"r.relation_type = :{rel_param}")
                    query_params[rel_param] = rel_type.strip()
                where_conditions.append(f"({' OR '.join(rel_conditions)})")
        
        # Etymology source filter
        if filters['etymology_source'] is not None:
            join_clauses.append("JOIN etymologies etym ON etym.word_id = w.id")
            where_conditions.append("etym.language_codes ILIKE :etymology_source")
            query_params['etymology_source'] = f"%{filters['etymology_source']}%"
        
        # Tag filter
        if filters['tag_contains'] is not None:
            where_conditions.append("w.tags ILIKE :tag_contains")
            query_params['tag_contains'] = f"%{filters['tag_contains']}%"
        
        # Completeness score filter
        if filters['min_completeness'] is not None:
            # For performance, let's do a rough filter in SQL
            # The precise filtering will happen in Python
            where_conditions.append("""
            (
                (SELECT COUNT(*) FROM definitions WHERE word_id = w.id) > 0 OR
                (SELECT COUNT(*) FROM etymologies WHERE word_id = w.id) > 0 OR
                (SELECT COUNT(*) FROM pronunciations WHERE word_id = w.id) > 0 OR
                w.has_baybayin = true OR
                EXISTS (SELECT 1 FROM relations WHERE from_word_id = w.id OR to_word_id = w.id)
            )
            """)
        
        # Deduplicate join clauses
        unique_join_clauses = []
        seen_joins = set()
        for join in join_clauses:
            join_key = join.split(' ')[1]  # Extract the table alias
            if join_key not in seen_joins:
                unique_join_clauses.append(join)
                seen_joins.add(join_key)
        
        # Construct final SQL
        main_sql = main_sql_base
        count_sql = count_sql_base
        
        if unique_join_clauses:
            join_clause = " ".join(unique_join_clauses)
            main_sql += " " + join_clause
            count_sql += " " + join_clause
            
        if where_conditions:
            where_clause = " WHERE " + " AND ".join(where_conditions)
            main_sql += where_clause
            count_sql += where_clause
        
        if having_conditions:
            having_clause = " GROUP BY w.id HAVING " + " AND ".join(having_conditions)
            main_sql += having_clause
            count_sql += having_clause
        elif unique_join_clauses:
            # If we have joins but no HAVING, we still need GROUP BY
            main_sql += " GROUP BY w.id"
            count_sql += " GROUP BY w.id"
        
        # Add ORDER BY clause based on sort and order parameters
        if sort_field == 'relevance':
            if query:
                # For relevance sorting, prioritize exact matches first
                if mode == 'fuzzy':
                    main_sql += f" ORDER BY SIMILARITY(w.lemma, :query) DESC"
                else:
                    main_sql += f" ORDER BY (CASE WHEN w.normalized_lemma = :normalized_query THEN 0 ELSE 1 END), w.lemma {sort_order}"
                    query_params['normalized_query'] = normalized_query
            else:
                # Default ordering when no query
                main_sql += f" ORDER BY w.lemma ASC"
        elif sort_field == 'alphabetical':
            main_sql += f" ORDER BY w.lemma {sort_order}"
        elif sort_field == 'created':
            main_sql += f" ORDER BY w.created_at {sort_order}"
        elif sort_field == 'updated':
            main_sql += f" ORDER BY w.updated_at {sort_order}"
        elif sort_field == 'completeness':
            main_sql += f" ORDER BY w.lemma {sort_order}"
        else:
            # Default fallback sorting
            main_sql += f" ORDER BY w.lemma ASC"
        
        # Add LIMIT and OFFSET
        main_sql += " LIMIT :limit OFFSET :offset"
        query_params['limit'] = limit
        query_params['offset'] = offset
        
        # Initialize result
        result = {
            'count': 0,
            'results': [],
            'filters': filters
        }
        
        # Execute count query first
        try:
            with db.engine.connect() as conn:
                logger.debug(f"Executing count query: {count_sql}")
                count_stmt = text(count_sql) 
                count_result = conn.execute(count_stmt, query_params).scalar()
                result['count'] = count_result or 0
        except Exception as count_error:
            logger.warning(f"Error executing count query: {str(count_error)}")
            # Continue with the main query
        
        # If count is zero, return empty results
        if result['count'] == 0 and not query_params.get('min_completeness'):
            return jsonify(result), 200
            
        # Execute main query for results
        with db.engine.connect() as conn:
            logger.debug(f"Executing main query: {main_sql}")
            rows = conn.execute(text(main_sql), query_params).fetchall()
            
            if include_full:
                # Full word details for each result
                word_ids = [row[0] for row in rows]
                full_words = []
                
                # Process word IDs in batches to reduce number of queries
                max_batch_size = 10  # Adjust based on your performance testing
                for i in range(0, len(word_ids), max_batch_size):
                    batch_ids = word_ids[i:i + max_batch_size]
                    
                    # Fetch all words in this batch
                    for word_id in batch_ids:
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
                            # Check completeness_score filter here
                            min_completeness = filters.get('min_completeness')
                            if min_completeness is not None and word.completeness_score < float(min_completeness):
                                continue
                                
                            schema = WordSchema()
                            serialized = schema.dump(word)
                            full_words.append(serialized)
                
                result['results'] = full_words
            else:
                # Basic info for each result
                basic_results = []
                completeness_filter = filters.get('min_completeness')
                
                if completeness_filter is not None:
                    # We need to fetch full objects to calculate completeness
                    for row in rows:
                        word_id = row[0]
                        word = _fetch_word_details(
                            word_id,
                            include_definitions=True,
                            include_etymologies=True,
                            include_pronunciations=True,
                            include_credits=False,
                            include_relations=True,
                            include_forms=False,
                            include_templates=False,
                            include_definition_relations=False
                        )
                        
                        if word and word.completeness_score >= float(completeness_filter):
                            basic_results.append({
                                'id': word.id,
                                'lemma': word.lemma,
                                'normalized_lemma': word.normalized_lemma,
                                'language_code': word.language_code,
                                'has_baybayin': bool(word.has_baybayin) if word.has_baybayin is not None else None,
                                'baybayin_form': word.baybayin_form,
                                'completeness_score': word.completeness_score
                            })
                    
                    # Update count to reflect filtered results
                    result['count'] = len(basic_results)
                else:
                    # No completeness filter, use the direct results
                    for row in rows:
                        basic_results.append({
                            'id': row[0],
                            'lemma': row[1],
                            'normalized_lemma': row[2],
                            'language_code': row[3],
                            'has_baybayin': bool(row[4]) if row[4] is not None else None,
                            'baybayin_form': row[5],
                            'completeness_score': None
                        })
                
                result['results'] = basic_results
        
        return jsonify(result), 200
    except Exception as e:
        # Make error handling more robust
        logger.error(f"Search function error: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="search", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"Unexpected error ({type(e).__name__})", 
            "message": str(e),
            "query": query if 'query' in locals() else None,
            "filters": filters if 'filters' in locals() else None
        }), 500

@bp.route("/random", methods=["GET"])
def get_random_word():
    """Get a random word with optional filters."""
    try:
        API_REQUESTS.labels(endpoint="get_random_word", method="GET").inc()
        
        # Parse filter parameters with defaults
        language = request.args.get("language")
        pos_code = request.args.get("pos")
        has_etymology = request.args.get("has_etymology", "false").lower() == "true"
        has_definitions = request.args.get("has_definitions", "true").lower() == "true"
        has_baybayin = request.args.get("has_baybayin", "false").lower() == "true"
        min_definitions = int(request.args.get("min_definitions", 1))
        
        # Add max attempts to avoid infinite loops
        max_attempts = 5
        attempts = 0
        word = None
        errors = []
        
        while attempts < max_attempts and word is None:
            attempts += 1
            try:
                # Build and execute query to find random word ID
                sql_query = build_random_word_query(
                    language, pos_code, has_etymology, 
                    has_definitions, has_baybayin, min_definitions
                )
                
                # Execute query
                result = db.session.execute(text(sql_query), {
                    "language": language, 
                    "pos_code": pos_code,
                    "min_defs": min_definitions
                }).fetchone()
                
                if not result:
                    logger.info(f"No words match the specified criteria for random selection: language={language}, pos={pos_code}, has_etymology={has_etymology}")
                    return jsonify({
                        "error": "No words match the specified criteria",
                        "criteria": {
                            "language": language,
                            "pos": pos_code,
                            "has_etymology": has_etymology,
                            "has_definitions": has_definitions,
                            "has_baybayin": has_baybayin,
                            "min_definitions": min_definitions
                        }
                    }), 404
                
                # Get the random word's details
                word = _fetch_word_details(
                    result.id,
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
                
                # Validate word data to ensure it's complete
                if word is None or not hasattr(word, 'lemma') or not word.lemma:
                    logger.warning(f"Random word attempt {attempts} returned incomplete data")
                    errors.append(f"Attempt {attempts}: Incomplete word data")
                    word = None  # Reset to try again
                    continue
                    
            except Exception as e:
                error_msg = f"Attempt {attempts} failed: {type(e).__name__}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                word = None  # Ensure word is None to try again
        
        if word is None:
            error_details = "\n".join(errors)
            logger.error(f"Failed to get random word after {max_attempts} attempts:\n{error_details}")
            return jsonify({
                "error": "Failed to get a random word after multiple attempts", 
                "details": errors,
                "suggestion": "Please try again or modify your filters"
            }), 500
        
        # Serialize and return the word data
        schema = WordSchema()
        result = schema.dump(word)
        
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
    definition = fields.Nested('backend.routes.DefinitionSchema', only=('id', 'definition_text'))
    related_word = fields.Nested('backend.routes.WordSchema', only=('id', 'lemma', 'language_code'))

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
                    # Wrap each batch in an explicit transaction
                    with db.session.begin():
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

# Add advanced statistical analysis endpoints
@bp.route("/statistics/advanced", methods=["GET"])
@cached_query(timeout=1800)  # Cache for 30 minutes as this is resource-intensive
def get_advanced_statistics():
    """Get advanced statistical analysis of the dictionary."""
    try:
        API_REQUESTS.labels(endpoint="get_advanced_statistics", method="GET").inc()
        start_time = time.time()
        
        # Get optional language filter
        language = request.args.get('language')
        
        # Initialize result dictionary
        result = {
            "word_counts": {},
            "pos_distribution": {},
            "etymology_sources": {},
            "relation_types": {},
            "baybayin_stats": {},
            "definition_counts": {},
            "complexity_distribution": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Base query condition
        base_condition = ""
        params = {}
        if language:
            base_condition = "WHERE language_code = :language"
            params["language"] = language
        
        # 1. Get word counts by language
        sql_lang = f"""
        SELECT language_code, COUNT(*) as count
        FROM words
        {base_condition}
        GROUP BY language_code
        ORDER BY count DESC
        """
        lang_counts = db.session.execute(text(sql_lang), params).fetchall()
        result["word_counts"] = {row.language_code: row.count for row in lang_counts}
        
        # 2. Get POS distribution
        sql_pos = f"""
        SELECT pos.code, pos.name_en, COUNT(DISTINCT d.word_id) as word_count
        FROM definitions d
        JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
        JOIN words w ON d.word_id = w.id
        {base_condition}
        GROUP BY pos.code, pos.name_en
        ORDER BY word_count DESC
        """
        pos_counts = db.session.execute(text(sql_pos), params).fetchall()
        result["pos_distribution"] = {row.code: {"name": row.name_en, "count": row.word_count} for row in pos_counts}
        
        # 3. Etymology source distribution
        sql_etym = f"""
        WITH extracted_langs AS (
          SELECT 
            e.word_id,
            jsonb_array_elements_text(
              CASE 
                WHEN e.language_codes IS NULL THEN '["unknown"]'
                WHEN e.language_codes = '' THEN '["unknown"]'
                ELSE e.language_codes::jsonb
              END
            ) as source_lang
          FROM etymologies e
          JOIN words w ON e.word_id = w.id
          {base_condition}
        )
        SELECT source_lang, COUNT(*) as count 
        FROM extracted_langs
        GROUP BY source_lang
        ORDER BY count DESC
        """
        etym_counts = db.session.execute(text(sql_etym), params).fetchall()
        result["etymology_sources"] = {row.source_lang: row.count for row in etym_counts}
        
        # 4. Relation type distribution
        sql_rel = f"""
        SELECT relation_type, COUNT(*) as count
        FROM relations r
        JOIN words w ON r.from_word_id = w.id
        {base_condition}
        GROUP BY relation_type
        ORDER BY count DESC
        """
        rel_counts = db.session.execute(text(sql_rel), params).fetchall()
        result["relation_types"] = {row.relation_type: row.count for row in rel_counts}
        
        # 5. Baybayin statistics
        sql_baybayin = f"""
        SELECT 
          SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END) as has_baybayin_count,
          COUNT(*) as total_count,
          ROUND((SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END)::float / COUNT(*)) * 100, 2) as percentage
        FROM words
        {base_condition}
        """
        baybayin_stats = db.session.execute(text(sql_baybayin), params).fetchone()
        result["baybayin_stats"] = {
            "with_baybayin": baybayin_stats.has_baybayin_count,
            "total_words": baybayin_stats.total_count,
            "percentage": baybayin_stats.percentage
        }
        
        # 6. Definition counts distribution
        sql_def_counts = f"""
        SELECT 
          definition_count,
          COUNT(*) as word_count
        FROM (
          SELECT 
            w.id,
            COUNT(d.id) as definition_count
          FROM words w
          LEFT JOIN definitions d ON w.id = d.word_id
          {base_condition}
          GROUP BY w.id
        ) as def_counts
        GROUP BY definition_count
        ORDER BY definition_count
        """
        def_count_dist = db.session.execute(text(sql_def_counts), params).fetchall()
        result["definition_counts"] = {row.definition_count: row.word_count for row in def_count_dist}
        
        # 7. Completeness score distribution (rounded to nearest 0.1)
        sql_completeness = f"""
        WITH word_scores AS (
          SELECT 
            w.id,
            CASE
              WHEN (SELECT COUNT(*) FROM definitions WHERE word_id = w.id) > 0 THEN 0.3
              ELSE 0
            END +
            CASE
              WHEN (SELECT COUNT(*) FROM etymologies WHERE word_id = w.id) > 0 THEN 0.15
              ELSE 0
            END +
            CASE
              WHEN (SELECT COUNT(*) FROM pronunciations WHERE word_id = w.id) > 0 THEN 0.1
              ELSE 0
            END +
            CASE
              WHEN has_baybayin = true THEN 0.1
              ELSE 0
            END +
            CASE
              WHEN (SELECT COUNT(*) FROM relations WHERE from_word_id = w.id OR to_word_id = w.id) > 0 THEN 0.15
              ELSE 0
            END +
            CASE
              WHEN (SELECT COUNT(*) FROM affixations WHERE root_word_id = w.id OR affixed_word_id = w.id) > 0 THEN 0.1
              ELSE 0
            END AS completeness_score
          FROM words w
          {base_condition}
        )
        SELECT 
          ROUND(completeness_score::numeric, 1) as score_rounded,
          COUNT(*) as word_count
        FROM word_scores
        GROUP BY score_rounded
        ORDER BY score_rounded
        """
        completeness_dist = db.session.execute(text(sql_completeness), params).fetchall()
        result["complexity_distribution"] = {float(row.score_rounded): row.word_count for row in completeness_dist}
        
        # Record execution time
        execution_time = time.time() - start_time
        logger.info(f"Generated advanced statistics in {execution_time:.2f}s")
        REQUEST_LATENCY.labels(endpoint="get_advanced_statistics").observe(execution_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating advanced statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_advanced_statistics", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "type": type(e).__name__
        }), 500

@bp.route("/statistics/timeseries", methods=["GET"])
@cached_query(timeout=3600)  # Cache for 1 hour
def get_timeseries_statistics():
    """Get time-based statistical analysis showing growth and changes over time."""
    try:
        API_REQUESTS.labels(endpoint="get_timeseries_statistics", method="GET").inc()
        
        # Get parameters
        interval = request.args.get('interval', 'month')  # day, week, month, year
        period = request.args.get('period', '1-year')  # 1-month, 3-month, 6-month, 1-year, all-time
        language = request.args.get('language')
        
        # Convert period to date
        end_date = datetime.now(timezone.utc)
        if period == '1-month':
            start_date = end_date - timedelta(days=30)
        elif period == '3-month':
            start_date = end_date - timedelta(days=90)
        elif period == '6-month':
            start_date = end_date - timedelta(days=180)
        elif period == '1-year':
            start_date = end_date - timedelta(days=365)
        else:  # all-time
            start_date = datetime(2000, 1, 1, tzinfo=timezone.utc)  # Arbitrarily old date
            
        # Build time grouping expression based on interval
        if interval == 'day':
            time_group = "DATE_TRUNC('day', created_at)"
            format_pattern = 'YYYY-MM-DD'
        elif interval == 'week':
            time_group = "DATE_TRUNC('week', created_at)"
            format_pattern = 'YYYY-WW'
        elif interval == 'year':
            time_group = "DATE_TRUNC('year', created_at)"
            format_pattern = 'YYYY'
        else:  # Default to month
            time_group = "DATE_TRUNC('month', created_at)"
            format_pattern = 'YYYY-MM'
            
        # Base query parameters
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add language condition if provided
        language_condition = ""
        if language:
            language_condition = "AND language_code = :language"
            params["language"] = language
            
        # Query for new words over time
        sql_new_words = f"""
        SELECT 
          {time_group} as time_period,
          TO_CHAR({time_group}, '{format_pattern}') as period_label,
          COUNT(*) as word_count
        FROM words
        WHERE created_at BETWEEN :start_date AND :end_date
        {language_condition}
        GROUP BY time_period
        ORDER BY time_period
        """
        new_words_data = db.session.execute(text(sql_new_words), params).fetchall()
        
        # Query for words updated over time
        sql_updated_words = f"""
        SELECT 
          {time_group} as time_period,
          TO_CHAR({time_group}, '{format_pattern}') as period_label,
          COUNT(*) as word_count
        FROM words
        WHERE updated_at BETWEEN :start_date AND :end_date
        AND updated_at > created_at  -- Only count actual updates
        {language_condition}
        GROUP BY time_period
        ORDER BY time_period
        """
        updated_words_data = db.session.execute(text(sql_updated_words), params).fetchall()
        
        # Query for new definitions over time
        sql_new_definitions = f"""
        SELECT 
          {time_group} as time_period,
          TO_CHAR({time_group}, '{format_pattern}') as period_label,
          COUNT(*) as definition_count
        FROM definitions d
        JOIN words w ON d.word_id = w.id
        WHERE d.created_at BETWEEN :start_date AND :end_date
        {language_condition}
        GROUP BY time_period
        ORDER BY time_period
        """
        new_definitions_data = db.session.execute(text(sql_new_definitions), params).fetchall()
        
        # Query for baybayin adoption over time
        sql_baybayin = f"""
        SELECT 
          {time_group} as time_period,
          TO_CHAR({time_group}, '{format_pattern}') as period_label,
          SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END) as baybayin_count,
          COUNT(*) as total_words,
          ROUND((SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END)::float / 
               NULLIF(COUNT(*), 0)) * 100, 2) as percentage
        FROM words
        WHERE created_at BETWEEN :start_date AND :end_date
        {language_condition}
        GROUP BY time_period
        ORDER BY time_period
        """
        baybayin_data = db.session.execute(text(sql_baybayin), params).fetchall()
        
        # Build the result
        result = {
            "interval": interval,
            "period": period,
            "language": language,
            "new_words": [{"period": row.period_label, "count": row.word_count} for row in new_words_data],
            "updated_words": [{"period": row.period_label, "count": row.word_count} for row in updated_words_data],
            "new_definitions": [{"period": row.period_label, "count": row.definition_count} for row in new_definitions_data],
            "baybayin_adoption": [
                {
                    "period": row.period_label, 
                    "baybayin_count": row.baybayin_count,
                    "total_words": row.total_words,
                    "percentage": row.percentage
                } 
                for row in baybayin_data
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating timeseries statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_timeseries_statistics", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "type": type(e).__name__
        }), 500

@bp.route("/statistics/language/<language_code>", methods=["GET"])
@cached_query(timeout=1800)  # Cache for 30 minutes
def get_language_statistics(language_code: str):
    """Get detailed statistics for a specific language."""
    try:
        API_REQUESTS.labels(endpoint="get_language_statistics", method="GET").inc()
        
        # Validate language code
        sql_check = "SELECT COUNT(*) FROM words WHERE language_code = :language_code"
        count = db.session.execute(text(sql_check), {"language_code": language_code}).scalar()
        
        if count == 0:
            return jsonify({"error": f"No words found for language code '{language_code}'"}), 404
            
        # Get language-specific statistics
        result = {
            "language_code": language_code,
            "total_words": count,
            "pos_distribution": {},
            "relation_stats": {},
            "etymology_sources": {},
            "baybayin_coverage": {},
            "quality_metrics": {}
        }
        
        # Get POS distribution
        sql_pos = """
        SELECT 
          pos.code, 
          pos.name_en,
          COUNT(DISTINCT d.word_id) as word_count
        FROM definitions d
        JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
        JOIN words w ON d.word_id = w.id
        WHERE w.language_code = :language_code
        GROUP BY pos.code, pos.name_en
        ORDER BY word_count DESC
        """
        pos_data = db.session.execute(text(sql_pos), {"language_code": language_code}).fetchall()
        result["pos_distribution"] = {row.code: {"name": row.name_en, "count": row.word_count} for row in pos_data}
        
        # Get relation statistics
        sql_relations = """
        SELECT 
          relation_type,
          COUNT(*) as relation_count,
          COUNT(DISTINCT from_word_id) as source_word_count
        FROM relations r
        JOIN words w ON r.from_word_id = w.id
        WHERE w.language_code = :language_code
        GROUP BY relation_type
        ORDER BY relation_count DESC
        """
        relations_data = db.session.execute(text(sql_relations), {"language_code": language_code}).fetchall()
        result["relation_stats"] = {
            row.relation_type: {
                "count": row.relation_count,
                "source_words": row.source_word_count
            } for row in relations_data
        }
        
        # Get etymology sources
        sql_etymology = """
        WITH extracted_langs AS (
          SELECT 
            e.word_id,
            jsonb_array_elements_text(
              CASE 
                WHEN e.language_codes IS NULL THEN '["unknown"]'
                WHEN e.language_codes = '' THEN '["unknown"]'
                ELSE e.language_codes::jsonb
              END
            ) as source_lang
          FROM etymologies e
          JOIN words w ON e.word_id = w.id
          WHERE w.language_code = :language_code
        )
        SELECT source_lang, COUNT(*) as count 
        FROM extracted_langs
        GROUP BY source_lang
        ORDER BY count DESC
        """
        etymology_data = db.session.execute(text(sql_etymology), {"language_code": language_code}).fetchall()
        result["etymology_sources"] = {row.source_lang: row.count for row in etymology_data}
        
        # Get Baybayin coverage
        sql_baybayin = """
        SELECT 
          SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END) as with_baybayin,
          COUNT(*) as total_words,
          ROUND((SUM(CASE WHEN has_baybayin = true THEN 1 ELSE 0 END)::float / COUNT(*)) * 100, 2) as percentage
        FROM words
        WHERE language_code = :language_code
        """
        baybayin_data = db.session.execute(text(sql_baybayin), {"language_code": language_code}).fetchone()
        result["baybayin_coverage"] = {
            "with_baybayin": baybayin_data.with_baybayin,
            "total_words": baybayin_data.total_words,
            "percentage": baybayin_data.percentage
        }
        
        # Get quality metrics
        sql_quality = """
        SELECT
          AVG(definition_count) as avg_definitions_per_word,
          SUM(CASE WHEN definition_count > 0 THEN 1 ELSE 0 END) as words_with_definitions,
          SUM(CASE WHEN etymology_count > 0 THEN 1 ELSE 0 END) as words_with_etymology,
          SUM(CASE WHEN pronunciation_count > 0 THEN 1 ELSE 0 END) as words_with_pronunciation,
          SUM(CASE WHEN relation_count > 0 THEN 1 ELSE 0 END) as words_with_relations,
          COUNT(*) as total_words
        FROM (
          SELECT 
            w.id,
            (SELECT COUNT(*) FROM definitions WHERE word_id = w.id) as definition_count,
            (SELECT COUNT(*) FROM etymologies WHERE word_id = w.id) as etymology_count,
            (SELECT COUNT(*) FROM pronunciations WHERE word_id = w.id) as pronunciation_count,
            (SELECT COUNT(*) FROM relations WHERE from_word_id = w.id OR to_word_id = w.id) as relation_count
          FROM words w
          WHERE w.language_code = :language_code
        ) word_stats
        """
        quality_data = db.session.execute(text(sql_quality), {"language_code": language_code}).fetchone()
        result["quality_metrics"] = {
            "avg_definitions_per_word": float(quality_data.avg_definitions_per_word or 0),
            "words_with_definitions_pct": round((quality_data.words_with_definitions / quality_data.total_words) * 100, 2) if quality_data.total_words > 0 else 0,
            "words_with_etymology_pct": round((quality_data.words_with_etymology / quality_data.total_words) * 100, 2) if quality_data.total_words > 0 else 0,
            "words_with_pronunciation_pct": round((quality_data.words_with_pronunciation / quality_data.total_words) * 100, 2) if quality_data.total_words > 0 else 0,
            "words_with_relations_pct": round((quality_data.words_with_relations / quality_data.total_words) * 100, 2) if quality_data.total_words > 0 else 0
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating language statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_language_statistics", error_type=type(e).__name__).inc()
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
            "type": type(e).__name__
        }), 500

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
                        include_definition_relations=True):  # Changed default to True
    """
    Fetch a word with specified relationships using direct SQL to avoid ORM issues.
    Returns a Word object with the requested relationships loaded, handling errors gracefully.
    """
    # Use the global cache_client defined at module level
    cache_key = None
    if cache_client:
        try:
            cache_key = f'word_details:{word_id}:{include_definitions}:{include_etymologies}:' \
                        f'{include_pronunciations}:{include_credits}:{include_relations}:' \
                        f'{include_affixations}:{include_root}:{include_derived}:' \
                        f'{include_forms}:{include_templates}:{include_definition_relations}'
            cached_data = cache_client.get(cache_key)
            if cached_data:
                try:
                    word = pickle.loads(cached_data)
                    if word:
                        return word
                except (pickle.PickleError, EOFError) as pickle_error:
                    # Handle corrupted cache data
                    logger.warning(f"Cache data corruption for word_id={word_id}: {pickle_error}")
        except Exception as e:
            logger.warning(f"Cache retrieval error for word_id={word_id}: {e}")
            # Continue execution with DB lookup instead of failing

    try:
        # First get the basic word data
        sql_word = """
        SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form,
               romanized_form, root_word_id, preferred_spelling, tags, source_info, 
               word_metadata, data_hash, search_text, idioms, pronunciation_data,
               badlit_form, hyphenation, is_proper_noun, is_abbreviation, is_initialism,
               created_at, updated_at
        FROM words
        WHERE id = :id
        """
        word_result = db.session.execute(text(sql_word), {"id": word_id}).fetchone()

        if not word_result:
            logger.warning(f"Word with ID {word_id} not found in database.")
            # Raise a specific error if the word itself isn't found
            raise ValueError(f"Word with ID {word_id} not found")

        # Create word object manually
        word = Word()
        for key in word_result.keys():
            if hasattr(word, key) and key != 'is_root':
                value = word_result[key]
                try:
                    # Handle baybayin_form specially to avoid validation errors
                    if key == 'baybayin_form':
                        try:
                            if value is not None:
                                setattr(word, key, value)
                            # If setting succeeds, ensure has_baybayin is consistent
                            if value and len(value.strip()) > 0:
                                word.has_baybayin = True
                            else:
                                word.has_baybayin = False
                        except ValueError as bve:
                            # Log but don't fail if baybayin validation fails
                            logger.warning(f"Value error in baybayin_form for word_id {word_id}: {str(bve)}")
                            word.baybayin_form = None
                            word.has_baybayin = False
                    else:
                        setattr(word, key, value)
                except Exception as te:
                    logger.warning(f"Error setting {key} for word {word_id}: {te}, value: {value}")
                    # Don't let other attribute errors prevent the word from loading
                    if key == 'baybayin_form':
                        word.baybayin_form = None
                        word.has_baybayin = False

        # --- Load related data with error handling for each section --- 

        # Load definitions if requested
        if include_definitions:
            try:
                sql_defs = """
                SELECT id, definition_text, original_pos, standardized_pos_id,
                       examples, usage_notes, tags, sources
                FROM definitions
                WHERE word_id = :word_id
                """
                defs_result = db.session.execute(text(sql_defs), {"word_id": word_id}).fetchall()
                word.definitions = []
                # --- Fetch all POS data in one go for efficiency ---
                pos_ids = [d.standardized_pos_id for d in defs_result if d.standardized_pos_id]
                pos_data = {}
                if pos_ids:
                    sql_pos = "SELECT id, code, name_en, name_tl, description FROM parts_of_speech WHERE id = ANY(:ids)"
                    pos_results = db.session.execute(text(sql_pos), {"ids": pos_ids}).fetchall()
                    for p_row in pos_results:
                        pos = PartOfSpeech()
                        pos.id = p_row.id
                        pos.code = p_row.code
                        pos.name_en = p_row.name_en
                        pos.name_tl = p_row.name_tl
                        pos.description = p_row.description
                        pos_data[p_row.id] = pos
                # --- End POS pre-fetch ---

                # --- Pre-fetch Definition related data ---
                definition_ids = [d.id for d in defs_result if d.id]
                categories_by_def_id = {}
                links_by_def_id = {}
                def_relations_by_def_id = {}

                if definition_ids:
                    # Categories (Assume always needed if definitions are included)
                    try:
                        # Modified SQL to only select columns that definitely exist
                        sql_categories = """
                        SELECT id, definition_id, category_name, category_kind, category_metadata, parents
                        FROM definition_categories
                        WHERE definition_id = ANY(:ids)
                        """
                        category_results = db.session.execute(text(sql_categories), {"ids": definition_ids}).fetchall()
                        for c_row in category_results:
                            category = DefinitionCategory()
                            category.id = c_row.id
                            category.definition_id = c_row.definition_id
                            category.category_name = c_row.category_name
                            category.category_kind = c_row.category_kind
                            # Get metadata and parents from the query results
                            category.category_metadata = c_row.category_metadata if c_row.category_metadata else {}
                            category.parents = c_row.parents if c_row.parents else []
                            
                            if c_row.definition_id not in categories_by_def_id:
                                categories_by_def_id[c_row.definition_id] = []
                            categories_by_def_id[c_row.definition_id].append(category)
                    except Exception as e:
                        logger.error(f"Error loading definition categories for word {word_id}: {e}", exc_info=False)
                        # Rollback just this query to keep transaction alive
                        db.session.rollback()
                        categories_by_def_id = {}

                    # Links (Assume always needed if definitions are included)
                    try:
                        # Modified SQL to only select columns that definitely exist
                        sql_links = """
                        SELECT id, definition_id, link_text, target_url, display_text, is_external
                        FROM definition_links
                        WHERE definition_id = ANY(:ids)
                        """
                        link_results = db.session.execute(text(sql_links), {"ids": definition_ids}).fetchall()
                        for l_row in link_results:
                            link = DefinitionLink()
                            link.id = l_row.id
                            link.definition_id = l_row.definition_id
                            link.link_text = l_row.link_text
                            link.target_url = l_row.target_url
                            link.display_text = l_row.display_text
                            link.is_external = l_row.is_external
                            
                            if l_row.definition_id not in links_by_def_id:
                                links_by_def_id[l_row.definition_id] = []
                            links_by_def_id[l_row.definition_id].append(link)
                    except Exception as e:
                        logger.error(f"Error loading definition links for word {word_id}: {e}", exc_info=False)
                        # Rollback just this query to keep transaction alive
                        db.session.rollback()
                        links_by_def_id = {}

                    # Definition Relations (Only if requested)
                    if include_definition_relations:
                        try:
                            # Modified SQL to only select columns that definitely exist
                            sql_rel = """
                            SELECT dr.id, dr.definition_id, dr.word_id as related_word_id, dr.relation_type,
                                   w.lemma as related_word_lemma, w.language_code as related_word_language_code
                            FROM definition_relations dr
                            JOIN words w ON dr.word_id = w.id
                            WHERE dr.definition_id = ANY(:ids)
                            """
                            rel_results = db.session.execute(text(sql_rel), {"ids": definition_ids}).fetchall()
                            for r_row in rel_results:
                                rel = DefinitionRelation()
                                # Assign fields safely
                                rel.id = r_row.id
                                rel.definition_id = r_row.definition_id
                                rel.relation_type = r_row.relation_type
                                # Create related word
                                related_word = Word()
                                related_word.id = r_row.related_word_id
                                related_word.lemma = r_row.related_word_lemma
                                related_word.language_code = r_row.related_word_language_code
                                rel.related_word = related_word
                                
                                # Initialize empty metadata
                                rel.relation_metadata = {}
                                
                                if r_row.definition_id not in relations_by_def_id:
                                    relations_by_def_id[r_row.definition_id] = []
                                relations_by_def_id[r_row.definition_id].append(rel)
                        except Exception as e:
                            logger.error(f"Error loading definition relations for word {word_id}: {e}", exc_info=False)
                            # Rollback just this query to keep transaction alive
                            db.session.rollback()
                            relations_by_def_id = {}
                # --- End pre-fetch ---


                for d in defs_result:
                    definition = Definition()
                    for key in d.keys():
                        if hasattr(definition, key):
                           try: # Added inner try-except for attribute setting
                                setattr(definition, key, d[key])
                           except Exception as attr_e:
                                logger.warning(f"Error setting definition attr {key} for word {word_id}: {attr_e}")
                    definition.word_id = word_id

                    # --- Attach pre-fetched POS object ---
                    if definition.standardized_pos_id and definition.standardized_pos_id in pos_data:
                        definition.standardized_pos = pos_data[definition.standardized_pos_id]
                    else:
                        definition.standardized_pos = None # Ensure attribute exists even if no POS found
                    # --- End POS attachment ---

                    # --- Attach pre-fetched definition related data ---
                    definition.categories = categories_by_def_id.get(definition.id, [])
                    definition.links = links_by_def_id.get(definition.id, [])
                    # Check flag before assigning definition relations
                    if include_definition_relations:
                         definition.definition_relations = def_relations_by_def_id.get(definition.id, [])
                    else: # Ensure attribute exists even if not included
                         definition.definition_relations = []
                    # --- End definition related attachment ---

                    word.definitions.append(definition)
            except Exception as e:
                logger.error(f"Error loading definitions for word {word_id}: {e}", exc_info=False) # Less verbose logging
                word.definitions = [] # Ensure it's an empty list on error
        else: word.definitions = []

        # Load etymologies if requested
        if include_etymologies:
            try:
                sql_etym = """
                SELECT id, etymology_text, normalized_components, etymology_structure,
                      language_codes, sources
                FROM etymologies
                WHERE word_id = :word_id
                """
                etym_result = db.session.execute(text(sql_etym), {"word_id": word_id}).fetchall()
                word.etymologies = []
                for e_row in etym_result:
                    etymology = Etymology()
                    for key in e_row.keys():
                        if hasattr(etymology, key):
                            try:
                                setattr(etymology, key, e_row[key])
                            except Exception as attr_e:
                                logger.warning(f"Error setting etymology attr {key} for word {word_id}: {attr_e}")
                    etymology.word_id = word_id
                    word.etymologies.append(etymology)
            except Exception as e:
                logger.error(f"Error loading etymologies for word {word_id}: {e}", exc_info=False)
                word.etymologies = []
        else: word.etymologies = []

        # Load pronunciations if requested
        if include_pronunciations:
            try:
                sql_pron = """
                SELECT id, type, value, tags, pronunciation_metadata, sources
                FROM pronunciations 
                WHERE word_id = :word_id
                """
                pron_result = db.session.execute(text(sql_pron), {"word_id": word_id}).fetchall()
                word.pronunciations = []
                for p_row in pron_result:
                    try:
                        pronunciation = Pronunciation()
                        pronunciation.id = p_row.id
                        pronunciation.type = p_row.type if p_row.type in ['ipa', 'x-sampa', 'pinyin', 'jyutping', 'romaji', 'audio', 'respelling', 'phonemic'] else 'respelling'
                        pronunciation.value = p_row.value
                        pronunciation.tags = p_row.tags if isinstance(p_row.tags, dict) else {}
                        pronunciation.pronunciation_metadata = p_row.pronunciation_metadata if isinstance(p_row.pronunciation_metadata, dict) else {}
                        pronunciation.sources = p_row.sources
                        pronunciation.word_id = word_id
                        word.pronunciations.append(pronunciation)
                    except Exception as inner_e:
                        logger.warning(f"Error processing pronunciation {p_row.id} for word {word_id}: {inner_e}")
                        continue
            except Exception as e:
                logger.error(f"Error loading pronunciations for word {word_id}: {e}", exc_info=False)
                word.pronunciations = []
        else: word.pronunciations = []

        # Load relations if requested
        if include_relations:
            try:
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
                    try: # Added inner try-except
                        relation = Relation()
                        # Manually set attributes to avoid issues with potential missing columns
                        relation.id = getattr(r, 'id', None)
                        relation.from_word_id = getattr(r, 'from_word_id', None)
                        relation.to_word_id = getattr(r, 'to_word_id', None)
                        relation.relation_type = getattr(r, 'relation_type', None)
                        relation.relation_data = getattr(r, 'relation_data', None)

                        target_word = Word()
                        target_word.id = getattr(r, 'target_id', None)
                        target_word.lemma = getattr(r, 'target_lemma', None)
                        target_word.language_code = getattr(r, 'target_language_code', None)
                        target_word.has_baybayin = getattr(r, 'target_has_baybayin', None)
                        
                        # Safe handling for baybayin form in relations
                        try:
                            baybayin_form = getattr(r, 'target_baybayin_form', None)
                            if baybayin_form and len(baybayin_form.strip()) > 0:
                                target_word.baybayin_form = baybayin_form
                            else:
                                target_word.baybayin_form = None
                                target_word.has_baybayin = False
                        except ValueError:
                            # Invalid baybayin form, set to None
                            target_word.baybayin_form = None
                            target_word.has_baybayin = False
                        
                        relation.target_word = target_word
                        word.outgoing_relations.append(relation)
                    except Exception as inner_e:
                        logger.warning(f"Error processing outgoing relation ID {getattr(r, 'id', 'UNKNOWN')} for word {word_id}: {inner_e}")

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
                    try: # Added inner try-except
                        relation = Relation()
                        relation.id = getattr(r, 'id', None)
                        relation.from_word_id = getattr(r, 'from_word_id', None)
                        relation.to_word_id = getattr(r, 'to_word_id', None)
                        relation.relation_type = getattr(r, 'relation_type', None)
                        relation.relation_data = getattr(r, 'relation_data', None)

                        source_word = Word()
                        source_word.id = getattr(r, 'source_id', None)
                        # Remove duplicate ID assignment
                        source_word.lemma = r.source_lemma
                        source_word.language_code = r.source_language_code
                        source_word.has_baybayin = r.source_has_baybayin
                        source_word.baybayin_form = r.source_baybayin_form
                        relation.source_word = source_word
                        word.incoming_relations.append(relation)
                    except Exception as inner_e:
                         logger.warning(f"Error processing incoming relation ID {r.id} for word {word_id}: {inner_e}")

            except Exception as e:
                logger.error(f"Error loading relations for word {word_id}: {e}", exc_info=False)
                word.outgoing_relations = []
                word.incoming_relations = []

        # Load root word if requested and exists
        if include_root and word.root_word_id:
            try:
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
            except Exception as e:
                 logger.error(f"Error loading root word for word {word_id}: {e}", exc_info=False)
                 word.root_word = None
        else:
             word.root_word = None # Ensure it's None if not included or no root_word_id

        # Load derived words if requested
        if include_derived:
            try:
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
            except Exception as e:
                logger.error(f"Error loading derived words for word {word_id}: {e}", exc_info=False)
                word.derived_words = []

        # Load credits if requested
        if include_credits:
            try:
                sql_credits = "SELECT id, credit FROM credits WHERE word_id = :word_id"
                cred_result = db.session.execute(text(sql_credits), {"word_id": word_id}).fetchall()
                word.credits = []
                for c_row in cred_result:
                    credit = Credit()
                    credit.id = c_row.id
                    credit.credit = c_row.credit
                    word.credits.append(credit)
            except Exception as e:
                logger.error(f"Error loading credits for word {word_id}: {e}", exc_info=False)
                word.credits = []
        else: word.credits = [] # Initialize if include_credits was false or loading failed

        # Load affixations if requested
        if include_affixations:
            try:
                # Root affixations (where this word is the root)
                sql_root_affix = """
                SELECT af.id, af.affix_type, af.sources,
                       w.id as affixed_word_id, w.lemma as affixed_word_lemma, w.language_code as affixed_word_language_code
                FROM affixations af
                JOIN words w ON af.affixed_word_id = w.id
                WHERE af.root_word_id = :word_id
                """
                root_affix_result = db.session.execute(text(sql_root_affix), {"word_id": word_id}).fetchall()
                word.root_affixations = []
                for r_row in root_affix_result:
                    affix = Affixation()
                    affix.id = r_row.id
                    affix.affix_type = r_row.affix_type
                    affix.sources = r_row.sources
                    affix.root_word_id = word_id # Set the known root word ID

                    affixed_word = Word()
                    affixed_word.id = r_row.affixed_word_id
                    affixed_word.lemma = r_row.affixed_word_lemma
                    affixed_word.language_code = r_row.affixed_word_language_code
                    affix.affixed_word = affixed_word
                    word.root_affixations.append(affix)

                # Affixed affixations (where this word is the result)
                sql_affixed_affix = """
                SELECT af.id, af.affix_type, af.sources,
                       w.id as root_word_id_val, w.lemma as root_word_lemma, w.language_code as root_word_language_code
                FROM affixations af
                JOIN words w ON af.root_word_id = w.id
                WHERE af.affixed_word_id = :word_id
                """
                affixed_affix_result = db.session.execute(text(sql_affixed_affix), {"word_id": word_id}).fetchall()
                word.affixed_affixations = []
                for a_row in affixed_affix_result:
                    affix = Affixation()
                    affix.id = a_row.id
                    affix.affix_type = a_row.affix_type
                    affix.sources = a_row.sources
                    affix.affixed_word_id = word_id # Set the known affixed word ID

                    root_word_affix = Word()
                    root_word_affix.id = a_row.root_word_id_val
                    root_word_affix.lemma = a_row.root_word_lemma
                    root_word_affix.language_code = a_row.root_word_language_code
                    affix.root_word = root_word_affix
                    word.affixed_affixations.append(affix)

            except Exception as e:
                logger.error(f"Error loading affixations for word {word_id}: {e}", exc_info=False)
                word.root_affixations = []
                word.affixed_affixations = []
        else:
             word.root_affixations = []
             word.affixed_affixations = []

        # Load forms if requested
        if include_forms:
            try:
                sql_forms = "SELECT id, form, tags, is_canonical, is_primary FROM word_forms WHERE word_id = :word_id"
                form_result = db.session.execute(text(sql_forms), {"word_id": word_id}).fetchall()
                word.forms = []
                for f_row in form_result:
                    form = WordForm()
                    form.id = f_row.id
                    form.form = f_row.form
                    form.tags = f_row.tags
                    form.is_canonical = f_row.is_canonical
                    form.is_primary = f_row.is_primary
                    form.word_id = word_id
                    word.forms.append(form)
            except Exception as e:
                logger.error(f"Error loading forms for word {word_id}: {e}", exc_info=False)
                word.forms = []
        else: word.forms = []

        # Load templates if requested
        if include_templates:
            try:
                sql_templates = "SELECT id, template_name, args, expansion FROM word_templates WHERE word_id = :word_id"
                template_result = db.session.execute(text(sql_templates), {"word_id": word_id}).fetchall()
                word.templates = []
                for t_row in template_result:
                    template = WordTemplate()
                    template.id = t_row.id
                    template.template_name = t_row.template_name
                    template.args = t_row.args
                    template.expansion = t_row.expansion
                    template.word_id = word_id
                    word.templates.append(template)
            except Exception as e:
                logger.error(f"Error loading templates for word {word_id}: {e}", exc_info=False)
                word.templates = []
        else: word.templates = []


        # Initialize other relationships as empty lists if not included/loaded
        # Ensures attributes exist even if loading fails or is skipped
        if not hasattr(word, 'root_affixations'): word.root_affixations = []
        if not hasattr(word, 'affixed_affixations'): word.affixed_affixations = []
        if not hasattr(word, 'forms'): word.forms = []
        if not hasattr(word, 'templates'): word.templates = []
        # Definition relations/categories/links are initialized within the definition loop
        if not hasattr(word, 'credits'): word.credits = [] # Ensure credits is initialized


        # --- Add loading logic for Affixations, Forms, Templates if needed ---
        # Remember to add try/except blocks if you implement them

        # Cache the result if we have a cache client
        if cache_client and cache_key:
            try:
                # Detach the object from the session before pickling to avoid session-related issues
                temp_word = db.session.merge(word) # Merge ensures it's in session
                db.session.expunge(temp_word)
                pickled_word = pickle.dumps(temp_word)
                # Set with a reasonable timeout and handle potential storage failures
                cache_set_success = cache_client.set(cache_key, pickled_word, timeout=600)  # Cache for 10 minutes
                if not cache_set_success and hasattr(cache_client, 'last_error'):
                    logger.warning(f"Cache storage failed for word_id={word_id}: {getattr(cache_client, 'last_error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"Cache storage error for word_id={word_id}: {e}")
                # Cache errors shouldn't affect the API response

        # Return the populated word object
        return word
    except ValueError as ve:
        # Handle the specific case where the word ID itself wasn't found
        logger.error(f"Value error in _fetch_word_details for word_id {word_id}: {ve}")
        # Preserve original error context with raise from
        raise ValueError(f"Word with ID {word_id} not found") from ve
    except SQLAlchemyError as db_error:
        logger.error(f"Database error in _fetch_word_details for word_id {word_id}: {db_error}", exc_info=True)
        # Add more specific context while preserving original error
        raise SQLAlchemyError(f"Database error retrieving details for word ID {word_id}") from db_error
    except Exception as e:
        logger.error(f"Unexpected error in _fetch_word_details for word_id {word_id}: {str(e)}", exc_info=True)
        # Preserve original error context for easier debugging
        raise Exception(f"Failed to retrieve details for word ID {word_id}") from e

# Add Baybayin-specific endpoints
@bp.route("/baybayin/search", methods=["GET"])
def search_baybayin():
    """
    Search for words with specific Baybayin characters.
    Supports partial matching on Baybayin forms.
    """
    API_REQUESTS.labels(endpoint="search_baybayin", method="GET").inc()
    start_time = time.time()
    
    query = request.args.get("query", "")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    language_code = request.args.get("language_code")
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
        
    # Ensure the query contains at least one Baybayin character
    if not re.search(r'[\u1700-\u171F]', query):
        return jsonify({"error": "Query must contain at least one Baybayin character"}), 400
    
    try:
        # Build the base SQL query
        sql = """
        SELECT w.id, w.lemma, w.language_code, w.baybayin_form, w.pos, w.completeness_score
        FROM words w
        WHERE w.has_baybayin = TRUE 
        AND w.baybayin_form ILIKE :query_pattern
        """
        
        # Add language filter if specified
        if language_code:
            sql += " AND w.language_code = :language_code"
        
        # Add ordering and pagination
        sql += """
        ORDER BY w.lemma DESC, w.lemma
        LIMIT :limit OFFSET :offset
        """
        
        # Count total results (base query without limit/offset)
        count_sql = """
        SELECT COUNT(*)
        FROM words w
        WHERE w.has_baybayin = TRUE 
        AND w.baybayin_form ILIKE :query_pattern
        """
        
        if language_code:
            count_sql += " AND w.language_code = :language_code"
        
        # Execute the queries
        params = {
            "query_pattern": f"%{query}%", 
            "limit": limit, 
            "offset": offset
        }
        
        if language_code:
            params["language_code"] = language_code
        
        count_result = db.session.execute(text(count_sql), params).scalar()
        
        if count_result == 0:
            execution_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="search_baybayin").observe(execution_time)
            return jsonify({"count": 0, "results": []})
        
        query_result = db.session.execute(text(sql), params)
        
        # Format the results
        results = []
        for row in query_result:
            results.append({
                "id": row.id,
                "lemma": row.lemma,
                "language_code": row.language_code,
                "baybayin_form": row.baybayin_form,
                "pos": row.pos,
                "completeness_score": row.completeness_score
            })
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search_baybayin").observe(execution_time)
        
        return jsonify({
            "count": count_result,
            "results": results
        })
    
    except Exception as e:
        logger.error(f"Error in baybayin search: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="search_baybayin", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred while searching"}), 500


@bp.route("/baybayin/statistics", methods=["GET"])
def get_baybayin_statistics():
    """
    Get detailed statistics about Baybayin usage in the dictionary.
    Includes character frequency, language distribution, and completeness metrics.
    """
    API_REQUESTS.labels(endpoint="get_baybayin_statistics", method="GET").inc()
    start_time = time.time()
    
    try:
        # 1. Overall Baybayin statistics
        sql_overview = """
        SELECT 
            COUNT(*) as total_words,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) as with_baybayin,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as percentage
        FROM words
        """
        
        # 2. Baybayin by language
        sql_by_language = """
        SELECT 
            language_code,
            COUNT(*) as total_words,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) as with_baybayin,
            SUM(CASE WHEN has_baybayin = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as percentage
        FROM words
        GROUP BY language_code
        ORDER BY with_baybayin DESC
        """
        
        # 3. Baybayin character frequency
        sql_char_frequency = """
        WITH characters AS (
            SELECT w.id, w.language_code, 
                   regexp_split_to_table(w.baybayin_form, '') as character
            FROM words w
            WHERE w.has_baybayin = TRUE AND w.baybayin_form IS NOT NULL
        )
        SELECT 
            character, 
            COUNT(*) as frequency,
            language_code
        FROM characters
        WHERE character ~ '[\u1700-\u171F]'
        GROUP BY character, language_code
        ORDER BY frequency DESC
        """
        
        # 4. Average completeness score for words with Baybayin
        sql_completeness = """
        SELECT 
            AVG(completeness_score) as avg_score_with_baybayin,
            (SELECT AVG(completeness_score) FROM words WHERE has_baybayin = FALSE) as avg_score_without_baybayin
        FROM words
        WHERE has_baybayin = TRUE
        """
        
        # Execute all queries
        overview = db.session.execute(text(sql_overview)).fetchone()
        by_language = db.session.execute(text(sql_by_language)).fetchall()
        char_frequency = db.session.execute(text(sql_char_frequency)).fetchall()
        completeness = db.session.execute(text(sql_completeness)).fetchone()
        
        # Format results
        result = {
            "overview": {
                "total_words": overview.total_words,
                "with_baybayin": overview.with_baybayin,
                "percentage": float(overview.percentage) if overview.percentage else 0
            },
            "by_language": [
                {
                    "language_code": row.language_code,
                    "total_words": row.total_words,
                    "with_baybayin": row.with_baybayin,
                    "percentage": float(row.percentage) if row.percentage else 0
                }
                for row in by_language
            ],
            "character_frequency": {
                row.language_code: {
                    row.character: row.frequency
                    for char_row in char_frequency if char_row.language_code == row.language_code
                }
                for row in by_language if row.with_baybayin > 0
            },
            "completeness": {
                "with_baybayin": float(completeness.avg_score_with_baybayin) if completeness.avg_score_with_baybayin else 0,
                "without_baybayin": float(completeness.avg_score_without_baybayin) if completeness.avg_score_without_baybayin else 0
            }
        }
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="get_baybayin_statistics").observe(execution_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating Baybayin statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_baybayin_statistics", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred while generating statistics"}), 500


@bp.route("/baybayin/convert", methods=["POST"])
def convert_to_baybayin():
    """
    Convert romanized text to Baybayin script.
    Looks up words in the dictionary and uses their Baybayin form when available.
    For unknown words, applies conversion rules based on phonetic patterns.
    """
    API_REQUESTS.labels(endpoint="convert_to_baybayin", method="POST").inc()
    start_time = time.time()
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Text parameter is required"}), 400
    
    input_text = data.get("text", "")
    language_code = data.get("language_code", "fil")  # Default to Filipino
    
    if not input_text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        # Tokenize the text
        words = re.findall(r'\b\w+\b', input_text.lower())
        
        # Query the database for known words
        if words:
            placeholders = ", ".join([f":word{i}" for i in range(len(words))])
            params = {f"word{i}": word for i, word in enumerate(words)}
            params["language_code"] = language_code
            
            sql = f"""
            SELECT lemma, baybayin_form
            FROM words
            WHERE lemma IN ({placeholders})
            AND language_code = :language_code
            AND has_baybayin = TRUE
            """
            
            word_mappings = {}
            results = db.session.execute(text(sql), params)
            
            for row in results:
                word_mappings[row.lemma.lower()] = row.baybayin_form
        
        # Process the text
        result_text = input_text
        for word in sorted(words, key=len, reverse=True):  # Process longer words first
            if word.lower() in word_mappings:
                # Replace the word with its Baybayin form
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                result_text = pattern.sub(word_mappings[word.lower()], result_text)
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="convert_to_baybayin").observe(execution_time)
        
        return jsonify({
            "original_text": input_text,
            "baybayin_text": result_text,
            "conversion_rate": len([w for w in words if w.lower() in word_mappings]) / len(words) if words else 0
        })
        
    except Exception as e:
        logger.error(f"Error converting to Baybayin: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="convert_to_baybayin", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred during conversion"}), 500


# Enhanced search with additional filtering capabilities
@bp.route("/search/advanced", methods=["GET"])
def advanced_search():
    """
    Advanced search with additional filtering capabilities.
    Supports complex filtering on multiple fields and returns detailed metrics.
    """
    API_REQUESTS.labels(endpoint="advanced_search", method="GET").inc()
    start_time = time.time()
    
    # Get query parameters with defaults
    query = request.args.get("query", "")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    include_details = request.args.get("include_details", "true").lower() == "true"
    
    # Additional filters
    min_completeness = request.args.get("min_completeness", type=float)
    max_completeness = request.args.get("max_completeness", type=float)
    date_added_from = request.args.get("date_added_from")
    date_added_to = request.args.get("date_added_to")
    date_modified_from = request.args.get("date_modified_from")
    date_modified_to = request.args.get("date_modified_to")
    min_definition_count = request.args.get("min_definition_count", type=int)
    max_definition_count = request.args.get("max_definition_count", type=int)
    min_relation_count = request.args.get("min_relation_count", type=int)
    max_relation_count = request.args.get("max_relation_count", type=int)
    has_etymology = request.args.get("has_etymology")
    
    # Extract filter parameters using existing schema
    try:
        filter_schema = SearchFilterSchema()
        filter_data = filter_schema.load(request.args)
    except ValidationError as e:
        return jsonify({"error": e.messages}), 400
    
    try:
        # Start building the SQL queries
        select_clause = """
        SELECT w.id, w.lemma, w.language_code, w.pos, w.has_baybayin, w.baybayin_form, 
               w.completeness_score, w.date_created, w.date_modified
        """
        
        count_clause = "SELECT COUNT(*)"
        
        from_clause = """
        FROM words w
        """
        
        where_conditions = []
        params = {}
        
        # Add basic search condition from original implementation
        if query:
            where_conditions.append("""
            (
                w.lemma ILIKE :query_pattern
                OR w.normalized_lemma ILIKE :normalized_query
                OR EXISTS (
                    SELECT 1 FROM definitions d
                    WHERE d.word_id = w.id
                    AND (
                        d.definition ILIKE :query_pattern
                        OR d.example ILIKE :query_pattern
                    )
                )
            )
            """)
            params["query_pattern"] = f"%{query}%"
            params["normalized_query"] = f"%{normalize_query(query)}%"
        
        # Process filter conditions from original implementation
        # Language filter
        if filter_data.get("language"):
            where_conditions.append("w.language_code = :language_code")
            params["language_code"] = filter_data["language"]
        
        # POS filter
        if filter_data.get("pos"):
            where_conditions.append("w.pos = :pos")
            params["pos"] = filter_data["pos"]
        
        # Has baybayin filter
        if filter_data.get("has_baybayin") is not None:
            if filter_data["has_baybayin"]:
                where_conditions.append("w.has_baybayin = TRUE AND w.baybayin_form IS NOT NULL AND TRIM(w.baybayin_form) != ''")
            else:
                where_conditions.append("(w.has_baybayin = FALSE OR w.baybayin_form IS NULL)")
        
        # Process new advanced filters
        # Completeness range
        if min_completeness is not None:
            where_conditions.append("w.completeness_score >= :min_completeness")
            params["min_completeness"] = min_completeness
        
        if max_completeness is not None:
            where_conditions.append("w.completeness_score <= :max_completeness")
            params["max_completeness"] = max_completeness
        
        # Date ranges
        if date_added_from:
            where_conditions.append("w.date_created >= :date_added_from")
            params["date_added_from"] = date_added_from
        
        if date_added_to:
            where_conditions.append("w.date_created <= :date_added_to")
            params["date_added_to"] = date_added_to
        
        if date_modified_from:
            where_conditions.append("w.date_modified >= :date_modified_from")
            params["date_modified_from"] = date_modified_from
        
        if date_modified_to:
            where_conditions.append("w.date_modified <= :date_modified_to")
            params["date_modified_to"] = date_modified_to
        
        # Definition count
        if min_definition_count is not None:
            from_clause += """
            LEFT JOIN (
                SELECT word_id, COUNT(*) as def_count
                FROM definitions
                GROUP BY word_id
            ) def_counts ON def_counts.word_id = w.id
            """
            where_conditions.append("def_counts.def_count >= :min_definition_count")
            params["min_definition_count"] = min_definition_count
        
        if max_definition_count is not None:
            if "def_counts" not in from_clause:
                from_clause += """
                LEFT JOIN (
                    SELECT word_id, COUNT(*) as def_count
                    FROM definitions
                    GROUP BY word_id
                ) def_counts ON def_counts.word_id = w.id
                """
            where_conditions.append("def_counts.def_count <= :max_definition_count")
            params["max_definition_count"] = max_definition_count
        
        # Relation count
        if min_relation_count is not None or max_relation_count is not None:
            from_clause += """
            LEFT JOIN (
                SELECT word_id, COUNT(*) as rel_count
                FROM (
                    SELECT source_id as word_id FROM word_relations
                    UNION ALL
                    SELECT target_id as word_id FROM word_relations
                ) all_relations
                GROUP BY word_id
            ) rel_counts ON rel_counts.word_id = w.id
            """
            
            if min_relation_count is not None:
                where_conditions.append("rel_counts.rel_count >= :min_relation_count")
                params["min_relation_count"] = min_relation_count
            
            if max_relation_count is not None:
                where_conditions.append("rel_counts.rel_count <= :max_relation_count")
                params["max_relation_count"] = max_relation_count
        
        # Has etymology filter
        if has_etymology is not None:
            has_etym = has_etymology.lower() == "true"
            from_clause += """
            LEFT JOIN etymologies e ON e.word_id = w.id
            """
            if has_etym:
                where_conditions.append("e.id IS NOT NULL")
            else:
                where_conditions.append("e.id IS NULL")
        
        # Build the final queries
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        count_query = f"{count_clause} {from_clause} WHERE {where_clause}"
        
        main_query = f"""
        {select_clause} {from_clause}
        WHERE {where_clause}
        ORDER BY w.lemma DESC, w.lemma
        LIMIT :limit OFFSET :offset
        """
        
        params["limit"] = limit
        params["offset"] = offset
        
        # Execute queries
        count_result = db.session.execute(text(count_query), params).scalar()
        
        result_dict = {
            "count": count_result,
            "results": []
        }
        
        if count_result == 0 and not min_completeness:
            execution_time = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="advanced_search").observe(execution_time)
            return jsonify(result_dict)
        
        query_result = db.session.execute(text(main_query), params)
        
        # Process results
        words = []
        for row in query_result:
            word_data = {
                "id": row.id,
                "lemma": row.lemma,
                "language_code": row.language_code,
                "pos": row.pos,
                "has_baybayin": row.has_baybayin,
                "baybayin_form": row.baybayin_form,
                "completeness_score": row.completeness_score,
                "date_created": row.date_created.isoformat() if row.date_created else None,
                "date_modified": row.date_modified.isoformat() if row.date_modified else None
            }
            
            # Add detailed information if requested
            if include_details:
                # Get definitions
                def_sql = """
                SELECT id, sense_number, definition, example, usage_notes, source, date_created
                FROM definitions
                WHERE word_id = :word_id
                ORDER BY sense_number
                """
                definitions = db.session.execute(text(def_sql), {"word_id": row.id}).fetchall()
                
                word_data["definitions"] = [
                    {
                        "id": d.id,
                        "sense_number": d.sense_number,
                        "definition": d.definition,
                        "example": d.example,
                        "usage_notes": d.usage_notes,
                        "source": d.source,
                        "date_created": d.date_created.isoformat() if d.date_created else None
                    }
                    for d in definitions
                ]
                
                # Get relation counts
                rel_sql = """
                SELECT 
                    (SELECT COUNT(*) FROM word_relations WHERE source_id = :word_id) as outgoing,
                    (SELECT COUNT(*) FROM word_relations WHERE target_id = :word_id) as incoming
                """
                rel_counts = db.session.execute(text(rel_sql), {"word_id": row.id}).fetchone()
                
                word_data["relation_counts"] = {
                    "outgoing": rel_counts.outgoing,
                    "incoming": rel_counts.incoming,
                    "total": rel_counts.outgoing + rel_counts.incoming
                }
                
                # Check for etymology
                etym_sql = "SELECT COUNT(*) FROM etymologies WHERE word_id = :word_id"
                has_etym = db.session.execute(text(etym_sql), {"word_id": row.id}).scalar() > 0
                
                word_data["has_etymology"] = has_etym
            
            words.append(word_data)
        
        result_dict["results"] = words
        
        execution_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="advanced_search").observe(execution_time)
        
        return jsonify(result_dict)
    
    except Exception as e:
        logger.error(f"Error in advanced search: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="advanced_search", error_type=type(e).__name__).inc()
        return jsonify({"error": "An error occurred while searching"}), 500
        