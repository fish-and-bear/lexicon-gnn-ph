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
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict

from backend.models import (
    Word, Definition, Etymology, Pronunciation, Relation, DefinitionCategory,
    DefinitionLink, DefinitionRelation, Affixation, WordForm, WordTemplate,
    PartOfSpeech, Credit
)
from backend.database import db, cached_query
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
    if isinstance(collector, MetricWrapperBase):
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
    etymology_data = fields.Dict()  # Updated to match renamed property
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
    popularity_score = fields.Float(dump_default=0.0)
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
    description = fields.Str()
    tags = fields.Dict()
    category_metadata = fields.Dict()
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
@cached_query(timeout=900, key_prefix="word_detail")  # Cache results for 15 minutes
def get_word(word: str):
    """Get detailed information about a specific word."""
    try:
        # Track metrics
        API_REQUESTS.labels(endpoint="get_word", method="GET").inc()

        # Parse query parameters for includes
        include_definitions = request.args.get("include_definitions", "true").lower() == "true"
        include_etymologies = request.args.get("include_etymologies", "true").lower() == "true"
        include_pronunciations = request.args.get("include_pronunciations", "true").lower() == "true"
        include_credits = request.args.get("include_credits", "true").lower() == "true"
        include_relations = request.args.get("include_relations", "true").lower() == "true"
        include_affixations = request.args.get("include_affixations", "true").lower() == "true"
        include_root = request.args.get("include_root", "true").lower() == "true"
        include_derived = request.args.get("include_derived", "true").lower() == "true"
        include_forms = request.args.get("include_forms", "true").lower() == "true"
        include_templates = request.args.get("include_templates", "true").lower() == "true"
        include_definition_relations = request.args.get("include_definition_relations", "true").lower() == "true"

        word_id = None
        # Try to parse word as an ID first
        try:
            word_id = int(word)
            logger.debug(f"Attempting to fetch word by ID: {word_id}")
        except ValueError:
            # Not an ID, look up by lemma to get the ID *without* fetching the full object yet
            word_entry_initial = Word.query.with_entities(Word.id).filter(
                or_(
                    func.lower(Word.normalized_lemma) == func.lower(normalize_lemma(word)),
                func.lower(Word.lemma) == func.lower(word)
                )
            ).first()
            if word_entry_initial:
                word_id = word_entry_initial.id
                logger.debug(f"Found word ID {word_id} by normalized lemma '{normalize_lemma(word)}' or original lemma '{word}'.")
            else:
                logger.warning(f"Word '{word}' not found by lemma or normalized lemma.")
                return jsonify({"error": f"Word '{word}' not found"}), 404

        # If we have a valid word_id, fetch details *once* using the helper
        if word_id is not None:
            word_entry = _fetch_word_details(
                word_id,
                include_definitions=include_definitions,
                include_etymologies=include_etymologies,
                include_pronunciations=include_pronunciations,
                include_credits=include_credits,
                include_relations=include_relations,
                include_affixations=include_affixations,
                include_root=include_root,
                include_derived=include_derived,
                include_forms=include_forms,
                include_templates=include_templates,
                include_definition_relations=include_definition_relations
            )
            logger.debug(f"Fetched details for word ID {word_id}. Result type: {type(word_entry)}")
        else:
            # This case should not be reached due to the lemma checks above, but included for safety
            word_entry = None

        if not word_entry:
            # Use the original input 'word' in the error message for clarity
            logger.warning(f"Word corresponding to '{word}' (ID: {word_id}) not found or details fetch failed.")
            return jsonify({"error": f"Word '{word}' not found"}), 404

        # Use WordSchema (defined in this file) to serialize the response
        schema = WordSchema()
        result = schema.dump(word_entry)

        # --- Manual exclusion based on flags (if needed, Schema might handle this via exclude/only) ---
        # This section might be redundant if _fetch_word_details correctly avoids loading
        # relationships based on flags and the schema only dumps loaded fields.
        # Keep it for now as explicit control.
        if not include_definitions:
            result.pop('definitions', None)
        if not include_etymologies:
            result.pop('etymologies', None)
        if not include_pronunciations:
            result.pop('pronunciations', None)
        if not include_credits:
            result.pop('credits', None)
        if not include_relations:
            result.pop('outgoing_relations', None)
            result.pop('incoming_relations', None)
        if not include_affixations:
            result.pop('root_affixations', None)
            result.pop('affixed_affixations', None)
        if not include_root:
            result.pop('root_word', None) # Ensure key matches schema definition
        if not include_derived:
            result.pop('derived_words', None) # Ensure key matches schema definition
        if not include_forms:
            result.pop('forms', None)
        if not include_templates:
            result.pop('templates', None)
        if not include_definition_relations:
            result.pop('definition_relations', None)
            result.pop('related_definitions', None)


        # Add data completeness manually as it uses the fetched object's state
        result["data_completeness"] = {
            "has_definitions": bool(getattr(word_entry, 'definitions', [])),
            "has_etymology": bool(getattr(word_entry, 'etymologies', [])),
            "has_pronunciations": bool(getattr(word_entry, 'pronunciations', [])),
            "has_baybayin": bool(word_entry.has_baybayin and word_entry.baybayin_form),
            "has_relations": bool(getattr(word_entry, 'outgoing_relations', []) or getattr(word_entry, 'incoming_relations', [])),
            "has_affixations": bool(getattr(word_entry, 'root_affixations', []) or getattr(word_entry, 'affixed_affixations', [])),
            "has_forms": bool(getattr(word_entry, 'forms', [])),
            "has_templates": bool(getattr(word_entry, 'templates', [])),
            "has_definition_relations": bool(getattr(word_entry, 'definition_relations', [])),
            "completeness_score": getattr(word_entry, 'completeness_score', 0) # Use the hybrid property
        }
        
        logger.debug(f"Returning result for word '{word}' using schema", data_keys=list(result.keys()))
        return jsonify(result)

    except SQLAlchemyError as e:
         # Let the SQLAlchemyError handler defined earlier catch this
        logger.error(f"Database error retrieving word '{word}'", error=str(e), exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Error retrieving word '{word}'", error=str(e), exc_info=True)
        API_ERRORS.labels(endpoint="get_word", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500


def _fetch_word_details(
    word_id,
    include_definitions=True,
    include_etymologies=False,
    include_pronunciations=False,
    include_credits=False,
    include_relations=False,
    include_affixations=False,
    include_root=False,
    include_derived=False,
    include_forms=False,
    include_templates=False,
    include_definition_relations=False
):
    """
    Helper function to fetch a word by ID with the requested level of relationship loading.
    This centralizes the logic for efficient eager loading.
    """
    # First check if it's in redis cache
    from backend.database import get_cache_client
    cache = get_cache_client()
    cache_key = f'word_details:{word_id}:{include_definitions}:{include_etymologies}:' \
                f'{include_pronunciations}:{include_credits}:{include_relations}:' \
                f'{include_affixations}:{include_root}:{include_derived}:' \
                f'{include_forms}:{include_templates}:{include_definition_relations}'
    
    try:
        cached_data = cache.get(cache_key)
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
    
    try:
        query = Word.query
        load_options = []
        
        # Configure eager loading options based on parameters
        if include_definitions:
            # Load definitions but only with the columns that exist in the database
            load_options.append(
                selectinload(Word.definitions).load_only(
                    Definition.id, Definition.word_id, Definition.definition_text,
                    Definition.original_pos, Definition.standardized_pos_id,
                    Definition.usage_notes, Definition.examples, Definition.tags,
                    Definition.sources, # Both definition_metadata and popularity_score removed
                    Definition.created_at, Definition.updated_at
                ).selectinload(Definition.standardized_pos)
            )
            
            # Skip loading links due to column mismatch
            # load_options.append(
            #     selectinload(Word.definitions).selectinload(Definition.links)
            # )
            
            load_options.append(
                selectinload(Word.definitions).selectinload(Definition.categories)
            )
        
        if include_etymologies:
            # Load etymologies but only with the columns that exist in the database
            from sqlalchemy.orm import Load
            load_options.append(
                selectinload(Word.etymologies).load_only(
                    Etymology.id, Etymology.word_id, Etymology.etymology_text,
                    Etymology.normalized_components, Etymology.etymology_structure,
                    Etymology.language_codes, Etymology.sources,
                    Etymology.created_at, Etymology.updated_at
                )
            )
            
        if include_pronunciations:
            load_options.append(selectinload(Word.pronunciations))
            
        if include_credits:
            load_options.append(selectinload(Word.credits))
            
        if include_relations:
            # Load relations but only with the columns that exist in the database
            load_options.append(
                selectinload(Word.outgoing_relations).load_only(
                    Relation.id, Relation.from_word_id, Relation.to_word_id,
                    Relation.relation_type, Relation.sources,
                    Relation.created_at, Relation.updated_at
                ).selectinload(Relation.target_word)
            )
            load_options.append(
                selectinload(Word.incoming_relations).load_only(
                    Relation.id, Relation.from_word_id, Relation.to_word_id,
                    Relation.relation_type, Relation.sources,
                    Relation.created_at, Relation.updated_at
                ).selectinload(Relation.source_word)
            )
            
        if include_affixations:
            load_options.append(selectinload(Word.root_affixations).selectinload(Affixation.affixed_word))
            load_options.append(selectinload(Word.affixed_affixations).selectinload(Affixation.root_word))
            
        if include_root:
            load_options.append(selectinload(Word.root_word))
            
        if include_derived:
            load_options.append(selectinload(Word.derived_words))
            
        if include_forms:
            load_options.append(selectinload(Word.forms))
            
        if include_templates:
            load_options.append(selectinload(Word.templates))
            
        if include_definition_relations:
            # Skip loading definition relations due to schema mismatch
            # log it so we know it was intentionally skipped
            logger.warning("Skipping definition_relations loading due to schema mismatch")
            # load_options.append(selectinload(Word.definition_relations))
            # load_options.append(selectinload(Word.related_definitions))
            
        if load_options:
            query = query.options(*load_options)
            
        word = query.get(word_id)
        
        if word:
            # Cache the word with its loaded relationships
            try:
                pickled_word = pickle.dumps(word)
                cache.set(cache_key, pickled_word, ex=600)  # Cache for 10 minutes
            except Exception as e:
                logger.warning(f"Cache storage error for word_id={word_id}: {e}")
                
        return word
        
    except SQLAlchemyError as e:
        logger.error(f"Database error in _fetch_word_details for word_id={word_id}", error=str(e))
        return None
    except Exception as e:
        logger.error(f"Unexpected error in _fetch_word_details for word_id={word_id}", error=str(e))
        return None

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
        SELECT DISTINCT ON (w.id) w.id, w.lemma, w.normalized_lemma, w.language_code, 
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
             OR LOWER(w.search_text) LIKE :contains_query
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
        
        # Add sort parameter handling
        sort_field = request.args.get('sort', 'relevance')
        sort_order = request.args.get('order', 'asc').lower()
        
        # Add appropriate ORDER BY based on sort field
        if sort_field == 'relevance':
            main_sql += """
            ORDER BY 
                CASE 
                    WHEN w.normalized_lemma = :normalized_query THEN 0
                    WHEN w.normalized_lemma LIKE :prefix_query THEN 1
                    ELSE 2 
                END,
                LENGTH(w.lemma),
                w.lemma
            """
            query_params['normalized_query'] = normalized_query
            query_params['prefix_query'] = f"{normalized_query}%"
        elif sort_field == 'alphabetical':
            main_sql += f" ORDER BY w.lemma {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'created':
            main_sql += f" ORDER BY w.created_at {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'updated':
            main_sql += f" ORDER BY w.updated_at {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'completeness':
            # Use lemma instead of completeness_score which is a hybrid property
            main_sql += f" ORDER BY w.lemma {'DESC' if sort_order == 'desc' else 'ASC'}"
        else:
            # Default to relevance
            main_sql += " ORDER BY CASE WHEN w.normalized_lemma = :normalized_query THEN 0 WHEN w.normalized_lemma LIKE :prefix_query THEN 1 ELSE 2 END, LENGTH(w.lemma), w.lemma"
            query_params['normalized_query'] = normalized_query
            query_params['prefix_query'] = f"{normalized_query}%"
            
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
        pos_code = request.args.get("pos") # Use pos_code to avoid conflict with pos variable later
        has_etymology = request.args.get("has_etymology", "false").lower() == "true"
        has_definitions = request.args.get("has_definitions", "true").lower() == "true" # Default to true to get meaningful words
        has_baybayin = request.args.get("has_baybayin", "false").lower() == "true"
        has_pronunciation = request.args.get("has_pronunciation", "false").lower() == "true"
        has_forms = request.args.get("has_forms", "false").lower() == "true"
        has_templates = request.args.get("has_templates", "false").lower() == "true"
        has_definition_relations = request.args.get("has_definition_relations", "false").lower() == "true"
        min_definitions = int(request.args.get("min_definitions", 1)) # Default to minimum 1 definition
        min_completeness = float(request.args.get("min_completeness", 0.1)) # Add minimum completeness

        # Base query - select only the ID first for efficiency
        query = Word.query.with_entities(Word.id)

        # Apply filters
        if language:
            query = query.filter(Word.language_code == language)
        if pos_code:
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(PartOfSpeech.code == pos_code)
        if has_etymology:
            # Ensure Etymology is joined correctly if not already implied
            query = query.join(Word.etymologies).group_by(Word.id).having(func.count(Etymology.id) > 0)
        if has_definitions or min_definitions > 0:
            # Ensure Definition is joined
            if pos_code is None: # Avoid double join if already joined for pos_code
                 query = query.join(Word.definitions)
            query = query.group_by(Word.id).having(func.count(Definition.id) >= min_definitions)
        if has_baybayin:
            query = query.filter(Word.has_baybayin == True, Word.baybayin_form.isnot(None))
        if has_pronunciation:
            query = query.join(Word.pronunciations).group_by(Word.id).having(func.count(Pronunciation.id) > 0)
        if has_forms:
            query = query.join(Word.forms).group_by(Word.id).having(func.count(WordForm.id) > 0)
        if has_templates:
            query = query.join(Word.templates).group_by(Word.id).having(func.count(WordTemplate.id) > 0)
        if has_definition_relations:
            query = query.join(Word.definition_relations).group_by(Word.id).having(func.count('definition_relations.id') > 0)
        # Commented out because completeness_score is a hybrid property, not a database column
        # if min_completeness > 0:
        #    query = query.filter(Word.completeness_score >= min_completeness)

        # Get a list of eligible word IDs
        eligible_ids = [item[0] for item in query.all()] # Extract IDs

        if not eligible_ids:
            logger.info("No words match the specified criteria for random selection.")
            return jsonify({"error": "No words match the specified criteria"}), 404

        # Choose a random ID from the list
        random_word_id = random.choice(eligible_ids)
        logger.debug(f"Selected random word ID: {random_word_id}")

        # Fetch full details for the selected word using the helper function
        word_with_details = _fetch_word_details(
            random_word_id,
            include_definitions=True, # Typically want full details for a random word endpoint
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

        if not word_with_details:
             logger.error(f"Failed to retrieve details for randomly selected word ID: {random_word_id}")
             # This indicates an issue if the ID was valid but fetch failed
             return jsonify({"error": "Failed to retrieve random word details"}), 500

        # Use WordSchema to serialize the result
        schema = WordSchema()
        result = schema.dump(word_with_details)

        # Add data completeness (might be redundant if already in schema)
        result["data_completeness"] = {
            "has_definitions": bool(getattr(word_with_details, 'definitions', [])),
            "has_etymology": bool(getattr(word_with_details, 'etymologies', [])),
            "has_pronunciations": bool(getattr(word_with_details, 'pronunciations', [])),
            "has_baybayin": bool(word_with_details.has_baybayin and word_with_details.baybayin_form),
            "has_relations": bool(getattr(word_with_details, 'outgoing_relations', []) or getattr(word_with_details, 'incoming_relations', [])),
            "has_affixations": bool(getattr(word_with_details, 'root_affixations', []) or getattr(word_with_details, 'affixed_affixations', [])),
            "has_forms": bool(getattr(word_with_details, 'forms', [])),
            "has_templates": bool(getattr(word_with_details, 'templates', [])),
            "has_definition_relations": bool(getattr(word_with_details, 'definition_relations', [])),
            "completeness_score": getattr(word_with_details, 'completeness_score', 0)
        }

        # Add filter criteria to response for transparency
        result["filter_criteria"] = {
            "language": language,
            "pos_code": pos_code,
            "has_etymology": has_etymology,
            "has_definitions": has_definitions,
            "has_baybayin": has_baybayin,
            "has_pronunciation": has_pronunciation,
            "has_forms": has_forms,
            "has_templates": has_templates,
            "has_definition_relations": has_definition_relations,
            "min_definitions": min_definitions,
            "min_completeness": min_completeness
        }

        logger.debug(f"Returning random word: {word_with_details.lemma} (ID: {random_word_id})")
        return jsonify(result)

    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving random word", error=str(e), exc_info=True)
        raise e # Let the handler manage it
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
def get_semantic_network(word: str):
    """Get semantic network for a word."""
    try:
        # Validate parameters
        depth = min(int(request.args.get("depth", 1)), 2)  # Max depth 2 to prevent large networks
        relation_types = request.args.get("relation_types", "").split(",") if request.args.get("relation_types") else []
        
        # Find the word entry
        word_entry = Word.query.filter(
            or_(
                Word.lemma == word,
                Word.normalized_lemma == normalize_lemma(word),
                Word.id == word if word.isdigit() else False
            )
        ).first()
        
        if not word_entry:
            return jsonify({"error": f"Word '{word}' not found"}), 404
        
        # Create network
        nodes = {}
        edges = []
        processed = set()
        
        def add_node(node_id, lemma, language_code):
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "lemma": lemma,
                    "language_code": language_code
                }
        
        def process_word(word_id, current_depth=0):
            if word_id in processed or current_depth > depth:
                return
                
            processed.add(word_id)
            
            # Get word with relations
            word_with_relations = _fetch_word_details(
                word_id,
                include_definitions=False,
                include_etymologies=False,
                include_pronunciations=False,
                include_credits=False,
                include_relations=True,
                include_affixations=False,
                include_root=False,
                include_derived=False,
                include_forms=False,
                include_templates=False,
                include_definition_relations=False
            )
            
            # Add current word as node
            add_node(word_with_relations.id, word_with_relations.lemma, word_with_relations.language_code)
            
            # Process outgoing relations
            for relation in word_with_relations.outgoing_relations:
                if relation_types and relation.relation_type not in relation_types:
                    continue
                    
                target_id = relation.target_word.id
                add_node(target_id, relation.target_word.lemma, relation.target_word.language_code)
                
                edge_id = f"{word_id}-{relation.relation_type}-{target_id}"
                edges.append({
                    "id": edge_id,
                    "source": word_id,
                    "target": target_id,
                    "type": relation.relation_type,
                    "metadata": relation.relation_data if hasattr(relation, 'relation_data') else {}
                })
                
                if current_depth < depth:
                    process_word(target_id, current_depth + 1)
            
            # Process incoming relations
            for relation in word_with_relations.incoming_relations:
                if relation_types and relation.relation_type not in relation_types:
                    continue
                    
                source_id = relation.source_word.id
                add_node(source_id, relation.source_word.lemma, relation.source_word.language_code)
                
                edge_id = f"{source_id}-{relation.relation_type}-{word_id}"
                edges.append({
                    "id": edge_id,
                    "source": source_id,
                    "target": word_id,
                    "type": relation.relation_type,
                    "metadata": relation.relation_data if hasattr(relation, 'relation_data') else {}
                })
                
                if current_depth < depth:
                    process_word(source_id, current_depth + 1)
        
        # Start processing from the initial word
        process_word(word_entry.id)
        
        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "depth": depth
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating semantic network for word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<int:word_id>/etymology/tree", methods=["GET"])
def get_etymology_tree(word_id: int):
    """Get etymology tree for a word."""
    try:
        # Validate parameters
        depth = min(int(request.args.get("depth", 2)), 3)  # Max depth 3 to prevent excessive recursion
        
        word = Word.query.get(word_id)
        if not word:
            return jsonify({"error": f"Word with ID {word_id} not found"}), 404
            
        # Helper functions for building etymology tree
        def get_word_by_lemma(lemma, language=None):
            query = Word.query.filter(Word.lemma == lemma)
            if language:
                query = query.filter(Word.language_code == language)
            return query.first()
        
        def get_etymologies(word_id):
            return Etymology.query.filter(Etymology.word_id == word_id).all()
        
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
                    components = extract_etymology_components(etym.normalized_components)
                    
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
            
        if filter_args.get('min_completeness') is not None:
            query = query.filter(Word.completeness_score >= filter_args['min_completeness'])
            
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
        
        # Apply pagination
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
        completeness_avg = db.session.query(func.avg(Word.completeness_score)).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).scalar() or 0.0
        
        results["completeness"]["average_score"] = round(float(completeness_avg), 2)
        
        # Completeness distribution
        completeness_distribution = db.session.query(
            case(
                (Word.completeness_score >= 0.9, "excellent"),
                (Word.completeness_score >= 0.7, "good"),
                (Word.completeness_score >= 0.5, "fair"),
                (Word.completeness_score >= 0.3, "poor"),
                else_="incomplete"
            ),
            func.count(Word.id)
        ).filter(
            Word.id.in_(query.with_entities(Word.id))
        ).group_by(1).all()
        
        for category, count in completeness_distribution:
            results["completeness"]["distribution"][category] = count
            results["completeness"]["percent_by_category"][category] = round((count / total_count) * 100, 1) if total_count > 0 else 0.0
        
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
        API_REQUESTS.labels(endpoint="get_parts_of_speech", method="GET").inc()
        
        # Query all parts of speech
        parts_of_speech = PartOfSpeech.query.all()
        
        # Frontend expects an array of objects with specific keys
        # Determine exactly what format the frontend expects by examining frontend code
        formatted_pos = []
        for pos in parts_of_speech:
            formatted_pos.append({
                "id": pos.id,
                "code": pos.code,
                "name": pos.name_en,  # Frontend may expect 'name' instead of 'name_en'
                "name_en": pos.name_en,
                "name_tl": pos.name_tl,
                "description": getattr(pos, 'description', ""),
                "count": 0  # Frontend might expect a count field
            })
        
        # Return the formatted result directly - matches the frontend expectation
        return jsonify(formatted_pos)
    except Exception as e:
        logger.error(f"Error retrieving parts of speech: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_parts_of_speech", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500

@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics summary."""
    try:
        API_REQUESTS.labels(endpoint="get_statistics", method="GET").inc()
        
        # Get basic counts without relying on the problematic columns
        total_words = db.session.query(func.count(Word.id)).scalar() or 0
        
        # Use a safer approach for definitions count
        total_definitions = db.session.query(func.count(Definition.id)).scalar() or 0
        
        # Continue with other counts
        total_etymologies = db.session.query(func.count(Etymology.id)).scalar() or 0
        total_relations = db.session.query(func.count(Relation.id)).scalar() or 0
        
        # Words with baybayin
        words_with_baybayin = db.session.query(func.count(Word.id)).filter(Word.has_baybayin == True).scalar() or 0
        
        # Words by language
        language_distribution = db.session.query(
            Word.language_code, func.count(Word.id)
        ).group_by(Word.language_code).all()
        
        languages = {lang: count for lang, count in language_distribution if lang}
        
        # Safer approach for POS distribution that avoids complex joins
        pos_counts = {}
        try:
            pos_list = PartOfSpeech.query.all()
            for pos in pos_list:
                count = db.session.query(func.count(Definition.id))\
                    .filter(Definition.standardized_pos_id == pos.id).scalar() or 0
                if count > 0 and pos.code:
                    pos_counts[pos.code] = count
        except Exception as e:
            logger.error(f"Error getting part of speech distribution: {str(e)}")
        
        # Format results
        results = {
            "total_words": total_words,
            "total_definitions": total_definitions,
            "total_etymologies": total_etymologies,
            "total_relations": total_relations,
            "words_with_baybayin": words_with_baybayin,
            "languages": languages,
            "parts_of_speech": pos_counts,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="get_statistics", error_type=type(e).__name__).inc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}", "type": type(e).__name__}), 500