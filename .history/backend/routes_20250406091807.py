"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, Pronunciation, Credit,
    WordForm, WordTemplate, DefinitionCategory, DefinitionLink, DefinitionRelation
)
from backend.database import db, cached_query
from backend.dictionary_manager import (
    normalize_lemma, extract_etymology_components, extract_language_codes,
    RelationshipType, RelationshipCategory, BaybayinRomanizer
)
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict
import logging
from sqlalchemy.exc import SQLAlchemyError
from flask_graphql import GraphQLView
import time
import random # Add random import
from flask_limiter.util import get_remote_address # Import limiter utility
from backend.extensions import limiter # Import the limiter instance from extensions
# from backend.search_tasks import log_search_query
# from backend.utils.normalization import normalize_tagalog
# from backend.utils.cache_helpers import invalidate_word_cache
# from backend.utils.baybayin_utils import baybayin_similarity
# from backend.utils.query_helpers import build_search_query, apply_filters, apply_sorting
# Import schemas that will be used

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
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True)
    metadata = fields.Dict()
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
    
    # Relationships
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_default=None)
    standardized_pos = fields.Nested('PartOfSpeechSchema', dump_default=None)
    categories = fields.List(fields.Nested('DefinitionCategorySchema', exclude=('definition',)))
    links = fields.List(fields.Nested('DefinitionLinkSchema', exclude=('definition',)))
    
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
    link_type = fields.Str(required=True)
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
    verified = fields.Bool(dump_default=False)
    verification_date = fields.DateTime(dump_only=True)
    verification_notes = fields.String()
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

# ------------------------------------

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    mode = fields.Str(validate=validate.OneOf([
        'all', 'exact', 'phonetic', 'baybayin', 'fuzzy', 'etymology',
        'semantic', 'root', 'affixed', 'prefix'
    ]), dump_default='all', load_default='all')
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(validate=validate.OneOf([
        'n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix',
        'lig', 'part', 'num', 'expr', 'punc', 'idm', 'col', 'tx', 'unc'
    ]), dump_default=None, load_default=None)
    
    # Data inclusion flags
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_etymology = fields.Bool(dump_default=True, load_default=True)
    include_pronunciation = fields.Bool(dump_default=True, load_default=True)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_examples = fields.Bool(dump_default=True, load_default=True)
    include_usage = fields.Bool(dump_default=True, load_default=True)
    include_baybayin = fields.Bool(dump_default=True, load_default=True)
    include_metadata = fields.Bool(dump_default=True, load_default=True)
    include_full = fields.Bool(dump_default=False, load_default=False)
    include_forms = fields.Bool(dump_default=False, load_default=False)
    include_templates = fields.Bool(dump_default=False, load_default=False)
    include_definition_relations = fields.Bool(dump_default=False, load_default=False)
    
    # Filter parameters
    verified = fields.Bool(dump_default=None, load_default=None)
    has_etymology = fields.Bool(dump_default=None, load_default=None)
    has_pronunciation = fields.Bool(dump_default=None, load_default=None)
    has_baybayin = fields.Bool(dump_default=None, load_default=None)
    has_forms = fields.Bool(dump_default=None, load_default=None)
    has_templates = fields.Bool(dump_default=None, load_default=None)
    sort = fields.Str(validate=validate.OneOf([
        'relevance', 'alphabetical', 'created', 'updated',
        'quality', 'frequency', 'complexity'
    ]), dump_default='relevance', load_default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), dump_default='desc', load_default='desc')
    limit = fields.Int(validate=validate.Range(min=1, max=100), dump_default=20, load_default=20)
    offset = fields.Int(validate=validate.Range(min=0), dump_default=0, load_default=0)

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
            load_options.append(selectinload(Word.definitions).selectinload(Definition.standardized_pos))
            load_options.append(selectinload(Word.definitions).selectinload(Definition.links))
            load_options.append(selectinload(Word.definitions).selectinload(Definition.categories))
        
        if include_etymologies:
            load_options.append(selectinload(Word.etymologies))
            
        if include_pronunciations:
            load_options.append(selectinload(Word.pronunciations))
            
        if include_credits:
            load_options.append(selectinload(Word.credits))
            
        if include_relations:
            load_options.append(selectinload(Word.outgoing_relations).selectinload(Relation.target_word))
            load_options.append(selectinload(Word.incoming_relations).selectinload(Relation.source_word))
            
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
            load_options.append(selectinload(Word.definition_relations))
            load_options.append(selectinload(Word.related_definitions))
            
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
    """Search for words with optimized performance for high traffic."""
    try:
        # --- Start: Validate input using SearchQuerySchema ---
        search_schema = SearchQuerySchema()
        try:
            # Load and validate query parameters from request.args
            search_args = search_schema.load(request.args)
        except ValidationError as err:
            logger.warning("Search query validation failed", errors=err.messages)
            # Return validation errors to the client
            return jsonify({"error": "Invalid search parameters", "details": err.messages}), 400
        # --- End: Validate input using SearchQuerySchema ---

        # Track request metrics
        REQUEST_COUNT.inc()
        API_REQUESTS.labels(endpoint="search", method="GET").inc()
        start_time = time.time()
        
        # Use validated arguments from the loaded schema data
        query = search_args['q']  # 'q' is required by the schema
        mode = search_args['mode'] # has default in schema
        language = search_args.get('language', 'tl') # Use .get with default, as schema might allow None
        pos = search_args.get('pos') # Use .get, as it's optional
        limit = search_args['limit'] # has default in schema
        offset = search_args['offset'] # has default in schema
        
        # Get filter parameters
        verified = search_args.get('verified')
        has_etymology = search_args.get('has_etymology')
        has_pronunciation = search_args.get('has_pronunciation')
        has_baybayin = search_args.get('has_baybayin')
        has_forms = search_args.get('has_forms')
        has_templates = search_args.get('has_templates')
        has_definition_relations = search_args.get('has_definition_relations')
        min_completeness = search_args.get('min_completeness')

        # Handle include_full separately for now, or add it to the schema
        include_full = request.args.get("include_full", "false").lower() == "true"
        
        # Add a hard limit on offset to prevent excessive deep pagination
        if offset > 1000:
            return jsonify({
                "error": "Pagination limit exceeded",
                "message": "Please use a more specific search query instead of deep pagination"
            }), 400
            
        # Add query execution timeout to prevent long-running queries
        # Consider setting this globally or per-request based on configuration
        try:
            db.session.execute(text("SET statement_timeout TO '3000'"))  # 3 seconds timeout
        except SQLAlchemyError as e:
            logger.warning(f"Could not set statement timeout: {e}")
        
        # Build the base query - use normalized_query for better index usage
        normalized_query = normalize_lemma(query)
        
        # Check for Baybayin query
        has_baybayin_query = any(0x1700 <= ord(c) <= 0x171F for c in query)
        baybayin_filter = None
        
        if has_baybayin_query:
            romanizer = BaybayinRomanizer()
            romanized_query = romanizer.romanize(query) if romanizer.validate_text(query) else ""
            baybayin_filter = Word.baybayin_form.ilike(f"%{query}%")
            if romanized_query:
                normalized_query = normalize_lemma(romanized_query)
        
        # Use direct SQL for count query for better performance
        count_sql = """
        SELECT COUNT(*) 
        FROM words w
        """
        
        # Add joins for filtering by related entities if needed
        if pos:
            count_sql += """
            JOIN definitions d ON w.id = d.word_id
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            """
            
        if has_etymology:
            count_sql += " JOIN etymologies e ON w.id = e.word_id"
            
        if has_pronunciation:
            count_sql += " JOIN pronunciations pr ON w.id = pr.word_id"
            
        if has_forms:
            count_sql += " JOIN word_forms wf ON w.id = wf.word_id"
            
        if has_templates:
            count_sql += " JOIN word_templates wt ON w.id = wt.word_id"
            
        if has_definition_relations:
            count_sql += " JOIN definition_relations dr ON w.id = dr.word_id"
            
        count_sql += " WHERE "
        
        # Optimize the query based on mode
        if mode == "exact":
            count_sql += "w.normalized_lemma = :normalized_query"
            if has_baybayin_query and baybayin_filter:
                count_sql += " OR w.baybayin_form = :query"
                
        elif mode == "prefix":
            count_sql += "w.normalized_lemma LIKE :prefix_query"
        
        elif mode == "baybayin":
            if has_baybayin_query:
                count_sql += "w.baybayin_form ILIKE :baybayin_query"
            else:
                count_sql += "w.has_baybayin = TRUE AND w.normalized_lemma ILIKE :prefix_query"
                
        else:  # Default "all" mode with optimized filter
            count_sql += """(
                w.normalized_lemma ILIKE :contains_query 
                OR w.lemma ILIKE :contains_query
            )"""
        
        # Add language filter
        if language and language != "all":
            count_sql += " AND w.language_code = :language"
        
        # Add part of speech filter
        if pos:
            count_sql += " AND p.code = :pos"
            
        # Add verified filter
        if verified is not None:
            count_sql += " AND w.verified = :verified"
            
        # Add baybayin filter (when not from the search query)
        if has_baybayin is not None and not has_baybayin_query:
            count_sql += " AND w.has_baybayin = :has_baybayin"
            
        # Add completeness filter
        if min_completeness is not None:
            count_sql += " AND w.completeness_score >= :min_completeness"
        
        # Prepare query parameters with percent wildcards
        count_params = {
            "query": query,
            "normalized_query": normalized_query,
            "prefix_query": f"{normalized_query}%",
            "contains_query": f"%{normalized_query}%",
            "baybayin_query": f"%{query}%" if has_baybayin_query else None,
            "language": language if language and language != "all" else None,
            "pos": pos,
            "verified": verified,
            "has_baybayin": has_baybayin,
            "min_completeness": min_completeness
        }
        
        # Execute count query with timeout protection
        try:
            total_count = db.session.execute(text(count_sql), count_params).scalar() or 0
        except Exception as e:
            logger.warning(f"Count query timed out, using estimate: {e}")
            total_count = 100  # Fallback to estimate
        
        # If no results found, return early
        if total_count == 0:
            return jsonify({
                "count": 0,
                "offset": offset,
                "limit": limit,
                "query": query,
                "mode": mode,
                "language": language,
                "results": []
            })
            
        # Build main query - similar structure but with ordering, limit, offset
        # Fetch only IDs first to keep the initial query fast
        main_sql = "SELECT w.id FROM words w "
        
        # Add joins for filtering by related entities if needed
        if pos:
            main_sql += """
            JOIN definitions d ON w.id = d.word_id
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            """
            
        if has_etymology:
            main_sql += " JOIN etymologies e ON w.id = e.word_id"
            
        if has_pronunciation:
            main_sql += " JOIN pronunciations pr ON w.id = pr.word_id"
            
        if has_forms:
            main_sql += " JOIN word_forms wf ON w.id = wf.word_id"
            
        if has_templates:
            main_sql += " JOIN word_templates wt ON w.id = wt.word_id"
            
        if has_definition_relations:
            main_sql += " JOIN definition_relations dr ON w.id = dr.word_id"
            
        # Add WHERE clause
        main_sql += " WHERE "
        
        # Reuse the same conditions as count query
        if mode == "exact":
            main_sql += "w.normalized_lemma = :normalized_query"
            if has_baybayin_query and baybayin_filter:
                main_sql += " OR w.baybayin_form = :query"
                
        elif mode == "prefix":
            main_sql += "w.normalized_lemma LIKE :prefix_query"
            
        elif mode == "baybayin":
            if has_baybayin_query:
                main_sql += "w.baybayin_form ILIKE :baybayin_query"
            else:
                main_sql += "w.has_baybayin = TRUE AND w.normalized_lemma ILIKE :prefix_query"
                
        else:  # Default "all" mode with optimized filter
            main_sql += """(
                w.normalized_lemma ILIKE :contains_query 
                OR w.lemma ILIKE :contains_query
            )"""
        
        # Add language filter
        if language and language != "all":
            main_sql += " AND w.language_code = :language"
            
        # Add part of speech filter
        if pos:
            main_sql += " AND p.code = :pos"
            
        # Add verified filter
        if verified is not None:
            main_sql += " AND w.verified = :verified"
            
        # Add baybayin filter (when not from the search query)
        if has_baybayin is not None and not has_baybayin_query:
            main_sql += " AND w.has_baybayin = :has_baybayin"
            
        # Add completeness filter
        if min_completeness is not None:
            main_sql += " AND w.completeness_score >= :min_completeness"
            
        # Add ORDER BY based on sort parameter
        sort_field = search_args.get('sort', 'relevance')
        sort_order = search_args.get('order', 'desc')
        
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
        elif sort_field == 'alphabetical':
            main_sql += f" ORDER BY w.lemma {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'created':
            main_sql += f" ORDER BY w.created_at {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'updated':
            main_sql += f" ORDER BY w.updated_at {'DESC' if sort_order == 'desc' else 'ASC'}"
        elif sort_field == 'completeness':
            main_sql += f" ORDER BY w.completeness_score {'DESC' if sort_order == 'desc' else 'ASC'}"
        else:
            # Default to relevance if sort field not recognized
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
        
        # Add LIMIT and OFFSET
        main_sql += " LIMIT :limit OFFSET :offset"
        
        # Add limit and offset to params
        main_params = dict(count_params)
        main_params["limit"] = limit
        main_params["offset"] = offset
        
        # Execute main query to get only the relevant Word IDs
        try:
            word_ids_result = db.session.execute(text(main_sql), main_params).fetchall()
            word_ids = [row[0] for row in word_ids_result] # Extract IDs from result rows
        except Exception as e:
            logger.error(f"Search query error fetching IDs: {e}", exc_info=True)
            return jsonify({
                "error": "Search query failed during ID retrieval, please refine your search",
                "count": total_count,
                "offset": offset,
                "limit": limit,
                "query": query,
                "results": []
            }), 500
        
        # Prepare result structure
        result = {
            "count": total_count,
            "offset": offset,
            "limit": limit,
            "query": query,
            "mode": mode,
            "language": language,
            "pos": pos,
            "results": []
        }
        
        # If no IDs found, return early
        if not word_ids:
            logger.debug(f"No matching word IDs found for search query: '{query}'")
            return jsonify(result)

        # --- Refactored section for include_full=True and include_full=False ---
        if include_full:
            # Fetch full Word objects with relationships efficiently for the found IDs
            # Use eager loading options within the query
            include_options = []
            
            if search_args.get('include_definitions', True):
                include_options.append(selectinload(Word.definitions).selectinload(Definition.standardized_pos))
                
            if search_args.get('include_etymology', True):
                include_options.append(selectinload(Word.etymologies))
                
            if search_args.get('include_pronunciation', True):
                include_options.append(selectinload(Word.pronunciations))
                
            if search_args.get('include_relations', True):
                include_options.append(selectinload(Word.outgoing_relations).selectinload(Relation.target_word))
                include_options.append(selectinload(Word.incoming_relations).selectinload(Relation.source_word))
                
            if search_args.get('include_forms', False):
                include_options.append(selectinload(Word.forms))
                
            if search_args.get('include_templates', False):
                include_options.append(selectinload(Word.templates))
                
            if search_args.get('include_definition_relations', False):
                include_options.append(selectinload(Word.definition_relations))
                include_options.append(selectinload(Word.related_definitions))
            
            full_words_query = Word.query
            
            if include_options:
                full_words_query = full_words_query.options(*include_options)
                
            full_words_query = full_words_query.filter(Word.id.in_(word_ids))

            # Preserve the order from the initial search
            # Create a mapping of ID to its order index
            id_order_map = {id_val: index for index, id_val in enumerate(word_ids)}
            # Fetch words and sort them based on the initial order
            full_words = sorted(full_words_query.all(), key=lambda w: id_order_map.get(w.id, float('inf')))

            # Serialize using WordSchema
            word_schema = WordSchema(many=True)
            result["results"] = word_schema.dump(full_words)

        else:
            # For simplified results, fetch basic data for the found IDs
            # This query could potentially be combined with the ID fetching step if needed,
            # but separating keeps the initial ID query potentially faster.
            basic_words_query = Word.query.with_entities(
                Word.id, Word.lemma, Word.normalized_lemma, Word.language_code,
                Word.has_baybayin, Word.baybayin_form, Word.romanized_form,
                Word.root_word_id, Word.verified, Word.completeness_score
            ).filter(Word.id.in_(word_ids))

            # Preserve order similar to the include_full case
            id_order_map = {id_val: index for index, id_val in enumerate(word_ids)}
            basic_words_data = sorted(basic_words_query.all(), key=lambda w: id_order_map.get(w.id, float('inf')))

            # Manually build the simplified dictionary list
            result["results"] = [
                {
                    "id": w.id,
                    "lemma": w.lemma,
                    "normalized_lemma": w.normalized_lemma,
                    "language_code": w.language_code,
                    "has_baybayin": w.has_baybayin,
                    "baybayin_form": w.baybayin_form,
                    "romanized_form": w.romanized_form,
                    "root_word_id": w.root_word_id,
                    "is_root": w.root_word_id is None,
                    "verified": w.verified,
                    "completeness_score": w.completeness_score
                }
                for w in basic_words_data
            ]
        
        # Add the applied filters to the response
        result["filters"] = {
            "verified": verified,
            "has_etymology": has_etymology,
            "has_pronunciation": has_pronunciation,
            "has_baybayin": has_baybayin,
            "has_forms": has_forms,
            "has_templates": has_templates,
            "has_definition_relations": has_definition_relations,
            "min_completeness": min_completeness
        }
        
        # Record request latency
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search").observe(request_time)
        
        return jsonify(result)
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
        verified = request.args.get("verified", "false").lower() == "true"
        min_definitions = int(request.args.get("min_definitions", 1)) # Default to minimum 1 definition
        min_completeness = float(request.args.get("min_completeness", 0.1)) # Add minimum completeness

        # Base query - select only the ID first for efficiency
        query = Word.query.with_entities(Word.id)

        # Apply filters
        if language:
            query = query.filter(Word.language_code == language)
        if pos_code:
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(PartOfSpeech.code == pos_code)
        if verified:
            query = query.filter(Word.verified == True)
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
        if min_completeness > 0:
            query = query.filter(Word.completeness_score >= min_completeness)

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
            "verified": verified,
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
                    "metadata": relation.metadata if hasattr(relation, 'metadata') else {}
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
                    "metadata": relation.metadata if hasattr(relation, 'metadata') else {}
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
            "derived_from": "derivational",
            "root_of": "derivational",
            "variant": "variant",
            "spelling_variant": "variant",
            "regional_variant": "variant",
            "compare_with": "usage",
            "see_also": "usage",
            "equals": "other"
        }
        
        # Categorize relationship types
        for rel_type in relation_types:
            category = category_mapping.get(rel_type, "other")
            result["categories"][category].append(rel_type)
                
            # Add description if available from RelationshipType
            if hasattr(RelationshipType, "get_description"):
                description = RelationshipType.get_description(rel_type)
                if description:
                    result["descriptions"][rel_type] = description
        
        return jsonify(result)
    
    except Exception as e:
        logger.error("Error retrieving relationship types", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

# Add schema for DefinitionRelation
class DefinitionRelationSchema(BaseSchema):
    """Schema for definition relation data."""
    relation_type = fields.Str(required=True)
    definition_id = fields.Int(required=True) 
    word_id = fields.Int(required=True)
    metadata = fields.Dict()
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