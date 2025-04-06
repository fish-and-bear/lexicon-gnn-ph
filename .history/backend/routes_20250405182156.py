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
    PartOfSpeech, Language, Pronunciation, Credit,
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
    metadata = fields.Dict()  # Changed from pronunciation_metadata to metadata to match the model
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
    definition_text = fields.Str(required=True)
    original_pos = fields.String()
    standardized_pos_id = fields.Int()
    examples = fields.String()
    usage_notes = fields.String()
    tags = fields.String()
    # Removed word nesting as it's implicit from WordSchema
    # word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))
    standardized_pos = fields.Nested('PartOfSpeechSchema') # Keep nested POS

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Int(dump_only=True)
    code = fields.Str(required=True)
    name_en = fields.Str(required=True)
    name_tl = fields.Str(required=True)
    description = fields.String()

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

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    mode = fields.Str(validate=validate.OneOf([
        'all', 'exact', 'phonetic', 'baybayin', 'fuzzy', 'etymology',
        'semantic', 'root', 'affixed'
    ]), dump_default='all', load_default='all')
    language = fields.Str(dump_default=None, load_default=None)
    pos = fields.Str(validate=validate.OneOf([
        'n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix'
    ]), dump_default=None, load_default=None)
    include_relations = fields.Bool(dump_default=True, load_default=True)
    include_etymology = fields.Bool(dump_default=True, load_default=True)
    include_pronunciation = fields.Bool(dump_default=True, load_default=True)
    include_definitions = fields.Bool(dump_default=True, load_default=True)
    include_examples = fields.Bool(dump_default=True, load_default=True)
    include_usage = fields.Bool(dump_default=True, load_default=True)
    include_baybayin = fields.Bool(dump_default=True, load_default=True)
    include_metadata = fields.Bool(dump_default=True, load_default=True)
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
        
        # Parse query parameters
        include_definitions = request.args.get("include_definitions", "true").lower() == "true"
        include_etymologies = request.args.get("include_etymologies", "true").lower() == "true"
        include_pronunciations = request.args.get("include_pronunciations", "true").lower() == "true"
        include_credits = request.args.get("include_credits", "true").lower() == "true"
        include_relations = request.args.get("include_relations", "true").lower() == "true"
        include_affixations = request.args.get("include_affixations", "true").lower() == "true"
        include_root = request.args.get("include_root", "true").lower() == "true"
        include_derived = request.args.get("include_derived", "true").lower() == "true"
        
        # Try to parse word as an ID
        try:
            word_id = int(word)
            word_entry = _fetch_word_details(
                word_id,
                include_definitions=include_definitions,
                include_etymologies=include_etymologies,
                include_pronunciations=include_pronunciations,
                include_credits=include_credits,
                include_relations=include_relations,
                include_affixations=include_affixations,
                include_root=include_root,
                include_derived=include_derived
            )
        except ValueError:
            # Not an ID, so look up by lemma
            word_entry_initial = Word.query.filter(
                func.lower(Word.lemma) == func.lower(word)
            ).first()
            logger.debug(f"Looked up word by lemma '{word}'. Found: {bool(word_entry_initial)}")
            
            if word_entry_initial:
                word_entry = _fetch_word_details(
                    word_entry_initial.id, # Use the found ID
                    include_definitions=include_definitions,
                    include_etymologies=include_etymologies,
                    include_pronunciations=include_pronunciations,
                    include_credits=include_credits,
                    include_relations=include_relations,
                    include_affixations=include_affixations,
                    include_root=include_root,
                    include_derived=include_derived
                )
                logger.debug(f"Fetched details for word ID {word_entry_initial.id}. Result: {type(word_entry)}")
            else:
                word_entry = None # Ensure word_entry is None if initial lookup failed
        
        if not word_entry:
            logger.warning(f"Word '{word}' not found in database or details fetch failed.")
            return jsonify({"error": f"Word '{word}' not found"}), 404
        
        # Build response
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "normalized_lemma": word_entry.normalized_lemma,
            "language_code": word_entry.language_code,
            "has_baybayin": word_entry.has_baybayin,
            "baybayin_form": word_entry.baybayin_form,
            "sources": word_entry.sources,
            "created_at": word_entry.created_at.isoformat() if word_entry.created_at else None,
            "updated_at": word_entry.updated_at.isoformat() if word_entry.updated_at else None
        }
        
        # Add optional data
        if include_definitions and hasattr(word_entry, 'definitions'):
            result["definitions"] = [
                {
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "original_pos": d.original_pos,
                    "standardized_pos": {
                        "id": d.standardized_pos.id if d.standardized_pos else None,
                        "code": d.standardized_pos.code if d.standardized_pos else None,
                        "name_en": d.standardized_pos.name_en if d.standardized_pos else None,
                        "name_tl": d.standardized_pos.name_tl if d.standardized_pos else None
                    } if d.standardized_pos else None,
                    "examples": d.examples,
                    "usage_notes": d.usage_notes,
                    "sources": d.sources
                }
                for d in word_entry.definitions
            ]
                
        if include_etymologies and hasattr(word_entry, 'etymologies'):
            result["etymologies"] = [
                {
                    "id": e.id,
                    "etymology_text": e.etymology_text,
                    "normalized_components": e.normalized_components,
                    "etymology_structure": e.etymology_structure,
                    "language_codes": e.language_codes,
                    "sources": e.sources
                }
                for e in word_entry.etymologies
            ]
                
        if include_pronunciations and hasattr(word_entry, 'pronunciations'):
            result["pronunciations"] = [
                {
                    "id": p.id,
                    "type": p.type,
                    "value": p.value,
                    "metadata": p.metadata,
                    "sources": p.sources
                }
                for p in word_entry.pronunciations
            ]
                
        if include_credits and hasattr(word_entry, 'credits'):
            result["credits"] = [
                {
                    "id": c.id,
                    "credit": c.credit,
                    "sources": c.sources
                }
                for c in word_entry.credits
            ]
        
        if include_relations:
            if hasattr(word_entry, 'outgoing_relations'):
                result["outgoing_relations"] = [
                    {
                        "id": r.id,
                        "relation_type": r.relation_type,
                        "target_word": {
                            "id": r.target_word.id,
                            "lemma": r.target_word.lemma,
                            "language_code": r.target_word.language_code,
                            "has_baybayin": r.target_word.has_baybayin,
                            "baybayin_form": r.target_word.baybayin_form
                        },
                        "sources": r.sources
                    }
                    for r in word_entry.outgoing_relations
                ]
            
            if hasattr(word_entry, 'incoming_relations'):
                result["incoming_relations"] = [
                    {
                        "id": r.id,
                        "relation_type": r.relation_type,
                        "source_word": {
                            "id": r.source_word.id,
                            "lemma": r.source_word.lemma,
                            "language_code": r.source_word.language_code,
                            "has_baybayin": r.source_word.has_baybayin,
                            "baybayin_form": r.source_word.baybayin_form
                        },
                        "sources": r.sources
                    }
                    for r in word_entry.incoming_relations
                ]
        
        if include_affixations:
            if hasattr(word_entry, 'root_affixations'):
                result["root_affixations"] = [
                    {
                        "id": aff.id,
                        "affix_type": aff.affix_type,
                        "affixed_word": {
                            "id": aff.affixed_word_id,
                            "lemma": aff.affixed_lemma,
                            "language_code": aff.affixed_language_code,
                            "has_baybayin": aff.affixed_has_baybayin,
                            "baybayin_form": aff.affixed_baybayin_form
                        },
                        "sources": aff.sources
                    }
                    for aff in word_entry.root_affixations
                ]
            
            if hasattr(word_entry, 'affixed_affixations'):
                result["affixed_affixations"] = [
                    {
                        "id": aff.id,
                        "affix_type": aff.affix_type,
                        "root_word": {
                            "id": aff.root_word_id,
                            "lemma": aff.root_lemma,
                            "language_code": aff.root_language_code,
                            "has_baybayin": aff.root_has_baybayin,
                            "baybayin_form": aff.root_baybayin_form
                        },
                        "sources": aff.sources
                    }
                    for aff in word_entry.affixed_affixations
                ]
        
        if include_root and word_entry.root_word_id and hasattr(word_entry, 'root_word'):
            result["root_word"] = {
                "id": word_entry.root_word.id,
                "lemma": word_entry.root_word.lemma,
                "language_code": word_entry.root_word.language_code,
                "has_baybayin": word_entry.root_word.has_baybayin,
                "baybayin_form": word_entry.root_word.baybayin_form
            }
                
        if include_derived and hasattr(word_entry, 'derived_words'):
            result["derived_words"] = [
                {
                    "id": d.id,
                    "lemma": d.lemma,
                    "language_code": d.language_code,
                    "has_baybayin": d.has_baybayin,
                    "baybayin_form": d.baybayin_form
                }
                for d in word_entry.derived_words
            ]
        
        # Add data completeness information
        result["data_completeness"] = {
            "has_definitions": bool(getattr(word_entry, 'definitions', [])),
            "has_etymology": bool(getattr(word_entry, 'etymologies', [])),
            "has_pronunciations": bool(getattr(word_entry, 'pronunciations', [])),
            "has_baybayin": bool(word_entry.has_baybayin and word_entry.baybayin_form),
            "has_relations": bool(getattr(word_entry, 'outgoing_relations', []) or getattr(word_entry, 'incoming_relations', [])),
            "has_affixations": bool(getattr(word_entry, 'root_affixations', []) or getattr(word_entry, 'affixed_affixations', [])),
            "completeness_score": getattr(word_entry, 'completeness_score', 0)
        }
        
        logger.debug(f"Returning result for word '{word}'", data=result) # Log the final result
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error retrieving word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


def _fetch_word_details(
    word_id: int,
    include_definitions: bool = True,
    include_etymologies: bool = True,
    include_pronunciations: bool = True,
    include_credits: bool = True,
    include_relations: bool = True,
    include_affixations: bool = True,
    include_root: bool = True,
    include_derived: bool = True,
) -> Optional[Word]:
    """
    Fetches a Word object with specified related data efficiently using SQLAlchemy loading strategies.

    Args:
        word_id: The ID of the word to fetch
        include_*: Flags to determine which related data to load

    Returns:
        The Word object with eagerly loaded data, or None if not found
    """
    options = []

    # Build query options based on what's needed
    if include_definitions:
        options.append(joinedload(Word.definitions).joinedload(Definition.standardized_pos))
    if include_etymologies:
        options.append(joinedload(Word.etymologies))
    if include_pronunciations:
        options.append(joinedload(Word.pronunciations))
    if include_credits:
        options.append(joinedload(Word.credits))
    if include_relations:
        options.append(joinedload(Word.outgoing_relations).joinedload(Relation.target_word))
        options.append(joinedload(Word.incoming_relations).joinedload(Relation.source_word))
    if include_affixations:
        options.append(joinedload(Word.root_affixations).joinedload(Affixation.affixed_word))
        options.append(joinedload(Word.affixed_affixations).joinedload(Affixation.root_word))
    if include_root:
        options.append(joinedload(Word.root_word))
    if include_derived:
        options.append(joinedload(Word.derived_words))

    # Fetch word with all requested relationships
    return Word.query.options(*options).get(word_id)


@bp.route("/search", methods=["GET"])
@cached_query(timeout=300)  # Increase cache timeout to 5 minutes for search results
@limiter.limit("20 per minute", key_func=get_remote_address) # Apply rate limit
def search():
    """Search for words with optimized performance for high traffic."""
    logger.debug("Entering search function")
    try:
        # Track request metrics
        logger.debug("Tracking metrics")
        REQUEST_COUNT.inc()
        API_REQUESTS.labels(endpoint="search", method="GET").inc()
        start_time = time.time()
        
        # Validate and parse search parameters
        logger.debug("Parsing parameters")
        query = request.args.get("q", "")
        if not query or len(query.strip()) < 1:
            return jsonify({"error": "Search query required"}), 400
        
        mode = request.args.get("mode", "all")
        language = request.args.get("language", "tl")
        pos = request.args.get("pos")
        
        # Enforce stricter limits for high traffic protection
        limit = min(int(request.args.get("limit", 20)), 50)  # Max 50 results
        offset = int(request.args.get("offset", 0))
        include_full = request.args.get("include_full", "false").lower() == "true"
        
        # Add a hard limit on offset to prevent excessive deep pagination
        if offset > 1000:
            return jsonify({
                "error": "Pagination limit exceeded",
                "message": "Please use a more specific search query instead of deep pagination"
            }), 400
            
        # Add query execution timeout to prevent long-running queries
        logger.debug("Setting statement timeout")
        db.session.execute(text("SET statement_timeout TO '3000'"))  # 3 seconds timeout
        
        # Build the base query - use normalized_query for better index usage
        logger.debug("Normalizing query")
        normalized_query = normalize_lemma(query)
        
        # Check for Baybayin query
        logger.debug("Checking for Baybayin")
        has_baybayin = any(0x1700 <= ord(c) <= 0x171F for c in query)
        baybayin_filter = None
        
        if has_baybayin:
            logger.debug("Processing Baybayin query")
            romanizer = BaybayinRomanizer()
            romanized_query = romanizer.romanize(query) if romanizer.validate_text(query) else ""
            baybayin_filter = Word.baybayin_form.ilike(f"%{query}%")
            if romanized_query:
                normalized_query = normalize_lemma(romanized_query)
        
        # Use direct SQL for count query for better performance
        logger.debug("Building count SQL")
        count_sql = """
        SELECT COUNT(*) 
        FROM words w
        WHERE """
        
        # Optimize the query based on mode
        if mode == "exact":
            count_sql += "w.normalized_lemma = :normalized_query"
            if has_baybayin and baybayin_filter:
                count_sql += " OR w.baybayin_form = :query"
                
        elif mode == "prefix":
            count_sql += "w.normalized_lemma LIKE :prefix_query"
        
        elif mode == "baybayin":
            if has_baybayin:
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
        
        # Add part of speech filter using JOIN
        if pos:
            count_sql = f"""
            SELECT COUNT(DISTINCT w.id) 
            FROM words w
            JOIN definitions d ON w.id = d.word_id
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            WHERE p.code = :pos AND ({count_sql.split('WHERE ')[1]})
            """
        
        # Prepare query parameters with percent wildcards
        logger.debug("Preparing count parameters")
        count_params = {
            "query": query,
            "normalized_query": normalized_query,
            "prefix_query": f"{normalized_query}%",
            "contains_query": f"%{normalized_query}%",
            "baybayin_query": f"%{query}%" if has_baybayin else None,
            "language": language if language and language != "all" else None,
            "pos": pos
        }
        
        # Execute count query with timeout protection
        logger.debug("Executing count query")
        try:
            total_count = db.session.execute(text(count_sql), count_params).scalar() or 0
        except Exception as e:
            logger.warning(f"Count query timed out, using estimate: {e}")
            total_count = 100  # Fallback to estimate
        
        logger.debug(f"Count query result: {total_count}")
        # If no results found, return early
        if total_count == 0:
            logger.debug("No results found, returning early")
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
        logger.debug("Building main SQL query")
        main_sql = """
        SELECT w.id, w.lemma, w.normalized_lemma, w.language_code, 
               w.has_baybayin, w.baybayin_form, w.romanized_form, w.root_word_id
        FROM words w
        """
        
        # Add joins if needed
        if pos:
            main_sql += """
            JOIN definitions d ON w.id = d.word_id
            JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            """
            
        # Add WHERE clause
        main_sql += " WHERE "
        
        # Reuse the same conditions as count query
        if mode == "exact":
            main_sql += "w.normalized_lemma = :normalized_query"
            if has_baybayin and baybayin_filter:
                main_sql += " OR w.baybayin_form = :query"
                
        elif mode == "prefix":
            main_sql += "w.normalized_lemma LIKE :prefix_query"
            
        elif mode == "baybayin":
            if has_baybayin:
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
            
        # Add GROUP BY if we used JOINs
        if pos:
            main_sql += " GROUP BY w.id"
            
        # Add ORDER BY
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
        logger.debug("Preparing main parameters")
        main_params = dict(count_params)
        main_params["limit"] = limit
        main_params["offset"] = offset
        
        # Execute main query with timeout protection
        logger.debug("Executing main query")
        try:
            words_result = db.session.execute(text(main_sql), main_params).fetchall()
        except Exception as e:
            logger.error(f"Search query error: {e}")
            return jsonify({
                "error": "Search query timed out, please refine your search",
                "count": total_count,
                "offset": offset,
                "limit": limit,
                "query": query,
                "results": []
            }), 500
        
        # Prepare result
        logger.debug("Preparing result dictionary")
        result = {
            "count": total_count,
            "offset": offset,
            "limit": limit,
            "query": query,
            "mode": mode,
            "language": language,
            "results": []
        }
        
        # Get word IDs for related data lookup
        logger.debug("Getting word IDs")
        word_ids = [w.id for w in words_result]
        
        if include_full and word_ids:
            logger.debug("Fetching full details")
            # For full results with related data, use the existing implementation
            # Load definitions with standardized POS
            definitions_query = db.session.query(
                Definition.id,
                Definition.word_id,
                Definition.definition_text,
                Definition.original_pos,
                Definition.standardized_pos_id,
                Definition.examples,
                Definition.usage_notes,
                Definition.tags,
                Definition.sources,
                PartOfSpeech.code,
                PartOfSpeech.name_en,
                PartOfSpeech.name_tl
            ).outerjoin(
                PartOfSpeech, Definition.standardized_pos_id == PartOfSpeech.id
            ).filter(
                Definition.word_id.in_(word_ids)
            ).all()
            
            # Group definitions by word_id
            definitions_by_word = {}
            for d in definitions_query:
                if d.word_id not in definitions_by_word:
                    definitions_by_word[d.word_id] = []
                    
                pos_info = None
                if d.standardized_pos_id:
                    pos_info = {
                        "id": d.standardized_pos_id,
                        "code": d.code,
                        "name_en": d.name_en,
                        "name_tl": d.name_tl
                    }
                    
                definitions_by_word[d.word_id].append({
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "original_pos": d.original_pos,
                    "standardized_pos_id": d.standardized_pos_id,
                    "standardized_pos": pos_info,
                    "examples": d.examples,
                    "usage_notes": d.usage_notes,
                    "tags": d.tags,
                    "sources": d.sources
                })
            
            # Get etymologies (if needed)
            etymologies_query = db.session.query(
                Etymology.id,
                Etymology.word_id,
                Etymology.etymology_text,
                Etymology.normalized_components,
                Etymology.etymology_structure,
                Etymology.language_codes,
                Etymology.sources
            ).filter(
                Etymology.word_id.in_(word_ids)
            ).all()
            
            etymologies_by_word = {}
            for e in etymologies_query:
                if e.word_id not in etymologies_by_word:
                    etymologies_by_word[e.word_id] = []
                    
                etymologies_by_word[e.word_id].append({
                    "id": e.id,
                    "etymology_text": e.etymology_text,
                    "normalized_components": e.normalized_components,
                    "etymology_structure": e.etymology_structure,
                    "language_codes": e.language_codes,
                    "sources": e.sources
                })
            
            # Get pronunciations
            pronunciations_query = db.session.query(
                Pronunciation.id,
                Pronunciation.word_id,
                Pronunciation.type,
                Pronunciation.value,
                Pronunciation.tags,
                Pronunciation.metadata.label('metadata'),
                Pronunciation.sources
            ).filter(
                Pronunciation.word_id.in_(word_ids)
            ).all()
            
            pronunciations_by_word = {}
            for p in pronunciations_query:
                if p.word_id not in pronunciations_by_word:
                    pronunciations_by_word[p.word_id] = []
                    
                pronunciations_by_word[p.word_id].append({
                    "id": p.id,
                    "type": p.type,
                    "value": p.value,
                    "tags": p.tags,
                    "metadata": p.metadata,  # Use the attribute name from the model
                    "sources": p.sources
                })
            
            # Prepare word results with relationships
            logger.debug("Processing full results")
            for word in words_result:
                word_dict = {
                    "id": word.id,
                    "lemma": word.lemma,
                    "normalized_lemma": word.normalized_lemma,
                    "language_code": word.language_code,
                    "has_baybayin": word.has_baybayin,
                    "baybayin_form": word.baybayin_form,
                    "romanized_form": word.romanized_form,
                    "root_word_id": word.root_word_id,
                }
                
                # Add related data
                word_dict["definitions"] = definitions_by_word.get(word.id, [])
                word_dict["etymologies"] = etymologies_by_word.get(word.id, [])
                word_dict["pronunciations"] = pronunciations_by_word.get(word.id, [])
                
                # Check if word is a root word
                word_dict["is_root"] = word.root_word_id is None
                
                result["results"].append(word_dict)
        else:
            logger.debug("Processing simplified results")
            # Simplified results without relationships
            for word in words_result:
                result["results"].append({
                    "id": word.id,
                    "lemma": word.lemma,
                    "normalized_lemma": word.normalized_lemma,
                    "language_code": word.language_code,
                    "has_baybayin": word.has_baybayin, 
                    "baybayin_form": word.baybayin_form,
                    "romanized_form": word.romanized_form,
                    "root_word_id": word.root_word_id,  # Added root_word_id
                    "is_root": word.root_word_id is None
                })
        
        # Record request latency
        logger.debug("Recording latency")
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search").observe(request_time)
        
        logger.debug("Returning JSON response")
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
        # Parse filter parameters
        language = request.args.get("language")
        pos = request.args.get("pos")
        has_etymology = request.args.get("has_etymology", "").lower() == "true"
        has_definitions = request.args.get("has_definitions", "").lower() == "true"
        has_baybayin = request.args.get("has_baybayin", "").lower() == "true"
        min_definitions = int(request.args.get("min_definitions", 0))
        
        # Base query
        query = Word.query
        
        # Apply filters
        if language:
            query = query.filter(Word.language_code == language)
        if pos:
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(PartOfSpeech.code == pos)
        if has_etymology:
            query = query.join(Word.etymologies).group_by(Word.id).having(func.count(Etymology.id) > 0)
        if has_definitions:
            query = query.join(Word.definitions).group_by(Word.id).having(func.count(Definition.id) > min_definitions)
        if has_baybayin:
            query = query.filter(Word.has_baybayin == True)
            
        # Count total eligible words
        count = query.count()
        
        if count == 0:
            return jsonify({"error": "No words match the specified criteria"}), 404
            
        # Get a random word
        random_offset = random.randint(0, count - 1)
        random_word = query.offset(random_offset).first()
        
        if not random_word:
            return jsonify({"error": "Failed to retrieve random word"}), 500
            
        # Fetch full details for the selected word
        word_with_details = _fetch_word_details(
            random_word.id,
            include_definitions=True,
            include_etymologies=True,
            include_pronunciations=True,
            include_credits=True,
            include_relations=True,
            include_affixations=True,
            include_root=True,
            include_derived=True
        )
        
        # Format result
        result = {
            "id": word_with_details.id,
            "lemma": word_with_details.lemma,
            "normalized_lemma": word_with_details.normalized_lemma,
            "language_code": word_with_details.language_code,
            "has_baybayin": word_with_details.has_baybayin,
            "baybayin_form": word_with_details.baybayin_form,
            "definitions": [
                {
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "original_pos": d.original_pos,
                    "standardized_pos": d.standardized_pos.name_en if d.standardized_pos else None,
                    "examples": d.examples,
                    "usage_notes": d.usage_notes,
                    "tags": d.tags,
                    "sources": d.sources
                }
                for d in word_with_details.definitions
            ],
            "etymologies": [
                {
                    "id": e.id,
                    "etymology_text": e.etymology_text,
                    "normalized_components": e.normalized_components,
                    "language_codes": e.language_codes,
                    "sources": e.sources
                }
                for e in word_with_details.etymologies
            ],
            "pronunciations": [
                {
                    "type": p.type,
                    "value": p.value,
                    "tags": p.tags,
                    "metadata": p.metadata,
                    "sources": p.sources
                }
                for p in word_with_details.pronunciations
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving random word", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get relations for a word."""
    try:
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
            
        # Load relations
        word_entry = _fetch_word_details(
            word_entry.id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=True,
            include_affixations=False,
            include_root=False,
            include_derived=False
        )
        
        # Format result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "outgoing_relations": [
                {
                    "id": r.id,
                    "relation_type": r.relation_type,
                    "target_word": {
                        "id": r.target_word.id,
                        "lemma": r.target_word.lemma,
                        "language_code": r.target_word.language_code,
                        "has_baybayin": r.target_word.has_baybayin,
                        "baybayin_form": r.target_word.baybayin_form
                    },
                    "sources": r.sources
                }
                for r in word_entry.outgoing_relations
            ],
            "incoming_relations": [
                {
                    "id": r.id,
                    "relation_type": r.relation_type,
                    "source_word": {
                        "id": r.source_word.id,
                        "lemma": r.source_word.lemma,
                        "language_code": r.source_word.language_code,
                        "has_baybayin": r.source_word.has_baybayin,
                        "baybayin_form": r.source_word.baybayin_form
                    },
                    "sources": r.sources
                }
                for r in word_entry.incoming_relations
            ]
        }
        
        # Group by relation type
        relation_types = {}
        
        for r in word_entry.outgoing_relations:
            if r.relation_type not in relation_types:
                relation_types[r.relation_type] = {"outgoing": [], "incoming": []}
            
            relation_types[r.relation_type]["outgoing"].append({
                "id": r.id,
                "word": {
                    "id": r.target_word.id,
                    "lemma": r.target_word.lemma,
                    "language_code": r.target_word.language_code
                },
                "sources": r.sources
            })
            
        for r in word_entry.incoming_relations:
            if r.relation_type not in relation_types:
                relation_types[r.relation_type] = {"outgoing": [], "incoming": []}
                
            relation_types[r.relation_type]["incoming"].append({
                "id": r.id,
                "word": {
                    "id": r.source_word.id,
                    "lemma": r.source_word.lemma,
                    "language_code": r.source_word.language_code
                },
                "sources": r.sources
            })
        
        result["relation_types"] = relation_types
        result["total_relations"] = len(word_entry.outgoing_relations) + len(word_entry.incoming_relations)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving relations for word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get affixations for a word."""
    try:
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
            
        # Load affixations
        word_entry = _fetch_word_details(
            word_entry.id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=True,
            include_root=True,
            include_derived=True
        )
        
        # Format result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "is_root": not word_entry.root_word_id,
            "root_affixations": [
                {
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "affixed_word": {
                        "id": aff.affixed_word.id,
                        "lemma": aff.affixed_word.lemma,
                        "language_code": aff.affixed_word.language_code,
                        "has_baybayin": aff.affixed_word.has_baybayin,
                        "baybayin_form": aff.affixed_word.baybayin_form
                    },
                    "sources": aff.sources
                }
                for aff in word_entry.root_affixations
            ],
            "affixed_affixations": [
                {
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "root_word": {
                        "id": aff.root_word.id,
                        "lemma": aff.root_word.lemma,
                        "language_code": aff.root_word.language_code,
                        "has_baybayin": aff.root_word.has_baybayin,
                        "baybayin_form": aff.root_word.baybayin_form
                    },
                    "sources": aff.sources
                }
                for aff in word_entry.affixed_affixations
            ]
        }
        
        # Add root word if available
        if word_entry.root_word_id and word_entry.root_word:
            result["root_word"] = {
                "id": word_entry.root_word.id,
                "lemma": word_entry.root_word.lemma,
                "language_code": word_entry.root_word.language_code,
                "has_baybayin": word_entry.root_word.has_baybayin,
                "baybayin_form": word_entry.root_word.baybayin_form
            }
        
        # Add derived words if available
        if hasattr(word_entry, 'derived_words') and word_entry.derived_words:
            result["derived_words"] = [
                {
                    "id": d.id,
                    "lemma": d.lemma,
                    "language_code": d.language_code,
                    "has_baybayin": d.has_baybayin,
                    "baybayin_form": d.baybayin_form
                }
                for d in word_entry.derived_words
            ]
        
        # Group by affix type
        affix_types = {}
        
        for aff in word_entry.root_affixations:
            if aff.affix_type not in affix_types:
                affix_types[aff.affix_type] = {"as_root": [], "as_affixed": []}
            
            affix_types[aff.affix_type]["as_root"].append({
                "id": aff.id,
                "word": {
                    "id": aff.affixed_word.id,
                    "lemma": aff.affixed_word.lemma,
                    "language_code": aff.affixed_word.language_code
                },
                "sources": aff.sources
            })
        
        for aff in word_entry.affixed_affixations:
            if aff.affix_type not in affix_types:
                affix_types[aff.affix_type] = {"as_root": [], "as_affixed": []}
                
            affix_types[aff.affix_type]["as_affixed"].append({
                "id": aff.id,
                "word": {
                    "id": aff.root_word.id,
                    "lemma": aff.root_word.lemma,
                    "language_code": aff.root_word.language_code
                },
                "sources": aff.sources
            })
        
        result["affix_types"] = affix_types
        result["total_affixations"] = len(word_entry.root_affixations) + len(word_entry.affixed_affixations)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving affixations for word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get pronunciation data for a word."""
    try:
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
            
        # Load pronunciations
        word_entry = _fetch_word_details(
            word_entry.id,
            include_definitions=False,
            include_etymologies=False,
            include_pronunciations=True,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False
        )
        
        # Format result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "pronunciations": [
                {
                    "type": p.type,
                    "value": p.value,
                    "tags": p.tags,
                    "metadata": p.metadata,
                    "sources": p.sources
                }
                for p in word_entry.pronunciations
            ]
        }
        
        # Group by pronunciation type
        pronunciation_types = {}
        
        for p in word_entry.pronunciations:
            if p.type not in pronunciation_types:
                pronunciation_types[p.type] = []
                
            pronunciation_types[p.type].append({
                "value": p.value,
                "tags": p.tags,
                "metadata": p.metadata,
                "sources": p.sources
            })
        
        result["pronunciation_types"] = pronunciation_types
        result["has_audio"] = any(p.type == "audio" for p in word_entry.pronunciations)
        result["has_ipa"] = any(p.type == "ipa" for p in word_entry.pronunciations)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving pronunciation for word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get etymology data for a word."""
    try:
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
            
        # Load etymologies
        word_entry = _fetch_word_details(
            word_entry.id,
            include_definitions=False,
            include_etymologies=True,
            include_pronunciations=False,
            include_credits=False,
            include_relations=False,
            include_affixations=False,
            include_root=False,
            include_derived=False
        )
        
        # Format result
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "language_code": word_entry.language_code,
            "etymologies": [
                {
                    "id": e.id,
                    "etymology_text": e.etymology_text,
                    "normalized_components": e.normalized_components,
                    "etymology_structure": e.etymology_structure,
                    "language_codes": e.language_codes,
                    "sources": e.sources
                }
                for e in word_entry.etymologies
            ]
        }
        
        # Extract components if available
        for e in result["etymologies"]:
            if e["normalized_components"]:
                e["components"] = extract_etymology_components(e["normalized_components"])
                
            if e["language_codes"]:
                e["languages"] = extract_language_codes(e["language_codes"])
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error retrieving etymology for word '{word}'", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

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
                include_derived=False
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
                "idiomatic": [],
                "formal": [],
                "other": []
            },
            "descriptions": {}
        }
        
        # Categorize relationship types
        for rel_type in relation_types:
            category = RelationshipCategory.get_category(rel_type)
            if category:
                result["categories"][category].append(rel_type)
                
            description = RelationshipType.get_description(rel_type)
            if description:
                result["descriptions"][rel_type] = description
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Error retrieving relationship types", error=str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500