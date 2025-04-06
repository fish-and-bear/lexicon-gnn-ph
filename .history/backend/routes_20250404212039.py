"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, Language, Pronunciation, Credit
)
from database import db, cached_query
from dictionary_manager import (
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
from .search_tasks import log_search_query

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__, url_prefix='/api/v2')

# GraphQL will be added later to avoid circular dependencies
# Instead of immediate import: from gql.schema import schema

def is_testing_db(engine):
    """Check if we're using a test database."""
    return engine.url.database.endswith('_test')

# Health check endpoint
@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db.session.execute(text('SELECT 1'))
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.error('Health check failed', error=str(e))
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

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

# API Routes

@bp.before_request
def before_request():
    """Log and track request metrics."""
    g.start_time = datetime.utcnow()
    API_REQUESTS.labels(
        endpoint=request.endpoint,
        method=request.method
    ).inc()

@bp.after_request
def after_request(response):
    """Log response metrics."""
    if hasattr(g, 'start_time'):
        duration = (datetime.utcnow() - g.start_time).total_seconds()
        REQUEST_LATENCY.labels(endpoint=request.endpoint).observe(duration)
        REQUEST_COUNT.inc()
    return response

@bp.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({"error": "Not found", "message": str(error)}), 404

@bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    db.session.rollback()
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@bp.errorhandler(SQLAlchemyError)
def database_error(error):
    """Handle database errors."""
    db.session.rollback()
    logger.error(f"Database error: {str(error)}")
    return jsonify({"error": "Database error"}), 500

@bp.errorhandler(Exception)
def handle_exception(error):
    """Handle unhandled exceptions."""
    db.session.rollback()
    logger.error(f"Unhandled exception: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>", methods=["GET"])
@cached_query(timeout=900, key_prefix="word_detail")  # Cache results for 15 minutes
def get_word(word: str):
    """Get a word entry by lemma or ID."""
    try:
        # Get query parameters with defaults
        include_definitions = request.args.get('include_definitions', 'true').lower() == 'true'
        include_relations = request.args.get('include_relations', 'true').lower() == 'true'
        include_affixations = request.args.get('include_affixations', 'true').lower() == 'true'
        include_etymologies = request.args.get('include_etymologies', 'true').lower() == 'true'
        include_pronunciations = request.args.get('include_pronunciations', 'true').lower() == 'true'
        include_credits = request.args.get('include_credits', 'true').lower() == 'true'
        include_root = request.args.get('include_root', 'true').lower() == 'true'
        include_derived = request.args.get('include_derived', 'true').lower() == 'true'
        include_all = request.args.get('include_all', 'false').lower() == 'true'
        
        if include_all:
            include_definitions = include_relations = include_affixations = include_etymologies = True
            include_pronunciations = include_credits = include_root = include_derived = True
        
        # Get the word by ID or lemma
        word_entry = None
        try:
            # Try to parse as integer (ID)
            word_id = int(word)
            word_entry = _fetch_word_details(
                word_id,
                include_definitions=include_definitions,
                include_relations=include_relations,
                include_affixations=include_affixations,
                include_etymologies=include_etymologies,
                include_pronunciations=include_pronunciations,
                include_credits=include_credits,
                include_root=include_root,
                include_derived=include_derived
            )
            if word_entry is None:
                abort(404, description=f"Word ID {word_id} not found")
        except ValueError:
            # Not an ID, look up by lemma
            word_entry = Word.query.filter(
                func.lower(Word.lemma) == func.lower(word)
            ).first()
            
            if not word_entry:
                # Try normalized lemma as a fallback
                normalized = normalize_lemma(word)
                word_entry = Word.query.filter(
                    Word.normalized_lemma == normalized
                ).first()
                
            if not word_entry:
                abort(404, description=f"Word '{word}' not found")
                
            # Now fetch full details for this word
            word_entry = _fetch_word_details(
                word_entry.id,
                include_definitions=include_definitions,
                include_relations=include_relations,
                include_affixations=include_affixations,
                include_etymologies=include_etymologies,
                include_pronunciations=include_pronunciations,
                include_credits=include_credits,
                include_root=include_root,
                include_derived=include_derived
            )
        
        # Serialize the result
        if include_all:
            # Include all relationships in the dictionary
            result = word_entry.to_dict(include_related=True)
        else:
            # Basic dictionary with selected relationships
            result = word_entry.to_dict()
            
            # Only add selected relationships to avoid excess data
            if include_definitions and hasattr(word_entry, 'definitions'):
                result["definitions"] = [d.to_dict() for d in word_entry.definitions]
                
            if include_etymologies and hasattr(word_entry, 'etymologies'):
                result["etymologies"] = [e.to_dict() for e in word_entry.etymologies]
                
            if include_pronunciations and hasattr(word_entry, 'pronunciations'):
                result["pronunciations"] = [p.to_dict() for p in word_entry.pronunciations]
                
            if include_credits and hasattr(word_entry, 'credits'):
                result["credits"] = [c.to_dict() for c in word_entry.credits]
                
            if include_relations:
                if hasattr(word_entry, 'outgoing_relations'):
                    result["outgoing_relations"] = [
                        {
                "id": rel.id,
                "relation_type": rel.relation_type,
                "target_word": {
                                "id": rel.target_word.id,
                                "lemma": rel.target_word.lemma,
                                "language_code": rel.target_word.language_code,
                                "has_baybayin": rel.target_word.has_baybayin,
                                "baybayin_form": rel.target_word.baybayin_form
                            },
                            "metadata": rel.metadata,
                "sources": rel.sources
                        }
                        for rel in word_entry.outgoing_relations
                    ]
            
                if hasattr(word_entry, 'incoming_relations'):
                    result["incoming_relations"] = [
                        {
                "id": rel.id,
                "relation_type": rel.relation_type,
                "source_word": {
                                "id": rel.source_word.id,
                                "lemma": rel.source_word.lemma,
                                "language_code": rel.source_word.language_code,
                                "has_baybayin": rel.source_word.has_baybayin,
                                "baybayin_form": rel.source_word.baybayin_form
                },
                "metadata": rel.metadata,
                "sources": rel.sources
                        }
                        for rel in word_entry.incoming_relations
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
def search():
    """Search for words with optimized performance for high traffic."""
    try:
        # Track request metrics
        REQUEST_COUNT.inc()
        API_REQUESTS.labels(endpoint="search", method="GET").inc()
        start_time = time.time()
        
        # Validate and parse search parameters
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
        db.session.execute(text("SET statement_timeout TO '3000'"))  # 3 seconds timeout
        
        # Build the base query - use normalized_query for better index usage
        normalized_query = normalize_lemma(query)
        
        # Check for Baybayin query
        has_baybayin = any(0x1700 <= ord(c) <= 0x171F for c in query)
        baybayin_filter = None
        
        if has_baybayin:
            romanizer = BaybayinRomanizer()
            romanized_query = romanizer.romanize(query) if romanizer.validate_text(query) else ""
            baybayin_filter = Word.baybayin_form.ilike(f"%{query}%")
            if romanized_query:
                normalized_query = normalize_lemma(romanized_query)
        
        # Use direct SQL for count query for better performance
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
        main_params = dict(count_params)
        main_params["limit"] = limit
        main_params["offset"] = offset
        
        # Execute main query with timeout protection
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
        word_ids = [w.id for w in words_result]
        
        if include_full and word_ids:
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
            for word in words:
                word_dict = word.to_dict()
                
                # Add related data
                word_dict["definitions"] = definitions_by_word.get(word.id, [])
                word_dict["etymologies"] = etymologies_by_word.get(word.id, [])
                word_dict["pronunciations"] = pronunciations_by_word.get(word.id, [])
                
                # Check if word is a root word
                word_dict["is_root"] = word.root_word_id is None
                
                result["results"].append(word_dict)
        else:
            # Simplified results without relationships
            for word in words:
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
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search").observe(request_time)
        
        return jsonify(result)
    except Exception as e:
        API_ERRORS.labels(endpoint="search", error_type=type(e).__name__).inc()
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route("/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get word relations."""
    try:
        # Get word ID first using direct SQL
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text(
            "SELECT id, lemma FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Get outgoing relations
        outgoing_relations = db.session.execute(text("""
            SELECT r.id, r.relation_type, r.metadata, 
                   w.id as target_id, w.lemma as target_lemma, 
                   w.language_code as target_language_code,
                   w.has_baybayin as target_has_baybayin,
                   w.baybayin_form as target_baybayin_form
            FROM relations r
            JOIN words w ON r.to_word_id = w.id
            WHERE r.from_word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        # Get incoming relations
        incoming_relations = db.session.execute(text("""
            SELECT r.id, r.relation_type, r.metadata, 
                   w.id as source_id, w.lemma as source_lemma, 
                   w.language_code as source_language_code,
                   w.has_baybayin as source_has_baybayin,
                   w.baybayin_form as source_baybayin_form
            FROM relations r
            JOIN words w ON r.from_word_id = w.id
            WHERE r.to_word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        result = {
            'outgoing_relations': [{
                "id": rel.id,
                "relation_type": rel.relation_type,
                "target_word": {
                    "id": rel.to_word_id,
                    "lemma": rel.target_lemma,
                    "language_code": rel.target_language_code,
                    "has_baybayin": rel.target_has_baybayin,
                    "baybayin_form": rel.target_baybayin_form
                },
                "metadata": rel.metadata,  # Keep using 'metadata' in API responses
                "sources": rel.sources
            } for rel in outgoing_relations],
            'incoming_relations': [{
                "id": rel.id,
                "relation_type": rel.relation_type,
                "source_word": {
                    "id": rel.from_word_id,
                    "lemma": rel.source_lemma,
                    "language_code": rel.source_language_code,
                    "has_baybayin": rel.source_has_baybayin,
                    "baybayin_form": rel.source_baybayin_form
                },
                "metadata": rel.metadata,
                "sources": rel.sources
            } for rel in incoming_relations]
                }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting word relations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get all affixations for a word."""
    try:
        # Set a query timeout
        db.session.execute(text("SET statement_timeout TO '5000'"))  # 5 seconds
        
        # Get word ID first using direct SQL
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text(
            "SELECT id, lemma FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Get affixations where this word is the root
        try:
            root_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, 
                       w.id as affixed_id, w.lemma as affixed_lemma, 
                       w.language_code as affixed_language_code,
                       w.has_baybayin as affixed_has_baybayin,
                       w.baybayin_form as affixed_baybayin_form
                FROM affixations a
                JOIN words w ON a.affixed_word_id = w.id
                WHERE a.root_word_id = :word_id
                LIMIT 50  /* Limit to prevent excessive results */
            """), {"word_id": word_id}).fetchall()
        except Exception as e:
            logger.error(f"Error querying root affixations: {str(e)}")
            root_affixations = []
        
        # Get affixations where this word is the affixed form
        try:
            affixed_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, 
                       w.id as root_id, w.lemma as root_lemma, 
                       w.language_code as root_language_code,
                       w.has_baybayin as root_has_baybayin,
                       w.baybayin_form as root_baybayin_form
                FROM affixations a
                JOIN words w ON a.root_word_id = w.id
                WHERE a.affixed_word_id = :word_id
                LIMIT 50  /* Limit to prevent excessive results */
            """), {"word_id": word_id}).fetchall()
        except Exception as e:
            logger.error(f"Error querying affixed affixations: {str(e)}")
            affixed_affixations = []
        
        # Format the results
        as_root = []
        for aff in root_affixations:
            try:
                as_root.append({
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "affixed_word": {
                        "id": aff.affixed_word_id,
                        "lemma": aff.affixed_lemma,
                        "language_code": aff.affixed_language_code,
                        "has_baybayin": aff.affixed_has_baybayin,
                        "baybayin_form": aff.affixed_baybayin_form
                    }
                })
            except Exception as e:
                logger.error(f"Error formatting root affixation: {str(e)}")
        
        as_affixed = []
        for aff in affixed_affixations:
            try:
                as_affixed.append({
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "root_word": {
                        "id": aff.root_word_id,
                        "lemma": aff.root_lemma,
                        "language_code": aff.root_language_code,
                        "has_baybayin": aff.root_has_baybayin,
                        "baybayin_form": aff.root_baybayin_form
                    }
                })
            except Exception as e:
                logger.error(f"Error formatting affixed affixation: {str(e)}")
        
        return jsonify({
            'as_root': as_root,
            'as_affixed': as_affixed
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting word affixations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get word pronunciation."""
    try:
        # Get word ID first using direct SQL
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text(
            "SELECT id, lemma FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Get pronunciations - use column name 'metadata' in SQL query
        pronunciations = db.session.execute(text("""
            SELECT id, type, value, tags, metadata
            FROM pronunciations
            WHERE word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        # Format the results - return as 'metadata' in API for backwards compatibility
        pronunciation_list = []
        for pron in pronunciations:
            pronunciation_list.append({
                "id": pron.id,
                "type": pron.type,
                "value": pron.value,
                "tags": pron.tags,
                "metadata": pron.metadata  # Keep as metadata in API response
            })
        
        return jsonify({
            'pronunciations': pronunciation_list,
            'has_pronunciation': bool(pronunciation_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting word pronunciation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics."""
    try:
        # Use direct SQL queries for all statistics
        stats = {}
        
        # Total word count
        result = db.session.execute(text("SELECT COUNT(*) FROM words")).scalar()
        stats['total_words'] = result
        
        # Total definition count
        result = db.session.execute(text("SELECT COUNT(*) FROM definitions")).scalar()
        stats['total_definitions'] = result
        
        # Total etymology count
        result = db.session.execute(text("SELECT COUNT(*) FROM etymologies")).scalar()
        stats['total_etymologies'] = result
        
        # Languages
        result = db.session.execute(text("""
            SELECT language_code, COUNT(*) as count
            FROM words
            GROUP BY language_code
            ORDER BY count DESC
        """)).fetchall()
        stats['words_by_language'] = {row.language_code: row.count for row in result}
        
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@bp.route("/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get word etymology."""
    try:
        # Set a timeout for these queries
        db.session.execute(text("SET statement_timeout TO '5000'"))  # 5 seconds
        
        # Get word ID first using direct SQL
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text(
            "SELECT id, lemma FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Get etymologies
        etymologies = db.session.execute(text("""
            SELECT id, etymology_text, normalized_components, language_codes, sources
            FROM etymologies
            WHERE word_id = :word_id
            LIMIT 10  /* Limit etymologies to prevent processing timeouts */
        """), {"word_id": word_id}).fetchall()
        
        # Format the results
        etymology_list = []
        for etym in etymologies:
            etymology_data = {
                "id": etym.id,
                "etymology_text": etym.etymology_text,
                "normalized_components": etym.normalized_components,
                "language_codes": etym.language_codes,
                "sources": etym.sources
            }
            
            # Add components if they exist
            try:
                etymology_data["components"] = extract_etymology_components(etym.etymology_text)
            except:
                etymology_data["components"] = []
                
            etymology_list.append(etymology_data)
        
        return jsonify({
            'etymologies': etymology_list,
            'has_etymology': bool(etymology_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting word etymology: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/semantic_network", methods=["GET"])
def get_semantic_network(word: str):
    """Get semantic network for a word."""
    try:
        # Set a timeout for these queries
        db.session.execute(text("SET statement_timeout TO '5000'"))  # 5 seconds
        
        # Get word ID first using direct SQL
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text(
            "SELECT id, lemma FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Get all relations for this word with a limit to prevent timeouts
        relations = db.session.execute(text("""
            SELECT r.id, r.relation_type,
                   sw.id as source_id, sw.lemma as source_lemma,
                   tw.id as target_id, tw.lemma as target_lemma
            FROM relations r
            JOIN words sw ON r.from_word_id = sw.id
            JOIN words tw ON r.to_word_id = tw.id
            WHERE r.from_word_id = :word_id OR r.to_word_id = :word_id
            LIMIT 100  /* Limit total relations to prevent timeout */
        """), {"word_id": word_id}).fetchall()
        
        # Build network
        nodes = {}
        edges = []
        
        for rel in relations:
            # Add source node if not already in nodes
            if rel.source_id not in nodes:
                nodes[rel.source_id] = {
                    'id': rel.source_lemma,
                    'label': rel.source_lemma
                }
            
            # Add target node if not already in nodes
            if rel.target_id not in nodes:
                nodes[rel.target_id] = {
                    'id': rel.target_lemma,
                    'label': rel.target_lemma
                }
            
            # Add edge
            edges.append({
                'source': rel.source_lemma,
                'target': rel.target_lemma,
                'type': rel.relation_type,
                'bidirectional': False  # Default to false since column doesn't exist
            })
        
        return jsonify({
            'nodes': list(nodes.values()),
            'edges': edges
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting semantic network: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/affixation_tree", methods=["GET"])
def get_affixation_tree(word: str):
    """Get the affixation tree for a word."""
    try:
        max_depth = int(request.args.get('max_depth', 3))
        include_definitions = request.args.get('include_definitions', 'true').lower() == 'true'
        include_baybayin = request.args.get('include_baybayin', 'true').lower() == 'true'
    except (ValueError, TypeError) as err:
        return jsonify({"error": str(err)}), 400

    # Get initial word using direct SQL
    normalized_word = normalize_lemma(word)
    word_entry = db.session.execute(text(
        """SELECT id, lemma, language_code, has_baybayin, baybayin_form, root_word_id 
           FROM words 
           WHERE normalized_lemma = :normalized"""
    ), {"normalized": normalized_word}).fetchone()

    if not word_entry:
        return jsonify({"error": "Word not found"}), 404

    # Session for all SQL queries
    session = db.session
    
    def get_word_by_id(word_id):
        """Get word data using direct SQL."""
        return session.execute(text(
            """SELECT id, lemma, language_code, has_baybayin, baybayin_form, root_word_id 
               FROM words WHERE id = :word_id"""
        ), {"word_id": word_id}).fetchone()
    
    def get_definitions(word_id, limit=2):
        """Get definitions using direct SQL."""
        defs = session.execute(text(
            """SELECT definition_text, standardized_pos_id 
               FROM definitions 
               WHERE word_id = :word_id 
               LIMIT :limit"""
        ), {"word_id": word_id, "limit": limit}).fetchall()
        
        return [{"text": d.definition_text, "pos_id": d.standardized_pos_id} for d in defs]
    
    def get_derived_affixations(word_id):
        """Get derived affixations using direct SQL."""
        return session.execute(text(
            """SELECT a.affixed_word_id, a.affix_type
               FROM affixations a
               WHERE a.root_word_id = :word_id"""
        ), {"word_id": word_id}).fetchall()
    
    def get_root_affixation(word_id):
        """Get root affixation using direct SQL."""
        return session.execute(text(
            """SELECT a.root_word_id, a.affix_type
               FROM affixations a
               WHERE a.affixed_word_id = :word_id"""
        ), {"word_id": word_id}).fetchone()

    def build_tree(word, depth=0, processed=None):
        if processed is None:
            processed = set()
            
        if depth > max_depth or word.id in processed:
            return None
            
        processed.add(word.id)
        
        node = {
            "id": word.id,
            "word": word.lemma,
            "language_code": word.language_code,
            "is_root": word.root_word_id is None
        }

        if include_definitions:
            node["definitions"] = get_definitions(word.id)

        if include_baybayin and word.has_baybayin:
            node["baybayin_form"] = word.baybayin_form

        # Get derived words
        derived = get_derived_affixations(word.id)

        if derived:
            node["derived"] = []
            for aff in derived:
                derived_word = get_word_by_id(aff.affixed_word_id)
                if derived_word:
                    child = build_tree(derived_word, depth + 1, processed)
                    if child:
                        child["affixation"] = {
                            "type": aff.affix_type
                        }
                        node["derived"].append(child)

        # Get root word if this is a derived word
        if not node["is_root"]:
            root_affixation = get_root_affixation(word.id)
            
            if root_affixation and root_affixation.root_word_id not in processed:
                root_word = get_word_by_id(root_affixation.root_word_id)
                if root_word:
                    node["root"] = build_tree(root_word, depth + 1, processed)
                    if node["root"]:
                        node["root"]["affixation"] = {
                            "type": root_affixation.affix_type
                        }

        return node

    # Build the complete tree
    tree = build_tree(word_entry)
    
    # Add metadata
    result = {
        "tree": tree,
        "metadata": {
            "max_depth": max_depth,
            "word": word_entry.lemma,
            "is_root": word_entry.root_word_id is None,
            "has_derived_forms": bool(tree and tree.get("derived")),
            "has_root_word": bool(tree and tree.get("root"))
        }
    }

    return jsonify(result), 200

# Helper functions
def get_word_suggestions(word: str) -> List[Dict[str, Any]]:
    """Get word suggestions for a failed lookup."""
    try:
        # Try fuzzy matching with direct SQL
        suggestions = db.session.execute(text("""
            SELECT id, lemma, language_code, has_baybayin, baybayin_form
            FROM words
            WHERE lemma ILIKE :pattern
            ORDER BY length(lemma)
            LIMIT 5
        """), {"pattern": f"%{word}%"}).fetchall()
        
        # If no suggestions, try with normalized_lemma
        if not suggestions:
            suggestions = db.session.execute(text("""
                SELECT id, lemma, language_code, has_baybayin, baybayin_form
                FROM words
                WHERE normalized_lemma ILIKE :pattern
                ORDER BY length(normalized_lemma)
                LIMIT 5
            """), {"pattern": f"%{normalize_lemma(word)}%"}).fetchall()
        
        # Create formatted list of suggestions
        result = []
        for w in suggestions:
            result.append({
                "id": w.id,
                "lemma": w.lemma,
                "language_code": w.language_code,
                "similarity": 1.0 if w.lemma.lower() == word.lower() else 0.5,
                "has_baybayin": w.has_baybayin,
                "baybayin_form": w.baybayin_form if w.has_baybayin else None
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting word suggestions: {str(e)}")
        return []

def calculate_data_completeness(word) -> Dict[str, Any]:
    """Calculate completeness metrics for a word entry."""
    total_fields = 12  # Total number of main data fields
    present_fields = 0
    
    # Check basic fields
    if word.lemma:
        present_fields += 1
    if word.language_code:
        present_fields += 1
    if word.has_baybayin and word.baybayin_form:
        present_fields += 1
    
    # Check definitions
    if word.definitions:
        present_fields += 1
        # Check definition quality
        definitions_with_examples = sum(1 for d in word.definitions if d.examples)
        definitions_with_usage = sum(1 for d in word.definitions if d.usage_notes)
        if definitions_with_examples > 0:
            present_fields += 1
        if definitions_with_usage > 0:
            present_fields += 1
    
    # Check etymology
    if word.etymologies:
        present_fields += 1
        # Check etymology quality
        etymologies_with_structure = sum(1 for e in word.etymologies if e.etymology_structure)
        if etymologies_with_structure > 0:
            present_fields += 1
    
    # Check relationships
    if word.relations_from or word.relations_to:
        present_fields += 1
    
    # Check pronunciations
    if word.pronunciations:
        present_fields += 1
        # Check pronunciation quality
        pronunciations_with_ipa = sum(1 for p in word.pronunciations if p.type == 'ipa')
        if pronunciations_with_ipa > 0:
            present_fields += 1
    
    # Calculate scores
    completeness = {
        "overall_score": round(present_fields / total_fields * 100, 2),
        "fields": {
            "basic_info": bool(word.lemma and word.language_code),
            "baybayin": bool(word.has_baybayin and word.baybayin_form),
            "definitions": bool(word.definitions),
            "definitions_with_examples": bool(definitions_with_examples),
            "definitions_with_usage": bool(definitions_with_usage),
            "etymology": bool(word.etymologies),
            "etymology_structured": bool(etymologies_with_structure),
            "relationships": bool(word.relations_from or word.relations_to),
            "pronunciation": bool(word.pronunciations),
            "pronunciation_ipa": bool(pronunciations_with_ipa)
        },
        "suggestions": []
    }
    
    # Add improvement suggestions
    if not completeness["fields"]["baybayin"] and word.language_code == 'tl':
        completeness["suggestions"].append({
            "field": "baybayin",
            "message": "Add Baybayin script representation"
        })
    if not completeness["fields"]["definitions_with_examples"]:
        completeness["suggestions"].append({
            "field": "examples",
            "message": "Add usage examples to definitions"
        })
    if not completeness["fields"]["etymology"]:
        completeness["suggestions"].append({
            "field": "etymology",
            "message": "Add etymology information"
        })
    if not completeness["fields"]["pronunciation_ipa"]:
        completeness["suggestions"].append({
            "field": "pronunciation",
            "message": "Add IPA pronunciation"
        })
    
    return completeness

def get_verification_history(word) -> List[Dict[str, Any]]:
    """Get verification history for a word entry."""
    # This would typically come from a separate verification_history table
    # For now, return a basic structure
    return [{
        "timestamp": word.updated_at.isoformat(),
        "status": word.verification_status,
        "verified_fields": ["lemma", "language_code"],
        "verifier": "system"
    }]

def get_edit_history(word) -> List[Dict[str, Any]]:
    """Get edit history for a word entry."""
    # This would typically come from a separate edit_history table
    # For now, return a basic structure
    return [{
        "timestamp": word.created_at.isoformat(),
        "type": "creation",
        "fields": ["lemma", "language_code"],
        "editor": "system"
    }]

def get_usage_statistics(word) -> Dict[str, Any]:
    """Get usage statistics for a word."""
    return {
        "frequency_score": word.usage_frequency or 0.0,
        "search_frequency": 0,  # Would come from analytics
        "citation_count": 0,    # Would come from references
        "last_accessed": word.updated_at.isoformat()
    }

def get_related_concepts(word) -> List[Dict[str, Any]]:
    """Get related concepts for a word."""
    concepts = []
    
    # Get semantic domains from definitions
    domains = set()
    for def_ in word.definitions:
        if def_.domain:
            domains.add(def_.domain)
    
    # Get related words through relationships
    related = (
        Relation.query
        .filter(
            or_(
                Relation.from_word_id == word.id,
                Relation.to_word_id == word.id
            ),
            Relation.relation_type.in_(['synonym', 'related', 'similar'])
        )
        .options(
            joinedload(Relation.from_word),
            joinedload(Relation.to_word)
        )
        .limit(5)
        .all()
    )
    
    for rel in related:
        related_word = rel.to_word if rel.from_word_id == word.id else rel.from_word
        concepts.append({
            "word": related_word.lemma,
            "type": rel.relation_type,
            "confidence": rel.confidence_score
        })
    
    return {
        "semantic_domains": list(domains),
        "related_words": concepts
    }

def get_dialectal_variations(word) -> List[Dict[str, Any]]:
    """Get dialectal variations for a word."""
    variations = []
    
    # Get regional variants through relationships
    variants = (
        Relation.query
        .filter(
            or_(
                Relation.from_word_id == word.id,
                Relation.to_word_id == word.id
            ),
            Relation.relation_type == 'regional_variant'
        )
        .options(
            joinedload(Relation.from_word),
            joinedload(Relation.to_word)
        )
        .all()
    )
    
    for var in variants:
        variant_word = var.to_word if var.from_word_id == word.id else var.from_word
        variations.append({
            "word": variant_word.lemma,
            "region": variant_word.geographic_region,
            "confidence": var.confidence_score
        })
    
    return variations

def get_semantic_domains(word) -> List[Dict[str, Any]]:
    """Get semantic domains for a word."""
    domains = set()
    
    # Collect domains from definitions
    for def_ in word.definitions:
        if def_.domain:
            domains.add(def_.domain)
    
    # Get domains from related words
    related = (
        Relation.query
        .filter(
            or_(
                Relation.from_word_id == word.id,
                Relation.to_word_id == word.id
            )
        )
        .options(
            joinedload(Relation.from_word).joinedload(Word.definitions),
            joinedload(Relation.to_word).joinedload(Word.definitions)
        )
        .all()
    )
    
    for rel in related:
        related_word = rel.to_word if rel.from_word_id == word.id else rel.from_word
        for def_ in related_word.definitions:
            if def_.domain:
                domains.add(def_.domain)
    
    return [{
        "domain": domain,
        "frequency": 1.0  # Would be calculated based on actual usage
    } for domain in domains]

def generate_search_facets(results):
    """Generate facets from search results."""
    facets = {
        'parts_of_speech': defaultdict(int),
        'languages': defaultdict(int),
        'verification_status': defaultdict(int),
        'has_baybayin': defaultdict(int),
        'has_etymology': defaultdict(int),
        'has_pronunciation': defaultdict(int)
    }

    for word in results:
        # Language facet
        facets['languages'][word.language_code] += 1
        
        # Verification status facet
        facets['verification_status'][word.verification_status] += 1
        
        # Baybayin facet
        facets['has_baybayin'][bool(word.has_baybayin)] += 1
        
        # Etymology facet
        facets['has_etymology'][bool(word.etymologies)] += 1
        
        # Pronunciation facet
        facets['has_pronunciation'][bool(word.pronunciation_data)] += 1
        
        # Parts of speech facet
        for definition in word.definitions:
            if definition.standardized_pos:
                facets['parts_of_speech'][definition.standardized_pos.code] += 1

    return {k: dict(v) for k, v in facets.items()}

def generate_search_suggestions(query, results):
    """Generate search suggestions based on query and results."""
    suggestions = []
    
    # Add spelling suggestions
    if len(results) < 5:
        # TODO: Implement fuzzy matching for spelling suggestions
        pass
        
    # Add related searches based on word relations
    for word in results[:5]:  # Limit to first 5 results
        for relation in word.relations_from:
            if relation.type in ['synonym', 'variant']:
                suggestions.append({
                    'type': 'related',
                    'text': relation.target_word.lemma,
                    'score': 1.0
                })
    
    return suggestions[:5]  # Limit to top 5 suggestions

def generate_quality_distribution() -> Dict[str, int]:
    """Generate quality score distribution."""
    return db.session.query(
        case(
            (Word.quality_score >= 0.8, "high"),
            (Word.quality_score >= 0.5, "medium"),
            else_="low"
        ).label("quality_level"),
        func.count(Word.id)
    ).group_by("quality_level").all()

def generate_update_frequency_stats() -> Dict[str, int]:
    """Generate update frequency statistics."""
    now = datetime.now(timezone.utc)
    return {
        "last_24h": Word.query.filter(
            Word.updated_at >= now - timedelta(days=1)
        ).count(),
        "last_week": Word.query.filter(
            Word.updated_at >= now - timedelta(weeks=1)
        ).count(),
        "last_month": Word.query.filter(
            Word.updated_at >= now - timedelta(days=30)
        ).count()
    }

@bp.route("/words/<int:word_id>/etymology/tree", methods=["GET"])
def get_etymology_tree(word_id: int):
    """Get the complete etymology tree for a word."""
    try:
        # Set a query timeout
        db.session.execute(text("SET statement_timeout TO '10000'"))  # 10 seconds
        
        # Get word using direct SQL
        word = db.session.execute(text(
            "SELECT id, lemma, language_code FROM words WHERE id = :word_id"
        ), {"word_id": word_id}).fetchone()
        
        if not word:
            return jsonify({"error": "Word not found"}), 404
        
        session = db.session
        
        def get_word_by_lemma(lemma, language=None):
            """Find word by lemma and language."""
            try:
                if language:
                    result = session.execute(text(
                        """SELECT id, lemma, language_code 
                           FROM words 
                           WHERE normalized_lemma = :norm AND language_code = :lang
                           LIMIT 1"""
                    ), {"norm": normalize_lemma(lemma), "lang": language}).fetchone()
                else:
                    result = session.execute(text(
                        """SELECT id, lemma, language_code 
                           FROM words 
                           WHERE normalized_lemma = :norm
                           LIMIT 1"""
                    ), {"norm": normalize_lemma(lemma)}).fetchone()
                return result
            except Exception as e:
                logger.error(f"Error getting word by lemma: {str(e)}")
                return None
        
        def get_etymologies(word_id):
            """Get etymologies for a word."""
            try:
                return session.execute(text(
                    """SELECT id, etymology_text, language_codes
                       FROM etymologies
                       WHERE word_id = :word_id
                       LIMIT 5"""  # Limit to 5 etymologies per word to prevent timeouts
                ), {"word_id": word_id}).fetchall()
            except Exception as e:
                logger.error(f"Error getting etymologies: {str(e)}")
                return []
        
        def build_etymology_tree(word, depth=0, max_depth=2, visited=None):
            # Limit max depth to 2 to prevent hanging
            if visited is None:
                visited = set()
            if depth > max_depth or word.id in visited or len(visited) > 30:  # Also limit total visited nodes
                return None
                
            visited.add(word.id)
            
            tree = {
                "word": word.lemma,
                "language": word.language_code,
                "etymologies": []
            }
            
            etymologies = get_etymologies(word.id)
            if not etymologies:
                return tree  # Return early if no etymologies
                
            for etymology in etymologies:
                etym_data = {
                    "text": etymology.etymology_text,
                    "languages": etymology.language_codes.split(',') if etymology.language_codes else []
                }
                
                # Skip component extraction if depth is already at limit
                if depth < max_depth:
                    # Extract components
                    try:
                        components = extract_etymology_components(etymology.etymology_text)
                        etym_data["components"] = components
                        
                        # Only process a limited number of components
                        processed = 0
                        # Make sure components is not None before trying to slice it
                        if components and isinstance(components, list):
                            for comp in components[:3]:  # Limit to first 3 components
                                if processed >= 2:  # Limit to 2 recursive lookups
                                    break
                                    
                                if comp.get('text') and comp.get('language'):
                                    related = get_word_by_lemma(comp['text'], comp['language'])
                                    if related and related.id not in visited:
                                        subtree = build_etymology_tree(
                                            related, depth + 1, max_depth, visited
                                        )
                                        if subtree:
                                            etym_data["derived_from"] = subtree
                                            processed += 1
                        # Handle case where components is a dictionary with original_text
                        elif components and isinstance(components, dict) and 'original_text' in components:
                            etym_data["components"] = [components]  # Wrap in list for consistent interface
                    except Exception as e:
                        etym_data["components"] = []
                        logger.error(f"Error extracting etymology components: {str(e)}")
                                
                tree["etymologies"].append(etym_data)
                
            return tree
            
        # Apply maximum timeout for tree building
        max_time = time.time() + 8  # 8 second max processing time
        etymology_tree = None
        
        try:
            etymology_tree = build_etymology_tree(word)
            
            # If we're taking too long, return what we have so far
            if time.time() > max_time:
                logger.warning("Etymology tree building timed out, returning partial result")
        except Exception as e:
            logger.error(f"Error building etymology tree: {str(e)}")
            return jsonify({
                "error": "Error building etymology tree",
                "message": str(e)
            }), 500
        
        if not etymology_tree:
            etymology_tree = {
                "word": word.lemma,
                "language": word.language_code,
                "etymologies": []
            }
        
        return jsonify({
            "word": word.lemma,
            "etymology_tree": etymology_tree,
            "complete": time.time() <= max_time
        }), 200
        
    except Exception as e:
        logger.error("Error processing etymology tree request",
                    word_id=word_id,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word_identifier>/relations/graph", methods=["GET"])
def get_relations_graph(word_identifier):
    """Get a graph representation of word relations, accepting ID or lemma."""
    try:
        # Set a timeout for these queries
        db.session.execute(text("SET statement_timeout TO '10000'"))  # 10 seconds
        
        word = None
        word_id = None

        # Try to convert to int for ID lookup
        try:
            word_id = int(word_identifier)
            word = db.session.execute(text(
                "SELECT id, lemma, language_code FROM words WHERE id = :word_id"
            ), {"word_id": word_id}).fetchone()
        except ValueError:
            # If not an integer, assume it's a lemma
            normalized_lemma = normalize_lemma(word_identifier)
            word = db.session.execute(text(
                "SELECT id, lemma, language_code FROM words WHERE normalized_lemma = :normalized"
            ), {"normalized": normalized_lemma}).fetchone()

        if not word:
            # Use get_word_suggestions if lookup failed
            suggestions = get_word_suggestions(word_identifier)
            return jsonify({
                "error": "Word not found",
                "suggestions": suggestions
            }), 404

        # Store the actual word ID found
        word_id = word.id

        # Continue with the rest of the original function logic using word_id
        session = db.session
        # Limit depth and complexity to avoid hanging
        max_depth = min(int(request.args.get('max_depth', 1)), 2)  # Limit max depth to 2
        include_bidirectional = request.args.get('bidirectional', 'true').lower() == 'true'
        
        # Get all relations with depth limit
        nodes = {}
        edges = []
        visited = set()
        
        def add_node(node_id, lemma, language_code):
            """Add node to graph if not already present."""
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id,
                    "label": lemma,
                    "language": language_code
                }
        
        def get_relations(node_id, depth=0):
            """Get relations recursively with depth limit."""
            if depth > max_depth or node_id in visited or len(nodes) > 100:  # Limit nodes to 100
                return
                
            visited.add(node_id)
            
            # Get outgoing relations
            outgoing = session.execute(text("""
                SELECT r.id, r.relation_type, 
                       w.id as target_id, w.lemma as target_lemma, w.language_code as target_language
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :node_id
                LIMIT 20  /* Limit number of relations */
            """), {"node_id": node_id}).fetchall()
            
            # Add relations to graph
            for rel in outgoing:
                add_node(rel.target_id, rel.target_lemma, rel.target_language)
                
                edges.append({
                    "source": node_id,
                    "target": rel.target_id,
                    "type": rel.relation_type,
                    "bidirectional": False  # Default to false since column doesn't exist
                })
                
                # Recurse if not at max depth
                if depth < max_depth:
                    get_relations(rel.target_id, depth + 1)
            
            # Get incoming relations if needed
            if include_bidirectional:
                incoming = session.execute(text("""
                    SELECT r.id, r.relation_type, 
                           w.id as source_id, w.lemma as source_lemma, w.language_code as source_language
                    FROM relations r
                    JOIN words w ON r.from_word_id = w.id
                    WHERE r.to_word_id = :node_id
                    LIMIT 20  /* Limit number of relations */
                """), {"node_id": node_id}).fetchall()
                
                # Add relations to graph
                for rel in incoming:
                    add_node(rel.source_id, rel.source_lemma, rel.source_language)
                    
                    # Only add edge if it's not already there (avoid duplicates)
                    edge_exists = any(e for e in edges if 
                                    e["source"] == rel.source_id and 
                                    e["target"] == node_id)
                    
                    if not edge_exists:
                        edges.append({
                            "source": rel.source_id,
                            "target": node_id,
                            "type": rel.relation_type,
                            "bidirectional": False  # Default to false since column doesn't exist
                        })
                    
                    # Recurse if not at max depth
                    if depth < max_depth:
                        get_relations(rel.source_id, depth + 1)
        
        # Add starting node
        add_node(word.id, word.lemma, word.language_code)
        
        # Build graph
        get_relations(word.id) # Use word.id here which is the correct ID
        
        # Return graph data
        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges,
            "metadata": {
                "word_id": word.id,
                "word": word.lemma,
                "max_depth": max_depth,
                "include_bidirectional": include_bidirectional,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error building relation graph: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route('/test', methods=['GET'])
def test_api():
    """Simple test endpoint to verify API is working."""
    try:
        # Get database info
        db_info = {}
        
        # Check if database is connected
        try:
            db_status = db.session.execute(text('SELECT 1')).scalar() is not None
            db_info["connected"] = db_status
            
            # Get some basic statistics with timeouts
            if db_status:
                try:
                    # Set a timeout for these queries
                    db.session.execute(text("SET statement_timeout TO '5000'"))
                    
                    word_count = db.session.execute(text("SELECT COUNT(*) FROM words")).scalar()
                    language_count = db.session.execute(text(
                        "SELECT COUNT(DISTINCT language_code) FROM words"
                    )).scalar()
                    
                    db_info["word_count"] = word_count
                    db_info["language_count"] = language_count
                except Exception as e:
                    # If queries time out, just return partial info
                    db_info["stats_error"] = str(e)
        except Exception as e:
            db_info["connected"] = False
            db_info["error"] = str(e)
        
        return jsonify({
            'status': 'success',
            'message': 'API is working properly!',
            'database': db_info,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'api_version': '2.0.0'
        })
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'API error occurred',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@bp.route("/words/<path:word>/comprehensive", methods=["GET"])
def get_word_comprehensive(word: str):
    """Get comprehensive word details including all relationships and metadata."""
    try:
        # Set a query timeout for safety
        db.session.execute(text("SET statement_timeout TO '10000'"))  # 10 seconds
        
        # Get word ID first using direct SQL - only include columns that exist in the database
        normalized_word = normalize_lemma(word)
        word_result = db.session.execute(text("""
            SELECT id, lemma, normalized_lemma, language_code, 
                   COALESCE(has_baybayin, FALSE) as has_baybayin, 
                   baybayin_form, romanized_form,
                   root_word_id, preferred_spelling, tags, 
                   data_hash, search_text,
                   created_at, updated_at
            FROM words WHERE normalized_lemma = :normalized
        """), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        word_id = word_result.id
        
        # Build comprehensive response with safe access for all fields
        response = {
            # Basic word info
            "id": word_id,
            "lemma": word_result.lemma,
            "normalized_lemma": word_result.normalized_lemma,
            "language_code": word_result.language_code,
            "has_baybayin": bool(word_result.has_baybayin),
            "baybayin_form": word_result.baybayin_form,
            "romanized_form": word_result.romanized_form,
            "root_word_id": word_result.root_word_id,
            "preferred_spelling": word_result.preferred_spelling,
            "tags": word_result.tags,
            "data_hash": word_result.data_hash,
            "search_text": word_result.search_text,
            "created_at": word_result.created_at.isoformat() if word_result.created_at else None,
            "updated_at": word_result.updated_at.isoformat() if word_result.updated_at else None,
        }
        
        # Get definitions with a new transaction
        try:
            definitions = db.session.execute(text("""
                SELECT id, definition_text, COALESCE(original_pos, '') as original_pos, 
                       standardized_pos_id, COALESCE(examples, '') as examples, 
                       COALESCE(usage_notes, '') as usage_notes, COALESCE(tags, '') as tags,
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM definitions WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["definitions"] = []
            for d in definitions:
                def_data = {
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "original_pos": d.original_pos,
                    "standardized_pos_id": d.standardized_pos_id,
                    "examples": d.examples,
                    "usage_notes": d.usage_notes,
                    "tags": d.tags,
                    "sources": d.sources,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "updated_at": d.updated_at.isoformat() if d.updated_at else None
                }
                
                if d.standardized_pos_id:
                    try:
                        pos_info = db.session.execute(text("""
                            SELECT id, code, name_en, name_tl, description
                            FROM parts_of_speech WHERE id = :pos_id
                        """), {"pos_id": d.standardized_pos_id}).fetchone()
                        
                        if pos_info:
                            def_data["standardized_pos"] = {
                                "id": pos_info.id,
                                "code": pos_info.code,
                                "name_en": pos_info.name_en,
                                "name_tl": pos_info.name_tl,
                                "description": pos_info.description
                            }
                    except Exception as e:
                        logger.error(f"Error fetching pos info: {str(e)}")
                        def_data["standardized_pos"] = None
                
                response["definitions"].append(def_data)
        except Exception as e:
            logger.error(f"Error fetching definitions: {str(e)}")
            response["definitions"] = []
            
        # Get etymologies with a new transaction
        try:
            etymologies = db.session.execute(text("""
                SELECT id, etymology_text, normalized_components, etymology_structure, language_codes,
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM etymologies WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["etymologies"] = []
            for etym in etymologies:
                etym_data = {
                    "id": etym.id,
                    "etymology_text": etym.etymology_text,
                    "normalized_components": etym.normalized_components,
                    "etymology_structure": etym.etymology_structure,
                    "language_codes": etym.language_codes,
                    "sources": etym.sources,
                    "created_at": etym.created_at.isoformat() if etym.created_at else None,
                    "updated_at": etym.updated_at.isoformat() if etym.updated_at else None
                }
                
                # Try to extract components if available
                try:
                    components = extract_etymology_components(etym.etymology_text)
                    if components:
                        if isinstance(components, dict) and 'original_text' in components:
                            etym_data["components"] = [components]
                        else:
                            etym_data["components"] = components
                except Exception as e:
                    logger.error(f"Error extracting etymology components: {str(e)}")
                    etym_data["components"] = []
                    
                response["etymologies"].append(etym_data)
        except Exception as e:
            logger.error(f"Error fetching etymologies: {str(e)}")
            response["etymologies"] = []
        
        # Get pronunciations with a new transaction
        try:
            pronunciations = db.session.execute(text("""
                SELECT id, type, value, COALESCE(tags, '{}') as tags, 
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM pronunciations WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["pronunciations"] = []
            for pron in pronunciations:
                response["pronunciations"].append({
                    "id": pron.id,
                    "type": pron.type,
                    "value": pron.value,
                    "tags": pron.tags,
                    "sources": pron.sources,
                    "created_at": pron.created_at.isoformat() if pron.created_at else None,
                    "updated_at": pron.updated_at.isoformat() if pron.updated_at else None
                })
        except Exception as e:
            logger.error(f"Error fetching pronunciations: {str(e)}")
            response["pronunciations"] = []
        
        # Get credits with a new transaction
        try:
            credits = db.session.execute(text("""
                SELECT id, credit, created_at, updated_at
                FROM credits WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["credits"] = []
            for credit in credits:
                response["credits"].append({
                    "id": credit.id,
                    "credit": credit.credit,
                    "created_at": credit.created_at.isoformat() if credit.created_at else None,
                    "updated_at": credit.updated_at.isoformat() if credit.updated_at else None
                })
        except Exception as e:
            logger.error(f"Error fetching credits: {str(e)}")
            response["credits"] = []
            
        # Get root word details if this word has a root
        response["root_word"] = None
        if word_result.root_word_id:
            try:
                root_word = db.session.execute(text("""
                    SELECT id, lemma, normalized_lemma, language_code, 
                           COALESCE(has_baybayin, FALSE) as has_baybayin, baybayin_form
                    FROM words WHERE id = :root_id
                """), {"root_id": word_result.root_word_id}).fetchone()
                
                if root_word:
                    response["root_word"] = {
                        "id": root_word.id,
                        "lemma": root_word.lemma,
                        "normalized_lemma": root_word.normalized_lemma,
                        "language_code": root_word.language_code,
                        "has_baybayin": bool(root_word.has_baybayin),
                        "baybayin_form": root_word.baybayin_form
                    }
            except Exception as e:
                logger.error(f"Error fetching root word: {str(e)}")
        
        # Get derived words with a new transaction
        try:
            derived_words = db.session.execute(text("""
                SELECT id, lemma, normalized_lemma, language_code, 
                       COALESCE(has_baybayin, FALSE) as has_baybayin, baybayin_form
                FROM words WHERE root_word_id = :word_id
                LIMIT 100
            """), {"word_id": word_id}).fetchall()
            
            response["derived_words"] = []
            for derived in derived_words:
                response["derived_words"].append({
                    "id": derived.id,
                    "lemma": derived.lemma,
                    "normalized_lemma": derived.normalized_lemma,
                    "language_code": derived.language_code,
                    "has_baybayin": bool(derived.has_baybayin),
                    "baybayin_form": derived.baybayin_form
                })
        except Exception as e:
            logger.error(f"Error fetching derived words: {str(e)}")
            response["derived_words"] = []
            
        # Get relations with a new transaction
        try:
            outgoing_relations = db.session.execute(text("""
                SELECT r.id, r.relation_type, COALESCE(r.metadata, '{}') as metadata, COALESCE(r.sources, '') as sources,
                       w.id as target_id, w.lemma as target_lemma, 
                       w.language_code as target_language_code,
                       COALESCE(w.has_baybayin, FALSE) as target_has_baybayin,
                       w.baybayin_form as target_baybayin_form
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["outgoing_relations"] = []
            for rel in outgoing_relations:
                response["outgoing_relations"].append({
                    "id": rel.id,
                    "relation_type": rel.relation_type,
                    "metadata": rel.metadata,
                    "sources": rel.sources,
                    "target_word": {
                        "id": rel.target_id,
                        "lemma": rel.target_lemma,
                        "language_code": rel.target_language_code,
                        "has_baybayin": bool(rel.target_has_baybayin),
                        "baybayin_form": rel.target_baybayin_form
                    }
                })
        except Exception as e:
            logger.error(f"Error fetching outgoing relations: {str(e)}")
            response["outgoing_relations"] = []
        
        # Get incoming relations with a new transaction
        try:
            incoming_relations = db.session.execute(text("""
                SELECT r.id, r.relation_type, COALESCE(r.metadata, '{}') as metadata, COALESCE(r.sources, '') as sources,
                       w.id as source_id, w.lemma as source_lemma, 
                       w.language_code as source_language_code,
                       COALESCE(w.has_baybayin, FALSE) as source_has_baybayin,
                       w.baybayin_form as source_baybayin_form
                FROM relations r
                JOIN words w ON r.from_word_id = w.id
                WHERE r.to_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["incoming_relations"] = []
            for rel in incoming_relations:
                response["incoming_relations"].append({
                    "id": rel.id,
                    "relation_type": rel.relation_type,
                    "metadata": rel.metadata,
                    "sources": rel.sources,
                    "source_word": {
                        "id": rel.source_id,
                        "lemma": rel.source_lemma,
                        "language_code": rel.source_language_code,
                        "has_baybayin": bool(rel.source_has_baybayin),
                        "baybayin_form": rel.source_baybayin_form
                    }
                })
        except Exception as e:
            logger.error(f"Error fetching incoming relations: {str(e)}")
            response["incoming_relations"] = []
        
        # Get affixations with a new transaction
        try:
            root_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.created_at, a.updated_at, COALESCE(a.sources, '') as sources,
                       w.id as affixed_id, w.lemma as affixed_lemma, 
                       w.language_code as affixed_language_code,
                       COALESCE(w.has_baybayin, FALSE) as affixed_has_baybayin,
                       w.baybayin_form as affixed_baybayin_form
                FROM affixations a
                JOIN words w ON a.affixed_word_id = w.id
                WHERE a.root_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["root_affixations"] = []
            for aff in root_affixations:
                response["root_affixations"].append({
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "sources": aff.sources,
                    "created_at": aff.created_at.isoformat() if aff.created_at else None,
                    "updated_at": aff.updated_at.isoformat() if aff.updated_at else None,
                    "affixed_word": {
                        "id": aff.affixed_id,
                        "lemma": aff.affixed_lemma,
                        "language_code": aff.affixed_language_code,
                        "has_baybayin": bool(aff.affixed_has_baybayin),
                        "baybayin_form": aff.affixed_baybayin_form
                    }
                })
        except Exception as e:
            logger.error(f"Error fetching root affixations: {str(e)}")
            response["root_affixations"] = []
        
        # Get affixed affixations with a new transaction
        try:
            affixed_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.created_at, a.updated_at, COALESCE(a.sources, '') as sources,
                       w.id as root_id, w.lemma as root_lemma, 
                       w.language_code as root_language_code,
                       COALESCE(w.has_baybayin, FALSE) as root_has_baybayin,
                       w.baybayin_form as root_baybayin_form
                FROM affixations a
                JOIN words w ON a.root_word_id = w.id
                WHERE a.affixed_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            response["affixed_affixations"] = []
            for aff in affixed_affixations:
                response["affixed_affixations"].append({
                    "id": aff.id,
                    "affix_type": aff.affix_type,
                    "sources": aff.sources,
                    "created_at": aff.created_at.isoformat() if aff.created_at else None,
                    "updated_at": aff.updated_at.isoformat() if aff.updated_at else None,
                    "root_word": {
                        "id": aff.root_id,
                        "lemma": aff.root_lemma,
                        "language_code": aff.root_language_code,
                        "has_baybayin": bool(aff.root_has_baybayin),
                        "baybayin_form": aff.root_baybayin_form
                    }
                })
        except Exception as e:
            logger.error(f"Error fetching affixed affixations: {str(e)}")
            response["affixed_affixations"] = []
            
        # Add data completeness metrics
        response["data_completeness"] = {
            "has_definitions": bool(response.get("definitions", [])),
            "has_etymology": bool(response.get("etymologies", [])),
            "has_pronunciations": bool(response.get("pronunciations", [])),
            "has_baybayin": bool(response.get("has_baybayin") and response.get("baybayin_form")),
            "has_relations": bool(response.get("outgoing_relations", []) or response.get("incoming_relations", [])),
            "has_affixations": bool(response.get("root_affixations", []) or response.get("affixed_affixations", [])),
        }
        
        # Calculate some summary stats
        relation_types = {}
        for rel in response.get("outgoing_relations", []):
            relation_types[rel["relation_type"]] = relation_types.get(rel["relation_type"], 0) + 1
        for rel in response.get("incoming_relations", []):
            relation_types[rel["relation_type"]] = relation_types.get(rel["relation_type"], 0) + 1
            
        response["relation_summary"] = relation_types
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting comprehensive word data: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Add an endpoint to get all relationship types
@bp.route("/relationships/types", methods=["GET"])
def get_relationship_types():
    """Get all available relationship types."""
    try:
        # Query unique relationship types from the database
        relationship_types = db.session.execute(text("""
            SELECT DISTINCT relation_type
            FROM relations
            ORDER BY relation_type
        """)).fetchall()
        
        types_list = [rel[0] for rel in relationship_types]
        
        # Add metadata about standard relationship types
        standard_types = {
            "synonym": {"bidirectional": True, "category": "semantic"},
            "antonym": {"bidirectional": True, "category": "semantic"},
            "related": {"bidirectional": True, "category": "semantic"},
            "similar": {"bidirectional": True, "category": "semantic"},
            "hypernym": {"bidirectional": False, "category": "taxonomic", "inverse": "hyponym"},
            "hyponym": {"bidirectional": False, "category": "taxonomic", "inverse": "hypernym"},
            "meronym": {"bidirectional": False, "category": "taxonomic", "inverse": "holonym"},
            "holonym": {"bidirectional": False, "category": "taxonomic", "inverse": "meronym"},
            "derived_from": {"bidirectional": False, "category": "derivational", "inverse": "root_of"},
            "root_of": {"bidirectional": False, "category": "derivational", "inverse": "derived_from"},
            "variant": {"bidirectional": True, "category": "variant"},
            "spelling_variant": {"bidirectional": True, "category": "variant"},
            "regional_variant": {"bidirectional": True, "category": "variant"},
            "compare_with": {"bidirectional": True, "category": "usage"},
            "see_also": {"bidirectional": True, "category": "usage"},
            "equals": {"bidirectional": True, "category": "other"}
        }
        
        result = []
        for rel_type in types_list:
            if rel_type in standard_types:
                result.append({
                    "type": rel_type,
                    "metadata": standard_types[rel_type]
                })
            else:
                result.append({
                    "type": rel_type,
                    "metadata": {"category": "unknown"}
                })
        
        return jsonify({
            "relationship_types": result,
            "total": len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting relationship types: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Add an endpoint to find words by relationship type
@bp.route("/relationships/<relationship_type>", methods=["GET"])
def get_words_by_relationship(relationship_type: str):
    """Find all word pairs with a specific relationship type."""
    try:
        # Pagination parameters
        limit = min(int(request.args.get('limit', 100)), 1000)  # Cap at 1000 to prevent excessive queries
        offset = int(request.args.get('offset', 0))
        
        # Query relationships with the specified type
        relationships = db.session.execute(text("""
            SELECT r.id, r.relation_type, r.metadata,
                   w1.id as source_id, w1.lemma as source_lemma, w1.language_code as source_language,
                   w2.id as target_id, w2.lemma as target_lemma, w2.language_code as target_language
            FROM relations r
            JOIN words w1 ON r.from_word_id = w1.id
            JOIN words w2 ON r.to_word_id = w2.id
            WHERE r.relation_type = :relationship_type
            ORDER BY w1.lemma, w2.lemma
            LIMIT :limit OFFSET :offset
        """), {"relationship_type": relationship_type, "limit": limit, "offset": offset}).fetchall()
        
        # Count total relationships of this type
        total_count = db.session.execute(text("""
            SELECT COUNT(*) FROM relations WHERE relation_type = :relationship_type
        """), {"relationship_type": relationship_type}).scalar()
        
        result = []
        for rel in relationships:
            result.append({
                "id": rel.id,
                "relation_type": rel.relation_type,
                "metadata": rel.metadata,
                "source_word": {
                    "id": rel.source_id,
                    "lemma": rel.source_lemma,
                    "language_code": rel.source_language
                },
                "target_word": {
                    "id": rel.target_id,
                    "lemma": rel.target_lemma,
                    "language_code": rel.target_language
                }
            })
        
        return jsonify({
            "relationship_type": relationship_type,
            "relationships": result,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting relationships of type {relationship_type}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# --- NEW RANDOM WORD ENDPOINT ---
@bp.route("/random", methods=["GET"])
def get_random_word():
    """Get comprehensive details for a randomly selected word."""
    try:
        # Set a query timeout for safety
        db.session.execute(text("SET statement_timeout TO '10000'"))  # 10 seconds

        # 1. Get Min/Max IDs
        id_range = db.session.execute(text(
            """SELECT MIN(id) as min_id, MAX(id) as max_id, COUNT(*) as count
               FROM words"""
        )).fetchone()

        if not id_range or id_range.count == 0:
            logger.warning("Attempted to get random word from empty table.")
            return jsonify({"error": "No words found in the dictionary."}), 404

        min_id, max_id = id_range.min_id, id_range.max_id
        if min_id is None or max_id is None: # Should not happen if count > 0, but safety check
             logger.error("Could not determine min/max ID range for random word.")
             return jsonify({"error": "Could not determine ID range."}), 500

        # 2. Find a random word ID efficiently
        random_word_id = None
        attempts = 0
        max_attempts = 10 # Prevent infinite loops in edge cases (e.g., very sparse IDs)

        while random_word_id is None and attempts < max_attempts:
            attempts += 1
            # Pick a random ID in the full range
            random_target_id = random.randint(min_id, max_id)

            # Try finding the first word ID >= target
            result = db.session.execute(text(
                """SELECT id FROM words WHERE id >= :target_id ORDER BY id LIMIT 1"""
            ), {"target_id": random_target_id}).fetchone()

            if result:
                random_word_id = result.id
            else:
                # If no word found >= target (likely due to gaps at the end),
                # try finding the last word ID <= target.
                result = db.session.execute(text(
                    """SELECT id FROM words WHERE id <= :target_id ORDER BY id DESC LIMIT 1"""
                ), {"target_id": random_target_id}).fetchone()
                if result:
                    random_word_id = result.id
                # If still no word found (highly unlikely unless table is *very* sparse), loop again

        if random_word_id is None:
            logger.error("Failed to find a random word ID after multiple attempts.")
            # Fallback: just get the first word
            result = db.session.execute(text("SELECT id FROM words ORDER BY id LIMIT 1")).fetchone()
            if not result:
                 return jsonify({"error": "Failed to select any word."}), 500
            random_word_id = result.id


        # 3. Fetch comprehensive data for the selected word ID using the helper
        logger.info(f"Fetching comprehensive details for random word_id: {random_word_id}")
        # Use the refactored helper function
        word_details = _fetch_word_details(random_word_id, session=db.session, include_all=True)

        if not word_details:
             # This should theoretically not happen if we found an ID above
             logger.error("Found random word ID but failed to fetch its data.", word_id=random_word_id)
             return jsonify({"error": "Could not fetch data for the selected random word."}), 404

        # 4. Serialize the result using WordSchema
        schema = WordSchema()
        result = schema.dump(word_details)

        # Add data completeness and summary (optional, copied from get_word_comprehensive refactor)
        completeness = {
            "has_definitions": bool(word_details.definitions),
            "has_etymology": bool(word_details.etymologies),
            "has_pronunciations": bool(word_details.pronunciations),
            "has_baybayin": bool(word_details.has_baybayin and word_details.baybayin_form),
            "has_relations": bool(word_details.outgoing_relations or word_details.incoming_relations),
            "has_affixations": bool(word_details.root_affixations or word_details.affixed_affixations),
        }
        result["data_completeness"] = completeness
        relation_summary = defaultdict(int)
        for rel in word_details.outgoing_relations:
            relation_summary[rel.relation_type] += 1
        for rel in word_details.incoming_relations:
            relation_summary[rel.relation_type] += 1
        result["relation_summary"] = dict(relation_summary)


        # 5. Return the response
        logger.info(f"Successfully retrieved random word: id={random_word_id}, lemma={word_details.lemma}")
        return jsonify(result), 200

    except SQLAlchemyError as db_err:
        db.session.rollback()
        logger.error(f"Database error fetching random word: {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "details": str(db_err)}), 500
    except Exception as e:
        db.session.rollback() # Rollback on any general error too
        logger.error(f"Error fetching random word: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# --- Helper Functions ---

def _fetch_word_details(
    word_id: int,
    session: Session, # Pass session for explicit control
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
        word_id: The ID of the word to fetch.
        session: The SQLAlchemy session to use.
        include_*: Flags to determine which related data to load.

    Returns:
        The Word object with eagerly loaded data, or None if not found.
    """
    options = []

    # Basic Word Data: Always fetched by get()

    # --- Relationships ---
    if include_definitions:
        # Use joinedload for PartOfSpeech as it's often needed with definitions (many-to-one)
        options.append(selectinload(Word.definitions).joinedload(Definition.standardized_pos))
    if include_etymologies:
        options.append(selectinload(Word.etymologies))
    if include_pronunciations:
        options.append(selectinload(Word.pronunciations)) # Assumes relationship name 'pronunciations'
    if include_credits:
        options.append(selectinload(Word.credits)) # Assumes relationship name 'credits'
    if include_relations:
         # Load related words along with the relation itself
         options.append(selectinload(Word.outgoing_relations).joinedload(Relation.target_word))
         options.append(selectinload(Word.incoming_relations).joinedload(Relation.source_word))
    if include_affixations:
         # Load related words along with the affixation itself
         # Assumes relationships 'root_affixations' and 'affixed_affixations'
         options.append(selectinload(Word.root_affixations).joinedload(Affixation.affixed_word))
         options.append(selectinload(Word.affixed_affixations).joinedload(Affixation.root_word))
    if include_root:
        # Root word is many-to-one, joinedload is efficient
        options.append(joinedload(Word.root_word))
    if include_derived:
        # Derived words are one-to-many, selectinload preferred
        options.append(selectinload(Word.derived_words)) # Assumes relationship name 'derived_words'

    # Fetch the word with the specified options
    # Use session.get() for primary key lookup if available (SQLAlchemy 2.0+)
    # Fallback to query.get() for compatibility
    if hasattr(session, 'get'):
        word = session.get(Word, word_id, options=options)
    else: # SQLAlchemy 1.x compatibility
        word = session.query(Word).options(*options).get(word_id) # Use session passed in

    return word

@bp.route("/api/search/suggestions", methods=["GET"])
@cached_query(timeout=60)  # Cache for 1 minute
def get_search_suggestions():
    """
    Get efficient search suggestions optimized for cloud deployment.
    
    Query parameters:
    - q: Search query (required)
    - language: Filter by language code (optional)
    - limit: Maximum number of suggestions (optional, default 10)
    """
    # Track request metrics
    REQUEST_COUNT.inc()
    API_REQUESTS.labels(endpoint="search_suggestions", method="GET").inc()
    start_time = time.time()
    
    query = request.args.get("q", "").strip().lower()
    if not query or len(query) < 2:
        return jsonify({"suggestions": []})
        
    language = request.args.get("language")
    limit = min(int(request.args.get("limit", 10)), 20)
    
    # Single optimized query that combines multiple suggestion types
    sql = """
    WITH suggestions AS (
        -- Prefix matches (highest efficiency, uses index)
        SELECT 
            w.id, 
            w.lemma as text,
            w.normalized_lemma,
            w.language_code,
            w.has_baybayin,
            w.baybayin_form,
            'prefix_match' as type,
            0.95 as confidence,
            NULL as definition_preview,
            NULL as pos_code
        FROM 
            words w
        WHERE 
            w.normalized_lemma LIKE :prefix_query
            AND (:language IS NULL OR w.language_code = :language)
        ORDER BY 
            CASE WHEN w.root_word_id IS NULL THEN 1 ELSE 0 END DESC,
            LENGTH(w.lemma) ASC,
            w.lemma ASC
        LIMIT :prefix_limit
        
        UNION ALL
        
        -- Popular exact matches (from materialized view for efficiency)
        -- Only include if the materialized view exists
        SELECT 
            w.id, 
            w.lemma as text,
            w.normalized_lemma,
            w.language_code,
            w.has_baybayin,
            w.baybayin_form,
            'popular_match' as type,
            0.9 as confidence,
            NULL as definition_preview,
            NULL as pos_code
        FROM 
            words w
        JOIN
            popular_words pw ON w.id = pw.word_id
        WHERE 
            w.normalized_lemma LIKE :prefix_query
            AND (:language IS NULL OR w.language_code = :language)
        ORDER BY 
            pw.search_count DESC
        LIMIT :popular_limit
        
        UNION ALL
        
        -- Trigram similarity (more expensive, limited strictly)
        SELECT 
            w.id, 
            w.lemma as text,
            w.normalized_lemma,
            w.language_code,
            w.has_baybayin,
            w.baybayin_form,
            'spelling_suggestion' as type,
            SIMILARITY(w.normalized_lemma, :query) as confidence,
            NULL as definition_preview,
            NULL as pos_code
        FROM 
            words w
        WHERE 
            SIMILARITY(w.normalized_lemma, :query) > 0.3
            AND w.normalized_lemma != :normalized_query
            AND (:language IS NULL OR w.language_code = :language)
        ORDER BY 
            confidence DESC
        LIMIT :trigram_limit
        
        UNION ALL
        
        -- Definition matches (with snippet)
        SELECT 
            w.id, 
            w.lemma as text,
            w.normalized_lemma,
            w.language_code,
            w.has_baybayin,
            w.baybayin_form,
            'definition_match' as type,
            0.7 as confidence,
            SUBSTRING(d.definition_text, 1, 60) as definition_preview,
            p.code as pos_code
        FROM 
            words w
        JOIN 
            definitions d ON w.id = d.word_id
        LEFT JOIN
            parts_of_speech p ON d.standardized_pos_id = p.id
        WHERE 
            to_tsvector('english', d.definition_text) @@ plainto_tsquery('english', :query)
            AND (:language IS NULL OR w.language_code = :language)
        LIMIT :definition_limit
    )
    
    -- Main selection with deduplication and priority ordering
    SELECT DISTINCT ON (text) 
        id, 
        text,
        normalized_lemma,
        language_code,
        has_baybayin,
        baybayin_form,
        type,
        confidence,
        definition_preview,
        pos_code
    FROM 
        suggestions
    ORDER BY 
        text,                -- Deduplication key 
        confidence DESC,     -- Prioritize by confidence
        type                 -- Then by suggestion type
    LIMIT :overall_limit;
    """
    
    # Set query parameters
    normalized_query = normalize_lemma(query)
    params = {
        "query": query,
        "normalized_query": normalized_query,
        "prefix_query": f"{normalized_query}%",
        "language": language,
        "prefix_limit": max(limit, 5),  # Get at least 5 for prefix matches
        "popular_limit": min(limit // 2, 5),
        "trigram_limit": min(limit // 3, 3),
        "definition_limit": min(limit // 3, 3),
        "overall_limit": limit
    }
    
    try:
        # Execute single optimized query
        result = db.session.execute(text(sql), params).fetchall()
        
        # Build suggestions list from result
        suggestions = []
        for row in result:
            suggestion = {
                "id": row.id,
                "text": row.text,
                "type": row.type,
                "confidence": float(row.confidence),
                "language": row.language_code
            }
            
            # Only add these fields if they have values (reduce payload size)
            if row.has_baybayin and row.baybayin_form:
                suggestion["has_baybayin"] = True
                suggestion["baybayin_form"] = row.baybayin_form
                
            if row.definition_preview:
                suggestion["definition_preview"] = row.definition_preview
                
            if row.pos_code:
                suggestion["pos"] = row.pos_code
                
            suggestions.append(suggestion)
            
        # Optional: very lightweight logging (async/background)
        if len(query) > 2:
            try:
                log_search_query(query, language, query_type="suggestion")
            except Exception:
                # Never fail the main request for logging issues
                pass
                
        # Record request latency
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="search_suggestions").observe(request_time)
        
        return jsonify({"suggestions": suggestions})
    
    except Exception as e:
        # Log the error and fallback to basic suggestions
        logger.error(f"Error in search suggestions: {str(e)}")
        API_ERRORS.labels(endpoint="search_suggestions", error_type=type(e).__name__).inc()
        return get_basic_fallback_suggestions(query, language, limit)


def get_basic_fallback_suggestions(query, language, limit):
    """Ultra-basic fallback if complex suggestions fail."""
    try:
        # Simple query as last resort
        simple_sql = """
        SELECT id, lemma as text, language_code
        FROM words
        WHERE lemma ILIKE :prefix
        AND (:language IS NULL OR language_code = :language)
        ORDER BY LENGTH(lemma), lemma
        LIMIT :limit
        """
        
        result = db.session.execute(text(simple_sql), {
            "prefix": f"{query}%", 
            "language": language, 
            "limit": limit
        }).fetchall()
        
        suggestions = [{"id": row.id, "text": row.text, "language": row.language_code} for row in result]
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        logger.error(f"Even basic fallback suggestions failed: {str(e)}")
        # Absolute last resort
        return jsonify({"suggestions": []})


@bp.route("/api/search/track-selection", methods=["POST"])
def track_search_selection():
    """Track which suggestions users select to improve future suggestions."""
    if not request.is_json:
        return jsonify({"error": "Expected JSON data"}), 400
        
    data = request.json
    query = data.get("query")
    selected_id = data.get("selected_id")
    selected_text = data.get("selected_text")
    
    if not query or (not selected_id and not selected_text):
        return jsonify({"error": "Missing required parameters"}), 400
        
    try:
        # Use asynchronous task to record the selection
        log_search_query(
            query=selected_text or query,
            word_id=selected_id,
            query_type="selection"
        )
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Error tracking search selection: {str(e)}")
        return jsonify({"status": "received"})  # Still return success to client