"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast, asc
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE, post_dump
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
import structlog
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, Pronunciation, Credit, Language
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
    # Standardize sources field
    sources = fields.String(required=False, allow_none=True) # Allow null for sources

    def standardize_sources(self, value):
        if not value:
            return None
        # Assuming standardize_source_identifier exists and works
        # Split if comma-separated, standardize each, join back
        if isinstance(value, str):
            source_list = [s.strip() for s in value.split(',') if s.strip()]
            standardized_list = [standardize_source_identifier(s) for s in source_list]
            return ', '.join(sorted(list(set(standardized_list)))) # Ensure unique and sorted
        return value

    # Use pre_load or post_dump if needed for complex source handling
    @post_dump
    def format_sources(self, data, **kwargs):
        """Standardize the sources field after dumping."""
        if 'sources' in data and data.get('sources') and isinstance(data['sources'], str):
            try:
                source_list = [s.strip() for s in data['sources'].split(',') if s.strip()]
                # Ensure standardize_source_identifier is available in this scope
                standardized_list = [standardize_source_identifier(s) for s in source_list]
                # Ensure unique and sorted
                data['sources'] = ', '.join(sorted(list(set(standardized_list))))
            except NameError:
                # Handle case where standardize_source_identifier might not be imported/available
                logger.warning("standardize_source_identifier not found during source formatting.")
            except Exception as e:
                logger.error(f"Error formatting sources in post_dump: {e}")
        return data

class PronunciationType(BaseSchema): # Inherit from BaseSchema
    """Schema for pronunciation data."""
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic']))
    value = fields.Str(required=True)
    tags = fields.Dict()
    metadata = fields.Dict(attribute="metadata")
    # No need for word here, it's nested under WordSchema

class CreditSchema(BaseSchema):
    """Schema for word credits."""
    credit = fields.Str(required=True)
    # No need for word here, it's nested under WordSchema

class EtymologySchema(BaseSchema):
    """Schema for etymology data."""
    etymology_text = fields.Str(required=True)
    normalized_components = fields.String()
    etymology_structure = fields.String()
    language_codes = fields.String()
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))
    standardized_pos_id = fields.Int(load_only=True) # Load only, use nested object for dump
    examples = fields.String(required=False, allow_none=True) # Allow null
    usage_notes = fields.String(required=False, allow_none=True) # Allow null
    tags = fields.String(required=False, allow_none=True) # Allow null
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'), dump_only=True)
    standardized_pos = fields.Nested('PartOfSpeechSchema', dump_only=True) # Use nested POS schema for output
    components = fields.Method("get_extracted_components", dump_only=True)

    def get_extracted_components(self, obj):
        """Extract etymology components during serialization."""
        if not obj.etymology_text:
            return []
        try:
            extracted = extract_etymology_components(obj.etymology_text)
            # Adapt based on the actual return type of the helper
            if isinstance(extracted, dict) and 'original_text' in extracted:
                return [extracted] # Return as list with single dict
            elif isinstance(extracted, list):
                 return extracted # Assume list contains serializable items
            return [] # Or handle other formats
        except Exception as e:
            logger.warning(f"Error extracting etymology components during serialization for Etymology ID {obj.id}: {e}")
            return []

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True)
    metadata = fields.Dict()
    source_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    target_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class AffixationSchema(BaseSchema):
    """Schema for word affixation data."""
    affix_type = fields.Str(required=True)
    root_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    affixed_word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class DefinitionSchema(BaseSchema):
    """Schema for word definitions."""
    definition_text = fields.Str(required=True)
    original_pos = fields.String()
    standardized_pos_id = fields.Int()
    examples = fields.String()
    usage_notes = fields.String()
    tags = fields.String()
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))
    standardized_pos = fields.Nested('PartOfSpeechSchema')

class PartOfSpeechSchema(Schema):
    """Schema for parts of speech."""
    id = fields.Int(dump_only=True)
    code = fields.Str(required=True)
    name_en = fields.Str(required=True)
    name_tl = fields.Str(required=True)
    description = fields.String()
    is_abbreviation = fields.Bool()
    is_initialism = fields.Bool()
    is_root = fields.Bool()

    # Updated Relationships using ORM relationships
    definitions = fields.List(fields.Nested(DefinitionSchema, exclude=('word',))) # Exclude backref
    etymologies = fields.List(fields.Nested(EtymologySchema, exclude=('word',))) # Exclude backref
    pronunciations = fields.List(fields.Nested(PronunciationType)) # Add pronunciations
    credits = fields.List(fields.Nested(CreditSchema)) # Add credits
    root_word = fields.Nested('self', only=('id', 'lemma', 'language_code'), dump_only=True)
    derived_words = fields.List(fields.Nested('self', only=('id', 'lemma', 'language_code'), dump_only=True))

    # Relations might need more specific schemas based on direction
    outgoing_relations = fields.List(fields.Nested(RelationSchema, exclude=('source_word',))) # Exclude backref
    incoming_relations = fields.List(fields.Nested(RelationSchema, exclude=('target_word',))) # Exclude backref

    # Affixations might need more specific schemas based on direction
    root_affixations = fields.List(fields.Nested(AffixationSchema, exclude=('root_word',))) # Exclude backref
    affixed_affixations = fields.List(fields.Nested(AffixationSchema, exclude=('affixed_word',))) # Exclude backref

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
    data_hash = fields.String()
    search_text = fields.String()
    badlit_form = fields.String()
    hyphenation = fields.Dict()
    is_proper_noun = fields.Bool()
    is_abbreviation = fields.Bool()
    is_initialism = fields.Bool()
    is_root = fields.Bool()
    
    # Relationships
    definitions = fields.List(fields.Nested(DefinitionSchema))
    etymologies = fields.List(fields.Nested(EtymologySchema))
    pronunciations = fields.List(fields.Nested(PronunciationType))
    credits = fields.List(fields.Nested(CreditSchema))
    root_word = fields.Nested('self', only=('id', 'lemma', 'language_code'))
    derived_words = fields.List(fields.Nested('self', only=('id', 'lemma', 'language_code')))
    outgoing_relations = fields.List(fields.Nested(RelationSchema))
    incoming_relations = fields.List(fields.Nested(RelationSchema))
    root_affixations = fields.List(fields.Nested(AffixationSchema))
    affixed_affixations = fields.List(fields.Nested(AffixationSchema))

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
def get_word(word: str):
    """Get word details using ORM and Marshmallow schema."""
    try:
        # Use the ORM helper function
        word_obj = _get_word_details_orm(word)

        if not word_obj:
            # Keep suggestion logic for now (can also be refactored to ORM later)
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep using direct SQL for suggestions for now
            }), 404
        
        # Serialize using the updated WordSchema
        # Limit nested data for this basic endpoint
        schema = WordSchema(exclude=(
            'etymologies', 'pronunciations', 'credits',
            'outgoing_relations', 'incoming_relations',
            'root_affixations', 'affixed_affixations',
            'derived_words', 'root_word',
            # Exclude fields not typically needed for basic lookup
            'tags', 'idioms', 'source_info', 'data_hash', 'search_text',
            'badlit_form', 'hyphenation', 'is_proper_noun', 'is_abbreviation',
            'is_initialism', 'is_root'
        ))
        result = schema.dump(word_obj)

        return jsonify(result), 200

    except ValidationError as err:
        logger.error(f"Schema validation error: {err.messages}", word=word)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        logger.error(f"Error processing word request (ORM): {str(e)}", word=word, exc_info=True)
        db.session.rollback() # Rollback in case of ORM errors
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/search", methods=["GET"])
def search_words():
    """Search for words using ORM and Marshmallow schema."""
    try:
        # Validate query parameters using SearchQuerySchema
        try:
            search_args = SearchQuerySchema().load(request.args)
        except ValidationError as err:
            return jsonify({"error": "Invalid search parameters", "details": err.messages}), 400

        query_term = search_args['q']
        limit = search_args['limit']
        offset = search_args['offset']
        language = search_args.get('language')
        pos_code = search_args.get('pos')
        sort_by = search_args['sort']
        sort_order = search_args['order']

        # Build base query using ORM
        base_query = Word.query

        # Apply search term filter (case-insensitive like)
        norm_query = normalize_lemma(query_term)
        base_query = base_query.filter(
            or_(
                Word.lemma.ilike(f'%{query_term}%'),
                Word.normalized_lemma.ilike(f'%{norm_query}%')
                # Add other search modes later (fuzzy, baybayin, etc.)
            )
        )

        # Apply filters
        if language:
            base_query = base_query.filter(Word.language_code == language)
        if pos_code:
            pos_id = get_pos_id_orm(pos_code)
            if pos_id:
                # Join Definitions and PartOfSpeech to filter by POS code
                base_query = base_query.join(Word.definitions).filter(Definition.standardized_pos_id == pos_id)
            else:
                logger.warning(f"Invalid POS code specified in search: {pos_code}")
                # Return empty results if POS code is invalid? Or ignore?
                # return jsonify({'total': 0, 'words': [], 'query': query_term, 'filters': search_args}), 200


        # Apply sorting (add more sophisticated sorting later, e.g., relevance)
        sort_column = Word.lemma # Default sort
        if sort_by == 'alphabetical':
            sort_column = Word.normalized_lemma
        elif sort_by == 'created':
            sort_column = Word.created_at
        elif sort_by == 'updated':
            sort_column = Word.updated_at
        # Add other sort options: quality, frequency, complexity requires additional fields/logic

        if sort_order == 'desc':
            base_query = base_query.order_by(desc(sort_column))
        else:
            base_query = base_query.order_by(asc(sort_column))

        # Get total count before pagination
        total_count = base_query.count()

        # Apply pagination and eager loading for definitions (limit 3)
        results = base_query.limit(limit).offset(offset).options(
             selectinload(Word.definitions).selectinload(Definition.standardized_pos) # Eager load needed data
             # Limit definitions per word *after* fetching? Difficult with ORM directly.
             # Alternative: fetch only word IDs, then fetch details separately.
             # Simpler for now: Fetch all definitions, schema will handle exclusion/limits later if needed.
         ).all()

        # Serialize results using a simplified schema for search results
        search_result_schema = WordSchema(many=True, exclude=(
             # Exclude heavy fields for search results
             'etymologies', 'pronunciations', 'credits',
             'outgoing_relations', 'incoming_relations',
             'root_affixations', 'affixed_affixations',
             'derived_words', 'root_word',
             'tags', 'idioms', 'source_info', 'data_hash', 'search_text',
             'badlit_form', 'hyphenation', 'is_proper_noun', 'is_abbreviation',
             'is_initialism', 'is_root'
             # Maybe limit definitions shown in schema?
         ))
        word_list = search_result_schema.dump(results)

        # Manually limit definitions in the dumped data if needed
        for word_data in word_list:
            if 'definitions' in word_data:
                word_data['definitions'] = word_data['definitions'][:3] # Limit to 3 definitions
        
        return jsonify({
            'total': total_count,
            'words': word_list,
            'query': query_term,
            'filters': search_args # Show applied filters
        })
        
    except ValidationError as err:
         logger.error(f"Schema validation/serialization error in search: {err.messages}")
         return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        logger.error(f"Error processing search request (ORM): {str(e)}", exc_info=True)
        db.session.rollback() # Rollback in case of ORM errors
        return jsonify({'error': 'Internal server error'}), 500

@bp.route("/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get word relations using ORM."""
    try:
        # Use the ORM helper, but only load relation data
        word_obj = Word.query.filter(Word.normalized_lemma == normalize_lemma(word)).options(
            selectinload(Word.outgoing_relations).selectinload(Relation.target_word),
            selectinload(Word.incoming_relations).selectinload(Relation.source_word)
        ).first()

        if not word_obj:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep suggestion helper
            }), 404
        
        # Serialize relations using RelationSchema
        relation_schema = RelationSchema(many=True)
        outgoing = relation_schema.dump(word_obj.outgoing_relations)
        incoming = relation_schema.dump(word_obj.incoming_relations)
        
        return jsonify({
            'outgoing_relations': outgoing,
            'incoming_relations': incoming
        }), 200
        
    except ValidationError as err:
        logger.error(f"Schema validation error getting relations: {err.messages}", word=word)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        logger.error(f"Error getting word relations (ORM): {str(e)}", word=word, exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get all affixations for a word using ORM."""
    try:
        # Set a statement timeout manually if needed (less ideal with ORM)
        # db.session.execute(text("SET statement_timeout TO '5000'"))

        # Fetch word with affixations eagerly loaded
        word_obj = Word.query.filter(Word.normalized_lemma == normalize_lemma(word)).options(
            selectinload(Word.root_affixations).selectinload(Affixation.affixed_word),
            selectinload(Word.affixed_affixations).selectinload(Affixation.root_word)
        ).first()

        if not word_obj:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep suggestion helper
            }), 404
        
        # Serialize using AffixationSchema
        affixation_schema = AffixationSchema(many=True)
        as_root = affixation_schema.dump(word_obj.root_affixations)
        as_affixed = affixation_schema.dump(word_obj.affixed_affixations)
        
        return jsonify({
            'as_root': as_root,
            'as_affixed': as_affixed
        }), 200
        
    except ValidationError as err:
        logger.error(f"Schema validation error getting affixations: {err.messages}", word=word)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        # Consider rolling back timeout if set manually
        # db.session.execute(text("SET statement_timeout TO DEFAULT"))
        logger.error(f"Error getting word affixations (ORM): {str(e)}", word=word, exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get word pronunciation using ORM."""
    try:
        # Fetch word with pronunciations eagerly loaded
        word_obj = Word.query.filter(Word.normalized_lemma == normalize_lemma(word)).options(
            selectinload(Word.pronunciations)
        ).first()

        if not word_obj:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep suggestion helper
            }), 404
        
        # Serialize using PronunciationType schema
        pronunciation_schema = PronunciationType(many=True)
        pronunciation_list = pronunciation_schema.dump(word_obj.pronunciations)
        
        return jsonify({
            'pronunciations': pronunciation_list,
            'has_pronunciation': bool(pronunciation_list)
        }), 200
        
    except ValidationError as err:
        logger.error(f"Schema validation error getting pronunciation: {err.messages}", word=word)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        logger.error(f"Error getting word pronunciation (ORM): {str(e)}", word=word, exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics using ORM aggregation."""
    try:
        # Use ORM count for better maintainability
        stats = {}
        
        # Total word count
        stats['total_words'] = db.session.query(func.count(Word.id)).scalar()
        
        # Total definition count
        stats['total_definitions'] = db.session.query(func.count(Definition.id)).scalar()
        
        # Total etymology count
        stats['total_etymologies'] = db.session.query(func.count(Etymology.id)).scalar()

        # Total relations count
        stats['total_relations'] = db.session.query(func.count(Relation.id)).scalar()

        # Total affixations count
        stats['total_affixations'] = db.session.query(func.count(Affixation.id)).scalar()

        # Words by language
        lang_counts = db.session.query(
            Word.language_code, func.count(Word.id)
        ).group_by(Word.language_code).order_by(desc(func.count(Word.id))).all()
        stats['words_by_language'] = {lang: count for lang, count in lang_counts}

        # Words by POS (more complex join)
        pos_counts = db.session.query(
            PartOfSpeech.code, func.count(distinct(Definition.word_id)) # Count distinct words per POS
        ).join(Definition.standardized_pos) \
         .group_by(PartOfSpeech.code) \
         .order_by(desc(func.count(distinct(Definition.word_id)))) \
         .all()
        stats['words_by_pos'] = {code: count for code, count in pos_counts}

        # Other stats can be added similarly using ORM queries
        
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting statistics (ORM): {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@bp.route("/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get word etymology using ORM."""
    try:
        # Set a timeout manually if needed
        # db.session.execute(text("SET statement_timeout TO '5000'"))

        # Fetch word with etymologies eagerly loaded
        word_obj = Word.query.filter(Word.normalized_lemma == normalize_lemma(word)).options(
            selectinload(Word.etymologies)
        ).first()

        if not word_obj:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep suggestion helper
            }), 404
        
        # Serialize using EtymologySchema
        etymology_schema = EtymologySchema(many=True)
        etymology_list = etymology_schema.dump(word_obj.etymologies)

        # Add extracted components manually after dumping if needed by schema/frontend
        for etym_data in etymology_list:
             if etym_data.get('etymology_text'):
                 try:
                     components = extract_etymology_components(etym_data['etymology_text'])
                     # Assuming components might be dict or list based on routes code
                     if isinstance(components, dict) and 'original_text' in components:
                         etym_data["components"] = [components] # Ensure list format
                     else:
                          etym_data["components"] = components
                 except Exception as e:
                     logger.warning(f"Error extracting etymology components during serialization: {e}")
                     etym_data["components"] = []
             else:
                 etym_data["components"] = []
        
        return jsonify({
            'etymologies': etymology_list,
            'has_etymology': bool(etymology_list)
        }), 200
        
    except ValidationError as err:
        logger.error(f"Schema validation error getting etymology: {err.messages}", word=word)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except SQLAlchemyError as e: # Catch specific DB errors
        # db.session.execute(text("SET statement_timeout TO DEFAULT"))
        logger.error(f"Error getting word etymology (ORM): {str(e)}", word=word, exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Database error"}), 500
    except Exception as e:
        # db.session.execute(text("SET statement_timeout TO DEFAULT"))
        logger.error(f"Error getting word etymology (ORM General): {str(e)}", word=word, exc_info=True)
        db.session.rollback()
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
    """Get comprehensive word details using ORM helper and schema."""
    try:
        # Set timeout if needed
        # db.session.execute(text("SET statement_timeout TO '10000'"))

        # Use the central ORM helper
        word_obj = _get_word_details_orm(word)

        if not word_obj:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word) # Keep suggestion helper
            }), 404
        
        # Serialize using the full WordSchema
        schema = WordSchema()
        result = schema.dump(word_obj)

        # Etymology components are now handled by the EtymologySchema Method field

        # Add data completeness metrics (adapt calculate_data_completeness to use ORM object)
        # result["data_completeness"] = calculate_data_completeness_orm(word_obj) # Needs refactoring

        # Add relation summary (can be calculated from dumped data)
        relation_types = defaultdict(int)
        for rel in result.get("outgoing_relations", []):
            relation_types[rel["relation_type"]] += 1
        for rel in result.get("incoming_relations", []):
            # Avoid double counting if bidirectional flag is reliable, otherwise count both
            relation_types[rel["relation_type"]] += 1
        result["relation_summary"] = dict(relation_types)


        return jsonify(result), 200

    except ValidationError as err:
        try:
            logger.error(f"Schema validation error getting comprehensive data: {err.messages}", word=word)
            return jsonify({"error": "Data serialization error", "details": err.messages}), 500
        except Exception as e:
            logger.error(f"Error getting comprehensive word data (ORM): {str(e)}", word=word, exc_info=True)
            db.session.rollback()
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Add an endpoint to get all relationship types
@bp.route("/relationships/types", methods=["GET"])
def get_relationship_types():
    """Get all available relationship types from Enum or DB."""
    try:
        # Option 1: Use the Enum definition (if comprehensive)
        # standard_types = RelationshipType # Assuming this Enum exists and is complete

        # Option 2: Query distinct types from DB (as before, but using ORM)
        distinct_types = db.session.query(distinct(Relation.relation_type)).order_by(Relation.relation_type).all()
        types_list = [rel[0] for rel in distinct_types]

        # Enrich with metadata from Enum if available
        standard_types_meta = {
             rel_enum.rel_value: {
                 "bidirectional": rel_enum.bidirectional,
                 "category": rel_enum.category.value,
                 "inverse": rel_enum.inverse
             } for rel_enum in RelationshipType
        }
        
        result = []
        for rel_type_str in types_list:
            metadata = standard_types_meta.get(rel_type_str, {"category": "unknown"})
            result.append({
                "type": rel_type_str,
                "metadata": metadata
            })
        return jsonify({
            "relationship_types": result,
            "total": len(result)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting relationship types (ORM): {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

# Add an endpoint to find words by relationship type
@bp.route("/relationships/<relationship_type>", methods=["GET"])
def get_words_by_relationship(relationship_type: str):
    """Find all word pairs with a specific relationship type using ORM."""
    try:
        # Pagination parameters
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        # Base query
        base_query = Relation.query.filter(Relation.relation_type == relationship_type)

        # Count total
        total_count = base_query.count()

        # Fetch relationships with pagination and eager loading
        relationships = base_query.order_by(
            Relation.from_word_id, Relation.to_word_id # Need join for lemma sorting
        ).limit(limit).offset(offset).options(
            selectinload(Relation.source_word), # Eager load words
            selectinload(Relation.target_word)
        ).all()

        # Serialize results
        # Need a schema that includes nested source/target word basic info
        relation_with_words_schema = RelationSchema(many=True) # Use existing, ensure it nests words
        result = relation_with_words_schema.dump(relationships)
        
        return jsonify({
            "relationship_type": relationship_type,
            "relationships": result,
            "total": total_count,
            "limit": limit,
            "offset": offset
        }), 200
        
    except ValidationError as err:
        logger.error(f"Schema validation error getting relationships by type: {err.messages}", type=relationship_type)
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except Exception as e:
        logger.error(f"Error getting relationships of type {relationship_type} (ORM): {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

# --- NEW ORM HELPER ---
def _get_word_details_orm(word_identifier: Union[str, int]) -> Optional[Word]:
    """
    Fetches a Word object with its related data using SQLAlchemy ORM.
    Loads relationships efficiently using selectinload.
    Handles lookup by ID or normalized lemma.
    """
    query = Word.query

    # Determine if identifier is ID or lemma
    if isinstance(word_identifier, int):
        query = query.filter(Word.id == word_identifier)
    elif isinstance(word_identifier, str):
        normalized_word = normalize_lemma(word_identifier)
        query = query.filter(Word.normalized_lemma == normalized_word)
    else:
        logger.error(f"Invalid word_identifier type: {type(word_identifier)}")
        return None

    # Eagerly load relationships to avoid N+1 queries
    query = query.options(
        selectinload(Word.definitions).selectinload(Definition.standardized_pos),
        selectinload(Word.etymologies),
        selectinload(Word.pronunciations),
        selectinload(Word.credits),
        selectinload(Word.outgoing_relations).selectinload(Relation.target_word),
        selectinload(Word.incoming_relations).selectinload(Relation.source_word),
        selectinload(Word.root_affixations).selectinload(Affixation.affixed_word),
        selectinload(Word.affixed_affixations).selectinload(Affixation.root_word),
        selectinload(Word.root_word), # Load the root word if it exists
        selectinload(Word.derived_words) # Load derived words if needed (might be heavy)
    )

    word_obj = query.first()
    return word_obj

# --- REFACTORED RANDOM WORD ENDPOINT ---
@bp.route("/random", methods=["GET"])
def get_random_word():
    """Get comprehensive details for a randomly selected word using ORM."""
    try:
        # Set timeout if needed
        # db.session.execute(text("SET statement_timeout TO '10000'"))

        # More efficient random word ID selection for PostgreSQL
        result = db.session.query(func.min(Word.id), func.max(Word.id), func.count(Word.id)).one()
        min_id, max_id, word_count = result

        if not word_count or word_count == 0:
            logger.warning("Attempted to get random word from empty table.")
            return jsonify({"error": "No words found in the dictionary."}), 404

        # Attempt to find a random ID within the range
        random_word_id = None
        attempts = 0
        max_attempts = 10 # Prevent infinite loops

        while random_word_id is None and attempts < max_attempts:
            attempts += 1
            if min_id is None or max_id is None:
                 logger.error("Min/Max ID is None, cannot select random word.")
                 return jsonify({"error": "Could not determine ID range."}), 500
            # Pick a random ID in the full range
            random_target_id = random.randint(min_id, max_id)

            # Find first existing ID >= target
            result = db.session.query(Word.id).filter(Word.id >= random_target_id).order_by(asc(Word.id)).first()

            if result:
                random_word_id = result.id
            else:
                # If no word found >= target (gap at end), try finding last word ID <= target
                result = db.session.query(Word.id).filter(Word.id <= random_target_id).order_by(desc(Word.id)).first()
                if result:
                    random_word_id = result.id

        if random_word_id is None:
            logger.error("Failed to find a random word ID after multiple attempts. Falling back to first word.")
            first_word = db.session.query(Word.id).order_by(asc(Word.id)).first()
            if not first_word:
                 return jsonify({"error": "Failed to select any word."}), 500
            random_word_id = first_word.id

        # Fetch comprehensive data using the ORM helper
        word_obj = _get_word_details_orm(random_word_id)

        if not word_obj:
             # This should be unlikely if ID was found
             logger.error("Found random word ID but failed to fetch its data via ORM.", word_id=random_word_id)
             return jsonify({"error": "Could not fetch data for the selected random word."}), 404

        # Serialize using the full WordSchema
        schema = WordSchema()
        result = schema.dump(word_obj)

        # Etymology components handled by schema

        # Add completeness and summary (as in comprehensive endpoint)
        # result["data_completeness"] = calculate_data_completeness_orm(word_obj) # Needs refactoring
        relation_types = defaultdict(int)
        for rel in result.get("outgoing_relations", []):
            relation_types[rel["relation_type"]] += 1
        for rel in result.get("incoming_relations", []):
             relation_types[rel["relation_type"]] += 1
        result["relation_summary"] = dict(relation_types)


        # Return the response
        return jsonify(result), 200

    except ValidationError as err:
        logger.error(f"Schema validation error getting random word: {err.messages}")
        return jsonify({"error": "Data serialization error", "details": err.messages}), 500
    except SQLAlchemyError as db_err:
        db.session.rollback()
        logger.error(f"Database error fetching random word (ORM): {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "details": str(db_err)}), 500
    except Exception as e:
        db.session.rollback() # Rollback on any general error too
        logger.error(f"Error fetching random word (ORM): {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500