"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast
from sqlalchemy.orm import joinedload, contains_eager, selectinload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, Language
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

class CreditSchema(BaseSchema):
    """Schema for word credits."""
    credit = fields.Str(required=True)
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code'))

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
    """Get a word entry by lemma or ID."""
    try:
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
        
        # Try to parse as integer (ID)
        try:
            word_id = int(word)
            word_entry = Word.query.get_or_404(word_id)
        except ValueError:
            # Look up by lemma
            word_entry = Word.query.filter(
                func.lower(Word.lemma) == func.lower(word)
            ).first_or_404()
            word_id = word_entry.id
        
        result = {
            "id": word_entry.id,
            "lemma": word_entry.lemma,
            "normalized_lemma": word_entry.normalized_lemma,
            "language_code": word_entry.language_code,
            "has_baybayin": word_entry.has_baybayin,
            "baybayin_form": word_entry.baybayin_form,
            "romanized_form": word_entry.romanized_form,
            "root_word_id": word_entry.root_word_id,
            "preferred_spelling": word_entry.preferred_spelling,
            "tags": word_entry.tags,
            "idioms": word_entry.idioms,
            "source_info": word_entry.source_info,
            "word_metadata": word_entry.word_metadata,  # Added word_metadata field
            "data_hash": word_entry.data_hash,
            "search_text": word_entry.search_text,
            "badlit_form": word_entry.badlit_form,
            "hyphenation": word_entry.hyphenation,
            "is_proper_noun": word_entry.is_proper_noun,
            "is_abbreviation": word_entry.is_abbreviation,
            "is_initialism": word_entry.is_initialism,
            "is_root": word_entry.is_root,
            "created_at": word_entry.created_at.isoformat() if word_entry.created_at else None,
            "updated_at": word_entry.updated_at.isoformat() if word_entry.updated_at else None,
        }
        
        # Include definitions if requested
        if include_definitions:
            # Get definitions with standardized part of speech
            definitions = db.session.execute(text("""
                SELECT d.id, d.definition_text, d.original_pos, d.standardized_pos_id, 
                       d.examples, d.usage_notes, d.tags, d.sources,
                       pos.code, pos.name_en, pos.name_tl
                FROM definitions d
                LEFT JOIN parts_of_speech pos ON d.standardized_pos_id = pos.id
                WHERE d.word_id = :word_id
                ORDER BY d.standardized_pos_id
            """), {"word_id": word_id}).fetchall()
            
            result["definitions"] = []
            for def_row in definitions:
                pos_info = None
                if def_row.standardized_pos_id:
                    pos_info = {
                        "id": def_row.standardized_pos_id,
                        "code": def_row.code,
                        "name_en": def_row.name_en,
                        "name_tl": def_row.name_tl
                    }
                
                result["definitions"].append({
                    "id": def_row.id,
                    "definition_text": def_row.definition_text,
                    "original_pos": def_row.original_pos,
                    "standardized_pos_id": def_row.standardized_pos_id,
                    "standardized_pos": pos_info,
                    "examples": def_row.examples,
                    "usage_notes": def_row.usage_notes,
                    "tags": def_row.tags,
                    "sources": def_row.sources
                })
        
        # Include relationships if requested
        if include_relations:
            # Get outgoing relations
            outgoing = db.session.execute(text("""
                SELECT r.id, r.relation_type, r.to_word_id, r.sources, r.metadata,
                       w.lemma, w.language_code, w.has_baybayin, w.baybayin_form
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            # Get incoming relations
            incoming = db.session.execute(text("""
                SELECT r.id, r.relation_type, r.from_word_id, r.sources, r.metadata,
                       w.lemma, w.language_code, w.has_baybayin, w.baybayin_form
                FROM relations r
                JOIN words w ON r.from_word_id = w.id
                WHERE r.to_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            result["outgoing_relations"] = [{
                "id": rel.id,
                "relation_type": rel.relation_type,
                "target_word": {
                    "id": rel.to_word_id,
                    "lemma": rel.lemma,
                    "language_code": rel.language_code,
                    "has_baybayin": rel.has_baybayin,
                    "baybayin_form": rel.baybayin_form
                },
                "metadata": rel.metadata,
                "sources": rel.sources
            } for rel in outgoing]
            
            result["incoming_relations"] = [{
                "id": rel.id,
                "relation_type": rel.relation_type,
                "source_word": {
                    "id": rel.from_word_id,
                    "lemma": rel.lemma,
                    "language_code": rel.language_code,
                    "has_baybayin": rel.has_baybayin,
                    "baybayin_form": rel.baybayin_form
                },
                "metadata": rel.metadata,
                "sources": rel.sources
            } for rel in incoming]
        
        # Include affixations if requested
        if include_affixations:
            # Get root affixations
            root_affixes = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.affixed_word_id, a.sources,
                       w.lemma, w.language_code, w.has_baybayin, w.baybayin_form
                FROM affixations a
                JOIN words w ON a.affixed_word_id = w.id
                WHERE a.root_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            # Get affixed affixations
            affixed = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.root_word_id, a.sources,
                       w.lemma, w.language_code, w.has_baybayin, w.baybayin_form
                FROM affixations a
                JOIN words w ON a.root_word_id = w.id
                WHERE a.affixed_word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            result["root_affixations"] = [{
                "id": aff.id,
                "affix_type": aff.affix_type,
                "affixed_word": {
                    "id": aff.affixed_word_id,
                    "lemma": aff.lemma,
                    "language_code": aff.language_code,
                    "has_baybayin": aff.has_baybayin,
                    "baybayin_form": aff.baybayin_form
                },
                "sources": aff.sources
            } for aff in root_affixes]
            
            result["affixed_affixations"] = [{
                "id": aff.id,
                "affix_type": aff.affix_type,
                "root_word": {
                    "id": aff.root_word_id,
                    "lemma": aff.lemma,
                    "language_code": aff.language_code,
                    "has_baybayin": aff.has_baybayin,
                    "baybayin_form": aff.baybayin_form
                },
                "sources": aff.sources
            } for aff in affixed]
        
        # Include etymologies if requested
        if include_etymologies:
            etymologies = db.session.execute(text("""
                SELECT id, etymology_text, normalized_components, etymology_structure, 
                       language_codes, sources
                FROM etymologies
                WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            result["etymologies"] = [{
                "id": etym.id,
                "etymology_text": etym.etymology_text,
                "normalized_components": etym.normalized_components,
                "etymology_structure": etym.etymology_structure,
                "language_codes": etym.language_codes,
                "sources": etym.sources
            } for etym in etymologies]
        
        # Include pronunciations if requested
        if include_pronunciations:
            # Get pronunciations
            pronunciations = db.session.execute(text("""
                SELECT id, type, value, tags, metadata
                FROM pronunciations
                WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            result["pronunciations"] = []
            for pron in pronunciations:
                result["pronunciations"].append({
                    "id": pron.id,
                    "type": pron.type,
                    "value": pron.value,
                    "tags": pron.tags,
                    "metadata": pron.metadata
                })
        
        # Include credits if requested
        if include_credits:
            credits = db.session.execute(text("""
                SELECT id, credit, sources
                FROM credits
                WHERE word_id = :word_id
            """), {"word_id": word_id}).fetchall()
            
            result["credits"] = [{
                "id": credit.id,
                "credit": credit.credit,
                "sources": credit.sources
            } for credit in credits]
        
        # Include root word if requested
        if include_root and word_entry.root_word_id:
            root = Word.query.get(word_entry.root_word_id)
            if root:
                result["root_word"] = {
                    "id": root.id,
                    "lemma": root.lemma,
                    "language_code": root.language_code
                }
        
        # Include derived words if requested
        if include_derived:
            derived = Word.query.filter_by(root_word_id=word_id).all()
            result["derived_words"] = [{
                "id": d.id,
                "lemma": d.lemma,
                "language_code": d.language_code
            } for d in derived]
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error retrieving word '{word}': {str(e)}")
        return jsonify({"error": str(e)}), 500

@bp.route("/search", methods=["GET"])
def search_words():
    """Search for words with various matching modes."""
    try:
        # Parse query parameters
        query = request.args.get('q', '')
        language = request.args.get('language')
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        if not query:
            return jsonify({
                'error': 'Missing query parameter',
                'message': 'Please provide a search query with the q parameter'
            }), 400
            
        # Build SQL query
        sql = """
            SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form, romanized_form
            FROM words
            WHERE (lemma ILIKE :pattern OR normalized_lemma ILIKE :pattern)
        """
        
        params = {"pattern": f'%{query.lower()}%'}
        
        if language:
            sql += " AND language_code = :language"
            params["language"] = language
            
        sql += " ORDER BY lemma ASC LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset
        
        # Execute the query
        results = db.session.execute(text(sql), params).fetchall()
        
        # Format the results
        word_list = []
        for row in results:
            word_id = row.id
            
            # Get definitions for this word
            definitions = db.session.execute(text(
                "SELECT id, definition_text, original_pos "
                "FROM definitions WHERE word_id = :word_id LIMIT 3"
            ), {"word_id": word_id}).fetchall()
            
            word_data = {
                "id": row.id,
                "lemma": row.lemma,
                "normalized_lemma": row.normalized_lemma,
                "language_code": row.language_code,
                "has_baybayin": row.has_baybayin,
                "baybayin_form": row.baybayin_form,
                "romanized_form": row.romanized_form,
                "definitions": [
                    {
                        "id": d.id,
                        "definition_text": d.definition_text,
                        "part_of_speech": d.original_pos
                    }
                    for d in definitions
                ]
            }
            word_list.append(word_data)
        
        return jsonify({
            'total': len(word_list),
            'words': word_list,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

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
        
        # Format the results
        outgoing = []
        for rel in outgoing_relations:
            outgoing.append({
                "id": rel.id,
                "relation_type": rel.relation_type,
                "metadata": rel.metadata,
                "target_word": {
                    "id": rel.target_id,
                    "lemma": rel.target_lemma,
                    "language_code": rel.target_language_code,
                    "has_baybayin": rel.target_has_baybayin,
                    "baybayin_form": rel.target_baybayin_form
                }
            })
        
        incoming = []
        for rel in incoming_relations:
            incoming.append({
                "id": rel.id,
                "relation_type": rel.relation_type,
                "metadata": rel.metadata,
                "source_word": {
                    "id": rel.source_id,
                    "lemma": rel.source_lemma,
                    "language_code": rel.source_language_code,
                    "has_baybayin": rel.source_has_baybayin,
                    "baybayin_form": rel.source_baybayin_form
                }
            })
        
        return jsonify({
            'outgoing_relations': outgoing,
            'incoming_relations': incoming
        }), 200
        
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
                        "id": aff.affixed_id,
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
                        "id": aff.root_id,
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
        
        # Get pronunciations
        pronunciations = db.session.execute(text("""
            SELECT id, type, value, tags, metadata
            FROM pronunciations
            WHERE word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        # Format the results
        pronunciation_list = []
        for pron in pronunciations:
            pronunciation_list.append({
                "id": pron.id,
                "type": pron.type,
                "value": pron.value,
                "tags": pron.tags,
                "metadata": pron.metadata
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
        # Set a timeout for these queries
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


        # 3. Fetch comprehensive data for the selected word ID
        # Re-use the direct SQL logic from get_word_comprehensive for efficiency

        word_result = db.session.execute(text("""
            SELECT id, lemma, normalized_lemma, language_code,
                   COALESCE(has_baybayin, FALSE) as has_baybayin,
                   baybayin_form, romanized_form,
                   root_word_id, preferred_spelling, tags,
                   data_hash, search_text,
                   created_at, updated_at
            FROM words WHERE id = :word_id
        """), {"word_id": random_word_id}).fetchone()

        if not word_result:
             # This should theoretically not happen if we found an ID above
             logger.error("Found random word ID but failed to fetch its data.", word_id=random_word_id)
             return jsonify({"error": "Could not fetch data for the selected random word."}), 404

        # Build comprehensive response (copied & adapted from get_word_comprehensive)
        response = {
            # Basic word info
            "id": word_result.id,
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

        # Fetch related data (Definitions, Etymologies, Pronunciations, Credits, Relations, Affixations)
        # using separate queries for clarity and potential performance isolation.
        # NOTE: This is duplicated logic from get_word_comprehensive. Consider refactoring
        # into a helper function if maintainability becomes an issue.

        # Definitions
        try:
            definitions = db.session.execute(text("""
                SELECT id, definition_text, COALESCE(original_pos, '') as original_pos,
                       standardized_pos_id, COALESCE(examples, '') as examples,
                       COALESCE(usage_notes, '') as usage_notes, COALESCE(tags, '') as tags,
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM definitions WHERE word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()

            response["definitions"] = []
            for d in definitions:
                def_data = {
                    "id": d.id, "definition_text": d.definition_text, "original_pos": d.original_pos,
                    "standardized_pos_id": d.standardized_pos_id, "examples": d.examples,
                    "usage_notes": d.usage_notes, "tags": d.tags, "sources": d.sources,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                    "standardized_pos": None # Placeholder, fetch POS details below
                }
                # Fetch PartOfSpeech details if ID exists
                if d.standardized_pos_id:
                     try:
                         pos_info = db.session.execute(text("""
                             SELECT id, code, name_en, name_tl, description
                             FROM parts_of_speech WHERE id = :pos_id
                         """), {"pos_id": d.standardized_pos_id}).fetchone()
                         if pos_info:
                             def_data["standardized_pos"] = {k: getattr(pos_info, k) for k in pos_info.keys()}
                     except Exception as pos_e:
                         logger.warning(f"Error fetching POS details for definition {d.id}: {pos_e}")

                response["definitions"].append(def_data)
        except Exception as e:
            logger.error(f"Error fetching definitions for random word {random_word_id}: {e}")
            response["definitions"] = []

        # Etymologies
        try:
            etymologies = db.session.execute(text("""
                SELECT id, etymology_text, normalized_components, etymology_structure, language_codes,
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM etymologies WHERE word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["etymologies"] = [{k: getattr(etym, k) for k in etym.keys()} for etym in etymologies]
             # Add extracted components (optional, consider performance impact)
            for etym_data in response["etymologies"]:
                try:
                    components = extract_etymology_components(etym_data.get("etymology_text", ""))
                    if components:
                        if isinstance(components, dict) and 'original_text' in components:
                            etym_data["components"] = [components]
                        else:
                            etym_data["components"] = components
                except Exception as comp_e:
                     logger.warning(f"Error extracting components for etymology {etym_data.get('id')}: {comp_e}")
                     etym_data["components"] = []
                # Convert datetime objects
                etym_data["created_at"] = etym_data["created_at"].isoformat() if etym_data.get("created_at") else None
                etym_data["updated_at"] = etym_data["updated_at"].isoformat() if etym_data.get("updated_at") else None

        except Exception as e:
            logger.error(f"Error fetching etymologies for random word {random_word_id}: {e}")
            response["etymologies"] = []

        # Pronunciations
        try:
            pronunciations = db.session.execute(text("""
                SELECT id, type, value, COALESCE(tags, '{}') as tags,
                       created_at, updated_at, COALESCE(sources, '') as sources
                FROM pronunciations WHERE word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["pronunciations"] = []
            for pron in pronunciations:
                 pron_data = {k: getattr(pron, k) for k in pron.keys()}
                 pron_data["created_at"] = pron_data["created_at"].isoformat() if pron_data.get("created_at") else None
                 pron_data["updated_at"] = pron_data["updated_at"].isoformat() if pron_data.get("updated_at") else None
                 response["pronunciations"].append(pron_data)
        except Exception as e:
            logger.error(f"Error fetching pronunciations for random word {random_word_id}: {e}")
            response["pronunciations"] = []

        # Credits
        try:
            credits = db.session.execute(text("""
                SELECT id, credit, created_at, updated_at
                FROM credits WHERE word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["credits"] = []
            for cred in credits:
                 cred_data = {k: getattr(cred, k) for k in cred.keys()}
                 cred_data["created_at"] = cred_data["created_at"].isoformat() if cred_data.get("created_at") else None
                 cred_data["updated_at"] = cred_data["updated_at"].isoformat() if cred_data.get("updated_at") else None
                 response["credits"].append(cred_data)
        except Exception as e:
            logger.error(f"Error fetching credits for random word {random_word_id}: {e}")
            response["credits"] = []

        # Root Word
        response["root_word"] = None
        if word_result.root_word_id:
            try:
                root_word_res = db.session.execute(text("""
                    SELECT id, lemma, normalized_lemma, language_code,
                           COALESCE(has_baybayin, FALSE) as has_baybayin, baybayin_form
                    FROM words WHERE id = :root_id
                """), {"root_id": word_result.root_word_id}).fetchone()
                if root_word_res:
                    response["root_word"] = {k: getattr(root_word_res, k) for k in root_word_res.keys()}
                    response["root_word"]["has_baybayin"] = bool(response["root_word"]["has_baybayin"]) # Ensure boolean
            except Exception as e:
                logger.error(f"Error fetching root word {word_result.root_word_id} for random word {random_word_id}: {e}")

        # Derived Words
        try:
            derived_words_res = db.session.execute(text("""
                SELECT id, lemma, normalized_lemma, language_code,
                       COALESCE(has_baybayin, FALSE) as has_baybayin, baybayin_form
                FROM words WHERE root_word_id = :word_id
                LIMIT 100
            """), {"word_id": random_word_id}).fetchall()
            response["derived_words"] = []
            for derived in derived_words_res:
                derived_data = {k: getattr(derived, k) for k in derived.keys()}
                derived_data["has_baybayin"] = bool(derived_data["has_baybayin"]) # Ensure boolean
                response["derived_words"].append(derived_data)
        except Exception as e:
            logger.error(f"Error fetching derived words for random word {random_word_id}: {e}")
            response["derived_words"] = []

        # Outgoing Relations
        try:
            outgoing_relations = db.session.execute(text("""
                SELECT r.id, r.relation_type, COALESCE(r.metadata, '{}') as metadata, COALESCE(r.sources, '') as sources,
                       w.id as target_id, w.lemma as target_lemma,
                       w.language_code as target_language_code,
                       COALESCE(w.has_baybayin, FALSE) as target_has_baybayin,
                       w.baybayin_form as target_baybayin_form
                FROM relations r JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["outgoing_relations"] = []
            for rel in outgoing_relations:
                 response["outgoing_relations"].append({
                     "id": rel.id, "relation_type": rel.relation_type, "metadata": rel.metadata, "sources": rel.sources,
                     "target_word": {
                         "id": rel.target_id, "lemma": rel.target_lemma, "language_code": rel.target_language_code,
                         "has_baybayin": bool(rel.target_has_baybayin), "baybayin_form": rel.target_baybayin_form
                     }
                 })
        except Exception as e:
            logger.error(f"Error fetching outgoing relations for random word {random_word_id}: {e}")
            response["outgoing_relations"] = []

        # Incoming Relations
        try:
            incoming_relations = db.session.execute(text("""
                SELECT r.id, r.relation_type, COALESCE(r.metadata, '{}') as metadata, COALESCE(r.sources, '') as sources,
                       w.id as source_id, w.lemma as source_lemma,
                       w.language_code as source_language_code,
                       COALESCE(w.has_baybayin, FALSE) as source_has_baybayin,
                       w.baybayin_form as source_baybayin_form
                FROM relations r JOIN words w ON r.from_word_id = w.id
                WHERE r.to_word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["incoming_relations"] = []
            for rel in incoming_relations:
                 response["incoming_relations"].append({
                     "id": rel.id, "relation_type": rel.relation_type, "metadata": rel.metadata, "sources": rel.sources,
                     "source_word": {
                         "id": rel.source_id, "lemma": rel.source_lemma, "language_code": rel.source_language_code,
                         "has_baybayin": bool(rel.source_has_baybayin), "baybayin_form": rel.source_baybayin_form
                     }
                 })
        except Exception as e:
            logger.error(f"Error fetching incoming relations for random word {random_word_id}: {e}")
            response["incoming_relations"] = []

        # Root Affixations
        try:
            root_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.created_at, a.updated_at, COALESCE(a.sources, '') as sources,
                       w.id as affixed_id, w.lemma as affixed_lemma,
                       w.language_code as affixed_language_code,
                       COALESCE(w.has_baybayin, FALSE) as affixed_has_baybayin,
                       w.baybayin_form as affixed_baybayin_form
                FROM affixations a JOIN words w ON a.affixed_word_id = w.id
                WHERE a.root_word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["root_affixations"] = []
            for aff in root_affixations:
                 aff_data = {
                     "id": aff.id, "affix_type": aff.affix_type, "sources": aff.sources,
                     "created_at": aff.created_at.isoformat() if aff.created_at else None,
                     "updated_at": aff.updated_at.isoformat() if aff.updated_at else None,
                     "affixed_word": {
                         "id": aff.affixed_id, "lemma": aff.affixed_lemma, "language_code": aff.affixed_language_code,
                         "has_baybayin": bool(aff.affixed_has_baybayin), "baybayin_form": aff.affixed_baybayin_form
                     }
                 }
                 response["root_affixations"].append(aff_data)
        except Exception as e:
            logger.error(f"Error fetching root affixations for random word {random_word_id}: {e}")
            response["root_affixations"] = []

        # Affixed Affixations
        try:
            affixed_affixations = db.session.execute(text("""
                SELECT a.id, a.affix_type, a.created_at, a.updated_at, COALESCE(a.sources, '') as sources,
                       w.id as root_id, w.lemma as root_lemma,
                       w.language_code as root_language_code,
                       COALESCE(w.has_baybayin, FALSE) as root_has_baybayin,
                       w.baybayin_form as root_baybayin_form
                FROM affixations a JOIN words w ON a.root_word_id = w.id
                WHERE a.affixed_word_id = :word_id
            """), {"word_id": random_word_id}).fetchall()
            response["affixed_affixations"] = []
            for aff in affixed_affixations:
                 aff_data = {
                     "id": aff.id, "affix_type": aff.affix_type, "sources": aff.sources,
                     "created_at": aff.created_at.isoformat() if aff.created_at else None,
                     "updated_at": aff.updated_at.isoformat() if aff.updated_at else None,
                     "root_word": {
                         "id": aff.root_id, "lemma": aff.root_lemma, "language_code": aff.root_language_code,
                         "has_baybayin": bool(aff.root_has_baybayin), "baybayin_form": aff.root_baybayin_form
                     }
                 }
                 response["affixed_affixations"].append(aff_data)
        except Exception as e:
            logger.error(f"Error fetching affixed affixations for random word {random_word_id}: {e}")
            response["affixed_affixations"] = []

        # Add data completeness and summary (optional, can be simplified/removed if slow)
        response["data_completeness"] = {
            "has_definitions": bool(response.get("definitions", [])),
            "has_etymology": bool(response.get("etymologies", [])),
            "has_pronunciations": bool(response.get("pronunciations", [])),
            "has_baybayin": bool(response.get("has_baybayin") and response.get("baybayin_form")),
            "has_relations": bool(response.get("outgoing_relations", []) or response.get("incoming_relations", [])),
            "has_affixations": bool(response.get("root_affixations", []) or response.get("affixed_affixations", [])),
        }
        relation_types = {}
        for rel in response.get("outgoing_relations", []):
            relation_types[rel["relation_type"]] = relation_types.get(rel["relation_type"], 0) + 1
        for rel in response.get("incoming_relations", []):
            relation_types[rel["relation_type"]] = relation_types.get(rel["relation_type"], 0) + 1
        response["relation_summary"] = relation_types

        # 4. Return the response
        return jsonify(response), 200

    except SQLAlchemyError as db_err:
        db.session.rollback()
        logger.error(f"Database error fetching random word: {db_err}", exc_info=True)
        return jsonify({"error": "Database error", "details": str(db_err)}), 500
    except Exception as e:
        db.session.rollback() # Rollback on any general error too
        logger.error(f"Error fetching random word: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500