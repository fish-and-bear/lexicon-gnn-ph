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
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic']))
    value = fields.Str(required=True)
    tags = fields.Dict()
    pronunciation_metadata = fields.Dict()
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
    """Get word details."""
    try:
        # Try direct SQL approach first
        normalized_word = normalize_lemma(word)
        
        # Direct database query
        word_result = db.session.execute(text(
            "SELECT id, lemma, normalized_lemma, language_code, has_baybayin, baybayin_form, romanized_form "
            "FROM words WHERE normalized_lemma = :normalized"
        ), {"normalized": normalized_word}).fetchone()
        
        if not word_result:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404
        
        # Get definitions
        word_id = word_result.id
        definitions = db.session.execute(text(
            "SELECT id, definition_text, original_pos, examples "
            "FROM definitions WHERE word_id = :word_id"
        ), {"word_id": word_id}).fetchall()
        
        # Build the response manually
        response = {
            "id": word_id,
            "lemma": word_result.lemma,
            "normalized_lemma": word_result.normalized_lemma,
            "language_code": word_result.language_code,
            "has_baybayin": word_result.has_baybayin,
            "baybayin_form": word_result.baybayin_form,
            "romanized_form": word_result.romanized_form,
            "definitions": [
                {
                    "id": d.id,
                    "definition_text": d.definition_text,
                    "part_of_speech": d.original_pos,
                    "examples": d.examples
                }
                for d in definitions
            ]
        }
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing word request: {str(e)}", word=word)
        return jsonify({"error": "Internal server error"}), 500

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
            SELECT r.id, r.type, r.metadata, 
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
            SELECT r.id, r.type, r.metadata, 
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
                "relation_type": rel.type,
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
                "relation_type": rel.type,
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
        root_affixations = db.session.execute(text("""
            SELECT a.id, a.affix_type, a.position, 
                   w.id as affixed_id, w.lemma as affixed_lemma, 
                   w.language_code as affixed_language_code,
                   w.has_baybayin as affixed_has_baybayin,
                   w.baybayin_form as affixed_baybayin_form
            FROM affixations a
            JOIN words w ON a.affixed_word_id = w.id
            WHERE a.root_word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        # Get affixations where this word is the affixed form
        affixed_affixations = db.session.execute(text("""
            SELECT a.id, a.affix_type, a.position, 
                   w.id as root_id, w.lemma as root_lemma, 
                   w.language_code as root_language_code,
                   w.has_baybayin as root_has_baybayin,
                   w.baybayin_form as root_baybayin_form
            FROM affixations a
            JOIN words w ON a.root_word_id = w.id
            WHERE a.affixed_word_id = :word_id
        """), {"word_id": word_id}).fetchall()
        
        # Format the results
        as_root = []
        for aff in root_affixations:
            as_root.append({
                "id": aff.id,
                "affix_type": aff.affix_type,
                "position": aff.position,
                "affixed_word": {
                    "id": aff.affixed_id,
                    "lemma": aff.affixed_lemma,
                    "language_code": aff.affixed_language_code,
                    "has_baybayin": aff.affixed_has_baybayin,
                    "baybayin_form": aff.affixed_baybayin_form
                }
            })
        
        as_affixed = []
        for aff in affixed_affixations:
            as_affixed.append({
                "id": aff.id,
                "affix_type": aff.affix_type,
                "position": aff.position,
                "root_word": {
                    "id": aff.root_id,
                    "lemma": aff.root_lemma,
                    "language_code": aff.root_language_code,
                    "has_baybayin": aff.root_has_baybayin,
                    "baybayin_form": aff.root_baybayin_form
                }
            })
        
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
            SELECT id, type, value, tags, pronunciation_metadata
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
                "pronunciation_metadata": pron.pronunciation_metadata
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
        
        # Get all relations for this word
        relations = db.session.execute(text("""
            SELECT r.id, r.type, r.bidirectional,
                   sw.id as source_id, sw.lemma as source_lemma,
                   tw.id as target_id, tw.lemma as target_lemma
            FROM relations r
            JOIN words sw ON r.from_word_id = sw.id
            JOIN words tw ON r.to_word_id = tw.id
            WHERE r.from_word_id = :word_id OR r.to_word_id = :word_id
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
                'type': rel.type,
                'bidirectional': rel.bidirectional
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
            """SELECT a.affixed_word_id, a.affix_type, a.affix_value, a.position
               FROM affixations a
               WHERE a.root_word_id = :word_id"""
        ), {"word_id": word_id}).fetchall()
    
    def get_root_affixation(word_id):
        """Get root affixation using direct SQL."""
        return session.execute(text(
            """SELECT a.root_word_id, a.affix_type, a.affix_value, a.position
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
                            "type": aff.affix_type,
                            "value": aff.affix_value,
                            "position": aff.position
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
                            "type": root_affixation.affix_type,
                            "value": root_affixation.affix_value,
                            "position": root_affixation.position
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
            WHERE similarity(lemma, :word) > 0.3
            ORDER BY similarity(lemma, :word) DESC
            LIMIT 5
        """), {"word": word}).fetchall()
        
        # If no suggestions, try with normalized_lemma
        if not suggestions:
            suggestions = db.session.execute(text("""
                SELECT id, lemma, language_code, has_baybayin, baybayin_form
                FROM words
                WHERE similarity(normalized_lemma, :word) > 0.3
                ORDER BY similarity(normalized_lemma, :word) DESC
                LIMIT 5
            """), {"word": normalize_lemma(word)}).fetchall()
        
        # Create formatted list of suggestions
        result = []
        for w in suggestions:
            similarity = db.session.execute(text("""
                SELECT similarity(:word, :lemma) as score
            """), {"word": word, "lemma": w.lemma}).scalar() or 0.0
            
            result.append({
                "id": w.id,
                "lemma": w.lemma,
                "language_code": w.language_code,
                "similarity": similarity,
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
@cached_query(timeout=3600)
def get_etymology_tree(word_id: int):
    """Get the complete etymology tree for a word."""
    try:
        # Get word using direct SQL
        word = db.session.execute(text(
            "SELECT id, lemma, language_code FROM words WHERE id = :word_id"
        ), {"word_id": word_id}).fetchone()
        
        if not word:
            return jsonify({"error": "Word not found"}), 404
        
        session = db.session
        
        def get_word_by_lemma(lemma, language=None):
            """Find word by lemma and language."""
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
        
        def get_etymologies(word_id):
            """Get etymologies for a word."""
            return session.execute(text(
                """SELECT id, etymology_text, language_codes, confidence_score
                   FROM etymologies
                   WHERE word_id = :word_id"""
            ), {"word_id": word_id}).fetchall()
        
        def build_etymology_tree(word, depth=0, max_depth=5, visited=None):
            if visited is None:
                visited = set()
            if depth > max_depth or word.id in visited:
                return None
                
            visited.add(word.id)
            
            tree = {
                "word": word.lemma,
                "language": word.language_code,
                "etymologies": []
            }
            
            for etymology in get_etymologies(word.id):
                etym_data = {
                    "text": etymology.etymology_text,
                    "languages": etymology.language_codes.split(',') if etymology.language_codes else [],
                    "confidence": etymology.confidence_score
                }
                
                # Extract components
                try:
                    components = extract_etymology_components(etymology.etymology_text)
                    etym_data["components"] = components
                    
                    # Find related words
                    for comp in components:
                        if comp.get('text') and comp.get('language'):
                            related = get_word_by_lemma(comp['text'], comp['language'])
                            if related and related.id not in visited:
                                subtree = build_etymology_tree(
                                    related, depth + 1, max_depth, visited
                                )
                                if subtree:
                                    etym_data["derived_from"] = subtree
                except Exception as e:
                    etym_data["components"] = []
                    logger.error(f"Error extracting etymology components: {str(e)}")
                                
                tree["etymologies"].append(etym_data)
                
            return tree
            
        etymology_tree = build_etymology_tree(word)
        
        return jsonify({
            "word": word.lemma,
            "etymology_tree": etymology_tree
        }), 200
        
    except Exception as e:
        API_ERRORS.labels(endpoint='/words/etymology/tree', error_type='processing').inc()
        logger.error("Error processing etymology tree request",
                    word_id=word_id,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<int:word_id>/relations/graph", methods=["GET"])
@cached_query(timeout=3600)
def get_relations_graph(word_id: int):
    """Get a graph representation of word relations."""
    try:
        # Get word using direct SQL
        word = db.session.execute(text(
            "SELECT id, lemma, language_code FROM words WHERE id = :word_id"
        ), {"word_id": word_id}).fetchone()
        
        if not word:
            return jsonify({"error": "Word not found"}), 404
        
        session = db.session
        max_depth = min(int(request.args.get('max_depth', 2)), 3)  # Limit max depth to 3
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
            if depth > max_depth or node_id in visited:
                return
                
            visited.add(node_id)
            
            # Get outgoing relations
            outgoing = session.execute(text("""
                SELECT r.id, r.type, r.bidirectional, 
                       w.id as target_id, w.lemma as target_lemma, w.language_code as target_language
                FROM relations r
                JOIN words w ON r.to_word_id = w.id
                WHERE r.from_word_id = :node_id
            """), {"node_id": node_id}).fetchall()
            
            # Add relations to graph
            for rel in outgoing:
                add_node(rel.target_id, rel.target_lemma, rel.target_language)
                
                edges.append({
                    "source": node_id,
                    "target": rel.target_id,
                    "type": rel.type,
                    "bidirectional": rel.bidirectional
                })
                
                # Recurse if not at max depth
                if depth < max_depth:
                    get_relations(rel.target_id, depth + 1)
            
            # Get incoming relations if needed
            if include_bidirectional:
                incoming = session.execute(text("""
                    SELECT r.id, r.type, r.bidirectional, 
                           w.id as source_id, w.lemma as source_lemma, w.language_code as source_language
                    FROM relations r
                    JOIN words w ON r.from_word_id = w.id
                    WHERE r.to_word_id = :node_id
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
                            "type": rel.type,
                            "bidirectional": rel.bidirectional
                        })
                    
                    # Recurse if not at max depth
                    if depth < max_depth:
                        get_relations(rel.source_id, depth + 1)
        
        # Add starting node
        add_node(word.id, word.lemma, word.language_code)
        
        # Build graph
        get_relations(word.id)
        
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
            
            # Get some basic statistics
            if db_status:
                word_count = db.session.execute(text("SELECT COUNT(*) FROM words")).scalar()
                language_count = db.session.execute(text(
                    "SELECT COUNT(DISTINCT language_code) FROM words"
                )).scalar()
                
                db_info["word_count"] = word_count
                db_info["language_count"] = language_count
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