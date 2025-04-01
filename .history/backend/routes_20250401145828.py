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
    RelationshipType, RelationshipCategory, BaybayinRomanizer,
    normalize_lemma, extract_etymology_components, extract_language_codes
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
        # Get word entry with all relationships loaded
        word_entry = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos),
            joinedload(Word.etymologies),
            joinedload(Word.pronunciations),
            joinedload(Word.credits),
            joinedload(Word.root_word),
            joinedload(Word.derived_words),
            joinedload(Word.outgoing_relations),
            joinedload(Word.incoming_relations),
            joinedload(Word.root_affixations),
            joinedload(Word.affixed_affixations)
        ).filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first()

        if not word_entry:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404

        # Convert to dictionary with all details
        schema = WordSchema()
        result = schema.dump(word_entry)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing word request: {str(e)}", word=word)
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/search", methods=["GET"])
def search_words():
    """Search for words with various matching modes."""
    try:
        # Parse and validate query parameters
        schema = SearchQuerySchema()
        params = schema.load(request.args)
        
        # Normalize search text
        search_text = params['q'].strip().lower()
        
        # Base query with eager loading of relationships
        query = Word.query.options(
            selectinload(Word.definitions).selectinload(Definition.standardized_pos),
            selectinload(Word.definitions).selectinload(Definition.definition_relations),
            selectinload(Word.related_definitions),
            selectinload(Word.outgoing_relations),
            selectinload(Word.incoming_relations)
        )
        
        # Apply text matching filter
        query = query.filter(
            or_(
                Word.normalized_lemma.ilike(f'%{search_text}%'),
                Word.lemma.ilike(f'%{search_text}%')
            )
        )
        
        # Apply additional filters
        if params.get('language'):
            query = query.filter(Word.language_code == params['language'])
        if params.get('pos'):
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == params['pos']
            ).distinct()
        
        # Apply sorting
        if params['sort'] == 'alphabetical':
            query = query.order_by(Word.lemma.asc() if params['order'] == 'asc' else Word.lemma.desc())
        elif params['sort'] == 'created':
            query = query.order_by(Word.created_at.asc() if params['order'] == 'asc' else Word.created_at.desc())
        elif params['sort'] == 'updated':
            query = query.order_by(Word.updated_at.asc() if params['order'] == 'asc' else Word.updated_at.desc())
        else:  # Default to alphabetical
            query = query.order_by(Word.lemma.asc())
        
        # Apply pagination
        query = query.offset(params['offset']).limit(params['limit'])
        
        # Execute query and get total count
        words = query.all()
        
        # Serialize results
        schema = WordSchema(many=True)
        result = schema.dump(words)
        
        return jsonify({
            'total': len(result),
            'words': result
        })
        
    except Exception as e:
        current_app.logger.error(f"Error processing search request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@bp.route("/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get word relations."""
    try:
        word_obj = Word.query.options(
            joinedload(Word.outgoing_relations).joinedload(Relation.target_word),
            joinedload(Word.incoming_relations).joinedload(Relation.source_word)
        ).filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first_or_404()
        
        # Process relations
        schema = RelationSchema(many=True)
        outgoing = schema.dump(word_obj.outgoing_relations)
        incoming = schema.dump(word_obj.incoming_relations)
        
        return jsonify({
            'outgoing_relations': outgoing,
            'incoming_relations': incoming
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word relations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get all affixations for a word."""
    try:
        word_obj = Word.query.options(
            joinedload(Word.root_affixations),
            joinedload(Word.affixed_affixations)
        ).filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first_or_404()
        
        schema = AffixationSchema(many=True)
        root_affixations = schema.dump(word_obj.root_affixations)
        affixed_affixations = schema.dump(word_obj.affixed_affixations)
        
        return jsonify({
            'as_root': root_affixations,
            'as_affixed': affixed_affixations
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word affixations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get word pronunciation."""
    try:
        word_obj = Word.query.options(
            joinedload(Word.pronunciations)
        ).filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first_or_404()
        
        schema = PronunciationType(many=True)
        pronunciations = schema.dump(word_obj.pronunciations)
        
        return jsonify({
            'pronunciations': pronunciations,
            'has_pronunciation': bool(pronunciations)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word pronunciation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics."""
    try:
        stats = {
            'total_words': Word.query.count(),
            'total_definitions': Definition.query.count(),
            'total_etymologies': Etymology.query.count(),
            'total_relations': Relation.query.count(),
            'total_affixations': Affixation.query.count(),
            'words_with_examples': db.session.query(
                func.count(distinct(Word.id))
            ).join(Word.definitions).filter(
                Definition.examples.isnot(None)
            ).scalar(),
            'words_with_etymology': db.session.query(
                func.count(distinct(Word.id))
            ).join(Word.etymologies).scalar(),
            'words_with_relations': db.session.query(
                func.count(distinct(Word.id))
            ).join(Word.outgoing_relations).scalar(),
            'words_with_baybayin': Word.query.filter(
                Word.has_baybayin == True
            ).count(),
            'words_by_language': dict(
                db.session.query(
                    Word.language_code,
                    func.count(Word.id)
                ).group_by(Word.language_code).all()
            ),
            'words_by_pos': dict(
                db.session.query(
                    PartOfSpeech.code,
                    func.count(distinct(Word.id))
                ).join(Definition).join(PartOfSpeech).group_by(PartOfSpeech.code).all()
            ),
            'verification_stats': {},  # Removed as we don't track verification status
            'quality_distribution': {},  # Removed as we don't track quality scores
            'update_frequency': generate_update_frequency_stats()
        }
        return jsonify(stats), 200
    except Exception as e:
        current_app.logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@bp.route("/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get word etymology."""
    try:
        word_obj = Word.query.options(
            joinedload(Word.etymologies)
        ).filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first_or_404()
        
        schema = EtymologySchema(many=True)
        etymologies = schema.dump(word_obj.etymologies)
        
        return jsonify({
            'etymologies': etymologies,
            'has_etymology': bool(etymologies)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word etymology: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/words/<path:word>/semantic_network", methods=["GET"])
def get_semantic_network(word: str):
    """Get semantic network for a word."""
    try:
        # Get word entry
        word_entry = Word.query.filter(
            Word.normalized_lemma == word.lower()
        ).first()

        if not word_entry:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404

        # Get all relations
        relations = (
            Relation.query
            .filter(
                or_(
                    Relation.source_word_id == word_entry.id,
                    Relation.target_word_id == word_entry.id
                )
            )
            .options(
                joinedload(Relation.source_word),
                joinedload(Relation.target_word)
            )
            .all()
        )

        # Build network
        nodes = set()
        edges = []
        for rel in relations:
            source = rel.source_word
            target = rel.target_word
            nodes.add(source)
            nodes.add(target)
            edges.append({
                'source': source.lemma,
                'target': target.lemma,
                'type': rel.type,
                'bidirectional': rel.bidirectional
            })

        return jsonify({
            'nodes': [{'id': node.lemma, 'label': node.lemma} for node in nodes],
            'edges': edges
        }), 200

    except Exception as e:
        logger.error("Error getting semantic network", error=str(e))
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

    # Get initial word
    word_entry = Word.query.filter(
        Word.normalized_lemma == normalize_lemma(word)
    ).first()

    if not word_entry:
        return jsonify({"error": "Word not found"}), 404

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
            "is_root": word.is_root
        }

        if include_definitions:
            node["definitions"] = [
                {
                    "text": d.definition_text,
                    "pos": d.standardized_pos
                }
                for d in word.definitions[:2]
            ]

        if include_baybayin and word.has_baybayin:
            node["baybayin_form"] = word.baybayin_form

        # Get derived words
        derived = (
            Affixation.query
            .filter(Affixation.root_word_id == word.id)
            .options(joinedload(Affixation.affixed_word))
            .all()
        )

        if derived:
            node["derived"] = []
            for aff in derived:
                child = build_tree(aff.affixed_word, depth + 1, processed)
                if child:
                    child["affixation"] = {
                        "type": aff.affix_type,
                        "value": aff.affix_value,
                        "position": aff.position
                    }
                    node["derived"].append(child)

        # Get root word if this is a derived word
        if not word.is_root:
            root_affixation = (
                Affixation.query
                .filter(Affixation.affixed_word_id == word.id)
                .options(joinedload(Affixation.root_word))
                .first()
            )
            
            if root_affixation and root_affixation.root_word_id not in processed:
                node["root"] = build_tree(root_affixation.root_word, depth + 1, processed)
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
            "is_root": word_entry.is_root,
            "has_derived_forms": bool(tree.get("derived")),
            "has_root_word": bool(tree.get("root"))
        }
    }

    return jsonify(result), 200

# Helper functions
def get_word_suggestions(word: str) -> List[Dict[str, Any]]:
    """Get word suggestions for a failed lookup."""
    # Try fuzzy matching
    matches = Word.query.filter(
        Word.search_text.match(word, postgresql_regconfig='simple')
    ).order_by(
        func.similarity(Word.lemma, word).desc()
    ).limit(5).all()

    return [
        {
            "id": w.id,
            "lemma": w.lemma,
            "language_code": w.language_code,
            "similarity": w.calculate_similarity_score(word),
            "has_baybayin": w.has_baybayin,
            "baybayin_form": w.baybayin_form if w.has_baybayin else None
        }
        for w in matches
    ]

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
        word = Word.query.get_or_404(word_id)
        
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
            
            for etymology in word.etymologies:
                etym_data = {
                    "text": etymology.etymology_text,
                    "languages": etymology.get_language_codes_list(),
                    "components": etymology.get_components_list(),
                    "confidence": etymology.confidence_score
                }
                
                # Find related words
                for comp in etymology.get_components_list():
                    if comp.get('text'):
                        related = Word.query.filter(
                            Word.normalized_lemma == normalize_lemma(comp['text']),
                            Word.language_code == comp.get('language', '')
                        ).first()
                        if related and related.id not in visited:
                            subtree = build_etymology_tree(
                                related, depth + 1, max_depth, visited
                            )
                            if subtree:
                                etym_data["derived_from"] = subtree
                                
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
@cached_query(timeout=1800)
def get_relation_graph(word_id: int):
    """Get a graph of word relationships."""
    try:
        depth = int(request.args.get('depth', 2))
        include_definitions = request.args.get('include_definitions', 'true').lower() == 'true'
        include_etymology = request.args.get('include_etymology', 'false').lower() == 'true'
        min_confidence = float(request.args.get('min_confidence', 0.0))
        
        word = Word.query.get_or_404(word_id)
        
        # Build graph
        nodes = {}
        edges = []
        
        def add_node(word):
            if word.id not in nodes:
                node_data = {
                    "id": word.id,
                    "label": word.lemma,
                    "language": word.language_code,
                    "has_baybayin": word.has_baybayin,
                    "baybayin_form": word.baybayin_form if word.has_baybayin else None
                }
                
                if include_definitions:
                    node_data["definitions"] = [
                        {
                            "text": d.definition_text,
                            "pos": d.standardized_pos.code if d.standardized_pos else None
                        }
                        for d in word.definitions[:2]
                    ]
                    
                if include_etymology and word.etymologies:
                    node_data["etymology"] = {
                        "text": word.etymologies[0].etymology_text,
                        "confidence": word.etymologies[0].confidence_score
                    }
                    
                nodes[word.id] = node_data
                
        def process_relations(word, current_depth=0):
            if current_depth >= depth:
                return
                
            add_node(word)
            
            # Process outgoing relations
            for relation in word.relations_from:
                if relation.confidence_score >= min_confidence:
                    add_node(relation.to_word)
                    edges.append({
                        "source": word.id,
                        "target": relation.to_word_id,
                        "type": relation.relation_type,
                        "bidirectional": relation.bidirectional,
                        "confidence": relation.confidence_score
                    })
                    process_relations(relation.to_word, current_depth + 1)
                    
            # Process incoming relations
            for relation in word.relations_to:
                if relation.confidence_score >= min_confidence:
                    add_node(relation.from_word)
                    if not relation.bidirectional:
                        edges.append({
                            "source": relation.from_word_id,
                            "target": word.id,
                            "type": relation.relation_type,
                            "bidirectional": relation.bidirectional,
                            "confidence": relation.confidence_score
                        })
                    process_relations(relation.from_word, current_depth + 1)
                    
        # Build the graph
        process_relations(word)
        
        return jsonify({
            "nodes": list(nodes.values()),
            "edges": edges,
            "metadata": {
                "root_word": word.lemma,
                "depth": depth,
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }), 200
        
    except Exception as e:
        API_ERRORS.labels(endpoint='/words/relations/graph', error_type='processing').inc()
        logger.error("Error processing relation graph request",
                    word_id=word_id,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route('/test', methods=['GET'])
def test_api():
    """Simple test endpoint to verify API is working."""
    return jsonify({
        'status': 'success',
        'message': 'API is working properly!',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })