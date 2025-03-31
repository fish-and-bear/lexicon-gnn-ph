"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast
from sqlalchemy.orm import joinedload, contains_eager
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, db
)
from dictionary_manager import (
    RelationshipType, RelationshipCategory, BaybayinRomanizer,
    normalize_lemma, extract_etymology_components, extract_language_codes
)
from database import cached_query
import json
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict
import logging

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__)

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

# Schema definitions
class BaseSchema(Schema):
    """Base schema with common metadata fields."""
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))
    quality_score = fields.Float()
    metadata = fields.Dict(keys=fields.Str(), values=fields.Raw())
    sources = fields.List(fields.Str())

class PronunciationSchema(BaseSchema):
    """Schema for pronunciation data."""
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic']))
    value = fields.Str(required=True)
    variants = fields.List(fields.Str())
    phonemes = fields.List(fields.Str())
    stress_pattern = fields.Str()
    syllable_count = fields.Int()
    is_primary = fields.Bool()
    dialect = fields.Str()
    region = fields.Str()
    usage_frequency = fields.Float()
    
class EtymologySchema(BaseSchema):
    """Schema for etymology data."""
    etymology_text = fields.Str(required=True)
    language_codes = fields.List(fields.Str())
    components = fields.List(fields.Dict())
    structure = fields.Dict()
    confidence_score = fields.Float()
    period = fields.Str()  # Historical period
    reconstructed = fields.Bool()
    uncertain = fields.Bool()
    notes = fields.Str()

class RelationSchema(BaseSchema):
    """Schema for word relationships."""
    relation_type = fields.Str(required=True, validate=validate.OneOf([
        rel.value for rel in RelationshipType
    ]))
    category = fields.Str(validate=validate.OneOf([
        cat.value for cat in RelationshipCategory
    ]))
    bidirectional = fields.Bool()
    strength = fields.Float()  # Confidence/strength of the relationship
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))

class AffixationSchema(BaseSchema):
    """Schema for word affixation data."""
    affix_type = fields.Str(validate=validate.OneOf([
        'prefix', 'infix', 'suffix', 'circumfix', 'reduplication', 'compound'
    ]))
    position = fields.Str(validate=validate.OneOf([
        'initial', 'medial', 'final', 'both'
    ]))
    value = fields.Str()
    word = fields.Nested('WordSchema', only=('id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form'))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')

class DefinitionSchema(BaseSchema):
    """Schema for word definitions."""
    definition_text = fields.Str(required=True)
    part_of_speech = fields.Str()
    standardized_pos = fields.Str()
    examples = fields.List(fields.Dict(keys=fields.Str()))
    usage_notes = fields.Str()
    register = fields.Str()  # Formal, informal, etc.
    domain = fields.Str()  # Subject domain
    dialect = fields.Str()
    region = fields.Str()
    time_period = fields.Str()
    frequency = fields.Float()
    confidence_score = fields.Float()
    related_definitions = fields.List(fields.Nested('self'))

class WordSchema(BaseSchema):
    """Schema for word entries."""
    id = fields.Int(dump_only=True)
    lemma = fields.Str(required=True)
    normalized_lemma = fields.Str()
    language_code = fields.Str(required=True)
    has_baybayin = fields.Bool()
    baybayin_form = fields.Str()
    romanized_form = fields.Str()
    is_root = fields.Bool()
    root_word = fields.Nested('self', only=('id', 'lemma', 'language_code'))
    preferred_spelling = fields.Str()
    alternative_spellings = fields.List(fields.Str())
    syllable_count = fields.Int()
    pronunciation_guide = fields.Str()
    stress_pattern = fields.Str()
    formality_level = fields.Str()
    usage_frequency = fields.Float()
    geographic_region = fields.Str()
    time_period = fields.Str()
    cultural_notes = fields.Str()
    grammatical_categories = fields.List(fields.Str())
    semantic_domains = fields.List(fields.Str())
    etymology_confidence = fields.Float()
    data_quality_score = fields.Float()
    pronunciation_data = fields.Dict()
    tags = fields.List(fields.Str())
    idioms = fields.List(fields.Dict())
    source_info = fields.Dict()
    metadata = fields.Dict()
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))
    verification_notes = fields.Str()
    last_verified_at = fields.DateTime()
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)
    etymology = fields.Nested(EtymologySchema)
    definitions = fields.List(fields.Nested(DefinitionSchema))
    pronunciations = fields.List(fields.Nested(PronunciationSchema))
    relations = fields.List(fields.Nested(RelationSchema))
    affixations = fields.Dict(keys=fields.Str(), values=fields.List(fields.Nested(AffixationSchema)))

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
    min_quality = fields.Float(validate=validate.Range(min=0.0, max=1.0), dump_default=None, load_default=None)
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]), dump_default=None, load_default=None)
    sort = fields.Str(validate=validate.OneOf([
        'relevance', 'alphabetical', 'created', 'updated',
        'quality', 'frequency', 'complexity'
    ]), dump_default='relevance', load_default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), dump_default='desc', load_default='desc')
    limit = fields.Int(validate=validate.Range(min=1, max=100), dump_default=20, load_default=20)
    offset = fields.Int(validate=validate.Range(min=0), dump_default=0, load_default=0)
    region = fields.Str(dump_default=None, load_default=None)
    period = fields.Str(dump_default=None, load_default=None)
    formality = fields.Str(dump_default=None, load_default=None)
    min_frequency = fields.Float(dump_default=None, load_default=None)
    semantic_domain = fields.Str(dump_default=None, load_default=None)
    grammatical_category = fields.Str(dump_default=None, load_default=None)

class StatisticsSchema(Schema):
    """Schema for dictionary statistics."""
    total_words = fields.Int()
    words_by_language = fields.Dict(keys=fields.Str(), values=fields.Int())
    words_by_pos = fields.Dict(keys=fields.Str(), values=fields.Int())
    words_with_baybayin = fields.Int()
    words_with_etymology = fields.Int()
    total_definitions = fields.Int()
    total_examples = fields.Int()
    total_relations = fields.Int()
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
    return response

@bp.route("/api/v2/words/<path:word>", methods=["GET"])
def get_word(word: str):
    """Get word details."""
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

        # Convert to dictionary with all details
        result = word_entry.to_dict(
            include_definitions=True,
            include_etymology=True,
            include_relations=True,
            include_metadata=True
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing word request: {str(e)}", word=word)
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/search", methods=["GET"])
def search_words():
    """Search for words with comprehensive filtering and sorting."""
    schema = SearchQuerySchema()
    try:
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    # Build base query
    query = Word.query.options(
        joinedload(Word.definitions).joinedload(Definition.standardized_pos),
        joinedload(Word.etymologies),
        joinedload(Word.relations_from),
        joinedload(Word.relations_to)
    )

    # Apply filters based on search mode
    search_text = params['q'].lower()
    if params['mode'] == 'exact':
        query = query.filter(Word.normalized_lemma == search_text)
    elif params['mode'] == 'phonetic':
        # Add phonetic matching logic
        pass
    elif params['mode'] == 'baybayin':
        query = query.filter(Word.has_baybayin == True)
        query = query.filter(Word.baybayin_form.ilike(f"%{search_text}%"))
    else:  # Default to 'all' mode
        query = query.filter(
            or_(
                Word.normalized_lemma.ilike(f"%{search_text}%"),
                Word.search_text.match(search_text)
            )
        )

    # Apply additional filters
    if params.get('language'):
        query = query.filter(Word.language_code == params['language'])
    if params.get('pos'):
        query = query.join(Definition).filter(Definition.standardized_pos.has(code=params['pos']))
    if params.get('min_quality'):
        query = query.filter(Word.quality_score >= params['min_quality'])
    if params.get('verification_status'):
        query = query.filter(Word.verification_status == params['verification_status'])

    # Apply sorting
    if params['sort'] == 'alphabetical':
        query = query.order_by(Word.normalized_lemma.asc() if params['order'] == 'asc' else Word.normalized_lemma.desc())
    elif params['sort'] == 'created':
        query = query.order_by(Word.created_at.asc() if params['order'] == 'asc' else Word.created_at.desc())
    elif params['sort'] == 'updated':
        query = query.order_by(Word.updated_at.asc() if params['order'] == 'asc' else Word.updated_at.desc())
    elif params['sort'] == 'quality':
        query = query.order_by(Word.quality_score.asc() if params['order'] == 'asc' else Word.quality_score.desc())
    else:  # Default to relevance
        query = query.order_by(
            func.ts_rank_cd(Word.search_text, func.to_tsquery('simple', search_text)).desc(),
            func.similarity(Word.lemma, search_text).desc()
        )

    # Get total count before pagination
    total_count = query.count()

    # Apply pagination
    query = query.offset(params['offset']).limit(params['limit'])

    # Execute query and format results
    results = query.all()
    words = [
        word.to_dict(
            include_definitions=params['include_definitions'],
            include_etymology=params['include_etymology'],
            include_relations=params['include_relations'],
            include_metadata=params['include_metadata']
        )
        for word in results
    ]

    # Generate facets and suggestions
    facets = generate_search_facets(results)
    suggestions = generate_search_suggestions(params['q'], results)

    return jsonify({
        "words": words,
        "total": total_count,
        "offset": params['offset'],
        "limit": params['limit'],
        "query": params['q'],
        "mode": params['mode'],
        "facets": facets,
        "suggestions": suggestions
    }), 200

@bp.route("/api/v2/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word):
    """Get word relations."""
    try:
        word_obj = Word.query.filter(Word.normalized_lemma == word.lower()).first_or_404()
        
        # Get all relations
        relations = {
            'synonyms': [],
            'antonyms': [],
            'variants': [],
            'related': [],
            'affixations': {
                'as_root': [],
                'as_affixed': []
            }
        }
        
        # Process outgoing relations
        for rel in word_obj.relations_from:
            target = rel.target_word
            relation_data = {
                'word': target.lemma,
                'normalized_lemma': target.normalized_lemma,
                'language_code': target.language_code,
                'type': rel.type,
                'direction': 'outgoing',
                'metadata': rel.metadata or {}
            }
            
            if rel.type == 'affixation':
                relations['affixations']['as_root'].append(relation_data)
            else:
                relations[rel.type + 's'].append(relation_data)
        
        # Process incoming relations
        for rel in word_obj.relations_to:
            source = rel.source_word
            relation_data = {
                'word': source.lemma,
                'normalized_lemma': source.normalized_lemma,
                'language_code': source.language_code,
                'type': rel.type,
                'direction': 'incoming',
                'metadata': rel.metadata or {}
            }
            
            if rel.type == 'affixation':
                relations['affixations']['as_affixed'].append(relation_data)
            else:
                relations[rel.type + 's'].append(relation_data)
        
        return jsonify(relations), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word relations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get all affixations for a word."""
    schema = AffixationSchema()
    try:
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    # Get word entry
    word_entry = Word.query.filter(
        Word.normalized_lemma == word.lower()
    ).first()

    if not word_entry:
        return jsonify({
            "error": "Word not found",
            "suggestions": get_word_suggestions(word)
        }), 404

    # Get affixations
    affixations = Affixation.query.filter(
        Affixation.root_word_id == word_entry.id
    ).all()

    result = {
        "word": word_entry.lemma,
        "affixations": [
            {
                "affix_type": aff.affix_type,
                "position": aff.position,
                "value": aff.value,
                "word": {
                    "id": aff.affixed_word.id,
                    "lemma": aff.affixed_word.lemma,
                    "language_code": aff.affixed_word.language_code,
                    "has_baybayin": aff.affixed_word.has_baybayin,
                    "baybayin_form": aff.affixed_word.baybayin_form
                } if aff.affixed_word else None
            }
            for aff in affixations
        ]
    }

    return jsonify(result), 200

@bp.route("/api/v2/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word):
    """Get word pronunciation."""
    try:
        word_obj = Word.query.filter(Word.normalized_lemma == word.lower()).first_or_404()
        
        pronunciation_data = word_obj.pronunciation_data or {}
        
        # Format pronunciation data to match test expectations
        pronunciations = []
        if pronunciation_data.get('ipa'):
            pronunciations.append({
                'type': 'ipa',
                'value': pronunciation_data['ipa'],
                'metadata': pronunciation_data.get('ipa_metadata', {})
            })
        if pronunciation_data.get('respelling'):
            pronunciations.append({
                'type': 'respelling',
                'value': pronunciation_data['respelling'],
                'metadata': pronunciation_data.get('respelling_metadata', {})
            })
        
        return jsonify({
            'pronunciations': pronunciations,
            'has_pronunciation': bool(pronunciations),
            'audio_url': pronunciation_data.get('audio_url'),
            'syllables': pronunciation_data.get('syllables'),
            'stress_pattern': pronunciation_data.get('stress_pattern'),
            'notes': pronunciation_data.get('notes')
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word pronunciation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/baybayin/process", methods=["POST"])
def process_baybayin():
    """Process text for Baybayin conversion with enhanced accuracy."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        romanizer = BaybayinRomanizer()
        
        # Process each word
        results = []
        for word in text.split():
            baybayin_form = None
            romanized_form = None
            
            # Check if word is already in Baybayin
            if romanizer.is_baybayin(word):
                baybayin_form = word
                romanized_form = romanizer.romanize(word)
            else:
                # Attempt to transliterate to Baybayin
                try:
                    baybayin_form = transliterate_to_baybayin(word)
                    if baybayin_form:
                        romanized_form = word
                except Exception:
                    pass
            
            if baybayin_form:
                results.append({
                    'original': word,
                    'baybayin': baybayin_form,
                    'romanized': romanized_form,
                    'is_valid': romanizer.validate_text(baybayin_form) if baybayin_form else False
                })
            else:
                results.append({
                    'original': word,
                    'error': 'Could not process text'
                })

        return jsonify({
            'results': results,
            'success': True
        })

    except Exception as e:
        logger.error(f"Error processing Baybayin text: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route("/api/v2/statistics", methods=["GET"])
def get_statistics():
    """Get dictionary statistics."""
    try:
        total_words = Word.query.count()
        total_definitions = Definition.query.count()
        total_etymologies = Etymology.query.count()
        total_relations = Relation.query.count()
        total_affixations = Affixation.query.count()

        # Count words with examples
        words_with_examples = db.session.query(func.count(Definition.id)).filter(
            cast(Definition.examples, db.String) != cast('[]', db.String)
        ).scalar()

        # Count words by language
        words_by_language = db.session.query(
            Word.language_code,
            func.count(Word.id).label('count')
        ).group_by(Word.language_code).all()

        # Count words by verification status
        words_by_status = db.session.query(
            Word.verification_status,
            func.count(Word.id).label('count')
        ).group_by(Word.verification_status).all()

        return jsonify({
            'total_words': total_words,
            'total_definitions': total_definitions,
            'total_etymologies': total_etymologies,
            'total_relations': total_relations,
            'total_affixations': total_affixations,
            'words_with_examples': words_with_examples,
            'words_by_language': {
                lang: count for lang, count in words_by_language
            },
            'words_by_status': {
                status if status else 'unverified': count
                for status, count in words_by_status
            }
        })
    except Exception as e:
        logger.error("Error getting statistics", error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word):
    """Get word etymology."""
    try:
        word_obj = Word.query.filter(Word.normalized_lemma == word.lower()).first_or_404()
        
        if not word_obj.etymologies:
            return jsonify({
                'etymologies': [],
                'has_etymology': False
            }), 200
            
        etymologies = []
        for etym in word_obj.etymologies:
            etymology_data = {
                'id': etym.id,
                'etymology_text': etym.etymology_text,
                'components': etym.components or [],
                'language_codes': etym.language_codes or [],
                'confidence_score': etym.confidence_score,
                'verification_status': etym.verification_status,
                'sources': etym.sources or []
            }
            etymologies.append(etymology_data)
            
        return jsonify({
            'etymologies': etymologies,
            'has_etymology': True
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting word etymology: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<path:word>/semantic_network", methods=["GET"])
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

@bp.route("/api/v2/words/<path:word>/affixation_tree", methods=["GET"])
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

@bp.route("/api/v2/words/<int:word_id>/etymology/tree", methods=["GET"])
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
        API_ERRORS.labels(endpoint='/api/v2/words/etymology/tree', error_type='processing').inc()
        logger.error("Error processing etymology tree request",
                    word_id=word_id,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<int:word_id>/relations/graph", methods=["GET"])
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
        API_ERRORS.labels(endpoint='/api/v2/words/relations/graph', error_type='processing').inc()
        logger.error("Error processing relation graph request",
                    word_id=word_id,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500