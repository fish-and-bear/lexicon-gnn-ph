"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort
from sqlalchemy import or_, and_, func, desc, text, distinct
from sqlalchemy.orm import joinedload, contains_eager
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from models import (
    Word, Definition, Etymology, Relation, DefinitionRelation, 
    Affixation, PartOfSpeech, Pronunciation, db
)
from dictionary_manager import (
    RelationshipType, RelationshipCategory, BaybayinRomanizer,
    normalize_lemma, extract_etymology_components, extract_language_codes
)
from database import cached_query
import json
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.metrics import MetricWrapperBase

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
    ]), default='all')
    language = fields.Str()  # No validation - accept any language code
    pos = fields.Str(validate=validate.OneOf([
        'n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix'
    ]))
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=True)
    include_definitions = fields.Bool(default=True)
    include_examples = fields.Bool(default=True)
    include_usage = fields.Bool(default=True)
    include_baybayin = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)
    min_quality = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))
    sort = fields.Str(validate=validate.OneOf([
        'relevance', 'alphabetical', 'created', 'updated',
        'quality', 'frequency', 'complexity'
    ]), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')
    limit = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    offset = fields.Int(validate=validate.Range(min=0), default=0)
    region = fields.Str()  # For filtering by geographic region
    period = fields.Str()  # For filtering by time period
    formality = fields.Str()  # For filtering by formality level
    min_frequency = fields.Float()  # For filtering by usage frequency
    semantic_domain = fields.Str()  # For filtering by semantic domain
    grammatical_category = fields.Str()  # For filtering by grammatical category

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
@cached_query(timeout=300)
def get_word(word: str):
    """Get detailed information about a word."""
    schema = WordSchema()
    
    try:
        # Query word with all relationships
        query = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word),
            joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
            joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word),
            joinedload(Word.pronunciations)
        )

        # Try exact match first
        word_entry = query.filter(
            Word.normalized_lemma == normalize_lemma(word)
        ).first()

        # If not found, try fuzzy match
        if not word_entry:
            word_entry = query.filter(
                Word.search_text.match(word, postgresql_regconfig='simple')
            ).order_by(
                func.similarity(Word.lemma, word).desc()
            ).first()

        if not word_entry:
            return jsonify({
                "error": "Word not found",
                "suggestions": get_word_suggestions(word)
            }), 404

        # Get complete word data
        result = schema.dump(word_entry)
        
        # Add additional metadata
        result.update({
            "data_completeness": calculate_data_completeness(word_entry),
            "verification_history": get_verification_history(word_entry),
            "edit_history": get_edit_history(word_entry),
            "usage_statistics": get_usage_statistics(word_entry),
            "related_concepts": get_related_concepts(word_entry),
            "dialectal_variations": get_dialectal_variations(word_entry),
            "semantic_domains": get_semantic_domains(word_entry)
        })

        return jsonify(result), 200
        
    except Exception as e:
        API_ERRORS.labels(endpoint='/api/v2/words', error_type='processing').inc()
        logger.error("Error processing word request",
                    word=word,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/search", methods=["GET"])
def search_words():
    """Search for words with comprehensive filtering and sorting."""
    schema = SearchQuerySchema()
    try:
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    try:
        # Build base query
        query = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from),
            joinedload(Word.relations_to),
            joinedload(Word.pronunciations)
        )

        # Apply search mode
        search_term = params['q']
        if params['mode'] == 'exact':
            query = query.filter(Word.normalized_lemma == normalize_lemma(search_term))
        elif params['mode'] == 'baybayin':
            query = query.filter(
                Word.has_baybayin == True,
                Word.baybayin_form.ilike(f"%{search_term}%")
            )
        elif params['mode'] == 'phonetic':
            query = query.join(Word.pronunciations).filter(
                Pronunciation.value.ilike(f"%{search_term}%")
            )
        elif params['mode'] == 'etymology':
            query = query.join(Word.etymologies).filter(
                Etymology.etymology_text.ilike(f"%{search_term}%")
            )
        elif params['mode'] == 'semantic':
            query = query.join(Word.definitions).filter(
                Definition.definition_text.match(search_term, postgresql_regconfig='simple')
            )
        elif params['mode'] == 'root':
            query = query.filter(Word.is_root == True)
        elif params['mode'] == 'affixed':
            query = query.filter(Word.root_word_id.isnot(None))
        else:  # 'all' or 'fuzzy'
            query = query.filter(
                Word.search_text.match(search_term, postgresql_regconfig='simple')
            )

        # Apply filters
        if params.get('language'):
            query = query.filter(Word.language_code == params['language'])
        if params.get('pos'):
            query = query.join(Word.definitions).filter(
                Definition.standardized_pos.has(PartOfSpeech.code == params['pos'])
            )
        if params.get('min_quality'):
            query = query.filter(Word.quality_score >= params['min_quality'])
        if params.get('verification_status'):
            query = query.filter(Word.verification_status == params['verification_status'])
            
        # Apply sorting
        if params['sort'] == 'alphabetical':
            query = query.order_by(
                Word.normalized_lemma.asc() if params['order'] == 'asc' 
                else Word.normalized_lemma.desc()
            )
        elif params['sort'] == 'created':
            query = query.order_by(
                Word.created_at.asc() if params['order'] == 'asc'
                else Word.created_at.desc()
            )
        elif params['sort'] == 'updated':
            query = query.order_by(
                Word.updated_at.asc() if params['order'] == 'asc'
                else Word.updated_at.desc()
            )
        elif params['sort'] == 'quality':
            query = query.order_by(
                Word.quality_score.asc() if params['order'] == 'asc'
                else Word.quality_score.desc()
            )
        elif params['sort'] == 'frequency':
            query = query.order_by(
                Word.usage_frequency.asc() if params['order'] == 'asc'
                else Word.usage_frequency.desc()
            )
        elif params['sort'] == 'complexity':
            query = query.order_by(
                func.length(Word.lemma).asc() if params['order'] == 'asc'
                else func.length(Word.lemma).desc()
            )
        else:  # 'relevance'
            query = query.order_by(
                func.ts_rank_cd(Word.search_text, func.to_tsquery('simple', search_term)).desc(),
                func.similarity(Word.lemma, search_term).desc()
            )

        # Apply pagination
        total_count = query.count()
        query = query.offset(params['offset']).limit(params['limit'])

        # Execute query
        results = query.all()

        # Format results
        schema = WordSchema(
            only=(
                'id', 'lemma', 'language_code', 'has_baybayin', 'baybayin_form',
                'definitions' if params['include_definitions'] else None,
                'etymology' if params['include_etymology'] else None,
                'pronunciations' if params['include_pronunciation'] else None,
                'relations' if params['include_relations'] else None,
                'metadata' if params['include_metadata'] else None
            )
        )
        
        words = [schema.dump(word) for word in results]

        return jsonify({
            "query": search_term,
            "mode": params['mode'],
            "total_count": total_count,
            "offset": params['offset'],
            "limit": params['limit'],
            "words": words,
            "facets": generate_search_facets(results),
            "suggestions": generate_search_suggestions(search_term, results)
        }), 200
        
    except Exception as e:
        API_ERRORS.labels(endpoint='/api/v2/search', error_type='processing').inc()
        logger.error("Error processing search request",
                    params=params,
                    error=str(e))
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word: str):
    """Get word relationships with enhanced categorization."""
    try:
        # Get word from database
        word_obj = Word.query.filter(
            or_(
                Word.lemma == word,
                Word.normalized_lemma == normalize_lemma(word)
            )
        ).first()

        if not word_obj:
            return jsonify({"error": "Word not found"}), 404

        # Get all relations
        relations = {}
        for rel_type in RelationshipType:
            relations[rel_type.value] = {
                'category': rel_type.category.value,
                'bidirectional': rel_type.bidirectional,
                'strength': rel_type.strength,
                'words': []
            }

        # Process outgoing relations
        for rel in word_obj.relations_from:
            if rel.to_word:
                rel_type = RelationshipType.from_string(rel.relation_type)
                relations[rel_type.value]['words'].append({
                    'id': rel.to_word.id,
                    'lemma': rel.to_word.lemma,
                    'language_code': rel.to_word.language_code,
                    'has_baybayin': rel.to_word.has_baybayin,
                    'baybayin_form': rel.to_word.baybayin_form,
                    'confidence_score': rel.confidence_score,
                    'sources': rel.get_sources_list()
                })

        # Process incoming relations for bidirectional types
        for rel in word_obj.relations_to:
            if rel.from_word:
                rel_type = RelationshipType.from_string(rel.relation_type)
                if rel_type.bidirectional:
                    relations[rel_type.value]['words'].append({
                        'id': rel.from_word.id,
                        'lemma': rel.from_word.lemma,
                        'language_code': rel.from_word.language_code,
                        'has_baybayin': rel.from_word.has_baybayin,
                        'baybayin_form': rel.from_word.baybayin_form,
                        'confidence_score': rel.confidence_score,
                        'sources': rel.get_sources_list()
                    })
                elif rel_type.inverse:
                    # Add to inverse relationship type
                    inverse_type = rel_type.get_inverse()
                    relations[inverse_type.value]['words'].append({
                        'id': rel.from_word.id,
                        'lemma': rel.from_word.lemma,
                        'language_code': rel.from_word.language_code,
                        'has_baybayin': rel.from_word.has_baybayin,
                        'baybayin_form': rel.from_word.baybayin_form,
                        'confidence_score': rel.confidence_score,
                        'sources': rel.get_sources_list()
                    })

        # Group by categories
        categorized = {}
        for cat in RelationshipCategory:
            categorized[cat.value] = {
                'description': cat.description,
                'relations': {
                    rel_type: data
                    for rel_type, data in relations.items()
                    if RelationshipType.from_string(rel_type).category == cat
                }
            }

        return jsonify({
            'word': word_obj.lemma,
            'categories': categorized,
            'total_relations': sum(len(rel['words']) for rel in relations.values())
        })

    except Exception as e:
        logger.error(f"Error getting word relations: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        Word.language_code == params['language'],
        Word.normalized_lemma == word.lower()
    ).first()
    
    if not word_entry:
        return jsonify({"error": "Word not found"}), 404

    # Build query based on affixation type
    query = Affixation.query.options(
        joinedload(Affixation.root_word),
        joinedload(Affixation.affixed_word)
    )

    if params.get('type'):
        query = query.filter(Affixation.affix_type == params['type'])
    if params.get('position'):
        query = query.filter(Affixation.position == params['position'])
    if params.get('verification_status'):
        query = query.filter(Affixation.verification_status == params['verification_status'])

    # Get both as root and as affixed
    affixations = {
        "as_root": [],
        "as_affixed": []
    }

    # Word as root
    as_root = query.filter(Affixation.root_word_id == word_entry.id).all()
    affixations["as_root"] = [
        {
            "type": aff.affix_type,
            "position": aff.position,
            "value": aff.affix_value,
            "word": {
                "id": aff.affixed_word.id,
                "lemma": aff.affixed_word.lemma,
                "language_code": aff.affixed_word.language_code,
                "has_baybayin": aff.affixed_word.has_baybayin,
                "baybayin_form": aff.affixed_word.baybayin_form if aff.affixed_word.has_baybayin else None
            },
            "verification_status": aff.verification_status if params['include_verification'] else None,
            "metadata": aff.metadata if params['include_metadata'] else None,
            "source_info": aff.get_sources_list() if params['include_source_info'] else None
        }
        for aff in as_root
    ]

    # Word as affixed
    as_affixed = query.filter(Affixation.affixed_word_id == word_entry.id).all()
    affixations["as_affixed"] = [
        {
            "type": aff.affix_type,
            "position": aff.position,
            "value": aff.affix_value,
            "word": {
                "id": aff.root_word.id,
                "lemma": aff.root_word.lemma,
                "language_code": aff.root_word.language_code,
                "has_baybayin": aff.root_word.has_baybayin,
                "baybayin_form": aff.root_word.baybayin_form if aff.root_word.has_baybayin else None
            },
            "verification_status": aff.verification_status if params['include_verification'] else None,
            "metadata": aff.metadata if params['include_metadata'] else None,
            "source_info": aff.get_sources_list() if params['include_source_info'] else None
        }
        for aff in as_affixed
    ]

    return jsonify({
        "word": word_entry.lemma,
        "language_code": word_entry.language_code,
        "affixation_type": params.get('type'),
        "position": params.get('position'),
        "affixations": affixations
    }), 200

@bp.route("/api/v2/words/<path:word>/pronunciation", methods=["GET"])
def get_word_pronunciation(word: str):
    """Get pronunciation information for a word."""
    schema = PronunciationSchema()
    try:
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    # Get word entry
    word_entry = Word.query.filter(
        Word.normalized_lemma == word.lower()
    ).options(
        joinedload(Word.pronunciations)
    ).first()

    if not word_entry:
        return jsonify({"error": "Word not found"}), 404

    # Filter pronunciations by type if specified
    pronunciations = word_entry.pronunciations
    if params.get('type'):
        pronunciations = [p for p in pronunciations if p.type == params['type']]

    # Format pronunciation data
    result = {
        "word": word_entry.lemma,
        "language_code": word_entry.language_code,
        "pronunciations": [
            {
                "type": p.type,
                "value": p.value,
                "metadata": p.metadata if params['include_metadata'] else None,
                "source_info": p.get_sources_list() if params['include_source_info'] else None
            }
            for p in pronunciations
        ]
    }

    # Include pronunciation data from word if available
    if word_entry.pronunciation_data and params['include_variants']:
        pron_data = word_entry.get_pronunciation_data()
        if pron_data:
            result["additional_data"] = pron_data

    return jsonify(result), 200

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
    """Get comprehensive dictionary statistics."""
    try:
        stats = {
            "total_words": Word.query.count(),
            "words_by_language": dict(db.session.query(
                Word.language_code,
                func.count(Word.id)
            ).group_by(Word.language_code).all()),
            "words_by_pos": dict(db.session.query(
                PartOfSpeech.code,
                func.count(distinct(Word.id))
            ).join(Definition, Definition.word_id == Word.id)
             .join(PartOfSpeech, Definition.standardized_pos_id == PartOfSpeech.id)
             .group_by(PartOfSpeech.code).all()),
            "words_with_baybayin": Word.query.filter(Word.has_baybayin == True).count(),
            "words_with_etymology": Word.query.join(Word.etymologies).distinct().count(),
            "total_definitions": Definition.query.count(),
            "total_examples": Definition.query.filter(Definition.examples.isnot(None)).count(),
            "total_relations": Relation.query.count(),
            "verification_stats": dict(db.session.query(
                Word.verification_status,
                func.count(Word.id)
            ).group_by(Word.verification_status).all()),
            "quality_distribution": generate_quality_distribution(),
            "update_frequency": generate_update_frequency_stats()
        }
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route("/api/v2/words/<path:word>/etymology", methods=["GET"])
def get_word_etymology(word: str):
    """Get word etymology with enhanced processing."""
    try:
        # Get word from database
        word_obj = Word.query.filter(
            or_(
                Word.lemma == word,
                Word.normalized_lemma == normalize_lemma(word)
            )
        ).first()

        if not word_obj:
            return jsonify({"error": "Word not found"}), 404

        etymologies = []
        for etym in word_obj.etymologies:
            # Extract components using enhanced processing
            components = extract_etymology_components(etym.etymology_text)
            
            # Extract language codes
            language_codes = extract_language_codes(etym.etymology_text)
            
            etymology_data = {
                'etymology_text': etym.etymology_text,
                'components': components,
                'language_codes': language_codes,
                'structure': etym.etymology_structure,
                'confidence_score': etym.confidence_score,
                'sources': etym.get_sources_list(),
                'verification_status': etym.verification_status,
                'metadata': {
                    'has_uncertain_elements': any('?' in comp['text'] for comp in components),
                    'has_reconstructed_forms': any('*' in comp['text'] for comp in components),
                    'language_count': len(language_codes),
                    'component_count': len(components)
                }
            }
            
            # Add related words if they exist in our database
            related_words = []
            for comp in components:
                if comp.get('text'):
                    related = Word.query.filter(
                        Word.normalized_lemma == normalize_lemma(comp['text']),
                        Word.language_code == comp.get('language', '')
                    ).first()
                    if related:
                        related_words.append({
                            'id': related.id,
                            'lemma': related.lemma,
                            'language_code': related.language_code,
                            'component_text': comp['text']
                        })
            
            if related_words:
                etymology_data['related_words'] = related_words

            etymologies.append(etymology_data)

        return jsonify({
            'word': word_obj.lemma,
            'etymologies': etymologies,
            'has_complete_etymology': bool(etymologies),
            'etymology_quality_score': max((e['confidence_score'] or 0) for e in etymologies) if etymologies else 0
        })

    except Exception as e:
        logger.error(f"Error getting word etymology: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route("/api/v2/words/<path:word>/semantic_network", methods=["GET"])
def get_semantic_network(word: str):
    """Get the semantic network for a word."""
    try:
        depth = int(request.args.get('depth', 1))
        include_definitions = request.args.get('include_definitions', 'true').lower() == 'true'
        include_etymology = request.args.get('include_etymology', 'false').lower() == 'true'
        min_confidence = float(request.args.get('min_confidence', 0.0))
    except (ValueError, TypeError) as err:
        return jsonify({"error": str(err)}), 400

    # Get initial word
    word_entry = Word.query.filter(
        Word.normalized_lemma == normalize_lemma(word)
    ).first()

    if not word_entry:
        return jsonify({"error": "Word not found"}), 404

    # Build semantic network
    network = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "root_word": word_entry.lemma,
            "depth": depth,
            "total_nodes": 0,
            "total_edges": 0
        }
    }

    processed_words = set()
    words_to_process = [(word_entry, 0)]

    while words_to_process:
        current_word, current_depth = words_to_process.pop(0)
        
        if current_word.id in processed_words or current_depth > depth:
            continue

        # Add node
        node_data = {
            "id": current_word.id,
            "label": current_word.lemma,
            "language": current_word.language_code,
            "type": "root" if current_word.id == word_entry.id else "related",
            "has_baybayin": current_word.has_baybayin,
            "baybayin_form": current_word.baybayin_form if current_word.has_baybayin else None
        }

        if include_definitions:
            node_data["definitions"] = [
                {
                    "text": d.definition_text,
                    "pos": d.standardized_pos
                }
                for d in current_word.definitions[:2]  # Limit to top 2 definitions
            ]

        if include_etymology and current_word.etymologies:
            node_data["etymology"] = {
                "text": current_word.etymologies[0].etymology_text,
                "confidence": current_word.etymology_confidence
            }

        network["nodes"].append(node_data)
        processed_words.add(current_word.id)

        # Process relations if not at max depth
        if current_depth < depth:
            relations = (
                Relation.query
                .filter(
                    or_(
                        Relation.from_word_id == current_word.id,
                        Relation.to_word_id == current_word.id
                    ),
                    Relation.confidence_score >= min_confidence
                )
                .options(
                    joinedload(Relation.from_word),
                    joinedload(Relation.to_word)
                )
                .all()
            )

            for relation in relations:
                related_word = (
                    relation.to_word if relation.from_word_id == current_word.id
                    else relation.from_word
                )

                if related_word.id not in processed_words:
                    words_to_process.append((related_word, current_depth + 1))

                # Add edge
                edge = {
                    "source": relation.from_word_id,
                    "target": relation.to_word_id,
                    "type": relation.relation_type,
                    "bidirectional": relation.bidirectional,
                    "confidence": relation.confidence_score
                }
                network["edges"].append(edge)

    # Update metadata
    network["metadata"].update({
        "total_nodes": len(network["nodes"]),
        "total_edges": len(network["edges"])
    })

    return jsonify(network), 200

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

def generate_search_facets(results) -> Dict[str, Any]:
    """Generate faceted search results."""
    facets = {
        "languages": {},
        "pos": {},
        "verification_status": {},
        "has_baybayin": 0,
        "has_etymology": 0,
        "has_pronunciation": 0
    }
    
    for word in results:
        # Count languages
        facets["languages"][word.language_code] = (
            facets["languages"].get(word.language_code, 0) + 1
        )
        
        # Count parts of speech
        for def_ in word.definitions:
            if def_.standardized_pos:
                facets["pos"][def_.standardized_pos] = (
                    facets["pos"].get(def_.standardized_pos, 0) + 1
                )
        
        # Count verification status
        facets["verification_status"][word.verification_status] = (
            facets["verification_status"].get(word.verification_status, 0) + 1
        )
        
        # Count features
        if word.has_baybayin:
            facets["has_baybayin"] += 1
        if word.etymologies:
            facets["has_etymology"] += 1
        if word.pronunciations:
            facets["has_pronunciation"] += 1
    
    return facets

def generate_search_suggestions(query: str, results) -> List[Dict[str, Any]]:
    """Generate search suggestions based on results."""
    suggestions = []
    
    # Add spelling suggestions
    if len(results) == 0:
        # Get similar words using trigram similarity
        similar = Word.query.filter(
            func.similarity(Word.lemma, query) > 0.3
        ).order_by(
            func.similarity(Word.lemma, query).desc()
        ).limit(3).all()
        
        for word in similar:
            suggestions.append({
                "type": "spelling",
                "suggestion": word.lemma,
                "similarity": word.calculate_similarity_score(query)
            })
    
    # Add related searches based on semantic relationships
    if len(results) > 0:
        sample_word = results[0]
        related = (
            Relation.query
            .filter(
                Relation.from_word_id == sample_word.id,
                Relation.relation_type.in_(['synonym', 'related'])
            )
            .options(joinedload(Relation.to_word))
            .limit(3)
            .all()
        )
        
        for rel in related:
            suggestions.append({
                "type": "related",
                "suggestion": rel.to_word.lemma,
                "relation_type": rel.relation_type
            })
    
    return suggestions

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