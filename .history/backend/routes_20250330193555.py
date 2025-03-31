"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort
from sqlalchemy import or_, and_, func, desc, text
from sqlalchemy.orm import joinedload, contains_eager
from models import (
    Word, Definition, Etymology, Relation, DefinitionRelation, 
    Affixation, PartOfSpeech, Pronunciation, db
)
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import structlog
from baybayin_romanizer import BaybayinRomanizer

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__)

# Schema definitions
class WordQuerySchema(Schema):
    """Schema for word query parameters."""
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb', 'chv', 'en', 'es', 'spl', 'psp']), default='tl')
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=True)
    include_idioms = fields.Bool(default=True)
    include_source_info = fields.Bool(default=True)
    include_baybayin = fields.Bool(default=True)
    include_romanized = fields.Bool(default=True)
    include_affixations = fields.Bool(default=True)

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix']))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb', 'chv', 'en', 'es', 'spl', 'psp']), default='tl')
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf([
        'all', 'exact', 'phonetic', 'baybayin', 'fuzzy', 'etymology'
    ]), default='all')
    sort = fields.Str(validate=validate.OneOf([
        'relevance', 'alphabetical', 'created', 'updated', 'etymology', 'quality'
    ]), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')
    include_definitions = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=True)
    include_idioms = fields.Bool(default=True)
    include_source_info = fields.Bool(default=True)
    filter_has_etymology = fields.Bool(default=False)
    filter_has_pronunciation = fields.Bool(default=False)
    filter_has_idioms = fields.Bool(default=False)
    filter_has_baybayin = fields.Bool(default=False)
    filter_verified = fields.Bool(default=False)

class RelationQuerySchema(Schema):
    """Schema for relation query parameters."""
    type = fields.Str(validate=validate.OneOf([
        # Semantic relationships
        'synonym', 'antonym', 'related', 'similar',
        # Hierarchical relationships
        'hypernym', 'hyponym', 'meronym', 'holonym',
        # Derivational relationships
        'derived_from', 'root_of',
        # Variant relationships
        'variant', 'spelling_variant', 'regional_variant',
        # Usage relationships
        'compare_with', 'see_also',
        # Other relationships
        'equals'
    ]))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb', 'chv', 'en', 'es', 'spl', 'psp']), default='tl')
    include_metadata = fields.Bool(default=True)
    include_source_info = fields.Bool(default=True)
    include_confidence = fields.Bool(default=True)
    include_verification = fields.Bool(default=True)
    min_confidence = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.0)
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))

class AffixationQuerySchema(Schema):
    """Schema for affixation query parameters."""
    type = fields.Str(validate=validate.OneOf([
        'prefix', 'infix', 'suffix', 'circumfix', 'reduplication', 'compound'
    ]))
    position = fields.Str(validate=validate.OneOf([
        'initial', 'medial', 'final', 'both'
    ]))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb', 'chv', 'en', 'es', 'spl', 'psp']), default='tl')
    include_metadata = fields.Bool(default=True)
    include_source_info = fields.Bool(default=True)
    include_verification = fields.Bool(default=True)
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))

class BaybayinQuerySchema(Schema):
    """Schema for Baybayin query parameters."""
    text = fields.Str(required=True, validate=validate.Length(min=1, max=1000))
    mode = fields.Str(validate=validate.OneOf(['romanize', 'transliterate']), default='romanize')
    validate_text = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=True)

class PronunciationQuerySchema(Schema):
    """Schema for pronunciation query parameters."""
    type = fields.Str(validate=validate.OneOf(['ipa', 'respelling', 'audio', 'phonemic']))
    include_metadata = fields.Bool(default=True)
    include_source_info = fields.Bool(default=True)
    include_variants = fields.Bool(default=True)

# API Routes
@bp.route("/api/v2/words/<path:word>", methods=["GET"])
def get_word(word: str):
    """Get detailed information about a word."""
    schema = WordQuerySchema()
    try:
        params = schema.load(request.args)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

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
        Word.language_code == params['language_code'],
        Word.normalized_lemma == word.lower()
    ).first()

    # If not found, try fuzzy match
    if not word_entry:
        word_entry = query.filter(
            Word.language_code == params['language_code'],
            Word.search_text.match(word, postgresql_regconfig='simple')
        ).order_by(
            func.similarity(Word.lemma, word).desc()
        ).first()

    if not word_entry:
        return jsonify({
            "error": "Word not found",
            "suggestions": get_word_suggestions(word, params['language_code'])
        }), 404

    # Convert to dictionary with requested includes
    return jsonify(word_entry.to_dict(
        include_definitions=params['include_definitions'],
        include_etymology=params['include_etymology'],
        include_relations=params['include_relations'],
        include_metadata=params['include_metadata']
    )), 200

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
        joinedload(Word.relations_to),
        joinedload(Word.pronunciations)
    )

    # Apply language filter
    if params['language']:
        query = query.filter(Word.language_code == params['language'])

    # Apply search mode
    search_term = params['q']
    if params['mode'] == 'exact':
        query = query.filter(Word.normalized_lemma == search_term.lower())
    elif params['mode'] == 'baybayin':
        query = query.filter(
            Word.has_baybayin == True,
            Word.baybayin_form.ilike(f"%{search_term}%")
        )
    elif params['mode'] == 'phonetic':
        query = query.filter(
            Word.pronunciation_data.op('->>')('ipa').ilike(f"%{search_term}%")
        )
    elif params['mode'] == 'etymology':
        query = query.join(Word.etymologies).filter(
            Etymology.etymology_text.ilike(f"%{search_term}%")
        )
    else:  # 'all' or 'fuzzy'
        query = query.filter(
            Word.search_text.match(search_term, postgresql_regconfig='simple')
        )

    # Apply filters
    if params['pos']:
        query = query.join(Word.definitions).filter(
            Definition.standardized_pos.has(PartOfSpeech.code == params['pos'])
        )
    if params['filter_has_etymology']:
        query = query.filter(Word.etymologies.any())
    if params['filter_has_pronunciation']:
        query = query.filter(Word.pronunciations.any())
    if params['filter_has_idioms']:
        query = query.filter(Word.idioms != '[]')
    if params['filter_has_baybayin']:
        query = query.filter(Word.has_baybayin == True)
    if params['filter_verified']:
        query = query.filter(Word.verification_status == 'verified')

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
    elif params['sort'] == 'etymology':
        query = query.outerjoin(Word.etymologies).group_by(Word.id).order_by(
            func.count(Etymology.id).asc() if params['order'] == 'asc'
            else func.count(Etymology.id).desc()
        )
    elif params['sort'] == 'quality':
        # Use calculated quality score for sorting
        query = query.order_by(
            Word.calculate_data_quality_score().asc() if params['order'] == 'asc'
            else Word.calculate_data_quality_score().desc()
        )
    else:  # 'relevance'
        query = query.order_by(
            func.similarity(Word.lemma, search_term).desc()
        )

    # Execute query with limit
    results = query.limit(params['limit']).all()

    # Convert results to dictionaries with requested includes
    words = [
        word.to_dict(
            include_definitions=params['include_definitions'],
            include_etymology=params['include_etymology'],
            include_relations=False,  # Don't include relations in search results
            include_metadata=params['include_metadata']
        )
        for word in results
    ]

    return jsonify({
        "query": search_term,
        "language": params['language'],
        "mode": params['mode'],
        "count": len(words),
        "words": words
    }), 200

@bp.route("/api/v2/words/<path:word>/relations", methods=["GET"])
def get_word_relations(word: str):
    """Get all relationships for a word."""
    schema = RelationQuerySchema()
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

    # Build query based on relation type
    query = Relation.query.options(
        joinedload(Relation.from_word),
        joinedload(Relation.to_word)
    )

    if params['type']:
        query = query.filter(Relation.relation_type == params['type'])
    if params.get('min_confidence'):
        query = query.filter(Relation.confidence_score >= params['min_confidence'])
    if params.get('verification_status'):
        query = query.filter(Relation.verification_status == params['verification_status'])

    # Get both outgoing and incoming relations
    relations = []
    
    # Outgoing relations
    outgoing = query.filter(Relation.from_word_id == word_entry.id).all()
    relations.extend([
        {
            "type": rel.relation_type,
            "direction": "outgoing",
            "word": {
                "id": rel.to_word.id,
                "lemma": rel.to_word.lemma,
                "language_code": rel.to_word.language_code,
                "has_baybayin": rel.to_word.has_baybayin,
                "baybayin_form": rel.to_word.baybayin_form if rel.to_word.has_baybayin else None
            },
            "bidirectional": rel.bidirectional,
            "confidence_score": rel.confidence_score,
            "verification_status": rel.verification_status if params['include_verification'] else None,
            "metadata": rel.metadata if params['include_metadata'] else None,
            "source_info": rel.get_sources_list() if params['include_source_info'] else None
        }
        for rel in outgoing
    ])

    # Incoming relations
    incoming = query.filter(Relation.to_word_id == word_entry.id).all()
    relations.extend([
        {
            "type": rel.relation_type,
            "direction": "incoming",
            "word": {
                "id": rel.from_word.id,
                "lemma": rel.from_word.lemma,
                "language_code": rel.from_word.language_code,
                "has_baybayin": rel.from_word.has_baybayin,
                "baybayin_form": rel.from_word.baybayin_form if rel.from_word.has_baybayin else None
            },
            "bidirectional": rel.bidirectional,
            "confidence_score": rel.confidence_score,
            "verification_status": rel.verification_status if params['include_verification'] else None,
            "metadata": rel.metadata if params['include_metadata'] else None,
            "source_info": rel.get_sources_list() if params['include_source_info'] else None
        }
        for rel in incoming
    ])

    return jsonify({
        "word": word_entry.lemma,
        "language_code": word_entry.language_code,
        "relation_type": params.get('type'),
        "relations": relations
    }), 200

@bp.route("/api/v2/words/<path:word>/affixations", methods=["GET"])
def get_word_affixations(word: str):
    """Get all affixations for a word."""
    schema = AffixationQuerySchema()
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
    schema = PronunciationQuerySchema()
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
    """Process Baybayin text for romanization or transliteration."""
    schema = BaybayinQuerySchema()
    try:
        params = schema.load(request.json)
    except ValidationError as err:
        return jsonify({"error": str(err)}), 400

    text = params['text']
    mode = params['mode']
    
    # Initialize Baybayin processor
    processor = BaybayinRomanizer()

    # Validate text if requested
    if params['validate_text'] and not processor.validate_text(text):
        return jsonify({
            "error": "Invalid Baybayin text",
            "details": "Text contains invalid Baybayin characters or combinations"
        }), 400

    try:
        if mode == 'romanize':
            result = processor.romanize(text)
        else:  # transliterate
            result = processor.transliterate_to_baybayin(text)

        response = {
            "input": text,
            "mode": mode,
            "result": result
        }

        # Include additional metadata if requested
        if params['include_metadata']:
            response["metadata"] = {
                "character_count": len(text),
                "syllable_count": len(text.split()),
                "valid_baybayin": processor.is_baybayin(text)
            }

        # Include pronunciation if requested
        if params['include_pronunciation']:
            response["pronunciation"] = {
                "ipa": get_pronunciation(result) if mode == 'romanize' else get_pronunciation(text),
                "respelling": get_respelling(result) if mode == 'romanize' else get_respelling(text)
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

# Helper functions
def get_word_suggestions(word: str, language_code: str) -> List[Dict[str, Any]]:
    """Get word suggestions for a failed lookup."""
    # Try fuzzy matching
    matches = Word.query.filter(
        Word.language_code == language_code,
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