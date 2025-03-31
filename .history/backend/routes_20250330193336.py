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
                        "id": word.id,
                        "lemma": word.lemma,
                        "normalized_lemma": word.normalized_lemma,
                        "language_code": word.language_code,
                        "quality_score": word.quality_score
                    },
                    "category": category_name,
                    "priority": priority_level,
                    "suggestion": suggestion_text,
                    "impact": impact_text
                })
        
        for word in words:
            # Critical suggestions
            if not word.has_definitions:
                add_suggestion(
                    word,
                    "definitions",
                    "critical",
                    "Add at least one definition",
                    "Essential for word meaning and usability"
                )
            
            # High priority suggestions
            if not word.has_baybayin and word.language_code == 'tl':
                add_suggestion(
                    word,
                    "basic_info",
                    "high",
                    "Add Baybayin script representation",
                    "Improves cultural authenticity and search capabilities"
                )
            
            if word.has_definitions and word.definitions_with_examples == 0:
                add_suggestion(
                    word,
                    "definitions",
                    "high",
                    "Add usage examples to definitions",
                    "Helps users understand word usage in context"
                )
            
            if not word.has_etymology:
                add_suggestion(
                    word,
                    "etymology",
                    "high",
                    "Add etymology information",
                    "Provides word origin and historical context"
                )
            
            # Medium priority suggestions
            if not word.has_pronunciation:
                add_suggestion(
                    word,
                    "basic_info",
                    "medium",
                    "Add pronunciation data",
                    "Helps users pronounce the word correctly"
                )
            
            if word.has_etymology and word.etymologies_with_components == 0:
                add_suggestion(
                    word,
                    "etymology",
                    "medium",
                    "Add etymology components",
                    "Clarifies word formation and relationships"
                )
            
            if word.has_definitions and word.definitions_with_usage_notes == 0:
                add_suggestion(
                    word,
                    "definitions",
                    "medium",
                    "Add usage notes to definitions",
                    "Provides additional context and guidance"
                )
            
            if not word.has_relations:
                add_suggestion(
                    word,
                    "relationships",
                    "medium",
                    "Add word relationships",
                    "Improves word network and discoverability"
                )
            
            # Low priority suggestions
            if not word.has_idioms:
                add_suggestion(
                    word,
                    "additional_features",
                    "low",
                    "Add idiomatic expressions",
                    "Enriches usage understanding"
                )
        
        # Sort suggestions by priority and quality score
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda x: (
            priority_order[x["priority"]],
            x["word"]["quality_score"]
        ))
        
        # Group suggestions by category
        grouped_suggestions = {}
        for suggestion in suggestions:
            cat = suggestion["category"]
            if cat not in grouped_suggestions:
                grouped_suggestions[cat] = {
                    "total": 0,
                    "by_priority": {
                        "critical": [],
                        "high": [],
                        "medium": [],
                        "low": []
                    }
                }
            grouped_suggestions[cat]["total"] += 1
            grouped_suggestions[cat]["by_priority"][suggestion["priority"]].append(suggestion)
        
        # Calculate statistics
        stats = {
            "total_suggestions": len(suggestions),
            "by_priority": {
                "critical": len([s for s in suggestions if s["priority"] == "critical"]),
                "high": len([s for s in suggestions if s["priority"] == "high"]),
                "medium": len([s for s in suggestions if s["priority"] == "medium"]),
                "low": len([s for s in suggestions if s["priority"] == "low"])
            },
            "by_category": {
                cat: {
                    "total": data["total"],
                    "by_priority": {
                        p: len(suggestions)
                        for p, suggestions in data["by_priority"].items()
                    }
                }
                for cat, data in grouped_suggestions.items()
            }
        }
        
        return success_response({
            "suggestions": suggestions[:limit],
            "statistics": stats,
            "filters": {
                "language": language,
                "min_quality": min_quality,
                "max_quality": max_quality,
                "category": category,
                "priority": priority
            },
            "total_words_analyzed": len(words)
        })
    except Exception as e:
        logger.error(f"Error in get_quality_suggestions: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve quality suggestions")

@bp.route("/api/v2/baybayin/process", methods=["POST"])
@validate_query_params(BaybayinQuerySchema)
def process_baybayin_text(**params):
    """
    Process Baybayin text - either romanize or transliterate.
    
    Query Parameters:
        text (str): The text to process
        mode (str): Either 'romanize' or 'transliterate'
        validate (bool): Whether to validate the text
        include_metadata (bool): Include metadata in response
        include_pronunciation (bool): Include pronunciation data
    
    Returns:
        Processed text with metadata
    """
    try:
        text = params.get('text')
        mode = params.get('mode', 'romanize')
        validate_text = params.get('validate', True)
        include_metadata = params.get('include_metadata', False)
        include_pronunciation = params.get('include_pronunciation', False)
        
        # Process the text
        if mode == 'romanize':
            romanizer = BaybayinRomanizer()
            if validate_text and not romanizer.validate_text(text):
                return error_response("Invalid Baybayin text", 400)
            result = romanizer.romanize(text)
            processed_text = result
        else:
            processed_text = transliterate_to_baybayin(text)
        
        # Prepare response
        response_data = {
            "original_text": text,
            "processed_text": processed_text,
            "mode": mode
        }
        
        if include_metadata:
            response_data["metadata"] = {
                "has_baybayin": True if mode == 'transliterate' else romanizer.is_baybayin(text),
                "character_count": len(text),
                "processed_character_count": len(processed_text)
            }
            
        if include_pronunciation and mode == 'romanize':
            response_data["pronunciation"] = {
                "ipa": get_pronunciation(processed_text),
                "respelling": get_respelling(processed_text)
            }
            
        return success_response(response_data)
        
    except Exception as e:
        logger.error(f"Error processing Baybayin text: {str(e)}", exc_info=True)
        return error_response("Failed to process Baybayin text")

@bp.route("/api/v2/words/<path:word>/etymology", methods=["GET"])
@cached(prefix="word_etymology", ttl=3600)
@validate_query_params(EtymologyQuerySchema)
def get_word_etymology(word, **params):
    """
    Get detailed etymology information for a word.
    
    Path Parameters:
        word (str): The word to get etymology for
        
    Query Parameters:
        language_codes (List[str]): Filter by language codes
        include_components (bool): Include etymology components
        include_structure (bool): Include etymology structure
        include_confidence (bool): Include confidence scores
        include_source_info (bool): Include source information
    
    Returns:
        Etymology details for the word
    """
    try:
        # Extract parameters
        language_codes = params.get('language_codes')
        include_components = params.get('include_components', True)
        include_structure = params.get('include_structure', False)
        include_confidence = params.get('include_confidence', False)
        include_source_info = params.get('include_source_info', False)
        
        # Get word
        word_obj = Word.query.filter(
            func.lower(Word.lemma) == word.lower()
        ).first()
        
        if not word_obj:
            return error_response("Word not found", 404)
            
        # Get etymologies
        etymologies = Etymology.query.filter_by(word_id=word_obj.id)
        if language_codes:
            etymologies = etymologies.filter(
                Etymology.language_codes.op('&&')(language_codes)
            )
            
        etymology_data = []
        for etym in etymologies:
            etym_entry = {
                "etymology_text": etym.etymology_text,
                "language_codes": etym.language_codes.split(',') if etym.language_codes else []
            }
            
            if include_components and etym.normalized_components:
                try:
                    etym_entry["components"] = json.loads(etym.normalized_components)
                except json.JSONDecodeError:
                    etym_entry["components"] = [
                        {"text": comp.strip()} 
                        for comp in etym.normalized_components.split(';')
                        if comp.strip()
                    ]
                    
            if include_structure and etym.etymology_structure:
                try:
                    etym_entry["structure"] = json.loads(etym.etymology_structure)
                except json.JSONDecodeError:
                    etym_entry["structure"] = None
                    
            if include_confidence and etym.etymology_structure:
                try:
                    structure = json.loads(etym.etymology_structure)
                    etym_entry["confidence"] = structure.get('confidence')
                except json.JSONDecodeError:
                    etym_entry["confidence"] = None
                    
            if include_source_info:
                etym_entry["sources"] = etym.sources.split(', ') if etym.sources else []
                
            etymology_data.append(etym_entry)
            
        return success_response({
            "word": word_obj.lemma,
            "etymologies": etymology_data
        })
        
    except Exception as e:
        logger.error(f"Error getting etymology: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve etymology")

@bp.route("/api/v2/words/<path:word>/pronunciation", methods=["GET"])
@cached(prefix="word_pronunciation", ttl=3600)
@validate_query_params(PronunciationQuerySchema)
def get_word_pronunciation(word, **params):
    """
    Get pronunciation information for a word.
    
    Path Parameters:
        word (str): The word to get pronunciation for
        
    Query Parameters:
        type (str): Pronunciation type (ipa, respelling, audio)
        include_variants (bool): Include pronunciation variants
        include_phonemes (bool): Include phoneme breakdown
        include_stress (bool): Include stress pattern
        include_source_info (bool): Include source information
    
    Returns:
        Pronunciation details for the word
    """
    try:
        # Extract parameters
        pron_type = params.get('type', 'ipa')
        include_variants = params.get('include_variants', False)
        include_phonemes = params.get('include_phonemes', False)
        include_stress = params.get('include_stress', False)
        include_source_info = params.get('include_source_info', False)
        
        # Get word
        word_obj = Word.query.filter(
            func.lower(Word.lemma) == word.lower()
        ).first()
        
        if not word_obj:
            return error_response("Word not found", 404)
            
        # Get pronunciations
        pronunciations = Pronunciation.query.filter_by(word_id=word_obj.id)
        
        pronunciation_data = []
        for pron in pronunciations:
            try:
                pron_info = json.loads(pron.value) if isinstance(pron.value, str) else pron.value
            except json.JSONDecodeError:
                pron_info = {"value": pron.value}
                
            pron_entry = {
                "type": pron.type,
                "value": pron_info.get("value", pron.value)
            }
            
            if include_variants and "variants" in pron_info:
                pron_entry["variants"] = pron_info["variants"]
                
            if include_phonemes and "phonemes" in pron_info:
                pron_entry["phonemes"] = pron_info["phonemes"]
                
            if include_stress and "stress_pattern" in pron_info:
                pron_entry["stress_pattern"] = pron_info["stress_pattern"]
                
            if include_source_info:
                pron_entry["sources"] = pron.sources.split(', ') if pron.sources else []
                
            pronunciation_data.append(pron_entry)
            
        return success_response({
            "word": word_obj.lemma,
            "pronunciations": pronunciation_data
        })
        
    except Exception as e:
        logger.error(f"Error getting pronunciation: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve pronunciation")

# Create and configure the Flask application instance if running directly
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    app.register_blueprint(bp)
    
    # Initialize rate limiter
    with app.app_context():
        init_rate_limiter(app)
    
    # Run the application for development (not for production)
    app.run(host="127.0.0.1", port=8000, debug=True)