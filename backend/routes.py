from flask import Blueprint, jsonify, request, current_app
from sqlalchemy.orm import joinedload, selectinload, load_only
from sqlalchemy import or_, func, desc
from backend.models import Word, Definition, Etymology, Relation, DefinitionRelation, Affixation, PartOfSpeech, db
from datetime import datetime
from unidecode import unidecode
from functools import lru_cache, wraps
import re
from backend.caching import multi_level_cache
from urllib.parse import unquote
from fuzzywuzzy import fuzz, process
from sqlalchemy.schema import Index
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, validate
import logging

bp = Blueprint("api", __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(
    app=current_app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@lru_cache(maxsize=1000)
def normalize_word(word):
    return unidecode(str(word).lower()) if word else None

def get_word_details(word_entry):
    """Get comprehensive word details including all relationships and metadata."""
    if not word_entry:
        return None

    # Process etymologies with components and language codes
    etymologies = []
    for etym in word_entry.etymologies:
        etymology_data = {
            "text": etym.etymology_text,
            "normalized_components": etym.normalized_components.split(", ") if etym.normalized_components else [],
            "language_codes": etym.language_codes.split(", ") if etym.language_codes else [],
            "sources": etym.sources
        }
        etymologies.append(etymology_data)

    # Process definitions with POS and relations
    definitions = []
    for defn in word_entry.definitions:
        definition_data = {
            "text": defn.definition_text,
            "part_of_speech": {
                "code": defn.part_of_speech.code if defn.part_of_speech else None,
                "name_en": defn.part_of_speech.name_en if defn.part_of_speech else None,
                "name_tl": defn.part_of_speech.name_tl if defn.part_of_speech else None
            } if defn.part_of_speech else None,
            "examples": defn.examples.split("\n") if defn.examples else [],
            "usage_notes": defn.usage_notes.split("; ") if defn.usage_notes else [],
            "sources": defn.sources,
            "relations": [
                {
                    "word": rel.word.lemma,
                    "type": rel.relation_type,
                    "sources": rel.sources
                }
                for rel in defn.definition_relations
            ]
        }
        definitions.append(definition_data)

    # Process all word relations
    relations = {
        "synonyms": [],
        "antonyms": [],
        "related": [],
        "derived": [],
        "root": None
    }

    for rel in word_entry.relations_from:
        if rel.relation_type == "synonym":
            relations["synonyms"].append({"word": rel.to_word.lemma, "sources": rel.sources})
        elif rel.relation_type == "antonym":
            relations["antonyms"].append({"word": rel.to_word.lemma, "sources": rel.sources})
        elif rel.relation_type == "related":
            relations["related"].append({"word": rel.to_word.lemma, "sources": rel.sources})
        elif rel.relation_type == "derived_from":
            relations["root"] = {"word": rel.to_word.lemma, "sources": rel.sources}

    for rel in word_entry.relations_to:
        if rel.relation_type == "derived_from":
            relations["derived"].append({"word": rel.from_word.lemma, "sources": rel.sources})

    # Process affixations
    affixations = {
        "as_root": [
            {
                "affixed_word": aff.affixed_word.lemma,
                "type": aff.affix_type,
                "sources": aff.sources
            }
            for aff in word_entry.affixations_as_root
        ],
        "as_affixed": [
            {
                "root_word": aff.root_word.lemma,
                "type": aff.affix_type,
                "sources": aff.sources
            }
            for aff in word_entry.affixations_as_affixed
        ]
    }

    return {
        "meta": {
            "version": "2.0",
            "word": word_entry.lemma,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        "data": {
            "word": word_entry.lemma,
            "normalized_lemma": word_entry.normalized_lemma,
            "language_code": word_entry.language_code,
            "baybayin": {
                "has_baybayin": word_entry.has_baybayin,
                "form": word_entry.baybayin_form,
                "romanized": word_entry.romanized_form
            } if word_entry.has_baybayin else None,
            "pronunciation": word_entry.pronunciation_data,
            "preferred_spelling": word_entry.preferred_spelling,
            "tags": word_entry.tags.split(", ") if word_entry.tags else [],
            "idioms": word_entry.idioms,
            "etymologies": etymologies,
            "definitions": definitions,
            "relations": relations,
            "affixations": affixations,
            "source_info": word_entry.source_info
        }
    }

@bp.route("/api/v1/words", methods=["GET"])
def get_words():
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    search = request.args.get("search", "")
    exclude_baybayin = request.args.get("exclude_baybayin", "true").lower() == "true"
    pos_filter = request.args.get("pos")
    source_filter = request.args.get("source")

    query = Word.query.options(
        joinedload(Word.definitions).joinedload(Definition.part_of_speech)
    )

    if exclude_baybayin:
        query = query.filter(Word.has_baybayin == False)

    if search:
        normalized_search = normalize_word(search)
        query = query.filter(
            or_(
                func.lower(func.unaccent(Word.lemma)).like(f"{normalized_search}%"),
                Word.search_text.match(normalized_search)
            )
        )

    if pos_filter:
        query = query.join(Word.definitions).join(Definition.part_of_speech).filter(
            PartOfSpeech.code == pos_filter
        )

    if source_filter:
        query = query.join(Word.definitions).filter(
            Definition.sources.like(f"%{source_filter}%")
        )

    total = query.count()
    words = query.order_by(Word.lemma).offset((page - 1) * per_page).limit(per_page).all()

    return jsonify({
        "words": [{"word": w.lemma, "id": w.id} for w in words],
        "page": page,
        "perPage": per_page,
        "total": total,
    })

def cache_key_with_params(*args, **kwargs):
    """Generate cache key including query parameters."""
    sorted_params = sorted(request.args.items())
    param_str = '&'.join(f"{k}={v}" for k, v in sorted_params)
    return f"{request.path}?{param_str}"

@bp.route("/api/v1/words/<path:word>", methods=["GET"])
@multi_level_cache
def get_word(word):
    try:
        word_entry = get_word_with_relations(word)
        if word_entry is None:
            return error_response("Word not found", 404)

        return jsonify(get_word_details(word_entry))
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word")

def get_word_with_relations(word):
    normalized_word = normalize_word(word)
    
    # Use select_from to optimize join order
    word_entry = Word.query.select_from(Word)\
        .options(
            joinedload(Word.definitions).joinedload(Definition.part_of_speech),
            joinedload(Word.definitions).joinedload(Definition.definition_relations),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word),
            joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
            joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word)
        )\
        .filter(
            Word.normalized_lemma == normalized_word
        )\
        .execution_options(max_row_buffer=100)\
        .first()
    
    return word_entry

@bp.route("/api/v1/check_word/<path:word>", methods=["GET"])
@multi_level_cache
def check_word(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = Word.query.filter(
            func.lower(func.unaccent(Word.normalized_lemma)) == normalized_word
        ).first()
        return jsonify({
            "exists": bool(word_entry),
            "word": word_entry.lemma if word_entry else None,
        })
    except Exception as e:
        logger.error(f"Error in check_word: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route("/api/v1/word_network/<path:word>", methods=["GET"])
@multi_level_cache
def get_word_network(word):
    try:
        depth = min(int(request.args.get("depth", 2)), 5)
        breadth = min(int(request.args.get("breadth", 10)), 20)
        relation_types = request.args.getlist("relation_types")

        if not word:
            return jsonify({"error": "Word not provided"}), 400

        normalized_word = normalize_word(unquote(word))
        network = build_word_network(normalized_word, depth, breadth, relation_types)

        if not network:
            return jsonify({"error": "Word not found"}), 404

        return jsonify(network)
    except Exception as e:
        logger.error(f"Error in get_word_network: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

def build_word_network(word, depth=2, breadth=10, relation_types=None):
    """Build a network of related words."""
    if not relation_types:
        relation_types = ["synonym", "antonym", "derived_from", "related"]

    visited = set()
    queue = [(word, 0)]
    network = {}
    max_network_size = current_app.config.get("MAX_NETWORK_SIZE", 100)

    while queue and len(network) < max_network_size:
        current_word, current_depth = queue.pop(0)
        
        if current_word in visited or current_depth > depth:
            continue

        visited.add(current_word)
        word_entry = Word.query.options(
            joinedload(Word.definitions),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word)
        ).filter(
            func.lower(func.unaccent(Word.normalized_lemma)) == normalize_word(current_word)
        ).first()

        if word_entry:
            # Get first definition
            definition = next((d.definition_text for d in word_entry.definitions), "No definition available")
            
            # Get related words based on relation types
            related_words = {
                "synonyms": [],
                "antonyms": [],
                "derived": [],
                "related": [],
                "root": None
            }

            for rel in word_entry.relations_from:
                if rel.relation_type in relation_types:
                    if rel.relation_type == "derived_from":
                        related_words["root"] = rel.to_word.lemma
                    else:
                        related_words[rel.relation_type + "s"].append(rel.to_word.lemma)

            for rel in word_entry.relations_to:
                if rel.relation_type == "derived_from" and "derived_from" in relation_types:
                    related_words["derived"].append(rel.from_word.lemma)

            network[current_word] = {
                "word": word_entry.lemma,
                "definition": definition,
                **related_words
            }

            if current_depth < depth:
                new_words = set()
                for rel_words in related_words.values():
                    if isinstance(rel_words, list):
                        new_words.update(rel_words)
                    elif rel_words:  # For root word
                        new_words.add(rel_words)

                new_words = list(new_words - visited)[:breadth]
                queue.extend((w, current_depth + 1) for w in new_words)

    return network

@bp.route("/api/v1/etymology/<path:word>", methods=["GET"])
@multi_level_cache
def get_etymology(word):
    try:
        normalized_word = normalize_word(word)
        word_entry = Word.query.options(
            joinedload(Word.etymologies)
        ).filter(
            func.lower(func.unaccent(Word.normalized_lemma)) == normalized_word
        ).first()
        
        if not word_entry:
            return jsonify({"error": "Word not found"}), 404

        etymologies = [
            {
                "text": etym.etymology_text,
                "normalized_components": etym.normalized_components.split(", ") if etym.normalized_components else [],
                "language_codes": etym.language_codes.split(", ") if etym.language_codes else [],
                "sources": etym.sources
            }
            for etym in word_entry.etymologies
        ]

        return jsonify({
            "word": word_entry.lemma,
            "etymologies": etymologies
        })
    except Exception as e:
        logger.error(f"Error in get_etymology: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route("/api/v1/bulk_words", methods=["POST"])
def bulk_get_words():
    try:
        words = request.json.get("words", [])
        if not words or not isinstance(words, list):
            return jsonify({"error": "Invalid input"}), 400

        normalized_words = [normalize_word(w) for w in words]
        word_entries = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.part_of_speech),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from),
            joinedload(Word.relations_to),
            joinedload(Word.affixations_as_root),
            joinedload(Word.affixations_as_affixed)
        ).filter(
            func.lower(func.unaccent(Word.normalized_lemma)).in_(normalized_words)
        ).all()

        return jsonify({
            "words": [get_word_details(w) for w in word_entries]
        })
    except Exception as e:
        logger.error(f"Error in bulk_get_words: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

class SearchParamsSchema(Schema):
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(load_default=10, validate=validate.Range(min=1, max=50))
    min_similarity = fields.Float(load_default=0.3, validate=validate.Range(min=0, max=1))

def validate_request(schema_class):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            schema = schema_class()
            errors = schema.validate(request.args)
            if errors:
                return error_response({"validation_errors": errors}, 400)
            return f(*args, **kwargs)
        return wrapper
    return decorator

@bp.route("/api/v1/search", methods=["GET"])
@validate_request(SearchParamsSchema)
@limiter.limit("20 per minute")
def search_words():
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 50)
    min_similarity = float(request.args.get("min_similarity", 0.3))
    
    if not query:
        return error_response("No search query provided", 400)

    try:
        normalized_query = normalize_word(query)
        tsquery = func.plainto_tsquery('english', normalized_query)
        
        # Use CTE for better performance
        results = db.session.query(
            Word.id,
            Word.lemma,
            Word.normalized_lemma,
            func.ts_rank(Word.search_text, tsquery).label('rank'),
            func.similarity(Word.normalized_lemma, normalized_query).label('similarity')
        ).cte('search_results')
        
        final_results = db.session.query(results)\
            .filter(
                or_(
                    results.c.rank > 0,
                    results.c.similarity > min_similarity
                )
            )\
            .order_by(
                desc(results.c.rank),
                desc(results.c.similarity)
            )\
            .limit(limit)\
            .all()

        return jsonify({
            "results": [
                {
                    "word": r.lemma,
                    "id": r.id,
                    "score": max(r.rank, r.similarity)
                } for r in final_results
            ]
        })
    except Exception as e:
        logger.error(f"Error in search_words: {str(e)}", exc_info=True)
        return error_response("Search failed")

@bp.route("/api/v1/pos", methods=["GET"])
@multi_level_cache
def get_parts_of_speech():
    try:
        pos_entries = PartOfSpeech.query.all()
        return jsonify({
            "parts_of_speech": [
                {
                    "code": pos.code,
                    "name_en": pos.name_en,
                    "name_tl": pos.name_tl,
                    "description": pos.description
                }
                for pos in pos_entries
            ]
        })
    except Exception as e:
        logger.error(f"Error in get_parts_of_speech: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

@bp.route("/favicon.ico")
def favicon():
    return "", 204

@bp.teardown_request
def remove_session(exception=None):
    db.session.remove()

# Add composite indexes for better query performance
__table_args__ = (
    Index('idx_words_normalized_lang', 'normalized_lemma', 'language_code'),
    Index('idx_words_search_text', 'search_text', postgresql_using='gin'),
)

def error_response(message, status_code=500):
    """Standardized error response."""
    return jsonify({
        "error": {
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }), status_code

@bp.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return error_response("An unexpected error occurred")
