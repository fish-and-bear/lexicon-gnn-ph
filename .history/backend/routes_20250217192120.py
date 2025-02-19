"""
API routes for the dictionary application.
"""

from flask import Blueprint, jsonify, request, current_app
from sqlalchemy.orm import joinedload, selectinload, load_only
from sqlalchemy import or_, func, desc, text
from backend.models import Word, Definition, Etymology, Relation, DefinitionRelation, Affixation, PartOfSpeech, db
from datetime import datetime, UTC
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
import json
import traceback
from marshmallow.exceptions import ValidationError
from sqlalchemy import event
from backend.language_utils import language_system
from backend.source_standardization import extract_etymology_components, extract_meaning

bp = Blueprint("api", __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize limiter without app context
limiter = None

def init_limiter(app):
    """Initialize the limiter with the app instance"""
    global limiter
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri="memory://",
        strategy="fixed-window",
        headers_enabled=True,
        retry_after="http-date",
        key_prefix="api_v2"
    )
    limiter.init_app(app)

def error_response(message: str, status_code: int = 500, errors: dict = None) -> tuple:
    """Generate consistent error responses with optional validation errors"""
    response = {
        "error": {
            "message": message,
            "status_code": status_code,
            "timestamp": datetime.now(UTC).isoformat()
        }
    }
    if errors:
        response["error"]["details"] = errors
    return jsonify(response), status_code

@lru_cache(maxsize=1000)
def normalize_word(word: str) -> str:
    """Normalize word for consistent lookup."""
    if not word:
        return None
    # Remove diacritics and convert to lowercase
    normalized = word.lower()
    normalized = re.sub(r'[^\w\s\-]', '', normalized)
    return normalized.strip()

def get_word_details(word_entry, include_definitions=True, include_relations=True, include_etymology=True):
    """Get comprehensive word details including all relationships and metadata."""
    if not word_entry:
        return None

    result = {
        "meta": {
            "version": "2.0",
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "data": {
            "word": word_entry.lemma,
            "normalized_lemma": word_entry.normalized_lemma,
            "language_code": word_entry.language_code,
            "tags": word_entry.tags.split(", ") if word_entry.tags else [],
            "preferred_spelling": word_entry.preferred_spelling,
            "source_info": word_entry.source_info,
            "root_word_id": word_entry.root_word_id,
            "baybayin": {
                "has_baybayin": word_entry.has_baybayin,
                "form": word_entry.baybayin_form,
                "romanized": word_entry.romanized_form
            } if word_entry.has_baybayin else None,
            "pronunciation_data": word_entry.pronunciation_data,
            "idioms": json.loads(word_entry.idioms) if word_entry.idioms else [],
            "created_at": word_entry.created_at.isoformat() if word_entry.created_at else None,
            "updated_at": word_entry.updated_at.isoformat() if word_entry.updated_at else None
        }
    }

    if include_definitions:
        result["data"]["definitions"] = [
            {
                "id": d.id,
                "text": d.definition_text,
                "original_pos": d.original_pos,
                "part_of_speech": {
                    "id": d.part_of_speech.id if d.part_of_speech else None,
                    "code": d.part_of_speech.code if d.part_of_speech else None,
                    "name_en": d.part_of_speech.name_en if d.part_of_speech else None,
                    "name_tl": d.part_of_speech.name_tl if d.part_of_speech else None,
                    "description": d.part_of_speech.description if d.part_of_speech else None
                } if d.part_of_speech else None,
                "examples": d.examples.split("\n") if d.examples else [],
                "usage_notes": d.usage_notes.split("; ") if d.usage_notes else [],
                "sources": d.sources.split(", ") if d.sources else [],
                "definition_relations": [
                    {
                        "word": dr.word.lemma,
                        "type": dr.relation_type,
                        "sources": dr.sources.split(", ") if dr.sources else []
                    }
                    for dr in d.definition_relations
                ],
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            }
            for d in word_entry.definitions
        ]

    if include_etymology and word_entry.etymologies:
        result["data"]["etymologies"] = [
            {
                "id": e.id,
                "text": e.etymology_text,
                "components": e.normalized_components.split(", ") if e.normalized_components else [],
                "language_codes": e.language_codes.split(", ") if e.language_codes else [],
                "sources": e.sources.split(", ") if e.sources else [],
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "updated_at": e.updated_at.isoformat() if e.updated_at else None
            }
            for e in word_entry.etymologies
        ]

    if include_relations:
        relations = {
            "synonyms": [],
            "antonyms": [],
            "root": None,
            "derived": [],
            "related": [],
            "affixations": {
                "as_root": [],
                "as_affixed": []
            }
        }

        # Process word relations
        for rel in word_entry.relations_from:
            rel_data = {
                "id": rel.id,
                "word": rel.to_word.lemma,
                "normalized_word": rel.to_word.normalized_lemma,
                "language_code": rel.to_word.language_code,
                "sources": rel.sources.split(", ") if rel.sources else [],
                "created_at": rel.created_at.isoformat() if rel.created_at else None
            }
            if rel.relation_type == "synonym":
                relations["synonyms"].append(rel_data)
            elif rel.relation_type == "antonym":
                relations["antonyms"].append(rel_data)
            elif rel.relation_type == "derived_from":
                relations["root"] = rel_data
            else:
                relations["related"].append(rel_data)

        for rel in word_entry.relations_to:
            if rel.relation_type == "derived_from":
                relations["derived"].append({
                    "id": rel.id,
                    "word": rel.from_word.lemma,
                    "normalized_word": rel.from_word.normalized_lemma,
                    "language_code": rel.from_word.language_code,
                    "sources": rel.sources.split(", ") if rel.sources else [],
                    "created_at": rel.created_at.isoformat() if rel.created_at else None
                })

        # Process affixations
        for aff in word_entry.affixations_as_root:
            relations["affixations"]["as_root"].append({
                "id": aff.id,
                "affixed_word": aff.affixed_word.lemma,
                "normalized_word": aff.affixed_word.normalized_lemma,
                "type": aff.affix_type,
                "sources": aff.sources.split(", ") if aff.sources else [],
                "created_at": aff.created_at.isoformat() if aff.created_at else None
            })

        for aff in word_entry.affixations_as_affixed:
            relations["affixations"]["as_affixed"].append({
                "id": aff.id,
                "root_word": aff.root_word.lemma,
                "normalized_word": aff.root_word.normalized_lemma,
                "type": aff.affix_type,
                "sources": aff.sources.split(", ") if aff.sources else [],
                "created_at": aff.created_at.isoformat() if aff.created_at else None
            })

        result["data"]["relations"] = relations

    return result

@bp.route("/api/v2/words", methods=["GET"])
@multi_level_cache
def get_words():
    """Get a paginated list of words with optional filtering."""
    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(int(request.args.get("per_page", 20)), 100)
        search = request.args.get("search", "").strip()
        language = request.args.get("language", "tl")
        pos = request.args.get("pos")
        has_baybayin = request.args.get("has_baybayin", type=bool)
        has_etymology = request.args.get("has_etymology", type=bool)

        query = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.part_of_speech)
        )

        # Apply filters
        if search:
            normalized_search = normalize_word(search)
            query = query.filter(
                or_(
                    Word.normalized_lemma.like(f"{normalized_search}%"),
                    Word.search_text.match(normalized_search)
                )
            )

        if language:
            query = query.filter(Word.language_code == language)

        if pos:
            query = query.join(Word.definitions).join(Definition.part_of_speech).filter(
                PartOfSpeech.code == pos
            )

        if has_baybayin is not None:
            query = query.filter(Word.has_baybayin == has_baybayin)

        if has_etymology:
            query = query.join(Word.etymologies)

        # Get total count and paginated results
        total = query.count()
        words = query.order_by(Word.lemma).offset((page - 1) * per_page).limit(per_page).all()

        return jsonify({
            "meta": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page
            },
            "data": [
                {
                    "word": w.lemma,
                    "language": w.language_code,
                    "has_baybayin": w.has_baybayin,
                    "pos_list": list(set(d.part_of_speech.code for d in w.definitions if d.part_of_speech))
                }
                for w in words
            ]
        })

    except Exception as e:
        logger.error(f"Error in get_words: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve words")

# Request Validation Schemas
class WordQuerySchema(Schema):
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)

class SearchQuerySchema(Schema):
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=50), default=10)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'part', 'conj', 'prep', 'pron', 'det', 'int']))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'phonetic', 'baybayin']), default='all')
    sort = fields.Str(validate=validate.OneOf(['relevance', 'alphabetical', 'created', 'updated']), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')

class PaginationSchema(Schema):
    page = fields.Int(validate=validate.Range(min=1), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    
def validate_request(schema_class):
    """Decorator for request validation"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            schema = schema_class()
            try:
                params = schema.load(request.args)
                return f(*args, **{**kwargs, **params})
            except ValidationError as err:
                return error_response(f"Invalid parameters: {err.messages}", 400)
        return wrapper
    return decorator

@bp.route("/api/v2/words/<path:word>", methods=["GET"])
@multi_level_cache
@validate_request(WordQuerySchema)
def get_word(word, **params):
    """Get detailed information about a specific word."""
    try:
        word_entry = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.part_of_speech),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word),
            joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
            joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word)
        ).filter(
            Word.normalized_lemma == normalize_word(word),
            Word.language_code == params["language_code"]
        ).first()
        
        if not word_entry:
            similar_words = Word.query.filter(
                Word.language_code == params["language_code"],
                func.similarity(Word.normalized_lemma, normalize_word(word)) > 0.3
            ).order_by(
                func.similarity(Word.normalized_lemma, normalize_word(word)).desc()
            ).limit(5).all()

            return error_response(
                "Word not found", 
                404, 
                {"suggestions": [w.lemma for w in similar_words] if similar_words else None}
            )

        return jsonify(get_word_details(
            word_entry,
            include_definitions=params["include_definitions"],
            include_relations=params["include_relations"],
            include_etymology=params["include_etymology"],
            include_metadata=params["include_metadata"]
        ))

    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word details")

@bp.route("/api/v2/search", methods=["GET"])
@multi_level_cache
@validate_request(SearchQuerySchema)
def search_words(**params):
    """Search words with advanced filtering and sorting."""
    try:
        # Build advanced search query with enhanced sorting
        sql = text("""
            WITH search_results AS (
                SELECT 
                    w.id,
                    w.lemma,
                    w.normalized_lemma,
                    w.language_code,
                    w.has_baybayin,
                    w.baybayin_form,
                    w.romanized_form,
                    w.preferred_spelling,
                    w.created_at,
                    w.updated_at,
                    w.source_info,
                    GREATEST(
                        similarity(w.normalized_lemma, :normalized_query),
                        similarity(w.lemma, :query),
                        CASE WHEN w.has_baybayin AND :include_baybayin 
                             THEN similarity(w.baybayin_form, :query)
                             ELSE 0 
                        END,
                        ts_rank(w.search_text, plainto_tsquery('simple', :query)) * 0.8,
                        CASE WHEN metaphone(w.normalized_lemma, 10) = metaphone(:normalized_query, 10)
                             THEN 0.7
                             ELSE 0
                        END,
                        CASE WHEN w.preferred_spelling = :query THEN 1.0 ELSE 0 END
                    ) as relevance,
                    EXISTS(SELECT 1 FROM etymologies e WHERE e.word_id = w.id) as has_etymology,
                    EXISTS(SELECT 1 FROM relations r WHERE r.from_word_id = w.id OR r.to_word_id = w.id) as has_relations
                FROM words w
                WHERE w.language_code = :language
                AND (
                    CASE 
                        WHEN :mode = 'exact' THEN w.normalized_lemma = :normalized_query
                        WHEN :mode = 'phonetic' THEN metaphone(w.normalized_lemma, 10) = metaphone(:normalized_query, 10)
                        WHEN :mode = 'baybayin' THEN w.has_baybayin = true AND w.baybayin_form LIKE :query || '%'
                        ELSE (
                            similarity(w.normalized_lemma, :normalized_query) > :min_similarity
                            OR w.search_text @@ plainto_tsquery('simple', :query)
                            OR metaphone(w.normalized_lemma, 10) = metaphone(:normalized_query, 10)
                            OR (w.has_baybayin AND w.baybayin_form LIKE :query || '%')
                            OR w.preferred_spelling = :query
                        )
                    END
                )
            )
            SELECT 
                sr.*,
                array_agg(DISTINCT p.code) as pos_codes,
                array_agg(DISTINCT d.definition_text) as definitions,
                array_agg(DISTINCT d.examples) FILTER (WHERE d.examples IS NOT NULL) as examples,
                COUNT(DISTINCT d.id) as definition_count,
                COUNT(DISTINCT e.id) as etymology_count,
                COUNT(DISTINCT r.id) as relation_count
            FROM search_results sr
            LEFT JOIN definitions d ON sr.id = d.word_id
            LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
            LEFT JOIN etymologies e ON sr.id = e.word_id
            LEFT JOIN relations r ON sr.id = r.from_word_id OR sr.id = r.to_word_id
            WHERE sr.relevance > :min_similarity
            GROUP BY 
                sr.id, sr.lemma, sr.normalized_lemma, sr.language_code,
                sr.has_baybayin, sr.baybayin_form, sr.romanized_form,
                sr.preferred_spelling, sr.relevance, sr.created_at, sr.updated_at,
                sr.source_info, sr.has_etymology, sr.has_relations
            ORDER BY 
                CASE 
                    WHEN :sort = 'relevance' THEN sr.relevance
                    WHEN :sort = 'alphabetical' THEN sr.lemma
                    WHEN :sort = 'created' THEN sr.created_at::text
                    WHEN :sort = 'updated' THEN sr.updated_at::text
                END DESC
            LIMIT :limit
        """)

        normalized_query = normalize_word(params["q"])
        results = db.session.execute(
            sql,
            {
                "query": params["q"],
                "normalized_query": normalized_query,
                "language": params["language"],
                "min_similarity": params["min_similarity"],
                "limit": params["limit"],
                "mode": params["mode"],
                "include_baybayin": params["include_baybayin"],
                "sort": params["sort"]
            }
        ).fetchall()

        return jsonify({
            "meta": {
                "query": params["q"],
                "normalized_query": normalized_query,
                "mode": params["mode"],
                "total": len(results),
                "params": {k: v for k, v in params.items() if k != 'q'}
            },
            "data": [
                {
                    "word": row.lemma,
                    "normalized_lemma": row.normalized_lemma,
                    "language": row.language_code,
                    "has_baybayin": row.has_baybayin,
                    "baybayin_form": row.baybayin_form if row.has_baybayin else None,
                    "romanized_form": row.romanized_form if row.has_baybayin else None,
                    "preferred_spelling": row.preferred_spelling,
                    "source_info": row.source_info,
                    "parts_of_speech": row.pos_codes if row.pos_codes[0] else [],
                    "definitions": row.definitions if row.definitions[0] else [],
                    "examples": row.examples if row.examples and row.examples[0] else [],
                    "counts": {
                        "definitions": row.definition_count,
                        "etymologies": row.etymology_count,
                        "relations": row.relation_count
                    },
                    "has_etymology": row.has_etymology,
                    "has_relations": row.has_relations,
                    "relevance": float(row.relevance),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None
                }
                for row in results
            ]
        })

    except Exception as e:
        logger.error(f"Error in search_words: {str(e)}", exc_info=True)
        return error_response("Failed to perform search")

@bp.route("/api/v2/parts-of-speech", methods=["GET"])
@multi_level_cache
def get_parts_of_speech():
    """Get all available parts of speech."""
    try:
        pos_list = PartOfSpeech.query.order_by(PartOfSpeech.code).all()
        return jsonify({
            "data": [
                {
                    "code": pos.code,
                    "name_en": pos.name_en,
                    "name_tl": pos.name_tl,
                    "description": pos.description
                }
                for pos in pos_list
            ]
        })
    except Exception as e:
        logger.error(f"Error in get_parts_of_speech: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve parts of speech")

@bp.route("/api/v2/statistics", methods=["GET"])
@multi_level_cache
def get_statistics():
    """Get comprehensive dictionary statistics."""
    try:
        stats = db.session.execute(text("""
            WITH pos_stats AS (
                SELECT 
                    p.code,
                    p.name_tl,
                    COUNT(DISTINCT d.word_id) as word_count,
                    COUNT(d.id) as definition_count
                FROM parts_of_speech p
                LEFT JOIN definitions d ON p.id = d.standardized_pos_id
                GROUP BY p.id, p.code, p.name_tl
            ),
            relation_stats AS (
                SELECT 
                    relation_type,
                    COUNT(*) as count
                FROM relations
                GROUP BY relation_type
            ),
            language_stats AS (
                SELECT 
                    language_code,
                    COUNT(*) as word_count,
                    COUNT(CASE WHEN has_baybayin THEN 1 END) as baybayin_count
                FROM words
                GROUP BY language_code
            )
            SELECT
                (SELECT COUNT(*) FROM words) as total_words,
                (SELECT COUNT(*) FROM words WHERE has_baybayin = true) as baybayin_words,
                (SELECT COUNT(*) FROM definitions) as total_definitions,
                (SELECT COUNT(*) FROM etymologies) as total_etymologies,
                (SELECT COUNT(*) FROM relations) as total_relations,
                (SELECT COUNT(*) FROM definition_relations) as total_definition_relations,
                (SELECT COUNT(*) FROM affixations) as total_affixations,
                (SELECT json_agg(pos_stats) FROM pos_stats) as pos_distribution,
                (SELECT json_agg(relation_stats) FROM relation_stats) as relation_distribution,
                (SELECT json_agg(language_stats) FROM language_stats) as language_distribution,
                (SELECT COUNT(DISTINCT language_code) FROM words) as language_count,
                (SELECT AVG(array_length(regexp_split_to_array(definition_text, '\s+'), 1))::float 
                 FROM definitions) as avg_definition_length,
                (SELECT COUNT(*) FROM words WHERE idioms IS NOT NULL) as words_with_idioms
        """)).fetchone()

        return jsonify({
            "data": {
                "totals": {
                    "words": stats.total_words,
                    "definitions": stats.total_definitions,
                    "etymologies": stats.total_etymologies,
                    "relations": stats.total_relations,
                    "definition_relations": stats.total_definition_relations,
                    "affixations": stats.total_affixations,
                    "words_with_idioms": stats.words_with_idioms
                },
                "baybayin": {
                    "total": stats.baybayin_words,
                    "percentage": (stats.baybayin_words / stats.total_words * 100) if stats.total_words > 0 else 0
                },
                "languages": {
                    "count": stats.language_count,
                    "distribution": stats.language_distribution
                },
                "parts_of_speech": stats.pos_distribution,
                "relations": stats.relation_distribution,
                "metrics": {
                    "avg_definition_length": stats.avg_definition_length
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve statistics")

@bp.route("/api/v2/random", methods=["GET"])
@multi_level_cache
def get_random_word():
    """Get a random word with optional filters."""
    try:
        language = request.args.get("language", "tl")
        has_baybayin = request.args.get("has_baybayin", type=bool)
        has_etymology = request.args.get("has_etymology", type=bool)

        query = Word.query

        if language:
            query = query.filter(Word.language_code == language)
        if has_baybayin is not None:
            query = query.filter(Word.has_baybayin == has_baybayin)
        if has_etymology:
            query = query.join(Word.etymologies)

        word = query.order_by(func.random()).first()
        if not word:
            return error_response("No words found matching criteria", 404)

        return jsonify(get_word_details(word))
    except Exception as e:
        logger.error(f"Error in get_random_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve random word")

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

# Add more comprehensive validation schemas
class WordRelationshipSchema(Schema):
    depth = fields.Int(validate=validate.Range(min=1, max=3), default=1)
    include_affixes = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    cluster_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)

class EtymologyTreeSchema(Schema):
    max_depth = fields.Int(validate=validate.Range(min=1, max=5), default=3)
    include_uncertain = fields.Bool(default=False)
    group_by_language = fields.Bool(default=False)

# Add performance optimizations
def get_word_with_full_relations(word_normalized: str, language_code: str = "tl"):
    """Efficiently load a word with all its relationships in a single query."""
    return Word.query.options(
        joinedload(Word.definitions).joinedload(Definition.part_of_speech),
        joinedload(Word.etymologies),
        joinedload(Word.relations_from).joinedload(Relation.to_word),
        joinedload(Word.relations_to).joinedload(Relation.from_word),
        joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
        joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word),
        joinedload(Word.definition_relations)
    ).filter(
        Word.normalized_lemma == word_normalized,
        Word.language_code == language_code
    ).first()

@bp.route("/api/v2/words/<path:word>/related", methods=["GET"], endpoint="word_relationships")
@multi_level_cache(ttl=3600)
@validate_request(WordRelationshipSchema)
def get_word_relationships(word, **params):
    """Get detailed relationship graph for a word with optimized loading."""
    try:
        # Check rate limit
        if limiter and not current_app.config.get('TESTING'):
            try:
                key = f"{get_remote_address()}:word_relationships"
                if not limiter.storage.get(key):
                    limiter.storage.incr(key, 1)
                    limiter.storage.expire(key, 60)  # 60 seconds
                elif int(limiter.storage.get(key)) >= 30:  # 30 requests per minute
                    return error_response("Rate limit exceeded. Please try again later.", 429)
                else:
                    limiter.storage.incr(key, 1)
            except Exception as e:
                logger.warning(f"Rate limit check failed: {str(e)}")

        word_entry = get_word_with_full_relations(normalize_word(word))
        if not word_entry:
            return error_response("Word not found", 404)

        graph = {
            "nodes": [],
            "edges": [],
            "clusters": {
                "etymology": [],
                "affixes": [],
                "synonyms": [],
                "variants": []
            }
        }
        
        visited = set()
        
        def add_node(word_obj, level=0):
            if word_obj.id in visited or level > params["depth"]:
                return
            visited.add(word_obj.id)
            
            node = {
                "id": word_obj.id,
                "word": word_obj.lemma,
                "type": "root" if word_obj.id == word_entry.id else "related",
                "has_baybayin": word_obj.has_baybayin,
                "language": word_obj.language_code
            }
            graph["nodes"].append(node)
            
            # Process relations
            for rel in word_obj.relations_from:
                if rel.to_word_id not in visited and level < params["depth"]:
                    add_node(rel.to_word, level + 1)
                    edge = {
                        "source": rel.from_word_id,
                        "target": rel.to_word_id,
                        "type": rel.relation_type,
                        "sources": rel.sources.split(", ") if rel.sources else []
                    }
                    graph["edges"].append(edge)
                    
                    if rel.relation_type == "synonym":
                        graph["clusters"]["synonyms"].append(rel.to_word_id)
                    elif rel.relation_type == "variant":
                        graph["clusters"]["variants"].append(rel.to_word_id)
            
            if params["include_affixes"]:
                for aff in word_obj.affixations_as_root:
                    if aff.affixed_word_id not in visited and level < params["depth"]:
                        add_node(aff.affixed_word, level + 1)
                        edge = {
                            "source": aff.root_word_id,
                            "target": aff.affixed_word_id,
                            "type": f"affix_{aff.affix_type}",
                            "sources": aff.sources.split(", ") if aff.sources else []
                        }
                        graph["edges"].append(edge)
                        graph["clusters"]["affixes"].append(aff.affixed_word_id)
            
            if params["include_etymology"] and word_obj.etymologies:
                for etym in word_obj.etymologies:
                    if etym.normalized_components:
                        components = etym.normalized_components.split(", ")
                        for comp in components:
                            comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                            if comp_word and comp_word.id not in visited and level < params["depth"]:
                                add_node(comp_word, level + 1)
                                edge = {
                                    "source": word_obj.id,
                                    "target": comp_word.id,
                                    "type": "etymology",
                                    "sources": etym.sources.split(", ") if etym.sources else []
                                }
                                graph["edges"].append(edge)
                                graph["clusters"]["etymology"].append(comp_word.id)
        
        add_node(word_entry)
        
        return jsonify({
            "meta": {
                "root_word": word_entry.lemma,
                "depth": params["depth"],
                "total_nodes": len(graph["nodes"]),
                "total_edges": len(graph["edges"])
            },
            "data": graph
        })
        
    except Exception as e:
        logger.error(f"Error in get_word_relationships: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word relationships")

@bp.route("/api/v2/words/<path:word>/etymology-tree", methods=["GET"], endpoint="etymology_tree")
@multi_level_cache(ttl=3600)
@validate_request(EtymologyTreeSchema)
def get_etymology_tree(word, **params):
    """Get the complete etymology tree for a word with optimized loading."""
    try:
        # Check rate limit
        if limiter and not current_app.config.get('TESTING'):
            try:
                key = f"{get_remote_address()}:etymology_tree"
                if not limiter.storage.get(key):
                    limiter.storage.incr(key, 1)
                    limiter.storage.expire(key, 60)  # 60 seconds
                elif int(limiter.storage.get(key)) >= 30:  # 30 requests per minute
                    return error_response("Rate limit exceeded. Please try again later.", 429)
                else:
                    limiter.storage.incr(key, 1)
            except Exception as e:
                logger.warning(f"Rate limit check failed: {str(e)}")

        word_entry = get_word_with_full_relations(normalize_word(word))
        if not word_entry:
            return error_response("Word not found", 404)

        def build_etymology_tree(word_obj, visited=None):
            if visited is None:
                visited = set()
            if word_obj.id in visited:
                return None
            visited.add(word_obj.id)
            
            tree = {
                "word": word_obj.lemma,
                "language": word_obj.language_code,
                "etymologies": [],
                "components": []
            }
            
            for etym in word_obj.etymologies:
                etymology_entry = {
                    "text": etym.etymology_text,
                    "languages": etym.language_codes.split(", ") if etym.language_codes else [],
                    "sources": etym.sources.split(", ") if etym.sources else []
                }
                
                if etym.normalized_components:
                    components = etym.normalized_components.split(", ")
                    for comp in components:
                        comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                        if comp_word:
                            comp_tree = build_etymology_tree(comp_word, visited)
                            if comp_tree:
                                tree["components"].append(comp_tree)
                
                tree["etymologies"].append(etymology_entry)
            
            return tree
        
        etymology_tree = build_etymology_tree(word_entry)
        
        return jsonify({
            "meta": {
                "word": word_entry.lemma,
                "language": word_entry.language_code
            },
            "data": etymology_tree
        })
        
    except Exception as e:
        logger.error(f"Error in get_etymology_tree: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve etymology tree")

# Add index hints for better query planning
@event.listens_for(Word.__table__, 'after_create')
def create_word_indexes(target, connection, **kw):
    """Create additional indexes for better query performance."""
    connection.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_word_relations ON relations (from_word_id, to_word_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_word_affixations ON affixations (root_word_id, affixed_word_id, affix_type);
        CREATE INDEX IF NOT EXISTS idx_word_etymologies ON etymologies (word_id, language_codes);
        CREATE INDEX IF NOT EXISTS idx_word_definitions ON definitions (word_id, standardized_pos_id);
        CREATE INDEX IF NOT EXISTS idx_definition_relations ON definition_relations (definition_id, word_id);
    """))

# Add query optimization hints
def optimize_query(query):
    """Add optimization hints to complex queries."""
    return query.execution_options(
        postgresql_hint="""
            SET enable_seqscan = off;
            SET random_page_cost = 1.1;
            SET effective_cache_size = '4GB';
        """
    )
