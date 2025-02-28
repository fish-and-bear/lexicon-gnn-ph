"""
API routes for the Filipino Dictionary application.
This module provides RESTful endpoints for accessing the dictionary data,
with support for searching, detailed word information, Baybayin script,
etymology relationships, and statistical analysis.
"""

from flask import Flask, Blueprint, jsonify, request, current_app, g
from sqlalchemy.orm import joinedload, selectinload, load_only
from sqlalchemy import or_, func, desc, text, event
from models import Word, Definition, Etymology, Relation, DefinitionRelation, Affixation, PartOfSpeech, db
from datetime import datetime, timezone
from unidecode import unidecode
from functools import lru_cache, wraps
import re
import json
import traceback
import logging
import structlog
import sys
from urllib.parse import unquote
from fuzzywuzzy import fuzz, process
from sqlalchemy.schema import Index
import redis
from marshmallow import Schema, fields, validate
from marshmallow.exceptions import ValidationError
import language_utils as language_utils
from source_standardization import SourceStandardization
from security import validate_json_request, validate_query_params
from caching import multi_level_cache, cache, invalidate_cache_prefix
from typing import Tuple, Optional, Any, List, Dict, Union

# Initialize blueprint
bp = Blueprint("api", __name__)

# Enhanced logging setup
logger = structlog.get_logger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize Redis client
redis_client = None

def init_rate_limiter(app):
    """Initialize Redis client for rate limiting with enhanced error handling."""
    global redis_client
    try:
        redis_url = app.config.get('REDIS_URL', 'redis://localhost:6379/0')
        
        # Normalize the URL for Windows
        if 'redis:6379' in redis_url:
            redis_url = 'redis://localhost:6379/0'
            
        print(f"Attempting to connect to Redis at: {redis_url}")
        
        redis_client = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        redis_client.ping()
        print(f"Successfully connected to Redis rate limiter at {redis_url}")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        print("Rate limiting will be disabled")
        redis_client = None

def check_rate_limit(key: str, limit: int = 30, window: int = 60) -> Tuple[bool, Optional[int]]:
    """
    Check if a request is within the configured rate limit.
    """
    if not redis_client or current_app.config.get('TESTING'):
        return True, None
        
    try:
        pipe = redis_client.pipeline()
        now = datetime.now(timezone.utc).timestamp()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window)
        _, request_count, *_ = pipe.execute()
        remaining = max(0, limit - request_count)
        is_allowed = request_count < limit
        if not is_allowed:
            logger.warning(
                "Rate limit exceeded",
                key=key,
                limit=limit,
                window=window,
                count=request_count
            )
        return is_allowed, remaining
    except redis.RedisError as e:
        logger.error(
            "Redis error during rate limit check",
            error=str(e),
            key=key,
            error_type=type(e).__name__
        )
        return True, None  # Fail open
    except Exception as e:
        logger.error(
            "Unexpected error during rate limit check",
            error=str(e),
            key=key,
            error_type=type(e).__name__
        )
        return True, None  # Fail open

def log_request_info():
    """
    Log detailed request information for debugging and monitoring.
    """
    filtered_headers = {k: v for k, v in request.headers.items()
                        if k.lower() not in ('authorization', 'cookie')}
    logger.info(
        "request_received",
        method=request.method,
        path=request.path,
        query_params=dict(request.args),
        headers=filtered_headers,
        remote_addr=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )

@bp.before_request
def before_request():
    """
    Pre-request processing: record start time, generate request ID, log info and check rate limit.
    """
    g.start_time = datetime.now(timezone.utc)
    g.request_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S') + '-' + str(hash(request.remote_addr))[:4]
    log_request_info()
    if request.endpoint and not request.endpoint.endswith('static'):
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}")
        if not is_allowed:
            logger.warning(
                "rate_limit_exceeded",
                remote_addr=request.remote_addr,
                endpoint=request.endpoint,
                request_id=g.request_id
            )
            return rate_limit_exceeded_response()
        if remaining is not None:
            g.rate_limit_remaining = remaining

@bp.after_request
def after_request(response):
    """
    Post-request processing: log response details and add common headers.
    """
    duration = int((datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.path,
        status_code=response.status_code,
        duration_ms=duration,
        response_size=len(response.get_data()),
        request_id=getattr(g, 'request_id', 'unknown')
    )
    if hasattr(g, 'rate_limit_remaining'):
        response.headers['X-RateLimit-Remaining'] = str(g.rate_limit_remaining)
    response.headers['X-Response-Time'] = str(duration)
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    return response

@bp.errorhandler(Exception)
def handle_exception(e):
    """
    Global exception handler with detailed logging.
    """
    error_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    logger.error(
        "unhandled_exception",
        error_id=error_id,
        error_type=type(e).__name__,
        error_message=str(e),
        traceback=traceback.format_exc(),
        request_method=request.method,
        request_path=request.path,
        request_args=dict(request.args),
        request_headers=dict(request.headers),
        request_id=getattr(g, 'request_id', 'unknown')
    )
    if isinstance(e, ValidationError):
        return error_response("Validation error", 400, e.messages, "ERR_VALIDATION")
    if isinstance(e, redis.RedisError):
        return error_response("Cache service temporarily unavailable", 503, {"error_id": error_id}, "ERR_REDIS")
    return error_response("Internal server error", 500, {"error_id": error_id}, "ERR_INTERNAL")

def error_response(message: str, status_code: int = 500, errors: dict = None, error_code: str = None) -> tuple:
    """
    Generate detailed error responses.
    """
    error_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    response = {
        "error": {
            "message": message,
            "status_code": status_code,
            "code": error_code or f"ERR_{status_code}",
            "request_id": getattr(g, 'request_id', 'unknown'),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.path,
            "method": request.method
        }
    }
    if errors:
        response["error"]["details"] = errors
    if current_app.debug:
        response["error"]["debug_info"] = {
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get('User-Agent'),
            "query_params": dict(request.args)
        }
    headers = {
        'Content-Type': 'application/json',
        'X-Error-Code': error_code or f"ERR_{status_code}",
        'X-Error-ID': error_id
    }
    return jsonify(response), status_code, headers

def success_response(data: Any, message: str = None, meta: dict = None) -> tuple:
    """
    Generate consistent success responses.
    """
    response = {
        "meta": {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(g, 'request_id', 'unknown'),
            **(meta or {})
        },
        "data": data
    }
    if message:
        response["meta"]["message"] = message
    headers = {
        'Content-Type': 'application/json',
        'X-Response-Time': str(int((datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000)) if hasattr(g, 'start_time') else None,
        'X-Request-ID': getattr(g, 'request_id', 'unknown')
    }
    return jsonify(response), 200, headers

@lru_cache(maxsize=1000)
def normalize_word(text: str) -> str:
    """
    Normalize a word for consistent lookup.
    """
    if not text:
        return ""
    normalized = unidecode(text.lower())
    normalized = re.sub(r'[^\w\s\-]', '', normalized)
    return normalized.strip()

def parse_components_field(components_data: str) -> List[str]:
    """
    Parse etymology components from various formats.
    """
    if not components_data:
        return []
    try:
        parsed = json.loads(components_data)
        if isinstance(parsed, list):
            return parsed
        return []
    except json.JSONDecodeError:
        if ';' in components_data:
            return [comp.strip() for comp in components_data.split(';') if comp.strip()]
        elif ',' in components_data:
            return [comp.strip() for comp in components_data.split(',') if comp.strip()]
        return [components_data] if components_data.strip() else []
    except Exception:
        return []

def get_word_details(word_entry, include_definitions=True, include_relations=True, include_etymology=True, include_metadata=True):
    """
    Generate comprehensive word details including relationships and metadata.
    """
    if not word_entry:
        return None

    result = {
        "meta": {
            "version": "2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(g, 'request_id', 'unknown')
        },
        "data": {
            "word": word_entry.lemma,
            "normalized_lemma": word_entry.normalized_lemma,
            "language_code": word_entry.language_code,
            "tags": word_entry.tags.split(",") if word_entry.tags else [],
            "preferred_spelling": word_entry.preferred_spelling,
            "baybayin": {
                "has_baybayin": word_entry.has_baybayin,
                "form": word_entry.baybayin_form,
                "romanized": word_entry.romanized_form
            } if word_entry.has_baybayin else None,
            "source_info": word_entry.source_info,
            "pronunciation_data": word_entry.pronunciation_data,
            "idioms": json.loads(word_entry.idioms) if word_entry.idioms and word_entry.idioms != '[]' else [],
            "created_at": word_entry.created_at.isoformat() if word_entry.created_at else None,
            "updated_at": word_entry.updated_at.isoformat() if word_entry.updated_at else None,
            "root_word_id": word_entry.root_word_id
        }
    }
    if include_metadata:
        result["data"].update({
            "source_info": word_entry.source_info,
            "pronunciation_data": word_entry.pronunciation_data,
            "idioms": json.loads(word_entry.idioms) if word_entry.idioms and word_entry.idioms != '[]' else [],
            "created_at": word_entry.created_at.isoformat() if word_entry.created_at else None,
            "updated_at": word_entry.updated_at.isoformat() if word_entry.updated_at else None,
            "root_word_id": word_entry.root_word_id
        })
    if include_definitions and hasattr(word_entry, 'definitions'):
        result["data"]["definitions"] = []
        for d in word_entry.definitions:
            try:
                examples = json.loads(d.examples) if d.examples else []
                if not isinstance(examples, list):
                    examples = [str(examples)]
            except Exception:
                examples = [line.strip() for line in d.examples.split('\n') if line.strip()] if d.examples else []
            try:
                usage_notes = json.loads(d.usage_notes) if d.usage_notes else []
                if not isinstance(usage_notes, list):
                    usage_notes = [str(usage_notes)]
            except Exception:
                usage_notes = [line.strip() for line in d.usage_notes.split('\n') if line.strip()] if d.usage_notes else []
            pos_data = None
            if hasattr(d, 'standardized_pos') and d.standardized_pos:
                pos_data = {
                    "id": d.standardized_pos.id,
                    "code": d.standardized_pos.code,
                    "name_en": d.standardized_pos.name_en,
                    "name_tl": d.standardized_pos.name_tl,
                    "description": d.standardized_pos.description
                }
            definition_obj = {
                "id": d.id,
                "text": d.definition_text,
                "original_pos": d.original_pos,
                "part_of_speech": pos_data,
                "examples": examples,
                "usage_notes": usage_notes,
                "sources": d.sources.split(", ") if d.sources else [],
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            }
            if hasattr(d, 'definition_relations'):
                definition_obj["definition_relations"] = [
                    {
                        "word": dr.word.lemma,
                        "type": dr.relation_type,
                        "sources": dr.sources.split(", ") if dr.sources else []
                    }
                    for dr in d.definition_relations
                ]
            result["data"]["definitions"].append(definition_obj)
    if include_etymology and hasattr(word_entry, 'etymologies') and word_entry.etymologies:
        result["data"]["etymologies"] = []
        for e in word_entry.etymologies:
            etymology_obj = {
                "id": e.id,
                "text": e.etymology_text,
                "components": parse_components_field(e.normalized_components),
                "language_codes": e.language_codes.split(", ") if e.language_codes else [],
                "sources": e.sources.split(", ") if e.sources else [],
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "updated_at": e.updated_at.isoformat() if e.updated_at else None
            }
            result["data"]["etymologies"].append(etymology_obj)
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
        if hasattr(word_entry, 'relations_from'):
        for rel in word_entry.relations_from:
            rel_data = {
                "id": rel.id,
                "word": rel.to_word.lemma,
                "normalized_word": rel.to_word.normalized_lemma,
                "language_code": rel.to_word.language_code,
                "sources": rel.sources.split(", ") if rel.sources else [],
                "created_at": rel.created_at.isoformat() if rel.created_at else None
            }
                rel_type = rel.relation_type.lower()
                if rel_type == "synonym":
                relations["synonyms"].append(rel_data)
                elif rel_type == "antonym":
                relations["antonyms"].append(rel_data)
                elif rel_type == "derived_from":
                relations["root"] = rel_data
            else:
                relations["related"].append(rel_data)
        if hasattr(word_entry, 'relations_to'):
        for rel in word_entry.relations_to:
                if rel.relation_type.lower() == "derived_from":
                relations["derived"].append({
                    "id": rel.id,
                    "word": rel.from_word.lemma,
                    "normalized_word": rel.from_word.normalized_lemma,
                    "language_code": rel.from_word.language_code,
                    "sources": rel.sources.split(", ") if rel.sources else [],
                    "created_at": rel.created_at.isoformat() if rel.created_at else None
                })
        if hasattr(word_entry, 'affixations_as_root'):
        for aff in word_entry.affixations_as_root:
            relations["affixations"]["as_root"].append({
                "id": aff.id,
                "affixed_word": aff.affixed_word.lemma,
                "normalized_word": aff.affixed_word.normalized_lemma,
                "type": aff.affix_type,
                "sources": aff.sources.split(", ") if aff.sources else [],
                "created_at": aff.created_at.isoformat() if aff.created_at else None
            })
        if hasattr(word_entry, 'affixations_as_affixed'):
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

@bp.route("/", methods=["GET"])
def index():
    """API root endpoint with version and feature information."""
    return jsonify({
        "name": "Filipino Dictionary API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": getattr(g, 'request_id', 'unknown'),
        "features": {
            "baybayin_support": True,
            "etymology_tracking": True,
            "word_relationships": True
        },
        "endpoints": {
            "root": "/",
            "words": "/api/v2/words",
            "word_details": "/api/v2/words/<word>",
            "search": "/api/v2/search",
            "statistics": "/api/v2/statistics",
            "random": "/api/v2/random",
            "baybayin": "/api/v2/baybayin",
            "parts_of_speech": "/api/v2/parts-of-speech",
            "related_words": "/api/v2/words/<word>/related",
            "etymology_tree": "/api/v2/words/<word>/etymology-tree"
        }
    })

@bp.route("/api/v2/words", methods=["GET"])
@multi_level_cache(prefix="words_list")
def get_words():
    """
    Get a paginated list of words with filtering options.
    """
    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(int(request.args.get("per_page", 20)), 100)
        search = request.args.get("search", "").strip()
        language = request.args.get("language", "tl")
        pos = request.args.get("pos")
        has_baybayin = request.args.get("has_baybayin", type=bool)
        has_etymology = request.args.get("has_etymology", type=bool)

        query = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos)
        )
        if search:
            normalized_search = normalize_word(search)
            query = query.filter(
                or_(
                    Word.normalized_lemma.like(f"{normalized_search}%"),
                    Word.search_text.op('@@')(func.plainto_tsquery('simple', normalized_search))
                )
            )
        if language:
            query = query.filter(Word.language_code == language)
        if pos:
            query = query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == pos
            )
        if has_baybayin is not None:
            query = query.filter(Word.has_baybayin == has_baybayin)
        if has_etymology:
            query = query.join(Word.etymologies)
        total = query.count()
        words = query.order_by(Word.lemma).offset((page - 1) * per_page).limit(per_page).all()
        word_list = []
        for w in words:
            pos_list = []
            if hasattr(w, 'definitions'):
                pos_list = list(set(
                    d.standardized_pos.code for d in w.definitions 
                    if hasattr(d, 'standardized_pos') and d.standardized_pos
                ))
            word_list.append({
                    "word": w.lemma,
                    "language": w.language_code,
                    "has_baybayin": w.has_baybayin,
                "baybayin_form": w.baybayin_form if w.has_baybayin else None,
                "romanized_form": w.romanized_form if w.has_baybayin else None,
                "pos_list": pos_list
            })
        return success_response(
            word_list,
            meta={
                "page": page,
                "per_page": per_page,
                "total": total,
                "pages": (total + per_page - 1) // per_page,
                "filters": {
                    "search": search if search else None,
                    "language": language,
                    "pos": pos,
                    "has_baybayin": has_baybayin,
                    "has_etymology": has_etymology
                }
            }
        )
    except Exception as e:
        logger.error(
            "Error in get_words",
            error=str(e),
            traceback=traceback.format_exc(),
            request_id=getattr(g, 'request_id', 'unknown')
        )
        return error_response("Failed to retrieve words")

class WordQuerySchema(Schema):
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)

class SearchQuerySchema(Schema):
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=50), default=10)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix', 'idm', 'col', 'syn', 'ant', 'eng', 'spa', 'tx', 'var', 'unc']))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'phonetic', 'baybayin']), default='all')
    sort = fields.Str(validate=validate.OneOf(['relevance', 'alphabetical', 'created', 'updated']), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')

class PaginationSchema(Schema):
    page = fields.Int(validate=validate.Range(min=1), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    
class WordDetailSchema(Schema):
    include_definitions = fields.Boolean(missing=True)
    include_relations = fields.Boolean(missing=True)
    include_etymology = fields.Boolean(missing=True)
    include_metadata = fields.Boolean(missing=True)

@bp.route("/api/v2/words/<path:word>", methods=["GET"])
@multi_level_cache(prefix="word_detail")
@validate_query_params(WordDetailSchema)
def get_word(word, **params):
    """Get detailed information about a specific word."""
    try:
        include_definitions = params.get('include_definitions', True)
        include_relations = params.get('include_relations', True)
        include_etymology = params.get('include_etymology', True)
        language_code = request.args.get('language_code', 'tl')
        
        if language_code not in ['tl', 'ceb']:
            language_code = 'tl'
            
        cache_key = f"word:{word}:{language_code}:{include_definitions}:{include_relations}:{include_etymology}"
        cached_response = cache.get(cache_key)
        if cached_response:
            return cached_response

        query_options = []
        if include_definitions:
            query_options.append(joinedload(Word.definitions).joinedload(Definition.standardized_pos))
        if include_etymology:
            query_options.append(joinedload(Word.etymologies))
        if include_relations:
            query_options.append(joinedload(Word.relations_from).joinedload(Relation.to_word))
            query_options.append(joinedload(Word.relations_to).joinedload(Relation.from_word))
            query_options.append(joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word))
            query_options.append(joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word))
            
        word_entry = Word.query.options(*query_options).filter(
            Word.normalized_lemma == normalize_word(word),
            Word.language_code == language_code
        ).first()
        
        if not word_entry:
            similar_words = Word.query.filter(
                Word.language_code == language_code,
                func.similarity(Word.normalized_lemma, normalize_word(word)) > 0.3
            ).order_by(
                func.similarity(Word.normalized_lemma, normalize_word(word)).desc()
            ).limit(5).all()

            response = error_response(
                "Word not found", 
                404, 
                {"suggestions": [w.lemma for w in similar_words] if similar_words else None}
            )
            cache.set(cache_key, response, timeout=300)
            return response

        word_details = get_word_details(
            word_entry,
            include_definitions=include_definitions,
            include_relations=include_relations,
            include_etymology=include_etymology
        )
        
        response = success_response(word_details["data"], meta=word_details["meta"])
        cache.set(cache_key, response, timeout=3600)
        return response

    except ValidationError as err:
        return error_response("Invalid parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word details")

@bp.route("/api/v2/search", methods=["GET"])
@multi_level_cache(prefix="search", ttl=300)
@validate_query_params(SearchQuerySchema)
def search_words():
    """Search words with advanced filtering and sorting."""
    try:
        params = SearchQuerySchema().load(request.args)
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:search")
        if not is_allowed:
            return rate_limit_exceeded_response()
        
        sql = text("""
            WITH RECURSIVE search_results AS (
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
                    w.tags,
                    GREATEST(
                        similarity(w.normalized_lemma, :normalized_query),
                        similarity(w.lemma, :query),
                        CASE WHEN w.has_baybayin AND :include_baybayin 
                             THEN similarity(w.baybayin_form, :query)
                             ELSE 0 
                        END,
                        ts_rank_cd(
                            w.search_text,
                            plainto_tsquery('simple', :query)
                        ) * 0.8,
                        CASE WHEN metaphone(w.normalized_lemma, 10) = metaphone(:normalized_query, 10)
                             THEN 0.7
                             ELSE 0
                        END,
                        CASE WHEN w.preferred_spelling = :query THEN 1.0 ELSE 0 END
                    ) as relevance,
                    EXISTS(SELECT 1 FROM etymologies e WHERE e.word_id = w.id) as has_etymology,
                    EXISTS(SELECT 1 FROM relations r WHERE r.from_word_id = w.id OR r.to_word_id = w.id) as has_relations,
                    array_agg(DISTINCT d.definition_text) OVER (PARTITION BY w.id) as definitions,
                    array_agg(DISTINCT p.code) OVER (PARTITION BY w.id) as pos_codes,
                    COUNT(DISTINCT d.id) OVER (PARTITION BY w.id) as definition_count,
                    COUNT(DISTINCT e.id) OVER (PARTITION BY w.id) as etymology_count,
                    COUNT(DISTINCT r.id) OVER (PARTITION BY w.id) as relation_count
                FROM words w
                LEFT JOIN definitions d ON w.id = d.word_id
                LEFT JOIN parts_of_speech p ON d.standardized_pos_id = p.id
                LEFT JOIN etymologies e ON w.id = e.word_id
                LEFT JOIN relations r ON w.id = r.from_word_id OR w.id = r.to_word_id
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
                AND (:pos IS NULL OR p.code = :pos)
            )
            SELECT DISTINCT ON (
                CASE 
                    WHEN :sort = 'relevance' THEN sr.relevance
                    WHEN :sort = 'alphabetical' THEN sr.lemma
                    WHEN :sort = 'created' THEN sr.created_at::text
                    WHEN :sort = 'updated' THEN sr.updated_at::text
                END,
                sr.id
            )
                sr.*
            FROM search_results sr
            WHERE sr.relevance >= :min_similarity
            ORDER BY 
                CASE 
                    WHEN :sort = 'relevance' THEN sr.relevance
                    WHEN :sort = 'alphabetical' THEN sr.lemma
                    WHEN :sort = 'created' THEN sr.created_at::text
                    WHEN :sort = 'updated' THEN sr.updated_at::text
                END DESC,
                sr.id
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
                "sort": params["sort"],
                "pos": params.get("pos")
            }
        ).fetchall()

        search_results = []
        for row in results:
            tags = row.tags.split(',') if row.tags else []
            definitions = [d for d in row.definitions if d] if row.definitions and row.definitions[0] else []
            pos_codes = [p for p in row.pos_codes if p] if row.pos_codes and row.pos_codes[0] else []
            search_results.append({
                    "word": row.lemma,
                    "normalized_lemma": row.normalized_lemma,
                    "language": row.language_code,
                    "has_baybayin": row.has_baybayin,
                    "baybayin_form": row.baybayin_form if row.has_baybayin else None,
                    "romanized_form": row.romanized_form if row.has_baybayin else None,
                    "preferred_spelling": row.preferred_spelling,
                "tags": tags,
                "parts_of_speech": pos_codes,
                "definitions": definitions[:3] if definitions else [],
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
            })
        
        response_meta = {
            "query": params["q"],
            "normalized_query": normalized_query,
            "mode": params["mode"],
            "total": len(search_results),
            "params": {k: v for k, v in params.items() if k != 'q'},
            "execution_time_ms": int((datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000)
        }
        
        return success_response(search_results, meta=response_meta)
    except ValidationError as err:
        return error_response("Invalid search parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Error in search_words: {str(e)}", exc_info=True)
        return error_response("Failed to perform search", 500, error_code="ERR_SEARCH")

@bp.route("/api/v2/baybayin", methods=["GET"])
@multi_level_cache(prefix="baybayin_words", ttl=3600)
def get_baybayin_words():
    """Get words with Baybayin forms."""
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        page = max(int(request.args.get("page", 1)), 1)
        language = request.args.get("language", "tl")
        query = Word.query.filter(Word.has_baybayin == True, Word.language_code == language)
        total = query.count()
        words = query.order_by(Word.lemma).offset((page - 1) * limit).limit(limit).all()
        results = []
        for word in words:
            results.append({
                "word": word.lemma,
                "normalized_lemma": word.normalized_lemma,
                "language": word.language_code,
                "baybayin_form": word.baybayin_form,
                "romanized_form": word.romanized_form,
                "has_etymology": bool(word.etymologies),
                "has_definitions": bool(word.definitions)
            })
        return success_response(
            results,
            meta={
                "total": total,
                "page": page,
                "limit": limit,
                "pages": (total + limit - 1) // limit
            }
        )
    except Exception as e:
        logger.error(f"Error in get_baybayin_words: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve Baybayin words")

@bp.route("/api/v2/parts-of-speech", methods=["GET"])
@multi_level_cache(prefix="parts_of_speech")
def get_parts_of_speech():
    """Get all available parts of speech."""
    try:
        pos_list = PartOfSpeech.query.order_by(PartOfSpeech.code).all()
        results = []
        for pos in pos_list:
            results.append({
                    "code": pos.code,
                    "name_en": pos.name_en,
                    "name_tl": pos.name_tl,
                    "description": pos.description
        })
        return success_response(results)
    except Exception as e:
        logger.error(f"Error in get_parts_of_speech: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve parts of speech")

@bp.route("/api/v2/statistics", methods=["GET"])
@multi_level_cache(prefix="statistics")
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
                (SELECT AVG(array_length(regexp_split_to_array(definition_text, '[[:space:]]+'), 1))::float 
                 FROM definitions) as avg_definition_length,
                (SELECT COUNT(*) FROM words WHERE idioms IS NOT NULL AND idioms != '[]') as words_with_idioms
        """)).fetchone()
        return success_response({
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
        })
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve statistics")

@bp.route("/api/v2/random", methods=["GET"])
@multi_level_cache(prefix="random_word", ttl=60)
def get_random_word():
    """Get a random word with optional filters."""
    try:
        language = request.args.get("language", "tl")
        has_baybayin = request.args.get("has_baybayin", type=bool)
        has_etymology = request.args.get("has_etymology", type=bool)
        has_definitions = request.args.get("has_definitions", True, type=bool)
        query = Word.query
        if language:
            query = query.filter(Word.language_code == language)
        if has_baybayin is not None:
            query = query.filter(Word.has_baybayin == has_baybayin)
        if has_etymology:
            query = query.join(Word.etymologies)
        if has_definitions:
            query = query.join(Word.definitions)
        word = query.order_by(func.random()).first()
        if not word:
            return error_response("No words found matching criteria", 404)
        word_details = get_word_details(word)
        return success_response(word_details["data"], meta={"random": True})
    except Exception as e:
        logger.error(f"Error in get_random_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve random word")

class WordRelationshipSchema(Schema):
    depth = fields.Int(validate=validate.Range(min=1, max=3), default=1)
    include_affixes = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    cluster_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)

class EtymologyTreeSchema(Schema):
    max_depth = fields.Int(validate=validate.Range(min=1, max=5), default=3)
    include_uncertain = fields.Bool(default=False)
    group_by_language = fields.Bool(default=False)

@bp.route("/api/v2/words/<path:word>/related", methods=["GET"])
@validate_query_params(WordRelationshipSchema)
@multi_level_cache(prefix="word_relationships", ttl=3600)
def get_word_relationship_graph(word, **params):
    """Get detailed relationship graph for a word."""
    try:
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:word_relationships")
        if not is_allowed:
            return rate_limit_exceeded_response()
        params = WordRelationshipSchema().load(request.args)
        word_entry = Word.query.options(
            joinedload(Word.definitions),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word),
            joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
            joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word)
        ).filter(Word.normalized_lemma == normalize_word(word)).first()
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
                    if rel.relation_type.lower() == "synonym":
                        graph["clusters"]["synonyms"].append(rel.to_word_id)
                    elif rel.relation_type.lower() == "variant":
                        graph["clusters"]["variants"].append(rel.to_word_id)
            if params.get("include_affixes", True):
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
            if params.get("include_etymology", True) and word_obj.etymologies:
                for etym in word_obj.etymologies:
                    components = parse_components_field(etym.normalized_components)
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
        return success_response(
            graph,
            meta={
                "root_word": word_entry.lemma,
                "depth": params["depth"],
                "total_nodes": len(graph["nodes"]),
                "total_edges": len(graph["edges"])
            }
        )
    except ValidationError as err:
        return error_response("Invalid parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Error in get_word_relationship_graph: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word relationships")

@bp.route("/api/v2/words/<path:word>/etymology-tree", methods=["GET"])
@validate_query_params(EtymologyTreeSchema)
@multi_level_cache(prefix="etymology_tree", ttl=3600)
def get_etymology_tree(word, **params):
    """Get the complete etymology tree for a word."""
    try:
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:etymology_tree")
        if not is_allowed:
            return rate_limit_exceeded_response()
        params = EtymologyTreeSchema().load(request.args)
        word_entry = Word.query.options(
            joinedload(Word.etymologies)
        ).filter(Word.normalized_lemma == normalize_word(word)).first()
        if not word_entry:
            return error_response("Word not found", 404)
        def build_etymology_tree(word_obj, visited=None, current_depth=0):
            if visited is None:
                visited = set()
            if word_obj.id in visited or current_depth >= params["max_depth"]:
                return None
            visited.add(word_obj.id)
            tree = {
                "word": word_obj.lemma,
                "language": word_obj.language_code,
                "has_baybayin": word_obj.has_baybayin,
                "baybayin_form": word_obj.baybayin_form if word_obj.has_baybayin else None,
                "etymologies": [],
                "components": []
            }
            for etym in word_obj.etymologies:
                etymology_entry = {
                    "text": etym.etymology_text,
                    "languages": etym.language_codes.split(", ") if etym.language_codes else [],
                    "sources": etym.sources.split(", ") if etym.sources else []
                }
                components = parse_components_field(etym.normalized_components)
                    for comp in components:
                        comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                    if comp_word and comp_word.id not in visited and current_depth < params["max_depth"]:
                        comp_tree = build_etymology_tree(comp_word, visited, current_depth + 1)
                            if comp_tree:
                                tree["components"].append(comp_tree)
                tree["etymologies"].append(etymology_entry)
            return tree
        etymology_tree = build_etymology_tree(word_entry)
        return success_response(
            etymology_tree,
            meta={
                "word": word_entry.lemma,
                "language": word_entry.language_code,
                "max_depth": params["max_depth"]
            }
        )
    except ValidationError as err:
        return error_response("Invalid parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Error in get_etymology_tree: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve etymology tree")

@bp.route("/favicon.ico")
def favicon():
    return "", 204

@bp.teardown_request
def remove_session(exception=None):
    db.session.remove()

def rate_limit_exceeded_response(window: int = 60) -> tuple:
    """Generate standardized rate limit exceeded response."""
    return error_response("Rate limit exceeded", 429, {"retry_after": window}, "ERR_RATE_LIMIT")

@event.listens_for(Word.__table__, 'after_create')
def create_word_indexes(target, connection, **kw):
    """Create additional indexes for better query performance."""
    connection.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_word_normalized_lemma ON words (normalized_lemma);
        CREATE INDEX IF NOT EXISTS idx_word_language_code ON words (language_code);
        CREATE INDEX IF NOT EXISTS idx_word_has_baybayin ON words (has_baybayin);
        CREATE INDEX IF NOT EXISTS idx_word_search_text ON words USING gin(search_text);
        CREATE INDEX IF NOT EXISTS idx_word_relations ON relations (from_word_id, to_word_id, relation_type);
        CREATE INDEX IF NOT EXISTS idx_word_affixations ON affixations (root_word_id, affixed_word_id, affix_type);
        CREATE INDEX IF NOT EXISTS idx_word_etymologies ON etymologies (word_id);
        CREATE INDEX IF NOT EXISTS idx_word_definitions ON definitions (word_id, standardized_pos_id);
        CREATE INDEX IF NOT EXISTS idx_definition_relations ON definition_relations (definition_id, word_id);
    """))

# Create and configure the Flask application instance
app = Flask(__name__)
app.register_blueprint(bp)

# (Optional) Load configuration from environment or a config file
# app.config.from_pyfile('config.py')
init_rate_limiter(app)

if __name__ == "__main__":
    # For local testing, run the built-in server (not for production)
    app.run(host="0.0.0.0", port=8000, debug=True)
