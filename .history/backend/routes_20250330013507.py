"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data,
with enhanced support for searching, detailed word information, Baybayin script,
etymology relationships, and statistical analysis.
"""

from flask import Flask, Blueprint, jsonify, request, current_app, g, abort
from sqlalchemy.orm import joinedload, selectinload, contains_eager
from sqlalchemy import or_, and_, func, desc, text, event, not_, case
from models import Word, Definition, Etymology, Relation, DefinitionRelation, Affixation, PartOfSpeech, db
from datetime import datetime, timezone, timedelta
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
import redis
from marshmallow import Schema, fields, validate, ValidationError
from typing import Tuple, Optional, Any, List, Dict, Union
from werkzeug.exceptions import NotFound, BadRequest, Forbidden, TooManyRequests, InternalServerError

# Initialize blueprint
bp = Blueprint("api", __name__)

# Enhanced logging setup
logger = structlog.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
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
            
        logger.info(f"Connecting to Redis at: {redis_url}")
        
        redis_client = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        redis_client.ping()
        logger.info(f"Successfully connected to Redis rate limiter at {redis_url}")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        logger.warning("Rate limiting will be disabled")
        redis_client = None

def check_rate_limit(key: str, limit: int = 30, window: int = 60) -> Tuple[bool, Optional[int]]:
    """
    Check if a request is within the configured rate limit.
    
    Args:
        key: Unique identifier for the rate limit
        limit: Maximum number of requests allowed in the window
        window: Time window in seconds
        
    Returns:
        Tuple of (is_allowed, remaining_requests)
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
                "rate_limit_exceeded",
                key=key,
                limit=limit,
                window=window,
                count=request_count
            )
        return is_allowed, remaining
    except redis.RedisError as e:
        logger.error(
            "redis_error",
            error=str(e),
            key=key,
            error_type=type(e).__name__
        )
        return True, None  # Fail open
    except Exception as e:
        logger.error(
            "unexpected_error",
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
    
    # Apply rate limiting to non-static endpoints
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
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
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
        request_headers={k: v for k, v in request.headers.items() 
                         if k.lower() not in ('authorization', 'cookie')},
        request_id=getattr(g, 'request_id', 'unknown')
    )
    
    # Handle specific exception types
    if isinstance(e, ValidationError):
        return error_response("Validation error", 400, e.messages, "ERR_VALIDATION")
    if isinstance(e, redis.RedisError):
        return error_response("Cache service temporarily unavailable", 503, {"error_id": error_id}, "ERR_REDIS")
    if isinstance(e, NotFound):
        return error_response("Resource not found", 404, {"error_id": error_id}, "ERR_NOT_FOUND")
    if isinstance(e, BadRequest):
        return error_response("Bad request", 400, {"error_id": error_id}, "ERR_BAD_REQUEST")
    if isinstance(e, Forbidden):
        return error_response("Forbidden", 403, {"error_id": error_id}, "ERR_FORBIDDEN")
    if isinstance(e, TooManyRequests):
        return error_response("Too many requests", 429, {"error_id": error_id}, "ERR_RATE_LIMIT")
    
    # Default to internal server error
    return error_response("Internal server error", 500, {"error_id": error_id}, "ERR_INTERNAL")

def error_response(message: str, status_code: int = 500, errors: dict = None, error_code: str = None) -> tuple:
    """
    Generate detailed error responses.
    
    Args:
        message: Error message
        status_code: HTTP status code
        errors: Additional error details
        error_code: Error code identifier
        
    Returns:
        Tuple of (response, status_code, headers)
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
    
    Args:
        data: Response data
        message: Optional success message
        meta: Additional metadata
        
    Returns:
        Tuple of (response, status_code, headers)
    """
    duration = None
    if hasattr(g, 'start_time'):
        duration = int((datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000)
        
    response = {
        "meta": {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(g, 'request_id', 'unknown'),
            **(meta or {})
        },
        "data": data
    }
    
    if duration:
        response["meta"]["duration_ms"] = duration
        
    if message:
        response["meta"]["message"] = message
        
    headers = {
        'Content-Type': 'application/json',
        'X-Response-Time': str(duration) if duration else None,
        'X-Request-ID': getattr(g, 'request_id', 'unknown')
    }
    
    return jsonify(response), 200, headers

# Cache utility for frequent operations
class ResponseCache:
    """
    Simple in-memory cache for API responses.
    """
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached item if exists and not expired."""
        if key not in self.cache:
            return None
            
        # Check expiration
        cached_time = self.access_times.get(key)
        if cached_time and (datetime.now(timezone.utc) - cached_time).total_seconds() > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
            
        # Update access time
        self.access_times[key] = datetime.now(timezone.utc)
        return self.cache[key]
        
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cache item with optional TTL."""
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            
        self.cache[key] = value
        self.access_times[key] = datetime.now(timezone.utc)
        
    def clear(self, prefix: str = None) -> None:
        """Clear entire cache or items with given prefix."""
        if prefix:
            keys_to_remove = [k for k in self.cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self.cache[k]
                del self.access_times[k]
        else:
            self.cache.clear()
            self.access_times.clear()

# Initialize response cache
response_cache = ResponseCache()

def cached(prefix: str, ttl: int = 3600):
    """
    Decorator for caching API responses.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Skip caching for non-GET requests
            if request.method != 'GET':
                return f(*args, **kwargs)
                
            # Skip caching if specifically requested
            if request.args.get('nocache') == 'true':
                return f(*args, **kwargs)
                
            # Generate cache key
            key = f"{prefix}:{request.path}:{request.query_string.decode()}"
            cached_response = response_cache.get(key)
            
            # Return cached response if available
            if cached_response:
                return cached_response
                
            # Generate new response
            response = f(*args, **kwargs)
            
            # Cache response
            response_cache.set(key, response, ttl)
            
            return response
        return wrapped
    return decorator

@lru_cache(maxsize=1000)
def normalize_word(text: str) -> str:
    """
    Normalize a word for consistent lookup.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    normalized = unidecode(text.lower())
    normalized = re.sub(r'[^\w\s\-]', '', normalized)
    return normalized.strip()

def parse_components_field(components_data: str) -> List[str]:
    """
    Parse etymology components from various formats.
    
    Args:
        components_data: Raw components data
        
    Returns:
        List of component strings
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

def extract_components_from_text(etymology_text: str) -> List[str]:
    """
    Extract components directly from etymology text when normalized_components is not available.
    
    Args:
        etymology_text: The etymology text to parse
        
    Returns:
        List of component strings
    """
    if not etymology_text:
        return []
    
    components = []
    
    # Clean the text
    text = etymology_text.strip()
    
    # Pattern for "From X + Y" or "X + Y"
    if '+' in text:
        # Remove common prefixes like "From" or "Derived from"
        cleaned_text = re.sub(r'^(From|Derived from|Compound of)\s+', '', text, flags=re.IGNORECASE)
        # Remove trailing punctuation
        cleaned_text = re.sub(r'[.,;:]$', '', cleaned_text.strip())
        
        # Split by '+' and clean each component
        parts = [part.strip() for part in cleaned_text.split('+')]
        for part in parts:
            # Remove any trailing/leading punctuation or quotes
            clean_part = re.sub(r'^[\s"\'(]|[\s"\'.),;:]$', '', part.strip())
            if clean_part:
                components.append(clean_part)
    
    # Pattern for "Borrowed from X" or "from X"
    borrowed_match = re.search(r'[Bb]orrowed from\s+([^,.;]+)', text)
    if borrowed_match:
        borrowed_from = borrowed_match.group(1).strip()
        if borrowed_from:
            components.append(borrowed_from)
    
    # Pattern for "from X" when not already matched
    if not components and re.search(r'[Ff]rom\s+([^,.;+]+)', text):
        from_matches = re.findall(r'[Ff]rom\s+([^,.;+]+)', text)
        for match in from_matches:
            clean_match = match.strip()
            if clean_match and not any(c in clean_match for c in [',', '.', ';']):
                components.append(clean_match)
    
    # Pattern for "compound of X and Y"
    compound_match = re.search(r'[Cc]ompound of\s+([^,]+)\s+and\s+([^,.;]+)', text)
    if compound_match:
        components.append(compound_match.group(1).strip())
        components.append(compound_match.group(2).strip())
    
    # Pattern for "blend of X and Y"
    blend_match = re.search(r'[Bb]lend of\s+([^,]+)\s+and\s+([^,.;]+)', text)
    if blend_match:
        components.append(blend_match.group(1).strip())
        components.append(blend_match.group(2).strip())
    
    # Extract language-specific components
    language_patterns = [
        r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+([^,.;]+)',  # from Language Word
        r'from\s+([^,.;]+)\s+\(([^)]+)\)',  # from Word (meaning)
        r'from\s+Proto-[A-Za-z-]+\s+\*([^,.;]+)'  # from Proto-X *word
    ]
    
    for pattern in language_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            for group in groups:
                if group and len(group) > 1:  # Avoid single characters
                    clean_group = group.strip()
                    # Remove asterisks from proto-forms
                    if clean_group.startswith('*'):
                        clean_group = clean_group[1:]
                    components.append(clean_group)
    
    # Extract root words
    root_patterns = [
        r'with the root from\s+([^,.;]+)',
        r'root word\s+([^,.;]+)',
        r'derived from\s+([^,.;]+)'
    ]
    
    for pattern in root_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            root = match.group(1).strip()
            if root:
                components.append(root)
    
    # Clean up components
    cleaned_components = []
    for comp in components:
        # Remove quotes, parentheses, and other punctuation at the beginning/end
        clean_comp = re.sub(r'^[\s"\'(*]|[\s"\'.),;:*]$', '', comp.strip())
        
        # Skip components that are too short or contain unwanted patterns
        if (clean_comp and len(clean_comp) > 1 and 
            not clean_comp.lower().startswith(('see ', 'compare ', 'cognate ', 'doublet '))):
            # Remove any language name prefixes like "English" or "Spanish" if they're standalone
            if not re.match(r'^[A-Z][a-z]+$', clean_comp):  # Don't remove if it's just the language name
                cleaned_components.append(clean_comp)
    
    return cleaned_components

# Schema definitions for request validation
class WordQuerySchema(Schema):
    """Schema for word query parameters."""
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=False)
    include_idioms = fields.Bool(default=False)
    include_source_info = fields.Bool(default=False)

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix', 'idm', 'col', 'syn', 'ant', 'eng', 'spa', 'tx', 'var', 'unc']))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'phonetic', 'baybayin', 'fuzzy', 'etymology']), default='all')
    sort = fields.Str(validate=validate.OneOf(['relevance', 'alphabetical', 'created', 'updated', 'etymology']), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')
    include_definitions = fields.Bool(default=True)
    include_etymology = fields.Bool(default=False)
    include_pronunciation = fields.Bool(default=False)
    include_idioms = fields.Bool(default=False)
    include_source_info = fields.Bool(default=False)
    filter_has_etymology = fields.Bool(default=False)
    filter_has_pronunciation = fields.Bool(default=False)
    filter_has_idioms = fields.Bool(default=False)

class PaginationSchema(Schema):
    """Schema for pagination parameters."""
    page = fields.Int(validate=validate.Range(min=1), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)

class WordDetailSchema(Schema):
    """Schema for word detail parameters."""
    include_definitions = fields.Bool(missing=True)
    include_relations = fields.Bool(missing=True)
    include_etymology = fields.Bool(missing=True)
    include_metadata = fields.Bool(missing=True)
    include_pronunciation = fields.Bool(missing=False)
    include_idioms = fields.Bool(missing=False)
    include_source_info = fields.Bool(missing=False)
    include_affixes = fields.Bool(missing=True)
    include_definition_relations = fields.Bool(missing=False)

class WordRelationshipSchema(Schema):
    """Schema for word relationship parameters."""
    depth = fields.Int(validate=validate.Range(min=1, max=3), default=1)
    include_affixes = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_definitions = fields.Bool(default=False)
    cluster_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)

class EtymologyTreeSchema(Schema):
    """Schema for etymology tree parameters."""
    max_depth = fields.Int(validate=validate.Range(min=1, max=5), default=3)
    include_uncertain = fields.Bool(default=False)
    group_by_language = fields.Bool(default=False)

class RandomQuerySchema(Schema):
    """Schema for random word parameters."""
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    has_baybayin = fields.Bool()
    has_etymology = fields.Bool(default=False)
    has_definitions = fields.Bool(default=True)
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)

class BaybayinQuerySchema(Schema):
    """Schema for baybayin query parameters."""
    limit = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    page = fields.Int(validate=validate.Range(min=1), default=1)
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')

class BaybayinTransliterationSchema(Schema):
    """Schema for baybayin transliteration parameters."""
    text = fields.Str(required=True, validate=validate.Length(min=1, max=1000))
    to_baybayin = fields.Bool(missing=True)
    
class RelationsQuerySchema(Schema):
    """Schema for relations query parameters."""
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    type = fields.Str(validate=validate.OneOf([
        # Basic semantic relationships
        'synonym', 'antonym', 'variant', 'spelling_variant',
        # Hierarchical relationships
        'hypernym_of', 'hyponym_of', 'meronym_of', 'holonym_of',
        # Derivational relationships
        'derived_from', 'root_of', 'derived', 'base_of',
        # Etymology relationships
        'cognate', 'borrowed_from', 'loaned_to', 'descendant_of', 'ancestor_of',
        # Structural relationships
        'component_of', 'abbreviation_of', 'has_abbreviation', 'initialism_of', 'has_initialism',
        # General relationships
        'related'
    ], error="Invalid relation type specified"), missing=None)
    include_metadata = fields.Bool(default=False)
    include_source_info = fields.Bool(default=False)
    include_strength = fields.Bool(default=False)
    
class AffixesQuerySchema(Schema):
    """Schema for affixes query parameters."""
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    type = fields.Str(validate=validate.OneOf([
        'prefix', 'infix', 'suffix', 'circumfix', 'reduplication', 'compound'
    ], error="Invalid affix type specified"), missing=None)
    include_source_info = fields.Bool(default=False)
    include_examples = fields.Bool(default=False)
    
def validate_query_params(schema_cls):
    """
    Decorator to validate query parameters using marshmallow schema.
    
    Args:
        schema_cls: Schema class for validation
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Create schema instance
                schema = schema_cls()
                
                # Validate query parameters
                params = schema.load(request.args)
                
                # Add validated parameters to kwargs
                kwargs.update(params)
                
                return f(*args, **kwargs)
            except ValidationError as err:
                return error_response("Invalid query parameters", 400, err.messages, "ERR_VALIDATION")
        return wrapper
    return decorator

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
            "word_relationships": True,
            "multilingual": True
        },
        "endpoints": {
            "root": "/",
            "words": "/api/v2/words",
            "word_details": "/api/v2/words/<word>",
            "search": "/api/v2/search",
            "statistics": "/api/v2/statistics",
            "random": "/api/v2/random",
            "baybayin": "/api/v2/baybayin",
            "baybayin_transliterate": "/api/v2/baybayin/transliterate",
            "word_baybayin": "/api/v2/words/<word>/baybayin",
            "parts_of_speech": "/api/v2/parts-of-speech",
            "related_words": "/api/v2/words/<word>/related",
            "etymology_tree": "/api/v2/words/<word>/etymology-tree",
            "affixes": "/api/v2/affixes",
            "relations": "/api/v2/relations",
        },
        "supported_languages": [
            {"code": "tl", "name": "Tagalog / Filipino"},
            {"code": "ceb", "name": "Cebuano / Bisaya"}
        ]
    })

@bp.route("/api/v2/words", methods=["GET"])
@cached(prefix="words_list", ttl=1800)
@validate_query_params(PaginationSchema)
def get_words(**params):
    """
    Get a paginated list of words with filtering options.
    
    Query Parameters:
        page (int): Page number (default: 1)
        per_page (int): Items per page (default: 20, max: 100)
        search (str): Optional search term for filtering
        language (str): Language code filter (default: 'tl')
        pos (str): Part of speech filter
        has_baybayin (bool): Filter for words with Baybayin script
        has_etymology (bool): Filter for words with etymology
    
    Returns:
        Paginated list of words matching the filters
    """
    try:
        # Extract pagination parameters
        page = params.get('page', 1)
        per_page = min(params.get('per_page', 20), 100)
        
        # Extract filter parameters
        search = request.args.get("search", "").strip()
        language = request.args.get("language", "tl")
        pos = request.args.get("pos")
        has_baybayin = request.args.get("has_baybayin", type=bool)
        has_etymology = request.args.get("has_etymology", type=bool)

        # Build query with eager loading
        query = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos)
        )
        
        # Apply filters
        if search:
            normalized_search = normalize_word(search)
            query = query.filter(
                or_(
                    Word.normalized_lemma.like(f"{normalized_search}%"),
                    Word.lemma.ilike(f"%{search}%"),
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
        
        # Get total count for pagination
        total = query.count()
        
        # Apply pagination and ordering
        words = query.order_by(Word.lemma).offset((page - 1) * per_page).limit(per_page).all()
        
        # Format response data
        word_list = []
        for w in words:
            pos_list = []
            if hasattr(w, 'definitions'):
                pos_list = list(set(
                    d.standardized_pos.code for d in w.definitions 
                    if hasattr(d, 'standardized_pos') and d.standardized_pos
                ))
                
            word_list.append({
                "id": w.id,
                "word": w.lemma,
                "normalized_lemma": w.normalized_lemma,
                "language": w.language_code,
                "has_baybayin": w.has_baybayin,
                "baybayin_form": w.baybayin_form if w.has_baybayin else None,
                "romanized_form": w.romanized_form if w.has_baybayin else None,
                "parts_of_speech": pos_list,
                "definition_count": len(w.definitions),
                "has_etymology": bool(w.etymologies),
                "tags": w.get_tags_list(),
                "created_at": w.created_at.isoformat() if w.created_at else None
            })
            
        # Create response with pagination metadata
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

@bp.route("/api/v2/words/<path:word>", methods=["GET"])
@cached(prefix="word_detail", ttl=3600)
@validate_query_params(WordDetailSchema)
def get_word(word, **params):
    """
    Get detailed information about a specific word.
    
    Path Parameters:
        word (str): The word to retrieve
        
    Query Parameters:
        language_code (str): Language code (default: 'tl')
        include_definitions (bool): Include definitions (default: True)
        include_relations (bool): Include word relationships (default: True)
        include_etymology (bool): Include etymology (default: True)
        include_metadata (bool): Include additional metadata (default: True)
        include_pronunciation (bool): Include pronunciation (default: False)
        include_idioms (bool): Include idioms (default: False)
        include_source_info (bool): Include source information (default: False)
        include_affixes (bool): Include affix relationships (default: True)
        include_definition_relations (bool): Include definition relations (default: False)
    
    Returns:
        Detailed word information
    """
    try:
        # Extract parameters
        include_definitions = params.get('include_definitions', True)
        include_relations = params.get('include_relations', True)
        include_etymology = params.get('include_etymology', True)
        include_metadata = params.get('include_metadata', True)
        include_pronunciation = params.get('include_pronunciation', False)
        include_idioms = params.get('include_idioms', False)
        include_source_info = params.get('include_source_info', False)
        include_affixes = params.get('include_affixes', True)
        include_definition_relations = params.get('include_definition_relations', False)
        language_code = request.args.get('language_code', 'tl')
        
        # Validate language code
        if language_code not in ['tl', 'ceb']:
            language_code = 'tl'
        
        # Build query with eager loading based on parameters
        query_options = []
        if include_definitions:
            query_options.append(joinedload(Word.definitions).joinedload(Definition.standardized_pos))
            if include_definition_relations:
                query_options.append(joinedload(Word.definitions).joinedload(Definition.definition_relations).joinedload(DefinitionRelation.word))
        if include_etymology:
            query_options.append(joinedload(Word.etymologies))
        if include_relations:
            query_options.append(joinedload(Word.relations_from).joinedload(Relation.to_word))
            query_options.append(joinedload(Word.relations_to).joinedload(Relation.from_word))
        if include_affixes:
            query_options.append(joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word))
            query_options.append(joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word))
        
        # Find word by normalized lemma
        word_entry = Word.query.options(*query_options).filter(
            Word.normalized_lemma == normalize_word(word),
            Word.language_code == language_code
        ).first()
        
        # If word not found, try to find similar words as suggestions
        if not word_entry:
            similar_words = Word.query.filter(
                Word.language_code == language_code,
                func.similarity(Word.normalized_lemma, normalize_word(word)) > 0.3
            ).order_by(
                func.similarity(Word.normalized_lemma, normalize_word(word)).desc()
            ).limit(5).all()

            suggestions = []
            if similar_words:
                suggestions = [
                    {
                        "word": w.lemma,
                        "normalized_lemma": w.normalized_lemma,
                        "similarity": round(fuzz.ratio(normalize_word(word), w.normalized_lemma) / 100, 2)
                    } 
                    for w in similar_words
                ]

            return error_response(
                "Word not found", 
                404, 
                {"suggestions": suggestions if suggestions else None}
            )
        
        # Convert word to dictionary with requested includes
        word_data = word_entry.to_dict(
            include_definitions=include_definitions,
            include_relations=include_relations,
            include_etymology=include_etymology,
            include_metadata=include_metadata
        )
        
        # Add additional requested data
        if include_pronunciation and word_entry.pronunciation_data:
            word_data["pronunciation"] = word_entry.pronunciation_data
            
        if include_idioms:
            word_data["idioms"] = word_entry.get_idioms_list()
            
        if include_source_info:
            word_data["source_info"] = word_entry.source_info
            
        # Return success response
        return success_response(
            word_data,
            meta={
                "version": "2.0",
                "includes": {
                    "definitions": include_definitions,
                    "relations": include_relations,
                    "etymology": include_etymology,
                    "metadata": include_metadata,
                    "pronunciation": include_pronunciation,
                    "idioms": include_idioms,
                    "source_info": include_source_info,
                    "affixes": include_affixes,
                    "definition_relations": include_definition_relations
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in get_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve word details")

@bp.route("/api/v2/search", methods=["GET"])
@cached(prefix="search", ttl=300)
@validate_query_params(SearchQuerySchema)
def search_words(**params):
    """
    Search words with enhanced filtering and sorting.
    
    Query Parameters:
        q (str): Search query
        limit (int): Maximum results to return (default: 20, max: 100)
        pos (str): Filter by part of speech
        language (str): Language code (default: 'tl')
        include_baybayin (bool): Include baybayin matches (default: True)
        min_similarity (float): Minimum similarity threshold (default: 0.3)
        mode (str): Search mode (all, exact, phonetic, baybayin, fuzzy, etymology) (default: all)
        sort (str): Sorting field (relevance, alphabetical, created, updated, etymology) (default: relevance)
        order (str): Sort order (asc, desc) (default: desc)
        include_definitions (bool): Include definitions in results (default: True)
        include_etymology (bool): Include etymology in results (default: False)
        include_pronunciation (bool): Include pronunciation data (default: False)
        include_idioms (bool): Include idioms (default: False)
        include_source_info (bool): Include source information (default: False)
        filter_has_etymology (bool): Filter words with etymology (default: False)
        filter_has_pronunciation (bool): Filter words with pronunciation (default: False)
        filter_has_idioms (bool): Filter words with idioms (default: False)
    """
    try:
        # Log search request
        logger.info(
            "search_request",
            params=params,
            remote_addr=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        # Check rate limit
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:search", limit=60, window=60)
        if not is_allowed:
            return rate_limit_exceeded_response()
        
        # Extract parameters
        query = params.get("q", "")
        limit = min(params.get("limit", 20), 100)
        language = params.get("language", "tl")
        include_baybayin = params.get("include_baybayin", True)
        min_similarity = params.get("min_similarity", 0.3)
        mode = params.get("mode", "all")
        sort = params.get("sort", "relevance")
        order = params.get("order", "desc")
        pos = params.get("pos")
        
        # Extract include flags
        include_definitions = params.get("include_definitions", True)
        include_etymology = params.get("include_etymology", False)
        include_pronunciation = params.get("include_pronunciation", False)
        include_idioms = params.get("include_idioms", False)
        include_source_info = params.get("include_source_info", False)
        
        # Extract filter flags
        filter_has_etymology = params.get("filter_has_etymology", False)
        filter_has_pronunciation = params.get("filter_has_pronunciation", False)
        filter_has_idioms = params.get("filter_has_idioms", False)
        
        # Validate query
        if not query or len(query.strip()) == 0:
            return error_response("Search query cannot be empty", 400, error_code="ERR_VALIDATION")
        
        # Normalize query
        normalized_query = normalize_word(query)
        if not normalized_query:
            normalized_query = query.lower()
        
        # Build base query with eager loading based on includes
        base_query = Word.query
        
        # Add eager loading options
        if include_definitions:
            base_query = base_query.options(
                joinedload(Word.definitions).joinedload(Definition.standardized_pos)
            )
        if include_etymology:
            base_query = base_query.options(
                joinedload(Word.etymologies)
            )
        
        # Apply language filter
        base_query = base_query.filter(Word.language_code == language)
        
        # Apply mode-specific filters
        if mode == 'exact':
            base_query = base_query.filter(Word.normalized_lemma == normalized_query)
        elif mode == 'phonetic':
            try:
                # Try PostgreSQL metaphone
                test_query = db.session.query(func.metaphone('test', 4))
                test_query.first()
                base_query = base_query.filter(
                    func.metaphone(Word.normalized_lemma, 10) == func.metaphone(normalized_query, 10)
                )
            except Exception:
                # Fallback to prefix matching
                if len(normalized_query) >= 3:
                    base_query = base_query.filter(Word.normalized_lemma.like(f'{normalized_query[:3]}%'))
                else:
                    base_query = base_query.filter(Word.normalized_lemma.like(f'{normalized_query}%'))
        elif mode == 'baybayin' and include_baybayin:
            base_query = base_query.filter(
                Word.has_baybayin == True,
                Word.baybayin_form.isnot(None),
                Word.baybayin_form.like(f'{query}%')
            )
        elif mode == 'fuzzy':
            try:
                # Try PostgreSQL similarity
                test_query = db.session.query(func.similarity('test', 'test'))
                test_query.first()
                base_query = base_query.filter(
                    func.similarity(Word.normalized_lemma, normalized_query) > min_similarity
                )
            except Exception:
                # Fallback to LIKE
                base_query = base_query.filter(
                    or_(
                        Word.lemma.ilike(f'%{query}%'),
                        Word.normalized_lemma.ilike(f'%{normalized_query}%')
                    )
                )
        elif mode == 'etymology':
            base_query = base_query.join(Word.etymologies).filter(
                or_(
                    Etymology.etymology_text.ilike(f'%{query}%'),
                    Etymology.language_codes.ilike(f'%{query}%'),
                    Etymology.normalized_components.ilike(f'%{query}%')
                )
            )
        else:  # 'all' mode
            base_query = base_query.filter(
                or_(
                    Word.normalized_lemma == normalized_query,
                    Word.normalized_lemma.like(f'{normalized_query}%'),
                    Word.lemma.ilike(f'%{query}%'),
                    Word.normalized_lemma.ilike(f'%{normalized_query}%'),
                    Word.preferred_spelling == query,
                    and_(
                        Word.has_baybayin == True,
                        Word.baybayin_form.isnot(None),
                        Word.baybayin_form.like(f'{query}%')
                    ) if include_baybayin else False,
                    Word.search_text.op('@@')(func.plainto_tsquery('simple', normalized_query))
                )
            )
        
        # Apply additional filters
        if pos:
            base_query = base_query.join(Word.definitions).join(Definition.standardized_pos).filter(
                PartOfSpeech.code == pos
            ).distinct()
            
        if filter_has_etymology:
            base_query = base_query.join(Word.etymologies)
            
        if filter_has_pronunciation:
            base_query = base_query.filter(Word.pronunciation_data.isnot(None))
            
        if filter_has_idioms:
            base_query = base_query.filter(Word.idioms != '[]')
        
        # Apply sorting
        if sort == 'alphabetical':
            base_query = base_query.order_by(Word.lemma.asc() if order == 'asc' else Word.lemma.desc())
        elif sort == 'created':
            base_query = base_query.order_by(Word.created_at.asc() if order == 'asc' else Word.created_at.desc())
        elif sort == 'updated':
            base_query = base_query.order_by(Word.updated_at.asc() if order == 'asc' else Word.updated_at.desc())
        elif sort == 'etymology':
            base_query = base_query.outerjoin(Word.etymologies).group_by(Word.id).order_by(
                func.count(Etymology.id).asc() if order == 'asc' else func.count(Etymology.id).desc()
            )
        else:  # relevance
            try:
                if order == 'asc':
                    base_query = base_query.order_by(
                        (Word.normalized_lemma != normalized_query).asc(),
                        (not_(Word.normalized_lemma.like(f'{normalized_query}%'))).asc(),
                        Word.lemma.asc()
                    )
                else:
                    base_query = base_query.order_by(
                        (Word.normalized_lemma == normalized_query).desc(),
                        (Word.normalized_lemma.like(f'{normalized_query}%')).desc(),
                        Word.lemma.asc()
                    )
            except Exception:
                # Fallback to simpler ordering
                if order == 'asc':
                    base_query = base_query.order_by(Word.lemma.asc())
                else:
                    try:
                        exact_matches = [w.id for w in db.session.query(Word.id).filter(
                            Word.normalized_lemma == normalized_query,
                            Word.language_code == language
                        ).all()]
                        
                        startswith_matches = [w.id for w in db.session.query(Word.id).filter(
                            Word.normalized_lemma.like(f'{normalized_query}%'),
                            Word.normalized_lemma != normalized_query,
                            Word.language_code == language
                        ).all()]
                        
                        base_query = base_query.order_by(
                            case((Word.id.in_(exact_matches), 1), else_=
                                case((Word.id.in_(startswith_matches), 2), else_=3)
                            ),
                            Word.lemma.asc()
                        )
                    except Exception:
                        base_query = base_query.order_by(Word.lemma.asc())
        
        # Execute query
        words = base_query.limit(limit).all()
        
        # Format results
        search_results = []
        for word in words:
            # Calculate relevance score
            relevance = 0.0
            if word.normalized_lemma == normalized_query:
                relevance = 1.0
            elif word.normalized_lemma.startswith(normalized_query):
                relevance = 0.8
            elif normalized_query in word.normalized_lemma:
                relevance = 0.6
            elif word.lemma.lower() == query.lower():
                relevance = 0.9
            elif word.preferred_spelling == query:
                relevance = 0.7
            elif word.has_baybayin and word.baybayin_form and query in word.baybayin_form:
                relevance = 0.7
            else:
                relevance = 0.4
            
            # Get definitions if requested
            word_definitions = []
            if include_definitions and word.definitions:
                word_definitions = [d.definition_text for d in word.definitions[:3]]
            
            # Get parts of speech
            pos_codes = []
            if word.definitions:
                pos_codes = list(set(
                    d.standardized_pos.code for d in word.definitions 
                    if hasattr(d, 'standardized_pos') and d.standardized_pos
                ))
            
            # Get counts
            definition_count = len(word.definitions) if word.definitions else 0
            etymology_count = len(word.etymologies) if word.etymologies else 0
            relation_count = (
                len(word.relations_from) if hasattr(word, 'relations_from') and word.relations_from else 0
            ) + (
                len(word.relations_to) if hasattr(word, 'relations_to') and word.relations_to else 0
            )
            
            # Extract tags
            tags = word.get_tags_list() if hasattr(word, 'get_tags_list') else []
            
            # Build result
            result = {
                "id": word.id,
                "word": word.lemma,
                "normalized_lemma": word.normalized_lemma,
                "language": word.language_code,
                "has_baybayin": word.has_baybayin,
                "baybayin_form": word.baybayin_form if word.has_baybayin else None,
                "romanized_form": word.romanized_form if word.has_baybayin else None,
                "preferred_spelling": word.preferred_spelling,
                "tags": tags,
                "parts_of_speech": pos_codes,
                "counts": {
                    "definitions": definition_count,
                    "etymologies": etymology_count,
                    "relations": relation_count
                },
                "has_etymology": bool(word.etymologies),
                "has_pronunciation": bool(word.pronunciation_data),
                "has_idioms": bool(word.idioms and word.idioms != '[]'),
                "has_relations": bool(hasattr(word, 'relations_from') and (word.relations_from or word.relations_to)),
                "relevance": relevance,
                "created_at": word.created_at.isoformat() if word.created_at else None,
                "updated_at": word.updated_at.isoformat() if word.updated_at else None
            }
            
            # Add definitions if requested
            if include_definitions and word_definitions:
                result["definitions"] = word_definitions
            
            # Add etymology if requested
            if include_etymology and word.etymologies:
                etymology_data = {
                    "text": word.etymologies[0].etymology_text,
                    "sources": word.etymologies[0].get_sources_list() if hasattr(word.etymologies[0], 'get_sources_list') else [],
                    "languages": word.etymologies[0].get_language_codes_list() if hasattr(word.etymologies[0], 'get_language_codes_list') else [],
                    "components": word.etymologies[0].get_components_list() if hasattr(word.etymologies[0], 'get_components_list') else []
                }
                result["etymology"] = etymology_data
            
            # Add pronunciation if requested
            if include_pronunciation and word.pronunciation_data:
                result["pronunciation"] = word.pronunciation_data
            
            # Add idioms if requested
            if include_idioms and word.idioms and word.idioms != '[]':
                result["idioms"] = word.get_idioms_list()
            
            # Add source info if requested
            if include_source_info and word.source_info:
                result["source_info"] = word.source_info
            
            search_results.append(result)
        
        # Create response metadata
        response_meta = {
            "query": query,
            "normalized_query": normalized_query,
            "mode": mode,
            "total": len(search_results),
            "params": {
                "limit": limit,
                "language": language,
                "pos": pos,
                "min_similarity": min_similarity,
                "sort": sort,
                "order": order,
                "include_definitions": include_definitions,
                "include_etymology": include_etymology,
                "include_pronunciation": include_pronunciation,
                "include_idioms": include_idioms,
                "include_source_info": include_source_info,
                "include_baybayin": include_baybayin,
                "filter_has_etymology": filter_has_etymology,
                "filter_has_pronunciation": filter_has_pronunciation,
                "filter_has_idioms": filter_has_idioms
            },
            "execution_time_ms": int((datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000)
        }
        
        logger.info(
            "search_completed",
            query=query,
            results_count=len(search_results),
            execution_time_ms=response_meta["execution_time_ms"]
        )
        
        return success_response(search_results, meta=response_meta)
    except ValidationError as err:
        logger.error(f"Validation error in search_words: {str(err)}")
        return error_response("Invalid search parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Unexpected error in search_words: {str(e)}", exc_info=True)
        return error_response(
            "Failed to perform search", 
            500, 
            {
                "error_details": str(e),
                "query": params.get("q", ""),
                "debug_info": {
                    "remote_addr": request.remote_addr,
                    "user_agent": request.headers.get('User-Agent'),
                    "query_params": dict(request.args)
                }
            }, 
            "ERR_SEARCH"
        )

@bp.route("/api/v2/baybayin", methods=["GET"])
@cached(prefix="baybayin_words", ttl=3600)
@validate_query_params(BaybayinQuerySchema)
def get_baybayin_words(**params):
    """
    Get words with Baybayin script representation.
    
    Query Parameters:
        limit (int): Maximum results to return (default: 20, max: 100)
        page (int): Page number for pagination (default: 1)
        language (str): Language code (default: 'tl')
    
    Returns:
        List of words with Baybayin script
    """
    try:
        # Extract parameters
        limit = min(params.get("limit", 20), 100)
        page = max(params.get("page", 1), 1)
        language = params.get("language", "tl")
        
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Get words with Baybayin script
        words = Word.query.filter(
            Word.has_baybayin == True,
            Word.baybayin_form.isnot(None),
            Word.language_code == language
        ).order_by(
            Word.lemma.asc()
        ).offset(offset).limit(limit).all()
        
        # Count total words with Baybayin
        total_baybayin = Word.query.filter(
            Word.has_baybayin == True,
            Word.baybayin_form.isnot(None),
            Word.language_code == language
        ).count()
        
        # Format results
        results = []
        for word in words:
            results.append({
                "id": word.id,
                "word": word.lemma,
                "normalized_lemma": word.normalized_lemma,
                "baybayin_form": word.baybayin_form,
                "romanized_form": word.romanized_form
            })
        
        # Return response
        return success_response(
            results,
            meta={
                "page": page,
                "per_page": limit,
                "total": total_baybayin,
                "total_pages": (total_baybayin + limit - 1) // limit,
                "language": language
            }
        )
    except Exception as e:
        logger.error(f"Error in get_baybayin_words: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve Baybayin words")

@bp.route("/api/v2/baybayin/transliterate", methods=["GET"])
@validate_query_params(BaybayinTransliterationSchema)
def transliterate_baybayin(**params):
    """
    Transliterate text between Latin and Baybayin scripts.
    
    Query Parameters:
        text (str): The text to transliterate
        to_baybayin (bool): Whether to convert to Baybayin (True) or from Baybayin to Latin (False)
        
    Returns:
        Transliteration result
    """
    try:
        # Extract validated parameters
        text = params.get("text")
        to_baybayin = params.get("to_baybayin", True)
        
        # Import necessary functions from dictionary_manager
        from dictionary_manager import transliterate_to_baybayin, get_romanized_text
        
        # Detect script type for mixed script handling
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_baybayin = bool(re.search(r'[\u1700-\u171F]', text))
        
        # Handle case with mixed scripts
        if has_latin and has_baybayin:
            # If mixed, process each part separately based on the primary direction
            if to_baybayin:
                # Extract Latin parts for Baybayin conversion
                latin_parts = re.split(r'[\u1700-\u171F\s]+', text)
                result = text
                for part in latin_parts:
                    if part.strip():
                        baybayin_part = transliterate_to_baybayin(part)
                        result = result.replace(part, baybayin_part)
            else:
                # Extract Baybayin parts for Latin conversion
                baybayin_parts = re.findall(r'[\u1700-\u171F]+', text)
                result = text
                for part in baybayin_parts:
                    if part.strip():
                        latin_part = get_romanized_text(part)
                        result = result.replace(part, latin_part)
        else:
            # Perform transliteration based on direction
            if to_baybayin:
                result = transliterate_to_baybayin(text)
            else:
                result = get_romanized_text(text)
            
        # Return response
        return success_response(
            {
                "original_text": text,
                "transliterated_text": result,
                "direction": "to_baybayin" if to_baybayin else "to_latin"
            },
            meta={
                "length": len(text),
                "duration_ms": getattr(g, 'request_duration_ms', 0)
            }
        )
    except Exception as e:
        logger.error(f"Error in transliterate_baybayin: {str(e)}", exc_info=True)
        return error_response(
            "Failed to transliterate text", 
            error_code="ERR_TRANSLITERATION",
            errors={"text": "Text contains characters that cannot be transliterated properly"}
        )

@bp.route("/api/v2/words/<path:word>/baybayin", methods=["GET"])
@cached(prefix="word_baybayin", ttl=3600)
def get_word_baybayin(word):
    """
    Get the Baybayin representation of a specific word.
    
    Path Parameters:
        word (str): The word to get Baybayin for
        
    Returns:
        Word with Baybayin representation
    """
    try:
        # Find word
        word_entry = Word.query.filter(
            Word.normalized_lemma == normalize_word(word)
        ).first()
        
        if not word_entry:
            return error_response("Word not found", 404)
            
        # Check if word has Baybayin
        if not word_entry.has_baybayin or not word_entry.baybayin_form:
            # Try to generate Baybayin if it doesn't exist
            from dictionary_manager import transliterate_to_baybayin
            generated_baybayin = transliterate_to_baybayin(word_entry.lemma)
            
            return success_response(
                {
                    "id": word_entry.id,
                    "word": word_entry.lemma,
                    "normalized_lemma": word_entry.normalized_lemma, 
                    "baybayin_form": generated_baybayin,
                    "romanized_form": word_entry.lemma,
                    "is_generated": True
                }
            )
            
        # Return Baybayin representation
        return success_response(
            {
                "id": word_entry.id,
                "word": word_entry.lemma,
                "normalized_lemma": word_entry.normalized_lemma,
                "baybayin_form": word_entry.baybayin_form,
                "romanized_form": word_entry.romanized_form,
                "is_generated": False
            }
        )
    except Exception as e:
        logger.error(f"Error in get_word_baybayin: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve Baybayin representation")

@bp.route("/api/v2/parts-of-speech", methods=["GET"])
@cached(prefix="parts_of_speech", ttl=86400)  # Cache for 24 hours
def get_parts_of_speech():
    """
    Get all available parts of speech.
    
    Returns:
        List of parts of speech with codes and descriptions
    """
    try:
        # Fetch all parts of speech
        pos_list = PartOfSpeech.query.order_by(PartOfSpeech.code).all()
        
        # Format results
        results = []
        for pos in pos_list:
            results.append({
                "id": pos.id,
                "code": pos.code,
                "name_en": pos.name_en,
                "name_tl": pos.name_tl,
                "description": pos.description
            })
            
        # Return response
        return success_response(results)
    except Exception as e:
        logger.error(f"Error in get_parts_of_speech: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve parts of speech")

@bp.route("/api/v2/statistics", methods=["GET"])
@cached(prefix="statistics", ttl=3600)  # Cache for 1 hour
def get_statistics():
    """
    Get comprehensive dictionary statistics.
    
    Returns:
        Dictionary statistics including:
        - Total counts for words, definitions, etymologies, relations, etc.
        - Language-specific statistics
        - Part of speech distribution
        - Relation type distribution
        - Affix type distribution
        - Baybayin coverage
        - Etymology coverage
        - Daily additions
        - Quality metrics
    """
    try:
        # Use a single SQL query for efficiency
        stats = db.session.execute(text("""
            WITH pos_stats AS (
                SELECT 
                    p.code,
                    p.name_en,
                    p.name_tl,
                    COUNT(DISTINCT d.word_id) as word_count,
                    COUNT(d.id) as definition_count,
                    COUNT(DISTINCT dr.word_id) as related_words_count
                FROM parts_of_speech p
                LEFT JOIN definitions d ON p.id = d.standardized_pos_id
                LEFT JOIN definition_relations dr ON d.id = dr.definition_id
                GROUP BY p.id, p.code, p.name_en, p.name_tl
                ORDER BY p.code
            ),
            relation_stats AS (
                SELECT 
                    relation_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT from_word_id) as from_words_count,
                    COUNT(DISTINCT to_word_id) as to_words_count,
                    COUNT(CASE WHEN metadata IS NOT NULL AND metadata != '{}'::jsonb THEN 1 END) as with_metadata_count
                FROM relations
                GROUP BY relation_type
                ORDER BY count DESC
            ),
            language_stats AS (
                SELECT 
                    language_code,
                    COUNT(*) as word_count,
                    COUNT(CASE WHEN has_baybayin THEN 1 END) as baybayin_count,
                    COUNT(CASE WHEN pronunciation_data IS NOT NULL THEN 1 END) as pronunciation_count,
                    COUNT(CASE WHEN idioms != '[]' THEN 1 END) as idioms_count,
                    COUNT(CASE WHEN source_info != '{}' THEN 1 END) as with_sources_count,
                    MIN(created_at) as first_added,
                    MAX(created_at) as last_added
                FROM words
                GROUP BY language_code
            ),
            affixation_stats AS (
                SELECT 
                    affix_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT root_word_id) as root_words_count,
                    COUNT(DISTINCT affixed_word_id) as affixed_words_count,
                    array_agg(DISTINCT sources) as source_types
                FROM affixations
                GROUP BY affix_type
                ORDER BY count DESC
            ),
            daily_stats AS (
                SELECT 
                    date_trunc('day', created_at) as day,
                    COUNT(*) as words_added,
                    COUNT(CASE WHEN has_baybayin THEN 1 END) as baybayin_added
                FROM words
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY day
                ORDER BY day
            ),
            etymology_stats AS (
                SELECT 
                    COUNT(DISTINCT e.word_id) as words_with_etymology,
                    COUNT(DISTINCT e.language_codes) as unique_language_codes,
                    COUNT(CASE WHEN e.normalized_components IS NOT NULL THEN 1 END) as with_components,
                    COUNT(CASE WHEN e.etymology_structure IS NOT NULL THEN 1 END) as with_structure
                FROM etymologies e
            ),
            definition_stats AS (
                SELECT 
                    COUNT(CASE WHEN examples IS NOT NULL THEN 1 END) as with_examples,
                    COUNT(CASE WHEN usage_notes IS NOT NULL THEN 1 END) as with_usage_notes,
                    COUNT(CASE WHEN tags IS NOT NULL THEN 1 END) as with_tags,
                    AVG(array_length(regexp_split_to_array(definition_text, E'\\\\s+'), 1)) as avg_length
                FROM definitions
            ),
            quality_metrics AS (
                SELECT
                    COUNT(CASE WHEN data_quality_score >= 80 THEN 1 END) as high_quality_entries,
                    COUNT(CASE WHEN data_quality_score >= 50 AND data_quality_score < 80 THEN 1 END) as medium_quality_entries,
                    COUNT(CASE WHEN data_quality_score < 50 THEN 1 END) as low_quality_entries,
                    AVG(data_quality_score) as avg_quality_score
                FROM (
                    SELECT w.id,
                           CASE
                               WHEN w.has_baybayin THEN 15
                               ELSE 0
                           END +
                           CASE
                               WHEN w.pronunciation_data IS NOT NULL THEN 10
                               ELSE 0
                           END +
                           CASE
                               WHEN w.idioms != '[]' THEN 10
                               ELSE 0
                           END +
                           CASE
                               WHEN EXISTS (SELECT 1 FROM etymologies e WHERE e.word_id = w.id) THEN 20
                               ELSE 0
                           END +
                           CASE
                               WHEN EXISTS (SELECT 1 FROM definitions d WHERE d.word_id = w.id) THEN 25
                               ELSE 0
                           END +
                           CASE
                               WHEN EXISTS (SELECT 1 FROM relations r WHERE r.from_word_id = w.id OR r.to_word_id = w.id) THEN 20
                               ELSE 0
                           END as data_quality_score
                    FROM words w
                ) quality_scores
            )
            SELECT
                (SELECT COUNT(*) FROM words) as total_words,
                (SELECT COUNT(*) FROM words WHERE has_baybayin = true) as baybayin_words,
                (SELECT COUNT(*) FROM definitions) as total_definitions,
                (SELECT COUNT(*) FROM etymologies) as total_etymologies,
                (SELECT COUNT(*) FROM relations) as total_relations,
                (SELECT COUNT(*) FROM definition_relations) as total_definition_relations,
                (SELECT COUNT(*) FROM affixations) as total_affixations,
                (SELECT COUNT(*) FROM words WHERE pronunciation_data IS NOT NULL) as total_pronunciations,
                (SELECT COUNT(*) FROM words WHERE idioms != '[]') as total_idioms,
                (SELECT json_agg(pos_stats) FROM pos_stats) as pos_distribution,
                (SELECT json_agg(relation_stats) FROM relation_stats) as relation_distribution,
                (SELECT json_agg(language_stats) FROM language_stats) as language_distribution,
                (SELECT json_agg(affixation_stats) FROM affixation_stats) as affixation_distribution,
                (SELECT json_agg(daily_stats) FROM daily_stats) as daily_additions,
                (SELECT json_agg(etymology_stats) FROM etymology_stats) as etymology_stats,
                (SELECT json_agg(definition_stats) FROM definition_stats) as definition_stats,
                (SELECT json_agg(quality_metrics) FROM quality_metrics) as quality_metrics,
                (SELECT COUNT(DISTINCT language_code) FROM words) as language_count,
                (SELECT COUNT(*) FROM words WHERE root_word_id IS NULL) as root_words_count,
                (SELECT COUNT(*) FROM words WHERE root_word_id IS NOT NULL) as derived_words_count,
                (SELECT COUNT(DISTINCT root_word_id) FROM affixations) as words_with_affixes,
                (SELECT COUNT(DISTINCT word_id) FROM definitions WHERE examples IS NOT NULL) as words_with_examples,
                (SELECT COUNT(DISTINCT word_id) FROM definitions WHERE usage_notes IS NOT NULL) as words_with_usage_notes
        """)).fetchone()
        
        # Format the response
        return success_response({
            "totals": {
                "words": stats.total_words,
                "definitions": stats.total_definitions,
                "etymologies": stats.total_etymologies,
                "relations": stats.total_relations,
                "definition_relations": stats.total_definition_relations,
                "affixations": stats.total_affixations,
                "pronunciations": stats.total_pronunciations,
                "idioms": stats.total_idioms,
                "root_words": stats.root_words_count,
                "derived_words": stats.derived_words_count,
                "words_with_affixes": stats.words_with_affixes,
                "words_with_examples": stats.words_with_examples,
                "words_with_usage_notes": stats.words_with_usage_notes
            },
            "baybayin": {
                "total": stats.baybayin_words,
                "percentage": round((stats.baybayin_words / stats.total_words * 100), 2) if stats.total_words > 0 else 0
            },
            "languages": {
                "count": stats.language_count,
                "distribution": stats.language_distribution
            },
            "parts_of_speech": stats.pos_distribution,
            "relations": stats.relation_distribution,
            "affixations": stats.affixation_distribution,
            "daily_additions": stats.daily_additions,
            "etymology": stats.etymology_stats[0] if stats.etymology_stats else None,
            "definitions": stats.definition_stats[0] if stats.definition_stats else None,
            "quality": {
                "metrics": stats.quality_metrics[0] if stats.quality_metrics else None,
                "distribution": {
                    "high_quality": {
                        "count": stats.quality_metrics[0]["high_quality_entries"] if stats.quality_metrics else 0,
                        "percentage": round((stats.quality_metrics[0]["high_quality_entries"] / stats.total_words * 100), 2) if stats.total_words > 0 and stats.quality_metrics else 0
                    },
                    "medium_quality": {
                        "count": stats.quality_metrics[0]["medium_quality_entries"] if stats.quality_metrics else 0,
                        "percentage": round((stats.quality_metrics[0]["medium_quality_entries"] / stats.total_words * 100), 2) if stats.total_words > 0 and stats.quality_metrics else 0
                    },
                    "low_quality": {
                        "count": stats.quality_metrics[0]["low_quality_entries"] if stats.quality_metrics else 0,
                        "percentage": round((stats.quality_metrics[0]["low_quality_entries"] / stats.total_words * 100), 2) if stats.total_words > 0 and stats.quality_metrics else 0
                    },
                    "average_score": round(stats.quality_metrics[0]["avg_quality_score"], 2) if stats.quality_metrics else 0
                }
            }
        })
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve statistics")

@bp.route("/api/v2/random", methods=["GET"])
@validate_query_params(RandomQuerySchema)
def get_random_word(**params):
    """
    Get a random word with optional filters.
    
    Query Parameters:
        language (str): Language code (default: 'tl')
        has_baybayin (bool): Filter for words with Baybayin
        has_etymology (bool): Filter for words with etymology
        has_definitions (bool): Filter for words with definitions (default: True)
        include_definitions (bool): Include definitions in response (default: True)
        include_relations (bool): Include relations in response (default: True)
        include_etymology (bool): Include etymology in response (default: True)
        include_metadata (bool): Include additional metadata (default: True)
    
    Returns:
        Random word details
    """
    try:
        # Extract parameters
        language = params.get("language", "tl")
        has_baybayin = params.get("has_baybayin")
        has_etymology = params.get("has_etymology", False)
        has_definitions = params.get("has_definitions", True)
        include_definitions = params.get("include_definitions", True)
        include_relations = params.get("include_relations", True)
        include_etymology = params.get("include_etymology", True)
        include_metadata = params.get("include_metadata", True)
        
        # Build query
        query = Word.query
        
        # Apply filters
        if language:
            query = query.filter(Word.language_code == language)
            
        if has_baybayin is not None:
            query = query.filter(Word.has_baybayin == has_baybayin)
            
        if has_etymology:
            query = query.join(Word.etymologies)
            
        if has_definitions:
            query = query.join(Word.definitions)
        
        # Get a random word
        word = query.order_by(func.random()).first()
        
        # Check if a word was found
        if not word:
            return error_response("No words found matching criteria", 404)
        
        # Load related data
        if include_definitions and not has_definitions:
            db.session.query(Definition).options(
                joinedload(Definition.standardized_pos)
            ).filter(Definition.word_id == word.id).all()
            
        if include_etymology and not has_etymology:
            db.session.query(Etymology).filter(Etymology.word_id == word.id).all()
            
        if include_relations:
            db.session.query(Relation).options(
                joinedload(Relation.to_word)
            ).filter(Relation.from_word_id == word.id).all()
            
            db.session.query(Relation).options(
                joinedload(Relation.from_word)
            ).filter(Relation.to_word_id == word.id).all()
            
            db.session.query(Affixation).options(
                joinedload(Affixation.affixed_word)
            ).filter(Affixation.root_word_id == word.id).all()
            
            db.session.query(Affixation).options(
                joinedload(Affixation.root_word)
            ).filter(Affixation.affixed_word_id == word.id).all()
        
        # Convert word to dictionary
        word_data = word.to_dict(
            include_definitions=include_definitions,
            include_relations=include_relations,
            include_etymology=include_etymology,
            include_metadata=include_metadata
        )
        
        # Return response
        return success_response(
            word_data,
            meta={
                "random": True,
                "filters": {
                    "language": language,
                    "has_baybayin": has_baybayin,
                    "has_etymology": has_etymology,
                    "has_definitions": has_definitions
                }
            }
        )
    except Exception as e:
        logger.error(f"Error in get_random_word: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve random word")

@bp.route("/api/v2/words/<path:word>/related", methods=["GET"])
@cached(prefix="word_relationships", ttl=3600)
@validate_query_params(WordRelationshipSchema)
def get_word_relationship_graph(word, **params):
    """
    Get detailed relationship graph for a word.
    
    Path Parameters:
        word (str): The word to build relationships for
        
    Query Parameters:
        depth (int): Maximum relationship depth (default: 1, max: 3)
        include_affixes (bool): Include affix relationships (default: True)
        include_etymology (bool): Include etymology relationships (default: True)
        include_definitions (bool): Include definition details (default: False)
        cluster_threshold (float): Similarity threshold for clustering (default: 0.3)
    
    Returns:
        Word relationship graph with nodes and edges
    """
    try:
        # Check rate limit for relationship graph
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:word_relationships")
        if not is_allowed:
            return rate_limit_exceeded_response()
            
        # Extract parameters
        depth = params.get("depth", 1)
        include_affixes = params.get("include_affixes", True)
        include_etymology = params.get("include_etymology", True)
        include_definitions = params.get("include_definitions", False)
        cluster_threshold = params.get("cluster_threshold", 0.3)
        
        # Find word with eager loading
        word_entry = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.standardized_pos) if include_definitions else joinedload(Word.definitions),
            joinedload(Word.etymologies),
            joinedload(Word.relations_from).joinedload(Relation.to_word),
            joinedload(Word.relations_to).joinedload(Relation.from_word),
            joinedload(Word.affixations_as_root).joinedload(Affixation.affixed_word),
            joinedload(Word.affixations_as_affixed).joinedload(Affixation.root_word)
        ).filter(Word.normalized_lemma == normalize_word(word)).first()
        
        # Check if word exists
        if not word_entry:
            return error_response("Word not found", 404)
            
        # Initialize graph
        graph = {
            "nodes": [],
            "edges": [],
            "clusters": {
                "etymology": [],
                "affixes": [],
                "synonyms": [],
                "antonyms": [],
                "variants": [],
                "root_words": [],
                "derived_words": []
            }
        }
        
        # Track visited words to avoid cycles
        visited = set()
        
        # Function to add node and its relationships recursively
        def add_node(word_obj, level=0, relation_path=None):
            # Skip if already visited or beyond depth limit
            if word_obj.id in visited or level > depth:
                return
                
            # Mark as visited
            visited.add(word_obj.id)
            
            # Create node data
            node = {
                "id": word_obj.id,
                "word": word_obj.lemma,
                "normalized_lemma": word_obj.normalized_lemma,
                "type": "root" if word_obj.id == word_entry.id else "related",
                "has_baybayin": word_obj.has_baybayin,
                "baybayin_form": word_obj.baybayin_form if word_obj.has_baybayin else None,
                "language": word_obj.language_code,
                "path": relation_path or []
            }
            
            # Add definitions if requested
            if include_definitions and word_obj.definitions:
                node["definitions"] = [
                    {
                        "text": d.definition_text,
                        "pos": d.standardized_pos.code if d.standardized_pos else None
                    }
                    for d in word_obj.definitions[:3]  # Limit to 3 definitions
                ]
                
            # Add node to graph
            graph["nodes"].append(node)
            
            # Process outgoing relations
            for rel in word_obj.relations_from:
                if rel.to_word_id not in visited and level < depth:
                    # Add target node
                    new_path = (relation_path or []) + [{"type": rel.relation_type, "word": rel.to_word.lemma}]
                    add_node(rel.to_word, level + 1, new_path)
                    
                    # Add edge
                    edge = {
                        "source": rel.from_word_id,
                        "target": rel.to_word_id,
                        "type": rel.relation_type,
                        "sources": rel.get_sources_list() if hasattr(rel, 'get_sources_list') else []
                    }
                    
                    # Include metadata if available
                    if hasattr(rel, 'metadata') and rel.metadata:
                        edge["metadata"] = rel.metadata
                    
                    graph["edges"].append(edge)
                    
                    # Add to appropriate cluster
                    rel_type = rel.relation_type.lower()
                    if rel_type == "synonym":
                        graph["clusters"]["synonyms"].append(rel.to_word_id)
                    elif rel_type == "antonym":
                        graph["clusters"]["antonyms"].append(rel.to_word_id)
                    elif rel_type == "variant":
                        graph["clusters"]["variants"].append(rel.to_word_id)
                    elif rel_type == "derived_from":
                        graph["clusters"]["root_words"].append(rel.to_word_id)
            
            # Process incoming derived_from relations
            for rel in word_obj.relations_to:
                if rel.relation_type.lower() == "derived_from" and rel.from_word_id not in visited and level < depth:
                    # Add source node
                    new_path = (relation_path or []) + [{"type": "derived", "word": rel.from_word.lemma}]
                    add_node(rel.from_word, level + 1, new_path)
                    
                    # Add edge
                    edge = {
                        "source": rel.from_word_id,
                        "target": rel.to_word_id,
                        "type": "derived",
                        "sources": rel.get_sources_list() if hasattr(rel, 'get_sources_list') else []
                    }
                    
                    # Include metadata if available
                    if hasattr(rel, 'metadata') and rel.metadata:
                        edge["metadata"] = rel.metadata
                    
                    graph["edges"].append(edge)
                    
                    # Add to derived words cluster
                    graph["clusters"]["derived_words"].append(rel.from_word_id)
            
            # Process affixations if requested
            if include_affixes:
                for aff in word_obj.affixations_as_root:
                    if aff.affixed_word_id not in visited and level < depth:
                        # Add affixed word node
                        new_path = (relation_path or []) + [{"type": f"affix_{aff.affix_type}", "word": aff.affixed_word.lemma}]
                        add_node(aff.affixed_word, level + 1, new_path)
                        
                        # Add edge
                        edge = {
                            "source": aff.root_word_id,
                            "target": aff.affixed_word_id,
                            "type": f"affix_{aff.affix_type}",
                            "sources": aff.get_sources_list() if hasattr(aff, 'get_sources_list') else []
                        }
                        graph["edges"].append(edge)
                        
                        # Add to affixes cluster
                        graph["clusters"]["affixes"].append(aff.affixed_word_id)
                
                # Process as affixed word (to find root)
                for aff in word_obj.affixations_as_affixed:
                    if aff.root_word_id not in visited and level < depth:
                        # Add root word node
                        new_path = (relation_path or []) + [{"type": f"root_of_{aff.affix_type}", "word": aff.root_word.lemma}]
                        add_node(aff.root_word, level + 1, new_path)
                        
                        # Add edge
                        edge = {
                            "source": aff.affixed_word_id,
                            "target": aff.root_word_id,
                            "type": f"root_of_{aff.affix_type}",
                            "sources": aff.get_sources_list() if hasattr(aff, 'get_sources_list') else []
                        }
                        graph["edges"].append(edge)
                        
                        # Add to root words cluster
                        graph["clusters"]["root_words"].append(aff.root_word_id)
            
            # Process etymology components if requested
            if include_etymology and word_obj.etymologies:
                for etym in word_obj.etymologies:
                    components = parse_components_field(etym.normalized_components)
                    for comp in components:
                        comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                        if comp_word and comp_word.id not in visited and level < depth:
                            # Add component word node
                            new_path = (relation_path or []) + [{"type": "etymology", "word": comp_word.lemma}]
                            add_node(comp_word, level + 1, new_path)
                            
                            # Add edge
                            edge = {
                                "source": word_obj.id,
                                "target": comp_word.id,
                                "type": "etymology",
                                "sources": etym.get_sources_list() if hasattr(etym, 'get_sources_list') else []
                            }
                            graph["edges"].append(edge)
                            
                            # Add to etymology cluster
                            graph["clusters"]["etymology"].append(comp_word.id)
        
        # Start building graph from the root word
        add_node(word_entry)
        
        # Clean up clusters (remove duplicates)
        for cluster_name, cluster_ids in graph["clusters"].items():
            graph["clusters"][cluster_name] = list(set(cluster_ids))
        
        # Return response
        return success_response(
            graph,
            meta={
                "root_word": word_entry.lemma,
                "normalized_lemma": word_entry.normalized_lemma,
                "language_code": word_entry.language_code,
                "depth": depth,
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
@cached(prefix="etymology_tree", ttl=3600)
@validate_query_params(EtymologyTreeSchema)
def get_etymology_tree(word, **params):
    """
    Get the complete etymology tree for a word.
    
    Path Parameters:
        word (str): The word to build etymology tree for
        
    Query Parameters:
        max_depth (int): Maximum tree depth (default: 3, max: 5)
        include_uncertain (bool): Include uncertain etymologies (default: False)
        group_by_language (bool): Group components by language (default: False)
    
    Returns:
        Etymology tree showing word origins
    """
    try:
        # Check rate limit for etymology tree
        is_allowed, remaining = check_rate_limit(f"rate_limit:{request.remote_addr}:etymology_tree")
        if not is_allowed:
            return rate_limit_exceeded_response()
            
        # Extract parameters
        max_depth = params.get("max_depth", 3)
        include_uncertain = params.get("include_uncertain", False)
        group_by_language = params.get("group_by_language", False)
            
        # Find word with eager loading of etymologies
        word_entry = Word.query.options(
            joinedload(Word.etymologies)
        ).filter(Word.normalized_lemma == normalize_word(word)).first()
        
        # Check if word exists
        if not word_entry:
            return error_response("Word not found", 404)

        # Function to build etymology tree recursively
        def build_etymology_tree(word_obj, visited=None, current_depth=0):
            if visited is None:
                visited = set()
                
            # Skip if already visited or beyond depth limit
            if word_obj.id in visited or current_depth >= max_depth:
                return None
                
            # Mark as visited
            visited.add(word_obj.id)
            
            # Create tree node
            tree = {
                "id": word_obj.id,
                "word": word_obj.lemma,
                "normalized_lemma": word_obj.normalized_lemma,
                "language": word_obj.language_code,
                "has_baybayin": word_obj.has_baybayin,
                "baybayin_form": word_obj.baybayin_form if word_obj.has_baybayin else None,
                "romanized_form": word_obj.romanized_form if word_obj.has_baybayin else None,
                "etymologies": [],
                "components": [],
                "component_words": []  # Renamed from component_objects for clarity
            }
            
            # Process etymologies
            for etym in word_obj.etymologies:
                # Create etymology entry
                etymology_entry = {
                    "id": etym.id,
                    "text": etym.etymology_text,
                    "languages": etym.get_language_codes_list() if hasattr(etym, 'get_language_codes_list') else [],
                    "sources": etym.get_sources_list() if hasattr(etym, 'get_sources_list') else []
                }
                
                # Extract components
                components = parse_components_field(etym.normalized_components)
                
                # If no components were found in the normalized_components field,
                # try to extract them directly from the etymology text
                if not components:
                    components = extract_components_from_text(etym.etymology_text)
                
                # Add component strings to the top-level components array
                tree["components"].extend(components)
                
                # Group components by language if requested
                if group_by_language:
                    components_by_lang = {}
                    
                    for comp in components:
                        comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                        if comp_word and comp_word.id not in visited and current_depth < max_depth:
                            lang = comp_word.language_code
                            if lang not in components_by_lang:
                                components_by_lang[lang] = []
                                
                            comp_tree = build_etymology_tree(comp_word, visited, current_depth + 1)
                            if comp_tree:
                                components_by_lang[lang].append(comp_tree)
                    
                    # Add components grouped by language
                    etymology_entry["components_by_language"] = components_by_lang
                else:
                    # Process components individually
                    for comp in components:
                        comp_word = Word.query.filter(Word.normalized_lemma == normalize_word(comp)).first()
                        if comp_word and comp_word.id not in visited and current_depth < max_depth:
                            comp_tree = build_etymology_tree(comp_word, visited, current_depth + 1)
                            if comp_tree:
                                # Add a reference to the component word
                                tree["component_words"].append({
                                    "component": comp,
                                    "word": comp_tree
                                })
                
                tree["etymologies"].append(etymology_entry)
            
            # Deduplicate string components
            tree["components"] = list(set(tree["components"]))
            
            # Sort components alphabetically for consistency
            tree["components"].sort()
            
            return tree
        
        # Build etymology tree
        etymology_tree = build_etymology_tree(word_entry)
        
        # Return response
        return success_response(
            etymology_tree,
            meta={
                "word": word_entry.lemma,
                "normalized_lemma": word_entry.normalized_lemma,
                "language": word_entry.language_code,
                "max_depth": max_depth,
                "group_by_language": group_by_language
            }
        )
    except ValidationError as err:
        return error_response("Invalid parameters", 400, err.messages, "ERR_VALIDATION")
    except Exception as e:
        logger.error(f"Error in get_etymology_tree: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve etymology tree")

@bp.route("/api/v2/affixes", methods=["GET"])
@cached(prefix="affixes", ttl=86400)  # Cache for 24 hours
def get_affixes():
    """
    Get all available affixes and their usage statistics.
    
    Query Parameters:
        language (str): Language code filter (default: 'tl')
        type (str): Filter by affix type
    
    Returns:
        List of affixes with usage statistics
    """
    try:
        # Extract parameters
        language = request.args.get("language", "tl")
        affix_type = request.args.get("type")
        
        # Build query for affix statistics with a simpler approach
        sql = text("""
            SELECT 
                a.affix_type,
                COUNT(*) as usage_count,
                COUNT(DISTINCT a.root_word_id) as root_words_count,
                COUNT(DISTINCT a.affixed_word_id) as affixed_words_count,
                array_to_json(array(
                    SELECT DISTINCT w.lemma
                    FROM affixations a2
                    JOIN words w ON a2.root_word_id = w.id
                    WHERE a2.affix_type = a.affix_type
                    AND w.language_code = :language
                    LIMIT 5
                )) as sample_roots,
                array_to_json(array(
                    SELECT DISTINCT w2.lemma
                    FROM affixations a3
                    JOIN words w2 ON a3.affixed_word_id = w2.id
                    WHERE a3.affix_type = a.affix_type
                    AND w2.language_code = :language
                    LIMIT 5
                )) as sample_affixed
            FROM affixations a
            JOIN words w ON a.root_word_id = w.id
            JOIN words w2 ON a.affixed_word_id = w2.id
            WHERE w.language_code = :language
            AND (:affix_type IS NULL OR a.affix_type = :affix_type)
            GROUP BY a.affix_type
            ORDER BY usage_count DESC
        """)
        
        # Execute query
        results = db.session.execute(
            sql,
            {
                "language": language,
                "affix_type": affix_type
            }
        ).fetchall()
        
        # Format results
        affixes = []
        for row in results:
            # Get affix type description
            description = get_affix_description(row.affix_type)
            
            # Format result
            affixes.append({
                "type": row.affix_type,
                "description": description,
                "usage_count": row.usage_count,
                "root_words_count": row.root_words_count,
                "affixed_words_count": row.affixed_words_count,
                "sample_roots": row.sample_roots if row.sample_roots else [],
                "sample_affixed": row.sample_affixed if row.sample_affixed else []
            })
        
        # Return response
        return success_response(
            affixes,
            meta={
                "language": language,
                "affix_type": affix_type,
                "total": len(affixes)
            }
        )
    except Exception as e:
        logger.error(f"Error in get_affixes: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve affixes")

@bp.route("/api/v2/relations", methods=["GET"])
@cached(prefix="relations", ttl=86400)  # Cache for 24 hours
def get_relations():
    """
    Get all available relation types and their usage statistics.
    
    Query Parameters:
        language (str): Language code filter (default: 'tl')
        type (str): Filter by relation type
    
    Returns:
        List of relation types with usage statistics
    """
    try:
        # Extract parameters
        language = request.args.get("language", "tl")
        relation_type = request.args.get("type")
        
        # Build query for relation statistics with a simpler approach
        sql = text("""
            SELECT 
                r.relation_type,
                COUNT(*) as usage_count,
                COUNT(DISTINCT r.from_word_id) as from_words_count,
                COUNT(DISTINCT r.to_word_id) as to_words_count,
                SUM(CASE WHEN r.metadata IS NOT NULL AND r.metadata != '{}'::jsonb THEN 1 ELSE 0 END) as with_metadata_count,
                array_to_json(array(
                    SELECT DISTINCT w.lemma
                    FROM relations r2
                    JOIN words w ON r2.from_word_id = w.id
                    WHERE r2.relation_type = r.relation_type
                    AND w.language_code = :language
                    LIMIT 5
                )) as sample_from,
                array_to_json(array(
                    SELECT DISTINCT w2.lemma
                    FROM relations r3
                    JOIN words w2 ON r3.to_word_id = w2.id
                    WHERE r3.relation_type = r.relation_type
                    AND w2.language_code = :language
                    LIMIT 5
                )) as sample_to
            FROM relations r
            JOIN words w ON r.from_word_id = w.id
            JOIN words w2 ON r.to_word_id = w2.id
            WHERE w.language_code = :language
            AND (:relation_type IS NULL OR r.relation_type = :relation_type)
            GROUP BY r.relation_type
            ORDER BY usage_count DESC
        """)
        
        # Execute query
        results = db.session.execute(
            sql,
            {
                "language": language,
                "relation_type": relation_type
            }
        ).fetchall()
        
        # Format results
        relations = []
        for row in results:
            # Get relation type description
            description = get_relation_description(row.relation_type)
            
            # Format result
            relations.append({
                "type": row.relation_type,
                "description": description,
                "usage_count": row.usage_count,
                "from_words_count": row.from_words_count,
                "to_words_count": row.to_words_count,
                "with_metadata_count": row.with_metadata_count,
                "has_metadata": row.with_metadata_count > 0,
                "sample_from": row.sample_from if row.sample_from else [],
                "sample_to": row.sample_to if row.sample_to else []
            })
        
        # Return response
        return success_response(
            relations,
            meta={
                "language": language,
                "relation_type": relation_type,
                "total": len(relations)
            }
        )
    except Exception as e:
        logger.error(f"Error in get_relations: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve relations")

@bp.route("/favicon.ico")
def favicon():
    """Serve favicon or return empty response."""
    return "", 204

@bp.teardown_request
def remove_session(exception=None):
    """Clean up database session after request."""
    db.session.remove()

def rate_limit_exceeded_response(window: int = 60) -> tuple:
    """Generate standardized rate limit exceeded response."""
    return error_response(
        "Rate limit exceeded", 
        429, 
        {"retry_after": window, "window_seconds": window}, 
        "ERR_RATE_LIMIT"
    )

def get_affix_description(affix_type: str) -> str:
    """Get a description for an affix type."""
    descriptions = {
        "prefix": "An affix attached to the beginning of a word",
        "infix": "An affix inserted within a word",
        "suffix": "An affix attached to the end of a word",
        "circumfix": "An affix attached to both the beginning and end of a word",
        "reduplication": "Repetition of a syllable or word element",
        "compound": "Combination of two or more words to form a new word"
    }
    return descriptions.get(affix_type, "")

def get_relation_description(relation_type: str) -> str:
    """Get a description for a relation type."""
    descriptions = {
        # Basic semantic relationships
        "synonym": "Words with the same or similar meaning",
        "antonym": "Words with opposite meanings",
        "variant": "Alternative forms or spellings of the same word",
        "spelling_variant": "Alternative spelling of the same word",
        
        # Hierarchical relationships
        "hypernym_of": "Word with a broader meaning that includes more specific words",
        "hyponym_of": "Word with a more specific meaning included in a broader term",
        "meronym_of": "Word that denotes a part of another word's concept",
        "holonym_of": "Word that denotes a whole containing the part denoted by another word",
        
        # Derivational relationships
        "derived_from": "Word formed from another word through derivation",
        "root_of": "Base word from which other words are derived",
        "derived": "Word derived from a base word through affixation or other morphological processes",
        "base_of": "Base word from which another word is derived",
        
        # Etymology relationships
        "cognate": "Words sharing the same linguistic origin",
        "borrowed_from": "Word borrowed or adopted from another language",
        "loaned_to": "Word that has been adopted by another language",
        "descendant_of": "Word evolved from an earlier form",
        "ancestor_of": "Word from which another word evolved",
        
        # Structural relationships
        "component_of": "Word that forms part of another word or phrase",
        "abbreviation_of": "Shortened form of a word or phrase",
        "has_abbreviation": "Word or phrase that has a shortened form",
        "initialism_of": "Acronym formed from initial letters of a phrase",
        "has_initialism": "Phrase that has an acronym form",
        
        # General relationships
        "related": "Words with a semantic or etymological connection"
    }
    return descriptions.get(relation_type, "Relationship between words")

@bp.route("/api/v2/words/<path:word>/pronunciation", methods=["GET"])
@cached(prefix="word_pronunciation", ttl=3600)
@validate_query_params(PronunciationSchema)
def get_word_pronunciation(word, **params):
    """Get pronunciation details for a word."""
    try:
        word_entry = Word.query.filter(
            Word.normalized_lemma == normalize_word(word),
            Word.language_code == params.get("language", "tl")
        ).first()
        
        if not word_entry:
            return error_response("Word not found", 404)
            
        if not word_entry.pronunciation_data:
            return error_response("No pronunciation data available", 404)
            
        # Extract pronunciation data based on parameters
        pronunciation = {}
        if params.get("include_ipa", True) and "ipa" in word_entry.pronunciation_data:
            pronunciation["ipa"] = word_entry.pronunciation_data["ipa"]
            
        if params.get("include_audio", True) and "audio" in word_entry.pronunciation_data:
            pronunciation["audio"] = word_entry.pronunciation_data["audio"]
            
        if params.get("include_hyphenation", True) and "hyphenation" in word_entry.pronunciation_data:
            pronunciation["hyphenation"] = word_entry.pronunciation_data["hyphenation"]
            
        if "sounds" in word_entry.pronunciation_data:
            pronunciation["sounds"] = word_entry.pronunciation_data["sounds"]
            
        return success_response(pronunciation)
    except Exception as e:
        logger.error(f"Error in get_word_pronunciation: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve pronunciation data")

@bp.route("/api/v2/words/<path:word>/etymology/details", methods=["GET"])
@cached(prefix="etymology_details", ttl=3600)
@validate_query_params(EtymologySchema)
def get_etymology_details(word, **params):
    """Get detailed etymology information for a word."""
    try:
        word_entry = Word.query.options(
            joinedload(Word.etymologies)
        ).filter(
            Word.normalized_lemma == normalize_word(word),
            Word.language_code == params.get("language", "tl")
        ).first()
        
        if not word_entry:
            return error_response("Word not found", 404)
            
        if not word_entry.etymologies:
            return error_response("No etymology data available", 404)
            
        etymology_data = []
        for etymology in word_entry.etymologies:
            entry = {
                "text": etymology.etymology_text,
                "sources": etymology.get_sources_list()
            }
            
            if params.get("include_components", True):
                entry["components"] = etymology.get_components_list()
                
            if params.get("include_language_codes", True):
                entry["language_codes"] = etymology.get_language_codes_list()
                
            if params.get("include_structure", False) and etymology.etymology_structure:
                try:
                    entry["structure"] = json.loads(etymology.etymology_structure)
                except json.JSONDecodeError:
                    entry["structure"] = None
                    
            etymology_data.append(entry)
            
        # Group by language if requested
        if params.get("group_by_language", False):
            grouped_data = {}
            for etym in etymology_data:
                for lang in etym.get("language_codes", []):
                    if lang not in grouped_data:
                        grouped_data[lang] = []
                    grouped_data[lang].append(etym)
            return success_response(grouped_data)
            
        return success_response(etymology_data)
    except Exception as e:
        logger.error(f"Error in get_etymology_details: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve etymology details")

@bp.route("/api/v2/words/<path:word>/idioms", methods=["GET"])
@cached(prefix="word_idioms", ttl=3600)
def get_word_idioms(word):
    """Get idioms associated with a word."""
    try:
        word_entry = Word.query.filter(
            Word.normalized_lemma == normalize_word(word)
        ).first()
        
        if not word_entry:
            return error_response("Word not found", 404)
            
        idioms = word_entry.get_idioms_list()
        if not idioms:
            return error_response("No idioms found for this word", 404)
            
        return success_response(idioms)
    except Exception as e:
        logger.error(f"Error in get_word_idioms: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve idioms")

@bp.route("/api/v2/words/<path:word>/definition-relations", methods=["GET"])
@cached(prefix="definition_relations", ttl=3600)
def get_definition_relations(word):
    """Get relations specific to word definitions."""
    try:
        word_entry = Word.query.options(
            joinedload(Word.definitions).joinedload(Definition.definition_relations)
        ).filter(
            Word.normalized_lemma == normalize_word(word)
        ).first()
        
        if not word_entry:
            return error_response("Word not found", 404)
            
        relations = []
        for definition in word_entry.definitions:
            def_relations = []
            for relation in definition.definition_relations:
                def_relations.append({
                    "word": relation.word.lemma,
                    "type": relation.relation_type,
                    "sources": relation.get_sources_list()
                })
            
            if def_relations:
                relations.append({
                    "definition": definition.definition_text,
                    "relations": def_relations
                })
                
        return success_response(relations)
    except Exception as e:
        logger.error(f"Error in get_definition_relations: {str(e)}", exc_info=True)
        return error_response("Failed to retrieve definition relations")

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