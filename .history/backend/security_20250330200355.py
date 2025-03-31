"""
Comprehensive security configuration for the application.
Includes rate limiting, CORS, request validation, and security headers.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Callable, Type, Union
from functools import wraps
from datetime import datetime, timedelta
import jwt
from flask import request, current_app, g, abort, Response
from werkzeug.security import generate_password_hash, check_password_hash
import redis
from prometheus_client import Counter
import logging
import re
import hashlib
import secrets
from dataclasses import dataclass
from marshmallow import Schema, fields, validate, ValidationError
from flask import jsonify
import structlog

# Set up logging
logger = structlog.get_logger(__name__)

# Security Metrics
BLOCKED_REQUESTS = Counter('blocked_requests_total', 'Blocked requests', ['reason'])
AUTH_FAILURES = Counter('auth_failures_total', 'Authentication failures', ['type'])
RATE_LIMITS = Counter('rate_limits_total', 'Rate limit hits', ['endpoint'])

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = 'HS256'
    JWT_EXPIRATION: int = 3600  # 1 hour
    RATE_LIMIT_DEFAULT: str = '200 per day'
    RATE_LIMIT_SEARCH: str = '60 per minute'
    RATE_LIMIT_WRITE: str = '30 per minute'
    MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_ORIGINS: List[str] = None
    ALLOWED_HEADERS: List[str] = None
    CORS_MAX_AGE: int = 600
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_SPECIAL: bool = True
    REQUEST_TIMEOUT: int = 30
    MAX_NESTING_DEPTH: int = 10
    MAX_ARRAY_LENGTH: int = 1000
    SANITIZE_PATTERNS: List[Tuple[str, str]] = None
    CSP_POLICY: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.ALLOWED_ORIGINS is None:
            self.ALLOWED_ORIGINS = [
                'http://localhost:3000',
                'https://example.com'
            ]
        if self.ALLOWED_HEADERS is None:
            self.ALLOWED_HEADERS = [
                'Content-Type',
                'Authorization',
                'X-Request-ID',
                'X-Client-Version'
            ]
        if self.SANITIZE_PATTERNS is None:
            self.SANITIZE_PATTERNS = [
                (r'<script.*?>.*?</script>', ''),  # Remove script tags
                (r'javascript:', ''),  # Remove javascript: protocol
                (r'on\w+=".*?"', ''),  # Remove event handlers
                (r'data:', 'data-safe:')  # Sanitize data: URLs
            ]
        if self.CSP_POLICY is None:
            self.CSP_POLICY = {
                'default-src': ["'self'"],
                'script-src': ["'self'", "'unsafe-inline'"],
                'style-src': ["'self'", "'unsafe-inline'"],
                'img-src': ["'self'", 'data:'],
                'font-src': ["'self'"],
                'connect-src': ["'self'"]
            }

class RateLimiter:
    """Rate limiting implementation using Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_limit = (200, 'day')  # 200 requests per day
        self.limits = {
            'search': (60, 'minute'),  # 60 requests per minute
            'write': (30, 'minute')    # 30 requests per minute
        }

    def _get_time_window(self, period: str) -> int:
        """Convert time period to seconds."""
        periods = {
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        return periods.get(period, 86400)

    def is_rate_limited(self, key: str, endpoint_type: str = 'default') -> bool:
        """Check if request should be rate limited."""
        limit, period = self.limits.get(endpoint_type, self.default_limit)
        window = self._get_time_window(period)
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        current_time = datetime.utcnow().timestamp()
        
        # Clean old requests
        pipe.zremrangebyscore(key, 0, current_time - window)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Count requests in window
        pipe.zcard(key)
        
        # Set key expiration
        pipe.expire(key, window)
        
        # Execute pipeline
        _, _, request_count, _ = pipe.execute()
        
        if request_count > limit:
            RATE_LIMITS.labels(endpoint=endpoint_type).inc()
            return True
            
        return False

class RequestValidator:
    """Validate and sanitize incoming requests."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sanitize_patterns = [
            (re.compile(pattern), repl)
            for pattern, repl in config.SANITIZE_PATTERNS
        ]

    def validate_content_length(self, request) -> bool:
        """Check if request content length is within limits."""
        content_length = request.content_length or 0
        if content_length > self.config.MAX_CONTENT_LENGTH:
            BLOCKED_REQUESTS.labels(reason='content_length').inc()
            return False
        return True

    def validate_json_structure(self, data: Any, depth: int = 0) -> bool:
        """Validate JSON structure depth and array lengths."""
        if depth > self.config.MAX_NESTING_DEPTH:
            BLOCKED_REQUESTS.labels(reason='nesting_depth').inc()
            return False
            
        if isinstance(data, dict):
            return all(
                self.validate_json_structure(value, depth + 1)
                for value in data.values()
            )
        elif isinstance(data, list):
            if len(data) > self.config.MAX_ARRAY_LENGTH:
                BLOCKED_REQUESTS.labels(reason='array_length').inc()
                return False
            return all(
                self.validate_json_structure(item, depth + 1)
                for item in data
            )
            
        return True

    def sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        result = value
        for pattern, repl in self.sanitize_patterns:
            result = pattern.sub(repl, result)
        return result

    def sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize data structure."""
        if isinstance(data, dict):
            return {
                key: self.sanitize_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        elif isinstance(data, str):
            return self.sanitize_string(data)
        return data

class SecurityHeaders:
    """Security headers management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config

    def get_csp_header(self) -> str:
        """Generate Content Security Policy header."""
        policies = []
        for directive, sources in self.config.CSP_POLICY.items():
            policies.append(f"{directive} {' '.join(sources)}")
        return "; ".join(policies)

    def apply_security_headers(self, response: Response) -> Response:
        """Apply security headers to response."""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Content-Security-Policy'] = self.get_csp_header()
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        return response

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            AUTH_FAILURES.labels(type='missing_key').inc()
            abort(401, description="API key required")
            
        # Validate API key (implement your validation logic)
        if not validate_api_key(api_key):
            AUTH_FAILURES.labels(type='invalid_key').inc()
            abort(401, description="Invalid API key")
            
        return f(*args, **kwargs)
    return decorated

def rate_limit(limit_type: str = 'default'):
    """Decorator to apply rate limiting."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            limiter = current_app.rate_limiter
            key = f"rate_limit:{request.remote_addr}:{f.__name__}"
            
            if limiter.is_rate_limited(key, limit_type):
                abort(429, description="Rate limit exceeded")
                
            return f(*args, **kwargs)
        return decorated
    return decorator

def validate_json_request(schema_cls: Type[Schema]):
    """Decorator to validate JSON request body using marshmallow schema."""
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Ensure request has JSON content type
                if not request.is_json:
                    return {
                        "error": {
                            "message": "Request must be JSON",
                            "code": "ERR_INVALID_CONTENT_TYPE"
                        }
                    }, 415
                
                # Create schema instance
                schema = schema_cls()
                
                # Validate request data
                data = schema.load(request.get_json())
                
                # Add validated data to kwargs
                kwargs['data'] = data
                
                return f(*args, **kwargs)
            except ValidationError as err:
                logger.warning(
                    "JSON request validation failed",
                    error=err.messages,
                    path=request.path,
                    data=request.get_json()
                )
                return {
                    "error": {
                        "message": "Invalid request data",
                        "details": err.messages,
                        "code": "ERR_VALIDATION"
                    }
                }, 400
        return wrapper
    return decorator

def validate_query_params(schema_cls: Type[Schema]):
    """Decorator to validate query parameters using marshmallow schema."""
    def decorator(f: Callable):
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
                logger.warning(
                    "Query parameter validation failed",
                    error=err.messages,
                    path=request.path,
                    params=dict(request.args)
                )
                return {
                    "error": {
                        "message": "Invalid query parameters",
                        "details": err.messages,
                        "code": "ERR_VALIDATION"
                    }
                }, 400
        return wrapper
    return decorator

def init_security(app, redis_client: redis.Redis) -> None:
    """Initialize security components."""
    # Load security configuration
    app.config['SECURITY'] = SecurityConfig(
        JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY', secrets.token_hex(32)),
        ALLOWED_ORIGINS=os.getenv('ALLOWED_ORIGINS', '').split(','),
        RATE_LIMIT_DEFAULT=os.getenv('RATE_LIMIT_DEFAULT', '200 per day'),
        RATE_LIMIT_SEARCH=os.getenv('RATE_LIMIT_SEARCH', '60 per minute'),
        RATE_LIMIT_WRITE=os.getenv('RATE_LIMIT_WRITE', '30 per minute')
    )
    
    # Initialize components
    app.rate_limiter = RateLimiter(redis_client)
    app.request_validator = RequestValidator(app.config['SECURITY'])
    app.security_headers = SecurityHeaders(app.config['SECURITY'])
    
    # Add security middleware
    @app.after_request
    def add_security_headers(response):
        return app.security_headers.apply_security_headers(response)
        
    @app.before_request
    def validate_request_base():
        # Basic request validation
        validator = app.request_validator
        if not validator.validate_content_length(request):
            abort(413, description="Request too large")
            
    # Add CORS support
    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get('Origin')
        if origin in app.config['SECURITY'].ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = ', '.join(
                app.config['SECURITY'].ALLOWED_HEADERS
            )
            response.headers['Access-Control-Max-Age'] = str(
                app.config['SECURITY'].CORS_MAX_AGE
            )
        return response

# Request validation schemas
class SearchSchema(Schema):
    """Schema for search query parameters."""
    query = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    page = fields.Int(validate=validate.Range(min=1, max=100), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    language = fields.Str()  # Accept any valid language code
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)
    include_pronunciation = fields.Bool(default=True)
    include_metadata = fields.Bool(default=True)
    min_quality = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    sort = fields.Str(validate=validate.OneOf([
        'relevance', 'alphabetical', 'created', 'updated',
        'quality', 'frequency', 'complexity'
    ]), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')

class WordSchema(Schema):
    """Schema for word data validation."""
    lemma = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    language_code = fields.Str(required=True)  # Accept any valid language code
    definitions = fields.List(fields.Dict(), validate=validate.Length(max=50))
    etymologies = fields.List(fields.Dict(), validate=validate.Length(max=10))
    has_baybayin = fields.Bool()
    baybayin_form = fields.Str()
    romanized_form = fields.Str()
    preferred_spelling = fields.Str()
    alternative_spellings = fields.List(fields.Str())
    syllable_count = fields.Int()
    pronunciation_guide = fields.Str()
    stress_pattern = fields.Str()
    formality_level = fields.Str()
    usage_frequency = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    geographic_region = fields.Str()
    time_period = fields.Str()
    cultural_notes = fields.Str()
    grammatical_categories = fields.List(fields.Str())
    semantic_domains = fields.List(fields.Str())
    etymology_confidence = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    data_quality_score = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    verification_status = fields.Str(validate=validate.OneOf([
        'unverified', 'verified', 'needs_review', 'disputed'
    ]))

# Utility functions
def generate_api_key() -> str:
    """Generate a new API key."""
    return secrets.token_urlsafe(32)

def validate_api_key(api_key: str) -> bool:
    """Validate an API key."""
    # Implement your API key validation logic
    # This is a placeholder implementation
    return bool(api_key and len(api_key) >= 32)

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.blake2b(api_key.encode()).hexdigest()

def generate_jwt_token(user_id: int, roles: List[str] = None) -> str:
    """Generate a JWT token."""
    payload = {
        'user_id': user_id,
        'roles': roles or [],
        'exp': datetime.utcnow() + timedelta(
            seconds=current_app.config['SECURITY'].JWT_EXPIRATION
        )
    }
    return jwt.encode(
        payload,
        current_app.config['SECURITY'].JWT_SECRET_KEY,
        algorithm=current_app.config['SECURITY'].JWT_ALGORITHM
    )

class WordQuerySchema(Schema):
    """Schema for word query parameters."""
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_definitions = fields.Bool(default=True)
    include_relations = fields.Bool(default=True)
    include_etymology = fields.Bool(default=True)

class SearchQuerySchema(Schema):
    """Schema for search query parameters."""
    q = fields.Str(required=True, validate=validate.Length(min=1))
    limit = fields.Int(validate=validate.Range(min=1, max=50), default=10)
    pos = fields.Str(validate=validate.OneOf(['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix']))
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')
    include_baybayin = fields.Bool(default=True)
    min_similarity = fields.Float(validate=validate.Range(min=0.0, max=1.0), default=0.3)
    mode = fields.Str(validate=validate.OneOf(['all', 'exact', 'phonetic', 'baybayin']), default='all')
    sort = fields.Str(validate=validate.OneOf(['relevance', 'alphabetical', 'created', 'updated']), default='relevance')
    order = fields.Str(validate=validate.OneOf(['asc', 'desc']), default='desc')

class PaginationSchema(Schema):
    """Schema for pagination parameters."""
    page = fields.Int(validate=validate.Range(min=1), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)

class WordDetailSchema(Schema):
    """Schema for word detail parameters."""
    include_definitions = fields.Boolean(missing=True)
    include_relations = fields.Boolean(missing=True)
    include_etymology = fields.Boolean(missing=True)

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    if not text:
        return ""
    
    # Remove any non-word characters except spaces, hyphens, and apostrophes
    text = re.sub(r'[^\w\s\-\']', '', text)
    
    # Remove trailing numbers
    text = re.sub(r'\d+$', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def validate_language_code(code: str) -> bool:
    """Validate language code."""
    if not code or not isinstance(code, str):
        return False
    if len(code) > 16:
        return False
    # Only allow lowercase letters and hyphens
    return bool(re.match(r'^[a-z-]+$', code))

def validate_pos_code(code: str) -> bool:
    """Validate part of speech code."""
    valid_codes = ['n', 'v', 'adj', 'adv', 'pron', 'prep', 'conj', 'intj', 'det', 'affix']
    return code in valid_codes

def validate_relation_type(rel_type: str) -> bool:
    """Validate relation type."""
    valid_types = [
        'synonym', 'antonym', 'variant', 'derived_from', 'component_of', 
        'related', 'cognate', 'hypernym', 'hyponym', 'meronym', 'holonym',
        'regional_variant', 'spelling_variant', 'historical_variant',
        'root_of', 'derived_form'
    ]
    return rel_type in valid_types

def validate_affix_type(affix_type: str) -> bool:
    """Validate affix type."""
    valid_types = ['prefix', 'infix', 'suffix', 'circumfix', 'reduplication', 'compound']
    return affix_type in valid_types

def validate_baybayin_text(text: str) -> bool:
    """Validate Baybayin text."""
    if not text:
        return False
    
    # Check if text contains only valid Baybayin characters
    baybayin_pattern = r'^[\u1700-\u171F\s]*$'
    return bool(re.match(baybayin_pattern, text))

def validate_sources(sources: List[str]) -> bool:
    """Validate source references."""
    if not sources:
        return False
    
    # Check if sources are in valid format and not empty
    return all(isinstance(s, str) and s.strip() for s in sources)

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate metadata structure."""
    if not isinstance(metadata, dict):
        return False
    
    # Check for required metadata fields
    required_fields = ['strength', 'confidence', 'source']
    return all(field in metadata for field in required_fields)

def validate_pronunciation_data(data: Dict[str, Any]) -> bool:
    """Validate pronunciation data structure."""
    if not isinstance(data, dict):
        return False
    
    # Check for valid pronunciation types
    valid_types = ['ipa', 'respelling', 'audio', 'phonemic']
    if 'type' in data and data['type'] not in valid_types:
        return False
    
    return True

def validate_etymology_data(data: Dict[str, Any]) -> bool:
    """Validate etymology data structure."""
    if not isinstance(data, dict):
        return False
    
    # Check for required etymology fields
    required_fields = ['text', 'components', 'confidence']
    return all(field in data for field in required_fields)

def validate_definition_data(data: Dict[str, Any]) -> bool:
    """Validate definition data structure."""
    if not isinstance(data, dict):
        return False
    
    # Check for required definition fields
    required_fields = ['text', 'pos']
    if not all(field in data for field in required_fields):
        return False
    
    # Validate POS code if present
    if 'pos' in data and not validate_pos_code(data['pos']):
        return False
    
    return True

def check_request_origin():
    """Check if request origin is allowed."""
    origin = request.headers.get('Origin')
    if not origin:
        return True
    
    allowed_origins = current_app.config.get('ALLOWED_ORIGINS', [])
    return origin in allowed_origins

def check_api_key():
    """Check if API key is valid."""
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return False
    
    valid_keys = current_app.config.get('API_KEYS', [])
    return api_key in valid_keys 