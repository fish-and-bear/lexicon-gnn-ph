"""
Comprehensive security configuration for the application.
Includes rate limiting, CORS, request validation, and security headers.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
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
from marshmallow import Schema, fields, validate
from marshmallow.exceptions import ValidationError
from flask import jsonify

logger = logging.getLogger(__name__)

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

def validate_json_request(schema: Schema):
    """Decorator to validate request data against schema."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            validator = current_app.request_validator
            
            # Validate content length
            if not validator.validate_content_length(request):
                abort(413, description="Request too large")
                
            # Validate and sanitize JSON
            if request.is_json:
                data = request.get_json()
                if not validator.validate_json_structure(data):
                    abort(400, description="Invalid request structure")
                    
                try:
                    # Validate against schema
                    validated_data = schema.load(data)
                    # Sanitize data
                    sanitized_data = validator.sanitize_data(validated_data)
                    # Store sanitized data for the route
                    g.validated_data = sanitized_data
                except Exception as e:
                    abort(400, description=str(e))
                    
            return f(*args, **kwargs)
        return decorated
    return decorator

def validate_query_params(schema: Schema):
    """Decorator to validate query parameters."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                # Fixed: Convert request.args to dict before passing to schema.load
                query_dict = request.args.to_dict()
                params = schema.load(query_dict)
                # Add validated parameters to kwargs
                kwargs.update(params)
                return f(*args, **kwargs)
            except ValidationError as err:
                return jsonify({
                    "error": {
                        "message": "Invalid parameters",
                        "details": err.messages,
                        "status_code": 400,
                        "timestamp": datetime.now().isoformat()
                    }
                }), 400
        # Preserve the original endpoint name
        decorated.__name__ = f.__name__
        return decorated
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
    query = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    page = fields.Int(validate=validate.Range(min=1, max=100), default=1)
    per_page = fields.Int(validate=validate.Range(min=1, max=100), default=20)
    language = fields.Str(validate=validate.OneOf(['tl', 'ceb']), default='tl')

class WordSchema(Schema):
    lemma = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    language_code = fields.Str(validate=validate.OneOf(['tl', 'ceb']), required=True)
    definitions = fields.List(fields.Dict(), validate=validate.Length(max=50))
    etymologies = fields.List(fields.Dict(), validate=validate.Length(max=10))

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