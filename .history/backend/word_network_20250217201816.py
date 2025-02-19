"""
Word network utilities and monitoring components.
"""

import logging
import os
from datetime import datetime, UTC
import time
from functools import wraps
import redis
from prometheus_client import Counter, Histogram
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('word_network.log')
    ]
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
DB_QUERY_LATENCY = Histogram('db_query_duration_seconds', 'Database query latency')
CACHE_HIT_COUNT = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISS_COUNT = Counter('cache_misses_total', 'Total cache misses')

# Initialize Redis
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=int(os.environ.get("REDIS_DB", 0)),
    decode_responses=True
)

def monitor_performance(endpoint_name):
    """Monitor endpoint performance."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                status = result[1] if isinstance(result, tuple) else 200
                REQUEST_COUNT.labels(
                    method=f.__name__,
                    endpoint=endpoint_name,
                    status=status
                ).inc()
                return result
            finally:
                REQUEST_LATENCY.observe(time.time() - start_time)
        return wrapped
    return decorator

def monitor_query(f):
    """Monitor database query performance."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            DB_QUERY_LATENCY.observe(duration)
    return wrapped

def apply_rate_limit(limit_per_minute):
    """Apply rate limiting to endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            from flask import request, jsonify
            
            key = f"rate_limit:{request.remote_addr}:{f.__name__}"
            current = redis_client.get(key)
            
            if current is None:
                redis_client.setex(key, 60, 1)
            elif int(current) >= limit_per_minute:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": redis_client.ttl(key)
                }), 429
            else:
                redis_client.incr(key)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator
