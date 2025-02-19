"""
Enhanced caching system with Redis and in-memory caching.
Provides multi-level caching, circuit breaker pattern, and monitoring.
"""

from functools import wraps
from cachetools import TTLCache, LRUCache
import redis
import pickle
import os
from dotenv import load_dotenv
import json
import functools
from datetime import datetime, UTC, timedelta
from flask import current_app, has_app_context, request
import logging
from prometheus_client import Counter, Histogram, Gauge
import hashlib
import zlib
from typing import Any, Optional, Union, Dict, List, Callable
import threading
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

# Enhanced Metrics
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses', ['cache_type'])
CACHE_ERRORS = Counter('cache_errors_total', 'Total cache errors', ['cache_type', 'error_type'])
CACHE_OPERATION_DURATION = Histogram('cache_operation_duration_seconds', 'Cache operation duration')
CACHE_SIZE = Gauge('cache_size_bytes', 'Current cache size in bytes', ['cache_type'])
CACHE_ITEMS = Gauge('cache_items_total', 'Total number of items in cache', ['cache_type'])
CIRCUIT_BREAKER_STATUS = Gauge('circuit_breaker_status', 'Circuit breaker status (0=open, 1=closed)', ['cache_type'])

# Initialize caches with size tracking
memory_cache = {
    'default': TTLCache(maxsize=1000, ttl=3600),
    'short': TTLCache(maxsize=500, ttl=300),
    'long': TTLCache(maxsize=200, ttl=86400),
    'permanent': LRUCache(maxsize=100),
    'hot': TTLCache(maxsize=100, ttl=60)  # For frequently accessed items
}

# Redis client
redis_client = None

# Circuit breaker state
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.status = "closed"
        self._lock = threading.Lock()

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.status = "open"
                CIRCUIT_BREAKER_STATUS.labels(cache_type='redis').set(0)

    def record_success(self):
        with self._lock:
            self.failures = 0
            self.status = "closed"
            CIRCUIT_BREAKER_STATUS.labels(cache_type='redis').set(1)

    def can_execute(self) -> bool:
        if self.status == "closed":
            return True
        
        if self.last_failure_time and time.time() - self.last_failure_time > self.reset_timeout:
            with self._lock:
                self.status = "half-open"
                return True
        
        return False

circuit_breaker = CircuitBreaker()

class CacheConfig:
    """Enhanced cache configuration settings."""
    DEFAULT_EXPIRY = 3600
    SHORT_EXPIRY = 300
    LONG_EXPIRY = 86400
    COMPRESSION_THRESHOLD = 1024  # Compress data larger than 1KB
    MAX_KEY_LENGTH = 200
    CACHE_PREFIX = "api:v2:"
    WARM_UP_KEYS = [  # Keys to pre-warm
        "statistics",
        "parts_of_speech",
        "language_metadata"
    ]
    HOT_KEY_THRESHOLD = 10  # Number of accesses to consider a key "hot"
    CACHE_JITTER = 0.1  # 10% jitter for cache expiration
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 0.1  # seconds

class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass

class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass

class CacheSerializationError(CacheError):
    """Raised when serialization/deserialization fails."""
    pass

def apply_jitter(expiry: int) -> int:
    """Add random jitter to cache expiration to prevent cache stampede."""
    jitter = random.uniform(-CacheConfig.CACHE_JITTER, CacheConfig.CACHE_JITTER)
    return int(expiry * (1 + jitter))

@retry(stop=stop_after_attempt(CacheConfig.MAX_RETRY_ATTEMPTS), 
       wait=wait_exponential(multiplier=CacheConfig.RETRY_DELAY))
def init_cache(redis_url: Optional[str] = None) -> Optional[redis.Redis]:
    """Initialize Redis client with enhanced error handling and connection pooling."""
    global redis_client
    if redis_client is not None:
        return redis_client

    try:
        if redis_url:
            connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=10,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True
            )
        else:
            connection_pool = redis.ConnectionPool(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD', ''),
                decode_responses=True,
                max_connections=10,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True
            )
        
        redis_client = redis.Redis(connection_pool=connection_pool)
        redis_client.ping()  # Test connection
        logger.info("Redis cache initialized successfully")
        
        # Warm up cache
        warm_up_cache()
        
        # Start cache maintenance thread
        start_cache_maintenance()
        
        return redis_client
        
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis connection failed: {str(e)}")
        CACHE_ERRORS.labels(cache_type='redis', error_type='connection').inc()
        circuit_breaker.record_failure()
        redis_client = None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {str(e)}")
        CACHE_ERRORS.labels(cache_type='redis', error_type='unknown').inc()
        circuit_breaker.record_failure()
        redis_client = None

    return None

def warm_up_cache():
    """Pre-warm cache with frequently accessed data."""
    if not should_use_redis():
        return

    for key in CacheConfig.WARM_UP_KEYS:
        try:
            # Call the appropriate function to generate the data
            if key == "statistics":
                from backend.routes import get_statistics
                data = get_statistics()
            elif key == "parts_of_speech":
                from backend.routes import get_parts_of_speech
                data = get_parts_of_speech()
            elif key == "language_metadata":
                from backend.language_utils import get_language_metadata
                data = {code: get_language_metadata(code) for code in ['tl', 'ceb']}
            
            # Cache the data
            cache_key = f"{CacheConfig.CACHE_PREFIX}warmup:{key}"
            add_to_cache(cache_key, data, CacheConfig.LONG_EXPIRY)
            logger.info(f"Warmed up cache for {key}")
            
        except Exception as e:
            logger.error(f"Failed to warm up cache for {key}: {str(e)}")

def start_cache_maintenance():
    """Start background thread for cache maintenance."""
    def maintenance_worker():
        while True:
            try:
                # Clean up expired keys
                clean_expired_keys()
                
                # Update cache size metrics
                update_cache_metrics()
                
                # Promote hot keys
                promote_hot_keys()
                
                # Wait for next maintenance cycle
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
    
    maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
    maintenance_thread.start()

def clean_expired_keys():
    """Clean up expired keys from Redis."""
    if not should_use_redis():
        return

    try:
        pattern = f"{CacheConfig.CACHE_PREFIX}*"
        for key in redis_client.scan_iter(pattern):
            if not redis_client.ttl(key):
                redis_client.delete(key)
    except Exception as e:
        logger.error(f"Failed to clean expired keys: {str(e)}")

def update_cache_metrics():
    """Update cache size and item count metrics."""
    # Update memory cache metrics
    for cache_type, cache in memory_cache.items():
        CACHE_ITEMS.labels(cache_type=f'memory_{cache_type}').set(len(cache))
        
    # Update Redis metrics if available
    if should_use_redis():
        try:
            info = redis_client.info(section="memory")
            CACHE_SIZE.labels(cache_type='redis').set(info.get('used_memory', 0))
            CACHE_ITEMS.labels(cache_type='redis').set(
                redis_client.dbsize()
            )
        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {str(e)}")

def promote_hot_keys():
    """Promote frequently accessed keys to the hot cache."""
    if not should_use_redis():
        return

    try:
        # Get access counts from Redis
        pattern = f"{CacheConfig.CACHE_PREFIX}*"
        for key in redis_client.scan_iter(pattern):
            access_count = int(redis_client.object('idletime', key))
            if access_count > CacheConfig.HOT_KEY_THRESHOLD:
                # Move to hot cache
                value = redis_client.get(key)
                if value:
                    memory_cache['hot'][key] = deserialize_data(value)
    except Exception as e:
        logger.error(f"Failed to promote hot keys: {str(e)}")

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a consistent and safe cache key."""
    # Create a string representation of args and kwargs
    key_parts = [prefix]
    if args:
        key_parts.append(str(args))
    if kwargs:
        # Sort kwargs for consistency
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(sorted_kwargs))
    
    # Join parts and create hash for long keys
    key = ":".join(key_parts)
    if len(key) > CacheConfig.MAX_KEY_LENGTH:
        key = hashlib.sha256(key.encode()).hexdigest()
    
    return f"{CacheConfig.CACHE_PREFIX}{key}"

def should_use_redis() -> bool:
    """Check if Redis should be used based on configuration and availability."""
    if has_app_context():
        if current_app.config.get('TESTING', False):
            return False
        if current_app.config.get('DISABLE_REDIS', False):
            return False
    return redis_client is not None and circuit_breaker.can_execute()

def serialize_data(data: Any) -> bytes:
    """Serialize data with optional compression."""
    try:
        serialized = pickle.dumps(data)
        if len(serialized) > CacheConfig.COMPRESSION_THRESHOLD:
            return zlib.compress(serialized)
        return serialized
    except Exception as e:
        logger.error(f"Serialization error: {str(e)}")
        raise CacheError(f"Failed to serialize data: {str(e)}")

def deserialize_data(data: bytes) -> Any:
    """Deserialize data with automatic decompression if needed."""
    try:
        try:
            return pickle.loads(zlib.decompress(data))
        except zlib.error:
            return pickle.loads(data)
    except Exception as e:
        logger.error(f"Deserialization error: {str(e)}")
        CACHE_ERRORS.labels(cache_type='serialization', error_type='deserialize').inc()
        raise CacheSerializationError(f"Failed to deserialize data: {str(e)}")

def multi_level_cache(ttl: int = CacheConfig.DEFAULT_EXPIRY, prefix: str = ""):
    """Enhanced multi-level cache decorator with metrics and error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(prefix or func.__name__, *args, **kwargs)

            with CACHE_OPERATION_DURATION.time():
                try:
                    # Try Redis first if available and circuit breaker is closed
                    if should_use_redis():
                        try:
                            cached_data = redis_client.get(cache_key)
                            if cached_data:
                                CACHE_HITS.labels(cache_type='redis').inc()
                                circuit_breaker.record_success()
                                return deserialize_data(cached_data)
                        except Exception as e:
                            logger.warning(f"Redis cache error: {str(e)}")
                            CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()
                            circuit_breaker.record_failure()

                    # Try memory cache
                    cache_level = 'default'
                    if ttl <= CacheConfig.SHORT_EXPIRY:
                        cache_level = 'short'
                    elif ttl >= CacheConfig.LONG_EXPIRY:
                        cache_level = 'long'
                    
                    if cache_key in memory_cache[cache_level]:
                        CACHE_HITS.labels(cache_type='memory').inc()
                        return memory_cache[cache_level][cache_key]
                    
                    # Cache miss - execute function
                    CACHE_MISSES.labels(cache_type='all').inc()
                    result = func(*args, **kwargs)

                    # Store in both caches
                    if should_use_redis():
                        try:
                            redis_client.setex(
                                cache_key,
                                apply_jitter(ttl),  # Add jitter to prevent thundering herd
                                serialize_data(result)
                            )
                            circuit_breaker.record_success()
                        except Exception as e:
                            logger.warning(f"Redis cache write error: {str(e)}")
                            CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
                            circuit_breaker.record_failure()
            
                    memory_cache[cache_level][cache_key] = result
                    return result

                except Exception as e:
                    logger.error(f"Cache error in {func.__name__}: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='general', error_type='unknown').inc()
                    return func(*args, **kwargs)

        # Preserve the original endpoint name
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def cache_response(expiry: int = CacheConfig.DEFAULT_EXPIRY):
    """Enhanced cache decorator for API responses with better error handling."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not should_use_redis():
                return f(*args, **kwargs)

            # Include request method and query parameters in cache key
            cache_key = generate_cache_key(
                f.__name__,
                request.method,
                request.args.to_dict(),
                *args,
                **kwargs
            )
            
            with CACHE_OPERATION_DURATION.time():
                try:
                    # Try to get cached result
                    cached_result = redis_client.get(cache_key)
                    if cached_result:
                        CACHE_HITS.labels(cache_type='redis').inc()
                        circuit_breaker.record_success()
                        return json.loads(cached_result)
                    
                    # Cache miss
                    CACHE_MISSES.labels(cache_type='redis').inc()
                    result = f(*args, **kwargs)
                    
                    # Cache the result with jitter
                    redis_client.setex(
                        cache_key,
                        apply_jitter(expiry),
                        json.dumps(result, default=str)
                    )
                    circuit_breaker.record_success()
                    return result
                    
                except Exception as e:
                    logger.error(f"Cache error in {f.__name__}: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='response', error_type='unknown').inc()
                    circuit_breaker.record_failure()
                return f(*args, **kwargs)
                
        return wrapper
    return decorator

# Initialize Redis on module import
init_cache()

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics and metrics."""
    stats = {
        "memory_cache": {
            cache_type: {
                "size": len(cache),
                "maxsize": cache.maxsize,
                "currsize": cache.currsize,
                "ttl": cache.ttl if hasattr(cache, 'ttl') else None
            }
            for cache_type, cache in memory_cache.items()
        },
        "redis": {
            "connected": should_use_redis(),
            "circuit_breaker": {
                "status": circuit_breaker.status,
                "failures": circuit_breaker.failures,
                "last_failure": circuit_breaker.last_failure_time
            },
            "info": None
        },
        "metrics": {
            "hits": {
                "redis": CACHE_HITS.labels(cache_type='redis')._value.get(),
                "memory": CACHE_HITS.labels(cache_type='memory')._value.get()
            },
            "misses": {
                "all": CACHE_MISSES.labels(cache_type='all')._value.get()
            },
            "errors": {
                cache_type: {
                    error_type: CACHE_ERRORS.labels(
                        cache_type=cache_type, 
                        error_type=error_type
                    )._value.get()
                    for error_type in ['connection', 'read', 'write', 'delete', 'clear', 'unknown']
                }
                for cache_type in ['redis', 'memory', 'general']
            }
        }
    }
    
    if should_use_redis():
        try:
            info = redis_client.info()
            stats["redis"]["info"] = {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days"),
                "hit_rate": info.get("keyspace_hits", 0) / (
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1)
                ) * 100,
                "total_keys": redis_client.dbsize(),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {str(e)}")
    
    return stats
