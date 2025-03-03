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
from datetime import datetime, timezone, timedelta
from flask import current_app, has_app_context, request, g
import logging
from prometheus_client import Counter, Histogram, Gauge
import hashlib
import zlib
from typing import Any, Optional, Union, Dict, List, Callable
import threading
import time
import random
import structlog

load_dotenv()

logger = structlog.get_logger(__name__)

# Initialize cache
cache = None

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
        "baybayin_words"
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

def init_cache(app=None):
    """Initialize Redis cache client."""
    global redis_client, cache
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        
        # Normalize the URL for Windows
        if 'redis:6379' in redis_url:
            redis_url = 'redis://localhost:6379/0'

        logger.info(f"Connecting to Redis cache at {redis_url}")

        redis_client = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test connection
        redis_client.ping()
        logger.info("Successfully connected to Redis cache")
        
        # Initialize Flask-Caching
        if app:
            cache_config = {
                'CACHE_TYPE': 'redis',
                'CACHE_REDIS_URL': redis_url,
                'CACHE_DEFAULT_TIMEOUT': CacheConfig.DEFAULT_EXPIRY
            }
            from flask_caching import Cache
            cache = Cache(config=cache_config)
            cache.init_app(app)
        
        # Set circuit breaker to closed state
        CIRCUIT_BREAKER_STATUS.labels(cache_type='redis').set(1)
        
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Redis cache: {e}")
        
        # Use simple dict as fallback if Redis is unavailable
        if app:
            from flask_caching import Cache
            cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
            cache.init_app(app)
            
        return False

def warm_up_cache():
    """Pre-warm cache with frequently accessed data."""
    if not should_use_redis():
        return

    for key in CacheConfig.WARM_UP_KEYS:
        try:
            cache_key = f"{CacheConfig.CACHE_PREFIX}warmup:{key}"
            add_to_cache(cache_key, {"warmed_up": True}, CacheConfig.LONG_EXPIRY)
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
        for key in redis_client.scan_iter(match=pattern):
            if not redis_client.ttl(key):
                redis_client.delete(key)
    except Exception as e:
        logger.error(f"Failed to clean expired keys: {str(e)}")

def update_cache_metrics():
    """Update cache size and item count metrics."""
    # Update memory cache metrics
    for cache_type, cache_obj in memory_cache.items():
        CACHE_ITEMS.labels(cache_type=f'memory_{cache_type}').set(len(cache_obj))
        
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
        pattern = f"{CacheConfig.CACHE_PREFIX}*"
        for key in redis_client.scan_iter(match=pattern):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            meta_key = f"{key_str}:meta"
            access_count_val = redis_client.hget(meta_key, "access_count")
            if access_count_val:
                access_count = int(access_count_val.decode('utf-8') if isinstance(access_count_val, bytes) else access_count_val)
                if access_count >= CacheConfig.HOT_KEY_THRESHOLD:
                    # Get the data and TTL from Redis
                    data = redis_client.get(key)
                    ttl = redis_client.ttl(key)
                    if data and ttl > 0:
                        # Add to hot memory cache
                        memory_cache['hot'][key_str] = deserialize_data(data)
    except Exception as e:
        logger.error(f"Failed to promote hot keys: {str(e)}")

def add_to_cache(key: str, data: Any, expiry: int = CacheConfig.DEFAULT_EXPIRY) -> bool:
    """Add data to both Redis and memory cache with proper error handling."""
    try:
        if not key or len(key) > CacheConfig.MAX_KEY_LENGTH:
            logger.warning(f"Invalid cache key: {key}")
            return False

        # Apply jitter to prevent cache stampede
        final_expiry = apply_jitter(expiry)
        
        # Add to memory cache
        memory_cache['default'][key] = data
        
        # Add to Redis if available
        if should_use_redis():
            try:
                serialized_data = serialize_data(data)
                redis_client.setex(key, final_expiry, serialized_data)
                redis_client.hset(f"{key}:meta", mapping={"access_count": 0})
                CACHE_SIZE.labels(cache_type='redis').inc(len(serialized_data))
                CACHE_ITEMS.labels(cache_type='redis').inc()
                circuit_breaker.record_success()
            except Exception as e:
                logger.error(f"Failed to add to Redis cache: {str(e)}")
                circuit_breaker.record_failure()
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add to cache: {str(e)}")
        CACHE_ERRORS.labels(cache_type='memory', error_type='add').inc()
        return False

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Convert args and kwargs to strings
    args_str = [str(arg) for arg in args]
    kwargs_str = [f"{k}:{v}" for k, v in sorted(kwargs.items())]
    
    # Create key components
    components = [prefix] + args_str + kwargs_str
    
    # Add request-specific components if available
    if request:
        path = request.path
        args_dict = request.args.to_dict() if hasattr(request.args, 'to_dict') else {}
        components.extend([
            path,
            str(args_dict),
            request.headers.get('Accept-Language', '')
        ])
    
    # Join and hash components
    key_string = ":".join(str(c) for c in components)
    return f"{CacheConfig.CACHE_PREFIX}{prefix}:{hashlib.md5(key_string.encode()).hexdigest()}"

def should_use_redis() -> bool:
    """Check if Redis should be used based on configuration and availability."""
    if has_app_context():
        if current_app.config.get('TESTING', False):
            return False
        if not current_app.config.get('REDIS_ENABLED', True):
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
        CACHE_ERRORS.labels(cache_type='serialization', error_type='serialize').inc()
        raise CacheSerializationError(f"Failed to serialize data: {str(e)}")

def deserialize_data(data: bytes) -> Any:
    """Deserialize data with automatic decompression if needed."""
    if not data:
        return None
        
    try:
        try:
            return pickle.loads(zlib.decompress(data))
        except zlib.error:
            return pickle.loads(data)
    except Exception as e:
        logger.error(f"Deserialization error: {str(e)}")
        CACHE_ERRORS.labels(cache_type='serialization', error_type='deserialize').inc()
        raise CacheSerializationError(f"Failed to deserialize data: {str(e)}")

def multi_level_cache(prefix: str, ttl: Optional[int] = None):
    """
    Multi-level cache decorator that uses both memory and Redis caching.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if no app context or in debug mode
            if not has_app_context() or current_app.debug:
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = generate_cache_key(prefix, *args, **kwargs)
            
            # First check memory cache
            if cache_key in memory_cache['default']:
                CACHE_HITS.labels(cache_type='memory').inc()
                return memory_cache['default'][cache_key]
            
            # Then check Redis cache if available
            if should_use_redis() and cache:
                try:
                    cached_value = cache.get(cache_key)
                    if cached_value is not None:
                        CACHE_HITS.labels(cache_type='redis').inc()
                        # Add to memory cache for faster future access
                        memory_cache['default'][cache_key] = cached_value
                        return cached_value
                except Exception as e:
                    logger.error(f"Redis cache error: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()
            
            # Cache miss, execute function
            CACHE_MISSES.labels(cache_type='all').inc()
            result = func(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                memory_cache['default'][cache_key] = result
                
                if should_use_redis() and cache:
                    try:
                        cache.set(cache_key, result, timeout=ttl or CacheConfig.DEFAULT_EXPIRY)
                    except Exception as e:
                        logger.error(f"Failed to set Redis cache: {str(e)}")
                        CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
            
            return result
        return wrapper
    return decorator

def cache_response(expiry: int = CacheConfig.DEFAULT_EXPIRY):
    """Enhanced cache decorator for API responses with better error handling."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Skip caching if no app context, in debug mode, or during testing
            if not has_app_context() or current_app.debug or current_app.config.get('TESTING', False):
                return f(*args, **kwargs)

            # Include request method and query parameters in cache key
            cache_key = generate_cache_key(
                f.__name__,
                request.method,
                request.args.to_dict() if hasattr(request.args, 'to_dict') else {},
                *args,
                **kwargs
            )
            
            # First check memory cache
            if cache_key in memory_cache['default']:
                CACHE_HITS.labels(cache_type='memory').inc()
                return memory_cache['default'][cache_key]
            
            # Then check Redis if available
            if should_use_redis() and redis_client:
                with CACHE_OPERATION_DURATION.time():
                    try:
                        # Try to get cached result
                        cached_result = redis_client.get(cache_key)
                        if cached_result:
                            # Increment access count for promotion to hot cache
                            redis_client.hincrby(f"{cache_key}:meta", "access_count", 1)
                            
                            CACHE_HITS.labels(cache_type='redis').inc()
                            circuit_breaker.record_success()
                            
                            # Deserialize data
                            result = deserialize_data(cached_result)
                            # Add to memory cache for faster future access
                            memory_cache['default'][cache_key] = result
                            return result
                    except Exception as e:
                        logger.error(f"Redis cache error: {str(e)}")
                        CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()
                        circuit_breaker.record_failure()
            
            # Cache miss, execute function
            CACHE_MISSES.labels(cache_type='all').inc()
            result = f(*args, **kwargs)
            
            # Cache the result
            if result is not None:
                memory_cache['default'][cache_key] = result
                
                if should_use_redis() and redis_client:
                    try:
                        # Serialize and store in Redis
                        serialized_data = serialize_data(result)
                        redis_client.setex(
                            cache_key,
                            apply_jitter(expiry),
                            serialized_data
                        )
                        
                        # Initialize metadata
                        redis_client.hset(f"{cache_key}:meta", mapping={"access_count": 1})
                        circuit_breaker.record_success()
                    except Exception as e:
                        logger.error(f"Failed to set Redis cache: {str(e)}")
                        CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
                        circuit_breaker.record_failure()
            
            return result
        return wrapper
    return decorator

def invalidate_cache_prefix(prefix: str) -> bool:
    """Invalidate all cache entries with given prefix."""
    success = True
    
    # Clear memory cache entries
    prefix_pattern = f"{CacheConfig.CACHE_PREFIX}{prefix}"
    for cache_type, cache_obj in memory_cache.items():
        keys_to_delete = [k for k in cache_obj.keys() if str(k).startswith(prefix_pattern)]
        for key in keys_to_delete:
            try:
                del cache_obj[key]
            except KeyError:
                pass
    
    # Clear Redis cache entries if available
    if should_use_redis() and redis_client:
        try:
            pattern = f"{CacheConfig.CACHE_PREFIX}{prefix}*"
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, match=pattern)
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Failed to invalidate Redis cache: {str(e)}")
            CACHE_ERRORS.labels(cache_type='redis', error_type='delete').inc()
            success = False
    
    # Clear Flask-Caching cache if available
    if cache:
        try:
            if hasattr(cache, 'delete_pattern'):
                cache.delete_pattern(f"{CacheConfig.CACHE_PREFIX}{prefix}*")
            elif hasattr(cache, 'delete_many'):
                # If delete_pattern is not available, fall back to basic cache
                keys = []
                if should_use_redis() and redis_client:
                    pattern = f"{CacheConfig.CACHE_PREFIX}{prefix}*"
                    cursor = 0
                    while True:
                        cursor, matching_keys = redis_client.scan(cursor, match=pattern)
                        keys.extend([k.decode('utf-8') if isinstance(k, bytes) else k for k in matching_keys])
                        if cursor == 0:
                            break
                
                if keys:
                    cache.delete_many(*keys)
        except Exception as e:
            logger.error(f"Failed to invalidate Flask-Caching cache: {str(e)}")
            success = False
    
    return success

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics and metrics."""
    stats = {
        "memory_cache": {
            cache_type: {
                "size": len(cache_obj),
                "maxsize": getattr(cache_obj, 'maxsize', 0),
                "currsize": getattr(cache_obj, 'currsize', len(cache_obj)),
                "ttl": getattr(cache_obj, 'ttl', None)
            }
            for cache_type, cache_obj in memory_cache.items()
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
                "redis": CACHE_HITS.labels(cache_type='redis')._value.get() if hasattr(CACHE_HITS.labels(cache_type='redis'), '_value') else 0,
                "memory": CACHE_HITS.labels(cache_type='memory')._value.get() if hasattr(CACHE_HITS.labels(cache_type='memory'), '_value') else 0
            },
            "misses": {
                "all": CACHE_MISSES.labels(cache_type='all')._value.get() if hasattr(CACHE_MISSES.labels(cache_type='all'), '_value') else 0
            },
            "errors": {}
        }
    }
    
    # Collect error metrics
    for cache_type in ['redis', 'memory', 'general']:
        stats["metrics"]["errors"][cache_type] = {}
        for error_type in ['connection', 'read', 'write', 'delete', 'clear', 'unknown']:
            counter = CACHE_ERRORS.labels(cache_type=cache_type, error_type=error_type)
            stats["metrics"]["errors"][cache_type][error_type] = counter._value.get() if hasattr(counter, '_value') else 0
    
    # Get Redis info if available
    if should_use_redis() and redis_client:
        try:
            info = redis_client.info()
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 1)
            hit_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
            
            stats["redis"]["info"] = {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_days": info.get("uptime_in_days"),
                "hit_rate": hit_rate,
                "total_keys": redis_client.dbsize(),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {str(e)}")
    
    return stats