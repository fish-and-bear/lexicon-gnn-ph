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
from flask_caching import Cache
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

# Initialize cache
cache = Cache(config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379/0'),
    'CACHE_DEFAULT_TIMEOUT': 300
})

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

@retry(stop=stop_after_attempt(CacheConfig.MAX_RETRY_ATTEMPTS), 
       wait=wait_exponential(multiplier=CacheConfig.RETRY_DELAY))
def init_cache(redis_url: Optional[str] = None) -> Optional[redis.Redis]:
    """Initialize Redis client with enhanced error handling and connection pooling."""
    global redis_client
    if redis_client is not None:
        return redis_client

    try:
        # Normalize Redis URL for Windows compatibility
        if redis_url and 'redis:6379' in redis_url:
            redis_url = 'redis://localhost:6379/0'
            logger.info(f"Converted Redis URL to {redis_url} for Windows compatibility")

        if redis_url:
            connection_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=10,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
        else:
            connection_pool = redis.ConnectionPool(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD', ''),
                decode_responses=False,  # Keep as bytes for pickle compatibility
                max_connections=10,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
        
        redis_client = redis.Redis(connection_pool=connection_pool)
        redis_client.ping()  # Test connection
        logger.info("Redis cache initialized successfully")
        
        return redis_client
        
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis connection failed: {str(e)}. Using in-memory cache only.")
        redis_client = None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {str(e)}")
        redis_client = None

    return None

def warm_up_cache():
    """Pre-warm cache with frequently accessed data."""
    if not should_use_redis():
        return

    # Import here to avoid circular imports
    try:
        import backend.routes as routes
        import backend.language_utils as language_utils
        
        for key in CacheConfig.WARM_UP_KEYS:
            try:
                # Call the appropriate function to generate the data
                if key == "statistics":
                    # Handle statistics (don't call the function directly)
                    cache_key = f"{CacheConfig.CACHE_PREFIX}warmup:statistics"
                    add_to_cache(cache_key, {"warmed_up": True}, CacheConfig.LONG_EXPIRY)
                elif key == "parts_of_speech":
                    # Handle parts of speech
                    cache_key = f"{CacheConfig.CACHE_PREFIX}parts_of_speech"
                    add_to_cache(cache_key, {"warmed_up": True}, CacheConfig.LONG_EXPIRY)
                elif key == "baybayin_words":
                    # Prepare for baybayin words
                    cache_key = f"{CacheConfig.CACHE_PREFIX}baybayin_words"
                    add_to_cache(cache_key, {"warmed_up": True}, CacheConfig.LONG_EXPIRY)
                
                logger.info(f"Warmed up cache for {key}")
                
            except Exception as e:
                logger.error(f"Failed to warm up cache for {key}: {str(e)}")
    except ImportError as e:
        logger.warning(f"Could not import modules for cache warming: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to warm up cache: {str(e)}")

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
        pattern = f"{CacheConfig.CACHE_PREFIX}*"
        for key in redis_client.scan_iter(match=pattern):
            meta_key = f"{key.decode('utf-8') if isinstance(key, bytes) else key}:meta"
            access_count_val = redis_client.hget(meta_key, "access_count")
            if access_count_val:
                access_count = int(access_count_val.decode('utf-8') if isinstance(access_count_val, bytes) else access_count_val)
                if access_count >= CacheConfig.HOT_KEY_THRESHOLD:
                    # Get the data and TTL from Redis
                    data = redis_client.get(key)
                    ttl = redis_client.ttl(key)
                    if data and ttl > 0:
                        # Add to hot memory cache
                        key_str = key.decode('utf-8') if isinstance(key, bytes) else key
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

def multi_level_cache(ttl: int = CacheConfig.DEFAULT_EXPIRY, prefix: str = ""):
    """Enhanced multi-level cache decorator with metrics and error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(prefix or func.__name__, *args, **kwargs)

            with CACHE_OPERATION_DURATION.time():
                try:
                    # Try memory cache first (fastest)
                    cache_level = 'default'
                    if ttl <= CacheConfig.SHORT_EXPIRY:
                        cache_level = 'short'
                    elif ttl >= CacheConfig.LONG_EXPIRY:
                        cache_level = 'long'
                    
                    if cache_key in memory_cache[cache_level]:
                        CACHE_HITS.labels(cache_type='memory').inc()
                        return memory_cache[cache_level][cache_key]
                    
                    # Try Redis if available and circuit breaker is closed
                    if should_use_redis():
                        try:
                            cached_data = redis_client.get(cache_key)
                            if cached_data:
                                # Increment access count for promotion to hot cache
                                redis_client.hincrby(f"{cache_key}:meta", "access_count", 1)
                                
                                CACHE_HITS.labels(cache_type='redis').inc()
                                circuit_breaker.record_success()
                                
                                result = deserialize_data(cached_data)
                                # Also add to memory cache
                                memory_cache[cache_level][cache_key] = result
                                return result
                                
                        except Exception as e:
                            logger.warning(f"Redis cache error: {str(e)}")
                            CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()
                            circuit_breaker.record_failure()

                    # Cache miss - execute function
                    CACHE_MISSES.labels(cache_type='all').inc()
                    result = func(*args, **kwargs)

                    # Store in both caches
                    memory_cache[cache_level][cache_key] = result
                    
                    if should_use_redis():
                        try:
                            serialized_data = serialize_data(result)
                            redis_client.setex(
                                cache_key,
                                apply_jitter(ttl),  # Add jitter to prevent thundering herd
                                serialized_data
                            )
                            # Initialize metadata
                            redis_client.hset(f"{cache_key}:meta", mapping={"access_count": 1})
                            circuit_breaker.record_success()
                        except Exception as e:
                            logger.warning(f"Redis cache write error: {str(e)}")
                            CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
                            circuit_breaker.record_failure()
            
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
                        # Increment access count for promotion to hot cache
                        redis_client.hincrby(f"{cache_key}:meta", "access_count", 1)
                        
                        CACHE_HITS.labels(cache_type='redis').inc()
                        circuit_breaker.record_success()
                        
                        # Check if it's a serialized object or JSON string
                        try:
                            return deserialize_data(cached_result)
                        except:
                            # Try as JSON string
                            json_str = cached_result.decode('utf-8') if isinstance(cached_result, bytes) else cached_result
                            return json.loads(json_str)
                    
                    # Cache miss
                    CACHE_MISSES.labels(cache_type='redis').inc()
                    result = f(*args, **kwargs)
                    
                    # Cache the result with jitter
                    try:
                        # Try to serialize using pickle
                        serialized_data = serialize_data(result)
                        redis_client.setex(
                            cache_key,
                            apply_jitter(expiry),
                            serialized_data
                        )
                    except:
                        # Fallback to JSON if pickle fails
                        redis_client.setex(
                            cache_key,
                            apply_jitter(expiry),
                            json.dumps(result, default=str)
                        )
                    
                    # Initialize metadata
                    redis_client.hset(f"{cache_key}:meta", mapping={"access_count": 1})
                    circuit_breaker.record_success()
                    return result
                    
                except Exception as e:
                    logger.error(f"Cache error in {f.__name__}: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='response', error_type='unknown').inc()
                    circuit_breaker.record_failure()
                    return f(*args, **kwargs)
                
        return wrapper
    return decorator

def invalidate_cache_prefix(prefix: str):
    """Invalidate all cache entries with the given prefix."""
    if not should_use_redis():
        return
        
    try:
        pattern = f"{CacheConfig.CACHE_PREFIX}{prefix}*"
        keys_to_delete = []
        
        # Find all keys with the prefix
        for key in redis_client.scan_iter(match=pattern):
            keys_to_delete.append(key)
            # Also delete any metadata keys
            meta_key = f"{key.decode('utf-8') if isinstance(key, bytes) else key}:meta"
            keys_to_delete.append(meta_key)
            
        # Delete the keys in batches
        if keys_to_delete:
            for i in range(0, len(keys_to_delete), 100):
                batch = keys_to_delete[i:i+100]
                if batch:
                    redis_client.delete(*batch)
                    
        logger.info(f"Invalidated {len(keys_to_delete)//2} cache entries with prefix {prefix}")
        
        # Also clear memory cache entries with the same prefix
        for cache_type in memory_cache:
            keys_to_delete = [k for k in memory_cache[cache_type] if k.startswith(f"{CacheConfig.CACHE_PREFIX}{prefix}")]
            for k in keys_to_delete:
                del memory_cache[cache_type][k]
                
    except Exception as e:
        logger.error(f"Failed to invalidate cache with prefix {prefix}: {str(e)}")
        CACHE_ERRORS.labels(cache_type='redis', error_type='invalidate').inc()

# Initialize Redis on module import
init_cache()

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics and metrics."""
    stats = {
        "memory_cache": {
            cache_type: {
                "size": len(cache),
                "maxsize": getattr(cache, 'maxsize', 0),
                "currsize": getattr(cache, 'currsize', len(cache)),
                "ttl": getattr(cache, 'ttl', None)
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