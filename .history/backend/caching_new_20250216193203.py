"""
Caching utilities for managing Redis and in-memory caches.
Provides multi-level caching with metrics and error handling.
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

def should_use_redis() -> bool:
    """Check if Redis should be used based on configuration and availability."""
    if has_app_context():
        if current_app.config.get('TESTING', False):
            return False
        if current_app.config.get('DISABLE_REDIS', False):
            return False
    return redis_client is not None

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
    try:
        try:
            return pickle.loads(zlib.decompress(data))
        except zlib.error:
            return pickle.loads(data)
    except Exception as e:
        logger.error(f"Deserialization error: {str(e)}")
        CACHE_ERRORS.labels(cache_type='serialization', error_type='deserialize').inc()
        raise CacheSerializationError(f"Failed to deserialize data: {str(e)}")

def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a consistent and safe cache key."""
    key_parts = [prefix]
    if args:
        key_parts.append(str(args))
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(sorted_kwargs))
    
    key = ":".join(key_parts)
    if len(key) > CacheConfig.MAX_KEY_LENGTH:
        key = hashlib.sha256(key.encode()).hexdigest()
    
    return f"{CacheConfig.CACHE_PREFIX}{key}"

def multi_level_cache(ttl: int = CacheConfig.DEFAULT_EXPIRY, prefix: str = ""):
    """Enhanced multi-level cache decorator with metrics and error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(prefix or func.__name__, *args, **kwargs)

            with CACHE_OPERATION_DURATION.time():
                try:
                    # Try Redis first
                    if should_use_redis():
                        try:
                            cached_data = redis_client.get(cache_key)
                            if cached_data:
                                CACHE_HITS.labels(cache_type='redis').inc()
                                return deserialize_data(cached_data)
                        except Exception as e:
                            logger.warning(f"Redis cache error: {str(e)}")
                            CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()

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
                    try:
                        if should_use_redis():
                            redis_client.setex(
                                cache_key,
                                ttl,
                                serialize_data(result)
                            )
                    except Exception as e:
                        logger.warning(f"Redis cache write error: {str(e)}")
                        CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
            
                    memory_cache[cache_level][cache_key] = result
                    return result

                except Exception as e:
                    logger.error(f"Cache error in {func.__name__}: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='general', error_type='unknown').inc()
                    return func(*args, **kwargs)

        return wrapper
    return decorator

def cache_response(expiry: int = CacheConfig.DEFAULT_EXPIRY):
    """Enhanced cache decorator for API responses with better error handling."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not should_use_redis():
                return f(*args, **kwargs)

            cache_key = generate_cache_key(
                f.__name__,
                request.method,
                request.args.to_dict(),
                *args,
                **kwargs
            )
            
            with CACHE_OPERATION_DURATION.time():
                try:
                    cached_result = redis_client.get(cache_key)
                    if cached_result:
                        CACHE_HITS.labels(cache_type='redis').inc()
                        return json.loads(cached_result)
                    
                    CACHE_MISSES.labels(cache_type='redis').inc()
                    result = f(*args, **kwargs)
                    
                    redis_client.setex(
                        cache_key,
                        expiry,
                        json.dumps(result, default=str)
                    )
                    return result
                    
                except Exception as e:
                    logger.error(f"Cache error in {f.__name__}: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='response', error_type='unknown').inc()
                    return f(*args, **kwargs)
                
        return wrapper
    return decorator

def get_from_cache(key: str, cache_type: str = 'default') -> Optional[Any]:
    """Retrieve an item from the cache with metrics."""
    with CACHE_OPERATION_DURATION.time():
        try:
            if should_use_redis():
                try:
                    data = redis_client.get(key)
                    if data:
                        CACHE_HITS.labels(cache_type='redis').inc()
                        return deserialize_data(data)
                except Exception as e:
                    logger.warning(f"Redis get error: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='redis', error_type='read').inc()
            
            if key in memory_cache[cache_type]:
                CACHE_HITS.labels(cache_type='memory').inc()
                return memory_cache[cache_type][key]
            
            CACHE_MISSES.labels(cache_type='all').inc()
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            CACHE_ERRORS.labels(cache_type='general', error_type='read').inc()
            return None

def add_to_cache(key: str, value: Any, expiry: int = CacheConfig.DEFAULT_EXPIRY, 
                cache_type: str = 'default') -> bool:
    """Add an item to the cache with metrics and error handling."""
    with CACHE_OPERATION_DURATION.time():
        try:
            if should_use_redis():
                try:
                    redis_client.setex(key, expiry, serialize_data(value))
                except Exception as e:
                    logger.warning(f"Redis set error: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='redis', error_type='write').inc()
            
            memory_cache[cache_type][key] = value
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            CACHE_ERRORS.labels(cache_type='general', error_type='write').inc()
            return False

def remove_from_cache(key: str, cache_type: str = 'default') -> bool:
    """Remove an item from the cache with metrics."""
    with CACHE_OPERATION_DURATION.time():
        try:
            success = True
            if should_use_redis():
                try:
                    redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis delete error: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='redis', error_type='delete').inc()
                    success = False
            
            if key in memory_cache[cache_type]:
                del memory_cache[cache_type][key]
            
            return success
            
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            CACHE_ERRORS.labels(cache_type='general', error_type='delete').inc()
            return False

def clear_cache(cache_type: Optional[str] = None) -> bool:
    """Clear the entire cache or a specific cache type with metrics."""
    with CACHE_OPERATION_DURATION.time():
        try:
            success = True
            if should_use_redis():
                try:
                    if cache_type:
                        pattern = f"{CacheConfig.CACHE_PREFIX}{cache_type}:*"
                        keys = redis_client.keys(pattern)
                        if keys:
                            redis_client.delete(*keys)
                    else:
                        redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis clear error: {str(e)}")
                    CACHE_ERRORS.labels(cache_type='redis', error_type='clear').inc()
                    success = False
            
            if cache_type:
                if cache_type in memory_cache:
                    memory_cache[cache_type].clear()
            else:
                for cache in memory_cache.values():
                    cache.clear()
            
            return success
            
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            CACHE_ERRORS.labels(cache_type='general', error_type='clear').inc()
            return False

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics and metrics."""
    stats = {
        "memory_cache": {
            cache_type: {
                "size": len(cache),
                "maxsize": cache.maxsize,
                "currsize": cache.currsize
            }
            for cache_type, cache in memory_cache.items()
        },
        "redis": {
            "connected": should_use_redis(),
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
                ) * 100
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis info: {str(e)}")
    
    return stats 