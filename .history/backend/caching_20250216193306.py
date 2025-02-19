from functools import wraps
from cachetools import TTLCache
import redis
import pickle
import os
from dotenv import load_dotenv
import json
import functools
from datetime import datetime
from flask import current_app, has_app_context
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the cache
cache = TTLCache(maxsize=1000, ttl=3600)

# In-memory cache for when Redis is not available
local_cache = TTLCache(maxsize=1000, ttl=3600)

# Initialize Redis client
redis_client = None

def init_cache(redis_url=None):
    """Initialize Redis client with error handling."""
    global redis_client
    if redis_client is not None:
        return

    try:
        if redis_url:
            redis_client = redis.from_url(redis_url)
        else:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                password=os.getenv('REDIS_PASSWORD', ''),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1
            )
        # Test the connection
        redis_client.ping()
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.warning(f"Redis connection failed: {str(e)}")
        redis_client = None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {str(e)}")
        redis_client = None

def should_use_redis():
    """Check if Redis should be used based on configuration and availability."""
    if has_app_context() and current_app.config.get('TESTING', False):
        return False
    if has_app_context() and current_app.config.get('DISABLE_REDIS', False):
        return False
    return redis_client is not None

def multi_level_cache(func):
    """Decorator that implements multi-level caching with Redis and local cache."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not should_use_redis():
            return func(*args, **kwargs)

        cache_key = f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}"
        
        try:
            # Try Redis first
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)

            # If not in Redis, check local cache
            local_result = local_cache.get(cache_key)
            if local_result:
                return local_result

            # If not found, compute result
            result = func(*args, **kwargs)

            # Store in both caches
            try:
                redis_client.setex(
                    cache_key,
                    int(os.getenv('CACHE_EXPIRATION', 3600)),
                    json.dumps(result)
                )
            except:
                pass  # Silently fail Redis storage
            
            local_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Cache error: {str(e)}")
            return func(*args, **kwargs)

    return wrapper

def cache_response(expiry=3600):
    """Cache decorator that stores API responses."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not should_use_redis():
                return f(*args, **kwargs)

            cache_key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
            
            try:
                # Try to get cached result
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # If no cached result, execute function and cache result
                result = f(*args, **kwargs)
                redis_client.setex(
                    cache_key,
                    expiry,
                    json.dumps(result, default=str)
                )
                return result
            except Exception as e:
                logger.error(f"Cache error: {str(e)}")
                return f(*args, **kwargs)
                
        return wrapper
    return decorator

# Initialize Redis on module import
init_cache()

# Helper functions for cache management
def get_from_cache(key):
    """Retrieve an item from the cache by key."""
    if should_use_redis():
        try:
            return redis_client.get(key)
        except:
            pass
    return local_cache.get(key)

def add_to_cache(key, value, expiry=3600):
    """Add an item to the cache with a specified key and value."""
    if should_use_redis():
        try:
            redis_client.setex(key, expiry, value)
        except:
            pass
    local_cache[key] = value

def remove_from_cache(key):
    """Remove an item from the cache by key."""
    if should_use_redis():
        try:
            redis_client.delete(key)
        except:
            pass
    if key in local_cache:
        del local_cache[key]

def clear_cache():
    """Clear the entire cache."""
    if should_use_redis():
        try:
            redis_client.flushdb()
        except:
            pass
    local_cache.clear()
