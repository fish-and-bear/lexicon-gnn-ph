from functools import wraps
from cachetools import TTLCache
import redis
import pickle
import os
from dotenv import load_dotenv
import json


load_dotenv()

# Initialize the cache
cache = TTLCache(maxsize=1000, ttl=3600)

# In-memory cache
local_cache = TTLCache(maxsize=1000, ttl=3600)

# Redis cache
redis_url = os.getenv('REDIS_URL')
redis_client = redis.Redis.from_url(redis_url)

def init_cache(redis_url):
    global redis_client
    redis_client = redis.Redis.from_url(redis_url)

def multi_level_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not redis_client:
            return func(*args, **kwargs)

        cache_key = f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}"
        cached_result = redis_client.get(cache_key)

        if cached_result:
            return json.loads(cached_result)

        result = func(*args, **kwargs)
        redis_client.setex(cache_key, int(os.getenv('CACHE_EXPIRATION', 3600)), json.dumps(result))
        return result

    return wrapper

# Optionally, you can add helper functions to interact with the cache.

def get_from_cache(key):
    """Retrieve an item from the cache by key."""
    return cache.get(key)

def add_to_cache(key, value):
    """Add an item to the cache with a specified key and value."""
    cache[key] = value

def remove_from_cache(key):
    """Remove an item from the cache by key."""
    if key in cache:
        del cache[key]

def clear_cache():
    """Clear the entire cache."""
    cache.clear()
