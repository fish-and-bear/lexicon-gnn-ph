from functools import wraps
from cachetools import TTLCache
import redis
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

# In-memory cache
local_cache = TTLCache(maxsize=1000, ttl=3600)

# Redis cache
redis_url = os.getenv('REDIS_URL')
if redis_url:
    redis_client = redis.from_url(redis_url)
else:
    redis_client = None

def multi_level_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = f"{func.__name__}:{args}:{kwargs}"
        
        # Check local cache
        if key in local_cache:
            return local_cache[key]
        
        # Check Redis cache
        if redis_client:
            redis_result = redis_client.get(key)
            if redis_result:
                result = pickle.loads(redis_result)
                local_cache[key] = result
                return result
        
        # If not in cache, call the function
        result = func(*args, **kwargs)
        
        # Store in local cache
        local_cache[key] = result
        
        # Store in Redis cache if available
        if redis_client:
            redis_client.set(key, pickle.dumps(result), ex=3600)
        
        return result
    return wrapper