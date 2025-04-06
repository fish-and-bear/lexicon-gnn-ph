"""Caching utilities for Flask routes."""

from functools import wraps
from flask import request, current_app, make_response
import hashlib
import time
import json

def get_cache_key(prefix, request_args=None):
    """
    Generate a cache key based on the request path and query parameters.
    
    Args:
        prefix (str): Prefix for the cache key
        request_args (dict, optional): Request arguments to include in the cache key
        
    Returns:
        str: The cache key
    """
    if request_args is None:
        request_args = request.args
        
    # Sort the arguments to ensure consistent cache keys
    args_str = "&".join(f"{k}={v}" for k, v in sorted(request_args.items()))
    path = request.path
    
    # Create a hash of the path and arguments
    key_data = f"{path}?{args_str}"
    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    
    return f"{prefix}:{key_hash}"

def cached_query(timeout=300, key_prefix="query"):
    """
    Decorator for caching Flask route responses.
    
    Args:
        timeout (int): Cache timeout in seconds
        key_prefix (str): Prefix for the cache key
        
    Returns:
        function: Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if caching is enabled
            if not current_app.config.get("CACHE_ENABLED", True):
                return f(*args, **kwargs)
                
            # Generate a cache key
            cache_key = get_cache_key(key_prefix, request.args)
            
            # Check if the response is in the cache
            cached = current_app.cache.get(cache_key)
            
            if cached is not None:
                # Return the cached response
                return make_response(cached)
                
            # Get the response from the decorated function
            response = f(*args, **kwargs)
            
            # Cache the response
            current_app.cache.set(cache_key, response.get_data(), timeout=timeout)
            
            return response
        
        return decorated_function
    
    return decorator 