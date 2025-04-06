"""
Simplified database module with basic SQLAlchemy setup.
This version has fewer dependencies for easier testing.
"""

from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import logging
import time
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# In-memory cache for simple function results
_cache = {}
_cache_timeout = {}

def init_db(app):
    """Initialize database."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")

def teardown_db():
    """Clean up database resources."""
    if hasattr(db, 'session'):
        db.session.remove()

def cached_query(timeout: int = 300, key_prefix: str = None):
    """
    Decorator to cache function results in memory.
    Much simpler than the original Redis-based implementation.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Optional prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            key = f"{key_prefix or func.__name__}:{hash(str(args))}:{hash(str(kwargs))}"
            
            # Check if result is in cache and not expired
            if key in _cache and key in _cache_timeout:
                if _cache_timeout[key] > datetime.now():
                    return _cache[key]
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            _cache[key] = result
            _cache_timeout[key] = datetime.now() + timedelta(seconds=timeout)
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str = None):
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Optional pattern to match cache keys
    """
    if pattern is None:
        _cache.clear()
        _cache_timeout.clear()
        return
    
    # Convert glob pattern to regex for simple matching
    import re
    pattern = pattern.replace('*', '.*')
    matcher = re.compile(pattern)
    
    # Remove matching keys
    for key in list(_cache.keys()):
        if matcher.match(key):
            _cache.pop(key, None)
            _cache_timeout.pop(key, None) 
Simplified database module with basic SQLAlchemy setup.
This version has fewer dependencies for easier testing.
"""

from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import logging
import time
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# In-memory cache for simple function results
_cache = {}
_cache_timeout = {}

def init_db(app):
    """Initialize database."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")

def teardown_db():
    """Clean up database resources."""
    if hasattr(db, 'session'):
        db.session.remove()

def cached_query(timeout: int = 300, key_prefix: str = None):
    """
    Decorator to cache function results in memory.
    Much simpler than the original Redis-based implementation.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Optional prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            key = f"{key_prefix or func.__name__}:{hash(str(args))}:{hash(str(kwargs))}"
            
            # Check if result is in cache and not expired
            if key in _cache and key in _cache_timeout:
                if _cache_timeout[key] > datetime.now():
                    return _cache[key]
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            _cache[key] = result
            _cache_timeout[key] = datetime.now() + timedelta(seconds=timeout)
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str = None):
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Optional pattern to match cache keys
    """
    if pattern is None:
        _cache.clear()
        _cache_timeout.clear()
        return
    
    # Convert glob pattern to regex for simple matching
    import re
    pattern = pattern.replace('*', '.*')
    matcher = re.compile(pattern)
    
    # Remove matching keys
    for key in list(_cache.keys()):
        if matcher.match(key):
            _cache.pop(key, None)
            _cache_timeout.pop(key, None) 
Simplified database module with basic SQLAlchemy setup.
This version has fewer dependencies for easier testing.
"""

from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import logging
import time
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# In-memory cache for simple function results
_cache = {}
_cache_timeout = {}

def init_db(app):
    """Initialize database."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")

def teardown_db():
    """Clean up database resources."""
    if hasattr(db, 'session'):
        db.session.remove()

def cached_query(timeout: int = 300, key_prefix: str = None):
    """
    Decorator to cache function results in memory.
    Much simpler than the original Redis-based implementation.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Optional prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            key = f"{key_prefix or func.__name__}:{hash(str(args))}:{hash(str(kwargs))}"
            
            # Check if result is in cache and not expired
            if key in _cache and key in _cache_timeout:
                if _cache_timeout[key] > datetime.now():
                    return _cache[key]
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            _cache[key] = result
            _cache_timeout[key] = datetime.now() + timedelta(seconds=timeout)
            
            return result
        return wrapper
    return decorator

def invalidate_cache(pattern: str = None):
    """
    Invalidate cache entries matching pattern.
    
    Args:
        pattern: Optional pattern to match cache keys
    """
    if pattern is None:
        _cache.clear()
        _cache_timeout.clear()
        return
    
    # Convert glob pattern to regex for simple matching
    import re
    pattern = pattern.replace('*', '.*')
    matcher = re.compile(pattern)
    
    # Remove matching keys
    for key in list(_cache.keys()):
        if matcher.match(key):
            _cache.pop(key, None)
            _cache_timeout.pop(key, None) 