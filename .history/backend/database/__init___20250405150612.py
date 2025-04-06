"""
Database initialization and connection management.
"""

import logging
import threading
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from flask_sqlalchemy import SQLAlchemy
from contextlib import contextmanager
import os
from typing import Set, Optional, Callable
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# Global state
_initialized_features: Set[str] = set()
_state_lock = threading.RLock()  # Reentrant lock for all state changes

# Redis configuration
REDIS_TIMEOUT = int(os.getenv('REDIS_TIMEOUT', 3600))  # 1 hour default

def create_tables(app):
    """Create all database tables."""
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

def init_db(app):
    """Initialize database with Flask app."""
    try:
        # Get database configuration from environment variables
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'fil_dict_db')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', 'postgres')
        
        # Construct database URI
        database_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Configure SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
            'max_overflow': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
            'pool_timeout': int(os.getenv('DB_STATEMENT_TIMEOUT', 30000)),
            'pool_recycle': int(os.getenv('DB_IDLE_TIMEOUT', 60000)),
        }
        
        # Initialize SQLAlchemy with app
        db.init_app(app)
        
        # Create Session within app context
        with app.app_context():
            # Create scoped session
            Session = scoped_session(sessionmaker(bind=db.get_engine(app)))
            
            # Create tables
            db.create_all()
        
        logger.info("Database initialized successfully")
        return Session
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

def teardown_db():
    """Clean up database resources."""
    try:
        # Remove all sessions
        db.session.remove()
        
        # Dispose of the engine
        if hasattr(db, 'engine') and db.engine:
            db.engine.dispose()
            
        logger.info("Database resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up database resources: {str(e)}")
        raise

def is_feature_initialized(feature: str) -> bool:
    """Check if a feature has been initialized."""
    with _state_lock:
        return feature in _initialized_features

def mark_feature_initialized(feature: str) -> None:
    """Mark a feature as initialized."""
    with _state_lock:
        _initialized_features.add(feature)

@contextmanager
def feature_lock(feature: str):
    """Lock for feature initialization."""
    with _state_lock:
        if is_feature_initialized(feature):
            yield False  # Feature already initialized
            return
        yield True  # Feature needs initialization
        mark_feature_initialized(feature)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = db.session
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def get_db():
    """Get database session."""
    return db.session 

def cached_query(timeout: int = REDIS_TIMEOUT, key_prefix: str = "") -> Callable:
    """
    Decorator for caching function results in Redis.
    
    Args:
        timeout: Cache timeout in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not redis_client:
                return f(*args, **kwargs)
            
            # Generate cache key
            cache_key_str = f"{key_prefix}:{f.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = redis_client.get(cache_key_str)
            if cached_value:
                try:
                    return json.loads(cached_value)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode cached value for key: {cache_key_str}")
            
            # Get fresh value
            result = f(*args, **kwargs)
            
            # Cache the result
            try:
                redis_client.setex(
                    cache_key_str,
                    timeout,
                    json.dumps(result, default=str)
                )
            except (TypeError, json.JSONEncodeError) as e:
                logger.warning(f"Failed to cache result for key {cache_key_str}: {e}")
            
            return result
        return wrapper
    return decorator 