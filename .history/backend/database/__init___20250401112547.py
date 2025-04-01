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
from typing import Set, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

# Global state
_initialized_features: Set[str] = set()
_state_lock = threading.RLock()  # Reentrant lock for all state changes

def init_db(app):
    """Initialize database with Flask app."""
    try:
        # Configure SQLAlchemy
        app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 1800,
        }
        
        # Initialize SQLAlchemy with app
        db.init_app(app)
        
        # Create scoped session
        Session = scoped_session(sessionmaker(bind=db.get_engine(app)))
        
        logger.info("Database initialized successfully")
        return Session
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
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