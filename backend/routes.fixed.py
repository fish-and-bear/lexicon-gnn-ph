"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, request, jsonify, send_file, abort, current_app, g, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast, not_, case, exists, extract
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload, raiseload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
import logging
from sqlalchemy.exc import SQLAlchemyError
import time
import random
import json
import csv
import os
import io
import re
import zipfile
import math
import pickle
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Histogram, REGISTRY
# Comment out problematic import
# from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict

from backend.models import (
    Word, Definition, Etymology, Pronunciation, Relation, DefinitionCategory,
    DefinitionLink, DefinitionRelation, Affixation, WordForm, WordTemplate,
    PartOfSpeech, Credit
)
from backend.database import db, cached_query, get_cache_client
from backend.dictionary_manager import (
    normalize_lemma, extract_etymology_components, extract_language_codes,
    RelationshipType, RelationshipCategory, BaybayinRomanizer
)
from backend.utils.word_processing import normalize_word
from backend.utils.rate_limiting import limiter
from backend.utils.ip import get_remote_address

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__, url_prefix='/api/v2')

# Get a cache client - initialize it once
try:
    cache_client = get_cache_client()
except Exception as e:
    logger.error(f"Failed to initialize cache client: {e}")
    cache_client = None

# Define request latency histogram
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Flask Request Latency',
    ['endpoint']
)

# Define request counter
REQUEST_COUNT = Counter(
    'request_count', 
    'Flask Request Count',
    ['method', 'endpoint', 'status']
)

# Unregister existing metrics to avoid duplication
for collector in list(REGISTRY._collector_to_names.keys()):
    # Check if the collector is a metric by checking for _type attribute
    if hasattr(collector, '_type'):
        try:
            REGISTRY.unregister(collector)
        except Exception as e:
            logger.error(f"Error unregistering metric: {e}")
            pass

# Test endpoint - quick connection test without hitting the database
@bp.route('/test', methods=['GET'])
def test_api():
    """Simple test endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'API is running',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

# Health check endpoint
# @bp.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint."""
#     try:
#         # Check database connection
#         db.session.execute(text('SELECT 1'))
#         return jsonify({
#             'status': 'healthy',
#             'database': 'connected',
#             'timestamp': datetime.now(timezone.utc).isoformat()
#         })
#     except Exception as e:
#         logger.error('Health check failed', error=str(e))
#         return jsonify({
#             'status': 'unhealthy',
#             'error': str(e),
#             'timestamp': datetime.now(timezone.utc).isoformat()
#         })                    derived.baybayin_form = d.baybayin_form
                    word.derived_words.append(derived)
            except Exception as e:
                logger.error(f"Error loading derived words for word {word_id}: {e}", exc_info=False)
                word.derived_words = []
        else:
            word.derived_words = []

        # Load credits if requested
        if include_credits:
            try:
                sql_credits = "SELECT id, credit FROM credits WHERE word_id = :word_id"
                cred_result = db.session.execute(text(sql_credits), {"word_id": word_id}).fetchall()
                word.credits = []
                for c_row in cred_result:
                    credit = Credit()
                    credit.id = c_row.id
                    credit.credit = c_row.credit
                    word.credits.append(credit)
            except Exception as e:
                logger.error(f"Error loading credits for word {word_id}: {e}", exc_info=False)
                word.credits = []

        # Initialize other relationships as empty lists if not included/loaded
        # Ensures attributes exist even if loading fails or is skipped
        if not hasattr(word, 'root_affixations'): word.root_affixations = []
        if not hasattr(word, 'affixed_affixations'): word.affixed_affixations = []
        if not hasattr(word, 'forms'): word.forms = []
        if not hasattr(word, 'templates'): word.templates = []
        if not hasattr(word, 'definition_relations'): word.definition_relations = []
        if not hasattr(word, 'related_definitions'): word.related_definitions = []
        if not hasattr(word, 'credits'): word.credits = [] # Initialize if include_credits was false

        # --- Add loading logic for Affixations, Forms, Templates if needed --- 
        # Remember to add try/except blocks if you implement them

        # Cache the result if we have a cache client
        if cache_client and cache_key:
            try:
                # Detach the object from the session before pickling to avoid session-related issues
                temp_word = db.session.merge(word) # Merge ensures it's in session
                db.session.expunge(temp_word)
                pickled_word = pickle.dumps(temp_word)
                cache_client.set(cache_key, pickled_word, timeout=600)  # Cache for 10 minutes
            except Exception as e:
                logger.warning(f"Cache storage error for word_id={word_id}: {e}")

        # Return the populated word object
        return word
    except ValueError as ve:
        # Handle the specific case where the word ID itself wasn't found
         logger.error(f"Value error in _fetch_word_details for word_id {word_id}: {ve}")
         raise Exception(f"Failed to retrieve details: {ve}") from ve # Re-raise with specific message
    except SQLAlchemyError as db_error:
        logger.error(f"Database error in _fetch_word_details for word_id {word_id}: {db_error}", exc_info=True)
        raise Exception(f"Database error retrieving details for word ID {word_id}") from db_error
    except Exception as e:
        logger.error(f"Unexpected error in _fetch_word_details for word_id {word_id}: {str(e)}", exc_info=True)
        raise Exception(f"Failed to retrieve details for word ID {word_id}") from e
