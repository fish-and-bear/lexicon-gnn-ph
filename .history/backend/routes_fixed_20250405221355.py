"""
API routes for the Filipino Dictionary application.
This module provides comprehensive RESTful endpoints for accessing the dictionary data.
"""

from flask import Blueprint, jsonify, request, current_app, g, abort, send_file, make_response
from sqlalchemy import or_, and_, func, desc, text, distinct, cast
from sqlalchemy.orm import joinedload, contains_eager, selectinload, Session, subqueryload
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import structlog
from backend.models import (
    Word, Definition, Etymology, Relation, Affixation,
    PartOfSpeech, Language, Pronunciation, Credit,
    WordForm, WordTemplate, DefinitionCategory, DefinitionLink, DefinitionRelation
)
from backend.database import db, cached_query
from backend.dictionary_manager import (
    normalize_lemma, extract_etymology_components, extract_language_codes,
    RelationshipType, RelationshipCategory, BaybayinRomanizer
)
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_client.metrics import MetricWrapperBase
from collections import defaultdict
import logging
from sqlalchemy.exc import SQLAlchemyError
from flask_graphql import GraphQLView
import time
import random # Add random import
from flask_limiter.util import get_remote_address # Import limiter utility
from backend.extensions import limiter # Import the limiter instance from extensions
# from backend.search_tasks import log_search_query

# Set up logging
logger = structlog.get_logger(__name__)

# Initialize blueprint
bp = Blueprint("api", __name__, url_prefix='/api/v2')

@bp.route("/search", methods=["GET"])
@cached_query(timeout=300)  # Increase cache timeout to 5 minutes for search results
@limiter.limit("20 per minute", key_func=get_remote_address) # Apply rate limit
def search():
    """Search for words with optimized performance for high traffic."""
    try:
        # --- Start: Validate input using SearchQuerySchema ---
        search_schema = SearchQuerySchema()
        try:
            # Load and validate query parameters from request.args
            search_args = search_schema.load(request.args)
        except ValidationError as err:
            logger.warning("Search query validation failed", errors=err.messages)
            # Return validation errors to the client
            return jsonify({"error": "Invalid search parameters", "details": err.messages}), 400
        # --- End: Validate input using SearchQuerySchema ---

        # Track request metrics
        REQUEST_COUNT.inc()
        API_REQUESTS.labels(endpoint="search", method="GET").inc()
        start_time = time.time()
        
        # Use validated arguments from the loaded schema data
        query = search_args['q'] # 'q' is required by the schema
        mode = search_args['mode'] # has default in schema
        language = search_args.get('language', 'tl') # Use .get with default, as schema might allow None
        pos = search_args.get('pos') # Use .get, as it's optional
        limit = search_args['limit'] # has default in schema
        offset = search_args['offset'] # has default in schema

        # Handle include_full separately for now, or add it to the schema
        include_full = request.args.get("include_full", "false").lower() == "true"
        
        # Add a hard limit on offset to prevent excessive deep pagination
        if offset > 1000:
            return jsonify({
                "error": "Pagination limit exceeded",
                "message": "Please use a more specific search query instead of deep pagination"
            }), 400
            
        # Add query execution timeout to prevent long-running queries
        # Consider setting this globally or per-request based on configuration
        try:
            db.session.execute(text("SET statement_timeout TO '3000'"))  # 3 seconds timeout
        except SQLAlchemyError as e:
            logger.warning(f"Could not set statement timeout: {e}")
        
        # Build the base query - use normalized_query for better index usage
        normalized_query = normalize_lemma(query)
        
        # Check for Baybayin query
        has_baybayin = any(0x1700 <= ord(c) <= 0x171F for c in query)
        baybayin_filter = None

        # Rest of the search function implementation
        # ...
        
    except Exception as e:
        logger.error(f"Search function error: {str(e)}", exc_info=True)
        API_ERRORS.labels(endpoint="search", error_type=type(e).__name__).inc()
        return jsonify({"error": f"Unexpected error ({type(e).__name__})", "message": str(e)}), 500 