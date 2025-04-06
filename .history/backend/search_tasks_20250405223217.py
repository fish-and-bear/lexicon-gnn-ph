"""
Search tasks module for background processing of search queries.

This module provides functions for logging search queries and updating popular words.
"""

import logging
import time
from collections import defaultdict
from threading import Lock
import os

# Setup logger
logger = logging.getLogger(__name__)

# In-memory storage for search queries (will be lost on restart)
recent_searches = []
popular_words = []
search_counts = defaultdict(int)
search_lock = Lock()

def log_search_query(query, user_id=None, ip=None, results_count=0, language=None):
    """
    Log a search query to track popular searches.
    
    Args:
        query (str): The search query
        user_id (str, optional): Anonymous user identifier
        ip (str, optional): IP address (hashed)
        results_count (int, optional): Number of results returned
        language (str, optional): Language of the search
        
    Returns:
        bool: True if logging was successful
    """
    if not query or not isinstance(query, str):
        return False
        
    # Normalize query
    query = query.lower().strip()
    if not query:
        return False

    # Log the search
    timestamp = time.time()
    with search_lock:
        # Add to recent searches
        search_entry = {
            'query': query,
            'timestamp': timestamp,
            'results': results_count,
            'language': language
        }
        recent_searches.append(search_entry)
        
        # Limit the size of recent searches
        if len(recent_searches) > 1000:
            recent_searches.pop(0)
            
        # Update search counts
        search_counts[query] += 1
    
    logger.info(f"Search query logged: {query} (results: {results_count})")
    return True

def get_popular_searches(limit=10, language=None):
    """
    Get the most popular search queries.
    
    Args:
        limit (int): Maximum number of results to return
        language (str, optional): Filter by language
        
    Returns:
        list: List of popular search queries
    """
    with search_lock:
        # Filter by language if specified
        if language:
            filtered_searches = {k: v for k, v in search_counts.items() 
                               if any(s['language'] == language for s in recent_searches 
                                     if s['query'] == k)}
        else:
            filtered_searches = search_counts
            
        # Sort by count (descending)
        sorted_searches = sorted(filtered_searches.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [{'query': q, 'count': c} for q, c in sorted_searches[:limit]]

def refresh_popular_words():
    """
    Refresh the list of popular words based on recent searches.
    This would typically connect to the database and update a cache.
    
    Returns:
        list: Updated list of popular words
    """
    global popular_words
    
    with search_lock:
        # In a real implementation, this would query the database
        # For now, just use the in-memory search counts
        sorted_words = sorted(search_counts.items(), 
                             key=lambda x: x[1], reverse=True)
        popular_words = [word for word, _ in sorted_words[:50]]
        
    logger.info(f"Popular words refreshed: {len(popular_words)} words")
    return popular_words

def get_popular_words(limit=10):
    """
    Get the current list of popular words.
    
    Args:
        limit (int): Maximum number of words to return
        
    Returns:
        list: List of popular words
    """
    return popular_words[:limit]

def cleanup():
    """
    Clean up resources used by the search tasks module.
    """
    global recent_searches, popular_words, search_counts
    
    with search_lock:
        recent_searches = []
        popular_words = []
        search_counts = defaultdict(int)
    
    logger.info("Search tasks resources cleaned up") 