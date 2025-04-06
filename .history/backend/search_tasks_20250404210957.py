"""
Background tasks for search functionality.
This module handles asynchronous tasks for logging search queries and refreshing materialized views.
"""

import logging
from sqlalchemy import text
from database import db
import threading
import time
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Global thread-safe queue for background processing
query_log_queue = []
queue_lock = threading.Lock()

# Flag to control the background thread
keep_running = True

def log_search_query(query, language=None, word_id=None, query_type="search"):
    """
    Queue a search query to be logged asynchronously.
    
    Args:
        query: The search query text
        language: Optional language code 
        word_id: Optional word ID if known
        query_type: Type of query (search, suggestion, selection)
    """
    if not query:
        return
        
    with queue_lock:
        query_log_queue.append({
            "query": query,
            "language": language,
            "word_id": word_id,
            "type": query_type,
            "timestamp": datetime.now().isoformat()
        })

def refresh_popular_words():
    """Refresh the materialized view of popular words."""
    try:
        with db.engine.connect() as conn:
            conn.execute(text("REFRESH MATERIALIZED VIEW popular_words;"))
            conn.commit()
        logger.info("Successfully refreshed popular_words materialized view")
        return True
    except Exception as e:
        logger.error(f"Failed to refresh popular_words view: {e}")
        return False

def _log_processor_thread():
    """Background thread to process the query log queue."""
    logger.info("Starting search log processor thread")
    
    last_refresh_time = time.time()
    REFRESH_INTERVAL = 3600  # Refresh popular_words view every hour
    
    while keep_running:
        # Process any pending log entries
        entries_to_process = []
        with queue_lock:
            if query_log_queue:
                # Take up to 50 entries to process in a batch
                entries_to_process = query_log_queue[:50]
                del query_log_queue[:len(entries_to_process)]
        
        if entries_to_process:
            try:
                with db.engine.connect() as conn:
                    # Prepare the batch insert
                    stmt = text("""
                    INSERT INTO search_logs (query_text, language, type, word_id)
                    VALUES (:query, :language, :type, :word_id)
                    """)
                    
                    # Execute the batch insert
                    params = [
                        {
                            "query": entry["query"],
                            "language": entry["language"],
                            "type": entry["type"],
                            "word_id": entry["word_id"]
                        } for entry in entries_to_process
                    ]
                    
                    conn.execute(stmt, params)
                    conn.commit()
                    logger.debug(f"Logged {len(entries_to_process)} search queries")
            except Exception as e:
                logger.error(f"Error logging search queries: {e}")
                # If processing fails, put entries back in queue
                with queue_lock:
                    query_log_queue.extend(entries_to_process)
        
        # Check if we should refresh the popular_words view
        current_time = time.time()
        if current_time - last_refresh_time > REFRESH_INTERVAL:
            refresh_popular_words()
            last_refresh_time = current_time
        
        # Sleep to avoid busy waiting
        time.sleep(1)
    
    logger.info("Search log processor thread stopped")

def start_background_tasks():
    """Start the background processing thread."""
    global keep_running
    keep_running = True
    
    # Create and start the background thread
    thread = threading.Thread(target=_log_processor_thread, daemon=True)
    thread.start()
    logger.info("Search background tasks started")
    return thread

def stop_background_tasks():
    """Stop the background processing thread."""
    global keep_running
    keep_running = False
    logger.info("Search background tasks stopping...")

# Register for application startup/shutdown
def initialize():
    """Initialize search tasks when application starts."""
    thread = start_background_tasks()
    return thread

def cleanup():
    """Clean up search tasks when application stops."""
    stop_background_tasks()
    # Process any remaining entries in the queue
    with queue_lock:
        if query_log_queue:
            logger.info(f"Processing {len(query_log_queue)} remaining search log entries")
            try:
                with db.engine.connect() as conn:
                    stmt = text("""
                    INSERT INTO search_logs (query_text, language, type, word_id)
                    VALUES (:query, :language, :type, :word_id)
                    """)
                    
                    params = [
                        {
                            "query": entry["query"],
                            "language": entry["language"],
                            "type": entry["type"],
                            "word_id": entry["word_id"]
                        } for entry in query_log_queue
                    ]
                    
                    conn.execute(stmt, params)
                    conn.commit()
            except Exception as e:
                logger.error(f"Error processing remaining search logs: {e}") 