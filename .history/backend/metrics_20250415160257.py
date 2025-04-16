"""
Metrics module that reexports metrics from monitoring.py for backward compatibility.
"""

from backend.monitoring import (
    API_REQUESTS, 
    REQUEST_LATENCY, 
    API_RESPONSE_SIZE,
    DB_QUERY_LATENCY,
    CACHE_HITS,
    CACHE_MISSES,
    ERROR_COUNT,
    ACTIVE_USERS,
    WORD_COUNTS,
    MEMORY_USAGE,
    CPU_USAGE,
    SLOW_QUERIES,
    CACHE_SIZE
)

# Define API_ERRORS directly since it's not in monitoring.py
from prometheus_client import Counter
API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
) 