"""
Simplified metrics module for Baybayin routes
"""

from prometheus_client import Counter, Histogram, REGISTRY

# Clear existing metrics to avoid duplication
for collector in list(REGISTRY._collector_to_names.keys()):
    if hasattr(collector, '_type'):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

# Define metrics that aren't already defined in routes.py
# These metrics are imported in routes.py
API_REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method']
)

API_ERRORS = Counter(
    'api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

REQUEST_LATENCY = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

# Additional metrics not used in routes.py
API_RESPONSE_SIZE = Counter(
    'api_response_size_bytes',
    'API response size in bytes',
    ['endpoint']
)

DB_QUERY_LATENCY = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['endpoint']
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Cache hits',
    ['endpoint']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Cache misses',
    ['endpoint']
)

ERROR_COUNT = Counter(
    'error_count_total',
    'Total error count',
    ['endpoint']
)

ACTIVE_USERS = Counter(
    'active_users_total',
    'Total active users',
    ['endpoint']
)

WORD_COUNTS = Counter(
    'word_counts_total',
    'Total word counts',
    ['endpoint']
)

MEMORY_USAGE = Counter(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['endpoint']
)

CPU_USAGE = Counter(
    'cpu_usage_seconds',
    'CPU usage in seconds',
    ['endpoint']
)

SLOW_QUERIES = Counter(
    'slow_queries_total',
    'Total slow queries',
    ['endpoint']
)

CACHE_SIZE = Counter(
    'cache_size_bytes',
    'Cache size in bytes',
    ['endpoint']
)

# Define REQUEST_COUNT here so it's only defined once
REQUEST_COUNT = Counter(
    'request_count',
    'Flask Request Count',
    ['method', 'endpoint', 'status']
) 