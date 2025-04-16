"""
Simplified metrics module for Baybayin routes
"""

# Try to import prometheus_client, create dummy metrics if not available
try:
    from prometheus_client import Counter, Histogram
    
    # Define basic metrics
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
except ImportError:
    # Create dummy metric classes if prometheus_client is not available
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
            
        def labels(self, *args, **kwargs):
            return self
            
        def inc(self, *args, **kwargs):
            pass
            
        def observe(self, *args, **kwargs):
            pass
    
    API_REQUESTS = DummyMetric()
    API_ERRORS = DummyMetric()
    REQUEST_LATENCY = DummyMetric()
    API_RESPONSE_SIZE = DummyMetric()
    DB_QUERY_LATENCY = DummyMetric()
    CACHE_HITS = DummyMetric()
    CACHE_MISSES = DummyMetric()
    ERROR_COUNT = DummyMetric()
    ACTIVE_USERS = DummyMetric()
    WORD_COUNTS = DummyMetric()
    MEMORY_USAGE = DummyMetric()
    CPU_USAGE = DummyMetric()
    SLOW_QUERIES = DummyMetric()
    CACHE_SIZE = DummyMetric() 