"""
Comprehensive monitoring system for the application.
Includes metrics, tracing, error tracking, and performance monitoring.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from datetime import datetime, UTC
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Summary
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from datadog import initialize, statsd

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
API_REQUESTS = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)
DB_QUERY_LATENCY = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation']
)
CACHE_HITS = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['type', 'code'])
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
WORD_COUNTS = Gauge('word_counts', 'Word counts', ['language'])
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

# Performance Metrics
SLOW_QUERIES = Summary(
    'slow_queries_seconds',
    'Slow query duration in seconds',
    ['query_type']
)
CACHE_SIZE = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'])
API_RESPONSE_SIZE = Histogram(
    'api_response_size_bytes',
    'API response size in bytes',
    ['endpoint']
)

class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self._metrics_lock = threading.Lock()
        self._last_collection = 0
        self._collection_interval = 60  # seconds

    def collect_system_metrics(self):
        """Collect system-level metrics."""
        import psutil
        
        try:
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            MEMORY_USAGE.set(memory_info.rss)
            
            # CPU metrics
            CPU_USAGE.set(process.cpu_percent())
            
            # File descriptor metrics
            if hasattr(process, 'num_fds'):
                Gauge('open_files', 'Number of open files').set(process.num_fds())
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            ERROR_COUNT.labels(type='metrics_collection', code='system').inc()

    def collect_application_metrics(self, app):
        """Collect application-specific metrics."""
        try:
            with app.app_context():
                # Database metrics
                from models import Word, Definition
                WORD_COUNTS.labels(language='tl').set(
                    Word.query.filter_by(language_code='tl').count()
                )
                WORD_COUNTS.labels(language='ceb').set(
                    Word.query.filter_by(language_code='ceb').count()
                )
                
                # Cache metrics
                from caching import get_cache_stats
                cache_stats = get_cache_stats()
                CACHE_SIZE.labels(cache_type='redis').set(
                    cache_stats['redis']['info']['used_memory']
                )
                
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            ERROR_COUNT.labels(type='metrics_collection', code='application').inc()

    def start_collection(self, app):
        """Start periodic metrics collection."""
        def collect_metrics():
            while True:
                try:
                    current_time = time.time()
                    if current_time - self._last_collection >= self._collection_interval:
                        with self._metrics_lock:
                            self.collect_system_metrics()
                            self.collect_application_metrics(app)
                            self._last_collection = current_time
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in metrics collection thread: {e}")
                    time.sleep(5)  # Back off on error

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

def init_monitoring(app, env: str = 'production'):
    """Initialize all monitoring systems."""
    config = app.config
    
    # Initialize Sentry for error tracking
    if config.get('SENTRY_DSN'):
        sentry_sdk.init(
            dsn=config['SENTRY_DSN'],
            environment=env,
            integrations=[
                FlaskIntegration(),
                RedisIntegration(),
                SqlalchemyIntegration(),
            ],
            traces_sample_rate=float(config.get('SENTRY_SAMPLE_RATE', 0.1)),
            profiles_sample_rate=float(config.get('SENTRY_PROFILE_RATE', 0.1)),
            send_default_pii=False,
            before_send=before_send_event,
            before_breadcrumb=before_breadcrumb
        )

    # Initialize OpenTelemetry for distributed tracing
    if config.get('JAEGER_HOST'):
        trace.set_tracer_provider(TracerProvider())
        jaeger_exporter = JaegerExporter(
            agent_host_name=config['JAEGER_HOST'],
            agent_port=int(config.get('JAEGER_PORT', 6831)),
        )
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        # Instrument Flask
        FlaskInstrumentor().instrument_app(app)
        RequestsInstrumentor().instrument()

    # Initialize DataDog for APM
    if config.get('DATADOG_API_KEY'):
        initialize(
            api_key=config['DATADOG_API_KEY'],
            app_key=config['DATADOG_APP_KEY'],
            hostname=config.get('DATADOG_HOSTNAME')
        )

    # Start metrics collection
    metrics_collector = MetricsCollector()
    metrics_collector.start_collection(app)

    return metrics_collector

def before_send_event(event: Dict, hint: Dict) -> Optional[Dict]:
    """Process events before sending to Sentry."""
    # Filter out specific errors
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        if isinstance(exc_value, (KeyError, ValueError)):
            # Don't send common programming errors
            return None
    
    # Sanitize sensitive data
    if 'request' in event and 'headers' in event['request']:
        headers = event['request']['headers']
        if 'Authorization' in headers:
            headers['Authorization'] = '[FILTERED]'
            
    return event

def before_breadcrumb(breadcrumb: Dict, hint: Dict) -> Optional[Dict]:
    """Process breadcrumbs before adding to Sentry."""
    # Filter out noise
    if breadcrumb.get('category') == 'http':
        # Don't track health check endpoints
        if breadcrumb.get('data', {}).get('url', '').endswith('/health'):
            return None
    return breadcrumb

def trace_function(name: str = None):
    """Decorator to add tracing to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            function_name = name or func.__name__
            with trace.get_tracer(__name__).start_as_current_span(function_name) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute('status', 'success')
                    return result
                except Exception as e:
                    span.set_attribute('status', 'error')
                    span.set_attribute('error.type', e.__class__.__name__)
                    span.set_attribute('error.message', str(e))
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute('duration', duration)
                    if duration > 1.0:  # Slow operation threshold
                        logger.warning(f"Slow operation detected: {function_name} took {duration:.2f}s")
                        SLOW_QUERIES.labels(query_type=function_name).observe(duration)
        return wrapper
    return decorator

def monitor_performance(endpoint_name: str = None):
    """Decorator to monitor endpoint performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = endpoint_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = getattr(result, 'status_code', 200)
                API_REQUESTS.labels(
                    method=request.method,
                    endpoint=name,
                    status=status
                ).inc()
                return result
            except Exception as e:
                status = getattr(e, 'code', 500)
                API_REQUESTS.labels(
                    method=request.method,
                    endpoint=name,
                    status=status
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_LATENCY.labels(endpoint=name).observe(duration)
                if hasattr(result, 'get_data'):
                    response_size = len(result.get_data())
                    API_RESPONSE_SIZE.labels(endpoint=name).observe(response_size)
        return wrapper
    return decorator 