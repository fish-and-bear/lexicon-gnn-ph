"""
Database utilities for managing connections and database operations.
Provides connection pooling, retry logic, monitoring, and advanced functionality.
"""

from sqlalchemy import create_engine, text, func
import os
from dotenv import load_dotenv
import logging
from sqlalchemy.orm import scoped_session, sessionmaker
import psycopg2
from psycopg2.extensions import (
    ISOLATION_LEVEL_AUTOCOMMIT,
    ISOLATION_LEVEL_READ_COMMITTED,
    ISOLATION_LEVEL_REPEATABLE_READ
)
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import DictCursor, execute_batch
import time
from contextlib import contextmanager
from typing import Optional, Generator, Dict, Any, List, Tuple
from prometheus_client import Counter, Histogram, Gauge, Summary
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import traceback
import structlog
from functools import wraps
import json
from datetime import datetime, timedelta

# Set up structured logging
logger = structlog.get_logger(__name__)

load_dotenv()

# Metrics
DB_ERRORS = Counter('database_errors_total', 'Total database errors', ['operation'])
DB_OPERATION_DURATION = Histogram('database_operation_duration_seconds', 'Database operation duration')
DB_CONNECTIONS = Counter('database_connections_total', 'Total database connections')
DB_CONNECTION_ERRORS = Counter('database_connection_errors_total', 'Total database connection errors')
DB_POOL_SIZE = Gauge('database_pool_size', 'Current database pool size')
DB_POOL_AVAILABLE = Gauge('database_pool_available', 'Available connections in pool')
DB_POOL_IN_USE = Gauge('database_pool_in_use', 'Connections currently in use')
DB_TRANSACTION_COUNT = Counter('database_transaction_total', 'Total database transactions')
DB_TRANSACTION_ERRORS = Counter('database_transaction_errors_total', 'Total transaction errors')
DB_TRANSACTION_DURATION = Histogram('database_transaction_duration_seconds', 'Transaction duration')
DB_QUERY_DURATION = Summary('database_query_duration_seconds', 'Query duration', ['query_type'])
DB_BATCH_OPERATIONS = Counter('database_batch_operations_total', 'Total batch operations', ['operation'])
DB_CACHE_HITS = Counter('database_cache_hits_total', 'Total cache hits')
DB_CACHE_MISSES = Counter('database_cache_misses_total', 'Total cache misses')

# Global connection pool
connection_pool = None

# Query cache
query_cache = {}
CACHE_TIMEOUT = 300  # 5 minutes

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass

class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass

class TransactionError(DatabaseError):
    """Raised when transaction operations fail."""
    pass

class QueryError(DatabaseError):
    """Raised when query execution fails."""
    pass

def get_base_config() -> Dict[str, Any]:
    """Get base database configuration without database name."""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', 5)),
        'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', 20)),
        'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 10)),
        'application_name': os.getenv('APP_NAME', 'dictionary_api'),
        'statement_timeout': int(os.getenv('DB_STATEMENT_TIMEOUT', 30000)),  # 30 seconds
        'idle_in_transaction_session_timeout': int(os.getenv('DB_IDLE_TIMEOUT', 60000)),  # 60 seconds
        'tcp_keepalives': 1,
        'tcp_keepalives_idle': 60,
        'tcp_keepalives_interval': 10,
        'tcp_keepalives_count': 3,
        'client_encoding': 'UTF8',
        'timezone': 'UTC',
        'application_type': os.getenv('APP_TYPE', 'dictionary'),
        'max_prepared_transactions': int(os.getenv('DB_MAX_PREPARED_TRANSACTIONS', 0)),
        'max_worker_processes': int(os.getenv('DB_MAX_WORKER_PROCESSES', 8)),
        'max_parallel_workers': int(os.getenv('DB_MAX_PARALLEL_WORKERS', 8)),
        'max_parallel_workers_per_gather': int(os.getenv('DB_MAX_PARALLEL_WORKERS_PER_GATHER', 2)),
        'lc_collate': 'en_US.UTF-8',  # For proper sorting of all languages
        'lc_ctype': 'en_US.UTF-8',    # For proper character handling
        'default_text_search_config': 'pg_catalog.simple',  # For text search across languages
        'standard_conforming_strings': 'on',
        'synchronous_commit': 'off',  # For better performance
        'effective_cache_size': '4GB',  # Adjust based on available memory
        'maintenance_work_mem': '256MB',
        'max_stack_depth': '7MB',
        'temp_file_limit': '1GB',
        'work_mem': '64MB',
        'random_page_cost': 1.1  # Assuming SSD storage
    }

def check_database_exists(dbname: str) -> bool:
    """Check if a database exists with enhanced error handling."""
    try:
        config = get_base_config()
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database='postgres',
            connect_timeout=config['connect_timeout']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cur.fetchone() is not None
            
        conn.close()
        return exists
    except Exception:
        return False

def get_db_config() -> Dict[str, Any]:
    """Get database configuration from environment variables with enhanced settings."""
    # List of possible database names in order of preference
    possible_db_names = ['fil_dict_db', 'fil_dict', 'dictionary']
    
    # Get the first existing database or default to fil_dict_db
    database_name = next(
        (name for name in possible_db_names if check_database_exists(name)),
        os.getenv('DB_NAME', 'fil_dict_db')
    )
    
    config = get_base_config()
    config['database'] = database_name
    return config

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def create_database_if_not_exists() -> bool:
    """Create the database if it doesn't exist, with retry logic."""
    config = get_db_config()
    
    try:
        # Connect to default database to create new database
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database='postgres',
            connect_timeout=config['connect_timeout'],
            application_name=f"{config['application_name']}_init"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database'],))
            if not cur.fetchone():
                # Create database with proper encoding and locale
                cur.execute(f"""
                    CREATE DATABASE {config['database']}
                    WITH 
                    OWNER = {config['user']}
                    ENCODING = 'UTF8'
                    LC_COLLATE = 'en_US.UTF-8'
                    LC_CTYPE = 'en_US.UTF-8'
                    TEMPLATE = template0
                    CONNECTION LIMIT = -1;
                    
                    -- Set database configuration
                    ALTER DATABASE {config['database']} SET timezone TO 'UTC';
                    ALTER DATABASE {config['database']} SET max_worker_processes TO {config['max_worker_processes']};
                    ALTER DATABASE {config['database']} SET max_parallel_workers TO {config['max_parallel_workers']};
                    ALTER DATABASE {config['database']} SET max_parallel_workers_per_gather TO {config['max_parallel_workers_per_gather']};
                    ALTER DATABASE {config['database']} SET standard_conforming_strings TO 'on';
                    ALTER DATABASE {config['database']} SET synchronous_commit TO 'off';
                    ALTER DATABASE {config['database']} SET effective_cache_size TO '4GB';
                    ALTER DATABASE {config['database']} SET maintenance_work_mem TO '256MB';
                    ALTER DATABASE {config['database']} SET max_stack_depth TO '7MB';
                    ALTER DATABASE {config['database']} SET temp_file_limit TO '1GB';
                    ALTER DATABASE {config['database']} SET work_mem TO '64MB';
                    ALTER DATABASE {config['database']} SET random_page_cost TO 1.1;
                    ALTER DATABASE {config['database']} SET default_text_search_config TO 'pg_catalog.simple';
                """)
                logger.info(f"Created database {config['database']}")
            return True
        
    except Exception as e:
        logger.error("Database creation failed", error=str(e), traceback=traceback.format_exc())
        DB_ERRORS.labels(operation='create_database').inc()
        raise ConnectionError(f"Failed to create database: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

def setup_extensions(conn, required_only=False) -> bool:
    """Set up required PostgreSQL extensions with enhanced functionality."""
    try:
        with conn.cursor() as cur:
            # Essential extensions that should always be available
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS pg_trgm;
                CREATE EXTENSION IF NOT EXISTS unaccent;
                CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
                CREATE EXTENSION IF NOT EXISTS btree_gin;
            """)
            
            if not required_only:
                try:
                    # Optional extensions that enhance functionality but aren't required
                    cur.execute("""
                        CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
                        CREATE EXTENSION IF NOT EXISTS hstore;
                        CREATE EXTENSION IF NOT EXISTS pgcrypto;
                        CREATE EXTENSION IF NOT EXISTS dict_xsyn;
                        CREATE EXTENSION IF NOT EXISTS tsm_system_rows;
                        CREATE EXTENSION IF NOT EXISTS pg_similarity;
                    """)
                except Exception as e:
                    logger.warning("Some optional extensions could not be installed",
                                error=str(e))
            
            # Create custom collations if supported
            try:
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_collation WHERE collname = 'case_insensitive'
                        ) THEN
                            CREATE COLLATION case_insensitive (
                                provider = icu,
                                locale = 'und-u-ks-level2',
                                deterministic = false
                            );
                        END IF;

                        IF NOT EXISTS (
                            SELECT 1 FROM pg_collation WHERE collname = 'ph_standard'
                        ) THEN
                            CREATE COLLATION ph_standard (
                                provider = icu,
                                locale = 'fil-PH-u-co-standard',
                                deterministic = false
                            );
                        END IF;
                    END
                    $$;
                """)
            except Exception as e:
                logger.warning("Custom collations could not be created",
                            error=str(e))
            
            # Create text search configurations
            try:
                cur.execute("""
                    CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS ph_default ( COPY = simple );
                    ALTER TEXT SEARCH CONFIGURATION ph_default
                        ALTER MAPPING FOR asciiword, word, numword, asciihword, hword, numhword
                        WITH unaccent, simple;
                """)
            except Exception as e:
                logger.warning("Text search configuration could not be created",
                            error=str(e))
            
            logger.info("Database extensions and language support setup completed")
            return True
        
    except Exception as e:
        logger.error("Failed to set up extensions",
                    error=str(e),
                    traceback=traceback.format_exc())
        if required_only:
            raise DatabaseError(f"Failed to set up required extensions: {str(e)}")
        else:
            logger.warning("Some extensions could not be installed",
                        error=str(e))
            return False

def init_connection_pool() -> None:
    """Initialize the database connection pool with enhanced monitoring."""
    global connection_pool
    config = get_db_config()
    
    if connection_pool is not None:
        return
    
    try:
        connection_pool = ThreadedConnectionPool(
            config['min_connections'],
            config['max_connections'],
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password'],
            connect_timeout=config['connect_timeout'],
            application_name=config['application_name'],
            options=f'-c statement_timeout={config["statement_timeout"]} -c idle_in_transaction_session_timeout={config["idle_in_transaction_session_timeout"]}',
            keepalives=config['tcp_keepalives'],
            keepalives_idle=config['tcp_keepalives_idle'],
            keepalives_interval=config['tcp_keepalives_interval'],
            keepalives_count=config['tcp_keepalives_count'],
            client_encoding=config['client_encoding']
        )
        DB_POOL_SIZE.set(config['max_connections'])
        DB_POOL_AVAILABLE.set(config['max_connections'])
        DB_POOL_IN_USE.set(0)
        logger.info("Database connection pool initialized", 
                   min_connections=config['min_connections'],
                   max_connections=config['max_connections'])
    except Exception as e:
        logger.error("Failed to initialize connection pool", 
                    error=str(e),
                    traceback=traceback.format_exc())
        DB_ERRORS.labels(operation='init_pool').inc()
        raise ConnectionError(f"Failed to initialize database connection pool: {str(e)}")

@contextmanager
def get_db_connection(isolation_level=ISOLATION_LEVEL_READ_COMMITTED) -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection from the pool with configurable isolation level."""
    conn = None
    start_time = time.time()
    
    try:
        conn = connection_pool.getconn()
        conn.set_session(isolation_level=isolation_level)
        DB_CONNECTIONS.inc()
        current_available = DB_POOL_AVAILABLE._value.get()
        current_in_use = DB_POOL_IN_USE._value.get()
        DB_POOL_AVAILABLE.set(max(0, current_available - 1))
        DB_POOL_IN_USE.set(current_in_use + 1)
        
        # Ensure connection is valid
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        
        yield conn
        
    except Exception as e:
        DB_CONNECTION_ERRORS.inc()
        logger.error("Database connection error", 
                    error=str(e),
                    traceback=traceback.format_exc())
        raise ConnectionError(f"Failed to get database connection: {str(e)}")
    finally:
        if conn is not None:
            try:
                if conn.status in (psycopg2.extensions.STATUS_IN_TRANSACTION,
                                 psycopg2.extensions.STATUS_PREPARED):
                    conn.rollback()
                connection_pool.putconn(conn)
                current_available = DB_POOL_AVAILABLE._value.get()
                current_in_use = DB_POOL_IN_USE._value.get()
                DB_POOL_AVAILABLE.set(current_available + 1)
                DB_POOL_IN_USE.set(max(0, current_in_use - 1))
            except Exception as e:
                logger.error("Error returning connection to pool",
                           error=str(e),
                           traceback=traceback.format_exc())
        DB_OPERATION_DURATION.observe(time.time() - start_time)

@contextmanager
def transaction(isolation_level=ISOLATION_LEVEL_READ_COMMITTED):
    """Enhanced transaction context manager with configurable isolation level."""
    with get_db_connection(isolation_level) as conn:
        start_time = time.time()
        DB_TRANSACTION_COUNT.inc()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            DB_TRANSACTION_ERRORS.inc()
            logger.error("Transaction error",
                        error=str(e),
                        traceback=traceback.format_exc())
            raise TransactionError(f"Transaction failed: {str(e)}")
        finally:
            DB_TRANSACTION_DURATION.observe(time.time() - start_time)

def with_transaction(isolation_level=ISOLATION_LEVEL_READ_COMMITTED):
    """Enhanced decorator to wrap function in a transaction."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with transaction(isolation_level) as cur:
                return func(cur, *args, **kwargs)
        return wrapper
    return decorator

def execute_batch_operation(operation: str, data: List[Tuple], page_size: int = 1000) -> int:
    """Execute a batch operation with progress tracking."""
    DB_BATCH_OPERATIONS.labels(operation=operation).inc()
    start_time = time.time()
    
    try:
        with transaction() as cur:
            execute_batch(cur, operation, data, page_size=page_size)
            row_count = cur.rowcount
            logger.info(f"Batch operation completed",
                       operation=operation,
                       rows=row_count,
                       duration=time.time() - start_time)
            return row_count
    except Exception as e:
        logger.error("Batch operation failed",
                    operation=operation,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise

def cached_query(timeout: int = CACHE_TIMEOUT):
    """Decorator for caching query results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached_result = query_cache.get(key)
            if cached_result:
                cache_time, result = cached_result
                if datetime.now() - cache_time < timedelta(seconds=timeout):
                    DB_CACHE_HITS.inc()
                    return result
            
            # Execute query
            DB_CACHE_MISSES.inc()
            result = func(*args, **kwargs)
            
            # Cache result
            query_cache[key] = (datetime.now(), result)
            return result
        return wrapper
    return decorator

def clear_query_cache():
    """Clear the query cache."""
    query_cache.clear()

def execute_query(query: str, params: Optional[Dict] = None, fetch: bool = True) -> Any:
    """Execute a query with timing and monitoring."""
    start_time = time.time()
    query_type = query.split()[0].lower()
    
    try:
        with transaction() as cur:
            cur.execute(query, params or {})
            result = cur.fetchall() if fetch else cur.rowcount
            DB_QUERY_DURATION.labels(query_type=query_type).observe(time.time() - start_time)
            return result
    except Exception as e:
        logger.error("Query execution failed",
                    query=query,
                    params=params,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise QueryError(f"Query execution failed: {str(e)}")

def get_table_stats() -> Dict[str, Any]:
    """Get statistics for all tables."""
    query = """
    SELECT 
        schemaname,
        relname as table_name,
        n_live_tup as row_count,
        n_dead_tup as dead_rows,
        n_mod_since_analyze as modifications,
        last_vacuum,
        last_autovacuum,
        last_analyze,
        last_autoanalyze
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    """
    return execute_query(query)

def get_index_stats() -> Dict[str, Any]:
    """Get statistics for all indexes."""
    query = """
    SELECT 
        schemaname,
        relname as table_name,
        indexrelname as index_name,
        idx_scan as scans,
        idx_tup_read as tuples_read,
        idx_tup_fetch as tuples_fetched
    FROM pg_stat_user_indexes
    WHERE schemaname = 'public'
    """
    return execute_query(query)

def get_query_stats() -> Dict[str, Any]:
    """Get statistics for queries."""
    query = """
    SELECT 
        query,
        calls,
        total_time,
        min_time,
        max_time,
        mean_time,
        rows
    FROM pg_stat_statements
    ORDER BY total_time DESC
    LIMIT 100
    """
    return execute_query(query)

def vacuum_analyze(table_name: Optional[str] = None):
    """Perform VACUUM ANALYZE on specified table or all tables."""
    with get_db_connection(ISOLATION_LEVEL_AUTOCOMMIT) as conn:
        with conn.cursor() as cur:
            if table_name:
                cur.execute(f"VACUUM ANALYZE {table_name}")
                logger.info(f"VACUUM ANALYZE completed", table=table_name)
            else:
                cur.execute("VACUUM ANALYZE")
                logger.info("VACUUM ANALYZE completed on all tables")

def check_db_health() -> Dict[str, Any]:
    """Enhanced database health check with detailed metrics."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Check basic connectivity
                cur.execute("SELECT 1")
                
                # Check pool status
                pool_info = {
                    "max_connections": connection_pool.maxconn,
                    "used_connections": len(connection_pool._used),
                    "available_connections": connection_pool.maxconn - len(connection_pool._used)
                }
                
                # Check PostgreSQL statistics
                cur.execute("""
                    SELECT * FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                stats = dict(zip([col[0] for col in cur.description], cur.fetchone()))
                
                # Check table statistics
                cur.execute("""
                    SELECT 
                        schemaname,
                        relname,
                        n_live_tup,
                        n_dead_tup,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                """)
                table_stats = [dict(zip([col[0] for col in cur.description], row))
                             for row in cur.fetchall()]
                
                # Check index health
                cur.execute("""
                    SELECT 
                        schemaname,
                        relname as table_name,
                        indexrelname as index_name,
                        idx_scan as scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    WHERE schemaname = 'public'
                """)
                index_stats = [dict(zip([col[0] for col in cur.description], row))
                             for row in cur.fetchall()]
                
                return {
                    "status": "healthy",
                    "pool": pool_info,
                    "statistics": {
                        "commits": stats.get("xact_commit", 0),
                        "rollbacks": stats.get("xact_rollback", 0),
                        "blocks_read": stats.get("blks_read", 0),
                        "blocks_hit": stats.get("blks_hit", 0),
                        "rows_returned": stats.get("tup_returned", 0),
                        "rows_fetched": stats.get("tup_fetched", 0),
                        "rows_inserted": stats.get("tup_inserted", 0),
                        "rows_updated": stats.get("tup_updated", 0),
                        "rows_deleted": stats.get("tup_deleted", 0)
                    },
                    "tables": table_stats,
                    "indexes": index_stats,
                    "vacuum_status": {
                        "running": bool(stats.get("vacuum_count", 0)),
                        "analyze_count": stats.get("analyze_count", 0),
                        "autovacuum_count": stats.get("autovacuum_count", 0),
                        "autoanalyze_count": stats.get("autoanalyze_count", 0)
                    }
                }
    except Exception as e:
        logger.error("Database health check failed",
                    error=str(e),
                    traceback=traceback.format_exc())
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def init_db(required_extensions_only=False) -> None:
    """Initialize the database and connection pool with enhanced setup."""
    try:
        create_database_if_not_exists()
        init_connection_pool()
        with get_db_connection() as conn:
            setup_extensions(conn, required_only=required_extensions_only)
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error("Database initialization failed",
                    error=str(e),
                    traceback=traceback.format_exc())
        raise

def close_db() -> None:
    """Close all database connections and clean up resources."""
    global connection_pool
    if connection_pool is not None:
        connection_pool.closeall()
        connection_pool = None
        clear_query_cache()
        logger.info("Database connections closed and resources cleaned up")

# Initialize database on module import
try:
    init_db()
except Exception as e:
    logger.error(f"Failed to initialize database on module import: {str(e)}")