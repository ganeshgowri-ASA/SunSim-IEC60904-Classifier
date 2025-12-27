"""
Database connection module with lazy-loading pattern.

This module implements lazy-loaded database connections to prevent 502 errors
when database credentials are missing or invalid. Connections are only
established when explicitly requested, allowing pages without database
dependencies to function normally.

Key design principles:
1. NO connections initialized at import time
2. Lazy initialization on first access
3. Graceful degradation when database is unavailable
4. Proper connection pooling and cleanup
"""

import os
import logging
from typing import Optional, Any
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection state - NOT initialized at import time
_connection_pool: Optional[Any] = None
_connection_lock = Lock()
_initialization_attempted = False
_last_error: Optional[str] = None


class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors."""
    pass


def _get_database_url() -> Optional[str]:
    """
    Get database URL from environment variables.
    Supports Railway's automatic DATABASE_URL and manual configuration.

    Railway provides DATABASE_URL automatically when a database is attached.
    For local development, users can set individual connection parameters.
    """
    # Check for Railway's automatic DATABASE_URL first
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Railway uses postgres:// but psycopg2 requires postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url

    # Check for individual connection parameters
    db_host = os.environ.get("DB_HOST") or os.environ.get("PGHOST")
    db_name = os.environ.get("DB_NAME") or os.environ.get("PGDATABASE")
    db_user = os.environ.get("DB_USER") or os.environ.get("PGUSER")
    db_password = os.environ.get("DB_PASSWORD") or os.environ.get("PGPASSWORD")
    db_port = os.environ.get("DB_PORT") or os.environ.get("PGPORT") or "5432"

    if all([db_host, db_name, db_user, db_password]):
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    return None


def is_database_configured() -> bool:
    """
    Check if database configuration is available.
    Does NOT attempt to connect - just checks for config presence.
    """
    return _get_database_url() is not None


def get_connection_status() -> dict:
    """
    Get the current database connection status.
    Useful for health checks and status pages.
    """
    global _connection_pool, _initialization_attempted, _last_error

    return {
        "configured": is_database_configured(),
        "connected": _connection_pool is not None,
        "initialization_attempted": _initialization_attempted,
        "last_error": _last_error,
    }


def _initialize_connection_pool() -> Any:
    """
    Initialize the database connection pool.
    This is called lazily on first connection request.
    """
    global _connection_pool, _initialization_attempted, _last_error

    try:
        import psycopg2
        from psycopg2 import pool
    except ImportError as e:
        _last_error = f"psycopg2 not installed: {e}"
        logger.error(_last_error)
        raise DatabaseConnectionError(_last_error)

    database_url = _get_database_url()
    if not database_url:
        _last_error = "Database not configured. Set DATABASE_URL or individual connection parameters."
        logger.warning(_last_error)
        raise DatabaseConnectionError(_last_error)

    try:
        # Create a connection pool with min 1, max 10 connections
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=database_url
        )
        _last_error = None
        logger.info("Database connection pool initialized successfully")
        return _connection_pool
    except psycopg2.Error as e:
        _last_error = f"Failed to connect to database: {e}"
        logger.error(_last_error)
        raise DatabaseConnectionError(_last_error)
    except Exception as e:
        _last_error = f"Unexpected error initializing database: {e}"
        logger.error(_last_error)
        raise DatabaseConnectionError(_last_error)


def _get_pool() -> Any:
    """
    Get the connection pool, initializing it lazily if needed.
    Thread-safe implementation.
    """
    global _connection_pool, _initialization_attempted

    if _connection_pool is not None:
        return _connection_pool

    with _connection_lock:
        # Double-check after acquiring lock
        if _connection_pool is not None:
            return _connection_pool

        _initialization_attempted = True
        return _initialize_connection_pool()


@contextmanager
def get_database_connection():
    """
    Context manager for database connections.
    Lazily initializes the connection pool on first use.

    Usage:
        try:
            with get_database_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM measurements")
                    results = cur.fetchall()
        except DatabaseConnectionError as e:
            st.error(f"Database unavailable: {e}")

    Raises:
        DatabaseConnectionError: If database is not configured or connection fails
    """
    pool = _get_pool()
    conn = None

    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            pool.putconn(conn)


def close_all_connections():
    """
    Close all database connections and clean up the pool.
    Should be called on application shutdown.
    """
    global _connection_pool, _initialization_attempted, _last_error

    with _connection_lock:
        if _connection_pool is not None:
            try:
                _connection_pool.closeall()
                logger.info("All database connections closed")
            except Exception as e:
                logger.error(f"Error closing connections: {e}")
            finally:
                _connection_pool = None
                _initialization_attempted = False
                _last_error = None


def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Optional[list]:
    """
    Execute a database query with automatic connection handling.

    Args:
        query: SQL query string
        params: Query parameters (optional)
        fetch: Whether to fetch and return results

    Returns:
        List of results if fetch=True, None otherwise

    Raises:
        DatabaseConnectionError: If database is not available
    """
    with get_database_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None


def execute_query_safe(query: str, params: tuple = None, fetch: bool = True, default=None):
    """
    Execute a query with graceful error handling.
    Returns default value if database is unavailable.

    This is useful for non-critical database operations where
    the application should continue even if the query fails.
    """
    try:
        return execute_query(query, params, fetch)
    except DatabaseConnectionError as e:
        logger.warning(f"Database unavailable for query: {e}")
        return default
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return default


def init_schema():
    """
    Initialize the database schema if tables don't exist.
    Called lazily when database features are first accessed.
    """
    schema = """
    -- Classification results table
    CREATE TABLE IF NOT EXISTS classification_results (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        simulator_id VARCHAR(100),
        spectral_class VARCHAR(1),
        uniformity_class VARCHAR(1),
        temporal_class VARCHAR(1),
        overall_class VARCHAR(3),
        notes TEXT
    );

    -- Spectral match measurements
    CREATE TABLE IF NOT EXISTS spectral_measurements (
        id SERIAL PRIMARY KEY,
        result_id INTEGER REFERENCES classification_results(id),
        wavelength_start REAL,
        wavelength_end REAL,
        measured_ratio REAL,
        reference_ratio REAL,
        deviation_percent REAL,
        pass_fail BOOLEAN
    );

    -- Uniformity measurements
    CREATE TABLE IF NOT EXISTS uniformity_measurements (
        id SERIAL PRIMARY KEY,
        result_id INTEGER REFERENCES classification_results(id),
        position_x REAL,
        position_y REAL,
        irradiance_value REAL,
        deviation_from_mean REAL
    );

    -- Temporal stability measurements
    CREATE TABLE IF NOT EXISTS temporal_measurements (
        id SERIAL PRIMARY KEY,
        result_id INTEGER REFERENCES classification_results(id),
        timestamp_offset_ms INTEGER,
        irradiance_value REAL,
        deviation_percent REAL
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_results_created ON classification_results(created_at);
    CREATE INDEX IF NOT EXISTS idx_results_class ON classification_results(overall_class);
    CREATE INDEX IF NOT EXISTS idx_spectral_result ON spectral_measurements(result_id);
    CREATE INDEX IF NOT EXISTS idx_uniformity_result ON uniformity_measurements(result_id);
    CREATE INDEX IF NOT EXISTS idx_temporal_result ON temporal_measurements(result_id);
    """

    try:
        with get_database_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema)
        logger.info("Database schema initialized successfully")
        return True
    except DatabaseConnectionError:
        logger.warning("Database not available - schema initialization skipped")
        return False
    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False
