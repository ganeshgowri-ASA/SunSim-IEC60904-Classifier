"""
Database utilities for SunSim IEC60904 Classifier.

Uses lazy-loaded connections with @st.cache_resource to avoid
initialization at import time, preventing 502 errors on Railway.
"""

import os
from contextlib import contextmanager
from typing import Optional, Generator, Any

import streamlit as st

# Database availability flag - checked before operations
_db_available: Optional[bool] = None


def get_database_url() -> Optional[str]:
    """Get database URL from environment variables.

    Returns:
        Database URL string or None if not configured.
    """
    # Railway provides DATABASE_URL automatically
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


@st.cache_resource
def _get_connection_pool():
    """Create a lazy-loaded database connection pool.

    This function is decorated with @st.cache_resource to ensure:
    - Connection pool is only created when first accessed
    - Pool is reused across reruns
    - No initialization happens at module import time

    Returns:
        Connection pool or None if database is not available.
    """
    database_url = get_database_url()

    if not database_url:
        return None

    try:
        import psycopg2
        from psycopg2 import pool

        # Create a threaded connection pool
        connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=database_url
        )
        return connection_pool
    except ImportError:
        st.warning("psycopg2 not installed. Database features disabled.")
        return None
    except Exception as e:
        st.warning(f"Could not connect to database: {e}")
        return None


def is_database_available() -> bool:
    """Check if database is available without triggering errors.

    Returns:
        True if database connection can be established, False otherwise.
    """
    global _db_available

    if _db_available is not None:
        return _db_available

    pool = _get_connection_pool()
    _db_available = pool is not None
    return _db_available


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Get a database connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()

    Yields:
        Database connection or None if unavailable.

    Raises:
        DatabaseUnavailableError: If database is not configured.
    """
    pool = _get_connection_pool()

    if pool is None:
        raise DatabaseUnavailableError("Database connection not available")

    conn = None
    try:
        conn = pool.getconn()
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            pool.putconn(conn)


class DatabaseUnavailableError(Exception):
    """Raised when database operations are attempted without a connection."""
    pass


def init_database() -> bool:
    """Initialize database schema if needed.

    This function should only be called explicitly when database
    operations are needed, not at module import time.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    if not is_database_available():
        return False

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Create tables if they don't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS classifications (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        simulator_name VARCHAR(255),
                        spectral_class VARCHAR(10),
                        uniformity_class VARCHAR(10),
                        temporal_class VARCHAR(10),
                        overall_class VARCHAR(10),
                        metadata JSONB
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS measurements (
                        id SERIAL PRIMARY KEY,
                        classification_id INTEGER REFERENCES classifications(id),
                        measurement_type VARCHAR(50),
                        data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_classifications_created_at
                    ON classifications(created_at)
                """)

        return True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        return False


def save_classification(
    simulator_name: str,
    spectral_class: str,
    uniformity_class: str,
    temporal_class: str,
    overall_class: str,
    metadata: Optional[dict] = None
) -> Optional[int]:
    """Save a classification result to the database.

    Args:
        simulator_name: Name of the sun simulator
        spectral_class: Spectral match classification (A/B/C)
        uniformity_class: Uniformity classification (A/B/C)
        temporal_class: Temporal stability classification (A/B/C)
        overall_class: Overall classification (A/B/C)
        metadata: Optional additional metadata

    Returns:
        Classification ID if saved successfully, None otherwise.
    """
    if not is_database_available():
        return None

    try:
        import json
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO classifications
                    (simulator_name, spectral_class, uniformity_class,
                     temporal_class, overall_class, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    simulator_name,
                    spectral_class,
                    uniformity_class,
                    temporal_class,
                    overall_class,
                    json.dumps(metadata) if metadata else None
                ))
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        st.error(f"Failed to save classification: {e}")
        return None


def get_classifications(limit: int = 100) -> list:
    """Retrieve recent classifications from the database.

    Args:
        limit: Maximum number of records to retrieve.

    Returns:
        List of classification records or empty list if unavailable.
    """
    if not is_database_available():
        return []

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, created_at, simulator_name, spectral_class,
                           uniformity_class, temporal_class, overall_class, metadata
                    FROM classifications
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
                return cur.fetchall()
    except Exception as e:
        st.error(f"Failed to retrieve classifications: {e}")
        return []


def require_database(func):
    """Decorator to gracefully handle missing database.

    Use this decorator on functions that require database access.
    If database is unavailable, shows a warning and returns None.

    Usage:
        @require_database
        def my_db_function():
            with get_connection() as conn:
                ...
    """
    def wrapper(*args, **kwargs):
        if not is_database_available():
            st.warning(
                "Database not configured. Some features are disabled. "
                "Set DATABASE_URL environment variable to enable."
            )
            return None
        return func(*args, **kwargs)
    return wrapper
