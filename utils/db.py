"""
Database connection utilities with lazy loading for Streamlit.

This module provides lazy-loaded database connections using @st.cache_resource
to prevent 502 errors on Railway and other cloud platforms. Database connections
are only established when first needed, not at import time.

Features:
- Lazy-loaded PostgreSQL connections via @st.cache_resource
- Graceful fallback when database is unavailable
- Connection health checking and auto-reconnection
- Thread-safe connection pool management
"""

import os
import logging
from typing import Optional, Any, List, Dict, Tuple
from contextlib import contextmanager
from functools import wraps

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database availability flag - set after first connection attempt
_db_available: Optional[bool] = None


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class DatabaseUnavailable(DatabaseError):
    """Exception raised when database is not available."""
    pass


def get_database_url() -> Optional[str]:
    """
    Get database URL from environment variables.

    Checks for common PostgreSQL environment variable names used by
    Railway, Heroku, and other cloud platforms.

    Returns:
        Database URL string or None if not configured.
    """
    # Check common environment variable names
    env_vars = [
        'DATABASE_URL',
        'POSTGRES_URL',
        'POSTGRESQL_URL',
        'DATABASE_PRIVATE_URL',
        'POSTGRES_PRISMA_URL',
    ]

    for var in env_vars:
        url = os.environ.get(var)
        if url:
            # Handle Railway's postgres:// vs postgresql:// issue
            if url.startswith('postgres://'):
                url = url.replace('postgres://', 'postgresql://', 1)
            return url

    return None


@st.cache_resource(show_spinner=False)
def get_db_engine():
    """
    Create and cache a SQLAlchemy engine with lazy loading.

    This function is decorated with @st.cache_resource to ensure the engine
    is only created once and reused across reruns. The connection is lazy-loaded
    meaning it's only established when this function is first called, not at
    import time.

    Returns:
        SQLAlchemy Engine object or None if database is unavailable.
    """
    global _db_available

    database_url = get_database_url()

    if not database_url:
        logger.warning("No database URL configured. Running in offline mode.")
        _db_available = False
        return None

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool

        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections every 30 minutes
            pool_pre_ping=True,  # Enable connection health checks
            echo=False,
        )

        # Test the connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")

        logger.info("Database connection established successfully.")
        _db_available = True
        return engine

    except ImportError:
        logger.error("SQLAlchemy not installed. Install with: pip install sqlalchemy psycopg2-binary")
        _db_available = False
        return None

    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        _db_available = False
        return None


@st.cache_resource(show_spinner=False)
def get_psycopg2_connection():
    """
    Create and cache a psycopg2 connection with lazy loading.

    Alternative to SQLAlchemy for direct PostgreSQL access.
    Uses @st.cache_resource for connection caching.

    Returns:
        psycopg2 connection object or None if unavailable.
    """
    global _db_available

    database_url = get_database_url()

    if not database_url:
        logger.warning("No database URL configured.")
        _db_available = False
        return None

    try:
        import psycopg2
        from psycopg2 import pool

        # Create a connection pool
        connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=database_url,
        )

        # Test the connection
        conn = connection_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        connection_pool.putconn(conn)

        logger.info("PostgreSQL connection pool established.")
        _db_available = True
        return connection_pool

    except ImportError:
        logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
        _db_available = False
        return None

    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection: {e}")
        _db_available = False
        return None


def is_database_available() -> bool:
    """
    Check if database connection is available.

    This performs a lazy check - it will attempt to connect on first call
    if not already attempted.

    Returns:
        True if database is available, False otherwise.
    """
    global _db_available

    if _db_available is None:
        # Trigger lazy connection attempt
        get_db_engine()

    return _db_available or False


def require_database(func):
    """
    Decorator that requires database to be available.

    If database is unavailable, displays a warning and returns None
    instead of raising an exception.

    Usage:
        @require_database
        def get_user_data(user_id):
            # Database operations here
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_database_available():
            st.warning(
                "Database is currently unavailable. "
                "Some features may be limited.",
                icon="exclamation-triangle"
            )
            return None

        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database operation failed in {func.__name__}: {e}")
            st.error(f"Database operation failed: {str(e)}")
            return None

    return wrapper


@contextmanager
def get_db_session():
    """
    Context manager for database sessions with automatic cleanup.

    Provides a SQLAlchemy session that automatically commits on success
    and rolls back on failure.

    Usage:
        with get_db_session() as session:
            result = session.execute(query)

    Yields:
        SQLAlchemy Session object or None if unavailable.
    """
    engine = get_db_engine()

    if engine is None:
        yield None
        return

    try:
        from sqlalchemy.orm import sessionmaker
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    except ImportError:
        logger.error("SQLAlchemy not installed")
        yield None


def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    fetch: str = 'all'
) -> Optional[List[Tuple]]:
    """
    Execute a SQL query with graceful error handling.

    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        fetch: 'all', 'one', or 'none' for fetchall, fetchone, or no fetch

    Returns:
        Query results or None if database is unavailable/query fails.
    """
    engine = get_db_engine()

    if engine is None:
        logger.warning("Cannot execute query - database unavailable")
        return None

    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))

            if fetch == 'all':
                return result.fetchall()
            elif fetch == 'one':
                row = result.fetchone()
                return [row] if row else []
            else:
                conn.commit()
                return []

    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return None


def execute_query_safe(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    default: Any = None,
    show_warning: bool = True
) -> Any:
    """
    Execute a SQL query with graceful fallback to default value.

    This is the safest way to execute queries - it will never raise
    an exception and always returns the default value on any failure.

    Args:
        query: SQL query string
        params: Optional dictionary of query parameters
        default: Default value to return on failure
        show_warning: Whether to show a Streamlit warning on failure

    Returns:
        Query results or default value.
    """
    try:
        result = execute_query(query, params)
        if result is not None:
            return result
    except Exception as e:
        logger.error(f"Safe query execution failed: {e}")

    if show_warning:
        st.warning(
            "Could not load data from database. Using default values.",
            icon="exclamation-triangle"
        )

    return default


def check_connection_health() -> Dict[str, Any]:
    """
    Check the health of the database connection.

    Returns:
        Dictionary with connection health information.
    """
    health = {
        'configured': get_database_url() is not None,
        'connected': False,
        'latency_ms': None,
        'error': None,
    }

    if not health['configured']:
        health['error'] = 'No database URL configured'
        return health

    engine = get_db_engine()
    if engine is None:
        health['error'] = 'Failed to create database engine'
        return health

    try:
        import time
        from sqlalchemy import text

        start = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health['latency_ms'] = round((time.time() - start) * 1000, 2)
        health['connected'] = True

    except Exception as e:
        health['error'] = str(e)

    return health


def clear_connection_cache():
    """
    Clear cached database connections.

    Call this to force reconnection on next database access.
    Useful when database credentials have changed.
    """
    global _db_available
    _db_available = None

    # Clear Streamlit's cached resources
    get_db_engine.clear()
    get_psycopg2_connection.clear()

    logger.info("Database connection cache cleared")


def show_database_status():
    """
    Display database connection status in the Streamlit sidebar.

    Shows a status indicator with connection health information.
    """
    health = check_connection_health()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Database Status**")

        if health['connected']:
            st.success(f"Connected ({health['latency_ms']}ms)")
        elif health['configured']:
            st.error(f"Connection Failed: {health['error']}")
        else:
            st.warning("Not Configured (Offline Mode)")
