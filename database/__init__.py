# Database module with lazy-loaded connections
# This module NEVER initializes connections at import time to prevent 502 errors

from database.connection import (
    get_database_connection,
    get_connection_status,
    is_database_configured,
    close_all_connections,
    DatabaseConnectionError,
)

__all__ = [
    "get_database_connection",
    "get_connection_status",
    "is_database_configured",
    "close_all_connections",
    "DatabaseConnectionError",
]
