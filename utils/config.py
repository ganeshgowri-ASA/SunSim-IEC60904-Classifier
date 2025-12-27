"""
Configuration management for SunSim-IEC60904-Classifier.

This module handles all configuration without triggering database connections
at import time. All database-related configuration is deferred until needed.
"""

import os
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Get application configuration from environment variables.
    Cached for performance - called once per application lifecycle.
    """
    return {
        # Application settings
        "app_name": os.environ.get("APP_NAME", "SunSim-IEC60904-Classifier"),
        "app_version": os.environ.get("APP_VERSION", "1.0.0"),
        "debug": os.environ.get("DEBUG", "false").lower() == "true",

        # Railway-specific settings
        "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT"),
        "railway_service_name": os.environ.get("RAILWAY_SERVICE_NAME"),
        "port": int(os.environ.get("PORT", 8501)),

        # Feature flags
        "enable_database": os.environ.get("ENABLE_DATABASE", "true").lower() == "true",
        "enable_reports": os.environ.get("ENABLE_REPORTS", "true").lower() == "true",
        "enable_spc": os.environ.get("ENABLE_SPC", "true").lower() == "true",

        # IEC 60904-9 standard configuration
        "iec_standard_version": os.environ.get("IEC_STANDARD_VERSION", "Ed.3"),
    }


def is_production() -> bool:
    """Check if running in production environment (Railway)."""
    config = get_config()
    return config["railway_environment"] is not None


def is_railway() -> bool:
    """Check if running on Railway platform."""
    return os.environ.get("RAILWAY_ENVIRONMENT") is not None


def get_app_name() -> str:
    """Get the application name."""
    return get_config()["app_name"]


def get_port() -> int:
    """Get the configured port number."""
    return get_config()["port"]


def database_features_enabled() -> bool:
    """
    Check if database features should be enabled.
    This does NOT check if database is actually connected.
    """
    return get_config()["enable_database"]


def get_railway_info() -> Optional[dict]:
    """
    Get Railway deployment information if available.
    Returns None if not running on Railway.
    """
    if not is_railway():
        return None

    return {
        "environment": os.environ.get("RAILWAY_ENVIRONMENT"),
        "service_name": os.environ.get("RAILWAY_SERVICE_NAME"),
        "replica_id": os.environ.get("RAILWAY_REPLICA_ID"),
        "deployment_id": os.environ.get("RAILWAY_DEPLOYMENT_ID"),
        "project_id": os.environ.get("RAILWAY_PROJECT_ID"),
    }
