"""
Database utilities for SunSim-IEC60904-Classifier.
Supports PostgreSQL (via DATABASE_URL or Streamlit secrets) for production
and SQLite for local development.

Database connections are lazy-loaded to avoid initialization errors during import.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Set up logging for database operations
logger = logging.getLogger(__name__)

# Flag to track if database has been initialized
_db_initialized = False


def get_database_url() -> str:
    """Get database URL from Streamlit secrets, environment, or use SQLite fallback."""
    database_url = None

    # Priority 1: Check Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            database_url = st.secrets["database"].get("DATABASE_URL")
    except Exception:
        pass

    # Priority 2: Check environment variable (for Railway or local)
    if not database_url:
        database_url = os.environ.get('DATABASE_URL')

    if database_url:
        # Railway/Heroku use postgres:// but SQLAlchemy needs postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        return database_url
    else:
        # Local development fallback to SQLite
        db_path = Path(__file__).parent.parent / "data" / "sunsim.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"


def get_engine():
    """
    Get SQLAlchemy engine with caching support.

    Uses @st.cache_resource when running in Streamlit context to cache
    the database engine and avoid creating new connections on every rerun.
    """
    try:
        import streamlit as st
        # Use cached version if in Streamlit context
        return _get_cached_engine()
    except Exception:
        # Fallback for non-Streamlit environments (e.g., testing)
        return _create_engine()


def _create_engine():
    """Create a new SQLAlchemy engine (internal use)."""
    database_url = get_database_url()
    try:
        if database_url.startswith('sqlite'):
            return create_engine(database_url, connect_args={"check_same_thread": False})
        else:
            return create_engine(database_url, poolclass=NullPool)
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def _get_cached_engine():
    """Get cached SQLAlchemy engine using Streamlit's cache_resource."""
    import streamlit as st

    @st.cache_resource
    def _cached_engine():
        return _create_engine()

    return _cached_engine()


def init_database() -> bool:
    """
    Initialize database with required tables.

    Returns:
        bool: True if initialization succeeded, False otherwise.

    This function is safe to call multiple times - it will only
    create tables if they don't already exist.
    """
    global _db_initialized

    if _db_initialized:
        return True

    try:
        engine = get_engine()
        database_url = get_database_url()
        is_postgres = not database_url.startswith('sqlite')
    except Exception as e:
        logger.error(f"Failed to get database engine during initialization: {e}")
        return False

    try:
        with engine.connect() as conn:
            # SPC Data table for control chart measurements
            if is_postgres:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spc_data (
                        id SERIAL PRIMARY KEY,
                        simulator_id TEXT NOT NULL,
                        sample_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        parameter_name TEXT NOT NULL,
                        measured_value REAL NOT NULL,
                        ucl REAL,
                        lcl REAL,
                        cl REAL,
                        subgroup_number INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spc_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        simulator_id TEXT NOT NULL,
                        sample_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        parameter_name TEXT NOT NULL,
                        measured_value REAL NOT NULL,
                        ucl REAL,
                        lcl REAL,
                        cl REAL,
                        subgroup_number INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

            # MSA Studies table for Gage R&R data
            if is_postgres:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS msa_studies (
                        id SERIAL PRIMARY KEY,
                        simulator_id TEXT NOT NULL,
                        study_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        study_name TEXT,
                        operator TEXT NOT NULL,
                        part_id TEXT NOT NULL,
                        trial INTEGER NOT NULL,
                        measured_value REAL NOT NULL,
                        grr_pct REAL,
                        repeatability_pct REAL,
                        reproducibility_pct REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS msa_studies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        simulator_id TEXT NOT NULL,
                        study_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        study_name TEXT,
                        operator TEXT NOT NULL,
                        part_id TEXT NOT NULL,
                        trial INTEGER NOT NULL,
                        measured_value REAL NOT NULL,
                        grr_pct REAL,
                        repeatability_pct REAL,
                        reproducibility_pct REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

            # Simulators table for reference
            if is_postgres:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS simulators (
                        id SERIAL PRIMARY KEY,
                        simulator_id TEXT UNIQUE NOT NULL,
                        name TEXT,
                        location TEXT,
                        classification TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS simulators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        simulator_id TEXT UNIQUE NOT NULL,
                        name TEXT,
                        location TEXT,
                        classification TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

            # Spectral match results table
            if is_postgres:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spectral_match_results (
                        id SERIAL PRIMARY KEY,
                        test_id TEXT NOT NULL,
                        simulator_id TEXT NOT NULL,
                        test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        method TEXT NOT NULL,
                        wavelength_range TEXT,
                        grade TEXT NOT NULL,
                        min_ratio REAL,
                        max_ratio REAL,
                        mean_ratio REAL,
                        spectral_mismatch_factor REAL,
                        weighted_deviation_pct REAL,
                        intervals_data TEXT,
                        operator TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spectral_match_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        test_id TEXT NOT NULL,
                        simulator_id TEXT NOT NULL,
                        test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        method TEXT NOT NULL,
                        wavelength_range TEXT,
                        grade TEXT NOT NULL,
                        min_ratio REAL,
                        max_ratio REAL,
                        mean_ratio REAL,
                        spectral_mismatch_factor REAL,
                        weighted_deviation_pct REAL,
                        intervals_data TEXT,
                        operator TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

            # Capability history table
            if is_postgres:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS capability_history (
                        id SERIAL PRIMARY KEY,
                        simulator_id TEXT NOT NULL,
                        parameter_name TEXT NOT NULL,
                        sample_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cp REAL,
                        cpk REAL,
                        pp REAL,
                        ppk REAL,
                        usl REAL,
                        lsl REAL,
                        target REAL,
                        mean_value REAL,
                        std_dev REAL,
                        sample_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            else:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS capability_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        simulator_id TEXT NOT NULL,
                        parameter_name TEXT NOT NULL,
                        sample_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        cp REAL,
                        cpk REAL,
                        pp REAL,
                        ppk REAL,
                        usl REAL,
                        lsl REAL,
                        target REAL,
                        mean_value REAL,
                        std_dev REAL,
                        sample_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))

            conn.commit()

        _db_initialized = True
        logger.info("Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


# SPC Data Functions
def insert_spc_data(simulator_id: str, parameter_name: str, measured_value: float,
                    subgroup_number: int, ucl: Optional[float] = None,
                    lcl: Optional[float] = None, cl: Optional[float] = None,
                    sample_date: Optional[datetime] = None) -> bool:
    """
    Insert a single SPC measurement.

    Returns:
        bool: True if insert succeeded, False otherwise.
    """
    if not ensure_database_ready():
        logger.error("Database not ready, cannot insert SPC data")
        return False

    try:
        engine = get_engine()

        if sample_date is None:
            sample_date = datetime.now()

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO spc_data (simulator_id, sample_date, parameter_name,
                                     measured_value, ucl, lcl, cl, subgroup_number)
                VALUES (:simulator_id, :sample_date, :parameter_name, :measured_value,
                        :ucl, :lcl, :cl, :subgroup_number)
            """), {
                "simulator_id": simulator_id,
                "sample_date": sample_date,
                "parameter_name": parameter_name,
                "measured_value": measured_value,
                "ucl": ucl,
                "lcl": lcl,
                "cl": cl,
                "subgroup_number": subgroup_number
            })
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to insert SPC data: {e}")
        return False


def insert_spc_batch(df: pd.DataFrame, simulator_id: str, parameter_name: str):
    """Insert batch of SPC measurements from DataFrame."""
    engine = get_engine()

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO spc_data (simulator_id, sample_date, parameter_name,
                                     measured_value, ucl, lcl, cl, subgroup_number)
                VALUES (:simulator_id, :sample_date, :parameter_name, :measured_value,
                        :ucl, :lcl, :cl, :subgroup_number)
            """), {
                "simulator_id": simulator_id,
                "sample_date": row.get('sample_date', datetime.now()),
                "parameter_name": parameter_name,
                "measured_value": row['measured_value'],
                "ucl": row.get('ucl'),
                "lcl": row.get('lcl'),
                "cl": row.get('cl'),
                "subgroup_number": row.get('subgroup_number', 1)
            })
        conn.commit()


def get_spc_data(simulator_id: Optional[str] = None,
                 parameter_name: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Retrieve SPC data with optional filters."""
    if not ensure_database_ready():
        logger.warning("Database not ready, returning empty DataFrame")
        return pd.DataFrame()

    try:
        engine = get_engine()

        query = "SELECT * FROM spc_data WHERE 1=1"
        params = {}

        if simulator_id:
            query += " AND simulator_id = :simulator_id"
            params["simulator_id"] = simulator_id
        if parameter_name:
            query += " AND parameter_name = :parameter_name"
            params["parameter_name"] = parameter_name
        if start_date:
            query += " AND sample_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND sample_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY sample_date, subgroup_number"

        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve SPC data: {e}")
        return pd.DataFrame()


def get_spc_parameters() -> list:
    """Get list of unique parameter names in SPC data."""
    if not ensure_database_ready():
        return []

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT parameter_name FROM spc_data ORDER BY parameter_name"))
            params = [row[0] for row in result.fetchall()]
        return params
    except Exception as e:
        logger.error(f"Failed to get SPC parameters: {e}")
        return []


def get_simulator_ids() -> list:
    """Get list of unique simulator IDs."""
    if not ensure_database_ready():
        return ["SIM-001", "SIM-002", "SIM-003"]

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT simulator_id FROM spc_data
                UNION
                SELECT DISTINCT simulator_id FROM msa_studies
                ORDER BY simulator_id
            """))
            ids = [row[0] for row in result.fetchall()]
        return ids if ids else ["SIM-001", "SIM-002", "SIM-003"]
    except Exception as e:
        logger.error(f"Failed to get simulator IDs: {e}")
        return ["SIM-001", "SIM-002", "SIM-003"]


# MSA Data Functions
def insert_msa_measurement(simulator_id: str, operator: str, part_id: str,
                           trial: int, measured_value: float,
                           study_name: Optional[str] = None,
                           study_date: Optional[datetime] = None):
    """Insert a single MSA measurement."""
    engine = get_engine()

    if study_date is None:
        study_date = datetime.now()

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO msa_studies (simulator_id, study_date, study_name, operator,
                                    part_id, trial, measured_value)
            VALUES (:simulator_id, :study_date, :study_name, :operator,
                    :part_id, :trial, :measured_value)
        """), {
            "simulator_id": simulator_id,
            "study_date": study_date,
            "study_name": study_name,
            "operator": operator,
            "part_id": part_id,
            "trial": trial,
            "measured_value": measured_value
        })
        conn.commit()


def insert_msa_batch(df: pd.DataFrame, simulator_id: str, study_name: str):
    """Insert batch of MSA measurements from DataFrame."""
    engine = get_engine()
    study_date = datetime.now()

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO msa_studies (simulator_id, study_date, study_name, operator,
                                        part_id, trial, measured_value)
                VALUES (:simulator_id, :study_date, :study_name, :operator,
                        :part_id, :trial, :measured_value)
            """), {
                "simulator_id": simulator_id,
                "study_date": row.get('study_date', study_date),
                "study_name": study_name,
                "operator": row['operator'],
                "part_id": row['part_id'],
                "trial": row['trial'],
                "measured_value": row['measured_value']
            })
        conn.commit()


def update_msa_results(simulator_id: str, study_name: str,
                       grr_pct: float, repeatability_pct: float,
                       reproducibility_pct: float):
    """Update MSA study with calculated results."""
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE msa_studies
            SET grr_pct = :grr_pct, repeatability_pct = :repeatability_pct,
                reproducibility_pct = :reproducibility_pct
            WHERE simulator_id = :simulator_id AND study_name = :study_name
        """), {
            "grr_pct": grr_pct,
            "repeatability_pct": repeatability_pct,
            "reproducibility_pct": reproducibility_pct,
            "simulator_id": simulator_id,
            "study_name": study_name
        })
        conn.commit()


def get_msa_data(simulator_id: Optional[str] = None,
                 study_name: Optional[str] = None) -> pd.DataFrame:
    """Retrieve MSA study data with optional filters."""
    engine = get_engine()

    query = "SELECT * FROM msa_studies WHERE 1=1"
    params = {}

    if simulator_id:
        query += " AND simulator_id = :simulator_id"
        params["simulator_id"] = simulator_id
    if study_name:
        query += " AND study_name = :study_name"
        params["study_name"] = study_name

    query += " ORDER BY study_date, operator, part_id, trial"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)
    return df


def get_msa_studies() -> list:
    """Get list of unique MSA study names."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT DISTINCT study_name FROM msa_studies WHERE study_name IS NOT NULL ORDER BY study_name"
        ))
        studies = [row[0] for row in result.fetchall()]
    return studies


# Capability History Functions
def insert_capability_record(simulator_id: str, parameter_name: str,
                            cp: float, cpk: float, pp: float, ppk: float,
                            usl: float, lsl: float, target: float,
                            mean_value: float, std_dev: float, sample_size: int):
    """Insert capability analysis record."""
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO capability_history (simulator_id, parameter_name, cp, cpk,
                                           pp, ppk, usl, lsl, target, mean_value,
                                           std_dev, sample_size)
            VALUES (:simulator_id, :parameter_name, :cp, :cpk, :pp, :ppk, :usl,
                    :lsl, :target, :mean_value, :std_dev, :sample_size)
        """), {
            "simulator_id": simulator_id,
            "parameter_name": parameter_name,
            "cp": cp,
            "cpk": cpk,
            "pp": pp,
            "ppk": ppk,
            "usl": usl,
            "lsl": lsl,
            "target": target,
            "mean_value": mean_value,
            "std_dev": std_dev,
            "sample_size": sample_size
        })
        conn.commit()


def get_capability_history(simulator_id: Optional[str] = None,
                           parameter_name: Optional[str] = None) -> pd.DataFrame:
    """Retrieve capability history data."""
    engine = get_engine()

    query = "SELECT * FROM capability_history WHERE 1=1"
    params = {}

    if simulator_id:
        query += " AND simulator_id = :simulator_id"
        params["simulator_id"] = simulator_id
    if parameter_name:
        query += " AND parameter_name = :parameter_name"
        params["parameter_name"] = parameter_name

    query += " ORDER BY sample_date"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)
    return df


def clear_spc_data(simulator_id: Optional[str] = None, parameter_name: Optional[str] = None):
    """Clear SPC data with optional filters."""
    engine = get_engine()

    query = "DELETE FROM spc_data WHERE 1=1"
    params = {}

    if simulator_id:
        query += " AND simulator_id = :simulator_id"
        params["simulator_id"] = simulator_id
    if parameter_name:
        query += " AND parameter_name = :parameter_name"
        params["parameter_name"] = parameter_name

    with engine.connect() as conn:
        conn.execute(text(query), params)
        conn.commit()


def clear_msa_data(simulator_id: Optional[str] = None, study_name: Optional[str] = None):
    """Clear MSA data with optional filters."""
    engine = get_engine()

    query = "DELETE FROM msa_studies WHERE 1=1"
    params = {}

    if simulator_id:
        query += " AND simulator_id = :simulator_id"
        params["simulator_id"] = simulator_id
    if study_name:
        query += " AND study_name = :study_name"
        params["study_name"] = study_name

    with engine.connect() as conn:
        conn.execute(text(query), params)
        conn.commit()


def ensure_database_ready() -> bool:
    """
    Ensure database is initialized before performing operations.

    Call this function before any database operation to ensure tables exist.
    This is a lazy initialization pattern - the database is only initialized
    when first needed, not at import time.

    Returns:
        bool: True if database is ready, False if initialization failed.
    """
    return init_database()


# Spectral Match Results Functions
def insert_spectral_match_result(
    test_id: str,
    simulator_id: str,
    method: str,
    grade: str,
    min_ratio: float,
    max_ratio: float,
    mean_ratio: float,
    spectral_mismatch_factor: float,
    weighted_deviation_pct: float,
    intervals_data: str,
    wavelength_range: Optional[str] = None,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
    test_date: Optional[datetime] = None
) -> bool:
    """
    Insert a spectral match analysis result.

    Args:
        test_id: Unique identifier for this test
        simulator_id: Simulator equipment ID
        method: 'SPD' or 'SPC'
        grade: Classification grade (A+, A, B, C, Fail)
        min_ratio: Minimum spectral ratio
        max_ratio: Maximum spectral ratio
        mean_ratio: Mean spectral ratio
        spectral_mismatch_factor: SPD mismatch factor M
        weighted_deviation_pct: Weighted deviation percentage
        intervals_data: JSON string of interval data
        wavelength_range: e.g., "400-1100nm"
        operator: Test operator name
        notes: Additional notes
        test_date: Date/time of test

    Returns:
        bool: True if insert succeeded
    """
    if not ensure_database_ready():
        logger.error("Database not ready, cannot insert spectral match result")
        return False

    try:
        engine = get_engine()

        if test_date is None:
            test_date = datetime.now()

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO spectral_match_results
                (test_id, simulator_id, test_date, method, wavelength_range, grade,
                 min_ratio, max_ratio, mean_ratio, spectral_mismatch_factor,
                 weighted_deviation_pct, intervals_data, operator, notes)
                VALUES (:test_id, :simulator_id, :test_date, :method, :wavelength_range,
                        :grade, :min_ratio, :max_ratio, :mean_ratio, :spectral_mismatch_factor,
                        :weighted_deviation_pct, :intervals_data, :operator, :notes)
            """), {
                "test_id": test_id,
                "simulator_id": simulator_id,
                "test_date": test_date,
                "method": method,
                "wavelength_range": wavelength_range,
                "grade": grade,
                "min_ratio": min_ratio,
                "max_ratio": max_ratio,
                "mean_ratio": mean_ratio,
                "spectral_mismatch_factor": spectral_mismatch_factor,
                "weighted_deviation_pct": weighted_deviation_pct,
                "intervals_data": intervals_data,
                "operator": operator,
                "notes": notes
            })
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to insert spectral match result: {e}")
        return False


def get_spectral_match_results(
    simulator_id: Optional[str] = None,
    test_id: Optional[str] = None,
    method: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Retrieve spectral match results with optional filters."""
    if not ensure_database_ready():
        logger.warning("Database not ready, returning empty DataFrame")
        return pd.DataFrame()

    try:
        engine = get_engine()

        query = "SELECT * FROM spectral_match_results WHERE 1=1"
        params = {}

        if simulator_id:
            query += " AND simulator_id = :simulator_id"
            params["simulator_id"] = simulator_id
        if test_id:
            query += " AND test_id = :test_id"
            params["test_id"] = test_id
        if method:
            query += " AND method = :method"
            params["method"] = method
        if start_date:
            query += " AND test_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND test_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY test_date DESC"

        with engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=params)
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve spectral match results: {e}")
        return pd.DataFrame()


def get_spectral_match_test_ids() -> list:
    """Get list of unique test IDs from spectral match results."""
    if not ensure_database_ready():
        return []

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT DISTINCT test_id FROM spectral_match_results ORDER BY test_id DESC"
            ))
            test_ids = [row[0] for row in result.fetchall()]
        return test_ids
    except Exception as e:
        logger.error(f"Failed to get spectral match test IDs: {e}")
        return []


# NOTE: Database initialization is now lazy-loaded via ensure_database_ready()
# or by calling init_database() explicitly. This prevents connection errors
# during module import when the database may not be available yet.
