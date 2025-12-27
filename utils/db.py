"""
Database utilities for SunSim-IEC60904-Classifier.
Handles SQLite database operations for SPC and MSA data storage.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd


DB_PATH = Path(__file__).parent.parent / "data" / "sunsim.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection, creating directory if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database with required tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # SPC Data table for control chart measurements
    cursor.execute("""
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
    """)

    # MSA Studies table for Gage R&R data
    cursor.execute("""
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
    """)

    # Simulators table for reference
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS simulators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            simulator_id TEXT UNIQUE NOT NULL,
            name TEXT,
            location TEXT,
            classification TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Capability history table
    cursor.execute("""
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
    """)

    conn.commit()
    conn.close()


# SPC Data Functions
def insert_spc_data(simulator_id: str, parameter_name: str, measured_value: float,
                    subgroup_number: int, ucl: Optional[float] = None,
                    lcl: Optional[float] = None, cl: Optional[float] = None,
                    sample_date: Optional[datetime] = None):
    """Insert a single SPC measurement."""
    conn = get_connection()
    cursor = conn.cursor()

    if sample_date is None:
        sample_date = datetime.now()

    cursor.execute("""
        INSERT INTO spc_data (simulator_id, sample_date, parameter_name,
                             measured_value, ucl, lcl, cl, subgroup_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (simulator_id, sample_date, parameter_name, measured_value, ucl, lcl, cl, subgroup_number))

    conn.commit()
    conn.close()


def insert_spc_batch(df: pd.DataFrame, simulator_id: str, parameter_name: str):
    """Insert batch of SPC measurements from DataFrame."""
    conn = get_connection()

    for _, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO spc_data (simulator_id, sample_date, parameter_name,
                                 measured_value, ucl, lcl, cl, subgroup_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            simulator_id,
            row.get('sample_date', datetime.now()),
            parameter_name,
            row['measured_value'],
            row.get('ucl'),
            row.get('lcl'),
            row.get('cl'),
            row.get('subgroup_number', 1)
        ))

    conn.commit()
    conn.close()


def get_spc_data(simulator_id: Optional[str] = None,
                 parameter_name: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Retrieve SPC data with optional filters."""
    conn = get_connection()

    query = "SELECT * FROM spc_data WHERE 1=1"
    params = []

    if simulator_id:
        query += " AND simulator_id = ?"
        params.append(simulator_id)
    if parameter_name:
        query += " AND parameter_name = ?"
        params.append(parameter_name)
    if start_date:
        query += " AND sample_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND sample_date <= ?"
        params.append(end_date)

    query += " ORDER BY sample_date, subgroup_number"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_spc_parameters() -> list:
    """Get list of unique parameter names in SPC data."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT parameter_name FROM spc_data ORDER BY parameter_name")
    params = [row[0] for row in cursor.fetchall()]
    conn.close()
    return params


def get_simulator_ids() -> list:
    """Get list of unique simulator IDs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT simulator_id FROM spc_data
        UNION
        SELECT DISTINCT simulator_id FROM msa_studies
        ORDER BY simulator_id
    """)
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids if ids else ["SIM-001", "SIM-002", "SIM-003"]


# MSA Data Functions
def insert_msa_measurement(simulator_id: str, operator: str, part_id: str,
                           trial: int, measured_value: float,
                           study_name: Optional[str] = None,
                           study_date: Optional[datetime] = None):
    """Insert a single MSA measurement."""
    conn = get_connection()
    cursor = conn.cursor()

    if study_date is None:
        study_date = datetime.now()

    cursor.execute("""
        INSERT INTO msa_studies (simulator_id, study_date, study_name, operator,
                                part_id, trial, measured_value)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (simulator_id, study_date, study_name, operator, part_id, trial, measured_value))

    conn.commit()
    conn.close()


def insert_msa_batch(df: pd.DataFrame, simulator_id: str, study_name: str):
    """Insert batch of MSA measurements from DataFrame."""
    conn = get_connection()

    study_date = datetime.now()

    for _, row in df.iterrows():
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO msa_studies (simulator_id, study_date, study_name, operator,
                                    part_id, trial, measured_value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            simulator_id,
            row.get('study_date', study_date),
            study_name,
            row['operator'],
            row['part_id'],
            row['trial'],
            row['measured_value']
        ))

    conn.commit()
    conn.close()


def update_msa_results(simulator_id: str, study_name: str,
                       grr_pct: float, repeatability_pct: float,
                       reproducibility_pct: float):
    """Update MSA study with calculated results."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE msa_studies
        SET grr_pct = ?, repeatability_pct = ?, reproducibility_pct = ?
        WHERE simulator_id = ? AND study_name = ?
    """, (grr_pct, repeatability_pct, reproducibility_pct, simulator_id, study_name))

    conn.commit()
    conn.close()


def get_msa_data(simulator_id: Optional[str] = None,
                 study_name: Optional[str] = None) -> pd.DataFrame:
    """Retrieve MSA study data with optional filters."""
    conn = get_connection()

    query = "SELECT * FROM msa_studies WHERE 1=1"
    params = []

    if simulator_id:
        query += " AND simulator_id = ?"
        params.append(simulator_id)
    if study_name:
        query += " AND study_name = ?"
        params.append(study_name)

    query += " ORDER BY study_date, operator, part_id, trial"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_msa_studies() -> list:
    """Get list of unique MSA study names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT study_name FROM msa_studies WHERE study_name IS NOT NULL ORDER BY study_name")
    studies = [row[0] for row in cursor.fetchall()]
    conn.close()
    return studies


# Capability History Functions
def insert_capability_record(simulator_id: str, parameter_name: str,
                            cp: float, cpk: float, pp: float, ppk: float,
                            usl: float, lsl: float, target: float,
                            mean_value: float, std_dev: float, sample_size: int):
    """Insert capability analysis record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO capability_history (simulator_id, parameter_name, cp, cpk,
                                       pp, ppk, usl, lsl, target, mean_value,
                                       std_dev, sample_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (simulator_id, parameter_name, cp, cpk, pp, ppk, usl, lsl, target,
          mean_value, std_dev, sample_size))

    conn.commit()
    conn.close()


def get_capability_history(simulator_id: Optional[str] = None,
                           parameter_name: Optional[str] = None) -> pd.DataFrame:
    """Retrieve capability history data."""
    conn = get_connection()

    query = "SELECT * FROM capability_history WHERE 1=1"
    params = []

    if simulator_id:
        query += " AND simulator_id = ?"
        params.append(simulator_id)
    if parameter_name:
        query += " AND parameter_name = ?"
        params.append(parameter_name)

    query += " ORDER BY sample_date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def clear_spc_data(simulator_id: Optional[str] = None, parameter_name: Optional[str] = None):
    """Clear SPC data with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "DELETE FROM spc_data WHERE 1=1"
    params = []

    if simulator_id:
        query += " AND simulator_id = ?"
        params.append(simulator_id)
    if parameter_name:
        query += " AND parameter_name = ?"
        params.append(parameter_name)

    cursor.execute(query, params)
    conn.commit()
    conn.close()


def clear_msa_data(simulator_id: Optional[str] = None, study_name: Optional[str] = None):
    """Clear MSA data with optional filters."""
    conn = get_connection()
    cursor = conn.cursor()

    query = "DELETE FROM msa_studies WHERE 1=1"
    params = []

    if simulator_id:
        query += " AND simulator_id = ?"
        params.append(simulator_id)
    if study_name:
        query += " AND study_name = ?"
        params.append(study_name)

    cursor.execute(query, params)
    conn.commit()
    conn.close()


# Initialize database on import
init_database()
