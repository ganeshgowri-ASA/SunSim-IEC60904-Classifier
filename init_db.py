#!/usr/bin/env python3
"""
Database initialization script for SunSim-IEC60904-Classifier.
Run this script to create all required tables in PostgreSQL.

Usage:
    python init_db.py

Requires DATABASE_URL environment variable to be set for PostgreSQL.
Falls back to SQLite if not set.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def get_database_url() -> str:
    """Get database URL from environment or use SQLite fallback."""
    database_url = os.environ.get('DATABASE_URL')

    if database_url:
        # Railway uses postgres:// but SQLAlchemy needs postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        return database_url
    else:
        print("WARNING: DATABASE_URL not set, using SQLite fallback")
        return "sqlite:///data/sunsim.db"


def create_tables(engine, is_postgres: bool):
    """Create all required tables."""

    with engine.connect() as conn:
        # SPC Data table for control chart measurements
        print("Creating spc_data table...")
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
        print("Creating msa_studies table...")
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
        print("Creating simulators table...")
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

        # Capability history table
        print("Creating capability_history table...")
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

        # Create indexes for better query performance
        print("Creating indexes...")
        if is_postgres:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_spc_simulator_param
                ON spc_data(simulator_id, parameter_name)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_spc_sample_date
                ON spc_data(sample_date)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_msa_simulator_study
                ON msa_studies(simulator_id, study_name)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_capability_simulator_param
                ON capability_history(simulator_id, parameter_name)
            """))

        conn.commit()
        print("All tables created successfully!")


def verify_tables(engine):
    """Verify that all tables exist."""
    expected_tables = ['spc_data', 'msa_studies', 'simulators', 'capability_history']

    with engine.connect() as conn:
        # Check for PostgreSQL or SQLite
        try:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ))
            tables = [row[0] for row in result.fetchall()]
        except Exception:
            # SQLite fallback
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ))
            tables = [row[0] for row in result.fetchall()]

        print("\nVerifying tables:")
        all_exist = True
        for table in expected_tables:
            exists = table in tables
            status = "OK" if exists else "MISSING"
            print(f"  {table}: {status}")
            if not exists:
                all_exist = False

        return all_exist


def main():
    print("=" * 60)
    print("SunSim-IEC60904-Classifier Database Initialization")
    print("=" * 60)

    database_url = get_database_url()
    is_postgres = not database_url.startswith('sqlite')

    db_type = "PostgreSQL" if is_postgres else "SQLite"
    print(f"\nDatabase type: {db_type}")

    if is_postgres:
        # Mask password in URL for display
        display_url = database_url.split('@')[1] if '@' in database_url else database_url
        print(f"Host: {display_url.split('/')[0]}")
    else:
        print(f"Path: {database_url}")

    print("\nInitializing database...")

    try:
        if database_url.startswith('sqlite'):
            engine = create_engine(database_url, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(database_url, poolclass=NullPool)

        create_tables(engine, is_postgres)

        if verify_tables(engine):
            print("\n" + "=" * 60)
            print("Database initialization completed successfully!")
            print("=" * 60)
            return 0
        else:
            print("\nERROR: Some tables are missing!")
            return 1

    except Exception as e:
        print(f"\nERROR: Database initialization failed!")
        print(f"Details: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
