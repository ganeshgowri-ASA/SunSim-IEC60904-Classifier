"""
Sun Simulator Classification System - Database Module
Railway PostgreSQL Connection and Table Definitions

This module provides database connectivity and ORM models for storing
sun simulator measurement data and classification results.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Text, DateTime,
    Boolean, ForeignKey, JSON, Enum as SQLEnum, UniqueConstraint,
    Index, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import QueuePool
import enum

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

Base = declarative_base()


def get_database_url() -> str:
    """
    Get database URL from environment variables.
    Supports Railway PostgreSQL connection format.

    Returns:
        PostgreSQL connection URL
    """
    # Try Railway's DATABASE_URL first
    url = os.getenv('DATABASE_URL')
    if url:
        # Railway uses postgres:// but SQLAlchemy 2.0 requires postgresql://
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        return url

    # Build from individual components (Railway format)
    host = os.getenv('PGHOST', os.getenv('DB_HOST', 'localhost'))
    port = os.getenv('PGPORT', os.getenv('DB_PORT', '5432'))
    database = os.getenv('PGDATABASE', os.getenv('DB_NAME', 'sunsim'))
    user = os.getenv('PGUSER', os.getenv('DB_USER', 'postgres'))
    password = os.getenv('PGPASSWORD', os.getenv('DB_PASSWORD', ''))

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


# =============================================================================
# ENUMS
# =============================================================================

class ClassificationGrade(enum.Enum):
    """IEC 60904-9 Classification Grades"""
    A_PLUS = 'A+'
    A = 'A'
    B = 'B'
    C = 'C'
    NA = 'N/A'


class MeasurementType(enum.Enum):
    """Types of measurements"""
    SPECTRAL = 'spectral'
    UNIFORMITY = 'uniformity'
    TEMPORAL_STI = 'temporal_sti'
    TEMPORAL_LTI = 'temporal_lti'


class LampType(enum.Enum):
    """Solar simulator lamp types"""
    XENON = 'xenon'
    LED = 'led'
    HALOGEN = 'halogen'
    MULTI_SOURCE = 'multi_source'
    OTHER = 'other'


# =============================================================================
# DATABASE MODELS
# =============================================================================

class Manufacturer(Base):
    """Manufacturer information for solar simulators."""
    __tablename__ = 'manufacturers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    country = Column(String(100))
    website = Column(String(255))
    contact_email = Column(String(255))
    contact_phone = Column(String(50))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    simulators = relationship("Simulator", back_populates="manufacturer")

    def __repr__(self):
        return f"<Manufacturer(name='{self.name}')>"


class ReferenceModule(Base):
    """Reference modules used for calibration and testing."""
    __tablename__ = 'reference_modules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    serial_number = Column(String(100), nullable=False, unique=True)
    module_type = Column(String(100))  # e.g., 'c-Si', 'mc-Si', 'CdTe', etc.
    manufacturer = Column(String(255))
    calibration_date = Column(DateTime)
    calibration_lab = Column(String(255))
    isc_stc = Column(Float)  # Short-circuit current at STC (A)
    voc_stc = Column(Float)  # Open-circuit voltage at STC (V)
    pmax_stc = Column(Float)  # Maximum power at STC (W)
    area = Column(Float)  # Active area (m²)
    spectral_response = Column(JSON)  # Spectral response data
    temperature_coefficient = Column(Float)  # %/°C
    uncertainty = Column(Float)  # Calibration uncertainty (%)
    certificate_number = Column(String(100))
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    measurements = relationship("Measurement", back_populates="reference_module")

    def __repr__(self):
        return f"<ReferenceModule(serial='{self.serial_number}')>"


class Simulator(Base):
    """Solar simulator equipment information."""
    __tablename__ = 'simulators'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    serial_number = Column(String(100), unique=True)
    manufacturer_id = Column(Integer, ForeignKey('manufacturers.id'))
    model = Column(String(255))
    lamp_type = Column(String(50))  # Using string for flexibility
    lamp_count = Column(Integer, default=1)
    target_irradiance = Column(Float, default=1000.0)  # W/m²
    test_area = Column(Float)  # Test area in m²
    installation_date = Column(DateTime)
    last_calibration = Column(DateTime)
    next_calibration = Column(DateTime)

    # Current classification
    spectral_class = Column(String(10))
    uniformity_class = Column(String(10))
    temporal_class = Column(String(10))
    overall_class = Column(String(10))

    location = Column(String(255))
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    manufacturer = relationship("Manufacturer", back_populates="simulators")
    measurements = relationship("Measurement", back_populates="simulator")
    lamp_history = relationship("LampHistory", back_populates="simulator")

    __table_args__ = (
        Index('idx_simulator_serial', 'serial_number'),
        Index('idx_simulator_manufacturer', 'manufacturer_id'),
    )

    def __repr__(self):
        return f"<Simulator(name='{self.name}', model='{self.model}')>"


class Measurement(Base):
    """Individual measurement sessions."""
    __tablename__ = 'measurements'

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulator_id = Column(Integer, ForeignKey('simulators.id'), nullable=False)
    reference_module_id = Column(Integer, ForeignKey('reference_modules.id'))
    measurement_date = Column(DateTime, default=datetime.utcnow)
    operator = Column(String(255))

    # Environmental conditions
    ambient_temperature = Column(Float)  # °C
    relative_humidity = Column(Float)  # %
    atmospheric_pressure = Column(Float)  # hPa

    # Measurement settings
    irradiance_setpoint = Column(Float, default=1000.0)  # W/m²
    measured_irradiance = Column(Float)  # W/m²

    # Classification results
    spectral_class = Column(String(10))
    uniformity_class = Column(String(10))
    sti_class = Column(String(10))
    lti_class = Column(String(10))
    overall_class = Column(String(10))

    # Calculated values
    spectral_mismatch = Column(Float)  # Maximum spectral mismatch (%)
    non_uniformity = Column(Float)  # Spatial non-uniformity (%)
    sti_value = Column(Float)  # Short-term instability (%)
    lti_value = Column(Float)  # Long-term instability (%)
    spc = Column(Float)  # Spectral Performance Category
    spd = Column(Float)  # Spectral Performance Deviation

    notes = Column(Text)
    is_valid = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    simulator = relationship("Simulator", back_populates="measurements")
    reference_module = relationship("ReferenceModule", back_populates="measurements")
    spectral_data = relationship("SpectralData", back_populates="measurement",
                                  cascade="all, delete-orphan")
    uniformity_data = relationship("UniformityData", back_populates="measurement",
                                    cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_measurement_date', 'measurement_date'),
        Index('idx_measurement_simulator', 'simulator_id'),
    )

    def __repr__(self):
        return f"<Measurement(id={self.id}, date='{self.measurement_date}')>"


class SpectralData(Base):
    """Spectral irradiance data for measurements."""
    __tablename__ = 'spectral_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'), nullable=False)

    # Band-by-band data (IEC 60904-9 bands)
    band_number = Column(Integer, nullable=False)  # 1-7 for standard bands
    wavelength_start = Column(Float, nullable=False)  # nm
    wavelength_end = Column(Float, nullable=False)  # nm
    band_name = Column(String(50))

    # Measured values
    measured_irradiance = Column(Float)  # W/m²
    reference_irradiance = Column(Float)  # W/m² (AM1.5G)
    ratio = Column(Float)  # Measured/Reference ratio
    deviation = Column(Float)  # Deviation from 1.0 in %

    # Classification for this band
    band_class = Column(String(10))

    # Full spectral data (optional, stored as JSON)
    wavelength_data = Column(JSON)  # Array of wavelengths
    irradiance_data = Column(JSON)  # Array of irradiance values

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    measurement = relationship("Measurement", back_populates="spectral_data")

    __table_args__ = (
        Index('idx_spectral_measurement', 'measurement_id'),
        UniqueConstraint('measurement_id', 'band_number', name='uq_spectral_band'),
    )

    def __repr__(self):
        return f"<SpectralData(band={self.band_number}, ratio={self.ratio})>"


class UniformityData(Base):
    """Spatial uniformity data for measurements."""
    __tablename__ = 'uniformity_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'), nullable=False)

    # Grid position
    row = Column(Integer, nullable=False)
    col = Column(Integer, nullable=False)
    x_position = Column(Float)  # mm from reference point
    y_position = Column(Float)  # mm from reference point

    # Measured values
    irradiance = Column(Float, nullable=False)  # W/m²
    normalized_irradiance = Column(Float)  # Normalized to mean
    deviation_from_mean = Column(Float)  # % deviation

    # Reference cell position marker
    is_reference_position = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    measurement = relationship("Measurement", back_populates="uniformity_data")

    __table_args__ = (
        Index('idx_uniformity_measurement', 'measurement_id'),
        UniqueConstraint('measurement_id', 'row', 'col', name='uq_uniformity_position'),
    )

    def __repr__(self):
        return f"<UniformityData(row={self.row}, col={self.col}, irradiance={self.irradiance})>"


class LampHistory(Base):
    """Lamp usage and replacement history."""
    __tablename__ = 'lamp_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulator_id = Column(Integer, ForeignKey('simulators.id'), nullable=False)

    lamp_type = Column(String(50))
    lamp_serial = Column(String(100))
    lamp_manufacturer = Column(String(255))

    installation_date = Column(DateTime, nullable=False)
    removal_date = Column(DateTime)
    operating_hours = Column(Float, default=0)
    flash_count = Column(Integer, default=0)

    # Performance tracking
    initial_irradiance = Column(Float)  # W/m² when new
    current_irradiance = Column(Float)  # W/m² current
    irradiance_decay = Column(Float)  # % decay

    replacement_reason = Column(String(255))
    notes = Column(Text)
    is_current = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    simulator = relationship("Simulator", back_populates="lamp_history")

    __table_args__ = (
        Index('idx_lamp_simulator', 'simulator_id'),
        Index('idx_lamp_current', 'is_current'),
    )

    def __repr__(self):
        return f"<LampHistory(serial='{self.lamp_serial}', hours={self.operating_hours})>"


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """
    Database connection manager for the Sun Simulator Classification System.
    Handles connection pooling and session management.
    """

    _instance = None
    _engine = None
    _SessionFactory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the database engine with connection pooling."""
        database_url = get_database_url()

        self._engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=os.getenv('DB_ECHO', 'false').lower() == 'true'
        )

        self._SessionFactory = sessionmaker(bind=self._engine)

    @property
    def engine(self):
        """Get the database engine."""
        return self._engine

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self._engine)

    def drop_tables(self):
        """Drop all database tables (use with caution)."""
        Base.metadata.drop_all(self._engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._SessionFactory()

    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db_manager.session_scope() as session:
                session.add(obj)
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self._engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the singleton database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@contextmanager
def get_db_session():
    """
    Context manager for database sessions.

    Usage:
        with get_db_session() as session:
            simulators = session.query(Simulator).all()
    """
    db_manager = get_db_manager()
    with db_manager.session_scope() as session:
        yield session


def init_database():
    """Initialize the database and create all tables."""
    db_manager = get_db_manager()
    db_manager.create_tables()
    return db_manager.test_connection()


# =============================================================================
# DATA ACCESS FUNCTIONS
# =============================================================================

def get_all_simulators(session: Session) -> List[Simulator]:
    """Get all active simulators."""
    return session.query(Simulator).filter(Simulator.is_active == True).all()


def get_simulator_by_id(session: Session, simulator_id: int) -> Optional[Simulator]:
    """Get a simulator by ID."""
    return session.query(Simulator).filter(Simulator.id == simulator_id).first()


def get_recent_measurements(session: Session, limit: int = 10) -> List[Measurement]:
    """Get recent measurements."""
    return (session.query(Measurement)
            .filter(Measurement.is_valid == True)
            .order_by(Measurement.measurement_date.desc())
            .limit(limit)
            .all())


def get_measurements_by_simulator(session: Session, simulator_id: int) -> List[Measurement]:
    """Get all measurements for a simulator."""
    return (session.query(Measurement)
            .filter(Measurement.simulator_id == simulator_id)
            .filter(Measurement.is_valid == True)
            .order_by(Measurement.measurement_date.desc())
            .all())


def get_classification_statistics(session: Session) -> Dict[str, Any]:
    """Get classification statistics summary."""
    from sqlalchemy import func

    stats = {}

    # Count by classification grade
    for grade in ['A+', 'A', 'B', 'C']:
        count = (session.query(func.count(Measurement.id))
                 .filter(Measurement.overall_class == grade)
                 .filter(Measurement.is_valid == True)
                 .scalar())
        stats[grade] = count or 0

    # Total measurements
    stats['total'] = sum(stats.values())

    # Recent measurements count (last 30 days)
    from datetime import timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    stats['recent'] = (session.query(func.count(Measurement.id))
                       .filter(Measurement.measurement_date >= thirty_days_ago)
                       .filter(Measurement.is_valid == True)
                       .scalar() or 0)

    return stats


def save_measurement(session: Session, measurement_data: Dict[str, Any]) -> Measurement:
    """Save a new measurement."""
    measurement = Measurement(**measurement_data)
    session.add(measurement)
    session.flush()
    return measurement


def save_spectral_data(session: Session, measurement_id: int,
                       spectral_data: List[Dict[str, Any]]) -> List[SpectralData]:
    """Save spectral data for a measurement."""
    data_objects = []
    for data in spectral_data:
        data['measurement_id'] = measurement_id
        obj = SpectralData(**data)
        session.add(obj)
        data_objects.append(obj)
    session.flush()
    return data_objects


def save_uniformity_data(session: Session, measurement_id: int,
                         uniformity_data: List[Dict[str, Any]]) -> List[UniformityData]:
    """Save uniformity data for a measurement."""
    data_objects = []
    for data in uniformity_data:
        data['measurement_id'] = measurement_id
        obj = UniformityData(**data)
        session.add(obj)
        data_objects.append(obj)
    session.flush()
    return data_objects
