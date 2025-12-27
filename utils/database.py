"""
Database models and utilities for SunSim-IEC60904-Classifier.

Contains tables for:
- Lamp tracking and management
- Spectrum drift monitoring (UV/NIR shift tracking per TÜV paper)
- Flash repeatability records
- Calibration history
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
)
from sqlalchemy.engine import Engine

Base = declarative_base()

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sunsim.db"
)


class Lamp(Base):
    """
    Lamp tracking table for sun simulator lamp management.

    Tracks lamp identification, operating hours, flash counts,
    and replacement history per IEC 60904-9 requirements.
    """
    __tablename__ = "lamps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lamp_id = Column(String(50), unique=True, nullable=False, index=True)
    manufacturer = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)
    lamp_type = Column(String(50), default="Xenon")  # Xenon, LED, Metal Halide, etc.
    serial_number = Column(String(100))

    # Operating metrics
    flash_count = Column(Integer, default=0)
    operating_hours = Column(Float, default=0.0)
    max_flash_count = Column(Integer, default=100000)  # Manufacturer rated limit
    max_operating_hours = Column(Float, default=1000.0)

    # Installation and status
    installation_date = Column(DateTime, default=datetime.utcnow)
    last_flash_date = Column(DateTime)
    is_active = Column(Boolean, default=True)
    status = Column(String(20), default="active")  # active, aging, warning, replaced

    # Calibration tracking
    last_calibration_date = Column(DateTime)
    next_calibration_date = Column(DateTime)
    calibration_interval_days = Column(Integer, default=365)

    # Power settings
    rated_power_watts = Column(Float)
    current_power_percent = Column(Float, default=100.0)

    # Aging thresholds (percentage of life)
    warning_threshold_percent = Column(Float, default=80.0)
    critical_threshold_percent = Column(Float, default=95.0)

    # Notes
    notes = Column(Text)

    # Relationships
    calibrations = relationship("LampCalibration", back_populates="lamp", cascade="all, delete-orphan")
    flash_records = relationship("FlashRecord", back_populates="lamp", cascade="all, delete-orphan")
    spectrum_drifts = relationship("SpectrumDrift", back_populates="lamp", cascade="all, delete-orphan")
    repeatability_records = relationship("RepeatabilityRecord", back_populates="lamp", cascade="all, delete-orphan")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def life_percentage(self) -> float:
        """Calculate lamp life percentage based on flash count."""
        if self.max_flash_count and self.max_flash_count > 0:
            return min(100.0, (self.flash_count / self.max_flash_count) * 100)
        return 0.0

    @property
    def hours_life_percentage(self) -> float:
        """Calculate lamp life percentage based on operating hours."""
        if self.max_operating_hours and self.max_operating_hours > 0:
            return min(100.0, (self.operating_hours / self.max_operating_hours) * 100)
        return 0.0

    @property
    def calibration_due(self) -> bool:
        """Check if calibration is due."""
        if self.next_calibration_date:
            return datetime.utcnow() >= self.next_calibration_date
        return True

    @property
    def days_until_calibration(self) -> Optional[int]:
        """Calculate days until next calibration."""
        if self.next_calibration_date:
            delta = self.next_calibration_date - datetime.utcnow()
            return delta.days
        return None

    def update_status(self) -> str:
        """Update lamp status based on life percentage."""
        life_pct = max(self.life_percentage, self.hours_life_percentage)

        if life_pct >= self.critical_threshold_percent:
            self.status = "critical"
        elif life_pct >= self.warning_threshold_percent:
            self.status = "warning"
        elif life_pct > 0:
            self.status = "aging"
        else:
            self.status = "active"

        return self.status


class LampCalibration(Base):
    """
    Lamp calibration history tracking.

    Records calibration events, results, and certificates.
    """
    __tablename__ = "lamp_calibrations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lamp_id = Column(Integer, ForeignKey("lamps.id"), nullable=False, index=True)

    calibration_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    calibration_type = Column(String(50), default="routine")  # routine, initial, post-repair

    # Calibration results
    spectral_match_class = Column(String(5))  # A+, A, B, C
    uniformity_class = Column(String(5))
    temporal_stability_class = Column(String(5))
    overall_class = Column(String(5))

    # Spectral data at calibration
    uv_deviation_percent = Column(Float)  # 300-400nm deviation
    vis_deviation_percent = Column(Float)  # 400-700nm deviation
    nir_deviation_percent = Column(Float)  # 700-1100nm deviation

    # Reference values
    reference_irradiance = Column(Float)  # W/m² at calibration
    flash_count_at_calibration = Column(Integer)

    # Certification
    certificate_number = Column(String(100))
    calibrated_by = Column(String(100))
    laboratory = Column(String(200))

    # Attachments and notes
    calibration_data = Column(JSON)  # Full spectral data if needed
    notes = Column(Text)

    # Relationship
    lamp = relationship("Lamp", back_populates="calibrations")

    created_at = Column(DateTime, default=datetime.utcnow)


class FlashRecord(Base):
    """
    Individual flash event records for detailed tracking.

    Used for repeatability analysis and trend monitoring.
    """
    __tablename__ = "flash_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lamp_id = Column(Integer, ForeignKey("lamps.id"), nullable=False, index=True)

    flash_number = Column(Integer, nullable=False)  # Sequential flash number for this lamp
    flash_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Flash measurements
    irradiance = Column(Float)  # W/m²
    pulse_duration_ms = Column(Float)  # milliseconds

    # Spectral measurements
    uv_ratio = Column(Float)  # UV content ratio
    vis_ratio = Column(Float)  # Visible content ratio
    nir_ratio = Column(Float)  # NIR content ratio

    # Power settings during flash
    power_percent = Column(Float)

    # Quality metrics
    temporal_stability_percent = Column(Float)
    uniformity_percent = Column(Float)

    # Batch/session tracking
    session_id = Column(String(50), index=True)
    batch_id = Column(String(50))

    # Relationship
    lamp = relationship("Lamp", back_populates="flash_records")

    # Indexes for efficient querying
    __table_args__ = (
        Index("ix_flash_lamp_timestamp", "lamp_id", "flash_timestamp"),
        Index("ix_flash_lamp_number", "lamp_id", "flash_number"),
    )


class SpectrumDrift(Base):
    """
    Spectrum drift tracking per TÜV paper findings.

    Monitors UV/NIR degradation (Xenon aging), blue-shift during pulse,
    and lamp power adjustment effects.

    Key metrics tracked:
    - UV degradation over flash count (Xenon specific)
    - NIR stability monitoring
    - Blue-shift phenomena during individual pulses
    - Spectral shift vs lamp power settings
    """
    __tablename__ = "spectrum_drift"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lamp_id = Column(Integer, ForeignKey("lamps.id"), nullable=False, index=True)

    measurement_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    flash_count_at_measurement = Column(Integer, nullable=False)

    # UV Region (300-400nm) - Critical for Xenon aging
    uv_300_350_percent = Column(Float)  # 300-350nm deviation from reference
    uv_350_400_percent = Column(Float)  # 350-400nm deviation from reference
    uv_total_shift_percent = Column(Float)  # Total UV shift from baseline

    # Visible Region (400-700nm)
    vis_400_500_percent = Column(Float)  # Blue region
    vis_500_600_percent = Column(Float)  # Green region
    vis_600_700_percent = Column(Float)  # Red region
    vis_total_shift_percent = Column(Float)

    # NIR Region (700-1100nm)
    nir_700_800_percent = Column(Float)
    nir_800_900_percent = Column(Float)
    nir_900_1000_percent = Column(Float)
    nir_1000_1100_percent = Column(Float)
    nir_total_shift_percent = Column(Float)

    # Blue-shift during pulse tracking (TÜV paper finding)
    blue_shift_detected = Column(Boolean, default=False)
    blue_shift_magnitude_nm = Column(Float)  # Peak wavelength shift in nm
    blue_shift_timing_ms = Column(Float)  # When during pulse (start vs end)

    # Power adjustment effects
    power_setting_percent = Column(Float)
    power_adjusted_shift = Column(Float)  # Shift attributable to power change

    # Reference comparison
    reference_date = Column(DateTime)  # Date of reference spectrum
    overall_spectral_mismatch = Column(Float)  # Overall SM from IEC 60904-9

    # Classification impact
    classification_before = Column(String(5))  # Class before drift
    classification_after = Column(String(5))  # Class after drift
    classification_changed = Column(Boolean, default=False)

    # Trend indicators
    trend_direction = Column(String(20))  # improving, stable, degrading
    rate_of_change_per_1000_flashes = Column(Float)  # % change per 1000 flashes

    # Manufacturer comparison data
    manufacturer = Column(String(100))
    lamp_model = Column(String(100))

    # Full spectral data (optional, for detailed analysis)
    wavelengths = Column(JSON)  # Array of wavelength values
    intensities = Column(JSON)  # Array of intensity values
    reference_intensities = Column(JSON)  # Reference spectrum intensities

    notes = Column(Text)

    # Relationship
    lamp = relationship("Lamp", back_populates="spectrum_drifts")

    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes for trend analysis
    __table_args__ = (
        Index("ix_drift_lamp_date", "lamp_id", "measurement_date"),
        Index("ix_drift_lamp_flash", "lamp_id", "flash_count_at_measurement"),
        Index("ix_drift_manufacturer", "manufacturer", "lamp_model"),
    )


class RepeatabilityRecord(Base):
    """
    Flash-to-flash repeatability tracking.

    Records statistical metrics for repeatability analysis
    with target of 0.09% per IEC 60904-9 requirements.
    """
    __tablename__ = "repeatability_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lamp_id = Column(Integer, ForeignKey("lamps.id"), nullable=False, index=True)

    measurement_date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    session_id = Column(String(50), index=True)

    # Number of flashes in this measurement set
    flash_count = Column(Integer, nullable=False)
    start_flash_number = Column(Integer)
    end_flash_number = Column(Integer)

    # Statistical metrics - Irradiance
    irradiance_mean = Column(Float)
    irradiance_std_dev = Column(Float)
    irradiance_min = Column(Float)
    irradiance_max = Column(Float)
    irradiance_range = Column(Float)
    irradiance_cv_percent = Column(Float)  # Coefficient of Variation (%)

    # Repeatability metric (target: 0.09%)
    repeatability_percent = Column(Float)  # Main repeatability metric
    repeatability_pass = Column(Boolean)  # Pass/Fail vs 0.09% target

    # Control chart data
    ucl = Column(Float)  # Upper Control Limit
    lcl = Column(Float)  # Lower Control Limit
    centerline = Column(Float)  # Center Line (target/mean)

    # Out of control indicators
    out_of_control = Column(Boolean, default=False)
    out_of_control_reason = Column(String(200))

    # Trend analysis
    trend_direction = Column(String(20))  # improving, stable, degrading
    consecutive_improving = Column(Integer, default=0)
    consecutive_degrading = Column(Integer, default=0)

    # Environmental conditions during measurement
    ambient_temp_c = Column(Float)
    humidity_percent = Column(Float)

    # Flash count at this measurement
    lamp_total_flash_count = Column(Integer)

    notes = Column(Text)

    # Relationship
    lamp = relationship("Lamp", back_populates="repeatability_records")

    created_at = Column(DateTime, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index("ix_repeat_lamp_date", "lamp_id", "measurement_date"),
        Index("ix_repeat_session", "session_id"),
    )


class LampReplacementHistory(Base):
    """
    Track lamp replacement events for audit trail and trending.
    """
    __tablename__ = "lamp_replacement_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Old lamp info
    old_lamp_id = Column(String(50), nullable=False)
    old_lamp_manufacturer = Column(String(100))
    old_lamp_model = Column(String(100))
    old_lamp_flash_count = Column(Integer)
    old_lamp_operating_hours = Column(Float)
    old_lamp_installation_date = Column(DateTime)

    # New lamp info
    new_lamp_id = Column(String(50), nullable=False)
    new_lamp_manufacturer = Column(String(100))
    new_lamp_model = Column(String(100))

    # Replacement details
    replacement_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    replacement_reason = Column(String(200))  # aging, failure, upgrade, calibration

    # Performance comparison
    old_lamp_last_class = Column(String(5))
    new_lamp_initial_class = Column(String(5))

    replaced_by = Column(String(100))
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(db_path: Optional[str] = None) -> Engine:
    """
    Create and return database engine.

    Args:
        db_path: Optional path to SQLite database. Uses default if not specified.

    Returns:
        SQLAlchemy Engine instance.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        future=True,
    )
    return engine


def get_session(engine: Optional[Engine] = None) -> Session:
    """
    Create and return a database session.

    Args:
        engine: Optional SQLAlchemy engine. Creates default if not specified.

    Returns:
        SQLAlchemy Session instance.
    """
    if engine is None:
        engine = get_engine()

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def init_database(engine: Optional[Engine] = None) -> Engine:
    """
    Initialize database and create all tables.

    Args:
        engine: Optional SQLAlchemy engine. Creates default if not specified.

    Returns:
        SQLAlchemy Engine instance used for initialization.
    """
    if engine is None:
        engine = get_engine()

    Base.metadata.create_all(bind=engine)
    return engine


# Convenience functions for common queries

def get_active_lamps(session: Session) -> List[Lamp]:
    """Get all active lamps."""
    return session.query(Lamp).filter(Lamp.is_active == True).all()


def get_lamps_needing_calibration(session: Session, days_warning: int = 30) -> List[Lamp]:
    """Get lamps with calibration due within specified days."""
    cutoff_date = datetime.utcnow() + timedelta(days=days_warning)
    return session.query(Lamp).filter(
        Lamp.is_active == True,
        Lamp.next_calibration_date <= cutoff_date
    ).all()


def get_lamps_at_warning_threshold(session: Session) -> List[Lamp]:
    """Get lamps that have reached warning threshold."""
    lamps = session.query(Lamp).filter(Lamp.is_active == True).all()
    return [lamp for lamp in lamps if lamp.life_percentage >= lamp.warning_threshold_percent]


def get_recent_drift_records(
    session: Session,
    lamp_id: int,
    limit: int = 100
) -> List[SpectrumDrift]:
    """Get recent spectrum drift records for a lamp."""
    return session.query(SpectrumDrift).filter(
        SpectrumDrift.lamp_id == lamp_id
    ).order_by(SpectrumDrift.measurement_date.desc()).limit(limit).all()


def get_repeatability_history(
    session: Session,
    lamp_id: int,
    limit: int = 100
) -> List[RepeatabilityRecord]:
    """Get repeatability history for a lamp."""
    return session.query(RepeatabilityRecord).filter(
        RepeatabilityRecord.lamp_id == lamp_id
    ).order_by(RepeatabilityRecord.measurement_date.desc()).limit(limit).all()
