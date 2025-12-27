"""
Sun Simulator Classification System - Utilities Package
"""

# Database module
from .db import (
    DatabaseManager,
    get_db_session,
    get_db_manager,
    init_database,
    Simulator,
    Measurement,
    SpectralData,
    UniformityData,
    SPCStudy,
    SPCDataPoint,
    MSAStudy,
    MSAMeasurement
)

# IEC 60904-9 Calculations
from .calculations import (
    SpectralCalculator,
    UniformityCalculator,
    TemporalCalculator,
    calculate_spc,
    calculate_spd
)

# SPC Calculations (Phase 2)
from .spc_calculations import (
    SPCCalculator,
    CapabilityCalculator,
    HistogramCalculator,
    generate_spc_sample_data,
    generate_capability_sample_data
)

# MSA Calculations (Phase 2)
from .msa_calculations import (
    GRRCalculator,
    GRRResult,
    GRRRating,
    RangeMethodCalculator,
    generate_grr_sample_data,
    calculate_variance_chart_data
)

__all__ = [
    # Database
    'DatabaseManager',
    'get_db_session',
    'get_db_manager',
    'init_database',
    'Simulator',
    'Measurement',
    'SpectralData',
    'UniformityData',
    'SPCStudy',
    'SPCDataPoint',
    'MSAStudy',
    'MSAMeasurement',

    # IEC Calculations
    'SpectralCalculator',
    'UniformityCalculator',
    'TemporalCalculator',
    'calculate_spc',
    'calculate_spd',

    # SPC
    'SPCCalculator',
    'CapabilityCalculator',
    'HistogramCalculator',
    'generate_spc_sample_data',
    'generate_capability_sample_data',

    # MSA
    'GRRCalculator',
    'GRRResult',
    'GRRRating',
    'RangeMethodCalculator',
    'generate_grr_sample_data',
    'calculate_variance_chart_data'
]
