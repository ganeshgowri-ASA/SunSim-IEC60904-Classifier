"""
Sun Simulator Classification System - Utilities Package

This package provides core functionality for:
- Database operations and ORM models
- IEC 60904-9 classification calculations
- IEC 60904-14 capacitive effects analysis
- Angular distribution and FOV analysis
"""

from .db import (
    DatabaseManager,
    get_db_session,
    get_db_manager,
    init_database,
    # Models
    Simulator,
    Measurement,
    SpectralData,
    UniformityData,
    User,
    SystemSettings,
    AlarmThreshold,
    # Settings functions
    get_all_users,
    get_setting,
    save_setting,
    get_alarm_thresholds,
    initialize_default_settings,
    initialize_default_alarm_thresholds
)

from .calculations import (
    SpectralCalculator,
    UniformityCalculator,
    TemporalCalculator,
    calculate_spc,
    calculate_spd
)

from .capacitive_effects import (
    estimate_module_capacitance,
    calculate_capacitance_from_transient,
    optimize_sweep_rate,
    calculate_sweep_rate_for_module,
    calculate_correction_factors,
    apply_capacitive_correction,
    get_four_wire_connection_guide,
    analyze_capacitive_effects,
    get_capacitance_summary_table,
    CapacitanceResult,
    SweepRateOptimization,
    IVCorrectionFactors,
    FourWireGuide
)

from .angular_distribution import (
    calculate_solid_angle,
    calculate_half_angle,
    get_sun_solid_angle,
    analyze_beam_collimation,
    measure_beam_angle_from_profile,
    analyze_angle_of_incidence,
    calculate_cosine_correction,
    analyze_diffuse_direct_ratio,
    analyze_angular_profile,
    analyze_fov_mismatch,
    generate_gaussian_profile,
    generate_flat_top_profile,
    comprehensive_angular_analysis,
    get_angular_summary_table,
    BeamCollimationResult,
    AngleOfIncidenceResult,
    DiffuseDirectRatio,
    AngularDistributionProfile,
    FOVAnalysisResult
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
    'User',
    'SystemSettings',
    'AlarmThreshold',
    'get_all_users',
    'get_setting',
    'save_setting',
    'get_alarm_thresholds',
    'initialize_default_settings',
    'initialize_default_alarm_thresholds',

    # Calculations
    'SpectralCalculator',
    'UniformityCalculator',
    'TemporalCalculator',
    'calculate_spc',
    'calculate_spd',

    # Capacitive Effects
    'estimate_module_capacitance',
    'calculate_capacitance_from_transient',
    'optimize_sweep_rate',
    'calculate_sweep_rate_for_module',
    'calculate_correction_factors',
    'apply_capacitive_correction',
    'get_four_wire_connection_guide',
    'analyze_capacitive_effects',
    'get_capacitance_summary_table',
    'CapacitanceResult',
    'SweepRateOptimization',
    'IVCorrectionFactors',
    'FourWireGuide',

    # Angular Distribution
    'calculate_solid_angle',
    'calculate_half_angle',
    'get_sun_solid_angle',
    'analyze_beam_collimation',
    'measure_beam_angle_from_profile',
    'analyze_angle_of_incidence',
    'calculate_cosine_correction',
    'analyze_diffuse_direct_ratio',
    'analyze_angular_profile',
    'analyze_fov_mismatch',
    'generate_gaussian_profile',
    'generate_flat_top_profile',
    'comprehensive_angular_analysis',
    'get_angular_summary_table',
    'BeamCollimationResult',
    'AngleOfIncidenceResult',
    'DiffuseDirectRatio',
    'AngularDistributionProfile',
    'FOVAnalysisResult'
]
