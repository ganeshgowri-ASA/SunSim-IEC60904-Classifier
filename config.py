"""
Sun Simulator Classification System - Configuration
IEC 60904-9:2020 Ed.3 Standard Compliance

This module contains all configuration parameters for the Sun Simulator
Classification System following IEC 60904-9 Ed.3 standards.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# =============================================================================
# IEC 60904-9:2020 Ed.3 CLASSIFICATION LIMITS
# =============================================================================

@dataclass(frozen=True)
class ClassificationLimits:
    """IEC 60904-9:2020 classification limits for each parameter."""

    # Spectral Match Classification (% deviation from reference)
    # Each wavelength band must meet these criteria
    SPECTRAL_MATCH: Dict[str, Tuple[float, float]] = None

    # Spatial Non-Uniformity Classification (%)
    UNIFORMITY: Dict[str, float] = None

    # Temporal Instability - Short Term (STI) (%)
    STI: Dict[str, float] = None

    # Temporal Instability - Long Term (LTI) (%)
    LTI: Dict[str, float] = None

    def __post_init__(self):
        object.__setattr__(self, 'SPECTRAL_MATCH', {
            'A+': (0.875, 1.125),  # ±12.5% (0.875 to 1.125 ratio)
            'A': (0.75, 1.25),     # ±25% (0.75 to 1.25 ratio)
            'B': (0.6, 1.4),       # ±40% (0.6 to 1.4 ratio)
            'C': (0.0, float('inf'))  # Outside B limits
        })
        object.__setattr__(self, 'UNIFORMITY', {
            'A+': 1.0,   # ≤1%
            'A': 2.0,    # ≤2%
            'B': 5.0,    # ≤5%
            'C': 10.0    # ≤10%
        })
        object.__setattr__(self, 'STI', {
            'A+': 0.5,   # ≤0.5%
            'A': 2.0,    # ≤2%
            'B': 5.0,    # ≤5%
            'C': 10.0    # ≤10%
        })
        object.__setattr__(self, 'LTI', {
            'A+': 1.0,   # ≤1%
            'A': 2.0,    # ≤2%
            'B': 5.0,    # ≤5%
            'C': 10.0    # ≤10%
        })


# Initialize classification limits
CLASSIFICATION = ClassificationLimits()

# =============================================================================
# IEC 60904-9:2020 WAVELENGTH BANDS
# =============================================================================

# Six spectral bands as per IEC 60904-9:2020 Ed.3
# Extended range from 300nm to 1200nm
WAVELENGTH_BANDS: List[Tuple[int, int, str]] = [
    (300, 400, "UV-A"),
    (400, 500, "Blue"),
    (500, 600, "Green"),
    (600, 700, "Red"),
    (700, 800, "Near-IR 1"),
    (800, 900, "Near-IR 2"),
    (900, 1100, "Near-IR 3"),
]

# Additional band for extended spectrum (IEC 60904-9:2020)
EXTENDED_WAVELENGTH_BANDS: List[Tuple[int, int, str]] = [
    (300, 400, "UV-A"),
    (400, 500, "Blue"),
    (500, 600, "Green"),
    (600, 700, "Red"),
    (700, 800, "Near-IR 1"),
    (800, 900, "Near-IR 2"),
    (900, 1100, "Near-IR 3"),
    (1100, 1200, "Near-IR 4"),
]

# Wavelength range for full spectrum analysis
WAVELENGTH_RANGE = {
    'min': 300,
    'max': 1200,
    'reference_min': 300,
    'reference_max': 1100,  # Standard reference range
    'extended_max': 1200    # Extended range for bifacial modules
}

# =============================================================================
# REFERENCE SPECTRUM CONFIGURATION (AM1.5G)
# =============================================================================

AM15G_CONFIG = {
    'standard': 'IEC 60904-3:2019',
    'name': 'AM1.5 Global',
    'total_irradiance': 1000.0,  # W/m²
    'airmass': 1.5,
    'tilt_angle': 37,  # degrees
    'description': 'Standard Test Conditions (STC) reference spectrum'
}

# =============================================================================
# CLASSIFICATION BADGE COLORS
# =============================================================================

BADGE_COLORS = {
    'A+': '#10b981',  # Emerald green
    'A': '#3b82f6',   # Blue
    'B': '#f59e0b',   # Amber
    'C': '#ef4444',   # Red
    'N/A': '#6b7280'  # Gray
}

BADGE_COLORS_LIGHT = {
    'A+': '#d1fae5',  # Light emerald
    'A': '#dbeafe',   # Light blue
    'B': '#fef3c7',   # Light amber
    'C': '#fee2e2',   # Light red
    'N/A': '#f3f4f6'  # Light gray
}

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_CONFIG = {
    'title': 'Sun Simulator Classification System',
    'subtitle': 'IEC 60904-9:2020 Ed.3 Compliance',
    'version': '1.0.0',
    'author': 'PV Testing Laboratory',
    'page_icon': '☀️',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Theme colors
THEME = {
    'sidebar_bg': '#1e293b',        # Dark slate
    'sidebar_text': '#e2e8f0',      # Light slate
    'primary': '#3b82f6',           # Blue
    'secondary': '#10b981',         # Green
    'background': '#0f172a',        # Dark background
    'surface': '#1e293b',           # Card surface
    'text_primary': '#f8fafc',      # White text
    'text_secondary': '#94a3b8',    # Muted text
    'border': '#334155',            # Border color
    'success': '#10b981',           # Success green
    'warning': '#f59e0b',           # Warning amber
    'error': '#ef4444',             # Error red
    'info': '#3b82f6'               # Info blue
}

# Chart configuration
CHART_CONFIG = {
    'template': 'plotly_dark',
    'color_scale': 'Viridis',
    'font_family': 'Inter, sans-serif',
    'title_font_size': 16,
    'axis_font_size': 12,
    'legend_font_size': 10,
    'height': 400,
    'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60}
}

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_CONFIG = {
    'provider': 'postgresql',
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 1800,
    'echo': False
}

# Environment variable names for database connection
DB_ENV_VARS = {
    'url': 'DATABASE_URL',
    'host': 'PGHOST',
    'port': 'PGPORT',
    'database': 'PGDATABASE',
    'user': 'PGUSER',
    'password': 'PGPASSWORD'
}

# =============================================================================
# FILE UPLOAD CONFIGURATION
# =============================================================================

UPLOAD_CONFIG = {
    'max_file_size_mb': 50,
    'allowed_extensions': ['.csv', '.xlsx', '.xls', '.txt'],
    'spectral_file_columns': ['wavelength', 'irradiance'],
    'uniformity_file_columns': ['x', 'y', 'irradiance'],
    'temporal_file_columns': ['time', 'irradiance']
}

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

EXPORT_CONFIG = {
    'formats': ['PDF', 'Excel', 'Word', 'CSV'],
    'pdf_template': 'IEC_60904-9_Report',
    'company_logo_path': 'assets/logo.png',
    'report_footer': 'Generated by Sun Simulator Classification System v1.0.0'
}

# =============================================================================
# MEASUREMENT CONFIGURATION
# =============================================================================

MEASUREMENT_CONFIG = {
    # Uniformity grid settings
    'min_grid_size': 3,
    'max_grid_size': 15,
    'default_grid_size': 5,

    # Temporal stability settings
    'min_samples': 100,
    'max_samples': 10000,
    'default_sample_rate': 1000,  # Hz
    'sti_window_ms': 1,           # Short term window
    'lti_window_s': 60,           # Long term window

    # Spectral settings
    'wavelength_resolution': 1,   # nm
    'min_wavelength': 300,        # nm
    'max_wavelength': 1200,       # nm
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_classification(value: float, parameter: str) -> str:
    """
    Determine classification grade based on value and parameter type.

    Args:
        value: The measured value (ratio for spectral, % for others)
        parameter: One of 'spectral', 'uniformity', 'sti', 'lti'

    Returns:
        Classification grade: 'A+', 'A', 'B', or 'C'
    """
    if parameter == 'spectral':
        limits = CLASSIFICATION.SPECTRAL_MATCH
        if limits['A+'][0] <= value <= limits['A+'][1]:
            return 'A+'
        elif limits['A'][0] <= value <= limits['A'][1]:
            return 'A'
        elif limits['B'][0] <= value <= limits['B'][1]:
            return 'B'
        else:
            return 'C'
    else:
        if parameter == 'uniformity':
            limits = CLASSIFICATION.UNIFORMITY
        elif parameter == 'sti':
            limits = CLASSIFICATION.STI
        elif parameter == 'lti':
            limits = CLASSIFICATION.LTI
        else:
            return 'N/A'

        if value <= limits['A+']:
            return 'A+'
        elif value <= limits['A']:
            return 'A'
        elif value <= limits['B']:
            return 'B'
        else:
            return 'C'


def get_overall_classification(*classifications: str) -> str:
    """
    Determine overall classification from multiple individual classifications.
    The overall classification is the worst of all individual classifications.

    Args:
        *classifications: Variable number of classification grades

    Returns:
        Overall classification grade
    """
    priority = {'C': 0, 'B': 1, 'A': 2, 'A+': 3, 'N/A': -1}
    valid_classes = [c for c in classifications if c != 'N/A']

    if not valid_classes:
        return 'N/A'

    return min(valid_classes, key=lambda x: priority.get(x, -1))


def get_database_url() -> str:
    """
    Get database URL from environment variables.

    Returns:
        PostgreSQL connection URL
    """
    # Try direct DATABASE_URL first
    url = os.getenv(DB_ENV_VARS['url'])
    if url:
        return url

    # Build from components
    host = os.getenv(DB_ENV_VARS['host'], 'localhost')
    port = os.getenv(DB_ENV_VARS['port'], '5432')
    database = os.getenv(DB_ENV_VARS['database'], 'sunsim')
    user = os.getenv(DB_ENV_VARS['user'], 'postgres')
    password = os.getenv(DB_ENV_VARS['password'], '')

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"
