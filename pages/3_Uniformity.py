"""
Uniformity Analysis Page - Enhanced v2.1
IEC 60904-9 Ed.3 Solar Simulator Classification

Non-Uniformity analysis with irradiance uniformity measurements
across the test plane using a grid-based measurement approach.

Features:
- CSV/Excel file upload for measurement data
- Variable grid sizes with user-defined test plane dimensions
- Detector size dropdown (10x10mm, 20x20mm, 50x50mm, 100x100mm, custom)
- Test plane size configuration with optimal position calculation
- Grid size selector (3x3, 5x5, 7x7, 9x9, custom NxM) with visual preview
- Purpose selector: Classification / Reference Cell Positioning / Uncertainty Calculations
- Reference cell position correction factor calculation
- Interactive 2D heatmap and 3D surface plots
- Row/column statistics and position analysis
- PostgreSQL database storage with test_id linkage
- Excel/CSV export functionality

Version: 2.1 - Uniformity Enhanced
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
import sys
from pathlib import Path
import uuid

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    UniformityMeasurement,
    UniformityResult,
    get_grade_color,
)

# Import database utilities
try:
    from utils.db import get_engine, ensure_database_ready, insert_simulator_selection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

from utils.simulator_ui import (
    render_simulator_selector,
    render_simulator_summary_card,
    get_selected_simulator,
    get_simulator_id_for_db,
)

# Page configuration
st.set_page_config(
    page_title="Uniformity Analysis | SunSim",
    page_icon=":material/grid_on:",
    layout="wide",
)

# Custom CSS matching Classification Dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    .grade-badge-large {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 16px;
        min-width: 120px;
    }

    .grade-a-plus { background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; }
    .grade-a { background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); color: white; }
    .grade-b { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: white; }
    .grade-c { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; }
    .grade-fail { background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%); color: white; }

    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .formula-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 1rem 0;
    }

    .stats-table {
        width: 100%;
        border-collapse: collapse;
    }

    .stats-table th, .stats-table td {
        padding: 8px 12px;
        text-align: center;
        border: 1px solid #E2E8F0;
    }

    .stats-table th {
        background: #F1F5F9;
        font-weight: 600;
        color: #1E3A5F;
    }

    .ref-cell-marker {
        background: #EF4444;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .upload-section {
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .grade-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366F1;
    }

    .config-section {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
    }

    .purpose-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 2px solid #E2E8F0;
        cursor: pointer;
        transition: all 0.2s;
    }

    .purpose-card.selected {
        border-color: #6366F1;
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
    }

    .purpose-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }

    .grid-preview {
        background: white;
        border: 2px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .detector-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .position-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #6366F1;
        border-radius: 50%;
        margin: 2px;
    }

    .optimal-info {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 4px solid #10B981;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_grade_class(grade: ClassificationGrade) -> str:
    """Get CSS class for grade badge"""
    grade_classes = {
        ClassificationGrade.A_PLUS: "grade-a-plus",
        ClassificationGrade.A: "grade-a",
        ClassificationGrade.B: "grade-b",
        ClassificationGrade.C: "grade-c",
        ClassificationGrade.FAIL: "grade-fail",
    }
    return grade_classes.get(grade, "grade-fail")


# ============================================================================
# DETECTOR SIZE CONFIGURATION
# ============================================================================

# Predefined detector sizes (width x height in mm)
DETECTOR_SIZES = {
    "10x10mm": (10.0, 10.0),
    "20x20mm": (20.0, 20.0),
    "50x50mm": (50.0, 50.0),
    "100x100mm": (100.0, 100.0),
    "Custom": None,  # User-defined
}

# Predefined grid configurations
GRID_PRESETS = {
    "3x3": (3, 3),
    "5x5": (5, 5),
    "7x7": (7, 7),
    "9x9": (9, 9),
    "11x11": (11, 11),
    "15x15": (15, 15),
    "Custom": None,  # User-defined NxM
}

# Purpose-specific configurations
UNIFORMITY_PURPOSES = {
    "Classification": {
        "description": "IEC 60904-9 classification of solar simulator non-uniformity",
        "icon": "📊",
        "requirements": "Full test plane coverage, minimum 121 points recommended",
        "formula": "Standard IEC formula: ((E_max - E_min) / (E_max + E_min)) × 100",
        "output_focus": "Classification grade (A+/A/B/C)",
    },
    "Reference Cell Positioning": {
        "description": "Determine optimal reference cell placement based on uniformity map",
        "icon": "🎯",
        "requirements": "Higher resolution grid for accurate position identification",
        "formula": "Position correction factor: Grid_Average / Ref_Cell_Value",
        "output_focus": "Best position coordinates and correction factors",
    },
    "Uncertainty Calculations": {
        "description": "Calculate measurement uncertainty contributions from non-uniformity",
        "icon": "📐",
        "requirements": "Statistical analysis with uncertainty propagation",
        "formula": "Expanded uncertainty: U = k × σ (k=2 for 95% confidence)",
        "output_focus": "Uncertainty budget and expanded uncertainty values",
    },
}


def calculate_optimal_grid_size(plane_width: float, plane_height: float,
                                  detector_width: float, detector_height: float) -> tuple:
    """
    Calculate optimal measurement grid size based on test plane and detector dimensions.

    Per IEC 60904-9:
    - Minimum 121 points (11x11) for classification
    - Grid spacing should not exceed 2x detector dimension
    - Points should cover at least 90% of test plane area
    """
    if detector_width <= 0 or detector_height <= 0:
        # Default to 11x11 for point measurements
        return 11, 11, "Point measurement mode"

    # Calculate maximum recommended grid spacing (2x detector size)
    max_spacing_x = detector_width * 2
    max_spacing_y = detector_height * 2

    # Calculate minimum points needed for coverage
    min_cols = max(3, int(np.ceil(plane_width / max_spacing_x)) + 1)
    min_rows = max(3, int(np.ceil(plane_height / max_spacing_y)) + 1)

    # Ensure at least 121 points (11x11) for IEC compliance
    if min_rows * min_cols < 121:
        # Scale up proportionally
        scale_factor = np.sqrt(121 / (min_rows * min_cols))
        min_rows = max(11, int(np.ceil(min_rows * scale_factor)))
        min_cols = max(11, int(np.ceil(min_cols * scale_factor)))

    # Calculate actual grid spacing
    actual_spacing_x = plane_width / (min_cols - 1) if min_cols > 1 else plane_width
    actual_spacing_y = plane_height / (min_rows - 1) if min_rows > 1 else plane_height

    info = f"Grid spacing: {actual_spacing_x:.1f}mm × {actual_spacing_y:.1f}mm"

    return min_rows, min_cols, info


def calculate_measurement_positions(plane_width: float, plane_height: float,
                                     grid_rows: int, grid_cols: int,
                                     detector_width: float = 0, detector_height: float = 0) -> dict:
    """
    Calculate measurement positions and provide coverage analysis.

    Returns dict with:
    - positions: list of (x, y) coordinates
    - coverage_percent: percentage of test plane covered
    - spacing: (dx, dy) grid spacing
    - recommendations: list of recommendations
    """
    x = np.linspace(-plane_width / 2, plane_width / 2, grid_cols)
    y = np.linspace(-plane_height / 2, plane_height / 2, grid_rows)

    positions = [(xi, yi) for yi in y for xi in x]

    # Calculate spacing
    dx = plane_width / (grid_cols - 1) if grid_cols > 1 else plane_width
    dy = plane_height / (grid_rows - 1) if grid_rows > 1 else plane_height

    # Coverage analysis
    total_points = grid_rows * grid_cols
    covered_width = plane_width
    covered_height = plane_height

    # If detector size specified, calculate effective coverage
    if detector_width > 0 and detector_height > 0:
        effective_coverage_x = min(1.0, (grid_cols * detector_width) / plane_width)
        effective_coverage_y = min(1.0, (grid_rows * detector_height) / plane_height)
        coverage_percent = effective_coverage_x * effective_coverage_y * 100
    else:
        # For point measurements, estimate based on grid density
        coverage_percent = min(100, (total_points / 121) * 90)  # 121 points = 90% reference

    # Generate recommendations
    recommendations = []
    if total_points < 121:
        recommendations.append("⚠️ IEC 60904-9 recommends minimum 121 measurement points")
    if total_points >= 121 and total_points < 225:
        recommendations.append("✓ Meets minimum IEC requirements")
    if total_points >= 225:
        recommendations.append("✓ Exceeds IEC requirements - High resolution measurement")

    if detector_width > 0 and detector_height > 0:
        if dx > detector_width * 2 or dy > detector_height * 2:
            recommendations.append("⚠️ Grid spacing exceeds 2× detector size - consider denser grid")
        elif dx < detector_width * 0.5 or dy < detector_height * 0.5:
            recommendations.append("ℹ️ Significant overlap between detector positions")

    return {
        "positions": positions,
        "coverage_percent": coverage_percent,
        "spacing": (dx, dy),
        "total_points": total_points,
        "recommendations": recommendations,
    }


def create_grid_preview(plane_width: float, plane_height: float,
                        grid_rows: int, grid_cols: int,
                        detector_width: float = 0, detector_height: float = 0,
                        ref_pos: tuple = None) -> go.Figure:
    """Create visual preview of measurement grid with detector positions"""

    x = np.linspace(-plane_width / 2, plane_width / 2, grid_cols)
    y = np.linspace(-plane_height / 2, plane_height / 2, grid_rows)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    # Add test plane boundary
    fig.add_trace(go.Scatter(
        x=[-plane_width/2, plane_width/2, plane_width/2, -plane_width/2, -plane_width/2],
        y=[-plane_height/2, -plane_height/2, plane_height/2, plane_height/2, -plane_height/2],
        mode='lines',
        name='Test Plane',
        line=dict(color='#1E3A5F', width=3),
        fill='toself',
        fillcolor='rgba(30, 58, 95, 0.05)',
    ))

    # Add measurement points
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        name='Measurement Points',
        marker=dict(
            size=8 if detector_width <= 0 else max(6, min(15, detector_width / 3)),
            color='#6366F1',
            symbol='circle',
            line=dict(width=1, color='white'),
        ),
        hovertemplate='X: %{x:.1f}mm<br>Y: %{y:.1f}mm<extra></extra>',
    ))

    # Add detector areas if specified
    if detector_width > 0 and detector_height > 0:
        shapes = []
        for xi in x:
            for yi in y:
                shapes.append(dict(
                    type="rect",
                    x0=xi - detector_width/2,
                    y0=yi - detector_height/2,
                    x1=xi + detector_width/2,
                    y1=yi + detector_height/2,
                    line=dict(color="rgba(99, 102, 241, 0.3)", width=1),
                    fillcolor="rgba(99, 102, 241, 0.1)",
                ))
        fig.update_layout(shapes=shapes)

    # Mark reference cell position if specified
    if ref_pos:
        ref_row, ref_col = ref_pos
        if 0 <= ref_row < grid_rows and 0 <= ref_col < grid_cols:
            ref_x = x[ref_col]
            ref_y = y[ref_row]
            fig.add_trace(go.Scatter(
                x=[ref_x],
                y=[ref_y],
                mode='markers+text',
                name='Reference Cell',
                marker=dict(size=15, color='#EF4444', symbol='x',
                           line=dict(width=3, color='white')),
                text=['REF'],
                textposition='top center',
                textfont=dict(color='#EF4444', size=10, family='Arial Black'),
            ))

    fig.update_layout(
        title=dict(
            text=f"Measurement Grid Preview ({grid_cols}×{grid_rows} = {grid_cols*grid_rows} points)",
            font=dict(size=14, color="#1E3A5F")
        ),
        xaxis=dict(
            title="X Position (mm)",
            range=[-plane_width/2 - 20, plane_width/2 + 20],
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Y Position (mm)",
            range=[-plane_height/2 - 20, plane_height/2 + 20],
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=60, r=60, t=50, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
    )

    return fig


def calculate_uncertainty_metrics(Z: np.ndarray, metrics: dict, k_factor: float = 2.0) -> dict:
    """
    Calculate uncertainty metrics for uncertainty calculation purpose.

    Args:
        Z: Irradiance measurement grid
        metrics: Standard uniformity metrics
        k_factor: Coverage factor (default 2.0 for 95% confidence)

    Returns:
        Dict with uncertainty analysis results
    """
    # Type A uncertainty (statistical)
    n_points = Z.size
    type_a_std = metrics["std"] / np.sqrt(n_points)

    # Non-uniformity contribution to uncertainty
    non_uniformity_contribution = metrics["non_uniformity"] / 100 * metrics["mean"] / np.sqrt(3)

    # Combined standard uncertainty
    combined_uncertainty = np.sqrt(type_a_std**2 + non_uniformity_contribution**2)

    # Expanded uncertainty
    expanded_uncertainty = k_factor * combined_uncertainty
    expanded_uncertainty_percent = (expanded_uncertainty / metrics["mean"]) * 100

    # Relative uncertainties
    rel_type_a = (type_a_std / metrics["mean"]) * 100
    rel_non_uniformity = (non_uniformity_contribution / metrics["mean"]) * 100

    return {
        "n_points": n_points,
        "type_a_std": type_a_std,
        "type_a_relative": rel_type_a,
        "non_uniformity_contribution": non_uniformity_contribution,
        "non_uniformity_relative": rel_non_uniformity,
        "combined_uncertainty": combined_uncertainty,
        "combined_relative": (combined_uncertainty / metrics["mean"]) * 100,
        "k_factor": k_factor,
        "expanded_uncertainty": expanded_uncertainty,
        "expanded_uncertainty_percent": expanded_uncertainty_percent,
    }


def find_best_reference_positions(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                   metrics: dict, n_positions: int = 5) -> list:
    """
    Find the best positions for reference cell placement.

    Criteria:
    - Closest to mean irradiance
    - Minimal local variation
    - Preference for central positions

    Returns list of best positions with their characteristics
    """
    rows, cols = Z.shape
    positions = []

    # Calculate local variation for each position
    for i in range(rows):
        for j in range(cols):
            # Distance from mean
            deviation_from_mean = abs(Z[i, j] - metrics["mean"])
            deviation_percent = (deviation_from_mean / metrics["mean"]) * 100

            # Local variation (3x3 neighborhood)
            i_start = max(0, i - 1)
            i_end = min(rows, i + 2)
            j_start = max(0, j - 1)
            j_end = min(cols, j + 2)
            local_std = np.std(Z[i_start:i_end, j_start:j_end])
            local_variation = (local_std / metrics["mean"]) * 100

            # Distance from center (normalized)
            center_distance = np.sqrt((X[i, j])**2 + (Y[i, j])**2)
            max_distance = np.sqrt((X[0, 0])**2 + (Y[0, 0])**2)
            center_factor = center_distance / max_distance if max_distance > 0 else 0

            # Composite score (lower is better)
            # Weight: 50% deviation from mean, 30% local variation, 20% distance from center
            score = 0.5 * deviation_percent + 0.3 * local_variation + 0.2 * (center_factor * 5)

            correction_factor = metrics["mean"] / Z[i, j] if Z[i, j] > 0 else 1.0

            positions.append({
                "row": i,
                "col": j,
                "x_mm": X[i, j],
                "y_mm": Y[i, j],
                "irradiance": Z[i, j],
                "deviation_percent": deviation_percent,
                "local_variation": local_variation,
                "center_factor": center_factor,
                "score": score,
                "correction_factor": correction_factor,
            })

    # Sort by score and return top N
    positions.sort(key=lambda p: p["score"])
    return positions[:n_positions]


def calculate_uniformity_grade(non_uniformity_percent: float) -> ClassificationGrade:
    """Calculate grade based on non-uniformity percentage per IEC 60904-9"""
    # Note: The user requested ±2%, ±5%, ±10%, ±15% but IEC 60904-9 uses 1%, 2%, 5%, 10%
    # Using standard IEC thresholds
    if non_uniformity_percent <= 1.0:
        return ClassificationGrade.A_PLUS
    elif non_uniformity_percent <= 2.0:
        return ClassificationGrade.A
    elif non_uniformity_percent <= 5.0:
        return ClassificationGrade.B
    elif non_uniformity_percent <= 10.0:
        return ClassificationGrade.C
    else:
        return ClassificationGrade.FAIL


def generate_sample_uniformity_data(
    grid_size: int = 11,
    plane_width: float = 200.0,
    plane_height: float = 200.0,
    quality: str = "A+"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample uniformity measurement data"""

    # Quality settings - controls the irradiance variation
    quality_variation = {
        "A+": 0.003,   # ~0.6% non-uniformity
        "A": 0.008,    # ~1.5% non-uniformity
        "B": 0.020,    # ~4% non-uniformity
        "C": 0.040,    # ~8% non-uniformity
    }

    variation = quality_variation.get(quality, 0.003)
    np.random.seed(42)

    # Target irradiance (1000 W/m² = 1 sun)
    target_irradiance = 1000.0

    # Generate grid positions
    x = np.linspace(-plane_width / 2, plane_width / 2, grid_size)
    y = np.linspace(-plane_height / 2, plane_height / 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create irradiance grid with edge falloff and noise
    Z = np.zeros_like(X)
    max_distance = np.sqrt((plane_width/2)**2 + (plane_height/2)**2)

    for i in range(grid_size):
        for j in range(grid_size):
            distance = np.sqrt(X[i, j]**2 + Y[i, j]**2)
            edge_effect = 1 - 0.008 * (distance / max_distance)
            noise = np.random.uniform(-variation, variation)
            Z[i, j] = target_irradiance * edge_effect * (1 + noise)

    return X, Y, Z


def parse_uploaded_data(uploaded_file, grid_size: int, plane_width: float, plane_height: float) -> tuple:
    """Parse uploaded CSV/Excel file into grid data"""
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel file."

        # Check for required columns or grid format
        if 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower() and 'irradiance' in df.columns.str.lower():
            # XYZ format: columns for x, y, irradiance
            df.columns = df.columns.str.lower()
            x_vals = df['x'].values
            y_vals = df['y'].values
            z_vals = df['irradiance'].values

            # Create grid from scattered data
            x_unique = np.sort(np.unique(x_vals))
            y_unique = np.sort(np.unique(y_vals))

            X, Y = np.meshgrid(x_unique, y_unique)
            Z = np.zeros_like(X)

            for idx in range(len(x_vals)):
                i = np.where(y_unique == y_vals[idx])[0][0]
                j = np.where(x_unique == x_vals[idx])[0][0]
                Z[i, j] = z_vals[idx]

            return (X, Y, Z), None

        else:
            # Grid format: assume data is a matrix of irradiance values
            # First column might be row labels, first row might be column labels
            if df.iloc[0, 0] == '' or pd.isna(df.iloc[0, 0]):
                # Has headers
                Z = df.iloc[1:, 1:].values.astype(float)
            else:
                # Pure data matrix
                Z = df.values.astype(float)

            rows, cols = Z.shape
            x = np.linspace(-plane_width / 2, plane_width / 2, cols)
            y = np.linspace(-plane_height / 2, plane_height / 2, rows)
            X, Y = np.meshgrid(x, y)

            return (X, Y, Z), None

    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def calculate_position_averaged_irradiance(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                           cell_width: float, cell_height: float) -> np.ndarray:
    """
    Calculate position-averaged irradiance accounting for detector cell size.
    Uses weighted averaging based on cell coverage at each measurement position.
    """
    if cell_width <= 0 or cell_height <= 0:
        return Z

    rows, cols = Z.shape
    Z_averaged = np.zeros_like(Z)

    # Calculate grid spacing
    dx = (X[0, 1] - X[0, 0]) if cols > 1 else cell_width
    dy = (Y[1, 0] - Y[0, 0]) if rows > 1 else cell_height

    # Simple averaging using neighboring cells if cell is larger than grid spacing
    cells_x = max(1, int(np.ceil(cell_width / dx)))
    cells_y = max(1, int(np.ceil(cell_height / dy)))

    for i in range(rows):
        for j in range(cols):
            # Define averaging window
            i_start = max(0, i - cells_y // 2)
            i_end = min(rows, i + cells_y // 2 + 1)
            j_start = max(0, j - cells_x // 2)
            j_end = min(cols, j + cells_x // 2 + 1)

            # Calculate weighted average
            Z_averaged[i, j] = np.mean(Z[i_start:i_end, j_start:j_end])

    return Z_averaged


def calculate_uniformity_metrics(Z: np.ndarray) -> dict:
    """Calculate all uniformity metrics from irradiance grid"""
    E_min = np.min(Z)
    E_max = np.max(Z)
    E_mean = np.mean(Z)
    E_std = np.std(Z)
    E_median = np.median(Z)

    # IEC 60904-9 Non-Uniformity formula
    non_uniformity = ((E_max - E_min) / (E_max + E_min)) * 100

    # Calculate grade
    grade = calculate_uniformity_grade(non_uniformity)

    # Calculate row and column statistics
    row_means = np.mean(Z, axis=1)
    row_stds = np.std(Z, axis=1)
    col_means = np.mean(Z, axis=0)
    col_stds = np.std(Z, axis=0)

    # Coefficient of variation
    cv = (E_std / E_mean) * 100 if E_mean > 0 else 0

    return {
        "min": E_min,
        "max": E_max,
        "mean": E_mean,
        "std": E_std,
        "median": E_median,
        "range": E_max - E_min,
        "non_uniformity": non_uniformity,
        "grade": grade,
        "cv": cv,
        "row_means": row_means,
        "row_stds": row_stds,
        "col_means": col_means,
        "col_stds": col_stds,
    }


def calculate_reference_cell_correction(Z: np.ndarray, ref_row: int, ref_col: int) -> dict:
    """Calculate reference cell position correction factor"""
    grid_average = np.mean(Z)
    ref_cell_value = Z[ref_row, ref_col]
    correction_factor = grid_average / ref_cell_value if ref_cell_value > 0 else 1.0

    return {
        "grid_average": grid_average,
        "ref_cell_value": ref_cell_value,
        "correction_factor": correction_factor,
        "deviation_percent": ((ref_cell_value - grid_average) / grid_average) * 100 if grid_average > 0 else 0,
    }


def create_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, metrics: dict,
                   ref_pos: tuple = None, show_values: bool = False) -> go.Figure:
    """Create irradiance heatmap visualization with annotations"""

    # Create custom hover text
    hover_text = []
    for i in range(Z.shape[0]):
        row_text = []
        for j in range(Z.shape[1]):
            deviation = ((Z[i, j] - metrics["mean"]) / metrics["mean"]) * 100
            text = f"X: {X[i, j]:.1f}mm<br>Y: {Y[i, j]:.1f}mm<br>"
            text += f"Irradiance: {Z[i, j]:.2f} W/m²<br>"
            text += f"Deviation: {deviation:+.2f}%"
            row_text.append(text)
        hover_text.append(row_text)

    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=Z,
        x=X[0, :],
        y=Y[:, 0],
        colorscale='RdYlGn',
        reversescale=False,
        zmin=metrics["min"] - 2,
        zmax=metrics["max"] + 2,
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertext=hover_text,
        hoverinfo='text',
    ))

    # Add value annotations if requested
    if show_values:
        annotations = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                annotations.append(dict(
                    x=X[i, j],
                    y=Y[i, j],
                    text=f"{Z[i, j]:.0f}",
                    showarrow=False,
                    font=dict(size=8, color='black'),
                ))
        fig.update_layout(annotations=annotations)

    # Mark reference cell position
    if ref_pos:
        ref_row, ref_col = ref_pos
        if 0 <= ref_row < Z.shape[0] and 0 <= ref_col < Z.shape[1]:
            fig.add_trace(go.Scatter(
                x=[X[ref_row, ref_col]],
                y=[Y[ref_row, ref_col]],
                mode='markers+text',
                name='Reference Cell',
                marker=dict(size=20, color='#EF4444', symbol='x',
                           line=dict(width=3, color='white')),
                text=['REF'],
                textposition='top center',
                textfont=dict(color='#EF4444', size=12, family='Arial Black'),
            ))

    # Mark min/max positions
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)

    fig.add_trace(go.Scatter(
        x=[X[min_idx]],
        y=[Y[min_idx]],
        mode='markers',
        name=f'Min: {metrics["min"]:.1f}',
        marker=dict(size=15, color='blue', symbol='triangle-down',
                   line=dict(width=2, color='white')),
    ))

    fig.add_trace(go.Scatter(
        x=[X[max_idx]],
        y=[Y[max_idx]],
        mode='markers',
        name=f'Max: {metrics["max"]:.1f}',
        marker=dict(size=15, color='red', symbol='triangle-up',
                   line=dict(width=2, color='white')),
    ))

    fig.update_layout(
        title=dict(
            text="Irradiance Distribution Heatmap",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="X Position (mm)",
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Y Position (mm)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=550,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    return fig


def create_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, metrics: dict) -> go.Figure:
    """Create 3D surface plot of irradiance distribution"""

    fig = go.Figure(data=[go.Surface(
        z=Z,
        x=X,
        y=Y,
        colorscale='RdYlGn',
        reversescale=False,
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    )])

    # Add mean plane
    fig.add_trace(go.Surface(
        z=np.full_like(Z, metrics["mean"]),
        x=X,
        y=Y,
        colorscale=[[0, 'rgba(99, 102, 241, 0.3)'], [1, 'rgba(99, 102, 241, 0.3)']],
        showscale=False,
        name='Mean Plane',
        hovertemplate='Mean: %{z:.1f} W/m²<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="3D Irradiance Surface",
            font=dict(size=16, color="#1E3A5F")
        ),
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Irradiance (W/m²)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor="white",
        height=550,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_contour_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, metrics: dict) -> go.Figure:
    """Create irradiance contour plot"""

    fig = go.Figure(data=go.Contour(
        z=Z,
        x=X[0, :],
        y=Y[:, 0],
        colorscale='RdYlGn',
        reversescale=False,
        contours=dict(
            start=metrics["min"],
            end=metrics["max"],
            size=(metrics["max"] - metrics["min"]) / 10,
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Irradiance Contour Map",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="X Position (mm)",
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Y Position (mm)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def create_histogram(Z: np.ndarray, metrics: dict) -> go.Figure:
    """Create histogram of irradiance values"""

    irradiances = Z.flatten()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=irradiances,
        nbinsx=20,
        marker=dict(
            color='#6366F1',
            line=dict(color='white', width=1)
        ),
        hovertemplate='Irradiance: %{x:.1f} W/m²<br>Count: %{y}<extra></extra>'
    ))

    # Add mean line
    fig.add_vline(
        x=metrics["mean"],
        line_dash="solid",
        line_color="#1E3A5F",
        line_width=2,
        annotation_text=f"Mean: {metrics['mean']:.1f}",
        annotation_position="top"
    )

    # Add min/max lines
    fig.add_vline(x=metrics["min"], line_dash="dot", line_color="#EF4444", line_width=1)
    fig.add_vline(x=metrics["max"], line_dash="dot", line_color="#EF4444", line_width=1)

    fig.update_layout(
        title=dict(
            text="Irradiance Distribution",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Irradiance (W/m²)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Number of Points",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=60, r=40, t=60, b=60),
        bargap=0.05
    )

    return fig


def create_row_column_chart(metrics: dict, grid_size: int) -> go.Figure:
    """Create row and column statistics chart"""

    fig = go.Figure()

    # Row means
    fig.add_trace(go.Scatter(
        y=list(range(len(metrics["row_means"]))),
        x=metrics["row_means"],
        mode='markers+lines',
        name='Row Means',
        marker=dict(size=10, color='#6366F1'),
        line=dict(color='#6366F1', width=2),
    ))

    # Column means
    fig.add_trace(go.Scatter(
        y=list(range(len(metrics["col_means"]))),
        x=metrics["col_means"],
        mode='markers+lines',
        name='Column Means',
        marker=dict(size=10, color='#F59E0B'),
        line=dict(color='#F59E0B', width=2),
    ))

    # Add overall mean line
    fig.add_vline(
        x=metrics["mean"],
        line_dash="dash",
        line_color="#1E3A5F",
        line_width=2,
        annotation_text=f"Overall Mean: {metrics['mean']:.1f}",
        annotation_position="top"
    )

    fig.update_layout(
        title=dict(
            text="Row and Column Mean Analysis",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Irradiance (W/m²)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Row/Column Index",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    return fig


def save_to_database(test_id: str, metrics: dict, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                     plane_width: float, plane_height: float, ref_correction: dict = None) -> bool:
    """Save uniformity results to PostgreSQL database"""
    if not DB_AVAILABLE:
        return False

    try:
        if not ensure_database_ready():
            return False

        engine = get_engine()
        from sqlalchemy import text

        with engine.connect() as conn:
            # Check if uniformity_results table exists, create if not
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS uniformity_results (
                    id SERIAL PRIMARY KEY,
                    test_id TEXT UNIQUE NOT NULL,
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    grid_rows INTEGER NOT NULL,
                    grid_cols INTEGER NOT NULL,
                    plane_width_mm REAL NOT NULL,
                    plane_height_mm REAL NOT NULL,
                    min_irradiance REAL NOT NULL,
                    max_irradiance REAL NOT NULL,
                    mean_irradiance REAL NOT NULL,
                    std_irradiance REAL NOT NULL,
                    non_uniformity_pct REAL NOT NULL,
                    grade TEXT NOT NULL,
                    cv_percent REAL,
                    ref_cell_correction REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))

            # Create table for measurement data
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS uniformity_measurements (
                    id SERIAL PRIMARY KEY,
                    test_id TEXT NOT NULL,
                    x_position REAL NOT NULL,
                    y_position REAL NOT NULL,
                    irradiance REAL NOT NULL,
                    row_index INTEGER NOT NULL,
                    col_index INTEGER NOT NULL,
                    FOREIGN KEY (test_id) REFERENCES uniformity_results(test_id)
                )
            """))

            # Insert summary results
            conn.execute(text("""
                INSERT INTO uniformity_results
                (test_id, grid_rows, grid_cols, plane_width_mm, plane_height_mm,
                 min_irradiance, max_irradiance, mean_irradiance, std_irradiance,
                 non_uniformity_pct, grade, cv_percent, ref_cell_correction)
                VALUES (:test_id, :grid_rows, :grid_cols, :plane_width, :plane_height,
                        :min_irr, :max_irr, :mean_irr, :std_irr, :non_unif, :grade, :cv,
                        :ref_corr)
                ON CONFLICT (test_id) DO UPDATE SET
                    grid_rows = EXCLUDED.grid_rows,
                    grid_cols = EXCLUDED.grid_cols,
                    min_irradiance = EXCLUDED.min_irradiance,
                    max_irradiance = EXCLUDED.max_irradiance,
                    mean_irradiance = EXCLUDED.mean_irradiance,
                    std_irradiance = EXCLUDED.std_irradiance,
                    non_uniformity_pct = EXCLUDED.non_uniformity_pct,
                    grade = EXCLUDED.grade
            """), {
                "test_id": test_id,
                "grid_rows": Z.shape[0],
                "grid_cols": Z.shape[1],
                "plane_width": plane_width,
                "plane_height": plane_height,
                "min_irr": float(metrics["min"]),
                "max_irr": float(metrics["max"]),
                "mean_irr": float(metrics["mean"]),
                "std_irr": float(metrics["std"]),
                "non_unif": float(metrics["non_uniformity"]),
                "grade": metrics["grade"].value,
                "cv": float(metrics["cv"]),
                "ref_corr": float(ref_correction["correction_factor"]) if ref_correction else None,
            })

            # Delete existing measurements for this test
            conn.execute(text("DELETE FROM uniformity_measurements WHERE test_id = :test_id"),
                        {"test_id": test_id})

            # Insert individual measurements
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    conn.execute(text("""
                        INSERT INTO uniformity_measurements
                        (test_id, x_position, y_position, irradiance, row_index, col_index)
                        VALUES (:test_id, :x, :y, :irr, :row, :col)
                    """), {
                        "test_id": test_id,
                        "x": float(X[i, j]),
                        "y": float(Y[i, j]),
                        "irr": float(Z[i, j]),
                        "row": i,
                        "col": j,
                    })

            conn.commit()
        return True
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False


def export_to_excel(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, metrics: dict,
                    ref_correction: dict = None, test_id: str = None) -> BytesIO:
    """Export results to Excel file"""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Parameter": [
                "Test ID", "Test Date", "Grid Size", "Non-Uniformity (%)",
                "Grade", "Min Irradiance (W/m²)", "Max Irradiance (W/m²)",
                "Mean Irradiance (W/m²)", "Std Dev (W/m²)", "Range (W/m²)",
                "CV (%)", "Median (W/m²)"
            ],
            "Value": [
                test_id or "N/A",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{Z.shape[0]}x{Z.shape[1]}",
                f"{metrics['non_uniformity']:.3f}",
                metrics['grade'].value,
                f"{metrics['min']:.2f}",
                f"{metrics['max']:.2f}",
                f"{metrics['mean']:.2f}",
                f"{metrics['std']:.3f}",
                f"{metrics['range']:.2f}",
                f"{metrics['cv']:.3f}",
                f"{metrics['median']:.2f}",
            ]
        }

        if ref_correction:
            summary_data["Parameter"].extend([
                "Ref Cell Value (W/m²)", "Correction Factor", "Ref Cell Deviation (%)"
            ])
            summary_data["Value"].extend([
                f"{ref_correction['ref_cell_value']:.2f}",
                f"{ref_correction['correction_factor']:.4f}",
                f"{ref_correction['deviation_percent']:.3f}",
            ])

        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # Irradiance grid sheet
        df_grid = pd.DataFrame(Z,
                               columns=[f"Col_{j}" for j in range(Z.shape[1])],
                               index=[f"Row_{i}" for i in range(Z.shape[0])])
        df_grid.to_excel(writer, sheet_name='Irradiance Grid')

        # Position data sheet
        positions = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                deviation = ((Z[i, j] - metrics["mean"]) / metrics["mean"]) * 100
                positions.append({
                    "Row": i,
                    "Column": j,
                    "X (mm)": X[i, j],
                    "Y (mm)": Y[i, j],
                    "Irradiance (W/m²)": Z[i, j],
                    "Deviation from Mean (%)": deviation,
                })
        pd.DataFrame(positions).to_excel(writer, sheet_name='Position Data', index=False)

        # Row/Column statistics sheet
        stats_data = []
        for i in range(len(metrics["row_means"])):
            stats_data.append({
                "Index": i,
                "Row Mean (W/m²)": metrics["row_means"][i],
                "Row Std (W/m²)": metrics["row_stds"][i],
                "Col Mean (W/m²)": metrics["col_means"][i] if i < len(metrics["col_means"]) else None,
                "Col Std (W/m²)": metrics["col_stds"][i] if i < len(metrics["col_stds"]) else None,
            })
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Row-Column Stats', index=False)

        # Classification thresholds reference
        thresholds = {
            "Grade": ["A+", "A", "B", "C"],
            "Max Non-Uniformity (%)": ["≤ 1%", "≤ 2%", "≤ 5%", "≤ 10%"],
        }
        pd.DataFrame(thresholds).to_excel(writer, sheet_name='Thresholds', index=False)

    output.seek(0)
    return output


def export_to_csv(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, metrics: dict) -> str:
    """Export measurement data to CSV format"""
    rows = []
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            deviation = ((Z[i, j] - metrics["mean"]) / metrics["mean"]) * 100
            rows.append({
                "x": X[i, j],
                "y": Y[i, j],
                "irradiance": Z[i, j],
                "deviation_pct": deviation,
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main():
    """Uniformity analysis page - Enhanced v2.1"""

    # Header
    st.markdown('<h1 class="main-title">Non-Uniformity Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Spatial Irradiance Uniformity Classification per IEC 60904-9 Ed.3 | v2.1 Enhanced</p>',
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'uniformity_data' not in st.session_state:
        st.session_state.uniformity_data = None
    if 'test_id' not in st.session_state:
        st.session_state.test_id = f"UNIF-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
    if 'selected_purpose' not in st.session_state:
        st.session_state.selected_purpose = "Classification"

    # =========================================================================
    # SIDEBAR CONFIGURATION
    # =========================================================================
    with st.sidebar:
        st.markdown("## Equipment Selection")
        selected_simulator, sim_metadata = render_simulator_selector(
            key_prefix="uniformity",
            show_specs=True,
            show_custom_option=True,
            compact=False
        )

        # Save selection to database if a simulator is selected
        if selected_simulator and DB_AVAILABLE:
            insert_simulator_selection(
                simulator_id=sim_metadata.get("simulator_id", ""),
                manufacturer=selected_simulator.manufacturer_name,
                model=selected_simulator.model_name,
                lamp_type=selected_simulator.lamp_type.value,
                classification=selected_simulator.typical_classification,
                test_plane_size=selected_simulator.test_plane_size,
                irradiance_min=selected_simulator.irradiance_range.min_wm2,
                irradiance_max=selected_simulator.irradiance_range.max_wm2,
                illumination_mode=selected_simulator.illumination_mode.value,
                is_custom=sim_metadata.get("is_custom", False),
                notes=selected_simulator.notes
            )

        st.markdown("---")
        # ----- PURPOSE SELECTOR -----
        st.markdown("### Analysis Purpose")
        selected_purpose = st.selectbox(
            "Select Analysis Purpose",
            options=list(UNIFORMITY_PURPOSES.keys()),
            index=list(UNIFORMITY_PURPOSES.keys()).index(st.session_state.selected_purpose),
            format_func=lambda x: f"{UNIFORMITY_PURPOSES[x]['icon']} {x}",
            help="Different purposes optimize the analysis for specific outcomes"
        )
        st.session_state.selected_purpose = selected_purpose

        purpose_info = UNIFORMITY_PURPOSES[selected_purpose]
        st.caption(purpose_info["description"])

        st.markdown("---")

        # ----- DATA SOURCE -----
        st.markdown("### Data Source")
        data_source = st.radio(
            "Select data source",
            ["Generate Sample Data", "Upload CSV/Excel File"],
            index=0
        )

        st.markdown("---")

        # ----- TEST PLANE CONFIGURATION -----
        st.markdown("### Test Plane Configuration")

        plane_width = st.number_input(
            "Test Plane Width (mm)",
            min_value=50.0,
            max_value=2000.0,
            value=200.0,
            step=10.0,
            help="Width of the test area where measurements will be taken"
        )

        plane_height = st.number_input(
            "Test Plane Height (mm)",
            min_value=50.0,
            max_value=2000.0,
            value=200.0,
            step=10.0,
            help="Height of the test area where measurements will be taken"
        )

        st.markdown("---")

        # ----- DETECTOR SIZE CONFIGURATION -----
        st.markdown("### Detector Size Configuration")

        detector_size_option = st.selectbox(
            "Detector Size",
            options=list(DETECTOR_SIZES.keys()),
            index=1,  # Default to 20x20mm
            help="Select predefined detector size or choose Custom for manual entry"
        )

        if detector_size_option == "Custom":
            det_col1, det_col2 = st.columns(2)
            with det_col1:
                cell_width = st.number_input(
                    "Width (mm)",
                    min_value=0.0,
                    max_value=200.0,
                    value=20.0,
                    step=1.0,
                    key="custom_det_width"
                )
            with det_col2:
                cell_height = st.number_input(
                    "Height (mm)",
                    min_value=0.0,
                    max_value=200.0,
                    value=20.0,
                    step=1.0,
                    key="custom_det_height"
                )
        else:
            detector_dims = DETECTOR_SIZES[detector_size_option]
            cell_width, cell_height = detector_dims
            st.markdown(f'<span class="detector-badge">{cell_width:.0f} × {cell_height:.0f} mm</span>',
                       unsafe_allow_html=True)

        st.markdown("---")

        # ----- GRID SIZE CONFIGURATION -----
        st.markdown("### Measurement Grid Configuration")

        grid_preset = st.selectbox(
            "Grid Size",
            options=list(GRID_PRESETS.keys()),
            index=4,  # Default to 11x11
            help="Select predefined grid or choose Custom for NxM configuration"
        )

        if grid_preset == "Custom":
            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                grid_rows = st.number_input(
                    "Rows (N)",
                    min_value=3,
                    max_value=51,
                    value=11,
                    step=2,
                    key="custom_grid_rows"
                )
            with grid_col2:
                grid_cols = st.number_input(
                    "Columns (M)",
                    min_value=3,
                    max_value=51,
                    value=11,
                    step=2,
                    key="custom_grid_cols"
                )
        else:
            grid_dims = GRID_PRESETS[grid_preset]
            grid_rows, grid_cols = grid_dims

        total_points = grid_rows * grid_cols
        st.info(f"**{grid_rows}×{grid_cols} = {total_points} points**")

        # Show optimal grid recommendation
        opt_rows, opt_cols, opt_info = calculate_optimal_grid_size(
            plane_width, plane_height, cell_width, cell_height
        )
        if (opt_rows, opt_cols) != (grid_rows, grid_cols):
            st.caption(f"💡 Recommended: {opt_cols}×{opt_rows} ({opt_info})")

        if data_source == "Generate Sample Data":
            st.markdown("---")
            st.markdown("### Simulation Settings")
            quality_sim = st.select_slider(
                "Simulation Quality",
                options=["C", "B", "A", "A+"],
                value="A+"
            )
            grid_size = max(grid_rows, grid_cols)  # For sample generation

        st.markdown("---")

        # ----- REFERENCE CELL SETTINGS -----
        st.markdown("### Reference Cell Settings")
        use_ref_cell = st.checkbox("Enable Reference Cell Analysis", value=True)

    # =========================================================================
    # MAIN CONTENT AREA
    # =========================================================================

    # ----- PURPOSE INFORMATION SECTION -----
    st.markdown('<div class="section-header">Analysis Configuration</div>', unsafe_allow_html=True)

    config_col1, config_col2 = st.columns([2, 1])

    with config_col1:
        purpose_info = UNIFORMITY_PURPOSES[selected_purpose]
        st.markdown(f"""
        <div class="config-section">
            <h4>{purpose_info['icon']} {selected_purpose}</h4>
            <p><strong>Purpose:</strong> {purpose_info['description']}</p>
            <p><strong>Requirements:</strong> {purpose_info['requirements']}</p>
            <p><strong>Formula:</strong> <code>{purpose_info['formula']}</code></p>
            <p><strong>Focus:</strong> {purpose_info['output_focus']}</p>
        </div>
        """, unsafe_allow_html=True)

    with config_col2:
        # Configuration summary
        st.markdown(f"""
        <div class="config-section">
            <h4>📐 Configuration Summary</h4>
            <p><strong>Test Plane:</strong> {plane_width:.0f} × {plane_height:.0f} mm</p>
            <p><strong>Detector:</strong> {cell_width:.0f} × {cell_height:.0f} mm</p>
            <p><strong>Grid:</strong> {grid_rows} × {grid_cols} ({grid_rows * grid_cols} points)</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ----- DATA INPUT SECTION -----
    if data_source == "Upload CSV/Excel File":
        st.markdown('<div class="section-header">Data Input</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>File Upload Instructions:</strong><br>
            Upload a CSV or Excel file with uniformity measurement data. Supported formats:<br>
            <ul>
                <li><b>XYZ Format:</b> Columns named 'x', 'y', 'irradiance'</li>
                <li><b>Grid Format:</b> Matrix of irradiance values (rows × columns)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload measurement data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with uniformity measurements"
        )

        if uploaded_file is not None:
            grid_size_upload = st.number_input("Grid size (if grid format)", min_value=3, max_value=50, value=11)
            data, error = parse_uploaded_data(uploaded_file, grid_size_upload, plane_width, plane_height)

            if error:
                st.error(error)
                return

            X, Y, Z = data
            # Update grid dimensions from loaded data
            grid_rows, grid_cols = Z.shape
            st.success(f"Data loaded successfully! Grid size: {Z.shape[0]}x{Z.shape[1]}")
        else:
            # Show grid preview even without data
            st.markdown('<div class="section-header">Grid Preview</div>', unsafe_allow_html=True)
            preview_fig = create_grid_preview(plane_width, plane_height, grid_rows, grid_cols,
                                             cell_width, cell_height)
            st.plotly_chart(preview_fig, use_container_width=True)

            # Show position analysis
            pos_info = calculate_measurement_positions(plane_width, plane_height, grid_rows, grid_cols,
                                                       cell_width, cell_height)
            for rec in pos_info["recommendations"]:
                if rec.startswith("✓"):
                    st.success(rec)
                elif rec.startswith("⚠️"):
                    st.warning(rec)
                else:
                    st.info(rec)

            st.info("Please upload a measurement file to proceed with analysis.")
            return

    else:
        # Generate sample data with the configured grid
        X, Y, Z = generate_sample_uniformity_data(max(grid_rows, grid_cols), plane_width, plane_height, quality_sim)
        # Resize to match configured grid if different
        if Z.shape[0] != grid_rows or Z.shape[1] != grid_cols:
            from scipy.ndimage import zoom
            zoom_factors = (grid_rows / Z.shape[0], grid_cols / Z.shape[1])
            Z = zoom(Z, zoom_factors, order=1)
            x = np.linspace(-plane_width / 2, plane_width / 2, grid_cols)
            y = np.linspace(-plane_height / 2, plane_height / 2, grid_rows)
            X, Y = np.meshgrid(x, y)

    # Apply position averaging if detector cell size is specified
    if cell_width > 0 or cell_height > 0:
        Z = calculate_position_averaged_irradiance(Z, X, Y, cell_width, cell_height)
        st.markdown(f"""
        <div class="warning-box">
            <strong>Position Averaging Applied:</strong> Detector cell size {cell_width:.1f}x{cell_height:.1f} mm
        </div>
        """, unsafe_allow_html=True)

    # Calculate metrics
    metrics = calculate_uniformity_metrics(Z)

    # Reference cell configuration
    ref_correction = None
    ref_pos = None
    if use_ref_cell:
        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            ref_row = st.number_input("Reference Cell Row", min_value=0, max_value=Z.shape[0]-1,
                                      value=Z.shape[0]//2)
        with ref_col2:
            ref_col = st.number_input("Reference Cell Column", min_value=0, max_value=Z.shape[1]-1,
                                      value=Z.shape[1]//2)
        ref_pos = (ref_row, ref_col)
        ref_correction = calculate_reference_cell_correction(Z, ref_row, ref_col)

    # =========================================================================
    # GRID PREVIEW SECTION
    # =========================================================================
    st.markdown('<div class="section-header">Measurement Grid Preview</div>', unsafe_allow_html=True)

    grid_preview_col1, grid_preview_col2 = st.columns([2, 1])

    with grid_preview_col1:
        preview_fig = create_grid_preview(plane_width, plane_height, Z.shape[0], Z.shape[1],
                                         cell_width, cell_height, ref_pos)
        st.plotly_chart(preview_fig, use_container_width=True)

    with grid_preview_col2:
        # Position analysis info
        pos_info = calculate_measurement_positions(plane_width, plane_height, Z.shape[0], Z.shape[1],
                                                   cell_width, cell_height)

        st.markdown(f"""
        <div class="config-section">
            <h4>📍 Position Analysis</h4>
            <p><strong>Total Points:</strong> {pos_info['total_points']}</p>
            <p><strong>Grid Spacing:</strong> {pos_info['spacing'][0]:.1f} × {pos_info['spacing'][1]:.1f} mm</p>
            <p><strong>Coverage:</strong> {pos_info['coverage_percent']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        for rec in pos_info["recommendations"]:
            if rec.startswith("✓"):
                st.success(rec)
            elif rec.startswith("⚠️"):
                st.warning(rec)
            else:
                st.info(rec)

    st.divider()

    # =========================================================================
    # RESULTS SECTION - PURPOSE SPECIFIC
    # =========================================================================
    st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)

    # Classification result metrics (always shown)
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="color: #64748B; font-size: 0.875rem; margin-bottom: 0.5rem;">
                NON-UNIFORMITY GRADE
            </div>
            <div class="grade-badge-large {get_grade_class(metrics['grade'])}">
                {metrics['grade'].value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['non_uniformity']:.2f}%</div>
            <div class="metric-label">Non-Uniformity</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['mean']:.1f}</div>
            <div class="metric-label">Mean (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['min']:.1f}</div>
            <div class="metric-label">Min (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['max']:.1f}</div>
            <div class="metric-label">Max (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    # Formula box
    st.markdown("""
    <div class="formula-box">
        <strong>Non-Uniformity Formula (IEC 60904-9):</strong><br><br>
        Non-Uniformity (%) = ((E<sub>max</sub> - E<sub>min</sub>) / (E<sub>max</sub> + E<sub>min</sub>)) × 100
    </div>
    """, unsafe_allow_html=True)

    # Reference cell correction display
    if ref_correction:
        st.markdown(f"""
        <div class="success-box">
            <strong>Reference Cell Position Correction:</strong><br>
            Grid Average: {ref_correction['grid_average']:.2f} W/m² |
            Ref Cell Value: {ref_correction['ref_cell_value']:.2f} W/m² |
            <b>Correction Factor: {ref_correction['correction_factor']:.4f}</b> |
            Deviation: {ref_correction['deviation_percent']:+.2f}%
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # PURPOSE-SPECIFIC ANALYSIS SECTIONS
    # =========================================================================
    st.divider()

    if selected_purpose == "Reference Cell Positioning":
        # Show best reference cell positions
        st.markdown('<div class="section-header">🎯 Optimal Reference Cell Positions</div>',
                   unsafe_allow_html=True)

        best_positions = find_best_reference_positions(Z, X, Y, metrics, n_positions=5)

        st.markdown("""
        <div class="info-box">
            <strong>Position Selection Criteria:</strong><br>
            Positions are ranked based on: proximity to mean irradiance (50%),
            local stability (30%), and distance from edges (20%).
        </div>
        """, unsafe_allow_html=True)

        # Create columns for top positions
        pos_cols = st.columns(min(5, len(best_positions)))

        for idx, pos in enumerate(best_positions[:5]):
            with pos_cols[idx]:
                rank_color = "#10B981" if idx == 0 else "#6366F1" if idx < 3 else "#94A3B8"
                st.markdown(f"""
                <div class="config-section" style="text-align: center; border-left: 4px solid {rank_color};">
                    <h4>#{idx + 1}</h4>
                    <p><strong>Position:</strong><br>({pos['x_mm']:.0f}, {pos['y_mm']:.0f}) mm</p>
                    <p><strong>Grid Index:</strong><br>Row {pos['row']}, Col {pos['col']}</p>
                    <p><strong>Irradiance:</strong><br>{pos['irradiance']:.1f} W/m²</p>
                    <p><strong>Dev from Mean:</strong><br>{pos['deviation_percent']:+.2f}%</p>
                    <p><strong>Correction:</strong><br>{pos['correction_factor']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

        # Detailed position table
        with st.expander("View All Position Rankings"):
            all_positions = find_best_reference_positions(Z, X, Y, metrics, n_positions=Z.size)
            df_positions = pd.DataFrame([{
                "Rank": idx + 1,
                "Row": p["row"],
                "Col": p["col"],
                "X (mm)": f"{p['x_mm']:.1f}",
                "Y (mm)": f"{p['y_mm']:.1f}",
                "Irradiance (W/m²)": f"{p['irradiance']:.2f}",
                "Deviation (%)": f"{p['deviation_percent']:+.3f}",
                "Local Var (%)": f"{p['local_variation']:.3f}",
                "Correction Factor": f"{p['correction_factor']:.4f}",
                "Score": f"{p['score']:.3f}",
            } for idx, p in enumerate(all_positions)])
            st.dataframe(df_positions, use_container_width=True, hide_index=True, height=300)

    elif selected_purpose == "Uncertainty Calculations":
        # Show uncertainty analysis
        st.markdown('<div class="section-header">📐 Uncertainty Analysis</div>',
                   unsafe_allow_html=True)

        k_factor = st.slider("Coverage Factor (k)", min_value=1.0, max_value=3.0,
                            value=2.0, step=0.1,
                            help="k=2 for 95% confidence, k=3 for 99.7% confidence")

        uncertainty = calculate_uncertainty_metrics(Z, metrics, k_factor)

        # Uncertainty budget table
        st.markdown("### Uncertainty Budget")

        unc_col1, unc_col2 = st.columns(2)

        with unc_col1:
            st.markdown(f"""
            <div class="config-section">
                <h4>Type A Uncertainty (Statistical)</h4>
                <table class="stats-table" style="width: 100%;">
                    <tr><td>Number of Points</td><td><strong>{uncertainty['n_points']}</strong></td></tr>
                    <tr><td>Standard Uncertainty</td><td><strong>{uncertainty['type_a_std']:.4f} W/m²</strong></td></tr>
                    <tr><td>Relative Uncertainty</td><td><strong>{uncertainty['type_a_relative']:.4f}%</strong></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with unc_col2:
            st.markdown(f"""
            <div class="config-section">
                <h4>Type B Uncertainty (Non-Uniformity)</h4>
                <table class="stats-table" style="width: 100%;">
                    <tr><td>Non-Uniformity</td><td><strong>{metrics['non_uniformity']:.2f}%</strong></td></tr>
                    <tr><td>Standard Uncertainty</td><td><strong>{uncertainty['non_uniformity_contribution']:.4f} W/m²</strong></td></tr>
                    <tr><td>Relative Uncertainty</td><td><strong>{uncertainty['non_uniformity_relative']:.4f}%</strong></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Combined uncertainty
        st.markdown(f"""
        <div class="optimal-info">
            <h4>Combined Uncertainty</h4>
            <table class="stats-table" style="width: 100%;">
                <tr>
                    <td>Combined Standard Uncertainty (u<sub>c</sub>)</td>
                    <td><strong>{uncertainty['combined_uncertainty']:.4f} W/m²</strong></td>
                    <td><strong>{uncertainty['combined_relative']:.4f}%</strong></td>
                </tr>
                <tr>
                    <td>Expanded Uncertainty (U, k={k_factor:.1f})</td>
                    <td><strong>{uncertainty['expanded_uncertainty']:.4f} W/m²</strong></td>
                    <td><strong>{uncertainty['expanded_uncertainty_percent']:.4f}%</strong></td>
                </tr>
            </table>
            <br>
            <p><strong>Result:</strong> Irradiance = {metrics['mean']:.2f} ± {uncertainty['expanded_uncertainty']:.2f} W/m²
            (k={k_factor:.1f}, ~{int(95 if k_factor == 2 else 99.7 if k_factor == 3 else 68)}% confidence)</p>
        </div>
        """, unsafe_allow_html=True)

    else:  # Classification purpose (default)
        pass  # The standard classification display is shown above

    st.divider()

    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Heatmap", "3D Surface", "Contour Map", "Distribution", "Position Analysis"
    ])

    with tab1:
        show_values = st.checkbox("Show irradiance values on heatmap", value=False)
        fig_heatmap = create_heatmap(X, Y, Z, metrics, ref_pos, show_values)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab2:
        fig_3d = create_3d_surface(X, Y, Z, metrics)
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        fig_contour = create_contour_plot(X, Y, Z, metrics)
        st.plotly_chart(fig_contour, use_container_width=True)

    with tab4:
        fig_hist = create_histogram(Z, metrics)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Additional statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Standard Deviation", f"{metrics['std']:.2f} W/m²")
        with stat_col2:
            st.metric("Coefficient of Variation", f"{metrics['cv']:.3f}%")
        with stat_col3:
            st.metric("Range", f"{metrics['range']:.2f} W/m²")
        with stat_col4:
            st.metric("Median", f"{metrics['median']:.1f} W/m²")

    with tab5:
        st.markdown("### Row/Column Statistics Analysis")
        fig_rc = create_row_column_chart(metrics, Z.shape[0])
        st.plotly_chart(fig_rc, use_container_width=True)

        # Detailed position analysis table
        st.markdown("### Detailed Position Analysis")

        # Create position data table
        position_data = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                deviation = ((Z[i, j] - metrics["mean"]) / metrics["mean"]) * 100
                position_data.append({
                    "Row": i,
                    "Col": j,
                    "X (mm)": f"{X[i, j]:.1f}",
                    "Y (mm)": f"{Y[i, j]:.1f}",
                    "Irradiance (W/m²)": f"{Z[i, j]:.2f}",
                    "Dev from Mean (%)": f"{deviation:+.2f}",
                })

        df_positions = pd.DataFrame(position_data)
        st.dataframe(df_positions, use_container_width=True, hide_index=True, height=400)

    st.divider()

    # Classification thresholds reference
    st.markdown("### Classification Thresholds (IEC 60904-9 Ed.3)")

    thresh_col1, thresh_col2 = st.columns(2)

    with thresh_col1:
        st.markdown("""
        | Grade | Maximum Non-Uniformity |
        |-------|----------------------|
        | **A+** | ≤ 1% |
        | **A** | ≤ 2% |
        | **B** | ≤ 5% |
        | **C** | ≤ 10% |
        """)

    with thresh_col2:
        st.markdown(f"""
        **Current Measurement Summary:**
        - Test Plane: {plane_width:.0f} × {plane_height:.0f} mm
        - Grid: {Z.shape[0]} × {Z.shape[1]} ({Z.shape[0] * Z.shape[1]} points)
        - Non-Uniformity: **{metrics['non_uniformity']:.2f}%**
        - Classification: **{metrics['grade'].value}**
        """)

    st.divider()

    # Export and Save section
    st.markdown("### Export & Save Results")

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        excel_data = export_to_excel(X, Y, Z, metrics, ref_correction, st.session_state.test_id)
        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=f"uniformity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with export_col2:
        csv_data = export_to_csv(X, Y, Z, metrics)
        st.download_button(
            label="Download CSV Data",
            data=csv_data,
            file_name=f"uniformity_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col3:
        if DB_AVAILABLE:
            if st.button("Save to Database", use_container_width=True, type="primary"):
                if save_to_database(st.session_state.test_id, metrics, X, Y, Z,
                                   plane_width, plane_height, ref_correction):
                    st.success(f"Results saved! Test ID: {st.session_state.test_id}")
                    # Generate new test ID for next save
                    st.session_state.test_id = f"UNIF-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
                else:
                    st.error("Failed to save results to database.")
        else:
            st.info("Database not configured")

    # Raw data expander
    with st.expander("View Raw Measurement Data"):
        st.markdown(f"**Test ID:** {st.session_state.test_id}")
        df_raw = pd.DataFrame(Z,
                             columns=[f"Col {j}" for j in range(Z.shape[1])],
                             index=[f"Row {i}" for i in range(Z.shape[0])])
        st.dataframe(df_raw, use_container_width=True)


if __name__ == "__main__":
    main()
