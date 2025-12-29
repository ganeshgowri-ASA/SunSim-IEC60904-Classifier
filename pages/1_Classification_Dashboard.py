"""
IEC 60904-9 Ed.3 Classification Dashboard
==========================================

Comprehensive solar simulator classification dashboard with:
- Spectral Match analysis with 6 wavelength bands (AM1.5G reference)
- SPD (Spectral Power Distribution) integrated calculations
- SPC (Statistical Process Control) analysis
- Non-Uniformity heatmap with reference cell position marking
- Temporal Stability (STI/LTI) time-series analysis
- Overall classification summary visualization

Follows IEC 60904-9:2020 (Edition 3) standard requirements.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from io import BytesIO

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    WAVELENGTH_INTERVALS_ED2,
    get_grade_color,
    get_grade_description,
)

# Page configuration
st.set_page_config(
    page_title="IEC 60904-9 Classification | SunSim",
    page_icon=":material/dashboard:",
    layout="wide",
)

# IEC 60904-9 Ed.3 Wavelength Bands (6 intervals per Ed.2/Ed.3 simplified)
WAVELENGTH_BANDS = [
    {"range": "400-500nm", "start": 400, "end": 500, "color": "#8B5CF6", "name": "UV-Violet", "am15g_fraction": 18.4},
    {"range": "500-600nm", "start": 500, "end": 600, "color": "#22C55E", "name": "Green", "am15g_fraction": 19.9},
    {"range": "600-700nm", "start": 600, "end": 700, "color": "#EAB308", "name": "Yellow-Orange", "am15g_fraction": 18.4},
    {"range": "700-800nm", "start": 700, "end": 800, "color": "#F97316", "name": "Red", "am15g_fraction": 14.9},
    {"range": "800-900nm", "start": 800, "end": 900, "color": "#EF4444", "name": "Near-IR 1", "am15g_fraction": 12.5},
    {"range": "900-1100nm", "start": 900, "end": 1100, "color": "#991B1B", "name": "Near-IR 2", "am15g_fraction": 15.9},
]

# Custom CSS
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

    .overall-classification-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D4A6F 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 58, 95, 0.3);
    }

    .overall-title {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .overall-grade {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: 4px;
        margin: 1rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .grade-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        height: 100%;
    }

    .grade-card-title {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 16px;
        min-width: 100px;
        margin-bottom: 1rem;
    }

    .grade-a-plus { background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; }
    .grade-a { background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); color: white; }
    .grade-b { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: white; }
    .grade-c { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; }

    .grade-value {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.5rem;
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
        font-size: 1.75rem;
        font-weight: 700;
        color: #1E3A5F;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .spc-card {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .ref-cell-marker {
        background: #EF4444;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def calculate_spd_metrics(spectral_data: list) -> dict:
    """
    Calculate SPD (Spectral Power Distribution) metrics per IEC 60904-9

    Uses trapezoidal integration for spectral irradiance calculations.
    Note: numpy.trapezoid is used (numpy 2.0+ compatible, replaces deprecated trapz)
    """
    # Extract wavelength midpoints and measured fractions
    wavelengths = [(d["start"] + d["end"]) / 2 for d in spectral_data]
    measured_fractions = [d["measured"] for d in spectral_data]
    reference_fractions = [d["reference"] for d in spectral_data]

    # Calculate integrated spectral irradiance using trapezoidal rule
    # Use numpy.trapezoid (numpy 2.0+) with fallback to trapz for older versions
    try:
        total_measured = np.trapezoid(measured_fractions, wavelengths)
        total_reference = np.trapezoid(reference_fractions, wavelengths)
    except AttributeError:
        # Fallback for numpy < 2.0
        total_measured = np.trapz(measured_fractions, wavelengths)
        total_reference = np.trapz(reference_fractions, wavelengths)

    # Calculate spectral mismatch factor (M)
    spectral_mismatch = total_measured / total_reference if total_reference > 0 else 1.0

    # Calculate weighted spectral deviation
    ratios = [d["ratio"] for d in spectral_data]
    weighted_deviation = np.sqrt(np.mean([(r - 1)**2 for r in ratios])) * 100

    return {
        "total_measured_irradiance": total_measured,
        "total_reference_irradiance": total_reference,
        "spectral_mismatch_factor": spectral_mismatch,
        "weighted_deviation_percent": weighted_deviation,
        "mean_ratio": np.mean(ratios),
        "std_ratio": np.std(ratios),
    }


def calculate_spc_metrics(uniformity_grid: np.ndarray, temporal_data: dict) -> dict:
    """
    Calculate SPC (Statistical Process Control) metrics

    Includes control limits, capability indices, and process stability indicators.
    """
    # Uniformity SPC
    flat_grid = uniformity_grid.flatten()
    u_mean = np.mean(flat_grid)
    u_std = np.std(flat_grid)
    u_range = np.max(flat_grid) - np.min(flat_grid)

    # Control limits (3-sigma)
    ucl = u_mean + 3 * u_std  # Upper Control Limit
    lcl = u_mean - 3 * u_std  # Lower Control Limit

    # Process Capability Index (Cp) - assumes spec limits of +/- 2%
    spec_limit = u_mean * 0.02  # 2% of mean
    usl = u_mean + spec_limit  # Upper Spec Limit
    lsl = u_mean - spec_limit  # Lower Spec Limit

    cp = (usl - lsl) / (6 * u_std) if u_std > 0 else float('inf')
    cpk = min((usl - u_mean), (u_mean - lsl)) / (3 * u_std) if u_std > 0 else float('inf')

    # Temporal SPC
    sti_data = temporal_data.get("sti_data", np.array([1000]))
    t_mean = np.mean(sti_data)
    t_std = np.std(sti_data)
    t_ucl = t_mean + 3 * t_std
    t_lcl = t_mean - 3 * t_std

    # Count out-of-control points
    ooc_uniformity = np.sum((flat_grid > ucl) | (flat_grid < lcl))
    ooc_temporal = np.sum((sti_data > t_ucl) | (sti_data < t_lcl))

    return {
        "uniformity": {
            "mean": u_mean,
            "std": u_std,
            "range": u_range,
            "ucl": ucl,
            "lcl": lcl,
            "usl": usl,
            "lsl": lsl,
            "cp": cp,
            "cpk": cpk,
            "ooc_count": ooc_uniformity,
            "ooc_percent": (ooc_uniformity / len(flat_grid)) * 100,
        },
        "temporal": {
            "mean": t_mean,
            "std": t_std,
            "ucl": t_ucl,
            "lcl": t_lcl,
            "ooc_count": ooc_temporal,
            "ooc_percent": (ooc_temporal / len(sti_data)) * 100,
        }
    }


def generate_comprehensive_sample_data():
    """Generate comprehensive sample data for all classification parameters"""
    np.random.seed(42)

    # Spectral Match Data - 6 wavelength bands
    spectral_data = []
    for band in WAVELENGTH_BANDS:
        # Simulate measured fraction with slight deviation from reference
        noise = np.random.uniform(-0.08, 0.08)
        measured = band["am15g_fraction"] * (1 + noise)
        ratio = measured / band["am15g_fraction"]
        spectral_data.append({
            "band": band["range"],
            "name": band["name"],
            "color": band["color"],
            "start": band["start"],
            "end": band["end"],
            "reference": band["am15g_fraction"],
            "measured": measured,
            "ratio": ratio,
            "in_spec": 0.875 <= ratio <= 1.125
        })

    # Calculate SPD metrics
    spd_metrics = calculate_spd_metrics(spectral_data)

    # Uniformity Grid Data - 11x11 grid with reference cell position
    grid_size = 11
    plane_size = 200  # mm
    uniformity_grid = np.zeros((grid_size, grid_size))
    target_irradiance = 1000.0

    for i in range(grid_size):
        for j in range(grid_size):
            x = -plane_size/2 + i * plane_size/(grid_size-1)
            y = -plane_size/2 + j * plane_size/(grid_size-1)
            # Edge falloff effect
            distance = np.sqrt(x**2 + y**2)
            max_dist = np.sqrt(2) * plane_size/2
            edge_effect = 1 - 0.008 * (distance / max_dist)
            noise = np.random.uniform(-0.003, 0.003)
            uniformity_grid[i, j] = target_irradiance * edge_effect * (1 + noise)

    # Reference cell position (center-right area)
    ref_cell_pos = (6, 5)  # Grid indices
    ref_cell_actual = uniformity_grid[ref_cell_pos]
    ref_cell_avg = np.mean(uniformity_grid)
    correction_factor = ref_cell_avg / ref_cell_actual

    # Temporal Stability Data
    duration = 60  # seconds
    sampling_rate = 100  # Hz
    num_samples = duration * sampling_rate
    t = np.linspace(0, duration, num_samples)

    # STI data (short-term, 1 second windows)
    sti_base = target_irradiance
    sti_noise = np.random.normal(0, 2, num_samples)  # Small noise for A+ grade
    sti_drift = 1.5 * np.sin(2 * np.pi * t / 30)  # Slow drift
    sti_data = sti_base + sti_noise + sti_drift

    # LTI data (long-term, over minutes)
    lti_duration = 600  # 10 minutes
    lti_samples = 600
    lti_t = np.linspace(0, lti_duration, lti_samples)
    lti_drift = 3 * np.sin(2 * np.pi * lti_t / 300) + np.random.normal(0, 1.5, lti_samples)
    lti_data = target_irradiance + lti_drift

    temporal_data = {
        "sti_time": t,
        "sti_data": sti_data,
        "sti_value": (np.max(sti_data) - np.min(sti_data)) / (np.max(sti_data) + np.min(sti_data)) * 100,
        "lti_time": lti_t,
        "lti_data": lti_data,
        "lti_value": (np.max(lti_data) - np.min(lti_data)) / (np.max(lti_data) + np.min(lti_data)) * 100,
        "sti_grade": ClassificationGrade.A,
        "lti_grade": ClassificationGrade.A,
        "overall_grade": ClassificationGrade.A,
    }

    # Calculate SPC metrics
    spc_metrics = calculate_spc_metrics(uniformity_grid, temporal_data)

    return {
        "spectral": {
            "data": spectral_data,
            "min_ratio": min(d["ratio"] for d in spectral_data),
            "max_ratio": max(d["ratio"] for d in spectral_data),
            "grade": ClassificationGrade.A_PLUS,
            "spd_metrics": spd_metrics,
        },
        "uniformity": {
            "grid": uniformity_grid,
            "grid_size": grid_size,
            "plane_size": plane_size,
            "min": np.min(uniformity_grid),
            "max": np.max(uniformity_grid),
            "mean": np.mean(uniformity_grid),
            "non_uniformity": (np.max(uniformity_grid) - np.min(uniformity_grid)) /
                             (np.max(uniformity_grid) + np.min(uniformity_grid)) * 100,
            "ref_cell_pos": ref_cell_pos,
            "ref_cell_actual": ref_cell_actual,
            "ref_cell_avg": ref_cell_avg,
            "correction_factor": correction_factor,
            "grade": ClassificationGrade.A_PLUS,
        },
        "temporal": temporal_data,
        "spc": spc_metrics,
        "equipment": {
            "manufacturer": "SunSim Technologies",
            "model": "SS-3000 Pro",
            "serial": "SS3K-2024-0042",
            "lamp_type": "Xenon Arc",
            "lamp_hours": 245.5,
            "calibration_date": datetime.now() - timedelta(days=45),
            "next_calibration": datetime.now() + timedelta(days=320),
        },
        "test_info": {
            "test_date": datetime.now(),
            "operator": "John Smith",
            "laboratory": "PV Testing Lab - Building A",
            "certificate": "IEC-2024-SS-0042",
            "ambient_temp": 23.5,
            "humidity": 45.2,
        }
    }


def get_grade_class(grade: ClassificationGrade) -> str:
    """Get CSS class for grade badge"""
    return {
        ClassificationGrade.A_PLUS: "grade-a-plus",
        ClassificationGrade.A: "grade-a",
        ClassificationGrade.B: "grade-b",
        ClassificationGrade.C: "grade-c",
    }.get(grade, "grade-c")


def create_spectral_comparison_chart(spectral_data: list) -> go.Figure:
    """Create spectral distribution comparison chart with 6 wavelength bands"""
    fig = go.Figure()

    bands = [d["band"] for d in spectral_data]
    reference = [d["reference"] for d in spectral_data]
    measured = [d["measured"] for d in spectral_data]
    colors = [d["color"] for d in spectral_data]

    # Reference spectrum bars
    fig.add_trace(go.Bar(
        name='AM1.5G Reference',
        x=bands,
        y=reference,
        marker_color='rgba(99, 102, 241, 0.7)',
        marker_line_color='#4F46E5',
        marker_line_width=2,
        hovertemplate='<b>%{x}</b><br>Reference: %{y:.2f}%<extra></extra>'
    ))

    # Measured spectrum bars
    fig.add_trace(go.Bar(
        name='Measured Spectrum',
        x=bands,
        y=measured,
        marker_color=colors,
        marker_line_color='#1E3A5F',
        marker_line_width=2,
        hovertemplate='<b>%{x}</b><br>Measured: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Spectral Distribution: Measured vs AM1.5G Reference",
            'font': {'size': 16, 'color': "#1E3A5F"}
        },
        xaxis={'title': "Wavelength Band", 'gridcolor': "#E2E8F0"},
        yaxis={'title': "Spectral Fraction (%)", 'gridcolor': "#E2E8F0"},
        barmode='group',
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin={'l': 60, 'r': 40, 't': 60, 'b': 60},
        legend={
            'orientation': "h",
            'yanchor': "bottom",
            'y': 1.02,
            'xanchor': "center",
            'x': 0.5
        },
        hovermode="x unified"
    )

    return fig


def create_spectral_ratio_chart(spectral_data: list) -> go.Figure:
    """Create spectral match ratio chart with threshold bands"""
    fig = go.Figure()

    bands = [d["band"] for d in spectral_data]
    ratios = [d["ratio"] for d in spectral_data]

    # Threshold bands
    fig.add_hrect(y0=0.4, y1=2.0, fillcolor="rgba(239, 68, 68, 0.1)", line_width=0)
    fig.add_hrect(y0=0.6, y1=1.4, fillcolor="rgba(245, 158, 11, 0.1)", line_width=0)
    fig.add_hrect(y0=0.75, y1=1.25, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0)
    fig.add_hrect(y0=0.875, y1=1.125, fillcolor="rgba(16, 185, 129, 0.2)", line_width=0)

    # Reference line at 1.0
    fig.add_hline(y=1.0, line_dash="solid", line_color="#1E3A5F", line_width=2)

    # Threshold lines
    fig.add_hline(y=0.875, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ min", annotation_position="right")
    fig.add_hline(y=1.125, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ max", annotation_position="right")

    # Ratio markers
    marker_colors = ["#10B981" if 0.875 <= r <= 1.125 else "#EF4444" for r in ratios]
    fig.add_trace(go.Scatter(
        x=bands,
        y=ratios,
        mode='markers+lines',
        name='Spectral Match Ratio',
        marker={'size': 14, 'color': marker_colors, 'line': {'width': 2, 'color': 'white'}},
        line={'color': '#1E3A5F', 'width': 2},
        hovertemplate='<b>%{x}</b><br>Ratio: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "Spectral Match Ratio per Wavelength Band", 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "Wavelength Band", 'gridcolor': "#E2E8F0"},
        yaxis={'title': "Ratio (Measured/Reference)", 'gridcolor': "#E2E8F0", 'range': [0.6, 1.5]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin={'l': 60, 'r': 80, 't': 60, 'b': 60},
        showlegend=False,
    )

    return fig


def create_uniformity_heatmap(data: dict) -> go.Figure:
    """Create uniformity heatmap with reference cell position marked"""
    grid = data["grid"]
    grid_size = data["grid_size"]
    plane_size = data["plane_size"]
    ref_pos = data["ref_cell_pos"]

    # Create coordinate arrays
    x = np.linspace(-plane_size/2, plane_size/2, grid_size)
    y = np.linspace(-plane_size/2, plane_size/2, grid_size)

    fig = go.Figure()

    # Heatmap - using modern colorbar syntax
    fig.add_trace(go.Heatmap(
        z=grid,
        x=x,
        y=y,
        colorscale='RdYlGn',
        reversescale=False,
        zmin=data["min"] - 2,
        zmax=data["max"] + 2,
        colorbar={
            'title': {'text': 'W/m²', 'side': 'right'},
            'tickformat': '.1f'
        },
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    ))

    # Mark reference cell position
    ref_x = x[ref_pos[0]]
    ref_y = y[ref_pos[1]]

    fig.add_trace(go.Scatter(
        x=[ref_x],
        y=[ref_y],
        mode='markers+text',
        name='Reference Cell',
        marker={'size': 20, 'color': '#EF4444', 'symbol': 'x', 'line': {'width': 3, 'color': 'white'}},
        text=['REF'],
        textposition='top center',
        textfont={'color': '#EF4444', 'size': 12, 'family': 'Arial Black'},
        hovertemplate='<b>Reference Cell Position</b><br>X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: ' +
                      f'{data["ref_cell_actual"]:.1f} W/m²<extra></extra>'
    ))

    # Add grid lines
    for xi in x:
        fig.add_vline(x=xi, line_width=0.5, line_color="rgba(0,0,0,0.1)")
    for yi in y:
        fig.add_hline(y=yi, line_width=0.5, line_color="rgba(0,0,0,0.1)")

    fig.update_layout(
        title={'text': "Irradiance Uniformity Map (11x11 Grid)", 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "X Position (mm)", 'gridcolor': "#E2E8F0", 'scaleanchor': "y", 'scaleratio': 1},
        yaxis={'title': "Y Position (mm)", 'gridcolor': "#E2E8F0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin={'l': 60, 'r': 80, 't': 60, 'b': 60},
    )

    return fig


def create_reference_cell_chart(data: dict) -> go.Figure:
    """Create reference cell position visualization with correction factor"""
    fig = go.Figure()

    # Bar comparison
    categories = ['Grid Average', 'Ref Cell Actual']
    values = [data["ref_cell_avg"], data["ref_cell_actual"]]
    colors = ['#6366F1', '#EF4444']

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='outside',
        hovertemplate='%{x}: %{y:.2f} W/m²<extra></extra>'
    ))

    # Add correction factor annotation
    cf = data["correction_factor"]
    fig.add_annotation(
        x=0.5,
        y=max(values) + 5,
        text=f"Correction Factor: {cf:.4f}",
        showarrow=False,
        font={'size': 14, 'color': "#1E3A5F", 'family': "Arial Black"},
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#1E3A5F",
        borderwidth=1,
        borderpad=8
    )

    fig.update_layout(
        title={'text': "Reference Cell Position Analysis", 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "", 'gridcolor': "#E2E8F0"},
        yaxis={'title': "Irradiance (W/m²)", 'gridcolor': "#E2E8F0", 'range': [min(values) - 10, max(values) + 15]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin={'l': 60, 'r': 40, 't': 60, 'b': 40},
        showlegend=False,
    )

    return fig


def create_sti_chart(data: dict) -> go.Figure:
    """Create Short-Term Instability (STI) time series chart"""
    t = data["sti_time"]
    irr = data["sti_data"]

    # Downsample for display
    step = max(1, len(t) // 1000)
    t_ds = t[::step]
    irr_ds = irr[::step]

    fig = go.Figure()

    # Main trace
    fig.add_trace(go.Scatter(
        x=t_ds,
        y=irr_ds,
        mode='lines',
        name='Irradiance',
        line={'color': '#6366F1', 'width': 1},
        hovertemplate='Time: %{x:.2f}s<br>Irradiance: %{y:.2f} W/m²<extra></extra>'
    ))

    # Mean line
    mean_val = np.mean(irr)
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#1E3A5F", line_width=2,
                  annotation_text=f"Mean: {mean_val:.1f} W/m²", annotation_position="right")

    # Min/Max bounds
    min_val = np.min(irr)
    max_val = np.max(irr)
    fig.add_hrect(y0=min_val, y1=max_val, fillcolor="rgba(99, 102, 241, 0.1)", line_width=0)
    fig.add_hline(y=min_val, line_dash="dot", line_color="#EF4444", line_width=1)
    fig.add_hline(y=max_val, line_dash="dot", line_color="#EF4444", line_width=1)

    sti_val = data["sti_value"]
    fig.add_annotation(
        x=t_ds[-1] * 0.02,
        y=max_val + 2,
        text=f"STI: {sti_val:.3f}%",
        showarrow=False,
        font={'size': 12, 'color': "#1E3A5F"},
        bgcolor="white",
        bordercolor="#1E3A5F",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title={'text': "Short-Term Instability (STI) - 60s Measurement", 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "Time (seconds)", 'gridcolor': "#E2E8F0"},
        yaxis={'title': "Irradiance (W/m²)", 'gridcolor': "#E2E8F0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin={'l': 60, 'r': 80, 't': 60, 'b': 60},
        showlegend=False,
    )

    return fig


def create_lti_chart(data: dict) -> go.Figure:
    """Create Long-Term Instability (LTI) time series chart"""
    t = data["lti_time"] / 60  # Convert to minutes
    irr = data["lti_data"]

    fig = go.Figure()

    # Main trace
    fig.add_trace(go.Scatter(
        x=t,
        y=irr,
        mode='lines',
        name='Irradiance',
        line={'color': '#F59E0B', 'width': 2},
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
        hovertemplate='Time: %{x:.1f} min<br>Irradiance: %{y:.2f} W/m²<extra></extra>'
    ))

    # Mean line
    mean_val = np.mean(irr)
    fig.add_hline(y=mean_val, line_dash="dash", line_color="#1E3A5F", line_width=2,
                  annotation_text=f"Mean: {mean_val:.1f} W/m²", annotation_position="right")

    # Min/Max bounds
    min_val = np.min(irr)
    max_val = np.max(irr)
    fig.add_hline(y=min_val, line_dash="dot", line_color="#EF4444", line_width=1,
                  annotation_text=f"Min: {min_val:.1f}", annotation_position="left")
    fig.add_hline(y=max_val, line_dash="dot", line_color="#EF4444", line_width=1,
                  annotation_text=f"Max: {max_val:.1f}", annotation_position="left")

    lti_val = data["lti_value"]
    fig.add_annotation(
        x=t[-1] * 0.95,
        y=max_val + 2,
        text=f"LTI: {lti_val:.3f}%",
        showarrow=False,
        font={'size': 12, 'color': "#1E3A5F"},
        bgcolor="white",
        bordercolor="#1E3A5F",
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title={'text': "Long-Term Instability (LTI) - 10 Minute Measurement", 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "Time (minutes)", 'gridcolor': "#E2E8F0"},
        yaxis={'title': "Irradiance (W/m²)", 'gridcolor': "#E2E8F0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin={'l': 80, 'r': 80, 't': 60, 'b': 60},
        showlegend=False,
    )

    return fig


def create_spc_control_chart(spc_data: dict, data_type: str = "uniformity") -> go.Figure:
    """Create SPC control chart with control limits"""
    fig = go.Figure()

    if data_type == "uniformity":
        spc = spc_data["uniformity"]
        title = "Uniformity SPC Control Chart"
        y_label = "Irradiance (W/m²)"
    else:
        spc = spc_data["temporal"]
        title = "Temporal SPC Control Chart"
        y_label = "Irradiance (W/m²)"

    mean_val = spc["mean"]
    ucl = spc["ucl"]
    lcl = spc["lcl"]

    # Control limit bands
    fig.add_hrect(y0=lcl, y1=ucl, fillcolor="rgba(34, 197, 94, 0.1)", line_width=0)

    # Control lines
    fig.add_hline(y=mean_val, line_dash="solid", line_color="#1E3A5F", line_width=2,
                  annotation_text=f"Mean: {mean_val:.2f}", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dash", line_color="#EF4444", line_width=1,
                  annotation_text=f"UCL: {ucl:.2f}", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", line_width=1,
                  annotation_text=f"LCL: {lcl:.2f}", annotation_position="right")

    # Add warning limits (2-sigma)
    uwl = mean_val + 2 * spc["std"]
    lwl = mean_val - 2 * spc["std"]
    fig.add_hline(y=uwl, line_dash="dot", line_color="#F59E0B", line_width=1)
    fig.add_hline(y=lwl, line_dash="dot", line_color="#F59E0B", line_width=1)

    fig.update_layout(
        title={'text': title, 'font': {'size': 16, 'color': "#1E3A5F"}},
        xaxis={'title': "Sample", 'gridcolor': "#E2E8F0"},
        yaxis={'title': y_label, 'gridcolor': "#E2E8F0"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin={'l': 60, 'r': 100, 't': 60, 'b': 40},
        showlegend=False,
    )

    return fig


def create_classification_summary_chart(spectral_grade, uniformity_grade, temporal_grade) -> go.Figure:
    """Create overall classification summary visualization"""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.1
    )

    # Grade to numeric for gauge
    grade_values = {
        ClassificationGrade.A_PLUS: 100,
        ClassificationGrade.A: 75,
        ClassificationGrade.B: 50,
        ClassificationGrade.C: 25,
        ClassificationGrade.FAIL: 0
    }

    grades = [
        ("Spectral Match", spectral_grade),
        ("Uniformity", uniformity_grade),
        ("Temporal Stability", temporal_grade)
    ]

    for i, (name, grade) in enumerate(grades, 1):
        color = get_grade_color(grade)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=grade_values[grade],
                title={'text': f"{name}<br><b style='font-size:24px'>{grade.value}</b>",
                       'font': {'size': 14, 'color': '#64748B'}},
                number={'font': {'size': 1, 'color': 'rgba(0,0,0,0)'}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': color, 'thickness': 0.8},
                    'bgcolor': "#E2E8F0",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 25], 'color': 'rgba(239,68,68,0.2)'},
                        {'range': [25, 50], 'color': 'rgba(245,158,11,0.2)'},
                        {'range': [50, 75], 'color': 'rgba(34,197,94,0.2)'},
                        {'range': [75, 100], 'color': 'rgba(16,185,129,0.2)'},
                    ],
                }
            ),
            row=1, col=i
        )

    fig.update_layout(
        height=250,
        margin={'l': 30, 'r': 30, 't': 50, 'b': 30},
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig


# =============================================================================
# ISO 17025 PDF REPORT GENERATION
# =============================================================================

def generate_iso17025_pdf_report(data: dict) -> bytes:
    """
    Generate ISO 17025 compliant PDF classification report.

    Args:
        data: Complete classification data from generate_comprehensive_sample_data()

    Returns:
        PDF file as bytes, or None if reportlab is not available
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12
    )
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=8
    )
    normal_style = styles['Normal']

    elements = []

    # Overall classification grade
    overall_grade = (
        f"{data['spectral']['grade'].value}"
        f"{data['uniformity']['grade'].value}"
        f"{data['temporal']['overall_grade'].value}"
    )

    # Title
    elements.append(Paragraph("IEC 60904-9 Solar Simulator Classification Report", title_style))
    elements.append(Paragraph("ISO 17025 Compliant Test Report", ParagraphStyle(
        'Subtitle', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceAfter=20
    )))
    elements.append(Spacer(1, 12))

    # Report Information Table
    equip = data['equipment']
    test = data['test_info']

    info_data = [
        ['Report Date:', test['test_date'].strftime('%Y-%m-%d %H:%M:%S')],
        ['Simulator ID:', equip['serial']],
        ['Simulator Model:', f"{equip['manufacturer']} {equip['model']}"],
        ['Test Location:', test['laboratory']],
        ['Operator:', test['operator']],
        ['Certificate Number:', test['certificate']],
        ['Overall Classification:', overall_grade],
    ]

    info_table = Table(info_data, colWidths=[140, 330])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#E8F5E9')),
        ('FONTNAME', (1, -1), (1, -1), 'Helvetica-Bold'),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))

    # Classification Summary Section
    elements.append(Paragraph("Classification Summary", heading_style))

    summary_data = [
        ['Parameter', 'Measured Value', 'Grade', 'IEC 60904-9 Criteria'],
        [
            'Spectral Match',
            f"Ratio: {data['spectral']['min_ratio']:.3f} - {data['spectral']['max_ratio']:.3f}",
            data['spectral']['grade'].value,
            'A+: 0.875-1.125 | A: 0.75-1.25'
        ],
        [
            'Non-Uniformity',
            f"{data['uniformity']['non_uniformity']:.2f}%",
            data['uniformity']['grade'].value,
            'A+: ≤1% | A: ≤2% | B: ≤5%'
        ],
        [
            'Temporal Stability (STI)',
            f"{data['temporal']['sti_value']:.3f}%",
            data['temporal']['sti_grade'].value,
            'A+: ≤0.5% | A: ≤2% | B: ≤5%'
        ],
        [
            'Temporal Stability (LTI)',
            f"{data['temporal']['lti_value']:.3f}%",
            data['temporal']['lti_grade'].value,
            'A+: ≤1% | A: ≤2% | B: ≤5%'
        ],
    ]

    summary_table = Table(summary_data, colWidths=[120, 120, 60, 170])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Spectral Match Analysis Section
    elements.append(Paragraph("Spectral Match Analysis (IEC 60904-9 Ed.3)", heading_style))

    spectral_data_rows = [['Wavelength Band', 'Reference (%)', 'Measured (%)', 'Ratio', 'Status']]
    for band_data in data['spectral']['data']:
        status = 'PASS' if band_data['in_spec'] else 'FAIL'
        spectral_data_rows.append([
            band_data['band'],
            f"{band_data['reference']:.1f}%",
            f"{band_data['measured']:.1f}%",
            f"{band_data['ratio']:.3f}",
            status
        ])

    spectral_table = Table(spectral_data_rows, colWidths=[100, 80, 80, 70, 70])
    spectral_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(spectral_table)
    elements.append(Spacer(1, 15))

    # SPD Metrics
    spd = data['spectral']['spd_metrics']
    elements.append(Paragraph("SPD Integrated Metrics", subheading_style))
    spd_data = [
        ['Metric', 'Value'],
        ['Spectral Mismatch Factor (M)', f"{spd['spectral_mismatch_factor']:.4f}"],
        ['Mean Ratio', f"{spd['mean_ratio']:.4f}"],
        ['Standard Deviation', f"{spd['std_ratio']:.4f}"],
        ['Weighted Deviation', f"{spd['weighted_deviation_percent']:.2f}%"],
    ]
    spd_table = Table(spd_data, colWidths=[180, 120])
    spd_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366F1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(spd_table)
    elements.append(Spacer(1, 15))

    # Uniformity Section
    elements.append(Paragraph("Non-Uniformity Analysis", heading_style))
    unif = data['uniformity']
    unif_data = [
        ['Parameter', 'Value'],
        ['Grid Size', f"{unif['grid_size']} x {unif['grid_size']} points"],
        ['Test Plane Size', f"{unif['plane_size']} mm x {unif['plane_size']} mm"],
        ['Min Irradiance', f"{unif['min']:.1f} W/m²"],
        ['Max Irradiance', f"{unif['max']:.1f} W/m²"],
        ['Mean Irradiance', f"{unif['mean']:.1f} W/m²"],
        ['Non-Uniformity', f"{unif['non_uniformity']:.2f}%"],
        ['Reference Cell Position', f"({unif['ref_cell_pos'][0]}, {unif['ref_cell_pos'][1]})"],
        ['Correction Factor', f"{unif['correction_factor']:.4f}"],
    ]
    unif_table = Table(unif_data, colWidths=[180, 150])
    unif_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(unif_table)
    elements.append(Spacer(1, 15))

    # Temporal Stability Section
    elements.append(Paragraph("Temporal Stability Analysis", heading_style))
    temp = data['temporal']
    temp_data = [
        ['Parameter', 'Value', 'Grade'],
        ['Short-Term Instability (STI)', f"{temp['sti_value']:.3f}%", temp['sti_grade'].value],
        ['Long-Term Instability (LTI)', f"{temp['lti_value']:.3f}%", temp['lti_grade'].value],
    ]
    temp_table = Table(temp_data, colWidths=[180, 100, 60])
    temp_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(temp_table)
    elements.append(Spacer(1, 15))

    # SPC Analysis Section
    elements.append(Paragraph("Statistical Process Control (SPC) Analysis", heading_style))
    spc_u = data['spc']['uniformity']
    spc_t = data['spc']['temporal']
    spc_data = [
        ['SPC Metric', 'Uniformity', 'Temporal'],
        ['Mean', f"{spc_u['mean']:.2f} W/m²", f"{spc_t['mean']:.2f} W/m²"],
        ['Standard Deviation', f"{spc_u['std']:.4f}", f"{spc_t['std']:.4f}"],
        ['UCL (3-sigma)', f"{spc_u['ucl']:.2f}", f"{spc_t['ucl']:.2f}"],
        ['LCL (3-sigma)', f"{spc_u['lcl']:.2f}", f"{spc_t['lcl']:.2f}"],
        ['Cp Index', f"{spc_u['cp']:.3f}", 'N/A'],
        ['Cpk Index', f"{spc_u['cpk']:.3f}", 'N/A'],
        ['Out-of-Control Points', f"{spc_u['ooc_count']} ({spc_u['ooc_percent']:.1f}%)",
         f"{spc_t['ooc_count']} ({spc_t['ooc_percent']:.1f}%)"],
    ]
    spc_table = Table(spc_data, colWidths=[150, 110, 110])
    spc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F59E0B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(spc_table)
    elements.append(Spacer(1, 15))

    # Equipment Information
    elements.append(Paragraph("Equipment Information", heading_style))
    equip_data = [
        ['Parameter', 'Value'],
        ['Manufacturer', equip['manufacturer']],
        ['Model', equip['model']],
        ['Serial Number', equip['serial']],
        ['Lamp Type', equip['lamp_type']],
        ['Lamp Hours', f"{equip['lamp_hours']:.1f} hrs"],
        ['Last Calibration', equip['calibration_date'].strftime('%Y-%m-%d')],
        ['Next Calibration Due', equip['next_calibration'].strftime('%Y-%m-%d')],
    ]
    equip_table = Table(equip_data, colWidths=[150, 200])
    equip_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(equip_table)
    elements.append(Spacer(1, 15))

    # Test Conditions
    elements.append(Paragraph("Test Conditions", heading_style))
    cond_data = [
        ['Parameter', 'Value'],
        ['Ambient Temperature', f"{test['ambient_temp']:.1f} °C"],
        ['Relative Humidity', f"{test['humidity']:.1f}%"],
    ]
    cond_table = Table(cond_data, colWidths=[150, 100])
    cond_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(cond_table)
    elements.append(Spacer(1, 20))

    # ISO 17025 Compliance Section
    elements.append(Paragraph("ISO 17025 Compliance Information", heading_style))

    compliance_text = """
    This classification report was prepared in accordance with IEC 60904-9:2020 (Edition 3)
    standard requirements and ISO 17025 quality management system guidelines.

    The measurement uncertainty and traceability of calibration equipment are documented
    in the laboratory's quality management system records.
    """
    elements.append(Paragraph(compliance_text, normal_style))
    elements.append(Spacer(1, 10))

    # Signature fields
    sig_data = [
        ['Measurement Equipment Calibration Status:', '________________________'],
        ['Calibration Certificate Number:', '________________________'],
        ['Measurement Uncertainty (k=2):', '________________________'],
        ['', ''],
        ['Reviewed By:', '________________________  Date: ____________'],
        ['Approved By:', '________________________  Date: ____________'],
    ]
    sig_table = Table(sig_data, colWidths=[200, 270])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(sig_table)

    # Footer note
    elements.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey
    )
    elements.append(Paragraph(
        f"Report generated by SunSim IEC 60904-9 Classification System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def main():
    """Main dashboard page with comprehensive visualizations"""

    # Header
    st.markdown('<h1 class="main-title">IEC 60904-9 Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Solar Simulator Classification per IEC 60904-9:2020 (Edition 3)</p>',
        unsafe_allow_html=True
    )

    # Load comprehensive sample data
    data = generate_comprehensive_sample_data()

    # Overall Classification Card
    overall_grade = (
        f"{data['spectral']['grade'].value}"
        f"{data['uniformity']['grade'].value}"
        f"{data['temporal']['overall_grade'].value}"
    )

    st.markdown(f"""
    <div class="overall-classification-card">
        <div class="overall-title">Overall Classification</div>
        <div class="overall-grade">{overall_grade}</div>
        <div style="opacity: 0.8;">
            Spectral Match: {data['spectral']['grade'].value} |
            Uniformity: {data['uniformity']['grade'].value} |
            Temporal Stability: {data['temporal']['overall_grade'].value}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grade cards row
    col1, col2, col3 = st.columns(3)

    with col1:
        grade = data['spectral']['grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Spectral Match (SPD)</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                Ratio Range: {data['spectral']['min_ratio']:.3f} - {data['spectral']['max_ratio']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        grade = data['uniformity']['grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Non-Uniformity</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                Non-Uniformity: {data['uniformity']['non_uniformity']:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        grade = data['temporal']['overall_grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Temporal Stability</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                STI: {data['temporal']['sti_value']:.2f}% | LTI: {data['temporal']['lti_value']:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Tabbed Analysis Sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/ssid_chart: Spectral Match",
        ":material/grid_on: Non-Uniformity",
        ":material/timeline: Temporal Stability",
        ":material/analytics: SPC Analysis",
        ":material/check_circle: Classification Result"
    ])

    # Tab 1: Spectral Match
    with tab1:
        st.markdown("### Spectral Distribution Analysis (SPD)")
        st.markdown("""
        <div class="info-box">
            <strong>IEC 60904-9 Ed.3 Spectral Match:</strong> Comparison of simulator spectrum to AM1.5G
            reference across 6 wavelength bands (400-1100nm). Each band ratio must fall within classification thresholds.
        </div>
        """, unsafe_allow_html=True)

        # SPD Metrics
        spd = data['spectral']['spd_metrics']
        spd_col1, spd_col2, spd_col3, spd_col4 = st.columns(4)

        with spd_col1:
            st.metric("Spectral Mismatch Factor (M)", f"{spd['spectral_mismatch_factor']:.4f}")
        with spd_col2:
            st.metric("Mean Ratio", f"{spd['mean_ratio']:.4f}")
        with spd_col3:
            st.metric("Std Dev (Ratio)", f"{spd['std_ratio']:.4f}")
        with spd_col4:
            st.metric("Weighted Deviation", f"{spd['weighted_deviation_percent']:.2f}%")

        # Spectral comparison chart
        fig_spectral = create_spectral_comparison_chart(data['spectral']['data'])
        st.plotly_chart(fig_spectral, use_container_width=True)

        # Ratio analysis
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_ratio = create_spectral_ratio_chart(data['spectral']['data'])
            st.plotly_chart(fig_ratio, use_container_width=True)

        with col_b:
            st.markdown("#### Band Analysis")
            for band in data['spectral']['data']:
                status = "Pass" if band["in_spec"] else "Review"
                status_color = "#10B981" if band["in_spec"] else "#EF4444"
                st.markdown(f"""
                **{band['name']}** ({band['band']})
                - Ratio: {band['ratio']:.3f}
                - Status: <span style="color:{status_color};font-weight:600">{status}</span>
                """, unsafe_allow_html=True)

    # Tab 2: Non-Uniformity
    with tab2:
        st.markdown("### Spatial Uniformity Analysis")
        st.markdown("""
        <div class="info-box">
            <strong>IEC 60904-9 Ed.3 Non-Uniformity:</strong> Irradiance measured across 11x11 grid points.
            Non-uniformity = (E_max - E_min) / (E_max + E_min) x 100%. Reference cell position marked.
        </div>
        """, unsafe_allow_html=True)

        # Uniformity heatmap
        fig_heatmap = create_uniformity_heatmap(data['uniformity'])
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Stats and reference cell
        col_u1, col_u2, col_u3 = st.columns([1, 1, 1])

        with col_u1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{data['uniformity']['non_uniformity']:.2f}%</div>
                <div class="metric-label">Non-Uniformity</div>
            </div>
            """, unsafe_allow_html=True)

        with col_u2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{data['uniformity']['mean']:.1f}</div>
                <div class="metric-label">Mean Irradiance (W/m²)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_u3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{data['uniformity']['max'] - data['uniformity']['min']:.1f}</div>
                <div class="metric-label">Irradiance Range (W/m²)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Reference Cell Position Analysis")

        ref_col1, ref_col2 = st.columns([1, 1])
        with ref_col1:
            fig_ref = create_reference_cell_chart(data['uniformity'])
            st.plotly_chart(fig_ref, use_container_width=True)

        with ref_col2:
            st.markdown(f"""
            **Reference Cell Details:**
            - Grid Position: ({data['uniformity']['ref_cell_pos'][0]}, {data['uniformity']['ref_cell_pos'][1]})
            - Actual Reading: {data['uniformity']['ref_cell_actual']:.2f} W/m²
            - Grid Average: {data['uniformity']['ref_cell_avg']:.2f} W/m²
            - **Correction Factor: {data['uniformity']['correction_factor']:.4f}**

            The correction factor adjusts measurements taken at the reference
            cell position to represent the average irradiance across the test plane.
            """)

    # Tab 3: Temporal Stability
    with tab3:
        st.markdown("### Temporal Stability Analysis")
        st.markdown("""
        <div class="info-box">
            <strong>IEC 60904-9 Ed.3 Temporal Instability:</strong><br>
            <b>STI (Short-Term Instability):</b> Measured during I-V sweep (~1s to 60s)<br>
            <b>LTI (Long-Term Instability):</b> Measured over extended periods (minutes to hours)<br>
            Formula: Instability = (E_max - E_min) / (E_max + E_min) x 100%
        </div>
        """, unsafe_allow_html=True)

        # STI and LTI grade cards
        sti_col, lti_col = st.columns(2)

        with sti_col:
            sti_grade = data['temporal']['sti_grade']
            st.markdown(f"""
            <div class="metric-card">
                <div class="grade-badge {get_grade_class(sti_grade)}" style="font-size:1.5rem;padding:0.5rem 1rem;">
                    {sti_grade.value}
                </div>
                <div class="metric-value">{data['temporal']['sti_value']:.3f}%</div>
                <div class="metric-label">Short-Term Instability (STI)</div>
            </div>
            """, unsafe_allow_html=True)

        with lti_col:
            lti_grade = data['temporal']['lti_grade']
            st.markdown(f"""
            <div class="metric-card">
                <div class="grade-badge {get_grade_class(lti_grade)}" style="font-size:1.5rem;padding:0.5rem 1rem;">
                    {lti_grade.value}
                </div>
                <div class="metric-value">{data['temporal']['lti_value']:.3f}%</div>
                <div class="metric-label">Long-Term Instability (LTI)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # STI Chart
        fig_sti = create_sti_chart(data['temporal'])
        st.plotly_chart(fig_sti, use_container_width=True)

        # LTI Chart
        fig_lti = create_lti_chart(data['temporal'])
        st.plotly_chart(fig_lti, use_container_width=True)

    # Tab 4: SPC Analysis
    with tab4:
        st.markdown("### Statistical Process Control (SPC) Analysis")
        st.markdown("""
        <div class="spc-card">
            <strong>SPC Metrics:</strong> Control charts and capability indices for monitoring
            solar simulator performance stability. UCL/LCL = 3-sigma control limits.
        </div>
        """, unsafe_allow_html=True)

        spc = data['spc']

        # Uniformity SPC
        st.markdown("#### Uniformity SPC Metrics")
        spc_u = spc['uniformity']

        u_col1, u_col2, u_col3, u_col4 = st.columns(4)
        with u_col1:
            st.metric("Cp (Capability)", f"{spc_u['cp']:.3f}")
        with u_col2:
            st.metric("Cpk (Capability Index)", f"{spc_u['cpk']:.3f}")
        with u_col3:
            st.metric("Out-of-Control %", f"{spc_u['ooc_percent']:.2f}%")
        with u_col4:
            st.metric("Standard Deviation", f"{spc_u['std']:.3f} W/m²")

        fig_spc_u = create_spc_control_chart(spc, "uniformity")
        st.plotly_chart(fig_spc_u, use_container_width=True)

        st.markdown("---")

        # Temporal SPC
        st.markdown("#### Temporal Stability SPC Metrics")
        spc_t = spc['temporal']

        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        with t_col1:
            st.metric("Mean", f"{spc_t['mean']:.2f} W/m²")
        with t_col2:
            st.metric("Std Dev", f"{spc_t['std']:.3f} W/m²")
        with t_col3:
            st.metric("UCL (3-sigma)", f"{spc_t['ucl']:.2f} W/m²")
        with t_col4:
            st.metric("LCL (3-sigma)", f"{spc_t['lcl']:.2f} W/m²")

        fig_spc_t = create_spc_control_chart(spc, "temporal")
        st.plotly_chart(fig_spc_t, use_container_width=True)

    # Tab 5: Classification Result Summary
    with tab5:
        st.markdown("### Overall Classification Summary")

        # Summary visualization
        fig_summary = create_classification_summary_chart(
            data['spectral']['grade'],
            data['uniformity']['grade'],
            data['temporal']['overall_grade']
        )
        st.plotly_chart(fig_summary, use_container_width=True)

        # Detailed results table
        st.markdown("#### Classification Details")

        results_df = pd.DataFrame([
            {
                "Parameter": "Spectral Match",
                "Grade": data['spectral']['grade'].value,
                "Measured Value": f"{data['spectral']['min_ratio']:.3f} - {data['spectral']['max_ratio']:.3f}",
                "Threshold (A+)": "0.875 - 1.125",
                "Status": "PASS" if data['spectral']['grade'] in [ClassificationGrade.A_PLUS, ClassificationGrade.A] else "REVIEW"
            },
            {
                "Parameter": "Non-Uniformity",
                "Grade": data['uniformity']['grade'].value,
                "Measured Value": f"{data['uniformity']['non_uniformity']:.2f}%",
                "Threshold (A+)": "<= 1%",
                "Status": "PASS" if data['uniformity']['grade'] in [ClassificationGrade.A_PLUS, ClassificationGrade.A] else "REVIEW"
            },
            {
                "Parameter": "Temporal Stability (STI)",
                "Grade": data['temporal']['sti_grade'].value,
                "Measured Value": f"{data['temporal']['sti_value']:.3f}%",
                "Threshold (A+)": "<= 0.5%",
                "Status": "PASS" if data['temporal']['sti_grade'] in [ClassificationGrade.A_PLUS, ClassificationGrade.A] else "REVIEW"
            },
            {
                "Parameter": "Temporal Stability (LTI)",
                "Grade": data['temporal']['lti_grade'].value,
                "Measured Value": f"{data['temporal']['lti_value']:.3f}%",
                "Threshold (A+)": "<= 1%",
                "Status": "PASS" if data['temporal']['lti_grade'] in [ClassificationGrade.A_PLUS, ClassificationGrade.A] else "REVIEW"
            },
        ])

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Equipment and Test info
        st.divider()
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown("#### Equipment Information")
            equip = data['equipment']
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Manufacturer | {equip['manufacturer']} |
            | Model | {equip['model']} |
            | Serial Number | {equip['serial']} |
            | Lamp Type | {equip['lamp_type']} |
            | Lamp Hours | {equip['lamp_hours']:.1f} hrs |
            | Calibration Date | {equip['calibration_date'].strftime('%Y-%m-%d')} |
            """)

        with info_col2:
            st.markdown("#### Test Information")
            test = data['test_info']
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Test Date | {test['test_date'].strftime('%Y-%m-%d %H:%M')} |
            | Operator | {test['operator']} |
            | Laboratory | {test['laboratory']} |
            | Certificate No. | {test['certificate']} |
            | Ambient Temp | {test['ambient_temp']:.1f} C |
            """)

    # Thresholds Reference
    st.divider()
    with st.expander("IEC 60904-9 Ed.3 Classification Thresholds Reference"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            **Spectral Match Thresholds** (Ratio Range)

            | Grade | Min Ratio | Max Ratio |
            |-------|-----------|-----------|
            | A+ | 0.875 | 1.125 |
            | A | 0.75 | 1.25 |
            | B | 0.6 | 1.4 |
            | C | 0.4 | 2.0 |
            """)

        with col_b:
            st.markdown("""
            **Uniformity & Temporal Stability Thresholds**

            | Grade | Uniformity | STI | LTI |
            |-------|------------|-----|-----|
            | A+ | <= 1% | <= 0.5% | <= 1% |
            | A | <= 2% | <= 2% | <= 2% |
            | B | <= 5% | <= 5% | <= 5% |
            | C | <= 10% | <= 10% | <= 10% |
            """)

    # Export button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Generate report filename
        report_filename = f"IEC60904_Classification_{data['equipment']['serial']}_{datetime.now().strftime('%Y%m%d')}.pdf"

        # Generate PDF report
        pdf_bytes = generate_iso17025_pdf_report(data)

        if pdf_bytes:
            st.download_button(
                label="Download ISO 17025 Report (PDF)",
                data=pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        else:
            st.error("ReportLab library is required for PDF generation. Install with: pip install reportlab")


if __name__ == "__main__":
    main()
