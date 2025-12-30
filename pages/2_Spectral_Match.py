"""
Spectral Match Analysis Page
IEC 60904-9 Ed.3 Solar Simulator Classification

Comprehensive spectral match analysis with:
- CSV file upload for spectral data (wavelength vs irradiance)
- SPD (Spectral Power Distribution) method calculation
- SPC (Spectral Performance Coefficient) method calculation
- Classification grades (A+, A, B, C) per IEC 60904-9
- Interactive Plotly charts
- Database storage with test_id linkage
- Export functionality (CSV/Excel)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO
import json
import uuid
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    WAVELENGTH_INTERVALS_ED3,
    WAVELENGTH_INTERVALS_ED2,
    SpectralMatchData,
    SpectralMatchResult,
    get_grade_color,
    get_grade_description,
)
from utils.db import (
    get_simulator_ids,
    insert_spectral_match_result,
    get_spectral_match_results,
)

# Page configuration
st.set_page_config(
    page_title="Spectral Match Analysis | SunSim",
    page_icon=":material/ssid_chart:",
    layout="wide",
)

# IEC 60904-9 Ed.3 Wavelength Bands (8 intervals for SPD method, 300-1200nm)
# Extended spectral range per IEC 60904-9:2020 Edition 3 requirements
# AM1.5G fractions calculated from ASTM G173-03 reference spectrum
WAVELENGTH_BANDS = [
    {"range": "300-400nm", "start": 300, "end": 400, "color": "#7C3AED", "name": "UV", "am15g_fraction": 5.4},
    {"range": "400-500nm", "start": 400, "end": 500, "color": "#8B5CF6", "name": "Violet-Blue", "am15g_fraction": 17.4},
    {"range": "500-600nm", "start": 500, "end": 600, "color": "#22C55E", "name": "Green", "am15g_fraction": 18.9},
    {"range": "600-700nm", "start": 600, "end": 700, "color": "#EAB308", "name": "Yellow-Orange", "am15g_fraction": 17.4},
    {"range": "700-800nm", "start": 700, "end": 800, "color": "#F97316", "name": "Red", "am15g_fraction": 14.1},
    {"range": "800-900nm", "start": 800, "end": 900, "color": "#EF4444", "name": "Near-IR 1", "am15g_fraction": 11.8},
    {"range": "900-1100nm", "start": 900, "end": 1100, "color": "#991B1B", "name": "Near-IR 2", "am15g_fraction": 12.0},
    {"range": "1100-1200nm", "start": 1100, "end": 1200, "color": "#7F1D1D", "name": "Near-IR 3", "am15g_fraction": 3.0},
]

# Standard reference cell spectral response (normalized, for SPC method)
# Typical crystalline silicon cell response (extended to 300nm)
# SR values based on typical c-Si cell characteristics with low UV response
REFERENCE_CELL_SR = {
    300: 0.00, 320: 0.01, 340: 0.02, 360: 0.04, 380: 0.08,
    400: 0.25, 450: 0.45, 500: 0.62, 550: 0.72,
    600: 0.78, 650: 0.82, 700: 0.85, 750: 0.87, 800: 0.88,
    850: 0.86, 900: 0.80, 950: 0.65, 1000: 0.45, 1050: 0.25,
    1100: 0.10, 1150: 0.02, 1200: 0.00
}

# AM1.5G Reference Spectrum (W/m2/nm) - key wavelengths
AM15G_REFERENCE = {
    300: 0.0341, 350: 0.4891, 400: 1.1141, 450: 1.7491, 500: 1.8341,
    550: 1.6891, 600: 1.5391, 650: 1.4241, 700: 1.2891, 750: 1.1591,
    800: 1.0491, 850: 0.9141, 900: 0.8391, 950: 0.7091, 1000: 0.6391,
    1050: 0.5541, 1100: 0.4891, 1150: 0.3991, 1200: 0.3441
}

# Custom CSS matching Classification Dashboard
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

    .upload-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 2px dashed #CBD5E1;
        text-align: center;
        margin: 1rem 0;
    }

    .method-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
    }

    .method-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }

    .threshold-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .in-spec { background: #10B981; }
    .out-of-spec { background: #EF4444; }
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


def parse_spectral_csv(uploaded_file) -> pd.DataFrame:
    """
    Parse uploaded CSV file with spectral data.

    Expected format:
    - Column 1: Wavelength (nm)
    - Column 2: Irradiance (W/m2/nm or relative units)
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Try to identify wavelength and irradiance columns
        col_names = df.columns.str.lower()

        wavelength_col = None
        irradiance_col = None

        for col in df.columns:
            col_lower = col.lower()
            if any(w in col_lower for w in ['wavelength', 'wave', 'nm', 'lambda']):
                wavelength_col = col
            elif any(i in col_lower for i in ['irradiance', 'irrad', 'intensity', 'power', 'spectral']):
                irradiance_col = col

        # If not found by name, use first two columns
        if wavelength_col is None:
            wavelength_col = df.columns[0]
        if irradiance_col is None:
            irradiance_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        result = pd.DataFrame({
            'wavelength': pd.to_numeric(df[wavelength_col], errors='coerce'),
            'irradiance': pd.to_numeric(df[irradiance_col], errors='coerce')
        })

        # Remove NaN values
        result = result.dropna()

        # Sort by wavelength
        result = result.sort_values('wavelength').reset_index(drop=True)

        return result

    except Exception as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        return pd.DataFrame()


def interpolate_to_wavelengths(df: pd.DataFrame, target_wavelengths: list) -> dict:
    """Interpolate spectral data to specific wavelengths"""
    if df.empty:
        return {}

    result = {}
    for wl in target_wavelengths:
        if wl < df['wavelength'].min() or wl > df['wavelength'].max():
            result[wl] = 0.0
        else:
            result[wl] = np.interp(wl, df['wavelength'], df['irradiance'])

    return result


def calculate_spd_method(df: pd.DataFrame) -> dict:
    """
    Calculate spectral match using SPD (Spectral Power Distribution) method.

    Per IEC 60904-9, calculates the ratio of measured to reference spectral
    irradiance fraction in each wavelength interval.
    """
    if df.empty:
        return None

    # Calculate total integrated irradiance using trapezoidal rule
    try:
        total_measured = np.trapezoid(df['irradiance'], df['wavelength'])
    except AttributeError:
        total_measured = np.trapz(df['irradiance'], df['wavelength'])

    results = []

    for band in WAVELENGTH_BANDS:
        start, end = band['start'], band['end']

        # Filter data for this wavelength band
        band_data = df[(df['wavelength'] >= start) & (df['wavelength'] <= end)]

        if len(band_data) < 2:
            # Interpolate if no data points in range
            wl_range = np.linspace(start, end, 20)
            irr_interp = np.interp(wl_range, df['wavelength'], df['irradiance'])
            try:
                band_integral = np.trapezoid(irr_interp, wl_range)
            except AttributeError:
                band_integral = np.trapz(irr_interp, wl_range)
        else:
            try:
                band_integral = np.trapezoid(band_data['irradiance'], band_data['wavelength'])
            except AttributeError:
                band_integral = np.trapz(band_data['irradiance'], band_data['wavelength'])

        # Calculate measured fraction
        measured_fraction = (band_integral / total_measured) * 100 if total_measured > 0 else 0

        # Reference fraction from AM1.5G
        reference_fraction = band['am15g_fraction']

        # Calculate ratio
        ratio = measured_fraction / reference_fraction if reference_fraction > 0 else 0

        results.append({
            'band': band['range'],
            'name': band['name'],
            'color': band['color'],
            'start': start,
            'end': end,
            'reference': reference_fraction,
            'measured': measured_fraction,
            'ratio': ratio,
            'in_spec_aplus': 0.875 <= ratio <= 1.125,
            'in_spec_a': 0.75 <= ratio <= 1.25,
            'in_spec_b': 0.6 <= ratio <= 1.4,
            'in_spec_c': 0.4 <= ratio <= 2.0,
        })

    # Determine overall grade
    ratios = [r['ratio'] for r in results]
    min_ratio = min(ratios)
    max_ratio = max(ratios)

    if all(r['in_spec_aplus'] for r in results):
        grade = ClassificationGrade.A_PLUS
    elif all(r['in_spec_a'] for r in results):
        grade = ClassificationGrade.A
    elif all(r['in_spec_b'] for r in results):
        grade = ClassificationGrade.B
    elif all(r['in_spec_c'] for r in results):
        grade = ClassificationGrade.C
    else:
        grade = ClassificationGrade.FAIL

    # Calculate SPD metrics
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)

    # Spectral mismatch factor (M)
    try:
        spectral_mismatch = np.trapezoid(df['irradiance'], df['wavelength'])
    except AttributeError:
        spectral_mismatch = np.trapz(df['irradiance'], df['wavelength'])

    # Normalize to reference total
    ref_total = sum(b['am15g_fraction'] for b in WAVELENGTH_BANDS)
    measured_total = sum(r['measured'] for r in results)
    mismatch_factor = measured_total / ref_total if ref_total > 0 else 1.0

    # Weighted deviation
    weighted_deviation = np.sqrt(np.mean([(r['ratio'] - 1)**2 for r in results])) * 100

    return {
        'intervals': results,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'grade': grade,
        'spectral_mismatch_factor': mismatch_factor,
        'weighted_deviation_pct': weighted_deviation,
        'total_measured': total_measured,
    }


def calculate_spc_method(df: pd.DataFrame) -> dict:
    """
    Calculate spectral match using SPC (Spectral Performance Coefficient) method.

    Uses reference cell spectral response to calculate weighted spectral mismatch.
    SPC = integral(E_meas * SR * E_ref) / integral(E_ref * SR * E_meas)
    """
    if df.empty:
        return None

    # Get common wavelength range
    wl_min = max(df['wavelength'].min(), min(REFERENCE_CELL_SR.keys()))
    wl_max = min(df['wavelength'].max(), max(REFERENCE_CELL_SR.keys()))

    # Create common wavelength array
    common_wl = np.linspace(wl_min, wl_max, 100)

    # Interpolate measured spectrum
    measured_interp = np.interp(common_wl, df['wavelength'], df['irradiance'])

    # Interpolate reference cell SR
    sr_wavelengths = sorted(REFERENCE_CELL_SR.keys())
    sr_values = [REFERENCE_CELL_SR[w] for w in sr_wavelengths]
    sr_interp = np.interp(common_wl, sr_wavelengths, sr_values)

    # Interpolate AM1.5G reference
    am15g_wavelengths = sorted(AM15G_REFERENCE.keys())
    am15g_values = [AM15G_REFERENCE[w] for w in am15g_wavelengths]
    am15g_interp = np.interp(common_wl, am15g_wavelengths, am15g_values)

    # Calculate SPC integrals
    # Numerator: integral of (E_measured * SR)
    try:
        num_meas = np.trapezoid(measured_interp * sr_interp, common_wl)
        num_ref = np.trapezoid(am15g_interp * sr_interp, common_wl)
        denom_meas = np.trapezoid(measured_interp, common_wl)
        denom_ref = np.trapezoid(am15g_interp, common_wl)
    except AttributeError:
        num_meas = np.trapz(measured_interp * sr_interp, common_wl)
        num_ref = np.trapz(am15g_interp * sr_interp, common_wl)
        denom_meas = np.trapz(measured_interp, common_wl)
        denom_ref = np.trapz(am15g_interp, common_wl)

    # SPC = (I_meas/I_ref) * (integral(E_ref*SR)/integral(E_meas*SR))
    if num_meas > 0 and denom_meas > 0:
        spc = (denom_meas / denom_ref) * (num_ref / num_meas)
    else:
        spc = 1.0

    # Calculate spectral mismatch factor M
    if num_meas > 0:
        spectral_mismatch = num_meas / num_ref
    else:
        spectral_mismatch = 1.0

    # Determine grade based on SPC deviation from 1.0
    spc_deviation = abs(spc - 1.0)

    if spc_deviation <= 0.02:  # within 2%
        grade = ClassificationGrade.A_PLUS
    elif spc_deviation <= 0.05:  # within 5%
        grade = ClassificationGrade.A
    elif spc_deviation <= 0.10:  # within 10%
        grade = ClassificationGrade.B
    elif spc_deviation <= 0.20:  # within 20%
        grade = ClassificationGrade.C
    else:
        grade = ClassificationGrade.FAIL

    # Calculate interval-based analysis for compatibility
    intervals = []
    for band in WAVELENGTH_BANDS:
        start, end = band['start'], band['end']
        mask = (common_wl >= start) & (common_wl <= end)

        if np.any(mask):
            band_meas = measured_interp[mask]
            band_ref = am15g_interp[mask]
            band_sr = sr_interp[mask]

            try:
                meas_weighted = np.trapezoid(band_meas * band_sr, common_wl[mask])
                ref_weighted = np.trapezoid(band_ref * band_sr, common_wl[mask])
            except AttributeError:
                meas_weighted = np.trapz(band_meas * band_sr, common_wl[mask])
                ref_weighted = np.trapz(band_ref * band_sr, common_wl[mask])

            ratio = meas_weighted / ref_weighted if ref_weighted > 0 else 1.0
        else:
            ratio = 1.0

        intervals.append({
            'band': band['range'],
            'name': band['name'],
            'color': band['color'],
            'start': start,
            'end': end,
            'ratio': ratio,
            'in_spec_aplus': 0.875 <= ratio <= 1.125,
        })

    return {
        'spc': spc,
        'spectral_mismatch_factor': spectral_mismatch,
        'spc_deviation_pct': spc_deviation * 100,
        'grade': grade,
        'intervals': intervals,
        'min_ratio': min(i['ratio'] for i in intervals),
        'max_ratio': max(i['ratio'] for i in intervals),
        'mean_ratio': np.mean([i['ratio'] for i in intervals]),
        'weighted_deviation_pct': spc_deviation * 100,
    }


def create_spectrum_comparison_chart(df: pd.DataFrame, method_result: dict) -> go.Figure:
    """Create interactive spectrum comparison chart"""
    fig = go.Figure()

    # Measured spectrum
    fig.add_trace(go.Scatter(
        x=df['wavelength'],
        y=df['irradiance'],
        mode='lines',
        name='Measured Spectrum',
        line=dict(color='#F59E0B', width=2),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
        hovertemplate='<b>Measured</b><br>%{x:.0f} nm: %{y:.4f}<extra></extra>'
    ))

    # AM1.5G Reference (scaled)
    ref_wl = sorted(AM15G_REFERENCE.keys())
    ref_val = [AM15G_REFERENCE[w] for w in ref_wl]

    # Scale reference to match measured spectrum range
    scale_factor = df['irradiance'].max() / max(ref_val) if max(ref_val) > 0 else 1
    ref_scaled = [v * scale_factor for v in ref_val]

    fig.add_trace(go.Scatter(
        x=ref_wl,
        y=ref_scaled,
        mode='lines',
        name='AM1.5G Reference (scaled)',
        line=dict(color='#6366F1', width=2, dash='dash'),
        hovertemplate='<b>AM1.5G Ref</b><br>%{x:.0f} nm: %{y:.4f}<extra></extra>'
    ))

    # Add wavelength band boundaries
    for band in WAVELENGTH_BANDS:
        fig.add_vrect(
            x0=band['start'],
            x1=band['end'],
            fillcolor=band['color'],
            opacity=0.05,
            layer="below",
            line_width=0,
        )
        fig.add_vline(
            x=band['start'],
            line_dash="dot",
            line_color=band['color'],
            opacity=0.3
        )

    fig.update_layout(
        title=dict(
            text="Measured vs AM1.5G Reference Spectrum",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Wavelength (nm)",
            gridcolor="#E2E8F0",
            range=[300, 1200]
        ),
        yaxis=dict(
            title="Spectral Irradiance",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified"
    )

    return fig


def create_ratio_chart(intervals: list, method_name: str) -> go.Figure:
    """Create spectral match ratio chart with threshold bands"""
    fig = go.Figure()

    bands = [i['band'] for i in intervals]
    ratios = [i['ratio'] for i in intervals]

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
    fig.add_hline(y=0.75, line_dash="dash", line_color="#22C55E", line_width=1)
    fig.add_hline(y=1.25, line_dash="dash", line_color="#22C55E", line_width=1)

    # Determine colors based on threshold
    colors = []
    for r in ratios:
        if 0.875 <= r <= 1.125:
            colors.append("#10B981")
        elif 0.75 <= r <= 1.25:
            colors.append("#22C55E")
        elif 0.6 <= r <= 1.4:
            colors.append("#F59E0B")
        else:
            colors.append("#EF4444")

    # Ratio markers
    fig.add_trace(go.Scatter(
        x=bands,
        y=ratios,
        mode='markers+lines',
        name='Spectral Match Ratio',
        marker=dict(size=14, color=colors, line=dict(width=2, color='white')),
        line=dict(color='#1E3A5F', width=2),
        hovertemplate='<b>%{x}</b><br>Ratio: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"Spectral Match Ratio per Wavelength Band ({method_name} Method)",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(title="Wavelength Band", gridcolor="#E2E8F0"),
        yaxis=dict(title="Ratio (Measured/Reference)", gridcolor="#E2E8F0", range=[0.5, 1.6]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=80, t=60, b=60),
        showlegend=False,
    )

    return fig


def create_bar_comparison_chart(intervals: list) -> go.Figure:
    """Create bar chart comparing measured vs reference fractions"""
    fig = go.Figure()

    bands = [i['band'] for i in intervals]
    reference = [i.get('reference', 0) for i in intervals]
    measured = [i.get('measured', 0) for i in intervals]
    colors = [i['color'] for i in intervals]

    # Reference bars
    fig.add_trace(go.Bar(
        name='AM1.5G Reference',
        x=bands,
        y=reference,
        marker_color='rgba(99, 102, 241, 0.7)',
        marker_line_color='#4F46E5',
        marker_line_width=2,
        hovertemplate='<b>%{x}</b><br>Reference: %{y:.2f}%<extra></extra>'
    ))

    # Measured bars
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
        title=dict(
            text="Spectral Distribution: Measured vs AM1.5G Reference",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(title="Wavelength Band", gridcolor="#E2E8F0"),
        yaxis=dict(title="Spectral Fraction (%)", gridcolor="#E2E8F0"),
        barmode='group',
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified"
    )

    return fig


def create_ratio_histogram(intervals: list) -> go.Figure:
    """Create histogram of spectral match ratios"""
    ratios = [i['ratio'] for i in intervals]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ratios,
        nbinsx=12,
        marker=dict(
            color='#6366F1',
            line=dict(color='white', width=1)
        ),
        hovertemplate='Ratio: %{x:.3f}<br>Count: %{y}<extra></extra>'
    ))

    # Add threshold lines
    fig.add_vline(x=1.0, line_dash="solid", line_color="#1E3A5F", line_width=2)
    fig.add_vline(x=0.875, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ min", annotation_position="top")
    fig.add_vline(x=1.125, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ max", annotation_position="top")

    fig.update_layout(
        title=dict(
            text="Distribution of Spectral Match Ratios",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(title="Ratio (Measured/Reference)", gridcolor="#E2E8F0"),
        yaxis=dict(title="Number of Intervals", gridcolor="#E2E8F0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=60, r=40, t=60, b=60),
        bargap=0.05
    )

    return fig


def generate_sample_spectral_data() -> pd.DataFrame:
    """Generate sample spectral data for demonstration (300-1200nm range)"""
    np.random.seed(42)

    wavelengths = np.arange(300, 1201, 5)

    # Create realistic solar simulator spectrum with extended UV and IR range
    irradiance = []
    for wl in wavelengths:
        # Base AM1.5G-like shape with extended range
        if wl < 350:
            # UV region - low but increasing irradiance
            base = 0.1 + 0.4 * ((wl - 300) / 50)
        elif wl < 400:
            base = 0.5 + 0.5 * ((wl - 350) / 50)
        elif wl < 550:
            base = 1.0 + 0.5 * ((wl - 400) / 150)
        elif wl < 700:
            base = 1.5 - 0.3 * ((wl - 550) / 150)
        elif wl < 900:
            base = 1.2 - 0.4 * ((wl - 700) / 200)
        elif wl < 1100:
            base = 0.8 - 0.3 * ((wl - 900) / 200)
        else:
            # Extended IR region (1100-1200nm) - lower irradiance
            base = 0.5 - 0.2 * ((wl - 1100) / 100)

        # Add realistic noise and lamp characteristics
        noise = np.random.normal(0, 0.05)
        lamp_effect = 0.1 * np.sin(2 * np.pi * wl / 200)
        irradiance.append(max(0, base + noise + lamp_effect))

    return pd.DataFrame({
        'wavelength': wavelengths,
        'irradiance': irradiance
    })


def export_results_to_excel(results: dict, df: pd.DataFrame) -> BytesIO:
    """Export analysis results to Excel file"""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Parameter': ['Test ID', 'Date', 'Method', 'Grade', 'Min Ratio', 'Max Ratio',
                         'Mean Ratio', 'Spectral Mismatch Factor', 'Weighted Deviation (%)'],
            'Value': [
                results.get('test_id', 'N/A'),
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                results.get('method', 'SPD'),
                results.get('grade', ClassificationGrade.FAIL).value,
                f"{results.get('min_ratio', 0):.4f}",
                f"{results.get('max_ratio', 0):.4f}",
                f"{results.get('mean_ratio', 0):.4f}",
                f"{results.get('spectral_mismatch_factor', 0):.4f}",
                f"{results.get('weighted_deviation_pct', 0):.2f}"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # Interval analysis sheet
        if 'intervals' in results:
            intervals_df = pd.DataFrame(results['intervals'])
            intervals_df.to_excel(writer, sheet_name='Interval Analysis', index=False)

        # Raw spectral data sheet
        df.to_excel(writer, sheet_name='Spectral Data', index=False)

        # Thresholds reference
        thresholds_data = {
            'Grade': ['A+', 'A', 'B', 'C'],
            'Min Ratio': [0.875, 0.75, 0.6, 0.4],
            'Max Ratio': [1.125, 1.25, 1.4, 2.0],
            'Deviation': ['±12.5%', '±25%', '±40%', '±60%/+100%']
        }
        pd.DataFrame(thresholds_data).to_excel(writer, sheet_name='Thresholds', index=False)

    output.seek(0)
    return output


def export_results_to_csv(results: dict, df: pd.DataFrame) -> str:
    """Export analysis results to CSV string"""
    output_lines = []

    # Header
    output_lines.append("# IEC 60904-9 Spectral Match Analysis Results")
    output_lines.append(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output_lines.append(f"# Method: {results.get('method', 'SPD')}")
    output_lines.append(f"# Grade: {results.get('grade', ClassificationGrade.FAIL).value}")
    output_lines.append("")

    # Summary
    output_lines.append("# Summary")
    output_lines.append(f"Min Ratio,{results.get('min_ratio', 0):.4f}")
    output_lines.append(f"Max Ratio,{results.get('max_ratio', 0):.4f}")
    output_lines.append(f"Mean Ratio,{results.get('mean_ratio', 0):.4f}")
    output_lines.append(f"Spectral Mismatch Factor,{results.get('spectral_mismatch_factor', 0):.4f}")
    output_lines.append(f"Weighted Deviation (%),{results.get('weighted_deviation_pct', 0):.2f}")
    output_lines.append("")

    # Interval data
    output_lines.append("# Interval Analysis")
    output_lines.append("Band,Name,Start (nm),End (nm),Reference (%),Measured (%),Ratio,In Spec (A+)")
    for interval in results.get('intervals', []):
        output_lines.append(
            f"{interval['band']},{interval['name']},{interval['start']},{interval['end']},"
            f"{interval.get('reference', 0):.2f},{interval.get('measured', 0):.2f},"
            f"{interval['ratio']:.4f},{interval.get('in_spec_aplus', False)}"
        )

    return '\n'.join(output_lines)


def main():
    """Main Spectral Match Analysis page"""

    # Header
    st.markdown('<h1 class="main-title">Spectral Match Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">SPD & SPC Classification per IEC 60904-9 Ed.3</p>',
        unsafe_allow_html=True
    )

    # Initialize session state
    if 'spectral_data' not in st.session_state:
        st.session_state.spectral_data = None
    if 'spd_results' not in st.session_state:
        st.session_state.spd_results = None
    if 'spc_results' not in st.session_state:
        st.session_state.spc_results = None
    if 'test_id' not in st.session_state:
        st.session_state.test_id = f"SM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Analysis Settings")

        analysis_method = st.selectbox(
            "Analysis Method",
            ["SPD (Spectral Power Distribution)", "SPC (Spectral Performance Coefficient)", "Both Methods"],
            index=2
        )

        st.markdown("---")

        st.markdown("### Test Information")
        simulator_id = st.selectbox(
            "Simulator ID",
            get_simulator_ids() or ["SIM-001", "SIM-002", "SIM-003"],
            index=0
        )

        operator = st.text_input("Operator", value="")
        notes = st.text_area("Notes", value="", height=80)

        st.markdown("---")
        st.markdown("### Reference Spectrum")
        st.markdown("**AM1.5G Global**")
        st.markdown("IEC 60904-3")

        st.markdown("---")
        st.markdown("### Wavelength Range")
        st.markdown("**300-1200nm** (8 intervals)")

    # Data upload section
    st.markdown("### Upload Spectral Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>CSV File Format:</strong><br>
            Column 1: Wavelength (nm)<br>
            Column 2: Spectral Irradiance (W/m²/nm or relative units)<br>
            <em>Headers are auto-detected. Minimum range: 300-1200nm recommended for Ed.3.</em>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload spectral data with wavelength and irradiance columns"
        )

    with col2:
        st.markdown("**Or use sample data:**")
        if st.button("Load Sample Data", type="secondary", use_container_width=True):
            st.session_state.spectral_data = generate_sample_spectral_data()
            st.success("Sample spectral data loaded!")

    # Process uploaded file
    if uploaded_file is not None:
        df = parse_spectral_csv(uploaded_file)
        if not df.empty:
            st.session_state.spectral_data = df
            st.success(f"Loaded {len(df)} data points ({df['wavelength'].min():.0f} - {df['wavelength'].max():.0f} nm)")

    # Analysis section
    if st.session_state.spectral_data is not None:
        df = st.session_state.spectral_data

        st.markdown("---")

        # Run analysis button
        if st.button("Run Spectral Match Analysis", type="primary", use_container_width=True):
            with st.spinner("Calculating spectral match..."):
                if "SPD" in analysis_method or "Both" in analysis_method:
                    st.session_state.spd_results = calculate_spd_method(df)
                if "SPC" in analysis_method or "Both" in analysis_method:
                    st.session_state.spc_results = calculate_spc_method(df)

        # Display results if available
        spd_results = st.session_state.spd_results
        spc_results = st.session_state.spc_results

        if spd_results or spc_results:
            st.markdown("---")

            # Overall grade display
            if spd_results and spc_results:
                primary_grade = spd_results['grade']
                secondary_grade = spc_results['grade']

                st.markdown(f"""
                <div class="overall-classification-card">
                    <div class="overall-title">Spectral Match Classification</div>
                    <div class="overall-grade">{primary_grade.value}</div>
                    <div style="opacity: 0.8;">
                        SPD Method: {primary_grade.value} | SPC Method: {secondary_grade.value}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif spd_results:
                grade = spd_results['grade']
                st.markdown(f"""
                <div class="overall-classification-card">
                    <div class="overall-title">Spectral Match Classification (SPD)</div>
                    <div class="overall-grade">{grade.value}</div>
                    <div style="opacity: 0.8;">{get_grade_description(grade)}</div>
                </div>
                """, unsafe_allow_html=True)
            elif spc_results:
                grade = spc_results['grade']
                st.markdown(f"""
                <div class="overall-classification-card">
                    <div class="overall-title">Spectral Match Classification (SPC)</div>
                    <div class="overall-grade">{grade.value}</div>
                    <div style="opacity: 0.8;">{get_grade_description(grade)}</div>
                </div>
                """, unsafe_allow_html=True)

            # Tabbed results display
            tabs = st.tabs([
                ":material/ssid_chart: Spectrum Analysis",
                ":material/analytics: SPD Results" if spd_results else ":material/block: SPD N/A",
                ":material/calculate: SPC Results" if spc_results else ":material/block: SPC N/A",
                ":material/table_chart: Detailed Data",
                ":material/save: Save & Export"
            ])

            # Tab 1: Spectrum Analysis
            with tabs[0]:
                st.markdown("### Measured Spectrum vs Reference")

                active_results = spd_results or spc_results
                fig_spectrum = create_spectrum_comparison_chart(df, active_results)
                st.plotly_chart(fig_spectrum, use_container_width=True)

                # Data preview
                with st.expander("View Raw Data"):
                    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
                    st.caption(f"Showing first 50 of {len(df)} data points")

            # Tab 2: SPD Results
            with tabs[1]:
                if spd_results:
                    st.markdown("### SPD (Spectral Power Distribution) Analysis")

                    st.markdown("""
                    <div class="info-box">
                        <strong>SPD Method:</strong> Calculates the ratio of simulator to AM1.5G reference
                        spectral irradiance fraction in each of 8 wavelength intervals (300-1200nm).
                        Each ratio must fall within classification thresholds per IEC 60904-9 Ed.3.
                    </div>
                    """, unsafe_allow_html=True)

                    # Metrics row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spd_results['min_ratio']:.3f}</div>
                            <div class="metric-label">Min Ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spd_results['max_ratio']:.3f}</div>
                            <div class="metric-label">Max Ratio</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spd_results['spectral_mismatch_factor']:.4f}</div>
                            <div class="metric-label">Mismatch Factor (M)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spd_results['weighted_deviation_pct']:.2f}%</div>
                            <div class="metric-label">Weighted Deviation</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("")

                    # Charts
                    fig_ratio = create_ratio_chart(spd_results['intervals'], "SPD")
                    st.plotly_chart(fig_ratio, use_container_width=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig_bar = create_bar_comparison_chart(spd_results['intervals'])
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with col_b:
                        fig_hist = create_ratio_histogram(spd_results['intervals'])
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.info("SPD analysis not performed. Select 'SPD' or 'Both Methods' to run analysis.")

            # Tab 3: SPC Results
            with tabs[2]:
                if spc_results:
                    st.markdown("### SPC (Spectral Performance Coefficient) Analysis")

                    st.markdown("""
                    <div class="info-box">
                        <strong>SPC Method:</strong> Uses reference cell spectral response to calculate
                        weighted spectral mismatch. The SPC value indicates how the measured spectrum
                        would affect PV cell current compared to AM1.5G reference.
                    </div>
                    """, unsafe_allow_html=True)

                    # SPC metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spc_results['spc']:.4f}</div>
                            <div class="metric-label">SPC Value</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spc_results['spc_deviation_pct']:.2f}%</div>
                            <div class="metric-label">SPC Deviation</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{spc_results['spectral_mismatch_factor']:.4f}</div>
                            <div class="metric-label">Mismatch Factor</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        in_spec = sum(1 for i in spc_results['intervals'] if i['in_spec_aplus'])
                        total_intervals = len(spc_results['intervals'])
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{in_spec}/{total_intervals}</div>
                            <div class="metric-label">Intervals in A+ Spec</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("")

                    # SPC Interpretation
                    spc_val = spc_results['spc']
                    if spc_val > 1.02:
                        interpretation = "Simulator spectrum produces HIGHER short-circuit current than AM1.5G"
                        box_class = "warning-box"
                    elif spc_val < 0.98:
                        interpretation = "Simulator spectrum produces LOWER short-circuit current than AM1.5G"
                        box_class = "warning-box"
                    else:
                        interpretation = "Simulator spectrum closely matches AM1.5G for reference cell response"
                        box_class = "success-box"

                    st.markdown(f"""
                    <div class="{box_class}">
                        <strong>Interpretation:</strong> {interpretation}<br>
                        SPC = {spc_val:.4f} means the measured Isc would be {(spc_val - 1) * 100:+.2f}%
                        compared to measurement under AM1.5G reference.
                    </div>
                    """, unsafe_allow_html=True)

                    # Ratio chart
                    fig_spc_ratio = create_ratio_chart(spc_results['intervals'], "SPC")
                    st.plotly_chart(fig_spc_ratio, use_container_width=True)
                else:
                    st.info("SPC analysis not performed. Select 'SPC' or 'Both Methods' to run analysis.")

            # Tab 4: Detailed Data
            with tabs[3]:
                st.markdown("### Interval-wise Analysis")

                active_results = spd_results or spc_results
                if active_results:
                    # Create detailed table
                    table_data = []
                    for i, interval in enumerate(active_results['intervals']):
                        status = "Pass" if interval.get('in_spec_aplus', False) else "Review"
                        table_data.append({
                            "#": i + 1,
                            "Wavelength Range": interval['band'],
                            "Band Name": interval['name'],
                            "Reference (%)": f"{interval.get('reference', 0):.2f}" if 'reference' in interval else "N/A",
                            "Measured (%)": f"{interval.get('measured', 0):.2f}" if 'measured' in interval else "N/A",
                            "Ratio": f"{interval['ratio']:.4f}",
                            "Status": status
                        })

                    st.dataframe(
                        pd.DataFrame(table_data),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Status": st.column_config.TextColumn(
                                "Status",
                                help="Pass = within A+ threshold (0.875-1.125)"
                            )
                        }
                    )

                    st.markdown("---")

                    # Classification thresholds reference
                    st.markdown("### Classification Thresholds (IEC 60904-9)")

                    thresh_col1, thresh_col2 = st.columns(2)

                    with thresh_col1:
                        st.markdown("""
                        | Grade | Min Ratio | Max Ratio | Deviation |
                        |-------|-----------|-----------|-----------|
                        | **A+** | 0.875 | 1.125 | ±12.5% |
                        | **A** | 0.75 | 1.25 | ±25% |
                        | **B** | 0.6 | 1.4 | ±40% |
                        | **C** | 0.4 | 2.0 | ±60%/+100% |
                        """)

                    with thresh_col2:
                        grade = active_results['grade']
                        st.markdown(f"""
                        **Current Analysis Summary:**
                        - Classification: **{grade.value}** ({get_grade_description(grade)})
                        - Min Ratio: {active_results['min_ratio']:.4f}
                        - Max Ratio: {active_results['max_ratio']:.4f}
                        - Mean Ratio: {active_results['mean_ratio']:.4f}
                        - Total Intervals: {len(active_results['intervals'])}
                        """)

            # Tab 5: Save & Export
            with tabs[4]:
                st.markdown("### Save to Database")

                active_results = spd_results or spc_results
                method_name = "SPD" if spd_results else "SPC"

                col_save1, col_save2 = st.columns(2)

                with col_save1:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left;">
                        <div style="font-weight:600; color:#1E3A5F; margin-bottom:0.5rem;">Test Details</div>
                        <div style="font-size:0.9rem; color:#64748B;">
                            <strong>Test ID:</strong> {st.session_state.test_id}<br>
                            <strong>Simulator:</strong> {simulator_id}<br>
                            <strong>Method:</strong> {method_name}<br>
                            <strong>Grade:</strong> {active_results['grade'].value}<br>
                            <strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_save2:
                    if st.button("Save to Database", type="primary", use_container_width=True):
                        # Prepare intervals data as JSON
                        intervals_json = json.dumps([{
                            'band': i['band'],
                            'name': i['name'],
                            'ratio': float(i['ratio']),
                            'reference': float(i.get('reference', 0)),
                            'measured': float(i.get('measured', 0))
                        } for i in active_results['intervals']])

                        success = insert_spectral_match_result(
                            test_id=st.session_state.test_id,
                            simulator_id=simulator_id,
                            method=method_name,
                            grade=active_results['grade'].value,
                            min_ratio=float(active_results['min_ratio']),
                            max_ratio=float(active_results['max_ratio']),
                            mean_ratio=float(active_results['mean_ratio']),
                            spectral_mismatch_factor=float(active_results.get('spectral_mismatch_factor', 1.0)),
                            weighted_deviation_pct=float(active_results.get('weighted_deviation_pct', 0)),
                            intervals_data=intervals_json,
                            wavelength_range="300-1200nm",
                            operator=operator if operator else None,
                            notes=notes if notes else None
                        )

                        if success:
                            st.success(f"Results saved to database with Test ID: {st.session_state.test_id}")
                            # Generate new test ID for next test
                            st.session_state.test_id = f"SM-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
                        else:
                            st.error("Failed to save results to database. Check connection.")

                st.markdown("---")
                st.markdown("### Export Results")

                export_col1, export_col2 = st.columns(2)

                # Prepare export data
                export_results = active_results.copy()
                export_results['test_id'] = st.session_state.test_id
                export_results['method'] = method_name

                with export_col1:
                    excel_data = export_results_to_excel(export_results, df)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"spectral_match_{st.session_state.test_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                with export_col2:
                    csv_data = export_results_to_csv(export_results, df)
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name=f"spectral_match_{st.session_state.test_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.markdown("---")
                st.markdown("### Historical Results")

                history_df = get_spectral_match_results(simulator_id=simulator_id)
                if not history_df.empty:
                    st.dataframe(
                        history_df[['test_id', 'test_date', 'method', 'grade', 'min_ratio', 'max_ratio']].head(10),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No historical results found for this simulator.")

    else:
        # No data loaded yet - show instructions
        st.markdown("""
        <div class="info-box">
            <strong>Getting Started:</strong><br>
            1. Upload a CSV file with spectral data (wavelength vs irradiance)<br>
            2. Or click "Load Sample Data" to try with demonstration data<br>
            3. Select analysis method (SPD, SPC, or Both)<br>
            4. Click "Run Spectral Match Analysis" to classify the spectrum
        </div>
        """, unsafe_allow_html=True)

        # Classification thresholds reference
        st.markdown("---")
        st.markdown("### IEC 60904-9 Classification Thresholds")

        with st.expander("View Classification Requirements"):
            st.markdown("""
            **Spectral Match (SPD Method) - IEC 60904-9 Ed.3:**

            The spectral match is evaluated by comparing the simulator's spectral irradiance to the
            AM1.5G reference spectrum (IEC 60904-3) across 8 wavelength intervals (300-1200nm):

            | Wavelength Band | AM1.5G Fraction |
            |-----------------|-----------------|
            | 300-400 nm (UV) | 5.4% |
            | 400-500 nm | 17.4% |
            | 500-600 nm | 18.9% |
            | 600-700 nm | 17.4% |
            | 700-800 nm | 14.1% |
            | 800-900 nm | 11.8% |
            | 900-1100 nm | 12.0% |
            | 1100-1200 nm | 3.0% |

            **Classification Thresholds:**

            | Grade | Ratio Range | Description |
            |-------|-------------|-------------|
            | **A+** | 0.875 - 1.125 | Highest precision (calibration lab grade) |
            | **A** | 0.75 - 1.25 | High quality (standard testing) |
            | **B** | 0.6 - 1.4 | Moderate quality (general purpose) |
            | **C** | 0.4 - 2.0 | Basic quality (minimum acceptable) |
            """)


if __name__ == "__main__":
    main()
