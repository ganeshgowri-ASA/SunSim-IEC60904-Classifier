"""
IEC 60904-9 Sun Simulator Classification Dashboard.
Comprehensive classification of solar simulators per IEC 60904-9 standard.
Evaluates: Spectral Match, Non-Uniformity, and Temporal Instability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from io import BytesIO
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="IEC 60904-9 Classification | SunSim",
    page_icon="☀️",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .classification-card {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 25px;
        border-radius: 15px;
        border: 3px solid;
        text-align: center;
        margin: 10px 0;
    }
    .grade-display {
        font-size: 64px;
        font-weight: bold;
    }
    .grade-label {
        font-size: 14px;
        color: #888;
        margin-top: 5px;
    }
    .overall-grade {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        padding: 30px;
    }
    .interval-table {
        background-color: #1A1D24;
        border-radius: 10px;
        padding: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .grade-a-plus { color: #00D4AA; border-color: #00D4AA; }
    .grade-a { color: #4ECDC4; border-color: #4ECDC4; }
    .grade-b { color: #FFE66D; border-color: #FFE66D; }
    .grade-c { color: #FF6B6B; border-color: #FF6B6B; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# IEC 60904-9 CONSTANTS AND CLASSIFICATION CRITERIA
# =============================================================================

# Wavelength intervals per IEC 60904-9 Ed.3 (nm) - Extended to 300-1200nm
WAVELENGTH_INTERVALS = [
    (300, 400),
    (400, 500),
    (500, 600),
    (600, 700),
    (700, 800),
    (800, 900),
    (900, 1100),
    (1100, 1200)
]

# AM1.5G Reference spectrum fraction per interval (IEC 60904-9 Ed.3)
# Extended range from 300-1200nm per ASTM G173-03 reference spectrum
AM15G_REFERENCE_FRACTIONS = {
    (300, 400): 0.054,   # 5.4% (UV region)
    (400, 500): 0.174,   # 17.4%
    (500, 600): 0.189,   # 18.9%
    (600, 700): 0.174,   # 17.4%
    (700, 800): 0.141,   # 14.1%
    (800, 900): 0.118,   # 11.8%
    (900, 1100): 0.120,  # 12.0%
    (1100, 1200): 0.030  # 3.0% (Extended IR)
}

# Classification thresholds per IEC 60904-9
SPECTRAL_CLASS_THRESHOLDS = {
    'A+': (0.75, 1.25),
    'A': (0.6, 1.4),
    'B': (0.4, 2.0),
    'C': (0.0, float('inf'))
}

UNIFORMITY_CLASS_THRESHOLDS = {
    'A+': 2.0,
    'A': 5.0,
    'B': 10.0,
    'C': float('inf')
}

TEMPORAL_CLASS_THRESHOLDS = {
    'A+': 0.5,
    'A': 2.0,
    'B': 5.0,
    'C': float('inf')
}

GRADE_COLORS = {
    'A+': '#00D4AA',
    'A': '#4ECDC4',
    'B': '#FFE66D',
    'C': '#FF6B6B'
}


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_spectral_interval(spd_ratio: float) -> str:
    """Classify a single spectral interval based on SPD ratio."""
    for grade, (low, high) in SPECTRAL_CLASS_THRESHOLDS.items():
        if low <= spd_ratio <= high:
            return grade
    return 'C'


def calculate_spectral_classification(spectral_df: pd.DataFrame) -> dict:
    """
    Calculate spectral match classification per IEC 60904-9.

    Args:
        spectral_df: DataFrame with 'wavelength' and 'irradiance' columns

    Returns:
        dict with interval results and overall classification
    """
    results = {
        'intervals': [],
        'overall_class': 'A+',
        'valid': True
    }

    wavelengths = spectral_df['wavelength'].values
    irradiance = spectral_df['irradiance'].values

    # Calculate total irradiance in 300-1200nm range (IEC 60904-9 Ed.3)
    mask_total = (wavelengths >= 300) & (wavelengths <= 1200)
    total_irradiance = np.trapezoid(irradiance[mask_total], wavelengths[mask_total])

    if total_irradiance <= 0:
        results['valid'] = False
        return results

    worst_class = 'A+'
    class_order = ['A+', 'A', 'B', 'C']

    for interval in WAVELENGTH_INTERVALS:
        low, high = interval
        mask = (wavelengths >= low) & (wavelengths <= high)

        if not any(mask):
            results['intervals'].append({
                'interval': f"{low}-{high}nm",
                'measured_fraction': None,
                'reference_fraction': AM15G_REFERENCE_FRACTIONS[interval],
                'spd_ratio': None,
                'class': 'N/A'
            })
            continue

        # Calculate measured fraction
        interval_irradiance = np.trapezoid(irradiance[mask], wavelengths[mask])
        measured_fraction = interval_irradiance / total_irradiance

        # Calculate SPD ratio
        reference_fraction = AM15G_REFERENCE_FRACTIONS[interval]
        spd_ratio = measured_fraction / reference_fraction

        # Classify interval
        interval_class = classify_spectral_interval(spd_ratio)

        results['intervals'].append({
            'interval': f"{low}-{high}nm",
            'measured_fraction': measured_fraction,
            'reference_fraction': reference_fraction,
            'spd_ratio': spd_ratio,
            'class': interval_class
        })

        # Track worst class
        if class_order.index(interval_class) > class_order.index(worst_class):
            worst_class = interval_class

    results['overall_class'] = worst_class
    return results


def calculate_uniformity_classification(uniformity_df: pd.DataFrame) -> dict:
    """
    Calculate non-uniformity classification per IEC 60904-9.

    Args:
        uniformity_df: DataFrame with position and irradiance data

    Returns:
        dict with uniformity results and classification
    """
    irradiance_values = uniformity_df['irradiance'].values

    e_max = np.max(irradiance_values)
    e_min = np.min(irradiance_values)
    e_avg = np.mean(irradiance_values)

    # Non-uniformity formula: (Emax - Emin) / (Emax + Emin) × 100%
    non_uniformity = ((e_max - e_min) / (e_max + e_min)) * 100

    # Classify
    classification = 'C'
    for grade, threshold in UNIFORMITY_CLASS_THRESHOLDS.items():
        if non_uniformity <= threshold:
            classification = grade
            break

    return {
        'e_max': e_max,
        'e_min': e_min,
        'e_avg': e_avg,
        'non_uniformity_pct': non_uniformity,
        'classification': classification
    }


def calculate_temporal_classification(temporal_df: pd.DataFrame) -> dict:
    """
    Calculate temporal instability classification per IEC 60904-9.

    Args:
        temporal_df: DataFrame with 'time' and 'irradiance' columns

    Returns:
        dict with temporal stability results and classification
    """
    irradiance_values = temporal_df['irradiance'].values

    e_max = np.max(irradiance_values)
    e_min = np.min(irradiance_values)
    e_avg = np.mean(irradiance_values)

    # Temporal instability: (Emax - Emin) / (Emax + Emin) × 100%
    # This is STI (Short-Term Instability)
    sti = ((e_max - e_min) / (e_max + e_min)) * 100

    # Calculate standard deviation as percentage
    std_pct = (np.std(irradiance_values) / e_avg) * 100

    # Classify based on STI
    classification = 'C'
    for grade, threshold in TEMPORAL_CLASS_THRESHOLDS.items():
        if sti <= threshold:
            classification = grade
            break

    return {
        'e_max': e_max,
        'e_min': e_min,
        'e_avg': e_avg,
        'sti_pct': sti,
        'std_pct': std_pct,
        'classification': classification,
        'n_samples': len(irradiance_values),
        'duration': temporal_df['time'].max() - temporal_df['time'].min() if 'time' in temporal_df else None
    }


def get_overall_classification(spectral: str, uniformity: str, temporal: str) -> str:
    """Combine individual classifications into overall grade string."""
    return f"{spectral}{uniformity}{temporal}"


# =============================================================================
# SAMPLE DATA GENERATORS
# =============================================================================

def generate_sample_spectral_data(quality: str = 'A') -> pd.DataFrame:
    """Generate sample spectral irradiance data."""
    wavelengths = np.arange(350, 1150, 2)

    # Base AM1.5G-like spectrum (simplified Gaussian peaks)
    irradiance = np.zeros_like(wavelengths, dtype=float)

    # Add spectral features
    for wl, amp, width in [(480, 1.8, 50), (550, 1.9, 60), (650, 1.7, 70),
                            (750, 1.4, 80), (850, 1.2, 90), (1000, 1.0, 120)]:
        irradiance += amp * np.exp(-((wavelengths - wl) ** 2) / (2 * width ** 2))

    # Add noise based on quality
    noise_level = {'A+': 0.02, 'A': 0.05, 'B': 0.1, 'C': 0.2}
    noise = noise_level.get(quality, 0.05)
    irradiance *= (1 + np.random.normal(0, noise, len(wavelengths)))
    irradiance = np.maximum(irradiance, 0)

    return pd.DataFrame({'wavelength': wavelengths, 'irradiance': irradiance})


def generate_sample_uniformity_data(grid_size: int = 5, quality: str = 'A') -> pd.DataFrame:
    """Generate sample uniformity grid data."""
    positions = []

    # Non-uniformity levels
    nu_level = {'A+': 0.015, 'A': 0.035, 'B': 0.08, 'C': 0.15}
    variation = nu_level.get(quality, 0.035)

    base_irradiance = 1000  # W/m²

    for i in range(grid_size):
        for j in range(grid_size):
            # Add radial falloff pattern (typical for solar simulators)
            center = (grid_size - 1) / 2
            distance = np.sqrt((i - center)**2 + (j - center)**2)
            max_distance = np.sqrt(2) * center

            # Radial variation + random noise
            radial_factor = 1 - (distance / max_distance) * variation
            noise = np.random.normal(0, variation * 0.3)

            irradiance = base_irradiance * radial_factor * (1 + noise)

            positions.append({
                'x': i + 1,
                'y': j + 1,
                'row': i + 1,
                'col': j + 1,
                'irradiance': irradiance
            })

    return pd.DataFrame(positions)


def generate_sample_temporal_data(duration: float = 60, quality: str = 'A') -> pd.DataFrame:
    """Generate sample temporal stability data."""
    # Instability levels
    inst_level = {'A+': 0.003, 'A': 0.015, 'B': 0.04, 'C': 0.08}
    variation = inst_level.get(quality, 0.015)

    n_samples = int(duration * 10)  # 10 Hz sampling
    times = np.linspace(0, duration, n_samples)

    base_irradiance = 1000  # W/m²

    # Add low-frequency drift and high-frequency noise
    drift = np.sin(2 * np.pi * times / duration) * variation * 0.3
    noise = np.random.normal(0, variation, n_samples)

    irradiance = base_irradiance * (1 + drift + noise)

    return pd.DataFrame({'time': times, 'irradiance': irradiance})


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_spectral_plot(spectral_df: pd.DataFrame, results: dict) -> go.Figure:
    """Create spectral irradiance plot with interval highlighting."""
    fig = go.Figure()

    # Main spectrum line
    fig.add_trace(go.Scatter(
        x=spectral_df['wavelength'],
        y=spectral_df['irradiance'],
        mode='lines',
        name='Measured Spectrum',
        line=dict(color='#4ECDC4', width=2)
    ))

    # Add interval shading
    colors = {'A+': 'rgba(0,212,170,0.2)', 'A': 'rgba(78,205,196,0.2)',
              'B': 'rgba(255,230,109,0.2)', 'C': 'rgba(255,107,107,0.2)'}

    for interval_result in results['intervals']:
        interval_str = interval_result['interval']
        low, high = map(int, interval_str.replace('nm', '').split('-'))
        grade = interval_result['class']

        if grade != 'N/A':
            fig.add_vrect(
                x0=low, x1=high,
                fillcolor=colors.get(grade, 'rgba(128,128,128,0.1)'),
                layer="below",
                line_width=0,
                annotation_text=grade,
                annotation_position="top"
            )

    fig.update_layout(
        title=dict(text="Spectral Irradiance Distribution", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Wavelength (nm)",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            range=[350, 1150]
        ),
        yaxis=dict(
            title="Irradiance (W/m²/nm)",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_uniformity_heatmap(uniformity_df: pd.DataFrame, results: dict) -> go.Figure:
    """Create uniformity heatmap visualization."""
    # Pivot data to grid format
    grid_data = uniformity_df.pivot(index='row', columns='col', values='irradiance')

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=grid_data.values,
        x=grid_data.columns,
        y=grid_data.index,
        colorscale=[
            [0, '#FF6B6B'],
            [0.25, '#FFE66D'],
            [0.5, '#4ECDC4'],
            [0.75, '#00D4AA'],
            [1, '#00D4AA']
        ],
        colorbar=dict(
            title=dict(text="W/m²", font=dict(color='#FAFAFA')),
            tickfont=dict(color='#FAFAFA')
        ),
        hovertemplate='Position (%{x}, %{y})<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    ))

    # Add annotations for values
    for i, row in enumerate(grid_data.index):
        for j, col in enumerate(grid_data.columns):
            value = grid_data.loc[row, col]
            fig.add_annotation(
                x=col, y=row,
                text=f"{value:.0f}",
                showarrow=False,
                font=dict(color='#0E1117', size=10, weight='bold')
            )

    nu_pct = results['non_uniformity_pct']
    grade = results['classification']

    fig.update_layout(
        title=dict(
            text=f"Irradiance Uniformity Map (Non-uniformity: {nu_pct:.2f}% - Class {grade})",
            font=dict(color='#FAFAFA', size=16)
        ),
        xaxis=dict(
            title="Column",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            dtick=1
        ),
        yaxis=dict(
            title="Row",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            dtick=1,
            autorange='reversed'
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=450
    )

    return fig


def create_temporal_plot(temporal_df: pd.DataFrame, results: dict) -> go.Figure:
    """Create temporal stability time-series plot."""
    fig = go.Figure()

    # Main time series
    fig.add_trace(go.Scatter(
        x=temporal_df['time'],
        y=temporal_df['irradiance'],
        mode='lines',
        name='Irradiance',
        line=dict(color='#4ECDC4', width=1.5)
    ))

    # Add mean line
    e_avg = results['e_avg']
    fig.add_hline(
        y=e_avg,
        line_dash="solid",
        line_color="#00D4AA",
        annotation_text=f"Mean: {e_avg:.1f} W/m²",
        annotation_position="right"
    )

    # Add max/min lines
    fig.add_hline(y=results['e_max'], line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"Max: {results['e_max']:.1f}", annotation_position="right")
    fig.add_hline(y=results['e_min'], line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"Min: {results['e_min']:.1f}", annotation_position="right")

    sti = results['sti_pct']
    grade = results['classification']

    fig.update_layout(
        title=dict(
            text=f"Temporal Stability (STI: {sti:.3f}% - Class {grade})",
            font=dict(color='#FAFAFA', size=16)
        ),
        xaxis=dict(
            title="Time (s)",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="Irradiance (W/m²)",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=350,
        showlegend=False
    )

    return fig


def display_grade_card(grade: str, label: str, value: str = None):
    """Display a classification grade card."""
    color = GRADE_COLORS.get(grade, '#888888')
    value_text = f"<div style='font-size: 14px; color: #888;'>{value}</div>" if value else ""

    st.markdown(f"""
    <div class="classification-card" style="border-color: {color};">
        <div class="grade-display" style="color: {color};">{grade}</div>
        <div class="grade-label">{label}</div>
        {value_text}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(
    spectral_results: dict,
    uniformity_results: dict,
    temporal_results: dict,
    simulator_info: dict,
    spectral_df: pd.DataFrame = None,
    uniformity_df: pd.DataFrame = None,
    temporal_df: pd.DataFrame = None
) -> bytes:
    """Generate PDF classification report."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
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
    normal_style = styles['Normal']

    elements = []

    # Title
    elements.append(Paragraph("IEC 60904-9 Solar Simulator Classification Report", title_style))
    elements.append(Spacer(1, 12))

    # Report info
    overall_grade = get_overall_classification(
        spectral_results.get('overall_class', 'N/A') if spectral_results else 'N/A',
        uniformity_results.get('classification', 'N/A') if uniformity_results else 'N/A',
        temporal_results.get('classification', 'N/A') if temporal_results else 'N/A'
    )

    info_data = [
        ['Report Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Simulator ID:', simulator_info.get('simulator_id', 'N/A')],
        ['Simulator Model:', simulator_info.get('model', 'N/A')],
        ['Test Location:', simulator_info.get('location', 'N/A')],
        ['Operator:', simulator_info.get('operator', 'N/A')],
        ['Overall Classification:', overall_grade]
    ]

    info_table = Table(info_data, colWidths=[120, 350])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 20))

    # Classification Summary
    elements.append(Paragraph("Classification Summary", heading_style))

    summary_data = [
        ['Parameter', 'Result', 'Class', 'Criteria'],
        ['Spectral Match',
         f"Worst SPD: {min([i['spd_ratio'] for i in spectral_results['intervals'] if i['spd_ratio']], default=0):.3f}" if spectral_results else 'N/A',
         spectral_results.get('overall_class', 'N/A') if spectral_results else 'N/A',
         'A+: 0.75-1.25, A: 0.6-1.4, B: 0.4-2.0'],
        ['Non-Uniformity',
         f"{uniformity_results['non_uniformity_pct']:.2f}%" if uniformity_results else 'N/A',
         uniformity_results.get('classification', 'N/A') if uniformity_results else 'N/A',
         'A+: ≤2%, A: ≤5%, B: ≤10%'],
        ['Temporal Instability',
         f"{temporal_results['sti_pct']:.3f}%" if temporal_results else 'N/A',
         temporal_results.get('classification', 'N/A') if temporal_results else 'N/A',
         'A+: ≤0.5%, A: ≤2%, B: ≤5%']
    ]

    summary_table = Table(summary_data, colWidths=[100, 120, 60, 190])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A1D24')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Spectral Match Details
    if spectral_results and spectral_results.get('intervals'):
        elements.append(Paragraph("Spectral Match Analysis (IEC 60904-9)", heading_style))

        spectral_data = [['Wavelength Interval', 'Measured %', 'Reference %', 'SPD Ratio', 'Class']]
        for interval in spectral_results['intervals']:
            measured = f"{interval['measured_fraction']*100:.1f}%" if interval['measured_fraction'] else 'N/A'
            reference = f"{interval['reference_fraction']*100:.1f}%"
            spd = f"{interval['spd_ratio']:.3f}" if interval['spd_ratio'] else 'N/A'
            spectral_data.append([
                interval['interval'],
                measured,
                reference,
                spd,
                interval['class']
            ])

        spectral_table = Table(spectral_data, colWidths=[100, 80, 80, 80, 60])
        spectral_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A1D24')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(spectral_table)
        elements.append(Spacer(1, 15))

    # ISO 17025 Compliance Section
    elements.append(Paragraph("ISO 17025 Compliance Information", heading_style))

    compliance_text = """
    This classification was performed in accordance with IEC 60904-9 Ed.3 standard requirements.
    The measurement uncertainty and calibration status of equipment should be documented separately.

    Measurement Equipment Calibration Status: ______________________
    Calibration Certificate Number: ______________________
    Next Calibration Due: ______________________

    Reviewed By: ______________________  Date: ______________________
    Approved By: ______________________  Date: ______________________
    """

    elements.append(Paragraph(compliance_text, normal_style))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# MAIN PAGE CONTENT
# =============================================================================

st.title("☀️ IEC 60904-9 Solar Simulator Classification")
st.markdown("---")

st.markdown("""
Classify your solar simulator according to **IEC 60904-9** standard requirements.
This tool evaluates three key parameters:
- **Spectral Match** - How well the spectrum matches AM1.5G reference
- **Non-Uniformity** - Spatial variation across the test plane
- **Temporal Instability** - Stability over time
""")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Simulator Information")
    simulator_id = st.text_input("Simulator ID", value="SIM-001")
    simulator_model = st.text_input("Model", value="")
    test_location = st.text_input("Test Location", value="")
    operator_name = st.text_input("Operator", value="")

    st.markdown("---")
    st.subheader("Data Source")
    use_sample_data = st.checkbox("Use Sample Data", value=True)

    if use_sample_data:
        sample_quality = st.selectbox(
            "Sample Data Quality",
            ['A+', 'A', 'B', 'C'],
            index=1
        )
        grid_size = st.slider("Uniformity Grid Size", 3, 10, 5)
        temporal_duration = st.slider("Temporal Duration (s)", 10, 120, 60)

# Initialize session state
if 'spectral_data' not in st.session_state:
    st.session_state.spectral_data = None
if 'uniformity_data' not in st.session_state:
    st.session_state.uniformity_data = None
if 'temporal_data' not in st.session_state:
    st.session_state.temporal_data = None

# Create tabs for different measurement types
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Spectral Match",
    "🎯 Non-Uniformity",
    "⏱️ Temporal Stability",
    "📋 Classification Result"
])

# =============================================================================
# TAB 1: SPECTRAL MATCH
# =============================================================================
with tab1:
    st.subheader("Spectral Match Classification")

    st.markdown("""
    Upload spectral irradiance data or use sample data. The spectrum is evaluated against
    AM1.5G reference in six wavelength intervals per IEC 60904-9.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        if use_sample_data:
            if st.button("Generate Sample Spectral Data", type="primary", key="gen_spectral"):
                st.session_state.spectral_data = generate_sample_spectral_data(sample_quality)
                st.success("Sample spectral data generated!")
        else:
            uploaded_spectral = st.file_uploader(
                "Upload Spectral Data (CSV)",
                type=['csv'],
                key="spectral_upload",
                help="CSV with 'wavelength' (nm) and 'irradiance' (W/m²/nm) columns"
            )
            if uploaded_spectral:
                try:
                    st.session_state.spectral_data = pd.read_csv(uploaded_spectral)
                    st.success("Spectral data loaded!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Manual entry option
        with st.expander("Manual Entry"):
            st.markdown("Enter wavelength intervals manually:")
            manual_intervals = []
            for interval in WAVELENGTH_INTERVALS:
                low, high = interval
                fraction = st.number_input(
                    f"{low}-{high}nm fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=AM15G_REFERENCE_FRACTIONS[interval],
                    step=0.01,
                    key=f"manual_{low}_{high}"
                )
                manual_intervals.append({'interval': interval, 'fraction': fraction})

    with col2:
        if st.session_state.spectral_data is not None:
            spectral_df = st.session_state.spectral_data
            spectral_results = calculate_spectral_classification(spectral_df)

            if spectral_results['valid']:
                # Display spectral plot
                fig_spectral = create_spectral_plot(spectral_df, spectral_results)
                st.plotly_chart(fig_spectral, use_container_width=True)

                # Display interval results table
                st.markdown("**Interval Analysis:**")
                interval_df = pd.DataFrame(spectral_results['intervals'])
                interval_df['measured_fraction'] = interval_df['measured_fraction'].apply(
                    lambda x: f"{x*100:.1f}%" if x else "N/A"
                )
                interval_df['reference_fraction'] = interval_df['reference_fraction'].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                interval_df['spd_ratio'] = interval_df['spd_ratio'].apply(
                    lambda x: f"{x:.3f}" if x else "N/A"
                )
                st.dataframe(interval_df, hide_index=True, use_container_width=True)

                # Display overall spectral class
                display_grade_card(
                    spectral_results['overall_class'],
                    "Spectral Match Class",
                    "Based on worst interval"
                )
            else:
                st.error("Invalid spectral data. Please check your input.")
        else:
            st.info("Generate or upload spectral data to see analysis.")

# =============================================================================
# TAB 2: NON-UNIFORMITY
# =============================================================================
with tab2:
    st.subheader("Non-Uniformity Classification")

    st.markdown("""
    Measure irradiance at multiple positions across the test plane to evaluate spatial uniformity.
    Non-uniformity is calculated as: **(Emax - Emin) / (Emax + Emin) × 100%**
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        if use_sample_data:
            if st.button("Generate Sample Uniformity Data", type="primary", key="gen_uniformity"):
                st.session_state.uniformity_data = generate_sample_uniformity_data(grid_size, sample_quality)
                st.success("Sample uniformity data generated!")
        else:
            uploaded_uniformity = st.file_uploader(
                "Upload Uniformity Data (CSV)",
                type=['csv'],
                key="uniformity_upload",
                help="CSV with 'row', 'col', and 'irradiance' columns"
            )
            if uploaded_uniformity:
                try:
                    st.session_state.uniformity_data = pd.read_csv(uploaded_uniformity)
                    st.success("Uniformity data loaded!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Reference cell position
        st.markdown("---")
        st.markdown("**Reference Cell Position:**")
        if st.session_state.uniformity_data is not None:
            max_row = st.session_state.uniformity_data['row'].max()
            max_col = st.session_state.uniformity_data['col'].max()
            ref_row = st.number_input("Row", min_value=1, max_value=int(max_row), value=int((max_row+1)//2))
            ref_col = st.number_input("Column", min_value=1, max_value=int(max_col), value=int((max_col+1)//2))

    with col2:
        if st.session_state.uniformity_data is not None:
            uniformity_df = st.session_state.uniformity_data
            uniformity_results = calculate_uniformity_classification(uniformity_df)

            # Display heatmap
            fig_uniformity = create_uniformity_heatmap(uniformity_df, uniformity_results)
            st.plotly_chart(fig_uniformity, use_container_width=True)

            # Display metrics
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Max Irradiance", f"{uniformity_results['e_max']:.1f} W/m²")
            with met_col2:
                st.metric("Min Irradiance", f"{uniformity_results['e_min']:.1f} W/m²")
            with met_col3:
                st.metric("Average", f"{uniformity_results['e_avg']:.1f} W/m²")
            with met_col4:
                st.metric("Non-Uniformity", f"{uniformity_results['non_uniformity_pct']:.2f}%")

            # Reference cell correction factor
            if 'ref_row' in dir() and 'ref_col' in dir():
                ref_mask = (uniformity_df['row'] == ref_row) & (uniformity_df['col'] == ref_col)
                if ref_mask.any():
                    ref_irradiance = uniformity_df.loc[ref_mask, 'irradiance'].values[0]
                    correction_factor = uniformity_results['e_avg'] / ref_irradiance
                    st.info(f"Reference Cell Position: ({ref_row}, {ref_col}) | "
                           f"Irradiance: {ref_irradiance:.1f} W/m² | "
                           f"Correction Factor: {correction_factor:.4f}")

            # Display grade
            display_grade_card(
                uniformity_results['classification'],
                "Non-Uniformity Class",
                f"{uniformity_results['non_uniformity_pct']:.2f}%"
            )
        else:
            st.info("Generate or upload uniformity data to see analysis.")

# =============================================================================
# TAB 3: TEMPORAL STABILITY
# =============================================================================
with tab3:
    st.subheader("Temporal Instability Classification")

    st.markdown("""
    Measure irradiance over time to evaluate temporal stability.
    Short-Term Instability (STI) is calculated as: **(Emax - Emin) / (Emax + Emin) × 100%**
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        if use_sample_data:
            if st.button("Generate Sample Temporal Data", type="primary", key="gen_temporal"):
                st.session_state.temporal_data = generate_sample_temporal_data(temporal_duration, sample_quality)
                st.success("Sample temporal data generated!")
        else:
            uploaded_temporal = st.file_uploader(
                "Upload Temporal Data (CSV)",
                type=['csv'],
                key="temporal_upload",
                help="CSV with 'time' (s) and 'irradiance' (W/m²) columns"
            )
            if uploaded_temporal:
                try:
                    st.session_state.temporal_data = pd.read_csv(uploaded_temporal)
                    st.success("Temporal data loaded!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    with col2:
        if st.session_state.temporal_data is not None:
            temporal_df = st.session_state.temporal_data
            temporal_results = calculate_temporal_classification(temporal_df)

            # Display time series plot
            fig_temporal = create_temporal_plot(temporal_df, temporal_results)
            st.plotly_chart(fig_temporal, use_container_width=True)

            # Display metrics
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Max Irradiance", f"{temporal_results['e_max']:.1f} W/m²")
            with met_col2:
                st.metric("Min Irradiance", f"{temporal_results['e_min']:.1f} W/m²")
            with met_col3:
                st.metric("STI", f"{temporal_results['sti_pct']:.3f}%")
            with met_col4:
                st.metric("Std Dev", f"{temporal_results['std_pct']:.3f}%")

            # Display grade
            display_grade_card(
                temporal_results['classification'],
                "Temporal Instability Class",
                f"STI: {temporal_results['sti_pct']:.3f}%"
            )
        else:
            st.info("Generate or upload temporal data to see analysis.")

# =============================================================================
# TAB 4: CLASSIFICATION RESULT
# =============================================================================
with tab4:
    st.subheader("Overall Classification Result")

    # Check if all data is available
    has_spectral = st.session_state.spectral_data is not None
    has_uniformity = st.session_state.uniformity_data is not None
    has_temporal = st.session_state.temporal_data is not None

    if has_spectral and has_uniformity and has_temporal:
        # Calculate all results
        spectral_results = calculate_spectral_classification(st.session_state.spectral_data)
        uniformity_results = calculate_uniformity_classification(st.session_state.uniformity_data)
        temporal_results = calculate_temporal_classification(st.session_state.temporal_data)

        spectral_class = spectral_results['overall_class']
        uniformity_class = uniformity_results['classification']
        temporal_class = temporal_results['classification']

        overall_grade = get_overall_classification(spectral_class, uniformity_class, temporal_class)

        # Determine overall color (worst grade)
        grades = [spectral_class, uniformity_class, temporal_class]
        grade_order = ['A+', 'A', 'B', 'C']
        worst_grade = max(grades, key=lambda x: grade_order.index(x))
        overall_color = GRADE_COLORS[worst_grade]

        # Display overall grade prominently
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
                    border-radius: 20px; border: 3px solid {overall_color}; margin: 20px 0;">
            <div style="font-size: 24px; color: #888; margin-bottom: 10px;">IEC 60904-9 Classification</div>
            <div class="overall-grade" style="color: {overall_color};">{overall_grade}</div>
            <div style="font-size: 18px; color: #FAFAFA;">
                Spectral: {spectral_class} | Uniformity: {uniformity_class} | Temporal: {temporal_class}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Individual grade cards
        st.markdown("### Individual Classifications")
        grade_col1, grade_col2, grade_col3 = st.columns(3)

        with grade_col1:
            display_grade_card(spectral_class, "Spectral Match")
        with grade_col2:
            display_grade_card(uniformity_class, "Non-Uniformity")
        with grade_col3:
            display_grade_card(temporal_class, "Temporal Instability")

        # Summary table
        st.markdown("### Detailed Summary")

        summary_data = {
            'Parameter': ['Spectral Match', 'Non-Uniformity', 'Temporal Instability'],
            'Measured Value': [
                f"Worst SPD: {min([i['spd_ratio'] for i in spectral_results['intervals'] if i['spd_ratio']], default=0):.3f}",
                f"{uniformity_results['non_uniformity_pct']:.2f}%",
                f"{temporal_results['sti_pct']:.3f}%"
            ],
            'Class': [spectral_class, uniformity_class, temporal_class],
            'Criteria (A+)': ['0.75 - 1.25', '≤ 2%', '≤ 0.5%'],
            'Criteria (A)': ['0.60 - 1.40', '≤ 5%', '≤ 2%'],
            'Criteria (B)': ['0.40 - 2.00', '≤ 10%', '≤ 5%']
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # Classification legend
        with st.expander("Classification Criteria (IEC 60904-9)"):
            st.markdown("""
            | Class | Spectral Match (SPD) | Non-Uniformity | Temporal Instability |
            |-------|---------------------|----------------|---------------------|
            | **A+** | 0.75 - 1.25 | ≤ 2% | ≤ 0.5% |
            | **A** | 0.60 - 1.40 | ≤ 5% | ≤ 2% |
            | **B** | 0.40 - 2.00 | ≤ 10% | ≤ 5% |
            | **C** | Outside B limits | > 10% | > 5% |

            The overall classification is expressed as three letters (e.g., "AAA" or "A+BA"),
            representing Spectral Match, Non-Uniformity, and Temporal Instability respectively.
            """)

        # PDF Report Generation
        st.markdown("### Generate Report")

        simulator_info = {
            'simulator_id': simulator_id,
            'model': simulator_model,
            'location': test_location,
            'operator': operator_name
        }

        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_pdf_report(
                    spectral_results,
                    uniformity_results,
                    temporal_results,
                    simulator_info,
                    st.session_state.spectral_data,
                    st.session_state.uniformity_data,
                    st.session_state.temporal_data
                )

                if pdf_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"IEC60904_Classification_{simulator_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Report generated successfully!")
                else:
                    st.error("Could not generate PDF. ReportLab library may not be installed.")

    else:
        # Show what's missing
        st.warning("Complete all three measurements to see the overall classification.")

        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            if has_spectral:
                st.success("Spectral Data: Complete")
            else:
                st.error("Spectral Data: Missing")
        with status_col2:
            if has_uniformity:
                st.success("Uniformity Data: Complete")
            else:
                st.error("Uniformity Data: Missing")
        with status_col3:
            if has_temporal:
                st.success("Temporal Data: Complete")
            else:
                st.error("Temporal Data: Missing")

        st.info("Use the tabs above to enter or generate data for each measurement type.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    Classification per IEC 60904-9 Ed.3 | Solar Simulator Performance Requirements<br>
    Reference Spectrum: AM1.5G (IEC 60904-3)
</div>
""", unsafe_allow_html=True)
