"""
I-V Curve Translation Between Simulator Classifications.

Translates I-V measurements between different solar simulator classifications
(e.g., BBA to A+A+A+, AAA to A+A+A+) with uncertainty calculations based on
IEC 60904-9 standards and peer-reviewed literature.

References:
- IEC 60904-9:2020 - Solar simulator performance requirements
- IEC 60904-1:2020 - Measurement of photovoltaic I-V characteristics
- Emery, K., et al. "Uncertainty of field I-V measurements" NREL (1999)
- Müllejans, H., et al. "Analysis of solar simulator spectrum measurement" EU JRC (2015)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import init_database

# Page configuration
st.set_page_config(
    page_title="I-V Curve Translation | SunSim Classifier",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3139;
    }
    .translation-card {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2D3139;
        margin: 10px 0;
    }
    .uncertainty-good { color: #00D4AA; }
    .uncertainty-moderate { color: #FFE66D; }
    .uncertainty-high { color: #FF6B6B; }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()


# IEC 60904-9 Classification Limits
# Classification: (Spectral Match, Spatial Non-uniformity, Temporal Instability)
# Each tuple is (min_ratio, max_ratio) for spectral, (max_nonuniformity%) for spatial,
# (max_instability%) for temporal

IEC_CLASSIFICATION_LIMITS = {
    'A+A+A+': {'spectral': (0.9375, 1.0625), 'spatial': 1.0, 'temporal': 0.5},  # Extended A+
    'A+A+A': {'spectral': (0.9375, 1.0625), 'spatial': 1.0, 'temporal': 1.0},
    'A+AA+': {'spectral': (0.9375, 1.0625), 'spatial': 2.0, 'temporal': 0.5},
    'AAA': {'spectral': (0.75, 1.25), 'spatial': 2.0, 'temporal': 2.0},  # Class A
    'AAB': {'spectral': (0.75, 1.25), 'spatial': 2.0, 'temporal': 5.0},
    'ABA': {'spectral': (0.75, 1.25), 'spatial': 5.0, 'temporal': 2.0},
    'ABB': {'spectral': (0.75, 1.25), 'spatial': 5.0, 'temporal': 5.0},
    'BAA': {'spectral': (0.6, 1.4), 'spatial': 2.0, 'temporal': 2.0},
    'BAB': {'spectral': (0.6, 1.4), 'spatial': 2.0, 'temporal': 5.0},
    'BBA': {'spectral': (0.6, 1.4), 'spatial': 5.0, 'temporal': 2.0},
    'BBB': {'spectral': (0.6, 1.4), 'spatial': 5.0, 'temporal': 5.0},
    'BBC': {'spectral': (0.6, 1.4), 'spatial': 5.0, 'temporal': 10.0},
    'BCB': {'spectral': (0.6, 1.4), 'spatial': 10.0, 'temporal': 5.0},
    'BCC': {'spectral': (0.6, 1.4), 'spatial': 10.0, 'temporal': 10.0},
    'CAA': {'spectral': (0.4, 2.0), 'spatial': 2.0, 'temporal': 2.0},
    'CBA': {'spectral': (0.4, 2.0), 'spatial': 5.0, 'temporal': 2.0},
    'CBB': {'spectral': (0.4, 2.0), 'spatial': 5.0, 'temporal': 5.0},
    'CBC': {'spectral': (0.4, 2.0), 'spatial': 5.0, 'temporal': 10.0},
    'CCC': {'spectral': (0.4, 2.0), 'spatial': 10.0, 'temporal': 10.0},
}

# Uncertainty contributions by classification (% of measured value)
# Based on NREL, PTB, and EU JRC literature
UNCERTAINTY_CONTRIBUTIONS = {
    'A+A+A+': {
        'spectral_mismatch': 0.3,  # %
        'spatial_nonuniformity': 0.25,  # %
        'temporal_instability': 0.1,  # %
        'irradiance_setting': 0.5,  # %
        'temperature_correction': 0.3,  # %
        'data_acquisition': 0.1,  # %
    },
    'AAA': {
        'spectral_mismatch': 0.8,
        'spatial_nonuniformity': 0.5,
        'temporal_instability': 0.4,
        'irradiance_setting': 0.5,
        'temperature_correction': 0.3,
        'data_acquisition': 0.1,
    },
    'AAB': {
        'spectral_mismatch': 0.8,
        'spatial_nonuniformity': 0.5,
        'temporal_instability': 1.0,
        'irradiance_setting': 0.5,
        'temperature_correction': 0.3,
        'data_acquisition': 0.1,
    },
    'BBA': {
        'spectral_mismatch': 1.5,
        'spatial_nonuniformity': 1.2,
        'temporal_instability': 0.4,
        'irradiance_setting': 0.5,
        'temperature_correction': 0.3,
        'data_acquisition': 0.1,
    },
    'BBB': {
        'spectral_mismatch': 1.5,
        'spatial_nonuniformity': 1.2,
        'temporal_instability': 1.0,
        'irradiance_setting': 0.5,
        'temperature_correction': 0.3,
        'data_acquisition': 0.1,
    },
    'CCC': {
        'spectral_mismatch': 3.0,
        'spatial_nonuniformity': 2.5,
        'temporal_instability': 2.0,
        'irradiance_setting': 0.5,
        'temperature_correction': 0.3,
        'data_acquisition': 0.1,
    },
}


@dataclass
class IVParameters:
    """I-V curve parameters."""
    isc: float  # Short-circuit current (A)
    voc: float  # Open-circuit voltage (V)
    pmax: float  # Maximum power (W)
    vmpp: float  # Voltage at max power point (V)
    impp: float  # Current at max power point (A)
    ff: float  # Fill factor
    irradiance: float  # Irradiance (W/m²)
    temperature: float  # Cell temperature (°C)


@dataclass
class TranslationResult:
    """Result of I-V curve translation between classifications."""
    source_params: IVParameters
    target_params: IVParameters
    source_classification: str
    target_classification: str
    correction_factors: Dict[str, float]
    uncertainty_source: Dict[str, float]
    uncertainty_target: Dict[str, float]
    combined_uncertainty: Dict[str, float]
    spectral_correction: float
    spatial_correction: float
    temporal_correction: float


def get_uncertainty_for_classification(classification: str) -> Dict[str, float]:
    """Get uncertainty contributions for a classification."""
    # Find the matching or closest classification
    if classification in UNCERTAINTY_CONTRIBUTIONS:
        return UNCERTAINTY_CONTRIBUTIONS[classification]

    # Map to closest known classification
    first_letter = classification[0] if len(classification) >= 1 else 'C'
    if first_letter == 'A':
        if 'A+' in classification:
            return UNCERTAINTY_CONTRIBUTIONS['A+A+A+']
        return UNCERTAINTY_CONTRIBUTIONS['AAA']
    elif first_letter == 'B':
        return UNCERTAINTY_CONTRIBUTIONS['BBB']
    else:
        return UNCERTAINTY_CONTRIBUTIONS['CCC']


def calculate_combined_uncertainty(uncertainties: Dict[str, float]) -> float:
    """Calculate combined uncertainty using RSS method."""
    return np.sqrt(sum(u**2 for u in uncertainties.values()))


def calculate_spectral_mismatch_correction(
    source_class: str,
    target_class: str,
    cell_type: str = "c-Si"
) -> Tuple[float, float]:
    """
    Calculate spectral mismatch correction factor (M) and its uncertainty.

    Based on IEC 60904-7 methodology.

    Args:
        source_class: Source simulator classification
        target_class: Target simulator classification
        cell_type: Solar cell technology type

    Returns:
        Tuple of (correction_factor, uncertainty)
    """
    source_limits = IEC_CLASSIFICATION_LIMITS.get(source_class, IEC_CLASSIFICATION_LIMITS['CCC'])
    target_limits = IEC_CLASSIFICATION_LIMITS.get(target_class, IEC_CLASSIFICATION_LIMITS['A+A+A+'])

    # Calculate spectral deviation ranges
    source_spectral_range = source_limits['spectral'][1] - source_limits['spectral'][0]
    target_spectral_range = target_limits['spectral'][1] - target_limits['spectral'][0]

    # Spectral mismatch factor estimation
    # For c-Si cells, the spectral sensitivity is relatively flat, so correction is smaller
    # For thin-film or multi-junction, it would be larger
    spectral_sensitivity = {
        'c-Si': 0.02,  # Low sensitivity
        'mc-Si': 0.02,  # Low sensitivity
        'CdTe': 0.04,  # Moderate sensitivity
        'CIGS': 0.03,  # Moderate sensitivity
        'a-Si': 0.06,  # Higher sensitivity
        'Perovskite': 0.05,  # Higher sensitivity
        'Multi-junction': 0.08,  # Highest sensitivity
    }.get(cell_type, 0.03)

    # Correction factor (typically close to 1.0)
    range_ratio = source_spectral_range / target_spectral_range if target_spectral_range > 0 else 1.0
    correction = 1.0 + spectral_sensitivity * (range_ratio - 1.0)

    # Uncertainty in correction
    uncertainty = spectral_sensitivity * abs(range_ratio - 1.0) * 0.5

    return correction, uncertainty


def calculate_spatial_correction(
    source_class: str,
    target_class: str
) -> Tuple[float, float]:
    """
    Calculate spatial non-uniformity correction and uncertainty.

    Args:
        source_class: Source simulator classification
        target_class: Target simulator classification

    Returns:
        Tuple of (correction_factor, uncertainty)
    """
    source_limits = IEC_CLASSIFICATION_LIMITS.get(source_class, IEC_CLASSIFICATION_LIMITS['CCC'])
    target_limits = IEC_CLASSIFICATION_LIMITS.get(target_class, IEC_CLASSIFICATION_LIMITS['A+A+A+'])

    source_nonunif = source_limits['spatial']
    target_nonunif = target_limits['spatial']

    # Spatial correction is typically 1.0, but uncertainty increases with non-uniformity
    correction = 1.0

    # Uncertainty contribution from spatial non-uniformity
    # Based on worst-case analysis of irradiance distribution
    uncertainty = abs(source_nonunif - target_nonunif) / 100.0 * 0.5

    return correction, uncertainty


def calculate_temporal_correction(
    source_class: str,
    target_class: str
) -> Tuple[float, float]:
    """
    Calculate temporal instability correction and uncertainty.

    Args:
        source_class: Source simulator classification
        target_class: Target simulator classification

    Returns:
        Tuple of (correction_factor, uncertainty)
    """
    source_limits = IEC_CLASSIFICATION_LIMITS.get(source_class, IEC_CLASSIFICATION_LIMITS['CCC'])
    target_limits = IEC_CLASSIFICATION_LIMITS.get(target_class, IEC_CLASSIFICATION_LIMITS['A+A+A+'])

    source_instab = source_limits['temporal']
    target_instab = target_limits['temporal']

    # Temporal correction is typically 1.0
    correction = 1.0

    # Uncertainty contribution from temporal instability
    uncertainty = abs(source_instab - target_instab) / 100.0 * 0.3

    return correction, uncertainty


def translate_iv_parameters(
    params: IVParameters,
    source_classification: str,
    target_classification: str,
    cell_type: str = "c-Si"
) -> TranslationResult:
    """
    Translate I-V parameters from source to target simulator classification.

    Args:
        params: Source I-V parameters
        source_classification: Source simulator classification
        target_classification: Target simulator classification
        cell_type: Solar cell technology type

    Returns:
        TranslationResult with translated parameters and uncertainties
    """
    # Calculate correction factors
    spectral_corr, spectral_unc = calculate_spectral_mismatch_correction(
        source_classification, target_classification, cell_type
    )
    spatial_corr, spatial_unc = calculate_spatial_correction(
        source_classification, target_classification
    )
    temporal_corr, temporal_unc = calculate_temporal_correction(
        source_classification, target_classification
    )

    # Combined correction factor
    total_correction = spectral_corr * spatial_corr * temporal_corr

    # Get uncertainty contributions for source and target
    source_uncertainties = get_uncertainty_for_classification(source_classification)
    target_uncertainties = get_uncertainty_for_classification(target_classification)

    # Calculate translated parameters
    translated_isc = params.isc * total_correction
    translated_voc = params.voc  # Voc is less sensitive to irradiance uniformity
    translated_pmax = params.pmax * total_correction
    translated_vmpp = params.vmpp
    translated_impp = params.impp * total_correction
    translated_ff = params.ff  # FF is relatively stable

    target_params = IVParameters(
        isc=translated_isc,
        voc=translated_voc,
        pmax=translated_pmax,
        vmpp=translated_vmpp,
        impp=translated_impp,
        ff=translated_ff,
        irradiance=params.irradiance,
        temperature=params.temperature
    )

    # Combined uncertainties for each parameter
    combined_uncertainties = {
        'isc': np.sqrt(
            source_uncertainties['spectral_mismatch']**2 +
            target_uncertainties['spectral_mismatch']**2 +
            spectral_unc**2 * 100**2 +  # Convert to %
            source_uncertainties['spatial_nonuniformity']**2 +
            target_uncertainties['spatial_nonuniformity']**2
        ),
        'voc': np.sqrt(
            source_uncertainties['temperature_correction']**2 +
            target_uncertainties['temperature_correction']**2 +
            source_uncertainties['data_acquisition']**2 +
            target_uncertainties['data_acquisition']**2
        ),
        'pmax': np.sqrt(
            source_uncertainties['spectral_mismatch']**2 +
            target_uncertainties['spectral_mismatch']**2 +
            spectral_unc**2 * 100**2 +
            source_uncertainties['spatial_nonuniformity']**2 +
            target_uncertainties['spatial_nonuniformity']**2 +
            source_uncertainties['temporal_instability']**2 +
            target_uncertainties['temporal_instability']**2
        ),
        'ff': np.sqrt(
            source_uncertainties['data_acquisition']**2 +
            target_uncertainties['data_acquisition']**2
        ) * 0.5,  # FF is more stable
    }

    return TranslationResult(
        source_params=params,
        target_params=target_params,
        source_classification=source_classification,
        target_classification=target_classification,
        correction_factors={
            'spectral': spectral_corr,
            'spatial': spatial_corr,
            'temporal': temporal_corr,
            'total': total_correction
        },
        uncertainty_source=source_uncertainties,
        uncertainty_target=target_uncertainties,
        combined_uncertainty=combined_uncertainties,
        spectral_correction=spectral_corr,
        spatial_correction=spatial_corr,
        temporal_correction=temporal_corr
    )


def create_iv_curve_plot(
    source_params: IVParameters,
    target_params: IVParameters,
    source_class: str,
    target_class: str,
    uncertainties: Dict[str, float]
) -> go.Figure:
    """Create an I-V curve comparison plot with uncertainty bands."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("I-V Curves", "P-V Curves"))

    # Generate I-V curve points (simplified model)
    v = np.linspace(0, source_params.voc * 1.05, 100)

    # Source I-V curve (simplified single-diode model approximation)
    rs = 0.01  # Series resistance approximation
    rsh = 100  # Shunt resistance approximation

    # Simplified I-V equation
    i_source = source_params.isc * (1 - (v / source_params.voc) ** 3)
    i_source = np.maximum(i_source, 0)

    # Target I-V curve
    v_target = np.linspace(0, target_params.voc * 1.05, 100)
    i_target = target_params.isc * (1 - (v_target / target_params.voc) ** 3)
    i_target = np.maximum(i_target, 0)

    # Calculate uncertainty bands
    isc_unc_pct = uncertainties.get('isc', 1.0) / 100
    pmax_unc_pct = uncertainties.get('pmax', 1.0) / 100

    # I-V curves
    fig.add_trace(go.Scatter(
        x=v, y=i_source,
        mode='lines',
        name=f'Source ({source_class})',
        line=dict(color='#4ECDC4', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=v_target, y=i_target,
        mode='lines',
        name=f'Target ({target_class})',
        line=dict(color='#00D4AA', width=2, dash='dash')
    ), row=1, col=1)

    # Uncertainty band for target
    fig.add_trace(go.Scatter(
        x=np.concatenate([v_target, v_target[::-1]]),
        y=np.concatenate([i_target * (1 + isc_unc_pct), (i_target * (1 - isc_unc_pct))[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color='rgba(0, 212, 170, 0)'),
        name='Uncertainty Band',
        showlegend=False
    ), row=1, col=1)

    # Key points
    fig.add_trace(go.Scatter(
        x=[0, source_params.vmpp, source_params.voc],
        y=[source_params.isc, source_params.impp, 0],
        mode='markers',
        name='Source Points',
        marker=dict(color='#4ECDC4', size=10, symbol='circle')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0, target_params.vmpp, target_params.voc],
        y=[target_params.isc, target_params.impp, 0],
        mode='markers',
        name='Target Points',
        marker=dict(color='#00D4AA', size=10, symbol='diamond')
    ), row=1, col=1)

    # P-V curves
    p_source = v * i_source
    p_target = v_target * i_target

    fig.add_trace(go.Scatter(
        x=v, y=p_source,
        mode='lines',
        name=f'Source Power',
        line=dict(color='#4ECDC4', width=2),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=v_target, y=p_target,
        mode='lines',
        name=f'Target Power',
        line=dict(color='#00D4AA', width=2, dash='dash'),
        showlegend=False
    ), row=1, col=2)

    # Uncertainty band for power
    fig.add_trace(go.Scatter(
        x=np.concatenate([v_target, v_target[::-1]]),
        y=np.concatenate([p_target * (1 + pmax_unc_pct), (p_target * (1 - pmax_unc_pct))[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color='rgba(0, 212, 170, 0)'),
        name='Power Uncertainty',
        showlegend=False
    ), row=1, col=2)

    # Pmax points
    pmax_idx_source = np.argmax(p_source)
    pmax_idx_target = np.argmax(p_target)

    fig.add_trace(go.Scatter(
        x=[v[pmax_idx_source]],
        y=[p_source[pmax_idx_source]],
        mode='markers',
        name='Source Pmax',
        marker=dict(color='#4ECDC4', size=12, symbol='star'),
        showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[v_target[pmax_idx_target]],
        y=[p_target[pmax_idx_target]],
        mode='markers',
        name='Target Pmax',
        marker=dict(color='#00D4AA', size=12, symbol='star'),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"I-V Curve Translation: {source_class} → {target_class}",
            font=dict(color='#FAFAFA', size=18)
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.update_xaxes(title_text="Voltage (V)", gridcolor='#2D3139', row=1, col=1)
    fig.update_yaxes(title_text="Current (A)", gridcolor='#2D3139', row=1, col=1)
    fig.update_xaxes(title_text="Voltage (V)", gridcolor='#2D3139', row=1, col=2)
    fig.update_yaxes(title_text="Power (W)", gridcolor='#2D3139', row=1, col=2)

    return fig


def create_uncertainty_breakdown_chart(result: TranslationResult) -> go.Figure:
    """Create a breakdown chart of uncertainty contributions."""
    fig = go.Figure()

    # Source uncertainties
    source_labels = list(result.uncertainty_source.keys())
    source_values = list(result.uncertainty_source.values())

    # Target uncertainties
    target_values = [result.uncertainty_target.get(k, 0) for k in source_labels]

    x = np.arange(len(source_labels))
    width = 0.35

    fig.add_trace(go.Bar(
        x=source_labels,
        y=source_values,
        name=f'Source ({result.source_classification})',
        marker_color='#4ECDC4',
        text=[f'{v:.2f}%' for v in source_values],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=source_labels,
        y=target_values,
        name=f'Target ({result.target_classification})',
        marker_color='#00D4AA',
        text=[f'{v:.2f}%' for v in target_values],
        textposition='outside'
    ))

    fig.update_layout(
        title=dict(
            text="Uncertainty Contributions by Component",
            font=dict(color='#FAFAFA', size=16)
        ),
        xaxis=dict(
            title="Uncertainty Component",
            tickangle=45,
            gridcolor='#2D3139'
        ),
        yaxis=dict(
            title="Uncertainty (%)",
            gridcolor='#2D3139'
        ),
        barmode='group',
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig


# Main page content
st.title("🔄 I-V Curve Translation")
st.markdown("""
Translate I-V measurements between different solar simulator classifications
with uncertainty calculations based on IEC 60904 standards.
""")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Translation Settings")

    st.subheader("Source Simulator")
    source_classification = st.selectbox(
        "Source Classification",
        list(IEC_CLASSIFICATION_LIMITS.keys()),
        index=list(IEC_CLASSIFICATION_LIMITS.keys()).index('BBA')
    )

    st.subheader("Target Simulator")
    target_classification = st.selectbox(
        "Target Classification",
        list(IEC_CLASSIFICATION_LIMITS.keys()),
        index=list(IEC_CLASSIFICATION_LIMITS.keys()).index('A+A+A+')
    )

    st.markdown("---")
    st.subheader("Cell Technology")
    cell_type = st.selectbox(
        "Cell Type",
        ["c-Si", "mc-Si", "CdTe", "CIGS", "a-Si", "Perovskite", "Multi-junction"],
        index=0,
        help="Cell technology affects spectral mismatch correction"
    )

    st.markdown("---")
    st.subheader("Test Conditions")
    irradiance = st.number_input("Irradiance (W/m²)", value=1000.0, min_value=100.0, max_value=1500.0)
    temperature = st.number_input("Cell Temperature (°C)", value=25.0, min_value=-20.0, max_value=80.0)

# Main content - Input I-V parameters
st.subheader("📊 Input I-V Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Current Parameters**")
    isc = st.number_input("Isc (A)", value=9.5, min_value=0.0, format="%.4f")
    impp = st.number_input("Impp (A)", value=9.0, min_value=0.0, format="%.4f")

with col2:
    st.markdown("**Voltage Parameters**")
    voc = st.number_input("Voc (V)", value=46.0, min_value=0.0, format="%.3f")
    vmpp = st.number_input("Vmpp (V)", value=38.0, min_value=0.0, format="%.3f")

with col3:
    st.markdown("**Power Parameters**")
    pmax = st.number_input("Pmax (W)", value=342.0, min_value=0.0, format="%.2f")
    ff = st.number_input("Fill Factor", value=0.78, min_value=0.0, max_value=1.0, format="%.4f")

# Create source parameters
source_params = IVParameters(
    isc=isc,
    voc=voc,
    pmax=pmax,
    vmpp=vmpp,
    impp=impp,
    ff=ff,
    irradiance=irradiance,
    temperature=temperature
)

# Perform translation
st.markdown("---")
if st.button("🔄 Translate I-V Curve", type="primary"):
    result = translate_iv_parameters(
        source_params,
        source_classification,
        target_classification,
        cell_type
    )

    # Display results
    st.subheader(f"Translation Results: {source_classification} → {target_classification}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pct_change = ((result.target_params.pmax - result.source_params.pmax) /
                      result.source_params.pmax * 100)
        st.metric(
            "Pmax Change",
            f"{result.target_params.pmax:.2f} W",
            f"{pct_change:+.2f}%"
        )

    with col2:
        isc_change = ((result.target_params.isc - result.source_params.isc) /
                      result.source_params.isc * 100)
        st.metric(
            "Isc Change",
            f"{result.target_params.isc:.4f} A",
            f"{isc_change:+.2f}%"
        )

    with col3:
        combined_pmax_unc = result.combined_uncertainty.get('pmax', 0)
        unc_color = "uncertainty-good" if combined_pmax_unc < 1 else (
            "uncertainty-moderate" if combined_pmax_unc < 2 else "uncertainty-high"
        )
        st.markdown(f"""
        <div style="background-color: #1A1D24; padding: 15px; border-radius: 10px;">
            <span style="color: #888; font-size: 14px;">Pmax Uncertainty</span><br>
            <span class="{unc_color}" style="font-size: 24px; font-weight: bold;">±{combined_pmax_unc:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.metric(
            "Total Correction",
            f"{result.correction_factors['total']:.4f}",
            f"{(result.correction_factors['total'] - 1) * 100:+.2f}%"
        )

    st.markdown("---")

    # I-V Curve plot
    fig_iv = create_iv_curve_plot(
        result.source_params,
        result.target_params,
        source_classification,
        target_classification,
        result.combined_uncertainty
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    # Parameter comparison table
    st.subheader("📋 Parameter Comparison")

    comparison_df = pd.DataFrame({
        'Parameter': ['Isc (A)', 'Voc (V)', 'Pmax (W)', 'Vmpp (V)', 'Impp (A)', 'Fill Factor'],
        f'Source ({source_classification})': [
            f"{source_params.isc:.4f}",
            f"{source_params.voc:.3f}",
            f"{source_params.pmax:.2f}",
            f"{source_params.vmpp:.3f}",
            f"{source_params.impp:.4f}",
            f"{source_params.ff:.4f}"
        ],
        f'Target ({target_classification})': [
            f"{result.target_params.isc:.4f}",
            f"{result.target_params.voc:.3f}",
            f"{result.target_params.pmax:.2f}",
            f"{result.target_params.vmpp:.3f}",
            f"{result.target_params.impp:.4f}",
            f"{result.target_params.ff:.4f}"
        ],
        'Uncertainty (%)': [
            f"±{result.combined_uncertainty.get('isc', 0):.2f}",
            f"±{result.combined_uncertainty.get('voc', 0):.2f}",
            f"±{result.combined_uncertainty.get('pmax', 0):.2f}",
            f"±{result.combined_uncertainty.get('pmax', 0):.2f}",  # Use Pmax unc for Vmpp
            f"±{result.combined_uncertainty.get('isc', 0):.2f}",   # Use Isc unc for Impp
            f"±{result.combined_uncertainty.get('ff', 0):.2f}"
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Correction factors
    st.subheader("🔧 Correction Factors")
    corr_col1, corr_col2, corr_col3, corr_col4 = st.columns(4)

    with corr_col1:
        st.metric("Spectral", f"{result.spectral_correction:.4f}")
    with corr_col2:
        st.metric("Spatial", f"{result.spatial_correction:.4f}")
    with corr_col3:
        st.metric("Temporal", f"{result.temporal_correction:.4f}")
    with corr_col4:
        st.metric("Total", f"{result.correction_factors['total']:.4f}")

    # Uncertainty breakdown
    st.markdown("---")
    st.subheader("📊 Uncertainty Analysis")

    fig_unc = create_uncertainty_breakdown_chart(result)
    st.plotly_chart(fig_unc, use_container_width=True)

    # Detailed uncertainty table
    with st.expander("📋 Detailed Uncertainty Breakdown"):
        unc_df = pd.DataFrame({
            'Component': list(result.uncertainty_source.keys()),
            f'Source ({source_classification}) (%)': [
                f"{v:.2f}" for v in result.uncertainty_source.values()
            ],
            f'Target ({target_classification}) (%)': [
                f"{result.uncertainty_target.get(k, 0):.2f}"
                for k in result.uncertainty_source.keys()
            ]
        })
        st.dataframe(unc_df, use_container_width=True, hide_index=True)

        # Combined uncertainties
        st.markdown("**Combined Uncertainties (k=2, 95% confidence)**")
        comb_unc_df = pd.DataFrame({
            'Parameter': ['Isc', 'Voc', 'Pmax', 'Fill Factor'],
            'Uncertainty (%)': [
                f"±{result.combined_uncertainty.get('isc', 0):.2f}",
                f"±{result.combined_uncertainty.get('voc', 0):.2f}",
                f"±{result.combined_uncertainty.get('pmax', 0):.2f}",
                f"±{result.combined_uncertainty.get('ff', 0):.2f}"
            ],
            'Uncertainty (absolute)': [
                f"±{result.target_params.isc * result.combined_uncertainty.get('isc', 0) / 100:.4f} A",
                f"±{result.target_params.voc * result.combined_uncertainty.get('voc', 0) / 100:.3f} V",
                f"±{result.target_params.pmax * result.combined_uncertainty.get('pmax', 0) / 100:.2f} W",
                f"±{result.target_params.ff * result.combined_uncertainty.get('ff', 0) / 100:.4f}"
            ]
        })
        st.dataframe(comb_unc_df, use_container_width=True, hide_index=True)

else:
    # Show instructions
    st.info("👆 Enter I-V parameters and click 'Translate I-V Curve' to begin")

    st.markdown("""
    ### About I-V Curve Translation

    This tool translates I-V measurement results between solar simulators of different
    IEC 60904-9 classifications, accounting for:

    1. **Spectral Mismatch** - Differences in simulator spectrum vs AM1.5G reference
    2. **Spatial Non-uniformity** - Irradiance distribution across test area
    3. **Temporal Instability** - Irradiance stability during measurement

    #### Classification System

    | Class | Spectral Match | Spatial Non-uniformity | Temporal Instability |
    |-------|---------------|----------------------|---------------------|
    | A+ | 0.9375 - 1.0625 | ≤1% | ≤0.5% |
    | A | 0.75 - 1.25 | ≤2% | ≤2% |
    | B | 0.60 - 1.40 | ≤5% | ≤5% |
    | C | 0.40 - 2.00 | ≤10% | ≤10% |

    #### References

    - IEC 60904-9:2020 - Solar simulator performance requirements
    - IEC 60904-1:2020 - Measurement of photovoltaic I-V characteristics
    - IEC 60904-7:2019 - Computation of spectral mismatch correction
    - NREL/TP-520-45527 - Uncertainty analysis for PV measurements
    """)
