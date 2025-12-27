"""
Sun Simulator Classification System
Spectral Analysis Page - Spectral Match Classification

This page provides spectral match analysis with:
- File upload for spectral data
- 6-band classification table
- AM1.5G comparison chart
- SPC/SPD calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_CONFIG, THEME, BADGE_COLORS, BADGE_COLORS_LIGHT,
    WAVELENGTH_BANDS, CLASSIFICATION, get_classification
)
from utils.calculations import (
    SpectralCalculator, load_am15g_reference,
    generate_sample_spectral_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Spectral Analysis - " + APP_CONFIG['title'],
    page_icon="🌈",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] {
    background: #1e293b !important;
}

.page-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #475569;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
}

.result-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.result-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
}

.classification-large {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    border-radius: 16px;
    font-size: 2rem;
    font-weight: 700;
    color: white;
}

.badge-aplus { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
.badge-a { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
.badge-b { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
.badge-c { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }

.band-table {
    width: 100%;
    border-collapse: collapse;
}

.band-table th {
    background: #334155;
    color: #f8fafc;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.band-table td {
    padding: 0.75rem;
    border-bottom: 1px solid #475569;
    color: #e2e8f0;
}

.band-table tr:hover {
    background: rgba(59, 130, 246, 0.1);
}

.metric-box {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #3b82f6;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
}

.upload-zone {
    border: 2px dashed #475569;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: rgba(30, 41, 59, 0.5);
    transition: all 0.2s;
}

.upload-zone:hover {
    border-color: #3b82f6;
    background: rgba(59, 130, 246, 0.1);
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_badge_class(classification: str) -> str:
    """Get CSS class for classification badge."""
    return {
        'A+': 'badge-aplus',
        'A': 'badge-a',
        'B': 'badge-b',
        'C': 'badge-c'
    }.get(classification, 'badge-c')


def create_band_classification_table(band_details: list) -> str:
    """Create HTML table for band classifications."""
    rows = ""
    for band in band_details:
        badge_class = get_badge_class(band['classification'])
        deviation_color = '#10b981' if abs(band['deviation_percent']) < 12.5 else \
                          '#3b82f6' if abs(band['deviation_percent']) < 25 else \
                          '#f59e0b' if abs(band['deviation_percent']) < 40 else '#ef4444'

        rows += f"""
        <tr>
            <td><strong>Band {band['band']}</strong></td>
            <td>{band['range']}</td>
            <td>{band['name']}</td>
            <td>{band['measured_irradiance']:.2f}</td>
            <td>{band['reference_irradiance']:.2f}</td>
            <td>{band['ratio']:.3f}</td>
            <td style="color: {deviation_color};">{band['deviation_percent']:+.1f}%</td>
            <td>
                <span class="classification-large {badge_class}"
                      style="width: 40px; height: 30px; font-size: 0.9rem; border-radius: 6px;">
                    {band['classification']}
                </span>
            </td>
        </tr>
        """

    return f"""
    <table class="band-table">
        <thead>
            <tr>
                <th>Band</th>
                <th>Range</th>
                <th>Name</th>
                <th>Measured (W/m²)</th>
                <th>Reference (W/m²)</th>
                <th>Ratio</th>
                <th>Deviation</th>
                <th>Class</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🌈 Spectral Match Analysis</div>
        <div class="page-subtitle">
            Analyze spectral irradiance distribution across IEC 60904-9 wavelength bands
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'spectral_data' not in st.session_state:
        st.session_state.spectral_data = None
    if 'spectral_result' not in st.session_state:
        st.session_state.spectral_result = None

    # Sidebar options
    with st.sidebar:
        st.markdown("### Analysis Options")

        use_sample_data = st.checkbox("Use Sample Data", value=True,
                                       help="Use generated sample data for demonstration")

        if use_sample_data:
            noise_level = st.slider("Noise Level", 0.0, 0.3, 0.1, 0.01,
                                    help="Relative noise level for sample data")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        if use_sample_data:
            if st.button("Generate Sample Spectral Data", type="primary"):
                wavelength, irradiance = generate_sample_spectral_data(noise_level)
                st.session_state.spectral_data = {
                    'wavelength': wavelength,
                    'irradiance': irradiance
                }
                st.success("Sample spectral data generated!")
        else:
            uploaded_file = st.file_uploader(
                "Upload Spectral Data (CSV)",
                type=['csv', 'txt'],
                help="CSV file with 'wavelength' (nm) and 'irradiance' (W/m²/nm) columns"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) >= 2:
                        df.columns = ['wavelength', 'irradiance'] + list(df.columns[2:])
                        st.session_state.spectral_data = {
                            'wavelength': df['wavelength'].values,
                            'irradiance': df['irradiance'].values
                        }
                        st.success(f"Loaded {len(df)} data points")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    with col2:
        st.markdown('<div class="section-title">File Format</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.8); padding: 1rem; border-radius: 8px;
                    border: 1px solid #334155; color: #94a3b8; font-size: 0.85rem;">
            <strong style="color: #f8fafc;">Required columns:</strong><br>
            • wavelength (nm): 300-1200<br>
            • irradiance (W/m²/nm)<br><br>
            <strong style="color: #f8fafc;">Example:</strong><br>
            <code style="color: #10b981;">
            wavelength,irradiance<br>
            300,0.05<br>
            301,0.06<br>
            ...
            </code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis Section
    if st.session_state.spectral_data is not None:
        data = st.session_state.spectral_data
        wavelength = data['wavelength']
        irradiance = data['irradiance']

        # Run analysis
        calculator = SpectralCalculator()
        result = calculator.calculate_spectral_match(wavelength, irradiance)
        st.session_state.spectral_result = result

        # Results Overview
        st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            badge_class = get_badge_class(result.overall_classification)
            st.markdown(f"""
            <div class="result-card" style="text-align: center;">
                <div class="result-title">Overall Classification</div>
                <div class="classification-large {badge_class}">
                    {result.overall_classification}
                </div>
                <div style="margin-top: 0.5rem; color: #94a3b8; font-size: 0.85rem;">
                    Spectral Match
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{result.max_deviation:.1f}%</div>
                    <div class="metric-label">Max Deviation</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            spc_color = '#10b981' if 0.98 <= result.spc <= 1.02 else '#f59e0b'
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {spc_color}40; background: {spc_color}10;">
                    <div class="metric-value" style="color: {spc_color};">{result.spc:.4f}</div>
                    <div class="metric-label">SPC</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            spd_color = '#10b981' if result.spd < 10 else '#f59e0b' if result.spd < 20 else '#ef4444'
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {spd_color}40; background: {spd_color}10;">
                    <div class="metric-value" style="color: {spd_color};">{result.spd:.2f}%</div>
                    <div class="metric-label">SPD</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Band Classification Table
        st.markdown('<div class="section-title">Wavelength Band Analysis</div>', unsafe_allow_html=True)
        st.markdown(create_band_classification_table(result.band_details), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Spectral Comparison Chart
        st.markdown('<div class="section-title">Spectral Comparison with AM1.5G Reference</div>',
                    unsafe_allow_html=True)

        # Load reference
        reference = load_am15g_reference()

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Spectral Irradiance', 'Band Ratios'),
            vertical_spacing=0.15,
            row_heights=[0.65, 0.35]
        )

        # Spectral plot
        fig.add_trace(
            go.Scatter(
                x=reference['wavelength'],
                y=reference['irradiance'],
                name='AM1.5G Reference',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=irradiance,
                name='Measured',
                line=dict(color='#3b82f6', width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)'
            ),
            row=1, col=1
        )

        # Add band regions
        for i, (start, end, name) in enumerate(WAVELENGTH_BANDS):
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor='rgba(100, 100, 100, 0.1)',
                line_width=0,
                annotation_text=name,
                annotation_position="top",
                annotation=dict(font_size=9, font_color='#64748b'),
                row=1, col=1
            )

        # Band ratios bar chart
        bands = [f"Band {b['band']}" for b in result.band_details]
        ratios = [b['ratio'] for b in result.band_details]
        colors = [BADGE_COLORS[b['classification']] for b in result.band_details]

        fig.add_trace(
            go.Bar(
                x=bands,
                y=ratios,
                marker_color=colors,
                text=[f"{r:.3f}" for r in ratios],
                textposition='outside',
                textfont=dict(color='#f8fafc'),
                showlegend=False
            ),
            row=2, col=1
        )

        # Add reference lines for classification limits
        for limit, color, name in [
            (1.0, '#10b981', 'Ideal'),
            (0.875, '#3b82f6', 'A+ limit'),
            (1.125, '#3b82f6', ''),
            (0.75, '#f59e0b', 'A limit'),
            (1.25, '#f59e0b', ''),
        ]:
            fig.add_hline(
                y=limit,
                line_dash='dot',
                line_color=color,
                line_width=1,
                row=2, col=1
            )

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            margin=dict(l=60, r=40, t=60, b=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            font=dict(family='Inter', color='#e2e8f0')
        )

        fig.update_xaxes(
            title_text="Wavelength (nm)",
            showgrid=True,
            gridcolor='rgba(71,85,105,0.3)',
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Irradiance (W/m²/nm)",
            showgrid=True,
            gridcolor='rgba(71,85,105,0.3)',
            row=1, col=1
        )
        fig.update_xaxes(
            showgrid=False,
            row=2, col=1
        )
        fig.update_yaxes(
            title_text="Ratio",
            showgrid=True,
            gridcolor='rgba(71,85,105,0.3)',
            range=[0.5, 1.5],
            row=2, col=1
        )

        st.plotly_chart(fig, use_container_width=True)

        # Classification Limits Reference
        st.markdown('<div class="section-title">IEC 60904-9 Classification Limits</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Spectral Match Limits</h4>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> A+</td>
                        <td>0.875 - 1.125 (±12.5%)</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> A</td>
                        <td>0.75 - 1.25 (±25%)</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> B</td>
                        <td>0.6 - 1.4 (±40%)</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> C</td>
                        <td>Outside B limits</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Key Metrics Explained</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    <strong style="color: #3b82f6;">SPC</strong> - Spectral Performance Category:
                    Weighted mismatch factor for typical c-Si response. Ideal = 1.0<br><br>
                    <strong style="color: #3b82f6;">SPD</strong> - Spectral Performance Deviation:
                    RMS deviation of all band ratios. Lower is better.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # No data loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No Spectral Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV file or generate sample data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
