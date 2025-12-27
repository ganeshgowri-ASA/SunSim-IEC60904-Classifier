"""
Sun Simulator Classification System
Temporal Stability Page - STI/LTI Classification

This page provides temporal stability analysis with:
- Short-Term Instability (STI) analysis
- Long-Term Instability (LTI) analysis
- Pulse profile visualization
- IEC 60904-9 classification
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
    APP_CONFIG, THEME, BADGE_COLORS, CLASSIFICATION,
    get_classification, get_overall_classification, MEASUREMENT_CONFIG
)
from utils.calculations import (
    TemporalCalculator, generate_sample_temporal_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Temporal Stability - " + APP_CONFIG['title'],
    page_icon="⏱️",
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

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}

.config-panel {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
}

.pulse-info {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #334155;
    color: #e2e8f0;
}

.pulse-info:last-child {
    border-bottom: none;
}

.sti-lti-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

.stability-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.stability-title {
    font-size: 1rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.stability-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.stability-class {
    display: inline-block;
    padding: 0.25rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
}

.class-aplus { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.class-a { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
.class-b { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.class-c { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
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


def get_class_css(classification: str) -> str:
    """Get CSS class for pills."""
    return {
        'A+': 'class-aplus',
        'A': 'class-a',
        'B': 'class-b',
        'C': 'class-c'
    }.get(classification, 'class-c')


def create_pulse_profile_chart(time: np.ndarray, irradiance: np.ndarray,
                                normalized: np.ndarray) -> go.Figure:
    """Create pulse profile visualization."""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Irradiance Profile', 'Normalized Profile'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # Irradiance profile
    fig.add_trace(
        go.Scatter(
            x=time * 1000,  # Convert to ms
            y=irradiance,
            name='Irradiance',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)',
            hovertemplate='Time: %{x:.3f} ms<br>Irradiance: %{y:.1f} W/m²<extra></extra>'
        ),
        row=1, col=1
    )

    # Mean line
    mean_irr = np.mean(irradiance)
    fig.add_hline(
        y=mean_irr,
        line_dash='dash',
        line_color='#10b981',
        annotation_text=f'Mean: {mean_irr:.1f} W/m²',
        annotation_position='right',
        row=1, col=1
    )

    # Normalized profile
    fig.add_trace(
        go.Scatter(
            x=time * 1000,
            y=normalized,
            name='Normalized',
            line=dict(color='#f59e0b', width=2),
            hovertemplate='Time: %{x:.3f} ms<br>Normalized: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Reference line at 1.0
    fig.add_hline(y=1.0, line_dash='dash', line_color='#10b981', row=2, col=1)

    # Tolerance bands
    for offset, color in [(0.02, 'rgba(59, 130, 246, 0.2)'),
                          (0.05, 'rgba(245, 158, 11, 0.2)')]:
        fig.add_hrect(
            y0=1 - offset, y1=1 + offset,
            fillcolor=color,
            line_width=0,
            row=2, col=1
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=60, r=40, t=60, b=40),
        font=dict(family='Inter', color='#e2e8f0'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        showlegend=True
    )

    fig.update_xaxes(title_text="Time (ms)", showgrid=True,
                     gridcolor='rgba(71,85,105,0.3)', row=1, col=1)
    fig.update_xaxes(title_text="Time (ms)", showgrid=True,
                     gridcolor='rgba(71,85,105,0.3)', row=2, col=1)
    fig.update_yaxes(title_text="Irradiance (W/m²)", showgrid=True,
                     gridcolor='rgba(71,85,105,0.3)', row=1, col=1)
    fig.update_yaxes(title_text="Normalized", showgrid=True,
                     gridcolor='rgba(71,85,105,0.3)', row=2, col=1)

    return fig


def create_stability_analysis_chart(time: np.ndarray, irradiance: np.ndarray,
                                     sti: float, lti: float) -> go.Figure:
    """Create stability analysis chart with rolling statistics."""

    # Calculate rolling statistics
    window = max(1, len(irradiance) // 50)
    rolling_mean = pd.Series(irradiance).rolling(window, center=True).mean()
    rolling_std = pd.Series(irradiance).rolling(window, center=True).std()
    rolling_max = pd.Series(irradiance).rolling(window, center=True).max()
    rolling_min = pd.Series(irradiance).rolling(window, center=True).min()

    fig = go.Figure()

    # Add range band
    fig.add_trace(go.Scatter(
        x=np.concatenate([time * 1000, time[::-1] * 1000]),
        y=np.concatenate([rolling_max.values, rolling_min.values[::-1]]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Range',
        hoverinfo='skip'
    ))

    # Rolling mean
    fig.add_trace(go.Scatter(
        x=time * 1000,
        y=rolling_mean,
        name='Rolling Mean',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='Time: %{x:.3f} ms<br>Mean: %{y:.1f} W/m²<extra></extra>'
    ))

    # Overall mean
    overall_mean = np.mean(irradiance)
    fig.add_hline(
        y=overall_mean,
        line_dash='dash',
        line_color='#10b981',
        annotation_text=f'Overall Mean: {overall_mean:.1f}'
    )

    # STI/LTI bands
    fig.add_hline(y=overall_mean * (1 + sti/100), line_dash='dot', line_color='#f59e0b',
                  annotation_text=f'+STI ({sti:.2f}%)')
    fig.add_hline(y=overall_mean * (1 - sti/100), line_dash='dot', line_color='#f59e0b')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=40),
        font=dict(family='Inter', color='#e2e8f0'),
        xaxis_title="Time (ms)",
        yaxis_title="Irradiance (W/m²)",
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_histogram_chart(irradiance: np.ndarray) -> go.Figure:
    """Create histogram of irradiance values."""

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=irradiance,
        nbinsx=50,
        marker_color='#3b82f6',
        opacity=0.8,
        hovertemplate='Irradiance: %{x:.1f}<br>Count: %{y}<extra></extra>'
    ))

    mean_val = np.mean(irradiance)
    std_val = np.std(irradiance)

    fig.add_vline(x=mean_val, line_dash='dash', line_color='#10b981',
                  annotation_text=f'Mean: {mean_val:.1f}')
    fig.add_vline(x=mean_val - std_val, line_dash='dot', line_color='#f59e0b')
    fig.add_vline(x=mean_val + std_val, line_dash='dot', line_color='#f59e0b')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=60, r=40, t=40, b=40),
        xaxis_title="Irradiance (W/m²)",
        yaxis_title="Count",
        font=dict(family='Inter', color='#e2e8f0'),
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">⏱️ Temporal Stability Analysis</div>
        <div class="page-subtitle">
            Analyze short-term (STI) and long-term (LTI) temporal instability per IEC 60904-9
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'temporal_data' not in st.session_state:
        st.session_state.temporal_data = None
    if 'temporal_result' not in st.session_state:
        st.session_state.temporal_result = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Measurement Settings")

        sample_rate = st.number_input(
            "Sample Rate (Hz)",
            min_value=1000, max_value=1000000, value=100000,
            help="Data acquisition sample rate"
        )

        st.markdown("### Analysis Windows")

        sti_window = st.slider(
            "STI Window (ms)",
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            help="Short-term instability analysis window"
        )

        lti_window = st.slider(
            "LTI Window (s)",
            min_value=1.0, max_value=120.0, value=60.0, step=1.0,
            help="Long-term instability analysis window"
        )

        st.markdown("### Sample Data")

        pulse_duration = st.slider(
            "Pulse Duration (ms)",
            min_value=1.0, max_value=100.0, value=10.0, step=1.0,
            help="Flash pulse duration for sample data"
        )

        target_instability = st.slider(
            "Target Instability (%)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            help="Target instability for sample data"
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Generate Sample"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Temporal Data (CSV)",
                type=['csv', 'txt'],
                help="CSV file with 'time' (s) and 'irradiance' (W/m²) columns"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) >= 2:
                        df.columns = ['time', 'irradiance'] + list(df.columns[2:])
                        st.session_state.temporal_data = {
                            'time': df['time'].values,
                            'irradiance': df['irradiance'].values
                        }
                        st.success(f"Loaded {len(df)} data points")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with tab2:
            if st.button("Generate Sample Flash Data", type="primary"):
                time, irradiance = generate_sample_temporal_data(
                    duration_s=pulse_duration / 1000,
                    sample_rate=sample_rate,
                    instability=target_instability
                )
                st.session_state.temporal_data = {
                    'time': time,
                    'irradiance': irradiance
                }
                st.success(f"Generated {len(time)} sample points")

    with col2:
        st.markdown('<div class="section-title">Analysis Info</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="config-panel">
            <div class="pulse-info">
                <span style="color: #94a3b8;">Sample Rate</span>
                <span style="color: #f8fafc; font-weight: 600;">{sample_rate:,} Hz</span>
            </div>
            <div class="pulse-info">
                <span style="color: #94a3b8;">STI Window</span>
                <span style="color: #f8fafc; font-weight: 600;">{sti_window} ms</span>
            </div>
            <div class="pulse-info">
                <span style="color: #94a3b8;">LTI Window</span>
                <span style="color: #f8fafc; font-weight: 600;">{lti_window} s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis Results
    if st.session_state.temporal_data is not None:
        data = st.session_state.temporal_data
        time = data['time']
        irradiance = data['irradiance']

        # Calculate temporal stability
        result = TemporalCalculator.calculate_temporal_stability(
            time, irradiance,
            sample_rate=sample_rate,
            sti_window_ms=sti_window,
            lti_window_s=lti_window
        )
        st.session_state.temporal_result = result

        # Analyze pulse shape
        pulse_info = TemporalCalculator.analyze_pulse_shape(time, irradiance)

        # Results Overview
        st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])

        with col1:
            badge_class = get_badge_class(result.overall_classification)
            st.markdown(f"""
            <div class="result-card" style="text-align: center; height: 200px;
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    Overall Temporal
                </div>
                <div class="classification-large {badge_class}">
                    {result.overall_classification}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            sti_color = BADGE_COLORS[result.sti_classification]
            sti_class_css = get_class_css(result.sti_classification)
            st.markdown(f"""
            <div class="stability-card">
                <div class="stability-title">Short-Term Instability (STI)</div>
                <div class="stability-value" style="color: {sti_color};">
                    {result.sti:.3f}%
                </div>
                <span class="stability-class {sti_class_css}">
                    Class {result.sti_classification}
                </span>
                <div style="margin-top: 1rem; color: #64748b; font-size: 0.8rem;">
                    Within {sti_window}ms windows
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            lti_color = BADGE_COLORS[result.lti_classification]
            lti_class_css = get_class_css(result.lti_classification)
            st.markdown(f"""
            <div class="stability-card">
                <div class="stability-title">Long-Term Instability (LTI)</div>
                <div class="stability-value" style="color: {lti_color};">
                    {result.lti:.3f}%
                </div>
                <span class="stability-class {lti_class_css}">
                    Class {result.lti_classification}
                </span>
                <div style="margin-top: 1rem; color: #64748b; font-size: 0.8rem;">
                    Over full measurement period
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pulse Profile Visualization
        st.markdown('<div class="section-title">Pulse Profile Analysis</div>', unsafe_allow_html=True)

        fig = create_pulse_profile_chart(time, irradiance, result.pulse_profile)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Stability Analysis</div>', unsafe_allow_html=True)
            fig = create_stability_analysis_chart(time, irradiance, result.sti, result.lti)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Irradiance Distribution</div>', unsafe_allow_html=True)
            fig = create_histogram_chart(irradiance)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pulse Characteristics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Pulse Characteristics</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Pulse Width</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {pulse_info.get('pulse_width_ms', 0):.3f} ms
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Rise Time (10-90%)</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {pulse_info.get('rise_time_ms', 0):.3f} ms
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Fall Time (90-10%)</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {pulse_info.get('fall_time_ms', 0):.3f} ms
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Peak Irradiance</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {pulse_info.get('peak_irradiance', 0):.1f} W/m²
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Mean Irradiance</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {pulse_info.get('mean_irradiance', 0):.1f} W/m²
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)

            mean_irr = np.mean(irradiance)
            std_irr = np.std(irradiance)
            max_irr = np.max(irradiance)
            min_irr = np.min(irradiance)

            st.markdown(f"""
            <div class="result-card">
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Data Points</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {len(irradiance):,}
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Duration</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {(time[-1] - time[0]) * 1000:.3f} ms
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Mean</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {mean_irr:.2f} W/m²
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Std Dev</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {std_irr:.2f} W/m²
                    </span>
                </div>
                <div class="pulse-info">
                    <span style="color: #94a3b8;">Range</span>
                    <span style="color: #f8fafc; font-weight: 600;">
                        {min_irr:.1f} - {max_irr:.1f} W/m²
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Classification Limits Reference
        st.markdown('<div class="section-title">IEC 60904-9 Temporal Stability Limits</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">STI Classification</h4>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> A+</td>
                        <td>≤ 0.5%</td>
                        <td>{'✓' if result.sti <= 0.5 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> A</td>
                        <td>≤ 2.0%</td>
                        <td>{'✓' if 0.5 < result.sti <= 2.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> B</td>
                        <td>≤ 5.0%</td>
                        <td>{'✓' if 2.0 < result.sti <= 5.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> C</td>
                        <td>> 5.0%</td>
                        <td>{'✓' if result.sti > 5.0 else ''}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">LTI Classification</h4>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> A+</td>
                        <td>≤ 1.0%</td>
                        <td>{'✓' if result.lti <= 1.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> A</td>
                        <td>≤ 2.0%</td>
                        <td>{'✓' if 1.0 < result.lti <= 2.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> B</td>
                        <td>≤ 5.0%</td>
                        <td>{'✓' if 2.0 < result.lti <= 5.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> C</td>
                        <td>> 5.0%</td>
                        <td>{'✓' if result.lti > 5.0 else ''}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # Calculation Method
        st.markdown("""
        <div class="result-card" style="margin-top: 1rem;">
            <h4 style="color: #f8fafc; margin-bottom: 1rem;">Calculation Method (IEC 60904-9)</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Temporal instability is calculated as:<br><br>
                <code style="color: #3b82f6; background: rgba(59,130,246,0.1);
                             padding: 0.5rem; border-radius: 4px; display: block;">
                Instability = (E_max - E_min) / (E_max + E_min) × 100%
                </code><br>
                <strong style="color: #f8fafc;">STI:</strong> Evaluated within short time windows (typically 1ms)
                to capture fast fluctuations during the flash pulse.<br><br>
                <strong style="color: #f8fafc;">LTI:</strong> Evaluated over the entire measurement period
                to capture drift and longer-term variations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # No data loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⏱️</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No Temporal Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV file or generate sample flash data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
