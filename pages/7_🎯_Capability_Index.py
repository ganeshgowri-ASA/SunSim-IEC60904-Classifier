"""
Sun Simulator Classification System
Capability Index Page - Process Capability Analysis

This page provides capability analysis with:
- Visual gauge displays for Cp, Cpk, Pp, Ppk
- Process sigma level
- Capability trending charts
- Target specifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_CONFIG, THEME, BADGE_COLORS
from utils.spc_calculations import (
    CapabilityCalculator, HistogramCalculator,
    generate_capability_sample_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Capability Index - " + APP_CONFIG['title'],
    page_icon="🎯",
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

.capability-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}

.capability-value {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1.2;
}

.capability-label {
    font-size: 1.25rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.5rem;
}

.capability-rating {
    display: inline-block;
    padding: 0.25rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 0.75rem;
}

.rating-excellent {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.rating-good {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
}

.rating-marginal {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.rating-poor {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
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

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}

.sigma-display {
    font-size: 5rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
}

.spec-bar {
    background: linear-gradient(90deg,
        rgba(239, 68, 68, 0.3) 0%,
        rgba(245, 158, 11, 0.3) 25%,
        rgba(16, 185, 129, 0.3) 50%,
        rgba(245, 158, 11, 0.3) 75%,
        rgba(239, 68, 68, 0.3) 100%);
    height: 60px;
    border-radius: 8px;
    position: relative;
    margin: 2rem 0;
}

.capability-table {
    width: 100%;
    border-collapse: collapse;
}

.capability-table th {
    background: #334155;
    color: #f8fafc;
    padding: 0.75rem;
    text-align: left;
}

.capability-table td {
    padding: 0.75rem;
    border-bottom: 1px solid #475569;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_capability_color(value: float) -> str:
    """Get color based on capability index value."""
    if value >= 1.67:
        return BADGE_COLORS['A+']
    elif value >= 1.33:
        return BADGE_COLORS['A']
    elif value >= 1.0:
        return BADGE_COLORS['B']
    else:
        return BADGE_COLORS['C']


def get_rating_class(value: float) -> str:
    """Get CSS class for rating badge."""
    if value >= 1.67:
        return "rating-excellent"
    elif value >= 1.33:
        return "rating-good"
    elif value >= 1.0:
        return "rating-marginal"
    else:
        return "rating-poor"


def get_rating_text(value: float) -> str:
    """Get rating text."""
    if value >= 2.0:
        return "World Class"
    elif value >= 1.67:
        return "Excellent"
    elif value >= 1.33:
        return "Capable"
    elif value >= 1.0:
        return "Marginal"
    else:
        return "Not Capable"


def create_capability_gauge_large(value: float, label: str) -> go.Figure:
    """Create a large gauge chart for capability index."""
    color = get_capability_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 60, 'color': color}, 'valueformat': '.2f'},
        gauge={
            'axis': {
                'range': [0, 2.5],
                'tickwidth': 2,
                'tickcolor': '#64748b',
                'tickfont': {'color': '#94a3b8', 'size': 12}
            },
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': 'rgba(30, 41, 59, 0.8)',
            'borderwidth': 2,
            'bordercolor': '#475569',
            'steps': [
                {'range': [0, 1.0], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [1.0, 1.33], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [1.33, 1.67], 'color': 'rgba(59, 130, 246, 0.15)'},
                {'range': [1.67, 2.5], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#f8fafc', 'width': 3},
                'thickness': 0.8,
                'value': 1.33
            }
        },
        title={'text': label, 'font': {'color': '#94a3b8', 'size': 18}}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        height=280,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def create_process_distribution_chart(data: np.ndarray, result, lsl: float, usl: float) -> go.Figure:
    """Create process distribution visualization with spec limits."""
    hist_result = HistogramCalculator.calculate_histogram(data, lsl=lsl, usl=usl)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Bar(
        x=hist_result.bin_centers,
        y=hist_result.counts,
        marker_color='rgba(59, 130, 246, 0.6)',
        name='Data Distribution',
        hovertemplate='Value: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))

    # Normal curve
    x_curve = np.linspace(result.lsl - 3*result.sigma_overall,
                          result.usl + 3*result.sigma_overall, 200)
    bin_width = hist_result.bins[1] - hist_result.bins[0]
    y_curve = HistogramCalculator.fit_normal_curve(
        x_curve, result.process_mean, result.sigma_overall, len(data), bin_width
    )

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#f59e0b', width=3)
    ))

    # Specification limits
    max_y = max(hist_result.counts) * 1.2

    fig.add_vline(x=lsl, line_dash='dash', line_color='#ef4444', line_width=2)
    fig.add_vline(x=usl, line_dash='dash', line_color='#ef4444', line_width=2)

    # Add spec limit annotations
    fig.add_annotation(x=lsl, y=max_y, text=f"LSL<br>{lsl:.1f}",
                       showarrow=False, font=dict(color='#ef4444', size=12))
    fig.add_annotation(x=usl, y=max_y, text=f"USL<br>{usl:.1f}",
                       showarrow=False, font=dict(color='#ef4444', size=12))

    # Mean line
    fig.add_vline(x=result.process_mean, line_dash='solid',
                  line_color='#10b981', line_width=2)
    fig.add_annotation(x=result.process_mean, y=max_y * 0.9,
                       text=f"Mean: {result.process_mean:.2f}",
                       showarrow=False, font=dict(color='#10b981', size=12))

    # Target line
    if result.target is not None:
        fig.add_vline(x=result.target, line_dash='dot',
                      line_color='#8b5cf6', line_width=2)

    # Add 3σ zones
    for i in range(1, 4):
        fig.add_vrect(
            x0=result.process_mean - i * result.sigma_overall,
            x1=result.process_mean + i * result.sigma_overall,
            fillcolor=f'rgba(59, 130, 246, {0.05 * (4-i)})',
            line_width=0
        )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=60, r=40, t=40, b=60),
        xaxis_title='Measurement Value',
        yaxis_title='Frequency',
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

    fig.update_xaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_capability_trend_chart(n_periods: int = 12) -> go.Figure:
    """Create capability trending chart with sample data."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='M')
    cp_values = 1.5 + np.cumsum(np.random.normal(0, 0.05, n_periods))
    cpk_values = cp_values - np.abs(np.random.normal(0.1, 0.05, n_periods))
    cpk_values = np.clip(cpk_values, 0.8, 2.5)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=cp_values,
        mode='lines+markers',
        name='Cp',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10),
        hovertemplate='Date: %{x}<br>Cp: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=cpk_values,
        mode='lines+markers',
        name='Cpk',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10),
        hovertemplate='Date: %{x}<br>Cpk: %{y:.2f}<extra></extra>'
    ))

    # Reference lines
    fig.add_hline(y=1.33, line_dash='dash', line_color='#f59e0b',
                  annotation_text='Target (1.33)')
    fig.add_hline(y=1.0, line_dash='dash', line_color='#ef4444',
                  annotation_text='Minimum (1.0)')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=40),
        xaxis_title='Period',
        yaxis_title='Capability Index',
        font=dict(family='Inter', color='#e2e8f0'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)',
                     range=[0.5, 2.5])

    return fig


def create_sigma_level_chart(sigma: float) -> go.Figure:
    """Create sigma level visualization."""
    fig = go.Figure()

    # Background levels
    levels = [
        (0, 2, '#ef4444', '2σ'),
        (2, 3, '#f59e0b', '3σ'),
        (3, 4, '#3b82f6', '4σ'),
        (4, 5, '#10b981', '5σ'),
        (5, 6, '#8b5cf6', '6σ')
    ]

    for start, end, color, label in levels:
        fig.add_shape(
            type='rect',
            x0=start, x1=end,
            y0=0, y1=1,
            fillcolor=color,
            opacity=0.3,
            line_width=0
        )

    # Current sigma marker
    fig.add_trace(go.Scatter(
        x=[sigma],
        y=[0.5],
        mode='markers+text',
        marker=dict(size=30, color='#f8fafc', symbol='diamond',
                    line=dict(width=3, color=get_capability_color(sigma/3))),
        text=[f'{sigma:.2f}σ'],
        textposition='top center',
        textfont=dict(size=16, color='#f8fafc'),
        showlegend=False
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=150,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            range=[0, 6.5],
            showgrid=False,
            tickvals=[1, 2, 3, 4, 5, 6],
            ticktext=['1σ', '2σ', '3σ', '4σ', '5σ', '6σ'],
            tickfont=dict(size=14)
        ),
        yaxis=dict(visible=False),
        font=dict(family='Inter', color='#e2e8f0')
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🎯 Process Capability Analysis</div>
        <div class="page-subtitle">
            Comprehensive capability indices and sigma level analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'cap_data' not in st.session_state:
        st.session_state.cap_data = None
    if 'cap_result' not in st.session_state:
        st.session_state.cap_result = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Specification Limits")
        lsl = st.number_input("LSL (Lower Spec)", value=990.0, format="%.2f")
        usl = st.number_input("USL (Upper Spec)", value=1010.0, format="%.2f")
        target = st.number_input("Target", value=1000.0, format="%.2f")

        st.markdown("### Sample Data")
        n_samples = st.slider("Number of Samples", 50, 500, 200)
        process_mean = st.number_input("Process Mean", value=1000.0)
        process_std = st.number_input("Process Std Dev", value=3.0)

        st.markdown("### Subgroup Settings")
        subgroup_size = st.slider("Subgroup Size", 2, 10, 5)

    # Data input section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Generate Sample"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Measurement Data (CSV)",
                type=['csv', 'xlsx'],
                help="CSV/Excel file with measurement data"
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state.cap_data = df.iloc[:, 0].values
                    st.success(f"Loaded {len(st.session_state.cap_data)} measurements")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with tab2:
            if st.button("Generate Sample Data", type="primary"):
                data, _, _ = generate_capability_sample_data(
                    n_samples=n_samples,
                    mean=process_mean,
                    std=process_std,
                    lsl=lsl,
                    usl=usl
                )
                st.session_state.cap_data = data
                st.success(f"Generated {n_samples} sample measurements")

    with col2:
        st.markdown('<div class="section-title">Specifications</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div style="color: #94a3b8; line-height: 2;">
                <strong style="color: #f8fafc;">LSL:</strong> {lsl:.2f}<br>
                <strong style="color: #f8fafc;">USL:</strong> {usl:.2f}<br>
                <strong style="color: #f8fafc;">Target:</strong> {target:.2f}<br>
                <strong style="color: #f8fafc;">Tolerance:</strong> {usl - lsl:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis
    if st.session_state.cap_data is not None:
        data = st.session_state.cap_data

        # Calculate capability
        result = CapabilityCalculator.calculate_capability(
            data, lsl, usl, target, subgroup_size
        )
        st.session_state.cap_result = result

        # Capability Gauges
        st.markdown('<div class="section-title">Capability Indices</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig = create_capability_gauge_large(result.cp, "Cp")
            st.plotly_chart(fig, use_container_width=True)

            rating = get_rating_text(result.cp)
            rating_class = get_rating_class(result.cp)
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="capability-rating {rating_class}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig = create_capability_gauge_large(result.cpk, "Cpk")
            st.plotly_chart(fig, use_container_width=True)

            rating = get_rating_text(result.cpk)
            rating_class = get_rating_class(result.cpk)
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="capability-rating {rating_class}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            fig = create_capability_gauge_large(result.pp, "Pp")
            st.plotly_chart(fig, use_container_width=True)

            rating = get_rating_text(result.pp)
            rating_class = get_rating_class(result.pp)
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="capability-rating {rating_class}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            fig = create_capability_gauge_large(result.ppk, "Ppk")
            st.plotly_chart(fig, use_container_width=True)

            rating = get_rating_text(result.ppk)
            rating_class = get_rating_class(result.ppk)
            st.markdown(f"""
            <div style="text-align: center;">
                <span class="capability-rating {rating_class}">{rating}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sigma Level
        st.markdown('<div class="section-title">Process Sigma Level</div>', unsafe_allow_html=True)

        sigma_color = get_capability_color(result.sigma_level / 3)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"""
            <div class="capability-card">
                <div class="sigma-display" style="color: {sigma_color};">
                    {result.sigma_level:.2f}σ
                </div>
                <div class="capability-label">Process Sigma</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig = create_sigma_level_chart(result.sigma_level)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Process Distribution
        st.markdown('<div class="section-title">Process Distribution</div>', unsafe_allow_html=True)

        fig = create_process_distribution_chart(data, result, lsl, usl)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics and PPM
        st.markdown('<div class="section-title">Process Statistics</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Central Tendency</h4>
                <table class="capability-table">
                    <tr>
                        <td style="color: #94a3b8;">Process Mean</td>
                        <td style="text-align: right;">{result.process_mean:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">Target</td>
                        <td style="text-align: right;">{result.target:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">Deviation from Target</td>
                        <td style="text-align: right;">{abs(result.process_mean - result.target):.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Variation</h4>
                <table class="capability-table">
                    <tr>
                        <td style="color: #94a3b8;">σ Within (Short-term)</td>
                        <td style="text-align: right;">{result.sigma_within:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">σ Overall (Long-term)</td>
                        <td style="text-align: right;">{result.sigma_overall:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">6σ Spread</td>
                        <td style="text-align: right;">{6 * result.sigma_overall:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            ppm_color = '#10b981' if result.ppm_total < 66 else '#f59e0b' if result.ppm_total < 6210 else '#ef4444'
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Expected Defects (PPM)</h4>
                <table class="capability-table">
                    <tr>
                        <td style="color: #94a3b8;">Below LSL</td>
                        <td style="text-align: right;">{result.ppm_below_lsl:.1f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">Above USL</td>
                        <td style="text-align: right;">{result.ppm_above_usl:.1f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;"><strong>Total PPM</strong></td>
                        <td style="text-align: right; color: {ppm_color}; font-weight: 700;">
                            {result.ppm_total:.1f}
                        </td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;"><strong>Yield</strong></td>
                        <td style="text-align: right; color: #10b981; font-weight: 700;">
                            {result.yield_percent:.4f}%
                        </td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Capability Trend
        st.markdown('<div class="section-title">Capability Trend (Historical)</div>',
                    unsafe_allow_html=True)

        fig = create_capability_trend_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Reference Guide
        st.markdown('<div class="section-title">Capability Index Reference</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Index Definitions</h4>
                <table class="capability-table">
                    <tr>
                        <td style="color: #3b82f6; font-weight: 600;">Cp</td>
                        <td>Potential capability (uses σ within)</td>
                    </tr>
                    <tr>
                        <td style="color: #3b82f6; font-weight: 600;">Cpk</td>
                        <td>Actual capability accounting for centering</td>
                    </tr>
                    <tr>
                        <td style="color: #f59e0b; font-weight: 600;">Pp</td>
                        <td>Performance (uses σ overall)</td>
                    </tr>
                    <tr>
                        <td style="color: #f59e0b; font-weight: 600;">Ppk</td>
                        <td>Actual performance accounting for centering</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Industry Standards</h4>
                <table class="capability-table">
                    <tr>
                        <td><span style="color: #10b981;">■</span> Cpk ≥ 1.67</td>
                        <td>6σ Level - World Class</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> Cpk ≥ 1.33</td>
                        <td>4σ Level - Industry Standard</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> Cpk ≥ 1.00</td>
                        <td>3σ Level - Minimum Acceptable</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> Cpk < 1.00</td>
                        <td>Process improvement required</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No Capability Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV file or generate sample data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
