"""
Sun Simulator Classification System
SPC Analysis Page - Statistical Process Control

This page provides SPC analysis with:
- X-bar & R control charts
- Process capability indices (Cp, Cpk, Pp, Ppk)
- UCL/LCL/CL visualization
- Run rules detection
- Histogram distribution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_CONFIG, THEME, BADGE_COLORS
from utils.spc_calculations import (
    SPCCalculator, CapabilityCalculator, HistogramCalculator,
    generate_spc_sample_data, generate_capability_sample_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SPC Analysis - " + APP_CONFIG['title'],
    page_icon="📈",
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

.control-status {
    display: inline-flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.9rem;
}

.status-in-control {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.status-out-of-control {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.violation-item {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    color: #fca5a5;
}

.capability-gauge {
    text-align: center;
    padding: 1.5rem;
}

.gauge-value {
    font-size: 2.5rem;
    font-weight: 700;
}

.gauge-label {
    font-size: 1rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
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


def create_xbar_r_chart(result, subgroup_numbers=None) -> go.Figure:
    """Create X-bar and R control charts."""
    n_subgroups = len(result.subgroup_means)
    if subgroup_numbers is None:
        subgroup_numbers = list(range(1, n_subgroups + 1))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('X-bar Chart (Subgroup Means)', 'R Chart (Subgroup Ranges)'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # Identify out-of-control points
    ooc_set = set(result.out_of_control_points)

    # X-bar chart
    colors = ['#ef4444' if i in ooc_set else '#3b82f6' for i in range(n_subgroups)]

    fig.add_trace(
        go.Scatter(
            x=subgroup_numbers,
            y=result.subgroup_means,
            mode='lines+markers',
            name='X-bar',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=8, color=colors),
            hovertemplate='Subgroup: %{x}<br>Mean: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Control limits for X-bar
    fig.add_hline(y=result.ucl_x, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'UCL: {result.ucl_x:.3f}', row=1, col=1)
    fig.add_hline(y=result.cl_x, line_dash='solid', line_color='#10b981',
                  annotation_text=f'CL: {result.cl_x:.3f}', row=1, col=1)
    fig.add_hline(y=result.lcl_x, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'LCL: {result.lcl_x:.3f}', row=1, col=1)

    # Zone lines (1σ and 2σ)
    sigma = (result.ucl_x - result.cl_x) / 3
    for mult in [1, 2]:
        fig.add_hline(y=result.cl_x + mult * sigma, line_dash='dot',
                      line_color='rgba(148, 163, 184, 0.5)', row=1, col=1)
        fig.add_hline(y=result.cl_x - mult * sigma, line_dash='dot',
                      line_color='rgba(148, 163, 184, 0.5)', row=1, col=1)

    # R chart
    fig.add_trace(
        go.Scatter(
            x=subgroup_numbers,
            y=result.subgroup_ranges,
            mode='lines+markers',
            name='Range',
            line=dict(color='#f59e0b', width=2),
            marker=dict(size=8, color='#f59e0b'),
            hovertemplate='Subgroup: %{x}<br>Range: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    # Control limits for R
    fig.add_hline(y=result.ucl_r, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'UCL: {result.ucl_r:.3f}', row=2, col=1)
    fig.add_hline(y=result.cl_r, line_dash='solid', line_color='#10b981',
                  annotation_text=f'R-bar: {result.cl_r:.3f}', row=2, col=1)
    if result.lcl_r > 0:
        fig.add_hline(y=result.lcl_r, line_dash='dash', line_color='#ef4444',
                      annotation_text=f'LCL: {result.lcl_r:.3f}', row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=60, r=100, t=60, b=40),
        font=dict(family='Inter', color='#e2e8f0'),
        showlegend=False
    )

    fig.update_xaxes(title_text="Subgroup Number", showgrid=True,
                     gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_histogram_with_specs(data: np.ndarray, lsl: float, usl: float,
                                  target: float = None) -> go.Figure:
    """Create histogram with specification limits and normal curve."""
    hist_result = HistogramCalculator.calculate_histogram(data, lsl=lsl, usl=usl)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Bar(
        x=hist_result.bin_centers,
        y=hist_result.counts,
        marker_color='#3b82f6',
        opacity=0.8,
        name='Data',
        hovertemplate='Value: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))

    # Normal curve overlay
    x_curve = np.linspace(hist_result.min_val - hist_result.std,
                          hist_result.max_val + hist_result.std, 100)
    bin_width = hist_result.bins[1] - hist_result.bins[0]
    y_curve = HistogramCalculator.fit_normal_curve(
        x_curve, hist_result.mean, hist_result.std, len(data), bin_width
    )

    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        name='Normal Fit',
        line=dict(color='#f59e0b', width=2)
    ))

    # Specification limits
    fig.add_vline(x=lsl, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'LSL: {lsl}', annotation_position='top')
    fig.add_vline(x=usl, line_dash='dash', line_color='#ef4444',
                  annotation_text=f'USL: {usl}', annotation_position='top')

    # Mean line
    fig.add_vline(x=hist_result.mean, line_dash='solid', line_color='#10b981',
                  annotation_text=f'Mean: {hist_result.mean:.2f}')

    # Target line
    if target is not None:
        fig.add_vline(x=target, line_dash='dot', line_color='#8b5cf6',
                      annotation_text=f'Target: {target}')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=40),
        xaxis_title='Value',
        yaxis_title='Frequency',
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
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_capability_gauge(value: float, label: str, target: float = 1.33) -> go.Figure:
    """Create a gauge chart for capability index."""
    color = get_capability_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 40, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [0, 2.5], 'tickcolor': '#64748b',
                     'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': color},
            'bgcolor': '#1e293b',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 1.0], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [1.0, 1.33], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [1.33, 1.67], 'color': 'rgba(59, 130, 246, 0.2)'},
                {'range': [1.67, 2.5], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': '#f8fafc', 'width': 2},
                'thickness': 0.8,
                'value': target
            }
        },
        title={'text': label, 'font': {'color': '#94a3b8', 'size': 14}}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📈 Statistical Process Control (SPC)</div>
        <div class="page-subtitle">
            Control charts and process capability analysis per ISO 22514
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'spc_data' not in st.session_state:
        st.session_state.spc_data = None
    if 'spc_result' not in st.session_state:
        st.session_state.spc_result = None
    if 'cap_result' not in st.session_state:
        st.session_state.cap_result = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### SPC Settings")

        subgroup_size = st.slider(
            "Subgroup Size",
            min_value=2, max_value=10, value=5,
            help="Number of samples per subgroup"
        )

        st.markdown("### Specification Limits")
        lsl = st.number_input("LSL", value=990.0, format="%.2f")
        usl = st.number_input("USL", value=1010.0, format="%.2f")
        target = st.number_input("Target", value=1000.0, format="%.2f")

        st.markdown("### Sample Data")
        n_subgroups = st.slider("Number of Subgroups", 10, 50, 30)
        process_mean = st.number_input("Process Mean", value=1000.0)
        process_std = st.number_input("Process Std Dev", value=3.0)
        include_outliers = st.checkbox("Include OOC Points", value=True)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Generate Sample"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload SPC Data (CSV)",
                type=['csv', 'xlsx'],
                help="CSV/Excel file with measurement data"
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    st.session_state.spc_data = df.values.flatten()
                    st.success(f"Loaded {len(st.session_state.spc_data)} data points")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with tab2:
            if st.button("Generate Sample SPC Data", type="primary"):
                data = generate_spc_sample_data(
                    n_subgroups=n_subgroups,
                    subgroup_size=subgroup_size,
                    mean=process_mean,
                    std=process_std,
                    include_outliers=include_outliers
                )
                st.session_state.spc_data = data.flatten()
                st.success(f"Generated {n_subgroups} subgroups × {subgroup_size} samples")

    with col2:
        st.markdown('<div class="section-title">Quick Info</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">
                <strong style="color: #f8fafc;">Tolerance:</strong> {usl - lsl:.2f}<br>
                <strong style="color: #f8fafc;">Target:</strong> {target:.2f}<br>
                <strong style="color: #f8fafc;">Subgroup Size:</strong> {subgroup_size}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis
    if st.session_state.spc_data is not None:
        data = st.session_state.spc_data

        # Calculate SPC
        spc_result = SPCCalculator.calculate_xbar_r_chart(data, subgroup_size)
        st.session_state.spc_result = spc_result

        # Calculate Capability
        cap_result = CapabilityCalculator.calculate_capability(
            data, lsl, usl, target, subgroup_size
        )
        st.session_state.cap_result = cap_result

        # Control Status
        is_in_control = len(spc_result.out_of_control_points) == 0
        status_class = "status-in-control" if is_in_control else "status-out-of-control"
        status_text = "IN CONTROL" if is_in_control else "OUT OF CONTROL"

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span class="control-status {status_class}">
                {'✓' if is_in_control else '⚠'} Process is {status_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Process Statistics and Capability
        st.markdown('<div class="section-title">Process Capability Summary</div>',
                    unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig = create_capability_gauge(cap_result.cp, "Cp")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = create_capability_gauge(cap_result.cpk, "Cpk")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = create_capability_gauge(cap_result.pp, "Pp")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = create_capability_gauge(cap_result.ppk, "Ppk")
            st.plotly_chart(fig, use_container_width=True)

        # Additional Metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            sigma_color = get_capability_color(cap_result.sigma_level / 3)
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {sigma_color}40; background: {sigma_color}10;">
                    <div class="metric-value" style="color: {sigma_color};">
                        {cap_result.sigma_level:.2f}σ
                    </div>
                    <div class="metric-label">Sigma Level</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{cap_result.process_mean:.2f}</div>
                    <div class="metric-label">Process Mean</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{cap_result.sigma_within:.3f}</div>
                    <div class="metric-label">σ Within</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{cap_result.ppm_total:.0f}</div>
                    <div class="metric-label">PPM Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            yield_color = '#10b981' if cap_result.yield_percent >= 99.73 else '#f59e0b'
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {yield_color}40; background: {yield_color}10;">
                    <div class="metric-value" style="color: {yield_color};">
                        {cap_result.yield_percent:.2f}%
                    </div>
                    <div class="metric-label">Yield</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Control Charts
        st.markdown('<div class="section-title">Control Charts</div>', unsafe_allow_html=True)

        fig = create_xbar_r_chart(spc_result)
        st.plotly_chart(fig, use_container_width=True)

        # Run Rule Violations
        if spc_result.violations:
            st.markdown('<div class="section-title">Run Rule Violations</div>',
                        unsafe_allow_html=True)

            for violation in spc_result.violations[:5]:  # Show first 5
                st.markdown(f"""
                <div class="violation-item">
                    <strong>Point {violation['point'] + 1}:</strong> {violation['rule']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Histogram
        st.markdown('<div class="section-title">Process Distribution</div>',
                    unsafe_allow_html=True)

        fig = create_histogram_with_specs(data, lsl, usl, target)
        st.plotly_chart(fig, use_container_width=True)

        # Capability Rating Guide
        st.markdown('<div class="section-title">Capability Rating Guide</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            rating, desc = CapabilityCalculator.get_capability_rating(cap_result.cpk)
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc;">Current Rating: {rating} - {desc}</h4>
                <table style="width: 100%; color: #e2e8f0; margin-top: 1rem;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> Cpk ≥ 1.67</td>
                        <td>World Class</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> Cpk ≥ 1.33</td>
                        <td>Capable</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> Cpk ≥ 1.00</td>
                        <td>Marginally Capable</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> Cpk < 1.00</td>
                        <td>Not Capable</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc;">Detailed Statistics</h4>
                <table style="width: 100%; color: #e2e8f0; margin-top: 1rem;">
                    <tr>
                        <td style="color: #94a3b8;">X-bar (Grand Mean)</td>
                        <td style="text-align: right;">{spc_result.x_bar:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">R-bar (Avg Range)</td>
                        <td style="text-align: right;">{spc_result.r_bar:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">σ (Estimated)</td>
                        <td style="text-align: right;">{spc_result.sigma_estimate:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">UCL (X-bar)</td>
                        <td style="text-align: right;">{spc_result.ucl_x:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">LCL (X-bar)</td>
                        <td style="text-align: right;">{spc_result.lcl_x:.4f}</td>
                    </tr>
                    <tr>
                        <td style="color: #94a3b8;">OOC Points</td>
                        <td style="text-align: right;">{len(spc_result.out_of_control_points)}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No SPC Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV file or generate sample data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
