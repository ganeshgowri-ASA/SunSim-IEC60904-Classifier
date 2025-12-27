"""
Sun Simulator Classification System
MSA Gage R&R Page - Measurement System Analysis

This page provides Gage R&R analysis with:
- Repeatability & Reproducibility analysis
- Variance components (Equipment, Operator, Part-to-Part)
- GRR % calculation
- ndc (Number of Distinct Categories)
- Interactive variance charts
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
from utils.msa_calculations import (
    GRRCalculator, GRRRating,
    generate_grr_sample_data, calculate_variance_chart_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MSA Gage R&R - " + APP_CONFIG['title'],
    page_icon="🔬",
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

.grr-status {
    display: inline-flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 1rem;
}

.status-acceptable {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.status-marginal {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.status-unacceptable {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.variance-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

.variance-table th {
    background: #334155;
    color: #f8fafc;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.85rem;
}

.variance-table td {
    padding: 0.75rem;
    border-bottom: 1px solid #475569;
    color: #e2e8f0;
}

.variance-table tr:hover {
    background: rgba(59, 130, 246, 0.1);
}

.ndc-display {
    text-align: center;
    padding: 2rem;
}

.ndc-value {
    font-size: 4rem;
    font-weight: 700;
    color: #3b82f6;
}

.ndc-label {
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

def get_grr_color(grr_percent: float) -> str:
    """Get color based on GRR percentage."""
    if grr_percent < 10:
        return BADGE_COLORS['A+']
    elif grr_percent <= 30:
        return BADGE_COLORS['B']
    else:
        return BADGE_COLORS['C']


def get_ndc_color(ndc: int) -> str:
    """Get color based on ndc value."""
    if ndc >= 5:
        return BADGE_COLORS['A+']
    elif ndc >= 3:
        return BADGE_COLORS['B']
    else:
        return BADGE_COLORS['C']


def create_variance_components_chart(result) -> go.Figure:
    """Create stacked bar chart of variance components."""
    chart_data = calculate_variance_chart_data(result)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('% Contribution (Variance)', '% Study Variation (6σ)'),
        horizontal_spacing=0.15
    )

    colors = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981']
    components = ['Total GRR', 'Repeatability', 'Reproducibility', 'Part-to-Part']

    # % Contribution chart
    fig.add_trace(
        go.Bar(
            x=components,
            y=chart_data['pct_contribution'],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in chart_data['pct_contribution']],
            textposition='outside',
            textfont=dict(color='#f8fafc'),
            showlegend=False,
            hovertemplate='%{x}<br>Contribution: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # % Study Variation chart
    fig.add_trace(
        go.Bar(
            x=components,
            y=chart_data['pct_study_var'],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in chart_data['pct_study_var']],
            textposition='outside',
            textfont=dict(color='#f8fafc'),
            showlegend=False,
            hovertemplate='%{x}<br>Study Var: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # Add reference lines
    fig.add_hline(y=10, line_dash='dash', line_color='#10b981',
                  annotation_text='10% (Acceptable)', row=1, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#f59e0b',
                  annotation_text='30% (Marginal)', row=1, col=1)

    fig.add_hline(y=10, line_dash='dash', line_color='#10b981', row=1, col=2)
    fig.add_hline(y=30, line_dash='dash', line_color='#f59e0b', row=1, col=2)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=60, r=40, t=60, b=80),
        font=dict(family='Inter', color='#e2e8f0')
    )

    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)',
                     title_text='Percentage')

    return fig


def create_operator_chart(result) -> go.Figure:
    """Create operator comparison chart."""
    operators = list(result.operator_means.keys())
    means = list(result.operator_means.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=operators,
        y=means,
        marker_color='#3b82f6',
        text=[f"{v:.2f}" for v in means],
        textposition='outside',
        textfont=dict(color='#f8fafc'),
        hovertemplate='%{x}<br>Mean: %{y:.3f}<extra></extra>'
    ))

    # Overall mean line
    fig.add_hline(y=result.overall_mean, line_dash='dash', line_color='#10b981',
                  annotation_text=f'Overall Mean: {result.overall_mean:.3f}')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=60, r=40, t=40, b=40),
        xaxis_title='Operator',
        yaxis_title='Mean Measurement',
        font=dict(family='Inter', color='#e2e8f0')
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_part_chart(result) -> go.Figure:
    """Create part comparison chart."""
    parts = list(result.part_means.keys())
    means = list(result.part_means.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=parts,
        y=means,
        marker_color='#f59e0b',
        text=[f"{v:.2f}" for v in means],
        textposition='outside',
        textfont=dict(color='#f8fafc'),
        hovertemplate='%{x}<br>Mean: %{y:.3f}<extra></extra>'
    ))

    fig.add_hline(y=result.overall_mean, line_dash='dash', line_color='#10b981',
                  annotation_text=f'Overall Mean: {result.overall_mean:.3f}')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=60, r=40, t=40, b=60),
        xaxis_title='Part',
        yaxis_title='Mean Measurement',
        font=dict(family='Inter', color='#e2e8f0')
    )

    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_by_operator_by_part_chart(result) -> go.Figure:
    """Create measurements by operator by part chart."""
    data = result.measurement_data
    n_operators, n_parts, n_trials = data.shape

    operators = list(result.operator_means.keys())
    parts = list(result.part_means.keys())

    fig = go.Figure()

    colors = px.colors.qualitative.Set2[:n_operators]

    for i, op in enumerate(operators):
        # Calculate mean for each part for this operator
        op_part_means = np.mean(data[i, :, :], axis=1)

        fig.add_trace(go.Scatter(
            x=parts,
            y=op_part_means,
            mode='lines+markers',
            name=op,
            line=dict(color=colors[i], width=2),
            marker=dict(size=8),
            hovertemplate=f'{op}<br>Part: %{{x}}<br>Mean: %{{y:.3f}}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=60, r=40, t=40, b=60),
        xaxis_title='Part',
        yaxis_title='Measurement',
        font=dict(family='Inter', color='#e2e8f0'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_variance_table_html(result) -> str:
    """Create HTML table for variance components."""
    rows = f"""
    <tr>
        <td><strong>Total Gage R&R</strong></td>
        <td>{result.variance.total_grr_variance:.6f}</td>
        <td>{result.sigma_grr:.4f}</td>
        <td>{result.pct_contribution_grr:.2f}%</td>
        <td>{result.pct_study_var_grr:.2f}%</td>
    </tr>
    <tr>
        <td style="padding-left: 1.5rem;">Repeatability</td>
        <td>{result.variance.repeatability_variance:.6f}</td>
        <td>{result.sigma_repeatability:.4f}</td>
        <td>{result.pct_contribution_repeatability:.2f}%</td>
        <td>{result.pct_study_var_repeatability:.2f}%</td>
    </tr>
    <tr>
        <td style="padding-left: 1.5rem;">Reproducibility</td>
        <td>{result.variance.reproducibility_variance:.6f}</td>
        <td>{result.sigma_reproducibility:.4f}</td>
        <td>{result.pct_contribution_reproducibility:.2f}%</td>
        <td>{result.pct_study_var_reproducibility:.2f}%</td>
    </tr>
    <tr>
        <td><strong>Part-to-Part</strong></td>
        <td>{result.variance.part_to_part_variance:.6f}</td>
        <td>{result.sigma_part:.4f}</td>
        <td>{result.pct_contribution_part:.2f}%</td>
        <td>{result.pct_study_var_part:.2f}%</td>
    </tr>
    <tr style="background: rgba(59, 130, 246, 0.1);">
        <td><strong>Total Variation</strong></td>
        <td>{result.variance.total_variance:.6f}</td>
        <td>{result.sigma_total:.4f}</td>
        <td>100.00%</td>
        <td>100.00%</td>
    </tr>
    """

    return f"""
    <table class="variance-table">
        <thead>
            <tr>
                <th>Source</th>
                <th>Variance</th>
                <th>Std Dev (σ)</th>
                <th>% Contribution</th>
                <th>% Study Var</th>
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
        <div class="page-title">🔬 Measurement System Analysis (Gage R&R)</div>
        <div class="page-subtitle">
            Repeatability & Reproducibility analysis per AIAG MSA Manual
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'grr_data' not in st.session_state:
        st.session_state.grr_data = None
    if 'grr_result' not in st.session_state:
        st.session_state.grr_result = None
    if 'operators' not in st.session_state:
        st.session_state.operators = None
    if 'parts' not in st.session_state:
        st.session_state.parts = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Study Parameters")

        n_operators = st.slider("Number of Operators", 2, 5, 3)
        n_parts = st.slider("Number of Parts", 5, 15, 10)
        n_trials = st.slider("Number of Trials", 2, 5, 3)

        st.markdown("### Tolerance")
        tolerance = st.number_input("Tolerance (USL - LSL)", value=20.0, format="%.2f",
                                     help="Specification tolerance width")

        st.markdown("### Sample Data Settings")
        base_value = st.number_input("Base Value", value=100.0)
        part_var = st.slider("Part Variation (σ)", 1.0, 20.0, 10.0)
        operator_var = st.slider("Operator Variation (σ)", 0.0, 5.0, 2.0)
        equipment_var = st.slider("Equipment Variation (σ)", 0.1, 3.0, 1.0)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Generate Sample"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload GRR Study Data (CSV)",
                type=['csv', 'xlsx'],
                help="CSV with columns: Operator, Part, Trial, Measurement"
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Reshape to 3D array
                    operators = df['Operator'].unique().tolist()
                    parts = df['Part'].unique().tolist()
                    trials = df['Trial'].unique().tolist()

                    data = np.zeros((len(operators), len(parts), len(trials)))
                    for i, op in enumerate(operators):
                        for j, part in enumerate(parts):
                            for k, trial in enumerate(trials):
                                mask = (df['Operator'] == op) & (df['Part'] == part) & (df['Trial'] == trial)
                                if mask.any():
                                    data[i, j, k] = df.loc[mask, 'Measurement'].values[0]

                    st.session_state.grr_data = data
                    st.session_state.operators = operators
                    st.session_state.parts = parts
                    st.success(f"Loaded {len(operators)} operators × {len(parts)} parts × {len(trials)} trials")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with tab2:
            if st.button("Generate Sample GRR Data", type="primary"):
                data, operators, parts = generate_grr_sample_data(
                    n_operators=n_operators,
                    n_parts=n_parts,
                    n_trials=n_trials,
                    part_variation=part_var,
                    operator_variation=operator_var,
                    equipment_variation=equipment_var,
                    base_value=base_value
                )
                st.session_state.grr_data = data
                st.session_state.operators = operators
                st.session_state.parts = parts
                st.success(f"Generated {n_operators} operators × {n_parts} parts × {n_trials} trials")

    with col2:
        st.markdown('<div class="section-title">Study Setup</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.8;">
                <strong style="color: #f8fafc;">Operators:</strong> {n_operators}<br>
                <strong style="color: #f8fafc;">Parts:</strong> {n_parts}<br>
                <strong style="color: #f8fafc;">Trials:</strong> {n_trials}<br>
                <strong style="color: #f8fafc;">Total Measurements:</strong> {n_operators * n_parts * n_trials}<br>
                <strong style="color: #f8fafc;">Tolerance:</strong> {tolerance:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis
    if st.session_state.grr_data is not None:
        data = st.session_state.grr_data
        operators = st.session_state.operators
        parts = st.session_state.parts

        # Calculate GRR
        result = GRRCalculator.calculate_grr(
            data, operators, parts, tolerance=tolerance
        )
        st.session_state.grr_result = result

        # GRR Status
        rating, description = GRRCalculator.get_grr_rating(result.grr_percent)
        status_class = {
            GRRRating.ACCEPTABLE: "status-acceptable",
            GRRRating.MARGINAL: "status-marginal",
            GRRRating.UNACCEPTABLE: "status-unacceptable"
        }[rating]

        grr_color = get_grr_color(result.grr_percent)

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span class="grr-status {status_class}">
                {rating.value}: {description}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Key Metrics
        st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {grr_color}40; background: {grr_color}10;">
                    <div class="metric-value" style="color: {grr_color};">
                        {result.grr_percent:.1f}%
                    </div>
                    <div class="metric-label">GRR %</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            ndc_color = get_ndc_color(result.ndc)
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {ndc_color}40; background: {ndc_color}10;">
                    <div class="metric-value" style="color: {ndc_color};">
                        {result.ndc}
                    </div>
                    <div class="metric-label">ndc</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{result.pct_contribution_repeatability:.1f}%</div>
                    <div class="metric-label">Repeatability</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{result.pct_contribution_reproducibility:.1f}%</div>
                    <div class="metric-label">Reproducibility</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Variance Components Chart
        st.markdown('<div class="section-title">Variance Components</div>', unsafe_allow_html=True)

        fig = create_variance_components_chart(result)
        st.plotly_chart(fig, use_container_width=True)

        # Variance Table
        st.markdown('<div class="section-title">ANOVA Table</div>', unsafe_allow_html=True)
        st.markdown(create_variance_table_html(result), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">By Operator</div>', unsafe_allow_html=True)
            fig = create_operator_chart(result)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">By Part</div>', unsafe_allow_html=True)
            fig = create_part_chart(result)
            st.plotly_chart(fig, use_container_width=True)

        # Interaction Chart
        st.markdown('<div class="section-title">Operator × Part Interaction</div>',
                    unsafe_allow_html=True)
        fig = create_by_operator_by_part_chart(result)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation Guide
        st.markdown('<div class="section-title">Interpretation Guide</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">GRR % Guidelines (AIAG)</h4>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> &lt; 10%</td>
                        <td>Acceptable</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> 10% - 30%</td>
                        <td>Marginal (may be acceptable)</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> &gt; 30%</td>
                        <td>Unacceptable (needs improvement)</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            ndc_rating, ndc_desc = GRRCalculator.get_ndc_rating(result.ndc)
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">ndc Guidelines</h4>
                <p style="color: #94a3b8; margin-bottom: 1rem;">
                    Number of Distinct Categories the measurement system can reliably distinguish.
                </p>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> ndc ≥ 5</td>
                        <td>Acceptable</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> ndc 3-4</td>
                        <td>Limited discrimination</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> ndc &lt; 3</td>
                        <td>Cannot distinguish parts</td>
                    </tr>
                </table>
                <p style="color: #94a3b8; margin-top: 1rem;">
                    Current: <strong style="color: {ndc_color};">{ndc_rating}</strong> - {ndc_desc}
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No GRR Study Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV file or generate sample data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
