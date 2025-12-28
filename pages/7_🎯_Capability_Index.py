"""
Process Capability Dashboard.
Provides visual gauge displays for Cp, Cpk, Pp, Ppk with trending and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.spc_calculations import (
    calculate_capability_indices,
    generate_sample_data,
    get_capability_rating,
    calculate_process_sigma_level
)
from utils.db import (
    get_spc_data,
    get_capability_history,
    insert_capability_record,
    get_simulator_ids,
    get_spc_parameters,
    init_database
)

# Page configuration
st.set_page_config(
    page_title="Capability Index | SunSim Classifier",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .capability-gauge {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #2D3139;
        text-align: center;
        margin: 10px 0;
    }
    .sigma-badge {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 25px;
        font-size: 24px;
        font-weight: bold;
        margin: 10px;
    }
    .spec-card {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3139;
        margin: 5px 0;
    }
    .ppm-value {
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()


def create_capability_gauge(value: float, title: str, min_val: float = 0, max_val: float = 3) -> go.Figure:
    """Create a Plotly gauge chart for capability index."""
    rating, color = get_capability_rating(value)

    # Determine gauge color zones
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#FAFAFA'}},
        number={'font': {'size': 36, 'color': color}, 'suffix': '', 'valueformat': '.3f'},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickwidth': 1,
                'tickcolor': "#FAFAFA",
                'tickfont': {'color': '#FAFAFA'}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#2D3139",
            'borderwidth': 2,
            'bordercolor': "#555",
            'steps': [
                {'range': [0, 0.67], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [0.67, 1.0], 'color': 'rgba(255, 107, 107, 0.2)'},
                {'range': [1.0, 1.33], 'color': 'rgba(255, 230, 109, 0.2)'},
                {'range': [1.33, 1.67], 'color': 'rgba(78, 205, 196, 0.2)'},
                {'range': [1.67, 2.0], 'color': 'rgba(0, 212, 170, 0.2)'},
                {'range': [2.0, 3.0], 'color': 'rgba(0, 212, 170, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#FFE66D", 'width': 3},
                'thickness': 0.8,
                'value': 1.33
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1A1D24',
        font={'color': '#FAFAFA'},
        height=280,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    return fig


def create_sigma_level_gauge(sigma: float) -> go.Figure:
    """Create sigma level indicator gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sigma,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Process Sigma Level", 'font': {'size': 20, 'color': '#FAFAFA'}},
        number={'font': {'size': 48, 'color': '#00D4AA'}, 'suffix': 'σ', 'valueformat': '.2f'},
        gauge={
            'axis': {'range': [0, 7], 'tickwidth': 2, 'tickcolor': "#FAFAFA",
                    'tickfont': {'color': '#FAFAFA', 'size': 14}},
            'bar': {'color': '#00D4AA', 'thickness': 0.4},
            'bgcolor': "#2D3139",
            'borderwidth': 2,
            'bordercolor': "#00D4AA",
            'steps': [
                {'range': [0, 2], 'color': 'rgba(255, 107, 107, 0.4)'},
                {'range': [2, 3], 'color': 'rgba(255, 230, 109, 0.3)'},
                {'range': [3, 4], 'color': 'rgba(78, 205, 196, 0.3)'},
                {'range': [4, 5], 'color': 'rgba(0, 212, 170, 0.3)'},
                {'range': [5, 6], 'color': 'rgba(0, 212, 170, 0.4)'},
                {'range': [6, 7], 'color': 'rgba(0, 212, 170, 0.5)'}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1A1D24',
        font={'color': '#FAFAFA'},
        height=350,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    return fig


def create_process_distribution_chart(data: np.ndarray, usl: float, lsl: float,
                                        target: float, mean: float, std: float) -> go.Figure:
    """Create process distribution chart with spec limits."""
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=40,
        name='Process Data',
        marker_color='#4ECDC4',
        opacity=0.6,
        histnorm='probability density'
    ))

    # Normal curve
    x_range = np.linspace(mean - 4*std, mean + 4*std, 200)
    y_norm = stats.norm.pdf(x_range, mean, std)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_norm,
        mode='lines',
        name='Normal Fit',
        line=dict(color='#FFE66D', width=3)
    ))

    # Specification limits
    fig.add_vline(x=usl, line_dash="dash", line_color="#FF6B6B", line_width=3,
                  annotation_text=f"USL: {usl:.2f}", annotation_position="top right",
                  annotation_font_color="#FF6B6B")
    fig.add_vline(x=lsl, line_dash="dash", line_color="#FF6B6B", line_width=3,
                  annotation_text=f"LSL: {lsl:.2f}", annotation_position="top left",
                  annotation_font_color="#FF6B6B")
    fig.add_vline(x=target, line_dash="solid", line_color="#00D4AA", line_width=2,
                  annotation_text=f"Target: {target:.2f}", annotation_position="top",
                  annotation_font_color="#00D4AA")
    fig.add_vline(x=mean, line_dash="dot", line_color="#FFE66D", line_width=2,
                  annotation_text=f"Mean: {mean:.2f}", annotation_position="bottom",
                  annotation_font_color="#FFE66D")

    # Add 3-sigma lines
    for mult in [-3, 3]:
        sigma_val = mean + mult * std
        fig.add_vline(x=sigma_val, line_dash="dot", line_color="#555", line_width=1)

    fig.update_layout(
        title=dict(text="Process Distribution vs Specifications", font=dict(color='#FAFAFA', size=18)),
        xaxis=dict(
            title="Measured Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            zerolinecolor='#2D3139'
        ),
        yaxis=dict(
            title="Density",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def create_capability_trend_chart(history_df: pd.DataFrame) -> go.Figure:
    """Create capability trending chart over time."""
    fig = go.Figure()

    if 'sample_date' in history_df.columns and 'cpk' in history_df.columns:
        # Sort by date
        history_df = history_df.sort_values('sample_date')

        # Add Cpk trend
        fig.add_trace(go.Scatter(
            x=history_df['sample_date'],
            y=history_df['cpk'],
            mode='lines+markers',
            name='Cpk',
            line=dict(color='#00D4AA', width=3),
            marker=dict(size=8)
        ))

        # Add Ppk trend if available
        if 'ppk' in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df['sample_date'],
                y=history_df['ppk'],
                mode='lines+markers',
                name='Ppk',
                line=dict(color='#4ECDC4', width=2, dash='dash'),
                marker=dict(size=6)
            ))

        # Add threshold line
        fig.add_hline(y=1.33, line_dash="dash", line_color="#FFE66D",
                      annotation_text="Target: 1.33", annotation_position="right")
        fig.add_hline(y=1.0, line_dash="dot", line_color="#FF6B6B",
                      annotation_text="Min: 1.0", annotation_position="right")

    fig.update_layout(
        title=dict(text="Capability Trend Over Time", font=dict(color='#FAFAFA', size=18)),
        xaxis=dict(
            title="Date",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="Capability Index",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            range=[0, max(history_df['cpk'].max() * 1.2, 2) if len(history_df) > 0 else 2]
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=100, t=80, b=60)
    )

    return fig


def create_target_comparison_chart(target: float, actual_mean: float, usl: float, lsl: float) -> go.Figure:
    """Create target vs actual comparison visualization."""
    spec_range = usl - lsl
    offset = actual_mean - target
    offset_pct = (offset / (spec_range / 2)) * 100 if spec_range > 0 else 0

    fig = go.Figure()

    # Create horizontal bar showing spec range
    fig.add_trace(go.Bar(
        y=['Specification Range'],
        x=[spec_range],
        orientation='h',
        marker_color='#2D3139',
        name='Spec Range',
        base=lsl
    ))

    # Add markers
    fig.add_trace(go.Scatter(
        x=[lsl, target, usl],
        y=['Specification Range'] * 3,
        mode='markers+text',
        marker=dict(size=20, color=['#FF6B6B', '#00D4AA', '#FF6B6B'],
                   symbol=['line-ns', 'diamond', 'line-ns']),
        text=['LSL', 'Target', 'USL'],
        textposition='top center',
        textfont=dict(color='#FAFAFA', size=12),
        name='Limits'
    ))

    # Add actual mean marker
    fig.add_trace(go.Scatter(
        x=[actual_mean],
        y=['Specification Range'],
        mode='markers+text',
        marker=dict(size=25, color='#FFE66D', symbol='triangle-down'),
        text=[f'Mean: {actual_mean:.3f}'],
        textposition='bottom center',
        textfont=dict(color='#FFE66D', size=12),
        name='Actual Mean'
    ))

    fig.update_layout(
        title=dict(text=f"Target vs Actual (Offset: {offset:.3f}, {offset_pct:.1f}% of half-spec)",
                  font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            range=[lsl - spec_range * 0.1, usl + spec_range * 0.1]
        ),
        yaxis=dict(
            showticklabels=False
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=200,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def get_ppm_color(ppm: float) -> str:
    """Get color based on PPM level."""
    if ppm < 233:  # 6 sigma
        return '#00D4AA'
    elif ppm < 6210:  # 4 sigma
        return '#4ECDC4'
    elif ppm < 66807:  # 3 sigma
        return '#FFE66D'
    else:
        return '#FF6B6B'


# Main page content
st.title("🎯 Process Capability Dashboard")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Database", "Upload CSV"],
        index=0
    )

    if data_source == "Sample Data":
        st.subheader("Sample Data Settings")
        n_samples = st.slider("Sample Size", 50, 1000, 200)
        sample_mean = st.number_input("Process Mean", value=100.0)
        sample_std = st.number_input("Process Std Dev", value=2.0, min_value=0.1)
        seed = st.number_input("Random Seed", value=42, min_value=0)

    elif data_source == "Database":
        simulators = get_simulator_ids()
        selected_simulator = st.selectbox("Select Simulator", simulators)
        parameters = get_spc_parameters()
        if parameters:
            selected_parameter = st.selectbox("Select Parameter", parameters)
        else:
            st.info("No parameters in database.")
            selected_parameter = None

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    st.markdown("---")
    st.subheader("Specification Limits")

    usl = st.number_input("Upper Spec Limit (USL)", value=106.0)
    lsl = st.number_input("Lower Spec Limit (LSL)", value=94.0)
    target = st.number_input("Target Value", value=100.0)

    st.markdown("---")
    subgroup_size = st.slider("Subgroup Size", 2, 10, 5,
                              help="For within-subgroup variation calculation")

# Load data
data = None

if data_source == "Sample Data":
    data = generate_sample_data(n_samples, sample_mean, sample_std, seed)

elif data_source == "Database" and selected_parameter:
    df = get_spc_data(simulator_id=selected_simulator, parameter_name=selected_parameter)
    if not df.empty:
        data = df['measured_value'].values
    else:
        st.info("No data found. Try sample data or upload.")

elif data_source == "Upload CSV" and uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if 'measured_value' in df.columns:
                data = df['measured_value'].values
            elif 'value' in df.columns:
                data = df['value'].values
            else:
                data = df[numeric_cols[0]].values
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# Main analysis
if data is not None and len(data) > 0:
    # Calculate capability
    capability = calculate_capability_indices(data, usl, lsl, target, subgroup_size)

    # Summary metrics row
    st.subheader("📊 Process Summary")

    sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)

    with sum_col1:
        st.metric("Sample Size", f"{len(data)}")
    with sum_col2:
        st.metric("Process Mean", f"{capability.process_mean:.4f}")
    with sum_col3:
        st.metric("Std Dev (Within)", f"{capability.sigma_within:.4f}")
    with sum_col4:
        st.metric("Std Dev (Overall)", f"{capability.sigma_overall:.4f}")
    with sum_col5:
        rating, color = get_capability_rating(capability.cpk)
        st.markdown(f"""
        <div style="background-color: #1A1D24; padding: 10px; border-radius: 10px;
                    border: 2px solid {color};">
            <span style="color: #888; font-size: 12px;">Rating</span><br>
            <span style="color: {color}; font-size: 18px; font-weight: bold;">{rating}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Main capability gauges
    st.subheader("🎯 Capability Indices")

    gauge_col1, gauge_col2, gauge_col3, gauge_col4 = st.columns(4)

    with gauge_col1:
        fig_cp = create_capability_gauge(capability.cp, "Cp (Potential)")
        st.plotly_chart(fig_cp, use_container_width=True)

    with gauge_col2:
        fig_cpk = create_capability_gauge(capability.cpk, "Cpk (Actual)")
        st.plotly_chart(fig_cpk, use_container_width=True)

    with gauge_col3:
        fig_pp = create_capability_gauge(capability.pp, "Pp (Performance)")
        st.plotly_chart(fig_pp, use_container_width=True)

    with gauge_col4:
        fig_ppk = create_capability_gauge(capability.ppk, "Ppk (Actual Perf.)")
        st.plotly_chart(fig_ppk, use_container_width=True)

    st.markdown("---")

    # Sigma level and PPM
    sigma_col1, sigma_col2 = st.columns([1, 1])

    with sigma_col1:
        st.subheader("📈 Process Sigma Level")
        fig_sigma = create_sigma_level_gauge(capability.sigma_level)
        st.plotly_chart(fig_sigma, use_container_width=True)

    with sigma_col2:
        st.subheader("📉 Defect Rates (PPM)")

        ppm_col1, ppm_col2, ppm_col3 = st.columns(3)

        with ppm_col1:
            color = get_ppm_color(capability.ppm_above_usl)
            st.markdown(f"""
            <div class="spec-card" style="border-left: 4px solid {color};">
                <span style="color: #888;">PPM Above USL</span><br>
                <span class="ppm-value" style="color: {color};">{capability.ppm_above_usl:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        with ppm_col2:
            color = get_ppm_color(capability.ppm_below_lsl)
            st.markdown(f"""
            <div class="spec-card" style="border-left: 4px solid {color};">
                <span style="color: #888;">PPM Below LSL</span><br>
                <span class="ppm-value" style="color: {color};">{capability.ppm_below_lsl:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        with ppm_col3:
            color = get_ppm_color(capability.ppm_total)
            st.markdown(f"""
            <div class="spec-card" style="border-left: 4px solid {color};">
                <span style="color: #888;">PPM Total</span><br>
                <span class="ppm-value" style="color: {color};">{capability.ppm_total:.1f}</span>
            </div>
            """, unsafe_allow_html=True)

        # Z-scores
        st.markdown("#### Z-Scores")
        z_col1, z_col2 = st.columns(2)
        with z_col1:
            st.metric("Z (USL)", f"{capability.z_usl:.3f}")
        with z_col2:
            st.metric("Z (LSL)", f"{capability.z_lsl:.3f}")

    st.markdown("---")

    # Target comparison
    st.subheader("🎯 Target vs Actual")
    fig_target = create_target_comparison_chart(target, capability.process_mean, usl, lsl)
    st.plotly_chart(fig_target, use_container_width=True)

    # Process distribution
    st.subheader("📊 Process Distribution")
    fig_dist = create_process_distribution_chart(
        data, usl, lsl, target, capability.process_mean, capability.sigma_overall
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Capability Trending (if historical data available)
    if data_source == "Database":
        history_df = get_capability_history(
            simulator_id=selected_simulator if data_source == "Database" else None,
            parameter_name=selected_parameter if data_source == "Database" else None
        )
        if not history_df.empty:
            st.markdown("---")
            st.subheader("📈 Capability Trend")
            fig_trend = create_capability_trend_chart(history_df)
            st.plotly_chart(fig_trend, use_container_width=True)

    # Detailed statistics
    with st.expander("📋 Detailed Statistics"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("**Capability Indices**")
            cap_df = pd.DataFrame({
                'Index': ['Cp', 'Cpk', 'Pp', 'Ppk', 'Cpm'],
                'Value': [
                    f"{capability.cp:.4f}",
                    f"{capability.cpk:.4f}",
                    f"{capability.pp:.4f}",
                    f"{capability.ppk:.4f}",
                    f"{capability.cpm:.4f}"
                ],
                'Description': [
                    'Potential capability (spread vs specs)',
                    'Actual capability (accounts for centering)',
                    'Performance capability (overall variation)',
                    'Performance (accounts for centering)',
                    'Taguchi capability (accounts for target)'
                ]
            })
            st.dataframe(cap_df, hide_index=True, use_container_width=True)

        with detail_col2:
            st.markdown("**Specification Summary**")
            spec_df = pd.DataFrame({
                'Parameter': ['USL', 'LSL', 'Target', 'Spec Range', 'Process Mean', 'Offset from Target'],
                'Value': [
                    f"{usl:.4f}",
                    f"{lsl:.4f}",
                    f"{target:.4f}",
                    f"{usl - lsl:.4f}",
                    f"{capability.process_mean:.4f}",
                    f"{capability.process_mean - target:.4f}"
                ]
            })
            st.dataframe(spec_df, hide_index=True, use_container_width=True)

    # Save capability record
    with st.expander("💾 Save Capability Record"):
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            save_simulator = st.text_input("Simulator ID", value="SIM-001", key="cap_sim")
        with save_col2:
            save_param = st.text_input("Parameter Name", value="Irradiance", key="cap_param")

        if st.button("Save Record", type="primary"):
            insert_capability_record(
                save_simulator, save_param,
                capability.cp, capability.cpk, capability.pp, capability.ppk,
                usl, lsl, target, capability.process_mean,
                capability.sigma_overall, len(data)
            )
            st.success("✓ Capability record saved to database")

    # Interpretation guide
    with st.expander("📖 Capability Index Interpretation"):
        st.markdown("""
        ### Capability Index Guidelines

        | Index | < 1.0 | 1.0-1.33 | 1.33-1.67 | > 1.67 |
        |-------|-------|----------|-----------|--------|
        | Rating | ❌ Poor | ⚠️ Marginal | ✅ Good | ✅ Excellent |
        | Action | Immediate improvement needed | Monitor closely | Acceptable | World class |

        ### Index Definitions

        - **Cp (Process Capability)**: Compares specification spread to process spread (6σ)
          - Cp = (USL - LSL) / 6σ
          - Does NOT account for process centering

        - **Cpk (Process Capability Index)**: Accounts for process centering
          - Cpk = min[(USL - μ)/3σ, (μ - LSL)/3σ]
          - Always ≤ Cp

        - **Pp (Process Performance)**: Uses overall variation instead of within-subgroup
          - Same formula as Cp but uses overall σ

        - **Ppk (Process Performance Index)**: Cpk equivalent using overall variation

        - **Cpm (Taguchi Capability)**: Accounts for deviation from target
          - Useful when target ≠ midpoint of specs

        ### Sigma Level Reference

        | Sigma | Cpk | PPM (Defects) | Yield |
        |-------|-----|---------------|-------|
        | 2σ | 0.67 | 308,537 | 69.15% |
        | 3σ | 1.00 | 66,807 | 93.32% |
        | 4σ | 1.33 | 6,210 | 99.38% |
        | 5σ | 1.67 | 233 | 99.977% |
        | 6σ | 2.00 | 3.4 | 99.99966% |
        """)

else:
    st.info("👈 Configure data source and specification limits in the sidebar to begin analysis")

    st.markdown("""
    ### Getting Started

    1. **Load Data**: Use sample data, database, or upload a CSV
    2. **Set Specifications**: Define USL, LSL, and Target values
    3. **Analyze**: View capability indices, sigma level, and PPM

    #### What is Process Capability?

    Process capability measures how well a process can produce output within specification limits.
    A capable process has:
    - **Cpk ≥ 1.33**: Process mean centered within specs
    - **Low PPM**: Few defects expected
    - **High Sigma Level**: Process performance benchmark
    """)
