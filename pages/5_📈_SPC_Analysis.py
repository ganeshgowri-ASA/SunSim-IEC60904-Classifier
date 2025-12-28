"""
Statistical Process Control (SPC) Analysis Dashboard.
Provides X-bar & R control charts, process capability indices, and run rules detection.
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
    calculate_xbar_r_chart,
    calculate_xbar_s_chart,
    calculate_individuals_chart,
    calculate_capability_indices,
    detect_run_rules,
    generate_sample_data,
    get_capability_rating,
    CONTROL_CHART_CONSTANTS
)
from utils.db import (
    get_spc_data,
    insert_spc_batch,
    get_simulator_ids,
    get_spc_parameters,
    clear_spc_data,
    init_database
)

# Page configuration
st.set_page_config(
    page_title="SPC Analysis | SunSim Classifier",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3139;
    }
    .stMetric label {
        color: #FAFAFA !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2D3139;
        margin: 10px 0;
    }
    .status-good { color: #00D4AA; }
    .status-warning { color: #FFE66D; }
    .status-bad { color: #FF6B6B; }
    .chart-container {
        background-color: #1A1D24;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stExpander"] {
        background-color: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()


def create_control_chart(x_data: np.ndarray, y_data: np.ndarray, ucl: float, lcl: float, cl: float,
                          title: str, y_label: str, run_rule_violations: list = None) -> go.Figure:
    """Create an interactive Plotly control chart."""
    fig = go.Figure()

    # Determine point colors based on control limits and run rules
    colors = []
    violation_indices = set()
    if run_rule_violations:
        for rule in run_rule_violations:
            violation_indices.update(rule.violated_points)

    for i, y in enumerate(y_data):
        if y > ucl or y < lcl:
            colors.append('#FF6B6B')  # Out of control - red
        elif i in violation_indices:
            colors.append('#FFE66D')  # Run rule violation - yellow
        else:
            colors.append('#00D4AA')  # In control - green

    # Add data points
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers',
        name='Data',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(color=colors, size=8, line=dict(color='#FAFAFA', width=1)),
        hovertemplate='Point %{x}<br>Value: %{y:.4f}<extra></extra>'
    ))

    # Add control limits
    fig.add_hline(y=ucl, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"UCL: {ucl:.4f}", annotation_position="right")
    fig.add_hline(y=cl, line_dash="solid", line_color="#FFE66D",
                  annotation_text=f"CL: {cl:.4f}", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"LCL: {lcl:.4f}", annotation_position="right")

    # Add zone lines (1 and 2 sigma)
    sigma = (ucl - cl) / 3
    for mult in [1, 2]:
        fig.add_hline(y=cl + mult * sigma, line_dash="dot", line_color="#555555", line_width=1)
        fig.add_hline(y=cl - mult * sigma, line_dash="dot", line_color="#555555", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Sample",
            gridcolor='#2D3139',
            zerolinecolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title=y_label,
            gridcolor='#2D3139',
            zerolinecolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=120, t=50, b=50)
    )

    return fig


def create_histogram_with_normal(data: np.ndarray, usl: float = None, lsl: float = None,
                                  target: float = None) -> go.Figure:
    """Create histogram with normal distribution overlay."""
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Data',
        marker_color='#4ECDC4',
        opacity=0.7,
        histnorm='probability density'
    ))

    # Normal distribution overlay
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    x_norm = np.linspace(mean - 4*std, mean + 4*std, 100)
    y_norm = stats.norm.pdf(x_norm, mean, std)

    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='#FFE66D', width=3)
    ))

    # Add specification limits
    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color="#FF6B6B",
                      annotation_text=f"USL: {usl:.2f}", annotation_position="top")
    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color="#FF6B6B",
                      annotation_text=f"LSL: {lsl:.2f}", annotation_position="top")
    if target is not None:
        fig.add_vline(x=target, line_dash="solid", line_color="#00D4AA",
                      annotation_text=f"Target: {target:.2f}", annotation_position="top")

    fig.update_layout(
        title=dict(text="Process Distribution", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Measured Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
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
        margin=dict(l=60, r=40, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def display_run_rules_violations(violations: list):
    """Display run rules violations in an organized format."""
    if not violations:
        st.success("✓ No run rules violations detected")
        return

    st.warning(f"⚠ {len(violations)} run rule(s) violated")

    for violation in violations:
        severity_color = "#FF6B6B" if violation.severity == "violation" else "#FFE66D"
        st.markdown(f"""
        <div style="background-color: #1A1D24; padding: 10px; border-radius: 8px;
                    border-left: 4px solid {severity_color}; margin: 5px 0;">
            <strong style="color: {severity_color};">{violation.rule_name}</strong><br>
            <span style="color: #FAFAFA;">{violation.description}</span><br>
            <span style="color: #888;">Points: {violation.violated_points[:10]}{'...' if len(violation.violated_points) > 10 else ''}</span>
        </div>
        """, unsafe_allow_html=True)


# Main page content
st.title("📈 Statistical Process Control Analysis")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Database", "Sample Data", "Upload CSV"],
        index=1
    )

    if data_source == "Database":
        simulators = get_simulator_ids()
        selected_simulator = st.selectbox("Select Simulator", simulators)

        parameters = get_spc_parameters()
        if parameters:
            selected_parameter = st.selectbox("Select Parameter", parameters)
        else:
            st.info("No parameters in database. Add data first.")
            selected_parameter = None

    elif data_source == "Sample Data":
        st.subheader("Sample Data Settings")
        n_samples = st.slider("Number of Samples", 50, 500, 150)
        sample_mean = st.number_input("Process Mean", value=100.0)
        sample_std = st.number_input("Process Std Dev", value=2.0, min_value=0.1)
        seed = st.number_input("Random Seed", value=42, min_value=0)

    else:  # Upload CSV
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    st.markdown("---")
    st.subheader("Chart Settings")

    chart_type = st.selectbox(
        "Control Chart Type",
        ["X-bar & R", "X-bar & S", "I-MR (Individuals)"]
    )

    if chart_type != "I-MR (Individuals)":
        subgroup_size = st.slider("Subgroup Size", 2, 10, 5)
    else:
        subgroup_size = 1

    st.markdown("---")
    st.subheader("Specification Limits")
    use_specs = st.checkbox("Enable Specification Limits", value=True)
    if use_specs:
        usl = st.number_input("Upper Spec Limit (USL)", value=106.0)
        lsl = st.number_input("Lower Spec Limit (LSL)", value=94.0)
        target = st.number_input("Target", value=100.0)
    else:
        usl = lsl = target = None

# Load data based on source
data = None

if data_source == "Database":
    if selected_parameter:
        df = get_spc_data(simulator_id=selected_simulator, parameter_name=selected_parameter)
        if not df.empty:
            data = df['measured_value'].values
        else:
            st.info("No data found for selected parameters. Try sample data or upload.")

elif data_source == "Sample Data":
    data = generate_sample_data(n_samples, sample_mean, sample_std, seed)

elif data_source == "Upload CSV" and uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'measured_value' in df.columns:
            data = df['measured_value'].values
        elif 'value' in df.columns:
            data = df['value'].values
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data = df[numeric_cols[0]].values
                st.info(f"Using column '{numeric_cols[0]}' for analysis")
            else:
                st.error("No numeric columns found in CSV")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# Main analysis
if data is not None and len(data) > 0:
    # Calculate control chart statistics
    try:
        if chart_type == "X-bar & R":
            chart_result = calculate_xbar_r_chart(data, subgroup_size)
            x_bar_data = chart_result.x_bar
            range_data = chart_result.r_values
            range_label = "Range"
            range_ucl = chart_result.r_ucl
            range_lcl = chart_result.r_lcl
            range_cl = chart_result.r_cl
        elif chart_type == "X-bar & S":
            chart_result = calculate_xbar_s_chart(data, subgroup_size)
            x_bar_data = chart_result.x_bar
            range_data = chart_result.s_values
            range_label = "Std Dev"
            range_ucl = chart_result.s_ucl
            range_lcl = chart_result.s_lcl
            range_cl = chart_result.s_cl
        else:  # I-MR
            chart_result = calculate_individuals_chart(data)
            x_bar_data = chart_result['x_values']
            range_data = chart_result['mr_values']
            range_label = "Moving Range"
            range_ucl = chart_result['mr_ucl']
            range_lcl = chart_result['mr_lcl']
            range_cl = chart_result['mr_cl']

        # Get x-bar limits
        if chart_type in ["X-bar & R", "X-bar & S"]:
            x_bar_ucl = chart_result.x_bar_ucl
            x_bar_lcl = chart_result.x_bar_lcl
            x_bar_cl = chart_result.x_bar_cl
        else:
            x_bar_ucl = chart_result['x_ucl']
            x_bar_lcl = chart_result['x_lcl']
            x_bar_cl = chart_result['x_cl']

        # Detect run rules
        run_rules = detect_run_rules(x_bar_data, x_bar_cl, x_bar_ucl, x_bar_lcl)

        # Calculate capability indices if specs provided
        if use_specs and usl and lsl:
            capability = calculate_capability_indices(data, usl, lsl, target, subgroup_size)

        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Mean", f"{np.mean(data):.4f}")
        with col2:
            st.metric("Std Dev", f"{np.std(data, ddof=1):.4f}")
        with col3:
            st.metric("Samples", f"{len(data)}")
        with col4:
            ooc_count = sum(1 for x in x_bar_data if x > x_bar_ucl or x < x_bar_lcl)
            st.metric("Out of Control", f"{ooc_count}")
        with col5:
            st.metric("Run Rules", f"{len(run_rules)} violations")

        st.markdown("---")

        # Control Charts
        st.subheader("Control Charts")

        x_indices = np.arange(1, len(x_bar_data) + 1)

        # X-bar chart
        fig_xbar = create_control_chart(
            x_indices, x_bar_data, x_bar_ucl, x_bar_lcl, x_bar_cl,
            f"{chart_type.split('&')[0].strip()} Chart", "Subgroup Mean", run_rules
        )
        st.plotly_chart(fig_xbar, use_container_width=True)

        # Range/S/MR chart
        r_indices = np.arange(1, len(range_data) + 1)
        fig_range = create_control_chart(
            r_indices, range_data, range_ucl, range_lcl, range_cl,
            f"{range_label} Chart", range_label
        )
        st.plotly_chart(fig_range, use_container_width=True)

        # Run Rules Analysis
        st.markdown("---")
        st.subheader("Run Rules Analysis")
        display_run_rules_violations(run_rules)

        # Capability Analysis
        if use_specs and usl and lsl:
            st.markdown("---")
            st.subheader("Process Capability")

            cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)

            rating, color = get_capability_rating(capability.cpk)

            with cap_col1:
                st.metric("Cp", f"{capability.cp:.3f}")
            with cap_col2:
                st.markdown(f"""
                <div style="background-color: #1A1D24; padding: 15px; border-radius: 10px;
                            border: 2px solid {color};">
                    <span style="color: #888; font-size: 14px;">Cpk</span><br>
                    <span style="color: {color}; font-size: 28px; font-weight: bold;">{capability.cpk:.3f}</span><br>
                    <span style="color: {color}; font-size: 12px;">{rating}</span>
                </div>
                """, unsafe_allow_html=True)
            with cap_col3:
                st.metric("Pp", f"{capability.pp:.3f}")
            with cap_col4:
                st.metric("Ppk", f"{capability.ppk:.3f}")

            # Additional capability info
            cap_col5, cap_col6, cap_col7, cap_col8 = st.columns(4)
            with cap_col5:
                st.metric("Sigma Level", f"{capability.sigma_level:.2f}σ")
            with cap_col6:
                st.metric("PPM Total", f"{capability.ppm_total:.1f}")
            with cap_col7:
                st.metric("Z (USL)", f"{capability.z_usl:.2f}")
            with cap_col8:
                st.metric("Z (LSL)", f"{capability.z_lsl:.2f}")

        # Histogram
        st.markdown("---")
        st.subheader("Process Distribution")

        if use_specs:
            fig_hist = create_histogram_with_normal(data, usl, lsl, target)
        else:
            fig_hist = create_histogram_with_normal(data)

        st.plotly_chart(fig_hist, use_container_width=True)

        # Data Statistics Expander
        with st.expander("📊 Detailed Statistics"):
            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.markdown("**Descriptive Statistics**")
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Median', 'Skewness', 'Kurtosis'],
                    'Value': [
                        len(data),
                        f"{np.mean(data):.4f}",
                        f"{np.std(data, ddof=1):.4f}",
                        f"{np.min(data):.4f}",
                        f"{np.max(data):.4f}",
                        f"{np.ptp(data):.4f}",
                        f"{np.median(data):.4f}",
                        f"{stats.skew(data):.4f}",
                        f"{stats.kurtosis(data):.4f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

            with stat_col2:
                st.markdown("**Control Chart Constants**")
                if subgroup_size > 1 and subgroup_size in CONTROL_CHART_CONSTANTS:
                    const = CONTROL_CHART_CONSTANTS[subgroup_size]
                    const_df = pd.DataFrame({
                        'Constant': ['A2', 'D3', 'D4', 'A3', 'B3', 'B4', 'd2', 'c4'],
                        'Value': [const['A2'], const['D3'], const['D4'],
                                 const['A3'], const['B3'], const['B4'],
                                 const['d2'], const['c4']]
                    })
                    st.dataframe(const_df, hide_index=True, use_container_width=True)

        # Save to Database Option
        with st.expander("💾 Save Data to Database"):
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                save_simulator_id = st.text_input("Simulator ID", value="SIM-001")
            with save_col2:
                save_parameter = st.text_input("Parameter Name", value="Irradiance")

            if st.button("Save to Database", type="primary"):
                df_save = pd.DataFrame({
                    'measured_value': data,
                    'subgroup_number': np.repeat(np.arange(1, len(data) // subgroup_size + 2), subgroup_size)[:len(data)],
                    'ucl': x_bar_ucl,
                    'lcl': x_bar_lcl,
                    'cl': x_bar_cl
                })
                insert_spc_batch(df_save, save_simulator_id, save_parameter)
                st.success(f"✓ Saved {len(data)} measurements to database")

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        st.info("Try adjusting parameters or using different data.")

else:
    st.info("👈 Configure data source in the sidebar to begin analysis")

    # Show example
    st.markdown("""
    ### Getting Started

    1. **Sample Data**: Generate random process data for testing
    2. **Database**: Load previously saved SPC data
    3. **Upload CSV**: Import your own measurement data

    #### CSV Format
    Your CSV should have a `measured_value` or `value` column with numeric measurements.
    """)
