"""
Statistical Process Control (SPC) Analysis Dashboard.
Provides X-bar & R control charts, process capability indices, run rules detection,
and Reference Module monitoring for flasher/reference module performance tracking.
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
import io

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
    CONTROL_CHART_CONSTANTS,
    # Reference module functions
    analyze_ref_module_data,
    detect_western_electric_rules,
    generate_ref_module_sample_data,
    get_point_status,
    calculate_ref_module_cpk,
    RefModuleAnalysisResult,
    WesternElectricViolation
)
from utils.db import (
    get_spc_data,
    insert_spc_batch,
    get_simulator_ids,
    get_spc_parameters,
    clear_spc_data,
    init_database,
    get_ref_module_data,
    insert_ref_module_batch,
    get_ref_module_ids
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
    .violation-card {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .alert-critical {
        border-left: 4px solid #FF6B6B;
    }
    .alert-warning {
        border-left: 4px solid #FFE66D;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()


def create_control_chart(x_data: np.ndarray, y_data: np.ndarray, ucl: float, lcl: float, cl: float,
                          title: str, y_label: str, run_rule_violations: list = None,
                          warning_ucl: float = None, warning_lcl: float = None,
                          x_label: str = "Sample") -> go.Figure:
    """Create an interactive Plotly control chart with enhanced visualization."""
    fig = go.Figure()

    # Determine point colors based on control limits and run rules
    colors = []
    violation_indices = set()
    if run_rule_violations:
        for rule in run_rule_violations:
            if hasattr(rule, 'violated_points'):
                violation_indices.update(rule.violated_points)

    sigma = (ucl - cl) / 3
    warn_ucl = warning_ucl if warning_ucl else cl + 2 * sigma
    warn_lcl = warning_lcl if warning_lcl else cl - 2 * sigma

    for i, y in enumerate(y_data):
        if y > ucl or y < lcl:
            colors.append('#FF6B6B')  # Out of control - red
        elif y > warn_ucl or y < warn_lcl:
            colors.append('#FFE66D')  # Warning zone - yellow
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
    for mult in [1, 2]:
        fig.add_hline(y=cl + mult * sigma, line_dash="dot", line_color="#555555", line_width=1)
        fig.add_hline(y=cl - mult * sigma, line_dash="dot", line_color="#555555", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title=x_label,
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


def create_ref_module_control_chart(
    result: RefModuleAnalysisResult,
    title: str,
    y_label: str,
    x_axis_type: str = "flash_number",
    violations: list = None
) -> go.Figure:
    """Create a reference module Individual-X control chart with enhanced annotations."""
    fig = go.Figure()

    # Determine x-axis data
    if x_axis_type == "datetime" and result.timestamps is not None:
        x_data = result.timestamps
        x_title = "Date/Time"
    else:
        x_data = result.flash_numbers
        x_title = "Flash Number"

    # Calculate warning limits
    warning_ucl = result.x_cl + 2 * result.x_sigma
    warning_lcl = result.x_cl - 2 * result.x_sigma

    # Determine point colors
    colors = []
    violation_indices = set()
    if violations:
        for v in violations:
            violation_indices.update(v.violated_points)

    for i, val in enumerate(result.values):
        if val > result.x_ucl or val < result.x_lcl:
            colors.append('#FF6B6B')  # Out of control - red
        elif val > warning_ucl or val < warning_lcl:
            colors.append('#FFE66D')  # Warning zone - yellow
        elif i in violation_indices:
            colors.append('#FFE66D')  # Rule violation - yellow
        else:
            colors.append('#00D4AA')  # Normal - green

    # Add data points
    fig.add_trace(go.Scatter(
        x=x_data,
        y=result.values,
        mode='lines+markers',
        name='Measurements',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(color=colors, size=10, line=dict(color='#FAFAFA', width=1)),
        hovertemplate='%{x}<br>Value: %{y:.4f}<extra></extra>'
    ))

    # Add control limits
    fig.add_hline(y=result.x_ucl, line_dash="dash", line_color="#FF6B6B", line_width=2,
                  annotation_text=f"UCL: {result.x_ucl:.4f}", annotation_position="right")
    fig.add_hline(y=result.x_cl, line_dash="solid", line_color="#FFE66D", line_width=2,
                  annotation_text=f"CL: {result.x_cl:.4f}", annotation_position="right")
    fig.add_hline(y=result.x_lcl, line_dash="dash", line_color="#FF6B6B", line_width=2,
                  annotation_text=f"LCL: {result.x_lcl:.4f}", annotation_position="right")

    # Add warning limits
    fig.add_hline(y=warning_ucl, line_dash="dot", line_color="#FFE66D", line_width=1,
                  annotation_text="2σ", annotation_position="right")
    fig.add_hline(y=warning_lcl, line_dash="dot", line_color="#FFE66D", line_width=1)

    # Add 1-sigma lines
    fig.add_hline(y=result.x_cl + result.x_sigma, line_dash="dot", line_color="#555555", line_width=1)
    fig.add_hline(y=result.x_cl - result.x_sigma, line_dash="dot", line_color="#555555", line_width=1)

    # Add trend line if significant
    if result.has_significant_trend:
        trend_y = result.x_cl + result.trend_slope * np.arange(len(result.values))
        fig.add_trace(go.Scatter(
            x=x_data,
            y=trend_y,
            mode='lines',
            name='Trend',
            line=dict(color='#9B59B6', width=2, dash='dashdot'),
            hoverinfo='skip'
        ))

    # Add annotations for out-of-control points
    for i in result.ooc_points[:5]:  # Limit to first 5 annotations
        fig.add_annotation(
            x=x_data[i],
            y=result.values[i],
            text="OOC",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FF6B6B",
            font=dict(color="#FF6B6B", size=10),
            bgcolor="#1A1D24",
            bordercolor="#FF6B6B"
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title=x_title,
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
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=120, t=80, b=50)
    )

    return fig


def create_moving_range_chart(
    flash_numbers: np.ndarray,
    mr_values: np.ndarray,
    mr_ucl: float,
    mr_lcl: float,
    mr_cl: float,
    title: str = "Moving Range Chart"
) -> go.Figure:
    """Create a Moving Range control chart."""
    fig = go.Figure()

    # Determine point colors
    colors = ['#FF6B6B' if v > mr_ucl else '#00D4AA' for v in mr_values]

    # Add data points (MR has one less point than original data)
    x_data = flash_numbers[1:]  # Start from second point

    fig.add_trace(go.Scatter(
        x=x_data,
        y=mr_values,
        mode='lines+markers',
        name='Moving Range',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(color=colors, size=8, line=dict(color='#FAFAFA', width=1)),
        hovertemplate='Flash %{x}<br>MR: %{y:.4f}<extra></extra>'
    ))

    # Add control limits
    fig.add_hline(y=mr_ucl, line_dash="dash", line_color="#FF6B6B",
                  annotation_text=f"UCL: {mr_ucl:.4f}", annotation_position="right")
    fig.add_hline(y=mr_cl, line_dash="solid", line_color="#FFE66D",
                  annotation_text=f"MR̄: {mr_cl:.4f}", annotation_position="right")
    fig.add_hline(y=mr_lcl, line_dash="dash", line_color="#00D4AA",
                  annotation_text=f"LCL: {mr_lcl:.4f}", annotation_position="right")

    fig.update_layout(
        title=dict(text=title, font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Flash Number",
            gridcolor='#2D3139',
            zerolinecolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="Moving Range",
            gridcolor='#2D3139',
            zerolinecolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        showlegend=False,
        height=300,
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


def display_western_electric_violations(violations: list):
    """Display Western Electric rule violations with recommended actions."""
    if not violations:
        st.success("✓ No Western Electric rule violations detected - Process is in control")
        return

    critical_count = sum(1 for v in violations if v.severity == "critical")
    warning_count = sum(1 for v in violations if v.severity == "warning")

    if critical_count > 0:
        st.error(f"🚨 {critical_count} CRITICAL violation(s) detected - Immediate action required!")
    if warning_count > 0:
        st.warning(f"⚠️ {warning_count} WARNING(s) detected - Monitor closely")

    for violation in violations:
        if violation.severity == "critical":
            alert_class = "alert-critical"
            icon = "🔴"
        else:
            alert_class = "alert-warning"
            icon = "🟡"

        st.markdown(f"""
        <div class="violation-card {alert_class}">
            <strong style="color: {'#FF6B6B' if violation.severity == 'critical' else '#FFE66D'};">
                {icon} Rule {violation.rule_number}: {violation.rule_name}
            </strong><br>
            <span style="color: #FAFAFA;">{violation.description}</span><br>
            <span style="color: #888;">Affected points: {violation.violated_points[:10]}{'...' if len(violation.violated_points) > 10 else ''}</span><br>
            <div style="margin-top: 8px; padding: 8px; background-color: #2D3139; border-radius: 4px;">
                <strong style="color: #4ECDC4;">Recommended Action:</strong><br>
                <span style="color: #FAFAFA;">{violation.action}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def create_long_term_trend_chart(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Create a long-term trend analysis chart with rolling statistics."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Measurement Trend', 'Rolling Standard Deviation'),
        row_heights=[0.7, 0.3]
    )

    # Calculate rolling statistics
    window = min(20, len(df) // 5)
    if window < 3:
        window = 3

    rolling_mean = df[value_col].rolling(window=window, center=True).mean()
    rolling_std = df[value_col].rolling(window=window, center=True).std()

    x_data = df['flash_number'] if 'flash_number' in df.columns else df.index

    # Main trend chart
    fig.add_trace(
        go.Scatter(x=x_data, y=df[value_col], mode='markers', name='Measurements',
                   marker=dict(color='#4ECDC4', size=6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=rolling_mean, mode='lines', name=f'{window}-pt Moving Avg',
                   line=dict(color='#FFE66D', width=3)),
        row=1, col=1
    )

    # Rolling std dev chart
    fig.add_trace(
        go.Scatter(x=x_data, y=rolling_std, mode='lines', name='Rolling Std Dev',
                   line=dict(color='#9B59B6', width=2), fill='tozeroy',
                   fillcolor='rgba(155, 89, 182, 0.2)'),
        row=2, col=1
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color='#FAFAFA', size=16)),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(gridcolor='#2D3139', tickfont=dict(color='#FAFAFA'))
    fig.update_yaxes(gridcolor='#2D3139', tickfont=dict(color='#FAFAFA'))

    return fig


# Main page content
st.title("📈 Statistical Process Control Analysis")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Measurement Type Selector (NEW)
    measurement_type = st.radio(
        "📊 Measurement Type",
        ["Process Data", "Reference Module Isc", "Reference Module Pmax"],
        index=0
    )

    st.markdown("---")

    if measurement_type == "Process Data":
        # Original SPC functionality
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

    else:
        # Reference Module Configuration
        st.subheader("🔬 Reference Module Settings")

        ref_data_source = st.radio(
            "Data Source",
            ["Sample Data", "Database", "Upload CSV"],
            index=0
        )

        if ref_data_source == "Sample Data":
            st.subheader("Simulation Settings")
            n_flashes = st.slider("Number of Flashes", 50, 500, 150)

            if measurement_type == "Reference Module Isc":
                nominal_value = st.number_input("Nominal Isc (A)", value=8.5, format="%.3f")
                std_value = st.number_input("Std Dev (A)", value=0.02, min_value=0.001, format="%.4f")
            else:  # Pmax
                nominal_value = st.number_input("Nominal Pmax (W)", value=320.0, format="%.1f")
                std_value = st.number_input("Std Dev (W)", value=1.0, min_value=0.1)

            include_drift = st.checkbox("Include Degradation Drift", value=False)
            if include_drift:
                drift_start = st.slider("Drift Start (flash #)", 20, n_flashes - 10, 70)
                drift_rate = st.number_input("Drift Rate", value=0.001, format="%.4f")
            else:
                drift_start = 70
                drift_rate = 0.001

            include_anomalies = st.checkbox("Include Anomalies", value=False)
            seed = st.number_input("Random Seed", value=42, min_value=0)

        elif ref_data_source == "Database":
            ref_modules = get_ref_module_ids()
            selected_ref_module = st.selectbox("Select Reference Module", ref_modules)
            simulators = get_simulator_ids()
            selected_simulator = st.selectbox("Select Simulator", simulators)

        else:  # Upload CSV
            ref_uploaded_file = st.file_uploader(
                "Upload Reference Module CSV",
                type=['csv'],
                help="CSV should have columns: flash_number, isc (and optionally pmax, timestamp)"
            )

        st.markdown("---")
        st.subheader("Chart Settings")

        x_axis_type = st.radio(
            "X-Axis",
            ["Flash Number", "Date/Time"],
            index=0
        )

        st.markdown("---")
        st.subheader("Specification Limits")
        use_ref_specs = st.checkbox("Enable Spec Limits", value=True)
        if use_ref_specs:
            if measurement_type == "Reference Module Isc":
                ref_target = st.number_input("Target Isc (A)", value=8.5, format="%.3f")
                ref_tolerance_pct = st.slider("Tolerance (%)", 1.0, 10.0, 2.0)
            else:
                ref_target = st.number_input("Target Pmax (W)", value=320.0, format="%.1f")
                ref_tolerance_pct = st.slider("Tolerance (%)", 1.0, 10.0, 2.0)

            ref_usl = ref_target * (1 + ref_tolerance_pct / 100)
            ref_lsl = ref_target * (1 - ref_tolerance_pct / 100)
            st.text(f"USL: {ref_usl:.4f}\nLSL: {ref_lsl:.4f}")
        else:
            ref_usl = ref_lsl = ref_target = None


# ============================================================================
# REFERENCE MODULE MONITORING
# ============================================================================
if measurement_type in ["Reference Module Isc", "Reference Module Pmax"]:
    st.subheader(f"🔬 {measurement_type} Monitoring")
    st.markdown("Monitor flasher and reference module performance over time using Individual-X and Moving Range charts.")

    # Load reference module data
    ref_data = None
    value_column = 'isc' if measurement_type == "Reference Module Isc" else 'pmax'
    y_label = "Isc (A)" if measurement_type == "Reference Module Isc" else "Pmax (W)"

    if ref_data_source == "Sample Data":
        ref_df = generate_ref_module_sample_data(
            n_flashes=n_flashes,
            nominal_isc=nominal_value if measurement_type == "Reference Module Isc" else 8.5,
            nominal_pmax=nominal_value if measurement_type == "Reference Module Pmax" else 320.0,
            isc_std=std_value if measurement_type == "Reference Module Isc" else 0.02,
            pmax_std=std_value if measurement_type == "Reference Module Pmax" else 1.0,
            include_drift=include_drift,
            drift_start=drift_start,
            drift_rate=drift_rate,
            include_anomalies=include_anomalies,
            seed=seed
        )
        ref_data = ref_df[value_column].values

    elif ref_data_source == "Database":
        ref_df = get_ref_module_data(ref_module_id=selected_ref_module, simulator_id=selected_simulator)
        if not ref_df.empty and value_column in ref_df.columns:
            ref_data = ref_df[value_column].values
        else:
            st.info(f"No {value_column} data found for selected reference module.")

    elif ref_data_source == "Upload CSV" and ref_uploaded_file:
        try:
            ref_df = pd.read_csv(ref_uploaded_file)
            if value_column in ref_df.columns:
                ref_data = ref_df[value_column].values
            else:
                # Try alternative column names
                alt_cols = ['Isc', 'ISC', 'isc', 'Pmax', 'PMAX', 'pmax', 'power', 'Power']
                found_col = None
                for col in alt_cols:
                    if col in ref_df.columns:
                        found_col = col
                        break
                if found_col:
                    ref_data = ref_df[found_col].values
                    st.info(f"Using column '{found_col}' for analysis")
                else:
                    st.error(f"Could not find '{value_column}' column in CSV. Available: {list(ref_df.columns)}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # Perform analysis if data is available
    if ref_data is not None and len(ref_data) > 2:
        # Get flash numbers
        if 'flash_number' in ref_df.columns:
            flash_numbers = ref_df['flash_number'].values
        else:
            flash_numbers = np.arange(1, len(ref_data) + 1)

        # Get timestamps if available
        timestamps = None
        if 'timestamp' in ref_df.columns:
            timestamps = pd.to_datetime(ref_df['timestamp']).values
        elif 'measurement_time' in ref_df.columns:
            timestamps = pd.to_datetime(ref_df['measurement_time']).values

        # Analyze data
        result = analyze_ref_module_data(
            values=ref_data,
            flash_numbers=flash_numbers,
            timestamps=timestamps,
            nominal_value=ref_target if use_ref_specs else None
        )

        # Detect Western Electric rule violations
        we_violations = detect_western_electric_rules(
            data=ref_data,
            cl=result.x_cl,
            ucl=result.x_ucl,
            lcl=result.x_lcl
        )

        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Mean", f"{result.mean:.4f}")
        with col2:
            st.metric("Std Dev", f"{result.std_dev:.4f}")
        with col3:
            st.metric("Measurements", f"{len(ref_data)}")
        with col4:
            ooc_pct = (len(result.ooc_points) / len(ref_data)) * 100
            st.metric("Out of Control", f"{len(result.ooc_points)} ({ooc_pct:.1f}%)")
        with col5:
            st.metric("Rule Violations", f"{len(we_violations)}")

        # Trend and drift indicators
        trend_col1, trend_col2, trend_col3 = st.columns(3)

        with trend_col1:
            if result.has_significant_trend:
                trend_direction = "↗️ Increasing" if result.trend_slope > 0 else "↘️ Decreasing"
                st.warning(f"⚠️ Significant Trend Detected: {trend_direction}")
                st.caption(f"Slope: {result.trend_slope:.6f}/flash (p={result.trend_pvalue:.4f})")
            else:
                st.success("✓ No significant trend")

        with trend_col2:
            if result.drift_detected:
                st.error(f"🚨 Drift Detected at Flash #{result.drift_start_index + 1}")
            else:
                st.success("✓ No drift detected")

        with trend_col3:
            if use_ref_specs:
                cpk_result = calculate_ref_module_cpk(ref_data, ref_usl, ref_lsl, ref_target)
                rating, color = get_capability_rating(cpk_result['cpk'])
                st.markdown(f"""
                <div style="background-color: #1A1D24; padding: 10px; border-radius: 8px; border: 2px solid {color}; text-align: center;">
                    <span style="color: #888; font-size: 12px;">Cpk</span><br>
                    <span style="color: {color}; font-size: 24px; font-weight: bold;">{cpk_result['cpk']:.3f}</span><br>
                    <span style="color: {color}; font-size: 11px;">{rating}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Control Charts
        st.subheader("📊 Control Charts")

        # Individual-X Chart
        x_axis = "datetime" if x_axis_type == "Date/Time" and timestamps is not None else "flash_number"
        fig_x = create_ref_module_control_chart(
            result=result,
            title=f"Individual-X Chart: {measurement_type}",
            y_label=y_label,
            x_axis_type=x_axis,
            violations=we_violations
        )
        st.plotly_chart(fig_x, use_container_width=True)

        # Moving Range Chart
        fig_mr = create_moving_range_chart(
            flash_numbers=flash_numbers,
            mr_values=result.moving_ranges,
            mr_ucl=result.mr_ucl,
            mr_lcl=result.mr_lcl,
            mr_cl=result.mr_cl,
            title=f"Moving Range Chart: {measurement_type} Variation"
        )
        st.plotly_chart(fig_mr, use_container_width=True)

        st.markdown("---")

        # Western Electric Rules Analysis
        st.subheader("🚦 Western Electric Rules Analysis")
        display_western_electric_violations(we_violations)

        st.markdown("---")

        # Process Capability (if specs provided)
        if use_ref_specs:
            st.subheader("📐 Process Capability Analysis")

            cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)

            with cap_col1:
                st.metric("Cp", f"{cpk_result['cp']:.3f}")
            with cap_col2:
                st.metric("Cpk", f"{cpk_result['cpk']:.3f}")
            with cap_col3:
                st.metric("Cpm", f"{cpk_result['cpm']:.3f}")
            with cap_col4:
                st.metric("CPU / CPL", f"{cpk_result['cpu']:.3f} / {cpk_result['cpl']:.3f}")

            # Histogram with spec limits
            fig_hist = create_histogram_with_normal(ref_data, ref_usl, ref_lsl, ref_target)
            fig_hist.update_layout(title=dict(text=f"{measurement_type} Distribution"))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Long-term Monitoring Report
        st.subheader("📋 Long-term Monitoring Report")

        with st.expander("📈 Trend Analysis", expanded=True):
            ref_df_plot = pd.DataFrame({
                'flash_number': flash_numbers,
                value_column: ref_data
            })
            fig_trend = create_long_term_trend_chart(
                ref_df_plot,
                value_column,
                f"Long-term {measurement_type} Trend Analysis"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with st.expander("📊 Detailed Statistics"):
            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.markdown("**Descriptive Statistics**")
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Median', 'Skewness', 'Kurtosis'],
                    'Value': [
                        len(ref_data),
                        f"{result.mean:.4f}",
                        f"{result.std_dev:.4f}",
                        f"{result.min_value:.4f}",
                        f"{result.max_value:.4f}",
                        f"{result.max_value - result.min_value:.4f}",
                        f"{np.median(ref_data):.4f}",
                        f"{stats.skew(ref_data):.4f}",
                        f"{stats.kurtosis(ref_data):.4f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

            with stat_col2:
                st.markdown("**Control Limits**")
                limits_df = pd.DataFrame({
                    'Limit': ['UCL (3σ)', 'Warning UCL (2σ)', 'Center Line', 'Warning LCL (2σ)', 'LCL (3σ)', 'Sigma (σ)'],
                    'Value': [
                        f"{result.x_ucl:.4f}",
                        f"{result.x_cl + 2*result.x_sigma:.4f}",
                        f"{result.x_cl:.4f}",
                        f"{result.x_cl - 2*result.x_sigma:.4f}",
                        f"{result.x_lcl:.4f}",
                        f"{result.x_sigma:.4f}"
                    ]
                })
                st.dataframe(limits_df, hide_index=True, use_container_width=True)

        # Save to Database
        with st.expander("💾 Save Data to Database"):
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                save_ref_module_id = st.text_input("Reference Module ID", value="REF-001")
            with save_col2:
                save_simulator_id = st.text_input("Simulator ID", value="SIM-001", key="ref_save_sim")

            if st.button("Save to Database", type="primary", key="save_ref_data"):
                save_df = pd.DataFrame({
                    'flash_number': flash_numbers,
                    'isc': ref_df['isc'].values if 'isc' in ref_df.columns else None,
                    'pmax': ref_df['pmax'].values if 'pmax' in ref_df.columns else None
                })
                if insert_ref_module_batch(save_df, save_ref_module_id, save_simulator_id):
                    st.success(f"✓ Saved {len(ref_data)} measurements to database")
                else:
                    st.error("Failed to save data to database")

        # Export Report
        with st.expander("📥 Export Report"):
            # Generate summary report
            report_lines = [
                f"# Reference Module SPC Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"",
                f"## Summary",
                f"- Measurement Type: {measurement_type}",
                f"- Number of Measurements: {len(ref_data)}",
                f"- Mean: {result.mean:.4f}",
                f"- Std Dev: {result.std_dev:.4f}",
                f"- Out of Control Points: {len(result.ooc_points)}",
                f"",
                f"## Control Limits",
                f"- UCL: {result.x_ucl:.4f}",
                f"- CL: {result.x_cl:.4f}",
                f"- LCL: {result.x_lcl:.4f}",
                f"",
                f"## Trend Analysis",
                f"- Significant Trend: {'Yes' if result.has_significant_trend else 'No'}",
                f"- Drift Detected: {'Yes' if result.drift_detected else 'No'}",
                f"",
                f"## Western Electric Rule Violations",
            ]

            if we_violations:
                for v in we_violations:
                    report_lines.append(f"- Rule {v.rule_number}: {v.rule_name} - {v.severity.upper()}")
            else:
                report_lines.append("- No violations detected")

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Report (Markdown)",
                data=report_text,
                file_name=f"ref_module_spc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

            # Export data as CSV
            export_df = pd.DataFrame({
                'flash_number': flash_numbers,
                value_column: ref_data,
                'status': ['OOC' if i in result.ooc_points else 'Warning' if i in result.warning_points else 'Normal'
                          for i in range(len(ref_data))]
            })

            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="Download Data (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"ref_module_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    else:
        st.info("👈 Configure data source in the sidebar to begin reference module monitoring")

        st.markdown("""
        ### Reference Module Monitoring

        Monitor your flasher and reference module performance over time:

        1. **Sample Data**: Generate simulated data with optional drift and anomalies
        2. **Database**: Load previously saved reference module measurements
        3. **Upload CSV**: Import your own measurement data

        #### CSV Format
        Your CSV should have columns:
        - `flash_number`: Sequential flash number
        - `isc`: Short-circuit current (A)
        - `pmax`: Maximum power (W) - optional

        #### Features
        - Individual-X and Moving Range control charts
        - Western Electric rules for anomaly detection
        - Trend and drift detection
        - Process capability analysis
        - Long-term monitoring reports
        """)


# ============================================================================
# PROCESS DATA (Original SPC)
# ============================================================================
else:  # Process Data
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
