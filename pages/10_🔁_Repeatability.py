"""
Repeatability Analysis Page - SunSim-IEC60904-Classifier

Flash-to-flash repeatability and consistency analysis:
- Flash-to-flash repeatability tracking
- Statistical variation (mean, std dev, range)
- Trend charts with control limits
- SPC control charts (X-bar, R charts)
- Out-of-control detection

Target: 0.09% repeatability per IEC 60904-9 Ed.3
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import (
    get_engine,
    get_session,
    init_database,
    Lamp,
    FlashRecord,
    RepeatabilityRecord,
    get_active_lamps,
    get_repeatability_history,
)
from utils.drift_analysis import DriftAnalyzer, REPEATABILITY_TARGET

# Page configuration
st.set_page_config(
    page_title="Repeatability Analysis | SunSim Classifier",
    page_icon="🔁",
    layout="wide",
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .repeat-pass {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .repeat-fail {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .control-limit {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .ooc-warning {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .spc-info {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        margin: 10px 0;
    }
    .metric-highlight {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Western Electric rules for SPC
WESTERN_ELECTRIC_RULES = {
    "rule1": "One point beyond 3-sigma",
    "rule2": "Two of three consecutive points beyond 2-sigma",
    "rule3": "Four of five consecutive points beyond 1-sigma",
    "rule4": "Eight consecutive points on one side of centerline",
}


def create_repeatability_control_chart(
    records: List[RepeatabilityRecord],
    target: float = REPEATABILITY_TARGET
) -> go.Figure:
    """Create repeatability control chart with control limits."""
    if not records:
        fig = go.Figure()
        fig.add_annotation(
            text="No repeatability data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    dates = [r.measurement_date for r in records]
    repeatability = [r.repeatability_percent for r in records]

    # Calculate control limits from data
    mean_repeat = np.mean(repeatability)
    std_repeat = np.std(repeatability)

    ucl = mean_repeat + 3 * std_repeat
    lcl = max(0, mean_repeat - 3 * std_repeat)

    # Upper and lower warning limits (2-sigma)
    uwl = mean_repeat + 2 * std_repeat
    lwl = max(0, mean_repeat - 2 * std_repeat)

    fig = go.Figure()

    # Control limits
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL: {ucl:.4f}%")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL: {lcl:.4f}%")

    # Warning limits
    fig.add_hline(y=uwl, line_dash="dot", line_color="orange",
                  annotation_text=f"UWL: {uwl:.4f}%")
    fig.add_hline(y=lwl, line_dash="dot", line_color="orange")

    # Centerline (mean)
    fig.add_hline(y=mean_repeat, line_dash="solid", line_color="green",
                  annotation_text=f"Mean: {mean_repeat:.4f}%")

    # Target line
    fig.add_hline(y=target, line_dash="dashdot", line_color="blue",
                  annotation_text=f"Target: {target}%")

    # Data points with color coding
    colors = []
    symbols = []
    for r in repeatability:
        if r > ucl or r < lcl:
            colors.append('red')
            symbols.append('x')
        elif r > uwl or r < lwl:
            colors.append('orange')
            symbols.append('diamond')
        elif r > target:
            colors.append('yellow')
            symbols.append('circle')
        else:
            colors.append('green')
            symbols.append('circle')

    fig.add_trace(go.Scatter(
        x=dates,
        y=repeatability,
        mode='lines+markers',
        name='Repeatability',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=10, color=colors, symbol=symbols, line=dict(width=1, color='black')),
    ))

    fig.update_layout(
        title=f"Repeatability Control Chart (Target: {target}%)",
        xaxis_title="Date",
        yaxis_title="Repeatability (%)",
        height=400,
        showlegend=True,
        hovermode="x unified",
    )

    return fig


def create_xbar_r_charts(
    records: List[RepeatabilityRecord]
) -> go.Figure:
    """Create X-bar and R control charts for SPC analysis."""
    if len(records) < 5:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 5 subgroups for X-bar/R charts",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    dates = [r.measurement_date for r in records]
    means = [r.irradiance_mean for r in records]
    ranges = [r.irradiance_range for r in records]

    # Calculate control limits
    x_bar = np.mean(means)
    r_bar = np.mean(ranges)

    # Control chart constants for n=5 (typical subgroup size)
    A2 = 0.577
    D3 = 0
    D4 = 2.114

    x_ucl = x_bar + A2 * r_bar
    x_lcl = x_bar - A2 * r_bar
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("X-bar Chart (Subgroup Means)", "R Chart (Subgroup Ranges)"),
        vertical_spacing=0.15,
    )

    # X-bar chart
    x_colors = ['red' if m > x_ucl or m < x_lcl else 'blue' for m in means]
    fig.add_trace(go.Scatter(
        x=dates,
        y=means,
        mode='lines+markers',
        name='X-bar',
        marker=dict(color=x_colors, size=8),
        line=dict(color='#1f77b4'),
    ), row=1, col=1)

    fig.add_hline(y=x_ucl, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=x_lcl, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=x_bar, line_dash="solid", line_color="green", row=1, col=1)

    # R chart
    r_colors = ['red' if r > r_ucl else 'blue' for r in ranges]
    fig.add_trace(go.Scatter(
        x=dates,
        y=ranges,
        mode='lines+markers',
        name='Range',
        marker=dict(color=r_colors, size=8),
        line=dict(color='#ff7f0e'),
    ), row=2, col=1)

    fig.add_hline(y=r_ucl, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=r_lcl, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=r_bar, line_dash="solid", line_color="green", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
    )

    fig.update_yaxes(title_text="Irradiance (W/m²)", row=1, col=1)
    fig.update_yaxes(title_text="Range (W/m²)", row=2, col=1)

    return fig


def create_histogram(records: List[RepeatabilityRecord]) -> go.Figure:
    """Create histogram of repeatability values."""
    if not records:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    repeatability = [r.repeatability_percent for r in records]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=repeatability,
        nbinsx=20,
        name='Repeatability Distribution',
        marker_color='#1f77b4',
    ))

    # Add target line
    fig.add_vline(
        x=REPEATABILITY_TARGET,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target: {REPEATABILITY_TARGET}%",
    )

    # Add mean line
    mean_val = np.mean(repeatability)
    fig.add_vline(
        x=mean_val,
        line_dash="solid",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.4f}%",
    )

    fig.update_layout(
        title="Repeatability Distribution",
        xaxis_title="Repeatability (%)",
        yaxis_title="Frequency",
        height=350,
    )

    return fig


def create_trend_chart(records: List[RepeatabilityRecord]) -> go.Figure:
    """Create trend chart with moving average."""
    if len(records) < 5:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 5 data points for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    dates = [r.measurement_date for r in records]
    repeatability = [r.repeatability_percent for r in records]

    # Calculate moving averages
    window = min(5, len(repeatability) // 2)
    if window > 0:
        ma = pd.Series(repeatability).rolling(window=window).mean()
    else:
        ma = repeatability

    fig = go.Figure()

    # Raw data
    fig.add_trace(go.Scatter(
        x=dates,
        y=repeatability,
        mode='markers',
        name='Measurements',
        marker=dict(size=8, color='#1f77b4', opacity=0.6),
    ))

    # Moving average
    fig.add_trace(go.Scatter(
        x=dates,
        y=ma,
        mode='lines',
        name=f'{window}-Point Moving Avg',
        line=dict(color='red', width=2),
    ))

    # Target line
    fig.add_hline(
        y=REPEATABILITY_TARGET,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target: {REPEATABILITY_TARGET}%",
    )

    # Trend line
    x_numeric = np.arange(len(repeatability))
    coeffs = np.polyfit(x_numeric, repeatability, 1)
    trend = np.poly1d(coeffs)(x_numeric)

    fig.add_trace(go.Scatter(
        x=dates,
        y=trend,
        mode='lines',
        name='Linear Trend',
        line=dict(color='purple', width=1, dash='dot'),
    ))

    fig.update_layout(
        title="Repeatability Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Repeatability (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_flash_variation_chart(flash_records: List[FlashRecord]) -> go.Figure:
    """Create chart showing individual flash variation."""
    if not flash_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No flash data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    flash_numbers = [f.flash_number for f in flash_records]
    irradiances = [f.irradiance for f in flash_records]

    mean_irr = np.mean(irradiances)

    fig = go.Figure()

    # Irradiance values
    fig.add_trace(go.Scatter(
        x=flash_numbers,
        y=irradiances,
        mode='lines+markers',
        name='Irradiance',
        line=dict(color='#1f77b4', width=1),
        marker=dict(size=4),
    ))

    # Mean line
    fig.add_hline(
        y=mean_irr,
        line_dash="solid",
        line_color="green",
        annotation_text=f"Mean: {mean_irr:.2f} W/m²",
    )

    # +/- 0.09% bands
    upper = mean_irr * (1 + REPEATABILITY_TARGET / 100)
    lower = mean_irr * (1 - REPEATABILITY_TARGET / 100)

    fig.add_hline(y=upper, line_dash="dash", line_color="red")
    fig.add_hline(y=lower, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Flash-to-Flash Irradiance Variation",
        xaxis_title="Flash Number",
        yaxis_title="Irradiance (W/m²)",
        height=350,
    )

    return fig


def check_western_electric_rules(data: List[float]) -> Dict[str, List[int]]:
    """Check Western Electric SPC rules for out-of-control conditions."""
    violations = {rule: [] for rule in WESTERN_ELECTRIC_RULES}

    if len(data) < 8:
        return violations

    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)

    sigma1 = mean + std
    sigma2 = mean + 2 * std
    sigma3 = mean + 3 * std
    neg_sigma1 = mean - std
    neg_sigma2 = mean - 2 * std
    neg_sigma3 = mean - 3 * std

    for i, val in enumerate(data):
        # Rule 1: One point beyond 3-sigma
        if val > sigma3 or val < neg_sigma3:
            violations["rule1"].append(i)

    # Rule 2: Two of three consecutive beyond 2-sigma
    for i in range(2, len(data)):
        window = data[i-2:i+1]
        beyond_2sigma = sum((w > sigma2 or w < neg_sigma2) for w in window)
        if beyond_2sigma >= 2:
            violations["rule2"].append(i)

    # Rule 3: Four of five consecutive beyond 1-sigma
    for i in range(4, len(data)):
        window = data[i-4:i+1]
        beyond_1sigma = sum((w > sigma1 or w < neg_sigma1) for w in window)
        if beyond_1sigma >= 4:
            violations["rule3"].append(i)

    # Rule 4: Eight consecutive on one side
    for i in range(7, len(data)):
        window = data[i-7:i+1]
        if all(w > mean for w in window) or all(w < mean for w in window):
            violations["rule4"].append(i)

    return violations


def add_demo_repeatability_data(session, lamp_id: int):
    """Add demonstration repeatability data."""
    existing = session.query(RepeatabilityRecord).filter(
        RepeatabilityRecord.lamp_id == lamp_id
    ).first()
    if existing:
        return

    base_mean = 1000.0
    base_repeatability = 0.06

    for i in range(50):
        # Simulate some variation in repeatability
        if i > 40:  # Simulate degradation
            repeat_val = base_repeatability + 0.02 + np.random.uniform(-0.01, 0.02)
        elif i > 30:  # Simulate a problem period
            repeat_val = base_repeatability + np.random.uniform(-0.01, 0.04)
        else:
            repeat_val = base_repeatability + np.random.uniform(-0.02, 0.02)

        irr_mean = base_mean + np.random.uniform(-3, 3)
        irr_std = irr_mean * (repeat_val / 100)
        irr_range = irr_std * 4  # Approximate range from std

        record = RepeatabilityRecord(
            lamp_id=lamp_id,
            measurement_date=datetime.utcnow() - timedelta(days=50 - i),
            session_id=f"SESSION-R{i:04d}",
            flash_count=10,
            start_flash_number=i * 100,
            end_flash_number=i * 100 + 10,
            irradiance_mean=irr_mean,
            irradiance_std_dev=irr_std,
            irradiance_min=irr_mean - irr_range / 2,
            irradiance_max=irr_mean + irr_range / 2,
            irradiance_range=irr_range,
            irradiance_cv_percent=repeat_val,
            repeatability_percent=repeat_val,
            repeatability_pass=repeat_val <= REPEATABILITY_TARGET,
            ucl=REPEATABILITY_TARGET * 2,
            lcl=0,
            centerline=REPEATABILITY_TARGET,
            out_of_control=repeat_val > REPEATABILITY_TARGET * 1.5,
            trend_direction="stable" if i < 30 else "degrading",
            lamp_total_flash_count=i * 100 + 10,
        )
        session.add(record)

    session.commit()


def add_demo_flash_data(session, lamp_id: int):
    """Add demonstration flash data for recent session."""
    existing = session.query(FlashRecord).filter(
        FlashRecord.lamp_id == lamp_id
    ).first()
    if existing:
        return

    base_irr = 1000.0

    for i in range(100):
        flash = FlashRecord(
            lamp_id=lamp_id,
            flash_number=i + 1,
            flash_timestamp=datetime.utcnow() - timedelta(minutes=100 - i),
            irradiance=base_irr + np.random.normal(0, 0.5),
            pulse_duration_ms=10.0 + np.random.uniform(-0.05, 0.05),
            power_percent=95.0,
            session_id="CURRENT-SESSION",
        )
        session.add(flash)

    session.commit()


# Main page content
def main():
    st.title("Repeatability Analysis")
    st.markdown('<p class="section-header">Flash-to-Flash Consistency Tracking</p>', unsafe_allow_html=True)

    # SPC info
    st.markdown("""
    <div class="spc-info">
        <strong>IEC 60904-9 Requirement:</strong> Flash-to-flash repeatability shall be better than 0.09%
        for Class A temporal stability. This page provides SPC-based monitoring and Western Electric
        rules for out-of-control detection.
    </div>
    """, unsafe_allow_html=True)

    # Initialize database
    engine = init_database()
    session = get_session(engine)

    # Sidebar controls
    with st.sidebar:
        st.header("Repeatability Controls")

        # Lamp selection
        active_lamps = get_active_lamps(session)
        lamp_options = {f"{lamp.lamp_id} ({lamp.manufacturer})": lamp.id for lamp in active_lamps}

        if lamp_options:
            selected_lamp_name = st.selectbox(
                "Select Lamp",
                options=list(lamp_options.keys()),
            )
            selected_lamp_id = lamp_options[selected_lamp_name]

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Demo Data"):
                    add_demo_repeatability_data(session, selected_lamp_id)
                    add_demo_flash_data(session, selected_lamp_id)
                    st.success("Demo data loaded!")
                    st.rerun()
        else:
            st.warning("No lamps available. Add lamps in Lamp Monitor first.")
            selected_lamp_id = None

        st.divider()

        # Analysis options
        st.subheader("Analysis Options")
        record_limit = st.slider("Records to Analyze", 10, 100, 50)
        show_spc = st.checkbox("Show SPC Charts", value=True)
        show_we_rules = st.checkbox("Check Western Electric Rules", value=True)

        st.divider()

        # Custom target
        st.subheader("Target Settings")
        custom_target = st.number_input(
            "Repeatability Target (%)",
            min_value=0.01,
            max_value=1.0,
            value=REPEATABILITY_TARGET,
            step=0.01,
            format="%.4f"
        )

    if selected_lamp_id is None:
        st.info("Please select a lamp from the sidebar to view repeatability analysis.")
        return

    lamp = session.query(Lamp).filter(Lamp.id == selected_lamp_id).first()

    # Get repeatability records
    records = session.query(RepeatabilityRecord).filter(
        RepeatabilityRecord.lamp_id == selected_lamp_id
    ).order_by(RepeatabilityRecord.measurement_date.desc()).limit(record_limit).all()

    # Get recent flash records
    flash_records = session.query(FlashRecord).filter(
        FlashRecord.lamp_id == selected_lamp_id
    ).order_by(FlashRecord.flash_timestamp.desc()).limit(100).all()

    if not records:
        st.info("No repeatability records available. Click 'Load Demo Data' to add sample data.")
        return

    records = records[::-1]  # Reverse to chronological order

    # Current status summary
    latest = records[-1]

    st.subheader(f"Current Status: {lamp.lamp_id}")

    # Pass/Fail indicator
    if latest.repeatability_pass:
        st.markdown("""
        <div class="repeat-pass">
            <strong>PASS</strong> - Repeatability within target
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="repeat-fail">
            <strong>FAIL</strong> - Repeatability exceeds target
        </div>
        """, unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = "normal" if latest.repeatability_pass else "inverse"
        st.metric(
            "Current Repeatability",
            f"{latest.repeatability_percent:.4f}%",
            delta=f"Target: {custom_target}%",
            delta_color=color
        )

    with col2:
        st.metric("Mean Irradiance", f"{latest.irradiance_mean:.2f} W/m²")

    with col3:
        st.metric("Std Dev", f"{latest.irradiance_std_dev:.4f} W/m²")

    with col4:
        st.metric("Range", f"{latest.irradiance_range:.4f} W/m²")

    with col5:
        pass_rate = sum(1 for r in records if r.repeatability_pass) / len(records) * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")

    # Check for out-of-control conditions
    if show_we_rules:
        repeat_values = [r.repeatability_percent for r in records]
        violations = check_western_electric_rules(repeat_values)

        total_violations = sum(len(v) for v in violations.values())
        if total_violations > 0:
            st.markdown("""
            <div class="ooc-warning">
                <strong>OUT-OF-CONTROL CONDITIONS DETECTED</strong>
            </div>
            """, unsafe_allow_html=True)

            for rule, indices in violations.items():
                if indices:
                    st.warning(f"**{WESTERN_ELECTRIC_RULES[rule]}**: {len(indices)} occurrence(s)")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Control Chart",
        "SPC Charts",
        "Trend Analysis",
        "Flash Variation",
        "Statistics"
    ])

    with tab1:
        st.subheader("Repeatability Control Chart")

        control_chart = create_repeatability_control_chart(records, custom_target)
        st.plotly_chart(control_chart, use_container_width=True)

        # Legend explanation
        with st.expander("Chart Legend"):
            st.markdown("""
            - **Green circles**: Pass (below target)
            - **Yellow circles**: Marginal (above target but within warning limits)
            - **Orange diamonds**: Warning (beyond 2-sigma)
            - **Red X**: Out of control (beyond 3-sigma)
            - **Solid green line**: Mean (centerline)
            - **Dashed red lines**: Control limits (3-sigma)
            - **Dotted orange lines**: Warning limits (2-sigma)
            - **Blue dash-dot line**: Target (0.09%)
            """)

    with tab2:
        if show_spc:
            st.subheader("X-bar and R Control Charts")

            st.markdown("""
            <div class="spc-info">
                X-bar charts monitor the process mean, while R charts monitor the process variability.
                Points outside control limits indicate special cause variation requiring investigation.
            </div>
            """, unsafe_allow_html=True)

            spc_charts = create_xbar_r_charts(records)
            st.plotly_chart(spc_charts, use_container_width=True)

            # Control limits summary
            if len(records) >= 5:
                means = [r.irradiance_mean for r in records]
                ranges = [r.irradiance_range for r in records]

                x_bar = np.mean(means)
                r_bar = np.mean(ranges)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**X-bar Chart Limits**")
                    st.write(f"- UCL: {x_bar + 0.577 * r_bar:.4f} W/m²")
                    st.write(f"- Centerline: {x_bar:.4f} W/m²")
                    st.write(f"- LCL: {x_bar - 0.577 * r_bar:.4f} W/m²")

                with col2:
                    st.markdown("**R Chart Limits**")
                    st.write(f"- UCL: {2.114 * r_bar:.4f} W/m²")
                    st.write(f"- Centerline: {r_bar:.4f} W/m²")
                    st.write(f"- LCL: 0 W/m²")

    with tab3:
        st.subheader("Trend Analysis")

        trend_chart = create_trend_chart(records)
        st.plotly_chart(trend_chart, use_container_width=True)

        # Trend statistics
        repeat_values = [r.repeatability_percent for r in records]
        x_numeric = np.arange(len(repeat_values))
        coeffs = np.polyfit(x_numeric, repeat_values, 1)

        slope_per_day = coeffs[0]
        trend_direction = "degrading" if slope_per_day > 0.0001 else "improving" if slope_per_day < -0.0001 else "stable"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Trend Direction", trend_direction.title())

        with col2:
            st.metric("Slope", f"{slope_per_day:.6f}%/measurement")

        with col3:
            # Predict when target will be exceeded
            if slope_per_day > 0 and repeat_values[-1] < custom_target:
                measurements_to_fail = (custom_target - repeat_values[-1]) / slope_per_day
                st.metric("Measurements to Target", f"{int(measurements_to_fail)}")
            else:
                st.metric("Measurements to Target", "N/A")

    with tab4:
        st.subheader("Flash-to-Flash Variation")

        if flash_records:
            flash_chart = create_flash_variation_chart(flash_records[::-1])
            st.plotly_chart(flash_chart, use_container_width=True)

            # Real-time statistics
            irradiances = [f.irradiance for f in flash_records if f.irradiance]
            if irradiances:
                analyzer = DriftAnalyzer()
                result = analyzer.analyze_repeatability(irradiances, custom_target)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Flash Count", len(irradiances))

                with col2:
                    st.metric("Mean", f"{result.mean:.4f} W/m²")

                with col3:
                    st.metric("Std Dev", f"{result.std_dev:.4f} W/m²")

                with col4:
                    color = "normal" if result.passes_target else "inverse"
                    st.metric(
                        "CV%",
                        f"{result.cv_percent:.4f}%",
                        delta="PASS" if result.passes_target else "FAIL",
                        delta_color=color
                    )
        else:
            st.info("No flash records available for this lamp.")

    with tab5:
        st.subheader("Statistical Summary")

        repeat_values = [r.repeatability_percent for r in records]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Repeatability Statistics**")

            stats_df = pd.DataFrame({
                'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Range', 'Median', 'Q1', 'Q3'],
                'Value': [
                    len(repeat_values),
                    f"{np.mean(repeat_values):.6f}%",
                    f"{np.std(repeat_values):.6f}%",
                    f"{np.min(repeat_values):.6f}%",
                    f"{np.max(repeat_values):.6f}%",
                    f"{np.max(repeat_values) - np.min(repeat_values):.6f}%",
                    f"{np.median(repeat_values):.6f}%",
                    f"{np.percentile(repeat_values, 25):.6f}%",
                    f"{np.percentile(repeat_values, 75):.6f}%",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**Distribution**")
            hist_chart = create_histogram(records)
            st.plotly_chart(hist_chart, use_container_width=True)

        # Process capability
        st.subheader("Process Capability")

        mean_val = np.mean(repeat_values)
        std_val = np.std(repeat_values)

        # Cp and Cpk calculations (using target as USL, 0 as LSL)
        usl = custom_target
        lsl = 0

        if std_val > 0:
            cp = (usl - lsl) / (6 * std_val)
            cpk_upper = (usl - mean_val) / (3 * std_val)
            cpk_lower = (mean_val - lsl) / (3 * std_val)
            cpk = min(cpk_upper, cpk_lower)
        else:
            cp = cpk = float('inf')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cp", f"{cp:.3f}" if cp < 10 else ">10")

        with col2:
            st.metric("Cpk", f"{cpk:.3f}" if cpk < 10 else ">10")

        with col3:
            if cpk >= 1.33:
                capability = "Excellent"
            elif cpk >= 1.0:
                capability = "Capable"
            elif cpk >= 0.67:
                capability = "Marginal"
            else:
                capability = "Incapable"
            st.metric("Capability Rating", capability)

        st.markdown("""
        **Capability Index Interpretation:**
        - Cpk ≥ 1.33: Excellent - process is highly capable
        - 1.0 ≤ Cpk < 1.33: Capable - process meets requirements
        - 0.67 ≤ Cpk < 1.0: Marginal - process needs improvement
        - Cpk < 0.67: Incapable - significant improvement needed
        """)

    # Data table
    st.divider()
    with st.expander("View Raw Data"):
        data_df = pd.DataFrame([{
            'Date': r.measurement_date,
            'Repeatability %': r.repeatability_percent,
            'Pass': 'Yes' if r.repeatability_pass else 'No',
            'Mean (W/m²)': r.irradiance_mean,
            'Std Dev': r.irradiance_std_dev,
            'Range': r.irradiance_range,
            'Flash Count': r.flash_count,
            'Session': r.session_id,
        } for r in records])

        st.dataframe(data_df, use_container_width=True)

        # Export button
        csv = data_df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name=f"repeatability_{lamp.lamp_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    session.close()


if __name__ == "__main__":
    main()
