"""
Lamp Monitor Page - SunSim-IEC60904-Classifier

Comprehensive lamp tracking with:
- Flash counter / operating hours tracker
- Calibration due date alerts
- Repeatability tracking (target: 0.09%)
- Aging warning thresholds
- Lamp replacement history

IEC 60904-9 Ed.3 compliant monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import (
    get_engine,
    get_session,
    init_database,
    Lamp,
    LampCalibration,
    FlashRecord,
    RepeatabilityRecord,
    LampReplacementHistory,
    get_active_lamps,
    get_lamps_needing_calibration,
    get_lamps_at_warning_threshold,
)

# Page configuration
st.set_page_config(
    page_title="Lamp Monitor | SunSim Classifier",
    page_icon="💡",
    layout="wide",
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-active { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-critical { color: #dc3545; font-weight: bold; }
    .status-aging { color: #17a2b8; font-weight: bold; }
    .calibration-due {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .calibration-overdue {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def get_status_color(status: str) -> str:
    """Get color class for lamp status."""
    colors = {
        "active": "status-active",
        "aging": "status-aging",
        "warning": "status-warning",
        "critical": "status-critical",
        "replaced": "status-critical",
    }
    return colors.get(status, "")


def get_status_emoji(status: str) -> str:
    """Get emoji for lamp status."""
    emojis = {
        "active": "",
        "aging": "",
        "warning": "",
        "critical": "",
        "replaced": "",
    }
    return emojis.get(status, "")


def create_life_gauge(life_percent: float, title: str) -> go.Figure:
    """Create a gauge chart for lamp life percentage."""
    # Determine color based on percentage
    if life_percent >= 95:
        color = "red"
    elif life_percent >= 80:
        color = "orange"
    elif life_percent >= 50:
        color = "yellow"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=life_percent,
        title={"text": title},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "lightyellow"},
                {"range": [80, 95], "color": "orange"},
                {"range": [95, 100], "color": "lightcoral"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 95,
            },
        },
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_flash_trend_chart(flash_records: List[FlashRecord]) -> go.Figure:
    """Create flash count trend over time."""
    if not flash_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No flash records available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    dates = [r.flash_timestamp for r in flash_records]
    flash_numbers = [r.flash_number for r in flash_records]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=flash_numbers,
        mode='lines+markers',
        name='Flash Count',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
    ))

    fig.update_layout(
        title="Flash Count Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Flash Count",
        height=300,
        margin=dict(l=50, r=20, t=50, b=50),
    )

    return fig


def create_repeatability_chart(
    repeatability_records: List[RepeatabilityRecord],
    target: float = 0.09
) -> go.Figure:
    """Create repeatability trend chart with control limits."""
    if not repeatability_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No repeatability records available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    dates = [r.measurement_date for r in repeatability_records]
    repeatability = [r.repeatability_percent for r in repeatability_records]
    ucl_values = [r.ucl if r.ucl else target * 2 for r in repeatability_records]
    lcl_values = [r.lcl if r.lcl else 0 for r in repeatability_records]

    fig = go.Figure()

    # Add control limits
    fig.add_trace(go.Scatter(
        x=dates,
        y=ucl_values,
        mode='lines',
        name='UCL',
        line=dict(color='red', width=1, dash='dash'),
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=lcl_values,
        mode='lines',
        name='LCL',
        line=dict(color='red', width=1, dash='dash'),
    ))

    # Target line
    fig.add_hline(
        y=target,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Target: {target}%",
        annotation_position="right"
    )

    # Repeatability data
    colors = ['red' if r > target else 'green' for r in repeatability]
    fig.add_trace(go.Scatter(
        x=dates,
        y=repeatability,
        mode='lines+markers',
        name='Repeatability',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color=colors),
    ))

    fig.update_layout(
        title=f"Repeatability Tracking (Target: {target}%)",
        xaxis_title="Date",
        yaxis_title="Repeatability (%)",
        height=350,
        margin=dict(l=50, r=20, t=50, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_calibration_timeline(calibrations: List[LampCalibration]) -> go.Figure:
    """Create calibration history timeline."""
    if not calibrations:
        fig = go.Figure()
        fig.add_annotation(
            text="No calibration history available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    df = pd.DataFrame([{
        'Date': c.calibration_date,
        'Type': c.calibration_type,
        'Class': c.overall_class or 'N/A',
        'UV Dev': c.uv_deviation_percent or 0,
        'VIS Dev': c.vis_deviation_percent or 0,
        'NIR Dev': c.nir_deviation_percent or 0,
    } for c in calibrations])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Calibration Timeline", "Deviations at Calibration"))

    # Timeline
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=[1] * len(df),
        mode='markers+text',
        marker=dict(size=15, color='#1f77b4'),
        text=df['Class'],
        textposition='top center',
        name='Calibrations',
    ), row=1, col=1)

    # Deviations bar chart
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['UV Dev'],
        name='UV',
        marker_color='purple',
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['VIS Dev'],
        name='VIS',
        marker_color='green',
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['NIR Dev'],
        name='NIR',
        marker_color='red',
    ), row=1, col=2)

    fig.update_layout(
        height=300,
        barmode='group',
        showlegend=True,
        margin=dict(l=50, r=20, t=50, b=50),
    )

    return fig


def display_calibration_alerts(lamp: Lamp):
    """Display calibration status and alerts."""
    if lamp.calibration_due:
        if lamp.next_calibration_date and lamp.next_calibration_date < datetime.utcnow():
            st.markdown("""
            <div class="calibration-overdue">
                <strong>CALIBRATION OVERDUE</strong><br>
                Calibration was due on: """ + lamp.next_calibration_date.strftime('%Y-%m-%d') + """
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="calibration-due">
                <strong>Calibration Due</strong><br>
                No calibration date set - calibration required.
            </div>
            """, unsafe_allow_html=True)
    else:
        days_until = lamp.days_until_calibration
        if days_until is not None and days_until <= 30:
            st.markdown(f"""
            <div class="calibration-due">
                <strong>Calibration Due Soon</strong><br>
                Calibration due in {days_until} days ({lamp.next_calibration_date.strftime('%Y-%m-%d')})
            </div>
            """, unsafe_allow_html=True)
        elif days_until is not None:
            st.info(f"Next calibration: {lamp.next_calibration_date.strftime('%Y-%m-%d')} ({days_until} days)")


def display_aging_warnings(lamp: Lamp):
    """Display lamp aging status and warnings."""
    flash_life = lamp.life_percentage
    hours_life = lamp.hours_life_percentage

    if flash_life >= lamp.critical_threshold_percent or hours_life >= lamp.critical_threshold_percent:
        st.error(f"""
        **CRITICAL: Lamp Replacement Required**
        - Flash life: {flash_life:.1f}%
        - Hours life: {hours_life:.1f}%

        Lamp has exceeded {lamp.critical_threshold_percent}% of rated life.
        Schedule replacement immediately to maintain measurement quality.
        """)
    elif flash_life >= lamp.warning_threshold_percent or hours_life >= lamp.warning_threshold_percent:
        st.warning(f"""
        **WARNING: Lamp Approaching End of Life**
        - Flash life: {flash_life:.1f}%
        - Hours life: {hours_life:.1f}%

        Plan for lamp replacement. Monitor spectral drift closely.
        """)


def add_demo_data(session):
    """Add demonstration data for testing."""
    # Check if demo lamp exists
    existing = session.query(Lamp).filter(Lamp.lamp_id == "DEMO-001").first()
    if existing:
        return existing

    # Create demo lamp
    demo_lamp = Lamp(
        lamp_id="DEMO-001",
        manufacturer="Atlas Material Testing",
        model="Xenon 6500W",
        lamp_type="Xenon",
        serial_number="XE2024001",
        flash_count=45000,
        operating_hours=450.5,
        max_flash_count=100000,
        max_operating_hours=1000.0,
        installation_date=datetime.utcnow() - timedelta(days=365),
        last_flash_date=datetime.utcnow() - timedelta(hours=2),
        is_active=True,
        status="aging",
        last_calibration_date=datetime.utcnow() - timedelta(days=180),
        next_calibration_date=datetime.utcnow() + timedelta(days=185),
        calibration_interval_days=365,
        rated_power_watts=6500,
        current_power_percent=95.0,
        warning_threshold_percent=80.0,
        critical_threshold_percent=95.0,
    )
    session.add(demo_lamp)
    session.commit()

    # Add calibration records
    for i in range(3):
        cal = LampCalibration(
            lamp_id=demo_lamp.id,
            calibration_date=datetime.utcnow() - timedelta(days=180 + i * 180),
            calibration_type="routine" if i > 0 else "initial",
            spectral_match_class="A" if i < 2 else "A+",
            uniformity_class="A",
            temporal_stability_class="A",
            overall_class="A" if i < 2 else "A+",
            uv_deviation_percent=np.random.uniform(5, 15),
            vis_deviation_percent=np.random.uniform(3, 10),
            nir_deviation_percent=np.random.uniform(5, 12),
            reference_irradiance=1000.0,
            flash_count_at_calibration=demo_lamp.flash_count - (i * 15000),
            certificate_number=f"CAL-2024-{100 + i}",
            calibrated_by="Test Lab",
            laboratory="Demo Calibration Lab",
        )
        session.add(cal)

    # Add repeatability records
    base_repeat = 0.06
    for i in range(20):
        repeat = RepeatabilityRecord(
            lamp_id=demo_lamp.id,
            measurement_date=datetime.utcnow() - timedelta(days=i * 7),
            session_id=f"SESSION-{i:04d}",
            flash_count=10,
            start_flash_number=demo_lamp.flash_count - (i * 500),
            end_flash_number=demo_lamp.flash_count - (i * 500) + 10,
            irradiance_mean=1000.0 + np.random.uniform(-5, 5),
            irradiance_std_dev=np.random.uniform(0.5, 1.5),
            irradiance_min=995.0,
            irradiance_max=1005.0,
            irradiance_range=10.0,
            irradiance_cv_percent=base_repeat + np.random.uniform(-0.02, 0.03),
            repeatability_percent=base_repeat + np.random.uniform(-0.02, 0.03),
            repeatability_pass=True if np.random.random() > 0.1 else False,
            ucl=0.15,
            lcl=0.0,
            centerline=0.09,
            out_of_control=False,
            trend_direction="stable",
            lamp_total_flash_count=demo_lamp.flash_count - (i * 500),
        )
        session.add(repeat)

    # Add flash records
    for i in range(100):
        flash = FlashRecord(
            lamp_id=demo_lamp.id,
            flash_number=demo_lamp.flash_count - i,
            flash_timestamp=datetime.utcnow() - timedelta(minutes=i * 5),
            irradiance=1000.0 + np.random.uniform(-5, 5),
            pulse_duration_ms=10.0 + np.random.uniform(-0.1, 0.1),
            uv_ratio=0.03 + np.random.uniform(-0.005, 0.005),
            vis_ratio=0.45 + np.random.uniform(-0.02, 0.02),
            nir_ratio=0.52 + np.random.uniform(-0.02, 0.02),
            power_percent=95.0,
            temporal_stability_percent=0.5 + np.random.uniform(-0.1, 0.1),
            uniformity_percent=1.5 + np.random.uniform(-0.2, 0.2),
            session_id=f"SESSION-{i // 10:04d}",
        )
        session.add(flash)

    session.commit()
    return demo_lamp


# Main page content
def main():
    st.title("Lamp Monitor")
    st.markdown('<p class="section-header">Sun Simulator Lamp Tracking & Maintenance</p>', unsafe_allow_html=True)

    # Initialize database
    engine = init_database()
    session = get_session(engine)

    # Sidebar controls
    with st.sidebar:
        st.header("Lamp Monitor Controls")

        # Add demo data button
        if st.button("Load Demo Data", help="Load demonstration lamp data"):
            add_demo_data(session)
            st.success("Demo data loaded!")
            st.rerun()

        st.divider()

        # Lamp selection
        active_lamps = get_active_lamps(session)
        lamp_options = {f"{lamp.lamp_id} ({lamp.manufacturer})": lamp.id for lamp in active_lamps}

        if lamp_options:
            selected_lamp_name = st.selectbox(
                "Select Lamp",
                options=list(lamp_options.keys()),
                help="Choose a lamp to monitor"
            )
            selected_lamp_id = lamp_options[selected_lamp_name]
        else:
            st.info("No active lamps found. Click 'Load Demo Data' to add sample data.")
            selected_lamp_id = None

        st.divider()

        # Quick filters
        st.subheader("Quick Filters")
        show_calibration_due = st.checkbox("Show Calibration Due", value=True)
        show_warnings = st.checkbox("Show Aging Warnings", value=True)

    # Main content
    if selected_lamp_id is None:
        st.info("No lamps available. Please add lamp data or load demo data from the sidebar.")
        return

    lamp = session.query(Lamp).filter(Lamp.id == selected_lamp_id).first()
    if not lamp:
        st.error("Selected lamp not found.")
        return

    # Alerts section
    if show_calibration_due:
        display_calibration_alerts(lamp)
    if show_warnings:
        display_aging_warnings(lamp)

    # Lamp overview
    st.subheader(f"Lamp: {lamp.lamp_id}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_class = get_status_color(lamp.status)
        status_emoji = get_status_emoji(lamp.status)
        st.metric(
            label="Status",
            value=f"{status_emoji} {lamp.status.upper()}",
        )

    with col2:
        st.metric(
            label="Flash Count",
            value=f"{lamp.flash_count:,}",
            delta=f"of {lamp.max_flash_count:,} max",
        )

    with col3:
        st.metric(
            label="Operating Hours",
            value=f"{lamp.operating_hours:.1f}",
            delta=f"of {lamp.max_operating_hours:.0f} max",
        )

    with col4:
        st.metric(
            label="Power Setting",
            value=f"{lamp.current_power_percent:.0f}%",
        )

    # Life gauges
    st.subheader("Lamp Life Status")
    col1, col2 = st.columns(2)

    with col1:
        flash_life_gauge = create_life_gauge(lamp.life_percentage, "Flash Count Life")
        st.plotly_chart(flash_life_gauge, use_container_width=True)

    with col2:
        hours_life_gauge = create_life_gauge(lamp.hours_life_percentage, "Operating Hours Life")
        st.plotly_chart(hours_life_gauge, use_container_width=True)

    # Tabs for different monitoring views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Flash Tracking",
        "Repeatability",
        "Calibration History",
        "Lamp Details",
        "Replacement History"
    ])

    with tab1:
        st.subheader("Flash Count Tracking")

        # Recent flash activity
        flash_records = session.query(FlashRecord).filter(
            FlashRecord.lamp_id == lamp.id
        ).order_by(FlashRecord.flash_timestamp.desc()).limit(100).all()

        if flash_records:
            flash_chart = create_flash_trend_chart(flash_records[::-1])
            st.plotly_chart(flash_chart, use_container_width=True)

            # Flash statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_irradiance = np.mean([f.irradiance for f in flash_records if f.irradiance])
                st.metric("Avg Irradiance", f"{avg_irradiance:.1f} W/m²")
            with col2:
                if lamp.last_flash_date:
                    hours_since = (datetime.utcnow() - lamp.last_flash_date).total_seconds() / 3600
                    st.metric("Hours Since Last Flash", f"{hours_since:.1f}")
            with col3:
                flashes_today = len([f for f in flash_records if f.flash_timestamp.date() == datetime.utcnow().date()])
                st.metric("Flashes Today", flashes_today)

            # Recent flash data table
            with st.expander("Recent Flash Data"):
                flash_df = pd.DataFrame([{
                    'Timestamp': f.flash_timestamp,
                    'Flash #': f.flash_number,
                    'Irradiance (W/m²)': f.irradiance,
                    'Duration (ms)': f.pulse_duration_ms,
                    'UV Ratio': f.uv_ratio,
                    'Power %': f.power_percent,
                } for f in flash_records[:20]])
                st.dataframe(flash_df, use_container_width=True)
        else:
            st.info("No flash records available for this lamp.")

    with tab2:
        st.subheader("Repeatability Tracking (Target: 0.09%)")

        repeatability_records = session.query(RepeatabilityRecord).filter(
            RepeatabilityRecord.lamp_id == lamp.id
        ).order_by(RepeatabilityRecord.measurement_date.desc()).limit(50).all()

        if repeatability_records:
            repeat_chart = create_repeatability_chart(repeatability_records[::-1])
            st.plotly_chart(repeat_chart, use_container_width=True)

            # Latest repeatability metrics
            latest = repeatability_records[0]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                color = "normal" if latest.repeatability_pass else "inverse"
                st.metric(
                    "Current Repeatability",
                    f"{latest.repeatability_percent:.4f}%",
                    delta="PASS" if latest.repeatability_pass else "FAIL",
                    delta_color=color
                )

            with col2:
                st.metric("Mean Irradiance", f"{latest.irradiance_mean:.2f} W/m²")

            with col3:
                st.metric("Std Dev", f"{latest.irradiance_std_dev:.4f}")

            with col4:
                st.metric("CV%", f"{latest.irradiance_cv_percent:.4f}%")

            # Repeatability summary statistics
            with st.expander("Repeatability Statistics"):
                all_repeat = [r.repeatability_percent for r in repeatability_records]
                pass_rate = sum(1 for r in repeatability_records if r.repeatability_pass) / len(repeatability_records) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
                with col2:
                    st.metric("Avg Repeatability", f"{np.mean(all_repeat):.4f}%")
                with col3:
                    st.metric("Best Repeatability", f"{np.min(all_repeat):.4f}%")

                # Data table
                repeat_df = pd.DataFrame([{
                    'Date': r.measurement_date,
                    'Repeatability %': r.repeatability_percent,
                    'Pass': 'Yes' if r.repeatability_pass else 'No',
                    'Mean': r.irradiance_mean,
                    'Std Dev': r.irradiance_std_dev,
                    'Flash Count': r.flash_count,
                } for r in repeatability_records])
                st.dataframe(repeat_df, use_container_width=True)
        else:
            st.info("No repeatability records available for this lamp.")

    with tab3:
        st.subheader("Calibration History")

        calibrations = session.query(LampCalibration).filter(
            LampCalibration.lamp_id == lamp.id
        ).order_by(LampCalibration.calibration_date.desc()).all()

        if calibrations:
            cal_chart = create_calibration_timeline(calibrations)
            st.plotly_chart(cal_chart, use_container_width=True)

            # Calibration details
            for cal in calibrations:
                with st.expander(
                    f"{cal.calibration_date.strftime('%Y-%m-%d')} - {cal.calibration_type.title()} - Class {cal.overall_class or 'N/A'}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Certificate:** {cal.certificate_number or 'N/A'}")
                        st.write(f"**Laboratory:** {cal.laboratory or 'N/A'}")
                        st.write(f"**Calibrated By:** {cal.calibrated_by or 'N/A'}")
                    with col2:
                        st.write(f"**UV Deviation:** {cal.uv_deviation_percent:.2f}%" if cal.uv_deviation_percent else "N/A")
                        st.write(f"**VIS Deviation:** {cal.vis_deviation_percent:.2f}%" if cal.vis_deviation_percent else "N/A")
                        st.write(f"**NIR Deviation:** {cal.nir_deviation_percent:.2f}%" if cal.nir_deviation_percent else "N/A")
                        st.write(f"**Flash Count:** {cal.flash_count_at_calibration:,}" if cal.flash_count_at_calibration else "N/A")
        else:
            st.info("No calibration records available for this lamp.")

        # Add calibration button
        st.divider()
        if st.button("Record New Calibration"):
            st.session_state.show_calibration_form = True

        if st.session_state.get('show_calibration_form', False):
            with st.form("new_calibration"):
                st.subheader("Record New Calibration")
                col1, col2 = st.columns(2)

                with col1:
                    cal_date = st.date_input("Calibration Date", value=datetime.today())
                    cal_type = st.selectbox("Calibration Type", ["routine", "initial", "post-repair"])
                    overall_class = st.selectbox("Overall Class", ["A+", "A", "B", "C"])

                with col2:
                    uv_dev = st.number_input("UV Deviation (%)", value=10.0, step=0.1)
                    vis_dev = st.number_input("VIS Deviation (%)", value=8.0, step=0.1)
                    nir_dev = st.number_input("NIR Deviation (%)", value=10.0, step=0.1)

                certificate = st.text_input("Certificate Number")
                laboratory = st.text_input("Laboratory")

                submitted = st.form_submit_button("Save Calibration")
                if submitted:
                    new_cal = LampCalibration(
                        lamp_id=lamp.id,
                        calibration_date=datetime.combine(cal_date, datetime.min.time()),
                        calibration_type=cal_type,
                        overall_class=overall_class,
                        uv_deviation_percent=uv_dev,
                        vis_deviation_percent=vis_dev,
                        nir_deviation_percent=nir_dev,
                        certificate_number=certificate,
                        laboratory=laboratory,
                        flash_count_at_calibration=lamp.flash_count,
                    )
                    session.add(new_cal)

                    # Update lamp calibration dates
                    lamp.last_calibration_date = new_cal.calibration_date
                    lamp.next_calibration_date = new_cal.calibration_date + timedelta(days=lamp.calibration_interval_days)

                    session.commit()
                    st.success("Calibration recorded successfully!")
                    st.session_state.show_calibration_form = False
                    st.rerun()

    with tab4:
        st.subheader("Lamp Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Identification**")
            st.write(f"- Lamp ID: {lamp.lamp_id}")
            st.write(f"- Serial Number: {lamp.serial_number or 'N/A'}")
            st.write(f"- Manufacturer: {lamp.manufacturer}")
            st.write(f"- Model: {lamp.model}")
            st.write(f"- Type: {lamp.lamp_type}")

            st.write("**Power**")
            st.write(f"- Rated Power: {lamp.rated_power_watts or 'N/A'} W")
            st.write(f"- Current Power: {lamp.current_power_percent}%")

        with col2:
            st.write("**Dates**")
            st.write(f"- Installed: {lamp.installation_date.strftime('%Y-%m-%d') if lamp.installation_date else 'N/A'}")
            st.write(f"- Last Flash: {lamp.last_flash_date.strftime('%Y-%m-%d %H:%M') if lamp.last_flash_date else 'N/A'}")
            st.write(f"- Last Calibration: {lamp.last_calibration_date.strftime('%Y-%m-%d') if lamp.last_calibration_date else 'N/A'}")
            st.write(f"- Next Calibration: {lamp.next_calibration_date.strftime('%Y-%m-%d') if lamp.next_calibration_date else 'N/A'}")

            st.write("**Thresholds**")
            st.write(f"- Warning at: {lamp.warning_threshold_percent}%")
            st.write(f"- Critical at: {lamp.critical_threshold_percent}%")
            st.write(f"- Calibration Interval: {lamp.calibration_interval_days} days")

        # Notes
        st.divider()
        st.write("**Notes**")
        notes = st.text_area("Lamp Notes", value=lamp.notes or "", height=100)
        if st.button("Save Notes"):
            lamp.notes = notes
            session.commit()
            st.success("Notes saved!")

    with tab5:
        st.subheader("Lamp Replacement History")

        replacements = session.query(LampReplacementHistory).order_by(
            LampReplacementHistory.replacement_date.desc()
        ).all()

        if replacements:
            for rep in replacements:
                with st.expander(f"{rep.replacement_date.strftime('%Y-%m-%d')} - {rep.old_lamp_id} -> {rep.new_lamp_id}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Old Lamp**")
                        st.write(f"- ID: {rep.old_lamp_id}")
                        st.write(f"- Manufacturer: {rep.old_lamp_manufacturer}")
                        st.write(f"- Model: {rep.old_lamp_model}")
                        st.write(f"- Flash Count: {rep.old_lamp_flash_count:,}" if rep.old_lamp_flash_count else "N/A")
                        st.write(f"- Hours: {rep.old_lamp_operating_hours:.1f}" if rep.old_lamp_operating_hours else "N/A")
                    with col2:
                        st.write("**New Lamp**")
                        st.write(f"- ID: {rep.new_lamp_id}")
                        st.write(f"- Manufacturer: {rep.new_lamp_manufacturer}")
                        st.write(f"- Model: {rep.new_lamp_model}")
                    st.write(f"**Reason:** {rep.replacement_reason or 'N/A'}")
                    st.write(f"**Replaced By:** {rep.replaced_by or 'N/A'}")
        else:
            st.info("No replacement history available.")

        # Record replacement button
        st.divider()
        if st.button("Record Lamp Replacement"):
            st.session_state.show_replacement_form = True

        if st.session_state.get('show_replacement_form', False):
            with st.form("new_replacement"):
                st.subheader("Record Lamp Replacement")

                new_lamp_id = st.text_input("New Lamp ID")
                new_manufacturer = st.text_input("New Lamp Manufacturer")
                new_model = st.text_input("New Lamp Model")
                reason = st.selectbox("Replacement Reason", ["aging", "failure", "upgrade", "calibration"])
                replaced_by = st.text_input("Replaced By")

                submitted = st.form_submit_button("Record Replacement")
                if submitted and new_lamp_id:
                    # Record replacement history
                    replacement = LampReplacementHistory(
                        old_lamp_id=lamp.lamp_id,
                        old_lamp_manufacturer=lamp.manufacturer,
                        old_lamp_model=lamp.model,
                        old_lamp_flash_count=lamp.flash_count,
                        old_lamp_operating_hours=lamp.operating_hours,
                        old_lamp_installation_date=lamp.installation_date,
                        new_lamp_id=new_lamp_id,
                        new_lamp_manufacturer=new_manufacturer,
                        new_lamp_model=new_model,
                        replacement_reason=reason,
                        replaced_by=replaced_by,
                    )
                    session.add(replacement)

                    # Deactivate old lamp
                    lamp.is_active = False
                    lamp.status = "replaced"

                    # Create new lamp
                    new_lamp = Lamp(
                        lamp_id=new_lamp_id,
                        manufacturer=new_manufacturer,
                        model=new_model,
                        lamp_type=lamp.lamp_type,
                        installation_date=datetime.utcnow(),
                        is_active=True,
                        status="active",
                        max_flash_count=lamp.max_flash_count,
                        max_operating_hours=lamp.max_operating_hours,
                        calibration_interval_days=lamp.calibration_interval_days,
                        warning_threshold_percent=lamp.warning_threshold_percent,
                        critical_threshold_percent=lamp.critical_threshold_percent,
                    )
                    session.add(new_lamp)
                    session.commit()

                    st.success("Lamp replacement recorded successfully!")
                    st.session_state.show_replacement_form = False
                    st.rerun()

    # Close session
    session.close()


if __name__ == "__main__":
    main()
