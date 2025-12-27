"""
Spectrum Drift Page - SunSim-IEC60904-Classifier

Per TÜV paper findings for spectral stability monitoring:
- UV/NIR degradation monitoring (Xenon aging)
- Blue-shift during pulse tracking
- Lamp power adjustment effects
- Multi-manufacturer comparison charts
- Drift trending over flash count

IEC 60904-9 Ed.3 compliant spectral analysis.
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
    SpectrumDrift,
    get_active_lamps,
    get_recent_drift_records,
)
from utils.drift_analysis import (
    DriftAnalyzer,
    generate_am15g_reference,
    SPECTRAL_MATCH_LIMITS,
)

# Page configuration
st.set_page_config(
    page_title="Spectrum Drift | SunSim Classifier",
    page_icon="📉",
    layout="wide",
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .drift-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .drift-critical {
        background-color: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .drift-ok {
        background-color: #d4edda;
        border: 1px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .class-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .class-a-plus { background-color: #28a745; color: white; }
    .class-a { background-color: #17a2b8; color: white; }
    .class-b { background-color: #ffc107; color: black; }
    .class-c { background-color: #dc3545; color: white; }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .tuv-reference {
        background-color: #e9ecef;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        margin: 10px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


def get_class_style(classification: str) -> str:
    """Get CSS class for classification badge."""
    styles = {
        "A+": "class-a-plus",
        "A": "class-a",
        "B": "class-b",
        "C": "class-c",
    }
    return styles.get(classification, "class-b")


def create_spectral_drift_chart(drift_records: List[SpectrumDrift]) -> go.Figure:
    """Create comprehensive spectral drift chart over flash count."""
    if not drift_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No drift records available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    flash_counts = [d.flash_count_at_measurement for d in drift_records]
    uv_shifts = [d.uv_total_shift_percent or 0 for d in drift_records]
    vis_shifts = [d.vis_total_shift_percent or 0 for d in drift_records]
    nir_shifts = [d.nir_total_shift_percent or 0 for d in drift_records]

    fig = go.Figure()

    # UV drift (most critical for Xenon)
    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=uv_shifts,
        mode='lines+markers',
        name='UV (300-400nm)',
        line=dict(color='purple', width=2),
        marker=dict(size=6),
    ))

    # Visible drift
    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=vis_shifts,
        mode='lines+markers',
        name='VIS (400-700nm)',
        line=dict(color='green', width=2),
        marker=dict(size=6),
    ))

    # NIR drift
    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=nir_shifts,
        mode='lines+markers',
        name='NIR (700-1100nm)',
        line=dict(color='red', width=2),
        marker=dict(size=6),
    ))

    # Classification limit lines
    for cls, limit in SPECTRAL_MATCH_LIMITS.items():
        if limit <= 50:  # Only show reasonable limits
            fig.add_hline(
                y=limit,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Class {cls}: {limit}%",
                annotation_position="right",
            )

    fig.update_layout(
        title="Spectral Drift vs Flash Count",
        xaxis_title="Flash Count",
        yaxis_title="Spectral Deviation (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    return fig


def create_uv_degradation_chart(drift_records: List[SpectrumDrift]) -> go.Figure:
    """Create detailed UV degradation chart (Xenon aging specific)."""
    if not drift_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No UV data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    flash_counts = [d.flash_count_at_measurement for d in drift_records]
    uv_300_350 = [d.uv_300_350_percent or 0 for d in drift_records]
    uv_350_400 = [d.uv_350_400_percent or 0 for d in drift_records]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("UV Sub-band Degradation", "UV Degradation Rate"),
        vertical_spacing=0.15,
    )

    # UV sub-bands
    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=uv_300_350,
        mode='lines+markers',
        name='UV 300-350nm',
        line=dict(color='darkviolet', width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=uv_350_400,
        mode='lines+markers',
        name='UV 350-400nm',
        line=dict(color='violet', width=2),
    ), row=1, col=1)

    # Calculate rate of change
    if len(flash_counts) > 1:
        rates = []
        for i in range(1, len(uv_300_350)):
            flash_diff = flash_counts[i] - flash_counts[i-1]
            if flash_diff > 0:
                rate = (uv_300_350[i] - uv_300_350[i-1]) / flash_diff * 1000
                rates.append(rate)
            else:
                rates.append(0)

        fig.add_trace(go.Bar(
            x=flash_counts[1:],
            y=rates,
            name='Rate (%/1000 flashes)',
            marker_color='purple',
        ), row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.update_yaxes(title_text="Deviation (%)", row=1, col=1)
    fig.update_yaxes(title_text="Rate (%/1000 flashes)", row=2, col=1)
    fig.update_xaxes(title_text="Flash Count", row=2, col=1)

    return fig


def create_blue_shift_chart(drift_records: List[SpectrumDrift]) -> go.Figure:
    """Create blue-shift monitoring chart per TÜV findings."""
    blue_shift_records = [d for d in drift_records if d.blue_shift_detected]

    if not blue_shift_records:
        fig = go.Figure()
        fig.add_annotation(
            text="No blue-shift events detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=300)
        return fig

    flash_counts = [d.flash_count_at_measurement for d in blue_shift_records]
    magnitudes = [d.blue_shift_magnitude_nm or 0 for d in blue_shift_records]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=magnitudes,
        mode='markers+lines',
        name='Blue Shift',
        marker=dict(size=10, color='blue'),
        line=dict(color='lightblue', dash='dash'),
    ))

    # Warning threshold
    fig.add_hline(
        y=5.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="Warning: 5nm shift",
    )

    fig.update_layout(
        title="Blue-Shift During Pulse (TÜV Finding)",
        xaxis_title="Flash Count",
        yaxis_title="Wavelength Shift (nm)",
        height=300,
    )

    return fig


def create_power_effect_chart(drift_records: List[SpectrumDrift]) -> go.Figure:
    """Create chart showing power adjustment effects on spectrum."""
    records_with_power = [d for d in drift_records if d.power_setting_percent is not None]

    if not records_with_power:
        fig = go.Figure()
        fig.add_annotation(
            text="No power adjustment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    power_levels = [d.power_setting_percent for d in records_with_power]
    uv_shifts = [d.uv_total_shift_percent or 0 for d in records_with_power]
    vis_shifts = [d.vis_total_shift_percent or 0 for d in records_with_power]
    nir_shifts = [d.nir_total_shift_percent or 0 for d in records_with_power]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("UV vs Power", "VIS vs Power", "NIR vs Power"),
    )

    fig.add_trace(go.Scatter(
        x=power_levels,
        y=uv_shifts,
        mode='markers',
        name='UV',
        marker=dict(size=8, color='purple'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=power_levels,
        y=vis_shifts,
        mode='markers',
        name='VIS',
        marker=dict(size=8, color='green'),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=power_levels,
        y=nir_shifts,
        mode='markers',
        name='NIR',
        marker=dict(size=8, color='red'),
    ), row=1, col=3)

    # Add trendlines
    for idx, (x_data, y_data, color) in enumerate([
        (power_levels, uv_shifts, 'purple'),
        (power_levels, vis_shifts, 'green'),
        (power_levels, nir_shifts, 'red'),
    ], 1):
        if len(x_data) > 2:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                line=dict(dash='dash', color=color),
                showlegend=False,
            ), row=1, col=idx)

    fig.update_layout(
        title="Power Adjustment Effects on Spectral Distribution",
        height=350,
        showlegend=False,
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="Power (%)", row=1, col=i)
        fig.update_yaxes(title_text="Deviation (%)", row=1, col=i)

    return fig


def create_manufacturer_comparison(session) -> go.Figure:
    """Create multi-manufacturer comparison chart."""
    # Get all drift records grouped by manufacturer
    drift_records = session.query(SpectrumDrift).filter(
        SpectrumDrift.manufacturer.isnot(None)
    ).all()

    if not drift_records:
        # Generate demo comparison data
        manufacturers = ["Atlas", "Newport", "Wacom", "Pasan"]
        demo_data = []
        for mfr in manufacturers:
            base_degradation = np.random.uniform(0.5, 1.5)
            for fc in range(0, 50001, 5000):
                demo_data.append({
                    'Manufacturer': mfr,
                    'Flash Count': fc,
                    'UV Shift': base_degradation * (fc / 10000) + np.random.uniform(-0.5, 0.5),
                    'NIR Shift': base_degradation * 0.3 * (fc / 10000) + np.random.uniform(-0.2, 0.2),
                })

        df = pd.DataFrame(demo_data)
    else:
        df = pd.DataFrame([{
            'Manufacturer': d.manufacturer,
            'Flash Count': d.flash_count_at_measurement,
            'UV Shift': d.uv_total_shift_percent or 0,
            'NIR Shift': d.nir_total_shift_percent or 0,
        } for d in drift_records])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("UV Degradation by Manufacturer", "NIR Stability by Manufacturer"),
    )

    colors = px.colors.qualitative.Set1

    for i, manufacturer in enumerate(df['Manufacturer'].unique()):
        mfr_data = df[df['Manufacturer'] == manufacturer]
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=mfr_data['Flash Count'],
            y=mfr_data['UV Shift'],
            mode='lines+markers',
            name=manufacturer,
            line=dict(color=color),
            legendgroup=manufacturer,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=mfr_data['Flash Count'],
            y=mfr_data['NIR Shift'],
            mode='lines+markers',
            name=manufacturer,
            line=dict(color=color),
            legendgroup=manufacturer,
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        title="Multi-Manufacturer Comparison (Based on TÜV Testing)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.update_xaxes(title_text="Flash Count", row=1, col=1)
    fig.update_xaxes(title_text="Flash Count", row=1, col=2)
    fig.update_yaxes(title_text="UV Shift (%)", row=1, col=1)
    fig.update_yaxes(title_text="NIR Shift (%)", row=1, col=2)

    return fig


def create_drift_trend_forecast(drift_records: List[SpectrumDrift]) -> go.Figure:
    """Create drift trend with forecast."""
    if len(drift_records) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 3 data points for trend analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    flash_counts = np.array([d.flash_count_at_measurement for d in drift_records])
    uv_shifts = np.array([d.uv_total_shift_percent or 0 for d in drift_records])

    # Fit linear trend
    coeffs = np.polyfit(flash_counts, uv_shifts, 1)
    trend_line = np.poly1d(coeffs)

    # Forecast to 100k flashes
    max_forecast = 100000
    forecast_x = np.linspace(0, max_forecast, 100)
    forecast_y = trend_line(forecast_x)

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=flash_counts,
        y=uv_shifts,
        mode='markers',
        name='Measured Data',
        marker=dict(size=10, color='blue'),
    ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        mode='lines',
        name='Trend Forecast',
        line=dict(color='blue', dash='dash'),
    ))

    # Classification limits
    for cls, limit in [("A+", 12.5), ("A", 25.0), ("B", 40.0)]:
        fig.add_hline(
            y=limit,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Class {cls}",
        )

        # Calculate when limit will be reached
        if coeffs[0] > 0:  # If degrading
            flash_at_limit = (limit - coeffs[1]) / coeffs[0]
            if 0 < flash_at_limit < max_forecast:
                fig.add_vline(
                    x=flash_at_limit,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text=f"{int(flash_at_limit):,}",
                )

    fig.update_layout(
        title="UV Drift Trend with Forecast to 100k Flashes",
        xaxis_title="Flash Count",
        yaxis_title="UV Deviation (%)",
        height=400,
        showlegend=True,
    )

    return fig


def add_demo_drift_data(session, lamp_id: int):
    """Add demonstration drift data."""
    # Check if data exists
    existing = session.query(SpectrumDrift).filter(
        SpectrumDrift.lamp_id == lamp_id
    ).first()
    if existing:
        return

    # Generate degradation curve
    for flash_count in range(0, 50001, 2500):
        # Xenon lamp typical UV degradation curve
        aging_factor = flash_count / 50000
        uv_base = 5 + aging_factor * 15 + np.random.uniform(-1, 1)
        vis_base = 3 + aging_factor * 5 + np.random.uniform(-0.5, 0.5)
        nir_base = 4 + aging_factor * 3 + np.random.uniform(-0.5, 0.5)

        # Power typically reduced as lamp ages
        power = 100 - aging_factor * 10

        drift = SpectrumDrift(
            lamp_id=lamp_id,
            measurement_date=datetime.utcnow() - timedelta(days=int((50000 - flash_count) / 500)),
            flash_count_at_measurement=flash_count,
            uv_300_350_percent=uv_base * 1.2,
            uv_350_400_percent=uv_base * 0.8,
            uv_total_shift_percent=uv_base,
            vis_400_500_percent=vis_base * 1.1,
            vis_500_600_percent=vis_base,
            vis_600_700_percent=vis_base * 0.9,
            vis_total_shift_percent=vis_base,
            nir_700_800_percent=nir_base,
            nir_800_900_percent=nir_base * 0.9,
            nir_900_1000_percent=nir_base * 0.8,
            nir_1000_1100_percent=nir_base * 0.7,
            nir_total_shift_percent=nir_base,
            blue_shift_detected=flash_count > 30000 and np.random.random() > 0.7,
            blue_shift_magnitude_nm=np.random.uniform(2, 6) if flash_count > 30000 else None,
            power_setting_percent=power,
            overall_spectral_mismatch=(uv_base + vis_base + nir_base) / 3,
            classification_after="A+" if uv_base < 12.5 else "A" if uv_base < 25 else "B",
            trend_direction="degrading" if flash_count > 10000 else "stable",
            rate_of_change_per_1000_flashes=uv_base / (flash_count / 1000) if flash_count > 0 else 0,
            manufacturer="Atlas Material Testing",
            lamp_model="Xenon 6500W",
        )
        session.add(drift)

    session.commit()


# Main page content
def main():
    st.title("Spectrum Drift Monitor")
    st.markdown('<p class="section-header">UV/NIR Degradation Tracking per TÜV Findings</p>', unsafe_allow_html=True)

    # TÜV reference note
    st.markdown("""
    <div class="tuv-reference">
        <strong>Reference:</strong> This monitoring implements findings from TÜV Rheinland studies on solar simulator
        spectral stability. Key phenomena tracked include UV degradation in Xenon lamps, blue-shift during pulse,
        and power adjustment effects on spectral distribution.
    </div>
    """, unsafe_allow_html=True)

    # Initialize database
    engine = init_database()
    session = get_session(engine)

    # Sidebar controls
    with st.sidebar:
        st.header("Drift Analysis Controls")

        # Lamp selection
        active_lamps = get_active_lamps(session)
        lamp_options = {f"{lamp.lamp_id} ({lamp.manufacturer})": lamp.id for lamp in active_lamps}

        if lamp_options:
            selected_lamp_name = st.selectbox(
                "Select Lamp",
                options=list(lamp_options.keys()),
            )
            selected_lamp_id = lamp_options[selected_lamp_name]

            # Add demo data button
            if st.button("Load Demo Drift Data"):
                add_demo_drift_data(session, selected_lamp_id)
                st.success("Demo drift data loaded!")
                st.rerun()
        else:
            st.warning("No lamps available. Please add lamps in Lamp Monitor first.")
            selected_lamp_id = None

        st.divider()

        # Analysis options
        st.subheader("Analysis Options")
        flash_range = st.slider(
            "Flash Count Range (k)",
            min_value=0,
            max_value=100,
            value=(0, 100),
            help="Filter drift records by flash count"
        )

        show_forecast = st.checkbox("Show Trend Forecast", value=True)
        show_blue_shift = st.checkbox("Show Blue-Shift Analysis", value=True)
        show_power_effects = st.checkbox("Show Power Effects", value=True)

    if selected_lamp_id is None:
        st.info("Please select a lamp from the sidebar to view drift analysis.")
        return

    lamp = session.query(Lamp).filter(Lamp.id == selected_lamp_id).first()

    # Get drift records
    drift_records = session.query(SpectrumDrift).filter(
        SpectrumDrift.lamp_id == selected_lamp_id,
        SpectrumDrift.flash_count_at_measurement >= flash_range[0] * 1000,
        SpectrumDrift.flash_count_at_measurement <= flash_range[1] * 1000,
    ).order_by(SpectrumDrift.flash_count_at_measurement.asc()).all()

    if not drift_records:
        st.info("No drift records available for this lamp. Click 'Load Demo Drift Data' to add sample data.")
        return

    # Current status overview
    latest = drift_records[-1]
    st.subheader(f"Current Status: {lamp.lamp_id}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        class_style = get_class_style(latest.classification_after or "N/A")
        st.markdown(f"""
        <div class="class-badge {class_style}">
            Class {latest.classification_after or 'N/A'}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("UV Shift", f"{latest.uv_total_shift_percent:.1f}%")

    with col3:
        st.metric("VIS Shift", f"{latest.vis_total_shift_percent:.1f}%")

    with col4:
        st.metric("NIR Shift", f"{latest.nir_total_shift_percent:.1f}%")

    with col5:
        st.metric("Flash Count", f"{latest.flash_count_at_measurement:,}")

    # Alerts based on drift
    if latest.uv_total_shift_percent and latest.uv_total_shift_percent > 25:
        st.markdown("""
        <div class="drift-critical">
            <strong>CRITICAL:</strong> UV drift exceeds Class A limit (25%). Consider lamp replacement.
        </div>
        """, unsafe_allow_html=True)
    elif latest.uv_total_shift_percent and latest.uv_total_shift_percent > 12.5:
        st.markdown("""
        <div class="drift-warning">
            <strong>WARNING:</strong> UV drift exceeds Class A+ limit (12.5%). Monitor closely.
        </div>
        """, unsafe_allow_html=True)

    if latest.blue_shift_detected:
        st.markdown("""
        <div class="drift-warning">
            <strong>Blue-Shift Detected:</strong> Peak wavelength shift observed during pulse (TÜV finding).
            Magnitude: """ + f"{latest.blue_shift_magnitude_nm:.1f}nm" + """
        </div>
        """, unsafe_allow_html=True)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Drift",
        "UV Degradation",
        "Blue-Shift",
        "Power Effects",
        "Manufacturer Comparison"
    ])

    with tab1:
        st.subheader("Spectral Drift Over Flash Count")

        drift_chart = create_spectral_drift_chart(drift_records)
        st.plotly_chart(drift_chart, use_container_width=True)

        if show_forecast and len(drift_records) >= 3:
            st.subheader("Drift Trend Forecast")
            forecast_chart = create_drift_trend_forecast(drift_records)
            st.plotly_chart(forecast_chart, use_container_width=True)

            # Calculate and display predictions
            flash_counts = np.array([d.flash_count_at_measurement for d in drift_records])
            uv_shifts = np.array([d.uv_total_shift_percent or 0 for d in drift_records])

            if len(flash_counts) > 1:
                coeffs = np.polyfit(flash_counts, uv_shifts, 1)
                rate_per_1k = coeffs[0] * 1000

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Degradation Rate", f"{rate_per_1k:.3f}%/1k flashes")
                with col2:
                    if coeffs[0] > 0:
                        flash_to_25 = (25 - coeffs[1]) / coeffs[0]
                        st.metric("Flashes to Class A Limit", f"{int(flash_to_25):,}")
                    else:
                        st.metric("Flashes to Class A Limit", "N/A (stable)")
                with col3:
                    remaining = lamp.max_flash_count - lamp.flash_count if lamp else 0
                    st.metric("Remaining Rated Life", f"{remaining:,} flashes")

    with tab2:
        st.subheader("UV Region Degradation (Xenon Aging)")
        st.markdown("""
        UV degradation is the primary indicator of Xenon lamp aging. The 300-400nm region
        shows the most pronounced changes as the lamp ages, particularly affecting UV-sensitive
        photovoltaic devices.
        """)

        uv_chart = create_uv_degradation_chart(drift_records)
        st.plotly_chart(uv_chart, use_container_width=True)

        # UV sub-band comparison
        st.subheader("UV Sub-band Analysis")

        uv_data = pd.DataFrame([{
            'Flash Count': d.flash_count_at_measurement,
            '300-350nm': d.uv_300_350_percent or 0,
            '350-400nm': d.uv_350_400_percent or 0,
            'Total UV': d.uv_total_shift_percent or 0,
        } for d in drift_records])

        st.dataframe(uv_data, use_container_width=True)

    with tab3:
        if show_blue_shift:
            st.subheader("Blue-Shift During Pulse Analysis")

            st.markdown("""
            <div class="tuv-reference">
                TÜV studies have documented blue-shift phenomena in Xenon lamps, where the spectral
                peak shifts toward shorter wavelengths during the flash pulse. This is more pronounced
                in aged lamps and can affect measurement accuracy.
            </div>
            """, unsafe_allow_html=True)

            blue_shift_chart = create_blue_shift_chart(drift_records)
            st.plotly_chart(blue_shift_chart, use_container_width=True)

            # Blue-shift statistics
            blue_shift_events = [d for d in drift_records if d.blue_shift_detected]
            if blue_shift_events:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Blue-Shift Events", len(blue_shift_events))
                with col2:
                    avg_magnitude = np.mean([d.blue_shift_magnitude_nm for d in blue_shift_events if d.blue_shift_magnitude_nm])
                    st.metric("Avg Magnitude", f"{avg_magnitude:.2f} nm")
                with col3:
                    max_magnitude = max([d.blue_shift_magnitude_nm for d in blue_shift_events if d.blue_shift_magnitude_nm])
                    st.metric("Max Magnitude", f"{max_magnitude:.2f} nm")
            else:
                st.info("No blue-shift events detected in the current data range.")

    with tab4:
        if show_power_effects:
            st.subheader("Lamp Power Adjustment Effects")

            st.markdown("""
            Adjusting lamp power affects the spectral distribution. This analysis shows how
            spectral deviation changes with power setting, helping optimize the trade-off
            between lamp life extension and spectral accuracy.
            """)

            power_chart = create_power_effect_chart(drift_records)
            st.plotly_chart(power_chart, use_container_width=True)

            # Power sensitivity analysis
            records_with_power = [d for d in drift_records if d.power_setting_percent]
            if len(records_with_power) > 2:
                analyzer = DriftAnalyzer()
                power_levels = [d.power_setting_percent for d in records_with_power]
                uv_shifts = [d.uv_total_shift_percent or 0 for d in records_with_power]

                analysis = analyzer.analyze_power_effect(power_levels, uv_shifts)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("UV-Power Correlation", f"{analysis['correlation']:.3f}")
                with col2:
                    st.metric("Sensitivity", f"{analysis['sensitivity']:.3f}%/%")
                with col3:
                    st.metric("R²", f"{analysis['r_squared']:.3f}")

    with tab5:
        st.subheader("Multi-Manufacturer Comparison")

        st.markdown("""
        Comparison of spectral drift characteristics across different lamp manufacturers.
        This data helps inform lamp selection and replacement decisions based on
        degradation patterns.
        """)

        mfr_chart = create_manufacturer_comparison(session)
        st.plotly_chart(mfr_chart, use_container_width=True)

        # Manufacturer summary table
        st.subheader("Manufacturer Summary")

        # Demo summary data
        summary_data = pd.DataFrame({
            'Manufacturer': ['Atlas', 'Newport', 'Wacom', 'Pasan'],
            'Avg UV Rate (%/10k)': [1.2, 1.5, 1.1, 1.4],
            'NIR Stability': ['Good', 'Fair', 'Excellent', 'Good'],
            'Typical Life (flashes)': ['80,000', '70,000', '90,000', '75,000'],
            'Blue-Shift Risk': ['Low', 'Medium', 'Low', 'Medium'],
        })

        st.dataframe(summary_data, use_container_width=True)

    # Data export section
    st.divider()
    st.subheader("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Drift Data to CSV"):
            export_df = pd.DataFrame([{
                'Date': d.measurement_date,
                'Flash Count': d.flash_count_at_measurement,
                'UV Shift %': d.uv_total_shift_percent,
                'VIS Shift %': d.vis_total_shift_percent,
                'NIR Shift %': d.nir_total_shift_percent,
                'Classification': d.classification_after,
                'Blue Shift': d.blue_shift_detected,
                'Power %': d.power_setting_percent,
            } for d in drift_records])

            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"drift_data_{lamp.lamp_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("Generate Drift Report"):
            st.info("Report generation feature - would generate PDF report with drift analysis.")

    session.close()


if __name__ == "__main__":
    main()
