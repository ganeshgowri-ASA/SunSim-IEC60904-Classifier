"""
Temporal Stability Analysis Page
IEC 60904-9 Ed.3 Solar Simulator Classification

STI (Short Term Instability) and LTI (Long Term Instability) analysis
with temporal stability metrics and time-series visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    TemporalMeasurement,
    TemporalStabilityResult,
    get_grade_color,
)
from utils.simulator_ui import (
    render_simulator_selector,
    render_simulator_summary_card,
    get_selected_simulator,
    get_simulator_id_for_db,
)

# Import database utilities
try:
    from utils.db import insert_simulator_selection
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Temporal Stability | SunSim",
    page_icon=":material/timeline:",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }

    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    .grade-badge-large {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 16px;
        min-width: 120px;
    }

    .grade-badge-medium {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        min-width: 60px;
    }

    .grade-a-plus { background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; }
    .grade-a { background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); color: white; }
    .grade-b { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: white; }
    .grade-c { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; }

    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .formula-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 1rem 0;
    }

    .dual-grade-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        text-align: center;
    }

    .grade-label {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


def generate_sample_temporal_data(
    duration_seconds: float = 60.0,
    sampling_rate: float = 100.0,
    quality: str = "A"
) -> TemporalStabilityResult:
    """Generate sample temporal stability measurement data"""

    # Quality settings - controls temporal instability
    quality_instability = {
        "A+": 0.002,   # 0.2% base instability -> ~0.4% STI
        "A": 0.008,    # 0.8% base instability -> ~1.6% STI
        "B": 0.020,    # 2% base instability -> ~4% STI
        "C": 0.040,    # 4% base instability -> ~8% STI
    }

    base_instability = quality_instability.get(quality, 0.008)
    np.random.seed(42)

    result = TemporalStabilityResult(
        measurement_duration_s=duration_seconds,
        sampling_rate_hz=sampling_rate
    )

    # Target irradiance (1000 W/m²)
    target_irradiance = 1000.0

    # Number of samples
    num_samples = int(duration_seconds * sampling_rate)

    # Generate time series with realistic noise patterns
    t = np.linspace(0, duration_seconds, num_samples)

    # Add multiple noise components:
    # 1. High-frequency noise (electronic)
    hf_noise = np.random.normal(0, base_instability * 0.3, num_samples)

    # 2. Low-frequency drift (thermal)
    lf_drift = base_instability * 0.5 * np.sin(2 * np.pi * t / duration_seconds * 0.5)

    # 3. Random walk component
    random_walk = np.cumsum(np.random.normal(0, base_instability * 0.01, num_samples))
    random_walk = random_walk - np.mean(random_walk)  # Center around zero

    # 4. Occasional spikes (lamp flicker)
    spikes = np.zeros(num_samples)
    spike_positions = np.random.choice(num_samples, size=int(num_samples * 0.001), replace=False)
    spikes[spike_positions] = np.random.uniform(-base_instability, base_instability, len(spike_positions))

    # Combine all components
    total_variation = hf_noise + lf_drift + random_walk + spikes

    for i in range(num_samples):
        irradiance = target_irradiance * (1 + total_variation[i])
        measurement = TemporalMeasurement(
            timestamp=t[i],
            irradiance=irradiance
        )
        result.measurements.append(measurement)

    result.calculate_grade()
    return result


def get_grade_class(grade: ClassificationGrade) -> str:
    """Get CSS class for grade badge"""
    grade_classes = {
        ClassificationGrade.A_PLUS: "grade-a-plus",
        ClassificationGrade.A: "grade-a",
        ClassificationGrade.B: "grade-b",
        ClassificationGrade.C: "grade-c",
    }
    return grade_classes.get(grade, "grade-c")


def create_time_series_chart(result: TemporalStabilityResult, show_all: bool = False) -> go.Figure:
    """Create irradiance time series chart"""

    timestamps = [m.timestamp for m in result.measurements]
    irradiances = [m.irradiance for m in result.measurements]

    # Downsample for display if too many points
    if not show_all and len(timestamps) > 2000:
        step = len(timestamps) // 2000
        timestamps = timestamps[::step]
        irradiances = irradiances[::step]

    fig = go.Figure()

    # Add irradiance trace
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=irradiances,
        mode='lines',
        name='Irradiance',
        line=dict(color='#6366F1', width=1),
        hovertemplate='Time: %{x:.2f}s<br>Irradiance: %{y:.2f} W/m²<extra></extra>'
    ))

    # Add mean line
    fig.add_hline(
        y=result.mean_irradiance,
        line_dash="dash",
        line_color="#1E3A5F",
        line_width=2,
        annotation_text=f"Mean: {result.mean_irradiance:.1f} W/m²",
        annotation_position="top right"
    )

    # Add threshold bands for A+ classification
    threshold_sti = CLASSIFICATION_THRESHOLDS["temporal_sti"][ClassificationGrade.A_PLUS]
    upper_bound = result.mean_irradiance * (1 + threshold_sti / 100)
    lower_bound = result.mean_irradiance * (1 - threshold_sti / 100)

    fig.add_hrect(
        y0=lower_bound, y1=upper_bound,
        fillcolor="rgba(16, 185, 129, 0.1)",
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        title=dict(
            text="Irradiance Time Series",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Irradiance (W/m²)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
        hovermode="x unified"
    )

    return fig


def create_rolling_instability_chart(result: TemporalStabilityResult, window_seconds: float = 1.0) -> go.Figure:
    """Create rolling window instability chart"""

    timestamps = np.array([m.timestamp for m in result.measurements])
    irradiances = np.array([m.irradiance for m in result.measurements])

    # Calculate rolling window instability
    window_samples = int(window_seconds * result.sampling_rate_hz)
    if window_samples < 2:
        window_samples = 2

    rolling_instability = []
    rolling_times = []

    for i in range(0, len(irradiances) - window_samples, window_samples // 2):
        window = irradiances[i:i + window_samples]
        e_max = np.max(window)
        e_min = np.min(window)
        instability = (e_max - e_min) / (e_max + e_min) * 100
        rolling_instability.append(instability)
        rolling_times.append(timestamps[i + window_samples // 2])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_times,
        y=rolling_instability,
        mode='lines',
        name='Rolling Instability',
        line=dict(color='#F59E0B', width=2),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
        hovertemplate='Time: %{x:.2f}s<br>Instability: %{y:.3f}%<extra></extra>'
    ))

    # Add threshold lines
    for grade, threshold in CLASSIFICATION_THRESHOLDS["temporal_sti"].items():
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=get_grade_color(grade),
            line_width=1,
            annotation_text=f"{grade.value}: {threshold}%",
            annotation_position="right"
        )

    fig.update_layout(
        title=dict(
            text=f"Rolling Instability ({window_seconds}s window)",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Instability (%)",
            gridcolor="#E2E8F0",
            range=[0, max(rolling_instability) * 1.2] if rolling_instability else [0, 1]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=60, r=100, t=60, b=60),
        showlegend=False,
    )

    return fig


def create_frequency_spectrum(result: TemporalStabilityResult) -> go.Figure:
    """Create frequency spectrum analysis chart"""

    irradiances = np.array([m.irradiance for m in result.measurements])

    # Remove mean and compute FFT
    irradiances_centered = irradiances - np.mean(irradiances)
    n = len(irradiances_centered)
    fft_result = np.fft.rfft(irradiances_centered)
    frequencies = np.fft.rfftfreq(n, d=1/result.sampling_rate_hz)

    # Power spectrum (magnitude squared)
    power = np.abs(fft_result) ** 2

    # Normalize
    power = power / np.max(power) if np.max(power) > 0 else power

    # Limit to meaningful frequency range
    max_freq_idx = min(len(frequencies), int(len(frequencies) * 0.5))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frequencies[1:max_freq_idx],  # Skip DC component
        y=power[1:max_freq_idx],
        mode='lines',
        name='Power Spectrum',
        line=dict(color='#8B5CF6', width=1),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.1)',
        hovertemplate='Frequency: %{x:.2f} Hz<br>Power: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Frequency Spectrum Analysis",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Frequency (Hz)",
            gridcolor="#E2E8F0",
            type="log"
        ),
        yaxis=dict(
            title="Normalized Power",
            gridcolor="#E2E8F0",
            type="log"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
    )

    return fig


def create_histogram(result: TemporalStabilityResult) -> go.Figure:
    """Create histogram of irradiance values"""

    irradiances = [m.irradiance for m in result.measurements]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=irradiances,
        nbinsx=50,
        marker=dict(
            color='#6366F1',
            line=dict(color='white', width=0.5)
        ),
        hovertemplate='Irradiance: %{x:.1f} W/m²<br>Count: %{y}<extra></extra>'
    ))

    # Add mean and std lines
    fig.add_vline(
        x=result.mean_irradiance,
        line_dash="solid",
        line_color="#1E3A5F",
        line_width=2,
    )

    std = np.std(irradiances)
    fig.add_vline(x=result.mean_irradiance - std, line_dash="dash", line_color="#94A3B8", line_width=1)
    fig.add_vline(x=result.mean_irradiance + std, line_dash="dash", line_color="#94A3B8", line_width=1)

    fig.update_layout(
        title=dict(
            text="Irradiance Distribution",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Irradiance (W/m²)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Count",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=60, r=40, t=60, b=60),
        bargap=0.02
    )

    return fig


def main():
    """Temporal stability analysis page"""

    # Header
    st.markdown('<h1 class="main-title">Temporal Stability Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">STI (Short Term Instability) & LTI (Long Term Instability) per IEC 60904-9 Ed.3</p>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("## Equipment Selection")
        selected_simulator, sim_metadata = render_simulator_selector(
            key_prefix="temporal",
            show_specs=True,
            show_custom_option=True,
            compact=False
        )

        # Save selection to database if a simulator is selected
        if selected_simulator and DB_AVAILABLE:
            insert_simulator_selection(
                simulator_id=sim_metadata.get("simulator_id", ""),
                manufacturer=selected_simulator.manufacturer_name,
                model=selected_simulator.model_name,
                lamp_type=selected_simulator.lamp_type.value,
                classification=selected_simulator.typical_classification,
                test_plane_size=selected_simulator.test_plane_size,
                irradiance_min=selected_simulator.irradiance_range.min_wm2,
                irradiance_max=selected_simulator.irradiance_range.max_wm2,
                illumination_mode=selected_simulator.illumination_mode.value,
                is_custom=sim_metadata.get("is_custom", False),
                notes=selected_simulator.notes
            )

        st.markdown("---")
        st.markdown("### Analysis Settings")

        duration = st.slider(
            "Measurement Duration (s)",
            min_value=10,
            max_value=120,
            value=60,
            step=10
        )

        sampling_rate = st.select_slider(
            "Sampling Rate (Hz)",
            options=[10, 50, 100, 500, 1000],
            value=100
        )

        quality_sim = st.select_slider(
            "Simulation Quality",
            options=["C", "B", "A", "A+"],
            value="A"
        )

        st.markdown("---")
        st.markdown("### Measurement Info")
        st.markdown(f"**Duration:** {duration} seconds")
        st.markdown(f"**Sample Rate:** {sampling_rate} Hz")
        st.markdown(f"**Total Samples:** {duration * sampling_rate:,}")

    # Generate sample data
    result = generate_sample_temporal_data(duration, sampling_rate, quality_sim)

    # Classification results - Overall and STI/LTI breakdown
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="color: #64748B; font-size: 0.875rem; margin-bottom: 0.5rem;">
                OVERALL TEMPORAL GRADE
            </div>
            <div class="grade-badge-large {get_grade_class(result.overall_grade)}">
                {result.overall_grade.value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="dual-grade-card">
            <div class="grade-label">Short Term Instability (STI)</div>
            <div class="grade-badge-medium {get_grade_class(result.sti_grade)}">
                {result.sti_grade.value}
            </div>
            <div style="margin-top: 0.75rem; font-size: 1.25rem; font-weight: 600; color: #1E3A5F;">
                {result.sti_percent:.3f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="dual-grade-card">
            <div class="grade-label">Long Term Instability (LTI)</div>
            <div class="grade-badge-medium {get_grade_class(result.lti_grade)}">
                {result.lti_grade.value}
            </div>
            <div style="margin-top: 0.75rem; font-size: 1.25rem; font-weight: 600; color: #1E3A5F;">
                {result.lti_percent:.3f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Additional metrics
    st.markdown("")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.mean_irradiance:.1f}</div>
            <div class="metric-label">Mean Irradiance (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.min_irradiance:.1f}</div>
            <div class="metric-label">Minimum (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.max_irradiance:.1f}</div>
            <div class="metric-label">Maximum (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col4:
        range_val = result.max_irradiance - result.min_irradiance
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{range_val:.2f}</div>
            <div class="metric-label">Range (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    # Formula box
    st.markdown("""
    <div class="formula-box">
        <strong>Temporal Instability Formula (IEC 60904-9):</strong><br><br>
        Instability (%) = ((E<sub>max</sub> - E<sub>min</sub>) / (E<sub>max</sub> + E<sub>min</sub>)) × 100<br><br>
        <span style="font-size: 0.9rem; color: #64748B;">
        STI: Measured during a single I-V sweep (typically < 1s) | LTI: Measured over longer periods (minutes to hours)
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>IEC 60904-9 Ed.3 Temporal Stability Requirements:</strong><br>
        Temporal stability is measured in two categories: Short Term Instability (STI) during I-V curve measurement,
        and Long Term Instability (LTI) over extended operation. Both must meet the classification thresholds.
        The overall temporal grade is determined by the <strong>worse</strong> of STI and LTI grades.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Time series chart
    st.markdown("### Irradiance Time Series")
    fig_time = create_time_series_chart(result)
    st.plotly_chart(fig_time, use_container_width=True)

    # Two column charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        window_size = st.slider("Rolling Window Size (s)", 0.5, 5.0, 1.0, 0.5, key="window")
        fig_rolling = create_rolling_instability_chart(result, window_size)
        st.plotly_chart(fig_rolling, use_container_width=True)

    with chart_col2:
        fig_freq = create_frequency_spectrum(result)
        st.plotly_chart(fig_freq, use_container_width=True)

    # Distribution histogram
    st.markdown("### Irradiance Distribution")
    fig_hist = create_histogram(result)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistics
    irradiances = [m.irradiance for m in result.measurements]

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Standard Deviation", f"{np.std(irradiances):.3f} W/m²")

    with stat_col2:
        st.metric("Coefficient of Variation", f"{(np.std(irradiances)/np.mean(irradiances))*100:.4f}%")

    with stat_col3:
        st.metric("Peak-to-Peak", f"{result.max_irradiance - result.min_irradiance:.3f} W/m²")

    with stat_col4:
        st.metric("Samples Analyzed", f"{len(result.measurements):,}")

    st.divider()

    # Thresholds reference
    st.markdown("### Classification Thresholds")

    thresh_col1, thresh_col2 = st.columns(2)

    with thresh_col1:
        st.markdown("""
        **Short Term Instability (STI)**

        | Grade | Maximum STI |
        |-------|------------|
        | **A+** | <= 0.5% |
        | **A** | <= 2% |
        | **B** | <= 5% |
        | **C** | <= 10% |
        """)

    with thresh_col2:
        st.markdown("""
        **Long Term Instability (LTI)**

        | Grade | Maximum LTI |
        |-------|------------|
        | **A+** | <= 1% |
        | **A** | <= 2% |
        | **B** | <= 5% |
        | **C** | <= 10% |
        """)

    # Summary
    st.markdown(f"""
    **Current Measurement Summary:**
    - Duration: {result.measurement_duration_s:.1f} seconds
    - Sampling Rate: {result.sampling_rate_hz:.0f} Hz
    - STI: **{result.sti_percent:.3f}%** (Grade: {result.sti_grade.value})
    - LTI: **{result.lti_percent:.3f}%** (Grade: {result.lti_grade.value})
    - Overall Classification: **{result.overall_grade.value}**
    """)


if __name__ == "__main__":
    main()
