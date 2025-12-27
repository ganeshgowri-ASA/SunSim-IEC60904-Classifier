"""
Spectral Match Analysis Page
IEC 60904-9 Ed.3 Solar Simulator Classification

SPD (Spectral Power Distribution) analysis showing spectral match
calculations and charts across wavelength intervals.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    WAVELENGTH_INTERVALS_ED3,
    WAVELENGTH_INTERVALS_ED2,
    SpectralMatchData,
    SpectralMatchResult,
    get_grade_color,
)

# Page configuration
st.set_page_config(
    page_title="Spectral Match | SunSim",
    page_icon=":material/ssid_chart:",
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

    .threshold-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .in-spec { background: #10B981; }
    .out-of-spec { background: #EF4444; }
</style>
""", unsafe_allow_html=True)


def generate_sample_spectral_data(intervals: list, quality: str = "A+") -> SpectralMatchResult:
    """Generate sample spectral match data for demonstration"""

    # Quality settings for simulation
    quality_noise = {
        "A+": 0.06,  # Very tight match
        "A": 0.12,   # Good match
        "B": 0.25,   # Moderate match
        "C": 0.40,   # Poor match
    }

    noise_level = quality_noise.get(quality, 0.06)
    np.random.seed(42)  # Reproducible results

    result = SpectralMatchResult(
        wavelength_range="300-1200nm" if len(intervals) > 10 else "400-1100nm"
    )

    for start, end, ref_fraction in intervals:
        # Simulate measured fraction with some noise
        noise = np.random.uniform(-noise_level, noise_level)
        measured_fraction = ref_fraction * (1 + noise)

        interval = SpectralMatchData(
            interval_start_nm=start,
            interval_end_nm=end,
            reference_fraction=ref_fraction,
            measured_fraction=measured_fraction,
        )
        result.intervals.append(interval)

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


def create_spectral_ratio_chart(result: SpectralMatchResult) -> go.Figure:
    """Create spectral match ratio chart with threshold bands"""

    wavelengths = [(i.interval_start_nm + i.interval_end_nm) / 2 for i in result.intervals]
    ratios = [i.ratio for i in result.intervals]

    fig = go.Figure()

    # Add threshold bands
    thresholds = [
        (0.4, 2.0, "C", "rgba(239, 68, 68, 0.1)"),
        (0.6, 1.4, "B", "rgba(245, 158, 11, 0.1)"),
        (0.75, 1.25, "A", "rgba(34, 197, 94, 0.1)"),
        (0.875, 1.125, "A+", "rgba(16, 185, 129, 0.15)"),
    ]

    for lower, upper, label, color in thresholds:
        fig.add_hrect(
            y0=lower, y1=upper,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    # Add threshold lines
    fig.add_hline(y=1.0, line_dash="solid", line_color="#1E3A5F", line_width=2)
    fig.add_hline(y=0.875, line_dash="dash", line_color="#10B981", line_width=1)
    fig.add_hline(y=1.125, line_dash="dash", line_color="#10B981", line_width=1)
    fig.add_hline(y=0.75, line_dash="dash", line_color="#22C55E", line_width=1)
    fig.add_hline(y=1.25, line_dash="dash", line_color="#22C55E", line_width=1)

    # Determine colors based on A+ threshold
    colors = []
    for r in ratios:
        if 0.875 <= r <= 1.125:
            colors.append("#10B981")  # In A+ spec
        elif 0.75 <= r <= 1.25:
            colors.append("#22C55E")  # In A spec
        elif 0.6 <= r <= 1.4:
            colors.append("#F59E0B")  # In B spec
        else:
            colors.append("#EF4444")  # Out of spec

    # Add ratio line and markers
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=ratios,
        mode='lines+markers',
        name='Measured/Reference Ratio',
        line=dict(color='#1E3A5F', width=2),
        marker=dict(size=8, color=colors, line=dict(width=1, color='white')),
        hovertemplate='<b>%{x:.0f} nm</b><br>Ratio: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Spectral Match Ratio by Wavelength",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Wavelength (nm)",
            gridcolor="#E2E8F0",
            range=[280, 1220]
        ),
        yaxis=dict(
            title="Ratio (Measured/Reference)",
            gridcolor="#E2E8F0",
            range=[0.3, 2.1]
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
        hovermode="x unified"
    )

    return fig


def create_spectral_distribution_chart(result: SpectralMatchResult) -> go.Figure:
    """Create spectral power distribution comparison chart"""

    wavelengths = [(i.interval_start_nm + i.interval_end_nm) / 2 for i in result.intervals]
    reference = [i.reference_fraction for i in result.intervals]
    measured = [i.measured_fraction for i in result.intervals]

    fig = go.Figure()

    # Reference spectrum (AM1.5G)
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=reference,
        mode='lines',
        name='AM1.5G Reference',
        line=dict(color='#6366F1', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.1)',
        hovertemplate='<b>Reference</b><br>%{x:.0f} nm: %{y:.2f}%<extra></extra>'
    ))

    # Measured spectrum
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=measured,
        mode='lines',
        name='Simulator Measured',
        line=dict(color='#F59E0B', width=2),
        hovertemplate='<b>Measured</b><br>%{x:.0f} nm: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Spectral Power Distribution Comparison",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Wavelength (nm)",
            gridcolor="#E2E8F0",
            range=[280, 1220]
        ),
        yaxis=dict(
            title="Relative Spectral Power (%)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified"
    )

    return fig


def create_ratio_histogram(result: SpectralMatchResult) -> go.Figure:
    """Create histogram of spectral match ratios"""

    ratios = [i.ratio for i in result.intervals]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ratios,
        nbinsx=20,
        marker=dict(
            color='#6366F1',
            line=dict(color='white', width=1)
        ),
        hovertemplate='Ratio: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))

    # Add threshold lines
    fig.add_vline(x=1.0, line_dash="solid", line_color="#1E3A5F", line_width=2)
    fig.add_vline(x=0.875, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ min", annotation_position="top")
    fig.add_vline(x=1.125, line_dash="dash", line_color="#10B981", line_width=1,
                  annotation_text="A+ max", annotation_position="top")

    fig.update_layout(
        title=dict(
            text="Distribution of Spectral Match Ratios",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="Ratio (Measured/Reference)",
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Number of Intervals",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=60, r=40, t=60, b=60),
        bargap=0.05
    )

    return fig


def main():
    """Spectral Match analysis page"""

    # Header
    st.markdown('<h1 class="main-title">Spectral Match Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">SPD (Spectral Power Distribution) Classification per IEC 60904-9 Ed.3</p>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Analysis Settings")

        wavelength_range = st.selectbox(
            "Wavelength Range",
            ["300-1200nm (Ed.3)", "400-1100nm (Ed.2 Legacy)"],
            index=0
        )

        quality_sim = st.select_slider(
            "Simulation Quality",
            options=["C", "B", "A", "A+"],
            value="A+"
        )

        st.markdown("---")
        st.markdown("### Reference Spectrum")
        st.markdown("AM1.5G Global (IEC 60904-3)")

    # Generate data based on settings
    if "Ed.3" in wavelength_range:
        intervals = WAVELENGTH_INTERVALS_ED3
    else:
        intervals = WAVELENGTH_INTERVALS_ED2

    result = generate_sample_spectral_data(intervals, quality_sim)

    # Classification result
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="color: #64748B; font-size: 0.875rem; margin-bottom: 0.5rem;">
                SPECTRAL MATCH GRADE
            </div>
            <div class="grade-badge-large {get_grade_class(result.grade)}">
                {result.grade.value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.min_ratio:.3f}</div>
            <div class="metric-label">Minimum Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.max_ratio:.3f}</div>
            <div class="metric-label">Maximum Ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        in_spec = sum(1 for i in result.intervals if 0.875 <= i.ratio <= 1.125)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{in_spec}/{len(result.intervals)}</div>
            <div class="metric-label">Intervals in A+ Spec</div>
        </div>
        """, unsafe_allow_html=True)

    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>IEC 60904-9 Ed.3 Spectral Match Requirements:</strong><br>
        The spectral match is evaluated by comparing the simulator's spectral irradiance to the AM1.5G
        reference spectrum across wavelength intervals. Each interval ratio (measured/reference) must
        fall within the classification thresholds. The overall grade is determined by the worst-performing interval.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Charts
    st.markdown("### Spectral Match Ratio Analysis")
    fig_ratio = create_spectral_ratio_chart(result)
    st.plotly_chart(fig_ratio, use_container_width=True)

    # Two column layout for additional charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_spd = create_spectral_distribution_chart(result)
        st.plotly_chart(fig_spd, use_container_width=True)

    with chart_col2:
        fig_hist = create_ratio_histogram(result)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Detailed data table
    st.markdown("### Interval Data")

    # Create dataframe for display
    df_data = []
    for interval in result.intervals:
        status = "Pass" if 0.875 <= interval.ratio <= 1.125 else "Review"
        df_data.append({
            "Wavelength Range (nm)": f"{interval.interval_start_nm:.0f} - {interval.interval_end_nm:.0f}",
            "Reference (%)": f"{interval.reference_fraction:.2f}",
            "Measured (%)": f"{interval.measured_fraction:.2f}",
            "Ratio": f"{interval.ratio:.3f}",
            "Status": status,
        })

    df = pd.DataFrame(df_data)

    # Show with highlighting
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Status": st.column_config.TextColumn(
                "Status",
                help="Pass = within A+ threshold (0.875-1.125)"
            )
        }
    )

    # Thresholds reference
    st.divider()
    st.markdown("### Classification Thresholds")

    thresh_col1, thresh_col2 = st.columns(2)

    with thresh_col1:
        st.markdown("""
        | Grade | Min Ratio | Max Ratio | Deviation |
        |-------|-----------|-----------|-----------|
        | **A+** | 0.875 | 1.125 | ±12.5% |
        | **A** | 0.75 | 1.25 | ±25% |
        | **B** | 0.6 | 1.4 | ±40%/+40% |
        | **C** | 0.4 | 2.0 | ±60%/+100% |
        """)

    with thresh_col2:
        st.markdown(f"""
        **Current Measurement Summary:**
        - Wavelength Range: {result.wavelength_range}
        - Total Intervals: {len(result.intervals)}
        - Measurement Date: {result.measurement_date.strftime('%Y-%m-%d %H:%M')}
        - Classification: **{result.grade.value}**
        """)


if __name__ == "__main__":
    main()
