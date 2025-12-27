"""
Uniformity Analysis Page
IEC 60904-9 Ed.3 Solar Simulator Classification

Non-Uniformity analysis with irradiance uniformity measurements
across the test plane using a grid-based measurement approach.
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
    UniformityMeasurement,
    UniformityResult,
    get_grade_color,
)

# Page configuration
st.set_page_config(
    page_title="Uniformity Analysis | SunSim",
    page_icon=":material/grid_on:",
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

    .formula-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def generate_sample_uniformity_data(
    grid_size: int = 11,
    plane_size: float = 200.0,
    quality: str = "A+"
) -> UniformityResult:
    """Generate sample uniformity measurement data"""

    # Quality settings - controls the irradiance variation
    quality_variation = {
        "A+": 0.005,   # 0.5% variation -> ~1% non-uniformity
        "A": 0.010,    # 1% variation -> ~2% non-uniformity
        "B": 0.025,    # 2.5% variation -> ~5% non-uniformity
        "C": 0.050,    # 5% variation -> ~10% non-uniformity
    }

    variation = quality_variation.get(quality, 0.005)
    np.random.seed(42)

    result = UniformityResult(
        test_plane_size_mm=(plane_size, plane_size),
        grid_points=(grid_size, grid_size)
    )

    # Target irradiance (1000 W/m² = 1 sun)
    target_irradiance = 1000.0

    # Generate grid positions and measurements
    half_size = plane_size / 2
    step = plane_size / (grid_size - 1)

    for i in range(grid_size):
        for j in range(grid_size):
            x = -half_size + i * step
            y = -half_size + j * step

            # Create realistic variation pattern (slightly lower at edges)
            distance_from_center = np.sqrt(x**2 + y**2)
            max_distance = np.sqrt(2) * half_size
            edge_effect = 1 - 0.005 * (distance_from_center / max_distance)

            # Add random noise
            noise = np.random.uniform(-variation, variation)

            irradiance = target_irradiance * edge_effect * (1 + noise)

            measurement = UniformityMeasurement(
                x_position=x,
                y_position=y,
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


def create_heatmap(result: UniformityResult) -> go.Figure:
    """Create irradiance heatmap visualization"""

    grid_size = result.grid_points[0]
    irradiances = [m.irradiance for m in result.measurements]

    # Reshape to 2D grid
    z = np.array(irradiances).reshape(grid_size, grid_size)

    # Create position arrays
    half_size = result.test_plane_size_mm[0] / 2
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='RdYlGn',
        reversescale=False,
        zmin=result.min_irradiance - 2,
        zmax=result.max_irradiance + 2,
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Irradiance Distribution Heatmap",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="X Position (mm)",
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Y Position (mm)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def create_contour_plot(result: UniformityResult) -> go.Figure:
    """Create irradiance contour plot"""

    grid_size = result.grid_points[0]
    irradiances = [m.irradiance for m in result.measurements]

    # Reshape to 2D grid
    z = np.array(irradiances).reshape(grid_size, grid_size)

    # Create position arrays
    half_size = result.test_plane_size_mm[0] / 2
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)

    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale='RdYlGn',
        reversescale=False,
        contours=dict(
            start=result.min_irradiance,
            end=result.max_irradiance,
            size=(result.max_irradiance - result.min_irradiance) / 10,
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="Irradiance Contour Map",
            font=dict(size=16, color="#1E3A5F")
        ),
        xaxis=dict(
            title="X Position (mm)",
            gridcolor="#E2E8F0",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Y Position (mm)",
            gridcolor="#E2E8F0",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def create_3d_surface(result: UniformityResult) -> go.Figure:
    """Create 3D surface plot of irradiance distribution"""

    grid_size = result.grid_points[0]
    irradiances = [m.irradiance for m in result.measurements]

    # Reshape to 2D grid
    z = np.array(irradiances).reshape(grid_size, grid_size)

    # Create position arrays
    half_size = result.test_plane_size_mm[0] / 2
    x = np.linspace(-half_size, half_size, grid_size)
    y = np.linspace(-half_size, half_size, grid_size)

    fig = go.Figure(data=[go.Surface(
        z=z,
        x=x,
        y=y,
        colorscale='RdYlGn',
        reversescale=False,
        colorbar=dict(
            title=dict(text='W/m²', side='right'),
            tickformat='.1f'
        ),
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    )])

    fig.update_layout(
        title=dict(
            text="3D Irradiance Surface",
            font=dict(size=16, color="#1E3A5F")
        ),
        scene=dict(
            xaxis=dict(title="X (mm)"),
            yaxis=dict(title="Y (mm)"),
            zaxis=dict(title="Irradiance (W/m²)"),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor="white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


def create_histogram(result: UniformityResult) -> go.Figure:
    """Create histogram of irradiance values"""

    irradiances = [m.irradiance for m in result.measurements]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=irradiances,
        nbinsx=20,
        marker=dict(
            color='#6366F1',
            line=dict(color='white', width=1)
        ),
        hovertemplate='Irradiance: %{x:.1f} W/m²<br>Count: %{y}<extra></extra>'
    ))

    # Add mean line
    fig.add_vline(
        x=result.mean_irradiance,
        line_dash="solid",
        line_color="#1E3A5F",
        line_width=2,
        annotation_text=f"Mean: {result.mean_irradiance:.1f}",
        annotation_position="top"
    )

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
            title="Number of Points",
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
    """Uniformity analysis page"""

    # Header
    st.markdown('<h1 class="main-title">Non-Uniformity Analysis</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Spatial Irradiance Uniformity Classification per IEC 60904-9 Ed.3</p>',
        unsafe_allow_html=True
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Analysis Settings")

        grid_size = st.select_slider(
            "Measurement Grid",
            options=[5, 7, 9, 11, 15, 21],
            value=11,
            format_func=lambda x: f"{x}x{x} ({x*x} points)"
        )

        plane_size = st.slider(
            "Test Plane Size (mm)",
            min_value=100,
            max_value=400,
            value=200,
            step=50
        )

        quality_sim = st.select_slider(
            "Simulation Quality",
            options=["C", "B", "A", "A+"],
            value="A+"
        )

        st.markdown("---")
        st.markdown("### Measurement Settings")
        st.markdown(f"**Grid Points:** {grid_size}x{grid_size}")
        st.markdown(f"**Total Points:** {grid_size * grid_size}")
        st.markdown(f"**Test Area:** {plane_size}x{plane_size} mm")

    # Generate sample data
    result = generate_sample_uniformity_data(grid_size, plane_size, quality_sim)

    # Classification result metrics
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <div style="color: #64748B; font-size: 0.875rem; margin-bottom: 0.5rem;">
                NON-UNIFORMITY GRADE
            </div>
            <div class="grade-badge-large {get_grade_class(result.grade)}">
                {result.grade.value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.non_uniformity_percent:.2f}%</div>
            <div class="metric-label">Non-Uniformity</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.mean_irradiance:.1f}</div>
            <div class="metric-label">Mean (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.min_irradiance:.1f}</div>
            <div class="metric-label">Min (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result.max_irradiance:.1f}</div>
            <div class="metric-label">Max (W/m²)</div>
        </div>
        """, unsafe_allow_html=True)

    # Formula box
    st.markdown("""
    <div class="formula-box">
        <strong>Non-Uniformity Formula (IEC 60904-9):</strong><br><br>
        Non-Uniformity (%) = ((E<sub>max</sub> - E<sub>min</sub>) / (E<sub>max</sub> + E<sub>min</sub>)) × 100
    </div>
    """, unsafe_allow_html=True)

    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>IEC 60904-9 Ed.3 Non-Uniformity Requirements:</strong><br>
        The test plane is divided into a minimum 11×11 grid (121 points). Irradiance is measured
        at each point and the non-uniformity is calculated from the maximum and minimum values.
        The classification grade is determined by the calculated non-uniformity percentage.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["Heatmap", "Contour Map", "3D Surface"])

    with tab1:
        fig_heatmap = create_heatmap(result)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab2:
        fig_contour = create_contour_plot(result)
        st.plotly_chart(fig_contour, use_container_width=True)

    with tab3:
        fig_3d = create_3d_surface(result)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.divider()

    # Distribution histogram
    st.markdown("### Irradiance Distribution Analysis")
    fig_hist = create_histogram(result)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Statistics
    irradiances = [m.irradiance for m in result.measurements]

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Standard Deviation", f"{np.std(irradiances):.2f} W/m²")

    with stat_col2:
        st.metric("Coefficient of Variation", f"{(np.std(irradiances)/np.mean(irradiances))*100:.3f}%")

    with stat_col3:
        st.metric("Range", f"{result.max_irradiance - result.min_irradiance:.2f} W/m²")

    with stat_col4:
        st.metric("Median", f"{np.median(irradiances):.1f} W/m²")

    st.divider()

    # Thresholds reference
    st.markdown("### Classification Thresholds")

    thresh_col1, thresh_col2 = st.columns(2)

    with thresh_col1:
        st.markdown("""
        | Grade | Maximum Non-Uniformity |
        |-------|----------------------|
        | **A+** | <= 1% |
        | **A** | <= 2% |
        | **B** | <= 5% |
        | **C** | <= 10% |
        """)

    with thresh_col2:
        st.markdown(f"""
        **Current Measurement Summary:**
        - Test Plane: {result.test_plane_size_mm[0]:.0f} × {result.test_plane_size_mm[1]:.0f} mm
        - Grid: {result.grid_points[0]} × {result.grid_points[1]} ({len(result.measurements)} points)
        - Non-Uniformity: **{result.non_uniformity_percent:.2f}%**
        - Classification: **{result.grade.value}**
        """)

    # Data table expander
    with st.expander("View Raw Measurement Data"):
        df_data = [{
            "X (mm)": f"{m.x_position:.1f}",
            "Y (mm)": f"{m.y_position:.1f}",
            "Irradiance (W/m²)": f"{m.irradiance:.2f}",
            "Deviation from Mean (%)": f"{((m.irradiance - result.mean_irradiance) / result.mean_irradiance * 100):.2f}"
        } for m in result.measurements]

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)


if __name__ == "__main__":
    main()
