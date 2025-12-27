"""
Sun Simulator Classification System
Uniformity Analysis Page - Spatial Non-Uniformity Classification

This page provides spatial uniformity analysis with:
- Configurable measurement grid
- Heatmap visualization
- Reference cell position marker
- IEC 60904-9 classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_CONFIG, THEME, BADGE_COLORS, CLASSIFICATION,
    get_classification, MEASUREMENT_CONFIG
)
from utils.calculations import (
    UniformityCalculator, generate_sample_uniformity_data
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Uniformity - " + APP_CONFIG['title'],
    page_icon="🗺️",
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

.classification-large {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 80px;
    height: 80px;
    border-radius: 16px;
    font-size: 2rem;
    font-weight: 700;
    color: white;
}

.badge-aplus { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
.badge-a { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
.badge-b { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
.badge-c { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }

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

.grid-config {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #334155;
    color: #e2e8f0;
}

.stat-item:last-child {
    border-bottom: none;
}

.stat-label {
    color: #94a3b8;
}

.stat-value {
    font-weight: 600;
    color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_badge_class(classification: str) -> str:
    """Get CSS class for classification badge."""
    return {
        'A+': 'badge-aplus',
        'A': 'badge-a',
        'B': 'badge-b',
        'C': 'badge-c'
    }.get(classification, 'badge-c')


def create_uniformity_heatmap(data: np.ndarray, deviation_map: np.ndarray,
                               ref_row: int = None, ref_col: int = None,
                               width_mm: float = 500, height_mm: float = 500) -> go.Figure:
    """Create an interactive uniformity heatmap."""

    rows, cols = data.shape

    # Create position arrays
    x_pos = np.linspace(0, width_mm, cols)
    y_pos = np.linspace(0, height_mm, rows)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Irradiance Map (W/m²)', 'Deviation Map (%)'),
        horizontal_spacing=0.1
    )

    # Irradiance heatmap
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_pos,
            y=y_pos,
            colorscale='Viridis',
            colorbar=dict(
                title='W/m²',
                x=0.45,
                len=0.8
            ),
            hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
        ),
        row=1, col=1
    )

    # Deviation heatmap
    fig.add_trace(
        go.Heatmap(
            z=deviation_map,
            x=x_pos,
            y=y_pos,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(
                title='%',
                x=1.0,
                len=0.8
            ),
            hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Deviation: %{z:+.2f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # Add reference cell marker
    if ref_row is not None and ref_col is not None:
        ref_x = x_pos[ref_col]
        ref_y = y_pos[ref_row]

        for col_idx in [1, 2]:
            fig.add_trace(
                go.Scatter(
                    x=[ref_x],
                    y=[ref_y],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='rgba(255, 255, 255, 0.8)',
                        symbol='x',
                        line=dict(width=3, color='#ef4444')
                    ),
                    name='Reference Cell',
                    showlegend=(col_idx == 1),
                    hovertemplate='Reference Cell Position<extra></extra>'
                ),
                row=1, col=col_idx
            )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family='Inter', color='#e2e8f0'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(title_text="X Position (mm)", showgrid=False)
    fig.update_yaxes(title_text="Y Position (mm)", showgrid=False)

    return fig


def create_histogram(data: np.ndarray) -> go.Figure:
    """Create histogram of irradiance values."""
    flat = data.flatten()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=flat,
        nbinsx=20,
        marker_color='#3b82f6',
        opacity=0.8,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))

    # Add mean line
    mean_val = np.mean(flat)
    fig.add_vline(
        x=mean_val,
        line_dash='dash',
        line_color='#10b981',
        annotation_text=f'Mean: {mean_val:.1f}',
        annotation_position='top'
    )

    # Add std lines
    std_val = np.std(flat)
    fig.add_vline(x=mean_val - std_val, line_dash='dot', line_color='#f59e0b')
    fig.add_vline(x=mean_val + std_val, line_dash='dot', line_color='#f59e0b')

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=60, r=40, t=40, b=40),
        xaxis_title='Irradiance (W/m²)',
        yaxis_title='Count',
        font=dict(family='Inter', color='#e2e8f0'),
        showlegend=False
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

    return fig


def create_3d_surface(data: np.ndarray, width_mm: float, height_mm: float) -> go.Figure:
    """Create 3D surface plot of irradiance."""
    rows, cols = data.shape
    x = np.linspace(0, width_mm, cols)
    y = np.linspace(0, height_mm, rows)

    fig = go.Figure(data=[go.Surface(
        z=data,
        x=x,
        y=y,
        colorscale='Viridis',
        hovertemplate='X: %{x:.0f}mm<br>Y: %{y:.0f}mm<br>Irradiance: %{z:.1f} W/m²<extra></extra>'
    )])

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Irradiance (W/m²)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(71,85,105,0.3)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(71,85,105,0.3)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(71,85,105,0.3)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        font=dict(family='Inter', color='#e2e8f0')
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🗺️ Spatial Uniformity Analysis</div>
        <div class="page-subtitle">
            Measure and classify spatial irradiance uniformity per IEC 60904-9
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'uniformity_data' not in st.session_state:
        st.session_state.uniformity_data = None
    if 'uniformity_result' not in st.session_state:
        st.session_state.uniformity_result = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### Grid Configuration")

        grid_rows = st.slider(
            "Grid Rows",
            min_value=MEASUREMENT_CONFIG['min_grid_size'],
            max_value=MEASUREMENT_CONFIG['max_grid_size'],
            value=MEASUREMENT_CONFIG['default_grid_size'],
            help="Number of measurement rows"
        )

        grid_cols = st.slider(
            "Grid Columns",
            min_value=MEASUREMENT_CONFIG['min_grid_size'],
            max_value=MEASUREMENT_CONFIG['max_grid_size'],
            value=MEASUREMENT_CONFIG['default_grid_size'],
            help="Number of measurement columns"
        )

        st.markdown("### Test Area")
        test_width = st.number_input("Width (mm)", value=500, min_value=100, max_value=2000)
        test_height = st.number_input("Height (mm)", value=500, min_value=100, max_value=2000)

        st.markdown("### Reference Cell")
        ref_row = st.number_input("Row", value=grid_rows // 2, min_value=0, max_value=grid_rows - 1)
        ref_col = st.number_input("Column", value=grid_cols // 2, min_value=0, max_value=grid_cols - 1)

        st.markdown("### Sample Data")
        target_uniformity = st.slider(
            "Target Non-Uniformity (%)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            help="Target non-uniformity for sample data generation"
        )

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Data Input</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Generate Sample"])

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload Uniformity Data (CSV)",
                type=['csv', 'xlsx'],
                help="CSV/Excel file with irradiance grid data"
            )

            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, header=None)
                    else:
                        df = pd.read_excel(uploaded_file, header=None)

                    st.session_state.uniformity_data = df.values
                    st.success(f"Loaded {df.shape[0]}x{df.shape[1]} grid")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with tab2:
            if st.button("Generate Sample Data", type="primary"):
                data = generate_sample_uniformity_data(
                    rows=grid_rows,
                    cols=grid_cols,
                    non_uniformity=target_uniformity
                )
                st.session_state.uniformity_data = data
                st.success(f"Generated {grid_rows}x{grid_cols} sample grid")

    with col2:
        st.markdown('<div class="section-title">Grid Preview</div>', unsafe_allow_html=True)

        # Show grid configuration
        st.markdown(f"""
        <div class="grid-config">
            <div class="stat-item">
                <span class="stat-label">Grid Size</span>
                <span class="stat-value">{grid_rows} × {grid_cols}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Test Area</span>
                <span class="stat-value">{test_width} × {test_height} mm</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Measurement Points</span>
                <span class="stat-value">{grid_rows * grid_cols}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Reference Cell</span>
                <span class="stat-value">({ref_row}, {ref_col})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis Results
    if st.session_state.uniformity_data is not None:
        data = st.session_state.uniformity_data

        # Calculate uniformity
        result = UniformityCalculator.calculate_uniformity(data)
        st.session_state.uniformity_result = result

        # Results Overview
        st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

        with col1:
            badge_class = get_badge_class(result.classification)
            st.markdown(f"""
            <div class="result-card" style="text-align: center;">
                <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                    Classification
                </div>
                <div class="classification-large {badge_class}">
                    {result.classification}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            nu_color = BADGE_COLORS[result.classification]
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: {nu_color}40; background: {nu_color}10;">
                    <div class="metric-value" style="color: {nu_color};">
                        {result.non_uniformity:.2f}%
                    </div>
                    <div class="metric-label">Non-Uniformity</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box">
                    <div class="metric-value">{result.mean_irradiance:.1f}</div>
                    <div class="metric-label">Mean (W/m²)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: #10b98140; background: #10b98110;">
                    <div class="metric-value" style="color: #10b981;">
                        {result.max_irradiance:.1f}
                    </div>
                    <div class="metric-label">Max (W/m²)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="result-card">
                <div class="metric-box" style="border-color: #ef444440; background: #ef444410;">
                    <div class="metric-value" style="color: #ef4444;">
                        {result.min_irradiance:.1f}
                    </div>
                    <div class="metric-label">Min (W/m²)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualization Tabs
        st.markdown('<div class="section-title">Uniformity Visualization</div>', unsafe_allow_html=True)

        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🗺️ Heatmap", "📊 3D Surface", "📈 Distribution"])

        with viz_tab1:
            fig = create_uniformity_heatmap(
                data, result.deviation_map,
                ref_row=ref_row, ref_col=ref_col,
                width_mm=test_width, height_mm=test_height
            )
            st.plotly_chart(fig, use_container_width=True)

        with viz_tab2:
            fig = create_3d_surface(data, test_width, test_height)
            st.plotly_chart(fig, use_container_width=True)

        with viz_tab3:
            col1, col2 = st.columns(2)

            with col1:
                fig = create_histogram(data)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Statistics summary
                st.markdown("""
                <div class="result-card">
                    <h4 style="color: #f8fafc; margin-bottom: 1rem;">Statistical Summary</h4>
                """, unsafe_allow_html=True)

                stats = [
                    ("Mean Irradiance", f"{result.mean_irradiance:.2f} W/m²"),
                    ("Standard Deviation", f"{result.std_deviation:.2f} W/m²"),
                    ("Maximum", f"{result.max_irradiance:.2f} W/m²"),
                    ("Minimum", f"{result.min_irradiance:.2f} W/m²"),
                    ("Range", f"{result.max_irradiance - result.min_irradiance:.2f} W/m²"),
                    ("CV (%)", f"{(result.std_deviation / result.mean_irradiance * 100):.2f}%"),
                ]

                for label, value in stats:
                    st.markdown(f"""
                    <div class="stat-item">
                        <span class="stat-label">{label}</span>
                        <span class="stat-value">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Data Table
        with st.expander("📋 View Raw Data Grid"):
            df_display = pd.DataFrame(
                data,
                index=[f"Row {i+1}" for i in range(data.shape[0])],
                columns=[f"Col {j+1}" for j in range(data.shape[1])]
            )
            st.dataframe(df_display.style.format("{:.1f}").background_gradient(cmap='viridis'),
                        use_container_width=True)

        # Classification Limits
        st.markdown('<div class="section-title">IEC 60904-9 Uniformity Limits</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Classification Limits</h4>
                <table style="width: 100%; color: #e2e8f0;">
                    <tr>
                        <td><span style="color: #10b981;">■</span> A+</td>
                        <td>≤ 1.0%</td>
                        <td>{'✓' if result.non_uniformity <= 1.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #3b82f6;">■</span> A</td>
                        <td>≤ 2.0%</td>
                        <td>{'✓' if 1.0 < result.non_uniformity <= 2.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #f59e0b;">■</span> B</td>
                        <td>≤ 5.0%</td>
                        <td>{'✓' if 2.0 < result.non_uniformity <= 5.0 else ''}</td>
                    </tr>
                    <tr>
                        <td><span style="color: #ef4444;">■</span> C</td>
                        <td>> 5.0%</td>
                        <td>{'✓' if result.non_uniformity > 5.0 else ''}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="result-card">
                <h4 style="color: #f8fafc; margin-bottom: 1rem;">Calculation Method</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Non-uniformity is calculated per IEC 60904-9 as:<br><br>
                    <code style="color: #3b82f6; background: rgba(59,130,246,0.1);
                                 padding: 0.5rem; border-radius: 4px; display: block;">
                    NU = (E_max - E_min) / (E_max + E_min) × 100%
                    </code><br>
                    where E_max and E_min are the maximum and minimum
                    irradiance values within the test area.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # No data loaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(30, 41, 59, 0.5);
                    border: 2px dashed #475569; border-radius: 12px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🗺️</div>
            <div style="color: #f8fafc; font-size: 1.2rem; margin-bottom: 0.5rem;">
                No Uniformity Data Loaded
            </div>
            <div style="color: #94a3b8;">
                Upload a CSV/Excel file or generate sample data to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
