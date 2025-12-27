"""
Uniformity Analysis Page

Analyzes spatial uniformity of irradiance across the test plane.
This page does NOT require database connectivity to function.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import calculations (no database dependency)
from utils.calculations import calculate_uniformity_class, IEC_60904_9_LIMITS

st.set_page_config(
    page_title="Uniformity | SunSim Classifier",
    page_icon="black_square_button",
    layout="wide"
)

st.title("Non-Uniformity of Irradiance Analysis")
st.markdown("#### IEC 60904-9 Ed.3 Uniformity Classification")

st.markdown("""
This analysis evaluates the spatial uniformity of irradiance across your
sun simulator's test plane according to IEC 60904-9 Ed.3.

**Classification Limits:**
- Class A: Non-uniformity within +/-2%
- Class B: Non-uniformity within +/-5%
- Class C: Non-uniformity within +/-10%
""")

st.markdown("---")

# Input configuration
col1, col2 = st.columns(2)

with col1:
    grid_rows = st.number_input("Grid Rows", min_value=3, max_value=20, value=5)
    grid_cols = st.number_input("Grid Columns", min_value=3, max_value=20, value=5)

with col2:
    nominal_irradiance = st.number_input(
        "Nominal Irradiance (W/m²)",
        min_value=100.0,
        max_value=2000.0,
        value=1000.0,
        step=10.0
    )

# Input method
input_method = st.radio(
    "Select Input Method",
    ["Generate Demo Data", "Manual Entry", "Upload CSV"],
    horizontal=True
)

irradiance_map = None

if input_method == "Generate Demo Data":
    st.markdown("### Demo Data Configuration")

    col1, col2 = st.columns(2)
    with col1:
        uniformity_quality = st.selectbox(
            "Simulated Quality",
            ["Class A (~1% variation)", "Class B (~3% variation)", "Class C (~7% variation)"]
        )

    # Generate appropriate demo data
    np.random.seed(42)

    if "Class A" in uniformity_quality:
        variation = 0.01
    elif "Class B" in uniformity_quality:
        variation = 0.03
    else:
        variation = 0.07

    # Generate realistic irradiance map with some spatial pattern
    base_map = np.ones((grid_rows, grid_cols)) * nominal_irradiance

    # Add center-weighted distribution (common in sun simulators)
    for i in range(grid_rows):
        for j in range(grid_cols):
            # Distance from center
            center_i, center_j = grid_rows / 2, grid_cols / 2
            dist = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
            max_dist = np.sqrt(center_i ** 2 + center_j ** 2)

            # Slight drop-off from center
            spatial_factor = 1 - (dist / max_dist) * variation * 0.5

            # Random variation
            random_factor = 1 + np.random.uniform(-variation, variation)

            base_map[i, j] = nominal_irradiance * spatial_factor * random_factor

    irradiance_map = base_map

    st.success(f"Generated {grid_rows}x{grid_cols} irradiance map")

elif input_method == "Manual Entry":
    st.markdown("### Enter Irradiance Values")
    st.caption(f"Enter values for each position in the {grid_rows}x{grid_cols} grid")

    irradiance_map = np.zeros((grid_rows, grid_cols))

    for i in range(grid_rows):
        cols = st.columns(grid_cols)
        for j in range(grid_cols):
            with cols[j]:
                irradiance_map[i, j] = st.number_input(
                    f"({i},{j})",
                    min_value=0.0,
                    max_value=2000.0,
                    value=nominal_irradiance,
                    step=1.0,
                    key=f"cell_{i}_{j}",
                    label_visibility="collapsed"
                )

elif input_method == "Upload CSV":
    st.markdown("### Upload Irradiance Map")
    st.caption("CSV should contain a grid of irradiance values (no headers)")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            irradiance_map = df.values
            st.success(f"Loaded {irradiance_map.shape[0]}x{irradiance_map.shape[1]} irradiance map")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

st.markdown("---")

# Analysis
if irradiance_map is not None:
    st.markdown("### Analysis Results")

    # Perform calculation
    result = calculate_uniformity_class(irradiance_map)

    # Display classification
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        class_color = {
            "A": "green",
            "B": "orange",
            "C": "red"
        }
        st.markdown(
            f"<h1 style='text-align: center; color: {class_color[result.classification]};'>"
            f"Class {result.classification}</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='text-align: center;'>Non-uniformity: {result.uniformity_percentage:.2f}%</p>",
            unsafe_allow_html=True
        )

    # Statistics
    st.markdown("### Irradiance Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean Irradiance", f"{result.mean_irradiance:.1f} W/m²")

    with col2:
        st.metric("Min Irradiance", f"{result.min_irradiance:.1f} W/m²")

    with col3:
        st.metric("Max Irradiance", f"{result.max_irradiance:.1f} W/m²")

    with col4:
        st.metric("Max Deviation", f"{result.max_deviation:.1f} W/m²")

    # Heatmap visualization
    st.markdown("### Irradiance Map Visualization")

    # Create a DataFrame for display
    map_df = pd.DataFrame(
        irradiance_map,
        columns=[f"Col {j+1}" for j in range(irradiance_map.shape[1])],
        index=[f"Row {i+1}" for i in range(irradiance_map.shape[0])]
    )

    # Calculate deviation from mean for coloring
    deviation_map = ((irradiance_map - result.mean_irradiance) / result.mean_irradiance) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Absolute Irradiance (W/m²)**")
        st.dataframe(
            map_df.style.background_gradient(cmap="RdYlGn", vmin=result.min_irradiance, vmax=result.max_irradiance),
            use_container_width=True
        )

    with col2:
        st.markdown("**Deviation from Mean (%)**")
        deviation_df = pd.DataFrame(
            deviation_map,
            columns=[f"Col {j+1}" for j in range(deviation_map.shape[1])],
            index=[f"Row {i+1}" for i in range(deviation_map.shape[0])]
        )
        st.dataframe(
            deviation_df.style.background_gradient(cmap="RdYlGn_r", vmin=-10, vmax=10).format("{:.2f}"),
            use_container_width=True
        )

    # Save/Export options
    st.markdown("---")
    st.markdown("### Save Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save to Database", type="primary"):
            try:
                from database import is_database_configured

                if not is_database_configured():
                    st.warning("Database not configured. Results cannot be saved.")
                    st.info("To enable database storage, configure DATABASE_URL environment variable.")
                else:
                    st.info("Database save functionality available when connected.")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("Export as CSV"):
            csv = map_df.to_csv()
            st.download_button(
                label="Download Irradiance Map CSV",
                data=csv,
                file_name="uniformity_results.csv",
                mime="text/csv"
            )

# Sidebar reference
with st.sidebar:
    st.markdown("### Classification Limits")
    st.markdown("""
    | Class | Non-uniformity |
    |-------|----------------|
    | A | +/-2% |
    | B | +/-5% |
    | C | +/-10% |
    """)

    st.markdown("---")
    st.markdown("### Measurement Guidelines")
    st.markdown("""
    - Minimum 3x3 grid recommended
    - Larger grids provide better resolution
    - Measurements should cover full test area
    - Use calibrated reference cell
    """)
