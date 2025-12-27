"""
Spectral Match Analysis Page

Analyzes spectral distribution of sun simulator against IEC 60904-9 Ed.3 reference.
This page does NOT require database connectivity to function.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import calculations (no database dependency)
from utils.calculations import (
    calculate_spectral_match_class,
    IEC_60904_9_LIMITS,
)

st.set_page_config(
    page_title="Spectral Match | SunSim Classifier",
    page_icon="rainbow",
    layout="wide"
)

st.title("Spectral Match Analysis")
st.markdown("#### IEC 60904-9 Ed.3 Spectral Classification")

# Reference data
WAVELENGTH_BANDS = IEC_60904_9_LIMITS["wavelength_bands"]
REFERENCE_PERCENTAGES = [band["reference"] for band in WAVELENGTH_BANDS]

st.markdown("""
This analysis evaluates the spectral distribution of your sun simulator
against the AM1.5G reference spectrum according to IEC 60904-9 Ed.3.

**Classification Limits:**
- Class A: Deviation within +/-25%
- Class B: Deviation within +/-40%
- Class C: Deviation exceeds +/-40%
""")

st.markdown("---")

# Input method selection
input_method = st.radio(
    "Select Input Method",
    ["Manual Entry", "Upload CSV", "Demo Data"],
    horizontal=True
)

measured_percentages = None

if input_method == "Manual Entry":
    st.markdown("### Enter Measured Spectral Percentages")
    st.caption("Enter the percentage of total irradiance in each wavelength band")

    cols = st.columns(3)
    measured_percentages = []

    for i, band in enumerate(WAVELENGTH_BANDS):
        col_idx = i % 3
        with cols[col_idx]:
            value = st.number_input(
                f"{band['start']}-{band['end']} nm",
                min_value=0.0,
                max_value=100.0,
                value=float(band["reference"]),
                step=0.1,
                key=f"band_{i}"
            )
            measured_percentages.append(value)

elif input_method == "Upload CSV":
    st.markdown("### Upload Spectral Data")
    st.caption("CSV should have columns: wavelength_start, wavelength_end, percentage")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "percentage" in df.columns:
                measured_percentages = df["percentage"].tolist()[:6]
                st.success(f"Loaded {len(measured_percentages)} wavelength bands")
                st.dataframe(df)
            else:
                st.error("CSV must contain a 'percentage' column")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

else:  # Demo Data
    st.markdown("### Demo Data")
    st.info("Using sample data for demonstration")

    # Generate realistic demo data with some variation
    np.random.seed(42)
    variation = np.random.uniform(-0.15, 0.15, 6)
    measured_percentages = [
        ref * (1 + var) for ref, var in zip(REFERENCE_PERCENTAGES, variation)
    ]

    # Normalize to 100%
    total = sum(measured_percentages)
    measured_percentages = [p * 100 / total for p in measured_percentages]

st.markdown("---")

# Analysis section
if measured_percentages is not None and len(measured_percentages) == 6:
    st.markdown("### Analysis Results")

    # Perform calculation
    result = calculate_spectral_match_class(measured_percentages, REFERENCE_PERCENTAGES)

    # Display classification prominently
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
            f"<p style='text-align: center;'>Maximum Deviation: {result.max_deviation:.1f}%</p>",
            unsafe_allow_html=True
        )

    # Detailed results table
    st.markdown("### Band-by-Band Results")

    results_df = pd.DataFrame(result.band_results)
    results_df = results_df.rename(columns={
        "band": "Wavelength Band",
        "measured": "Measured (%)",
        "reference": "Reference (%)",
        "ratio": "Ratio",
        "deviation_percent": "Deviation (%)",
        "classification": "Class"
    })

    # Style the dataframe
    def color_class(val):
        if val == "A":
            return "background-color: #90EE90"
        elif val == "B":
            return "background-color: #FFD700"
        else:
            return "background-color: #FF6B6B"

    styled_df = results_df.style.applymap(
        color_class,
        subset=["Class"]
    ).format({
        "Measured (%)": "{:.2f}",
        "Reference (%)": "{:.2f}",
        "Ratio": "{:.3f}",
        "Deviation (%)": "{:.2f}"
    })

    st.dataframe(styled_df, use_container_width=True)

    # Visualization
    st.markdown("### Spectral Comparison Chart")

    chart_data = pd.DataFrame({
        "Band": [b["band"] for b in result.band_results],
        "Measured": [b["measured"] for b in result.band_results],
        "Reference": [b["reference"] for b in result.band_results],
    })

    st.bar_chart(
        chart_data.set_index("Band"),
        use_container_width=True
    )

    # Save to database option (with graceful handling)
    st.markdown("---")
    st.markdown("### Save Results")

    if st.button("Save to Database", type="primary"):
        try:
            from database import is_database_configured, get_database_connection, DatabaseConnectionError

            if not is_database_configured():
                st.warning("Database not configured. Results cannot be saved.")
                st.info("To enable database storage, configure DATABASE_URL environment variable.")
            else:
                try:
                    # Attempt to save (lazy connection)
                    st.info("Database save functionality available when connected.")
                    # In production, this would save the results
                except DatabaseConnectionError as e:
                    st.error(f"Database connection failed: {e}")
                    st.info("Results can still be exported manually.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Export option (always available)
    if st.button("Export as CSV"):
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="spectral_match_results.csv",
            mime="text/csv"
        )

else:
    st.info("Please enter or upload spectral measurement data to begin analysis.")

# Sidebar info
with st.sidebar:
    st.markdown("### Reference: AM1.5G Spectrum")
    ref_df = pd.DataFrame({
        "Band (nm)": [f"{b['start']}-{b['end']}" for b in WAVELENGTH_BANDS],
        "Reference (%)": REFERENCE_PERCENTAGES
    })
    st.dataframe(ref_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Classification Limits")
    st.markdown("""
    | Class | Limit |
    |-------|-------|
    | A | +/-25% |
    | B | +/-40% |
    | C | >40% |
    """)
