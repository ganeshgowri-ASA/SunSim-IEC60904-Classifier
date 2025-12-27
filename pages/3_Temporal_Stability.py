"""
Temporal Stability Analysis Page

Analyzes temporal stability of irradiance during measurement period.
This page does NOT require database connectivity to function.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import calculations (no database dependency)
from utils.calculations import calculate_temporal_stability_class, IEC_60904_9_LIMITS

st.set_page_config(
    page_title="Temporal Stability | SunSim Classifier",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("Temporal Instability Analysis")
st.markdown("#### IEC 60904-9 Ed.3 Temporal Stability Classification")

st.markdown("""
This analysis evaluates the temporal stability of your sun simulator's
irradiance output according to IEC 60904-9 Ed.3.

**Classification is based on two metrics:**
- **Short-Term Instability (STI)**: Variation within millisecond timeframes
- **Long-Term Instability (LTI)**: Overall variation during measurement period

**Classification Limits:**
| Class | STI | LTI |
|-------|-----|-----|
| A | 0.5% | 2% |
| B | 2% | 5% |
| C | 10% | 10% |
""")

st.markdown("---")

# Configuration
col1, col2, col3 = st.columns(3)

with col1:
    sample_rate = st.number_input(
        "Sample Rate (Hz)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

with col2:
    duration_ms = st.number_input(
        "Duration (ms)",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )

with col3:
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
    ["Generate Demo Data", "Upload CSV"],
    horizontal=True
)

time_series = None
timestamps = None

if input_method == "Generate Demo Data":
    st.markdown("### Demo Data Configuration")

    col1, col2 = st.columns(2)

    with col1:
        stability_quality = st.selectbox(
            "Simulated Quality",
            [
                "Class A (STI<0.5%, LTI<2%)",
                "Class B (STI<2%, LTI<5%)",
                "Class C (STI>2%, LTI>5%)"
            ]
        )

    with col2:
        noise_type = st.selectbox(
            "Noise Pattern",
            ["Random", "Sinusoidal + Random", "Step Changes"]
        )

    # Generate data based on selection
    n_samples = int(sample_rate * duration_ms / 1000)
    timestamps = np.linspace(0, duration_ms, n_samples)

    np.random.seed(42)

    if "Class A" in stability_quality:
        sti_level = 0.003
        lti_level = 0.015
    elif "Class B" in stability_quality:
        sti_level = 0.015
        lti_level = 0.04
    else:
        sti_level = 0.08
        lti_level = 0.08

    # Base signal
    time_series = np.ones(n_samples) * nominal_irradiance

    if noise_type == "Random":
        # Pure random noise
        time_series += np.random.normal(0, nominal_irradiance * sti_level, n_samples)
        # Add slow drift for LTI
        time_series += np.linspace(0, nominal_irradiance * lti_level, n_samples)

    elif noise_type == "Sinusoidal + Random":
        # High frequency noise (STI)
        time_series += np.random.normal(0, nominal_irradiance * sti_level * 0.5, n_samples)
        # Low frequency sinusoidal variation (LTI)
        time_series += nominal_irradiance * lti_level * 0.5 * np.sin(2 * np.pi * timestamps / duration_ms)

    else:  # Step Changes
        # Random noise
        time_series += np.random.normal(0, nominal_irradiance * sti_level * 0.3, n_samples)
        # Step changes
        n_steps = 3
        for i in range(1, n_steps + 1):
            step_idx = int(n_samples * i / (n_steps + 1))
            step_size = np.random.uniform(-lti_level, lti_level) * nominal_irradiance
            time_series[step_idx:] += step_size

    st.success(f"Generated {n_samples} samples over {duration_ms} ms")

elif input_method == "Upload CSV":
    st.markdown("### Upload Time Series Data")
    st.caption("CSV should have columns: timestamp_ms, irradiance")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "irradiance" in df.columns:
                time_series = df["irradiance"].values
                if "timestamp_ms" in df.columns:
                    timestamps = df["timestamp_ms"].values
                else:
                    timestamps = np.arange(len(time_series))
                st.success(f"Loaded {len(time_series)} samples")
            else:
                st.error("CSV must contain an 'irradiance' column")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

st.markdown("---")

# Analysis
if time_series is not None:
    st.markdown("### Analysis Results")

    # Perform calculation
    result = calculate_temporal_stability_class(time_series, sample_rate)

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

    # Instability metrics
    col1, col2 = st.columns(2)

    limits = IEC_60904_9_LIMITS["temporal_stability"]

    with col1:
        sti_status = "Pass" if result.short_term_instability <= limits[result.classification]["sti"] else "Marginal"
        st.metric(
            "Short-Term Instability (STI)",
            f"{result.short_term_instability:.3f}%",
            delta=f"Limit: {limits[result.classification]['sti']}%",
            delta_color="off"
        )

    with col2:
        lti_status = "Pass" if result.long_term_instability <= limits[result.classification]["lti"] else "Marginal"
        st.metric(
            "Long-Term Instability (LTI)",
            f"{result.long_term_instability:.3f}%",
            delta=f"Limit: {limits[result.classification]['lti']}%",
            delta_color="off"
        )

    # Time series visualization
    st.markdown("### Irradiance Time Series")

    chart_df = pd.DataFrame({
        "Time (ms)": timestamps if timestamps is not None else np.arange(len(time_series)),
        "Irradiance (W/m²)": time_series
    })

    st.line_chart(chart_df.set_index("Time (ms)"), use_container_width=True)

    # Statistics
    st.markdown("### Signal Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{np.mean(time_series):.2f} W/m²")

    with col2:
        st.metric("Std Dev", f"{np.std(time_series):.2f} W/m²")

    with col3:
        st.metric("Min", f"{np.min(time_series):.2f} W/m²")

    with col4:
        st.metric("Max", f"{np.max(time_series):.2f} W/m²")

    # Histogram
    st.markdown("### Irradiance Distribution")

    hist_df = pd.DataFrame({"Irradiance": time_series})
    st.bar_chart(
        hist_df["Irradiance"].value_counts(bins=50).sort_index(),
        use_container_width=True
    )

    # Save/Export
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
            export_df = pd.DataFrame({
                "timestamp_ms": timestamps if timestamps is not None else np.arange(len(time_series)),
                "irradiance": time_series
            })
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Time Series CSV",
                data=csv,
                file_name="temporal_stability_data.csv",
                mime="text/csv"
            )

# Sidebar reference
with st.sidebar:
    st.markdown("### Classification Limits")

    limits = IEC_60904_9_LIMITS["temporal_stability"]
    limits_df = pd.DataFrame({
        "Class": ["A", "B", "C"],
        "STI (%)": [limits["A"]["sti"], limits["B"]["sti"], limits["C"]["sti"]],
        "LTI (%)": [limits["A"]["lti"], limits["B"]["lti"], limits["C"]["lti"]]
    })
    st.dataframe(limits_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Definitions")
    st.markdown("""
    **STI (Short-Term Instability)**
    - Variation within 1ms windows
    - Captures high-frequency fluctuations
    - Critical for fast I-V measurements

    **LTI (Long-Term Instability)**
    - Overall variation during test
    - Captures drift and slow changes
    - Important for measurement repeatability
    """)
