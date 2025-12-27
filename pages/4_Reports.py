"""
Classification Reports Page

Generates ISO 17025 compliant classification reports.
This page can function without database but has enhanced features with it.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Import utilities (no database dependency at import)
from utils.calculations import (
    get_overall_classification,
    interpret_classification,
    IEC_60904_9_LIMITS
)
from utils.config import get_config

st.set_page_config(
    page_title="Reports | SunSim Classifier",
    page_icon="clipboard",
    layout="wide"
)

st.title("Classification Reports")
st.markdown("#### ISO 17025 Compliant Report Generation")

st.markdown("""
Generate comprehensive classification reports for your sun simulator.
Reports include all three classification criteria and overall classification.
""")

st.markdown("---")

# Report input section
st.markdown("### Simulator Information")

col1, col2 = st.columns(2)

with col1:
    simulator_id = st.text_input("Simulator ID/Name", value="SunSim-001")
    manufacturer = st.text_input("Manufacturer", value="")
    model = st.text_input("Model", value="")

with col2:
    serial_number = st.text_input("Serial Number", value="")
    test_date = st.date_input("Test Date", value=datetime.now())
    operator = st.text_input("Operator Name", value="")

st.markdown("---")

# Classification input
st.markdown("### Classification Results")
st.caption("Enter the classification results from each analysis")

col1, col2, col3 = st.columns(3)

with col1:
    spectral_class = st.selectbox(
        "Spectral Match Class",
        options=["A", "B", "C"],
        index=0,
        help="Result from Spectral Match analysis"
    )
    spectral_deviation = st.number_input(
        "Max Spectral Deviation (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

with col2:
    uniformity_class = st.selectbox(
        "Uniformity Class",
        options=["A", "B", "C"],
        index=0,
        help="Result from Uniformity analysis"
    )
    uniformity_value = st.number_input(
        "Non-uniformity (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.5
    )

with col3:
    temporal_class = st.selectbox(
        "Temporal Stability Class",
        options=["A", "B", "C"],
        index=0,
        help="Result from Temporal Stability analysis"
    )
    sti_value = st.number_input("STI (%)", min_value=0.0, max_value=100.0, value=0.3)
    lti_value = st.number_input("LTI (%)", min_value=0.0, max_value=100.0, value=1.5)

# Calculate overall classification
overall_class = get_overall_classification(spectral_class, uniformity_class, temporal_class)
interpretation = interpret_classification(overall_class)

st.markdown("---")

# Display overall classification
st.markdown("### Overall Classification")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Color based on quality
    if overall_class == "AAA":
        color = "green"
    elif interpretation["quality_grade"] == "High Grade":
        color = "#228B22"
    elif interpretation["quality_grade"] == "Standard Grade":
        color = "orange"
    else:
        color = "#CD853F"

    st.markdown(
        f"<h1 style='text-align: center; color: {color}; font-size: 4em;'>"
        f"Class {overall_class}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h3 style='text-align: center;'>{interpretation['quality_grade']}</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center;'>{interpretation['description']}</p>",
        unsafe_allow_html=True
    )

st.markdown("---")

# Report preview
st.markdown("### Report Preview")

config = get_config()

report_content = f"""
# SUN SIMULATOR CLASSIFICATION REPORT
## IEC 60904-9 {config['iec_standard_version']} Compliance

---

### Simulator Information
| Field | Value |
|-------|-------|
| Simulator ID | {simulator_id} |
| Manufacturer | {manufacturer or 'N/A'} |
| Model | {model or 'N/A'} |
| Serial Number | {serial_number or 'N/A'} |
| Test Date | {test_date} |
| Operator | {operator or 'N/A'} |

---

### Classification Summary

## OVERALL CLASSIFICATION: {overall_class}
**Quality Grade: {interpretation['quality_grade']}**

{interpretation['description']}

---

### Individual Classifications

#### Spectral Match: Class {spectral_class}
- Maximum Deviation: {spectral_deviation:.1f}%
- Quality: {interpretation['spectral_quality']}
- Limit for Class {spectral_class}: +/-{IEC_60904_9_LIMITS['spectral_match'][spectral_class]*100:.0f}%

#### Non-Uniformity of Irradiance: Class {uniformity_class}
- Non-uniformity: {uniformity_value:.2f}%
- Quality: {interpretation['uniformity_quality']}
- Limit for Class {uniformity_class}: +/-{IEC_60904_9_LIMITS['uniformity'][uniformity_class]:.1f}%

#### Temporal Instability: Class {temporal_class}
- Short-Term Instability (STI): {sti_value:.3f}%
- Long-Term Instability (LTI): {lti_value:.3f}%
- Quality: {interpretation['temporal_quality']}
- Limits for Class {temporal_class}: STI {IEC_60904_9_LIMITS['temporal_stability'][temporal_class]['sti']}%, LTI {IEC_60904_9_LIMITS['temporal_stability'][temporal_class]['lti']}%

---

### Classification Criteria (IEC 60904-9 {config['iec_standard_version']})

| Characteristic | Class A | Class B | Class C |
|----------------|---------|---------|---------|
| Spectral Match | +/-25% | +/-40% | >40% |
| Non-uniformity | +/-2% | +/-5% | +/-10% |
| Temporal Instability (STI) | 0.5% | 2% | 10% |
| Temporal Instability (LTI) | 2% | 5% | 10% |

---

### Certification Statement

This sun simulator has been classified according to IEC 60904-9 {config['iec_standard_version']}
"Photovoltaic devices - Part 9: Classification of solar simulator characteristics."

The classification is valid for the conditions under which the measurements were performed.

---

*Report generated by SunSim-IEC60904-Classifier*
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

st.markdown(report_content)

st.markdown("---")

# Export options
st.markdown("### Export Report")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="Download as Markdown",
        data=report_content,
        file_name=f"classification_report_{simulator_id}_{test_date}.md",
        mime="text/markdown",
        type="primary"
    )

with col2:
    # Create CSV summary
    summary_data = {
        "Field": [
            "Simulator ID", "Test Date", "Spectral Class", "Uniformity Class",
            "Temporal Class", "Overall Class", "Quality Grade"
        ],
        "Value": [
            simulator_id, str(test_date), spectral_class, uniformity_class,
            temporal_class, overall_class, interpretation["quality_grade"]
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_content = summary_df.to_csv(index=False)

    st.download_button(
        label="Download Summary CSV",
        data=csv_content,
        file_name=f"classification_summary_{simulator_id}_{test_date}.csv",
        mime="text/csv"
    )

with col3:
    # Save to database option
    if st.button("Save to Database"):
        try:
            from database import is_database_configured, DatabaseConnectionError

            if not is_database_configured():
                st.warning("Database not configured.")
                st.info("Configure DATABASE_URL to enable saving.")
            else:
                try:
                    from database import get_database_connection

                    with get_database_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO classification_results
                                (simulator_id, spectral_class, uniformity_class,
                                 temporal_class, overall_class, notes)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (
                                simulator_id, spectral_class, uniformity_class,
                                temporal_class, overall_class,
                                f"Operator: {operator}"
                            ))
                            result_id = cur.fetchone()[0]
                            st.success(f"Saved to database with ID: {result_id}")
                except DatabaseConnectionError as e:
                    st.error(f"Database connection failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# Notes section
st.markdown("---")
st.markdown("### Additional Notes")

notes = st.text_area(
    "Enter any additional notes or observations",
    height=100,
    placeholder="Environmental conditions, special considerations, etc."
)

# Sidebar
with st.sidebar:
    st.markdown("### Quick Reference")
    st.markdown(f"""
    **Current Classification:**
    - Spectral: Class {spectral_class}
    - Uniformity: Class {uniformity_class}
    - Temporal: Class {temporal_class}
    - **Overall: {overall_class}**

    **Grade:** {interpretation['quality_grade']}
    """)

    st.markdown("---")
    st.markdown("### ISO 17025 Notes")
    st.markdown("""
    For full ISO 17025 compliance, ensure:
    - Calibrated measurement equipment
    - Documented procedures
    - Trained personnel
    - Environmental conditions recorded
    - Measurement uncertainty considered
    """)
