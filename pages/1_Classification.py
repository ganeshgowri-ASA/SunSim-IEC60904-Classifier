"""
Classification Page - Sun Simulator Analysis

Perform IEC 60904-9 Ed.3 classification analysis for sun simulators.
Handles database unavailability gracefully - core functionality works offline.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any

# Import database utilities with lazy loading
from utils.db import (
    is_database_available,
    execute_query_safe,
    get_db_session,
    show_database_status,
    require_database,
)

st.set_page_config(
    page_title="Classification - SunSim",
    page_icon="analytics",
    layout="wide",
)


def classify_spectral_match(data: Dict[str, Any]) -> str:
    """
    Classify spectral match according to IEC 60904-9.

    This is a core calculation that works without database.

    Args:
        data: Spectral irradiance data

    Returns:
        Classification grade (A+, A, B, or C)
    """
    # Placeholder classification logic
    # In production, this would analyze spectral irradiance ratios
    deviation = data.get('max_deviation', 0.15)

    if deviation <= 0.125:
        return 'A+'
    elif deviation <= 0.25:
        return 'A'
    elif deviation <= 0.40:
        return 'B'
    else:
        return 'C'


def classify_uniformity(data: Dict[str, Any]) -> str:
    """
    Classify spatial uniformity according to IEC 60904-9.

    Works without database connection.
    """
    non_uniformity = data.get('non_uniformity', 0.02)

    if non_uniformity <= 0.01:
        return 'A+'
    elif non_uniformity <= 0.02:
        return 'A'
    elif non_uniformity <= 0.05:
        return 'B'
    else:
        return 'C'


def classify_temporal_stability(data: Dict[str, Any]) -> str:
    """
    Classify temporal stability according to IEC 60904-9.

    Works without database connection.
    """
    instability = data.get('instability', 0.01)

    if instability <= 0.005:
        return 'A+'
    elif instability <= 0.01:
        return 'A'
    elif instability <= 0.02:
        return 'B'
    else:
        return 'C'


def get_overall_classification(spectral: str, uniformity: str, stability: str) -> str:
    """
    Determine overall classification from individual grades.

    The overall class is the lowest of the three individual classes.
    """
    grades = {'A+': 4, 'A': 3, 'B': 2, 'C': 1}
    reverse_grades = {4: 'A+', 3: 'A', 2: 'B', 1: 'C'}

    min_grade = min(grades[spectral], grades[uniformity], grades[stability])
    return reverse_grades[min_grade]


def save_classification_to_db(
    simulator_name: str,
    spectral_class: str,
    uniformity_class: str,
    stability_class: str,
    overall_class: str,
) -> Optional[int]:
    """
    Save classification results to database.

    Handles database unavailability gracefully.

    Returns:
        Classification ID if saved, None if database unavailable.
    """
    if not is_database_available():
        st.info(
            "Classification complete but not saved to database. "
            "Database is not available.",
            icon="info"
        )
        return None

    try:
        with get_db_session() as session:
            if session is None:
                return None

            from sqlalchemy import text

            result = session.execute(
                text("""
                    INSERT INTO classifications
                    (simulator_name, classification_date, spectral_class,
                     uniformity_class, stability_class, overall_class)
                    VALUES (:name, :date, :spectral, :uniformity, :stability, :overall)
                    RETURNING id
                """),
                {
                    'name': simulator_name,
                    'date': datetime.now(),
                    'spectral': spectral_class,
                    'uniformity': uniformity_class,
                    'stability': stability_class,
                    'overall': overall_class,
                }
            )

            classification_id = result.scalar()
            st.success(f"Classification saved with ID: {classification_id}")
            return classification_id

    except Exception as e:
        st.warning(
            f"Could not save classification to database: {str(e)}. "
            "Results are still valid but not persisted.",
            icon="exclamation-triangle"
        )
        return None


def load_saved_parameters(simulator_name: str) -> Optional[Dict]:
    """
    Load previously saved parameters for a simulator.

    Returns default values if database is unavailable.
    """
    default_params = {
        'irradiance_level': 1000.0,
        'test_area_size': 0.1,
        'measurement_duration': 60,
    }

    if not is_database_available():
        return default_params

    result = execute_query_safe(
        """
        SELECT irradiance_level, test_area_size, measurement_duration
        FROM simulator_parameters
        WHERE simulator_name = :name
        ORDER BY created_at DESC
        LIMIT 1
        """,
        params={'name': simulator_name},
        default=None,
        show_warning=False
    )

    if result and len(result) > 0:
        row = result[0]
        return {
            'irradiance_level': row[0],
            'test_area_size': row[1],
            'measurement_duration': row[2],
        }

    return default_params


def main():
    """Classification page main function."""

    # Sidebar with navigation and status
    with st.sidebar:
        st.title("Classification")
        show_database_status()

    st.title("Sun Simulator Classification")
    st.markdown("IEC 60904-9 Ed.3 Analysis")

    # Show warning if offline
    if not is_database_available():
        st.warning(
            "Running in offline mode. Classifications will not be saved to database, "
            "but all analysis features are fully functional.",
            icon="exclamation-triangle"
        )

    # Input section
    st.markdown("### Simulator Information")

    col1, col2 = st.columns(2)

    with col1:
        simulator_name = st.text_input(
            "Simulator Name",
            placeholder="e.g., XenonSun 3000A"
        )

    with col2:
        # Try to load saved parameters
        if simulator_name:
            params = load_saved_parameters(simulator_name)
        else:
            params = {'irradiance_level': 1000.0, 'test_area_size': 0.1, 'measurement_duration': 60}

        irradiance = st.number_input(
            "Target Irradiance (W/m²)",
            value=params['irradiance_level'],
            min_value=100.0,
            max_value=2000.0,
        )

    st.markdown("---")

    # Data input tabs
    tab1, tab2, tab3 = st.tabs(["Spectral Match", "Uniformity", "Temporal Stability"])

    spectral_data = {}
    uniformity_data = {}
    stability_data = {}

    with tab1:
        st.markdown("#### Spectral Match Analysis")
        st.markdown(
            "Enter spectral irradiance ratios for each wavelength interval "
            "relative to AM1.5G reference spectrum."
        )

        # Simplified input for demo
        spectral_deviation = st.slider(
            "Maximum Spectral Deviation",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Maximum deviation from unity in any wavelength interval"
        )
        spectral_data['max_deviation'] = spectral_deviation

        uploaded_spectral = st.file_uploader(
            "Or upload spectral data (CSV)",
            type=['csv'],
            key='spectral_upload'
        )

    with tab2:
        st.markdown("#### Spatial Uniformity Analysis")
        st.markdown(
            "Enter or calculate non-uniformity across the test plane."
        )

        non_uniformity = st.slider(
            "Non-uniformity (%)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
        ) / 100
        uniformity_data['non_uniformity'] = non_uniformity

        uploaded_uniformity = st.file_uploader(
            "Or upload uniformity map (CSV)",
            type=['csv'],
            key='uniformity_upload'
        )

    with tab3:
        st.markdown("#### Temporal Stability Analysis")
        st.markdown(
            "Enter or calculate temporal instability of irradiance."
        )

        instability = st.slider(
            "Temporal Instability (%)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        ) / 100
        stability_data['instability'] = instability

        uploaded_stability = st.file_uploader(
            "Or upload stability data (CSV)",
            type=['csv'],
            key='stability_upload'
        )

    st.markdown("---")

    # Classification button
    if st.button("Perform Classification", type="primary", disabled=not simulator_name):
        with st.spinner("Analyzing data..."):
            # Perform classification (works offline)
            spectral_class = classify_spectral_match(spectral_data)
            uniformity_class = classify_uniformity(uniformity_data)
            stability_class = classify_temporal_stability(stability_data)
            overall_class = get_overall_classification(
                spectral_class, uniformity_class, stability_class
            )

        # Display results
        st.markdown("### Classification Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Spectral Match", f"Class {spectral_class}")

        with col2:
            st.metric("Uniformity", f"Class {uniformity_class}")

        with col3:
            st.metric("Stability", f"Class {stability_class}")

        with col4:
            st.metric("Overall", f"Class {overall_class}")

        # Try to save to database (graceful if unavailable)
        save_classification_to_db(
            simulator_name,
            spectral_class,
            uniformity_class,
            stability_class,
            overall_class,
        )

        # Show report generation option
        st.markdown("---")
        st.markdown("### Generate Report")

        report_format = st.selectbox(
            "Report Format",
            options=["PDF", "Excel", "JSON"],
        )

        if st.button("Generate Report"):
            st.info("Report generation will be implemented in the Reports page.")


if __name__ == "__main__":
    main()
