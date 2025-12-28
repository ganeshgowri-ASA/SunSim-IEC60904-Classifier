"""
SunSim IEC 60904-9 Classifier - Main Application

Professional Sun Simulator Classification System compliant with IEC 60904-9 Ed.3.
Features:
- Spectral Match Analysis
- Uniformity Assessment
- Temporal Stability Analysis
- ISO 17025 Report Generation
- SPC/MSA Quality Control

Database connections are lazy-loaded to prevent startup failures on cloud platforms.
"""

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="SunSim IEC 60904-9 Classifier",
    page_icon="brightness_high",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import database utilities AFTER page config
# Note: This import does NOT establish database connections (lazy loading)
from utils.db import (
    is_database_available,
    show_database_status,
    execute_query_safe,
)


def init_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.offline_mode = False


def show_offline_warning():
    """Display warning when running in offline mode."""
    if not is_database_available():
        st.warning(
            "Running in offline mode. Database features are disabled. "
            "Historical data and report storage will not be available.",
            icon="exclamation-triangle"
        )
        st.session_state.offline_mode = True


def load_recent_classifications() -> list:
    """
    Load recent classifications from database with graceful fallback.

    Returns:
        List of recent classification records or empty list.
    """
    if not is_database_available():
        return []

    result = execute_query_safe(
        """
        SELECT id, classification_date, simulator_name, overall_class
        FROM classifications
        ORDER BY classification_date DESC
        LIMIT 5
        """,
        default=[],
        show_warning=False
    )

    return result if result else []


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("SunSim Classifier")
        st.markdown("IEC 60904-9 Ed.3 Compliant")

        st.markdown("---")
        st.markdown("**Navigation**")

        # Navigation links
        st.page_link("app.py", label="Home", icon="home")
        st.page_link("pages/1_Classification.py", label="Classification", icon="analytics")
        st.page_link("pages/2_Reports.py", label="Reports", icon="description")
        st.page_link("pages/3_Settings.py", label="Settings", icon="settings")

        # Show database status in sidebar
        show_database_status()

    # Main content
    st.title("Sun Simulator Classification System")
    st.markdown("### IEC 60904-9 Ed.3 Compliant Analysis")

    # Show offline warning if database is unavailable
    show_offline_warning()

    # Feature overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Spectral Match")
        st.markdown(
            "Analyze spectral irradiance distribution "
            "against AM1.5G reference spectrum."
        )

    with col2:
        st.markdown("#### Uniformity")
        st.markdown(
            "Assess spatial uniformity of irradiance "
            "across the test plane."
        )

    with col3:
        st.markdown("#### Temporal Stability")
        st.markdown(
            "Evaluate short-term and long-term "
            "irradiance stability."
        )

    st.markdown("---")

    # Recent classifications section (database dependent)
    st.markdown("### Recent Classifications")

    try:
        recent = load_recent_classifications()

        if recent:
            for record in recent:
                with st.container():
                    cols = st.columns([2, 2, 1])
                    cols[0].write(f"**{record[2]}**")  # simulator_name
                    cols[1].write(str(record[1]))  # date
                    cols[2].write(f"Class {record[3]}")  # overall_class
        elif is_database_available():
            st.info("No classifications yet. Start by classifying a sun simulator.")
        else:
            st.info(
                "Recent classifications unavailable in offline mode. "
                "Connect to database to view history."
            )

    except Exception as e:
        # Graceful fallback - never crash the main page
        st.info(
            "Could not load recent classifications. "
            "This feature requires database connectivity."
        )

    # Quick start section
    st.markdown("---")
    st.markdown("### Quick Start")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **New Classification**

            1. Navigate to the Classification page
            2. Upload spectral, uniformity, or stability data
            3. Review results and generate report
            """
        )

        if st.button("Start Classification", type="primary"):
            st.switch_page("pages/1_Classification.py")

    with col2:
        st.markdown(
            """
            **View Reports**

            Access previously generated classification reports
            and export them in various formats.
            """
        )

        if st.button("View Reports"):
            st.switch_page("pages/2_Reports.py")


if __name__ == "__main__":
    main()
