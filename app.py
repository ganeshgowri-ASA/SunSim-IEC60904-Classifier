"""
SunSim-IEC60904-Classifier - Main Application Entry Point

Professional Sun Simulator Classification System
IEC 60904-9 Ed.3 Compliant

This application provides classification analysis for sun simulators based on:
- Spectral Match
- Non-uniformity of Irradiance
- Temporal Instability

IMPORTANT: Database connections are LAZY-LOADED to prevent 502 errors.
The app will function normally even without database connectivity.
Pages that don't require database access will work independently.
"""

import streamlit as st
import logging

# Configure logging BEFORE any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration (no database connections here)
from utils.config import get_config, is_production, get_railway_info

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="SunSim Classifier | IEC 60904-9",
    page_icon="sun_with_face",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# SunSim-IEC60904-Classifier\nIEC 60904-9 Ed.3 Compliant Sun Simulator Classification System"
    }
)


def show_database_status():
    """Display database connection status in sidebar."""
    # Import here to avoid loading database module unnecessarily
    from database import is_database_configured, get_connection_status

    with st.sidebar:
        st.markdown("---")
        st.markdown("### System Status")

        if is_database_configured():
            # Only check actual connection status if configured
            status = get_connection_status()
            if status["connected"]:
                st.success("Database: Connected")
            elif status["initialization_attempted"]:
                st.warning("Database: Connection failed")
                if status["last_error"]:
                    st.caption(f"Error: {status['last_error'][:50]}...")
            else:
                st.info("Database: Available (not connected)")
        else:
            st.info("Database: Not configured")
            st.caption("Some features may be limited")

        # Railway deployment info
        railway_info = get_railway_info()
        if railway_info:
            st.markdown("---")
            st.caption(f"Environment: {railway_info['environment']}")


def main():
    """Main application entry point."""
    config = get_config()

    # Sidebar navigation
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/logo.svg", width=50)
        st.title("SunSim Classifier")
        st.caption(f"IEC 60904-9 {config['iec_standard_version']}")

        st.markdown("---")

        # Navigation info
        st.markdown("### Navigation")
        st.markdown("""
        Use the pages in the sidebar to:
        - **Spectral Match**: Analyze spectral distribution
        - **Uniformity**: Check irradiance uniformity
        - **Temporal Stability**: Measure temporal stability
        - **Reports**: Generate classification reports
        - **History**: View past classifications
        """)

    # Main content area
    st.title("Sun Simulator Classification System")
    st.markdown("#### IEC 60904-9 Ed.3 Compliant | ISO 17025 Ready")

    # Welcome content
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Welcome

        This application provides professional classification of sun simulators
        according to **IEC 60904-9 Edition 3** standards.

        #### Classification Criteria

        | Criterion | Class A | Class B | Class C |
        |-----------|---------|---------|---------|
        | Spectral Match | +/-25% | +/-40% | >40% |
        | Non-uniformity | +/-2% | +/-5% | +/-10% |
        | Temporal Instability (STI) | 0.5% | 2% | 10% |
        | Temporal Instability (LTI) | 2% | 5% | 10% |
        """)

    with col2:
        st.markdown("""
        ### Quick Start

        1. **Upload Data**: Import your measurement data
        2. **Analyze**: Run spectral, uniformity, and temporal analysis
        3. **Classify**: Get your overall classification (e.g., AAA, ABA)
        4. **Report**: Generate ISO 17025 compliant reports

        #### Features

        - Real-time classification calculations
        - Interactive visualization
        - SPC/MSA quality control charts
        - Historical trend analysis
        - PDF report generation
        """)

    st.markdown("---")

    # Status cards
    st.markdown("### System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Standard Version",
            value=config['iec_standard_version'],
            delta=None
        )

    with col2:
        # Check database status without connecting
        from database import is_database_configured
        db_status = "Available" if is_database_configured() else "Not Configured"
        st.metric(
            label="Database",
            value=db_status,
            delta=None
        )

    with col3:
        env = "Production" if is_production() else "Development"
        st.metric(
            label="Environment",
            value=env,
            delta=None
        )

    with col4:
        st.metric(
            label="App Version",
            value=config['app_version'],
            delta=None
        )

    # Show database status in sidebar
    show_database_status()

    # Footer
    st.markdown("---")
    st.caption(
        "SunSim-IEC60904-Classifier | "
        "Developed for IEC 60904-9 Ed.3 Compliance | "
        "ISO 17025 Report Generation Ready"
    )


if __name__ == "__main__":
    main()
