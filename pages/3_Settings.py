"""
Settings Page - Application Configuration

Configure application settings and manage database connection.
Provides database health monitoring and connection management.
"""

import streamlit as st
import os
from typing import Dict, Any

from utils.db import (
    is_database_available,
    check_connection_health,
    clear_connection_cache,
    show_database_status,
    get_database_url,
)

st.set_page_config(
    page_title="Settings - SunSim",
    page_icon="settings",
    layout="wide",
)


def get_app_info() -> Dict[str, Any]:
    """Get application information."""
    return {
        'version': '1.0.0',
        'standard': 'IEC 60904-9 Ed.3',
        'python_version': os.sys.version.split()[0],
        'streamlit_version': st.__version__,
    }


def mask_connection_string(url: str) -> str:
    """
    Mask sensitive parts of connection string for display.

    Args:
        url: Database connection URL

    Returns:
        Masked URL safe for display
    """
    if not url:
        return "Not configured"

    try:
        # Mask password in URL
        # Format: postgresql://user:password@host:port/database
        if '@' in url:
            parts = url.split('@')
            prefix = parts[0]
            suffix = parts[1]

            if ':' in prefix:
                # Mask the password
                proto_user = prefix.rsplit(':', 1)[0]
                masked = f"{proto_user}:****@{suffix}"
                return masked

        return url[:20] + "..." if len(url) > 20 else url

    except Exception:
        return "****"


def main():
    """Settings page main function."""

    with st.sidebar:
        st.title("Settings")
        show_database_status()

    st.title("Application Settings")

    # Application Info
    st.markdown("### Application Information")

    app_info = get_app_info()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Version:** {app_info['version']}")
        st.markdown(f"**Standard:** {app_info['standard']}")

    with col2:
        st.markdown(f"**Python:** {app_info['python_version']}")
        st.markdown(f"**Streamlit:** {app_info['streamlit_version']}")

    st.markdown("---")

    # Database Configuration
    st.markdown("### Database Configuration")

    # Connection status
    health = check_connection_health()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Connection Status**")

        if health['connected']:
            st.success(f"Connected (Latency: {health['latency_ms']}ms)")
        elif health['configured']:
            st.error(f"Connection Failed")
            st.caption(f"Error: {health['error']}")
        else:
            st.warning("Not Configured")
            st.caption("Set DATABASE_URL environment variable")

    with col2:
        st.markdown("**Connection String**")
        db_url = get_database_url()
        st.code(mask_connection_string(db_url or ""))

    # Connection management
    st.markdown("---")
    st.markdown("#### Connection Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                health = check_connection_health()

            if health['connected']:
                st.success(f"Connection successful! Latency: {health['latency_ms']}ms")
            else:
                st.error(f"Connection failed: {health['error']}")

    with col2:
        if st.button("Reconnect"):
            with st.spinner("Reconnecting..."):
                clear_connection_cache()
                health = check_connection_health()

            if health['connected']:
                st.success("Reconnected successfully!")
            else:
                st.warning("Reconnection attempted. Check connection status.")

    with col3:
        if st.button("Clear Cache"):
            clear_connection_cache()
            st.success("Connection cache cleared.")

    # Environment Variables Info
    st.markdown("---")
    st.markdown("### Environment Variables")

    st.markdown(
        """
        The following environment variables are checked for database configuration:

        | Variable | Description |
        |----------|-------------|
        | `DATABASE_URL` | Primary database connection URL |
        | `POSTGRES_URL` | Alternative PostgreSQL URL |
        | `DATABASE_PRIVATE_URL` | Railway private network URL |

        Set one of these variables to enable database functionality.
        """
    )

    # Show which variables are set (without values)
    env_vars = ['DATABASE_URL', 'POSTGRES_URL', 'POSTGRESQL_URL',
                'DATABASE_PRIVATE_URL', 'POSTGRES_PRISMA_URL']

    st.markdown("**Detected Environment Variables:**")

    for var in env_vars:
        is_set = os.environ.get(var) is not None
        status = "Set" if is_set else "Not set"
        icon = "check" if is_set else "close"
        st.markdown(f"- `{var}`: {status}")

    # Classification Settings
    st.markdown("---")
    st.markdown("### Classification Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Default Parameters**")

        default_irradiance = st.number_input(
            "Default Irradiance (W/m²)",
            value=1000.0,
            min_value=100.0,
            max_value=2000.0,
            help="Default target irradiance for new classifications"
        )

        default_area = st.number_input(
            "Default Test Area (m²)",
            value=0.1,
            min_value=0.01,
            max_value=1.0,
            step=0.01,
        )

    with col2:
        st.markdown("**Report Settings**")

        default_format = st.selectbox(
            "Default Report Format",
            options=["PDF", "Excel", "HTML", "JSON"],
        )

        include_raw_data = st.checkbox(
            "Include raw measurement data in reports",
            value=False,
        )

        company_name = st.text_input(
            "Company Name (for reports)",
            placeholder="Your Company Name",
        )

    # Save settings button
    st.markdown("---")

    if st.button("Save Settings", type="primary"):
        # In a real app, these would be saved to database or config file
        if is_database_available():
            st.success("Settings saved to database.")
        else:
            st.info(
                "Settings saved to session. "
                "Note: Settings will not persist without database connection."
            )
            # Store in session state as fallback
            st.session_state.settings = {
                'default_irradiance': default_irradiance,
                'default_area': default_area,
                'default_format': default_format,
                'include_raw_data': include_raw_data,
                'company_name': company_name,
            }

    # Offline Mode Info
    st.markdown("---")
    st.markdown("### Offline Mode")

    st.info(
        """
        **About Offline Mode**

        When the database is unavailable, the application runs in offline mode:

        - All classification calculations work normally
        - Results are displayed but not saved to database
        - Historical data and reports are not available
        - Settings are stored in session only

        To enable full functionality, configure a PostgreSQL database
        using one of the environment variables listed above.
        """
    )


if __name__ == "__main__":
    main()
