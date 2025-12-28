"""
Reports Page - Classification History and Report Generation

View past classifications and generate ISO 17025 compliant reports.
Gracefully handles database unavailability.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from utils.db import (
    is_database_available,
    execute_query_safe,
    show_database_status,
)

st.set_page_config(
    page_title="Reports - SunSim",
    page_icon="description",
    layout="wide",
)


def load_classification_history(
    days: int = 30,
    simulator_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load classification history from database.

    Returns empty list if database is unavailable.

    Args:
        days: Number of days to look back
        simulator_filter: Optional filter by simulator name

    Returns:
        List of classification records
    """
    if not is_database_available():
        return []

    query = """
        SELECT id, simulator_name, classification_date,
               spectral_class, uniformity_class, stability_class,
               overall_class
        FROM classifications
        WHERE classification_date >= :start_date
    """

    params = {'start_date': datetime.now() - timedelta(days=days)}

    if simulator_filter:
        query += " AND simulator_name ILIKE :simulator"
        params['simulator'] = f"%{simulator_filter}%"

    query += " ORDER BY classification_date DESC"

    result = execute_query_safe(query, params, default=[], show_warning=False)

    if not result:
        return []

    return [
        {
            'id': row[0],
            'simulator_name': row[1],
            'date': row[2],
            'spectral': row[3],
            'uniformity': row[4],
            'stability': row[5],
            'overall': row[6],
        }
        for row in result
    ]


def load_simulators_list() -> List[str]:
    """
    Load list of unique simulator names from database.

    Returns empty list if database unavailable.
    """
    if not is_database_available():
        return []

    result = execute_query_safe(
        "SELECT DISTINCT simulator_name FROM classifications ORDER BY simulator_name",
        default=[],
        show_warning=False
    )

    return [row[0] for row in result] if result else []


def generate_sample_report_data() -> Dict[str, Any]:
    """
    Generate sample report data for demo purposes.

    Used when database is unavailable.
    """
    return {
        'id': 'DEMO-001',
        'simulator_name': 'Demo Simulator',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'spectral': 'A',
        'uniformity': 'A',
        'stability': 'A+',
        'overall': 'A',
        'notes': 'This is a demonstration report generated in offline mode.',
    }


def main():
    """Reports page main function."""

    with st.sidebar:
        st.title("Reports")
        show_database_status()

    st.title("Classification Reports")
    st.markdown("View history and generate ISO 17025 compliant reports")

    # Check database availability with error handling
    try:
        db_available = is_database_available()
    except Exception as e:
        db_available = False
        st.error(f"Error checking database availability: {e}")

    if not db_available:
        st.warning(
            "Database is not available. Historical data cannot be loaded. "
            "You can still generate reports from new classifications.",
            icon="exclamation-triangle"
        )

    # Filters
    st.markdown("### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        date_range = st.selectbox(
            "Date Range",
            options=["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            index=1,
        )

        days_map = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "All time": 3650,
        }

    with col2:
        try:
            simulators = load_simulators_list()
        except Exception:
            simulators = []
        simulator_filter = st.selectbox(
            "Simulator",
            options=["All"] + simulators,
            disabled=not db_available,
        )

    with col3:
        class_filter = st.selectbox(
            "Classification",
            options=["All", "A+", "A", "B", "C"],
        )

    st.markdown("---")

    # Load and display history
    st.markdown("### Classification History")

    if db_available:
        try:
            history = load_classification_history(
                days=days_map[date_range],
                simulator_filter=simulator_filter if simulator_filter != "All" else None,
            )
        except Exception as e:
            history = []
            st.error(f"Error loading classification history: {e}")

        if class_filter != "All":
            history = [h for h in history if h['overall'] == class_filter]

        if history:
            # Display as table
            st.dataframe(
                [
                    {
                        'ID': h['id'],
                        'Simulator': h['simulator_name'],
                        'Date': h['date'],
                        'Spectral': h['spectral'],
                        'Uniformity': h['uniformity'],
                        'Stability': h['stability'],
                        'Overall': h['overall'],
                    }
                    for h in history
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Report generation for selected classification
            st.markdown("---")
            st.markdown("### Generate Report")

            selected_id = st.selectbox(
                "Select Classification",
                options=[h['id'] for h in history],
                format_func=lambda x: f"#{x} - {next(h['simulator_name'] for h in history if h['id'] == x)}",
            )

            col1, col2 = st.columns(2)

            with col1:
                report_format = st.selectbox(
                    "Format",
                    options=["PDF", "Excel", "JSON", "HTML"],
                )

            with col2:
                include_details = st.checkbox(
                    "Include detailed measurement data",
                    value=True,
                )

            if st.button("Generate Report", type="primary"):
                st.info(
                    f"Report generation for classification #{selected_id} "
                    f"in {report_format} format would be triggered here."
                )

        else:
            st.info("No classifications found matching the selected filters.")

    else:
        # Offline mode - show demo functionality
        st.info(
            "Database unavailable. Showing demonstration mode. "
            "Connect to database to view actual classification history."
        )

        # Demo data
        demo_data = generate_sample_report_data()

        st.markdown("#### Demo Classification")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spectral", demo_data['spectral'])
        col2.metric("Uniformity", demo_data['uniformity'])
        col3.metric("Stability", demo_data['stability'])
        col4.metric("Overall", demo_data['overall'])

        st.markdown("---")
        st.markdown("### Generate Demo Report")

        report_format = st.selectbox(
            "Format",
            options=["PDF", "Excel", "JSON", "HTML"],
        )

        if st.button("Generate Demo Report", type="primary"):
            st.success("Demo report generated successfully!")
            st.json(demo_data)

    # Export section
    st.markdown("---")
    st.markdown("### Bulk Export")

    st.markdown(
        "Export all classifications within the selected date range."
    )

    export_format = st.radio(
        "Export Format",
        options=["CSV", "Excel", "JSON"],
        horizontal=True,
    )

    if st.button("Export All", disabled=not db_available):
        if db_available:
            st.info(
                f"Exporting {len(history) if 'history' in dir() else 0} "
                f"classifications to {export_format}..."
            )
        else:
            st.warning("Export requires database connection.")


if __name__ == "__main__":
    main()
