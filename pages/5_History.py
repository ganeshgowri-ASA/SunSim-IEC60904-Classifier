"""
Classification History Page

View and analyze historical classification results.
This page requires database connectivity for full functionality.
Gracefully degrades when database is unavailable.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import utilities
from utils.config import get_config

st.set_page_config(
    page_title="History | SunSim Classifier",
    page_icon="file_cabinet",
    layout="wide"
)

st.title("Classification History")
st.markdown("#### Historical Results & Trend Analysis")


def check_database_status():
    """Check database availability and return status info."""
    try:
        from database import is_database_configured, get_connection_status

        if not is_database_configured():
            return {
                "available": False,
                "reason": "not_configured",
                "message": "Database not configured. Set DATABASE_URL to enable history tracking."
            }

        status = get_connection_status()
        if status["connected"]:
            return {
                "available": True,
                "reason": "connected",
                "message": "Connected to database"
            }
        elif status["initialization_attempted"] and status["last_error"]:
            return {
                "available": False,
                "reason": "connection_failed",
                "message": f"Connection failed: {status['last_error']}"
            }
        else:
            # Try to connect
            try:
                from database import get_database_connection
                with get_database_connection() as conn:
                    return {
                        "available": True,
                        "reason": "connected",
                        "message": "Connected to database"
                    }
            except Exception as e:
                return {
                    "available": False,
                    "reason": "connection_failed",
                    "message": f"Connection failed: {str(e)}"
                }
    except Exception as e:
        return {
            "available": False,
            "reason": "error",
            "message": f"Error checking database: {str(e)}"
        }


def load_history_from_database():
    """Load classification history from database."""
    try:
        from database import get_database_connection, init_schema

        # Ensure schema exists
        init_schema()

        with get_database_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id, created_at, simulator_id,
                        spectral_class, uniformity_class, temporal_class,
                        overall_class, notes
                    FROM classification_results
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
                rows = cur.fetchall()

                if not rows:
                    return None

                df = pd.DataFrame(rows, columns=[
                    "ID", "Date", "Simulator ID",
                    "Spectral", "Uniformity", "Temporal",
                    "Overall", "Notes"
                ])
                return df
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return None


def generate_demo_history():
    """Generate demo history data for display when database unavailable."""
    import numpy as np
    np.random.seed(42)

    dates = pd.date_range(
        end=datetime.now(),
        periods=20,
        freq='D'
    )

    classes = ["A", "B", "C"]
    weights = [0.6, 0.3, 0.1]

    data = []
    for i, date in enumerate(dates):
        spectral = np.random.choice(classes, p=weights)
        uniformity = np.random.choice(classes, p=weights)
        temporal = np.random.choice(classes, p=weights)
        overall = f"{spectral}{uniformity}{temporal}"

        data.append({
            "ID": i + 1,
            "Date": date,
            "Simulator ID": f"SunSim-{np.random.choice(['001', '002', '003'])}",
            "Spectral": spectral,
            "Uniformity": uniformity,
            "Temporal": temporal,
            "Overall": overall,
            "Notes": "Demo data"
        })

    return pd.DataFrame(data)


# Check database status
db_status = check_database_status()

# Display status
if db_status["available"]:
    st.success("Connected to database")
else:
    st.warning(db_status["message"])
    st.info("Showing demo data for preview. Configure database to store actual results.")

st.markdown("---")

# Load or generate data
if db_status["available"]:
    history_df = load_history_from_database()
    if history_df is None:
        st.info("No classification records found in database.")
        st.markdown("Run classifications and save them to populate history.")
        history_df = generate_demo_history()
        st.caption("Showing demo data for preview")
else:
    history_df = generate_demo_history()

# Filters
st.markdown("### Filters")

col1, col2, col3 = st.columns(3)

with col1:
    simulator_filter = st.multiselect(
        "Simulator ID",
        options=history_df["Simulator ID"].unique(),
        default=[]
    )

with col2:
    class_filter = st.multiselect(
        "Overall Class",
        options=sorted(history_df["Overall"].unique()),
        default=[]
    )

with col3:
    date_range = st.date_input(
        "Date Range",
        value=(
            history_df["Date"].min().date() if len(history_df) > 0 else datetime.now().date(),
            datetime.now().date()
        )
    )

# Apply filters
filtered_df = history_df.copy()

if simulator_filter:
    filtered_df = filtered_df[filtered_df["Simulator ID"].isin(simulator_filter)]

if class_filter:
    filtered_df = filtered_df[filtered_df["Overall"].isin(class_filter)]

if date_range and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df["Date"]).dt.date >= start_date) &
        (pd.to_datetime(filtered_df["Date"]).dt.date <= end_date)
    ]

st.markdown("---")

# Display results
st.markdown(f"### Classification Records ({len(filtered_df)} results)")


def color_class(val):
    """Color code classification cells."""
    if val == "A" or (isinstance(val, str) and val.count("A") == 3):
        return "background-color: #90EE90"
    elif val == "B" or (isinstance(val, str) and "C" not in val):
        return "background-color: #FFD700"
    elif val == "C" or (isinstance(val, str) and "C" in val):
        return "background-color: #FFB347"
    return ""


styled_df = filtered_df.style.applymap(
    color_class,
    subset=["Spectral", "Uniformity", "Temporal", "Overall"]
)

st.dataframe(styled_df, use_container_width=True)

st.markdown("---")

# Statistics
st.markdown("### Classification Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_records = len(filtered_df)
    st.metric("Total Records", total_records)

with col2:
    if total_records > 0:
        aaa_count = len(filtered_df[filtered_df["Overall"] == "AAA"])
        aaa_pct = (aaa_count / total_records) * 100
        st.metric("AAA Rate", f"{aaa_pct:.1f}%")
    else:
        st.metric("AAA Rate", "N/A")

with col3:
    if total_records > 0:
        # Most common classification
        most_common = filtered_df["Overall"].mode()
        if len(most_common) > 0:
            st.metric("Most Common", most_common.iloc[0])
        else:
            st.metric("Most Common", "N/A")
    else:
        st.metric("Most Common", "N/A")

with col4:
    unique_sims = filtered_df["Simulator ID"].nunique()
    st.metric("Unique Simulators", unique_sims)

# Class distribution
st.markdown("### Classification Distribution")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Overall Class Distribution**")
    if len(filtered_df) > 0:
        class_counts = filtered_df["Overall"].value_counts()
        st.bar_chart(class_counts)

with col2:
    st.markdown("**Individual Class Distribution**")
    if len(filtered_df) > 0:
        individual_counts = pd.DataFrame({
            "Spectral": filtered_df["Spectral"].value_counts(),
            "Uniformity": filtered_df["Uniformity"].value_counts(),
            "Temporal": filtered_df["Temporal"].value_counts()
        }).fillna(0)
        st.bar_chart(individual_counts)

# Trend over time
st.markdown("### Trend Analysis")

if len(filtered_df) > 1:
    # Convert class to numeric for trending
    class_map = {"A": 3, "B": 2, "C": 1}

    trend_df = filtered_df.copy()
    trend_df["Date"] = pd.to_datetime(trend_df["Date"])
    trend_df = trend_df.sort_values("Date")

    trend_df["Spectral_Score"] = trend_df["Spectral"].map(class_map)
    trend_df["Uniformity_Score"] = trend_df["Uniformity"].map(class_map)
    trend_df["Temporal_Score"] = trend_df["Temporal"].map(class_map)

    chart_data = trend_df.set_index("Date")[["Spectral_Score", "Uniformity_Score", "Temporal_Score"]]
    chart_data.columns = ["Spectral", "Uniformity", "Temporal"]

    st.line_chart(chart_data)
    st.caption("Score: A=3, B=2, C=1. Higher is better.")
else:
    st.info("Need at least 2 records for trend analysis")

# Export
st.markdown("---")
st.markdown("### Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"classification_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    st.button("Refresh Data", on_click=lambda: st.rerun())

# Sidebar
with st.sidebar:
    st.markdown("### Database Status")

    if db_status["available"]:
        st.success("Connected")
    else:
        st.warning("Not Connected")
        st.caption(db_status["reason"])

    st.markdown("---")
    st.markdown("### Quick Stats")

    if len(history_df) > 0:
        st.metric("Total Records", len(history_df))
        st.metric("Unique Simulators", history_df["Simulator ID"].nunique())

        # Latest classification
        latest = history_df.iloc[0]
        st.markdown(f"**Latest:** {latest['Overall']}")
        st.caption(f"{latest['Simulator ID']}")
