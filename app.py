"""
SunSim IEC 60904-9 Classification System
=========================================

Professional Sun Simulator Classification System
IEC 60904-9 Ed.3 Compliant | ISO 17025 Report Generation

Main Application Entry Point
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SunSim IEC 60904-9 Classifier",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "IEC 60904-9 Ed.3 Solar Simulator Classification System"
    }
)

# Custom CSS for professional styling
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        min-width: 80px;
    }

    .grade-a-plus {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }

    .grade-a {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }

    .grade-b {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }

    .grade-c {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }

    .nav-link {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        color: #475569;
        transition: all 0.2s;
        margin-bottom: 0.5rem;
    }

    .nav-link:hover {
        background: #F1F5F9;
        color: #1E3A5F;
    }

    .nav-link.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: #94A3B8;
        font-size: 0.875rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .css-1d391kg {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2D4A6F 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    .standard-ref {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #6366F1;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }

    .standard-ref-title {
        font-weight: 600;
        color: #4338CA;
        margin-bottom: 0.5rem;
    }

    .threshold-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }

    .threshold-table th, .threshold-table td {
        padding: 0.75rem;
        text-align: center;
        border-bottom: 1px solid #E2E8F0;
    }

    .threshold-table th {
        background: #F8FAFC;
        font-weight: 600;
        color: #1E3A5F;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Main application page - Welcome and Navigation"""

    # Header
    st.markdown(
        '<h1 class="main-header">SunSim IEC 60904-9 Classifier</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Professional Sun Simulator Classification System | '
        'IEC 60904-9 Ed.3 Compliant</p>',
        unsafe_allow_html=True
    )

    # Standard reference
    st.markdown(
        '<div class="standard-ref">'
        '<div class="standard-ref-title">IEC 60904-9:2020 (Edition 3)</div>'
        '<div>Photovoltaic devices - Part 9: Classification of solar simulator characteristics</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Overview columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
### Spectral Match (SPD)
Measures how closely the simulator's spectral distribution matches
the AM1.5G reference spectrum across wavelength intervals.

**Wavelength Range:** 300-1200nm (Ed.3)

| Grade | Ratio Range |
|-------|-------------|
| A+ | 0.875 - 1.125 |
| A | 0.75 - 1.25 |
| B | 0.6 - 1.4 |
| C | 0.4 - 2.0 |
""")

    with col2:
        st.markdown("""
### Non-Uniformity
Measures spatial uniformity of irradiance across the test plane.

**Formula:** ((Max - Min) / (Max + Min)) x 100%

| Grade | Max Non-Uniformity |
|-------|-------------------|
| A+ | <= 1% |
| A | <= 2% |
| B | <= 5% |
| C | <= 10% |
""")

    with col3:
        st.markdown("""
### Temporal Stability
Measures irradiance stability over time (STI & LTI).

**STI:** Short-term instability
**LTI:** Long-term instability

| Grade | STI | LTI |
|-------|-----|-----|
| A+ | <= 0.5% | <= 1% |
| A | <= 2% | <= 2% |
| B | <= 5% | <= 5% |
| C | <= 10% | <= 10% |
""")

    st.divider()

    # Navigation cards
    st.markdown("### Quick Navigation")

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

    with nav_col1:
        st.page_link(
            "pages/1_Classification_Dashboard.py",
            label="Classification Dashboard",
            icon="📊",
            use_container_width=True
        )

    with nav_col2:
        st.page_link(
            "pages/2_Spectral_Match.py",
            label="Spectral Match Analysis",
            icon="📈",
            use_container_width=True
        )

    with nav_col3:
        st.page_link(
            "pages/3_Uniformity.py",
            label="Uniformity Analysis",
            icon="🔲",
            use_container_width=True
        )

    with nav_col4:
        st.page_link(
            "pages/4_Temporal_Stability.py",
            label="Temporal Stability",
            icon="⏱️",
            use_container_width=True
        )

    # Footer
    st.markdown(
        '<div class="footer">'
        '<p>IEC 60904-9 Ed.3 Solar Simulator Classification System</p>'
        '<p>Compliant with ISO 17025 Laboratory Accreditation Requirements</p>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
