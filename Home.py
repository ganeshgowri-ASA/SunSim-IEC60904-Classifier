"""
SunSim-IEC60904-Classifier - Home Page

Professional Sun Simulator Classification System
IEC 60904-9 Ed.3 Compliant

Features:
- Spectral Match Analysis
- Uniformity Assessment
- Temporal Stability Analysis
- ISO 17025 Report Generation
- SPC/MSA Quality Control
- Lamp Monitoring & Drift Analysis
"""

import streamlit as st
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.database import (
    init_database,
    get_session,
    get_active_lamps,
    get_lamps_needing_calibration,
    get_lamps_at_warning_threshold,
)

# Page configuration
st.set_page_config(
    page_title="SunSim Classifier | IEC 60904-9",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .status-card {
        background-color: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
        text-align: center;
    }
    .status-good { border-left: 4px solid #28a745; }
    .status-warning { border-left: 4px solid #ffc107; }
    .status-critical { border-left: 4px solid #dc3545; }
    .iec-badge {
        background-color: #1f77b4;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .version-info {
        color: #888;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">SunSim-IEC60904-Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Sun Simulator Classification System</p>', unsafe_allow_html=True)

    # Compliance badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <span class="iec-badge">IEC 60904-9 Ed.3 Compliant</span>
            <span class="iec-badge" style="background-color: #28a745; margin-left: 10px;">ISO 17025 Ready</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Initialize database
    engine = init_database()
    session = get_session(engine)

    # System status overview
    st.subheader("System Status")

    active_lamps = get_active_lamps(session)
    lamps_cal_due = get_lamps_needing_calibration(session, days_warning=30)
    lamps_warning = get_lamps_at_warning_threshold(session)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_class = "status-good" if len(active_lamps) > 0 else "status-warning"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <h3>{len(active_lamps)}</h3>
            <p>Active Lamps</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status_class = "status-warning" if len(lamps_cal_due) > 0 else "status-good"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <h3>{len(lamps_cal_due)}</h3>
            <p>Calibration Due</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status_class = "status-critical" if len(lamps_warning) > 0 else "status-good"
        st.markdown(f"""
        <div class="status-card {status_class}">
            <h3>{len(lamps_warning)}</h3>
            <p>Aging Warnings</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="status-card status-good">
            <h3>{datetime.now().strftime('%H:%M')}</h3>
            <p>{datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Feature cards
    st.subheader("Monitoring Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Lamp Monitor</h4>
            <p>Comprehensive lamp tracking with flash counter, operating hours,
            calibration alerts, and aging warnings.</p>
            <ul>
                <li>Flash count tracking</li>
                <li>Calibration due dates</li>
                <li>Repeatability monitoring (0.09%)</li>
                <li>Replacement history</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>Spectrum Drift</h4>
            <p>UV/NIR degradation monitoring per TÜV paper findings with
            trend analysis and forecasting.</p>
            <ul>
                <li>UV degradation (Xenon aging)</li>
                <li>Blue-shift tracking</li>
                <li>Power adjustment effects</li>
                <li>Manufacturer comparison</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>Repeatability</h4>
            <p>Flash-to-flash consistency analysis with SPC control charts
            and process capability metrics.</p>
            <ul>
                <li>Control charts (X-bar, R)</li>
                <li>Western Electric rules</li>
                <li>Process capability (Cp, Cpk)</li>
                <li>Trend analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick navigation
    st.divider()
    st.subheader("Quick Navigation")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Go to Lamp Monitor", use_container_width=True):
            st.switch_page("pages/8_💡_Lamp_Monitor.py")

    with col2:
        if st.button("Go to Spectrum Drift", use_container_width=True):
            st.switch_page("pages/9_📉_Spectrum_Drift.py")

    with col3:
        if st.button("Go to Repeatability", use_container_width=True):
            st.switch_page("pages/10_🔁_Repeatability.py")

    # IEC 60904-9 Quick Reference
    st.divider()

    with st.expander("IEC 60904-9 Ed.3 Quick Reference"):
        st.markdown("""
        ### Spectral Match Classification

        | Class | Spectral Match Limits |
        |-------|----------------------|
        | A+ | 0.875 - 1.125 (±12.5%) |
        | A | 0.75 - 1.25 (±25%) |
        | B | 0.6 - 1.4 (±40%) |
        | C | 0.4 - 2.0 (>±40%) |

        ### Temporal Stability Requirements

        | Class | Short-term Instability | Long-term Instability |
        |-------|----------------------|----------------------|
        | A | ≤ 0.5% | ≤ 2% |
        | B | ≤ 2% | ≤ 5% |
        | C | ≤ 10% | ≤ 10% |

        ### Non-Uniformity Requirements

        | Class | Non-Uniformity |
        |-------|---------------|
        | A | ≤ 2% |
        | B | ≤ 5% |
        | C | ≤ 10% |

        ### Repeatability Target
        - Target: **0.09%** coefficient of variation for flash-to-flash repeatability

        ### Wavelength Range
        - 300nm to 1100nm (100nm intervals for spectral match)
        """)

    # About section
    with st.expander("About This System"):
        st.markdown("""
        ### SunSim-IEC60904-Classifier

        A professional-grade sun simulator classification and monitoring system designed
        for photovoltaic testing laboratories and calibration facilities.

        **Key Features:**
        - Full IEC 60904-9 Ed.3 compliance
        - Real-time lamp monitoring and alerts
        - Spectral drift analysis per TÜV research findings
        - Statistical process control (SPC) for repeatability
        - ISO 17025 compatible reporting

        **Standards Compliance:**
        - IEC 60904-9:2020 (Ed.3) - Solar simulator performance requirements
        - IEC 60904-3 - Measurement principles for terrestrial PV solar devices
        - ISO 17025 - General requirements for testing and calibration laboratories

        **Technical References:**
        - TÜV Rheinland publications on solar simulator spectral stability
        - ASTM G173-03 - Reference solar spectral irradiances (AM1.5G)
        """)

    # Version info
    st.markdown("""
    <p class="version-info">
        SunSim-IEC60904-Classifier v3.0 | Phase 3: Monitoring Modules | © 2024
    </p>
    """, unsafe_allow_html=True)

    session.close()


if __name__ == "__main__":
    main()
