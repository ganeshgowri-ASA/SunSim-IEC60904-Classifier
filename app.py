"""
SunSim-IEC60904-Classifier
Main application entry point for the Sun Simulator Classification System.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SunSim IEC60904 Classifier",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #00D4AA, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-card {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2D3139;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #00D4AA;
    }
    .feature-icon {
        font-size: 36px;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 20px;
        font-weight: bold;
        color: #FAFAFA;
        margin-bottom: 10px;
    }
    .feature-desc {
        color: #888;
        font-size: 14px;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
    }
    .stat-box {
        text-align: center;
        padding: 20px;
    }
    .stat-value {
        font-size: 36px;
        font-weight: bold;
        color: #00D4AA;
    }
    .stat-label {
        color: #888;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">☀️ SunSim IEC60904 Classifier</h1>', unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: #888; font-size: 18px; margin-bottom: 40px;">
    Comprehensive Quality Control System for Solar Simulator Classification per IEC 60904-9
</p>
""", unsafe_allow_html=True)

# Quick stats
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">IEC 60904</div>
        <div class="stat-label">Standard Compliance</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">ISO 22514</div>
        <div class="stat-label">SPC Methods</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">AIAG MSA</div>
        <div class="stat-label">Gage R&R Standard</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">6σ</div>
        <div class="stat-label">Capability Target</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Feature cards
st.subheader("Quality Control Modules")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">SPC Analysis</div>
        <div class="feature-desc">
            Statistical Process Control with X-bar & R charts, run rules detection,
            and real-time process monitoring. ISO 22514 compliant control limits.
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Gage R&R</div>
        <div class="feature-desc">
            Measurement System Analysis per AIAG MSA manual. Evaluate repeatability,
            reproducibility, and calculate number of distinct categories (ndc).
        </div>
    </div>
    """, unsafe_allow_html=True)

with feat_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Capability Index</div>
        <div class="feature-desc">
            Process capability analysis with Cp, Cpk, Pp, Ppk gauges.
            Sigma level calculation and PPM defect rate estimation.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Navigation help
st.subheader("🚀 Getting Started")

st.markdown("""
Use the **sidebar navigation** to access the Quality Control modules:

1. **📈 SPC Analysis** - Monitor process stability with control charts
2. **🔬 MSA Gage R&R** - Validate your measurement system
3. **🎯 Capability Index** - Assess process capability vs specifications

Each module supports:
- Sample data generation for testing
- Database storage for historical analysis
- CSV import for your own data
- Interactive Plotly visualizations
""")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **SunSim-IEC60904-Classifier**

    Quality control system for solar simulator
    classification per IEC 60904-9 standard.

    *Phase 2: Quality Control Modules*
    """)

    st.markdown("---")
    st.markdown("### Standards")
    st.markdown("""
    - IEC 60904-9 Ed.3
    - ISO 22514
    - AIAG MSA 4th Ed.
    """)
