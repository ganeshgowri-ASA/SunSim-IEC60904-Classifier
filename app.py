"""
SunSim IEC 60904-9 Classification System

Professional Sun Simulator Classification System
IEC 60904-9 Ed.3 Compliant | ISO 17025 Report Generation

Main Application Entry Point
SunSim-IEC60904-Classifier
Main application entry point for the Sun Simulator Classification System.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SunSim IEC 60904-9 Classifier",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>%E2%98%80%EF%B8%8F</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "IEC 60904-9 Ed.3 Solar Simulator Classification System"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
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

    /* Card styling */
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

    /* Grade badge styling */
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

    /* Navigation styling */
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

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94A3B8;
        font-size: 0.875rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2D4A6F 100%);
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    /* Standard reference box */
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

    /* Threshold table */
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
    page_title="SunSim IEC60904 Classifier",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;   
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
