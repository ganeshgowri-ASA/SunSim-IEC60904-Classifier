"""
Sun Simulator Classification System
IEC 60904-9:2020 Ed.3 Compliance

Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

from config import (
    APP_CONFIG, THEME, BADGE_COLORS, BADGE_COLORS_LIGHT,
    WAVELENGTH_BANDS, CLASSIFICATION, get_overall_classification
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title=APP_CONFIG['title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state=APP_CONFIG['initial_sidebar_state']
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for professional dark theme."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root Variables */
    :root {
        --sidebar-bg: #1e293b;
        --sidebar-text: #e2e8f0;
        --primary: #3b82f6;
        --secondary: #10b981;
        --background: #0f172a;
        --surface: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border: #334155;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
    }

    /* Main App Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #1e293b !important;
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #1e293b !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .main-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Classification Badge Styling */
    .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        min-width: 60px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .badge-aplus {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: 2px solid #34d399;
    }

    .badge-a {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: 2px solid #60a5fa;
    }

    .badge-b {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        border: 2px solid #fbbf24;
    }

    .badge-c {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: 2px solid #f87171;
    }

    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }

    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }

    .metric-delta.positive {
        color: #10b981;
    }

    .metric-delta.negative {
        color: #ef4444;
    }

    /* Info Card Styling */
    .info-card {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .info-card h4 {
        color: #3b82f6;
        margin-bottom: 0.5rem;
    }

    .info-card p {
        color: #94a3b8;
        margin: 0;
    }

    /* Table Styling */
    .dataframe {
        background: #1e293b !important;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background: #334155 !important;
        color: #f8fafc !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
    }

    .dataframe td {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e293b;
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        background: #3b82f6;
        color: white;
    }

    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: #1e293b;
        border-color: #475569;
        color: #f8fafc;
    }

    /* File Uploader Styling */
    .stFileUploader > div {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 12px;
    }

    .stFileUploader > div:hover {
        border-color: #3b82f6;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
    }

    /* Quick Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    /* Feature Card */
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        border-color: #3b82f6;
    }

    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
    }

    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #334155;
        margin: 2rem 0;
    }

    </style>
    """, unsafe_allow_html=True)


def create_badge_html(classification: str) -> str:
    """Create HTML for classification badge."""
    badge_class = {
        'A+': 'badge-aplus',
        'A': 'badge-a',
        'B': 'badge-b',
        'C': 'badge-c'
    }.get(classification, 'badge-c')

    return f'<span class="badge {badge_class}">{classification}</span>'


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #f8fafc; margin-bottom: 0.5rem;">☀️ SunSim</h2>
            <p style="color: #94a3b8; font-size: 0.85rem;">IEC 60904-9 Classifier</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="color: #94a3b8; font-size: 0.85rem;">
            <p><strong style="color: #f8fafc;">Version:</strong> 1.0.0</p>
            <p><strong style="color: #f8fafc;">Standard:</strong> IEC 60904-9:2020 Ed.3</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
            <p><strong style="color: #10b981;">A+</strong> - Excellent</p>
            <p><strong style="color: #3b82f6;">A</strong> - Good</p>
            <p><strong style="color: #f59e0b;">B</strong> - Acceptable</p>
            <p><strong style="color: #ef4444;">C</strong> - Needs Improvement</p>
        </div>
        """, unsafe_allow_html=True)

    # Main Content - Welcome Dashboard
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            ☀️ Sun Simulator Classification System
        </div>
        <div class="main-subtitle">
            Professional solar simulator performance classification per IEC 60904-9:2020 Ed.3
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Overview Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Simulators Tested</div>
            <div class="metric-value">24</div>
            <div class="metric-delta positive">+3 this month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">A+ Classifications</div>
            <div class="metric-value">8</div>
            <div class="metric-delta positive">33% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Measurements</div>
            <div class="metric-value">156</div>
            <div class="metric-delta positive">+12 this week</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Compliance Rate</div>
            <div class="metric-value">96%</div>
            <div class="metric-delta positive">+2% vs last quarter</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    st.markdown("### Quick Access")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Dashboard</div>
            <div class="feature-description">
                View overall classification results and performance trends
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌈</div>
            <div class="feature-title">Spectral Analysis</div>
            <div class="feature-description">
                Analyze spectral match across 6 wavelength bands
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🗺️</div>
            <div class="feature-title">Uniformity</div>
            <div class="feature-description">
                Measure spatial irradiance uniformity with heatmaps
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⏱️</div>
            <div class="feature-title">Temporal Stability</div>
            <div class="feature-description">
                Evaluate STI and LTI temporal performance
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # IEC 60904-9 Information
    st.markdown("### IEC 60904-9:2020 Ed.3 Classification Criteria")

    col1, col2 = st.columns(2)

    with col1:
        # Classification limits table
        limits_data = {
            'Parameter': ['Spectral Match', 'Spatial Uniformity', 'Temporal STI', 'Temporal LTI'],
            'A+ Limit': ['±12.5%', '≤1%', '≤0.5%', '≤1%'],
            'A Limit': ['±25%', '≤2%', '≤2%', '≤2%'],
            'B Limit': ['±40%', '≤5%', '≤5%', '≤5%'],
            'C Limit': ['>40%', '>5%', '>5%', '>5%']
        }
        df_limits = pd.DataFrame(limits_data)

        st.markdown("#### Classification Limits")
        st.dataframe(df_limits, use_container_width=True, hide_index=True)

    with col2:
        # Wavelength bands table
        bands_data = {
            'Band': [f"Band {i+1}" for i in range(len(WAVELENGTH_BANDS))],
            'Range (nm)': [f"{b[0]}-{b[1]}" for b in WAVELENGTH_BANDS],
            'Description': [b[2] for b in WAVELENGTH_BANDS]
        }
        df_bands = pd.DataFrame(bands_data)

        st.markdown("#### Spectral Wavelength Bands")
        st.dataframe(df_bands, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample Classification Chart
    st.markdown("### Recent Classification Trends")

    # Generate sample trend data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    trend_data = {
        'Date': dates,
        'Spectral': np.random.choice(['A+', 'A', 'A', 'A+', 'A'], 30),
        'Uniformity': np.random.choice(['A+', 'A+', 'A', 'A+', 'A'], 30),
        'STI': np.random.choice(['A+', 'A', 'A+', 'A+', 'A'], 30),
        'LTI': np.random.choice(['A+', 'A', 'A', 'A+', 'B'], 30)
    }

    # Create a summary chart
    classification_counts = {
        'A+': [8, 10, 12, 9],
        'A': [12, 10, 8, 11],
        'B': [3, 3, 3, 3],
        'C': [1, 1, 1, 1]
    }
    categories = ['Spectral', 'Uniformity', 'STI', 'LTI']

    fig = go.Figure()

    for classification, counts in classification_counts.items():
        fig.add_trace(go.Bar(
            name=classification,
            x=categories,
            y=counts,
            marker_color=BADGE_COLORS[classification],
            text=counts,
            textposition='inside',
            textfont=dict(color='white', size=14, family='Inter')
        ))

    fig.update_layout(
        barmode='stack',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#e2e8f0'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(71, 85, 105, 0.5)',
            title='Number of Measurements'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Info Section
    st.markdown("""
    <div class="info-card">
        <h4>About This System</h4>
        <p>
            This Sun Simulator Classification System provides comprehensive analysis and classification
            of solar simulator performance according to IEC 60904-9:2020 Ed.3 standards. The system
            evaluates spectral match, spatial uniformity, and temporal stability to provide
            accurate A+/A/B/C classifications for your solar simulators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Sun Simulator Classification System v1.0.0 | IEC 60904-9:2020 Ed.3 Compliant<br>
        <span style="color: #475569;">© 2024 PV Testing Laboratory</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
