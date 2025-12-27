"""
Sun Simulator Classification System
Dashboard Page - Overall Classification Results

This page displays the overall classification dashboard with A+/A/B/C badges,
quick statistics, and recent measurement summaries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_CONFIG, THEME, BADGE_COLORS, BADGE_COLORS_LIGHT,
    WAVELENGTH_BANDS, get_overall_classification
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Dashboard - " + APP_CONFIG['title'],
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] {
    background: #1e293b !important;
}

.dashboard-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #475569;
}

.dashboard-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.dashboard-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
}

.stat-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}

.stat-value {
    font-size: 2.25rem;
    font-weight: 700;
    color: #f8fafc;
}

.stat-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

.classification-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 60px;
    height: 60px;
    border-radius: 12px;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
}

.badge-aplus { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
.badge-a { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
.badge-b { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
.badge-c { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }

.simulator-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.simulator-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f8fafc;
}

.simulator-model {
    color: #94a3b8;
    font-size: 0.85rem;
}

.class-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.class-aplus { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.class-a { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
.class-b { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.class-c { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_badge_class(classification: str) -> str:
    """Get CSS class for classification badge."""
    return {
        'A+': 'badge-aplus',
        'A': 'badge-a',
        'B': 'badge-b',
        'C': 'badge-c'
    }.get(classification, 'badge-c')


def get_pill_class(classification: str) -> str:
    """Get CSS class for classification pill."""
    return {
        'A+': 'class-aplus',
        'A': 'class-a',
        'B': 'class-b',
        'C': 'class-c'
    }.get(classification, 'class-c')


def generate_sample_data():
    """Generate sample data for demonstration."""
    simulators = [
        {
            'id': 1, 'name': 'SunSim Pro 3000', 'manufacturer': 'Solar Tech Inc.',
            'model': 'SSP-3000X', 'spectral': 'A+', 'uniformity': 'A',
            'sti': 'A+', 'lti': 'A', 'overall': 'A',
            'last_calibration': datetime.now() - timedelta(days=15)
        },
        {
            'id': 2, 'name': 'FlashTest 5000', 'manufacturer': 'PV Systems Ltd.',
            'model': 'FT-5000', 'spectral': 'A', 'uniformity': 'A+',
            'sti': 'A', 'lti': 'A+', 'overall': 'A',
            'last_calibration': datetime.now() - timedelta(days=30)
        },
        {
            'id': 3, 'name': 'XenoFlash Elite', 'manufacturer': 'Xenon Labs',
            'model': 'XFE-2000', 'spectral': 'A+', 'uniformity': 'A+',
            'sti': 'A+', 'lti': 'A+', 'overall': 'A+',
            'last_calibration': datetime.now() - timedelta(days=7)
        },
        {
            'id': 4, 'name': 'LED Solar Sim', 'manufacturer': 'LED Photonics',
            'model': 'LSS-1500', 'spectral': 'B', 'uniformity': 'A',
            'sti': 'A+', 'lti': 'A+', 'overall': 'B',
            'last_calibration': datetime.now() - timedelta(days=45)
        },
        {
            'id': 5, 'name': 'MultiSpec 4000', 'manufacturer': 'Spectral Dynamics',
            'model': 'MS-4000', 'spectral': 'A', 'uniformity': 'B',
            'sti': 'A', 'lti': 'A', 'overall': 'B',
            'last_calibration': datetime.now() - timedelta(days=60)
        },
    ]

    measurements = []
    for i in range(20):
        sim = random.choice(simulators)
        measurements.append({
            'id': i + 1,
            'simulator': sim['name'],
            'date': datetime.now() - timedelta(days=random.randint(0, 30)),
            'spectral': random.choice(['A+', 'A', 'A', 'A+', 'B']),
            'uniformity': random.choice(['A+', 'A', 'A+', 'A', 'B']),
            'sti': random.choice(['A+', 'A+', 'A', 'A+', 'A']),
            'lti': random.choice(['A+', 'A', 'A+', 'A', 'A']),
        })
        measurements[-1]['overall'] = get_overall_classification(
            measurements[-1]['spectral'],
            measurements[-1]['uniformity'],
            measurements[-1]['sti'],
            measurements[-1]['lti']
        )

    return simulators, measurements


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">📊 Classification Dashboard</div>
        <div class="dashboard-subtitle">
            Overview of sun simulator classifications and performance metrics
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load sample data
    simulators, measurements = generate_sample_data()

    # Classification Summary Cards
    st.markdown('<div class="section-title">Classification Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Count classifications
    class_counts = {'A+': 0, 'A': 0, 'B': 0, 'C': 0}
    for sim in simulators:
        class_counts[sim['overall']] = class_counts.get(sim['overall'], 0) + 1

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="classification-badge badge-aplus">A+</div>
            <div class="stat-value" style="color: #10b981;">{class_counts['A+']}</div>
            <div class="stat-label">Excellent</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="classification-badge badge-a">A</div>
            <div class="stat-value" style="color: #3b82f6;">{class_counts['A']}</div>
            <div class="stat-label">Good</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="classification-badge badge-b">B</div>
            <div class="stat-value" style="color: #f59e0b;">{class_counts['B']}</div>
            <div class="stat-label">Acceptable</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="classification-badge badge-c">C</div>
            <div class="stat-value" style="color: #ef4444;">{class_counts['C']}</div>
            <div class="stat-label">Needs Improvement</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick Stats
    st.markdown('<div class="section-title">Quick Statistics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Simulators", len(simulators))

    with col2:
        st.metric("Total Measurements", len(measurements))

    with col3:
        recent = len([m for m in measurements if m['date'] > datetime.now() - timedelta(days=7)])
        st.metric("This Week", recent, delta=f"+{recent}")

    with col4:
        aplus_percent = (class_counts['A+'] / len(simulators) * 100) if simulators else 0
        st.metric("A+ Rate", f"{aplus_percent:.0f}%")

    with col5:
        compliance = ((class_counts['A+'] + class_counts['A']) / len(simulators) * 100) if simulators else 0
        st.metric("Compliance", f"{compliance:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Classification Distribution</div>', unsafe_allow_html=True)

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['A+', 'A', 'B', 'C'],
            values=[class_counts['A+'], class_counts['A'], class_counts['B'], class_counts['C']],
            hole=0.5,
            marker=dict(colors=[BADGE_COLORS['A+'], BADGE_COLORS['A'],
                               BADGE_COLORS['B'], BADGE_COLORS['C']]),
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])

        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            annotations=[dict(
                text=f'{len(simulators)}<br>Total',
                x=0.5, y=0.5,
                font=dict(size=24, color='#f8fafc', family='Inter'),
                showarrow=False
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Parameter Breakdown</div>', unsafe_allow_html=True)

        # Stacked bar chart by parameter
        params = ['Spectral', 'Uniformity', 'STI', 'LTI']
        param_counts = {
            'A+': [0, 0, 0, 0],
            'A': [0, 0, 0, 0],
            'B': [0, 0, 0, 0],
            'C': [0, 0, 0, 0]
        }

        for sim in simulators:
            param_counts[sim['spectral']][0] += 1
            param_counts[sim['uniformity']][1] += 1
            param_counts[sim['sti']][2] += 1
            param_counts[sim['lti']][3] += 1

        fig = go.Figure()

        for grade in ['A+', 'A', 'B', 'C']:
            fig.add_trace(go.Bar(
                name=grade,
                x=params,
                y=param_counts[grade],
                marker_color=BADGE_COLORS[grade],
                text=param_counts[grade],
                textposition='inside',
                textfont=dict(color='white', size=12)
            ))

        fig.update_layout(
            barmode='stack',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(71,85,105,0.3)')
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Simulators List
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-title">Registered Simulators</div>', unsafe_allow_html=True)

        for sim in simulators:
            overall_class = sim['overall']
            badge_class = get_badge_class(overall_class)

            st.markdown(f"""
            <div class="simulator-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="simulator-name">{sim['name']}</div>
                        <div class="simulator-model">{sim['manufacturer']} - {sim['model']}</div>
                    </div>
                    <div class="classification-badge {badge_class}" style="width: 50px; height: 50px; font-size: 1.25rem;">
                        {overall_class}
                    </div>
                </div>
                <div style="margin-top: 0.75rem;">
                    <span class="class-pill {get_pill_class(sim['spectral'])}">Spectral: {sim['spectral']}</span>
                    <span class="class-pill {get_pill_class(sim['uniformity'])}">Uniformity: {sim['uniformity']}</span>
                    <span class="class-pill {get_pill_class(sim['sti'])}">STI: {sim['sti']}</span>
                    <span class="class-pill {get_pill_class(sim['lti'])}">LTI: {sim['lti']}</span>
                </div>
                <div style="margin-top: 0.5rem; color: #64748b; font-size: 0.8rem;">
                    Last calibration: {sim['last_calibration'].strftime('%Y-%m-%d')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Recent Measurements</div>', unsafe_allow_html=True)

        # Sort by date
        recent_measurements = sorted(measurements, key=lambda x: x['date'], reverse=True)[:8]

        for m in recent_measurements:
            overall = m['overall']
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid #334155;
                        border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="color: #f8fafc; font-size: 0.9rem;">{m['simulator']}</div>
                        <div style="color: #64748b; font-size: 0.75rem;">
                            {m['date'].strftime('%Y-%m-%d %H:%M')}
                        </div>
                    </div>
                    <span class="class-pill {get_pill_class(overall)}" style="font-size: 0.85rem;">
                        {overall}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Trend Chart
    st.markdown('<div class="section-title">Classification Trends (Last 30 Days)</div>', unsafe_allow_html=True)

    # Generate trend data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    trend_data = []

    for date in dates:
        day_measurements = [m for m in measurements
                           if m['date'].date() == date.date()]
        counts = {'A+': 0, 'A': 0, 'B': 0, 'C': 0}
        for m in day_measurements:
            counts[m['overall']] = counts.get(m['overall'], 0) + 1

        # Add some random data for demonstration
        if not day_measurements:
            counts = {
                'A+': random.randint(0, 3),
                'A': random.randint(1, 4),
                'B': random.randint(0, 2),
                'C': random.randint(0, 1)
            }

        trend_data.append({
            'date': date,
            'A+': counts['A+'],
            'A': counts['A'],
            'B': counts['B'],
            'C': counts['C']
        })

    df_trend = pd.DataFrame(trend_data)

    fig = go.Figure()

    for grade in ['A+', 'A', 'B', 'C']:
        fig.add_trace(go.Scatter(
            x=df_trend['date'],
            y=df_trend[grade],
            name=grade,
            mode='lines+markers',
            line=dict(color=BADGE_COLORS[grade], width=2),
            marker=dict(size=6),
            fill='tonexty' if grade != 'A+' else None,
            stackgroup='one'
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            showgrid=False,
            tickformat='%b %d'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(71,85,105,0.3)',
            title='Number of Measurements'
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
