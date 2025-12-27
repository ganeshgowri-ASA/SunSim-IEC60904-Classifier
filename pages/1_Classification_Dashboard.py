"""
Classification Dashboard - Main Overview
IEC 60904-9 Ed.3 Solar Simulator Classification

Displays overall classification with Spectral Match, Uniformity,
and Temporal Stability grade badges.
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_models import (
    ClassificationGrade,
    CLASSIFICATION_THRESHOLDS,
    get_grade_color,
    get_grade_description,
)

# Page configuration
st.set_page_config(
    page_title="Classification Dashboard | SunSim",
    page_icon=":material/dashboard:",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }

    .overall-classification-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D4A6F 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 58, 95, 0.3);
    }

    .overall-title {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .overall-grade {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: 4px;
        margin: 1rem 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .grade-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        height: 100%;
    }

    .grade-card-title {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .grade-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 16px;
        min-width: 100px;
        margin-bottom: 1rem;
    }

    .grade-a-plus { background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; }
    .grade-a { background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); color: white; }
    .grade-b { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: white; }
    .grade-c { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; }

    .grade-value {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.5rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #F1F5F9;
    }

    .metric-label {
        color: #64748B;
    }

    .metric-value {
        font-weight: 600;
        color: #1E3A5F;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-pass {
        background: #D1FAE5;
        color: #065F46;
    }

    .status-warning {
        background: #FEF3C7;
        color: #92400E;
    }

    .info-section {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }

    .info-title {
        font-weight: 600;
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def generate_sample_data():
    """Generate sample classification data for demonstration"""
    return {
        "spectral_match": {
            "grade": ClassificationGrade.A_PLUS,
            "min_ratio": 0.92,
            "max_ratio": 1.08,
            "intervals_in_spec": 100,
            "total_intervals": 100,
        },
        "uniformity": {
            "grade": ClassificationGrade.A_PLUS,
            "non_uniformity": 0.78,
            "min_irradiance": 995.2,
            "max_irradiance": 1010.8,
            "mean_irradiance": 1002.5,
        },
        "temporal_stability": {
            "grade": ClassificationGrade.A,
            "sti": 0.42,
            "lti": 0.85,
            "min_irradiance": 998.1,
            "max_irradiance": 1006.3,
        },
        "equipment": {
            "manufacturer": "SunSim Technologies",
            "model": "SS-3000 Pro",
            "serial": "SS3K-2024-0042",
            "lamp_type": "Xenon Arc",
            "lamp_hours": 245.5,
            "calibration_date": datetime.now() - timedelta(days=45),
            "next_calibration": datetime.now() + timedelta(days=320),
        },
        "test_info": {
            "test_date": datetime.now(),
            "operator": "John Smith",
            "laboratory": "PV Testing Lab - Building A",
            "certificate": "IEC-2024-SS-0042",
            "ambient_temp": 23.5,
            "humidity": 45.2,
        }
    }


def get_grade_class(grade: ClassificationGrade) -> str:
    """Get CSS class for grade badge"""
    grade_classes = {
        ClassificationGrade.A_PLUS: "grade-a-plus",
        ClassificationGrade.A: "grade-a",
        ClassificationGrade.B: "grade-b",
        ClassificationGrade.C: "grade-c",
    }
    return grade_classes.get(grade, "grade-c")


def create_gauge_chart(value: float, max_value: float, title: str, grade: ClassificationGrade) -> go.Figure:
    """Create a gauge chart for metric visualization"""
    color = get_grade_color(grade)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#64748B'}},
        number={'suffix': '%', 'font': {'size': 24, 'color': '#1E3A5F'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': "#E2E8F0"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 0,
            'steps': [
                {'range': [0, max_value * 0.1], 'color': '#D1FAE5'},
                {'range': [max_value * 0.1, max_value * 0.2], 'color': '#FEF3C7'},
                {'range': [max_value * 0.2, max_value * 0.5], 'color': '#FEE2E2'},
                {'range': [max_value * 0.5, max_value], 'color': '#FECACA'},
            ],
            'threshold': {
                'line': {'color': "#1E3A5F", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1E3A5F'}
    )

    return fig


def main():
    """Main dashboard page"""

    # Header
    st.markdown('<h1 class="main-title">Classification Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">IEC 60904-9 Ed.3 Overall Classification Results</p>',
        unsafe_allow_html=True
    )

    # Load sample data
    data = generate_sample_data()

    # Overall Classification Card
    overall_grade = (
        f"{data['spectral_match']['grade'].value}"
        f"{data['uniformity']['grade'].value}"
        f"{data['temporal_stability']['grade'].value}"
    )

    st.markdown(f"""
    <div class="overall-classification-card">
        <div class="overall-title">Overall Classification</div>
        <div class="overall-grade">{overall_grade}</div>
        <div style="opacity: 0.8;">
            Spectral Match: {data['spectral_match']['grade'].value} |
            Uniformity: {data['uniformity']['grade'].value} |
            Temporal Stability: {data['temporal_stability']['grade'].value}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grade cards
    col1, col2, col3 = st.columns(3)

    with col1:
        grade = data['spectral_match']['grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Spectral Match (SPD)</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                Ratio Range: {data['spectral_match']['min_ratio']:.2f} - {data['spectral_match']['max_ratio']:.2f}
            </div>
            <div class="grade-value" style="margin-top: 0.5rem;">
                {data['spectral_match']['intervals_in_spec']}/{data['spectral_match']['total_intervals']} intervals in spec
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        grade = data['uniformity']['grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Non-Uniformity</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                Non-Uniformity: {data['uniformity']['non_uniformity']:.2f}%
            </div>
            <div class="grade-value" style="margin-top: 0.5rem;">
                Mean Irradiance: {data['uniformity']['mean_irradiance']:.1f} W/m²
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        grade = data['temporal_stability']['grade']
        st.markdown(f"""
        <div class="grade-card">
            <div class="grade-card-title">Temporal Stability</div>
            <div class="grade-badge {get_grade_class(grade)}">{grade.value}</div>
            <div class="grade-value">
                STI: {data['temporal_stability']['sti']:.2f}% | LTI: {data['temporal_stability']['lti']:.2f}%
            </div>
            <div class="grade-value" style="margin-top: 0.5rem;">
                Range: {data['temporal_stability']['min_irradiance']:.1f} - {data['temporal_stability']['max_irradiance']:.1f} W/m²
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Gauge charts row
    st.markdown("### Performance Metrics")

    gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

    with gauge_col1:
        deviation = max(
            abs(1 - data['spectral_match']['min_ratio']),
            abs(data['spectral_match']['max_ratio'] - 1)
        ) * 100
        fig = create_gauge_chart(
            deviation, 50, "Spectral Deviation",
            data['spectral_match']['grade']
        )
        st.plotly_chart(fig, use_container_width=True)

    with gauge_col2:
        fig = create_gauge_chart(
            data['uniformity']['non_uniformity'], 10, "Non-Uniformity",
            data['uniformity']['grade']
        )
        st.plotly_chart(fig, use_container_width=True)

    with gauge_col3:
        fig = create_gauge_chart(
            data['temporal_stability']['sti'], 5, "Temporal Instability (STI)",
            data['temporal_stability']['grade']
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Equipment and Test Information
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("### Equipment Information")
        equip = data['equipment']

        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Manufacturer | {equip['manufacturer']} |
        | Model | {equip['model']} |
        | Serial Number | {equip['serial']} |
        | Lamp Type | {equip['lamp_type']} |
        | Lamp Hours | {equip['lamp_hours']:.1f} hrs |
        | Calibration Date | {equip['calibration_date'].strftime('%Y-%m-%d')} |
        | Next Calibration | {equip['next_calibration'].strftime('%Y-%m-%d')} |
        """)

    with info_col2:
        st.markdown("### Test Information")
        test = data['test_info']

        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Test Date | {test['test_date'].strftime('%Y-%m-%d %H:%M')} |
        | Operator | {test['operator']} |
        | Laboratory | {test['laboratory']} |
        | Certificate No. | {test['certificate']} |
        | Ambient Temperature | {test['ambient_temp']:.1f} °C |
        | Relative Humidity | {test['humidity']:.1f} % |
        """)

    # Classification Summary
    st.divider()
    st.markdown("### IEC 60904-9 Ed.3 Classification Thresholds")

    threshold_data = []
    for param, thresholds in CLASSIFICATION_THRESHOLDS.items():
        if param == "spectral_match":
            continue  # Handle separately due to different format
        for grade, value in thresholds.items():
            threshold_data.append({
                "Parameter": param.replace("_", " ").title(),
                "Grade": grade.value,
                "Threshold": f"<= {value}%"
            })

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **Spectral Match Thresholds** (Ratio Range)

        | Grade | Min Ratio | Max Ratio |
        |-------|-----------|-----------|
        | A+ | 0.875 | 1.125 |
        | A | 0.75 | 1.25 |
        | B | 0.6 | 1.4 |
        | C | 0.4 | 2.0 |
        """)

    with col_b:
        st.markdown("""
        **Uniformity & Temporal Stability Thresholds**

        | Grade | Uniformity | STI | LTI |
        |-------|------------|-----|-----|
        | A+ | <= 1% | <= 0.5% | <= 1% |
        | A | <= 2% | <= 2% | <= 2% |
        | B | <= 5% | <= 5% | <= 5% |
        | C | <= 10% | <= 10% | <= 10% |
        """)

    # Export button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Generate ISO 17025 Report", type="primary", use_container_width=True):
            st.info("Report generation feature - Coming soon!")


if __name__ == "__main__":
    main()
