"""
Sun Simulator Classification System
Manufacturers Database Page

This page provides a comprehensive database of sun simulator manufacturers
with equipment profiles, performance specifications, and interactive comparison tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_CONFIG, THEME, BADGE_COLORS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Manufacturers - " + APP_CONFIG['title'],
    page_icon="🏭",
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

.page-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #475569;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.page-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
}

.manufacturer-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
}

.manufacturer-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    border-color: #3b82f6;
}

.manufacturer-name {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.manufacturer-country {
    color: #94a3b8;
    font-size: 0.9rem;
}

.model-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.model-name {
    font-size: 1rem;
    font-weight: 600;
    color: #f8fafc;
}

.model-type {
    color: #94a3b8;
    font-size: 0.85rem;
}

.spec-label {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.spec-value {
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 500;
}

.class-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.25rem;
}

.badge-aplus { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.badge-a { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
.badge-b { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.badge-c { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}

.stat-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.stat-value {
    font-size: 2rem;
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

.filter-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.comparison-table {
    background: #1e293b;
    border-radius: 8px;
    overflow: hidden;
}

.info-box {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.info-box h4 {
    color: #3b82f6;
    margin-bottom: 0.5rem;
}

.info-box p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.9rem;
}

.lamp-type-icon {
    font-size: 1.5rem;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_manufacturers_data():
    """Load manufacturer data from JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "manufacturers.json"
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading manufacturers data: {e}")
        return {"manufacturers": [], "metadata": {}}


def get_badge_class(classification: str) -> str:
    """Get CSS class for classification badge."""
    return {
        'A+': 'badge-aplus',
        'A': 'badge-a',
        'B': 'badge-b',
        'C': 'badge-c'
    }.get(classification, 'badge-c')


def get_lamp_icon(lamp_type: str) -> str:
    """Get icon for lamp type."""
    lamp_type_lower = lamp_type.lower() if lamp_type else ""
    if 'xenon' in lamp_type_lower:
        return "⚡"
    elif 'led' in lamp_type_lower:
        return "💡"
    elif 'halogen' in lamp_type_lower:
        return "🔆"
    else:
        return "☀️"


def get_overall_classification(typical: dict) -> str:
    """Calculate overall classification from typical values."""
    priority = {'C': 0, 'B': 1, 'A': 2, 'A+': 3}
    classes = [
        typical.get('spectral', 'C'),
        typical.get('uniformity', 'C'),
        typical.get('temporal', 'C')
    ]
    return min(classes, key=lambda x: priority.get(x, 0))


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Load data
    data = load_manufacturers_data()
    manufacturers = data.get("manufacturers", [])
    metadata = data.get("metadata", {})
    lamp_types = data.get("lamp_types", {})

    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🏭 Manufacturer Database</div>
        <div class="page-subtitle">
            Comprehensive profiles of sun simulator manufacturers and equipment specifications
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats
    total_manufacturers = len([m for m in manufacturers if m.get('id') != 'custom'])
    total_models = sum(len(m.get('models', [])) for m in manufacturers)
    xenon_models = sum(1 for m in manufacturers for model in m.get('models', [])
                       if 'xenon' in model.get('type', '').lower())
    led_models = sum(1 for m in manufacturers for model in m.get('models', [])
                     if 'led' in model.get('type', '').lower())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_manufacturers}</div>
            <div class="stat-label">Manufacturers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_models}</div>
            <div class="stat-label">Models</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{xenon_models}</div>
            <div class="stat-label">Xenon Systems</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{led_models}</div>
            <div class="stat-label">LED Systems</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Manufacturer Profiles",
        "📊 Model Comparison",
        "📈 Performance Analysis",
        "🔍 Lamp Technology"
    ])

    # ==========================================================================
    # TAB 1: Manufacturer Profiles
    # ==========================================================================
    with tab1:
        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            countries = list(set(m.get('country', 'Unknown') for m in manufacturers if m.get('id') != 'custom'))
            countries.sort()
            selected_country = st.selectbox(
                "Filter by Country",
                ["All Countries"] + countries
            )

        with col2:
            lamp_type_filter = st.selectbox(
                "Filter by Lamp Type",
                ["All Types", "Xenon Flash", "Xenon Arc", "LED", "Hybrid"]
            )

        with col3:
            class_filter = st.selectbox(
                "Minimum Classification",
                ["Any", "A+", "A", "B"]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter manufacturers
        filtered_manufacturers = manufacturers
        if selected_country != "All Countries":
            filtered_manufacturers = [m for m in filtered_manufacturers
                                     if m.get('country') == selected_country]

        # Display manufacturers
        for mfr in filtered_manufacturers:
            if mfr.get('id') == 'custom':
                continue

            # Filter models by lamp type
            models = mfr.get('models', [])
            if lamp_type_filter != "All Types":
                filter_map = {
                    "Xenon Flash": "pulsed_xenon",
                    "Xenon Arc": "xenon_continuous",
                    "LED": "led",
                    "Hybrid": "hybrid"
                }
                filter_key = filter_map.get(lamp_type_filter, "")
                models = [m for m in models if filter_key in m.get('type', '').lower()]

            # Filter by classification
            if class_filter != "Any":
                class_priority = {'A+': 3, 'A': 2, 'B': 1, 'C': 0}
                min_priority = class_priority.get(class_filter, 0)
                models = [m for m in models if class_priority.get(
                    get_overall_classification(m.get('typical_classification', {})), 0
                ) >= min_priority]

            if not models and (lamp_type_filter != "All Types" or class_filter != "Any"):
                continue

            with st.expander(f"**{mfr.get('name', 'Unknown')}** - {mfr.get('country', 'Unknown')}", expanded=False):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"""
                    <div style="color: #94a3b8; margin-bottom: 1rem;">
                        <strong>Founded:</strong> {mfr.get('founded', 'N/A')} |
                        <strong>Specialization:</strong> {mfr.get('specialization', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)

                    if mfr.get('website'):
                        st.markdown(f"🌐 [{mfr.get('website')}]({mfr.get('website')})")

                with col2:
                    st.metric("Models Available", len(models))

                st.markdown("---")

                # Display models
                for model in models:
                    typical = model.get('typical_classification', {})
                    specs = model.get('specifications', {})
                    overall_class = get_overall_classification(typical)
                    lamp_icon = get_lamp_icon(model.get('lamp_type', ''))

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-name">
                                <span class="lamp-type-icon">{lamp_icon}</span>
                                {model.get('name', 'Unknown')}
                            </div>
                            <div class="model-type">{model.get('lamp_type', 'N/A')} | {model.get('description', '')}</div>
                            <div style="margin-top: 0.5rem;">
                                <span class="class-badge {get_badge_class(typical.get('spectral', 'C'))}">
                                    Spectral: {typical.get('spectral', 'N/A')}
                                </span>
                                <span class="class-badge {get_badge_class(typical.get('uniformity', 'C'))}">
                                    Uniformity: {typical.get('uniformity', 'N/A')}
                                </span>
                                <span class="class-badge {get_badge_class(typical.get('temporal', 'C'))}">
                                    Temporal: {typical.get('temporal', 'N/A')}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        irr = model.get('irradiance_range', {})
                        st.markdown(f"""
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            <div><span class="spec-label">Irradiance:</span></div>
                            <div class="spec-value">{irr.get('min', 'N/A')}-{irr.get('max', 'N/A')} W/m²</div>
                            <div style="margin-top: 0.5rem;"><span class="spec-label">Test Areas:</span></div>
                            <div class="spec-value">{', '.join(str(a) for a in model.get('test_area_cm2', [])[:3])} cm²</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            <div><span class="spec-label">Drift:</span></div>
                            <div class="spec-value">{specs.get('drift_percent', 'N/A')}%</div>
                            <div style="margin-top: 0.5rem;"><span class="spec-label">Repeatability:</span></div>
                            <div class="spec-value">{specs.get('repeatability_percent', 'N/A')}%</div>
                        </div>
                        """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: Model Comparison
    # ==========================================================================
    with tab2:
        st.markdown('<div class="section-title">Interactive Model Comparison</div>', unsafe_allow_html=True)

        # Get all models for comparison
        all_models = []
        for mfr in manufacturers:
            for model in mfr.get('models', []):
                model_info = {
                    'Manufacturer': mfr.get('name', 'Unknown'),
                    'Model': model.get('name', 'Unknown'),
                    'Type': model.get('lamp_type', 'N/A'),
                    'Spectral': model.get('typical_classification', {}).get('spectral', 'N/A'),
                    'Uniformity': model.get('typical_classification', {}).get('uniformity', 'N/A'),
                    'Temporal': model.get('typical_classification', {}).get('temporal', 'N/A'),
                    'Drift %': model.get('specifications', {}).get('drift_percent'),
                    'Repeatability %': model.get('specifications', {}).get('repeatability_percent'),
                    'UV Coverage': '✓' if model.get('specifications', {}).get('uv_coverage') else '✗',
                    'Irr. Min': model.get('irradiance_range', {}).get('min'),
                    'Irr. Max': model.get('irradiance_range', {}).get('max'),
                }
                all_models.append(model_info)

        df = pd.DataFrame(all_models)

        # Model selection for comparison
        st.markdown("**Select Models to Compare:**")
        model_options = [f"{m['Manufacturer']} - {m['Model']}" for m in all_models]
        selected_models = st.multiselect(
            "Choose up to 6 models",
            model_options,
            default=model_options[:3] if len(model_options) >= 3 else model_options,
            max_selections=6
        )

        if selected_models:
            # Filter dataframe
            comparison_df = df[df.apply(
                lambda x: f"{x['Manufacturer']} - {x['Model']}" in selected_models, axis=1
            )]

            # Display comparison table
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Radar chart comparison
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Classification Comparison**")

                # Convert classifications to numeric
                class_map = {'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'N/A': 0}
                categories = ['Spectral', 'Uniformity', 'Temporal']

                fig = go.Figure()

                for _, row in comparison_df.iterrows():
                    values = [
                        class_map.get(row['Spectral'], 0),
                        class_map.get(row['Uniformity'], 0),
                        class_map.get(row['Temporal'], 0)
                    ]
                    values.append(values[0])  # Close the polygon

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        name=f"{row['Manufacturer'][:15]} - {row['Model'][:15]}",
                        fill='toself',
                        opacity=0.6
                    ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 4],
                            ticktext=['', 'C', 'B', 'A', 'A+'],
                            tickvals=[0, 1, 2, 3, 4]
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=-0.3,
                        xanchor='center',
                        x=0.5,
                        font=dict(size=10)
                    ),
                    margin=dict(l=60, r=60, t=40, b=80)
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Performance Metrics**")

                # Bar chart for drift and repeatability
                metrics_df = comparison_df[['Manufacturer', 'Model', 'Drift %', 'Repeatability %']].copy()
                metrics_df = metrics_df.dropna()

                if not metrics_df.empty:
                    metrics_df['Label'] = metrics_df['Manufacturer'].str[:10] + ' - ' + metrics_df['Model'].str[:10]

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        name='Drift %',
                        x=metrics_df['Label'],
                        y=metrics_df['Drift %'],
                        marker_color='#3b82f6'
                    ))

                    fig.add_trace(go.Bar(
                        name='Repeatability %',
                        x=metrics_df['Label'],
                        y=metrics_df['Repeatability %'],
                        marker_color='#10b981'
                    ))

                    fig.update_layout(
                        barmode='group',
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        xaxis=dict(showgrid=False, tickangle=-45),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(71,85,105,0.3)',
                            title='Percentage (%)'
                        ),
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='center',
                            x=0.5
                        ),
                        margin=dict(l=40, r=40, t=60, b=100)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No performance data available for selected models")

    # ==========================================================================
    # TAB 3: Performance Analysis
    # ==========================================================================
    with tab3:
        st.markdown('<div class="section-title">Performance Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Classification distribution
            st.markdown("**Classification Distribution by Parameter**")

            class_counts = {
                'Spectral': {'A+': 0, 'A': 0, 'B': 0, 'C': 0},
                'Uniformity': {'A+': 0, 'A': 0, 'B': 0, 'C': 0},
                'Temporal': {'A+': 0, 'A': 0, 'B': 0, 'C': 0}
            }

            for mfr in manufacturers:
                for model in mfr.get('models', []):
                    typical = model.get('typical_classification', {})
                    for param, key in [('Spectral', 'spectral'), ('Uniformity', 'uniformity'), ('Temporal', 'temporal')]:
                        grade = typical.get(key, 'C')
                        if grade in class_counts[param]:
                            class_counts[param][grade] += 1

            fig = go.Figure()

            for grade in ['A+', 'A', 'B', 'C']:
                fig.add_trace(go.Bar(
                    name=grade,
                    x=['Spectral', 'Uniformity', 'Temporal'],
                    y=[class_counts['Spectral'][grade],
                       class_counts['Uniformity'][grade],
                       class_counts['Temporal'][grade]],
                    marker_color=BADGE_COLORS[grade]
                ))

            fig.update_layout(
                barmode='stack',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                xaxis=dict(showgrid=False),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(71,85,105,0.3)',
                    title='Number of Models'
                ),
                margin=dict(l=40, r=40, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Lamp type distribution
            st.markdown("**Distribution by Lamp Technology**")

            lamp_counts = {}
            for mfr in manufacturers:
                for model in mfr.get('models', []):
                    lamp = model.get('lamp_type', 'Other')
                    if 'Xenon Flash' in lamp or 'pulsed' in model.get('type', '').lower():
                        key = 'Xenon Flash'
                    elif 'Xenon Arc' in lamp or 'continuous' in model.get('type', '').lower():
                        key = 'Xenon Arc'
                    elif 'LED' in lamp:
                        key = 'LED'
                    else:
                        key = 'Other'
                    lamp_counts[key] = lamp_counts.get(key, 0) + 1

            fig = go.Figure(data=[go.Pie(
                labels=list(lamp_counts.keys()),
                values=list(lamp_counts.values()),
                hole=0.5,
                marker=dict(colors=['#ef4444', '#f59e0b', '#10b981', '#6b7280']),
                textinfo='label+percent',
                textfont=dict(size=12, color='white')
            )])

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Drift vs Repeatability scatter plot
        st.markdown("**Drift vs Repeatability Analysis**")

        scatter_data = []
        for mfr in manufacturers:
            for model in mfr.get('models', []):
                specs = model.get('specifications', {})
                drift = specs.get('drift_percent')
                repeat = specs.get('repeatability_percent')
                if drift is not None and repeat is not None:
                    scatter_data.append({
                        'Manufacturer': mfr.get('name', 'Unknown'),
                        'Model': model.get('name', 'Unknown'),
                        'Drift': drift,
                        'Repeatability': repeat,
                        'Type': model.get('lamp_type', 'Unknown'),
                        'Overall': get_overall_classification(model.get('typical_classification', {}))
                    })

        if scatter_data:
            scatter_df = pd.DataFrame(scatter_data)

            fig = px.scatter(
                scatter_df,
                x='Drift',
                y='Repeatability',
                color='Overall',
                symbol='Type',
                hover_data=['Manufacturer', 'Model'],
                color_discrete_map={'A+': '#10b981', 'A': '#3b82f6', 'B': '#f59e0b', 'C': '#ef4444'}
            )

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(71,85,105,0.3)',
                    title='Drift (%)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(71,85,105,0.3)',
                    title='Repeatability (%)'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                margin=dict(l=60, r=40, t=80, b=60)
            )

            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))

            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # TAB 4: Lamp Technology
    # ==========================================================================
    with tab4:
        st.markdown('<div class="section-title">Lamp Technology Overview</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>⚡ Xenon Flash</h4>
                <p>
                    <strong>Advantages:</strong> Full spectrum UV-IR, high intensity pulses,
                    fast measurement capability<br>
                    <strong>Disadvantages:</strong> Limited lamp lifetime (2000-5000 hrs),
                    spectrum drift over time, heat management<br>
                    <strong>Best for:</strong> Production testing, flash IV measurements
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>💡 LED Arrays</h4>
                <p>
                    <strong>Advantages:</strong> Long lifetime (50,000+ hrs), excellent stability,
                    tunable spectrum, low heat<br>
                    <strong>Disadvantages:</strong> Limited UV coverage, higher initial cost,
                    complex multi-channel control<br>
                    <strong>Best for:</strong> Laboratory testing, R&D, continuous measurements
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🔆 Xenon Arc (Continuous)</h4>
                <p>
                    <strong>Advantages:</strong> Excellent AM1.5G match, full UV-IR spectrum,
                    proven technology<br>
                    <strong>Disadvantages:</strong> High power consumption, lamp aging effects,
                    requires cooling system<br>
                    <strong>Best for:</strong> Research applications, reference measurements
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box">
                <h4>🌐 Hybrid Systems</h4>
                <p>
                    <strong>Advantages:</strong> Combines benefits of multiple sources,
                    optimized spectral coverage<br>
                    <strong>Disadvantages:</strong> Complex calibration, higher cost,
                    maintenance of multiple sources<br>
                    <strong>Best for:</strong> Advanced testing, multi-technology modules
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Technology comparison chart
        st.markdown("**Technology Performance Comparison**")

        tech_comparison = {
            'Parameter': ['Spectral Match', 'UV Coverage', 'Stability', 'Lifetime', 'Heat Load', 'Cost'],
            'Xenon Flash': [5, 5, 3, 2, 3, 3],
            'Xenon Arc': [5, 5, 3, 2, 2, 3],
            'LED Array': [4, 2, 5, 5, 5, 3],
            'Hybrid': [5, 4, 4, 4, 4, 2]
        }

        fig = go.Figure()

        for tech in ['Xenon Flash', 'Xenon Arc', 'LED Array', 'Hybrid']:
            fig.add_trace(go.Scatterpolar(
                r=tech_comparison[tech] + [tech_comparison[tech][0]],
                theta=tech_comparison['Parameter'] + [tech_comparison['Parameter'][0]],
                name=tech,
                fill='toself',
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    ticktext=['', 'Poor', 'Fair', 'Good', 'Excellent', 'Best'],
                    tickvals=[0, 1, 2, 3, 4, 5]
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=80, r=80, t=40, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Manufacturer Database v{metadata.get('version', '1.0.0')} |
        Last Updated: {metadata.get('last_updated', 'N/A')} |
        Standard: {metadata.get('standard', 'IEC 60904-9:2020 Ed.3')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
