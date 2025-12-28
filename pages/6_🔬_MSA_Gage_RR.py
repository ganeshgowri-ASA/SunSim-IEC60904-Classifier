"""
Measurement System Analysis (MSA) - Gage R&R Study Interface.
Provides Gage R&R analysis with variance component breakdown and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.msa_calculations import (
    calculate_gage_rr_xbar_r,
    calculate_gage_rr_anova,
    generate_sample_msa_data,
    get_grr_status,
    get_ndc_status,
    create_variance_component_summary
)
from utils.db import (
    get_msa_data,
    insert_msa_batch,
    get_simulator_ids,
    get_msa_studies,
    update_msa_results,
    init_database
)

# Page configuration
st.set_page_config(
    page_title="Gage R&R | SunSim Classifier",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2D3139;
    }
    .grr-card {
        background: linear-gradient(135deg, #1A1D24 0%, #0E1117 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid;
        text-align: center;
        margin: 10px 0;
    }
    .grr-value {
        font-size: 48px;
        font-weight: bold;
    }
    .grr-label {
        font-size: 14px;
        color: #888;
        margin-top: 5px;
    }
    .status-pill {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-top: 10px;
    }
    div[data-testid="stDataFrame"] {
        background-color: #1A1D24;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_database()


def create_variance_pie_chart(result) -> go.Figure:
    """Create variance component pie chart."""
    labels = ['Repeatability', 'Reproducibility', 'Part-to-Part']
    values = [
        result.repeatability_pct_contribution,
        result.reproducibility_pct_contribution,
        result.part_to_part_pct_contribution
    ]
    colors = ['#FF6B6B', '#FFE66D', '#00D4AA']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(color='#FAFAFA', size=12),
        hovertemplate='%{label}<br>%{value:.1f}%<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text="Variance Components", font=dict(color='#FAFAFA', size=16)),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=50, b=80)
    )

    return fig


def create_study_variation_bar_chart(result) -> go.Figure:
    """Create study variation bar chart."""
    categories = ['Gage R&R', 'Repeatability', 'Reproducibility', 'Part-to-Part']
    values = [
        result.grr_pct_sv,
        result.repeatability_pct_sv,
        result.reproducibility_pct_sv,
        result.part_to_part_pct_sv
    ]

    # Color based on GRR threshold
    colors = []
    for i, v in enumerate(values):
        if i == 0:  # GRR
            if v < 10:
                colors.append('#00D4AA')
            elif v < 30:
                colors.append('#FFE66D')
            else:
                colors.append('#FF6B6B')
        else:
            colors.append('#4ECDC4')

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        textfont=dict(color='#FAFAFA')
    )])

    # Add threshold lines
    fig.add_hline(y=10, line_dash="dash", line_color="#00D4AA",
                  annotation_text="10% (Acceptable)", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color="#FF6B6B",
                  annotation_text="30% (Unacceptable)", annotation_position="right")

    fig.update_layout(
        title=dict(text="% Study Variation", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="% Study Variation",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA'),
            range=[0, max(values) * 1.2 + 10]
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=350,
        margin=dict(l=60, r=100, t=50, b=50)
    )

    return fig


def create_operator_part_chart(data: pd.DataFrame) -> go.Figure:
    """Create operator by part interaction chart."""
    operators = data['operator'].unique()
    parts = data['part_id'].unique()

    # Calculate means
    means = data.groupby(['operator', 'part_id'])['measured_value'].mean().unstack()

    fig = go.Figure()

    colors = ['#00D4AA', '#4ECDC4', '#FFE66D', '#FF6B6B', '#9B59B6']

    for i, op in enumerate(operators):
        if op in means.index:
            fig.add_trace(go.Scatter(
                x=means.columns.tolist(),
                y=means.loc[op].values,
                mode='lines+markers',
                name=op,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title=dict(text="Operator × Part Interaction", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Part",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="Measured Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=50)
    )

    return fig


def create_range_chart(data: pd.DataFrame) -> go.Figure:
    """Create range chart by operator."""
    # Calculate ranges within each operator-part-trial group
    ranges_data = []
    for (op, part), group in data.groupby(['operator', 'part_id']):
        range_val = group['measured_value'].max() - group['measured_value'].min()
        ranges_data.append({'operator': op, 'part_id': part, 'range': range_val})

    range_df = pd.DataFrame(ranges_data)

    operators = range_df['operator'].unique()
    colors = ['#00D4AA', '#4ECDC4', '#FFE66D']

    fig = go.Figure()

    for i, op in enumerate(operators):
        op_data = range_df[range_df['operator'] == op]
        fig.add_trace(go.Scatter(
            x=op_data['part_id'],
            y=op_data['range'],
            mode='lines+markers',
            name=op,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8)
        ))

    # Add R-bar line
    r_bar = range_df['range'].mean()
    fig.add_hline(y=r_bar, line_dash="solid", line_color="#FFE66D",
                  annotation_text=f"R̄: {r_bar:.4f}", annotation_position="right")

    fig.update_layout(
        title=dict(text="Range Chart by Operator", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(
            title="Part",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        yaxis=dict(
            title="Range",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=100, t=80, b=50)
    )

    return fig


def display_grr_gauge(grr_pct: float, title: str = "GRR %"):
    """Display GRR gauge with status color."""
    status, color = get_grr_status(grr_pct)

    st.markdown(f"""
    <div class="grr-card" style="border-color: {color};">
        <div class="grr-value" style="color: {color};">{grr_pct:.1f}%</div>
        <div class="grr-label">{title}</div>
        <div class="status-pill" style="background-color: {color}; color: #0E1117;">
            {status}
        </div>
    </div>
    """, unsafe_allow_html=True)


# Main page content
st.title("🔬 Measurement System Analysis - Gage R&R")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Study Configuration")

    # Data source selection
    data_source = st.radio(
        "Data Source",
        ["Sample Data", "Database", "Manual Entry", "Upload CSV"],
        index=0
    )

    if data_source == "Sample Data":
        st.subheader("Study Parameters")
        n_operators = st.slider("Number of Operators", 2, 5, 3)
        n_parts = st.slider("Number of Parts", 5, 15, 10)
        n_trials = st.slider("Number of Trials", 2, 5, 3)

        st.subheader("Variation Components")
        part_var = st.slider("Part Variation", 0.5, 10.0, 5.0)
        op_var = st.slider("Operator Variation", 0.0, 2.0, 0.5)
        repeat_var = st.slider("Repeatability", 0.1, 2.0, 0.3)

        seed = st.number_input("Random Seed", value=42, min_value=0)

    elif data_source == "Database":
        simulators = get_simulator_ids()
        selected_simulator = st.selectbox("Select Simulator", simulators)
        studies = get_msa_studies()
        if studies:
            selected_study = st.selectbox("Select Study", studies)
        else:
            st.info("No studies in database.")
            selected_study = None

    elif data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        st.info("CSV should have columns: operator, part_id, trial, measured_value")

    st.markdown("---")
    st.subheader("Analysis Method")
    analysis_method = st.radio(
        "Calculation Method",
        ["X-bar & R Method", "ANOVA Method"],
        index=0
    )

    st.markdown("---")
    st.subheader("Tolerance (Optional)")
    use_tolerance = st.checkbox("Include Tolerance", value=False)
    if use_tolerance:
        tolerance = st.number_input("Tolerance (USL - LSL)", value=10.0, min_value=0.1)
    else:
        tolerance = None

# Load or generate data
data = None

if data_source == "Sample Data":
    data = generate_sample_msa_data(
        n_operators=n_operators,
        n_parts=n_parts,
        n_trials=n_trials,
        part_variation=part_var,
        operator_variation=op_var,
        repeatability=repeat_var,
        seed=seed
    )

elif data_source == "Database" and selected_study:
    data = get_msa_data(simulator_id=selected_simulator, study_name=selected_study)
    if data.empty:
        st.info("No data found for selected study.")
        data = None

elif data_source == "Upload CSV" and uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        required_cols = ['operator', 'part_id', 'trial', 'measured_value']
        if not all(col in data.columns for col in required_cols):
            st.error(f"CSV must have columns: {required_cols}")
            data = None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        data = None

elif data_source == "Manual Entry":
    st.subheader("📝 Manual Data Entry")

    entry_col1, entry_col2 = st.columns(2)
    with entry_col1:
        num_operators = st.number_input("Number of Operators", 2, 5, 3, key="manual_ops")
        num_parts = st.number_input("Number of Parts", 2, 15, 5, key="manual_parts")
    with entry_col2:
        num_trials = st.number_input("Number of Trials", 2, 5, 2, key="manual_trials")

    operator_names = [st.text_input(f"Operator {i+1} Name", f"Operator {i+1}", key=f"op_{i}")
                     for i in range(num_operators)]
    part_ids = [f"Part {i+1}" for i in range(num_parts)]

    st.markdown("### Enter Measurements")

    # Create data entry grid
    entry_data = []
    for op_idx, op_name in enumerate(operator_names):
        st.markdown(f"**{op_name}**")
        cols = st.columns(num_parts)
        for part_idx, part_id in enumerate(part_ids):
            with cols[part_idx]:
                st.markdown(f"*{part_id}*")
                for trial in range(1, num_trials + 1):
                    value = st.number_input(
                        f"T{trial}",
                        value=100.0,
                        key=f"val_{op_idx}_{part_idx}_{trial}",
                        label_visibility="collapsed" if trial > 1 else "visible"
                    )
                    entry_data.append({
                        'operator': op_name,
                        'part_id': part_id,
                        'trial': trial,
                        'measured_value': value
                    })

    if st.button("Analyze Data", type="primary"):
        data = pd.DataFrame(entry_data)

# Main analysis
if data is not None and len(data) > 0:
    st.subheader("📊 Study Summary")

    # Display data summary
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Operators", data['operator'].nunique())
    with sum_col2:
        st.metric("Parts", data['part_id'].nunique())
    with sum_col3:
        st.metric("Trials", data.groupby(['operator', 'part_id']).size().mode().iloc[0])
    with sum_col4:
        st.metric("Total Measurements", len(data))

    st.markdown("---")

    # Perform Gage R&R analysis
    try:
        if analysis_method == "X-bar & R Method":
            result = calculate_gage_rr_xbar_r(data, tolerance)
            anova_results = None
        else:
            result, anova_results = calculate_gage_rr_anova(data, tolerance)

        # Main GRR Results
        st.subheader("🎯 Gage R&R Results")

        grr_col1, grr_col2, grr_col3 = st.columns(3)

        with grr_col1:
            display_grr_gauge(result.grr_pct_sv, "GRR (% Study Var)")

        with grr_col2:
            ndc_status, ndc_color = get_ndc_status(result.ndc)
            st.markdown(f"""
            <div class="grr-card" style="border-color: {ndc_color};">
                <div class="grr-value" style="color: {ndc_color};">{result.ndc}</div>
                <div class="grr-label">Number of Distinct Categories (ndc)</div>
                <div class="status-pill" style="background-color: {ndc_color}; color: #0E1117;">
                    {ndc_status}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with grr_col3:
            if tolerance:
                display_grr_gauge(result.grr_pct_tolerance, "GRR (% Tolerance)")
            else:
                st.markdown("""
                <div class="grr-card" style="border-color: #555;">
                    <div class="grr-value" style="color: #888;">--</div>
                    <div class="grr-label">GRR (% Tolerance)</div>
                    <div class="status-pill" style="background-color: #555; color: #FAFAFA;">
                        No Tolerance Set
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Variance Components
        st.subheader("📈 Variance Component Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pie = create_variance_pie_chart(result)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_bar = create_study_variation_bar_chart(result)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed breakdown
        st.subheader("📋 Variance Component Summary")
        summary_df = create_variance_component_summary(result)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # ANOVA Table (if using ANOVA method)
        if anova_results:
            st.subheader("📊 ANOVA Table")
            anova_df = pd.DataFrame([{
                'Source': r.source,
                'DF': r.df,
                'SS': f"{r.ss:.6f}",
                'MS': f"{r.ms:.6f}" if r.ms else "-",
                'F': f"{r.f_value:.4f}" if r.f_value else "-",
                'P-Value': f"{r.p_value:.4f}" if r.p_value else "-"
            } for r in anova_results])
            st.dataframe(anova_df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Graphical Analysis
        st.subheader("📉 Graphical Analysis")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            fig_interaction = create_operator_part_chart(data)
            st.plotly_chart(fig_interaction, use_container_width=True)

        with graph_col2:
            fig_range = create_range_chart(data)
            st.plotly_chart(fig_range, use_container_width=True)

        # Raw Data View
        with st.expander("📄 View Raw Data"):
            st.dataframe(data, hide_index=True, use_container_width=True)

        # Save to Database
        with st.expander("💾 Save Study to Database"):
            save_col1, save_col2 = st.columns(2)
            with save_col1:
                save_simulator_id = st.text_input("Simulator ID", value="SIM-001", key="save_sim")
            with save_col2:
                save_study_name = st.text_input("Study Name", value=f"GRR_{datetime.now().strftime('%Y%m%d')}")

            if st.button("Save Study", type="primary"):
                insert_msa_batch(data, save_simulator_id, save_study_name)
                update_msa_results(
                    save_simulator_id,
                    save_study_name,
                    result.grr_pct_sv,
                    result.repeatability_pct_sv,
                    result.reproducibility_pct_sv
                )
                st.success(f"✓ Saved study '{save_study_name}' to database")

        # Interpretation Guide
        with st.expander("📖 Interpretation Guide"):
            st.markdown("""
            ### GRR Acceptance Criteria (per AIAG MSA Manual)

            | GRR % | Status | Interpretation |
            |-------|--------|----------------|
            | < 10% | ✅ Acceptable | Measurement system is acceptable |
            | 10-30% | ⚠️ Marginal | May be acceptable based on application, cost, risk |
            | > 30% | ❌ Unacceptable | Measurement system needs improvement |

            ### Number of Distinct Categories (ndc)

            | ndc | Status | Interpretation |
            |-----|--------|----------------|
            | ≥ 5 | ✅ Acceptable | Adequate resolution for process control |
            | 3-4 | ⚠️ Marginal | Limited ability to distinguish parts |
            | < 3 | ❌ Unacceptable | Cannot adequately distinguish parts |

            ### Variance Components

            - **Repeatability (EV)**: Equipment variation - variation when same operator measures same part
            - **Reproducibility (AV)**: Appraiser variation - variation between different operators
            - **Part-to-Part (PV)**: Actual variation between parts

            Ideally, Part-to-Part variation should dominate (>90%), with GRR being minimal.
            """)

    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    if data_source != "Manual Entry":
        st.info("👈 Configure study parameters in the sidebar to begin analysis")

        st.markdown("""
        ### Getting Started with Gage R&R

        A Gage R&R study evaluates measurement system variation:

        1. **Configure Study**: Set number of operators, parts, and trials
        2. **Collect Data**: Each operator measures each part multiple times
        3. **Analyze**: Calculate repeatability and reproducibility components
        4. **Interpret**: Compare GRR% to acceptance criteria

        #### Recommended Study Design
        - **Operators**: 3 (minimum 2)
        - **Parts**: 10 (minimum 5)
        - **Trials**: 2-3 per operator-part combination
        """)
