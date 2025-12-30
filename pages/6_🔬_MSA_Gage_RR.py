"""
Measurement System Analysis (MSA) - Gage R&R Study Interface.
Provides Gage R&R analysis with variance component breakdown and visualization.
Includes Reference Module repeatability assessment for flasher/reference module analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path
import io

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
    init_database,
    get_ref_module_data,
    get_ref_module_ids
)
from utils.spc_calculations import (
    generate_ref_module_sample_data,
    get_capability_rating
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


# Reference Module Repeatability Analysis Functions
def calculate_ref_module_repeatability(data: pd.DataFrame, value_col: str = 'isc') -> dict:
    """
    Calculate repeatability metrics for reference module measurements.

    This assesses whether variation is from the flasher or the reference module.

    Args:
        data: DataFrame with flash measurements
        value_col: Column name for the measurement value

    Returns:
        Dictionary with repeatability statistics
    """
    values = data[value_col].values
    n = len(values)

    if n < 2:
        return None

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Calculate moving ranges
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr)

    # Estimate repeatability sigma using d2 = 1.128
    repeatability_sigma = mr_bar / 1.128

    # Calculate within-subgroup variation (repeatability)
    repeatability_pct = (repeatability_sigma / mean) * 100 if mean > 0 else 0

    # Range analysis
    range_val = np.max(values) - np.min(values)

    # Calculate coefficient of variation
    cv = (std / mean) * 100 if mean > 0 else 0

    # Calculate 6-sigma spread
    six_sigma = 6 * repeatability_sigma

    return {
        'n_measurements': n,
        'mean': mean,
        'std_dev': std,
        'repeatability_sigma': repeatability_sigma,
        'repeatability_pct': repeatability_pct,
        'cv': cv,
        'range': range_val,
        'mr_bar': mr_bar,
        'six_sigma': six_sigma,
        'min': np.min(values),
        'max': np.max(values)
    }


def generate_ref_module_msa_data(
    n_flashers: int = 2,
    n_ref_modules: int = 3,
    n_repeats: int = 10,
    nominal_value: float = 8.5,
    flasher_variation: float = 0.02,
    ref_module_variation: float = 0.01,
    repeatability: float = 0.005,
    seed: int = 42
) -> pd.DataFrame:
    """Generate sample data for Reference Module MSA study."""
    np.random.seed(seed)

    data = []
    flashers = [f"Flasher {i+1}" for i in range(n_flashers)]
    ref_modules = [f"RefMod {i+1}" for i in range(n_ref_modules)]

    flasher_effects = np.random.normal(0, flasher_variation, n_flashers)
    ref_module_effects = np.random.normal(0, ref_module_variation, n_ref_modules)

    for f_idx, flasher in enumerate(flashers):
        for r_idx, ref_module in enumerate(ref_modules):
            for repeat in range(1, n_repeats + 1):
                value = (nominal_value +
                        flasher_effects[f_idx] +
                        ref_module_effects[r_idx] +
                        np.random.normal(0, repeatability))
                data.append({
                    'flasher': flasher,
                    'ref_module': ref_module,
                    'repeat': repeat,
                    'isc': round(value, 4)
                })

    return pd.DataFrame(data)


def create_flasher_comparison_chart(data: pd.DataFrame, value_col: str = 'isc') -> go.Figure:
    """Create a chart comparing measurements across flashers."""
    fig = go.Figure()

    flashers = data['flasher'].unique()
    colors = ['#00D4AA', '#4ECDC4', '#FFE66D', '#FF6B6B', '#9B59B6']

    for i, flasher in enumerate(flashers):
        flasher_data = data[data['flasher'] == flasher][value_col]
        fig.add_trace(go.Box(
            y=flasher_data,
            name=flasher,
            marker_color=colors[i % len(colors)],
            boxmean='sd'
        ))

    fig.update_layout(
        title=dict(text="Flasher Comparison", font=dict(color='#FAFAFA', size=16)),
        yaxis=dict(
            title="Measured Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        xaxis=dict(tickfont=dict(color='#FAFAFA')),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400
    )

    return fig


def create_ref_module_comparison_chart(data: pd.DataFrame, value_col: str = 'isc') -> go.Figure:
    """Create a chart comparing measurements across reference modules."""
    fig = go.Figure()

    ref_modules = data['ref_module'].unique()
    colors = ['#00D4AA', '#4ECDC4', '#FFE66D', '#FF6B6B', '#9B59B6']

    for i, ref_mod in enumerate(ref_modules):
        ref_data = data[data['ref_module'] == ref_mod][value_col]
        fig.add_trace(go.Box(
            y=ref_data,
            name=ref_mod,
            marker_color=colors[i % len(colors)],
            boxmean='sd'
        ))

    fig.update_layout(
        title=dict(text="Reference Module Comparison", font=dict(color='#FAFAFA', size=16)),
        yaxis=dict(
            title="Measured Value",
            gridcolor='#2D3139',
            tickfont=dict(color='#FAFAFA')
        ),
        xaxis=dict(tickfont=dict(color='#FAFAFA')),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400
    )

    return fig


def create_interaction_heatmap(data: pd.DataFrame, value_col: str = 'isc') -> go.Figure:
    """Create a heatmap showing flasher x reference module interaction."""
    pivot = data.pivot_table(
        index='flasher',
        columns='ref_module',
        values=value_col,
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title=dict(text="Mean Value", font=dict(color='#FAFAFA')))
    ))

    fig.update_layout(
        title=dict(text="Flasher x Reference Module Interaction", font=dict(color='#FAFAFA', size=16)),
        xaxis=dict(title="Reference Module", tickfont=dict(color='#FAFAFA')),
        yaxis=dict(title="Flasher", tickfont=dict(color='#FAFAFA')),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#1A1D24',
        font=dict(color='#FAFAFA'),
        height=400
    )

    return fig


def calculate_ref_module_anova(data: pd.DataFrame, value_col: str = 'isc') -> dict:
    """
    Perform simplified ANOVA to separate flasher vs reference module variation.

    Returns variance components attributable to:
    - Flasher (reproducibility)
    - Reference Module (part variation)
    - Repeatability (measurement error)
    """
    from scipy import stats as scipy_stats

    flashers = data['flasher'].unique()
    ref_modules = data['ref_module'].unique()
    n_flashers = len(flashers)
    n_ref_modules = len(ref_modules)

    # Get repeat count
    repeats = data.groupby(['flasher', 'ref_module']).size()
    n_repeats = int(repeats.mode().iloc[0]) if len(repeats) > 0 else 1
    n_total = len(data)

    grand_mean = data[value_col].mean()

    # SS Total
    ss_total = ((data[value_col] - grand_mean) ** 2).sum()

    # SS Flashers
    flasher_means = data.groupby('flasher')[value_col].mean()
    ss_flashers = n_ref_modules * n_repeats * ((flasher_means - grand_mean) ** 2).sum()

    # SS Reference Modules
    ref_means = data.groupby('ref_module')[value_col].mean()
    ss_ref_modules = n_flashers * n_repeats * ((ref_means - grand_mean) ** 2).sum()

    # SS Interaction
    cell_means = data.groupby(['flasher', 'ref_module'])[value_col].mean()
    ss_interaction = 0
    for f in flashers:
        for r in ref_modules:
            if (f, r) in cell_means.index:
                cell_mean = cell_means[(f, r)]
                expected = grand_mean + (flasher_means[f] - grand_mean) + (ref_means[r] - grand_mean)
                ss_interaction += n_repeats * (cell_mean - expected) ** 2

    # SS Repeatability (Error)
    ss_repeatability = ss_total - ss_flashers - ss_ref_modules - ss_interaction

    # Degrees of freedom
    df_flashers = n_flashers - 1
    df_ref_modules = n_ref_modules - 1
    df_interaction = df_flashers * df_ref_modules
    df_repeatability = n_flashers * n_ref_modules * (n_repeats - 1)
    df_total = n_total - 1

    # Mean squares
    ms_flashers = ss_flashers / df_flashers if df_flashers > 0 else 0
    ms_ref_modules = ss_ref_modules / df_ref_modules if df_ref_modules > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

    # Variance components
    var_repeatability = ms_repeatability
    var_interaction = max(0, (ms_interaction - ms_repeatability) / n_repeats)
    var_flasher = max(0, (ms_flashers - ms_interaction) / (n_ref_modules * n_repeats))
    var_ref_module = max(0, (ms_ref_modules - ms_interaction) / (n_flashers * n_repeats))
    var_total = var_repeatability + var_interaction + var_flasher + var_ref_module

    # Percentage contributions
    if var_total > 0:
        pct_repeatability = (var_repeatability / var_total) * 100
        pct_flasher = (var_flasher / var_total) * 100
        pct_ref_module = (var_ref_module / var_total) * 100
        pct_interaction = (var_interaction / var_total) * 100
    else:
        pct_repeatability = pct_flasher = pct_ref_module = pct_interaction = 0

    # GRR = Repeatability + Flasher variation (reproducibility)
    var_grr = var_repeatability + var_flasher + var_interaction
    pct_grr = (var_grr / var_total) * 100 if var_total > 0 else 0

    # Standard deviations
    std_repeatability = np.sqrt(var_repeatability)
    std_flasher = np.sqrt(var_flasher)
    std_ref_module = np.sqrt(var_ref_module)
    std_grr = np.sqrt(var_grr)
    std_total = np.sqrt(var_total)

    # Study variation (5.15 sigma)
    sv_grr = 5.15 * std_grr
    sv_total = 5.15 * std_total

    # Percent study variation
    pct_sv_grr = (std_grr / std_total) * 100 if std_total > 0 else 0

    # Number of distinct categories
    ndc = 1.41 * (std_ref_module / std_grr) if std_grr > 0 else 0
    ndc = max(1, int(ndc))

    return {
        # Variance components
        'var_repeatability': var_repeatability,
        'var_flasher': var_flasher,
        'var_ref_module': var_ref_module,
        'var_interaction': var_interaction,
        'var_grr': var_grr,
        'var_total': var_total,
        # Standard deviations
        'std_repeatability': std_repeatability,
        'std_flasher': std_flasher,
        'std_ref_module': std_ref_module,
        'std_grr': std_grr,
        'std_total': std_total,
        # Percentages
        'pct_repeatability': pct_repeatability,
        'pct_flasher': pct_flasher,
        'pct_ref_module': pct_ref_module,
        'pct_interaction': pct_interaction,
        'pct_grr': pct_grr,
        'pct_sv_grr': pct_sv_grr,
        # NDC
        'ndc': ndc,
        # ANOVA table values
        'anova': {
            'flasher': {'df': df_flashers, 'ss': ss_flashers, 'ms': ms_flashers},
            'ref_module': {'df': df_ref_modules, 'ss': ss_ref_modules, 'ms': ms_ref_modules},
            'interaction': {'df': df_interaction, 'ss': ss_interaction, 'ms': ms_interaction},
            'repeatability': {'df': df_repeatability, 'ss': ss_repeatability, 'ms': ms_repeatability},
            'total': {'df': df_total, 'ss': ss_total}
        },
        # Study info
        'n_flashers': n_flashers,
        'n_ref_modules': n_ref_modules,
        'n_repeats': n_repeats
    }


# Main page content
st.title("🔬 Measurement System Analysis - Gage R&R")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Study Configuration")

    # Analysis Type Selection (NEW)
    analysis_type = st.radio(
        "📊 Analysis Type",
        ["Standard Gage R&R", "Reference Module Study"],
        index=0
    )

    st.markdown("---")

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

# ============================================================================
# REFERENCE MODULE STUDY
# ============================================================================
if analysis_type == "Reference Module Study":
    st.markdown("---")
    st.subheader("🔬 Reference Module Repeatability Study")
    st.markdown("""
    Assess whether measurement variation is from the **flasher** or the **reference module**.
    This helps identify if issues are with the sun simulator equipment or the reference modules themselves.
    """)

    # Reference Module Study Configuration in Sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔬 Reference Module Settings")

        ref_data_source = st.radio(
            "Data Source",
            ["Sample Data", "Upload CSV"],
            index=0,
            key="ref_data_source"
        )

        if ref_data_source == "Sample Data":
            n_flashers = st.slider("Number of Flashers", 2, 5, 2, key="n_flashers")
            n_ref_modules_study = st.slider("Number of Reference Modules", 2, 10, 3, key="n_ref_mods")
            n_repeats = st.slider("Repeats per Combination", 5, 20, 10, key="n_repeats")

            st.subheader("Variation Components")
            flasher_var = st.slider("Flasher Variation (σ)", 0.001, 0.05, 0.02, format="%.3f")
            ref_mod_var = st.slider("Ref Module Variation (σ)", 0.001, 0.05, 0.01, format="%.3f")
            repeat_var = st.slider("Repeatability (σ)", 0.001, 0.02, 0.005, format="%.3f")
            nominal_val = st.number_input("Nominal Isc (A)", value=8.5, format="%.3f")
            ref_seed = st.number_input("Random Seed", value=42, min_value=0, key="ref_seed")
        else:
            ref_uploaded = st.file_uploader(
                "Upload Reference Module Study CSV",
                type=['csv'],
                help="CSV should have columns: flasher, ref_module, repeat, isc",
                key="ref_upload"
            )

    # Load or generate reference module study data
    ref_study_data = None

    if ref_data_source == "Sample Data":
        ref_study_data = generate_ref_module_msa_data(
            n_flashers=n_flashers,
            n_ref_modules=n_ref_modules_study,
            n_repeats=n_repeats,
            nominal_value=nominal_val,
            flasher_variation=flasher_var,
            ref_module_variation=ref_mod_var,
            repeatability=repeat_var,
            seed=ref_seed
        )
    elif ref_data_source == "Upload CSV" and ref_uploaded:
        try:
            ref_study_data = pd.read_csv(ref_uploaded)
            required_cols = ['flasher', 'ref_module', 'repeat', 'isc']
            if not all(col in ref_study_data.columns for col in required_cols):
                # Try alternative column names
                col_mapping = {
                    'Flasher': 'flasher', 'FLASHER': 'flasher',
                    'RefModule': 'ref_module', 'ref_mod': 'ref_module', 'REF_MODULE': 'ref_module',
                    'Repeat': 'repeat', 'REPEAT': 'repeat', 'trial': 'repeat', 'Trial': 'repeat',
                    'Isc': 'isc', 'ISC': 'isc'
                }
                ref_study_data = ref_study_data.rename(columns=col_mapping)

                if not all(col in ref_study_data.columns for col in required_cols):
                    st.error(f"CSV must have columns: {required_cols}")
                    ref_study_data = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            ref_study_data = None

    if ref_study_data is not None and len(ref_study_data) > 0:
        # Study Summary
        st.subheader("📊 Study Summary")

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            st.metric("Flashers", ref_study_data['flasher'].nunique())
        with sum_col2:
            st.metric("Reference Modules", ref_study_data['ref_module'].nunique())
        with sum_col3:
            st.metric("Repeats/Combo", ref_study_data.groupby(['flasher', 'ref_module']).size().mode().iloc[0])
        with sum_col4:
            st.metric("Total Measurements", len(ref_study_data))

        st.markdown("---")

        # Perform ANOVA analysis
        try:
            anova_result = calculate_ref_module_anova(ref_study_data)

            # Main Results Display
            st.subheader("🎯 Variance Component Analysis")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                grr_status, grr_color = get_grr_status(anova_result['pct_sv_grr'])
                st.markdown(f"""
                <div class="grr-card" style="border-color: {grr_color};">
                    <div class="grr-value" style="color: {grr_color};">{anova_result['pct_sv_grr']:.1f}%</div>
                    <div class="grr-label">GRR (% Study Variation)</div>
                    <div class="status-pill" style="background-color: {grr_color}; color: #0E1117;">
                        {grr_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                ndc_status, ndc_color = get_ndc_status(anova_result['ndc'])
                st.markdown(f"""
                <div class="grr-card" style="border-color: {ndc_color};">
                    <div class="grr-value" style="color: {ndc_color};">{anova_result['ndc']}</div>
                    <div class="grr-label">Number of Distinct Categories</div>
                    <div class="status-pill" style="background-color: {ndc_color}; color: #0E1117;">
                        {ndc_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                # Determine primary source of variation
                if anova_result['pct_flasher'] > anova_result['pct_ref_module'] and anova_result['pct_flasher'] > anova_result['pct_repeatability']:
                    primary_source = "Flasher"
                    source_color = "#FF6B6B"
                    source_action = "Investigate flasher calibration and stability"
                elif anova_result['pct_ref_module'] > anova_result['pct_repeatability']:
                    primary_source = "Reference Module"
                    source_color = "#FFE66D"
                    source_action = "Reference modules show significant variation"
                else:
                    primary_source = "Repeatability"
                    source_color = "#00D4AA"
                    source_action = "Good measurement system - variation is random"

                st.markdown(f"""
                <div class="grr-card" style="border-color: {source_color};">
                    <div class="grr-value" style="color: {source_color}; font-size: 24px;">{primary_source}</div>
                    <div class="grr-label">Primary Variation Source</div>
                    <div style="font-size: 11px; color: #888; margin-top: 5px;">{source_action}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Variance Component Breakdown
            st.subheader("📈 Variance Component Breakdown")

            var_col1, var_col2 = st.columns(2)

            with var_col1:
                # Pie chart of variance components
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Repeatability', 'Flasher', 'Ref Module', 'Interaction'],
                    values=[
                        anova_result['pct_repeatability'],
                        anova_result['pct_flasher'],
                        anova_result['pct_ref_module'],
                        anova_result['pct_interaction']
                    ],
                    hole=0.4,
                    marker_colors=['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D'],
                    textinfo='label+percent',
                    textfont=dict(color='#FAFAFA', size=12)
                )])

                fig_pie.update_layout(
                    title=dict(text="Variance Components", font=dict(color='#FAFAFA', size=16)),
                    plot_bgcolor='#0E1117',
                    paper_bgcolor='#1A1D24',
                    font=dict(color='#FAFAFA'),
                    height=350
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with var_col2:
                # Bar chart of % Study Variation
                fig_bar = go.Figure(data=[go.Bar(
                    x=['GRR', 'Repeatability', 'Flasher', 'Ref Module'],
                    y=[
                        anova_result['pct_sv_grr'],
                        np.sqrt(anova_result['var_repeatability'] / anova_result['var_total']) * 100,
                        np.sqrt(anova_result['var_flasher'] / anova_result['var_total']) * 100,
                        np.sqrt(anova_result['var_ref_module'] / anova_result['var_total']) * 100
                    ],
                    marker_color=['#FF6B6B' if anova_result['pct_sv_grr'] > 30 else '#FFE66D' if anova_result['pct_sv_grr'] > 10 else '#00D4AA',
                                 '#4ECDC4', '#4ECDC4', '#4ECDC4'],
                    text=[f"{v:.1f}%" for v in [
                        anova_result['pct_sv_grr'],
                        np.sqrt(anova_result['var_repeatability'] / anova_result['var_total']) * 100,
                        np.sqrt(anova_result['var_flasher'] / anova_result['var_total']) * 100,
                        np.sqrt(anova_result['var_ref_module'] / anova_result['var_total']) * 100
                    ]],
                    textposition='outside',
                    textfont=dict(color='#FAFAFA')
                )])

                fig_bar.add_hline(y=10, line_dash="dash", line_color="#00D4AA",
                                  annotation_text="10% (Acceptable)", annotation_position="right")
                fig_bar.add_hline(y=30, line_dash="dash", line_color="#FF6B6B",
                                  annotation_text="30% (Unacceptable)", annotation_position="right")

                fig_bar.update_layout(
                    title=dict(text="% Study Variation", font=dict(color='#FAFAFA', size=16)),
                    xaxis=dict(tickfont=dict(color='#FAFAFA')),
                    yaxis=dict(title="% Study Variation", gridcolor='#2D3139', tickfont=dict(color='#FAFAFA')),
                    plot_bgcolor='#0E1117',
                    paper_bgcolor='#1A1D24',
                    font=dict(color='#FAFAFA'),
                    height=350
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")

            # Graphical Analysis
            st.subheader("📉 Graphical Analysis")

            graph_col1, graph_col2 = st.columns(2)

            with graph_col1:
                fig_flasher = create_flasher_comparison_chart(ref_study_data)
                st.plotly_chart(fig_flasher, use_container_width=True)

            with graph_col2:
                fig_ref = create_ref_module_comparison_chart(ref_study_data)
                st.plotly_chart(fig_ref, use_container_width=True)

            # Interaction Heatmap
            fig_heatmap = create_interaction_heatmap(ref_study_data)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("---")

            # ANOVA Table
            st.subheader("📊 ANOVA Table")

            anova_df = pd.DataFrame([
                {'Source': 'Flasher', 'DF': anova_result['anova']['flasher']['df'],
                 'SS': f"{anova_result['anova']['flasher']['ss']:.6f}",
                 'MS': f"{anova_result['anova']['flasher']['ms']:.6f}",
                 'VarComp': f"{anova_result['var_flasher']:.6f}",
                 '%Contribution': f"{anova_result['pct_flasher']:.2f}%"},
                {'Source': 'Ref Module', 'DF': anova_result['anova']['ref_module']['df'],
                 'SS': f"{anova_result['anova']['ref_module']['ss']:.6f}",
                 'MS': f"{anova_result['anova']['ref_module']['ms']:.6f}",
                 'VarComp': f"{anova_result['var_ref_module']:.6f}",
                 '%Contribution': f"{anova_result['pct_ref_module']:.2f}%"},
                {'Source': 'Flasher x Ref Mod', 'DF': anova_result['anova']['interaction']['df'],
                 'SS': f"{anova_result['anova']['interaction']['ss']:.6f}",
                 'MS': f"{anova_result['anova']['interaction']['ms']:.6f}",
                 'VarComp': f"{anova_result['var_interaction']:.6f}",
                 '%Contribution': f"{anova_result['pct_interaction']:.2f}%"},
                {'Source': 'Repeatability', 'DF': anova_result['anova']['repeatability']['df'],
                 'SS': f"{anova_result['anova']['repeatability']['ss']:.6f}",
                 'MS': f"{anova_result['anova']['repeatability']['ms']:.6f}",
                 'VarComp': f"{anova_result['var_repeatability']:.6f}",
                 '%Contribution': f"{anova_result['pct_repeatability']:.2f}%"},
                {'Source': 'Total', 'DF': anova_result['anova']['total']['df'],
                 'SS': f"{anova_result['anova']['total']['ss']:.6f}",
                 'MS': '-', 'VarComp': f"{anova_result['var_total']:.6f}",
                 '%Contribution': '100.00%'}
            ])

            st.dataframe(anova_df, hide_index=True, use_container_width=True)

            # Interpretation
            with st.expander("📖 Interpretation Guide"):
                st.markdown("""
                ### Reference Module Study Interpretation

                This analysis separates variation into:

                | Source | Description |
                |--------|-------------|
                | **Flasher** | Variation between different flashers (reproducibility) |
                | **Reference Module** | True variation between reference modules (part variation) |
                | **Flasher x Ref Mod** | Interaction between flasher and reference module |
                | **Repeatability** | Random measurement variation (equipment precision) |

                ### What the Results Mean

                - **High Flasher %**: Your flashers have significant differences - consider calibration
                - **High Ref Module %**: Expected - indicates reference modules are different (good for distinguishing)
                - **High Repeatability %**: Good - most variation is random measurement noise
                - **High Interaction %**: Flasher behavior differs by reference module - potential issue

                ### Actions Based on Results

                | Primary Source | Action |
                |---------------|--------|
                | Flasher dominant | Calibrate flashers, check lamp aging, verify alignment |
                | Ref Module dominant | Normal if modules are meant to differ; compare to specs |
                | Repeatability dominant | Measurement system is good |
                | Interaction dominant | Investigate specific flasher/module combinations |
                """)

            # Export
            with st.expander("📥 Export Results"):
                report_lines = [
                    "# Reference Module MSA Study Report",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "## Study Summary",
                    f"- Flashers: {anova_result['n_flashers']}",
                    f"- Reference Modules: {anova_result['n_ref_modules']}",
                    f"- Repeats per combination: {anova_result['n_repeats']}",
                    f"- Total measurements: {len(ref_study_data)}",
                    "",
                    "## Key Results",
                    f"- GRR (% Study Variation): {anova_result['pct_sv_grr']:.2f}%",
                    f"- Number of Distinct Categories: {anova_result['ndc']}",
                    "",
                    "## Variance Components",
                    f"- Flasher: {anova_result['pct_flasher']:.2f}%",
                    f"- Reference Module: {anova_result['pct_ref_module']:.2f}%",
                    f"- Interaction: {anova_result['pct_interaction']:.2f}%",
                    f"- Repeatability: {anova_result['pct_repeatability']:.2f}%"
                ]

                st.download_button(
                    label="Download Report (Markdown)",
                    data="\n".join(report_lines),
                    file_name=f"ref_module_msa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

                csv_buffer = io.StringIO()
                ref_study_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Data (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"ref_module_msa_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    else:
        st.info("👈 Configure Reference Module Study parameters in the sidebar")

        st.markdown("""
        ### Reference Module Study

        This study helps identify whether measurement variation comes from:

        - **Flasher (Sun Simulator)**: Equipment variation between different flashers
        - **Reference Module**: Actual differences between reference modules
        - **Repeatability**: Random measurement noise

        #### Study Design
        1. Use multiple flashers (minimum 2)
        2. Use multiple reference modules (minimum 2)
        3. Measure each combination multiple times (minimum 5 repeats)

        #### CSV Format
        Your CSV should have columns:
        - `flasher`: Flasher identifier (e.g., "Flasher 1")
        - `ref_module`: Reference module identifier (e.g., "RefMod 1")
        - `repeat`: Repeat number (1, 2, 3, ...)
        - `isc`: Measured Isc value
        """)
