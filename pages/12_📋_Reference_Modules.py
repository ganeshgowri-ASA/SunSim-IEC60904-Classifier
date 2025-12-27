"""
Sun Simulator Classification System
Reference Modules & Traceability Page

This page provides comprehensive management of reference modules used for
calibration and testing, including traceability chain validation per
IEC 60904-2 and IEC 60904-4 standards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APP_CONFIG, THEME, BADGE_COLORS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Reference Modules - " + APP_CONFIG['title'],
    page_icon="📋",
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

.module-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
}

.module-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}

.module-card.primary {
    border-left: 4px solid #10b981;
}

.module-card.secondary {
    border-left: 4px solid #3b82f6;
}

.module-card.working {
    border-left: 4px solid #f59e0b;
}

.module-serial {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f8fafc;
    font-family: 'Courier New', monospace;
}

.module-type {
    color: #94a3b8;
    font-size: 0.9rem;
}

.level-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

.level-primary {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.level-secondary {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
}

.level-working {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.level-production {
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-valid {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.status-expiring {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.status-expired {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
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

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #334155;
}

.chain-item {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
}

.chain-arrow {
    color: #3b82f6;
    font-size: 1.25rem;
    margin: 0 0.5rem;
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

.warning-box {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box h4 {
    color: #f59e0b;
    margin-bottom: 0.5rem;
}

.warning-box p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.9rem;
}

.error-box {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box h4 {
    color: #ef4444;
    margin-bottom: 0.5rem;
}

.error-box p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.9rem;
}

.spec-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
}

.spec-item {
    text-align: center;
    padding: 0.5rem;
}

.spec-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.spec-value {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 600;
}

.drift-indicator {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 500;
}

.drift-stable {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.drift-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
}

.drift-critical {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.match-excellent { color: #10b981; }
.match-good { color: #3b82f6; }
.match-acceptable { color: #f59e0b; }
.match-poor { color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_sample_reference_modules():
    """Generate sample reference module data for demonstration."""
    modules = [
        {
            'serial': 'REF-2024-001',
            'type': 'Monocrystalline Si',
            'manufacturer': 'ISTI-CNR',
            'model': 'Primary Reference Cell',
            'level': 'primary',
            'isc': 0.1580,
            'voc': 0.632,
            'pmax': 0.0850,
            'impp': 0.1495,
            'vmpp': 0.568,
            'ff': 85.2,
            'area': 4.0,
            'efficiency': 21.25,
            'calibration_date': datetime(2024, 6, 15),
            'expiry_date': datetime(2026, 6, 15),
            'calibration_lab': 'ISTI-CNR Pisa',
            'accreditation': 'ISO 17025',
            'certificate': 'ISTI-2024-00123',
            'uncertainty': 0.8,
            'traceability_chain': ['ISTI-CNR Primary', 'World Photovoltaic Scale'],
            'drift_isc': 0.05,
            'drift_pmax': 0.08,
            'drift_status': 'stable',
            'is_active': True
        },
        {
            'serial': 'REF-2024-002',
            'type': 'Monocrystalline Si',
            'manufacturer': 'Fraunhofer ISE',
            'model': 'Secondary Reference Cell',
            'level': 'secondary',
            'isc': 0.1565,
            'voc': 0.628,
            'pmax': 0.0838,
            'impp': 0.1478,
            'vmpp': 0.567,
            'ff': 85.0,
            'area': 4.0,
            'efficiency': 20.95,
            'calibration_date': datetime(2024, 3, 10),
            'expiry_date': datetime(2025, 3, 10),
            'calibration_lab': 'Fraunhofer ISE CalLab',
            'accreditation': 'ISO 17025',
            'certificate': 'ISE-2024-00456',
            'uncertainty': 1.2,
            'traceability_chain': ['Fraunhofer ISE', 'PTB Braunschweig', 'SI Units'],
            'drift_isc': 0.12,
            'drift_pmax': 0.15,
            'drift_status': 'stable',
            'is_active': True
        },
        {
            'serial': 'REF-2024-003',
            'type': 'PERC',
            'manufacturer': 'In-house',
            'model': 'Working Reference #1',
            'level': 'working',
            'isc': 9.85,
            'voc': 0.698,
            'pmax': 5.95,
            'impp': 9.45,
            'vmpp': 0.630,
            'ff': 86.5,
            'area': 243.36,
            'efficiency': 24.45,
            'calibration_date': datetime(2024, 9, 1),
            'expiry_date': datetime(2025, 3, 1),
            'calibration_lab': 'PV Test Laboratory',
            'accreditation': 'ISO 17025',
            'certificate': 'LAB-2024-00789',
            'uncertainty': 1.8,
            'traceability_chain': ['PV Test Lab', 'Fraunhofer ISE', 'PTB'],
            'drift_isc': 0.35,
            'drift_pmax': 0.42,
            'drift_status': 'stable',
            'is_active': True
        },
        {
            'serial': 'REF-2024-004',
            'type': 'HJT',
            'manufacturer': 'In-house',
            'model': 'Working Reference #2',
            'level': 'working',
            'isc': 10.12,
            'voc': 0.738,
            'pmax': 6.42,
            'impp': 9.68,
            'vmpp': 0.663,
            'ff': 86.0,
            'area': 243.36,
            'efficiency': 26.39,
            'calibration_date': datetime(2024, 7, 15),
            'expiry_date': datetime(2025, 1, 15),
            'calibration_lab': 'PV Test Laboratory',
            'accreditation': 'ISO 17025',
            'certificate': 'LAB-2024-00812',
            'uncertainty': 1.8,
            'traceability_chain': ['PV Test Lab', 'Fraunhofer ISE', 'PTB'],
            'drift_isc': 1.25,
            'drift_pmax': 1.48,
            'drift_status': 'drifting',
            'is_active': True
        },
        {
            'serial': 'REF-2023-005',
            'type': 'Multicrystalline Si',
            'manufacturer': 'CalLab',
            'model': 'Legacy Reference',
            'level': 'secondary',
            'isc': 0.1545,
            'voc': 0.618,
            'pmax': 0.0812,
            'impp': 0.1460,
            'vmpp': 0.556,
            'ff': 84.8,
            'area': 4.0,
            'efficiency': 20.30,
            'calibration_date': datetime(2023, 6, 1),
            'expiry_date': datetime(2024, 6, 1),
            'calibration_lab': 'CalLab PV Cells',
            'accreditation': 'ISO 17025',
            'certificate': 'CAL-2023-00234',
            'uncertainty': 1.5,
            'traceability_chain': ['CalLab', 'NREL', 'WRR'],
            'drift_isc': 2.8,
            'drift_pmax': 3.2,
            'drift_status': 'out_of_spec',
            'is_active': False
        }
    ]
    return modules


def generate_drift_history(serial: str, days: int = 180):
    """Generate sample drift history data."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_drift = random.uniform(-0.5, 0.5)

    drift_data = []
    for i, date in enumerate(dates):
        drift_data.append({
            'date': date,
            'isc_drift': base_drift + np.sin(i / 30) * 0.2 + random.uniform(-0.1, 0.1),
            'pmax_drift': base_drift * 1.2 + np.sin(i / 30) * 0.25 + random.uniform(-0.1, 0.1)
        })

    return pd.DataFrame(drift_data)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_level_class(level: str) -> str:
    """Get CSS class for reference level."""
    return f"level-{level}"


def get_status_class(status: str) -> str:
    """Get CSS class for calibration status."""
    return f"status-{status}"


def get_drift_class(drift: float, limit: float = 1.0) -> str:
    """Get CSS class for drift status."""
    if abs(drift) < limit * 0.5:
        return 'drift-stable'
    elif abs(drift) < limit:
        return 'drift-warning'
    else:
        return 'drift-critical'


def get_calibration_status(expiry_date: datetime) -> tuple:
    """Determine calibration status and days remaining."""
    now = datetime.now()
    days_remaining = (expiry_date - now).days

    if days_remaining < 0:
        return 'expired', days_remaining
    elif days_remaining < 30:
        return 'expiring', days_remaining
    else:
        return 'valid', days_remaining


def calculate_matching_quality(test_type: str, ref_type: str) -> str:
    """Calculate spectral matching quality between technologies."""
    silicon_types = {'Monocrystalline Si', 'Multicrystalline Si', 'PERC', 'TOPCon', 'HJT'}
    thin_film = {'CdTe', 'CIGS', 'Perovskite'}

    if test_type == ref_type:
        return 'excellent'
    elif test_type in silicon_types and ref_type in silicon_types:
        return 'good'
    elif test_type in thin_film and ref_type in thin_film:
        return 'acceptable'
    else:
        return 'poor'


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Load sample data
    modules = generate_sample_reference_modules()

    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📋 Reference Modules & Traceability</div>
        <div class="page-subtitle">
            Calibration chain management per IEC 60904-2 and IEC 60904-4 standards
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Stats
    active_modules = len([m for m in modules if m['is_active']])
    primary_count = len([m for m in modules if m['level'] == 'primary' and m['is_active']])
    expiring_count = len([m for m in modules
                          if get_calibration_status(m['expiry_date'])[0] == 'expiring'])
    expired_count = len([m for m in modules
                         if get_calibration_status(m['expiry_date'])[0] == 'expired'])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">{active_modules}</div>
            <div class="stat-label">Active References</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: #3b82f6;">{primary_count}</div>
            <div class="stat-label">Primary Standards</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        color = "#f59e0b" if expiring_count > 0 else "#10b981"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: {color};">{expiring_count}</div>
            <div class="stat-label">Expiring Soon</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        color = "#ef4444" if expired_count > 0 else "#10b981"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="color: {color};">{expired_count}</div>
            <div class="stat-label">Expired</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📦 Reference Modules",
        "🔗 Traceability Chain",
        "📈 Drift Monitoring",
        "🎯 Matching Criteria",
        "📊 Uncertainty Budget"
    ])

    # ==========================================================================
    # TAB 1: Reference Modules
    # ==========================================================================
    with tab1:
        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            level_filter = st.selectbox(
                "Filter by Level",
                ["All Levels", "Primary", "Secondary", "Working", "Production"]
            )

        with col2:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All Status", "Valid", "Expiring Soon", "Expired"]
            )

        with col3:
            active_filter = st.selectbox(
                "Active Status",
                ["All", "Active Only", "Inactive Only"]
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter modules
        filtered_modules = modules.copy()

        if level_filter != "All Levels":
            filtered_modules = [m for m in filtered_modules
                               if m['level'].lower() == level_filter.lower()]

        if status_filter != "All Status":
            status_map = {
                "Valid": "valid",
                "Expiring Soon": "expiring",
                "Expired": "expired"
            }
            filtered_modules = [m for m in filtered_modules
                               if get_calibration_status(m['expiry_date'])[0] == status_map.get(status_filter, '')]

        if active_filter == "Active Only":
            filtered_modules = [m for m in filtered_modules if m['is_active']]
        elif active_filter == "Inactive Only":
            filtered_modules = [m for m in filtered_modules if not m['is_active']]

        # Display modules
        for module in filtered_modules:
            status, days = get_calibration_status(module['expiry_date'])
            drift_class = get_drift_class(module['drift_pmax'])

            with st.expander(
                f"**{module['serial']}** - {module['type']} ({module['level'].title()})",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <span class="level-badge {get_level_class(module['level'])}">
                            {module['level'].upper()}
                        </span>
                        <span class="status-badge {get_status_class(status)}" style="margin-left: 0.5rem;">
                            {status.upper()} ({days} days)
                        </span>
                        <span class="drift-indicator {drift_class}" style="margin-left: 0.5rem;">
                            Drift: {module['drift_pmax']:.2f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    **Manufacturer:** {module['manufacturer']} | **Model:** {module['model']}

                    **Calibration Lab:** {module['calibration_lab']}
                    **Certificate:** {module['certificate']}
                    **Accreditation:** {module['accreditation']}
                    """)

                with col2:
                    st.markdown("**Uncertainty**")
                    st.metric("Expanded (k=2)", f"±{module['uncertainty']}%")

                st.markdown("---")

                # Electrical characteristics
                st.markdown("**Electrical Characteristics at STC**")
                cols = st.columns(6)

                with cols[0]:
                    st.metric("Isc", f"{module['isc']:.4f} A")
                with cols[1]:
                    st.metric("Voc", f"{module['voc']:.3f} V")
                with cols[2]:
                    st.metric("Pmax", f"{module['pmax']:.4f} W")
                with cols[3]:
                    st.metric("Impp", f"{module['impp']:.4f} A")
                with cols[4]:
                    st.metric("Vmpp", f"{module['vmpp']:.3f} V")
                with cols[5]:
                    st.metric("FF", f"{module['ff']:.1f}%")

                # Traceability chain preview
                st.markdown("**Traceability Chain**")
                chain_html = " → ".join([f"<span style='color: #3b82f6;'>{lab}</span>"
                                        for lab in module['traceability_chain']])
                st.markdown(f"<div style='color: #94a3b8;'>{chain_html}</div>", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: Traceability Chain
    # ==========================================================================
    with tab2:
        st.markdown('<div class="section-title">Calibration Traceability Hierarchy</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>IEC 60904-2 Calibration Hierarchy</h4>
            <p>
                Reference devices must be traceable to the World Radiometric Reference (WRR)
                through an unbroken chain of calibrations. The hierarchy ensures measurement
                accuracy and international comparability.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Visual hierarchy
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Traceability Levels**")

            levels = [
                ("🌍 World Radiometric Reference (WRR)", "International reference standard"),
                ("🏛️ National Metrology Institute", "PTB, NREL, ISTI-CNR, etc."),
                ("🔬 Primary Reference (k=2: ±0.8%)", "Calibrated by national lab"),
                ("📐 Secondary Reference (k=2: ±1.2%)", "Calibrated against primary"),
                ("🔧 Working Reference (k=2: ±1.8%)", "Used for routine measurements"),
                ("🏭 Production Reference (k=2: ±2.5%)", "For production testing only")
            ]

            for i, (title, desc) in enumerate(levels):
                arrow = "↓" if i < len(levels) - 1 else ""
                st.markdown(f"""
                <div class="chain-item">
                    <div>
                        <div style="color: #f8fafc; font-weight: 500;">{title}</div>
                        <div style="color: #64748b; font-size: 0.85rem;">{desc}</div>
                    </div>
                </div>
                {"<div style='text-align: center; color: #3b82f6; font-size: 1.5rem;'>↓</div>" if arrow else ""}
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**Current Traceability Status**")

            # Show chain for each active module
            for module in [m for m in modules if m['is_active']]:
                status, days = get_calibration_status(module['expiry_date'])

                box_class = "info-box" if status == "valid" else ("warning-box" if status == "expiring" else "error-box")

                chain_str = " → ".join(module['traceability_chain'])

                st.markdown(f"""
                <div class="{box_class}">
                    <h4>{module['serial']} ({module['level'].title()})</h4>
                    <p><strong>Chain:</strong> {chain_str}</p>
                    <p><strong>Uncertainty:</strong> ±{module['uncertainty']}% (k=2)</p>
                    <p><strong>Validity:</strong> {days} days remaining</p>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3: Drift Monitoring
    # ==========================================================================
    with tab3:
        st.markdown('<div class="section-title">Reference Module Drift Monitoring</div>',
                    unsafe_allow_html=True)

        # Select module for drift analysis
        active_serials = [m['serial'] for m in modules if m['is_active']]
        selected_serial = st.selectbox("Select Reference Module", active_serials)

        selected_module = next((m for m in modules if m['serial'] == selected_serial), None)

        if selected_module:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Generate drift history
                drift_data = generate_drift_history(selected_serial)

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   subplot_titles=('Isc Drift (%)', 'Pmax Drift (%)'),
                                   vertical_spacing=0.12)

                # Isc drift
                fig.add_trace(
                    go.Scatter(x=drift_data['date'], y=drift_data['isc_drift'],
                              mode='lines', name='Isc Drift',
                              line=dict(color='#3b82f6', width=2)),
                    row=1, col=1
                )

                # Drift limits for Isc
                limit = 1.0 if selected_module['level'] == 'secondary' else 2.0
                fig.add_hline(y=limit, line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"+{limit}% limit", row=1, col=1)
                fig.add_hline(y=-limit, line_dash="dash", line_color="#f59e0b",
                             annotation_text=f"-{limit}% limit", row=1, col=1)

                # Pmax drift
                fig.add_trace(
                    go.Scatter(x=drift_data['date'], y=drift_data['pmax_drift'],
                              mode='lines', name='Pmax Drift',
                              line=dict(color='#10b981', width=2)),
                    row=2, col=1
                )

                fig.add_hline(y=limit, line_dash="dash", line_color="#f59e0b", row=2, col=1)
                fig.add_hline(y=-limit, line_dash="dash", line_color="#f59e0b", row=2, col=1)

                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500,
                    showlegend=False,
                    margin=dict(l=60, r=40, t=60, b=40)
                )

                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(71,85,105,0.3)')

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Current Drift Status**")

                drift_isc = selected_module['drift_isc']
                drift_pmax = selected_module['drift_pmax']
                drift_limit = 1.0 if selected_module['level'] == 'secondary' else 2.0

                st.metric("Isc Drift", f"{drift_isc:.2f}%",
                         delta=f"{'Within' if abs(drift_isc) < drift_limit else 'Exceeds'} limit")
                st.metric("Pmax Drift", f"{drift_pmax:.2f}%",
                         delta=f"{'Within' if abs(drift_pmax) < drift_limit else 'Exceeds'} limit")

                st.markdown("---")

                st.markdown("**Drift Limits by Level**")
                limits_df = pd.DataFrame({
                    'Level': ['Primary', 'Secondary', 'Working', 'Production'],
                    'Limit (%)': [0.5, 1.0, 2.0, 3.0]
                })
                st.dataframe(limits_df, use_container_width=True, hide_index=True)

                st.markdown("---")

                if selected_module['drift_status'] == 'out_of_spec':
                    st.markdown("""
                    <div class="error-box">
                        <h4>⚠️ Action Required</h4>
                        <p>This reference module has drifted outside acceptable limits
                        and requires recalibration before further use.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif selected_module['drift_status'] == 'drifting':
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚡ Monitoring Recommended</h4>
                        <p>Drift is approaching limits. Increase monitoring frequency
                        and plan for recalibration.</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 4: Matching Criteria
    # ==========================================================================
    with tab4:
        st.markdown('<div class="section-title">Reference Module Matching Criteria</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>IEC 60904-7 Spectral Mismatch Considerations</h4>
            <p>
                The spectral response of the reference device should closely match
                the device under test (DUT) to minimize spectral mismatch errors.
                Matching quality depends on technology compatibility.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            test_type = st.selectbox(
                "Device Under Test (DUT) Technology",
                ['Monocrystalline Si', 'Multicrystalline Si', 'PERC', 'TOPCon',
                 'HJT', 'CdTe', 'CIGS', 'Perovskite', 'Bifacial']
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Matching Quality Legend:**")
            st.markdown("""
            <span class="match-excellent">● Excellent</span> |
            <span class="match-good">● Good</span> |
            <span class="match-acceptable">● Acceptable</span> |
            <span class="match-poor">● Poor</span>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Show matching recommendations
        st.markdown("**Recommended Reference Modules:**")

        for module in [m for m in modules if m['is_active']]:
            quality = calculate_matching_quality(test_type, module['type'])
            status, days = get_calibration_status(module['expiry_date'])

            quality_class = f"match-{quality}"

            st.markdown(f"""
            <div class="module-card {module['level']}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="module-serial">{module['serial']}</div>
                        <div class="module-type">{module['type']} | {module['manufacturer']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="{quality_class}" style="font-weight: 600; font-size: 1.1rem;">
                            {quality.upper()}
                        </div>
                        <div style="color: #64748b; font-size: 0.85rem;">
                            Uncertainty: ±{module['uncertainty']}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Matching matrix
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Technology Matching Matrix**")

        technologies = ['Mono-Si', 'Multi-Si', 'PERC', 'TOPCon', 'HJT', 'CdTe', 'CIGS']
        matrix_data = []

        for dut in technologies:
            row = {'DUT \\ Ref': dut}
            for ref in technologies:
                if dut == ref:
                    row[ref] = '●●●●'
                elif dut in ['Mono-Si', 'Multi-Si', 'PERC', 'TOPCon', 'HJT'] and \
                     ref in ['Mono-Si', 'Multi-Si', 'PERC', 'TOPCon', 'HJT']:
                    row[ref] = '●●●'
                elif dut in ['CdTe', 'CIGS'] and ref in ['CdTe', 'CIGS']:
                    row[ref] = '●●'
                else:
                    row[ref] = '●'
            matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data)
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    # ==========================================================================
    # TAB 5: Uncertainty Budget
    # ==========================================================================
    with tab5:
        st.markdown('<div class="section-title">Measurement Uncertainty Budget (IEC 60904-4)</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h4>Combined Uncertainty Calculation</h4>
            <p>
                The combined measurement uncertainty is calculated by root-sum-of-squares
                of all individual uncertainty components, then multiplied by the coverage
                factor (k=2) for 95% confidence level.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Uncertainty Components**")

            # Uncertainty budget components
            components = {
                'Reference module calibration': 1.5,
                'Spectral mismatch': 0.5,
                'Irradiance non-uniformity': 0.3,
                'Data acquisition': 0.2,
                'Temperature measurement': 0.3,
                'Repeatability': 0.2
            }

            # Allow user to modify
            modified_components = {}
            for name, default in components.items():
                modified_components[name] = st.slider(
                    name,
                    min_value=0.0,
                    max_value=3.0,
                    value=default,
                    step=0.1,
                    format="%.1f%%"
                )

        with col2:
            st.markdown("**Calculated Uncertainty**")

            # Calculate combined uncertainty
            sum_squares = sum(u**2 for u in modified_components.values())
            combined = np.sqrt(sum_squares)
            expanded = combined * 2.0

            st.metric("Combined Standard Uncertainty", f"±{combined:.2f}%")
            st.metric("Expanded Uncertainty (k=2)", f"±{expanded:.2f}%",
                     delta="95% confidence")

            # Pie chart of contributions
            fig = go.Figure(data=[go.Pie(
                labels=list(modified_components.keys()),
                values=[u**2 for u in modified_components.values()],
                hole=0.5,
                textinfo='label+percent',
                textfont=dict(size=10)
            )])

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                annotations=[dict(
                    text=f'±{expanded:.1f}%',
                    x=0.5, y=0.5,
                    font=dict(size=20, color='#f8fafc'),
                    showarrow=False
                )]
            )

            st.plotly_chart(fig, use_container_width=True)

        # Budget table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Uncertainty Budget Table**")

        budget_data = []
        for name, u in modified_components.items():
            budget_data.append({
                'Component': name,
                'Standard Uncertainty (%)': f"{u:.2f}",
                'Variance (%)': f"{u**2:.4f}",
                'Contribution (%)': f"{(u**2 / sum_squares * 100):.1f}"
            })

        budget_data.append({
            'Component': 'COMBINED (RSS)',
            'Standard Uncertainty (%)': f"{combined:.2f}",
            'Variance (%)': f"{sum_squares:.4f}",
            'Contribution (%)': '100.0'
        })

        budget_data.append({
            'Component': 'EXPANDED (k=2)',
            'Standard Uncertainty (%)': f"{expanded:.2f}",
            'Variance (%)': '-',
            'Contribution (%)': '-'
        })

        budget_df = pd.DataFrame(budget_data)
        st.dataframe(budget_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Reference Modules & Traceability Module v1.0.0 |
        IEC 60904-2:2015 | IEC 60904-4:2019 Compliant
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
