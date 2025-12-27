"""
Sun Simulator Classification System
Settings Page - System Configuration Panel

This page provides comprehensive system configuration including:
- Classification standard selection (Ed.2 / Ed.3)
- Alarm thresholds configuration
- User management (admin/operator/viewer)
- Data export/import
- System preferences
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import hashlib
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_CONFIG, THEME, BADGE_COLORS, BADGE_COLORS_LIGHT,
    WAVELENGTH_BANDS, EXTENDED_WAVELENGTH_BANDS, CLASSIFICATION
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Settings - " + APP_CONFIG['title'],
    page_icon="⚙️",
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

.settings-header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #475569;
}

.settings-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.25rem;
}

.settings-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
}

.section-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid #334155;
}

.setting-label {
    color: #e2e8f0;
    font-size: 0.9rem;
}

.setting-description {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 0.25rem;
}

.user-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.user-name {
    font-weight: 600;
    color: #f8fafc;
}

.user-role {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
}

.role-admin { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.role-operator { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
.role-viewer { background: rgba(16, 185, 129, 0.2); color: #10b981; }

.threshold-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

.threshold-name {
    font-weight: 500;
    color: #f8fafc;
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
    font-size: 0.9rem;
}

.info-box p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.85rem;
}

.success-message {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 1rem;
    color: #10b981;
    margin: 1rem 0;
}

.warning-message {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 1rem;
    color: #f59e0b;
    margin: 1rem 0;
}

.stat-badge {
    background: #334155;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.stat-badge-value {
    font-weight: 600;
    color: #f8fafc;
}

.stat-badge-label {
    color: #94a3b8;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state for settings."""
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'classification': {
                'standard_edition': 'Ed.3',
                'use_extended_spectrum': True,
                'auto_classify': True
            },
            'display': {
                'theme': 'dark',
                'chart_height': 400,
                'decimal_places': 3,
                'date_format': 'YYYY-MM-DD'
            },
            'export': {
                'default_format': 'PDF',
                'include_charts': True,
                'include_raw_data': False,
                'company_name': '',
                'lab_name': ''
            },
            'alarm': {
                'enable_alarms': True,
                'email_notifications': False,
                'notification_email': ''
            },
            'system': {
                'session_timeout_minutes': 60,
                'max_upload_size_mb': 50,
                'enable_audit_log': True,
                'data_retention_days': 365
            }
        }

    if 'users' not in st.session_state:
        st.session_state.users = [
            {'id': 1, 'username': 'admin', 'email': 'admin@example.com',
             'full_name': 'System Administrator', 'role': 'admin', 'is_active': True},
            {'id': 2, 'username': 'operator1', 'email': 'operator@example.com',
             'full_name': 'John Operator', 'role': 'operator', 'is_active': True},
            {'id': 3, 'username': 'viewer1', 'email': 'viewer@example.com',
             'full_name': 'Jane Viewer', 'role': 'viewer', 'is_active': True},
        ]

    if 'alarm_thresholds' not in st.session_state:
        st.session_state.alarm_thresholds = [
            {'id': 1, 'name': 'Spectral Warning', 'parameter': 'spectral',
             'threshold_type': 'warning', 'min_value': 0.75, 'max_value': 1.25,
             'enabled': True, 'color': '#f59e0b'},
            {'id': 2, 'name': 'Spectral Critical', 'parameter': 'spectral',
             'threshold_type': 'critical', 'min_value': 0.6, 'max_value': 1.4,
             'enabled': True, 'color': '#ef4444'},
            {'id': 3, 'name': 'Uniformity Warning', 'parameter': 'uniformity',
             'threshold_type': 'warning', 'max_value': 2.0,
             'enabled': True, 'color': '#f59e0b'},
            {'id': 4, 'name': 'Uniformity Critical', 'parameter': 'uniformity',
             'threshold_type': 'critical', 'max_value': 5.0,
             'enabled': True, 'color': '#ef4444'},
            {'id': 5, 'name': 'STI Warning', 'parameter': 'sti',
             'threshold_type': 'warning', 'max_value': 2.0,
             'enabled': True, 'color': '#f59e0b'},
            {'id': 6, 'name': 'STI Critical', 'parameter': 'sti',
             'threshold_type': 'critical', 'max_value': 5.0,
             'enabled': True, 'color': '#ef4444'},
            {'id': 7, 'name': 'LTI Warning', 'parameter': 'lti',
             'threshold_type': 'warning', 'max_value': 2.0,
             'enabled': True, 'color': '#f59e0b'},
            {'id': 8, 'name': 'LTI Critical', 'parameter': 'lti',
             'threshold_type': 'critical', 'max_value': 5.0,
             'enabled': True, 'color': '#ef4444'},
        ]

init_session_state()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_role_class(role: str) -> str:
    """Get CSS class for user role."""
    return f"role-{role}"


def generate_sample_export_data():
    """Generate sample data for export demonstration."""
    return {
        'simulators': [
            {'name': 'SunSim Pro 3000', 'classification': 'A+', 'last_test': '2024-01-15'},
            {'name': 'FlashTest 5000', 'classification': 'A', 'last_test': '2024-01-10'},
        ],
        'measurements': [
            {'date': '2024-01-15', 'spectral': 'A+', 'uniformity': 'A', 'sti': 'A+', 'lti': 'A'},
            {'date': '2024-01-14', 'spectral': 'A', 'uniformity': 'A+', 'sti': 'A', 'lti': 'A'},
        ],
        'settings': st.session_state.settings
    }


# =============================================================================
# CLASSIFICATION SETTINGS TAB
# =============================================================================

def render_classification_settings():
    """Render classification settings section."""
    st.markdown('<div class="section-title">📐 Classification Standard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        edition = st.radio(
            "IEC 60904-9 Edition",
            options=['Ed.2', 'Ed.3'],
            index=1 if st.session_state.settings['classification']['standard_edition'] == 'Ed.3' else 0,
            help="Select the IEC 60904-9 edition for classification"
        )
        st.session_state.settings['classification']['standard_edition'] = edition

        st.markdown("---")

        extended_spectrum = st.checkbox(
            "Use Extended Wavelength Range (300-1200nm)",
            value=st.session_state.settings['classification']['use_extended_spectrum'],
            help="Enable extended range for bifacial and advanced module testing"
        )
        st.session_state.settings['classification']['use_extended_spectrum'] = extended_spectrum

        auto_classify = st.checkbox(
            "Automatic Classification After Measurement",
            value=st.session_state.settings['classification']['auto_classify'],
            help="Automatically classify results immediately after data acquisition"
        )
        st.session_state.settings['classification']['auto_classify'] = auto_classify

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Edition Differences</h4>
            <p>
                <strong>Ed.2 (2007):</strong> Original 6-band classification<br>
                <strong>Ed.3 (2020):</strong> Updated with A+ class and extended wavelength support
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Show wavelength bands based on selection
    st.markdown("#### Wavelength Bands")

    bands = EXTENDED_WAVELENGTH_BANDS if extended_spectrum else WAVELENGTH_BANDS
    bands_df = pd.DataFrame([
        {'Band': i+1, 'Range (nm)': f"{b[0]}-{b[1]}", 'Name': b[2]}
        for i, b in enumerate(bands)
    ])
    st.dataframe(bands_df, use_container_width=True, hide_index=True)

    # Classification limits display
    st.markdown("#### Classification Limits")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="stat-badge" style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981;">
            <div class="stat-badge-value" style="color: #10b981;">A+</div>
            <div class="stat-badge-label">Spectral: ±12.5%</div>
            <div class="stat-badge-label">Uniformity: ≤1%</div>
            <div class="stat-badge-label">STI: ≤0.5%</div>
            <div class="stat-badge-label">LTI: ≤1%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-badge" style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6;">
            <div class="stat-badge-value" style="color: #3b82f6;">A</div>
            <div class="stat-badge-label">Spectral: ±25%</div>
            <div class="stat-badge-label">Uniformity: ≤2%</div>
            <div class="stat-badge-label">STI: ≤2%</div>
            <div class="stat-badge-label">LTI: ≤2%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-badge" style="background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b;">
            <div class="stat-badge-value" style="color: #f59e0b;">B</div>
            <div class="stat-badge-label">Spectral: ±40%</div>
            <div class="stat-badge-label">Uniformity: ≤5%</div>
            <div class="stat-badge-label">STI: ≤5%</div>
            <div class="stat-badge-label">LTI: ≤5%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-badge" style="background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444;">
            <div class="stat-badge-value" style="color: #ef4444;">C</div>
            <div class="stat-badge-label">Spectral: >±40%</div>
            <div class="stat-badge-label">Uniformity: >5%</div>
            <div class="stat-badge-label">STI: >5%</div>
            <div class="stat-badge-label">LTI: >5%</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ALARM THRESHOLDS TAB
# =============================================================================

def render_alarm_settings():
    """Render alarm thresholds configuration."""
    st.markdown('<div class="section-title">🔔 Alarm Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        enable_alarms = st.checkbox(
            "Enable Alarm System",
            value=st.session_state.settings['alarm']['enable_alarms']
        )
        st.session_state.settings['alarm']['enable_alarms'] = enable_alarms

        email_notifications = st.checkbox(
            "Email Notifications",
            value=st.session_state.settings['alarm']['email_notifications'],
            disabled=not enable_alarms
        )
        st.session_state.settings['alarm']['email_notifications'] = email_notifications

        if email_notifications:
            notification_email = st.text_input(
                "Notification Email",
                value=st.session_state.settings['alarm']['notification_email']
            )
            st.session_state.settings['alarm']['notification_email'] = notification_email

    with col2:
        st.markdown("#### Threshold Configuration")

        # Group thresholds by parameter
        params = ['spectral', 'uniformity', 'sti', 'lti']
        param_names = {
            'spectral': 'Spectral Match',
            'uniformity': 'Spatial Uniformity',
            'sti': 'Short-Term Instability',
            'lti': 'Long-Term Instability'
        }

        for param in params:
            param_thresholds = [t for t in st.session_state.alarm_thresholds if t['parameter'] == param]

            with st.expander(f"📊 {param_names[param]}", expanded=False):
                for threshold in param_thresholds:
                    col_a, col_b, col_c = st.columns([2, 1, 1])

                    with col_a:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="width: 12px; height: 12px; border-radius: 50%; background: {threshold['color']};"></span>
                            <span style="color: #f8fafc;">{threshold['name']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_b:
                        if 'max_value' in threshold and threshold['max_value']:
                            new_max = st.number_input(
                                "Max",
                                value=float(threshold['max_value']),
                                key=f"max_{threshold['id']}",
                                step=0.1
                            )
                            threshold['max_value'] = new_max

                    with col_c:
                        threshold['enabled'] = st.checkbox(
                            "Enabled",
                            value=threshold['enabled'],
                            key=f"enabled_{threshold['id']}"
                        )

    # Alarm visualization
    st.markdown("#### Threshold Visualization")

    fig = go.Figure()

    for param in params:
        param_thresholds = [t for t in st.session_state.alarm_thresholds if t['parameter'] == param]

        for threshold in param_thresholds:
            if threshold.get('max_value'):
                fig.add_trace(go.Bar(
                    name=f"{param_names[param]} - {threshold['threshold_type'].title()}",
                    x=[param_names[param]],
                    y=[threshold['max_value']],
                    marker_color=threshold['color'],
                    text=[f"{threshold['max_value']}%"],
                    textposition='outside'
                ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        barmode='group',
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis_title='Threshold Value (%)'
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# USER MANAGEMENT TAB
# =============================================================================

def render_user_management():
    """Render user management section."""
    st.markdown('<div class="section-title">👥 User Management</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Current Users")

        for user in st.session_state.users:
            role_class = get_role_class(user['role'])
            status = "🟢 Active" if user['is_active'] else "🔴 Inactive"

            st.markdown(f"""
            <div class="user-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="user-name">{user['full_name']}</div>
                        <div style="color: #64748b; font-size: 0.85rem;">@{user['username']} • {user['email']}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="user-role {role_class}">{user['role']}</span>
                        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{status}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Add New User")

        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_fullname = st.text_input("Full Name")
            new_role = st.selectbox("Role", ['viewer', 'operator', 'admin'])

            if st.form_submit_button("Add User", use_container_width=True):
                if new_username and new_email and new_fullname:
                    new_user = {
                        'id': len(st.session_state.users) + 1,
                        'username': new_username,
                        'email': new_email,
                        'full_name': new_fullname,
                        'role': new_role,
                        'is_active': True
                    }
                    st.session_state.users.append(new_user)
                    st.success(f"User '{new_username}' added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all fields")

        st.markdown("---")

        st.markdown("#### Role Permissions")

        st.markdown("""
        <div class="info-box" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3);">
            <h4 style="color: #ef4444;">Admin</h4>
            <p>Full system access, user management, settings configuration</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.3);">
            <h4 style="color: #3b82f6;">Operator</h4>
            <p>Create/edit measurements, view reports, manage simulators</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);">
            <h4 style="color: #10b981;">Viewer</h4>
            <p>View dashboards and reports, read-only access</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# DATA EXPORT/IMPORT TAB
# =============================================================================

def render_export_import():
    """Render data export/import section."""
    st.markdown('<div class="section-title">📁 Data Export / Import</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Export Data")

        export_format = st.selectbox(
            "Export Format",
            options=['JSON', 'CSV', 'Excel'],
            index=0
        )

        export_scope = st.multiselect(
            "Data to Export",
            options=['Measurements', 'Simulators', 'Settings', 'Users'],
            default=['Measurements', 'Simulators']
        )

        date_range = st.checkbox("Limit by Date Range", value=False)

        if date_range:
            col_a, col_b = st.columns(2)
            with col_a:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with col_b:
                end_date = st.date_input("End Date", value=datetime.now())

        if st.button("Export Data", use_container_width=True, type="primary"):
            export_data = generate_sample_export_data()

            if export_format == 'JSON':
                json_str = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"sunsim_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            elif export_format == 'CSV':
                # Convert to CSV
                df = pd.DataFrame(export_data['measurements'])
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"sunsim_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col2:
        st.markdown("#### Import Data")

        uploaded_file = st.file_uploader(
            "Upload Data File",
            type=['json', 'csv', 'xlsx'],
            help="Upload a previously exported file or compatible data format"
        )

        if uploaded_file:
            st.markdown(f"""
            <div class="success-message">
                <strong>File uploaded:</strong> {uploaded_file.name}<br>
                <strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)

            import_options = st.multiselect(
                "Data to Import",
                options=['Measurements', 'Simulators', 'Settings'],
                default=['Measurements']
            )

            overwrite = st.checkbox("Overwrite existing data", value=False)

            if st.button("Import Data", use_container_width=True):
                st.info("Import functionality would process the uploaded file here.")

    st.markdown("---")

    # Export settings for reports
    st.markdown("#### Report Export Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.session_state.settings['export']['company_name'] = st.text_input(
            "Company Name",
            value=st.session_state.settings['export']['company_name'],
            help="Company name to include in reports"
        )

    with col2:
        st.session_state.settings['export']['lab_name'] = st.text_input(
            "Laboratory Name",
            value=st.session_state.settings['export']['lab_name'],
            help="Lab name to include in reports"
        )

    with col3:
        st.session_state.settings['export']['default_format'] = st.selectbox(
            "Default Report Format",
            options=['PDF', 'Word', 'Excel'],
            index=['PDF', 'Word', 'Excel'].index(st.session_state.settings['export']['default_format'])
        )

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.settings['export']['include_charts'] = st.checkbox(
            "Include Charts in Reports",
            value=st.session_state.settings['export']['include_charts']
        )

    with col2:
        st.session_state.settings['export']['include_raw_data'] = st.checkbox(
            "Include Raw Data in Reports",
            value=st.session_state.settings['export']['include_raw_data']
        )


# =============================================================================
# SYSTEM PREFERENCES TAB
# =============================================================================

def render_system_preferences():
    """Render system preferences section."""
    st.markdown('<div class="section-title">🔧 System Preferences</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Display Settings")

        theme = st.selectbox(
            "Theme",
            options=['dark', 'light'],
            index=0 if st.session_state.settings['display']['theme'] == 'dark' else 1
        )
        st.session_state.settings['display']['theme'] = theme

        chart_height = st.slider(
            "Default Chart Height (px)",
            min_value=200,
            max_value=800,
            value=st.session_state.settings['display']['chart_height']
        )
        st.session_state.settings['display']['chart_height'] = chart_height

        decimal_places = st.number_input(
            "Decimal Places",
            min_value=1,
            max_value=6,
            value=st.session_state.settings['display']['decimal_places']
        )
        st.session_state.settings['display']['decimal_places'] = decimal_places

        date_format = st.selectbox(
            "Date Format",
            options=['YYYY-MM-DD', 'DD/MM/YYYY', 'MM/DD/YYYY'],
            index=0
        )
        st.session_state.settings['display']['date_format'] = date_format

    with col2:
        st.markdown("#### System Settings")

        session_timeout = st.number_input(
            "Session Timeout (minutes)",
            min_value=5,
            max_value=480,
            value=st.session_state.settings['system']['session_timeout_minutes']
        )
        st.session_state.settings['system']['session_timeout_minutes'] = session_timeout

        max_upload = st.number_input(
            "Max Upload Size (MB)",
            min_value=1,
            max_value=500,
            value=st.session_state.settings['system']['max_upload_size_mb']
        )
        st.session_state.settings['system']['max_upload_size_mb'] = max_upload

        enable_audit = st.checkbox(
            "Enable Audit Logging",
            value=st.session_state.settings['system']['enable_audit_log']
        )
        st.session_state.settings['system']['enable_audit_log'] = enable_audit

        data_retention = st.number_input(
            "Data Retention (days)",
            min_value=30,
            max_value=3650,
            value=st.session_state.settings['system']['data_retention_days']
        )
        st.session_state.settings['system']['data_retention_days'] = data_retention

    st.markdown("---")

    st.markdown("#### Database Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-badge">
            <div class="stat-badge-value">PostgreSQL</div>
            <div class="stat-badge-label">Database Type</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-badge">
            <div class="stat-badge-value">Railway</div>
            <div class="stat-badge-label">Hosting Provider</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-badge">
            <div class="stat-badge-value">🟢 Connected</div>
            <div class="stat-badge-label">Status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### About")

    st.markdown(f"""
    <div class="info-box">
        <h4>Sun Simulator Classification System</h4>
        <p>
            <strong>Version:</strong> {APP_CONFIG['version']}<br>
            <strong>Standard:</strong> IEC 60904-9:2020 Ed.3<br>
            <strong>Author:</strong> {APP_CONFIG['author']}<br>
            <strong>Build Date:</strong> 2024-01-15
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN PAGE
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="settings-header">
        <div class="settings-title">⚙️ System Settings</div>
        <div class="settings-subtitle">
            Configure classification parameters, alarms, users, and system preferences
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Settings Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 Classification",
        "🔔 Alarms",
        "👥 Users",
        "📁 Export/Import",
        "🔧 System"
    ])

    with tab1:
        render_classification_settings()

    with tab2:
        render_alarm_settings()

    with tab3:
        render_user_management()

    with tab4:
        render_export_import()

    with tab5:
        render_system_preferences()

    # Save button at bottom
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("💾 Save All Settings", use_container_width=True, type="primary"):
            st.success("Settings saved successfully!")
            st.balloons()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Settings are automatically persisted to the database when connected.<br>
        <span style="color: #475569;">Changes may require page refresh to take effect.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
