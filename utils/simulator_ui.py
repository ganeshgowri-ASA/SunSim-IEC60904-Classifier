"""
Simulator Selection UI Components
Shared UI components for selecting solar simulator manufacturer and model.

These components are designed to be used across all pages in the application
for consistent simulator selection experience.
"""

import streamlit as st
from typing import Optional, Tuple, Dict, Any
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator_database import (
    SimulatorManufacturer,
    SimulatorDatabase,
    get_simulator_database,
    create_custom_simulator,
    LampType,
    IlluminationMode,
    IrradianceRange,
    get_lamp_type_icon,
    get_classification_color,
    format_test_plane_size,
)


def render_simulator_selector(
    key_prefix: str = "sim",
    show_specs: bool = True,
    show_custom_option: bool = True,
    compact: bool = False
) -> Tuple[Optional[SimulatorManufacturer], Dict[str, Any]]:
    """
    Render the simulator manufacturer/model selection UI in the sidebar.

    Args:
        key_prefix: Prefix for session state keys to avoid conflicts
        show_specs: Whether to display simulator specifications
        show_custom_option: Whether to show the custom simulator input option
        compact: Whether to use a more compact layout

    Returns:
        Tuple of (selected_simulator, metadata_dict)
        metadata_dict includes additional info like simulator_id, custom flag, etc.
    """
    db = get_simulator_database()

    # Initialize session state
    if f'{key_prefix}_manufacturer' not in st.session_state:
        st.session_state[f'{key_prefix}_manufacturer'] = None
    if f'{key_prefix}_model' not in st.session_state:
        st.session_state[f'{key_prefix}_model'] = None
    if f'{key_prefix}_custom_mode' not in st.session_state:
        st.session_state[f'{key_prefix}_custom_mode'] = False

    # Get manufacturers list
    manufacturers = db.get_manufacturers()

    if show_custom_option:
        manufacturers = manufacturers + ["-- Custom Simulator --"]

    # Section header
    if not compact:
        st.markdown("### 🔧 Simulator Selection")
        st.markdown("---")

    # Manufacturer dropdown
    manufacturer_options = ["Select Manufacturer..."] + manufacturers

    selected_manufacturer = st.selectbox(
        "Manufacturer",
        options=manufacturer_options,
        key=f"{key_prefix}_manufacturer_select",
        help="Select the solar simulator manufacturer"
    )

    selected_simulator = None
    metadata = {
        "is_custom": False,
        "simulator_id": "",
        "manufacturer": "",
        "model": ""
    }

    # Handle custom simulator option
    if selected_manufacturer == "-- Custom Simulator --":
        st.session_state[f'{key_prefix}_custom_mode'] = True
        selected_simulator, metadata = _render_custom_simulator_form(key_prefix, compact)

    elif selected_manufacturer and selected_manufacturer != "Select Manufacturer...":
        st.session_state[f'{key_prefix}_custom_mode'] = False

        # Get models for selected manufacturer
        models = db.get_model_names_by_manufacturer(selected_manufacturer)
        model_options = ["Select Model..."] + models

        selected_model = st.selectbox(
            "Model",
            options=model_options,
            key=f"{key_prefix}_model_select",
            help="Select the simulator model"
        )

        if selected_model and selected_model != "Select Model...":
            selected_simulator = db.get_simulator(selected_manufacturer, selected_model)

            if selected_simulator:
                metadata = {
                    "is_custom": False,
                    "simulator_id": f"{selected_manufacturer}-{selected_model}".replace(" ", "-"),
                    "manufacturer": selected_manufacturer,
                    "model": selected_model
                }

                # Display specifications
                if show_specs:
                    _render_simulator_specs(selected_simulator, compact)

    # Save to session state
    if selected_simulator:
        st.session_state[f'{key_prefix}_selected_simulator'] = selected_simulator
        st.session_state[f'{key_prefix}_metadata'] = metadata

    return selected_simulator, metadata


def _render_custom_simulator_form(
    key_prefix: str,
    compact: bool = False
) -> Tuple[Optional[SimulatorManufacturer], Dict[str, Any]]:
    """Render the custom simulator input form"""

    if not compact:
        st.markdown("#### Custom Simulator Configuration")

    with st.expander("⚙️ Enter Custom Simulator Details", expanded=True):
        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            custom_manufacturer = st.text_input(
                "Manufacturer Name",
                key=f"{key_prefix}_custom_manufacturer",
                placeholder="e.g., MyCompany"
            )
        with col2:
            custom_model = st.text_input(
                "Model Name",
                key=f"{key_prefix}_custom_model",
                placeholder="e.g., SS-2000"
            )

        # Lamp type and classification
        col1, col2 = st.columns(2)
        with col1:
            lamp_types = [lt.value for lt in LampType]
            custom_lamp = st.selectbox(
                "Lamp Type",
                options=lamp_types,
                key=f"{key_prefix}_custom_lamp"
            )
        with col2:
            custom_classification = st.text_input(
                "Typical Classification",
                key=f"{key_prefix}_custom_classification",
                placeholder="e.g., AAA or A+A+A+",
                value="AAA"
            )

        # Test plane and irradiance
        col1, col2 = st.columns(2)
        with col1:
            custom_test_plane = st.text_input(
                "Test Plane Size",
                key=f"{key_prefix}_custom_test_plane",
                placeholder="e.g., 200x200mm",
                value="200x200mm"
            )
        with col2:
            illumination_modes = [m.value for m in IlluminationMode]
            custom_mode = st.selectbox(
                "Illumination Mode",
                options=illumination_modes,
                key=f"{key_prefix}_custom_illum_mode"
            )

        # Irradiance range
        col1, col2 = st.columns(2)
        with col1:
            irr_min = st.number_input(
                "Min Irradiance (W/m²)",
                min_value=0.0,
                max_value=2000.0,
                value=100.0,
                key=f"{key_prefix}_custom_irr_min"
            )
        with col2:
            irr_max = st.number_input(
                "Max Irradiance (W/m²)",
                min_value=0.0,
                max_value=2000.0,
                value=1000.0,
                key=f"{key_prefix}_custom_irr_max"
            )

        # Pulse duration (only shown for pulsed mode)
        pulse_duration = None
        if custom_mode in ["Pulsed", "Multi-Flash"]:
            pulse_duration = st.number_input(
                "Pulse Duration (ms)",
                min_value=0.1,
                max_value=500.0,
                value=10.0,
                key=f"{key_prefix}_custom_pulse"
            )

        # Spectral range
        custom_spectral = st.text_input(
            "Spectral Range",
            key=f"{key_prefix}_custom_spectral",
            placeholder="e.g., 300-1200nm",
            value="300-1200nm"
        )

        # Notes
        custom_notes = st.text_area(
            "Notes",
            key=f"{key_prefix}_custom_notes",
            placeholder="Additional specifications or notes...",
            height=68
        )

    # Create simulator if valid inputs
    selected_simulator = None
    metadata = {
        "is_custom": True,
        "simulator_id": "",
        "manufacturer": "",
        "model": ""
    }

    if custom_manufacturer and custom_model:
        try:
            selected_simulator = create_custom_simulator(
                manufacturer_name=custom_manufacturer,
                model_name=custom_model,
                lamp_type=custom_lamp,
                typical_classification=custom_classification,
                test_plane_size=custom_test_plane,
                irradiance_min=irr_min,
                irradiance_max=irr_max,
                illumination_mode=custom_mode,
                pulse_duration_ms=pulse_duration,
                spectral_range_nm=custom_spectral,
                notes=custom_notes
            )

            metadata = {
                "is_custom": True,
                "simulator_id": f"CUSTOM-{custom_manufacturer}-{custom_model}".replace(" ", "-"),
                "manufacturer": custom_manufacturer,
                "model": custom_model
            }

            # Validate and show warnings
            db = get_simulator_database()
            is_valid, messages = db.validate_simulator(selected_simulator)

            if messages:
                for msg in messages:
                    if msg.startswith("Error"):
                        st.error(msg)
                    else:
                        st.warning(msg)

            # Show specs for custom simulator
            if is_valid:
                _render_simulator_specs(selected_simulator, compact, is_custom=True)

        except Exception as e:
            st.error(f"Error creating custom simulator: {str(e)}")

    return selected_simulator, metadata


def _render_simulator_specs(
    simulator: SimulatorManufacturer,
    compact: bool = False,
    is_custom: bool = False
) -> None:
    """Render the simulator specifications display"""

    if not compact:
        st.markdown("---")
        st.markdown("#### Simulator Specifications")

    # Classification badge with color
    class_color = get_classification_color(simulator.typical_classification)
    lamp_icon = get_lamp_type_icon(simulator.lamp_type)

    # Main info card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid {class_color};
    ">
        <div style="font-weight: 600; color: #1e3a5f; font-size: 1rem;">
            {lamp_icon} {simulator.full_name}
        </div>
        <div style="
            display: inline-block;
            background: {class_color};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        ">
            Class {simulator.typical_classification}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Specs in expandable section
    with st.expander("📋 View Full Specifications", expanded=not compact):
        specs_data = {
            "🔆 Lamp Type": simulator.lamp_type.value,
            "📐 Test Plane": format_test_plane_size(simulator.test_plane_size),
            "☀️ Irradiance": str(simulator.irradiance_range),
            "⏱️ Mode": simulator.illumination_mode.value,
            "🌈 Spectral Range": simulator.spectral_range_nm,
        }

        if simulator.is_pulsed and simulator.pulse_duration_ms:
            specs_data["⚡ Pulse Duration"] = f"{simulator.pulse_duration_ms} ms"

        # Display specs in two columns
        col1, col2 = st.columns(2)
        items = list(specs_data.items())

        for i, (label, value) in enumerate(items):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"**{label}**")
                st.markdown(f"{value}")

        if simulator.notes:
            st.markdown("**📝 Notes**")
            st.markdown(f"_{simulator.notes}_")


def render_simulator_summary_card(
    simulator: SimulatorManufacturer,
    show_classification: bool = True
) -> None:
    """
    Render a compact simulator summary card for display in main content area.

    Args:
        simulator: The simulator to display
        show_classification: Whether to show classification badge
    """
    class_color = get_classification_color(simulator.typical_classification)
    lamp_icon = get_lamp_type_icon(simulator.lamp_type)

    classification_html = ""
    if show_classification:
        classification_html = f"""
        <span style="
            background: {class_color};
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        ">
            {simulator.typical_classification}
        </span>
        """

    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <span style="font-size: 1.1rem; font-weight: 600; color: #1e3a5f;">
                    {lamp_icon} {simulator.full_name}
                </span>
                {classification_html}
            </div>
        </div>
        <div style="
            display: flex;
            gap: 1.5rem;
            margin-top: 0.5rem;
            color: #64748b;
            font-size: 0.85rem;
        ">
            <span>📐 {format_test_plane_size(simulator.test_plane_size)}</span>
            <span>☀️ {simulator.irradiance_range}</span>
            <span>⏱️ {simulator.illumination_mode.value}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_selected_simulator(key_prefix: str = "sim") -> Optional[SimulatorManufacturer]:
    """
    Get the currently selected simulator from session state.

    Args:
        key_prefix: The prefix used when rendering the selector

    Returns:
        The selected simulator or None
    """
    return st.session_state.get(f'{key_prefix}_selected_simulator')


def get_selected_simulator_metadata(key_prefix: str = "sim") -> Dict[str, Any]:
    """
    Get metadata for the currently selected simulator.

    Args:
        key_prefix: The prefix used when rendering the selector

    Returns:
        Dictionary with simulator metadata
    """
    return st.session_state.get(f'{key_prefix}_metadata', {
        "is_custom": False,
        "simulator_id": "",
        "manufacturer": "",
        "model": ""
    })


def get_simulator_id_for_db(key_prefix: str = "sim") -> str:
    """
    Get a database-friendly simulator ID string.

    Args:
        key_prefix: The prefix used when rendering the selector

    Returns:
        Formatted simulator ID string for database storage
    """
    metadata = get_selected_simulator_metadata(key_prefix)
    return metadata.get("simulator_id", "UNKNOWN")


def render_simulator_filter_options(key_prefix: str = "filter") -> Dict[str, Any]:
    """
    Render filtering options for searching simulators.

    Args:
        key_prefix: Prefix for session state keys

    Returns:
        Dictionary of filter options
    """
    st.markdown("#### 🔍 Filter Simulators")

    filters = {}

    # Lamp type filter
    lamp_types = ["All"] + [lt.value for lt in LampType]
    selected_lamp = st.selectbox(
        "Lamp Type",
        options=lamp_types,
        key=f"{key_prefix}_lamp_filter"
    )
    if selected_lamp != "All":
        filters["lamp_type"] = LampType(selected_lamp)

    # Illumination mode filter
    modes = ["All"] + [m.value for m in IlluminationMode]
    selected_mode = st.selectbox(
        "Illumination Mode",
        options=modes,
        key=f"{key_prefix}_mode_filter"
    )
    if selected_mode != "All":
        filters["illumination_mode"] = IlluminationMode(selected_mode)

    # Classification filter
    classifications = ["All", "A+", "A", "B", "C"]
    selected_class = st.selectbox(
        "Classification Grade",
        options=classifications,
        key=f"{key_prefix}_class_filter"
    )
    if selected_class != "All":
        filters["classification_grade"] = selected_class

    # Minimum area filter
    min_area = st.number_input(
        "Min Test Area (mm²)",
        min_value=0,
        max_value=10000000,
        value=0,
        step=1000,
        key=f"{key_prefix}_area_filter"
    )
    if min_area > 0:
        filters["min_area_mm2"] = float(min_area)

    return filters


def search_and_display_simulators(filters: Dict[str, Any]) -> None:
    """
    Search simulators with given filters and display results.

    Args:
        filters: Dictionary of filter options from render_simulator_filter_options
    """
    db = get_simulator_database()
    results = db.search_simulators(**filters)

    st.markdown(f"**Found {len(results)} simulators**")

    for sim in results:
        render_simulator_summary_card(sim)
