"""
Sun Simulator Classification System - Capacitive Effects Module
IEC 60904-14 Compliance: Module Capacitance Measurement

This module provides calculations and analysis for capacitive effects
in PV modules during I-V curve measurements, including sweep rate
optimization and correction factors.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class ConnectionType(Enum):
    """PV module connection types for capacitance measurement."""
    TWO_WIRE = "2-wire"
    FOUR_WIRE = "4-wire"  # Kelvin connection - recommended


class SweepDirection(Enum):
    """I-V sweep direction."""
    ISC_TO_VOC = "isc_to_voc"  # Short-circuit to open-circuit
    VOC_TO_ISC = "voc_to_isc"  # Open-circuit to short-circuit


# IEC 60904-14 recommended sweep rate limits
SWEEP_RATE_LIMITS = {
    'min_ms_per_point': 0.001,  # 1 µs minimum
    'max_ms_per_point': 100,     # 100 ms maximum
    'recommended_points': 200,   # Recommended I-V points
    'min_points': 50,            # Minimum I-V points
}

# Typical capacitance values by cell technology (nF/cm²)
TYPICAL_CAPACITANCE = {
    'c-Si': {'min': 20, 'max': 80, 'typical': 40},
    'mc-Si': {'min': 15, 'max': 60, 'typical': 35},
    'PERC': {'min': 30, 'max': 100, 'typical': 60},
    'HJT': {'min': 50, 'max': 150, 'typical': 80},
    'TOPCon': {'min': 40, 'max': 120, 'typical': 70},
    'CdTe': {'min': 5, 'max': 30, 'typical': 15},
    'CIGS': {'min': 10, 'max': 50, 'typical': 25},
    'a-Si': {'min': 50, 'max': 200, 'typical': 100},
    'Perovskite': {'min': 100, 'max': 500, 'typical': 250},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CapacitanceResult:
    """Result of capacitance measurement/calculation."""
    capacitance_nF: float  # Total module capacitance in nF
    capacitance_per_area: float  # nF/cm²
    cell_area_cm2: float
    num_cells: int
    technology: str
    measurement_voltage: float  # V
    measurement_frequency: float  # Hz (if AC method used)
    method: str  # 'transient', 'ac_impedance', 'estimated'
    uncertainty_percent: float
    is_within_typical: bool
    notes: List[str]


@dataclass
class SweepRateOptimization:
    """Optimized sweep rate parameters."""
    recommended_sweep_time_ms: float
    min_sweep_time_ms: float
    max_sweep_time_ms: float
    recommended_points: int
    time_per_point_ms: float
    sweep_direction: SweepDirection
    capacitive_error_percent: float
    notes: List[str]


@dataclass
class IVCorrectionFactors:
    """Correction factors for capacitive effects."""
    current_correction_A: float
    power_correction_W: float
    fill_factor_correction: float
    efficiency_correction_percent: float
    corrected_isc: float
    corrected_voc: float
    corrected_pmax: float
    corrected_ff: float
    notes: List[str]


@dataclass
class FourWireGuide:
    """Four-terminal (Kelvin) connection guide."""
    sense_wire_placement: str
    force_wire_placement: str
    contact_resistance_max_ohm: float
    wire_gauge_recommendation: str
    connection_diagram: str
    troubleshooting_tips: List[str]


# =============================================================================
# CAPACITANCE MEASUREMENT FUNCTIONS
# =============================================================================

def estimate_module_capacitance(
    technology: str,
    cell_area_cm2: float,
    num_cells: int,
    voltage_bias: float = 0.0
) -> CapacitanceResult:
    """
    Estimate module capacitance based on cell technology and area.

    For series-connected cells: C_total = C_cell / n
    Capacitance varies with voltage (junction capacitance).

    Args:
        technology: Cell technology type
        cell_area_cm2: Active cell area in cm²
        num_cells: Number of cells in series
        voltage_bias: Voltage bias point for estimation (V)

    Returns:
        CapacitanceResult with estimated capacitance values
    """
    notes = []

    # Get typical capacitance for technology
    if technology in TYPICAL_CAPACITANCE:
        cap_data = TYPICAL_CAPACITANCE[technology]
        cap_per_area = cap_data['typical']
        notes.append(f"Using typical values for {technology} technology")
    else:
        # Default to c-Si values
        cap_data = TYPICAL_CAPACITANCE['c-Si']
        cap_per_area = cap_data['typical']
        notes.append(f"Technology '{technology}' not found, using c-Si defaults")

    # Calculate single cell capacitance
    cell_capacitance_nF = cap_per_area * cell_area_cm2

    # For series-connected cells, total capacitance is reduced
    # C_total = C_cell / n (for n cells in series)
    total_capacitance_nF = cell_capacitance_nF / num_cells

    # Voltage-dependent correction (junction capacitance decreases with forward bias)
    if voltage_bias > 0:
        # Approximate depletion capacitance reduction
        voltage_factor = 1 / (1 + voltage_bias / (0.6 * num_cells)) ** 0.5
        total_capacitance_nF *= voltage_factor
        notes.append(f"Applied voltage-dependent correction (V_bias={voltage_bias:.1f}V)")

    # Calculate effective capacitance per area
    total_area = cell_area_cm2 * num_cells
    effective_cap_per_area = (total_capacitance_nF / total_area) * num_cells

    # Check if within typical range
    is_within_typical = cap_data['min'] <= effective_cap_per_area <= cap_data['max']

    if not is_within_typical:
        notes.append("Estimated value outside typical range - consider measurement")

    return CapacitanceResult(
        capacitance_nF=total_capacitance_nF,
        capacitance_per_area=effective_cap_per_area,
        cell_area_cm2=cell_area_cm2,
        num_cells=num_cells,
        technology=technology,
        measurement_voltage=voltage_bias,
        measurement_frequency=0,  # Not applicable for estimation
        method='estimated',
        uncertainty_percent=30.0,  # High uncertainty for estimation
        is_within_typical=is_within_typical,
        notes=notes
    )


def calculate_capacitance_from_transient(
    voltage_data: np.ndarray,
    current_data: np.ndarray,
    time_data: np.ndarray,
    irradiance_W_m2: float = 1000.0
) -> CapacitanceResult:
    """
    Calculate capacitance from transient I-V response.

    Uses the relationship: C = I_cap / (dV/dt)
    where I_cap = I_measured - I_photo

    Args:
        voltage_data: Voltage measurements (V)
        current_data: Current measurements (A)
        time_data: Time stamps (s)
        irradiance_W_m2: Irradiance during measurement

    Returns:
        CapacitanceResult with measured capacitance
    """
    notes = []

    # Calculate dV/dt
    dv_dt = np.gradient(voltage_data, time_data)

    # Estimate photogenerated current (steady-state current at low voltage)
    # Use the current at minimum voltage as approximation
    min_v_idx = np.argmin(np.abs(voltage_data))
    i_photo = current_data[min_v_idx]

    # Capacitive current
    i_capacitive = current_data - i_photo

    # Calculate capacitance at each point
    with np.errstate(divide='ignore', invalid='ignore'):
        capacitance = np.abs(i_capacitive / dv_dt)
        capacitance = np.nan_to_num(capacitance, nan=0, posinf=0, neginf=0)

    # Use median value (more robust to outliers)
    valid_mask = (capacitance > 0) & (capacitance < 1e-3)  # Filter unrealistic values
    if np.sum(valid_mask) > 10:
        capacitance_F = np.median(capacitance[valid_mask])
    else:
        capacitance_F = np.median(capacitance[capacitance > 0])
        notes.append("Limited valid data points for capacitance calculation")

    capacitance_nF = capacitance_F * 1e9

    # Estimate uncertainty from spread of values
    if np.sum(valid_mask) > 10:
        std_cap = np.std(capacitance[valid_mask])
        uncertainty = (std_cap / capacitance_F) * 100 if capacitance_F > 0 else 50.0
    else:
        uncertainty = 50.0

    notes.append("Calculated from transient voltage response")
    notes.append(f"Used {np.sum(valid_mask)} valid data points")

    return CapacitanceResult(
        capacitance_nF=capacitance_nF,
        capacitance_per_area=0,  # Cannot determine without module info
        cell_area_cm2=0,
        num_cells=0,
        technology='unknown',
        measurement_voltage=np.mean(voltage_data),
        measurement_frequency=0,
        method='transient',
        uncertainty_percent=min(uncertainty, 100),
        is_within_typical=True,  # Cannot verify without technology info
        notes=notes
    )


# =============================================================================
# SWEEP RATE OPTIMIZATION
# =============================================================================

def optimize_sweep_rate(
    module_capacitance_nF: float,
    voc: float,
    isc: float,
    target_error_percent: float = 0.5,
    num_points: int = 200
) -> SweepRateOptimization:
    """
    Optimize I-V sweep rate to minimize capacitive effects.

    Based on IEC 60904-14 guidelines for sweep rate selection.

    Args:
        module_capacitance_nF: Module capacitance in nF
        voc: Open-circuit voltage (V)
        isc: Short-circuit current (A)
        target_error_percent: Target capacitive error (%)
        num_points: Number of I-V points to acquire

    Returns:
        SweepRateOptimization with recommended parameters
    """
    notes = []
    capacitance_F = module_capacitance_nF * 1e-9

    # Calculate characteristic time constant
    # tau = C * V / I (approximate)
    tau_s = capacitance_F * voc / isc if isc > 0 else 0.001
    tau_ms = tau_s * 1000

    notes.append(f"Module time constant: {tau_ms:.3f} ms")

    # For target error, sweep time should be much greater than tau
    # Error ≈ (tau / t_sweep) * 100%
    # t_sweep = tau * 100 / target_error
    min_sweep_time_ms = tau_ms * 100 / target_error_percent

    # Recommended sweep time (3x minimum for margin)
    recommended_sweep_time_ms = min_sweep_time_ms * 3

    # Maximum sweep time (limited by LTI considerations)
    max_sweep_time_ms = min(10000, recommended_sweep_time_ms * 10)  # Max 10 seconds

    # Time per point
    time_per_point = recommended_sweep_time_ms / num_points

    # Check against limits
    if time_per_point < SWEEP_RATE_LIMITS['min_ms_per_point']:
        notes.append("Warning: Time per point below minimum - increase sweep time")
        recommended_sweep_time_ms = SWEEP_RATE_LIMITS['min_ms_per_point'] * num_points
        time_per_point = recommended_sweep_time_ms / num_points

    if time_per_point > SWEEP_RATE_LIMITS['max_ms_per_point']:
        notes.append("Warning: Time per point exceeds maximum - reduce sweep time or add points")

    # Calculate expected capacitive error
    capacitive_error = (tau_ms / recommended_sweep_time_ms) * 100

    # Recommend sweep direction (Voc to Isc generally preferred)
    sweep_direction = SweepDirection.VOC_TO_ISC
    notes.append("Voc-to-Isc sweep direction recommended for most modules")

    return SweepRateOptimization(
        recommended_sweep_time_ms=recommended_sweep_time_ms,
        min_sweep_time_ms=min_sweep_time_ms,
        max_sweep_time_ms=max_sweep_time_ms,
        recommended_points=num_points,
        time_per_point_ms=time_per_point,
        sweep_direction=sweep_direction,
        capacitive_error_percent=capacitive_error,
        notes=notes
    )


def calculate_sweep_rate_for_module(
    module_power_W: float,
    technology: str = 'c-Si',
    target_accuracy: str = 'high'
) -> Dict[str, Any]:
    """
    Calculate recommended sweep rate based on module power rating.

    Simplified calculation for quick estimates.

    Args:
        module_power_W: Module rated power (W)
        technology: Cell technology
        target_accuracy: 'high', 'medium', or 'low'

    Returns:
        Dictionary with sweep rate recommendations
    """
    # Estimate Voc and Isc from power
    # Typical module: Vmp ≈ 0.82 * Voc, Imp ≈ 0.95 * Isc
    # Pmax ≈ Vmp * Imp ≈ 0.78 * Voc * Isc
    # Assume FF ≈ 0.78

    # For residential modules: ~40V Voc, varying current
    if module_power_W < 100:
        voc = 25
        isc = module_power_W / (0.78 * 25)
    elif module_power_W < 400:
        voc = 40
        isc = module_power_W / (0.78 * 40)
    else:
        voc = 50
        isc = module_power_W / (0.78 * 50)

    # Estimate capacitance
    cells_estimate = int(voc / 0.65)  # ~0.65V per cell
    cell_area = 166 * 166 / 100  # Half-cut cells, convert mm² to cm²

    cap_result = estimate_module_capacitance(technology, cell_area, cells_estimate)

    # Target error based on accuracy level
    target_errors = {'high': 0.3, 'medium': 0.5, 'low': 1.0}
    target_error = target_errors.get(target_accuracy, 0.5)

    sweep_opt = optimize_sweep_rate(
        cap_result.capacitance_nF, voc, isc, target_error
    )

    return {
        'estimated_voc': voc,
        'estimated_isc': isc,
        'estimated_capacitance_nF': cap_result.capacitance_nF,
        'recommended_sweep_time_ms': sweep_opt.recommended_sweep_time_ms,
        'time_per_point_ms': sweep_opt.time_per_point_ms,
        'expected_error_percent': sweep_opt.capacitive_error_percent,
        'technology': technology,
        'accuracy_level': target_accuracy
    }


# =============================================================================
# CORRECTION FACTORS
# =============================================================================

def calculate_correction_factors(
    voltage_forward: np.ndarray,
    current_forward: np.ndarray,
    voltage_reverse: np.ndarray,
    current_reverse: np.ndarray,
    module_capacitance_nF: float,
    sweep_time_ms: float
) -> IVCorrectionFactors:
    """
    Calculate correction factors from forward and reverse sweep comparison.

    IEC 60904-14 recommends comparing forward and reverse sweeps to
    quantify and correct for capacitive effects.

    Args:
        voltage_forward: Voltage data, forward sweep (V)
        current_forward: Current data, forward sweep (A)
        voltage_reverse: Voltage data, reverse sweep (V)
        current_reverse: Current data, reverse sweep (A)
        module_capacitance_nF: Module capacitance (nF)
        sweep_time_ms: Sweep time (ms)

    Returns:
        IVCorrectionFactors with correction values
    """
    notes = []

    # Interpolate reverse sweep to match forward sweep voltages
    current_reverse_interp = np.interp(
        voltage_forward, voltage_reverse[::-1], current_reverse[::-1]
    )

    # The true current is the average of forward and reverse
    # (capacitive effects are opposite in direction)
    current_corrected = (current_forward + current_reverse_interp) / 2

    # Current correction is the difference
    current_diff = current_forward - current_corrected
    current_correction = np.max(np.abs(current_diff))

    # Find MPP for both curves
    power_forward = voltage_forward * current_forward
    power_corrected = voltage_forward * current_corrected

    idx_mpp_forward = np.argmax(power_forward)
    idx_mpp_corrected = np.argmax(power_corrected)

    pmax_forward = power_forward[idx_mpp_forward]
    pmax_corrected = power_corrected[idx_mpp_corrected]
    power_correction = pmax_corrected - pmax_forward

    # Calculate Isc and Voc
    isc_forward = current_forward[np.argmin(np.abs(voltage_forward))]
    isc_corrected = current_corrected[np.argmin(np.abs(voltage_forward))]

    voc_idx_forward = np.argmin(np.abs(current_forward))
    voc_idx_corrected = np.argmin(np.abs(current_corrected))
    voc_forward = voltage_forward[voc_idx_forward]
    voc_corrected = voltage_forward[voc_idx_corrected]

    # Fill factor
    ff_forward = pmax_forward / (isc_forward * voc_forward) if (isc_forward * voc_forward) > 0 else 0
    ff_corrected = pmax_corrected / (isc_corrected * voc_corrected) if (isc_corrected * voc_corrected) > 0 else 0
    ff_correction = ff_corrected - ff_forward

    # Efficiency correction (relative)
    eff_correction = ((pmax_corrected - pmax_forward) / pmax_forward * 100) if pmax_forward > 0 else 0

    notes.append("Corrections calculated from forward/reverse sweep comparison")
    notes.append(f"Max current deviation: {current_correction*1000:.2f} mA")

    return IVCorrectionFactors(
        current_correction_A=current_correction,
        power_correction_W=power_correction,
        fill_factor_correction=ff_correction,
        efficiency_correction_percent=eff_correction,
        corrected_isc=isc_corrected,
        corrected_voc=voc_corrected,
        corrected_pmax=pmax_corrected,
        corrected_ff=ff_corrected,
        notes=notes
    )


def apply_capacitive_correction(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    capacitance_nF: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply capacitive correction to single-direction I-V sweep.

    Args:
        voltage: Voltage measurements (V)
        current: Current measurements (A)
        time: Time stamps (s)
        capacitance_nF: Module capacitance (nF)

    Returns:
        Tuple of (voltage_corrected, current_corrected)
    """
    capacitance_F = capacitance_nF * 1e-9

    # Calculate dV/dt
    dv_dt = np.gradient(voltage, time)

    # Capacitive current: I_cap = C * dV/dt
    i_capacitive = capacitance_F * dv_dt

    # Corrected current (remove capacitive component)
    current_corrected = current - i_capacitive

    return voltage, current_corrected


# =============================================================================
# FOUR-TERMINAL CONNECTION GUIDE
# =============================================================================

def get_four_wire_connection_guide(
    module_current_A: float,
    cable_length_m: float = 2.0
) -> FourWireGuide:
    """
    Generate four-terminal (Kelvin) connection guide.

    Args:
        module_current_A: Expected maximum current (A)
        cable_length_m: Cable length to module (m)

    Returns:
        FourWireGuide with connection recommendations
    """
    # Wire gauge recommendations based on current
    if module_current_A < 5:
        wire_gauge = "18 AWG (0.82 mm²)"
        contact_resistance_max = 0.01  # 10 mΩ
    elif module_current_A < 15:
        wire_gauge = "14 AWG (2.08 mm²)"
        contact_resistance_max = 0.005  # 5 mΩ
    elif module_current_A < 30:
        wire_gauge = "10 AWG (5.26 mm²)"
        contact_resistance_max = 0.003  # 3 mΩ
    else:
        wire_gauge = "8 AWG (8.37 mm²) or larger"
        contact_resistance_max = 0.002  # 2 mΩ

    # Connection diagram (ASCII art)
    diagram = """
    Four-Terminal (Kelvin) Connection Configuration
    ================================================

    Module Terminal (+)
         |
         +----[Force+ (High Current)]----> Current Source (+)
         |
         +----[Sense+ (Voltage)]----------> Voltmeter (+)

    Module Terminal (-)
         |
         +----[Force- (High Current)]----> Current Source (-)
         |
         +----[Sense- (Voltage)]----------> Voltmeter (-)

    Key Points:
    - Sense wires connect at the MODULE terminals, not the force wires
    - Sense wires carry minimal current (< 1 µA typical)
    - Force wires carry the full module current
    - Keep sense and force wire pairs twisted/shielded
    """

    troubleshooting = [
        "Verify sense wires connect directly to module terminals",
        "Check for oxidation or corrosion at contact points",
        "Ensure force wire connections are secure and low-resistance",
        "Use shielded cables to minimize noise pickup",
        "Measure contact resistance before testing (should be < specified max)",
        "Clean contacts with isopropyl alcohol if resistance is high",
        "Avoid running sense wires parallel to AC power cables",
        "Use spring-loaded or pneumatic probes for repeatable contact"
    ]

    return FourWireGuide(
        sense_wire_placement="Connect directly to module terminals, inside force wire connections",
        force_wire_placement="Connect to outer contact points, carry full current",
        contact_resistance_max_ohm=contact_resistance_max,
        wire_gauge_recommendation=wire_gauge,
        connection_diagram=diagram,
        troubleshooting_tips=troubleshooting
    )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_capacitive_effects(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    technology: str = 'c-Si',
    cell_area_cm2: float = 166 * 166 / 100,
    num_cells: int = 60
) -> Dict[str, Any]:
    """
    Comprehensive capacitive effects analysis.

    Args:
        voltage: I-V voltage data (V)
        current: I-V current data (A)
        time: Time stamps (s)
        technology: Cell technology
        cell_area_cm2: Cell area in cm²
        num_cells: Number of cells in series

    Returns:
        Dictionary with complete analysis results
    """
    # Estimate module parameters
    isc = current[np.argmin(np.abs(voltage))]
    voc_idx = np.argmin(np.abs(current))
    voc = voltage[voc_idx]

    # Estimate capacitance
    cap_estimate = estimate_module_capacitance(
        technology, cell_area_cm2, num_cells, voc / 2
    )

    # Try to measure from transient
    try:
        cap_measured = calculate_capacitance_from_transient(
            voltage, current, time
        )
        use_measured = cap_measured.uncertainty_percent < cap_estimate.uncertainty_percent
    except Exception:
        cap_measured = None
        use_measured = False

    capacitance_nF = cap_measured.capacitance_nF if use_measured else cap_estimate.capacitance_nF

    # Calculate sweep time
    sweep_time_ms = (time[-1] - time[0]) * 1000

    # Optimize sweep rate
    sweep_opt = optimize_sweep_rate(capacitance_nF, voc, isc)

    # Apply correction
    _, current_corrected = apply_capacitive_correction(
        voltage, current, time, capacitance_nF
    )

    # Calculate power curves
    power_original = voltage * current
    power_corrected = voltage * current_corrected

    pmax_original = np.max(power_original)
    pmax_corrected = np.max(power_corrected)

    # Get connection guide
    connection_guide = get_four_wire_connection_guide(isc)

    return {
        'capacitance': {
            'estimated': cap_estimate,
            'measured': cap_measured,
            'used_value_nF': capacitance_nF,
            'method': 'measured' if use_measured else 'estimated'
        },
        'sweep_analysis': {
            'actual_sweep_time_ms': sweep_time_ms,
            'recommended_sweep_time_ms': sweep_opt.recommended_sweep_time_ms,
            'is_sweep_adequate': sweep_time_ms >= sweep_opt.min_sweep_time_ms,
            'expected_error_percent': sweep_opt.capacitive_error_percent
        },
        'correction': {
            'pmax_original_W': pmax_original,
            'pmax_corrected_W': pmax_corrected,
            'power_difference_W': pmax_corrected - pmax_original,
            'power_difference_percent': (pmax_corrected - pmax_original) / pmax_original * 100 if pmax_original > 0 else 0,
            'current_corrected': current_corrected
        },
        'recommendations': sweep_opt.notes + [
            f"Module capacitance: {capacitance_nF:.1f} nF",
            f"Actual sweep: {sweep_time_ms:.1f} ms, Recommended: {sweep_opt.recommended_sweep_time_ms:.1f} ms"
        ],
        'four_wire_guide': connection_guide
    }


def get_capacitance_summary_table(technologies: List[str] = None) -> List[Dict[str, Any]]:
    """
    Get summary table of typical capacitance values by technology.

    Args:
        technologies: List of technologies to include (None for all)

    Returns:
        List of dictionaries with capacitance data
    """
    if technologies is None:
        technologies = list(TYPICAL_CAPACITANCE.keys())

    summary = []
    for tech in technologies:
        if tech in TYPICAL_CAPACITANCE:
            data = TYPICAL_CAPACITANCE[tech]
            summary.append({
                'Technology': tech,
                'Min (nF/cm²)': data['min'],
                'Typical (nF/cm²)': data['typical'],
                'Max (nF/cm²)': data['max'],
                'Relative': 'Low' if data['typical'] < 30 else ('High' if data['typical'] > 80 else 'Medium')
            })

    return summary
