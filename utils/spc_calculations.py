"""
Statistical Process Control (SPC) Calculations per ISO 22514.
Includes control chart calculations, capability indices, and run rules detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from scipy import stats


# Control chart constants (for subgroup sizes 2-10)
# A2, D3, D4 factors for X-bar and R charts
CONTROL_CHART_CONSTANTS = {
    2: {'A2': 1.880, 'D3': 0.000, 'D4': 3.267, 'A3': 2.659, 'B3': 0.000, 'B4': 3.267, 'd2': 1.128, 'c4': 0.7979},
    3: {'A2': 1.023, 'D3': 0.000, 'D4': 2.574, 'A3': 1.954, 'B3': 0.000, 'B4': 2.568, 'd2': 1.693, 'c4': 0.8862},
    4: {'A2': 0.729, 'D3': 0.000, 'D4': 2.282, 'A3': 1.628, 'B3': 0.000, 'B4': 2.266, 'd2': 2.059, 'c4': 0.9213},
    5: {'A2': 0.577, 'D3': 0.000, 'D4': 2.114, 'A3': 1.427, 'B3': 0.000, 'B4': 2.089, 'd2': 2.326, 'c4': 0.9400},
    6: {'A2': 0.483, 'D3': 0.000, 'D4': 2.004, 'A3': 1.287, 'B3': 0.030, 'B4': 1.970, 'd2': 2.534, 'c4': 0.9515},
    7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924, 'A3': 1.182, 'B3': 0.118, 'B4': 1.882, 'd2': 2.704, 'c4': 0.9594},
    8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864, 'A3': 1.099, 'B3': 0.185, 'B4': 1.815, 'd2': 2.847, 'c4': 0.9650},
    9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816, 'A3': 1.032, 'B3': 0.239, 'B4': 1.761, 'd2': 2.970, 'c4': 0.9693},
    10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777, 'A3': 0.975, 'B3': 0.284, 'B4': 1.716, 'd2': 3.078, 'c4': 0.9727},
}


@dataclass
class ControlChartResult:
    """Result container for control chart calculations."""
    x_bar: np.ndarray
    r_values: np.ndarray
    s_values: np.ndarray
    x_bar_cl: float
    x_bar_ucl: float
    x_bar_lcl: float
    r_cl: float
    r_ucl: float
    r_lcl: float
    s_cl: float
    s_ucl: float
    s_lcl: float
    subgroup_size: int


@dataclass
class CapabilityResult:
    """Result container for capability calculations."""
    cp: float
    cpk: float
    pp: float
    ppk: float
    cpm: float
    sigma_within: float
    sigma_overall: float
    process_mean: float
    usl: float
    lsl: float
    target: float
    ppm_above_usl: float
    ppm_below_lsl: float
    ppm_total: float
    z_usl: float
    z_lsl: float
    sigma_level: float


@dataclass
class RunRulesResult:
    """Result container for run rules violations."""
    rule_name: str
    description: str
    violated_points: List[int]
    severity: str  # 'warning' or 'violation'


@dataclass
class RefModuleControlLimits:
    """Control limits for reference module monitoring."""
    cl: float  # Center line (mean)
    ucl: float  # Upper control limit (3-sigma)
    lcl: float  # Lower control limit (3-sigma)
    warning_ucl: float  # Upper warning limit (2-sigma)
    warning_lcl: float  # Lower warning limit (2-sigma)
    sigma: float  # Estimated process sigma


@dataclass
class WesternElectricViolation:
    """Western Electric rule violation for reference module analysis."""
    rule_number: int
    rule_name: str
    description: str
    violated_points: List[int]
    severity: str  # 'violation' or 'warning'
    action: str  # Recommended action


@dataclass
class RefModuleAnalysisResult:
    """Complete analysis result for reference module SPC monitoring."""
    values: np.ndarray
    timestamps: Optional[np.ndarray]
    flash_numbers: np.ndarray
    moving_ranges: np.ndarray
    x_ucl: float
    x_lcl: float
    x_cl: float
    x_sigma: float
    mr_ucl: float
    mr_lcl: float
    mr_cl: float
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    ooc_points: List[int]
    warning_points: List[int]
    trend_slope: float
    trend_pvalue: float
    has_significant_trend: bool
    drift_detected: bool
    drift_start_index: Optional[int]


def calculate_subgroup_statistics(data: np.ndarray, subgroup_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate subgroup means, ranges, and standard deviations.

    Args:
        data: Raw measurement data
        subgroup_size: Number of measurements per subgroup

    Returns:
        Tuple of (subgroup means, subgroup ranges, subgroup std devs)
    """
    n_samples = len(data)
    n_subgroups = n_samples // subgroup_size

    # Reshape data into subgroups
    reshaped = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)

    x_bar = np.mean(reshaped, axis=1)
    r_values = np.ptp(reshaped, axis=1)  # Range (max - min)
    s_values = np.std(reshaped, axis=1, ddof=1)  # Sample std dev

    return x_bar, r_values, s_values


def calculate_xbar_r_chart(data: np.ndarray, subgroup_size: int = 5) -> ControlChartResult:
    """
    Calculate X-bar and R control chart limits per ISO 22514.

    Args:
        data: Raw measurement data
        subgroup_size: Number of measurements per subgroup (default 5)

    Returns:
        ControlChartResult with calculated limits
    """
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10")

    constants = CONTROL_CHART_CONSTANTS[subgroup_size]

    x_bar, r_values, s_values = calculate_subgroup_statistics(data, subgroup_size)

    # X-bar chart calculations
    x_bar_cl = np.mean(x_bar)
    r_bar = np.mean(r_values)

    x_bar_ucl = x_bar_cl + constants['A2'] * r_bar
    x_bar_lcl = x_bar_cl - constants['A2'] * r_bar

    # R chart calculations
    r_cl = r_bar
    r_ucl = constants['D4'] * r_bar
    r_lcl = constants['D3'] * r_bar

    # S chart calculations (alternative to R chart)
    s_bar = np.mean(s_values)
    s_cl = s_bar
    s_ucl = constants['B4'] * s_bar
    s_lcl = constants['B3'] * s_bar

    return ControlChartResult(
        x_bar=x_bar,
        r_values=r_values,
        s_values=s_values,
        x_bar_cl=x_bar_cl,
        x_bar_ucl=x_bar_ucl,
        x_bar_lcl=x_bar_lcl,
        r_cl=r_cl,
        r_ucl=r_ucl,
        r_lcl=r_lcl,
        s_cl=s_cl,
        s_ucl=s_ucl,
        s_lcl=s_lcl,
        subgroup_size=subgroup_size
    )


def calculate_xbar_s_chart(data: np.ndarray, subgroup_size: int = 5) -> ControlChartResult:
    """
    Calculate X-bar and S control chart limits per ISO 22514.

    Args:
        data: Raw measurement data
        subgroup_size: Number of measurements per subgroup (default 5)

    Returns:
        ControlChartResult with calculated limits
    """
    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10")

    constants = CONTROL_CHART_CONSTANTS[subgroup_size]

    x_bar, r_values, s_values = calculate_subgroup_statistics(data, subgroup_size)

    # X-bar chart calculations using S
    x_bar_cl = np.mean(x_bar)
    s_bar = np.mean(s_values)

    x_bar_ucl = x_bar_cl + constants['A3'] * s_bar
    x_bar_lcl = x_bar_cl - constants['A3'] * s_bar

    # R chart calculations (for reference)
    r_bar = np.mean(r_values)
    r_cl = r_bar
    r_ucl = constants['D4'] * r_bar
    r_lcl = constants['D3'] * r_bar

    # S chart calculations
    s_cl = s_bar
    s_ucl = constants['B4'] * s_bar
    s_lcl = constants['B3'] * s_bar

    return ControlChartResult(
        x_bar=x_bar,
        r_values=r_values,
        s_values=s_values,
        x_bar_cl=x_bar_cl,
        x_bar_ucl=x_bar_ucl,
        x_bar_lcl=x_bar_lcl,
        r_cl=r_cl,
        r_ucl=r_ucl,
        r_lcl=r_lcl,
        s_cl=s_cl,
        s_ucl=s_ucl,
        s_lcl=s_lcl,
        subgroup_size=subgroup_size
    )


def calculate_individuals_chart(data: np.ndarray) -> Dict:
    """
    Calculate I-MR (Individuals and Moving Range) chart limits.

    Args:
        data: Raw measurement data

    Returns:
        Dictionary with chart statistics
    """
    x_values = data
    mr_values = np.abs(np.diff(data))

    x_cl = np.mean(x_values)
    mr_bar = np.mean(mr_values)

    # Using d2 = 1.128 for moving range of 2
    sigma_estimate = mr_bar / 1.128

    x_ucl = x_cl + 3 * sigma_estimate
    x_lcl = x_cl - 3 * sigma_estimate

    mr_cl = mr_bar
    mr_ucl = 3.267 * mr_bar  # D4 for n=2
    mr_lcl = 0

    return {
        'x_values': x_values,
        'mr_values': mr_values,
        'x_cl': x_cl,
        'x_ucl': x_ucl,
        'x_lcl': x_lcl,
        'mr_cl': mr_cl,
        'mr_ucl': mr_ucl,
        'mr_lcl': mr_lcl,
        'sigma_estimate': sigma_estimate
    }


def calculate_capability_indices(data: np.ndarray, usl: float, lsl: float,
                                  target: Optional[float] = None,
                                  subgroup_size: int = 5) -> CapabilityResult:
    """
    Calculate process capability indices per ISO 22514.

    Args:
        data: Raw measurement data
        usl: Upper Specification Limit
        lsl: Lower Specification Limit
        target: Target value (defaults to midpoint of spec limits)
        subgroup_size: Subgroup size for within-subgroup variation

    Returns:
        CapabilityResult with all capability indices
    """
    if target is None:
        target = (usl + lsl) / 2

    process_mean = np.mean(data)
    overall_std = np.std(data, ddof=1)

    # Calculate within-subgroup standard deviation
    if len(data) >= subgroup_size * 2:
        constants = CONTROL_CHART_CONSTANTS.get(subgroup_size, CONTROL_CHART_CONSTANTS[5])
        x_bar, r_values, s_values = calculate_subgroup_statistics(data, subgroup_size)
        r_bar = np.mean(r_values)
        within_std = r_bar / constants['d2']
    else:
        # Fall back to moving range method
        mr_values = np.abs(np.diff(data))
        within_std = np.mean(mr_values) / 1.128

    spec_range = usl - lsl

    # Potential Capability (Cp) - uses within-subgroup variation
    cp = spec_range / (6 * within_std) if within_std > 0 else 0

    # Demonstrated Capability (Cpk) - accounts for centering
    cpu = (usl - process_mean) / (3 * within_std) if within_std > 0 else 0
    cpl = (process_mean - lsl) / (3 * within_std) if within_std > 0 else 0
    cpk = min(cpu, cpl)

    # Performance indices (Pp, Ppk) - uses overall variation
    pp = spec_range / (6 * overall_std) if overall_std > 0 else 0

    ppu = (usl - process_mean) / (3 * overall_std) if overall_std > 0 else 0
    ppl = (process_mean - lsl) / (3 * overall_std) if overall_std > 0 else 0
    ppk = min(ppu, ppl)

    # Cpm (Taguchi capability index)
    variance_from_target = np.mean((data - target) ** 2)
    cpm = spec_range / (6 * np.sqrt(variance_from_target)) if variance_from_target > 0 else 0

    # Z scores and PPM calculations
    z_usl = (usl - process_mean) / overall_std if overall_std > 0 else 0
    z_lsl = (process_mean - lsl) / overall_std if overall_std > 0 else 0

    ppm_above_usl = stats.norm.sf(z_usl) * 1_000_000
    ppm_below_lsl = stats.norm.cdf(-z_lsl) * 1_000_000
    ppm_total = ppm_above_usl + ppm_below_lsl

    # Sigma level (approximation)
    sigma_level = min(z_usl, z_lsl) + 1.5  # Adding 1.5 sigma shift

    return CapabilityResult(
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        cpm=cpm,
        sigma_within=within_std,
        sigma_overall=overall_std,
        process_mean=process_mean,
        usl=usl,
        lsl=lsl,
        target=target,
        ppm_above_usl=ppm_above_usl,
        ppm_below_lsl=ppm_below_lsl,
        ppm_total=ppm_total,
        z_usl=z_usl,
        z_lsl=z_lsl,
        sigma_level=sigma_level
    )


def detect_run_rules(data: np.ndarray, cl: float, ucl: float, lcl: float) -> List[RunRulesResult]:
    """
    Detect Western Electric / Nelson run rules violations.

    Args:
        data: Control chart data (subgroup means)
        cl: Center line
        ucl: Upper control limit
        lcl: Lower control limit

    Returns:
        List of RunRulesResult for each violated rule
    """
    violations = []
    n = len(data)

    sigma = (ucl - cl) / 3  # Estimate sigma from control limits
    one_sigma_above = cl + sigma
    one_sigma_below = cl - sigma
    two_sigma_above = cl + 2 * sigma
    two_sigma_below = cl - 2 * sigma

    # Rule 1: Point beyond 3 sigma (outside control limits)
    rule1_violations = []
    for i, value in enumerate(data):
        if value > ucl or value < lcl:
            rule1_violations.append(i)
    if rule1_violations:
        violations.append(RunRulesResult(
            rule_name="Rule 1: Beyond 3σ",
            description="Point(s) beyond control limits",
            violated_points=rule1_violations,
            severity="violation"
        ))

    # Rule 2: 8 consecutive points on same side of center line
    if n >= 8:
        rule2_violations = []
        for i in range(n - 7):
            window = data[i:i + 8]
            if all(v > cl for v in window) or all(v < cl for v in window):
                rule2_violations.extend(range(i, i + 8))
        if rule2_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 2: 8 Points Same Side",
                description="8 consecutive points above or below center line",
                violated_points=list(set(rule2_violations)),
                severity="violation"
            ))

    # Rule 3: 6 consecutive points increasing or decreasing
    if n >= 6:
        rule3_violations = []
        for i in range(n - 5):
            window = data[i:i + 6]
            increasing = all(window[j] < window[j + 1] for j in range(5))
            decreasing = all(window[j] > window[j + 1] for j in range(5))
            if increasing or decreasing:
                rule3_violations.extend(range(i, i + 6))
        if rule3_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 3: 6 Points Trending",
                description="6 consecutive points steadily increasing or decreasing",
                violated_points=list(set(rule3_violations)),
                severity="warning"
            ))

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        rule4_violations = []
        for i in range(n - 13):
            window = data[i:i + 14]
            alternating = True
            for j in range(12):
                if j % 2 == 0:  # Even index - should go up
                    if window[j] >= window[j + 1]:
                        alternating = False
                        break
                else:  # Odd index - should go down
                    if window[j] <= window[j + 1]:
                        alternating = False
                        break
            if alternating:
                rule4_violations.extend(range(i, i + 14))
        if rule4_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 4: 14 Points Alternating",
                description="14 consecutive points alternating up and down",
                violated_points=list(set(rule4_violations)),
                severity="warning"
            ))

    # Rule 5: 2 out of 3 points beyond 2 sigma
    if n >= 3:
        rule5_violations = []
        for i in range(n - 2):
            window = data[i:i + 3]
            above_2sigma = sum(1 for v in window if v > two_sigma_above)
            below_2sigma = sum(1 for v in window if v < two_sigma_below)
            if above_2sigma >= 2 or below_2sigma >= 2:
                rule5_violations.extend(range(i, i + 3))
        if rule5_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 5: 2 of 3 Beyond 2σ",
                description="2 out of 3 consecutive points beyond 2 sigma",
                violated_points=list(set(rule5_violations)),
                severity="warning"
            ))

    # Rule 6: 4 out of 5 points beyond 1 sigma
    if n >= 5:
        rule6_violations = []
        for i in range(n - 4):
            window = data[i:i + 5]
            above_1sigma = sum(1 for v in window if v > one_sigma_above)
            below_1sigma = sum(1 for v in window if v < one_sigma_below)
            if above_1sigma >= 4 or below_1sigma >= 4:
                rule6_violations.extend(range(i, i + 5))
        if rule6_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 6: 4 of 5 Beyond 1σ",
                description="4 out of 5 consecutive points beyond 1 sigma",
                violated_points=list(set(rule6_violations)),
                severity="warning"
            ))

    # Rule 7: 15 consecutive points within 1 sigma (stratification)
    if n >= 15:
        rule7_violations = []
        for i in range(n - 14):
            window = data[i:i + 15]
            within_1sigma = all(one_sigma_below <= v <= one_sigma_above for v in window)
            if within_1sigma:
                rule7_violations.extend(range(i, i + 15))
        if rule7_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 7: 15 Points Within 1σ",
                description="15 consecutive points within 1 sigma (stratification)",
                violated_points=list(set(rule7_violations)),
                severity="warning"
            ))

    # Rule 8: 8 consecutive points beyond 1 sigma (mixture)
    if n >= 8:
        rule8_violations = []
        for i in range(n - 7):
            window = data[i:i + 8]
            beyond_1sigma = all(v < one_sigma_below or v > one_sigma_above for v in window)
            if beyond_1sigma:
                rule8_violations.extend(range(i, i + 8))
        if rule8_violations:
            violations.append(RunRulesResult(
                rule_name="Rule 8: 8 Points Beyond 1σ",
                description="8 consecutive points beyond 1 sigma on either side (mixture)",
                violated_points=list(set(rule8_violations)),
                severity="warning"
            ))

    return violations


def generate_sample_data(n_samples: int = 100, mean: float = 100.0,
                         std: float = 2.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate sample SPC data for testing."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(mean, std, n_samples)


def generate_sample_data_with_shifts(n_samples: int = 100, mean: float = 100.0,
                                      std: float = 2.0, shift_points: List[Tuple[int, float]] = None,
                                      seed: Optional[int] = None) -> np.ndarray:
    """Generate sample SPC data with mean shifts for testing run rules."""
    if seed is not None:
        np.random.seed(seed)

    data = np.random.normal(mean, std, n_samples)

    if shift_points:
        for point, shift in shift_points:
            if point < n_samples:
                data[point:] += shift

    return data


def calculate_process_sigma_level(cpk: float) -> float:
    """
    Calculate approximate sigma level from Cpk.

    Args:
        cpk: Process capability index

    Returns:
        Sigma level (with 1.5 sigma shift)
    """
    return 3 * cpk + 1.5


def get_capability_rating(cpk: float) -> Tuple[str, str]:
    """
    Get capability rating and color based on Cpk value.

    Args:
        cpk: Process capability index

    Returns:
        Tuple of (rating, color)
    """
    if cpk >= 2.0:
        return "World Class", "#00D4AA"
    elif cpk >= 1.67:
        return "Excellent", "#00D4AA"
    elif cpk >= 1.33:
        return "Good", "#4ECDC4"
    elif cpk >= 1.0:
        return "Marginal", "#FFE66D"
    elif cpk >= 0.67:
        return "Poor", "#FF6B6B"
    else:
        return "Inadequate", "#FF0000"


# Reference Module Monitoring Functions
def calculate_ref_module_control_limits(
    values: np.ndarray,
    nominal_value: Optional[float] = None
) -> RefModuleControlLimits:
    """
    Calculate control limits for reference module I-MR chart.

    Uses individual values and moving range method appropriate for
    reference module monitoring where each measurement is a single value.

    Args:
        values: Array of measured values (Isc or Pmax)
        nominal_value: Expected nominal value (uses mean if None)

    Returns:
        RefModuleControlLimits with all limit values
    """
    n = len(values)

    # Calculate moving range
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr) if len(mr) > 0 else 0

    # Estimate sigma using d2 = 1.128 for moving range of 2
    sigma = mr_bar / 1.128 if mr_bar > 0 else np.std(values, ddof=1)

    # Center line
    cl = nominal_value if nominal_value is not None else np.mean(values)

    # 3-sigma control limits
    ucl = cl + 3 * sigma
    lcl = cl - 3 * sigma

    # 2-sigma warning limits
    warning_ucl = cl + 2 * sigma
    warning_lcl = cl - 2 * sigma

    return RefModuleControlLimits(
        cl=cl,
        ucl=ucl,
        lcl=lcl,
        warning_ucl=warning_ucl,
        warning_lcl=warning_lcl,
        sigma=sigma
    )


def detect_western_electric_rules(
    data: np.ndarray,
    cl: float,
    ucl: float,
    lcl: float
) -> List[WesternElectricViolation]:
    """
    Detect Western Electric rules violations for reference module monitoring.

    Implements 8 standard Western Electric rules with specific actions
    relevant to flasher/reference module anomaly detection.

    Args:
        data: Individual measurement values
        cl: Center line
        ucl: Upper control limit (3-sigma)
        lcl: Lower control limit (3-sigma)

    Returns:
        List of WesternElectricViolation for each violated rule
    """
    violations = []
    n = len(data)

    sigma = (ucl - cl) / 3
    one_sigma_above = cl + sigma
    one_sigma_below = cl - sigma
    two_sigma_above = cl + 2 * sigma
    two_sigma_below = cl - 2 * sigma

    # Rule 1: Point beyond 3-sigma (outside control limits)
    rule1_violations = []
    for i, value in enumerate(data):
        if value > ucl or value < lcl:
            rule1_violations.append(i)
    if rule1_violations:
        violations.append(WesternElectricViolation(
            rule_number=1,
            rule_name="Beyond 3σ",
            description="Point(s) beyond control limits - likely flasher malfunction or severe degradation",
            violated_points=rule1_violations,
            severity="violation",
            action="STOP: Immediately investigate flasher and reference module. Check for lamp issues, contact problems, or module damage."
        ))

    # Rule 2: 8 consecutive points on same side of center line
    if n >= 8:
        rule2_violations = []
        for i in range(n - 7):
            window = data[i:i + 8]
            if all(v > cl for v in window) or all(v < cl for v in window):
                rule2_violations.extend(range(i, i + 8))
        if rule2_violations:
            violations.append(WesternElectricViolation(
                rule_number=2,
                rule_name="8 Points Same Side",
                description="8 consecutive points above or below center line - systematic shift detected",
                violated_points=list(set(rule2_violations)),
                severity="violation",
                action="Recalibrate the flasher. Check for environmental changes (temperature, ventilation). Review reference module condition."
            ))

    # Rule 3: 6 consecutive points increasing or decreasing (trend)
    if n >= 6:
        rule3_violations = []
        for i in range(n - 5):
            window = data[i:i + 6]
            increasing = all(window[j] < window[j + 1] for j in range(5))
            decreasing = all(window[j] > window[j + 1] for j in range(5))
            if increasing or decreasing:
                rule3_violations.extend(range(i, i + 6))
        if rule3_violations:
            violations.append(WesternElectricViolation(
                rule_number=3,
                rule_name="6 Points Trending",
                description="6 consecutive points steadily increasing or decreasing - gradual drift detected",
                violated_points=list(set(rule3_violations)),
                severity="warning",
                action="Monitor closely. Check flasher lamp aging, reference module degradation, or temperature drift."
            ))

    # Rule 4: 14 consecutive points alternating up and down
    if n >= 14:
        rule4_violations = []
        for i in range(n - 13):
            window = data[i:i + 14]
            alternating = True
            for j in range(12):
                if j % 2 == 0:
                    if window[j] >= window[j + 1]:
                        alternating = False
                        break
                else:
                    if window[j] <= window[j + 1]:
                        alternating = False
                        break
            if alternating:
                rule4_violations.extend(range(i, i + 14))
        if rule4_violations:
            violations.append(WesternElectricViolation(
                rule_number=4,
                rule_name="14 Points Alternating",
                description="14 consecutive points alternating up and down - over-adjustment or control system oscillation",
                violated_points=list(set(rule4_violations)),
                severity="warning",
                action="Check for over-correction in control systems. Review operator procedures."
            ))

    # Rule 5: 2 out of 3 points beyond 2-sigma (same side)
    if n >= 3:
        rule5_violations = []
        for i in range(n - 2):
            window = data[i:i + 3]
            above_2sigma = sum(1 for v in window if v > two_sigma_above)
            below_2sigma = sum(1 for v in window if v < two_sigma_below)
            if above_2sigma >= 2 or below_2sigma >= 2:
                rule5_violations.extend(range(i, i + 3))
        if rule5_violations:
            violations.append(WesternElectricViolation(
                rule_number=5,
                rule_name="2 of 3 Beyond 2σ",
                description="2 out of 3 consecutive points beyond 2 sigma - process shift starting",
                violated_points=list(set(rule5_violations)),
                severity="warning",
                action="Increased monitoring required. Prepare for potential calibration adjustment."
            ))

    # Rule 6: 4 out of 5 points beyond 1-sigma (same side)
    if n >= 5:
        rule6_violations = []
        for i in range(n - 4):
            window = data[i:i + 5]
            above_1sigma = sum(1 for v in window if v > one_sigma_above)
            below_1sigma = sum(1 for v in window if v < one_sigma_below)
            if above_1sigma >= 4 or below_1sigma >= 4:
                rule6_violations.extend(range(i, i + 5))
        if rule6_violations:
            violations.append(WesternElectricViolation(
                rule_number=6,
                rule_name="4 of 5 Beyond 1σ",
                description="4 out of 5 consecutive points beyond 1 sigma - process shift approaching",
                violated_points=list(set(rule6_violations)),
                severity="warning",
                action="Schedule preventive calibration check within next 5-10 flashes."
            ))

    # Rule 7: 15 consecutive points within 1-sigma (stratification)
    if n >= 15:
        rule7_violations = []
        for i in range(n - 14):
            window = data[i:i + 15]
            within_1sigma = all(one_sigma_below <= v <= one_sigma_above for v in window)
            if within_1sigma:
                rule7_violations.extend(range(i, i + 15))
        if rule7_violations:
            violations.append(WesternElectricViolation(
                rule_number=7,
                rule_name="15 Points Within 1σ",
                description="15 consecutive points within 1 sigma - unusually low variation (stratification)",
                violated_points=list(set(rule7_violations)),
                severity="warning",
                action="Verify measurement system resolution. Check for averaged data or improper sampling."
            ))

    # Rule 8: 8 consecutive points beyond 1-sigma on either side (mixture)
    if n >= 8:
        rule8_violations = []
        for i in range(n - 7):
            window = data[i:i + 8]
            beyond_1sigma = all(v < one_sigma_below or v > one_sigma_above for v in window)
            if beyond_1sigma:
                rule8_violations.extend(range(i, i + 8))
        if rule8_violations:
            violations.append(WesternElectricViolation(
                rule_number=8,
                rule_name="8 Points Beyond 1σ",
                description="8 consecutive points beyond 1 sigma on either side - bimodal distribution (mixture)",
                violated_points=list(set(rule8_violations)),
                severity="warning",
                action="Investigate for multiple sources of variation. Check if multiple flashers or reference modules in use."
            ))

    return violations


def analyze_ref_module_data(
    values: np.ndarray,
    flash_numbers: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    historical_baseline: Optional[np.ndarray] = None,
    nominal_value: Optional[float] = None
) -> RefModuleAnalysisResult:
    """
    Perform comprehensive SPC analysis on reference module data.

    Args:
        values: Measured values (Isc or Pmax)
        flash_numbers: Flash sequence numbers (if None, uses indices)
        timestamps: Measurement timestamps (optional)
        historical_baseline: Historical data for establishing control limits (if None, uses values)
        nominal_value: Expected nominal value for centering

    Returns:
        RefModuleAnalysisResult with full analysis
    """
    n = len(values)

    if flash_numbers is None:
        flash_numbers = np.arange(1, n + 1)

    # Calculate moving ranges
    mr = np.abs(np.diff(values))
    mr_bar = np.mean(mr) if len(mr) > 0 else 0

    # Use historical baseline or current data for control limits
    baseline = historical_baseline if historical_baseline is not None else values
    limits = calculate_ref_module_control_limits(baseline, nominal_value)

    # Individual chart values
    x_cl = limits.cl
    x_ucl = limits.ucl
    x_lcl = limits.lcl
    x_sigma = limits.sigma

    # Moving range chart limits (using D4 = 3.267 for n=2)
    mr_cl = mr_bar
    mr_ucl = 3.267 * mr_bar
    mr_lcl = 0

    # Identify out-of-control and warning points
    ooc_points = []
    warning_points = []
    for i, v in enumerate(values):
        if v > x_ucl or v < x_lcl:
            ooc_points.append(i)
        elif v > limits.warning_ucl or v < limits.warning_lcl:
            warning_points.append(i)

    # Trend analysis using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.arange(n), values
    )
    has_significant_trend = p_value < 0.05 and abs(slope) > x_sigma * 0.1

    # Drift detection using CUSUM-like approach
    drift_detected = False
    drift_start_index = None
    if n >= 10:
        # Calculate cumulative sum of deviations from center line
        cusum = np.cumsum(values - x_cl)
        cusum_threshold = 4 * x_sigma * np.sqrt(n)

        for i in range(n):
            if abs(cusum[i]) > cusum_threshold:
                drift_detected = True
                drift_start_index = i
                break

    return RefModuleAnalysisResult(
        values=values,
        timestamps=timestamps,
        flash_numbers=flash_numbers,
        moving_ranges=mr,
        x_ucl=x_ucl,
        x_lcl=x_lcl,
        x_cl=x_cl,
        x_sigma=x_sigma,
        mr_ucl=mr_ucl,
        mr_lcl=mr_lcl,
        mr_cl=mr_cl,
        mean=np.mean(values),
        std_dev=np.std(values, ddof=1) if n > 1 else 0,
        min_value=np.min(values),
        max_value=np.max(values),
        ooc_points=ooc_points,
        warning_points=warning_points,
        trend_slope=slope,
        trend_pvalue=p_value,
        has_significant_trend=has_significant_trend,
        drift_detected=drift_detected,
        drift_start_index=drift_start_index
    )


def generate_ref_module_sample_data(
    n_flashes: int = 100,
    nominal_isc: float = 8.5,
    nominal_pmax: float = 320.0,
    isc_std: float = 0.02,
    pmax_std: float = 1.0,
    include_drift: bool = False,
    drift_start: int = 70,
    drift_rate: float = 0.001,
    include_anomalies: bool = False,
    anomaly_positions: Optional[List[int]] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate simulated reference module data for testing.

    Args:
        n_flashes: Number of flash measurements
        nominal_isc: Nominal short-circuit current (A)
        nominal_pmax: Nominal maximum power (W)
        isc_std: Standard deviation of Isc measurements
        pmax_std: Standard deviation of Pmax measurements
        include_drift: Whether to include gradual drift
        drift_start: Flash number where drift begins
        drift_rate: Rate of drift per flash
        include_anomalies: Whether to include anomalous points
        anomaly_positions: Specific positions for anomalies
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: flash_number, timestamp, isc, pmax
    """
    if seed is not None:
        np.random.seed(seed)

    flash_numbers = np.arange(1, n_flashes + 1)

    # Generate base measurements with random variation
    isc_values = np.random.normal(nominal_isc, isc_std, n_flashes)
    pmax_values = np.random.normal(nominal_pmax, pmax_std, n_flashes)

    # Add drift if requested
    if include_drift:
        drift_indices = np.arange(n_flashes) - drift_start
        drift_indices = np.maximum(drift_indices, 0)
        isc_drift = -drift_rate * drift_indices  # Degradation (decreasing Isc)
        pmax_drift = -drift_rate * 100 * drift_indices  # Proportional Pmax decrease
        isc_values += isc_drift
        pmax_values += pmax_drift

    # Add anomalies if requested
    if include_anomalies:
        if anomaly_positions is None:
            anomaly_positions = np.random.choice(n_flashes, size=3, replace=False)

        for pos in anomaly_positions:
            if pos < n_flashes:
                isc_values[pos] += np.random.choice([-1, 1]) * 4 * isc_std
                pmax_values[pos] += np.random.choice([-1, 1]) * 4 * pmax_std

    # Generate timestamps
    from datetime import datetime, timedelta
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i*8) for i in range(n_flashes)]

    return pd.DataFrame({
        'flash_number': flash_numbers,
        'timestamp': timestamps,
        'isc': np.round(isc_values, 4),
        'pmax': np.round(pmax_values, 2)
    })


def get_point_status(
    value: float,
    ucl: float,
    lcl: float,
    warning_ucl: float,
    warning_lcl: float
) -> Tuple[str, str]:
    """
    Get status and color for a data point.

    Args:
        value: Measured value
        ucl: Upper control limit
        lcl: Lower control limit
        warning_ucl: Upper warning limit (2-sigma)
        warning_lcl: Lower warning limit (2-sigma)

    Returns:
        Tuple of (status, color)
    """
    if value > ucl or value < lcl:
        return "Out of Control", "#FF6B6B"
    elif value > warning_ucl or value < warning_lcl:
        return "Warning", "#FFE66D"
    else:
        return "Normal", "#00D4AA"


def calculate_ref_module_cpk(
    values: np.ndarray,
    usl: float,
    lsl: float,
    target: Optional[float] = None
) -> Dict:
    """
    Calculate process capability indices for reference module monitoring.

    Args:
        values: Measured values
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (defaults to midpoint)

    Returns:
        Dictionary with Cp, Cpk, and related statistics
    """
    if target is None:
        target = (usl + lsl) / 2

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    spec_range = usl - lsl

    # Cp - Process potential
    cp = spec_range / (6 * std) if std > 0 else 0

    # Cpk - Process capability
    cpu = (usl - mean) / (3 * std) if std > 0 else 0
    cpl = (mean - lsl) / (3 * std) if std > 0 else 0
    cpk = min(cpu, cpl)

    # Cpm - Taguchi capability (accounts for deviation from target)
    variance_from_target = np.mean((values - target) ** 2)
    cpm = spec_range / (6 * np.sqrt(variance_from_target)) if variance_from_target > 0 else 0

    return {
        'cp': cp,
        'cpk': cpk,
        'cpm': cpm,
        'cpu': cpu,
        'cpl': cpl,
        'mean': mean,
        'std_dev': std,
        'usl': usl,
        'lsl': lsl,
        'target': target
    }
