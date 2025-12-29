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
