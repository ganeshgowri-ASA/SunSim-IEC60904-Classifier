"""
Measurement System Analysis (MSA) Calculations per AIAG MSA Manual.
Includes Gage R&R, variance component analysis, and ndc calculations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy import stats


# d2* values for Gage R&R calculations (rows: number of trials, columns: number of parts x operators)
D2_STAR_TABLE = {
    2: {1: 1.41, 2: 1.28, 3: 1.23, 4: 1.21, 5: 1.19, 6: 1.18, 7: 1.17, 8: 1.17, 9: 1.16, 10: 1.16,
        15: 1.15, 20: 1.14, 25: 1.14, 30: 1.14, 40: 1.13, 50: 1.13, 100: 1.13},
    3: {1: 1.91, 2: 1.81, 3: 1.77, 4: 1.75, 5: 1.74, 6: 1.73, 7: 1.73, 8: 1.72, 9: 1.72, 10: 1.72,
        15: 1.71, 20: 1.71, 25: 1.70, 30: 1.70, 40: 1.70, 50: 1.70, 100: 1.70},
}

# K1 factors for different number of trials
K1_FACTORS = {2: 4.56, 3: 3.05}

# K2 factors for different number of operators
K2_FACTORS = {2: 3.65, 3: 2.70}

# K3 factors for different number of parts
K3_FACTORS = {
    2: 3.65, 3: 2.70, 4: 2.30, 5: 2.08, 6: 1.93, 7: 1.82,
    8: 1.74, 9: 1.67, 10: 1.62, 15: 1.41, 20: 1.28, 25: 1.18
}


@dataclass
class GageRRResult:
    """Result container for Gage R&R analysis."""
    # Variance components
    repeatability_variance: float
    reproducibility_variance: float
    operator_variance: float
    operator_by_part_variance: float
    part_to_part_variance: float
    total_variance: float

    # Standard deviations
    repeatability_std: float
    reproducibility_std: float
    grr_std: float
    part_to_part_std: float
    total_std: float

    # Study variation (5.15 sigma)
    repeatability_sv: float
    reproducibility_sv: float
    grr_sv: float
    part_to_part_sv: float
    total_sv: float

    # Percent contribution
    repeatability_pct_contribution: float
    reproducibility_pct_contribution: float
    grr_pct_contribution: float
    part_to_part_pct_contribution: float

    # Percent study variation
    repeatability_pct_sv: float
    reproducibility_pct_sv: float
    grr_pct_sv: float
    part_to_part_pct_sv: float

    # Percent tolerance (if tolerance provided)
    repeatability_pct_tolerance: Optional[float]
    reproducibility_pct_tolerance: Optional[float]
    grr_pct_tolerance: Optional[float]
    part_to_part_pct_tolerance: Optional[float]

    # Number of distinct categories
    ndc: float

    # Summary
    n_operators: int
    n_parts: int
    n_trials: int
    tolerance: Optional[float]


@dataclass
class ANOVAResult:
    """ANOVA table result for Gage R&R."""
    source: str
    df: int
    ss: float
    ms: float
    f_value: Optional[float]
    p_value: Optional[float]


def calculate_gage_rr_xbar_r(data: pd.DataFrame, tolerance: Optional[float] = None) -> GageRRResult:
    """
    Calculate Gage R&R using the X-bar and R (Average and Range) method.

    Args:
        data: DataFrame with columns ['operator', 'part_id', 'trial', 'measured_value']
        tolerance: Specification tolerance (USL - LSL), optional

    Returns:
        GageRRResult with all calculations
    """
    # Get unique values
    operators = data['operator'].unique()
    parts = data['part_id'].unique()
    n_operators = len(operators)
    n_parts = len(parts)

    # Calculate number of trials
    trials_per_combo = data.groupby(['operator', 'part_id']).size()
    n_trials = int(trials_per_combo.mode().iloc[0]) if len(trials_per_combo) > 0 else 2

    # Create pivot table for analysis
    pivot = data.pivot_table(
        index=['operator', 'part_id'],
        values='measured_value',
        aggfunc=list
    )

    # Calculate ranges within each operator-part combination
    ranges = []
    for idx, measurements in pivot['measured_value'].items():
        if isinstance(measurements, list) and len(measurements) > 1:
            ranges.append(max(measurements) - min(measurements))

    r_bar = np.mean(ranges) if ranges else 0

    # Get K factors
    k1 = K1_FACTORS.get(n_trials, 3.05)

    # Get d2* value
    g = n_operators * n_parts  # Number of ranges
    d2_star = 1.128  # Default value
    if n_trials in D2_STAR_TABLE:
        for key in sorted(D2_STAR_TABLE[n_trials].keys()):
            if g <= key:
                d2_star = D2_STAR_TABLE[n_trials][key]
                break
            d2_star = D2_STAR_TABLE[n_trials][max(D2_STAR_TABLE[n_trials].keys())]

    # Repeatability (Equipment Variation - EV)
    ev = r_bar / d2_star
    repeatability_variance = ev ** 2

    # Calculate operator averages
    operator_means = data.groupby('operator')['measured_value'].mean()
    x_diff = operator_means.max() - operator_means.min()

    k2 = K2_FACTORS.get(n_operators, 2.70)

    # Reproducibility (Appraiser Variation - AV)
    av_squared = (x_diff / k2) ** 2 - (ev ** 2) / (n_parts * n_trials)
    av_squared = max(0, av_squared)  # Cannot be negative
    av = np.sqrt(av_squared)
    reproducibility_variance = av ** 2

    # Gage R&R
    grr = np.sqrt(ev ** 2 + av ** 2)
    grr_variance = grr ** 2

    # Part-to-part variation (PV)
    part_means = data.groupby('part_id')['measured_value'].mean()
    rp = part_means.max() - part_means.min()

    k3 = K3_FACTORS.get(n_parts, 1.62)
    pv = rp / k3
    part_to_part_variance = pv ** 2

    # Total variation
    tv = np.sqrt(grr ** 2 + pv ** 2)
    total_variance = tv ** 2

    # Study Variation (5.15 sigma for 99% coverage)
    sv_multiplier = 5.15
    repeatability_sv = ev * sv_multiplier
    reproducibility_sv = av * sv_multiplier
    grr_sv = grr * sv_multiplier
    part_to_part_sv = pv * sv_multiplier
    total_sv = tv * sv_multiplier

    # Percent Contribution (based on variance)
    if total_variance > 0:
        repeatability_pct_contribution = (repeatability_variance / total_variance) * 100
        reproducibility_pct_contribution = (reproducibility_variance / total_variance) * 100
        grr_pct_contribution = (grr_variance / total_variance) * 100
        part_to_part_pct_contribution = (part_to_part_variance / total_variance) * 100
    else:
        repeatability_pct_contribution = 0
        reproducibility_pct_contribution = 0
        grr_pct_contribution = 0
        part_to_part_pct_contribution = 0

    # Percent Study Variation
    if tv > 0:
        repeatability_pct_sv = (ev / tv) * 100
        reproducibility_pct_sv = (av / tv) * 100
        grr_pct_sv = (grr / tv) * 100
        part_to_part_pct_sv = (pv / tv) * 100
    else:
        repeatability_pct_sv = 0
        reproducibility_pct_sv = 0
        grr_pct_sv = 0
        part_to_part_pct_sv = 0

    # Percent Tolerance (if tolerance provided)
    if tolerance and tolerance > 0:
        repeatability_pct_tolerance = (repeatability_sv / tolerance) * 100
        reproducibility_pct_tolerance = (reproducibility_sv / tolerance) * 100
        grr_pct_tolerance = (grr_sv / tolerance) * 100
        part_to_part_pct_tolerance = (part_to_part_sv / tolerance) * 100
    else:
        repeatability_pct_tolerance = None
        reproducibility_pct_tolerance = None
        grr_pct_tolerance = None
        part_to_part_pct_tolerance = None

    # Number of Distinct Categories (ndc)
    ndc = 1.41 * (pv / grr) if grr > 0 else 0
    ndc = max(1, int(ndc))

    return GageRRResult(
        repeatability_variance=repeatability_variance,
        reproducibility_variance=reproducibility_variance,
        operator_variance=reproducibility_variance,  # In X-bar R method, same as reproducibility
        operator_by_part_variance=0,  # Not calculated in X-bar R method
        part_to_part_variance=part_to_part_variance,
        total_variance=total_variance,
        repeatability_std=ev,
        reproducibility_std=av,
        grr_std=grr,
        part_to_part_std=pv,
        total_std=tv,
        repeatability_sv=repeatability_sv,
        reproducibility_sv=reproducibility_sv,
        grr_sv=grr_sv,
        part_to_part_sv=part_to_part_sv,
        total_sv=total_sv,
        repeatability_pct_contribution=repeatability_pct_contribution,
        reproducibility_pct_contribution=reproducibility_pct_contribution,
        grr_pct_contribution=grr_pct_contribution,
        part_to_part_pct_contribution=part_to_part_pct_contribution,
        repeatability_pct_sv=repeatability_pct_sv,
        reproducibility_pct_sv=reproducibility_pct_sv,
        grr_pct_sv=grr_pct_sv,
        part_to_part_pct_sv=part_to_part_pct_sv,
        repeatability_pct_tolerance=repeatability_pct_tolerance,
        reproducibility_pct_tolerance=reproducibility_pct_tolerance,
        grr_pct_tolerance=grr_pct_tolerance,
        part_to_part_pct_tolerance=part_to_part_pct_tolerance,
        ndc=ndc,
        n_operators=n_operators,
        n_parts=n_parts,
        n_trials=n_trials,
        tolerance=tolerance
    )


def calculate_gage_rr_anova(data: pd.DataFrame, tolerance: Optional[float] = None) -> Tuple[GageRRResult, List[ANOVAResult]]:
    """
    Calculate Gage R&R using the ANOVA method.

    Args:
        data: DataFrame with columns ['operator', 'part_id', 'trial', 'measured_value']
        tolerance: Specification tolerance (USL - LSL), optional

    Returns:
        Tuple of (GageRRResult, list of ANOVAResult)
    """
    operators = data['operator'].unique()
    parts = data['part_id'].unique()
    n_operators = len(operators)
    n_parts = len(parts)

    trials_per_combo = data.groupby(['operator', 'part_id']).size()
    n_trials = int(trials_per_combo.mode().iloc[0]) if len(trials_per_combo) > 0 else 2

    n_total = len(data)
    grand_mean = data['measured_value'].mean()

    # Calculate sums of squares
    # SS Total
    ss_total = ((data['measured_value'] - grand_mean) ** 2).sum()

    # SS Parts
    part_means = data.groupby('part_id')['measured_value'].mean()
    ss_parts = n_operators * n_trials * ((part_means - grand_mean) ** 2).sum()

    # SS Operators
    operator_means = data.groupby('operator')['measured_value'].mean()
    ss_operators = n_parts * n_trials * ((operator_means - grand_mean) ** 2).sum()

    # SS Operator x Part interaction
    cell_means = data.groupby(['operator', 'part_id'])['measured_value'].mean()
    ss_interaction = 0
    for op in operators:
        for part in parts:
            if (op, part) in cell_means.index:
                cell_mean = cell_means[(op, part)]
                expected = grand_mean + (operator_means[op] - grand_mean) + (part_means[part] - grand_mean)
                ss_interaction += n_trials * (cell_mean - expected) ** 2

    # SS Repeatability (Error)
    ss_repeatability = ss_total - ss_parts - ss_operators - ss_interaction

    # Degrees of freedom
    df_parts = n_parts - 1
    df_operators = n_operators - 1
    df_interaction = df_parts * df_operators
    df_repeatability = n_parts * n_operators * (n_trials - 1)
    df_total = n_total - 1

    # Mean squares
    ms_parts = ss_parts / df_parts if df_parts > 0 else 0
    ms_operators = ss_operators / df_operators if df_operators > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

    # F-values
    f_parts = ms_parts / ms_interaction if ms_interaction > 0 else 0
    f_operators = ms_operators / ms_interaction if ms_interaction > 0 else 0
    f_interaction = ms_interaction / ms_repeatability if ms_repeatability > 0 else 0

    # P-values
    p_parts = 1 - stats.f.cdf(f_parts, df_parts, df_interaction) if f_parts > 0 else 1
    p_operators = 1 - stats.f.cdf(f_operators, df_operators, df_interaction) if f_operators > 0 else 1
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_repeatability) if f_interaction > 0 else 1

    # Create ANOVA table
    anova_results = [
        ANOVAResult("Part", df_parts, ss_parts, ms_parts, f_parts, p_parts),
        ANOVAResult("Operator", df_operators, ss_operators, ms_operators, f_operators, p_operators),
        ANOVAResult("Operator × Part", df_interaction, ss_interaction, ms_interaction, f_interaction, p_interaction),
        ANOVAResult("Repeatability", df_repeatability, ss_repeatability, ms_repeatability, None, None),
        ANOVAResult("Total", df_total, ss_total, None, None, None),
    ]

    # Variance components
    repeatability_variance = ms_repeatability

    # Check if interaction is significant (alpha = 0.25 per AIAG)
    if p_interaction <= 0.25:
        operator_by_part_variance = (ms_interaction - ms_repeatability) / n_trials
        operator_by_part_variance = max(0, operator_by_part_variance)
        reproducibility_variance = (ms_operators - ms_interaction) / (n_parts * n_trials)
        reproducibility_variance = max(0, reproducibility_variance)
    else:
        operator_by_part_variance = 0
        # Pool interaction with repeatability
        pooled_ms = (ss_interaction + ss_repeatability) / (df_interaction + df_repeatability)
        repeatability_variance = pooled_ms
        reproducibility_variance = (ms_operators - pooled_ms) / (n_parts * n_trials)
        reproducibility_variance = max(0, reproducibility_variance)

    operator_variance = reproducibility_variance
    part_to_part_variance = (ms_parts - ms_interaction) / (n_operators * n_trials)
    part_to_part_variance = max(0, part_to_part_variance)

    grr_variance = repeatability_variance + reproducibility_variance + operator_by_part_variance
    total_variance = grr_variance + part_to_part_variance

    # Standard deviations
    repeatability_std = np.sqrt(repeatability_variance)
    reproducibility_std = np.sqrt(reproducibility_variance + operator_by_part_variance)
    grr_std = np.sqrt(grr_variance)
    part_to_part_std = np.sqrt(part_to_part_variance)
    total_std = np.sqrt(total_variance)

    # Study Variation (5.15 sigma)
    sv_multiplier = 5.15
    repeatability_sv = repeatability_std * sv_multiplier
    reproducibility_sv = reproducibility_std * sv_multiplier
    grr_sv = grr_std * sv_multiplier
    part_to_part_sv = part_to_part_std * sv_multiplier
    total_sv = total_std * sv_multiplier

    # Percent Contribution
    if total_variance > 0:
        repeatability_pct_contribution = (repeatability_variance / total_variance) * 100
        reproducibility_pct_contribution = ((reproducibility_variance + operator_by_part_variance) / total_variance) * 100
        grr_pct_contribution = (grr_variance / total_variance) * 100
        part_to_part_pct_contribution = (part_to_part_variance / total_variance) * 100
    else:
        repeatability_pct_contribution = 0
        reproducibility_pct_contribution = 0
        grr_pct_contribution = 0
        part_to_part_pct_contribution = 0

    # Percent Study Variation
    if total_std > 0:
        repeatability_pct_sv = (repeatability_std / total_std) * 100
        reproducibility_pct_sv = (reproducibility_std / total_std) * 100
        grr_pct_sv = (grr_std / total_std) * 100
        part_to_part_pct_sv = (part_to_part_std / total_std) * 100
    else:
        repeatability_pct_sv = 0
        reproducibility_pct_sv = 0
        grr_pct_sv = 0
        part_to_part_pct_sv = 0

    # Percent Tolerance
    if tolerance and tolerance > 0:
        repeatability_pct_tolerance = (repeatability_sv / tolerance) * 100
        reproducibility_pct_tolerance = (reproducibility_sv / tolerance) * 100
        grr_pct_tolerance = (grr_sv / tolerance) * 100
        part_to_part_pct_tolerance = (part_to_part_sv / tolerance) * 100
    else:
        repeatability_pct_tolerance = None
        reproducibility_pct_tolerance = None
        grr_pct_tolerance = None
        part_to_part_pct_tolerance = None

    # Number of Distinct Categories
    ndc = 1.41 * (part_to_part_std / grr_std) if grr_std > 0 else 0
    ndc = max(1, int(ndc))

    result = GageRRResult(
        repeatability_variance=repeatability_variance,
        reproducibility_variance=reproducibility_variance + operator_by_part_variance,
        operator_variance=operator_variance,
        operator_by_part_variance=operator_by_part_variance,
        part_to_part_variance=part_to_part_variance,
        total_variance=total_variance,
        repeatability_std=repeatability_std,
        reproducibility_std=reproducibility_std,
        grr_std=grr_std,
        part_to_part_std=part_to_part_std,
        total_std=total_std,
        repeatability_sv=repeatability_sv,
        reproducibility_sv=reproducibility_sv,
        grr_sv=grr_sv,
        part_to_part_sv=part_to_part_sv,
        total_sv=total_sv,
        repeatability_pct_contribution=repeatability_pct_contribution,
        reproducibility_pct_contribution=reproducibility_pct_contribution,
        grr_pct_contribution=grr_pct_contribution,
        part_to_part_pct_contribution=part_to_part_pct_contribution,
        repeatability_pct_sv=repeatability_pct_sv,
        reproducibility_pct_sv=reproducibility_pct_sv,
        grr_pct_sv=grr_pct_sv,
        part_to_part_pct_sv=part_to_part_pct_sv,
        repeatability_pct_tolerance=repeatability_pct_tolerance,
        reproducibility_pct_tolerance=reproducibility_pct_tolerance,
        grr_pct_tolerance=grr_pct_tolerance,
        part_to_part_pct_tolerance=part_to_part_pct_tolerance,
        ndc=ndc,
        n_operators=n_operators,
        n_parts=n_parts,
        n_trials=n_trials,
        tolerance=tolerance
    )

    return result, anova_results


def get_grr_status(grr_pct: float) -> Tuple[str, str]:
    """
    Get GRR status and color based on percentage.

    Args:
        grr_pct: GRR percentage (of study variation or tolerance)

    Returns:
        Tuple of (status, color)
    """
    if grr_pct < 10:
        return "Acceptable", "#00D4AA"  # Green
    elif grr_pct < 30:
        return "Marginal", "#FFE66D"  # Yellow
    else:
        return "Unacceptable", "#FF6B6B"  # Red


def get_ndc_status(ndc: int) -> Tuple[str, str]:
    """
    Get ndc status and color.

    Args:
        ndc: Number of distinct categories

    Returns:
        Tuple of (status, color)
    """
    if ndc >= 5:
        return "Acceptable (≥5)", "#00D4AA"  # Green
    elif ndc >= 3:
        return "Marginal (3-4)", "#FFE66D"  # Yellow
    else:
        return "Unacceptable (<3)", "#FF6B6B"  # Red


def generate_sample_msa_data(n_operators: int = 3, n_parts: int = 10,
                              n_trials: int = 3, true_mean: float = 100.0,
                              part_variation: float = 5.0,
                              operator_variation: float = 0.5,
                              repeatability: float = 0.3,
                              seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate sample MSA data for testing.

    Args:
        n_operators: Number of operators
        n_parts: Number of parts
        n_trials: Number of trials per operator-part combination
        true_mean: True process mean
        part_variation: Standard deviation of part-to-part variation
        operator_variation: Standard deviation of operator-to-operator variation
        repeatability: Standard deviation of measurement repeatability
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns ['operator', 'part_id', 'trial', 'measured_value']
    """
    if seed is not None:
        np.random.seed(seed)

    operators = [f"Operator {i+1}" for i in range(n_operators)]
    parts = [f"Part {i+1}" for i in range(n_parts)]

    # Generate random effects
    part_effects = np.random.normal(0, part_variation, n_parts)
    operator_effects = np.random.normal(0, operator_variation, n_operators)

    data = []
    for i, op in enumerate(operators):
        for j, part in enumerate(parts):
            for trial in range(1, n_trials + 1):
                value = (true_mean +
                        part_effects[j] +
                        operator_effects[i] +
                        np.random.normal(0, repeatability))
                data.append({
                    'operator': op,
                    'part_id': part,
                    'trial': trial,
                    'measured_value': round(value, 4)
                })

    return pd.DataFrame(data)


def create_variance_component_summary(result: GageRRResult) -> pd.DataFrame:
    """Create a summary DataFrame of variance components."""
    data = {
        'Source': ['Gage R&R', '  Repeatability', '  Reproducibility', 'Part-to-Part', 'Total'],
        'VarComp': [
            result.grr_variance,
            result.repeatability_variance,
            result.reproducibility_variance,
            result.part_to_part_variance,
            result.total_variance
        ],
        'StdDev': [
            result.grr_std,
            result.repeatability_std,
            result.reproducibility_std,
            result.part_to_part_std,
            result.total_std
        ],
        'Study Var (5.15σ)': [
            result.grr_sv,
            result.repeatability_sv,
            result.reproducibility_sv,
            result.part_to_part_sv,
            result.total_sv
        ],
        '%Contribution': [
            result.grr_pct_contribution,
            result.repeatability_pct_contribution,
            result.reproducibility_pct_contribution,
            result.part_to_part_pct_contribution,
            100.0
        ],
        '%Study Var': [
            result.grr_pct_sv,
            result.repeatability_pct_sv,
            result.reproducibility_pct_sv,
            result.part_to_part_pct_sv,
            100.0
        ]
    }

    if result.tolerance:
        data['%Tolerance'] = [
            result.grr_pct_tolerance,
            result.repeatability_pct_tolerance,
            result.reproducibility_pct_tolerance,
            result.part_to_part_pct_tolerance,
            (result.total_sv / result.tolerance) * 100
        ]

    return pd.DataFrame(data)
