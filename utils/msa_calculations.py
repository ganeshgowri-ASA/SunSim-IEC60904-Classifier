"""
Sun Simulator Classification System - MSA Calculations Module
Measurement System Analysis / Gage R&R

This module provides calculations for:
- Gage Repeatability & Reproducibility (GRR)
- Variance Components Analysis
- Number of Distinct Categories (ndc)
- MSA Study Analysis per AIAG MSA Manual
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VarianceComponents:
    """Variance components from GRR study."""
    total_variance: float
    part_to_part_variance: float
    total_grr_variance: float
    repeatability_variance: float
    reproducibility_variance: float
    operator_variance: float
    operator_part_interaction_variance: float


@dataclass
class GRRResult:
    """Complete Gage R&R study results."""
    # Variance components
    variance: VarianceComponents

    # Standard deviations
    sigma_total: float
    sigma_part: float
    sigma_grr: float
    sigma_repeatability: float
    sigma_reproducibility: float
    sigma_operator: float
    sigma_interaction: float

    # Percentage contributions
    pct_contribution_part: float
    pct_contribution_grr: float
    pct_contribution_repeatability: float
    pct_contribution_reproducibility: float

    # Study variation (6*sigma as % of tolerance or total variation)
    pct_study_var_part: float
    pct_study_var_grr: float
    pct_study_var_repeatability: float
    pct_study_var_reproducibility: float

    # Tolerance analysis (if tolerance provided)
    pct_tolerance_grr: Optional[float]
    pct_tolerance_repeatability: Optional[float]
    pct_tolerance_reproducibility: Optional[float]

    # Key metrics
    ndc: int  # Number of Distinct Categories
    grr_percent: float  # Main GRR %

    # Study details
    n_parts: int
    n_operators: int
    n_trials: int

    # Data statistics
    overall_mean: float
    overall_std: float

    # Per-operator and per-part means
    operator_means: Dict[str, float]
    part_means: Dict[str, float]

    # Measurement data (for charts)
    measurement_data: np.ndarray


class GRRRating(Enum):
    """GRR study rating categories per AIAG MSA."""
    ACCEPTABLE = "Acceptable"
    MARGINAL = "Marginal"
    UNACCEPTABLE = "Unacceptable"


# =============================================================================
# GAGE R&R CALCULATOR
# =============================================================================

class GRRCalculator:
    """
    Gage R&R Calculator using ANOVA method.
    Implements AIAG MSA Manual methodology.
    """

    @staticmethod
    def calculate_grr(data: np.ndarray,
                      operators: List[str],
                      parts: List[str],
                      tolerance: Optional[float] = None,
                      alpha: float = 0.05) -> GRRResult:
        """
        Calculate Gage R&R using ANOVA method.

        Args:
            data: 3D array of shape (n_operators, n_parts, n_trials)
            operators: List of operator names/IDs
            parts: List of part names/IDs
            tolerance: Tolerance width (USL - LSL) for %Tolerance calculation
            alpha: Significance level for F-tests

        Returns:
            GRRResult with complete analysis
        """
        n_operators, n_parts, n_trials = data.shape
        N = n_operators * n_parts * n_trials

        # Grand mean
        grand_mean = np.mean(data)

        # Operator means
        operator_means_arr = np.mean(data, axis=(1, 2))
        operator_means = {op: operator_means_arr[i] for i, op in enumerate(operators)}

        # Part means
        part_means_arr = np.mean(data, axis=(0, 2))
        part_means = {part: part_means_arr[i] for i, part in enumerate(parts)}

        # Cell means (operator x part)
        cell_means = np.mean(data, axis=2)

        # Calculate Sum of Squares
        # SS Total
        ss_total = np.sum((data - grand_mean) ** 2)

        # SS Operators
        ss_operators = n_parts * n_trials * np.sum((operator_means_arr - grand_mean) ** 2)

        # SS Parts
        ss_parts = n_operators * n_trials * np.sum((part_means_arr - grand_mean) ** 2)

        # SS Operator x Part Interaction
        ss_interaction = n_trials * np.sum(
            (cell_means - operator_means_arr[:, np.newaxis] -
             part_means_arr[np.newaxis, :] + grand_mean) ** 2
        )

        # SS Equipment (Repeatability) - within cell variation
        ss_equipment = np.sum((data - cell_means[:, :, np.newaxis]) ** 2)

        # Degrees of Freedom
        df_operators = n_operators - 1
        df_parts = n_parts - 1
        df_interaction = df_operators * df_parts
        df_equipment = n_operators * n_parts * (n_trials - 1)
        df_total = N - 1

        # Mean Squares
        ms_operators = ss_operators / df_operators if df_operators > 0 else 0
        ms_parts = ss_parts / df_parts if df_parts > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_equipment = ss_equipment / df_equipment if df_equipment > 0 else 0

        # Variance Components (using expected mean squares)
        # Equipment variance (repeatability)
        var_equipment = ms_equipment

        # Interaction variance
        var_interaction = max(0, (ms_interaction - ms_equipment) / n_trials)

        # Operator variance
        var_operator = max(0, (ms_operators - ms_interaction) / (n_parts * n_trials))

        # Part-to-part variance
        var_parts = max(0, (ms_parts - ms_interaction) / (n_operators * n_trials))

        # Reproducibility variance (operator + interaction)
        var_reproducibility = var_operator + var_interaction

        # Total GRR variance
        var_grr = var_equipment + var_reproducibility

        # Total variance
        var_total = var_parts + var_grr

        # Create variance components object
        variance = VarianceComponents(
            total_variance=var_total,
            part_to_part_variance=var_parts,
            total_grr_variance=var_grr,
            repeatability_variance=var_equipment,
            reproducibility_variance=var_reproducibility,
            operator_variance=var_operator,
            operator_part_interaction_variance=var_interaction
        )

        # Standard deviations
        sigma_total = np.sqrt(var_total)
        sigma_part = np.sqrt(var_parts)
        sigma_grr = np.sqrt(var_grr)
        sigma_repeatability = np.sqrt(var_equipment)
        sigma_reproducibility = np.sqrt(var_reproducibility)
        sigma_operator = np.sqrt(var_operator)
        sigma_interaction = np.sqrt(var_interaction)

        # Percentage of Total Variance (Contribution)
        pct_var_part = (var_parts / var_total * 100) if var_total > 0 else 0
        pct_var_grr = (var_grr / var_total * 100) if var_total > 0 else 0
        pct_var_repeatability = (var_equipment / var_total * 100) if var_total > 0 else 0
        pct_var_reproducibility = (var_reproducibility / var_total * 100) if var_total > 0 else 0

        # Study Variation (% of Total Study Variation = 6*sigma)
        sv_total = 6 * sigma_total
        pct_sv_part = (6 * sigma_part / sv_total * 100) if sv_total > 0 else 0
        pct_sv_grr = (6 * sigma_grr / sv_total * 100) if sv_total > 0 else 0
        pct_sv_repeatability = (6 * sigma_repeatability / sv_total * 100) if sv_total > 0 else 0
        pct_sv_reproducibility = (6 * sigma_reproducibility / sv_total * 100) if sv_total > 0 else 0

        # Tolerance analysis
        pct_tol_grr = None
        pct_tol_repeatability = None
        pct_tol_reproducibility = None

        if tolerance is not None and tolerance > 0:
            pct_tol_grr = (6 * sigma_grr / tolerance * 100)
            pct_tol_repeatability = (6 * sigma_repeatability / tolerance * 100)
            pct_tol_reproducibility = (6 * sigma_reproducibility / tolerance * 100)

        # Number of Distinct Categories (ndc)
        # ndc = 1.41 * (sigma_part / sigma_grr)
        if sigma_grr > 0:
            ndc = int(1.41 * sigma_part / sigma_grr)
            ndc = max(1, ndc)
        else:
            ndc = 1

        # Main GRR % (study variation based)
        grr_percent = pct_sv_grr

        return GRRResult(
            variance=variance,
            sigma_total=sigma_total,
            sigma_part=sigma_part,
            sigma_grr=sigma_grr,
            sigma_repeatability=sigma_repeatability,
            sigma_reproducibility=sigma_reproducibility,
            sigma_operator=sigma_operator,
            sigma_interaction=sigma_interaction,
            pct_contribution_part=pct_var_part,
            pct_contribution_grr=pct_var_grr,
            pct_contribution_repeatability=pct_var_repeatability,
            pct_contribution_reproducibility=pct_var_reproducibility,
            pct_study_var_part=pct_sv_part,
            pct_study_var_grr=pct_sv_grr,
            pct_study_var_repeatability=pct_sv_repeatability,
            pct_study_var_reproducibility=pct_sv_reproducibility,
            pct_tolerance_grr=pct_tol_grr,
            pct_tolerance_repeatability=pct_tol_repeatability,
            pct_tolerance_reproducibility=pct_tol_reproducibility,
            ndc=ndc,
            grr_percent=grr_percent,
            n_parts=n_parts,
            n_operators=n_operators,
            n_trials=n_trials,
            overall_mean=grand_mean,
            overall_std=np.std(data),
            operator_means=operator_means,
            part_means=part_means,
            measurement_data=data
        )

    @staticmethod
    def get_grr_rating(grr_percent: float) -> Tuple[GRRRating, str]:
        """
        Get GRR rating per AIAG guidelines.

        Args:
            grr_percent: GRR % Study Variation

        Returns:
            Tuple of (rating enum, description)
        """
        if grr_percent < 10:
            return GRRRating.ACCEPTABLE, "Measurement system is acceptable"
        elif grr_percent <= 30:
            return GRRRating.MARGINAL, "May be acceptable based on application importance, cost, etc."
        else:
            return GRRRating.UNACCEPTABLE, "Measurement system needs improvement"

    @staticmethod
    def get_ndc_rating(ndc: int) -> Tuple[str, str]:
        """
        Get ndc rating.

        Args:
            ndc: Number of distinct categories

        Returns:
            Tuple of (rating, description)
        """
        if ndc >= 5:
            return "Acceptable", "Measurement system can distinguish parts adequately"
        elif ndc >= 3:
            return "Marginal", "Measurement system has limited discrimination"
        else:
            return "Unacceptable", "Measurement system cannot distinguish parts"


# =============================================================================
# RANGE METHOD CALCULATOR
# =============================================================================

class RangeMethodCalculator:
    """
    Quick GRR calculation using Range method (XbarR).
    Less accurate but simpler than ANOVA.
    """

    # Constants from AIAG MSA manual
    K1_TABLE = {1: 4.56, 2: 3.05, 3: 2.50}  # Based on number of trials
    K2_TABLE = {2: 3.65, 3: 2.70}  # Based on number of operators

    @staticmethod
    def calculate_grr_range(data: np.ndarray,
                             operators: List[str],
                             parts: List[str],
                             tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate GRR using Range method.

        Args:
            data: 3D array (n_operators, n_parts, n_trials)
            operators: Operator names
            parts: Part names
            tolerance: Tolerance width

        Returns:
            Dictionary with GRR results
        """
        n_operators, n_parts, n_trials = data.shape

        # Range for each cell
        ranges = np.ptp(data, axis=2)

        # Average range (R-bar)
        r_bar = np.mean(ranges)

        # Operator averages
        operator_avgs = np.mean(data, axis=(1, 2))
        x_diff = np.max(operator_avgs) - np.min(operator_avgs)

        # Get K factors
        k1 = RangeMethodCalculator.K1_TABLE.get(n_trials, 2.50)
        k2 = RangeMethodCalculator.K2_TABLE.get(n_operators, 2.70)

        # Repeatability (Equipment Variation)
        ev = r_bar * k1

        # Reproducibility (Appraiser Variation)
        av_squared = (x_diff * k2) ** 2 - (ev ** 2 / (n_parts * n_trials))
        av = np.sqrt(max(0, av_squared))

        # GRR
        grr = np.sqrt(ev ** 2 + av ** 2)

        # Part variation
        part_avgs = np.mean(data, axis=(0, 2))
        rp = np.max(part_avgs) - np.min(part_avgs)
        k3 = 2.08 if n_parts >= 5 else 1.91  # Simplified
        pv = rp * k3

        # Total variation
        tv = np.sqrt(grr ** 2 + pv ** 2)

        # Percentages
        pct_ev = (ev / tv * 100) if tv > 0 else 0
        pct_av = (av / tv * 100) if tv > 0 else 0
        pct_grr = (grr / tv * 100) if tv > 0 else 0
        pct_pv = (pv / tv * 100) if tv > 0 else 0

        # Tolerance %
        pct_tol = (grr / tolerance * 100) if tolerance and tolerance > 0 else None

        # ndc
        ndc = int(1.41 * pv / grr) if grr > 0 else 1
        ndc = max(1, ndc)

        return {
            'ev': ev,
            'av': av,
            'grr': grr,
            'pv': pv,
            'tv': tv,
            'pct_ev': pct_ev,
            'pct_av': pct_av,
            'pct_grr': pct_grr,
            'pct_pv': pct_pv,
            'pct_tolerance': pct_tol,
            'ndc': ndc,
            'r_bar': r_bar,
            'x_diff': x_diff
        }


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_grr_sample_data(n_operators: int = 3,
                              n_parts: int = 10,
                              n_trials: int = 3,
                              part_variation: float = 10.0,
                              operator_variation: float = 2.0,
                              equipment_variation: float = 1.0,
                              base_value: float = 100.0) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Generate sample GRR study data.

    Args:
        n_operators: Number of operators
        n_parts: Number of parts
        n_trials: Number of trials per operator-part combination
        part_variation: Part-to-part standard deviation
        operator_variation: Operator-to-operator standard deviation
        equipment_variation: Within-trial standard deviation (repeatability)
        base_value: Base measurement value

    Returns:
        Tuple of (data array, operator names, part names)
    """
    np.random.seed(42)

    operators = [f"Operator {i+1}" for i in range(n_operators)]
    parts = [f"Part {chr(65+i)}" for i in range(n_parts)]

    # Generate true part values
    true_parts = np.random.normal(0, part_variation, n_parts)

    # Generate operator bias
    operator_bias = np.random.normal(0, operator_variation, n_operators)

    # Generate data
    data = np.zeros((n_operators, n_parts, n_trials))

    for i in range(n_operators):
        for j in range(n_parts):
            for k in range(n_trials):
                data[i, j, k] = (base_value +
                                 true_parts[j] +
                                 operator_bias[i] +
                                 np.random.normal(0, equipment_variation))

    return data, operators, parts


def calculate_variance_chart_data(result: GRRResult) -> Dict[str, Any]:
    """
    Prepare data for variance component charts.

    Args:
        result: GRRResult from GRR calculation

    Returns:
        Dictionary with chart data
    """
    return {
        'components': ['Total GRR', 'Repeatability', 'Reproducibility', 'Part-to-Part'],
        'variances': [
            result.variance.total_grr_variance,
            result.variance.repeatability_variance,
            result.variance.reproducibility_variance,
            result.variance.part_to_part_variance
        ],
        'std_devs': [
            result.sigma_grr,
            result.sigma_repeatability,
            result.sigma_reproducibility,
            result.sigma_part
        ],
        'pct_contribution': [
            result.pct_contribution_grr,
            result.pct_contribution_repeatability,
            result.pct_contribution_reproducibility,
            result.pct_contribution_part
        ],
        'pct_study_var': [
            result.pct_study_var_grr,
            result.pct_study_var_repeatability,
            result.pct_study_var_reproducibility,
            result.pct_study_var_part
        ]
    }
