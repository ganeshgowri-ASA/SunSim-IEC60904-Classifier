"""
Sun Simulator Classification System - SPC Calculations Module
Statistical Process Control per ISO 22514

This module provides calculations for:
- Control Charts (X-bar, R, S charts)
- Process Capability Indices (Cp, Cpk, Pp, Ppk)
- Run Rules Detection
- Process Performance Metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# CONSTANTS - Control Chart Factors (ISO 7870-2)
# =============================================================================

# A2 factors for X-bar chart (based on subgroup size n)
A2_FACTORS = {
    2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
    6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308,
    11: 0.285, 12: 0.266, 13: 0.249, 14: 0.235, 15: 0.223,
    16: 0.212, 17: 0.203, 18: 0.194, 19: 0.187, 20: 0.180,
    21: 0.173, 22: 0.167, 23: 0.162, 24: 0.157, 25: 0.153
}

# D3 factors for R chart lower control limit
D3_FACTORS = {
    2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076,
    8: 0.136, 9: 0.184, 10: 0.223, 11: 0.256, 12: 0.283,
    13: 0.307, 14: 0.328, 15: 0.347, 16: 0.363, 17: 0.378,
    18: 0.391, 19: 0.403, 20: 0.415, 21: 0.425, 22: 0.434,
    23: 0.443, 24: 0.451, 25: 0.459
}

# D4 factors for R chart upper control limit
D4_FACTORS = {
    2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114,
    6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777,
    11: 1.744, 12: 1.717, 13: 1.693, 14: 1.672, 15: 1.653,
    16: 1.637, 17: 1.622, 18: 1.608, 19: 1.597, 20: 1.585,
    21: 1.575, 22: 1.566, 23: 1.557, 24: 1.548, 25: 1.541
}

# d2 factors for estimating sigma from R-bar
D2_FACTORS = {
    2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
    6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078,
    11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472,
    16: 3.532, 17: 3.588, 18: 3.640, 19: 3.689, 20: 3.735,
    21: 3.778, 22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931
}

# c4 factors for estimating sigma from S-bar
C4_FACTORS = {
    2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400,
    6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727,
    11: 0.9754, 12: 0.9776, 13: 0.9794, 14: 0.9810, 15: 0.9823,
    16: 0.9835, 17: 0.9845, 18: 0.9854, 19: 0.9862, 20: 0.9869,
    21: 0.9876, 22: 0.9882, 23: 0.9887, 24: 0.9892, 25: 0.9896
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class RunRuleViolation(Enum):
    """Western Electric Run Rules for control charts."""
    RULE_1_BEYOND_3_SIGMA = "1 point beyond 3σ limit"
    RULE_2_ZONE_A = "2 of 3 consecutive points in Zone A or beyond"
    RULE_3_ZONE_B = "4 of 5 consecutive points in Zone B or beyond"
    RULE_4_ZONE_C = "8 consecutive points on one side of centerline"
    RULE_5_TREND = "6 consecutive points trending up or down"
    RULE_6_ALTERNATING = "14 consecutive points alternating up and down"
    RULE_7_HUGGING = "15 consecutive points within Zone C"
    RULE_8_BEYOND_1_SIGMA = "8 consecutive points beyond 1σ on either side"


@dataclass
class ControlChartResult:
    """Results from control chart analysis."""
    subgroup_means: np.ndarray
    subgroup_ranges: np.ndarray
    subgroup_stds: np.ndarray
    x_bar: float
    r_bar: float
    s_bar: float
    sigma_estimate: float
    ucl_x: float
    lcl_x: float
    cl_x: float
    ucl_r: float
    lcl_r: float
    cl_r: float
    violations: List[Dict[str, Any]]
    out_of_control_points: List[int]


@dataclass
class CapabilityResult:
    """Results from capability analysis."""
    cp: float
    cpk: float
    cpl: float
    cpu: float
    pp: float
    ppk: float
    ppl: float
    ppu: float
    sigma_within: float
    sigma_overall: float
    process_mean: float
    lsl: float
    usl: float
    target: Optional[float]
    ppm_below_lsl: float
    ppm_above_usl: float
    ppm_total: float
    sigma_level: float
    yield_percent: float


@dataclass
class HistogramResult:
    """Results from histogram analysis."""
    bins: np.ndarray
    counts: np.ndarray
    bin_centers: np.ndarray
    mean: float
    std: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float


# =============================================================================
# CONTROL CHART CALCULATIONS
# =============================================================================

class SPCCalculator:
    """
    Statistical Process Control calculator.
    Implements X-bar/R charts per ISO 7870-2.
    """

    @staticmethod
    def calculate_xbar_r_chart(data: np.ndarray,
                                subgroup_size: int = 5) -> ControlChartResult:
        """
        Calculate X-bar and R control chart parameters.

        Args:
            data: 1D array of measurements or 2D array (subgroups x samples)
            subgroup_size: Number of samples per subgroup (if 1D data)

        Returns:
            ControlChartResult with all chart parameters
        """
        # Reshape data into subgroups if 1D
        if data.ndim == 1:
            n_complete = (len(data) // subgroup_size) * subgroup_size
            data = data[:n_complete].reshape(-1, subgroup_size)

        n_subgroups, n = data.shape

        # Calculate subgroup statistics
        subgroup_means = np.mean(data, axis=1)
        subgroup_ranges = np.ptp(data, axis=1)  # max - min
        subgroup_stds = np.std(data, axis=1, ddof=1)

        # Grand mean and average range
        x_bar = np.mean(subgroup_means)
        r_bar = np.mean(subgroup_ranges)
        s_bar = np.mean(subgroup_stds)

        # Get factors for subgroup size (cap at 25)
        n_capped = min(n, 25)
        A2 = A2_FACTORS.get(n_capped, 0.18)
        D3 = D3_FACTORS.get(n_capped, 0)
        D4 = D4_FACTORS.get(n_capped, 2.0)
        d2 = D2_FACTORS.get(n_capped, 3.0)

        # Estimate process sigma from R-bar
        sigma_estimate = r_bar / d2

        # X-bar chart control limits
        ucl_x = x_bar + A2 * r_bar
        lcl_x = x_bar - A2 * r_bar
        cl_x = x_bar

        # R chart control limits
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar
        cl_r = r_bar

        # Detect run rule violations
        violations, ooc_points = SPCCalculator._detect_run_rules(
            subgroup_means, cl_x, ucl_x, lcl_x
        )

        return ControlChartResult(
            subgroup_means=subgroup_means,
            subgroup_ranges=subgroup_ranges,
            subgroup_stds=subgroup_stds,
            x_bar=x_bar,
            r_bar=r_bar,
            s_bar=s_bar,
            sigma_estimate=sigma_estimate,
            ucl_x=ucl_x,
            lcl_x=lcl_x,
            cl_x=cl_x,
            ucl_r=ucl_r,
            lcl_r=lcl_r,
            cl_r=cl_r,
            violations=violations,
            out_of_control_points=ooc_points
        )

    @staticmethod
    def _detect_run_rules(values: np.ndarray, cl: float,
                          ucl: float, lcl: float) -> Tuple[List[Dict], List[int]]:
        """
        Detect Western Electric run rule violations.

        Args:
            values: Array of subgroup means
            cl: Center line
            ucl: Upper control limit
            lcl: Lower control limit

        Returns:
            Tuple of (violations list, out-of-control point indices)
        """
        violations = []
        ooc_points = set()

        sigma = (ucl - cl) / 3
        zone_a_upper = cl + 2 * sigma
        zone_a_lower = cl - 2 * sigma
        zone_b_upper = cl + 1 * sigma
        zone_b_lower = cl - 1 * sigma

        n = len(values)

        # Rule 1: Point beyond 3 sigma
        for i, v in enumerate(values):
            if v > ucl or v < lcl:
                violations.append({
                    'rule': RunRuleViolation.RULE_1_BEYOND_3_SIGMA.value,
                    'point': i,
                    'value': v
                })
                ooc_points.add(i)

        # Rule 4: 8 consecutive points on one side of centerline
        consecutive_above = 0
        consecutive_below = 0
        for i, v in enumerate(values):
            if v > cl:
                consecutive_above += 1
                consecutive_below = 0
            elif v < cl:
                consecutive_below += 1
                consecutive_above = 0
            else:
                consecutive_above = 0
                consecutive_below = 0

            if consecutive_above >= 8:
                violations.append({
                    'rule': RunRuleViolation.RULE_4_ZONE_C.value,
                    'point': i,
                    'start': i - 7
                })
                for j in range(i - 7, i + 1):
                    ooc_points.add(j)

            if consecutive_below >= 8:
                violations.append({
                    'rule': RunRuleViolation.RULE_4_ZONE_C.value,
                    'point': i,
                    'start': i - 7
                })
                for j in range(i - 7, i + 1):
                    ooc_points.add(j)

        # Rule 5: 6 consecutive trending points
        if n >= 6:
            for i in range(5, n):
                window = values[i-5:i+1]
                diffs = np.diff(window)
                if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                    violations.append({
                        'rule': RunRuleViolation.RULE_5_TREND.value,
                        'point': i,
                        'start': i - 5
                    })
                    for j in range(i - 5, i + 1):
                        ooc_points.add(j)

        return violations, list(sorted(ooc_points))

    @staticmethod
    def calculate_individuals_chart(data: np.ndarray) -> ControlChartResult:
        """
        Calculate I-MR (Individuals and Moving Range) chart.

        Args:
            data: 1D array of individual measurements

        Returns:
            ControlChartResult with I-MR chart parameters
        """
        n = len(data)
        if n < 2:
            raise ValueError("Need at least 2 data points")

        # Moving ranges
        mr = np.abs(np.diff(data))

        # Statistics
        x_bar = np.mean(data)
        mr_bar = np.mean(mr)

        # Sigma estimate (d2 for n=2)
        d2 = D2_FACTORS[2]
        sigma_estimate = mr_bar / d2

        # I chart limits
        ucl_x = x_bar + 3 * sigma_estimate
        lcl_x = x_bar - 3 * sigma_estimate

        # MR chart limits
        D4 = D4_FACTORS[2]
        ucl_r = D4 * mr_bar
        lcl_r = 0

        # Pad moving range for alignment
        mr_padded = np.concatenate([[np.nan], mr])

        violations, ooc_points = SPCCalculator._detect_run_rules(
            data, x_bar, ucl_x, lcl_x
        )

        return ControlChartResult(
            subgroup_means=data,
            subgroup_ranges=mr_padded,
            subgroup_stds=np.full_like(data, sigma_estimate),
            x_bar=x_bar,
            r_bar=mr_bar,
            s_bar=sigma_estimate,
            sigma_estimate=sigma_estimate,
            ucl_x=ucl_x,
            lcl_x=lcl_x,
            cl_x=x_bar,
            ucl_r=ucl_r,
            lcl_r=lcl_r,
            cl_r=mr_bar,
            violations=violations,
            out_of_control_points=ooc_points
        )


# =============================================================================
# CAPABILITY CALCULATIONS
# =============================================================================

class CapabilityCalculator:
    """
    Process capability calculator per ISO 22514.
    """

    @staticmethod
    def calculate_capability(data: np.ndarray,
                              lsl: float,
                              usl: float,
                              target: Optional[float] = None,
                              subgroup_size: int = 5) -> CapabilityResult:
        """
        Calculate process capability indices.

        Args:
            data: Measurement data (1D array)
            lsl: Lower specification limit
            usl: Upper specification limit
            target: Target value (defaults to midpoint of specs)
            subgroup_size: Subgroup size for within-subgroup sigma estimate

        Returns:
            CapabilityResult with all capability metrics
        """
        if target is None:
            target = (usl + lsl) / 2

        n = len(data)
        process_mean = np.mean(data)

        # Overall (long-term) sigma
        sigma_overall = np.std(data, ddof=1)

        # Within-subgroup (short-term) sigma using R-bar/d2 method
        if n >= subgroup_size * 2:
            n_complete = (n // subgroup_size) * subgroup_size
            subgroups = data[:n_complete].reshape(-1, subgroup_size)
            ranges = np.ptp(subgroups, axis=1)
            r_bar = np.mean(ranges)
            d2 = D2_FACTORS.get(min(subgroup_size, 25), 2.326)
            sigma_within = r_bar / d2
        else:
            # Fall back to moving range method
            mr = np.abs(np.diff(data))
            mr_bar = np.mean(mr)
            sigma_within = mr_bar / D2_FACTORS[2]

        # Ensure sigma_within is not zero
        sigma_within = max(sigma_within, 1e-10)
        sigma_overall = max(sigma_overall, 1e-10)

        # Specification width
        spec_width = usl - lsl

        # Cp (potential capability) - uses within sigma
        cp = spec_width / (6 * sigma_within)

        # Cpk (actual capability) - uses within sigma
        cpu = (usl - process_mean) / (3 * sigma_within)
        cpl = (process_mean - lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)

        # Pp (potential performance) - uses overall sigma
        pp = spec_width / (6 * sigma_overall)

        # Ppk (actual performance) - uses overall sigma
        ppu = (usl - process_mean) / (3 * sigma_overall)
        ppl = (process_mean - lsl) / (3 * sigma_overall)
        ppk = min(ppu, ppl)

        # PPM calculations (using normal distribution)
        from scipy.stats import norm

        # Z-scores
        z_lower = (lsl - process_mean) / sigma_overall
        z_upper = (usl - process_mean) / sigma_overall

        # PPM (parts per million)
        ppm_below_lsl = norm.cdf(z_lower) * 1e6
        ppm_above_usl = (1 - norm.cdf(z_upper)) * 1e6
        ppm_total = ppm_below_lsl + ppm_above_usl

        # Sigma level (based on Cpk)
        sigma_level = 3 * cpk

        # Yield
        yield_percent = (1 - ppm_total / 1e6) * 100

        return CapabilityResult(
            cp=cp,
            cpk=cpk,
            cpl=cpl,
            cpu=cpu,
            pp=pp,
            ppk=ppk,
            ppl=ppl,
            ppu=ppu,
            sigma_within=sigma_within,
            sigma_overall=sigma_overall,
            process_mean=process_mean,
            lsl=lsl,
            usl=usl,
            target=target,
            ppm_below_lsl=ppm_below_lsl,
            ppm_above_usl=ppm_above_usl,
            ppm_total=ppm_total,
            sigma_level=sigma_level,
            yield_percent=yield_percent
        )

    @staticmethod
    def capability_to_sigma_level(cpk: float) -> float:
        """Convert Cpk to sigma level."""
        return 3 * cpk

    @staticmethod
    def sigma_level_to_ppm(sigma: float) -> float:
        """Convert sigma level to expected PPM defects."""
        from scipy.stats import norm
        # Assumes 1.5 sigma shift
        return (1 - norm.cdf(sigma - 1.5) + norm.cdf(-sigma - 1.5)) * 1e6

    @staticmethod
    def get_capability_rating(cpk: float) -> Tuple[str, str]:
        """
        Get capability rating and description.

        Returns:
            Tuple of (rating, description)
        """
        if cpk >= 2.0:
            return "A+", "World Class"
        elif cpk >= 1.67:
            return "A", "Excellent"
        elif cpk >= 1.33:
            return "B", "Capable"
        elif cpk >= 1.0:
            return "C", "Marginally Capable"
        elif cpk >= 0.67:
            return "D", "Poor"
        else:
            return "F", "Inadequate"


# =============================================================================
# HISTOGRAM CALCULATIONS
# =============================================================================

class HistogramCalculator:
    """
    Histogram and distribution analysis calculator.
    """

    @staticmethod
    def calculate_histogram(data: np.ndarray,
                            n_bins: Optional[int] = None,
                            lsl: Optional[float] = None,
                            usl: Optional[float] = None) -> HistogramResult:
        """
        Calculate histogram with distribution statistics.

        Args:
            data: Measurement data
            n_bins: Number of bins (auto if None)
            lsl: Lower spec limit (for range extension)
            usl: Upper spec limit (for range extension)

        Returns:
            HistogramResult with histogram data and statistics
        """
        # Auto-calculate bins using Sturges' rule
        if n_bins is None:
            n_bins = int(np.ceil(np.log2(len(data)) + 1))
            n_bins = max(10, min(50, n_bins))

        # Extend range to include spec limits if provided
        data_min = np.min(data)
        data_max = np.max(data)

        if lsl is not None:
            data_min = min(data_min, lsl)
        if usl is not None:
            data_max = max(data_max, usl)

        # Add margin
        margin = (data_max - data_min) * 0.05
        bin_range = (data_min - margin, data_max + margin)

        # Calculate histogram
        counts, bins = np.histogram(data, bins=n_bins, range=bin_range)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Statistics
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        # Skewness
        n = len(data)
        skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)

        # Kurtosis (excess)
        kurtosis = ((n * (n + 1)) / ((n-1) * (n-2) * (n-3)) *
                    np.sum(((data - mean) / std) ** 4) -
                    (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))

        return HistogramResult(
            bins=bins,
            counts=counts,
            bin_centers=bin_centers,
            mean=mean,
            std=std,
            min_val=np.min(data),
            max_val=np.max(data),
            skewness=skewness,
            kurtosis=kurtosis
        )

    @staticmethod
    def fit_normal_curve(x: np.ndarray, mean: float, std: float,
                         n_samples: int, bin_width: float) -> np.ndarray:
        """
        Calculate normal distribution curve for overlay on histogram.

        Args:
            x: X values for curve
            mean: Distribution mean
            std: Distribution standard deviation
            n_samples: Number of samples (for scaling)
            bin_width: Histogram bin width (for scaling)

        Returns:
            Y values for normal curve
        """
        from scipy.stats import norm
        return n_samples * bin_width * norm.pdf(x, mean, std)


# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_spc_sample_data(n_subgroups: int = 30,
                              subgroup_size: int = 5,
                              mean: float = 1000.0,
                              std: float = 5.0,
                              include_outliers: bool = True) -> np.ndarray:
    """
    Generate sample SPC data for demonstration.

    Args:
        n_subgroups: Number of subgroups
        subgroup_size: Samples per subgroup
        mean: Process mean
        std: Process standard deviation
        include_outliers: Whether to include out-of-control points

    Returns:
        2D array of shape (n_subgroups, subgroup_size)
    """
    np.random.seed(42)
    data = np.random.normal(mean, std, (n_subgroups, subgroup_size))

    if include_outliers:
        # Add some out-of-control points
        data[5, 0] = mean + 4 * std  # Point beyond UCL
        data[15:23, :] = data[15:23, :] + 1.5 * std  # 8 consecutive above CL

    return data


def generate_capability_sample_data(n_samples: int = 200,
                                     mean: float = 1000.0,
                                     std: float = 3.0,
                                     lsl: float = 990.0,
                                     usl: float = 1010.0) -> Tuple[np.ndarray, float, float]:
    """
    Generate sample capability data.

    Args:
        n_samples: Number of samples
        mean: Process mean
        std: Process standard deviation
        lsl: Lower specification limit
        usl: Upper specification limit

    Returns:
        Tuple of (data, lsl, usl)
    """
    np.random.seed(42)
    data = np.random.normal(mean, std, n_samples)
    return data, lsl, usl
