"""
IEC 60904-9 Ed.3 Classification Calculations.

This module implements the classification algorithms for sun simulators
according to IEC 60904-9 Edition 3 standard.

Classification is based on three criteria:
1. Spectral Match (A, B, or C)
2. Non-uniformity of Irradiance (A, B, or C)
3. Temporal Instability (A, B, or C)

The overall classification combines all three (e.g., AAA, ABA, BBC, etc.)
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# IEC 60904-9 Ed.3 Classification Limits
IEC_60904_9_LIMITS = {
    "spectral_match": {
        # Deviation from reference spectrum per wavelength band
        "A": 0.25,   # +/- 25%
        "B": 0.40,   # +/- 40%
        "C": float('inf'),  # Greater than 40%
    },
    "uniformity": {
        # Maximum non-uniformity percentage
        "A": 2.0,    # +/- 2%
        "B": 5.0,    # +/- 5%
        "C": 10.0,   # +/- 10%
    },
    "temporal_stability": {
        # Short-term instability (STI) and Long-term instability (LTI)
        "A": {"sti": 0.5, "lti": 2.0},   # 0.5% STI, 2% LTI
        "B": {"sti": 2.0, "lti": 5.0},   # 2% STI, 5% LTI
        "C": {"sti": 10.0, "lti": 10.0}, # 10% STI, 10% LTI
    },
    "wavelength_bands": [
        # IEC 60904-9 Ed.3 wavelength bands (nm) and reference percentages
        {"start": 400, "end": 500, "reference": 18.4},
        {"start": 500, "end": 600, "reference": 19.9},
        {"start": 600, "end": 700, "reference": 18.4},
        {"start": 700, "end": 800, "reference": 14.9},
        {"start": 800, "end": 900, "reference": 12.5},
        {"start": 900, "end": 1100, "reference": 15.9},
    ]
}


@dataclass
class SpectralMatchResult:
    """Result of spectral match analysis."""
    classification: str
    band_results: List[Dict]
    max_deviation: float
    pass_all_bands: bool


@dataclass
class UniformityResult:
    """Result of uniformity analysis."""
    classification: str
    max_deviation: float
    min_irradiance: float
    max_irradiance: float
    mean_irradiance: float
    uniformity_percentage: float


@dataclass
class TemporalStabilityResult:
    """Result of temporal stability analysis."""
    classification: str
    short_term_instability: float
    long_term_instability: float


def calculate_spectral_match_class(
    measured_percentages: List[float],
    reference_percentages: Optional[List[float]] = None
) -> SpectralMatchResult:
    """
    Calculate spectral match classification according to IEC 60904-9 Ed.3.

    Args:
        measured_percentages: List of measured spectral percentages per band
        reference_percentages: Reference percentages (defaults to AM1.5G)

    Returns:
        SpectralMatchResult with classification and details
    """
    if reference_percentages is None:
        reference_percentages = [
            band["reference"] for band in IEC_60904_9_LIMITS["wavelength_bands"]
        ]

    if len(measured_percentages) != len(reference_percentages):
        raise ValueError(
            f"Measured ({len(measured_percentages)}) and reference "
            f"({len(reference_percentages)}) bands must match"
        )

    band_results = []
    max_deviation = 0.0
    all_pass = True
    worst_class = "A"

    limits = IEC_60904_9_LIMITS["spectral_match"]
    bands = IEC_60904_9_LIMITS["wavelength_bands"]

    for i, (measured, reference) in enumerate(zip(measured_percentages, reference_percentages)):
        # Calculate spectral match ratio
        if reference > 0:
            ratio = measured / reference
            deviation = abs(ratio - 1.0)
        else:
            ratio = 0.0
            deviation = 1.0

        max_deviation = max(max_deviation, deviation)

        # Determine class for this band
        if deviation <= limits["A"]:
            band_class = "A"
            passes = True
        elif deviation <= limits["B"]:
            band_class = "B"
            passes = True
            if worst_class == "A":
                worst_class = "B"
        else:
            band_class = "C"
            passes = True
            worst_class = "C"

        band_results.append({
            "band": f"{bands[i]['start']}-{bands[i]['end']} nm",
            "measured": measured,
            "reference": reference,
            "ratio": ratio,
            "deviation_percent": deviation * 100,
            "classification": band_class,
            "passes": passes,
        })

    return SpectralMatchResult(
        classification=worst_class,
        band_results=band_results,
        max_deviation=max_deviation * 100,
        pass_all_bands=all_pass,
    )


def calculate_uniformity_class(irradiance_map: np.ndarray) -> UniformityResult:
    """
    Calculate non-uniformity of irradiance according to IEC 60904-9 Ed.3.

    Args:
        irradiance_map: 2D array of irradiance values across measurement area

    Returns:
        UniformityResult with classification and details
    """
    # Calculate statistics
    mean_val = np.mean(irradiance_map)
    min_val = np.min(irradiance_map)
    max_val = np.max(irradiance_map)

    if mean_val > 0:
        # Non-uniformity as percentage deviation from mean
        max_deviation = max(abs(max_val - mean_val), abs(mean_val - min_val))
        uniformity_pct = (max_deviation / mean_val) * 100
    else:
        uniformity_pct = 100.0
        max_deviation = 0

    # Determine classification
    limits = IEC_60904_9_LIMITS["uniformity"]
    if uniformity_pct <= limits["A"]:
        classification = "A"
    elif uniformity_pct <= limits["B"]:
        classification = "B"
    elif uniformity_pct <= limits["C"]:
        classification = "C"
    else:
        classification = "C"  # Beyond C limits

    return UniformityResult(
        classification=classification,
        max_deviation=max_deviation,
        min_irradiance=float(min_val),
        max_irradiance=float(max_val),
        mean_irradiance=float(mean_val),
        uniformity_percentage=float(uniformity_pct),
    )


def calculate_temporal_stability_class(
    time_series: np.ndarray,
    sample_rate_hz: float = 1000.0
) -> TemporalStabilityResult:
    """
    Calculate temporal instability according to IEC 60904-9 Ed.3.

    Args:
        time_series: Array of irradiance values over time
        sample_rate_hz: Sampling rate in Hz

    Returns:
        TemporalStabilityResult with classification and details
    """
    if len(time_series) < 2:
        return TemporalStabilityResult(
            classification="C",
            short_term_instability=100.0,
            long_term_instability=100.0,
        )

    mean_val = np.mean(time_series)
    if mean_val <= 0:
        return TemporalStabilityResult(
            classification="C",
            short_term_instability=100.0,
            long_term_instability=100.0,
        )

    # Short-term instability (STI) - variation within short windows
    window_samples = int(0.001 * sample_rate_hz)  # 1ms window
    if window_samples < 1:
        window_samples = 1

    sti_values = []
    for i in range(0, len(time_series) - window_samples, window_samples):
        window = time_series[i:i + window_samples]
        window_range = np.max(window) - np.min(window)
        sti_values.append(window_range / mean_val * 100)

    sti = np.max(sti_values) if sti_values else 0

    # Long-term instability (LTI) - overall variation
    lti = (np.max(time_series) - np.min(time_series)) / mean_val * 100

    # Determine classification based on worst case
    limits = IEC_60904_9_LIMITS["temporal_stability"]

    if sti <= limits["A"]["sti"] and lti <= limits["A"]["lti"]:
        classification = "A"
    elif sti <= limits["B"]["sti"] and lti <= limits["B"]["lti"]:
        classification = "B"
    else:
        classification = "C"

    return TemporalStabilityResult(
        classification=classification,
        short_term_instability=float(sti),
        long_term_instability=float(lti),
    )


def get_overall_classification(
    spectral: str,
    uniformity: str,
    temporal: str
) -> str:
    """
    Combine individual classifications into overall class.

    Args:
        spectral: Spectral match class (A, B, or C)
        uniformity: Uniformity class (A, B, or C)
        temporal: Temporal stability class (A, B, or C)

    Returns:
        Overall classification string (e.g., "AAA", "ABA", "BBC")
    """
    valid_classes = {"A", "B", "C"}
    if spectral not in valid_classes or uniformity not in valid_classes or temporal not in valid_classes:
        raise ValueError(f"Invalid classification values: {spectral}, {uniformity}, {temporal}")

    return f"{spectral}{uniformity}{temporal}"


def interpret_classification(overall_class: str) -> Dict[str, str]:
    """
    Provide interpretation of the overall classification.

    Args:
        overall_class: Three-letter classification (e.g., "AAA")

    Returns:
        Dictionary with interpretation details
    """
    if len(overall_class) != 3:
        raise ValueError(f"Invalid classification: {overall_class}")

    class_quality = {
        "A": "Excellent",
        "B": "Good",
        "C": "Acceptable"
    }

    spectral, uniformity, temporal = overall_class

    # Determine overall quality
    if overall_class == "AAA":
        quality = "Reference Grade"
        description = "Highest quality sun simulator suitable for reference cell calibration."
    elif all(c in "AB" for c in overall_class):
        quality = "High Grade"
        description = "High quality sun simulator suitable for precision measurements."
    elif all(c in "ABC" for c in overall_class) and overall_class.count("C") == 1:
        quality = "Standard Grade"
        description = "Standard quality sun simulator suitable for routine testing."
    else:
        quality = "Basic Grade"
        description = "Basic sun simulator suitable for comparative testing."

    return {
        "overall_class": overall_class,
        "quality_grade": quality,
        "description": description,
        "spectral_quality": class_quality.get(spectral, "Unknown"),
        "uniformity_quality": class_quality.get(uniformity, "Unknown"),
        "temporal_quality": class_quality.get(temporal, "Unknown"),
    }
