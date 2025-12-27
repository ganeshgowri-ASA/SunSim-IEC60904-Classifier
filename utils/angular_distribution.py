"""
Sun Simulator Classification System - Angular Distribution Module
FOV/Solid Angle Analysis for Solar Simulator Characterization

This module provides calculations and analysis for angular distribution
characteristics of solar simulators, including beam collimation,
angle-of-incidence responsivity, and diffuse/direct irradiance ratios.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================

class BeamType(Enum):
    """Solar simulator beam types."""
    COLLIMATED = "collimated"
    SEMI_COLLIMATED = "semi_collimated"
    DIVERGENT = "divergent"
    DIFFUSE = "diffuse"


class SimulatorType(Enum):
    """Solar simulator optical configurations."""
    FRESNEL = "fresnel"  # Fresnel lens collimation
    PARABOLIC = "parabolic"  # Parabolic mirror
    ELLIPTICAL = "elliptical"  # Elliptical reflector
    DIRECT = "direct"  # No collimation optics
    INTEGRATING_SPHERE = "integrating_sphere"  # Diffuse light source


# IEC 60904-9 angular distribution requirements
ANGULAR_REQUIREMENTS = {
    'max_half_angle_deg': 3.0,  # Maximum half-angle for "collimated"
    'sun_half_angle_deg': 0.267,  # Natural sun half-angle (0.533° full)
    'target_solid_angle_sr': 6.8e-5,  # Target solid angle (sun)
}

# Typical solar simulator specifications
TYPICAL_SPECS = {
    'fresnel': {'half_angle': 2.5, 'collimation': 'high'},
    'parabolic': {'half_angle': 1.5, 'collimation': 'very_high'},
    'elliptical': {'half_angle': 5.0, 'collimation': 'medium'},
    'direct': {'half_angle': 15.0, 'collimation': 'low'},
    'integrating_sphere': {'half_angle': 90.0, 'collimation': 'none'},
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BeamCollimationResult:
    """Result of beam collimation analysis."""
    half_angle_deg: float
    full_angle_deg: float
    solid_angle_sr: float
    sun_ratio: float  # Ratio to natural sun solid angle
    beam_type: BeamType
    meets_iec_requirement: bool
    collimation_quality: str  # 'excellent', 'good', 'acceptable', 'poor'
    notes: List[str]


@dataclass
class AngleOfIncidenceResult:
    """Angle-of-incidence responsivity analysis."""
    angles_deg: np.ndarray
    responsivity: np.ndarray
    normalized_responsivity: np.ndarray
    cosine_factor: np.ndarray
    deviation_from_cosine: np.ndarray
    max_deviation_percent: float
    weighted_average_angle_deg: float
    effective_incidence_angle_deg: float
    notes: List[str]


@dataclass
class DiffuseDirectRatio:
    """Diffuse to direct irradiance ratio analysis."""
    total_irradiance_W_m2: float
    direct_irradiance_W_m2: float
    diffuse_irradiance_W_m2: float
    diffuse_fraction: float
    direct_fraction: float
    circumsolar_irradiance_W_m2: float
    horizon_brightening_W_m2: float
    ambient_light_W_m2: float
    meets_specification: bool
    notes: List[str]


@dataclass
class AngularDistributionProfile:
    """Complete angular distribution profile."""
    angles_deg: np.ndarray
    intensity: np.ndarray
    normalized_intensity: np.ndarray
    cumulative_energy: np.ndarray
    fwhm_deg: float  # Full width at half maximum
    beam_divergence_deg: float
    centroid_angle_deg: float
    symmetry_factor: float  # 1.0 = perfect symmetry
    profile_type: str  # 'gaussian', 'flat_top', 'donut', 'irregular'
    notes: List[str]


@dataclass
class FOVAnalysisResult:
    """Field of view analysis for detector/module."""
    detector_fov_deg: float
    source_angle_deg: float
    angular_mismatch_deg: float
    geometric_factor: float
    recommended_correction: float
    vignetting_present: bool
    notes: List[str]


# =============================================================================
# SOLID ANGLE CALCULATIONS
# =============================================================================

def calculate_solid_angle(half_angle_deg: float) -> float:
    """
    Calculate solid angle from half-angle of a cone.

    Solid angle = 2π(1 - cos(θ)) steradians

    Args:
        half_angle_deg: Half-angle of cone in degrees

    Returns:
        Solid angle in steradians
    """
    half_angle_rad = np.radians(half_angle_deg)
    return 2 * np.pi * (1 - np.cos(half_angle_rad))


def calculate_half_angle(solid_angle_sr: float) -> float:
    """
    Calculate half-angle from solid angle.

    Args:
        solid_angle_sr: Solid angle in steradians

    Returns:
        Half-angle in degrees
    """
    cos_theta = 1 - solid_angle_sr / (2 * np.pi)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))


def get_sun_solid_angle() -> Dict[str, float]:
    """
    Get natural sun angular parameters.

    Returns:
        Dictionary with sun angular specifications
    """
    sun_half_angle = ANGULAR_REQUIREMENTS['sun_half_angle_deg']
    solid_angle = calculate_solid_angle(sun_half_angle)

    return {
        'half_angle_deg': sun_half_angle,
        'full_angle_deg': sun_half_angle * 2,
        'solid_angle_sr': solid_angle,
        'sun_diameter_arcmin': sun_half_angle * 2 * 60  # ~32 arcmin
    }


# =============================================================================
# BEAM COLLIMATION ANALYSIS
# =============================================================================

def analyze_beam_collimation(
    half_angle_deg: float,
    source_type: str = 'unknown'
) -> BeamCollimationResult:
    """
    Analyze beam collimation quality.

    Args:
        half_angle_deg: Measured half-angle in degrees
        source_type: Type of light source/optics

    Returns:
        BeamCollimationResult with analysis
    """
    notes = []

    # Calculate solid angle
    solid_angle = calculate_solid_angle(half_angle_deg)
    sun_solid_angle = calculate_solid_angle(ANGULAR_REQUIREMENTS['sun_half_angle_deg'])
    sun_ratio = solid_angle / sun_solid_angle

    # Determine beam type
    if half_angle_deg <= ANGULAR_REQUIREMENTS['sun_half_angle_deg'] * 2:
        beam_type = BeamType.COLLIMATED
    elif half_angle_deg <= ANGULAR_REQUIREMENTS['max_half_angle_deg']:
        beam_type = BeamType.SEMI_COLLIMATED
    elif half_angle_deg <= 15:
        beam_type = BeamType.DIVERGENT
    else:
        beam_type = BeamType.DIFFUSE

    # Check IEC requirement
    meets_iec = half_angle_deg <= ANGULAR_REQUIREMENTS['max_half_angle_deg']

    # Determine quality
    if half_angle_deg <= 1.0:
        quality = 'excellent'
        notes.append("Excellent collimation - suitable for all module types")
    elif half_angle_deg <= ANGULAR_REQUIREMENTS['max_half_angle_deg']:
        quality = 'good'
        notes.append("Good collimation - meets IEC requirements")
    elif half_angle_deg <= 5.0:
        quality = 'acceptable'
        notes.append("Acceptable for standard modules, may affect concentrator testing")
    else:
        quality = 'poor'
        notes.append("Poor collimation - may affect measurement accuracy")

    # Add comparison to sun
    notes.append(f"Solid angle is {sun_ratio:.1f}x the natural sun")

    if source_type in TYPICAL_SPECS:
        expected = TYPICAL_SPECS[source_type]['half_angle']
        if half_angle_deg > expected * 1.5:
            notes.append(f"Warning: Higher than typical for {source_type} optics")
        elif half_angle_deg < expected * 0.5:
            notes.append(f"Better than typical for {source_type} optics")

    return BeamCollimationResult(
        half_angle_deg=half_angle_deg,
        full_angle_deg=half_angle_deg * 2,
        solid_angle_sr=solid_angle,
        sun_ratio=sun_ratio,
        beam_type=beam_type,
        meets_iec_requirement=meets_iec,
        collimation_quality=quality,
        notes=notes
    )


def measure_beam_angle_from_profile(
    position_mm: np.ndarray,
    intensity: np.ndarray,
    distance_mm: float
) -> float:
    """
    Calculate beam half-angle from intensity profile measurement.

    Args:
        position_mm: Position across beam in mm
        intensity: Intensity values at each position
        distance_mm: Distance from source to measurement plane

    Returns:
        Beam half-angle in degrees
    """
    # Normalize intensity
    intensity_norm = intensity / np.max(intensity)

    # Find FWHM
    half_max = 0.5
    above_half = intensity_norm >= half_max

    if np.sum(above_half) < 2:
        # Use 10% threshold for wide beams
        above_half = intensity_norm >= 0.1

    if np.sum(above_half) >= 2:
        indices = np.where(above_half)[0]
        beam_width_mm = position_mm[indices[-1]] - position_mm[indices[0]]
    else:
        # Use full width
        beam_width_mm = position_mm[-1] - position_mm[0]

    # Calculate half-angle
    half_width_mm = beam_width_mm / 2
    half_angle_rad = np.arctan(half_width_mm / distance_mm)

    return np.degrees(half_angle_rad)


# =============================================================================
# ANGLE-OF-INCIDENCE ANALYSIS
# =============================================================================

def analyze_angle_of_incidence(
    angles_deg: np.ndarray,
    response: np.ndarray,
    reference_angle_deg: float = 0.0
) -> AngleOfIncidenceResult:
    """
    Analyze angle-of-incidence responsivity.

    Args:
        angles_deg: Angles of incidence in degrees
        response: Measured response at each angle
        reference_angle_deg: Reference angle for normalization

    Returns:
        AngleOfIncidenceResult with analysis
    """
    notes = []

    # Normalize to reference angle
    ref_idx = np.argmin(np.abs(angles_deg - reference_angle_deg))
    ref_response = response[ref_idx]
    normalized = response / ref_response if ref_response > 0 else response

    # Calculate ideal cosine response
    angles_rad = np.radians(angles_deg)
    cosine_factor = np.cos(angles_rad)

    # Deviation from cosine law
    deviation = (normalized - cosine_factor) / cosine_factor * 100
    deviation = np.where(np.isfinite(deviation), deviation, 0)
    max_deviation = np.max(np.abs(deviation[np.abs(angles_deg) < 80]))

    # Calculate weighted average angle
    weights = response * np.cos(angles_rad)
    weighted_avg = np.average(np.abs(angles_deg), weights=weights)

    # Calculate effective incidence angle (angle at which response is 50%)
    half_response = 0.5 * np.max(normalized)
    above_half = normalized >= half_response
    if np.sum(above_half) > 0:
        indices = np.where(above_half)[0]
        effective_angle = np.abs(angles_deg[indices[-1]])
    else:
        effective_angle = 90.0

    # Assessment
    if max_deviation < 2:
        notes.append("Excellent cosine response - very close to ideal")
    elif max_deviation < 5:
        notes.append("Good cosine response - within typical specifications")
    elif max_deviation < 10:
        notes.append("Acceptable cosine response - may need correction for wide angles")
    else:
        notes.append("Significant deviation from cosine law - corrections recommended")

    return AngleOfIncidenceResult(
        angles_deg=angles_deg,
        responsivity=response,
        normalized_responsivity=normalized,
        cosine_factor=cosine_factor,
        deviation_from_cosine=deviation,
        max_deviation_percent=max_deviation,
        weighted_average_angle_deg=weighted_avg,
        effective_incidence_angle_deg=effective_angle,
        notes=notes
    )


def calculate_cosine_correction(angle_deg: float, n_glass: float = 1.526) -> float:
    """
    Calculate cosine correction factor including Fresnel reflection.

    For encapsulated cells, accounts for reflection at air-glass interface.

    Args:
        angle_deg: Angle of incidence in degrees
        n_glass: Refractive index of cover glass

    Returns:
        Correction factor to apply to measurement
    """
    angle_rad = np.radians(angle_deg)

    # Simple cosine factor
    cosine_factor = np.cos(angle_rad)

    # Fresnel reflection (approximate for unpolarized light)
    n1 = 1.0  # air
    n2 = n_glass

    # Snell's law for transmitted angle
    sin_t = n1 / n2 * np.sin(angle_rad)
    if np.abs(sin_t) > 1:
        return 0  # Total internal reflection

    angle_t = np.arcsin(sin_t)

    # Fresnel equations (average of s and p polarization)
    rs = ((n1 * np.cos(angle_rad) - n2 * np.cos(angle_t)) /
          (n1 * np.cos(angle_rad) + n2 * np.cos(angle_t))) ** 2
    rp = ((n1 * np.cos(angle_t) - n2 * np.cos(angle_rad)) /
          (n1 * np.cos(angle_t) + n2 * np.cos(angle_rad))) ** 2

    reflectance = (rs + rp) / 2
    transmittance = 1 - reflectance

    return cosine_factor * transmittance


# =============================================================================
# DIFFUSE/DIRECT RATIO ANALYSIS
# =============================================================================

def analyze_diffuse_direct_ratio(
    total_irradiance: float,
    direct_normal: float,
    diffuse_horizontal: float = None,
    circumsolar_contribution: float = 0.0,
    ambient_light: float = 0.0
) -> DiffuseDirectRatio:
    """
    Analyze diffuse to direct irradiance ratio.

    Args:
        total_irradiance: Total measured irradiance (W/m²)
        direct_normal: Direct normal irradiance (W/m²)
        diffuse_horizontal: Diffuse horizontal irradiance (W/m²)
        circumsolar_contribution: Circumsolar irradiance (W/m²)
        ambient_light: Ambient/stray light contribution (W/m²)

    Returns:
        DiffuseDirectRatio analysis result
    """
    notes = []

    # Calculate diffuse if not provided
    if diffuse_horizontal is None:
        diffuse_horizontal = total_irradiance - direct_normal - ambient_light

    # Ensure non-negative
    diffuse_horizontal = max(0, diffuse_horizontal)
    direct_normal = max(0, direct_normal)

    # Calculate fractions
    total_valid = total_irradiance - ambient_light
    if total_valid > 0:
        diffuse_fraction = diffuse_horizontal / total_valid
        direct_fraction = direct_normal / total_valid
    else:
        diffuse_fraction = 0
        direct_fraction = 0

    # Assess quality
    # For solar simulators, diffuse fraction should be low
    if diffuse_fraction < 0.02:
        notes.append("Excellent - minimal diffuse component")
        meets_spec = True
    elif diffuse_fraction < 0.05:
        notes.append("Good - low diffuse contribution")
        meets_spec = True
    elif diffuse_fraction < 0.10:
        notes.append("Acceptable - moderate diffuse component")
        meets_spec = True
    else:
        notes.append("Warning - high diffuse fraction may affect measurements")
        meets_spec = False

    # Circumsolar assessment
    if circumsolar_contribution > total_irradiance * 0.05:
        notes.append("Significant circumsolar contribution detected")

    # Ambient light assessment
    if ambient_light > total_irradiance * 0.01:
        notes.append(f"Ambient light contribution: {ambient_light:.1f} W/m² ({ambient_light/total_irradiance*100:.1f}%)")

    return DiffuseDirectRatio(
        total_irradiance_W_m2=total_irradiance,
        direct_irradiance_W_m2=direct_normal,
        diffuse_irradiance_W_m2=diffuse_horizontal,
        diffuse_fraction=diffuse_fraction,
        direct_fraction=direct_fraction,
        circumsolar_irradiance_W_m2=circumsolar_contribution,
        horizon_brightening_W_m2=0,  # Typically zero for simulators
        ambient_light_W_m2=ambient_light,
        meets_specification=meets_spec,
        notes=notes
    )


# =============================================================================
# ANGULAR DISTRIBUTION PROFILE
# =============================================================================

def analyze_angular_profile(
    angles_deg: np.ndarray,
    intensity: np.ndarray
) -> AngularDistributionProfile:
    """
    Analyze angular distribution profile of light source.

    Args:
        angles_deg: Angular positions in degrees
        intensity: Intensity at each angle

    Returns:
        AngularDistributionProfile analysis
    """
    notes = []

    # Normalize intensity
    max_intensity = np.max(intensity)
    normalized = intensity / max_intensity if max_intensity > 0 else intensity

    # Calculate FWHM
    half_max_indices = np.where(normalized >= 0.5)[0]
    if len(half_max_indices) >= 2:
        fwhm = angles_deg[half_max_indices[-1]] - angles_deg[half_max_indices[0]]
    else:
        fwhm = angles_deg[-1] - angles_deg[0]

    # Calculate centroid
    weights = normalized
    centroid = np.average(angles_deg, weights=weights)

    # Calculate cumulative energy
    sorted_idx = np.argsort(angles_deg)
    sorted_angles = angles_deg[sorted_idx]
    sorted_intensity = intensity[sorted_idx]
    cumulative = np.cumsum(sorted_intensity) / np.sum(sorted_intensity)

    # Beam divergence (angle containing 90% of energy)
    idx_90 = np.searchsorted(cumulative, 0.9)
    if idx_90 < len(sorted_angles):
        divergence = sorted_angles[idx_90] - sorted_angles[0]
    else:
        divergence = fwhm * 1.5

    # Symmetry factor
    center_idx = len(angles_deg) // 2
    left_half = normalized[:center_idx]
    right_half = normalized[center_idx:][::-1]
    min_len = min(len(left_half), len(right_half))
    if min_len > 0:
        symmetry = 1 - np.mean(np.abs(left_half[:min_len] - right_half[:min_len]))
    else:
        symmetry = 1.0

    # Determine profile type
    peak_idx = np.argmax(normalized)
    peak_pos = angles_deg[peak_idx]

    if np.abs(peak_pos - centroid) < fwhm * 0.1:
        # Peak near center
        if normalized[0] < 0.1 and normalized[-1] < 0.1:
            if np.std(normalized[normalized > 0.9]) < 0.05:
                profile_type = 'flat_top'
                notes.append("Flat-top profile - good for uniform illumination")
            else:
                profile_type = 'gaussian'
                notes.append("Gaussian-like profile - typical for collimated sources")
        else:
            profile_type = 'gaussian'
    else:
        if normalized[peak_idx] < normalized[0] or normalized[peak_idx] < normalized[-1]:
            profile_type = 'donut'
            notes.append("Donut profile - central minimum detected")
        else:
            profile_type = 'irregular'
            notes.append("Irregular profile - asymmetric distribution")

    # Quality assessment
    if symmetry > 0.95:
        notes.append("Excellent symmetry")
    elif symmetry > 0.9:
        notes.append("Good symmetry")
    else:
        notes.append("Asymmetric profile - may affect uniformity")

    return AngularDistributionProfile(
        angles_deg=angles_deg,
        intensity=intensity,
        normalized_intensity=normalized,
        cumulative_energy=cumulative,
        fwhm_deg=fwhm,
        beam_divergence_deg=divergence,
        centroid_angle_deg=centroid,
        symmetry_factor=symmetry,
        profile_type=profile_type,
        notes=notes
    )


# =============================================================================
# FOV ANALYSIS
# =============================================================================

def analyze_fov_mismatch(
    detector_fov_deg: float,
    source_angle_deg: float,
    detector_distance_mm: float = None,
    source_distance_mm: float = None
) -> FOVAnalysisResult:
    """
    Analyze field-of-view mismatch between detector and source.

    Args:
        detector_fov_deg: Detector field of view in degrees
        source_angle_deg: Source beam angle in degrees
        detector_distance_mm: Distance to detector (optional)
        source_distance_mm: Distance from source (optional)

    Returns:
        FOVAnalysisResult with analysis
    """
    notes = []

    # Angular mismatch
    mismatch = np.abs(detector_fov_deg - source_angle_deg)

    # Geometric factor (ratio of solid angles)
    detector_solid = calculate_solid_angle(detector_fov_deg / 2)
    source_solid = calculate_solid_angle(source_angle_deg / 2)
    geometric_factor = min(detector_solid, source_solid) / max(detector_solid, source_solid)

    # Check for vignetting
    vignetting = detector_fov_deg < source_angle_deg

    # Calculate correction
    if vignetting:
        # Detector sees less than full beam
        correction = source_solid / detector_solid
        notes.append("Vignetting present - detector FOV smaller than source angle")
        notes.append(f"Apply correction factor of {correction:.3f}")
    else:
        correction = 1.0
        notes.append("No vignetting - detector captures full beam")

    # Recommendations
    if mismatch < 1:
        notes.append("FOV well matched to source")
    elif mismatch < 5:
        notes.append("Minor FOV mismatch - acceptable for most applications")
    else:
        notes.append("Significant FOV mismatch - consider adjustment")

    return FOVAnalysisResult(
        detector_fov_deg=detector_fov_deg,
        source_angle_deg=source_angle_deg,
        angular_mismatch_deg=mismatch,
        geometric_factor=geometric_factor,
        recommended_correction=correction,
        vignetting_present=vignetting,
        notes=notes
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_gaussian_profile(
    half_angle_deg: float,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Gaussian angular distribution profile.

    Args:
        half_angle_deg: Half-angle at 1/e² intensity
        num_points: Number of points in profile

    Returns:
        Tuple of (angles, intensity)
    """
    angles = np.linspace(-half_angle_deg * 2, half_angle_deg * 2, num_points)
    sigma = half_angle_deg / 2  # 1/e² at half_angle

    intensity = np.exp(-angles**2 / (2 * sigma**2))

    return angles, intensity


def generate_flat_top_profile(
    half_angle_deg: float,
    edge_steepness: float = 0.9,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a flat-top angular distribution profile.

    Args:
        half_angle_deg: Half-angle of flat region
        edge_steepness: Steepness of edges (0-1)
        num_points: Number of points in profile

    Returns:
        Tuple of (angles, intensity)
    """
    angles = np.linspace(-half_angle_deg * 2, half_angle_deg * 2, num_points)

    # Super-Gaussian profile for flat top
    n = 2 + 8 * edge_steepness  # Exponent (higher = steeper edges)
    intensity = np.exp(-(np.abs(angles) / half_angle_deg) ** n)

    return angles, intensity


def calculate_effective_irradiance_angle(
    angles_deg: np.ndarray,
    intensity: np.ndarray,
    module_aoi_response: np.ndarray = None
) -> float:
    """
    Calculate effective angle of incidence for irradiance weighting.

    Args:
        angles_deg: Angular distribution angles
        intensity: Angular distribution intensity
        module_aoi_response: Module AOI response (optional)

    Returns:
        Effective angle in degrees
    """
    if module_aoi_response is None:
        # Use cosine response
        module_aoi_response = np.cos(np.radians(angles_deg))

    # Weight by both angular distribution and module response
    weights = intensity * module_aoi_response
    weights = np.maximum(weights, 0)

    if np.sum(weights) > 0:
        effective_angle = np.average(np.abs(angles_deg), weights=weights)
    else:
        effective_angle = 0.0

    return effective_angle


def get_angular_summary_table() -> List[Dict[str, Any]]:
    """
    Get summary table of angular specifications for different simulator types.

    Returns:
        List of dictionaries with angular specifications
    """
    summary = []

    for sim_type, specs in TYPICAL_SPECS.items():
        solid_angle = calculate_solid_angle(specs['half_angle'])
        sun_ratio = solid_angle / calculate_solid_angle(ANGULAR_REQUIREMENTS['sun_half_angle_deg'])

        summary.append({
            'Simulator Type': sim_type.replace('_', ' ').title(),
            'Half-Angle (deg)': specs['half_angle'],
            'Solid Angle (sr)': f"{solid_angle:.2e}",
            'Sun Ratio': f"{sun_ratio:.1f}x",
            'Collimation': specs['collimation'].replace('_', ' ').title(),
            'IEC Compliant': 'Yes' if specs['half_angle'] <= ANGULAR_REQUIREMENTS['max_half_angle_deg'] else 'No'
        })

    return summary


def comprehensive_angular_analysis(
    beam_half_angle_deg: float,
    aoi_angles_deg: np.ndarray = None,
    aoi_response: np.ndarray = None,
    total_irradiance: float = 1000.0,
    direct_fraction: float = 0.98,
    simulator_type: str = 'unknown'
) -> Dict[str, Any]:
    """
    Perform comprehensive angular distribution analysis.

    Args:
        beam_half_angle_deg: Beam half-angle in degrees
        aoi_angles_deg: AOI measurement angles (optional)
        aoi_response: AOI response values (optional)
        total_irradiance: Total irradiance W/m²
        direct_fraction: Fraction of direct irradiance
        simulator_type: Type of simulator

    Returns:
        Dictionary with complete analysis results
    """
    # Beam collimation analysis
    collimation = analyze_beam_collimation(beam_half_angle_deg, simulator_type)

    # Generate synthetic angular profile if not provided
    if aoi_angles_deg is None:
        aoi_angles_deg = np.linspace(-30, 30, 61)
        _, aoi_response = generate_gaussian_profile(beam_half_angle_deg)
        aoi_response = np.interp(aoi_angles_deg,
                                  np.linspace(-beam_half_angle_deg*2, beam_half_angle_deg*2, 100),
                                  aoi_response)

    # AOI analysis
    aoi_analysis = analyze_angle_of_incidence(aoi_angles_deg, aoi_response)

    # Angular profile analysis
    profile_analysis = analyze_angular_profile(aoi_angles_deg, aoi_response)

    # Diffuse/direct analysis
    direct_irradiance = total_irradiance * direct_fraction
    diffuse_irradiance = total_irradiance * (1 - direct_fraction)
    diffuse_direct = analyze_diffuse_direct_ratio(
        total_irradiance, direct_irradiance, diffuse_irradiance
    )

    # Calculate effective angle
    effective_angle = calculate_effective_irradiance_angle(
        aoi_angles_deg, aoi_response
    )

    return {
        'collimation': collimation,
        'angle_of_incidence': aoi_analysis,
        'angular_profile': profile_analysis,
        'diffuse_direct': diffuse_direct,
        'effective_angle_deg': effective_angle,
        'summary': {
            'beam_half_angle_deg': beam_half_angle_deg,
            'solid_angle_sr': collimation.solid_angle_sr,
            'collimation_quality': collimation.collimation_quality,
            'meets_iec': collimation.meets_iec_requirement,
            'profile_type': profile_analysis.profile_type,
            'fwhm_deg': profile_analysis.fwhm_deg,
            'symmetry': profile_analysis.symmetry_factor,
            'diffuse_fraction': diffuse_direct.diffuse_fraction
        }
    }
