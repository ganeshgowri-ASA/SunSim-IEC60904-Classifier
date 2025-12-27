"""
Sun Simulator Classification System - Calculations Module
IEC 60904-9:2020 Ed.3 Compliant Calculations

This module provides all calculation functions for:
- Spectral Match (6-band analysis)
- Spatial Uniformity
- Temporal Stability (STI/LTI)
- SPC (Spectral Performance Category)
- SPD (Spectral Performance Deviation)
"""

import numpy as np
import pandas as pd
from scipy import integrate, interpolate
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import os

from config import (
    WAVELENGTH_BANDS, EXTENDED_WAVELENGTH_BANDS,
    CLASSIFICATION, get_classification, get_overall_classification,
    WAVELENGTH_RANGE, AM15G_CONFIG
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SpectralMatchResult:
    """Results from spectral match calculation."""
    band_ratios: Dict[int, float]
    band_classifications: Dict[int, str]
    max_deviation: float
    overall_classification: str
    spc: float
    spd: float
    band_details: List[Dict[str, Any]]


@dataclass
class UniformityResult:
    """Results from uniformity calculation."""
    non_uniformity: float
    classification: str
    mean_irradiance: float
    max_irradiance: float
    min_irradiance: float
    std_deviation: float
    normalized_data: np.ndarray
    deviation_map: np.ndarray


@dataclass
class TemporalResult:
    """Results from temporal stability calculation."""
    sti: float
    lti: float
    sti_classification: str
    lti_classification: str
    overall_classification: str
    pulse_profile: np.ndarray
    time_series: np.ndarray


# =============================================================================
# REFERENCE SPECTRUM HANDLING
# =============================================================================

def load_am15g_reference(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load AM1.5G reference spectrum from file.

    Args:
        filepath: Path to CSV file with wavelength and irradiance columns

    Returns:
        DataFrame with wavelength (nm) and irradiance (W/m²/nm) columns
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'AM15G_reference.csv')

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Ensure proper column names
        if 'wavelength' not in df.columns:
            df.columns = ['wavelength', 'irradiance']
        return df
    else:
        # Generate synthetic reference if file not found
        return generate_am15g_reference()


def generate_am15g_reference() -> pd.DataFrame:
    """
    Generate AM1.5G reference spectrum based on IEC 60904-3.
    This is an approximation - for accurate work, use official data.

    Returns:
        DataFrame with wavelength and irradiance columns
    """
    wavelengths = np.arange(300, 1201, 1)

    # Simplified AM1.5G approximation using blackbody + atmospheric absorption
    # This follows the general shape of the solar spectrum
    T_sun = 5778  # Sun temperature in K
    h = 6.626e-34  # Planck constant
    c = 3e8  # Speed of light
    k = 1.381e-23  # Boltzmann constant

    # Wavelength in meters
    wl_m = wavelengths * 1e-9

    # Planck blackbody radiation (normalized)
    spectral = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T_sun)) - 1)
    spectral = spectral / np.max(spectral)

    # Apply atmospheric absorption approximation
    # O3 absorption around 300-350nm
    ozone_abs = np.exp(-0.5 * np.exp(-((wavelengths - 310) / 20)**2))

    # Water vapor absorption at various wavelengths
    water_abs = 1 - 0.3 * np.exp(-((wavelengths - 940) / 30)**2)
    water_abs *= 1 - 0.2 * np.exp(-((wavelengths - 1130) / 40)**2)

    # Oxygen absorption
    o2_abs = 1 - 0.15 * np.exp(-((wavelengths - 760) / 10)**2)

    # Apply all absorptions
    spectral = spectral * ozone_abs * water_abs * o2_abs

    # Scale to give approximately 1000 W/m² total irradiance
    # Typical AM1.5G has about 1.1-1.3 W/m²/nm peak
    spectral = spectral * 1.4

    df = pd.DataFrame({
        'wavelength': wavelengths,
        'irradiance': spectral
    })

    return df


def get_band_irradiance(wavelength: np.ndarray, irradiance: np.ndarray,
                        band_start: int, band_end: int) -> float:
    """
    Calculate integrated irradiance for a wavelength band.

    Args:
        wavelength: Array of wavelengths (nm)
        irradiance: Array of spectral irradiance (W/m²/nm)
        band_start: Start wavelength of band (nm)
        band_end: End wavelength of band (nm)

    Returns:
        Integrated irradiance (W/m²) for the band
    """
    mask = (wavelength >= band_start) & (wavelength <= band_end)

    if not np.any(mask):
        return 0.0

    wl_band = wavelength[mask]
    ir_band = irradiance[mask]

    # Integrate using trapezoidal rule
    return float(np.trapz(ir_band, wl_band))


# =============================================================================
# SPECTRAL MATCH CALCULATOR
# =============================================================================

class SpectralCalculator:
    """
    Calculator for spectral match classification per IEC 60904-9:2020.
    """

    def __init__(self, reference_spectrum: Optional[pd.DataFrame] = None):
        """
        Initialize the spectral calculator.

        Args:
            reference_spectrum: AM1.5G reference spectrum DataFrame
        """
        if reference_spectrum is None:
            self.reference = load_am15g_reference()
        else:
            self.reference = reference_spectrum

        # Pre-calculate reference band irradiances
        self.reference_bands = self._calculate_band_irradiances(
            self.reference['wavelength'].values,
            self.reference['irradiance'].values
        )

    def _calculate_band_irradiances(self, wavelength: np.ndarray,
                                     irradiance: np.ndarray) -> Dict[int, float]:
        """Calculate irradiance for each spectral band."""
        band_irradiances = {}

        for i, (start, end, name) in enumerate(WAVELENGTH_BANDS, 1):
            band_irradiances[i] = get_band_irradiance(
                wavelength, irradiance, start, end
            )

        return band_irradiances

    def calculate_spectral_match(self, wavelength: np.ndarray,
                                  irradiance: np.ndarray) -> SpectralMatchResult:
        """
        Calculate spectral match classification.

        Args:
            wavelength: Measured wavelength array (nm)
            irradiance: Measured spectral irradiance (W/m²/nm)

        Returns:
            SpectralMatchResult with all classification details
        """
        # Calculate measured band irradiances
        measured_bands = self._calculate_band_irradiances(wavelength, irradiance)

        # Calculate total irradiances
        total_measured = sum(measured_bands.values())
        total_reference = sum(self.reference_bands.values())

        # Calculate band ratios (normalized)
        band_ratios = {}
        band_classifications = {}
        band_details = []

        for band_num in measured_bands.keys():
            # Fraction of total for measured and reference
            measured_fraction = measured_bands[band_num] / total_measured if total_measured > 0 else 0
            reference_fraction = self.reference_bands[band_num] / total_reference if total_reference > 0 else 0

            # Ratio of fractions
            if reference_fraction > 0:
                ratio = measured_fraction / reference_fraction
            else:
                ratio = 1.0

            band_ratios[band_num] = ratio
            band_classifications[band_num] = get_classification(ratio, 'spectral')

            # Get band info
            band_info = WAVELENGTH_BANDS[band_num - 1]
            band_details.append({
                'band': band_num,
                'range': f"{band_info[0]}-{band_info[1]} nm",
                'name': band_info[2],
                'measured_irradiance': measured_bands[band_num],
                'reference_irradiance': self.reference_bands[band_num],
                'ratio': ratio,
                'deviation_percent': (ratio - 1.0) * 100,
                'classification': band_classifications[band_num]
            })

        # Calculate maximum deviation
        max_deviation = max(abs(r - 1.0) for r in band_ratios.values()) * 100

        # Overall classification (worst of all bands)
        overall_class = get_overall_classification(*band_classifications.values())

        # Calculate SPC and SPD
        spc = self.calculate_spc(wavelength, irradiance)
        spd = self.calculate_spd(wavelength, irradiance)

        return SpectralMatchResult(
            band_ratios=band_ratios,
            band_classifications=band_classifications,
            max_deviation=max_deviation,
            overall_classification=overall_class,
            spc=spc,
            spd=spd,
            band_details=band_details
        )

    def calculate_spc(self, wavelength: np.ndarray,
                      irradiance: np.ndarray,
                      spectral_response: Optional[np.ndarray] = None) -> float:
        """
        Calculate Spectral Performance Category (SPC) per IEC 60904-9.

        SPC quantifies how well the simulator spectrum matches AM1.5G
        when considering a typical c-Si spectral response.

        Args:
            wavelength: Measured wavelength array (nm)
            irradiance: Measured spectral irradiance (W/m²/nm)
            spectral_response: Device spectral response (optional)

        Returns:
            SPC value (closer to 1.0 is better)
        """
        # Use typical c-Si spectral response if not provided
        if spectral_response is None:
            spectral_response = self._get_csi_spectral_response(wavelength)

        # Interpolate reference to measured wavelengths
        ref_interp = np.interp(wavelength,
                                self.reference['wavelength'].values,
                                self.reference['irradiance'].values)

        # Calculate weighted integrals
        measured_weighted = np.trapz(irradiance * spectral_response, wavelength)
        reference_weighted = np.trapz(ref_interp * spectral_response, wavelength)

        measured_total = np.trapz(irradiance, wavelength)
        reference_total = np.trapz(ref_interp, wavelength)

        # SPC ratio
        if reference_weighted > 0 and measured_total > 0:
            spc = (measured_weighted / measured_total) / (reference_weighted / reference_total)
        else:
            spc = 1.0

        return spc

    def calculate_spd(self, wavelength: np.ndarray,
                      irradiance: np.ndarray) -> float:
        """
        Calculate Spectral Performance Deviation (SPD).

        SPD is the RMS deviation of band ratios from unity.

        Args:
            wavelength: Measured wavelength array (nm)
            irradiance: Measured spectral irradiance (W/m²/nm)

        Returns:
            SPD value in percent
        """
        measured_bands = self._calculate_band_irradiances(wavelength, irradiance)

        total_measured = sum(measured_bands.values())
        total_reference = sum(self.reference_bands.values())

        deviations = []
        for band_num in measured_bands.keys():
            measured_fraction = measured_bands[band_num] / total_measured if total_measured > 0 else 0
            reference_fraction = self.reference_bands[band_num] / total_reference if total_reference > 0 else 0

            if reference_fraction > 0:
                ratio = measured_fraction / reference_fraction
                deviations.append((ratio - 1.0) ** 2)

        if deviations:
            spd = np.sqrt(np.mean(deviations)) * 100
        else:
            spd = 0.0

        return spd

    def _get_csi_spectral_response(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Get typical crystalline silicon spectral response.

        Args:
            wavelength: Wavelength array (nm)

        Returns:
            Spectral response array (A/W equivalent)
        """
        # Typical c-Si response curve (simplified)
        sr = np.zeros_like(wavelength, dtype=float)

        # Response starts around 300nm, peaks around 900-950nm, drops by 1200nm
        mask = (wavelength >= 300) & (wavelength <= 1200)
        wl = wavelength[mask]

        # Approximate response shape
        response = np.zeros_like(wl, dtype=float)

        # Rising edge (300-500nm)
        mask1 = wl <= 500
        response[mask1] = 0.3 + 0.4 * (wl[mask1] - 300) / 200

        # Plateau (500-900nm)
        mask2 = (wl > 500) & (wl <= 900)
        response[mask2] = 0.7 + 0.2 * (wl[mask2] - 500) / 400

        # Falling edge (900-1200nm)
        mask3 = wl > 900
        response[mask3] = 0.9 * (1 - (wl[mask3] - 900) / 300)
        response[mask3] = np.maximum(response[mask3], 0)

        sr[mask] = response

        return sr


# =============================================================================
# UNIFORMITY CALCULATOR
# =============================================================================

class UniformityCalculator:
    """
    Calculator for spatial uniformity per IEC 60904-9:2020.
    """

    @staticmethod
    def calculate_uniformity(irradiance_grid: np.ndarray,
                             method: str = 'iec') -> UniformityResult:
        """
        Calculate spatial non-uniformity.

        Args:
            irradiance_grid: 2D array of irradiance values (W/m²)
            method: Calculation method ('iec' or 'std')

        Returns:
            UniformityResult with classification and details
        """
        if irradiance_grid.size == 0:
            raise ValueError("Empty irradiance grid")

        # Flatten grid for statistics
        flat = irradiance_grid.flatten()

        # Basic statistics
        mean_irr = np.mean(flat)
        max_irr = np.max(flat)
        min_irr = np.min(flat)
        std_irr = np.std(flat)

        # Calculate non-uniformity per IEC 60904-9
        if method == 'iec':
            # IEC definition: (max - min) / (max + min) * 100%
            if (max_irr + min_irr) > 0:
                non_uniformity = (max_irr - min_irr) / (max_irr + min_irr) * 100
            else:
                non_uniformity = 0.0
        else:
            # Alternative: relative standard deviation
            if mean_irr > 0:
                non_uniformity = (std_irr / mean_irr) * 100
            else:
                non_uniformity = 0.0

        # Classification
        classification = get_classification(non_uniformity, 'uniformity')

        # Normalized data (relative to mean)
        normalized = irradiance_grid / mean_irr if mean_irr > 0 else irradiance_grid

        # Deviation map (% from mean)
        deviation_map = ((irradiance_grid - mean_irr) / mean_irr * 100
                         if mean_irr > 0 else np.zeros_like(irradiance_grid))

        return UniformityResult(
            non_uniformity=non_uniformity,
            classification=classification,
            mean_irradiance=mean_irr,
            max_irradiance=max_irr,
            min_irradiance=min_irr,
            std_deviation=std_irr,
            normalized_data=normalized,
            deviation_map=deviation_map
        )

    @staticmethod
    def create_grid_positions(rows: int, cols: int,
                              width: float, height: float,
                              margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create x, y position arrays for uniformity grid.

        Args:
            rows: Number of rows
            cols: Number of columns
            width: Total width (mm)
            height: Total height (mm)
            margin: Edge margin (mm)

        Returns:
            Tuple of (x_positions, y_positions) meshgrid arrays
        """
        x = np.linspace(margin, width - margin, cols)
        y = np.linspace(margin, height - margin, rows)

        return np.meshgrid(x, y)

    @staticmethod
    def interpolate_grid(irradiance_grid: np.ndarray,
                         target_resolution: int = 100) -> np.ndarray:
        """
        Interpolate uniformity grid to higher resolution for visualization.

        Args:
            irradiance_grid: Original 2D array
            target_resolution: Target grid size

        Returns:
            Interpolated high-resolution grid
        """
        from scipy.ndimage import zoom

        rows, cols = irradiance_grid.shape
        zoom_factors = (target_resolution / rows, target_resolution / cols)

        return zoom(irradiance_grid, zoom_factors, order=3)


# =============================================================================
# TEMPORAL STABILITY CALCULATOR
# =============================================================================

class TemporalCalculator:
    """
    Calculator for temporal stability (STI/LTI) per IEC 60904-9:2020.
    """

    @staticmethod
    def calculate_temporal_stability(time_series: np.ndarray,
                                      irradiance: np.ndarray,
                                      sample_rate: float = 1000.0,
                                      sti_window_ms: float = 1.0,
                                      lti_window_s: float = 60.0) -> TemporalResult:
        """
        Calculate temporal instability (STI and LTI).

        Args:
            time_series: Time array (seconds)
            irradiance: Irradiance measurements (W/m²)
            sample_rate: Sample rate in Hz
            sti_window_ms: Short-term instability window (ms)
            lti_window_s: Long-term instability window (s)

        Returns:
            TemporalResult with STI, LTI, and classifications
        """
        if len(time_series) != len(irradiance):
            raise ValueError("Time and irradiance arrays must have same length")

        if len(irradiance) < 2:
            raise ValueError("Need at least 2 data points")

        # Calculate mean irradiance
        mean_irr = np.mean(irradiance)

        # Short-Term Instability (STI)
        # Maximum deviation within short windows
        sti_samples = max(1, int(sti_window_ms * sample_rate / 1000))
        sti = TemporalCalculator._calculate_instability(irradiance, sti_samples, mean_irr)

        # Long-Term Instability (LTI)
        # Maximum deviation across the entire measurement period
        lti_samples = max(1, int(lti_window_s * sample_rate))

        if len(irradiance) >= lti_samples:
            lti = TemporalCalculator._calculate_instability(irradiance, lti_samples, mean_irr)
        else:
            # Use entire measurement if shorter than LTI window
            lti = TemporalCalculator._calculate_instability(irradiance, len(irradiance), mean_irr)

        # Classifications
        sti_class = get_classification(sti, 'sti')
        lti_class = get_classification(lti, 'lti')
        overall_class = get_overall_classification(sti_class, lti_class)

        # Generate pulse profile (normalized)
        pulse_profile = irradiance / mean_irr if mean_irr > 0 else irradiance

        return TemporalResult(
            sti=sti,
            lti=lti,
            sti_classification=sti_class,
            lti_classification=lti_class,
            overall_classification=overall_class,
            pulse_profile=pulse_profile,
            time_series=time_series
        )

    @staticmethod
    def _calculate_instability(irradiance: np.ndarray,
                                window_size: int,
                                mean_value: float) -> float:
        """
        Calculate instability for a given window size.

        Args:
            irradiance: Irradiance array
            window_size: Window size in samples
            mean_value: Mean irradiance for normalization

        Returns:
            Instability percentage
        """
        if mean_value == 0:
            return 0.0

        max_deviation = 0.0
        n = len(irradiance)

        for i in range(0, n - window_size + 1, max(1, window_size // 2)):
            window = irradiance[i:i + window_size]
            window_max = np.max(window)
            window_min = np.min(window)

            # IEC definition: (max - min) / (max + min) * 100%
            if (window_max + window_min) > 0:
                deviation = (window_max - window_min) / (window_max + window_min) * 100
            else:
                deviation = 0.0

            max_deviation = max(max_deviation, deviation)

        return max_deviation

    @staticmethod
    def analyze_pulse_shape(time_series: np.ndarray,
                            irradiance: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
        """
        Analyze pulse shape characteristics.

        Args:
            time_series: Time array (seconds)
            irradiance: Irradiance array (W/m²)
            threshold: Threshold for pulse detection (fraction of max)

        Returns:
            Dictionary with pulse characteristics
        """
        max_irr = np.max(irradiance)
        threshold_value = max_irr * threshold

        # Find pulse start and end
        above_threshold = irradiance > threshold_value
        if not np.any(above_threshold):
            return {'pulse_width': 0, 'rise_time': 0, 'fall_time': 0}

        indices = np.where(above_threshold)[0]
        start_idx = indices[0]
        end_idx = indices[-1]

        # Pulse width
        pulse_width = time_series[end_idx] - time_series[start_idx]

        # Rise time (10% to 90%)
        rise_10 = 0.1 * max_irr
        rise_90 = 0.9 * max_irr

        rise_start = np.argmax(irradiance >= rise_10)
        rise_end = np.argmax(irradiance >= rise_90)
        rise_time = time_series[rise_end] - time_series[rise_start] if rise_end > rise_start else 0

        # Fall time (90% to 10%)
        peak_idx = np.argmax(irradiance)
        fall_irr = irradiance[peak_idx:]
        fall_time_arr = time_series[peak_idx:]

        fall_90_idx = np.argmax(fall_irr <= rise_90) if np.any(fall_irr <= rise_90) else 0
        fall_10_idx = np.argmax(fall_irr <= rise_10) if np.any(fall_irr <= rise_10) else 0
        fall_time = fall_time_arr[fall_10_idx] - fall_time_arr[fall_90_idx] if fall_10_idx > fall_90_idx else 0

        return {
            'pulse_width_ms': pulse_width * 1000,
            'rise_time_ms': rise_time * 1000,
            'fall_time_ms': fall_time * 1000,
            'peak_irradiance': max_irr,
            'mean_irradiance': np.mean(irradiance[above_threshold])
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_spc(wavelength: np.ndarray, irradiance: np.ndarray,
                  reference: Optional[pd.DataFrame] = None) -> float:
    """
    Convenience function to calculate SPC.

    Args:
        wavelength: Wavelength array (nm)
        irradiance: Irradiance array (W/m²/nm)
        reference: Reference spectrum (optional)

    Returns:
        SPC value
    """
    calc = SpectralCalculator(reference)
    return calc.calculate_spc(wavelength, irradiance)


def calculate_spd(wavelength: np.ndarray, irradiance: np.ndarray,
                  reference: Optional[pd.DataFrame] = None) -> float:
    """
    Convenience function to calculate SPD.

    Args:
        wavelength: Wavelength array (nm)
        irradiance: Irradiance array (W/m²/nm)
        reference: Reference spectrum (optional)

    Returns:
        SPD value in percent
    """
    calc = SpectralCalculator(reference)
    return calc.calculate_spd(wavelength, irradiance)


def classify_simulator(spectral_class: str, uniformity_class: str,
                       sti_class: str, lti_class: str) -> str:
    """
    Determine overall simulator classification.

    The overall classification is the worst of:
    - Spectral match
    - Spatial uniformity
    - Temporal stability (STI and LTI combined)

    Args:
        spectral_class: Spectral match classification
        uniformity_class: Uniformity classification
        sti_class: Short-term instability classification
        lti_class: Long-term instability classification

    Returns:
        Overall classification (A+, A, B, or C)
    """
    # Temporal class is the worst of STI and LTI
    temporal_class = get_overall_classification(sti_class, lti_class)

    # Overall is worst of all three main categories
    return get_overall_classification(spectral_class, uniformity_class, temporal_class)


def format_classification_string(spectral: str, uniformity: str,
                                  temporal: str) -> str:
    """
    Format classification as standard notation (e.g., 'A+AA').

    Args:
        spectral: Spectral classification
        uniformity: Uniformity classification
        temporal: Temporal classification

    Returns:
        Formatted classification string
    """
    return f"{spectral}{uniformity}{temporal}".replace('+', '+')


def generate_sample_spectral_data(noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample spectral data for testing/demonstration.

    Args:
        noise_level: Relative noise level (0-1)

    Returns:
        Tuple of (wavelength, irradiance) arrays
    """
    reference = load_am15g_reference()
    wavelength = reference['wavelength'].values
    irradiance = reference['irradiance'].values

    # Add random noise
    noise = 1 + (np.random.random(len(irradiance)) - 0.5) * 2 * noise_level
    measured = irradiance * noise

    return wavelength, measured


def generate_sample_uniformity_data(rows: int = 5, cols: int = 5,
                                     non_uniformity: float = 2.0) -> np.ndarray:
    """
    Generate sample uniformity grid for testing/demonstration.

    Args:
        rows: Number of grid rows
        cols: Number of grid columns
        non_uniformity: Target non-uniformity percentage

    Returns:
        2D array of irradiance values
    """
    # Create base pattern with slight gradient
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)

    # Radial falloff pattern (typical for reflector-based simulators)
    R = np.sqrt(X**2 + Y**2)
    pattern = 1 - 0.1 * R**2

    # Add random variation
    noise = 1 + (np.random.random((rows, cols)) - 0.5) * non_uniformity / 50

    # Scale to target irradiance
    grid = 1000 * pattern * noise

    return grid


def generate_sample_temporal_data(duration_s: float = 0.01,
                                   sample_rate: float = 100000,
                                   instability: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample temporal data (flash profile) for testing/demonstration.

    Args:
        duration_s: Total duration in seconds
        sample_rate: Sample rate in Hz
        instability: Target instability percentage

    Returns:
        Tuple of (time, irradiance) arrays
    """
    n_samples = int(duration_s * sample_rate)
    time = np.linspace(0, duration_s, n_samples)

    # Generate flash profile (typical xenon flash shape)
    t_rise = duration_s * 0.1
    t_flat = duration_s * 0.6
    t_fall = duration_s * 0.3

    irradiance = np.zeros(n_samples)

    # Rise phase
    rise_mask = time <= t_rise
    irradiance[rise_mask] = 1000 * (time[rise_mask] / t_rise) ** 0.5

    # Flat phase
    flat_mask = (time > t_rise) & (time <= t_rise + t_flat)
    irradiance[flat_mask] = 1000

    # Fall phase
    fall_mask = time > t_rise + t_flat
    t_fall_rel = time[fall_mask] - (t_rise + t_flat)
    irradiance[fall_mask] = 1000 * np.exp(-3 * t_fall_rel / t_fall)

    # Add instability noise
    noise = 1 + (np.random.random(n_samples) - 0.5) * instability / 50
    irradiance = irradiance * noise

    return time, irradiance
