"""
Spectrum drift analysis utilities for SunSim-IEC60904-Classifier.

Implements drift calculations per TÜV paper findings:
- UV/NIR degradation monitoring (Xenon aging)
- Blue-shift during pulse tracking
- Lamp power adjustment effects
- Multi-manufacturer comparison
- Drift trending over flash count

Reference: TÜV Rheinland publications on solar simulator spectral stability
IEC 60904-9 Ed.3 spectral match requirements
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.interpolate import interp1d


# IEC 60904-9 Ed.3 wavelength bands (nm)
IEC_BANDS = {
    "UV": [(300, 400)],
    "UV_detailed": [(300, 350), (350, 400)],
    "VIS": [(400, 500), (500, 600), (600, 700)],
    "VIS_total": [(400, 700)],
    "NIR": [(700, 800), (800, 900), (900, 1000), (1000, 1100)],
    "NIR_total": [(700, 1100)],
    "full": [(300, 1100)],
}

# IEC 60904-9 Ed.3 spectral match limits (% deviation from AM1.5G)
SPECTRAL_MATCH_LIMITS = {
    "A+": 12.5,
    "A": 25.0,
    "B": 40.0,
    "C": 100.0,  # >40% but classifiable
}

# Target repeatability per IEC 60904-9
REPEATABILITY_TARGET = 0.09  # 0.09%


@dataclass
class DriftResult:
    """Container for drift analysis results."""
    uv_shift_percent: float
    vis_shift_percent: float
    nir_shift_percent: float
    overall_shift_percent: float
    classification: str
    classification_changed: bool
    previous_classification: Optional[str]
    trend_direction: str
    rate_of_change: float  # % per 1000 flashes
    blue_shift_detected: bool
    blue_shift_magnitude_nm: Optional[float]


@dataclass
class RepeatabilityResult:
    """Container for repeatability analysis results."""
    mean: float
    std_dev: float
    cv_percent: float
    min_value: float
    max_value: float
    range_value: float
    repeatability_percent: float
    passes_target: bool
    ucl: float
    lcl: float
    out_of_control: bool
    trend_direction: str


@dataclass
class BlueShiftResult:
    """Container for blue-shift analysis results."""
    detected: bool
    magnitude_nm: float
    timing_ms: float
    confidence: float


class DriftAnalyzer:
    """
    Analyzer for spectrum drift and repeatability metrics.

    Implements TÜV paper findings for Xenon lamp degradation monitoring
    and IEC 60904-9 compliance checking.
    """

    def __init__(
        self,
        reference_wavelengths: Optional[np.ndarray] = None,
        reference_intensities: Optional[np.ndarray] = None,
    ):
        """
        Initialize drift analyzer.

        Args:
            reference_wavelengths: Reference spectrum wavelengths (nm)
            reference_intensities: Reference spectrum intensities
        """
        self.reference_wavelengths = reference_wavelengths
        self.reference_intensities = reference_intensities
        self._reference_interp = None

        if reference_wavelengths is not None and reference_intensities is not None:
            self._setup_reference_interpolation()

    def _setup_reference_interpolation(self):
        """Set up interpolation function for reference spectrum."""
        if self.reference_wavelengths is not None and self.reference_intensities is not None:
            self._reference_interp = interp1d(
                self.reference_wavelengths,
                self.reference_intensities,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )

    def set_reference(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray
    ) -> None:
        """
        Set reference spectrum for drift calculations.

        Args:
            wavelengths: Wavelength array in nm
            intensities: Intensity array
        """
        self.reference_wavelengths = np.array(wavelengths)
        self.reference_intensities = np.array(intensities)
        self._setup_reference_interpolation()

    def calculate_band_integral(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        band_start: float,
        band_end: float
    ) -> float:
        """
        Calculate integrated intensity over a wavelength band.

        Args:
            wavelengths: Wavelength array in nm
            intensities: Intensity array
            band_start: Start wavelength in nm
            band_end: End wavelength in nm

        Returns:
            Integrated intensity over the band
        """
        mask = (wavelengths >= band_start) & (wavelengths <= band_end)
        if not np.any(mask):
            return 0.0

        band_wl = wavelengths[mask]
        band_int = intensities[mask]

        # Trapezoidal integration
        if len(band_wl) < 2:
            return 0.0

        return np.trapz(band_int, band_wl)

    def calculate_spectral_deviation(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        band_start: float,
        band_end: float
    ) -> float:
        """
        Calculate spectral deviation from reference in a band.

        Args:
            wavelengths: Measured wavelength array
            intensities: Measured intensity array
            band_start: Band start wavelength
            band_end: Band end wavelength

        Returns:
            Deviation in percent from reference
        """
        if self._reference_interp is None:
            return 0.0

        # Calculate measured integral
        measured_integral = self.calculate_band_integral(
            wavelengths, intensities, band_start, band_end
        )

        # Calculate reference integral at same wavelengths
        ref_intensities = self._reference_interp(wavelengths)
        ref_integral = self.calculate_band_integral(
            wavelengths, ref_intensities, band_start, band_end
        )

        if ref_integral == 0:
            return 0.0

        # Deviation as percentage
        deviation = ((measured_integral - ref_integral) / ref_integral) * 100
        return deviation

    def analyze_drift(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        previous_classification: Optional[str] = None,
        flash_count: int = 0,
        previous_flash_count: int = 0,
        previous_shift: float = 0.0
    ) -> DriftResult:
        """
        Analyze spectrum drift from reference.

        Args:
            wavelengths: Current measured wavelengths
            intensities: Current measured intensities
            previous_classification: Previous classification if known
            flash_count: Current flash count
            previous_flash_count: Flash count at previous measurement
            previous_shift: Previous overall shift value

        Returns:
            DriftResult with comprehensive analysis
        """
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)

        # Calculate UV shift (300-400nm) - critical for Xenon aging
        uv_shift = self._calculate_region_shift(wavelengths, intensities, 300, 400)

        # Calculate visible shift (400-700nm)
        vis_shift = self._calculate_region_shift(wavelengths, intensities, 400, 700)

        # Calculate NIR shift (700-1100nm)
        nir_shift = self._calculate_region_shift(wavelengths, intensities, 700, 1100)

        # Overall weighted shift
        overall_shift = self._calculate_overall_shift(uv_shift, vis_shift, nir_shift)

        # Determine classification
        classification = self._determine_classification(overall_shift)
        classification_changed = (
            previous_classification is not None and
            classification != previous_classification
        )

        # Calculate trend
        trend_direction, rate_of_change = self._calculate_trend(
            overall_shift, previous_shift, flash_count, previous_flash_count
        )

        # Check for blue-shift (TÜV paper finding)
        blue_shift = self._detect_blue_shift(wavelengths, intensities)

        return DriftResult(
            uv_shift_percent=uv_shift,
            vis_shift_percent=vis_shift,
            nir_shift_percent=nir_shift,
            overall_shift_percent=overall_shift,
            classification=classification,
            classification_changed=classification_changed,
            previous_classification=previous_classification,
            trend_direction=trend_direction,
            rate_of_change=rate_of_change,
            blue_shift_detected=blue_shift.detected,
            blue_shift_magnitude_nm=blue_shift.magnitude_nm if blue_shift.detected else None
        )

    def _calculate_region_shift(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        start_nm: float,
        end_nm: float
    ) -> float:
        """Calculate shift for a spectral region."""
        if self._reference_interp is None:
            return 0.0

        mask = (wavelengths >= start_nm) & (wavelengths <= end_nm)
        if not np.any(mask):
            return 0.0

        region_wl = wavelengths[mask]
        region_int = intensities[mask]

        # Get reference values at same wavelengths
        ref_int = self._reference_interp(region_wl)

        # Calculate RMS deviation
        diff = region_int - ref_int
        ref_sum = np.sum(ref_int)

        if ref_sum == 0:
            return 0.0

        # Percentage shift
        shift = (np.sum(np.abs(diff)) / ref_sum) * 100
        return shift

    def _calculate_overall_shift(
        self,
        uv_shift: float,
        vis_shift: float,
        nir_shift: float
    ) -> float:
        """
        Calculate overall spectral shift with IEC weighting.

        UV gets higher weight due to Xenon degradation sensitivity.
        """
        # Weighted average - UV weighted more heavily
        weights = {
            'uv': 0.35,   # Higher weight due to Xenon sensitivity
            'vis': 0.40,  # Most important for cell performance
            'nir': 0.25,  # Important for multi-junction cells
        }

        overall = (
            weights['uv'] * uv_shift +
            weights['vis'] * vis_shift +
            weights['nir'] * nir_shift
        )

        return overall

    def _determine_classification(self, overall_shift: float) -> str:
        """Determine IEC classification based on overall shift."""
        abs_shift = abs(overall_shift)

        if abs_shift <= SPECTRAL_MATCH_LIMITS["A+"]:
            return "A+"
        elif abs_shift <= SPECTRAL_MATCH_LIMITS["A"]:
            return "A"
        elif abs_shift <= SPECTRAL_MATCH_LIMITS["B"]:
            return "B"
        else:
            return "C"

    def _calculate_trend(
        self,
        current_shift: float,
        previous_shift: float,
        current_flash: int,
        previous_flash: int
    ) -> Tuple[str, float]:
        """Calculate drift trend direction and rate."""
        flash_diff = current_flash - previous_flash

        if flash_diff <= 0:
            return "stable", 0.0

        shift_diff = abs(current_shift) - abs(previous_shift)

        # Rate per 1000 flashes
        rate = (shift_diff / flash_diff) * 1000

        if abs(rate) < 0.1:  # Less than 0.1% per 1000 flashes
            direction = "stable"
        elif rate > 0:
            direction = "degrading"
        else:
            direction = "improving"

        return direction, rate

    def _detect_blue_shift(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray
    ) -> BlueShiftResult:
        """
        Detect blue-shift phenomena per TÜV paper.

        Blue-shift is characterized by peak wavelength shifting
        toward shorter wavelengths, common in Xenon aging.
        """
        if self._reference_interp is None:
            return BlueShiftResult(False, 0.0, 0.0, 0.0)

        # Find peak in visible region
        vis_mask = (wavelengths >= 400) & (wavelengths <= 700)
        if not np.any(vis_mask):
            return BlueShiftResult(False, 0.0, 0.0, 0.0)

        vis_wl = wavelengths[vis_mask]
        vis_int = intensities[vis_mask]
        ref_int = self._reference_interp(vis_wl)

        # Find peaks
        if len(vis_int) < 3:
            return BlueShiftResult(False, 0.0, 0.0, 0.0)

        current_peak_idx = np.argmax(vis_int)
        ref_peak_idx = np.argmax(ref_int)

        current_peak_wl = vis_wl[current_peak_idx]
        ref_peak_wl = vis_wl[ref_peak_idx]

        shift_nm = current_peak_wl - ref_peak_wl

        # Blue shift is negative (toward shorter wavelengths)
        detected = shift_nm < -1.0  # More than 1nm shift

        return BlueShiftResult(
            detected=detected,
            magnitude_nm=abs(shift_nm) if detected else 0.0,
            timing_ms=0.0,  # Would need temporal data
            confidence=0.8 if detected else 0.0
        )

    def analyze_repeatability(
        self,
        measurements: List[float],
        target_percent: float = REPEATABILITY_TARGET,
        previous_mean: Optional[float] = None
    ) -> RepeatabilityResult:
        """
        Analyze flash-to-flash repeatability.

        Args:
            measurements: List of irradiance measurements
            target_percent: Target repeatability (default 0.09%)
            previous_mean: Previous mean for trend analysis

        Returns:
            RepeatabilityResult with statistical metrics
        """
        if not measurements or len(measurements) < 2:
            return RepeatabilityResult(
                mean=measurements[0] if measurements else 0.0,
                std_dev=0.0,
                cv_percent=0.0,
                min_value=measurements[0] if measurements else 0.0,
                max_value=measurements[0] if measurements else 0.0,
                range_value=0.0,
                repeatability_percent=0.0,
                passes_target=True,
                ucl=0.0,
                lcl=0.0,
                out_of_control=False,
                trend_direction="stable"
            )

        data = np.array(measurements)

        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Sample std dev
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val

        # Coefficient of variation (CV%)
        cv_percent = (std_dev / mean * 100) if mean != 0 else 0.0

        # Repeatability is typically CV or range-based
        # Using CV as primary metric
        repeatability_percent = cv_percent

        # Control limits (3-sigma)
        ucl = mean + 3 * std_dev
        lcl = mean - 3 * std_dev

        # Check for out-of-control points
        out_of_control = np.any((data > ucl) | (data < lcl))

        # Pass/fail vs target
        passes_target = repeatability_percent <= target_percent

        # Trend direction
        if previous_mean is not None and previous_mean != 0:
            pct_change = ((mean - previous_mean) / previous_mean) * 100
            if abs(pct_change) < 0.1:
                trend_direction = "stable"
            elif pct_change > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        return RepeatabilityResult(
            mean=mean,
            std_dev=std_dev,
            cv_percent=cv_percent,
            min_value=min_val,
            max_value=max_val,
            range_value=range_val,
            repeatability_percent=repeatability_percent,
            passes_target=passes_target,
            ucl=ucl,
            lcl=lcl,
            out_of_control=out_of_control,
            trend_direction=trend_direction
        )

    def calculate_control_limits(
        self,
        historical_means: List[float],
        historical_ranges: List[float],
        n_subgroup: int = 5
    ) -> Dict[str, float]:
        """
        Calculate control limits for X-bar and R charts.

        Args:
            historical_means: List of subgroup means
            historical_ranges: List of subgroup ranges
            n_subgroup: Subgroup size

        Returns:
            Dictionary with UCL, LCL, and centerlines for X-bar and R
        """
        # Control chart constants (for subgroup sizes 2-10)
        A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577,
              6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114,
              6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        n = min(max(n_subgroup, 2), 10)  # Clamp to valid range

        x_bar = np.mean(historical_means) if historical_means else 0
        r_bar = np.mean(historical_ranges) if historical_ranges else 0

        return {
            "xbar_ucl": x_bar + A2[n] * r_bar,
            "xbar_lcl": x_bar - A2[n] * r_bar,
            "xbar_centerline": x_bar,
            "r_ucl": D4[n] * r_bar,
            "r_lcl": D3[n] * r_bar,
            "r_centerline": r_bar,
        }

    def analyze_uv_degradation(
        self,
        flash_counts: List[int],
        uv_shifts: List[float]
    ) -> Dict[str, float]:
        """
        Analyze UV degradation trend (Xenon-specific).

        Fits a curve to UV shift vs flash count to predict
        future degradation.

        Args:
            flash_counts: List of flash counts
            uv_shifts: List of UV shift percentages

        Returns:
            Dictionary with trend parameters and predictions
        """
        if len(flash_counts) < 3:
            return {
                "slope": 0.0,
                "intercept": 0.0,
                "r_squared": 0.0,
                "predicted_at_50k": 0.0,
                "predicted_at_100k": 0.0,
                "flashes_to_limit": 0,
            }

        x = np.array(flash_counts)
        y = np.array(uv_shifts)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Predictions
        pred_50k = slope * 50000 + intercept
        pred_100k = slope * 100000 + intercept

        # Flashes until hitting Class B limit (40%)
        if slope > 0:
            flashes_to_limit = int((40.0 - intercept) / slope)
        else:
            flashes_to_limit = -1  # Not degrading

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "predicted_at_50k": pred_50k,
            "predicted_at_100k": pred_100k,
            "flashes_to_limit": max(0, flashes_to_limit),
        }

    def compare_manufacturers(
        self,
        manufacturer_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare drift characteristics across manufacturers.

        Args:
            manufacturer_data: Dict of manufacturer -> {flash_counts, uv_shifts, nir_shifts}

        Returns:
            Comparison metrics for each manufacturer
        """
        results = {}

        for manufacturer, data in manufacturer_data.items():
            flash_counts = data.get("flash_counts", [])
            uv_shifts = data.get("uv_shifts", [])
            nir_shifts = data.get("nir_shifts", [])

            uv_analysis = self.analyze_uv_degradation(flash_counts, uv_shifts)

            # NIR stability
            nir_std = np.std(nir_shifts) if nir_shifts else 0.0

            results[manufacturer] = {
                "uv_degradation_rate": uv_analysis["slope"] * 1000,  # per 1000 flashes
                "uv_r_squared": uv_analysis["r_squared"],
                "nir_stability": nir_std,
                "predicted_life_flashes": uv_analysis["flashes_to_limit"],
                "avg_uv_shift": np.mean(uv_shifts) if uv_shifts else 0.0,
                "avg_nir_shift": np.mean(nir_shifts) if nir_shifts else 0.0,
            }

        return results

    def analyze_power_effect(
        self,
        power_levels: List[float],
        spectral_shifts: List[float]
    ) -> Dict[str, float]:
        """
        Analyze effect of power adjustment on spectral shift.

        Args:
            power_levels: Power settings as percentage
            spectral_shifts: Corresponding spectral shifts

        Returns:
            Analysis of power-shift relationship
        """
        if len(power_levels) < 3:
            return {
                "correlation": 0.0,
                "sensitivity": 0.0,  # % shift per % power change
                "r_squared": 0.0,
            }

        x = np.array(power_levels)
        y = np.array(spectral_shifts)

        # Correlation
        correlation = np.corrcoef(x, y)[0, 1]

        # Linear fit for sensitivity
        slope, intercept, r_value, _, _ = stats.linregress(x, y)

        return {
            "correlation": correlation,
            "sensitivity": slope,  # % shift per % power
            "r_squared": r_value ** 2,
        }


def generate_am15g_reference(
    wavelength_start: float = 300,
    wavelength_end: float = 1100,
    resolution: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simplified AM1.5G reference spectrum.

    This is a simplified approximation for demonstration.
    Real implementation should use ASTM G173-03 data.

    Args:
        wavelength_start: Start wavelength in nm
        wavelength_end: End wavelength in nm
        resolution: Wavelength resolution in nm

    Returns:
        Tuple of (wavelengths, intensities)
    """
    wavelengths = np.arange(wavelength_start, wavelength_end + resolution, resolution)

    # Simplified black-body-like distribution with atmospheric absorption
    # This is a rough approximation - use real ASTM data in production

    # Base black-body at 5778K (sun temperature)
    T = 5778  # K
    h = 6.626e-34  # Planck
    c = 3e8  # Speed of light
    k = 1.38e-23  # Boltzmann

    wl_m = wavelengths * 1e-9  # Convert to meters

    # Planck's law (simplified)
    intensities = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T)) - 1)

    # Normalize to typical solar irradiance levels
    intensities = intensities / np.max(intensities)

    # Apply simplified atmospheric absorption
    # UV reduction
    uv_mask = wavelengths < 400
    intensities[uv_mask] *= 0.3 + 0.7 * (wavelengths[uv_mask] - 300) / 100

    # Water vapor absorption in NIR
    for abs_center, abs_width, abs_depth in [(940, 50, 0.3), (1130, 40, 0.2)]:
        if abs_center <= wavelength_end:
            abs_mask = np.abs(wavelengths - abs_center) < abs_width
            intensities[abs_mask] *= (1 - abs_depth)

    return wavelengths, intensities
