"""
Database Models for IEC 60904-9 Ed.3 Sun Simulator Classification System

This module defines the data structures for solar simulator classification
according to IEC 60904-9:2020 (Edition 3) standard requirements.

Classification Grades: A+, A, B, C
Three Classification Parameters:
1. Spectral Match (SPD) - Ratio of simulator to reference spectrum
2. Spatial Non-Uniformity - Irradiance distribution uniformity
3. Temporal Instability (STI/LTI) - Short/Long term stability
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class ClassificationGrade(Enum):
    """IEC 60904-9 Ed.3 Classification Grades"""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    FAIL = "Fail"


# IEC 60904-9 Ed.3 Classification Thresholds
CLASSIFICATION_THRESHOLDS = {
    # Spectral Match: (min_ratio, max_ratio)
    "spectral_match": {
        ClassificationGrade.A_PLUS: (0.875, 1.125),  # +/- 12.5%
        ClassificationGrade.A: (0.75, 1.25),          # +/- 25%
        ClassificationGrade.B: (0.6, 1.4),            # +/- 40%
        ClassificationGrade.C: (0.4, 2.0),            # +/- 60%/100%
    },
    # Spatial Non-Uniformity: max percentage
    "uniformity": {
        ClassificationGrade.A_PLUS: 1.0,   # <= 1%
        ClassificationGrade.A: 2.0,         # <= 2%
        ClassificationGrade.B: 5.0,         # <= 5%
        ClassificationGrade.C: 10.0,        # <= 10%
    },
    # Temporal Instability (STI): max percentage
    "temporal_sti": {
        ClassificationGrade.A_PLUS: 0.5,   # <= 0.5%
        ClassificationGrade.A: 2.0,         # <= 2%
        ClassificationGrade.B: 5.0,         # <= 5%
        ClassificationGrade.C: 10.0,        # <= 10%
    },
    # Temporal Instability (LTI): max percentage
    "temporal_lti": {
        ClassificationGrade.A_PLUS: 1.0,   # <= 1%
        ClassificationGrade.A: 2.0,         # <= 2%
        ClassificationGrade.B: 5.0,         # <= 5%
        ClassificationGrade.C: 10.0,        # <= 10%
    },
}

# IEC 60904-9 Ed.3 Spectral Wavelength Intervals (100 intervals from 300-1200nm)
# Standard AM1.5G reference spectrum intervals
WAVELENGTH_INTERVALS_ED3 = [
    # (start_nm, end_nm, AM1.5G_fraction_percentage)
    (300, 309, 0.14),
    (309, 318, 0.28),
    (318, 327, 0.43),
    (327, 336, 0.55),
    (336, 345, 0.63),
    (345, 354, 0.75),
    (354, 363, 0.88),
    (363, 372, 1.02),
    (372, 381, 1.18),
    (381, 390, 1.32),
    (390, 399, 1.43),
    (399, 408, 1.52),
    (408, 417, 1.58),
    (417, 426, 1.62),
    (426, 435, 1.65),
    (435, 444, 1.66),
    (444, 453, 1.68),
    (453, 462, 1.68),
    (462, 471, 1.68),
    (471, 480, 1.67),
    (480, 489, 1.66),
    (489, 498, 1.64),
    (498, 507, 1.62),
    (507, 516, 1.60),
    (516, 525, 1.57),
    (525, 534, 1.55),
    (534, 543, 1.52),
    (543, 552, 1.50),
    (552, 561, 1.47),
    (561, 570, 1.44),
    (570, 579, 1.41),
    (579, 588, 1.38),
    (588, 597, 1.35),
    (597, 606, 1.32),
    (606, 615, 1.29),
    (615, 624, 1.26),
    (624, 633, 1.23),
    (633, 642, 1.20),
    (642, 651, 1.17),
    (651, 660, 1.14),
    (660, 669, 1.11),
    (669, 678, 1.08),
    (678, 687, 1.05),
    (687, 696, 1.02),
    (696, 705, 0.99),
    (705, 714, 0.97),
    (714, 723, 0.94),
    (723, 732, 0.91),
    (732, 741, 0.89),
    (741, 750, 0.86),
    (750, 759, 0.84),
    (759, 768, 0.81),
    (768, 777, 0.79),
    (777, 786, 0.77),
    (786, 795, 0.74),
    (795, 804, 0.72),
    (804, 813, 0.70),
    (813, 822, 0.68),
    (822, 831, 0.66),
    (831, 840, 0.64),
    (840, 849, 0.62),
    (849, 858, 0.60),
    (858, 867, 0.58),
    (867, 876, 0.56),
    (876, 885, 0.55),
    (885, 894, 0.53),
    (894, 903, 0.51),
    (903, 912, 0.50),
    (912, 921, 0.48),
    (921, 930, 0.47),
    (930, 939, 0.45),
    (939, 948, 0.44),
    (948, 957, 0.42),
    (957, 966, 0.41),
    (966, 975, 0.40),
    (975, 984, 0.38),
    (984, 993, 0.37),
    (993, 1002, 0.36),
    (1002, 1011, 0.35),
    (1011, 1020, 0.34),
    (1020, 1029, 0.33),
    (1029, 1038, 0.32),
    (1038, 1047, 0.31),
    (1047, 1056, 0.30),
    (1056, 1065, 0.29),
    (1065, 1074, 0.28),
    (1074, 1083, 0.27),
    (1083, 1092, 0.26),
    (1092, 1101, 0.26),
    (1101, 1110, 0.25),
    (1110, 1119, 0.24),
    (1119, 1128, 0.23),
    (1128, 1137, 0.23),
    (1137, 1146, 0.22),
    (1146, 1155, 0.21),
    (1155, 1164, 0.21),
    (1164, 1173, 0.20),
    (1173, 1182, 0.20),
    (1182, 1191, 0.19),
    (1191, 1200, 0.19),
]

# Legacy Ed.2 intervals (400-1100nm) for backward compatibility
WAVELENGTH_INTERVALS_ED2 = [
    (400, 500, 18.4),
    (500, 600, 19.9),
    (600, 700, 18.4),
    (700, 800, 14.9),
    (800, 900, 12.5),
    (900, 1100, 15.9),
]


@dataclass
class SpectralMatchData:
    """Spectral Match (SPD) measurement data for a wavelength interval"""
    interval_start_nm: float
    interval_end_nm: float
    reference_fraction: float  # AM1.5G reference fraction
    measured_fraction: float   # Simulator measured fraction
    ratio: float = 0.0         # measured/reference ratio

    def __post_init__(self):
        if self.reference_fraction > 0:
            self.ratio = self.measured_fraction / self.reference_fraction


@dataclass
class SpectralMatchResult:
    """Complete spectral match classification result"""
    intervals: list[SpectralMatchData] = field(default_factory=list)
    min_ratio: float = 0.0
    max_ratio: float = 0.0
    grade: ClassificationGrade = ClassificationGrade.FAIL
    measurement_date: datetime = field(default_factory=datetime.now)
    wavelength_range: str = "300-1200nm"  # Ed.3 default

    def calculate_grade(self) -> ClassificationGrade:
        """Determine grade based on spectral match ratios"""
        if not self.intervals:
            return ClassificationGrade.FAIL

        ratios = [i.ratio for i in self.intervals]
        self.min_ratio = min(ratios)
        self.max_ratio = max(ratios)

        for grade in [ClassificationGrade.A_PLUS, ClassificationGrade.A,
                      ClassificationGrade.B, ClassificationGrade.C]:
            thresholds = CLASSIFICATION_THRESHOLDS["spectral_match"][grade]
            if self.min_ratio >= thresholds[0] and self.max_ratio <= thresholds[1]:
                self.grade = grade
                return grade

        self.grade = ClassificationGrade.FAIL
        return ClassificationGrade.FAIL


@dataclass
class UniformityMeasurement:
    """Single point irradiance measurement for uniformity analysis"""
    x_position: float  # mm from center
    y_position: float  # mm from center
    irradiance: float  # W/m²


@dataclass
class UniformityResult:
    """Non-uniformity classification result"""
    measurements: list[UniformityMeasurement] = field(default_factory=list)
    min_irradiance: float = 0.0
    max_irradiance: float = 0.0
    mean_irradiance: float = 0.0
    non_uniformity_percent: float = 0.0
    grade: ClassificationGrade = ClassificationGrade.FAIL
    measurement_date: datetime = field(default_factory=datetime.now)
    test_plane_size_mm: tuple[float, float] = (200, 200)
    grid_points: tuple[int, int] = (11, 11)  # 11x11 = 121 points minimum

    def calculate_grade(self) -> ClassificationGrade:
        """
        Calculate non-uniformity grade per IEC 60904-9 Ed.3
        Non-uniformity = ((max - min) / (max + min)) * 100%
        """
        if not self.measurements:
            return ClassificationGrade.FAIL

        irradiances = [m.irradiance for m in self.measurements]
        self.min_irradiance = min(irradiances)
        self.max_irradiance = max(irradiances)
        self.mean_irradiance = np.mean(irradiances)

        if (self.max_irradiance + self.min_irradiance) > 0:
            self.non_uniformity_percent = (
                (self.max_irradiance - self.min_irradiance) /
                (self.max_irradiance + self.min_irradiance)
            ) * 100

        for grade in [ClassificationGrade.A_PLUS, ClassificationGrade.A,
                      ClassificationGrade.B, ClassificationGrade.C]:
            threshold = CLASSIFICATION_THRESHOLDS["uniformity"][grade]
            if self.non_uniformity_percent <= threshold:
                self.grade = grade
                return grade

        self.grade = ClassificationGrade.FAIL
        return ClassificationGrade.FAIL


@dataclass
class TemporalMeasurement:
    """Single temporal stability measurement"""
    timestamp: float  # seconds from start
    irradiance: float  # W/m²


@dataclass
class TemporalStabilityResult:
    """Temporal instability (STI/LTI) classification result"""
    measurements: list[TemporalMeasurement] = field(default_factory=list)
    sti_percent: float = 0.0  # Short-term instability
    lti_percent: float = 0.0  # Long-term instability
    min_irradiance: float = 0.0
    max_irradiance: float = 0.0
    mean_irradiance: float = 0.0
    sti_grade: ClassificationGrade = ClassificationGrade.FAIL
    lti_grade: ClassificationGrade = ClassificationGrade.FAIL
    overall_grade: ClassificationGrade = ClassificationGrade.FAIL
    measurement_date: datetime = field(default_factory=datetime.now)
    measurement_duration_s: float = 0.0
    sampling_rate_hz: float = 1000.0

    def calculate_grade(self) -> ClassificationGrade:
        """
        Calculate temporal instability grades per IEC 60904-9 Ed.3
        STI = ((max - min) / (max + min)) * 100% over short periods
        LTI = similar calculation over longer measurement duration
        """
        if not self.measurements:
            return ClassificationGrade.FAIL

        irradiances = [m.irradiance for m in self.measurements]
        self.min_irradiance = min(irradiances)
        self.max_irradiance = max(irradiances)
        self.mean_irradiance = np.mean(irradiances)

        if (self.max_irradiance + self.min_irradiance) > 0:
            instability = (
                (self.max_irradiance - self.min_irradiance) /
                (self.max_irradiance + self.min_irradiance)
            ) * 100
            self.sti_percent = instability
            self.lti_percent = instability * 1.5  # Simplified LTI calculation

        # Determine STI grade
        for grade in [ClassificationGrade.A_PLUS, ClassificationGrade.A,
                      ClassificationGrade.B, ClassificationGrade.C]:
            threshold = CLASSIFICATION_THRESHOLDS["temporal_sti"][grade]
            if self.sti_percent <= threshold:
                self.sti_grade = grade
                break
        else:
            self.sti_grade = ClassificationGrade.FAIL

        # Determine LTI grade
        for grade in [ClassificationGrade.A_PLUS, ClassificationGrade.A,
                      ClassificationGrade.B, ClassificationGrade.C]:
            threshold = CLASSIFICATION_THRESHOLDS["temporal_lti"][grade]
            if self.lti_percent <= threshold:
                self.lti_grade = grade
                break
        else:
            self.lti_grade = ClassificationGrade.FAIL

        # Overall temporal grade is the worse of STI and LTI
        grade_order = [ClassificationGrade.A_PLUS, ClassificationGrade.A,
                       ClassificationGrade.B, ClassificationGrade.C, ClassificationGrade.FAIL]
        sti_idx = grade_order.index(self.sti_grade)
        lti_idx = grade_order.index(self.lti_grade)
        self.overall_grade = grade_order[max(sti_idx, lti_idx)]

        return self.overall_grade


@dataclass
class OverallClassification:
    """Complete IEC 60904-9 Ed.3 classification result"""
    spectral_match: SpectralMatchResult
    uniformity: UniformityResult
    temporal_stability: TemporalStabilityResult
    overall_classification: str = ""  # e.g., "A+A+A"
    measurement_date: datetime = field(default_factory=datetime.now)
    equipment_id: str = ""
    operator: str = ""
    laboratory: str = ""
    certificate_number: Optional[str] = None

    def calculate_overall(self) -> str:
        """
        Calculate overall classification string (e.g., A+A+A)
        Order: Spectral Match, Uniformity, Temporal Stability
        """
        self.overall_classification = (
            f"{self.spectral_match.grade.value}"
            f"{self.uniformity.grade.value}"
            f"{self.temporal_stability.overall_grade.value}"
        )
        return self.overall_classification


@dataclass
class SimulatorInfo:
    """Sun simulator equipment information"""
    manufacturer: str
    model: str
    serial_number: str
    lamp_type: str  # e.g., "Xenon", "LED", "Metal Halide"
    lamp_hours: float
    illumination_area_mm: tuple[float, float]
    calibration_date: datetime
    next_calibration_date: datetime
    notes: str = ""


def get_grade_color(grade: ClassificationGrade) -> str:
    """Get display color for grade badge"""
    colors = {
        ClassificationGrade.A_PLUS: "#10B981",  # Emerald green
        ClassificationGrade.A: "#22C55E",        # Green
        ClassificationGrade.B: "#F59E0B",        # Amber
        ClassificationGrade.C: "#EF4444",        # Red
        ClassificationGrade.FAIL: "#6B7280",     # Gray
    }
    return colors.get(grade, "#6B7280")


def get_grade_description(grade: ClassificationGrade) -> str:
    """Get description for classification grade"""
    descriptions = {
        ClassificationGrade.A_PLUS: "Highest precision - Calibration laboratory grade",
        ClassificationGrade.A: "High quality - Standard testing grade",
        ClassificationGrade.B: "Moderate quality - General purpose",
        ClassificationGrade.C: "Basic quality - Minimum acceptable",
        ClassificationGrade.FAIL: "Does not meet IEC 60904-9 requirements",
    }
    return descriptions.get(grade, "Unknown grade")
