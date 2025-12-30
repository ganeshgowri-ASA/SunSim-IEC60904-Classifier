"""
Simulator Manufacturer Database Module
IEC 60904-9 Sun Simulator Classification System

This module provides a comprehensive database of solar simulator manufacturers
and their models, with pre-populated data for major manufacturers.

Features:
- SimulatorManufacturer dataclass with full specifications
- Pre-populated database with industry-standard simulators
- Validation for irradiance ranges and lamp types
- Custom simulator support
- Rollback safety with validation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class LampType(Enum):
    """Solar simulator lamp types"""
    XENON = "Xenon"
    METAL_HALIDE = "Metal Halide"
    LED = "LED"
    HALOGEN = "Halogen"
    MIXED = "Mixed (Xenon/LED)"
    CUSTOM = "Custom"


class IlluminationMode(Enum):
    """Simulator illumination modes"""
    CONTINUOUS = "Continuous"
    PULSED = "Pulsed"
    MULTI_FLASH = "Multi-Flash"


@dataclass
class IrradianceRange:
    """Irradiance range specification in W/m²"""
    min_wm2: float
    max_wm2: float

    def __post_init__(self):
        if self.min_wm2 < 0 or self.max_wm2 < 0:
            raise ValueError("Irradiance values must be non-negative")
        if self.min_wm2 > self.max_wm2:
            raise ValueError("Minimum irradiance cannot exceed maximum")

    def __str__(self) -> str:
        return f"{self.min_wm2:.0f}-{self.max_wm2:.0f} W/m²"

    def contains(self, value: float) -> bool:
        """Check if a value is within the irradiance range"""
        return self.min_wm2 <= value <= self.max_wm2


@dataclass
class SimulatorManufacturer:
    """
    Complete solar simulator manufacturer and model specification.

    Attributes:
        manufacturer_name: Name of the manufacturer (e.g., "Wavelabs")
        model_name: Model identifier (e.g., "WXS-156S-10")
        lamp_type: Type of light source (Xenon, LED, Metal Halide, etc.)
        typical_classification: Expected IEC 60904-9 classification (e.g., "AAA")
        test_plane_size: Illumination area dimensions (e.g., "156x156mm")
        irradiance_range: Min and max irradiance in W/m²
        illumination_mode: Continuous, Pulsed, or Multi-Flash
        pulse_duration_ms: For pulsed simulators, flash duration in milliseconds
        spectral_range_nm: Wavelength coverage (e.g., "300-1200nm")
        notes: Additional specifications or notes
    """
    manufacturer_name: str
    model_name: str
    lamp_type: LampType
    typical_classification: str  # e.g., "AAA", "A+A+A+", "ABA"
    test_plane_size: str  # e.g., "156x156mm", "200x200mm"
    irradiance_range: IrradianceRange
    illumination_mode: IlluminationMode = IlluminationMode.CONTINUOUS
    pulse_duration_ms: Optional[float] = None
    spectral_range_nm: str = "300-1200nm"
    notes: str = ""

    def __post_init__(self):
        """Validate simulator specifications"""
        # Validate classification format (should be 3 letters/grades)
        valid_grades = ['A+', 'A', 'B', 'C']
        classification = self.typical_classification.replace('+', 'PLUS')

        # Validate pulse duration for pulsed simulators
        if self.illumination_mode == IlluminationMode.PULSED:
            if self.pulse_duration_ms is None:
                logger.warning(f"Pulsed simulator {self.model_name} missing pulse duration")

        # Validate test plane size format
        if 'x' not in self.test_plane_size.lower() and 'mm' not in self.test_plane_size.lower():
            logger.warning(f"Test plane size '{self.test_plane_size}' may not be in standard format")

    @property
    def full_name(self) -> str:
        """Return full manufacturer + model name"""
        return f"{self.manufacturer_name} {self.model_name}"

    @property
    def is_pulsed(self) -> bool:
        """Check if simulator uses pulsed illumination"""
        return self.illumination_mode in [IlluminationMode.PULSED, IlluminationMode.MULTI_FLASH]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "manufacturer_name": self.manufacturer_name,
            "model_name": self.model_name,
            "lamp_type": self.lamp_type.value,
            "typical_classification": self.typical_classification,
            "test_plane_size": self.test_plane_size,
            "irradiance_range": {
                "min": self.irradiance_range.min_wm2,
                "max": self.irradiance_range.max_wm2
            },
            "illumination_mode": self.illumination_mode.value,
            "pulse_duration_ms": self.pulse_duration_ms,
            "spectral_range_nm": self.spectral_range_nm,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulatorManufacturer':
        """Create instance from dictionary"""
        irr_data = data.get("irradiance_range", {})
        return cls(
            manufacturer_name=data["manufacturer_name"],
            model_name=data["model_name"],
            lamp_type=LampType(data.get("lamp_type", "Xenon")),
            typical_classification=data["typical_classification"],
            test_plane_size=data["test_plane_size"],
            irradiance_range=IrradianceRange(
                min_wm2=irr_data.get("min", 0),
                max_wm2=irr_data.get("max", 1000)
            ),
            illumination_mode=IlluminationMode(data.get("illumination_mode", "Continuous")),
            pulse_duration_ms=data.get("pulse_duration_ms"),
            spectral_range_nm=data.get("spectral_range_nm", "300-1200nm"),
            notes=data.get("notes", "")
        )


# =============================================================================
# PRE-POPULATED SIMULATOR DATABASE
# =============================================================================

SIMULATOR_DATABASE: List[SimulatorManufacturer] = [
    # -------------------------------------------------------------------------
    # WAVELABS (Germany) - LED-based solar simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Wavelabs",
        model_name="WXS-156S-10",
        lamp_type=LampType.LED,
        typical_classification="A+A+A+",
        test_plane_size="156x156mm",
        irradiance_range=IrradianceRange(100, 1200),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="Single cell tester, adjustable spectrum, IEC 60904-9 Ed.3 compliant"
    ),
    SimulatorManufacturer(
        manufacturer_name="Wavelabs",
        model_name="WXS-300S-50",
        lamp_type=LampType.LED,
        typical_classification="A+A+A+",
        test_plane_size="300x300mm",
        irradiance_range=IrradianceRange(100, 1100),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="Multi-cell tester, high uniformity, spectrum programmable"
    ),
    SimulatorManufacturer(
        manufacturer_name="Wavelabs",
        model_name="SINUS-220",
        lamp_type=LampType.LED,
        typical_classification="AAA",
        test_plane_size="220x220mm",
        irradiance_range=IrradianceRange(200, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="Production line solar simulator"
    ),

    # -------------------------------------------------------------------------
    # SPIRE CORPORATION (USA) - Xenon flash simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Spire",
        model_name="SPI-SUN SIMULATOR 4600",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2000x1000mm",
        irradiance_range=IrradianceRange(800, 1200),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=10.0,
        spectral_range_nm="300-1200nm",
        notes="Large area module tester, high throughput production"
    ),
    SimulatorManufacturer(
        manufacturer_name="Spire",
        model_name="SPI-SUN 240A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="1650x1000mm",
        irradiance_range=IrradianceRange(900, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=10.0,
        spectral_range_nm="300-1200nm",
        notes="Standard module tester, research and production"
    ),
    SimulatorManufacturer(
        manufacturer_name="Spire",
        model_name="SPI-SUN 5600",
        lamp_type=LampType.XENON,
        typical_classification="A+A+A",
        test_plane_size="2400x1200mm",
        irradiance_range=IrradianceRange(800, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=12.0,
        spectral_range_nm="300-1200nm",
        notes="Bifacial module testing capability"
    ),

    # -------------------------------------------------------------------------
    # PASAN (Switzerland, Meyer Burger) - Premium simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="PASAN",
        model_name="SINUS 400",
        lamp_type=LampType.XENON,
        typical_classification="A+A+A+",
        test_plane_size="400x400mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=50.0,
        spectral_range_nm="300-1200nm",
        notes="Cell and mini-module tester, highest accuracy class"
    ),
    SimulatorManufacturer(
        manufacturer_name="PASAN",
        model_name="HELIOS 3030",
        lamp_type=LampType.XENON,
        typical_classification="A+A+A+",
        test_plane_size="3000x3000mm",
        irradiance_range=IrradianceRange(800, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=100.0,
        spectral_range_nm="300-1200nm",
        notes="Large area module tester, calibration laboratory grade"
    ),
    SimulatorManufacturer(
        manufacturer_name="PASAN",
        model_name="HIGHLITE 5",
        lamp_type=LampType.LED,
        typical_classification="AAA",
        test_plane_size="2100x1100mm",
        irradiance_range=IrradianceRange(200, 1100),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="LED-based production line tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="PASAN",
        model_name="SUNSIM 3C",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2200x1200mm",
        irradiance_range=IrradianceRange(700, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=30.0,
        spectral_range_nm="300-1200nm",
        notes="Production module tester with IV curve measurement"
    ),

    # -------------------------------------------------------------------------
    # GSOLA (China) - Cost-effective simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Gsola",
        model_name="G-SIM AAA-200",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="200x200mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Research cell tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="Gsola",
        model_name="G-SIM AAA-2200",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2200x1100mm",
        irradiance_range=IrradianceRange(800, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=15.0,
        spectral_range_nm="300-1200nm",
        notes="Production module tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="Gsola",
        model_name="G-SIM LED-156",
        lamp_type=LampType.LED,
        typical_classification="AAA",
        test_plane_size="156x156mm",
        irradiance_range=IrradianceRange(100, 1200),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="LED cell tester with adjustable spectrum"
    ),
    SimulatorManufacturer(
        manufacturer_name="Gsola",
        model_name="G-SIM 3B",
        lamp_type=LampType.XENON,
        typical_classification="BAB",
        test_plane_size="300x300mm",
        irradiance_range=IrradianceRange(200, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Budget research simulator"
    ),

    # -------------------------------------------------------------------------
    # MBJ SOLUTIONS (Germany) - Industrial simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="MBJ",
        model_name="Solar Simulator SS-200",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="200x200mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Cell tester for R&D applications"
    ),
    SimulatorManufacturer(
        manufacturer_name="MBJ",
        model_name="Solar Simulator MEGA-2200",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2200x1100mm",
        irradiance_range=IrradianceRange(700, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=20.0,
        spectral_range_nm="300-1200nm",
        notes="Industrial module tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="MBJ",
        model_name="HighLINE 1",
        lamp_type=LampType.LED,
        typical_classification="A+A+A",
        test_plane_size="160x160mm",
        irradiance_range=IrradianceRange(100, 1300),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="350-1100nm",
        notes="High-accuracy LED cell tester"
    ),

    # -------------------------------------------------------------------------
    # ENDEAS (Finland) - Solar Constant series
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Endeas",
        model_name="Solar Constant 1200",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="1200x1200mm",
        irradiance_range=IrradianceRange(200, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1200nm",
        notes="Research and calibration applications"
    ),
    SimulatorManufacturer(
        manufacturer_name="Endeas",
        model_name="Solar Constant 2400",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2400x1200mm",
        irradiance_range=IrradianceRange(500, 1100),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1200nm",
        notes="Large module testing"
    ),
    SimulatorManufacturer(
        manufacturer_name="Endeas",
        model_name="QuickSun 160",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="160x160mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=5.0,
        spectral_range_nm="300-1100nm",
        notes="High-speed cell tester"
    ),

    # -------------------------------------------------------------------------
    # HALM (Germany) - High-precision simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Halm",
        model_name="cetisPV-CT-L1",
        lamp_type=LampType.XENON,
        typical_classification="A+A+A+",
        test_plane_size="165x165mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=30.0,
        spectral_range_nm="300-1200nm",
        notes="Single cell tester, highest precision class"
    ),
    SimulatorManufacturer(
        manufacturer_name="Halm",
        model_name="cetisPV-CT-M2",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="220x220mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=25.0,
        spectral_range_nm="300-1200nm",
        notes="Multi-cell tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="Halm",
        model_name="cetisPV-IUCT-2400",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="2400x1300mm",
        irradiance_range=IrradianceRange(700, 1100),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=50.0,
        spectral_range_nm="300-1200nm",
        notes="Full module IV curve tracer"
    ),
    SimulatorManufacturer(
        manufacturer_name="Halm",
        model_name="cetisPV-HT",
        lamp_type=LampType.XENON,
        typical_classification="A+A+A",
        test_plane_size="400x400mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.PULSED,
        pulse_duration_ms=40.0,
        spectral_range_nm="300-1200nm",
        notes="Half-cell and shingled cell tester"
    ),

    # -------------------------------------------------------------------------
    # AVALON INSTRUMENTS (Hong Kong) - Sun Simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Avalon",
        model_name="Sun Simulator SS-100A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="100x100mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Small area research tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="Avalon",
        model_name="Sun Simulator SS-200A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="200x200mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Standard cell tester"
    ),
    SimulatorManufacturer(
        manufacturer_name="Avalon",
        model_name="Sun Simulator SS-X",
        lamp_type=LampType.XENON,
        typical_classification="ABA",
        test_plane_size="160x160mm",
        irradiance_range=IrradianceRange(200, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Economical cell tester"
    ),

    # -------------------------------------------------------------------------
    # NEWPORT ORIEL (USA) - Laboratory simulators
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="Newport Oriel",
        model_name="Sol3A Class AAA 94083A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="200x200mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Research grade solar simulator with AM filters"
    ),
    SimulatorManufacturer(
        manufacturer_name="Newport Oriel",
        model_name="Sol3A Class AAA 94123A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="300x300mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Large beam research solar simulator"
    ),
    SimulatorManufacturer(
        manufacturer_name="Newport Oriel",
        model_name="LCS-100 94011A",
        lamp_type=LampType.XENON,
        typical_classification="ABB",
        test_plane_size="50x50mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Compact laboratory solar simulator"
    ),
    SimulatorManufacturer(
        manufacturer_name="Newport Oriel",
        model_name="Sol2A Class ABA 69911",
        lamp_type=LampType.XENON,
        typical_classification="ABA",
        test_plane_size="100x100mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Educational and demonstration solar simulator"
    ),

    # -------------------------------------------------------------------------
    # Additional manufacturers for comprehensive coverage
    # -------------------------------------------------------------------------
    SimulatorManufacturer(
        manufacturer_name="ABET Technologies",
        model_name="Sun 2000 11016A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="100x100mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Research laboratory solar simulator"
    ),
    SimulatorManufacturer(
        manufacturer_name="ABET Technologies",
        model_name="Sun 3000 11096A",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="200x200mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Advanced research solar simulator"
    ),
    SimulatorManufacturer(
        manufacturer_name="Wacom Electric",
        model_name="WXS-210S-L2",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="210x210mm",
        irradiance_range=IrradianceRange(100, 1000),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1100nm",
        notes="Japanese precision solar simulator"
    ),
    SimulatorManufacturer(
        manufacturer_name="Sciencetech",
        model_name="SS-1.6K",
        lamp_type=LampType.XENON,
        typical_classification="AAA",
        test_plane_size="400x400mm",
        irradiance_range=IrradianceRange(200, 1600),
        illumination_mode=IlluminationMode.CONTINUOUS,
        spectral_range_nm="300-1200nm",
        notes="High-power research simulator"
    ),
]


class SimulatorDatabase:
    """
    Manager class for the simulator database with rollback safety.

    Provides methods for querying, validating, and managing simulator data
    with support for custom simulators and rollback operations.
    """

    def __init__(self):
        self._simulators: List[SimulatorManufacturer] = list(SIMULATOR_DATABASE)
        self._custom_simulators: List[SimulatorManufacturer] = []
        self._backup: Optional[List[SimulatorManufacturer]] = None

    @property
    def all_simulators(self) -> List[SimulatorManufacturer]:
        """Get all simulators including custom ones"""
        return self._simulators + self._custom_simulators

    def get_manufacturers(self) -> List[str]:
        """Get unique list of manufacturer names"""
        manufacturers = set(s.manufacturer_name for s in self.all_simulators)
        return sorted(list(manufacturers))

    def get_models_by_manufacturer(self, manufacturer: str) -> List[SimulatorManufacturer]:
        """Get all models for a specific manufacturer"""
        return [s for s in self.all_simulators if s.manufacturer_name == manufacturer]

    def get_model_names_by_manufacturer(self, manufacturer: str) -> List[str]:
        """Get model names for a specific manufacturer"""
        return [s.model_name for s in self.get_models_by_manufacturer(manufacturer)]

    def get_simulator(self, manufacturer: str, model: str) -> Optional[SimulatorManufacturer]:
        """Get a specific simulator by manufacturer and model"""
        for s in self.all_simulators:
            if s.manufacturer_name == manufacturer and s.model_name == model:
                return s
        return None

    def search_simulators(
        self,
        lamp_type: Optional[LampType] = None,
        min_area_mm2: Optional[float] = None,
        classification_grade: Optional[str] = None,
        illumination_mode: Optional[IlluminationMode] = None
    ) -> List[SimulatorManufacturer]:
        """Search simulators with filters"""
        results = self.all_simulators

        if lamp_type:
            results = [s for s in results if s.lamp_type == lamp_type]

        if classification_grade:
            results = [s for s in results if classification_grade in s.typical_classification]

        if illumination_mode:
            results = [s for s in results if s.illumination_mode == illumination_mode]

        if min_area_mm2:
            filtered = []
            for s in results:
                try:
                    # Parse test plane size like "156x156mm"
                    size = s.test_plane_size.lower().replace('mm', '')
                    dims = size.split('x')
                    if len(dims) == 2:
                        area = float(dims[0]) * float(dims[1])
                        if area >= min_area_mm2:
                            filtered.append(s)
                except (ValueError, IndexError):
                    pass
            results = filtered

        return results

    def add_custom_simulator(self, simulator: SimulatorManufacturer) -> bool:
        """
        Add a custom simulator with validation.

        Args:
            simulator: The custom simulator to add

        Returns:
            bool: True if added successfully
        """
        # Validate that it doesn't duplicate an existing entry
        if self.get_simulator(simulator.manufacturer_name, simulator.model_name):
            logger.warning(f"Simulator {simulator.full_name} already exists")
            return False

        self._custom_simulators.append(simulator)
        return True

    def create_backup(self) -> None:
        """Create a backup of current state for rollback"""
        self._backup = list(self._custom_simulators)

    def rollback(self) -> bool:
        """
        Rollback to last backup state.

        Returns:
            bool: True if rollback successful
        """
        if self._backup is not None:
            self._custom_simulators = self._backup
            self._backup = None
            return True
        return False

    def clear_custom_simulators(self) -> None:
        """Remove all custom simulators"""
        self.create_backup()
        self._custom_simulators = []

    def validate_simulator(self, simulator: SimulatorManufacturer) -> Tuple[bool, List[str]]:
        """
        Validate a simulator configuration.

        Args:
            simulator: The simulator to validate

        Returns:
            Tuple of (is_valid, list of warning/error messages)
        """
        messages = []
        is_valid = True

        # Check required fields
        if not simulator.manufacturer_name:
            messages.append("Error: Manufacturer name is required")
            is_valid = False

        if not simulator.model_name:
            messages.append("Error: Model name is required")
            is_valid = False

        # Validate irradiance range
        if simulator.irradiance_range.max_wm2 > 2000:
            messages.append("Warning: Irradiance exceeds 2000 W/m², verify if correct")

        if simulator.irradiance_range.min_wm2 < 50:
            messages.append("Warning: Minimum irradiance below 50 W/m² may be too low")

        # Validate pulse duration for pulsed simulators
        if simulator.is_pulsed and simulator.pulse_duration_ms:
            if simulator.pulse_duration_ms < 1:
                messages.append("Warning: Pulse duration < 1ms may be too short for IV measurement")
            elif simulator.pulse_duration_ms > 200:
                messages.append("Warning: Pulse duration > 200ms is unusually long")

        # Validate classification format
        valid_chars = set('A+BC')
        class_chars = set(simulator.typical_classification.replace('+', ''))
        if not class_chars.issubset(valid_chars):
            messages.append(f"Warning: Classification '{simulator.typical_classification}' contains invalid characters")

        return is_valid, messages

    def to_json(self) -> str:
        """Export database to JSON"""
        data = {
            "standard_simulators": [s.to_dict() for s in self._simulators],
            "custom_simulators": [s.to_dict() for s in self._custom_simulators]
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'SimulatorDatabase':
        """Import database from JSON"""
        db = cls()
        data = json.loads(json_str)

        if "custom_simulators" in data:
            for sim_data in data["custom_simulators"]:
                try:
                    sim = SimulatorManufacturer.from_dict(sim_data)
                    db.add_custom_simulator(sim)
                except Exception as e:
                    logger.error(f"Failed to import custom simulator: {e}")

        return db


# Global database instance
_database: Optional[SimulatorDatabase] = None


def get_simulator_database() -> SimulatorDatabase:
    """Get the global simulator database instance"""
    global _database
    if _database is None:
        _database = SimulatorDatabase()
    return _database


def create_custom_simulator(
    manufacturer_name: str,
    model_name: str,
    lamp_type: str,
    typical_classification: str,
    test_plane_size: str,
    irradiance_min: float,
    irradiance_max: float,
    illumination_mode: str = "Continuous",
    pulse_duration_ms: Optional[float] = None,
    spectral_range_nm: str = "300-1200nm",
    notes: str = ""
) -> SimulatorManufacturer:
    """
    Helper function to create a custom simulator with string inputs.

    Args:
        manufacturer_name: Name of manufacturer
        model_name: Model identifier
        lamp_type: One of "Xenon", "LED", "Metal Halide", "Halogen", "Mixed", "Custom"
        typical_classification: e.g., "AAA", "A+A+A+"
        test_plane_size: e.g., "156x156mm"
        irradiance_min: Minimum irradiance in W/m²
        irradiance_max: Maximum irradiance in W/m²
        illumination_mode: "Continuous", "Pulsed", or "Multi-Flash"
        pulse_duration_ms: Flash duration for pulsed simulators
        spectral_range_nm: Wavelength range
        notes: Additional notes

    Returns:
        SimulatorManufacturer instance
    """
    # Convert string to enum
    lamp_type_enum = LampType(lamp_type)
    mode_enum = IlluminationMode(illumination_mode)

    return SimulatorManufacturer(
        manufacturer_name=manufacturer_name,
        model_name=model_name,
        lamp_type=lamp_type_enum,
        typical_classification=typical_classification,
        test_plane_size=test_plane_size,
        irradiance_range=IrradianceRange(irradiance_min, irradiance_max),
        illumination_mode=mode_enum,
        pulse_duration_ms=pulse_duration_ms,
        spectral_range_nm=spectral_range_nm,
        notes=notes
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_lamp_type_icon(lamp_type: LampType) -> str:
    """Get icon for lamp type"""
    icons = {
        LampType.XENON: "💡",
        LampType.LED: "🔆",
        LampType.METAL_HALIDE: "⚡",
        LampType.HALOGEN: "🔥",
        LampType.MIXED: "🌈",
        LampType.CUSTOM: "⚙️",
    }
    return icons.get(lamp_type, "💡")


def get_classification_color(classification: str) -> str:
    """Get color for classification grade"""
    if "A+" in classification or classification.startswith("A+"):
        return "#10B981"  # Emerald green
    elif classification.startswith("A"):
        return "#22C55E"  # Green
    elif classification.startswith("B"):
        return "#F59E0B"  # Amber
    elif classification.startswith("C"):
        return "#EF4444"  # Red
    return "#6B7280"  # Gray


def format_test_plane_size(size_str: str) -> str:
    """Format test plane size for display"""
    return size_str.replace("x", " × ")
