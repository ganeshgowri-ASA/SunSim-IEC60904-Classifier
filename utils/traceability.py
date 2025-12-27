"""
Sun Simulator Classification System - Traceability Module
IEC 60904-2 and IEC 60904-4 Compliance

This module provides calibration chain validation and traceability management
for reference modules and solar simulators.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import json


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ReferenceLevel(Enum):
    """Calibration hierarchy levels per IEC 60904-2."""
    PRIMARY = "primary"           # Calibrated by national lab (PTB, NREL, etc.)
    SECONDARY = "secondary"       # Calibrated against primary reference
    WORKING = "working"           # Calibrated against secondary reference
    PRODUCTION = "production"     # For routine production testing


class CertificateStatus(Enum):
    """Calibration certificate status."""
    VALID = "valid"
    EXPIRING_SOON = "expiring_soon"  # Within 30 days of expiration
    EXPIRED = "expired"
    PENDING = "pending"
    SUSPENDED = "suspended"


class ModuleType(Enum):
    """Reference module technology types."""
    MONO_SI = "mono_si"           # Monocrystalline Silicon
    MULTI_SI = "multi_si"         # Multicrystalline Silicon
    PERC = "perc"                 # PERC cells
    TOPCON = "topcon"             # TOPCon cells
    HJT = "hjt"                   # Heterojunction
    CDTE = "cdte"                 # Cadmium Telluride
    CIGS = "cigs"                 # Copper Indium Gallium Selenide
    PEROVSKITE = "perovskite"     # Perovskite
    BIFACIAL = "bifacial"         # Bifacial module
    OTHER = "other"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalibrationCertificate:
    """Represents a calibration certificate."""
    certificate_number: str
    issue_date: datetime
    expiration_date: datetime
    calibration_lab: str
    lab_accreditation: str  # e.g., "ISO 17025", "NVLAP"
    accreditation_number: Optional[str] = None
    calibration_method: str = "IEC 60904-2"
    traceability_chain: List[str] = field(default_factory=list)
    uncertainty_percent: float = 1.5
    coverage_factor: float = 2.0  # k=2 for 95% confidence
    temperature_range_c: Tuple[float, float] = (20.0, 30.0)
    irradiance_range_wm2: Tuple[float, float] = (800.0, 1200.0)
    notes: Optional[str] = None

    @property
    def status(self) -> CertificateStatus:
        """Determine current certificate status."""
        now = datetime.now()
        if now > self.expiration_date:
            return CertificateStatus.EXPIRED
        elif now > self.expiration_date - timedelta(days=30):
            return CertificateStatus.EXPIRING_SOON
        return CertificateStatus.VALID

    @property
    def days_until_expiration(self) -> int:
        """Calculate days until certificate expires."""
        delta = self.expiration_date - datetime.now()
        return max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "certificate_number": self.certificate_number,
            "issue_date": self.issue_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat(),
            "calibration_lab": self.calibration_lab,
            "lab_accreditation": self.lab_accreditation,
            "accreditation_number": self.accreditation_number,
            "calibration_method": self.calibration_method,
            "traceability_chain": self.traceability_chain,
            "uncertainty_percent": self.uncertainty_percent,
            "coverage_factor": self.coverage_factor,
            "status": self.status.value,
            "days_until_expiration": self.days_until_expiration
        }


@dataclass
class ReferenceModuleSpec:
    """Reference module specifications for traceability."""
    serial_number: str
    module_type: ModuleType
    manufacturer: str
    model: str
    reference_level: ReferenceLevel

    # Electrical characteristics at STC
    isc_stc_a: float          # Short-circuit current (A)
    voc_stc_v: float          # Open-circuit voltage (V)
    pmax_stc_w: float         # Maximum power (W)
    impp_a: float             # Current at MPP (A)
    vmpp_v: float             # Voltage at MPP (V)
    ff_percent: float         # Fill factor (%)

    # Physical characteristics
    area_cm2: float           # Active area
    cell_count: int = 1       # Number of cells

    # Temperature coefficients
    alpha_isc_pct_k: float = 0.05      # Isc temp coeff (%/K)
    beta_voc_pct_k: float = -0.30      # Voc temp coeff (%/K)
    gamma_pmax_pct_k: float = -0.40    # Pmax temp coeff (%/K)

    # Spectral response
    spectral_response_file: Optional[str] = None
    spectral_mismatch_factor: float = 1.0

    # Calibration info
    current_certificate: Optional[CalibrationCertificate] = None
    calibration_history: List[CalibrationCertificate] = field(default_factory=list)

    # Status
    is_active: bool = True
    notes: Optional[str] = None

    @property
    def efficiency_percent(self) -> float:
        """Calculate module efficiency."""
        # Assuming STC irradiance of 1000 W/m²
        irradiance = 1000  # W/m²
        area_m2 = self.area_cm2 / 10000
        return (self.pmax_stc_w / (irradiance * area_m2)) * 100

    @property
    def calibration_status(self) -> CertificateStatus:
        """Get current calibration status."""
        if self.current_certificate is None:
            return CertificateStatus.PENDING
        return self.current_certificate.status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "serial_number": self.serial_number,
            "module_type": self.module_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "reference_level": self.reference_level.value,
            "isc_stc_a": self.isc_stc_a,
            "voc_stc_v": self.voc_stc_v,
            "pmax_stc_w": self.pmax_stc_w,
            "impp_a": self.impp_a,
            "vmpp_v": self.vmpp_v,
            "ff_percent": self.ff_percent,
            "area_cm2": self.area_cm2,
            "efficiency_percent": round(self.efficiency_percent, 2),
            "calibration_status": self.calibration_status.value,
            "is_active": self.is_active
        }


@dataclass
class DriftRecord:
    """Records drift measurements over time."""
    measurement_date: datetime
    isc_measured_a: float
    voc_measured_v: float
    pmax_measured_w: float
    irradiance_wm2: float
    temperature_c: float
    spectral_mismatch: float = 1.0

    # Calculated drift values (vs reference)
    isc_drift_percent: Optional[float] = None
    pmax_drift_percent: Optional[float] = None

    notes: Optional[str] = None


# =============================================================================
# TRACEABILITY MANAGER
# =============================================================================

class TraceabilityManager:
    """
    Manages calibration chain and traceability per IEC 60904-2 and IEC 60904-4.

    The calibration chain typically follows:
    PRIMARY (National Lab) -> SECONDARY (Accredited Lab) -> WORKING (In-house)
    """

    # Standard calibration intervals (days)
    CALIBRATION_INTERVALS = {
        ReferenceLevel.PRIMARY: 365 * 2,     # 2 years
        ReferenceLevel.SECONDARY: 365,        # 1 year
        ReferenceLevel.WORKING: 180,          # 6 months
        ReferenceLevel.PRODUCTION: 90         # 3 months
    }

    # Maximum allowed drift before recalibration (%)
    MAX_DRIFT_LIMITS = {
        ReferenceLevel.PRIMARY: 0.5,
        ReferenceLevel.SECONDARY: 1.0,
        ReferenceLevel.WORKING: 2.0,
        ReferenceLevel.PRODUCTION: 3.0
    }

    # Uncertainty budget components (IEC 60904-4)
    UNCERTAINTY_COMPONENTS = {
        "spectral_mismatch": 0.5,        # %
        "irradiance_nonuniformity": 0.3,  # %
        "reference_module": 1.5,          # %
        "data_acquisition": 0.2,          # %
        "temperature": 0.3,               # %
        "repeatability": 0.2              # %
    }

    def __init__(self):
        self.reference_modules: Dict[str, ReferenceModuleSpec] = {}
        self.drift_records: Dict[str, List[DriftRecord]] = {}

    def add_reference_module(self, module: ReferenceModuleSpec) -> None:
        """Register a reference module."""
        self.reference_modules[module.serial_number] = module
        self.drift_records[module.serial_number] = []

    def add_drift_record(self, serial_number: str, record: DriftRecord) -> bool:
        """Add a drift measurement record."""
        if serial_number not in self.reference_modules:
            return False

        module = self.reference_modules[serial_number]

        # Calculate drift from reference values
        record.isc_drift_percent = (
            (record.isc_measured_a - module.isc_stc_a) / module.isc_stc_a * 100
        )
        record.pmax_drift_percent = (
            (record.pmax_measured_w - module.pmax_stc_w) / module.pmax_stc_w * 100
        )

        self.drift_records[serial_number].append(record)
        return True

    def check_calibration_validity(self, serial_number: str) -> Dict[str, Any]:
        """
        Check if a reference module's calibration is valid.

        Returns:
            Dictionary with validity status and details.
        """
        if serial_number not in self.reference_modules:
            return {"valid": False, "reason": "Module not found"}

        module = self.reference_modules[serial_number]
        result = {
            "serial_number": serial_number,
            "reference_level": module.reference_level.value,
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # Check certificate status
        if module.current_certificate is None:
            result["valid"] = False
            result["errors"].append("No calibration certificate on file")
        else:
            cert_status = module.current_certificate.status
            result["certificate_status"] = cert_status.value

            if cert_status == CertificateStatus.EXPIRED:
                result["valid"] = False
                result["errors"].append("Calibration certificate has expired")
            elif cert_status == CertificateStatus.EXPIRING_SOON:
                result["warnings"].append(
                    f"Certificate expires in {module.current_certificate.days_until_expiration} days"
                )

        # Check drift history
        if serial_number in self.drift_records and self.drift_records[serial_number]:
            latest_drift = self.drift_records[serial_number][-1]
            drift_limit = self.MAX_DRIFT_LIMITS[module.reference_level]

            if abs(latest_drift.isc_drift_percent or 0) > drift_limit:
                result["warnings"].append(
                    f"Isc drift ({latest_drift.isc_drift_percent:.2f}%) exceeds limit ({drift_limit}%)"
                )

            if abs(latest_drift.pmax_drift_percent or 0) > drift_limit:
                result["warnings"].append(
                    f"Pmax drift ({latest_drift.pmax_drift_percent:.2f}%) exceeds limit ({drift_limit}%)"
                )

        return result

    def calculate_combined_uncertainty(
        self,
        reference_uncertainty: float = 1.5,
        additional_components: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate combined measurement uncertainty per IEC 60904-4.

        Args:
            reference_uncertainty: Uncertainty of reference module (%)
            additional_components: Additional uncertainty components

        Returns:
            Combined expanded uncertainty (k=2)
        """
        components = dict(self.UNCERTAINTY_COMPONENTS)
        components["reference_module"] = reference_uncertainty

        if additional_components:
            components.update(additional_components)

        # Root sum of squares for uncorrelated components
        sum_squares = sum(u**2 for u in components.values())
        combined = (sum_squares ** 0.5)

        # Apply coverage factor k=2 for 95% confidence
        expanded = combined * 2.0

        return round(expanded, 2)

    def validate_traceability_chain(
        self,
        working_module: str,
        secondary_module: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate the complete traceability chain from working reference
        back to primary standard.

        Returns:
            Validation result with chain details.
        """
        result = {
            "valid": True,
            "chain": [],
            "total_uncertainty_percent": 0.0,
            "issues": []
        }

        # Check working module
        if working_module not in self.reference_modules:
            result["valid"] = False
            result["issues"].append(f"Working module {working_module} not registered")
            return result

        working = self.reference_modules[working_module]
        working_check = self.check_calibration_validity(working_module)

        result["chain"].append({
            "level": "working",
            "serial": working_module,
            "valid": working_check["valid"],
            "uncertainty": working.current_certificate.uncertainty_percent if working.current_certificate else None
        })

        if not working_check["valid"]:
            result["valid"] = False
            result["issues"].extend(working_check["errors"])

        # Check secondary module if specified
        if secondary_module:
            if secondary_module not in self.reference_modules:
                result["issues"].append(f"Secondary module {secondary_module} not registered")
            else:
                secondary = self.reference_modules[secondary_module]
                secondary_check = self.check_calibration_validity(secondary_module)

                result["chain"].append({
                    "level": "secondary",
                    "serial": secondary_module,
                    "valid": secondary_check["valid"],
                    "uncertainty": secondary.current_certificate.uncertainty_percent if secondary.current_certificate else None
                })

                if not secondary_check["valid"]:
                    result["valid"] = False
                    result["issues"].extend(secondary_check["errors"])

        # Calculate total uncertainty through chain
        uncertainties = [
            item["uncertainty"] for item in result["chain"]
            if item["uncertainty"] is not None
        ]
        if uncertainties:
            result["total_uncertainty_percent"] = self.calculate_combined_uncertainty(
                max(uncertainties)
            )

        return result

    def get_matching_criteria(
        self,
        test_device_type: ModuleType,
        reference_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Determine matching criteria between test device and reference module
        per IEC 60904-7.

        Args:
            test_device_type: Technology type of device under test
            reference_modules: List of available reference module serials

        Returns:
            Matching recommendations and spectral mismatch considerations.
        """
        result = {
            "test_device_type": test_device_type.value,
            "recommended_references": [],
            "matching_notes": [],
            "spectral_mismatch_risk": "low"
        }

        modules_to_check = reference_modules or list(self.reference_modules.keys())

        for serial in modules_to_check:
            if serial not in self.reference_modules:
                continue

            module = self.reference_modules[serial]

            # Check if module is active and valid
            if not module.is_active:
                continue

            validity = self.check_calibration_validity(serial)
            if not validity["valid"]:
                continue

            # Determine matching quality
            match_quality = self._calculate_match_quality(test_device_type, module.module_type)

            result["recommended_references"].append({
                "serial": serial,
                "type": module.module_type.value,
                "match_quality": match_quality,
                "uncertainty": module.current_certificate.uncertainty_percent if module.current_certificate else None
            })

        # Sort by match quality
        result["recommended_references"].sort(
            key=lambda x: ("excellent", "good", "acceptable", "poor").index(x["match_quality"])
        )

        # Add matching notes
        if test_device_type in [ModuleType.PEROVSKITE, ModuleType.CDTE, ModuleType.CIGS]:
            result["matching_notes"].append(
                "Thin-film technology - consider spectral mismatch correction"
            )
            result["spectral_mismatch_risk"] = "high"

        if test_device_type == ModuleType.BIFACIAL:
            result["matching_notes"].append(
                "Bifacial module - ensure reference captures both sides"
            )

        return result

    def _calculate_match_quality(
        self,
        test_type: ModuleType,
        reference_type: ModuleType
    ) -> str:
        """Determine spectral matching quality between technologies."""
        # Same technology family is excellent
        if test_type == reference_type:
            return "excellent"

        # Silicon types generally match well with each other
        silicon_types = {ModuleType.MONO_SI, ModuleType.MULTI_SI, ModuleType.PERC,
                        ModuleType.TOPCON, ModuleType.HJT}

        if test_type in silicon_types and reference_type in silicon_types:
            return "good"

        # Thin-film types need careful matching
        thin_film_types = {ModuleType.CDTE, ModuleType.CIGS, ModuleType.PEROVSKITE}

        if test_type in thin_film_types and reference_type in thin_film_types:
            return "acceptable"

        # Cross-technology matching is generally poor
        if (test_type in thin_film_types and reference_type in silicon_types) or \
           (test_type in silicon_types and reference_type in thin_film_types):
            return "poor"

        return "acceptable"

    def generate_traceability_report(self, serial_number: str) -> Dict[str, Any]:
        """
        Generate a complete traceability report for a reference module.

        Returns:
            Comprehensive traceability report.
        """
        if serial_number not in self.reference_modules:
            return {"error": "Module not found"}

        module = self.reference_modules[serial_number]
        validity = self.check_calibration_validity(serial_number)

        report = {
            "report_date": datetime.now().isoformat(),
            "module_info": module.to_dict(),
            "calibration_validity": validity,
            "calibration_history": [],
            "drift_analysis": {
                "records_count": 0,
                "latest_isc_drift_percent": None,
                "latest_pmax_drift_percent": None,
                "drift_trend": "stable"
            },
            "traceability_chain": [],
            "compliance": {
                "iec_60904_2": True,
                "iec_60904_4": True,
                "notes": []
            }
        }

        # Add certificate history
        if module.current_certificate:
            report["traceability_chain"] = module.current_certificate.traceability_chain

        for cert in module.calibration_history:
            report["calibration_history"].append(cert.to_dict())

        # Analyze drift
        if serial_number in self.drift_records and self.drift_records[serial_number]:
            records = self.drift_records[serial_number]
            report["drift_analysis"]["records_count"] = len(records)

            latest = records[-1]
            report["drift_analysis"]["latest_isc_drift_percent"] = latest.isc_drift_percent
            report["drift_analysis"]["latest_pmax_drift_percent"] = latest.pmax_drift_percent

            # Determine trend
            if len(records) >= 3:
                recent_drifts = [r.pmax_drift_percent for r in records[-3:] if r.pmax_drift_percent]
                if recent_drifts and len(recent_drifts) >= 2:
                    if all(d2 > d1 for d1, d2 in zip(recent_drifts, recent_drifts[1:])):
                        report["drift_analysis"]["drift_trend"] = "increasing"
                    elif all(d2 < d1 for d1, d2 in zip(recent_drifts, recent_drifts[1:])):
                        report["drift_analysis"]["drift_trend"] = "decreasing"

        # Check compliance
        if not validity["valid"]:
            report["compliance"]["iec_60904_2"] = False
            report["compliance"]["notes"].append("Calibration not valid - IEC 60904-2 compliance affected")

        return report


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_sample_reference_modules() -> List[ReferenceModuleSpec]:
    """Create sample reference modules for demonstration."""
    samples = [
        ReferenceModuleSpec(
            serial_number="REF-2024-001",
            module_type=ModuleType.MONO_SI,
            manufacturer="ISTI-CNR",
            model="Primary Reference Cell",
            reference_level=ReferenceLevel.PRIMARY,
            isc_stc_a=0.1580,
            voc_stc_v=0.632,
            pmax_stc_w=0.0850,
            impp_a=0.1495,
            vmpp_v=0.568,
            ff_percent=85.2,
            area_cm2=4.0,
            current_certificate=CalibrationCertificate(
                certificate_number="ISTI-2024-00123",
                issue_date=datetime(2024, 6, 15),
                expiration_date=datetime(2026, 6, 15),
                calibration_lab="ISTI-CNR Pisa",
                lab_accreditation="ISO 17025",
                accreditation_number="ACCREDIA 0123",
                traceability_chain=["ISTI-CNR Primary", "World Photovoltaic Scale"],
                uncertainty_percent=0.8
            )
        ),
        ReferenceModuleSpec(
            serial_number="REF-2024-002",
            module_type=ModuleType.MONO_SI,
            manufacturer="Fraunhofer ISE",
            model="Secondary Reference Cell",
            reference_level=ReferenceLevel.SECONDARY,
            isc_stc_a=0.1565,
            voc_stc_v=0.628,
            pmax_stc_w=0.0838,
            impp_a=0.1478,
            vmpp_v=0.567,
            ff_percent=85.0,
            area_cm2=4.0,
            current_certificate=CalibrationCertificate(
                certificate_number="ISE-2024-00456",
                issue_date=datetime(2024, 3, 10),
                expiration_date=datetime(2025, 3, 10),
                calibration_lab="Fraunhofer ISE CalLab",
                lab_accreditation="ISO 17025",
                accreditation_number="DAkkS D-K-15070-01-00",
                traceability_chain=["Fraunhofer ISE", "PTB Braunschweig", "SI Units"],
                uncertainty_percent=1.2
            )
        ),
        ReferenceModuleSpec(
            serial_number="REF-2024-003",
            module_type=ModuleType.PERC,
            manufacturer="In-house",
            model="Working Reference Module",
            reference_level=ReferenceLevel.WORKING,
            isc_stc_a=9.85,
            voc_stc_v=0.698,
            pmax_stc_w=5.95,
            impp_a=9.45,
            vmpp_v=0.630,
            ff_percent=86.5,
            area_cm2=243.36,
            cell_count=1,
            current_certificate=CalibrationCertificate(
                certificate_number="LAB-2024-00789",
                issue_date=datetime(2024, 9, 1),
                expiration_date=datetime(2025, 3, 1),
                calibration_lab="PV Test Laboratory",
                lab_accreditation="ISO 17025",
                traceability_chain=["PV Test Lab", "Fraunhofer ISE", "PTB"],
                uncertainty_percent=1.8
            )
        )
    ]

    return samples


def load_manufacturers_data(filepath: str = "data/manufacturers.json") -> Dict[str, Any]:
    """Load manufacturer data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"manufacturers": [], "metadata": {}}


def get_manufacturer_by_id(manufacturer_id: str, data: Optional[Dict] = None) -> Optional[Dict]:
    """Get manufacturer details by ID."""
    if data is None:
        data = load_manufacturers_data()

    for mfr in data.get("manufacturers", []):
        if mfr.get("id") == manufacturer_id:
            return mfr
    return None


def get_all_models(data: Optional[Dict] = None) -> List[Dict]:
    """Get all models from all manufacturers."""
    if data is None:
        data = load_manufacturers_data()

    all_models = []
    for mfr in data.get("manufacturers", []):
        for model in mfr.get("models", []):
            model_info = dict(model)
            model_info["manufacturer_id"] = mfr.get("id")
            model_info["manufacturer_name"] = mfr.get("name")
            all_models.append(model_info)

    return all_models


def filter_models_by_classification(
    target_class: str,
    parameter: str = "overall",
    data: Optional[Dict] = None
) -> List[Dict]:
    """Filter models by target classification."""
    models = get_all_models(data)
    filtered = []

    for model in models:
        typical = model.get("typical_classification", {})

        if parameter == "overall":
            # Check all parameters
            classes = [typical.get("spectral"), typical.get("uniformity"), typical.get("temporal")]
            overall = min(classes, key=lambda x: {"A+": 0, "A": 1, "B": 2, "C": 3}.get(x, 4))
            if overall == target_class:
                filtered.append(model)
        else:
            if typical.get(parameter) == target_class:
                filtered.append(model)

    return filtered
