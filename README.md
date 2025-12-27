# SunSim-IEC60904-Classifier

Professional Sun Simulator Classification System compliant with IEC 60904-9:2020 Ed.3 standards. Features comprehensive spectral match, spatial uniformity, and temporal stability analysis with ISO 17025 report generation and SPC/MSA quality control.

## Features

### Core Classification (IEC 60904-9:2020 Ed.3)

- **Spectral Match Analysis** - 6/7 wavelength band classification (300-1100nm / 300-1200nm extended)
- **Spatial Uniformity** - Irradiance distribution mapping with heatmap visualization
- **Temporal Stability** - Short-term (STI) and long-term (LTI) instability measurement
- **Classification Grades** - A+/A/B/C rating system per IEC standards

### Advanced Analysis

- **Capacitive Effects (IEC 60904-14)**
  - Module capacitance measurement and estimation
  - I-V sweep rate optimization
  - Correction factors calculation
  - Four-terminal (Kelvin) connection guide

- **Angular Distribution (FOV/Solid Angle)**
  - Beam collimation analysis
  - Angle-of-incidence responsivity
  - Diffuse vs. direct irradiance ratio

### System Features

- **Dashboard** - Real-time classification overview and statistics
- **Settings Panel** - Comprehensive system configuration
- **User Management** - Role-based access (Admin/Operator/Viewer)
- **Data Export/Import** - JSON, CSV, Excel format support
- **Alarm Thresholds** - Configurable warning and critical alerts

## Technology Stack

- **Frontend**: Streamlit with custom dark theme
- **Backend**: Python 3.9+
- **Database**: PostgreSQL (Railway-compatible)
- **Charts**: Plotly for interactive visualizations
- **ORM**: SQLAlchemy with connection pooling

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 13 or higher (or Railway PostgreSQL)
- pip package manager

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/SunSim-IEC60904-Classifier.git
cd SunSim-IEC60904-Classifier
```

2. **Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```env
# Database Configuration (Option 1: Full URL)
DATABASE_URL=postgresql://user:password@localhost:5432/sunsim

# Database Configuration (Option 2: Individual components)
PGHOST=localhost
PGPORT=5432
PGDATABASE=sunsim
PGUSER=postgres
PGPASSWORD=yourpassword

# Optional Settings
DB_ECHO=false
```

5. **Initialize the database**

```python
from utils.db import init_database
init_database()
```

6. **Run the application**

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Railway PostgreSQL Setup

### Creating a Railway Project

1. **Create a Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub for easy deployment

2. **Create a New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo" or "Empty Project"

3. **Add PostgreSQL Database**
   - In your project, click "New"
   - Select "Database" > "PostgreSQL"
   - Railway will automatically provision the database

4. **Get Connection Details**
   - Click on the PostgreSQL service
   - Go to "Variables" tab
   - Copy the `DATABASE_URL` or individual variables:
     - `PGHOST`
     - `PGPORT`
     - `PGDATABASE`
     - `PGUSER`
     - `PGPASSWORD`

### Environment Variables for Railway

Set these in your Railway service:

```
DATABASE_URL=${DATABASE_URL}
```

Railway automatically provides the database URL when linked.

## Streamlit Cloud Deployment

### Deploying to Streamlit Cloud

1. **Push to GitHub**
   Ensure your repository is on GitHub with all required files.

2. **Create Streamlit Cloud Account**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account

3. **Deploy the App**
   - Click "New app"
   - Select your repository
   - Set the main file path: `app.py`
   - Click "Deploy"

4. **Configure Secrets**
   In Streamlit Cloud, go to "Settings" > "Secrets" and add:

```toml
[database]
DATABASE_URL = "postgresql://user:password@host:port/database"
```

Or use individual variables:

```toml
[database]
PGHOST = "your-railway-host"
PGPORT = "5432"
PGDATABASE = "railway"
PGUSER = "postgres"
PGPASSWORD = "your-password"
```

### Required Files for Deployment

Ensure these files exist in your repository:

- `requirements.txt` - Python dependencies
- `app.py` - Main application entry point
- `.streamlit/config.toml` - (Optional) Streamlit configuration

Example `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f8fafc"
font = "sans serif"

[server]
maxUploadSize = 50
```

## Project Structure

```
SunSim-IEC60904-Classifier/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── pages/                      # Streamlit multipage app
│   ├── 1_📊_Dashboard.py       # Classification dashboard
│   ├── 2_🌈_Spectral_Analysis.py
│   ├── 3_🗺️_Uniformity.py
│   ├── 4_⏱️_Temporal_Stability.py
│   └── 14_⚙️_Settings.py       # System settings
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── db.py                   # Database models and functions
│   ├── calculations.py         # Classification calculations
│   ├── capacitive_effects.py   # IEC 60904-14 capacitance
│   └── angular_distribution.py # FOV/solid angle analysis
│
├── data/                       # Reference data
│   └── AM15G_reference.csv     # AM1.5G reference spectrum
│
└── .streamlit/                 # Streamlit configuration
    └── config.toml
```

## Database Schema

### Core Tables

- **manufacturers** - Equipment manufacturer information
- **simulators** - Solar simulator equipment records
- **reference_modules** - Calibration reference modules
- **measurements** - Individual measurement sessions
- **spectral_data** - Spectral irradiance measurements
- **uniformity_data** - Spatial uniformity grid data
- **lamp_history** - Lamp usage and replacement tracking

### Settings Tables

- **users** - User accounts with role-based access
- **system_settings** - Configuration key-value pairs
- **alarm_thresholds** - Configurable alarm limits
- **settings_audit_log** - Change tracking for compliance
- **data_export_import** - Export/import operation history

## IEC 60904-9:2020 Classification Limits

| Parameter | A+ | A | B | C |
|-----------|-----|-----|-----|-----|
| Spectral Match | ±12.5% | ±25% | ±40% | >40% |
| Spatial Uniformity | ≤1% | ≤2% | ≤5% | >5% |
| Temporal STI | ≤0.5% | ≤2% | ≤5% | >5% |
| Temporal LTI | ≤1% | ≤2% | ≤5% | >5% |

### Wavelength Bands

| Band | Range (nm) | Description |
|------|------------|-------------|
| 1 | 300-400 | UV-A |
| 2 | 400-500 | Blue |
| 3 | 500-600 | Green |
| 4 | 600-700 | Red |
| 5 | 700-800 | Near-IR 1 |
| 6 | 800-900 | Near-IR 2 |
| 7 | 900-1100 | Near-IR 3 |
| 8* | 1100-1200 | Near-IR 4 (Extended) |

*Extended range for bifacial and advanced module testing

## API Usage Examples

### Spectral Classification

```python
from utils.calculations import calculate_spectral_classification

result = calculate_spectral_classification(
    wavelengths=wavelength_array,
    irradiance=irradiance_array,
    reference_spectrum=am15g_reference
)

print(f"Overall Classification: {result['overall_class']}")
print(f"Band Results: {result['band_results']}")
```

### Capacitive Effects Analysis

```python
from utils.capacitive_effects import analyze_capacitive_effects

analysis = analyze_capacitive_effects(
    voltage=v_data,
    current=i_data,
    time=time_data,
    technology='PERC',
    cell_area_cm2=166*166/100,
    num_cells=60
)

print(f"Capacitance: {analysis['capacitance']['used_value_nF']:.1f} nF")
print(f"Recommended Sweep Time: {analysis['sweep_analysis']['recommended_sweep_time_ms']:.1f} ms")
```

### Angular Distribution

```python
from utils.angular_distribution import analyze_beam_collimation

result = analyze_beam_collimation(
    half_angle_deg=2.5,
    source_type='fresnel'
)

print(f"Beam Type: {result.beam_type}")
print(f"Meets IEC: {result.meets_iec_requirement}")
print(f"Quality: {result.collimation_quality}")
```

## Configuration Options

### Classification Settings

- **Standard Edition**: Ed.2 (2007) or Ed.3 (2020)
- **Extended Spectrum**: Enable 300-1200nm range
- **Auto-Classify**: Automatic classification after measurement

### Alarm Thresholds

Configure warning and critical thresholds for:
- Spectral match ratios
- Uniformity percentages
- STI/LTI instability values

### Display Settings

- Theme (Dark/Light)
- Chart height
- Decimal precision
- Date format

### System Settings

- Session timeout
- Max upload size
- Audit logging
- Data retention period

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IEC 60904-9:2020 Standard for Solar Simulator Classification
- IEC 60904-14 for Capacitive Effects Guidelines
- AM1.5G Reference Spectrum (IEC 60904-3:2019)

## Support

For issues and feature requests, please use the GitHub Issues page.

---

**Sun Simulator Classification System** - Professional PV Testing Solutions
