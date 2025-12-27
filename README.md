# SunSim-IEC60904-Classifier

Professional Sun Simulator Classification System | IEC 60904-9 Ed.3 Compliant | Spectral Match, Uniformity & Temporal Stability Analysis | ISO 17025 Report Generation | SPC/MSA Quality Control

## Features

- **SPC Analysis** - Statistical Process Control with X-bar & R charts, run rules detection
- **MSA Gage R&R** - Measurement System Analysis per AIAG MSA manual
- **Capability Index** - Process capability analysis with Cp, Cpk, Pp, Ppk gauges

## Quick Start (Local Development)

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/SunSim-IEC60904-Classifier.git
cd SunSim-IEC60904-Classifier

# Install dependencies
pip install -r requirements.txt

# Run the application (uses SQLite locally)
streamlit run app.py
```

## Deployment

### Streamlit Cloud (Recommended)

1. **Fork/Push to GitHub**
   - Ensure your repository is on GitHub

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository and branch `claude/deploy-railway-postgres-oTIHL`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Database (Railway PostgreSQL)**
   - In Streamlit Cloud dashboard, go to your app settings
   - Click "Secrets" in the left sidebar
   - Add your database configuration:

   ```toml
   [database]
   DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@YOUR_HOST.railway.app:5432/railway"
   ```

4. **Get Railway PostgreSQL Credentials**
   - Go to [Railway Dashboard](https://railway.app)
   - Create a new project with PostgreSQL database
   - Go to PostgreSQL service > Variables tab
   - Copy the `DATABASE_URL` value

### Railway PostgreSQL Setup

1. **Create PostgreSQL Database**
   - Log in to [Railway](https://railway.app)
   - Click "New Project" > "Provision PostgreSQL"
   - Wait for database to provision

2. **Get Connection String**
   - Click on PostgreSQL service
   - Go to "Variables" tab
   - Copy `DATABASE_URL`

3. **Initialize Database Tables**
   - Connect to database using Railway CLI or any PostgreSQL client
   - The app auto-creates tables on first connection

## Environment Variables

| Variable | Description | Where to Set |
|----------|-------------|--------------|
| `DATABASE_URL` | PostgreSQL connection string | Streamlit Cloud Secrets or Environment |

## Database Schema

The application uses 4 tables:

- **spc_data** - SPC control chart measurements
- **msa_studies** - Gage R&R study data
- **simulators** - Simulator reference data
- **capability_history** - Process capability records

## Tech Stack

- **Frontend**: Streamlit
- **Database**: PostgreSQL (production) / SQLite (development)
- **Charts**: Plotly
- **Data**: Pandas, NumPy, SciPy

## Project Structure

```
SunSim-IEC60904-Classifier/
├── app.py                    # Main Streamlit application
├── pages/
│   ├── 5_📈_SPC_Analysis.py  # SPC control charts
│   ├── 6_🔬_MSA_Gage_RR.py   # Measurement System Analysis
│   └── 7_🎯_Capability_Index.py  # Process capability
├── utils/
│   ├── db.py                 # Database operations
│   ├── spc_calculations.py   # SPC algorithms
│   └── msa_calculations.py   # MSA algorithms
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml.example  # Secrets template
├── requirements.txt          # Python dependencies
└── README.md
```

## Standards Compliance

- **IEC 60904-9 Ed.3** - Solar simulator classification
- **ISO 22514** - Statistical Process Control methods
- **AIAG MSA 4th Ed.** - Measurement System Analysis

## License

MIT License
