# SunSim-IEC60904-Classifier

Professional Sun Simulator Classification System | IEC 60904-9 Ed.3 Compliant

## Features

- **Spectral Match Analysis**: Classify spectral distribution against AM1.5G reference
- **Uniformity Analysis**: Evaluate spatial uniformity of irradiance
- **Temporal Stability Analysis**: Measure short-term and long-term instability
- **ISO 17025 Report Generation**: Generate compliant classification reports
- **SPC/MSA Quality Control**: Historical trend analysis and statistics

## Classification Criteria (IEC 60904-9 Ed.3)

| Characteristic | Class A | Class B | Class C |
|----------------|---------|---------|---------|
| Spectral Match | +/-25% | +/-40% | >40% |
| Non-uniformity | +/-2% | +/-5% | +/-10% |
| Temporal Instability (STI) | 0.5% | 2% | 10% |
| Temporal Instability (LTI) | 2% | 5% | 10% |

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-org/SunSim-IEC60904-Classifier.git
cd SunSim-IEC60904-Classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

**Note**: The application works without a database. Database features (history, saved reports) are optional.

## Deployment

### Railway Deployment

1. **Create a new project** on [Railway](https://railway.app)

2. **Connect your repository** or deploy directly:
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Configure environment variables** (optional, for database):
   - Go to your Railway project settings
   - Add a PostgreSQL database service (optional)
   - Railway automatically sets `DATABASE_URL` when you attach a database

4. **Environment Variables for Railway**:
   | Variable | Required | Description |
   |----------|----------|-------------|
   | `PORT` | Auto | Set automatically by Railway |
   | `DATABASE_URL` | No | Auto-set when PostgreSQL is attached |
   | `APP_NAME` | No | Application name (default: SunSim-IEC60904-Classifier) |
   | `DEBUG` | No | Enable debug mode (default: false) |

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect your repository
4. Configure secrets in Streamlit Cloud settings if using database

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Architecture

### Lazy-Loaded Database Connections

This application uses **lazy-loaded database connections** to prevent startup failures:

- Database connections are only established when actually needed
- Pages without database requirements work independently
- Application starts successfully even without database configuration
- Graceful error handling shows informative messages

```python
# Database is NOT connected at import time
from database import get_database_connection

# Connection only happens when explicitly requested
try:
    with get_database_connection() as conn:
        # Use connection
        pass
except DatabaseConnectionError as e:
    # Handle gracefully
    st.warning("Database unavailable")
```

### Project Structure

```
SunSim-IEC60904-Classifier/
├── app.py                    # Main Streamlit entry point
├── requirements.txt          # Python dependencies
├── Procfile                  # Railway/Heroku deployment
├── railway.json              # Railway configuration
├── nixpacks.toml             # Nixpacks build config
├── .env.example              # Environment template
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── database/
│   ├── __init__.py           # Lazy database module
│   └── connection.py         # Connection management
├── pages/
│   ├── 1_Spectral_Match.py   # Spectral analysis
│   ├── 2_Uniformity.py       # Uniformity analysis
│   ├── 3_Temporal_Stability.py # Temporal analysis
│   ├── 4_Reports.py          # Report generation
│   └── 5_History.py          # Historical data
└── utils/
    ├── __init__.py
    ├── config.py             # Configuration management
    └── calculations.py       # IEC 60904-9 calculations
```

## Troubleshooting

### 502 Errors on Deployment

If you encounter 502 errors:

1. **Check logs**: `railway logs` or check Railway dashboard
2. **Verify PORT**: Ensure the app binds to `$PORT` environment variable
3. **Database issues**: The app should work without database - check if database connection is blocking startup

### Database Connection Issues

The application is designed to work without a database:

- Analysis pages (Spectral, Uniformity, Temporal) work independently
- Report generation works without database
- Only History and saved results require database

To troubleshoot database issues:

```python
# Check database status
from database import get_connection_status
print(get_connection_status())
```

### Common Railway Issues

1. **PORT binding**: Ensure Procfile uses `$PORT`
2. **Health checks**: Application must respond to `/` within 300s
3. **Memory**: Default Railway containers have limited memory

## API Reference

### Database Module

```python
from database import (
    get_database_connection,  # Context manager for connections
    is_database_configured,   # Check if DB is configured
    get_connection_status,    # Get detailed status
    DatabaseConnectionError,  # Exception for connection errors
)
```

### Calculations Module

```python
from utils.calculations import (
    calculate_spectral_match_class,
    calculate_uniformity_class,
    calculate_temporal_stability_class,
    get_overall_classification,
    IEC_60904_9_LIMITS,
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (if available)
5. Submit a pull request

## License

See [LICENSE](LICENSE) file.

## Standards Reference

- IEC 60904-9 Ed.3: Solar simulator requirements for photovoltaic testing
- ISO 17025: General requirements for testing and calibration laboratories
