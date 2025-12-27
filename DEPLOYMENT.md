# Railway Deployment Guide for SunSim-IEC60904-Classifier

## Prerequisites

- A Railway account (https://railway.app)
- Git repository connected to Railway

## Deployment Steps

### 1. Create a New Project on Railway

1. Log in to Railway (https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose the `SunSim-IEC60904-Classifier` repository
5. Select the branch `claude/deploy-railway-postgres-oTIHL`

### 2. Add PostgreSQL Database

1. In your Railway project, click "New"
2. Select "Database" > "Add PostgreSQL"
3. Railway will automatically provision a PostgreSQL instance
4. The `DATABASE_URL` variable is automatically added to your app

### 3. Connect Database to App

1. Click on your web service
2. Go to "Variables" tab
3. Click "Add Variable Reference"
4. Select `DATABASE_URL` from the PostgreSQL service
5. This links the database connection string to your app

### 4. Initialize the Database

After deployment, run the database initialization:

**Option A: Via Railway CLI**
```bash
railway run python init_db.py
```

**Option B: Via Railway Shell**
1. In Railway dashboard, click your web service
2. Go to "Shell" tab
3. Run: `python init_db.py`

### 5. Verify Deployment

1. Click on your web service in Railway
2. Find the deployment URL (e.g., `your-app.up.railway.app`)
3. Access the application in your browser
4. Test all features:
   - SPC Analysis (sidebar menu)
   - MSA Gage R&R (sidebar menu)
   - Capability Index (sidebar menu)

## Environment Variables

| Variable | Description | Auto-configured |
|----------|-------------|-----------------|
| `DATABASE_URL` | PostgreSQL connection string | Yes (from PostgreSQL plugin) |
| `PORT` | Application port | Yes (by Railway) |

## Files Configuration

| File | Purpose |
|------|---------|
| `runtime.txt` | Python version (3.11.6) |
| `Procfile` | Start command for Streamlit |
| `requirements.txt` | Python dependencies |
| `railway.json` | Railway-specific settings |
| `nixpacks.toml` | Nixpacks build configuration |
| `init_db.py` | Database initialization script |

## Database Schema

The application creates 4 tables:

1. **spc_data** - SPC control chart measurements
2. **msa_studies** - Gage R&R study data
3. **simulators** - Simulator reference data
4. **capability_history** - Process capability records

## Troubleshooting

### Database Connection Issues
- Verify `DATABASE_URL` is set in Variables
- Check that PostgreSQL service is running
- Run `python init_db.py` to recreate tables

### Application Not Starting
- Check build logs for dependency errors
- Verify `Procfile` format is correct
- Ensure PORT is available

### Missing Tables
- Run `python init_db.py` via Railway shell
- Check database logs for errors

## Local Development

For local development, the app uses SQLite automatically:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (uses SQLite)
streamlit run app.py
```

To test with PostgreSQL locally:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/sunsim"
streamlit run app.py
```

## Support

For issues, check:
1. Railway deployment logs
2. Application logs in Railway dashboard
3. Database connection status
