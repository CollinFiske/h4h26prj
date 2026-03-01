# CA Fire Detection & Personalized Evacuation

A web application for California wildfire detection/prediction and personalized evacuation route planning, powered by an XGBoost model trained on MODIS satellite fire detection data.

## Project Structure

```
project-root/
|-- frontend/                    # React (CRA) client application
|   |-- public/
|   |-- src/
|   |   |-- components/
|   |   |   |-- LocationDisplay.js    # Reads/stores location via cookies
|   |   |   |-- EvacuationForm.js     # User info form (disability, pets, etc.)
|   |   |   |-- ResultsDisplay.js     # Renders backend prediction results
|   |   |-- App.js
|   |   |-- App.css                   # All styles (plain CSS, no Tailwind)
|   |   |-- index.js
|   |-- package.json
|
|-- backend/                     # Flask API server
|   |-- routes/
|   |   |-- __init__.py
|   |   |-- predict.py           # /api/predict endpoint
|   |-- app.py                   # Flask entry point
|   |-- requirements.txt
|
|-- machine-learning-stuff/      # ML model files & notebooks
|   |-- model.py                 # Model wrapper (loads .joblib, runs inference)
|   |-- ca_wildfire_model.joblib # <-- YOU add this (from Jupyter)
|   |-- ca_wildfire_scaler.joblib# <-- YOU add this (from Jupyter)
|   |-- README_ML.txt
```

## Prerequisites

- **Node.js** >= 18 and **npm** (for the frontend)
- **Python** >= 3.9 (for the backend)
- (Optional) A Python virtual environment tool (`venv`, `conda`, etc.)

---

## IMPORTANT -- Windows PowerShell Fix

If you see this error when running `npm install` or `venv\Scripts\activate`:

```
cannot be loaded because running scripts is disabled on this system
```

Pick ONE of these options:

### Option A: Change the execution policy (recommended, one-time fix)

Open PowerShell **as Administrator** and run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` to confirm. Close and reopen your terminal.

### Option B: Use Command Prompt instead of PowerShell

Open **Command Prompt** (`cmd.exe`) instead. It does not have this restriction.

### Option C: Bypass for the current session only

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

---

## Setup & Run

### 1. Add Your Model Files

Your Jupyter notebook exports two `.joblib` files. Copy them into `machine-learning-stuff/`:

```
machine-learning-stuff/
|-- ca_wildfire_model.joblib    <-- copy here
|-- ca_wildfire_scaler.joblib   <-- copy here
|-- model.py                    (already set up to load them)
```

The backend will detect and load them automatically on startup. If the files aren't there, it will run in placeholder mode and tell you in the terminal.

### 2. Backend (Flask)

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# macOS / Linux:
source venv/bin/activate
# Windows PowerShell (after fixing execution policy above):
venv\Scripts\activate
# Windows Command Prompt:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Start the Flask dev server (runs on port 5000)
python app.py
```

The backend will be available at `http://localhost:5000`.
Verify it works: `http://localhost:5000/api/health`

On startup you should see:
```
[model.py] Loaded model from .../ca_wildfire_model.joblib
[model.py] Loaded scaler from .../ca_wildfire_scaler.joblib
```

If you see "WARNING: .joblib files not found", double-check that you copied them into `machine-learning-stuff/`.

### 3. Frontend (React)

Open a **second terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start the React dev server (runs on port 3000)
npm start
```

The React app will open at `http://localhost:3000`.
It proxies `/api/*` requests to the Flask server on port 5000 (configured in `frontend/package.json` via the `"proxy"` field).

### 4. Using Both Together

1. Start the backend first (Terminal 1).
2. Start the frontend second (Terminal 2).
3. Open `http://localhost:3000` in your browser.
4. Allow location access when prompted (must be a California location for real predictions).
5. Fill out the evacuation form and click "Get Evacuation Plan".
6. You'll see: fire risk level, probability percentage, location details (region, zone, fire season), evacuation route text, and accommodation notes.

---

## How the ML Integration Works

```
[React Frontend]
    |
    | POST /api/predict  { latitude, longitude, has_disability, ... }
    |
[Flask Backend]  (backend/routes/predict.py)
    |
    | from model import predict
    |
[model.py]  (machine-learning-stuff/model.py)
    |
    | joblib.load("ca_wildfire_model.joblib")
    | joblib.load("ca_wildfire_scaler.joblib")
    | -> builds feature vector (same as notebook)
    | -> scaler.transform()
    | -> model.predict_proba()
    |
    | Returns: { fire_risk, probability, details, evacuation_route, notes }
    v
[React Frontend]  -- displays results
```

The feature vector matches your Jupyter notebook exactly:
`[lat, lon, month_sin, month_cos, doy_sin, doy_cos, hour_sin, hour_cos, is_day, is_fire_season, is_norcal, is_inland, confidence]`

---

## Debugging Tips

- **Flask auto-reloads** on file changes because `debug=True` is set in `app.py`.
- **React hot-reloads** via Create React App's built-in dev server.
- Check the browser console (F12) for frontend errors.
- Check the terminal running Flask for backend tracebacks.
- If the proxy isn't working, make sure Flask is running on port 5000 before starting React.
- The `libretranslate` dependency conflict warnings during `pip install` are from an unrelated package and do NOT affect this project.

---

## Next Steps

- Replace the map placeholder with a real map (Leaflet, Google Maps, Mapbox, etc.).
- Add real-time satellite confidence data instead of the `0` placeholder.
- Add persistent storage (database) for user profiles and past evacuations.
- Expand evacuation routing with actual road/map data.
