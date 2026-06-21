# configs/config.py

from pathlib import Path

# ── PROJECT ROOT ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── DATA PATHS ────────────────────────────────────────────────
NSRDB_DIR     = ROOT / "data" / "raw" / "nsrdb"
STATION_DIR   = ROOT / "data" / "raw" / "stations"
C13_PIXEL_DIR = ROOT / "data" / "raw" / "goes_c13" / "extracted_pixels"
C13_STAT_DIR  = ROOT / "data" / "raw" / "goes_c13" / "extracted_stations"

PROCESSED_DIR = ROOT / "data" / "processed"
BG_DIR        = PROCESSED_DIR / "background_field"
RESID_DIR     = PROCESSED_DIR / "residuals"
BASIS_DIR     = PROCESSED_DIR / "basis_matrix"
C13_FEAT_DIR  = PROCESSED_DIR / "c13_features"
TRAIN_DIR     = PROCESSED_DIR / "training_matrix"

OUTPUT_DIR    = ROOT / "outputs"
MODEL_DIR     = OUTPUT_DIR / "models"
PRED_DIR      = OUTPUT_DIR / "predictions"
VAL_DIR       = OUTPUT_DIR / "validation"
FIG_DIR       = OUTPUT_DIR / "figures"

# ── DOMAIN ────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 46.56, 46.82
LON_MIN, LON_MAX = -119.29, -119.05
KM_PER_LAT       = 111.0
KM_PER_LON        = 75.8   # at 46.7°N

# ── STATIONS ──────────────────────────────────────────────────
STATIONS = {
    'S1': {'lat': 46.59,  'lon': -119.150},
    'S2': {'lat': 46.82,  'lon': -119.160},
    'S3': {'lat': 46.82,  'lon': -119.150},
    'P2': {'lat': 46.78,  'lon': -119.228},
}

# ── NSRDB GRID ────────────────────────────────────────────────
NSRDB_LAT_MIN  = 46.56
NSRDB_LAT_MAX  = 46.82
NSRDB_LON_MIN  = -119.29
NSRDB_LON_MAX  = -119.05
NSRDB_RES      = 0.02      # degrees
NSRDB_SKIPROWS = 2

# ── BASIS FUNCTIONS ───────────────────────────────────────────
BASIS_LEVELS = [
    {'spacing_km': 10.0, 'theta_km': 25.0},
    {'spacing_km':  5.0, 'theta_km': 12.5},
    {'spacing_km':  2.5, 'theta_km':  6.25},
    {'spacing_km':  1.5, 'theta_km':  3.75},
]

# ── GOES-18 ───────────────────────────────────────────────────
GOES_RES_LAT   = 2.0 / KM_PER_LAT   # ~0.018°
GOES_RES_LON   = 2.0 / KM_PER_LON   # ~0.026°
GOES_SCALE_C13 = 0.1
GOES_BUFFER_M  = 1000
GEE_PROJECT    = "rachanaieee9500"
GEE_DATASET    = "NOAA/GOES/18/MCMIPC"

# ── MODEL ─────────────────────────────────────────────────────
N_COVARIATES   = 18
HIDDEN_SIZE    = 100
DROPOUT        = 0.5
WEIGHT_DECAY   = 1e-4
LEARNING_RATE  = 1e-3
BATCH_SIZE     = 512
MAX_EPOCHS     = 200
STATION_WEIGHT = 5.0

# ── TIMEZONE ──────────────────────────────────────────────────
LOCAL_TZ       = "America/Los_Angeles"