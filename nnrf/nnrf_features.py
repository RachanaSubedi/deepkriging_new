"""
src/nnrf_features.py

Prepares the "global" (NSRDB) feature side of the NNRF spatiotemporal
downscaling stage, matching the feature set used in the reference paper
(Asiedu et al. 2025, "Spatiotemporal Downscaling Model for Solar
Irradiance Forecast Using Nearest-Neighbor Random Forest and Gaussian
Process") and its actual code: Temperature, Pressure, Wind Speed, Dew
Point, global GHI, plus cyclical Hour/DayOfYear encodings.

Two things this script produces:

  1. Global (NSRDB) feature matrices at all 178 PVs.
     Each PV is matched to its SINGLE NEAREST NSRDB grid point (not an
     IDW blend across all 140 — the paper uses each site's own raw
     NSRDB cell directly, so we mirror that exactly). Multiple PVs may
     share the same NSRDB point, same as the paper's Site2/Site7
     sharing one 4km cell.

  2. k=3 nearest-PV neighbor map.
     For each PV, its 3 nearest OTHER PVs (Euclidean distance in km),
     used by nnrf_downscale.py to pull in neighbor GHI_global features
     exactly as the paper's find_nearest_neighbors() does.

Run:
    python src/nnrf_features.py

Outputs (data/processed/nnrf_features/):
    nsrdb_temperature_pvs.parquet   (T, 178)
    nsrdb_pressure_pvs.parquet      (T, 178)
    nsrdb_windspeed_pvs.parquet     (T, 178)
    nsrdb_dewpoint_pvs.parquet      (T, 178)
    nsrdb_ghi_pvs.parquet           (T, 178)   global GHI (NOT the DK/IDW target)
    pv_nsrdb_map.csv                pv_name -> matched NSRDB point + distance
    pv_neighbors_k3.csv             pv_name -> 3 nearest PV neighbors + distances
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PROCESSED_DIR, LOCAL_TZ, KM_PER_LAT, KM_PER_LON

NSRDB_DIR   = Path(__file__).parent.parent / "data" / "raw" / "nsrdb"
PV_FILE     = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
OUT_DIR     = PROCESSED_DIR / "nnrf_features"

K_NEIGHBORS = 3

NSRDB_COLUMNS = {
    'Temperature'      : 'temperature',
    'Pressure'         : 'pressure',
    'Wind Speed'       : 'windspeed',
    'Dew Point'        : 'dewpoint',
    'GHI'              : 'ghi_global',
}

FNAME_RE = re.compile(r'^(-?\d+\.\d+)_(-?\d+\.\d+)_\d+\.csv$')


def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * KM_PER_LAT
    dlon = (lon1 - lon2) * KM_PER_LON
    return np.sqrt(dlat ** 2 + dlon ** 2)


def list_nsrdb_points():
    """Parse lat/lon out of each NSRDB filename. Returns DataFrame [file, lat, lon]."""
    rows = []
    for f in sorted(NSRDB_DIR.glob("*.csv")):
        m = FNAME_RE.match(f.name)
        if m:
            rows.append({'file': f, 'lat': float(m.group(1)), 'lon': float(m.group(2))})
    if not rows:
        raise FileNotFoundError(
            f"No NSRDB files matched the {{lat}}_{{lon}}_year.csv pattern in {NSRDB_DIR}"
        )
    return pd.DataFrame(rows)


def load_nsrdb_series(fpath):
    """Read one NSRDB CSV, return DataFrame indexed by tz-aware local datetime."""
    df = pd.read_csv(fpath, skiprows=2, low_memory=False)
    dt_utc = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], utc=True)
    df.index = dt_utc.dt.tz_convert(LOCAL_TZ)
    df.index.name = 'datetime_local'
    return df[list(NSRDB_COLUMNS.keys())].rename(columns=NSRDB_COLUMNS)


def match_pvs_to_nsrdb(pv_df, nsrdb_points):
    """For each PV, find the single nearest NSRDB point. Returns match DataFrame."""
    rows = []
    for _, pv in pv_df.iterrows():
        d = dist_km(pv['pv_lat'], pv['pv_lon'],
                    nsrdb_points['lat'].values, nsrdb_points['lon'].values)
        j = d.argmin()
        rows.append({
            'pv_name': pv['pv_name'],
            'nsrdb_lat': nsrdb_points.iloc[j]['lat'],
            'nsrdb_lon': nsrdb_points.iloc[j]['lon'],
            'nsrdb_file': str(nsrdb_points.iloc[j]['file']),
            'distance_km': round(float(d[j]), 3),
        })
    return pd.DataFrame(rows)


def compute_k_neighbors(pv_df, k=K_NEIGHBORS):
    """For each PV, find its k nearest OTHER PVs. Returns wide DataFrame."""
    names = pv_df['pv_name'].values
    lats  = pv_df['pv_lat'].values
    lons  = pv_df['pv_lon'].values

    rows = []
    for i, name in enumerate(names):
        d = dist_km(lats[i], lons[i], lats, lons)
        d[i] = np.inf  # exclude self
        nearest_idx = np.argsort(d)[:k]
        row = {'pv_name': name}
        for rank, idx in enumerate(nearest_idx, start=1):
            row[f'neighbor_{rank}']    = names[idx]
            row[f'neighbor_{rank}_km'] = round(float(d[idx]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  nnrf_features.py — NSRDB global features + k-NN map")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading PV locations and NSRDB point list...")
    pv_df = pd.read_csv(PV_FILE)
    nsrdb_points = list_nsrdb_points()
    print(f"  PVs           : {len(pv_df)}")
    print(f"  NSRDB points  : {len(nsrdb_points)}")

    print("\n[2/4] Matching each PV to its nearest NSRDB point...")
    pv_nsrdb_map = match_pvs_to_nsrdb(pv_df, nsrdb_points)
    pv_nsrdb_map.to_csv(OUT_DIR / "pv_nsrdb_map.csv", index=False)
    n_unique = pv_nsrdb_map['nsrdb_file'].nunique()
    print(f"  Matched. {n_unique} unique NSRDB points serve all {len(pv_df)} PVs "
          f"(some PVs share a cell, same as the paper's Site2/Site7).")
    print(f"  Distance to matched NSRDB point: "
          f"min={pv_nsrdb_map['distance_km'].min():.2f}  "
          f"max={pv_nsrdb_map['distance_km'].max():.2f}  "
          f"mean={pv_nsrdb_map['distance_km'].mean():.2f} km")

    print("\n[3/4] Computing k=3 nearest-PV neighbor map...")
    neighbors = compute_k_neighbors(pv_df, k=K_NEIGHBORS)
    neighbors.to_csv(OUT_DIR / "pv_neighbors_k3.csv", index=False)
    dist_cols = [c for c in neighbors.columns if c.endswith('_km')]
    all_dists = neighbors[dist_cols].values.flatten()
    print(f"  Neighbor distances: min={all_dists.min():.3f}  "
          f"max={all_dists.max():.3f}  mean={all_dists.mean():.3f} km")

    print("\n[4/4] Loading NSRDB series and building per-PV feature matrices...")
    # Cache: only load each unique NSRDB file once, even if many PVs share it
    cache = {}
    for f in pv_nsrdb_map['nsrdb_file'].unique():
        cache[f] = load_nsrdb_series(Path(f))
    print(f"  Loaded {len(cache)} unique NSRDB series")

    # Build (T, 178) matrix per feature
    feature_cols = list(NSRDB_COLUMNS.values())
    pv_names = pv_df['pv_name'].tolist()
    master_index = next(iter(cache.values())).index

    mats = {feat: pd.DataFrame(index=master_index, columns=pv_names, dtype=np.float32)
            for feat in feature_cols}

    for _, row in pv_nsrdb_map.iterrows():
        series = cache[row['nsrdb_file']]
        for feat in feature_cols:
            mats[feat][row['pv_name']] = series[feat].reindex(master_index).values

    for feat in feature_cols:
        out_path = OUT_DIR / f"nsrdb_{feat}_pvs.parquet"
        mats[feat].index.name = 'datetime_local'
        mats[feat].to_parquet(out_path)
        print(f"  ✓ nsrdb_{feat}_pvs.parquet   {mats[feat].shape}")

    print(f"\n✓ nnrf_features.py complete")
    print(f"  Output dir: {OUT_DIR}")