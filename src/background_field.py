"""
src/background_field.py

Computes IDW-interpolated background CSI field from 182 NSRDB points
at all station and PV locations.

NSRDB filename pattern: {lat}_{lon}_{year}.csv
  e.g.  46.56_-119.05_2024.csv
        46.82_-119.29_2024.csv

Run:
    python src/background_field.py

Outputs (data/processed/background_field/):
    bg_csi_stations.parquet   shape (T, 4)   columns = S1 S2 S3 P2
    bg_csi_pvs.parquet        shape (T, 178) columns = pv_name
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    NSRDB_DIR, NSRDB_SKIPROWS, LOCAL_TZ,
    STATIONS, BG_DIR,
    KM_PER_LAT, KM_PER_LON,
)

# ── CONSTANTS ────────────────────────────────────────────────
IDW_POWER             = 2.0
CLEARSKY_NIGHT_W_M2   = 10.0   # clearsky GHI below this → nighttime → CSI = 0
CSI_CLIP_MAX          = 2.0    # clamp CSI upper bound (cloud-enhancement edge)


# ── STEP 1: PARSE LAT/LON FROM FILENAME ──────────────────────
def parse_lat_lon(filepath):
    """
    Extract lat, lon from filename like  46.56_-119.05_2024.csv
    stem.split('_') → ['46.56', '-119.05', '2024']
    """
    parts = filepath.stem.split('_')
    # parts[0] = lat, parts[1] = lon (negative, e.g. -119.05)
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


# ── STEP 2: LOAD ALL NSRDB FILES ─────────────────────────────
def load_nsrdb(nsrdb_dir):
    """
    Load all NSRDB CSVs from nsrdb_dir.

    Returns
    -------
    nsrdb_locs  : np.array  (N, 2)  [lat, lon] for each file
    ghi         : np.array  (N, T)  GHI values
    ghi_clear   : np.array  (N, T)  Clearsky GHI values
    timestamps  : pd.DatetimeIndex  local time, length T
    """
    files = sorted(nsrdb_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NSRDB CSVs found in {nsrdb_dir}")
    print(f"  Found {len(files)} NSRDB files")

    lats, lons       = [], []
    ghi_list         = []
    ghi_clear_list   = []
    timestamps       = None

    for f in files:
        lat, lon = parse_lat_lon(f)
        lats.append(lat)
        lons.append(lon)

        df = pd.read_csv(f, skiprows=NSRDB_SKIPROWS, low_memory=False)

        # Build timestamps once (all files share the same time axis)
        if timestamps is None:
            dt_utc = pd.to_datetime({
                'year':   df['Year'].astype(int),
                'month':  df['Month'].astype(int),
                'day':    df['Day'].astype(int),
                'hour':   df['Hour'].astype(int),
                'minute': df['Minute'].astype(int),
            }, utc=True)
            timestamps = pd.DatetimeIndex(dt_utc.dt.tz_convert(LOCAL_TZ))

        ghi_list.append(df['GHI'].values.astype(np.float32))
        ghi_clear_list.append(df['Clearsky GHI'].values.astype(np.float32))

    nsrdb_locs = np.column_stack([lats, lons])          # (N, 2)
    ghi        = np.stack(ghi_list,       axis=0)        # (N, T)
    ghi_clear  = np.stack(ghi_clear_list, axis=0)        # (N, T)

    return nsrdb_locs, ghi, ghi_clear, timestamps


# ── STEP 3: COMPUTE CSI ───────────────────────────────────────
def compute_csi(ghi, ghi_clear):
    """
    CSI = GHI / Clearsky_GHI

    Nighttime rule : clearsky GHI < CLEARSKY_NIGHT_W_M2  → CSI = 0.0
    Clip           : [0.0, CSI_CLIP_MAX]

    Parameters
    ----------
    ghi       : (N, T)
    ghi_clear : (N, T)

    Returns
    -------
    csi       : (N, T)  float32
    """
    csi       = np.zeros_like(ghi, dtype=np.float32)
    day_mask  = ghi_clear >= CLEARSKY_NIGHT_W_M2
    csi[day_mask] = ghi[day_mask] / ghi_clear[day_mask]
    np.clip(csi, 0.0, CSI_CLIP_MAX, out=csi)
    return csi


# ── STEP 4: PRECOMPUTE IDW WEIGHTS ───────────────────────────
def idw_weights(nsrdb_locs, target_locs, power=IDW_POWER):
    """
    Precompute IDW weight matrix.

    W[m, n] = weight of NSRDB point n for target location m
    Each row sums to 1.

    Parameters
    ----------
    nsrdb_locs  : (N, 2)
    target_locs : (M, 2)

    Returns
    -------
    W : (M, N)  float64
    """
    M  = len(target_locs)
    N  = len(nsrdb_locs)
    W  = np.zeros((M, N), dtype=np.float64)

    for m, (tlat, tlon) in enumerate(target_locs):
        dlat_km  = (nsrdb_locs[:, 0] - tlat) * KM_PER_LAT
        dlon_km  = (nsrdb_locs[:, 1] - tlon) * KM_PER_LON
        dist_km  = np.sqrt(dlat_km**2 + dlon_km**2)

        # Exact coincidence: target sits on an NSRDB grid point
        if dist_km.min() < 0.001:
            W[m, dist_km.argmin()] = 1.0
        else:
            w    = 1.0 / (dist_km ** power)
            W[m] = w / w.sum()

    return W   # (M, N)


# ── STEP 5: APPLY IDW ────────────────────────────────────────
def apply_idw(W, csi):
    """
    Matrix multiply to get background CSI at target locations.

    Parameters
    ----------
    W   : (M, N)  precomputed weights
    csi : (N, T)  CSI at NSRDB points

    Returns
    -------
    bg_csi : (M, T)  background CSI at targets
    """
    return (W @ csi).astype(np.float32)   # (M, N) @ (N, T)


# ── STEP 6: SANITY CHECK ─────────────────────────────────────
def sanity_check(df, label):
    print(f"\n── Sanity Check: {label} ──────────────────────")
    for col in df.columns:
        day_vals = df[col][df[col] > 0.01]
        if len(day_vals) == 0:
            print(f"  {col}: ALL NIGHT — check timezone or data")
        else:
            print(f"  {col}:  daytime mean={day_vals.mean():.3f}"
                  f"  max={day_vals.max():.3f}"
                  f"  daytime fraction={len(day_vals)/len(df[col]):.1%}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  background_field.py — IDW NSRDB CSI Interpolation")
    print("=" * 55)

    # ── 1. Load NSRDB ────────────────────────────────────────
    print("\n[1/5] Loading NSRDB files...")
    nsrdb_locs, ghi, ghi_clear, timestamps = load_nsrdb(NSRDB_DIR)
    print(f"      NSRDB grid points : {nsrdb_locs.shape[0]}")
    print(f"      Timesteps         : {ghi.shape[1]}")
    print(f"      Time range        : {timestamps[0]}  →  {timestamps[-1]}")

    # ── 2. Compute NSRDB CSI ─────────────────────────────────
    print("\n[2/5] Computing NSRDB CSI...")
    csi_nsrdb = compute_csi(ghi, ghi_clear)
    night_frac = (ghi_clear < CLEARSKY_NIGHT_W_M2).mean()
    print(f"      CSI range         : [{csi_nsrdb.min():.3f},  {csi_nsrdb.max():.3f}]")
    print(f"      Nighttime steps   : {night_frac:.1%}")

    # ── 3. Load target locations ─────────────────────────────
    print("\n[3/5] Loading target locations...")

    station_names = list(STATIONS.keys())
    station_locs  = np.array([[v['lat'], v['lon']]
                               for v in STATIONS.values()])
    print(f"      Stations          : {len(station_names)}  {station_names}")

    pv_path = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df   = pd.read_csv(pv_path)
    pv_locs  = pv_df[['pv_lat', 'pv_lon']].values
    pv_names = pv_df['pv_name'].tolist()
    print(f"      PV locations      : {len(pv_names)}")

    # ── 4. IDW weights ───────────────────────────────────────
    print("\n[4/5] Computing IDW weights...")
    W_stations = idw_weights(nsrdb_locs, station_locs)
    W_pvs      = idw_weights(nsrdb_locs, pv_locs)
    print(f"      W_stations        : {W_stations.shape}")
    print(f"      W_pvs             : {W_pvs.shape}")
    print(f"      Row-sum check     : stations={W_stations.sum(axis=1).min():.6f}"
          f"  pvs={W_pvs.sum(axis=1).min():.6f}  (should be 1.0)")

    # ── 5. Apply IDW + save ──────────────────────────────────
    print("\n[5/5] Applying IDW and saving...")

    bg_stations    = apply_idw(W_stations, csi_nsrdb)   # (4,   T)
    bg_pvs         = apply_idw(W_pvs,      csi_nsrdb)   # (178, T)
    bg_clearsky_st = apply_idw(W_stations, ghi_clear)   # (4,   T)  W/m²

    BG_DIR.mkdir(parents=True, exist_ok=True)

    # Stations DataFrame: (T, 4)
    df_stations = pd.DataFrame(
        bg_stations.T,
        index=timestamps,
        columns=station_names,
    )
    df_stations.index.name = 'datetime_local'
    df_stations.to_parquet(BG_DIR / "bg_csi_stations.parquet")

    # PV DataFrame: (T, 178)
    df_pvs = pd.DataFrame(
        bg_pvs.T,
        index=timestamps,
        columns=pv_names,
    )
    df_pvs.index.name = 'datetime_local'
    df_pvs.to_parquet(BG_DIR / "bg_csi_pvs.parquet")

    # Clearsky GHI at stations: (T, 4)
    df_clearsky = pd.DataFrame(
        bg_clearsky_st.T,
        index=timestamps,
        columns=station_names,
    )
    df_clearsky.index.name = 'datetime_local'
    df_clearsky.to_parquet(BG_DIR / "bg_clearsky_stations.parquet")

    print(f"      ✓ bg_csi_stations.parquet      {df_stations.shape}")
    print(f"      ✓ bg_csi_pvs.parquet           {df_pvs.shape}")
    print(f"      ✓ bg_clearsky_stations.parquet {df_clearsky.shape}")

    sanity_check(df_stations, "Station background CSI")

    print("\n✓ background_field.py complete")
    print(f"  Output dir: {BG_DIR}")