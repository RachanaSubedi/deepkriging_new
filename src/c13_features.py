"""
src/c13_features.py

Loads 33 GEE-extracted GOES-18 C13 pixel CSVs, resamples to 30-min,
computes BT features, and assigns to every PV and station location
via the pixel lookup table.

Prerequisites:
    - python src/pixel_mapping.py    (produces pv_pixel_map.csv)
    - GEE tasks complete + CSVs downloaded to data/raw/goes_c13/extracted_pixels/

Run:
    python src/c13_features.py

Outputs (data/processed/c13_features/):
    c13_bt_pixels.parquet     (T, 33)   BT at each pixel  — columns = pixel_id
    c13_bt_stations.parquet   (T, 4)    BT at each station pixel
    c13_bt_pvs.parquet        (T, 178)  BT at each PV's pixel
    c13_feat_stations.parquet (T, 12)   Engineered features for training
                                        MultiIndex columns: (station, feature)
    c13_feat_pvs.parquet      (T, 534)  Engineered features for inference
                                        MultiIndex columns: (pv_name, feature)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    C13_PIXEL_DIR, C13_FEAT_DIR,
    PROCESSED_DIR, LOCAL_TZ, STATIONS,
    GOES_SCALE_C13,
)

# ── CONSTANTS ─────────────────────────────────────────────────
BT_REF_K  = 270.0   # reference BT for normalisation
BT_SCALE_K = 50.0   # scale for normalisation


# ── STEP 1: LOAD ONE PIXEL CSV ────────────────────────────────
def load_pixel_csv(fpath):
    """
    Load a single GEE-exported C13 pixel CSV.

    GEE exports columns:
        system:index  datetime_utc  bt_c13_raw  pixel_id  .geo

    Returns a Series indexed by datetime_local (30-min resampled),
    with values = BT in Kelvin.
    """
    df = pd.read_csv(fpath, low_memory=False)

    # Drop GEE geometry column if present
    if '.geo' in df.columns:
        df = df.drop(columns=['.geo'])
    if 'system:index' in df.columns:
        df = df.drop(columns=['system:index'])

    # Parse UTC timestamp
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
    df = df.set_index('datetime_utc').sort_index()

    # Apply scale factor → Kelvin
    # Raw GEE values ~2900-3000 → × 0.1 → 290-300 K
    # Station-derived substitute already multiplied by 10, same logic applies
    raw_median = df['bt_c13_raw'].median()
    if raw_median > 400:
        df['bt_K'] = df['bt_c13_raw'] * GOES_SCALE_C13
    else:
        df['bt_K'] = df['bt_c13_raw'].astype(np.float32)

    # Get pixel_id
    pixel_id = (df['pixel_id'].dropna().iloc[0]
                if 'pixel_id' in df.columns else fpath.stem)

    # Resample to 30-min by averaging (GOES scans every ~5-10 min)
    bt_30min = df['bt_K'].resample('30T').mean()

    # Convert index to local time
    bt_30min.index = bt_30min.index.tz_convert(LOCAL_TZ)
    bt_30min.name  = pixel_id

    return bt_30min, pixel_id


# ── STEP 2: LOAD ALL 33 PIXEL CSVS ───────────────────────────
def load_all_pixels(pixel_dir):
    """
    Load all pixel CSVs from pixel_dir.
    Returns DataFrame (T, 33) indexed by datetime_local.
    """
    files = sorted(pixel_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No pixel CSVs found in {pixel_dir}\n"
            f"Download GEE outputs first and place them there."
        )
    print(f"  Found {len(files)} pixel CSV files")

    series_list = []
    for f in files:
        bt_series, pid = load_pixel_csv(f)
        series_list.append(bt_series)
        print(f"    {pid:45s}  {len(bt_series)} rows  "
              f"BT=[{bt_series.min():.1f}, {bt_series.max():.1f}] K")

    df = pd.concat(series_list, axis=1)
    df.index.name = 'datetime_local'
    return df


# ── STEP 3: COMPUTE BT FEATURES ──────────────────────────────
def compute_bt_features(bt_series):
    """
    Compute 3 BT features for one pixel/station time series:

        bt_norm  : (BT - 270) / 50
                   Normalised brightness temperature.
                   High bt_norm → warm → clear sky → high GHI.
                   Low bt_norm → cold cloud tops → thick cloud → low GHI.

        bt_lag30 : bt_norm at t-30 min
                   Captures cloud persistence — if cloudy 30 min ago,
                   likely still cloudy now.

        bt_diff  : bt_norm(t) − bt_norm(t-1)
                   Captures cloud trend direction.
                   Positive → sky clearing. Negative → cloud building.
                   More physically useful than a slow 7-day anomaly.

    Parameters
    ----------
    bt_series : pd.Series  raw BT in Kelvin, 30-min index

    Returns
    -------
    pd.DataFrame  with columns [bt_norm, bt_lag30, bt_diff]
    """
    bt = bt_series.copy()

    # Fill small gaps (up to 2 missing steps = 1 hour)
    bt = bt.interpolate(method='time', limit=2)

    bt_norm = (bt - BT_REF_K) / BT_SCALE_K
    bt_lag30 = bt_norm.shift(1)
    bt_diff = bt_norm - bt_lag30  # bt_norm(t) - bt_norm(t-30min)
    bt_diff60 = bt_norm - bt_norm.shift(2)  # bt_norm(t) - bt_norm(t-60min)

    return pd.DataFrame({
        'bt_norm': bt_norm,
        'bt_lag30': bt_lag30,
        'bt_diff': bt_diff,
        'bt_diff60': bt_diff60,
    })


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  c13_features.py — GOES-18 C13 Feature Engineering")
    print("=" * 60)

    # ── 1. Load pixel map ──────────────────────────────────────
    print("\n[1/5] Loading pixel map...")
    pv_map = pd.read_csv(PROCESSED_DIR / "pv_pixel_map.csv")

    station_map = pv_map[pv_map['pv_name'].str.startswith('STATION_')].copy()
    pv_map_only = pv_map[~pv_map['pv_name'].str.startswith('STATION_')].copy()
    station_map['station'] = station_map['pv_name'].str.replace('STATION_', '')

    print(f"  PV locations    : {len(pv_map_only)}")
    print(f"  Station entries : {len(station_map)}")

    # ── 2. Load all pixel BT time series ──────────────────────
    print("\n[2/5] Loading pixel CSVs...")
    bt_pixels = load_all_pixels(C13_PIXEL_DIR)
    print(f"\n  Pixel BT matrix : {bt_pixels.shape}")
    print(f"  Time range      : {bt_pixels.index[0]}  →  {bt_pixels.index[-1]}")
    vals = bt_pixels.values[~np.isnan(bt_pixels.values)]
    print(f"  BT range overall: [{vals.min():.1f},  {vals.max():.1f}] K")

    # ── 3. Assign BT to stations ───────────────────────────────
    print("\n[3/5] Assigning BT to station and PV locations...")
    station_names = list(STATIONS.keys())
    bt_st_dict = {}
    for _, row in station_map.iterrows():
        sname = row['station']
        pid   = row['pixel_id']
        if pid in bt_pixels.columns:
            bt_st_dict[sname] = bt_pixels[pid]
        else:
            print(f"  ⚠ Station {sname}: pixel {pid} not found")

    bt_stations = pd.DataFrame(bt_st_dict)[station_names]
    bt_stations.index.name = 'datetime_local'
    print(f"  bt_stations shape: {bt_stations.shape}")
    for s in station_names:
        v = bt_stations[s].dropna()
        print(f"    {s}: BT mean={v.mean():.1f}K  "
              f"min={v.min():.1f}K  max={v.max():.1f}K")

    # ── 4. Assign BT to PVs ────────────────────────────────────
    pv_names = pv_map_only['pv_name'].tolist()
    bt_pv_cols = {}
    for _, row in pv_map_only.iterrows():
        pid = row['pixel_id']
        if pid in bt_pixels.columns:
            bt_pv_cols[row['pv_name']] = bt_pixels[pid]
        else:
            print(f"  ⚠ PV {row['pv_name']}: pixel {pid} not found")

    bt_pvs = pd.DataFrame(bt_pv_cols)[pv_names]
    bt_pvs.index.name = 'datetime_local'
    print(f"  bt_pvs shape     : {bt_pvs.shape}")

    # ── 5. Save raw BT parquets ────────────────────────────────
    C13_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    bt_pixels.to_parquet(C13_FEAT_DIR   / "c13_bt_pixels.parquet")
    bt_stations.to_parquet(C13_FEAT_DIR / "c13_bt_stations.parquet")
    bt_pvs.to_parquet(C13_FEAT_DIR      / "c13_bt_pvs.parquet")
    print(f"\n  ✓ c13_bt_pixels.parquet    {bt_pixels.shape}")
    print(f"  ✓ c13_bt_stations.parquet  {bt_stations.shape}")
    print(f"  ✓ c13_bt_pvs.parquet       {bt_pvs.shape}")

    # ── 6. Compute engineered features ────────────────────────
    print("\n[4/5] Computing engineered BT features...")

    # Stations: 3 features × 4 stations
    feat_st_list = {}
    for s in station_names:
        feat_st_list[s] = compute_bt_features(bt_stations[s])
    feat_stations = pd.concat(feat_st_list, axis=1)
    feat_stations.index.name = 'datetime_local'

    # PVs: compute once per unique pixel, assign to all PVs in that pixel
    unique_pixels = pv_map_only['pixel_id'].unique()
    pixel_feats = {}
    for pid in unique_pixels:
        if pid in bt_pixels.columns:
            pixel_feats[pid] = compute_bt_features(bt_pixels[pid])

    feat_pv_list = {}
    for _, row in pv_map_only.iterrows():
        pid = row['pixel_id']
        if pid in pixel_feats:
            feat_pv_list[row['pv_name']] = pixel_feats[pid]

    feat_pvs = pd.concat(feat_pv_list, axis=1)
    feat_pvs.index.name = 'datetime_local'

    feat_stations.to_parquet(C13_FEAT_DIR / "c13_feat_stations.parquet")
    feat_pvs.to_parquet(C13_FEAT_DIR      / "c13_feat_pvs.parquet")
    print(f"  ✓ c13_feat_stations.parquet  {feat_stations.shape}")
    print(f"  ✓ c13_feat_pvs.parquet       {feat_pvs.shape}")

    # ── 7. Sanity check ────────────────────────────────────────
    print("\n[5/5] Sanity check — BT features at S1 (daytime only):")
    s1_feats = feat_stations['S1'].dropna()
    for col in s1_feats.columns:
        print(f"  {col:12s}  mean={s1_feats[col].mean():+.4f}  "
              f"std={s1_feats[col].std():.4f}  "
              f"range=[{s1_feats[col].min():.3f}, {s1_feats[col].max():.3f}]")

    print(f"\n✓ c13_features.py complete")
    print(f"  Features per location : bt_norm, bt_lag30, bt_diff")
    print(f"  Output dir: {C13_FEAT_DIR}")