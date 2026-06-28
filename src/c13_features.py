"""
src/c13_features.py

Loads GEE-extracted GOES-18 C13+C02 pixel CSVs, resamples to a 5-min
target grid (forward-filled from the native ~10-min GOES scan
cadence — see load_pixel_csv for details), computes BT and C02
reflectance features, and assigns to every PV and station location
via the pixel lookup table.

C13 = 10.3 um clean longwave IR   -> cloud-top temperature signal
C02 = 0.64 um visible reflectance -> cloud brightness/reflectance,
      physically distinct from C13. Null at night and on some scan
      gaps — this is expected (see compute_bt_features). Older
      C13-only pixel CSVs (no refl_c02_raw column) are handled
      gracefully — c02_norm will just be all-NaN for those pixels.

Prerequisites:
    - python src/pixel_mapping.py    (produces pv_pixel_map.csv)
    - GEE tasks complete + CSVs downloaded to data/raw/goes_c13/extracted_pixels/

Run:
    python src/c13_features.py

Outputs (data/processed/c13_features/):
    c13_bt_pixels.parquet      (T, n)   BT at each pixel  — columns = pixel_id
    c13_bt_stations.parquet    (T, 4)   BT at each station pixel
    c13_bt_pvs.parquet         (T, 178) BT at each PV's pixel
    c02_refl_pixels.parquet    (T, n)   C02 reflectance at each pixel
    c02_refl_stations.parquet  (T, 4)   C02 reflectance at each station pixel
    c02_refl_pvs.parquet       (T, 178) C02 reflectance at each PV's pixel
    c13_feat_stations.parquet  (T, 8)   Engineered features for training
                                        MultiIndex columns: (station, feature)
                                        Columns: bt_norm, c02_norm
    c13_feat_pvs.parquet       (T, 356) Engineered features for inference
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
BT_REF_K   = 270.0   # reference BT for normalisation
BT_SCALE_K = 50.0    # scale for normalisation

# Actual study period start (local time). Extractions deliberately
# start a few days earlier (UTC) to give margin around the
# Dec-31-local/Jan-1-UTC boundary — see the trim step in __main__.
STUDY_START_DATE = "2024-01-01"

# C02 (visible/near-IR reflectance) is a reflectance FACTOR, typically
# 0-1 over land/water, occasionally slightly >1 over bright cloud tops
# or snow due to calibration. No existing normalisation convention to
# match (this is a new feature) — center at 0.5, scale by 0.5, so the
# typical 0-1 range maps to roughly [-1, +1], same spirit as bt_norm.
REFL_REF   = 0.5
REFL_SCALE = 0.5


# ── STEP 1: LOAD ONE PIXEL CSV ────────────────────────────────
def load_pixel_csv(fpath):
    """
    Load a single GEE-exported C13+C02 pixel CSV.

    GEE exports columns:
        system:index  datetime_utc  bt_c13_raw  refl_c02_raw  pixel_id  .geo

    Returns (bt_series, refl_series, pixel_id), each a Series indexed
    by datetime_local (5-min, forward-filled from the native ~10-min
    GOES scan cadence). bt_series is BT in Kelvin; refl_series is raw
    C02 reflectance factor (NOT yet normalised — that happens in
    compute_bt_features / a parallel C02 feature step).

    refl_c02_raw will be null at night (no sunlight to reflect) and
    may have scattered nulls from satellite scan gaps, same as C13 —
    this is expected, not a bug. If the CSV has no refl_c02_raw
    column at all (older C13-only exports), refl_series is returned
    as all-NaN so downstream code doesn't need a separate code path.
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

    # C02 reflectance — GOES CMI_C02 comes back from GEE as a SCALED
    # value, not a calibrated reflectance factor directly (same issue
    # as C13's raw ~2900-3000 values needing × GOES_SCALE_C13).
    #
    # IMPORTANT: unlike GOES_SCALE_C13 (an established constant already
    # in this codebase), there is currently NO verified GOES-18 C02
    # scale constant here. The line below derives an empirical scale
    # from the 99th percentile of THIS pixel's own raw values, on the
    # assumption that near-maximum brightness (thick cloud/snow) should
    # land close to a reflectance factor of ~1.0. This is a heuristic,
    # not a verified calibration — sanity-check c02_norm against a
    # known clear vs. cloudy day before trusting it in training.
    if 'refl_c02_raw' in df.columns:
        refl_raw = df['refl_c02_raw'].astype(np.float32)
        refl_p99 = refl_raw.quantile(0.99)
        if refl_p99 > 5:  # raw values are clearly NOT already a 0-1 factor
            c02_scale = 1.0 / refl_p99 if refl_p99 > 0 else 1.0
            df['refl_c02'] = refl_raw * c02_scale
        else:
            df['refl_c02'] = refl_raw  # already looks like a 0-1 factor
    else:
        df['refl_c02'] = np.float32(np.nan)

    # Get pixel_id
    pixel_id = (df['pixel_id'].dropna().iloc[0]
                if 'pixel_id' in df.columns else fpath.stem)

    # Resample to 5-min target grid via forward-fill. GOES-18 MCMIPC
    # scans every ~10 min, NOT natively every 5 min — every other 5-min
    # slot will repeat the most recent scan rather than reflect a fresh
    # observation. This is a deliberate design choice (repeat rather than
    # interpolate/average) to match the 5-min station/NSRDB grid without
    # inventing values between real satellite observations.
    bt_5min = df['bt_K'].dropna().resample('5min').ffill()
    refl_5min = df['refl_c02'].dropna().resample('5min').ffill()

    # Convert index to local time
    bt_5min.index   = bt_5min.index.tz_convert(LOCAL_TZ)
    refl_5min.index = refl_5min.index.tz_convert(LOCAL_TZ)
    bt_5min.name    = pixel_id
    refl_5min.name  = pixel_id

    return bt_5min, refl_5min, pixel_id


# ── STEP 2: LOAD ALL DIRECTLY-ASSIGNED PIXEL CSVS ────────────
def load_all_pixels(pixel_dir):
    """
    Load pixel CSVs from pixel_dir, restricted to pixels that are
    DIRECTLY assigned to at least one PV or station (n_locations > 0
    in goes_pixel_list.csv).

    pixel_mapping.py extracts a larger set than this (it also adds
    each assigned pixel's 4 cardinal neighbors, for spatial-gradient
    features). Spatial C13 gradient features were tested earlier and
    degraded LOSO performance, so those neighbor-only pixels are
    skipped here — the files remain on disk (in case spatial
    gradients are revisited later) but are not loaded into the
    feature pipeline.

    Returns DataFrame (T, n_assigned_pixels) indexed by datetime_local.
    """
    pixel_list_path = PROCESSED_DIR / "goes_pixel_list.csv"
    if not pixel_list_path.exists():
        raise FileNotFoundError(
            f"{pixel_list_path} not found — run src/pixel_mapping.py first"
        )
    pixel_list = pd.read_csv(pixel_list_path, index_col='pixel_index')
    assigned_pixel_ids = set(
        pixel_list.loc[pixel_list['n_locations'] > 0, 'pixel_id']
    )
    n_total_in_list = len(pixel_list)
    print(f"  goes_pixel_list.csv: {n_total_in_list} total pixels "
          f"({len(assigned_pixel_ids)} directly assigned, "
          f"{n_total_in_list - len(assigned_pixel_ids)} neighbor-only — skipped)")

    files = sorted(pixel_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No pixel CSVs found in {pixel_dir}\n"
            f"Download GEE outputs first and place them there."
        )
    print(f"  Found {len(files)} pixel CSV files on disk")

    bt_series_list = []
    refl_series_list = []
    n_skipped = 0
    for f in files:
        bt_series, refl_series, pid = load_pixel_csv(f)
        if pid not in assigned_pixel_ids:
            n_skipped += 1
            continue
        bt_series_list.append(bt_series)
        refl_series_list.append(refl_series)
        n_refl_valid = refl_series.notna().sum()
        print(f"    {pid:45s}  {len(bt_series)} rows  "
              f"BT=[{bt_series.min():.1f}, {bt_series.max():.1f}] K  "
              f"C02_valid={n_refl_valid}")

    print(f"  Loaded {len(bt_series_list)} directly-assigned pixels "
          f"(skipped {n_skipped} neighbor-only files)")

    bt_df = pd.concat(bt_series_list, axis=1)
    bt_df.index.name = 'datetime_local'

    refl_df = pd.concat(refl_series_list, axis=1)
    refl_df.index.name = 'datetime_local'

    return bt_df, refl_df


# ── STEP 3: COMPUTE BT + C02 FEATURES ────────────────────────
def compute_bt_features(bt_series, refl_series=None):
    """
    Compute BT and (optionally) C02 reflectance feature(s) for one
    pixel/station time series.

    bt_norm   : (BT - 270) / 50
                Normalised brightness temperature.
                High bt_norm → warm → clear sky → high GHI.
                Low bt_norm → cold cloud tops → thick cloud → low GHI.

    c02_norm  : (refl_c02 - 0.5) / 0.5
                Normalised visible/near-IR reflectance. Physically
                distinct from bt_norm — reflectance (cloud brightness)
                rather than temperature (cloud-top altitude).

                Nulls come in two kinds, handled differently:
                  - Nighttime (no sunlight to reflect): long, multi-hour
                    contiguous blocks. Left as NaN — this is a real
                    physical state, not missing data, and filling it
                    would fabricate a reflectance value that never
                    existed.
                  - Daytime satellite scan gaps: short, isolated nulls
                    (a handful of 5-min steps) surrounded by valid
                    readings on both sides. These get the SAME limited
                    time-interpolation as bt_norm (limit=12, i.e. up to
                    1 hour) — a short daytime gap is a missing
                    observation, not a missing phenomenon, same
                    reasoning as bt_norm's gap-fill.
                A interpolate(limit=12) call cannot bridge a full night
                (many hours of contiguous NaN), so this one rule
                correctly handles both cases without needing an
                explicit day/night classifier.

                Only present if refl_series is provided (older
                C13-only pixel files have no C02 data — c02_norm will
                be all-NaN for those, which downstream training code
                must handle, e.g. via dropna or a presence mask).

    NOTE: bt_lag30 / bt_diff / bt_diff60 (temporal gradient features)
    are intentionally NOT included here for the 5-min rebuild. The
    60-min version (bt_diff60) was tested at 30-min resolution and
    degraded LOSO performance (R²: 0.909→0.884). Rather than port the
    same feature family forward unexamined at the new resolution
    (where "shift(1)" silently means a different time lag), bt_norm
    alone is the new baseline. Re-add a lag/diff feature (for BT or
    C02) only if a controlled ablation at 5-min resolution shows it
    helps.

    Parameters
    ----------
    bt_series   : pd.Series  raw BT in Kelvin, 5-min index
    refl_series : pd.Series or None  raw C02 reflectance factor,
                  5-min index, same index as bt_series

    Returns
    -------
    pd.DataFrame  with columns [bt_norm] or [bt_norm, c02_norm]
    """
    bt = bt_series.copy()

    # Fill small gaps (up to 12 missing steps = 1 hour at 5-min resolution)
    bt = bt.interpolate(method='time', limit=12)

    bt_norm = (bt - BT_REF_K) / BT_SCALE_K

    feat = {'bt_norm': bt_norm}

    if refl_series is not None:
        # limit=12 (1 hour) bridges short daytime scan gaps but cannot
        # bridge a full night — so nighttime nulls correctly remain NaN
        # without needing an explicit day/night classifier. Same logic
        # as bt_norm's gap-fill above.
        refl = refl_series.copy()
        refl = refl.interpolate(method='time', limit=12)
        c02_norm = (refl - REFL_REF) / REFL_SCALE
        feat['c02_norm'] = c02_norm

    return pd.DataFrame(feat)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  c13_features.py — GOES-18 C13 + C02 Feature Engineering")
    print("=" * 60)

    # ── 1. Load pixel map ──────────────────────────────────────
    print("\n[1/5] Loading pixel map...")
    pv_map = pd.read_csv(PROCESSED_DIR / "pv_pixel_map.csv")

    station_map = pv_map[pv_map['pv_name'].str.startswith('STATION_')].copy()
    pv_map_only = pv_map[~pv_map['pv_name'].str.startswith('STATION_')].copy()
    station_map['station'] = station_map['pv_name'].str.replace('STATION_', '')

    print(f"  PV locations    : {len(pv_map_only)}")
    print(f"  Station entries : {len(station_map)}")

    # ── 2. Load all pixel BT + C02 time series ─────────────────
    print("\n[2/5] Loading pixel CSVs...")
    bt_pixels, refl_pixels = load_all_pixels(C13_PIXEL_DIR)
    print(f"\n  Pixel BT matrix   : {bt_pixels.shape}")
    print(f"  Pixel C02 matrix  : {refl_pixels.shape}")
    print(f"  Time range (raw)  : {bt_pixels.index[0]}  →  {bt_pixels.index[-1]}")

    # Trim to the actual study period. Extractions were deliberately
    # started a few days before 2024-01-01 (UTC) to give the GEE export
    # margin to cover the Dec-31-local / Jan-1-UTC boundary cleanly —
    # this leaves a short lead-in window before the satellite's first
    # valid scan, which shows up as nulls right at the start of the
    # file. That's not a real data gap to fix; it's data outside the
    # study period that should simply be dropped.
    study_start = pd.Timestamp(STUDY_START_DATE, tz=LOCAL_TZ)
    bt_pixels   = bt_pixels[bt_pixels.index >= study_start]
    refl_pixels = refl_pixels[refl_pixels.index >= study_start]
    print(f"  Time range (trimmed to study period): "
          f"{bt_pixels.index[0]}  →  {bt_pixels.index[-1]}")

    bt_vals = bt_pixels.values[~np.isnan(bt_pixels.values)]
    print(f"  BT range overall  : [{bt_vals.min():.1f},  {bt_vals.max():.1f}] K")
    refl_vals = refl_pixels.values[~np.isnan(refl_pixels.values)]
    if len(refl_vals):
        print(f"  C02 range overall : [{refl_vals.min():.3f},  {refl_vals.max():.3f}]  "
              f"({refl_pixels.notna().values.sum()} valid / {refl_pixels.size} total)")
    else:
        print(f"  C02 range overall : no valid C02 data found "
              f"(all pixel CSVs are C13-only)")

    # ── 3. Assign BT + C02 to stations ──────────────────────────
    print("\n[3/5] Assigning BT and C02 to station and PV locations...")
    station_names = list(STATIONS.keys())
    bt_st_dict = {}
    refl_st_dict = {}
    for _, row in station_map.iterrows():
        sname = row['station']
        pid   = row['pixel_id']
        if pid in bt_pixels.columns:
            bt_st_dict[sname] = bt_pixels[pid]
            refl_st_dict[sname] = refl_pixels[pid]
        else:
            print(f"  ⚠ Station {sname}: pixel {pid} not found")

    bt_stations = pd.DataFrame(bt_st_dict)[station_names]
    bt_stations.index.name = 'datetime_local'
    refl_stations = pd.DataFrame(refl_st_dict)[station_names]
    refl_stations.index.name = 'datetime_local'
    print(f"  bt_stations shape: {bt_stations.shape}")
    for s in station_names:
        v = bt_stations[s].dropna()
        r = refl_stations[s].dropna()
        print(f"    {s}: BT mean={v.mean():.1f}K  "
              f"min={v.min():.1f}K  max={v.max():.1f}K  "
              f"| C02 valid={len(r)}"
              + (f"  mean={r.mean():.3f}" if len(r) else ""))

    # ── 4. Assign BT + C02 to PVs ───────────────────────────────
    pv_names = pv_map_only['pv_name'].tolist()
    bt_pv_cols = {}
    refl_pv_cols = {}
    for _, row in pv_map_only.iterrows():
        pid = row['pixel_id']
        if pid in bt_pixels.columns:
            bt_pv_cols[row['pv_name']] = bt_pixels[pid]
            refl_pv_cols[row['pv_name']] = refl_pixels[pid]
        else:
            print(f"  ⚠ PV {row['pv_name']}: pixel {pid} not found")

    bt_pvs = pd.DataFrame(bt_pv_cols)[pv_names]
    bt_pvs.index.name = 'datetime_local'
    refl_pvs = pd.DataFrame(refl_pv_cols)[pv_names]
    refl_pvs.index.name = 'datetime_local'
    print(f"  bt_pvs shape     : {bt_pvs.shape}")
    print(f"  refl_pvs shape   : {refl_pvs.shape}")

    # ── 5. Save raw BT + C02 parquets ───────────────────────────
    C13_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    bt_pixels.to_parquet(C13_FEAT_DIR     / "c13_bt_pixels.parquet")
    bt_stations.to_parquet(C13_FEAT_DIR   / "c13_bt_stations.parquet")
    bt_pvs.to_parquet(C13_FEAT_DIR        / "c13_bt_pvs.parquet")
    refl_pixels.to_parquet(C13_FEAT_DIR   / "c02_refl_pixels.parquet")
    refl_stations.to_parquet(C13_FEAT_DIR / "c02_refl_stations.parquet")
    refl_pvs.to_parquet(C13_FEAT_DIR      / "c02_refl_pvs.parquet")
    print(f"\n  ✓ c13_bt_pixels.parquet     {bt_pixels.shape}")
    print(f"  ✓ c13_bt_stations.parquet   {bt_stations.shape}")
    print(f"  ✓ c13_bt_pvs.parquet        {bt_pvs.shape}")
    print(f"  ✓ c02_refl_pixels.parquet   {refl_pixels.shape}")
    print(f"  ✓ c02_refl_stations.parquet {refl_stations.shape}")
    print(f"  ✓ c02_refl_pvs.parquet      {refl_pvs.shape}")

    # ── 6. Compute engineered features ────────────────────────
    print("\n[4/5] Computing engineered BT + C02 features...")

    # Stations
    feat_st_list = {}
    for s in station_names:
        feat_st_list[s] = compute_bt_features(bt_stations[s], refl_stations[s])
    feat_stations = pd.concat(feat_st_list, axis=1)
    feat_stations.index.name = 'datetime_local'

    # PVs: compute once per unique pixel, assign to all PVs in that pixel
    unique_pixels = pv_map_only['pixel_id'].unique()
    pixel_feats = {}
    for pid in unique_pixels:
        if pid in bt_pixels.columns:
            pixel_feats[pid] = compute_bt_features(bt_pixels[pid], refl_pixels[pid])

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
    print("\n[5/5] Sanity check — features at S1 (daytime only):")
    s1_feats = feat_stations['S1'].dropna(subset=['bt_norm'])
    for col in s1_feats.columns:
        v = s1_feats[col].dropna()
        if len(v):
            print(f"  {col:12s}  mean={v.mean():+.4f}  "
                  f"std={v.std():.4f}  "
                  f"range=[{v.min():.3f}, {v.max():.3f}]  "
                  f"(n={len(v)})")
        else:
            print(f"  {col:12s}  no valid data")

    print(f"\n✓ c13_features.py complete")
    print(f"  Features per location : bt_norm, c02_norm")
    print(f"  Output dir: {C13_FEAT_DIR}")