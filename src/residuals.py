"""
src/residuals.py

Computes CSI residuals at the 4 station locations:

    residual(s, t) = CSI_measured(s, t) − CSI_background(s, t)

where
    CSI_measured(s,t)    = GHI_station(s,t) / GHI_clearsky_idw(s,t)
    CSI_background(s,t)  = IDW-interpolated NSRDB CSI  (from background_field.py)
    GHI_clearsky_idw(s,t)= IDW-interpolated NSRDB clearsky GHI (from background_field.py)

Station file:  data/raw/stations/all_stations_GHI_30min_PST_filled.csv
               columns: datetime | GHI_S1 | GHI_S2 | GHI_S3 | GHI_P2

Run:
    python src/residuals.py

Outputs (data/processed/residuals/):
    csi_stations.parquet       (17520, 4)  measured CSI at stations
    residuals_stations.parquet (17520, 4)  CSI residual at stations
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    STATION_DIR, BG_DIR, RESID_DIR, STATIONS,
)

# ── CONSTANTS ────────────────────────────────────────────────
STATION_FILE      = STATION_DIR / "all_stations_GHI_30min_PST_filled.csv"
CLEARSKY_MIN_W_M2 = 10.0     # below this → nighttime → CSI = 0
CSI_CLIP_MAX      = 2.0      # cloud-enhancement cap


# ── STEP 1: LOAD STATION GHI ─────────────────────────────────
def load_station_ghi(filepath):
    """
    Load measured GHI for all 4 stations.

    datetime format: M/D/YYYY H:MM  (PST, no DST — tz_localize fixed offset)

    Returns
    -------
    df : DataFrame (T, 4)  index = tz-aware DatetimeIndex (America/Los_Angeles)
         columns = S1, S2, S3, P2
    """
    df = pd.read_csv(filepath, sep=None, engine='python', encoding='utf-8-sig')

    # Parse datetime — PST fixed offset (UTC-8), no DST shifts in data
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize('Etc/GMT+8')          # PST = UTC-8
    df.index = df.index.tz_convert('America/Los_Angeles')  # match background
    df.index.name = 'datetime_local'

    # Rename columns to match station keys: GHI_S1 → S1
    df.columns = [c.replace('GHI_', '') for c in df.columns]

    print(f"  Rows      : {len(df)}")
    print(f"  Columns   : {list(df.columns)}")
    print(f"  Time range: {df.index[0]}  →  {df.index[-1]}")
    print(f"  GHI range : [{df.values.min():.2f},  {df.values.max():.2f}] W/m²")

    return df


# ── STEP 2: COMPUTE MEASURED CSI ─────────────────────────────
def measured_csi(ghi_df, clearsky_df):
    """
    CSI_measured = GHI_station / GHI_clearsky_idw

    Nighttime rule : clearsky < CLEARSKY_MIN_W_M2  → CSI = 0
    Clip           : [0.0, CSI_CLIP_MAX]

    Parameters
    ----------
    ghi_df      : DataFrame (T, 4)  measured GHI  W/m²
    clearsky_df : DataFrame (T, 4)  IDW clearsky GHI  W/m²

    Returns
    -------
    csi_df : DataFrame (T, 4)
    """
    ghi = ghi_df.values.astype(np.float32)
    cs  = clearsky_df.values.astype(np.float32)

    csi = np.zeros_like(ghi, dtype=np.float32)
    day = cs >= CLEARSKY_MIN_W_M2
    csi[day] = ghi[day] / cs[day]
    np.clip(csi, 0.0, CSI_CLIP_MAX, out=csi)

    return pd.DataFrame(csi,
                        index=ghi_df.index,
                        columns=ghi_df.columns)


# ── STEP 3: ALIGN TIMESTAMPS ─────────────────────────────────
def align(df_station, df_background):
    """
    Inner-join on the datetime index.
    Both should be 17520 rows — any mismatch is flagged.
    """
    common = df_station.index.intersection(df_background.index)
    if len(common) != len(df_station):
        missing = len(df_station) - len(common)
        print(f"  ⚠ {missing} station timestamps have no background match "
              f"— they will be dropped")
    return df_station.loc[common], df_background.loc[common]


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  residuals.py — Station CSI Residuals")
    print("=" * 55)

    # ── 1. Load station GHI ──────────────────────────────────
    print("\n[1/4] Loading station GHI...")
    ghi_df = load_station_ghi(STATION_FILE)

    # ── 2. Load background fields ────────────────────────────
    print("\n[2/4] Loading background fields...")
    bg_csi      = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clearsky = pd.read_parquet(BG_DIR / "clearsky_pvlib_stations.parquet")
    print(f"  bg_csi      : {bg_csi.shape}   "
          f"range [{bg_csi.values.min():.3f}, {bg_csi.values.max():.3f}]")
    print(f"  bg_clearsky : {bg_clearsky.shape}   "
          f"range [{bg_clearsky.values.min():.1f}, {bg_clearsky.values.max():.1f}] W/m²")

    # ── 3. Align timestamps ───────────────────────────────────
    print("\n[3/4] Aligning timestamps...")
    ghi_aligned, cs_aligned = align(ghi_df, bg_clearsky)
    _,           bg_aligned = align(ghi_df, bg_csi)
    print(f"  Aligned rows: {len(ghi_aligned)}")

    # ── 4. Compute CSI and residuals ─────────────────────────
    print("\n[4/4] Computing CSI and residuals...")

    csi_df     = measured_csi(ghi_aligned, cs_aligned)
    resid_df   = csi_df - bg_aligned.values   # element-wise

    RESID_DIR.mkdir(parents=True, exist_ok=True)
    csi_df.to_parquet(RESID_DIR   / "csi_stations.parquet")
    #resid_df.to_parquet(RESID_DIR / "residuals_stations.parquet")

    # ── Sanity check ─────────────────────────────────────────
    print("\n── Sanity Check ────────────────────────────────────")
    station_names = list(STATIONS.keys())

    for s in station_names:
        day_mask   = cs_aligned[s] >= CLEARSKY_MIN_W_M2
        r_day      = resid_df[s][day_mask]
        csi_day    = csi_df[s][day_mask]
        bg_day     = bg_aligned[s][day_mask]

        print(f"\n  {s}:")
        print(f"    Measured CSI    mean={csi_day.mean():.3f}  "
              f"std={csi_day.std():.3f}  "
              f"max={csi_day.max():.3f}")
        print(f"    Background CSI  mean={bg_day.mean():.3f}  "
              f"std={bg_day.std():.3f}")
        print(f"    Residual        mean={r_day.mean():.4f}  "
              f"std={r_day.std():.3f}  "
              f"range=[{r_day.min():.3f}, {r_day.max():.3f}]")

    print(f"\n✓ residuals.py complete")
    print(f"  csi_stations.parquet       {csi_df.shape}")
    print(f"  residuals_stations.parquet {resid_df.shape}")
    print(f"  Output dir: {RESID_DIR}")