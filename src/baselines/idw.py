"""
src/idw_pipeline.py

Standalone Inverse-Distance-Weighting (IDW) baseline for solar irradiance
spatial interpolation. Fully self-contained and independent of the
DeepKriging pipeline — reads only the aligned station GHI file and the
NSRDB clear-sky background, writes everything to outputs/idw/.

IDW definition (matches the baseline used in the DeepKriging comparison):
    - power p = 1  (weights = 1 / distance, NOT 1 / distance^2)
    - straight-line distance in km via KM_PER_LAT / KM_PER_LON
    - applied directly to measured GHI (W/m^2), not to CSI

Three things this script produces:

  1. LOSO validation at the 4 stations
     Hold out each station, predict it from the other 3 by IDW, score
     RMSE / MAE / R^2 on GHI. This is the apples-to-apples number that
     goes head-to-head against DeepKriging's LOSO table.

  2. Full-field prediction at all 178 PV locations
     IDW-interpolate the 4 station GHI series to every PV, full year at
     5-min. This is the synthetic field that feeds the downstream
     spatiotemporal (NNRF) step.

  3. Spatial diversity analysis
     How many distinct GHI values IDW produces across the 178 PVs per
     timestep, and the per-PV mean spread. This is the quantity where
     DeepKriging was expected to win — reported here so the two methods
     can be compared directly.

Run:
    python src/idw_pipeline.py

Outputs (outputs/idw/):
    ghi_pvs_idw.parquet            (T, 178)  synthetic GHI at PVs
    idw_loso_metrics.csv           per-station RMSE/MAE/R2 + mean row
    idw_diversity.csv              spatial-diversity summary
    idw_summary.txt                human-readable summary of all three
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    STATIONS, STATION_DIR, BG_DIR, OUTPUT_DIR,
    KM_PER_LAT, KM_PER_LON,
)

# ── CONFIG ────────────────────────────────────────────────────
IDW_POWER      = 1.0     # weights = 1 / d^p
CLEARSKY_MIN   = 10.0    # W/m^2  daytime threshold (for masking / metrics)
DAYTIME_FIELD  = 100.0   # W/m^2  field-mean threshold for diversity calc

STATION_FILE   = STATION_DIR / "all_stations_GHI_5min_PST.csv"
PV_FILE        = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
OUT_DIR        = OUTPUT_DIR / "idw"

STATION_ORDER  = ['S1', 'S2', 'S3', 'P2']


# ── DISTANCE ──────────────────────────────────────────────────
def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * KM_PER_LAT
    dlon = (lon1 - lon2) * KM_PER_LON
    return np.sqrt(dlat ** 2 + dlon ** 2)


def idw_weights(target_lat, target_lon, src_lats, src_lons, power=IDW_POWER):
    """Return normalised IDW weights from one target to N sources."""
    d = dist_km(target_lat, target_lon, np.asarray(src_lats), np.asarray(src_lons))
    d = np.where(d < 1e-6, 1e-6, d)           # avoid divide-by-zero at coincident pts
    w = 1.0 / (d ** power)
    return w / w.sum()


# ── LOAD STATION GHI ──────────────────────────────────────────
def load_station_ghi():
    df = pd.read_csv(STATION_FILE, sep=None, engine='python', encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/Los_Angeles')
    df = df.set_index('datetime')
    df.columns = [c.replace('GHI_', '') for c in df.columns]
    return df[STATION_ORDER]


# ── 1. LOSO VALIDATION AT STATIONS ────────────────────────────
def run_loso(ghi_df):
    """Hold out each station, predict from other 3 by IDW, score on GHI."""
    rows = []
    for held in STATION_ORDER:
        others = [s for s in STATION_ORDER if s != held]
        w = idw_weights(
            STATIONS[held]['lat'], STATIONS[held]['lon'],
            [STATIONS[s]['lat'] for s in others],
            [STATIONS[s]['lon'] for s in others],
        )
        pred = (ghi_df[others].values * w).sum(axis=1)
        true = ghi_df[held].values

        # daytime only: score where the held-out station actually has sun
        day = true > CLEARSKY_MIN
        p, t = pred[day], true[day]

        rmse = np.sqrt(np.mean((p - t) ** 2))
        mae  = np.mean(np.abs(p - t))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot

        rows.append({'station': held, 'rmse': rmse, 'mae': mae, 'r2': r2,
                     'n_daytime': int(day.sum())})
        print(f"  {held}: RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:.4f}  "
              f"(n={day.sum()})")

    df = pd.DataFrame(rows)
    mean_row = {'station': 'Mean',
                'rmse': df['rmse'].mean(),
                'mae':  df['mae'].mean(),
                'r2':   df['r2'].mean(),
                'n_daytime': int(df['n_daytime'].sum())}
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    print(f"  Mean: RMSE={mean_row['rmse']:6.2f}  MAE={mean_row['mae']:6.2f}  "
          f"R2={mean_row['r2']:.4f}")
    return df


# ── 2. FULL-FIELD PREDICTION AT 178 PVs ───────────────────────
def predict_pvs(ghi_df, pv_df):
    """IDW-interpolate the 4 stations to all 178 PV locations."""
    pv_names = pv_df['pv_name'].tolist()
    src_lats = [STATIONS[s]['lat'] for s in STATION_ORDER]
    src_lons = [STATIONS[s]['lon'] for s in STATION_ORDER]

    ghi_arr = ghi_df[STATION_ORDER].values                  # (T, 4)
    out = np.empty((len(ghi_df), len(pv_names)), dtype=np.float32)

    for j, (_, row) in enumerate(pv_df.iterrows()):
        w = idw_weights(row['pv_lat'], row['pv_lon'], src_lats, src_lons)  # (4,)
        out[:, j] = (ghi_arr * w).sum(axis=1)

    ghi_pvs = pd.DataFrame(out, index=ghi_df.index, columns=pv_names)
    ghi_pvs.index.name = 'datetime_local'
    return ghi_pvs


# ── 3. SPATIAL DIVERSITY ──────────────────────────────────────
def diversity(ghi_pvs):
    day = ghi_pvs[ghi_pvs.mean(axis=1) > DAYTIME_FIELD]
    distinct = day.round(0).nunique(axis=1).mean()
    means = ghi_pvs.mean()
    spread = means.max() - means.min()
    return distinct, spread


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  idw_pipeline.py — IDW baseline (predict + LOSO + diversity)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading station GHI...")
    ghi_df = load_station_ghi()
    print(f"  Stations : {list(ghi_df.columns)}")
    print(f"  Shape    : {ghi_df.shape}")

    print("\n[2/4] LOSO validation at 4 stations...")
    loso = run_loso(ghi_df)
    loso.to_csv(OUT_DIR / "idw_loso_metrics.csv", index=False)

    print("\n[3/4] Predicting at 178 PV locations...")
    pv_df = pd.read_csv(PV_FILE)
    ghi_pvs = predict_pvs(ghi_df, pv_df)

    # Mask nighttime → NaN so downstream treats it like the DK output
    bg_cs = pd.read_parquet(BG_DIR / "bg_clearsky_pvs.parquet")
    common = ghi_pvs.index.intersection(bg_cs.index)
    ghi_pvs = ghi_pvs.loc[common]
    night = bg_cs.loc[common, ghi_pvs.columns] < CLEARSKY_MIN
    ghi_pvs = ghi_pvs.where(~night.values, np.nan).clip(lower=0)
    ghi_pvs.to_parquet(OUT_DIR / "ghi_pvs_idw.parquet")
    print(f"  Saved ghi_pvs_idw.parquet  {ghi_pvs.shape}")

    print("\n[4/4] Spatial diversity...")
    distinct, spread = diversity(ghi_pvs)
    div_df = pd.DataFrame([{
        'method': 'IDW',
        'distinct_values_per_timestep': round(distinct, 1),
        'n_pvs': ghi_pvs.shape[1],
        'per_pv_mean_spread_wm2': round(spread, 2),
    }])
    div_df.to_csv(OUT_DIR / "idw_diversity.csv", index=False)
    print(f"  Distinct GHI values / timestep : {distinct:.1f} / {ghi_pvs.shape[1]}")
    print(f"  Per-PV mean spread             : {spread:.2f} W/m²")

    # ── Human-readable summary ────────────────────────────────
    lines = [
        "IDW Baseline Summary", "=" * 45,
        f"IDW power p = {IDW_POWER}  (weights = 1/distance^{IDW_POWER:.0f})", "",
        "── LOSO validation (daytime GHI) ──",
    ]
    for _, r in loso.iterrows():
        lines.append(f"  {r['station']:>5}: RMSE={r['rmse']:6.2f}  "
                     f"MAE={r['mae']:6.2f}  R2={r['r2']:.4f}")
    lines += ["",
              "── Spatial diversity at 178 PVs ──",
              f"  Distinct values / timestep : {distinct:.1f} / {ghi_pvs.shape[1]}",
              f"  Per-PV mean spread         : {spread:.2f} W/m²", ""]
    (OUT_DIR / "idw_summary.txt").write_text('\n'.join(lines), encoding='utf-8')

    print(f"\n✓ idw_pipeline.py complete")
    print(f"  All outputs → {OUT_DIR}")