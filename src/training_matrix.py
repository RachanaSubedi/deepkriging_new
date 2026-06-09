"""
src/training_matrix.py

Assembles the complete training matrix for DeepKriging:

    X : (N, 426)   411 Wendland RBF basis + 15 covariates
    y : (N,)       CSI residual at station location
    fold_ids : (N,) station index 0-3 for LOSO CV

Only daytime rows (bg_clearsky >= 10 W/m²) included.

15 covariates:
    1  bg_csi              IDW NSRDB background CSI
    2  bg_csi_lag30        bg_csi at t-30 min
    3  bg_csi_diff         bg_csi(t) - bg_csi(t-1)
    4  clearsky_frac       clearsky / daily-max clearsky
    5  cos_zenith          cos(Solar Zenith Angle)
    6  bt_norm             (BT_C13 - 270) / 50
    7  bt_lag30            bt_norm at t-30 min
    8  bt_diff             bt_norm(t) - bt_norm(t-1)
    9  temperature         IDW temperature (normalised)
    10 rh                  IDW relative humidity (normalised)
    11 pressure            IDW pressure (normalised)
    12 pw                  IDW precipitable water (normalised)
    13 cloud_type          nearest-neighbour cloud type (normalised)
    14 elevation           IDW elevation, static (normalised)
    15 doy_sin             sin(2π × day-of-year / 365)

Run:
    python src/training_matrix.py

Outputs (data/processed/training_matrix/):
    X.npy, y.npy, fold_ids.npy, timestamps.npy
    scaler_mean.npy, scaler_std.npy
    feature_names.txt, training_summary.txt
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    STATIONS, BASIS_DIR, BG_DIR, RESID_DIR,
    C13_FEAT_DIR, TRAIN_DIR,
)

CLEARSKY_MIN  = 10.0     # W/m²  daytime threshold
CLEARSKY_NORM = 1000.0   # W/m²  normalise clearsky GHI

# Normalisation reference values for met variables
MET_NORM = {
    'temperature' : (15.0,  20.0),   # (ref, scale)  °C
    'rh'          : (70.0,  30.0),   # %
    'pressure'    : (990.0, 20.0),   # mbar
    'pw'          : (1.0,   1.0),    # cm
    'cos_zenith'  : (0.0,   1.0),    # already [-1,1]
    'cloud_type'  : (6.0,   6.0),    # 0-12 → centred
    'elevation'   : (300.0, 100.0),  # m
}


def norm(arr, ref, scale):
    return (arr - ref) / scale


# ── COVARIATE BUILDER ─────────────────────────────────────────
def build_covariates(timestamps, bg_csi_s, bg_clearsky_s,
                     c13_s, met_s, elev_s):
    """
    Build 15 covariates for one station's daytime time series.

    Parameters
    ----------
    timestamps    : DatetimeIndex (T,)
    bg_csi_s      : (T,)  background CSI
    bg_clearsky_s : (T,)  clearsky GHI W/m²
    c13_s         : DataFrame (T, 3)  bt_norm, bt_lag30, bt_diff
    met_s         : dict  key → (T,)  met variables
    elev_s        : float  scalar elevation for this station

    Returns
    -------
    cov   : (T, 15)  float32
    names : list[str]
    """
    bg   = bg_csi_s.astype(np.float32)
    bg_lag = np.concatenate([[np.nan], bg[:-1]])
    bg_dif = bg - bg_lag

    # Clearsky fraction of daily peak
    cs_ser    = pd.Series(bg_clearsky_s, index=timestamps)
    daily_max = cs_ser.groupby(cs_ser.index.date).transform('max').values
    daily_max = np.where(daily_max < 1.0, 1.0, daily_max)
    cs_frac   = (bg_clearsky_s / daily_max).astype(np.float32)

    # Day-of-year sine
    doy     = timestamps.day_of_year.values
    doy_sin = np.sin(2 * np.pi * doy / 365).astype(np.float32)

    # Elevation — broadcast scalar to (T,)
    elev_arr = np.full(len(timestamps), elev_s, dtype=np.float32)

    cov = np.column_stack([
        bg,                                                      # 1
        bg_lag,                                                  # 2
        bg_dif,                                                  # 3
        cs_frac,                                                 # 4
        norm(met_s['cos_zenith'], *MET_NORM['cos_zenith']),      # 5
        c13_s['bt_norm'].values.astype(np.float32),              # 6
        c13_s['bt_lag30'].values.astype(np.float32),             # 7
        c13_s['bt_diff'].values.astype(np.float32),              # 8
        norm(met_s['temperature'], *MET_NORM['temperature']),    # 9
        norm(met_s['rh'],          *MET_NORM['rh']),             # 10
        norm(met_s['pressure'],    *MET_NORM['pressure']),       # 11
        norm(met_s['pw'],          *MET_NORM['pw']),             # 12
        norm(met_s['cloud_type'],  *MET_NORM['cloud_type']),     # 13
        norm(elev_arr,             *MET_NORM['elevation']),      # 14
        doy_sin,                                                 # 15
    ]).astype(np.float32)

    names = [
        'bg_csi', 'bg_csi_lag30', 'bg_csi_diff',
        'clearsky_frac', 'cos_zenith',
        'bt_norm', 'bt_lag30', 'bt_diff',
        'temperature', 'rh', 'pressure', 'pw',
        'cloud_type', 'elevation', 'doy_sin',
    ]
    return cov, names


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  training_matrix.py — Assemble X, y, fold_ids")
    print("=" * 60)

    station_names = list(STATIONS.keys())

    # ── 1. Load inputs ────────────────────────────────────────
    print("\n[1/4] Loading inputs...")

    Phi_st    = np.load(BASIS_DIR / "Phi_stations_scaled.npy")   # (4, K)
    bg_csi    = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clear  = pd.read_parquet(BG_DIR / "bg_clearsky_stations.parquet")
    residuals = pd.read_parquet(RESID_DIR / "residuals_stations.parquet")
    c13_feats = pd.read_parquet(C13_FEAT_DIR / "c13_feat_stations.parquet")

    # Met variables
    met_keys  = ['temperature', 'rh', 'pressure', 'pw',
                 'cos_zenith', 'cloud_type']
    met_data  = {k: pd.read_parquet(BG_DIR / f"met_{k}_stations.parquet")
                 for k in met_keys}

    # Elevation (static per station)
    elev_st = np.load(BG_DIR / "elevation_stations.npy")   # (4,)

    print(f"  Phi_stations : {Phi_st.shape}")
    print(f"  bg_csi       : {bg_csi.shape}")
    print(f"  residuals    : {residuals.shape}")
    print(f"  c13_feats    : {c13_feats.shape}")
    for k in met_keys:
        print(f"  met_{k:12s}: {met_data[k].shape}")
    print(f"  elevations   : {elev_st.round(1)}")

    # ── 2. Align timestamps ───────────────────────────────────
    print("\n[2/4] Aligning timestamps...")
    common = residuals.index
    for df in [bg_csi, bg_clear, c13_feats] + list(met_data.values()):
        common = common.intersection(df.index)

    bg_csi    = bg_csi.loc[common]
    bg_clear  = bg_clear.loc[common]
    residuals = residuals.loc[common]
    c13_feats = c13_feats.loc[common]
    met_data  = {k: v.loc[common] for k, v in met_data.items()}
    print(f"  Common timesteps : {len(common)}")

    # ── 3. Build per-station matrices ─────────────────────────
    print("\n[3/4] Building station feature matrices...")

    X_list, y_list, fold_list, ts_list = [], [], [], []

    for s_idx, s_name in enumerate(station_names):
        print(f"\n  Station {s_name} (fold {s_idx})...")

        # Daytime mask
        day_mask = bg_clear[s_name].values >= CLEARSKY_MIN
        T_day    = day_mask.sum()
        ts       = common[day_mask]

        y_s = residuals.loc[ts, s_name].values.astype(np.float32)

        # C13 features for this station
        if s_name in c13_feats.columns.get_level_values(0):
            c13_s = c13_feats[s_name].loc[ts]
        else:
            print(f"    ⚠ C13 missing for {s_name} — using zeros")
            c13_s = pd.DataFrame(
                np.zeros((T_day, 3)), index=ts,
                columns=['bt_norm', 'bt_lag30', 'bt_diff'])

        # Met arrays for this station (daytime only)
        met_s = {k: met_data[k].loc[ts, s_name].values.astype(np.float32)
                 for k in met_keys}

        # Build covariates
        cov_s, cov_names = build_covariates(
            ts,
            bg_csi.loc[ts, s_name].values,
            bg_clear.loc[ts, s_name].values,
            c13_s, met_s,
            float(elev_st[s_idx]),
        )

        # Basis functions: same row for all timesteps of this station
        phi_rep = np.tile(Phi_st[s_idx], (T_day, 1))   # (T_day, K)

        # Concatenate [Phi | covariates]
        X_s = np.concatenate([phi_rep, cov_s], axis=1)  # (T_day, K+15)

        # Drop rows with any NaN (lag features at t=0, C13 gaps)
        nan_rows = np.isnan(X_s).any(axis=1) | np.isnan(y_s)
        X_s  = X_s[~nan_rows];   y_s  = y_s[~nan_rows]
        ts_s = ts[~nan_rows]
        fold_s = np.full(len(y_s), s_idx, dtype=np.int8)

        print(f"    Daytime rows  : {T_day}")
        print(f"    After NaN drop: {len(y_s)} rows")
        print(f"    X shape       : {X_s.shape}")
        print(f"    y mean / std  : {y_s.mean():.4f} / {y_s.std():.4f}")

        X_list.append(X_s);      y_list.append(y_s)
        fold_list.append(fold_s); ts_list.append(ts_s)

    # ── 4. Stack and save ─────────────────────────────────────
    print("\n[4/4] Stacking and saving...")

    X        = np.vstack(X_list).astype(np.float32)
    y        = np.concatenate(y_list).astype(np.float32)
    fold_ids = np.concatenate(fold_list).astype(np.int8)
    ts_ns    = np.concatenate([t.view('int64') for t in ts_list])

    K          = Phi_st.shape[1]
    feat_names = [f'phi_{i}' for i in range(K)] + cov_names

    # Global scaler
    sc_mean = np.nanmean(X, axis=0).astype(np.float32)
    sc_std  = np.nanstd(X,  axis=0).astype(np.float32)
    sc_std[sc_std < 1e-8] = 1.0

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_DIR / "X.npy",           X)
    np.save(TRAIN_DIR / "y.npy",           y)
    np.save(TRAIN_DIR / "fold_ids.npy",    fold_ids)
    np.save(TRAIN_DIR / "timestamps.npy",  ts_ns)
    np.save(TRAIN_DIR / "scaler_mean.npy", sc_mean)
    np.save(TRAIN_DIR / "scaler_std.npy",  sc_std)
    (TRAIN_DIR / "feature_names.txt").write_text('\n'.join(feat_names))

    # Summary
    lines = [
        "DeepKriging Training Matrix Summary", "=" * 45,
        f"Total samples : {len(y)}",
        f"Feature dim   : {X.shape[1]}  ({K} basis + 15 covariates)",
        f"Covariates    : {cov_names}",
        "", "Samples per fold:",
    ]
    for i, s in enumerate(station_names):
        lines.append(f"  Fold {i} ({s}) : {(fold_ids==i).sum()}")
    lines += ["", f"y mean={y.mean():.4f}  std={y.std():.4f}  "
              f"range=[{y.min():.3f}, {y.max():.3f}]"]
    (TRAIN_DIR / "training_summary.txt").write_text('\n'.join(lines))

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"  X shape     : {X.shape}   {X.nbytes/1e6:.1f} MB")
    print(f"  Feature dim : {X.shape[1]}  ({K} basis + 15 covariates)")
    for i, s in enumerate(station_names):
        print(f"  Fold {i} ({s}) : {(fold_ids==i).sum()} samples")
    print(f"\n✓ training_matrix.py complete")
    print(f"  Output dir: {TRAIN_DIR}")