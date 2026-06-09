"""
src/predict.py

Generate synthetic GHI at all 178 PV locations using an ensemble
of the 4 LOSO-trained DeepKriging models.

Each PV prediction = average of 4 model predictions.
Using all 4 models ensures every training station contributes.

Run:
    python src/predict.py

Outputs (outputs/predictions/):
    ghi_pvs.parquet       (T, 178)  GHI W/m²  — NaN at nighttime
    csi_pvs.parquet       (T, 178)  predicted CSI
    residual_pvs.parquet  (T, 178)  predicted CSI residual
"""

import numpy as np
import pandas as pd
import torch
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    STATIONS, BASIS_DIR, BG_DIR, C13_FEAT_DIR,
    MODEL_DIR, PRED_DIR,
    HIDDEN_SIZE, DROPOUT,
)
from src.model import DeepKriging

# ── CONSTANTS ─────────────────────────────────────────────────
CLEARSKY_MIN  = 10.0     # W/m²  daytime threshold
CLEARSKY_NORM = 1000.0
N_BASIS       = 411
N_FOLDS       = 4
DEVICE        = torch.device('cpu')

MET_NORM = {
    'temperature' : (15.0,  20.0),
    'rh'          : (70.0,  30.0),
    'pressure'    : (990.0, 20.0),
    'pw'          : (1.0,   1.0),
    'cos_zenith'  : (0.0,   1.0),
    'cloud_type'  : (6.0,   6.0),
    'elevation'   : (300.0, 100.0),
}
MET_KEYS = ['temperature', 'rh', 'pressure', 'pw', 'cos_zenith', 'cloud_type']


def norm(arr, ref, scale):
    return (arr - ref) / scale


# ── COVARIATE BUILDER (vectorised over all PVs) ───────────────
def build_cov_matrix(timestamps, bg_csi, bg_clearsky,
                     c13_norm, c13_lag30, c13_diff,
                     met_arrays, elev_vec):
    """
    Build covariate matrix for M PV locations at T timesteps.

    Parameters
    ----------
    timestamps   : DatetimeIndex (T,)
    bg_csi       : (T, M)
    bg_clearsky  : (T, M)
    c13_*        : (T, M)  BT features
    met_arrays   : dict key → (T, M)
    elev_vec     : (M,)  static elevation per PV

    Returns
    -------
    cov : (T, M, 15)  float32
    """
    T, M = bg_csi.shape

    bg      = bg_csi.astype(np.float32)
    bg_lag  = np.vstack([np.full((1, M), np.nan), bg[:-1]])
    bg_dif  = bg - bg_lag

    # Clearsky fraction of daily peak
    cs      = bg_clearsky.astype(np.float32)
    cs_df   = pd.DataFrame(cs, index=timestamps)
    daily_max = cs_df.groupby(cs_df.index.date).transform('max').values
    daily_max = np.where(daily_max < 1.0, 1.0, daily_max)
    cs_frac = (cs / daily_max).astype(np.float32)

    # DOY sine
    doy     = timestamps.day_of_year.values[:, None]   # (T, 1)
    doy_sin = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    doy_sin = np.broadcast_to(doy_sin, (T, M)).copy()

    # Elevation broadcast
    elev = np.broadcast_to(elev_vec[None, :], (T, M)).copy().astype(np.float32)

    cov = np.stack([
        bg,
        bg_lag,
        bg_dif,
        cs_frac,
        norm(met_arrays['cos_zenith'],   *MET_NORM['cos_zenith']),
        c13_norm.astype(np.float32),
        c13_lag30.astype(np.float32),
        c13_diff.astype(np.float32),
        norm(met_arrays['temperature'],  *MET_NORM['temperature']),
        norm(met_arrays['rh'],           *MET_NORM['rh']),
        norm(met_arrays['pressure'],     *MET_NORM['pressure']),
        norm(met_arrays['pw'],           *MET_NORM['pw']),
        norm(met_arrays['cloud_type'],   *MET_NORM['cloud_type']),
        norm(elev,                       *MET_NORM['elevation']),
        doy_sin,
    ], axis=2).astype(np.float32)   # (T, M, 15)

    return cov


# ── PREDICT BATCH ─────────────────────────────────────────────
def predict_batch(model, X_batch):
    """Run model on (N, 426) numpy array, return (N,) numpy array."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_batch, dtype=torch.float32))
    return out.cpu().numpy()


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  predict.py — DeepKriging PV Location Inference")
    print("=" * 60)

    # ── 1. Load models and scalers ────────────────────────────
    print(f"\n[1/5] Loading {N_FOLDS} LOSO models...")
    models, scalers = [], []
    for k in range(N_FOLDS):
        m = DeepKriging(N_BASIS + 15, HIDDEN_SIZE, DROPOUT).to(DEVICE)
        m.load_state_dict(
            torch.load(MODEL_DIR / f"fold_{k}_best.pt",
                       map_location=DEVICE))
        m.eval()
        sc_mean = np.load(MODEL_DIR / f"fold_{k}_scaler_mean.npy")
        sc_std  = np.load(MODEL_DIR / f"fold_{k}_scaler_std.npy")
        models.append(m)
        scalers.append((sc_mean, sc_std))
        print(f"  ✓ Fold {k} model loaded")

    # ── 2. Load PV spatial inputs ─────────────────────────────
    print("\n[2/5] Loading PV inputs...")

    Phi_pvs  = np.load(BASIS_DIR / "Phi_pvs_scaled.npy")   # (178, 411)
    pv_path  = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df    = pd.read_csv(pv_path)
    pv_names = pv_df['pv_name'].tolist()
    M        = len(pv_names)
    print(f"  PV locations : {M}")
    print(f"  Phi_pvs      : {Phi_pvs.shape}")

    # ── Mask untrained phi columns ────────────────────────────
    # Phi normalization was fit on 4 stations only. Any phi column
    # that is 0 for all stations was never trained (zero gradient
    # → weights stay at random init). Activating those columns with
    # PV phi values produces chaotic outputs. Zero them out so the
    # model stays in the regime it was trained on.
    Phi_st_ref = np.load(BASIS_DIR / "Phi_stations_scaled.npy")  # (4, 411)
    station_active = Phi_st_ref.max(axis=0) > 1e-6              # (411,) bool
    n_active = station_active.sum()
    Phi_pvs[:, ~station_active] = 0.0
    print(f"  Active phi cols : {n_active} / {len(station_active)} "
          f"({n_active/len(station_active):.1%} trained)")

    # Background fields
    bg_csi_pvs   = pd.read_parquet(BG_DIR / "bg_csi_pvs.parquet")
    bg_clear_pvs = pd.read_parquet(BG_DIR / "bg_clearsky_pvs.parquet")

    # Met variables
    met_pvs = {k: pd.read_parquet(BG_DIR / f"met_{k}_pvs.parquet")
               for k in MET_KEYS}

    # Elevation
    elev_pvs = np.load(BG_DIR / "elevation_pvs.npy")   # (178,)

    # C13 features — MultiIndex (pv_name, feature)
    c13_pvs = pd.read_parquet(C13_FEAT_DIR / "c13_feat_pvs.parquet")

    print(f"  bg_csi_pvs   : {bg_csi_pvs.shape}")
    print(f"  bg_clear_pvs : {bg_clear_pvs.shape}")
    print(f"  c13_feat_pvs : {c13_pvs.shape}")

    # ── 3. Align timestamps ───────────────────────────────────
    print("\n[3/5] Aligning timestamps...")
    common = bg_csi_pvs.index
    for df in [bg_clear_pvs, c13_pvs] + list(met_pvs.values()):
        common = common.intersection(df.index)

    bg_csi_pvs   = bg_csi_pvs.loc[common]
    bg_clear_pvs = bg_clear_pvs.loc[common]
    met_pvs      = {k: v.loc[common] for k, v in met_pvs.items()}
    c13_pvs      = c13_pvs.loc[common]
    T            = len(common)
    print(f"  Aligned timesteps : {T}")

    # ── 4. Build covariate matrix for all PVs ─────────────────
    print("\n[4/5] Building feature matrices and running ensemble...")

    # Raw arrays (T, M)
    bg_arr  = bg_csi_pvs[pv_names].values.astype(np.float32)
    cs_arr  = bg_clear_pvs[pv_names].values.astype(np.float32)
    met_arr = {k: met_pvs[k][pv_names].values.astype(np.float32)
               for k in MET_KEYS}

    # C13 features: MultiIndex → (T, M) per feature
    c13_norm_arr  = c13_pvs.xs('bt_norm',  axis=1, level=1)[pv_names].values
    c13_lag30_arr = c13_pvs.xs('bt_lag30', axis=1, level=1)[pv_names].values
    c13_diff_arr  = c13_pvs.xs('bt_diff',  axis=1, level=1)[pv_names].values

    # Build (T, M, 15) covariate tensor
    cov_all = build_cov_matrix(
        common, bg_arr, cs_arr,
        c13_norm_arr, c13_lag30_arr, c13_diff_arr,
        met_arr, elev_pvs,
    )   # (T, M, 15)

    # Daytime mask (T, M)
    day_mask = cs_arr >= CLEARSKY_MIN   # (T, M)

    # Output arrays
    resid_out = np.full((T, M), np.nan, dtype=np.float32)
    csi_out   = np.full((T, M), np.nan, dtype=np.float32)
    ghi_out   = np.full((T, M), np.nan, dtype=np.float32)

    t0 = time.time()

    # Process PV by PV (memory-efficient)
    for j, pv in enumerate(pv_names):

        day_j  = day_mask[:, j]       # (T,) boolean
        T_day  = day_j.sum()
        if T_day == 0:
            continue

        bg_j   = bg_arr[day_j, j]     # (T_day,)
        cs_j   = cs_arr[day_j, j]
        cov_j  = cov_all[day_j, j, :] # (T_day, 15)

        # Full feature: [phi_j repeated | covariates]
        phi_j  = np.tile(Phi_pvs[j], (T_day, 1))        # (T_day, 411)
        X_j    = np.concatenate([phi_j, cov_j], axis=1)  # (T_day, 426)

        # Ensemble: average predictions from all 4 models
        fold_preds = []
        for k, (model, (sc_mean, sc_std)) in enumerate(zip(models, scalers)):
            # Scale covariates only (basis cols have mean=0, std=1 → unchanged)
            X_j_sc = (X_j - sc_mean) / sc_std
            # Fill NaN lag features with 0 (standardised mean).
            # Training dropped NaN rows; prediction fills them so clip(NaN)≠0.
            X_j_sc = np.nan_to_num(X_j_sc, nan=0.0)
            resid_k = predict_batch(model, X_j_sc)
            fold_preds.append(resid_k)

        resid_j = np.mean(fold_preds, axis=0)             # (T_day,)
        csi_j   = np.clip(bg_j + resid_j, 0.0, 2.0)
        ghi_j   = csi_j * cs_j

        resid_out[day_j, j] = resid_j
        csi_out[day_j, j]   = csi_j
        ghi_out[day_j, j]   = ghi_j

        if (j + 1) % 30 == 0 or j == M - 1:
            elapsed = time.time() - t0
            print(f"  {j+1:>3}/{M} PVs done  "
                  f"({elapsed:.1f}s)  "
                  f"last: {pv}  "
                  f"GHI mean={np.nanmean(ghi_j[ghi_j>0]):.1f} W/m²")

    elapsed = time.time() - t0
    print(f"\n  Inference complete in {elapsed:.1f}s")

    # ── 5. Save ───────────────────────────────────────────────
    print("\n[5/5] Saving outputs...")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    def save_pred(arr, name):
        df = pd.DataFrame(arr, index=common, columns=pv_names)
        df.index.name = 'datetime_local'
        df.to_parquet(PRED_DIR / f"{name}.parquet")
        print(f"  ✓ {name}.parquet  {df.shape}  "
              f"non-NaN={df.notna().values.sum():,}")
        return df

    ghi_df   = save_pred(ghi_out,   "ghi_pvs")
    csi_df   = save_pred(csi_out,   "csi_pvs")
    resid_df = save_pred(resid_out, "residual_pvs")

    # ── Quick summary ─────────────────────────────────────────
    print(f"\n── Prediction Summary ───────────────────────────────")
    daytime_ghi = ghi_df.values[~np.isnan(ghi_df.values)]
    print(f"  Daytime GHI range  : [{daytime_ghi.min():.1f}, "
          f"{daytime_ghi.max():.1f}] W/m²")
    print(f"  Daytime GHI mean   : {daytime_ghi.mean():.1f} W/m²")
    print(f"  Daytime fraction   : "
          f"{(~np.isnan(ghi_df.values)).mean():.1%}")

    # Per-PV mean daytime GHI (show range across 178 PVs)
    pv_means = ghi_df.mean(skipna=True)
    print(f"  Per-PV mean GHI    : [{pv_means.min():.1f}, "
          f"{pv_means.max():.1f}] W/m²  "
          f"(spread across 178 PVs)")

    print(f"\n✓ predict.py complete")
    print(f"  Output dir: {PRED_DIR}")