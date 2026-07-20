"""
src/predictnew.py

ABLATION: PV-location inference using the basis-functions-only
DeepKriging models trained by trainnew.py (no covariates).

Loads the 4 LOSO models from outputs/models_nocov/ (each expecting
input_dim = 411, basis functions only) and runs inference at all
178 PV locations using Phi_pvs alone — no covariate matrix is built.

This mirrors predict.py's ensemble-averaging and low-sun blending
logic exactly, so the two outputs are directly comparable:
    outputs/predictions/ghi_pvs.parquet         ← predict.py (full model)
    outputs/predictions_nocov/ghi_pvs.parquet   ← this script (basis-only)

Run:
    python src/predictnew.py

Outputs (outputs/predictions_nocov/):
    ghi_pvs.parquet           (T, 178)  GHI W/m²  — NaN at nighttime
    csi_pvs.parquet           (T, 178)  predicted CSI
    csi_pred_raw_pvs.parquet  (T, 178)  predicted CSI residual (unclipped)
"""

import numpy as np
import pandas as pd
import torch
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    STATIONS, BASIS_DIR, BG_DIR,
    MODEL_DIR, PRED_DIR,
    HIDDEN_SIZE, DROPOUT,
)
from src.model import DeepKriging

# ── CONSTANTS ─────────────────────────────────────────────────
CLEARSKY_MIN  = 10.0     # W/m²  daytime threshold
N_BASIS       = 411
N_FOLDS       = 4
DEVICE        = torch.device('cpu')

# ── ABLATION: redirect to _nocov model / output dirs ──────────
MODEL_DIR = MODEL_DIR.parent / (MODEL_DIR.name + "_nocov")
PRED_DIR  = PRED_DIR.parent  / (PRED_DIR.name  + "_nocov")


# ── PREDICT BATCH ─────────────────────────────────────────────
def predict_batch(model, X_batch):
    """Run model on (N, 411) numpy array, return (N,) numpy array."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_batch, dtype=torch.float32))
    return out.cpu().numpy()


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  predictnew.py — DeepKriging PV Inference (ABLATION: basis-only)")
    print("=" * 60)

    # ── 1. Load models and scalers ────────────────────────────
    print(f"\n[1/5] Loading {N_FOLDS} LOSO models (input_dim={N_BASIS}, no covariates)...")
    models, scalers = [], []
    for k in range(N_FOLDS):
        m = DeepKriging(N_BASIS, HIDDEN_SIZE, DROPOUT).to(DEVICE)
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

    # Background fields — needed only for day-mask, low-sun blend,
    # and GHI = CSI × clearsky reconstruction. NOT fed to the model
    # as a covariate in this ablation.
    bg_csi_pvs   = pd.read_parquet(BG_DIR / "bg_csi_pvs.parquet")
    bg_clear_pvs = pd.read_parquet(BG_DIR / "clearsky_pvlib_pvs.parquet")

    print(f"  bg_csi_pvs   : {bg_csi_pvs.shape}")
    print(f"  bg_clear_pvs : {bg_clear_pvs.shape}")

    # ── 3. Align timestamps ───────────────────────────────────
    print("\n[3/5] Aligning timestamps...")
    common = bg_csi_pvs.index.intersection(bg_clear_pvs.index)

    bg_csi_pvs   = bg_csi_pvs.loc[common]
    bg_clear_pvs = bg_clear_pvs.loc[common]
    T            = len(common)
    print(f"  Aligned timesteps : {T}")

    # ── 4. Run ensemble inference for all PVs ─────────────────
    print("\n[4/5] Running ensemble inference (basis functions only)...")

    # Raw arrays (T, M)
    bg_arr = bg_csi_pvs[pv_names].values.astype(np.float32)
    cs_arr = bg_clear_pvs[pv_names].values.astype(np.float32)

    # Daytime mask (T, M)
    day_mask = cs_arr >= CLEARSKY_MIN   # (T, M)

    # Output arrays
    csi_raw_out = np.full((T, M), np.nan, dtype=np.float32)
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

        # Feature: basis functions only (no covariates in this ablation)
        X_j = np.tile(Phi_pvs[j], (T_day, 1))   # (T_day, 411)

        # Ensemble: average predictions from all 4 models
        fold_preds = []
        for k, (model, (sc_mean, sc_std)) in enumerate(zip(models, scalers)):
            # Scale (mean=0, std=1 in ablation — effectively identity,
            # kept for consistency with train-time standardise() path)
            X_j_sc = (X_j - sc_mean) / sc_std
            X_j_sc = np.nan_to_num(X_j_sc, nan=0.0)
            resid_k = predict_batch(model, X_j_sc)
            fold_preds.append(resid_k)

        csi_pred_j = np.mean(fold_preds, axis=0)
        csi_j = np.clip(csi_pred_j, 0.0, 1.3)  # initial ceiling

        # At very low clearsky (near sunrise/sunset), the model has insufficient
        # signal (unstable lag/BT features) and saturates near the clip ceiling.
        # Blend toward the NSRDB background CSI instead of trusting the raw
        # model output, with a tiered final cap since cloud enhancement is
        # not physically plausible at very low sun angles.
        LOW_SUN_BLEND_THRESHOLD = 200.0  # W/m²

        low_sun = cs_j < LOW_SUN_BLEND_THRESHOLD
        if low_sun.any():
            blend_w = np.clip(cs_j[low_sun] / LOW_SUN_BLEND_THRESHOLD, 0.0, 1.0)
            csi_j[low_sun] = (blend_w * csi_pred_j[low_sun] +
                              (1 - blend_w) * np.clip(bg_j[low_sun], 0.0, 1.0))

        # Tiered final cap based on clearsky level (after blend)
        # Applied to ALL timesteps, not just low-sun ones — this was
        # previously nested inside `if low_sun.any():`, which meant any
        # PV with zero low-sun timesteps in its series never had the cap
        # applied at all. Fixed June 2026 (matches predict.py fix).
        final_cap = np.where(cs_j < 100, 0.85,
                             np.where(cs_j < 200, 0.90,
                                      np.where(cs_j < 350, 0.95, 1.3)))
        csi_j = np.minimum(csi_j, final_cap)
        csi_j = np.clip(csi_j, 0.0, 1.3)

        ghi_j = csi_j * cs_j

        csi_raw_out[day_j, j] = csi_pred_j
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
    csi_raw_df = save_pred(csi_raw_out, "csi_pred_raw_pvs")

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

    print(f"\n✓ predictnew.py complete (ablation: basis-only)")
    print(f"  Output dir: {PRED_DIR}")