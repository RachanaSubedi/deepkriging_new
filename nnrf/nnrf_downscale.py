"""
src/nnrf_downscale.py

Nearest-Neighbor Random Forest (NNRF) spatiotemporal downscaling,
following the reference paper's actual implementation (Asiedu et al.
2025) as closely as possible, adapted to our 178-PV case:

  - ONE RandomForestRegressor(n_estimators=100) trained PER PV
    (not one shared model — matches the paper's per-site loop exactly)
  - Features per PV: Temperature, Pressure, Wind Speed, Dew Point,
    GHI_global (own NSRDB cell), Hour_sin/cos, DayOfYear_sin/cos,
    plus GHI_global of its k=3 nearest PV neighbors
    (GHI_neighbor_1/2/3 — same naming as the paper's code)
  - Target (GHI_local): whichever synthetic 178-PV field we're
    treating as ground truth — DeepKriging's or IDW's — selected
    with the TARGET argument. This is the substitution described in
    the pipeline discussion: DK/IDW manufactures the "local sensor"
    data the paper assumes already exists at every site.

Validation: fixed calendar holdout — December 31, hours 5-22 local —
matching the paper's day-ahead validation exactly (same date, same
hour window, trained on the other 364 days). A second model is
refit on the FULL year for the final deliverable series (the one
that actually feeds OpenDSS).

Run:
    python src/nnrf_downscale.py dk      # target = DeepKriging output
    python src/nnrf_downscale.py idw     # target = IDW output

Outputs (outputs/nnrf_{target}/):
    ghi_pvs_nnrf.parquet        (T, 178)  final downscaled GHI, full year
    nnrf_metrics.csv            per-PV RMSE / MAE / R2 / GoF on holdout
    nnrf_feature_importance.csv per-PV RF feature importances
    nnrf_summary.txt            human-readable summary
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PROCESSED_DIR, OUTPUT_DIR

# ── CONFIG ────────────────────────────────────────────────────
FEAT_DIR   = PROCESSED_DIR / "nnrf_features"
PV_FILE    = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"

TARGET_FILES = {
    'dk':  OUTPUT_DIR / "predictions" / "ghi_pvs.parquet",
    'idw': OUTPUT_DIR / "idw" / "ghi_pvs_idw.parquet",
}

# Last N calendar days held out for validation, all other days used for
# training. Switched from the paper's single-day (Dec 31) protocol because
# our NSRDB source was downloaded in UTC — the local-time year therefore
# truncates ~8 hours early on Dec 31 (UTC year-end = Dec 31 16:00 PST),
# cutting the last day's afternoon/evening short. A 5-day window absorbs
# that truncation without losing validation power on the other 4 days.
HOLDOUT_DAYS  = 5
N_ESTIMATORS  = 100     # matches the paper exactly
RANDOM_STATE  = 7       # matches the paper exactly

BASE_FEATURES = ['temperature', 'pressure', 'windspeed', 'dewpoint', 'ghi_global',
                  'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']


def load_global_features():
    """Load the 5 NSRDB feature matrices built by nnrf_features.py."""
    feats = {}
    for name in ['temperature', 'pressure', 'windspeed', 'dewpoint', 'ghi_global']:
        feats[name] = pd.read_parquet(FEAT_DIR / f"nsrdb_{name}_pvs.parquet")
    return feats


def add_cyclical_time(index):
    doy = index.day_of_year.values
    hour = index.hour.values + index.minute.values / 60.0
    return {
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'doy_sin':  np.sin(2 * np.pi * doy / 365),
        'doy_cos':  np.cos(2 * np.pi * doy / 365),
    }


def build_pv_dataframe(pv_name, feats, cyc, target_series, neighbor_names):
    """Assemble the feature+target DataFrame for one PV, paper-style."""
    df = pd.DataFrame(index=feats['ghi_global'].index)
    for name in ['temperature', 'pressure', 'windspeed', 'dewpoint', 'ghi_global']:
        df[name] = feats[name][pv_name].values
    for k, v in cyc.items():
        df[k] = v
    for rank, nb in enumerate(neighbor_names, start=1):
        df[f'ghi_neighbor_{rank}'] = feats['ghi_global'][nb].values
    df['ghi_local'] = target_series.values
    return df


def gof(y_true, y_pred):
    """Paper's Goodness-of-Fit metric: (1 - NRMSE) * 100."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    span = y_true.max() - y_true.min()
    if span < 1e-6:
        return np.nan
    return 100.0 * (1.0 - rmse / span)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in TARGET_FILES:
        print("Usage: python src/nnrf_downscale.py {dk|idw}")
        sys.exit(1)

    TARGET = sys.argv[1]
    TARGET_FILE = TARGET_FILES[TARGET]
    OUT_DIR = OUTPUT_DIR / f"nnrf_{TARGET}"

    print("=" * 60)
    print(f"  nnrf_downscale.py — target = {TARGET.upper()}")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading inputs...")
    feats = load_global_features()
    target_df = pd.read_parquet(TARGET_FILE)
    neighbors_df = pd.read_csv(FEAT_DIR / "pv_neighbors_k3.csv").set_index('pv_name')
    pv_names = pd.read_csv(PV_FILE)['pv_name'].tolist()

    common = feats['ghi_global'].index.intersection(target_df.index)
    for k in feats:
        feats[k] = feats[k].loc[common]
    target_df = target_df.loc[common]
    cyc = add_cyclical_time(common)
    print(f"  Target file    : {TARGET_FILE.name}")
    print(f"  Common timesteps: {len(common)}")
    print(f"  PVs            : {len(pv_names)}")

    holdout_start_date = (common.max() - pd.Timedelta(days=HOLDOUT_DAYS)).date()
    print(f"\n[2/5] Training {len(pv_names)} per-PV Random Forests "
          f"(n_estimators={N_ESTIMATORS}, holdout=last {HOLDOUT_DAYS} days "
          f"[{holdout_start_date} → {common.max().date()}])...")

    metrics_rows = []
    importance_rows = []
    final_pred = pd.DataFrame(index=common, columns=pv_names, dtype=np.float32)

    t0 = time.time()
    for j, pv in enumerate(pv_names):
        neighbor_names = [neighbors_df.loc[pv, f'neighbor_{r}'] for r in (1, 2, 3)]

        df = build_pv_dataframe(pv, feats, cyc, target_df[pv], neighbor_names)
        feature_cols = BASE_FEATURES + [f'ghi_neighbor_{r}' for r in (1, 2, 3)]

        # daytime only: where target is real (DK/IDW already NaN at night)
        day_mask = df['ghi_local'].notna() & df[feature_cols].notna().all(axis=1)
        df_day = df[day_mask]

        # Last HOLDOUT_DAYS calendar days -> validation, everything else
        # -> training. No fixed hour window (unlike the paper's single-day
        # protocol) since DK/IDW targets are already NaN at night.
        idx = df_day.index
        val_mask = pd.Series(idx.date, index=idx) >= holdout_start_date
        val_mask = val_mask.values
        train_mask = ~val_mask

        X_train = df_day.loc[train_mask, feature_cols].values
        y_train = df_day.loc[train_mask, 'ghi_local'].values
        X_val   = df_day.loc[val_mask, feature_cols].values
        y_val   = df_day.loc[val_mask, 'ghi_local'].values

        if len(X_train) < 50 or len(X_val) < 10:
            print(f"    ⚠ {pv}: insufficient data (train={len(X_train)}, "
                  f"val={len(X_val)}) — skipping")
            continue

        # --- Model A: holdout fit, for honest validation metrics ---
        model_val = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                          random_state=RANDOM_STATE, n_jobs=-1)
        model_val.fit(X_train, y_train)
        y_pred_val = model_val.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        mae  = mean_absolute_error(y_val, y_pred_val)
        r2   = r2_score(y_val, y_pred_val)
        g    = gof(y_val, y_pred_val)

        metrics_rows.append({'pv_name': pv, 'rmse': rmse, 'mae': mae,
                             'r2': r2, 'gof': g,
                             'n_train': len(X_train), 'n_val': len(X_val)})
        importance_rows.append(dict(zip(feature_cols, model_val.feature_importances_),
                                    pv_name=pv))

        # --- Model B: full-year refit, for the final deliverable series ---
        X_full = df_day[feature_cols].values
        y_full = df_day['ghi_local'].values
        model_full = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                           random_state=RANDOM_STATE, n_jobs=-1)
        model_full.fit(X_full, y_full)
        y_pred_full = model_full.predict(X_full)
        final_pred.loc[df_day.index, pv] = y_pred_full

        if (j + 1) % 20 == 0 or j == len(pv_names) - 1:
            elapsed = time.time() - t0
            print(f"  {j+1:>3}/{len(pv_names)} PVs done  ({elapsed:.0f}s)  "
                  f"last: {pv}  RMSE={rmse:.1f}  GoF={g:.1f}%")

    print(f"\n[3/5] Saving final downscaled series...")
    final_pred.index.name = 'datetime_local'
    final_pred.to_parquet(OUT_DIR / "ghi_pvs_nnrf.parquet")
    print(f"  ✓ ghi_pvs_nnrf.parquet  {final_pred.shape}")

    print(f"\n[4/5] Saving metrics and feature importances...")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "nnrf_metrics.csv", index=False)
    mean_row = metrics_df[['rmse', 'mae', 'r2', 'gof']].mean()
    print(f"  Mean over {len(metrics_df)} PVs: "
          f"RMSE={mean_row['rmse']:.2f}  MAE={mean_row['mae']:.2f}  "
          f"R2={mean_row['r2']:.4f}  GoF={mean_row['gof']:.2f}%")

    importance_df = pd.DataFrame(importance_rows).set_index('pv_name')
    importance_df.to_csv(OUT_DIR / "nnrf_feature_importance.csv")
    mean_importance = importance_df.mean().sort_values(ascending=False)
    print(f"\n  Mean feature importance across all PVs:")
    for feat, val in mean_importance.items():
        print(f"    {feat:20s} {val:.4f}")

    print(f"\n[5/5] Writing summary...")
    lines = [
        f"NNRF Downscaling Summary — target = {TARGET.upper()}",
        "=" * 50,
        f"n_estimators = {N_ESTIMATORS}, holdout = last {HOLDOUT_DAYS} days, "
        f"k neighbors = 3",
        "",
        f"Mean validation metrics ({len(metrics_df)} PVs):",
        f"  RMSE = {mean_row['rmse']:.2f} W/m2",
        f"  MAE  = {mean_row['mae']:.2f} W/m2",
        f"  R2   = {mean_row['r2']:.4f}",
        f"  GoF  = {mean_row['gof']:.2f}%",
        "",
        "Mean feature importance:",
    ]
    for feat, val in mean_importance.items():
        lines.append(f"  {feat:20s} {val:.4f}")
    (OUT_DIR / "nnrf_summary.txt").write_text('\n'.join(lines))

    print(f"\n✓ nnrf_downscale.py complete ({TARGET.upper()})")
    print(f"  Output dir: {OUT_DIR}")