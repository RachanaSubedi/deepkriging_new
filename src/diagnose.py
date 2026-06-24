"""
src/baseline_comparison.py

Comprehensive comparison of GHI prediction methods on the 4 LOSO
held-out stations, stratified by day type. Answers the question:
"Is DeepKriging actually better than simpler baselines?"

Methods compared (all evaluated on the SAME held-out station/timesteps):
    1. Nearest station copy   — predict = GHI of closest other station
    2. IDW station interpolation — predict = inverse-distance-weighted
       average of the OTHER 3 stations (not satellite — pure station IDW)
    3. NSRDB background only  — predict = bg_csi * pvlib_clearsky
       (i.e. what satellite reanalysis alone would give, no correction)
    4. DeepKriging             — your trained model's raw output
    5. DeepKriging + quantile correction — final corrected output

Metrics (for each method, each fold, AND pooled):
    RMSE, MAE, R², Bias (mean error)
    Peak RMSE   — RMSE restricted to top 10% GHI_true hours
    Ramp RMSE   — RMSE of |GHI(t) - GHI(t-30min)| (ramp magnitude)

Day-type stratification (using measured CSI variance per day as the
classifier — a simple, defensible regime split):
    Clear        — daily std(measured CSI) < 0.10  (smooth, high CSI)
    Cloudy-smooth — std in [0.10, 0.20)
    Broken-cloud — std >= 0.20  (high intra-day variability)

Run:
    python src/baseline_comparison.py

Outputs (outputs/validation/):
    baseline_comparison_overall.csv     — method x fold table
    baseline_comparison_by_regime.csv   — method x regime table
    baseline_comparison_summary.txt     — human-readable report
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import r2_score

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import STATIONS, RESID_DIR, BG_DIR, VAL_DIR, KM_PER_LAT, KM_PER_LON

CLEARSKY_MIN = 10.0
STATION_NAMES = list(STATIONS.keys())
LOCS = {s: (STATIONS[s]['lat'], STATIONS[s]['lon']) for s in STATION_NAMES}


# ── METRIC HELPERS ─────────────────────────────────────────────
def dist_km(a, b):
    dlat = (a[0] - b[0]) * KM_PER_LAT
    dlon = (a[1] - b[1]) * KM_PER_LON
    return np.sqrt(dlat**2 + dlon**2)


def compute_metrics(y_true, y_pred, ts=None):
    """Return dict of RMSE, MAE, R2, Bias, Peak RMSE, Ramp RMSE."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {k: np.nan for k in
                ['rmse', 'mae', 'r2', 'bias', 'peak_rmse', 'ramp_rmse', 'n']}

    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(err))

    # Peak RMSE: top 10% of measured GHI
    thresh = np.percentile(yt, 90)
    peak_mask = yt >= thresh
    peak_rmse = (float(np.sqrt(np.mean(err[peak_mask] ** 2)))
                 if peak_mask.sum() > 1 else np.nan)

    # Ramp RMSE: needs time order — only compute if ts provided
    ramp_rmse = np.nan
    if ts is not None:
        ts_m = pd.to_datetime(np.asarray(ts)[mask])
        order = np.argsort(ts_m)
        yt_o, yp_o = yt[order], yp[order]
        dt_true = np.diff(yt_o)
        dt_pred = np.diff(yp_o)
        ramp_err = dt_pred - dt_true
        ramp_rmse = float(np.sqrt(np.mean(ramp_err ** 2)))

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'bias': bias,
            'peak_rmse': peak_rmse, 'ramp_rmse': ramp_rmse, 'n': mask.sum()}


def classify_day(csi_series):
    """Classify each calendar day by std of measured CSI (daytime only)."""
    daily_std = csi_series.groupby(csi_series.index.date).std()
    regime = pd.Series(index=daily_std.index, dtype=object)
    regime[daily_std < 0.10] = 'clear'
    regime[(daily_std >= 0.10) & (daily_std < 0.20)] = 'cloudy_smooth'
    regime[daily_std >= 0.20] = 'broken_cloud'
    return regime  # indexed by date


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 65)
    print("  baseline_comparison.py — DeepKriging vs Simpler Baselines")
    print("=" * 65)

    # ── Load measured GHI/CSI at all stations ──────────────────
    print("\n[1/5] Loading station data...")
    csi = pd.read_parquet(RESID_DIR / "csi_stations.parquet")
    cs  = pd.read_parquet(BG_DIR / "clearsky_pvlib_stations.parquet")
    bg_csi = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    for d in [csi, cs, bg_csi]:
        d.index = pd.to_datetime(d.index)
    common = csi.index.intersection(cs.index).intersection(bg_csi.index)
    csi, cs, bg_csi = csi.loc[common], cs.loc[common], bg_csi.loc[common]
    ghi_measured = csi * cs
    ghi_nsrdb_only = bg_csi.clip(0, 1.3) * cs   # background-only "prediction"

    # ── Day regime classification per station ──────────────────
    print("[2/5] Classifying day regimes (clear / cloudy_smooth / broken_cloud)...")
    regimes = {}
    for s in STATION_NAMES:
        day_mask = cs[s] >= CLEARSKY_MIN
        regimes[s] = classify_day(csi[s][day_mask])
    regime_counts = pd.Series(
        pd.concat(regimes.values())).value_counts()
    print(f"  Day-type counts (pooled across stations):\n{regime_counts.to_string()}")

    # ── Load DeepKriging LOSO predictions ───────────────────────
    print("\n[3/5] Loading DeepKriging LOSO predictions...")
    dk_frames = []
    for k, s in enumerate(STATION_NAMES):
        df = pd.read_csv(VAL_DIR / f"fold_{k}_{s}_predictions.csv")
        df['datetime_local'] = pd.to_datetime(df['datetime_local'], utc=True) \
                                  .dt.tz_convert('America/Los_Angeles')
        df['station'] = s
        dk_frames.append(df)
    dk_all = pd.concat(dk_frames, ignore_index=True)

    # ── Quantile-corrected version: reuse correction_factors.csv ──
    cf_path = Path(VAL_DIR).parent / "predictions" / "correction_factors.csv"
    has_corrected = cf_path.exists()
    if has_corrected:
        cf = pd.read_csv(cf_path).set_index(['month', 'hour'])['correction_factor']
        dk_all['month'] = dk_all['datetime_local'].dt.month
        dk_all['hour']  = dk_all['datetime_local'].dt.hour
        def lookup_cf(row):
            key = (row['month'], row['hour'])
            return cf[key] if key in cf.index else 1.0
        dk_all['cf'] = dk_all.apply(lookup_cf, axis=1)
        dk_all['ghi_pred_corrected'] = dk_all['ghi_pred'] * dk_all['cf']
    else:
        dk_all['ghi_pred_corrected'] = dk_all['ghi_pred']
        print("  ⚠ correction_factors.csv not found — using uncorrected as 'corrected'")

    # ── Compute baselines per fold ───────────────────────────────
    print("\n[4/5] Computing all 5 methods per fold...")

    rows = []
    rows_regime = []

    for k, target in enumerate(STATION_NAMES):
        others = [s for s in STATION_NAMES if s != target]

        day_mask = cs[target].values >= CLEARSKY_MIN
        ts_target = cs.index[day_mask]
        y_true = ghi_measured[target].values[day_mask]

        # 1. Nearest station copy
        nearest = min(others, key=lambda s: dist_km(LOCS[target], LOCS[s]))
        y_nearest = ghi_measured[nearest].reindex(cs.index).values[day_mask]

        # 2. IDW of the other 3 stations (pure station interpolation)
        weights = np.array([1.0 / dist_km(LOCS[target], LOCS[s]) for s in others])
        weights /= weights.sum()
        y_idw = np.zeros(day_mask.sum())
        for w, s in zip(weights, others):
            y_idw += w * ghi_measured[s].reindex(cs.index).values[day_mask]

        # 3. NSRDB background only
        y_nsrdb = ghi_nsrdb_only[target].values[day_mask]

        # 4 & 5. DeepKriging raw / corrected — from LOSO predictions CSV
        dk_fold = dk_all[dk_all['station'] == target].set_index('datetime_local')
        dk_fold = dk_fold.reindex(ts_target)
        y_dk = dk_fold['ghi_pred'].values
        y_dk_corr = dk_fold['ghi_pred_corrected'].values

        methods = {
            'nearest_station': y_nearest,
            'idw_stations': y_idw,
            'nsrdb_only': y_nsrdb,
            'deepkriging': y_dk,
            'deepkriging_corrected': y_dk_corr,
        }

        regime_series = regimes[target].reindex(pd.Series(ts_target).dt.date.values)
        regime_arr = regime_series.values

        for method_name, y_pred in methods.items():
            m = compute_metrics(y_true, y_pred, ts=ts_target)
            m['method'] = method_name
            m['fold'] = k
            m['station'] = target
            rows.append(m)

            # by regime
            for regime_name in ['clear', 'cloudy_smooth', 'broken_cloud']:
                rmask = regime_arr == regime_name
                if rmask.sum() < 2:
                    continue
                mr = compute_metrics(y_true[rmask], y_pred[rmask])
                mr['method'] = method_name
                mr['station'] = target
                mr['regime'] = regime_name
                rows_regime.append(mr)

    overall_df = pd.DataFrame(rows)
    regime_df = pd.DataFrame(rows_regime)

    # ── Save ─────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    overall_df.to_csv(VAL_DIR / "baseline_comparison_overall.csv", index=False)
    regime_df.to_csv(VAL_DIR / "baseline_comparison_by_regime.csv", index=False)

    # ── Print summary tables ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("  OVERALL: Mean across 4 folds, per method")
    print("=" * 80)
    summary = (overall_df.groupby('method')
               [['rmse', 'mae', 'r2', 'bias', 'peak_rmse', 'ramp_rmse']]
               .mean().round(3))
    # Order methods logically
    order = ['nearest_station', 'idw_stations', 'nsrdb_only',
             'deepkriging', 'deepkriging_corrected']
    summary = summary.reindex(order)
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("  BY DAY REGIME: Mean RMSE_GHI per method x regime")
    print("=" * 80)
    if len(regime_df) > 0:
        pivot = regime_df.pivot_table(values='rmse', index='method',
                                      columns='regime', aggfunc='mean').round(2)
        pivot = pivot.reindex(order)
        regime_cols = [c for c in ['clear', 'cloudy_smooth', 'broken_cloud']
                       if c in pivot.columns]
        print(pivot[regime_cols].to_string())

        print("\n  BY DAY REGIME: Mean R2_GHI per method x regime")
        pivot_r2 = regime_df.pivot_table(values='r2', index='method',
                                         columns='regime', aggfunc='mean').round(3)
        pivot_r2 = pivot_r2.reindex(order)
        print(pivot_r2[regime_cols].to_string())

    # ── Write text summary ─────────────────────────────────────
    lines = [
        "Baseline Comparison: DeepKriging vs Simpler Methods",
        "=" * 55, "",
        "OVERALL (mean across 4 LOSO folds):",
        summary.to_string(), "",
    ]
    if len(regime_df) > 0:
        lines += ["BY DAY REGIME (mean RMSE_GHI):", pivot[regime_cols].to_string(), "",
                   "BY DAY REGIME (mean R2_GHI):", pivot_r2[regime_cols].to_string(), ""]

    dk_rmse = summary.loc['deepkriging', 'rmse']
    near_rmse = summary.loc['nearest_station', 'rmse']
    verdict = ("DeepKriging OUTPERFORMS nearest-station copy on average."
               if dk_rmse < near_rmse else
               "Nearest-station copy outperforms DeepKriging on average "
               "at the 4 VALIDATION stations. This is expected when stations "
               "are closely spaced; DeepKriging's value proposition is "
               "spatial diversity across the 178 PV locations where most "
               "points are far from any station, not necessarily lower "
               "RMSE at the 4 training/validation points themselves.")
    lines += ["VERDICT:", verdict]

    (VAL_DIR / "baseline_comparison_summary.txt").write_text('\n'.join(lines))
    print(f"\n✓ Saved:")
    print(f"  {VAL_DIR / 'baseline_comparison_overall.csv'}")
    print(f"  {VAL_DIR / 'baseline_comparison_by_regime.csv'}")
    print(f"  {VAL_DIR / 'baseline_comparison_summary.txt'}")
    print(f"\n{verdict}")