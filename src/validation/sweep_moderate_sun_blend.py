"""
sweep_moderate_sun_blend.py

Pure post-processing sweep — NO retraining.

Tests whether widening the low-sun blend (currently only active for
bg_clearsky < 200 W/m²) to a wider "moderate_sun" threshold, blended
with bg_csi, fixes the S1 morning overshoot WITHOUT hurting the other
three stations or non-morning hours.

For each (threshold, alpha) combination:
    moderate_sun = bg_clearsky < threshold
    csi_fixed = where(moderate_sun,
                       alpha * csi_pred_raw + (1-alpha) * bg_csi,
                       csi_pred)              # unchanged outside the zone
    csi_fixed = clip(csi_fixed, 0, 1.3)
    ghi_fixed = csi_fixed * bg_clearsky

Reports, per station AND pooled:
    - Overall RMSE/R² (GHI)            -- must not regress
    - Peak/noon (11:00-13:00) RMSE     -- must not regress
    - Clear-day RMSE (existing regime) -- the thing we're trying to fix

Requires fold CSVs to contain: csi_true, csi_pred, csi_pred_raw,
bg_csi, ghi_true, bg_clearsky, datetime_local.
(csi_pred_raw and bg_csi were added to train.py's CSV output —
 rerun training once before using this script.)

Run:
    python sweep_moderate_sun_blend.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

VAL_DIR = Path("outputs/validation")

FOLDS = {
    'S1': VAL_DIR / "fold_0_S1_predictions.csv",
    'S2': VAL_DIR / "fold_1_S2_predictions.csv",
    'S3': VAL_DIR / "fold_2_S3_predictions.csv",
    'P2': VAL_DIR / "fold_3_P2_predictions.csv",
}

THRESHOLDS = [200, 250, 300, 350, 400]   # 200 = current baseline threshold
ALPHAS     = [0.25, 0.50, 0.75, 1.00]     # 1.00 = no change (sanity check)

REQUIRED_COLS = ['csi_true', 'csi_pred', 'csi_pred_raw', 'bg_csi',
                  'ghi_true', 'bg_clearsky', 'datetime_local']


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def load_folds():
    dfs = {}
    for station, path in FOLDS.items():
        df = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} is missing columns {missing}. "
                f"Did you rerun train.py after adding csi_pred_raw/bg_csi "
                f"to the saved CSV? See script docstring."
            )
        df['datetime_local'] = pd.to_datetime(df['datetime_local'], utc=True) \
                                  .dt.tz_convert('America/Los_Angeles')
        df['hour'] = df['datetime_local'].dt.hour
        df['date'] = df['datetime_local'].dt.date
        dfs[station] = df
    return dfs


def classify_clear_days(df):
    """Reproduce diagnose.py's classify_day logic on this fold's csi_true."""
    daily_std = df.groupby('date')['csi_true'].std()
    regime = pd.Series(index=daily_std.index, dtype=object)
    regime[daily_std < 0.10] = 'clear'
    regime[(daily_std >= 0.10) & (daily_std < 0.20)] = 'cloudy_smooth'
    regime[daily_std >= 0.20] = 'broken_cloud'
    return df['date'].map(regime)


def apply_blend(df, threshold, alpha):
    """
    Apply the widened moderate-sun blend to csi_pred_raw, leaving
    csi_pred (already-blended/capped baseline) untouched outside the zone.
    """
    moderate_sun = df['bg_clearsky'].values < threshold
    csi_fixed = df['csi_pred'].values.copy()

    blended = (alpha * df['csi_pred_raw'].values +
               (1 - alpha) * np.clip(df['bg_csi'].values, 0.0, 1.0))
    csi_fixed[moderate_sun] = blended[moderate_sun]
    csi_fixed = np.clip(csi_fixed, 0.0, 1.3)

    ghi_fixed = csi_fixed * df['bg_clearsky'].values
    return csi_fixed, ghi_fixed, moderate_sun


def main():
    print("Loading fold predictions...")
    dfs = load_folds()

    for station, df in dfs.items():
        df['regime'] = classify_clear_days(df)

    print(f"\n{'='*100}")
    print("BASELINE (current pipeline, alpha=N/A, threshold=200 already applied in csi_pred)")
    print(f"{'='*100}")
    baseline_rows = []
    for station, df in dfs.items():
        overall_rmse = rmse(df['ghi_true'].values, df['ghi_pred'].values)
        clear_mask = df['regime'] == 'clear'
        clear_rmse = rmse(df.loc[clear_mask, 'ghi_true'].values,
                          (df.loc[clear_mask, 'csi_pred'].values * df.loc[clear_mask, 'bg_clearsky'].values))
        noon_mask = df['hour'].between(11, 13)
        noon_rmse = rmse(df.loc[noon_mask, 'ghi_true'].values,
                         (df.loc[noon_mask, 'csi_pred'].values * df.loc[noon_mask, 'bg_clearsky'].values))
        baseline_rows.append({'station': station, 'overall_rmse': overall_rmse,
                              'clear_rmse': clear_rmse, 'noon_rmse': noon_rmse})
        print(f"  {station}:  overall RMSE={overall_rmse:6.2f}  "
              f"clear RMSE={clear_rmse:6.2f}  noon RMSE={noon_rmse:6.2f}")

    base_df = pd.DataFrame(baseline_rows).set_index('station')

    print(f"\n{'='*100}")
    print("SWEEP: threshold x alpha  (delta vs baseline; negative = improvement)")
    print(f"{'='*100}")

    results = []
    for threshold in THRESHOLDS:
        for alpha in ALPHAS:
            row = {'threshold': threshold, 'alpha': alpha}
            station_overall, station_clear, station_noon = {}, {}, {}

            for station, df in dfs.items():
                csi_fixed, ghi_fixed, moderate_sun = apply_blend(df, threshold, alpha)

                overall_rmse = rmse(df['ghi_true'].values, ghi_fixed)
                clear_mask = (df['regime'] == 'clear').values
                clear_rmse = rmse(df['ghi_true'].values[clear_mask], ghi_fixed[clear_mask])
                noon_mask = df['hour'].between(11, 13).values
                noon_rmse = rmse(df['ghi_true'].values[noon_mask], ghi_fixed[noon_mask])

                station_overall[station] = overall_rmse
                station_clear[station]   = clear_rmse
                station_noon[station]    = noon_rmse

            row['pooled_overall_rmse'] = np.mean(list(station_overall.values()))
            row['pooled_clear_rmse']   = np.mean(list(station_clear.values()))
            row['pooled_noon_rmse']    = np.mean(list(station_noon.values()))
            row['S1_overall_rmse'] = station_overall['S1']
            row['S1_clear_rmse']   = station_clear['S1']
            row['S1_noon_rmse']    = station_noon['S1']
            # Worst-case degradation across the other 3 stations (overall RMSE)
            row['max_other_overall_delta'] = max(
                station_overall[s] - base_df.loc[s, 'overall_rmse']
                for s in ['S2', 'S3', 'P2']
            )
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df['S1_clear_delta'] = results_df['S1_clear_rmse'] - base_df.loc['S1', 'clear_rmse']
    results_df['S1_noon_delta']  = results_df['S1_noon_rmse']  - base_df.loc['S1', 'noon_rmse']

    print(results_df[['threshold', 'alpha', 'S1_clear_rmse', 'S1_clear_delta',
                      'S1_noon_rmse', 'S1_noon_delta',
                      'max_other_overall_delta', 'pooled_overall_rmse']]
          .round(2).to_string(index=False))

    out_csv = VAL_DIR / "moderate_sun_blend_sweep.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✓ Full sweep results saved: {out_csv}")

    # ── Acceptance check, per the agreed criteria ─────────────
    print(f"\n{'='*100}")
    print("ACCEPTANCE CHECK")
    print("  Criteria: S1 clear RMSE improves, S1 noon RMSE does not regress,")
    print("            other stations' overall RMSE does not regress (delta <= 1.0 W/m² tolerance)")
    print(f"{'='*100}")
    passing = results_df[
        (results_df['S1_clear_delta'] < 0) &
        (results_df['S1_noon_delta'] <= 1.0) &
        (results_df['max_other_overall_delta'] <= 1.0)
    ].sort_values('S1_clear_delta')

    if len(passing):
        print(passing[['threshold', 'alpha', 'S1_clear_delta', 'S1_noon_delta',
                       'max_other_overall_delta']].round(2).to_string(index=False))
        print(f"\n  -> Best candidate: threshold={passing.iloc[0]['threshold']:.0f}, "
              f"alpha={passing.iloc[0]['alpha']:.2f}")
    else:
        print("  No (threshold, alpha) combination passed all three criteria.")
        print("  The S1 clear-day failure may not be fixable by post-processing alone —")
        print("  consider this evidence the issue is in the raw model / training data,")
        print("  not the blending logic.")


if __name__ == "__main__":
    main()