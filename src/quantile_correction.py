"""
src/quantile_correction.py

Post-processing bias correction for DeepKriging GHI predictions.

Uses the LOSO held-out station predictions (genuinely out-of-sample)
to compute how much the model under-predicts GHI per month × hour bin.
Applies that correction factor to all 178 PV predictions.

This is valid because:
  - Each station's predictions come from a fold where it was held out
  - We compute average correction factors, not per-timestep fitting
  - All 178 PVs are in the same 30km domain as the 4 stations

Method:
  correction_factor(month, hour) = median(GHI_measured / GHI_predicted)
  GHI_corrected(pv, t) = GHI_pred(pv, t) × correction_factor(month(t), hour(t))

Run:
    python src/quantile_correction.py

Outputs (outputs/predictions/):
    ghi_pvs_corrected.parquet   (T, 178)  bias-corrected GHI
    ghi_pvs_corrected.csv       (T, 178)  same, with datetime column
    correction_factors.csv      correction table for documentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import VAL_DIR, PRED_DIR, FIG_DIR

CLEARSKY_MIN = 50.0    # W/m² — only compute correction where GHI is meaningful
MIN_RATIO    = 0.5     # clamp correction factor range
MAX_RATIO    = 1.5     # avoid overcorrecting

STATION_NAMES = ['S1', 'S2', 'S3', 'P2']

# ── STEP 1: LOAD LOSO PREDICTIONS ─────────────────────────────
print("=" * 55)
print("  quantile_correction.py — Peak Bias Correction")
print("=" * 55)

print("\n[1/4] Loading LOSO held-out predictions...")
frames = []
for k, s in enumerate(STATION_NAMES):
    path = VAL_DIR / f"fold_{k}_{s}_predictions.csv"
    df = pd.read_csv(path)
    df['datetime_local'] = (pd.to_datetime(df['datetime_local'], utc=True)
                            .dt.tz_convert('America/Los_Angeles'))
    frames.append(df)

all_preds = pd.concat(frames, ignore_index=True)
print(f"  Total held-out samples : {len(all_preds)}")

# Filter to meaningful daytime GHI
mask = (all_preds['ghi_true'] >= CLEARSKY_MIN) & \
       (all_preds['ghi_pred'] >= CLEARSKY_MIN)
all_preds = all_preds[mask].copy()
print(f"  After filtering (GHI ≥ {CLEARSKY_MIN} W/m²) : {len(all_preds)}")

# Extract month and hour
all_preds['month'] = all_preds['datetime_local'].dt.month
all_preds['hour']  = all_preds['datetime_local'].dt.hour
all_preds['ratio'] = all_preds['ghi_true'] / all_preds['ghi_pred']

print(f"\n  Ratio stats (measured / predicted):")
print(f"    mean   = {all_preds['ratio'].mean():.3f}")
print(f"    median = {all_preds['ratio'].median():.3f}")
print(f"    p90    = {all_preds['ratio'].quantile(0.90):.3f}")
print(f"    p10    = {all_preds['ratio'].quantile(0.10):.3f}")

# ── STEP 2: COMPUTE CORRECTION FACTORS ────────────────────────
print("\n[2/4] Computing correction factors by month × hour...")

PEAK_HOURS = range(9, 16)   # 09:00 – 15:00 PDT

cf_peak = (all_preds[all_preds['hour'].isin(PEAK_HOURS)]
           .groupby(['month', 'hour'])['ratio']
           .quantile(0.5)
           .reset_index()
           .rename(columns={'ratio': 'correction_factor'}))

cf_other = (all_preds[~all_preds['hour'].isin(PEAK_HOURS)]
            .groupby(['month', 'hour'])['ratio']
            .median()
            .reset_index()
            .rename(columns={'ratio': 'correction_factor'}))
cf_other['correction_factor'] = 1.0

cf = pd.concat([cf_peak, cf_other], ignore_index=True)

# Clamp to avoid overcorrection
cf['correction_factor'] = cf['correction_factor'].clip(MIN_RATIO, MAX_RATIO)

print(f"  Bins computed : {len(cf)}")
print(f"  CF range      : [{cf['correction_factor'].min():.3f}, "
      f"{cf['correction_factor'].max():.3f}]")
print(f"  CF mean       : {cf['correction_factor'].mean():.3f}")

cf.to_csv(PRED_DIR / "correction_factors.csv", index=False)
print(f"  ✓ correction_factors.csv saved")

# ── STEP 3: APPLY CORRECTION TO PV PREDICTIONS ────────────────
print("\n[3/4] Applying correction to 178 PV predictions...")

ghi = pd.read_parquet(PRED_DIR / "ghi_pvs.parquet")
ghi.index = pd.to_datetime(ghi.index, utc=True).tz_convert('America/Los_Angeles')

# Build a correction series aligned to ghi index
cf_lookup = cf.set_index(['month', 'hour'])['correction_factor']

months = ghi.index.month
hours  = ghi.index.hour

correction_series = pd.Series(index=ghi.index, dtype=np.float32)
for i, (m, h) in enumerate(zip(months, hours)):
    key = (m, h)
    if key in cf_lookup.index:
        correction_series.iloc[i] = np.float32(cf_lookup[key])
    else:
        correction_series.iloc[i] = 1.0  # no correction if bin missing

correction_series = correction_series.fillna(1.0)

# Apply: multiply each row by its correction factor
ghi_corr = ghi.multiply(correction_series, axis=0)

# Preserve NaN at nighttime
ghi_corr[ghi.isna()] = np.nan

daytime_orig = ghi.values[~np.isnan(ghi.values)]
daytime_corr = ghi_corr.values[~np.isnan(ghi_corr.values)]
print(f"  Original    max={daytime_orig.max():.1f}  "
      f"mean={daytime_orig.mean():.1f} W/m²")
print(f"  Corrected   max={daytime_corr.max():.1f}  "
      f"mean={daytime_corr.mean():.1f} W/m²")

# ── STEP 4: SAVE ───────────────────────────────────────────────
print("\n[4/4] Saving corrected predictions...")

ghi_corr.index.name = 'datetime_local'
ghi_corr.to_parquet(PRED_DIR / "ghi_pvs_corrected.parquet")

# CSV with PDT datetime column
ghi_csv = ghi_corr.copy()
ghi_csv.index = (pd.to_datetime(ghi_csv.index, utc=True)
                   .tz_convert('America/Los_Angeles')
                   .tz_localize(None))
ghi_csv.index.name = 'datetime'
ghi_csv = ghi_csv.map(lambda x: round(x, 4) if pd.notna(x) else x)
ghi_csv.to_csv(PRED_DIR / "ghi_pvs_corrected.csv", index=True)

print(f"  ✓ ghi_pvs_corrected.parquet  {ghi_corr.shape}")
print(f"  ✓ ghi_pvs_corrected.csv")

# ── PLOT: CORRECTION FACTOR HEATMAP ───────────────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)

cf_pivot = cf.pivot(index='hour', columns='month', values='correction_factor')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('GHI Bias Correction: Measured / Predicted Ratio\n'
             '(from LOSO held-out station predictions)',
             fontsize=12, fontweight='bold')

# Heatmap
im = ax1.imshow(cf_pivot.values, aspect='auto', cmap='RdYlGn',
                vmin=0.8, vmax=1.4,
                extent=[0.5, 12.5, cf_pivot.index.max()+0.5,
                        cf_pivot.index.min()-0.5])
plt.colorbar(im, ax=ax1, label='Correction Factor (measured/predicted)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Hour (PDT)')
ax1.set_title('Correction Factor Heatmap')
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])

# Distribution of ratios
ax2.hist(all_preds['ratio'], bins=50, color='steelblue',
         edgecolor='white', linewidth=0.4)
ax2.axvline(1.0, color='black', lw=1.5, ls='--', label='Perfect (ratio=1.0)')
ax2.axvline(all_preds['ratio'].median(), color='red', lw=1.5,
            label=f"Median = {all_preds['ratio'].median():.3f}")
ax2.set_xlabel('GHI_measured / GHI_predicted')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Prediction Ratios\n(all held-out stations)')
ax2.legend()
ax2.grid(alpha=0.25)

plt.tight_layout()
out = FIG_DIR / "fig_correction_factors.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ fig_correction_factors.png")

print(f"\n✓ Done")
print(f"  Use ghi_pvs_corrected.parquet for OpenDSS integration")
print(f"  Use ghi_pvs.parquet for reporting uncorrected model output")