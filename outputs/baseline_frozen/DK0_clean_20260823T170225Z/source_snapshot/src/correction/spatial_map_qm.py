"""
src/spatial_map_qm.py

Time-series visualisation of the quantile-mapped (QM) GHI predictions
(ghi_pvs_qm.parquet), plotted directly against the uncorrected baseline
(ghi_pvs.parquet) and station measurements — so the effect of
spatial_qm.py is visible, not just the corrected result in isolation.

Produces (outputs/figures/correction/):
  fig_{TARGET_DATE}_qm_vs_baseline_timeseries.png
      all 178 PV QM predictions (faint) + QM median + baseline median
      + station measurements, for one day
  fig_{TARGET_DATE}_qm_vs_baseline_diff.png
      per-timestep (QM - baseline) for all PVs, to see where/when the
      correction shifts things and by how much

Run:
    python src/correction/spatial_map_qm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import FIG_DIR, STATIONS, PRED_DIR

# ── Which date to plot — change this, nothing else needs editing ──
TARGET_DATE = "2024-12-31"

# ── PATHS ─────────────────────────────────────────────────────
QM_PARQUET       = PRED_DIR / "ghi_pvs_qm.parquet"
BASELINE_PARQUET = PRED_DIR / "ghi_pvs.parquet"
PV_CSV      = Path(__file__).parent.parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
STATION_CSV = (Path(__file__).parent.parent.parent / "data" / "raw" / "stations"
               / "all_stations_GHI_5min_PST.csv")

QM_OUT_DIR = FIG_DIR / "correction"
QM_OUT_DIR.mkdir(parents=True, exist_ok=True)

STATION_COLORS = {'S1': '#e63946', 'S2': '#2a9d8f',
                  'S3': '#e76f51', 'P2': '#264653'}


def load_ghi(path, label):
    df = pd.read_parquet(path)
    df.index = df.index.tz_localize(None) if df.index.tz is None \
        else df.index.tz_convert('America/Los_Angeles').tz_localize(None)
    print(f"  {label}: {df.shape}  [{df.index[0]} -> {df.index[-1]}]")
    return df


# ── LOAD ─────────────────────────────────────────────────────
print("Loading QM-corrected and baseline predictions...")
ghi_qm = load_ghi(QM_PARQUET, "QM-corrected")
ghi_base = load_ghi(BASELINE_PARQUET, "Baseline")

pv_df = pd.read_csv(PV_CSV)
pv_names = pv_df['pv_name'].tolist()

# guard against column-set mismatches between the two parquet files
common_pvs = [p for p in pv_names if p in ghi_qm.columns and p in ghi_base.columns]
if len(common_pvs) < len(pv_names):
    print(f"  ⚠ Only {len(common_pvs)}/{len(pv_names)} PVs present in both "
          f"QM and baseline files — plotting the common subset.")
pv_names = common_pvs

try:
    st = pd.read_csv(STATION_CSV, sep=None, engine='python',
                     encoding='utf-8-sig', index_col=0, parse_dates=True)
    st.index = (pd.to_datetime(st.index, utc=True)
                .tz_convert('America/Los_Angeles')
                .tz_localize(None))
    st.columns = [c.replace('GHI_', '') for c in st.columns]
    have_stations = True
    print("  Station data loaded")
except Exception as e:
    print(f"  ⚠ Station data unavailable: {e}")
    have_stations = False

target_date_obj = pd.Timestamp(TARGET_DATE).date()
date_tag = TARGET_DATE

day_qm = ghi_qm[ghi_qm.index.date == target_date_obj].dropna(how='all')
day_base = ghi_base[ghi_base.index.date == target_date_obj].dropna(how='all')

if day_qm.empty or day_base.empty:
    raise ValueError(
        f"No predictions found for {TARGET_DATE} in one or both files. "
        f"QM range: {ghi_qm.index[0].date()}–{ghi_qm.index[-1].date()}, "
        f"baseline range: {ghi_base.index[0].date()}–{ghi_base.index[-1].date()}"
    )

# align on common timestamps for this day (QM and baseline should share
# the same index, but don't assume it)
common_idx = day_qm.index.intersection(day_base.index)
day_qm = day_qm.loc[common_idx]
day_base = day_base.loc[common_idx]

print(f"\n{TARGET_DATE} summary:")
print(f"  Common daytime rows : {len(common_idx)}")
print(f"  QM max GHI          : {day_qm.max().max():.1f} W/m²")
print(f"  Baseline max GHI    : {day_base.max().max():.1f} W/m²")


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — QM vs baseline vs station measurements, time series
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))

# faint individual PV lines (QM only, to avoid visual clutter)
for col in pv_names:
    ax.plot(day_qm.index, day_qm[col], color='seagreen',
            lw=0.8, alpha=0.25, zorder=1)

qm_med = day_qm[pv_names].median(axis=1)
base_med = day_base[pv_names].median(axis=1)

ax.plot(day_qm.index, qm_med, color='seagreen', lw=2.4,
        label='QM-corrected median', zorder=4)
ax.plot(day_base.index, base_med, color='steelblue', lw=2.0, ls='-.',
        label='Baseline (uncorrected) median', zorder=3)

if have_stations:
    day_st = st[st.index.date == target_date_obj]
    for s, color in STATION_COLORS.items():
        if s in day_st.columns:
            ax.plot(day_st.index, day_st[s], color=color,
                    lw=2.2, ls='--', label=f'{s} measured', zorder=5)

ax.set_title(f'{pd.Timestamp(TARGET_DATE).strftime("%B %d, %Y")} — '
             f'QM-Corrected vs Baseline GHI at 178 PV Locations\n',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Time (PDT)')
ax.set_ylabel('GHI (W/m²)')
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(alpha=0.25)
ax.set_xlim(common_idx[0], common_idx[-1])

out1 = QM_OUT_DIR / f"fig_{date_tag}_qm_vs_baseline_timeseries.png"
plt.tight_layout()
plt.savefig(out1, dpi=160, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {out1.relative_to(FIG_DIR.parent)}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — (QM - baseline) diff, all PVs, to see shift magnitude
# ══════════════════════════════════════════════════════════════
diff = day_qm[pv_names] - day_base[pv_names]

fig, ax = plt.subplots(figsize=(13, 5))
for col in pv_names:
    ax.plot(diff.index, diff[col], color='darkorange', lw=0.7, alpha=0.3, zorder=1)

diff_med = diff.median(axis=1)
ax.plot(diff.index, diff_med, color='darkorange', lw=2.2,
        label='Median (QM − baseline)', zorder=3)
ax.axhline(0, color='grey', lw=1, ls=':')

ax.set_title(f'{pd.Timestamp(TARGET_DATE).strftime("%B %d, %Y")} — '
             f'Correction Magnitude (QM − Baseline) Across 178 PVs\n',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Time (PDT)')
ax.set_ylabel('ΔGHI (W/m²)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.25)
ax.set_xlim(diff.index[0], diff.index[-1])

out2 = QM_OUT_DIR / f"fig_{date_tag}_qm_vs_baseline_diff.png"
plt.tight_layout()
plt.savefig(out2, dpi=160, bbox_inches='tight')
plt.close()
print(f"  ✓ {out2.relative_to(FIG_DIR.parent)}")

print(f"\n✓ All QM figures saved to {QM_OUT_DIR}")