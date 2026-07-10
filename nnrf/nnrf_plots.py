"""
src/nnrf_plots.py

Diagnostic plots for the NNRF downscaling stage, mirroring the key
figures from the reference paper (Asiedu et al. 2025):

  Fig 3/4 (paper) -> annual heatmaps of Global (NSRDB) vs Local
                     (DK or IDW target) GHI, day-of-year x hour-of-day,
                     for one PV.
  Fig 5   (paper) -> deviation heatmap: Local - Global, same layout.
  Fig 8   (paper) -> day-ahead time series: Global GHI, Local GHI
                     (true target), NNRF Downscaled GHI, all overlaid
                     on the holdout day (Dec 31, 5am-22:00 — same
                     window nnrf_downscale.py validates on).
  Fig 9   (paper) -> boxplots of (Actual - Predicted) residuals across
                     a handful of PVs on the holdout day.

Run:
    python src/nnrf_plots.py dk             # random PV, DK target
    python src/nnrf_plots.py idw            # random PV, IDW target
    python src/nnrf_plots.py dk pv_1051     # specific PV

Outputs (outputs/figures/):
    fig_nnrf_{target}_{pv}_heatmap_global.png
    fig_nnrf_{target}_{pv}_heatmap_local.png
    fig_nnrf_{target}_{pv}_heatmap_deviation.png
    fig_nnrf_{target}_{pv}_dayahead.png
    fig_nnrf_{target}_residual_boxplot.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PROCESSED_DIR, OUTPUT_DIR, FIG_DIR

FEAT_DIR = PROCESSED_DIR / "nnrf_features"

TARGET_FILES = {
    'dk':  OUTPUT_DIR / "predictions" / "ghi_pvs.parquet",
    'idw': OUTPUT_DIR / "idw" / "ghi_pvs_idw.parquet",
}

HOLDOUT_DATE     = "2024-12-31"   # matches nnrf_downscale.py exactly
HOLDOUT_HOUR_MIN = 5
HOLDOUT_HOUR_MAX = 22

N_BOXPLOT_PVS = 8   # same count as the paper's 8 sites


def annual_heatmap(ax, series, title, cmap='viridis', vmax=None):
    """Day-of-year (y) x Hour-of-day (x) heatmap, paper's Fig 3/4 style."""
    df = pd.DataFrame({'value': series.values,
                       'doy': series.index.day_of_year,
                       'hour': series.index.hour})
    pivot = df.groupby(['doy', 'hour'])['value'].mean().unstack()
    pivot = pivot.reindex(index=range(1, 367), columns=range(24))

    im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                   cmap=cmap, vmax=vmax,
                   extent=[0, 24, 1, 366])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=9)
    ax.set_ylabel('Day of Year', fontsize=9)
    return im


def deviation_heatmap(ax, local_series, global_series, title):
    df = pd.DataFrame({
        'dev': local_series.values - global_series.values,
        'doy': local_series.index.day_of_year,
        'hour': local_series.index.hour,
    })
    pivot = df.groupby(['doy', 'hour'])['dev'].mean().unstack()
    pivot = pivot.reindex(index=range(1, 367), columns=range(24))

    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   extent=[0, 24, 1, 366])
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=9)
    ax.set_ylabel('Day of Year', fontsize=9)
    return im


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TARGET_FILES:
        print("Usage: python src/nnrf_plots.py {dk|idw} [pv_name]")
        sys.exit(1)

    TARGET = sys.argv[1]
    NNRF_DIR = OUTPUT_DIR / f"nnrf_{TARGET}"

    print("=" * 60)
    print(f"  nnrf_plots.py — target = {TARGET.upper()}")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading data...")
    global_ghi = pd.read_parquet(FEAT_DIR / "nsrdb_ghi_global_pvs.parquet")
    local_ghi  = pd.read_parquet(TARGET_FILES[TARGET])
    nnrf_ghi   = pd.read_parquet(NNRF_DIR / "ghi_pvs_nnrf.parquet")

    common = global_ghi.index.intersection(local_ghi.index).intersection(nnrf_ghi.index)
    global_ghi, local_ghi, nnrf_ghi = (global_ghi.loc[common], local_ghi.loc[common],
                                        nnrf_ghi.loc[common])
    pv_names = list(local_ghi.columns)

    if len(sys.argv) == 3:
        pv = sys.argv[2]
        if pv not in pv_names:
            print(f"  ⚠ {pv} not found. Choosing random PV instead.")
            pv = np.random.choice(pv_names)
    else:
        pv = np.random.choice(pv_names)
    print(f"  PV selected: {pv}")

    g = global_ghi[pv].fillna(0)
    l = local_ghi[pv].fillna(0)
    n = nnrf_ghi[pv].dropna()

    # ── [2/4] Annual heatmaps (paper Fig 3/4/5) ────────────────
    print(f"\n[2/4] Annual heatmaps for {pv}...")
    vmax_shared = max(g.max(), l.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Annual Irradiance Profile — {pv}  ({TARGET.upper()} target)',
                 fontsize=13, fontweight='bold')

    im1 = annual_heatmap(axes[0], g, 'Global (NSRDB) GHI', cmap='viridis', vmax=vmax_shared)
    im2 = annual_heatmap(axes[1], l, f'Local GHI ({TARGET.upper()} target)',
                         cmap='magma', vmax=vmax_shared)
    im3 = deviation_heatmap(axes[2], l, g, 'Deviation (Local − Global)')

    fig.colorbar(im1, ax=axes[0], shrink=0.8, label='W/m²')
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label='W/m²')
    fig.colorbar(im3, ax=axes[2], shrink=0.8, label='W/m²')

    plt.tight_layout()
    out = FIG_DIR / f"fig_nnrf_{TARGET}_{pv}_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")

    # ── [3/4] Day-ahead comparison (paper Fig 8) ───────────────
    print(f"\n[3/4] Day-ahead comparison for {pv} ({HOLDOUT_DATE})...")
    day_mask = (common.date == pd.Timestamp(HOLDOUT_DATE).date())
    day_idx = common[day_mask]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(day_idx, g.loc[day_idx], color='#2a9d8f', lw=2, label='Global GHI (NSRDB)')
    ax.plot(day_idx, l.loc[day_idx], color='#264653', lw=2, label=f'Local GHI ({TARGET.upper()}, true)')
    nnrf_day = n.reindex(day_idx)
    ax.plot(day_idx, nnrf_day, color='#e63946', lw=2, ls='--', label='NNRF Downscaled GHI')

    ax.axvspan(pd.Timestamp(f"{HOLDOUT_DATE} {HOLDOUT_HOUR_MIN}:00", tz=common.tz),
              pd.Timestamp(f"{HOLDOUT_DATE} {HOLDOUT_HOUR_MAX}:00", tz=common.tz),
              alpha=0.08, color='grey', label='Validation window')

    ax.set_title(f'Day-Ahead GHI Prediction — {pv}  ({TARGET.upper()} target)\n{HOLDOUT_DATE}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('GHI (W/m²)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / f"fig_nnrf_{TARGET}_{pv}_dayahead.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")

    # ── [4/4] Residual boxplots across PVs (paper Fig 9) ───────
    print(f"\n[4/4] Residual boxplots ({N_BOXPLOT_PVS} PVs, {HOLDOUT_DATE})...")
    rng = np.random.default_rng(42)
    box_pvs = [pv] + list(rng.choice([p for p in pv_names if p != pv],
                                     size=N_BOXPLOT_PVS - 1, replace=False))

    residuals = {}
    for p in box_pvs:
        true_day = local_ghi.loc[day_idx, p]
        pred_day = nnrf_ghi.reindex(day_idx)[p]
        mask = true_day.notna() & pred_day.notna()
        residuals[p] = (true_day[mask] - pred_day[mask]).values

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([residuals[p] for p in box_pvs], labels=box_pvs,
                    patch_artist=True, showfliers=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e76f51')
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', lw=1, ls='--', alpha=0.5)

    ax.set_title(f'NNRF Residuals (Actual − Predicted) — {N_BOXPLOT_PVS} PVs  '
                f'({TARGET.upper()} target, {HOLDOUT_DATE})',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (W/m²)')
    ax.set_xlabel('PV')
    ax.grid(alpha=0.25, axis='y')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    out = FIG_DIR / f"fig_nnrf_{TARGET}_residual_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")

    print(f"\n✓ nnrf_plots.py complete ({TARGET.upper()}, PV={pv})")
    print(f"  Output dir: {FIG_DIR}")