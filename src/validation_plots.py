"""
src/validation_plots.py

Generates validation figures from LOSO predictions:

  1. fig_scatter_ghi.png      predicted vs measured GHI, one panel per fold
  2. fig_timeseries.png       sample days time series, one row per station
  3. fig_metrics_bar.png      RMSE / R² bar chart across folds

Reads:
    outputs/validation/fold_{k}_{station}_predictions.csv
    outputs/validation/loso_results.csv

Run:
    python src/validation_plots.py

Outputs → outputs/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import VAL_DIR, FIG_DIR, STATIONS

STATION_NAMES = list(STATIONS.keys())
STATION_COLORS = {'S1': '#e63946', 'S2': '#2a9d8f',
                  'S3': '#e76f51', 'P2': '#264653'}


# ── LOAD PREDICTION FILES ─────────────────────────────────────
def load_fold_predictions():
    """Load all fold prediction CSVs into a dict keyed by station."""
    preds = {}
    for k, station in enumerate(STATION_NAMES):
        fpath = VAL_DIR / f"fold_{k}_{station}_predictions.csv"
        if not fpath.exists():
            print(f"  ⚠ Missing: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df['datetime_local'] = pd.to_datetime(df['datetime_local'], utc=True)
        preds[station] = df
    return preds


# ── FIGURE 1: SCATTER PLOTS ───────────────────────────────────
def plot_scatter(preds, results):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    fig.suptitle('LOSO Validation: Predicted vs Measured GHI '
                 '(held-out station)', fontsize=14, fontweight='bold')

    for ax, station in zip(axes, STATION_NAMES):
        df = preds[station]
        day = df['bg_clearsky'] >= 10
        x = df.loc[day, 'ghi_true'].values
        y = df.loc[day, 'ghi_pred'].values

        ax.scatter(x, y, s=3, alpha=0.15,
                   color=STATION_COLORS[station], rasterized=True)

        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)

        row = results[results['test_station'] == station].iloc[0]
        ax.set_title(f"{station}  (held out)\n"
                     f"R²={row.r2_ghi:.3f}   RMSE={row.rmse_ghi:.1f} W/m²",
                     fontsize=11)
        ax.set_xlabel('Measured GHI (W/m²)')
        if station == STATION_NAMES[0]:
            ax.set_ylabel('Predicted GHI (W/m²)')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = FIG_DIR / "fig_scatter_ghi.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ── FIGURE 2: TIME SERIES (sample days) ───────────────────────
def plot_timeseries(preds, n_days=5):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)
    fig.suptitle(f'LOSO Validation: GHI Time Series '
                 f'({n_days} sample summer days)',
                 fontsize=14, fontweight='bold')

    for ax, station in zip(axes, STATION_NAMES):
        df = preds[station].copy()

        # filter by month/day
        dt = df['datetime_local']
        mask = (dt.dt.month == 7) & (dt.dt.day >= 10) & (dt.dt.day < 10 + n_days)
        window = df[mask].set_index('datetime_local').sort_index()

        ax.plot(window.index, window['ghi_true'], color='black',
                lw=1.4, label='Measured', zorder=3)
        ax.plot(window.index, window['ghi_pred'],
                color=STATION_COLORS[station], lw=1.4, ls='--',
                label='Predicted (DeepKriging)', zorder=2)
        ax.fill_between(window.index, 0, window['bg_clearsky'],
                        color='gold', alpha=0.12, label='Clearsky envelope')

        ax.set_title(f"{station} (held out)", fontsize=11, loc='left')
        ax.set_ylabel('GHI (W/m²)')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    axes[-1].set_xlabel('Date (2024)')
    plt.tight_layout()
    out = FIG_DIR / "fig_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ── FIGURE 3: METRICS BAR CHART ───────────────────────────────
def plot_metrics(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('LOSO Performance by Held-Out Station',
                 fontsize=14, fontweight='bold')

    stations = results['test_station'].tolist()
    colors   = [STATION_COLORS[s] for s in stations]

    # R² GHI
    ax1.bar(stations, results['r2_ghi'], color=colors, alpha=0.85)
    ax1.axhline(results['r2_ghi'].mean(), color='gray',
                ls='--', lw=1, label=f"mean = {results['r2_ghi'].mean():.3f}")
    ax1.set_title('R²  (GHI)')
    ax1.set_ylabel('R²')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.25)
    for i, v in enumerate(results['r2_ghi']):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

    # RMSE GHI
    ax2.bar(stations, results['rmse_ghi'], color=colors, alpha=0.85)
    ax2.axhline(results['rmse_ghi'].mean(), color='gray',
                ls='--', lw=1, label=f"mean = {results['rmse_ghi'].mean():.1f}")
    ax2.set_title('RMSE  (GHI, W/m²)')
    ax2.set_ylabel('RMSE (W/m²)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.25)
    for i, v in enumerate(results['rmse_ghi']):
        ax2.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig_metrics_bar.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  validation_plots.py")
    print("=" * 55)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading prediction files...")
    preds   = load_fold_predictions()
    results = pd.read_csv(VAL_DIR / "loso_results.csv")
    print(f"  Loaded {len(preds)} folds")

    print("\nGenerating figures...")
    plot_scatter(preds, results)
    plot_timeseries(preds)
    plot_metrics(results)

    # Print summary table for convenience
    print("\n── LOSO Summary ──────────────────────────────────")
    print(f"{'Station':<8} {'R²_GHI':>8} {'RMSE_GHI':>10}")
    for _, r in results.iterrows():
        print(f"{r.test_station:<8} {r.r2_ghi:>8.3f} {r.rmse_ghi:>10.1f}")
    print(f"{'Mean':<8} {results.r2_ghi.mean():>8.3f} "
          f"{results.rmse_ghi.mean():>10.1f}")

    print(f"\n✓ All figures saved to {FIG_DIR}")