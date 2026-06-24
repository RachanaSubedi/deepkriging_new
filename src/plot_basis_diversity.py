"""
plot_basis_diversity.py

Visualizes spatial diversity coming from the Wendland RBF basis
functions ALONE — using the basis-only ("_nocov") DeepKriging model's
predictions at a single peak-GHI timestep on a chosen day.

Shows that even with zero time-varying covariates, the 411 basis
functions alone produce real PV-to-PV spatial variation (not a flat
field) — confirming that the spatial-diversity advantage documented
in the five-method baseline comparison is attributable to the basis
functions, not the covariate stack.

Inputs expected:
    outputs/predictions_nocov/ghi_pvs.parquet   (from predictnew.py)
    data/raw/pv_nn_assignments.csv              (pv_name, pv_lat, pv_lon)

Run:
    python plot_basis_diversity.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no popup window
import matplotlib.pyplot as plt
from pathlib import Path

# ── PATHS — edit these to match your environment ──────────────
PRED_PARQUET = Path("outputs/predictions_nocov/ghi_pvs.parquet")
PV_CSV       = Path("data/raw/pv_nn_assignments.csv")
OUT_PNG      = Path("outputs/figures_nocov/fig_basis_only_spatial_diversity.png")

# ── Which day/timestep to visualize ────────────────────────────
TARGET_DATE = "2024-03-22"   # set to any date present in your predictions

STATIONS = {
    'S1': {'lat': 46.59,  'lon': -119.150},
    'S2': {'lat': 46.82,  'lon': -119.160},
    'S3': {'lat': 46.82,  'lon': -119.150},
    'P2': {'lat': 46.78,  'lon': -119.228},
}


def main():
    # ── Load predictions ────────────────────────────────────────
    ghi_all = pd.read_parquet(PRED_PARQUET)
    ghi_all.index = ghi_all.index.tz_localize(None)   # fix: align tz with downstream plotting

    # ── Load PV coordinates ──────────────────────────────────────
    pv_df    = pd.read_csv(PV_CSV)
    pv_names = pv_df['pv_name'].tolist()
    lats_map = pv_df.set_index('pv_name')['pv_lat']
    lons_map = pv_df.set_index('pv_name')['pv_lon']

    # ── Pick the peak-GHI timestep on the target day ─────────────
    day      = ghi_all[ghi_all.index.date == pd.Timestamp(TARGET_DATE).date()]
    daytime  = day.dropna(how='all')
    if daytime.empty:
        raise ValueError(f"No daytime predictions found for {TARGET_DATE}")

    row_mean = daytime.mean(axis=1)
    peak_ts  = row_mean.idxmax()
    row      = daytime.loc[peak_ts]

    vals      = row[pv_names].values.astype(float)
    n_unique  = pd.Series(vals).round(1).nunique()
    vmin, vmax = vals.min(), vals.max()

    print(f"Timestamp        : {peak_ts}")
    print(f"Distinct values  : {n_unique} / {len(pv_names)}")
    print(f"Mean / Std       : {vals.mean():.2f} / {vals.std():.3f}")
    print(f"Range            : [{vmin:.2f}, {vmax:.2f}] W/m²")

    # ── Plot ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7.5))

    sc = ax.scatter(lons_map[pv_names], lats_map[pv_names],
                    c=vals, cmap='RdYlGn',
                    vmin=vmin, vmax=vmax,
                    s=90, edgecolors='grey', linewidths=0.4, zorder=3)

    for sname, info in STATIONS.items():
        ax.scatter(info['lon'], info['lat'], marker='*', s=320,
                   color='black', edgecolors='white', lw=0.6, zorder=5)
        ax.annotate(sname, (info['lon'], info['lat']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Predicted GHI (W/m²)', fontsize=10)

    ax.set_title(
        f"Spatial Diversity from Basis Functions Alone (No Covariates)\n"
        f"{peak_ts.strftime('%b %d, %Y %H:%M')} PDT  —  "
        f"{n_unique} distinct values across {len(pv_names)} PVs\n"
        f"mean={vals.mean():.1f}  std={vals.std():.2f}  "
        f"range=[{vmin:.1f}, {vmax:.1f}] W/m²",
        fontsize=11.5, fontweight='bold'
    )
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(alpha=0.2, ls='--')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
