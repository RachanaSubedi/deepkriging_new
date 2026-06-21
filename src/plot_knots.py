import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import(
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    KM_PER_LAT, KM_PER_LON, BASIS_LEVELS,
    STATIONS, FIG_DIR,
)

LEVEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
LEVEL_SIZES = [120, 60, 25, 10]

domain = {'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
          'lon_min': LON_MIN, 'lon_max': LON_MAX}

knot_levels = []

for lv in BASIS_LEVELS:
    step_lat = lv['spacing_km']/KM_PER_LAT
    step_lon = lv['spacing_km']/KM_PER_LON
    lat_k = np.arange(LAT_MIN, LAT_MAX + step_lat * 0.5, step_lat)
    lon_k = np.arange(LON_MIN, LON_MAX + step_lon * 0.5, step_lon)
    lg, ng = np.meshgrid(lat_k, lon_k, indexing='ij')
    knots = np.column_stack([lg.ravel(), ng.ravel()])
    knot_levels.append((knots, lv['theta_km'], lv['spacing_km']))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             gridspec_kw={'width_ratios': [1.8, 1]})

    fig.suptitle('Multi-Resolution Wendland RBF Knot Grid\n'
                 'DeepKriging Basis Functions — IEEE 9500-Node S2 Feeder Domain',
                 fontsize=13, fontweight='bold')

    # ── LEFT: all levels overlaid ─────────────────────────────────
    ax = axes[0]
    legend_handles = []

    for i, (knots, theta_km, spacing_km) in enumerate(knot_levels):
        color = LEVEL_COLORS[i]
        size = LEVEL_SIZES[i]
        label = (f"Level {i + 1}  (spacing={spacing_km} km, "
                 f"θ={theta_km} km,  {len(knots)} knots)")

        ax.scatter(knots[:, 1], knots[:, 0],
                   c=color, s=size, zorder=3 + i,
                   edgecolors='white', linewidths=0.4,
                   label=label, alpha=0.85)
        legend_handles.append(
            mpatches.Patch(color=color, label=label)
        )

    # Domain boundary
    rect = mpatches.FancyBboxPatch(
        (LON_MIN, LAT_MIN),
        LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
        boxstyle="square,pad=0",
        linewidth=1.5, edgecolor='black', facecolor='none',
        linestyle='--', zorder=1
    )
    ax.add_patch(rect)

    # Stations
    for sname, info in STATIONS.items():
        ax.scatter(info['lon'], info['lat'], marker='*',
                   s=350, color='black', edgecolors='white',
                   linewidths=0.7, zorder=10)
        ax.annotate(sname, (info['lon'], info['lat']),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_title('All Levels Overlaid  (★ = measurement station)', fontsize=11)
    ax.legend(handles=legend_handles, loc='lower right',
              fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.2, linestyle='--')

    total_knots = sum(len(k) for k, _, _ in knot_levels)
    ax.text(0.02, 0.98, f'Total knots: {total_knots}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # ── RIGHT: one panel per level (2×2 sub-grid in one axis) ────
    ax2 = axes[1]
    ax2.axis('off')

    # Draw 4 mini subplots manually
    sub_positions = [(0.0, 0.5, 0.5, 0.5),  # L1: left-top
                     (0.5, 0.5, 0.5, 0.5),  # L2: right-top
                     (0.0, 0.0, 0.5, 0.5),  # L3: left-bottom
                     (0.5, 0.0, 0.5, 0.5)]  # L4: right-bottom

    fig_pos = ax2.get_position()

    for i, (knots, theta_km, spacing_km) in enumerate(knot_levels):
        x0_rel, y0_rel, w_rel, h_rel = sub_positions[i]

        left = fig_pos.x0 + x0_rel * fig_pos.width
        bottom = fig_pos.y0 + y0_rel * fig_pos.height
        width = w_rel * fig_pos.width
        height = h_rel * fig_pos.height

        sub = fig.add_axes([left, bottom, width, height])

        sub.scatter(knots[:, 1], knots[:, 0],
                    c=LEVEL_COLORS[i], s=max(LEVEL_SIZES[i], 8),
                    edgecolors='white', linewidths=0.3, alpha=0.9)

        # Draw support radius circles for a few central knots
        center_idx = len(knots) // 2
        for ci in [center_idx, center_idx + 2, center_idx - 2]:
            ci = max(0, min(ci, len(knots) - 1))
            theta_lat = theta_km / KM_PER_LAT
            theta_lon = theta_km / KM_PER_LON
            ell = mpatches.Ellipse(
                (knots[ci, 1], knots[ci, 0]),
                width=2 * theta_lon, height=2 * theta_lat,
                linewidth=0.8, edgecolor=LEVEL_COLORS[i],
                facecolor=LEVEL_COLORS[i], alpha=0.08, zorder=1
            )
            sub.add_patch(ell)

        # Stations
        for sname, info in STATIONS.items():
            sub.scatter(info['lon'], info['lat'], marker='*',
                        s=80, color='black', zorder=5)

        sub.set_xlim(LON_MIN - 0.01, LON_MAX + 0.01)
        sub.set_ylim(LAT_MIN - 0.01, LAT_MAX + 0.01)
        sub.set_title(f"Level {i + 1}  |  {spacing_km} km  |  "
                      f"θ={theta_km} km  |  {len(knots)} knots",
                      fontsize=7.5, color=LEVEL_COLORS[i],
                      fontweight='bold', pad=2)
        sub.tick_params(labelsize=5.5)
        sub.grid(alpha=0.15, linestyle='--')

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_knot_grid.png"
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {out}")

    # Print summary
    print("\nKnot grid summary:")
    print(f"{'Level':<8} {'Spacing':>10} {'θ (km)':>8} {'Knots':>8}")
    print("─" * 38)
    for i, (knots, theta_km, spacing_km) in enumerate(knot_levels):
        print(f"  {i + 1:<6} {spacing_km:>8.1f} km {theta_km:>8.1f}   {len(knots):>6}")
    print(f"  {'Total':30} {sum(len(k) for k, _, _ in knot_levels):>6}")