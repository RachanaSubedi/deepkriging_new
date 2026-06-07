"""
figures/plot_goes_pixel_map.py

Generates a map showing:
  - 178 PV locations (all same color)
  - GOES-18 pixel boundary boxes (no centre markers)
  - 4 measurement stations

Run:
    python figures/plot_goes_pixel_map.py

Output:
    outputs/figures/goes_pixel_map.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv(
    r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar"
    r"\data\processed\pv_pixel_map.csv"
)

station_df = df[df['pv_name'].str.startswith('STATION_')].copy()
pv_df      = df[~df['pv_name'].str.startswith('STATION_')].copy()

station_df['label'] = station_df['pv_name'].str.replace('STATION_', '')

unique_pixels    = pv_df['pixel_id'].unique()
unique_pixel_df  = pv_df[['pixel_id', 'pixel_lat', 'pixel_lon']].drop_duplicates()

# ── Plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#f8f8f8')
ax.set_facecolor('#eef2f7')

# 1. PV locations — all same colour
ax.scatter(pv_df['pv_lon'], pv_df['pv_lat'],
           color='#4a90d9',
           s=18, alpha=0.75, zorder=2, linewidths=0)

# 2. GOES pixel boundary boxes only (no centre markers, no labels)
goes_res_lat = 2.0 / 111.0
goes_res_lon = 2.0 / 75.8
for _, row in unique_pixel_df.iterrows():
    rect = mpatches.Rectangle(
        (row['pixel_lon'] - goes_res_lon / 2,
         row['pixel_lat'] - goes_res_lat / 2),
        goes_res_lon, goes_res_lat,
        linewidth=0.6, edgecolor='#555555',
        facecolor='none', linestyle='--', zorder=3, alpha=0.5
    )
    ax.add_patch(rect)

# 3. Stations — dot markers
station_colors = {'S1': '#e63946', 'S2': '#2a9d8f',
                  'S3': '#e9c46a', 'P2': '#9b5de5'}
for _, row in station_df.iterrows():
    ax.scatter(row['pv_lon'], row['pv_lat'],
               marker='.', s=350,
               color=station_colors.get(row['label'], 'red'),
               edgecolors='black', linewidths=0.7,
               zorder=6)
    ax.text(row['pv_lon'] + 0.003, row['pv_lat'] + 0.002,
            row['label'], fontsize=9, fontweight='bold',
            color='black', zorder=7)

# ── Legend ─────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a90d9',
           markersize=7, label='PV location (178 total)'),
    mpatches.Patch(facecolor='none', edgecolor='#555555',
                   linestyle='--', linewidth=1.2,
                   label='GOES-18 pixel boundary (2 km)'),
]
for name, col in station_colors.items():
    legend_elements.append(
        Line2D([0], [0], marker='.', color='w',
               markerfacecolor=col, markeredgecolor='black',
               markersize=13, label=f'Station {name}')
    )
ax.legend(handles=legend_elements, loc='upper left',
          fontsize=8, framealpha=0.9)

# ── Formatting ─────────────────────────────────────────────────
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title(
    'GOES-18 Pixel Assignment for 178 PV Locations\n'
    'IEEE 9500-Node',
    fontsize=12, fontweight='bold'
)
ax.grid(True, linestyle='--', alpha=0.4, color='white')
ax.tick_params(labelsize=9)

plt.tight_layout()

out_dir = Path(r"C:\Users\C838122727\Documents\CSU\research"
               r"\deepkriging_solar\outputs\figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "goes_pixel_map.png"
plt.savefig(out_path, dpi=180, bbox_inches='tight')
print(f"✓ Saved: {out_path}")
plt.show()