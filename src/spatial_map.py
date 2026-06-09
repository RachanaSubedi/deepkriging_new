"""
src/spatial_map.py

Spatial GHI visualisation across 178 PV locations.

Produces:
  fig_april30_timeseries.png   all PV predictions vs station measurements Apr 30
  fig_april30_spatial.png      4 spatial snapshots during Apr 30
  fig_spatial_4panel.png       best clear / partly-cloudy / overcast auto-selected

Run:
    python src/spatial_map.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import FIG_DIR, STATIONS

# ── PATHS ─────────────────────────────────────────────────────
PRED_CSV    = Path(__file__).parent.parent / "outputs" / "predictions" / "ghi_pvs.csv"
PV_CSV      = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
STATION_CSV = (Path(__file__).parent.parent / "data" / "raw" / "stations"
               / "all_stations_GHI_30min_PST_filled.csv")

STATION_COLORS = {'S1': '#e63946', 'S2': '#2a9d8f',
                  'S3': '#e76f51', 'P2': '#264653'}

# ── LOAD ─────────────────────────────────────────────────────
print("Loading predictions...")
ghi_all = pd.read_csv(PRED_CSV, index_col='datetime', parse_dates=True)
ghi_all.index = pd.to_datetime(ghi_all.index, format='%m/%d/%Y %H:%M')

pv_df    = pd.read_csv(PV_CSV)
pv_names = pv_df['pv_name'].tolist()
lats_map = pv_df.set_index('pv_name')['pv_lat']
lons_map = pv_df.set_index('pv_name')['pv_lon']

try:
    st = pd.read_csv(STATION_CSV, sep=None, engine='python',
                     encoding='utf-8-sig', index_col=0, parse_dates=True)
    st.index = pd.to_datetime(st.index, format='ISO8601')
    st.columns = [c.replace('GHI_', '') for c in st.columns]
    have_stations = True
    print("  Station data loaded")
except Exception as e:
    print(f"  ⚠ Station data unavailable: {e}")
    have_stations = False

print(f"  GHI shape : {ghi_all.shape}")
print(f"  Date range: {ghi_all.index[0]}  →  {ghi_all.index[-1]}")

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── SPATIAL PLOT HELPER ───────────────────────────────────────
def spatial_ax(ax, row_series, vmin, vmax, title):
    """Plot one spatial snapshot on ax. row_series indexed by pv_name."""
    vals = row_series[pv_names].values.astype(float)

    sc = ax.scatter(lons_map[pv_names], lats_map[pv_names],
                    c=vals, cmap='RdYlGn',
                    vmin=vmin, vmax=vmax,
                    s=65, edgecolors='grey', linewidths=0.3, zorder=3)

    for sname, info in STATIONS.items():
        ax.scatter(info['lon'], info['lat'], marker='*', s=260,
                   color='black', edgecolors='white', lw=0.5, zorder=5)
        ax.annotate(sname, (info['lon'], info['lat']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=7.5, fontweight='bold')

    ax.set_title(f"{title}\nmean={np.nanmean(vals):.0f}  "
                 f"std={np.nanstd(vals):.0f}  max={np.nanmax(vals):.0f} W/m²",
                 fontsize=9)
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    ax.grid(alpha=0.2, ls='--')
    ax.tick_params(labelsize=7.5)
    return sc


# ══════════════════════════════════════════════════════════════
# FIGURE SET 1 — APRIL 30 ANALYSIS
# ══════════════════════════════════════════════════════════════
apr30 = ghi_all[ghi_all.index.date == pd.Timestamp('2024-04-30').date()]
daytime = apr30.dropna(how='all')

print(f"\nApril 30 summary:")
print(f"  Daytime rows  : {len(daytime)}")
print(f"  Max GHI       : {daytime.max().max():.1f} W/m²")
print(f"  Mean daytime  : {daytime.values[~np.isnan(daytime.values)].mean():.1f} W/m²")

# ── Fig 1a: Time series ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

pv_max = daytime.max(axis=1)
pv_min = daytime.min(axis=1)
pv_med = daytime.median(axis=1)

ax.fill_between(daytime.index, pv_min, pv_max,
                color='steelblue', alpha=0.18, label='PV prediction range (min–max)')
ax.plot(daytime.index, pv_med, color='steelblue', lw=2,
        label='PV median prediction', zorder=3)

# Faint individual PV lines
for col in pv_names[::10]:   # every 10th PV for legibility
    ax.plot(daytime.index, daytime[col], color='lightsteelblue',
            lw=0.5, alpha=0.5, zorder=1)

if have_stations:
    apr30_st = st[st.index.date == pd.Timestamp('2024-04-30').date()]
    for s, col in STATION_COLORS.items():
        if s in apr30_st.columns:
            ax.plot(apr30_st.index, apr30_st[s], color=col,
                    lw=2.2, ls='--', label=f'{s} measured', zorder=4)

ax.set_title('April 30, 2024 — Predicted GHI at 178 PV Locations vs Station Measurements\n'
             'Kennewick/Richland, WA  (IEEE 9500-Node S2 Feeder)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Time (PDT)')
ax.set_ylabel('GHI (W/m²)')
ax.legend(loc='upper left', fontsize=8, ncol=4)
ax.grid(alpha=0.25)
ax.set_xlim(daytime.index[0], daytime.index[-1])

out = FIG_DIR / "fig_april30_timeseries.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {out.name}")

# ── Fig 1b: 4 spatial snapshots across April 30 ──────────────
row_std  = daytime.std(axis=1)
row_mean = daytime.mean(axis=1)

morning_ts  = daytime.index[(daytime.index.hour == 9)][0] \
              if (daytime.index.hour == 9).any() else daytime.index[0]
peak_var_ts = row_std.idxmax()
peak_ghi_ts = row_mean.idxmax()
aft_mask    = (daytime.index.hour >= 13) & (daytime.index.hour <= 15)
afternoon_ts = row_mean[aft_mask].idxmax() if aft_mask.any() else daytime.index[-4]

snapshots = [
    (morning_ts,   f"Morning  {morning_ts.strftime('%H:%M')} PDT"),
    (peak_var_ts,  f"Peak Variability  {peak_var_ts.strftime('12:30')} PDT"),
    (peak_ghi_ts,  f"Peak GHI  {peak_ghi_ts.strftime('%H:%M')} PDT"),
    (afternoon_ts, f"Afternoon  {afternoon_ts.strftime('%H:%M')} PDT"),
]

print("\n  Snapshot times on April 30:")
for ts, label in snapshots:
    r = daytime.loc[ts]
    print(f"    {label:45s} mean={r.mean():.0f}  max={r.max():.0f}  std={r.std():.1f}")

all_day_vals = daytime.values[~np.isnan(daytime.values)]
vmax_shared  = np.percentile(all_day_vals, 98)

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
fig.suptitle('April 30, 2024 — Predicted GHI Spatial Distribution\n'
             'IEEE 9500-Node S2 Feeder  (178 PV locations,  colorbar = 0–{:.0f} W/m²)'.format(vmax_shared),
             fontsize=12, fontweight='bold')

for ax, (ts, label) in zip(axes, snapshots):
    sc = spatial_ax(ax, daytime.loc[ts], vmin=0, vmax=vmax_shared, title=label)

cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.7, pad=0.02)
cbar.set_label('GHI (W/m²)', fontsize=11)

out = FIG_DIR / "fig_april30_spatial.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.close()
print(f"  ✓ {out.name}")


# ══════════════════════════════════════════════════════════════
# FIGURE SET 2 — BEST AUTO-SELECTED CLEAR / CLOUDY / OVERCAST
# ══════════════════════════════════════════════════════════════
peak_hours = ghi_all[ghi_all.index.hour.isin(range(9, 16))].dropna(how='all')
rmean = peak_hours.mean(axis=1)
rstd  = peak_hours.std(axis=1)

clear_ts   = rmean[(rmean > 450) & (rstd < 30)].idxmax() \
             if ((rmean > 450) & (rstd < 30)).any() else rmean.idxmax()
partly_ts  = rstd[(rmean > 150) & (rmean < 500)].idxmax()
partly2_ts = rstd[(rmean > 150) & (rmean < 500) &
                  (peak_hours.index.date != partly_ts.date())].idxmax()
over_ts    = rmean[(rmean < 120) & (rmean > 5)].nsmallest(3).index[-1] \
             if ((rmean < 120) & (rmean > 5)).any() else rmean.idxmin()

panels = [
    (clear_ts,   'Clear Sky'),
    (partly_ts,  'Partly Cloudy'),
    (partly2_ts, 'Partly Cloudy (different day)'),
    (over_ts,    'Overcast'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Predicted GHI — Spatial Distribution Under Different Sky Conditions\n'
             'IEEE 9500-Node S2 Feeder  (178 PV locations)',
             fontsize=13, fontweight='bold')

ax_flat = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
for ax, (ts, label) in zip(ax_flat, panels):
    row = peak_hours.loc[ts]
    vals = row[pv_names].values.astype(float)
    finite = vals[np.isfinite(vals)]
    vmin_p = max(0, np.percentile(finite, 3))
    vmax_p = np.percentile(finite, 97)
    ts_str = ts.strftime('%b %d %H:%M PDT') if not hasattr(ts, 'tz_convert') \
             else ts.strftime('%b %d %H:%M')
    sc = spatial_ax(ax, row, vmin=vmin_p, vmax=vmax_p,
                    title=f"{label}\n{ts_str}")

cbar = fig.colorbar(sc, ax=ax_flat, shrink=0.55, pad=0.02)
cbar.set_label('GHI (W/m²)', fontsize=11)

out = FIG_DIR / "fig_spatial_4panel.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {out.name}")

print(f"\n✓ All figures saved to {FIG_DIR}")