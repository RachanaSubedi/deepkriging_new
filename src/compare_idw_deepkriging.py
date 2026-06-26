"""
src/compare_idw_deepkriging.py

Time-series comparison for ONE day: DeepKriging predictions (178 PVs)
vs IDW-of-stations predictions (178 PVs) vs actual station measurements
(S1/S2/S3/P2), all on the same axes.

IDW formula matches diagnose.py's "idw_stations" baseline exactly:
power = 1 (weights = 1/distance, NOT 1/distance^2), straight-line
distance in km using KM_PER_LAT/KM_PER_LON from configs.config,
applied directly to measured GHI (not CSI).

Produces (outputs/figures/):
  fig_{TARGET_DATE}_idw_vs_deepkriging.png

Run:
    python src/compare_idw_deepkriging.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import FIG_DIR, STATIONS, KM_PER_LAT, KM_PER_LON

# ── Which date to plot — change this, nothing else needs editing ──
TARGET_DATE = "2024-03-23"

# ── PATHS ─────────────────────────────────────────────────────
DK_PRED_CSV = Path(__file__).parent.parent / "outputs" / "predictions" / "ghi_pvs_corrected.csv"
PV_CSV      = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
STATION_CSV = (Path(__file__).parent.parent / "data" / "raw" / "stations"
               / "all_stations_GHI_30min_PST_filled.csv")

STATION_COLORS = {'S1': '#e63946', 'S2': '#2a9d8f',
                  'S3': '#e76f51', 'P2': '#264653'}
STATION_ORDER  = ['S1', 'S2', 'S3', 'P2']


def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * KM_PER_LAT
    dlon = (lon1 - lon2) * KM_PER_LON
    return np.sqrt(dlat**2 + dlon**2)


# ── LOAD: DeepKriging predictions ─────────────────────────────
print("Loading DeepKriging predictions...")
dk_all = pd.read_csv(DK_PRED_CSV, index_col='datetime', parse_dates=True)
dk_all.index = pd.to_datetime(dk_all.index, format='%m/%d/%Y %H:%M')

pv_df    = pd.read_csv(PV_CSV)
pv_names = pv_df['pv_name'].tolist()

# ── LOAD: station measurements ────────────────────────────────
print("Loading station measurements...")
st = pd.read_csv(STATION_CSV, sep=None, engine='python',
                 encoding='utf-8-sig', index_col=0, parse_dates=True)
st.index = (pd.to_datetime(st.index)
            .tz_localize('Etc/GMT+8')
            .tz_convert('America/Los_Angeles')
            .tz_localize(None))
st.columns = [c.replace('GHI_', '') for c in st.columns]
st = st[~st.index.duplicated(keep='first')]   # drop DST fall-back duplicate rows (Nov 3, 2024 01:00/01:30)

# ── COMPUTE: IDW field at all 178 PVs (power=1, matches diagnose.py) ──
print("Computing IDW field at 178 PV locations (power=1)...")
weight_rows = []
for _, row in pv_df.iterrows():
    dists = np.array([
        dist_km(row['pv_lat'], row['pv_lon'],
                STATIONS[s]['lat'], STATIONS[s]['lon'])
        for s in STATION_ORDER
    ])
    w = 1.0 / dists
    w /= w.sum()
    weight_rows.append(w)
W = np.array(weight_rows)  # (178, 4)

common_idx = st.index.intersection(dk_all.index)
station_arr = st.loc[common_idx, STATION_ORDER].values  # (T, 4)
idw_arr = station_arr @ W.T                              # (T, 178)
idw_all = pd.DataFrame(idw_arr, index=common_idx, columns=pv_names)

print(f"  DeepKriging shape : {dk_all.shape}")
print(f"  IDW shape         : {idw_all.shape}")
print(f"  Station shape     : {st.shape}")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PLOT: single day, all three sources overlaid
# ══════════════════════════════════════════════════════════════
target_date_obj = pd.Timestamp(TARGET_DATE).date()
date_tag = TARGET_DATE

dk_day  = dk_all[dk_all.index.date == target_date_obj].dropna(how='all')
idw_day = idw_all[idw_all.index.date == target_date_obj]
st_day  = st[st.index.date == target_date_obj]

if dk_day.empty:
    raise ValueError(
        f"No DeepKriging predictions found for {TARGET_DATE}. "
        f"Available range: {dk_all.index[0].date()} to {dk_all.index[-1].date()}"
    )

print(f"\n{TARGET_DATE} summary:")
print(f"  DeepKriging max GHI : {dk_day.max().max():.1f} W/m²")
print(f"  IDW max GHI         : {idw_day.max().max():.1f} W/m²")

fig, ax = plt.subplots(figsize=(14, 5.5))

# DeepKriging: faint individual PV lines (subsampled) + median
for col in pv_names[::8]:
    ax.plot(dk_day.index, dk_day[col], color='steelblue',
            lw=0.6, alpha=0.4, zorder=1)
dk_med = dk_day.median(axis=1)
ax.plot(dk_day.index, dk_med, color='steelblue', lw=2.6,
        label='DeepKriging (PV median)', zorder=3)

# IDW: faint individual PV lines (subsampled) + median
for col in pv_names[::8]:
    ax.plot(idw_day.index, idw_day[col], color='darkorange',
            lw=0.6, alpha=0.4, zorder=1)
idw_med = idw_day.median(axis=1)
ax.plot(idw_day.index, idw_med, color='darkorange', lw=2.6, ls='-.',
        label='IDW (PV median)', zorder=3)

# Station measurements
for s in STATION_ORDER:
    if s in st_day.columns:
        ax.plot(st_day.index, st_day[s], color=STATION_COLORS[s],
                lw=2.0, ls='--', label=f'{s} measured', zorder=4)

ax.set_title(f'{pd.Timestamp(TARGET_DATE).strftime("%B %d, %Y")} — '
             f'DeepKriging vs IDW vs Station Measurements (178 PV locations)\n',
             fontsize=12.5, fontweight='bold')
ax.set_xlabel('Time (PDT)')
ax.set_ylabel('GHI (W/m²)')
ax.legend(loc='upper left', fontsize=8.5, ncol=3)
ax.grid(alpha=0.25)
if len(dk_day):
    ax.set_xlim(dk_day.index[0], dk_day.index[-1])

out = FIG_DIR / f"fig_{date_tag}_idw_vs_deepkriging.png"
plt.tight_layout()
plt.savefig(out, dpi=160, bbox_inches='tight')
plt.close()
print(f"\n✓ {out}")