"""
src/diagnose_qm_extrapolation.py

Diagnoses the Dec 31, 2024 10:00-10:45 PDT overshoot seen in
fig_2024-12-31_qm_vs_baseline_diff.png (QM median hitting +40 to +48
W/m2 above baseline, while stations measured ~90-110 W/m2 and QM
predicted ~130-140).

Checks THREE possible causes, in order:

  1. EXTRAPOLATION: is the raw model prediction (in logit(scaled CSI)
     space) falling outside the quantile range each transfer function
     was actually FIT on? interp1d(..., fill_value='extrapolate')
     will silently extend the boundary slope arbitrarily far outside
     the fitted range -- a modest excursion in logit space can become
     a large excursion in GHI space, especially since a few W/m2 near
     the sensitive part of the logit curve can correspond to a big
     jump.

  2. DOMINANT STATION: for the PVs showing the worst overshoot, which
     of the 4 stations dominates their IDW blend? (Ties back to S1's
     already-flagged tail/plateau reliability concern.)

  3. IN-SAMPLE vs OUT-OF-SAMPLE: was Dec 31, 2024 actually part of the
     LOSO HELD-OUT fold for the dominant station, or is this date
     outside the range the fold file covers (meaning something else
     is going on, e.g. wrong fold being applied)?

Run:
    python src/diagnose_qm_extrapolation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import PRED_DIR, FIG_DIR, STATIONS, KM_PER_LAT, KM_PER_LON

# reuse the actual pipeline's transform + IDW code rather than
# reimplementing it, so this diagnostic is checking the real logic
from spatial_map_qm import (
    to_logit_space, from_logit_space, CSI_CAP, EPS,
    STATION_NAMES, dist_km, idw_weights, FOLD_FILES,
)

# ── CONFIG — the window flagged in the diff plot ──────────────
TARGET_DATE = "2024-12-31"
WINDOW_START = "10:00"
WINDOW_END = "10:45"

QM_DIAG_DIR = FIG_DIR / "qm"
QM_DIAG_DIR.mkdir(parents=True, exist_ok=True)


def load_local(path_or_df, label):
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_parquet(path_or_df)
    df.index = df.index.tz_localize(None) if df.index.tz is None \
        else df.index.tz_convert('America/Los_Angeles').tz_localize(None)
    print(f"  {label}: {df.shape}")
    return df


print("=" * 70)
print("  QM extrapolation diagnostic — Dec 31, 2024 overshoot")
print("=" * 70)

# ──────────────────────────────────────────────────────────────
# STEP 1 — Load raw (pre-QM) CSI predictions + fitted quantile bounds
# ──────────────────────────────────────────────────────────────
print("\n[1/4] Loading raw CSI predictions + fitted transfer function bounds...")

csi_raw_pvs = load_local(PRED_DIR / "csi_pred_raw_pvs.parquet", "Raw CSI predictions (178 PVs)")

tf_npz = np.load(PRED_DIR / "transfer_functions_qm.npz")
station_bounds = {}  # station -> (model_q_min, model_q_max) in logit space
for s in STATION_NAMES:
    model_q = tf_npz[f"{s}_model_q"]
    station_bounds[s] = (model_q.min(), model_q.max())
    print(f"  {s}: fitted logit-space range = [{model_q.min():.3f}, {model_q.max():.3f}]")

pv_file = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
pv_df = pd.read_csv(pv_file)

# ──────────────────────────────────────────────────────────────
# STEP 2 — Isolate the flagged window, convert raw CSI -> logit space
# ──────────────────────────────────────────────────────────────
print(f"\n[2/4] Isolating {TARGET_DATE} {WINDOW_START}-{WINDOW_END} PDT window...")

date_obj = pd.Timestamp(TARGET_DATE).date()
day_raw = csi_raw_pvs[csi_raw_pvs.index.date == date_obj]
window = day_raw.between_time(WINDOW_START, WINDOW_END)

if window.empty:
    raise ValueError(f"No raw predictions found for {TARGET_DATE} {WINDOW_START}-{WINDOW_END}. "
                      f"Available range: {csi_raw_pvs.index[0]} -> {csi_raw_pvs.index[-1]}")

print(f"  {len(window)} timesteps in window")

pv_names = [p for p in pv_df['pv_name'].tolist() if p in window.columns]
src_lats = [STATIONS[s]['lat'] for s in STATION_NAMES]
src_lons = [STATIONS[s]['lon'] for s in STATION_NAMES]

# ──────────────────────────────────────────────────────────────
# STEP 3 — Per-PV: dominant station + out-of-range fraction (weighted
# by whichever station(s) actually drive that PV's blend)
# ──────────────────────────────────────────────────────────────
print("\n[3/4] Checking extrapolation + dominant station per PV...")

results = []
for pv in pv_names:
    row = pv_df[pv_df['pv_name'] == pv].iloc[0]
    w = idw_weights(row['pv_lat'], row['pv_lon'], src_lats, src_lons)
    dominant_idx = int(np.argmax(w))
    dominant_station = STATION_NAMES[dominant_idx]
    dominant_weight = w[dominant_idx]

    raw_csi = window[pv].values.astype(float)
    valid = ~np.isnan(raw_csi)
    if valid.sum() == 0:
        continue
    t_raw = to_logit_space(raw_csi[valid])

    # out-of-range check against EACH station's fitted bounds, weighted
    # by that station's IDW contribution to this PV
    weighted_oor_frac = 0.0
    max_excess = 0.0  # how far past the nearest boundary, in logit units
    for k, s in enumerate(STATION_NAMES):
        lo, hi = station_bounds[s]
        below = t_raw < lo
        above = t_raw > hi
        oor = below | above
        weighted_oor_frac += w[k] * oor.mean()
        if oor.any():
            excess = np.maximum(lo - t_raw[below], np.array([0.0])).max() if below.any() else 0.0
            excess2 = np.maximum(t_raw[above] - hi, np.array([0.0])).max() if above.any() else 0.0
            max_excess = max(max_excess, excess, excess2)

    results.append({
        'pv_name': pv,
        'dominant_station': dominant_station,
        'dominant_weight': dominant_weight,
        'weighted_oor_frac': weighted_oor_frac,
        'max_logit_excess': max_excess,
        't_raw_max': t_raw.max(),
        't_raw_min': t_raw.min(),
    })

res_df = pd.DataFrame(results).sort_values('weighted_oor_frac', ascending=False)

print(f"\n  Top 10 PVs by weighted out-of-range fraction during the window:")
print(res_df.head(10).to_string(index=False))

n_any_oor = (res_df['weighted_oor_frac'] > 0).sum()
print(f"\n  {n_any_oor}/{len(res_df)} PVs have at least some timesteps outside "
      f"the fitted quantile range during this window.")
print(f"  Dominant-station breakdown among top-20 worst PVs:")
print(res_df.head(20)['dominant_station'].value_counts().to_string())

# ──────────────────────────────────────────────────────────────
# STEP 4 — Was Dec 31, 2024 in-sample or out-of-sample for the
# dominant station's LOSO fold?
# ──────────────────────────────────────────────────────────────
print(f"\n[4/4] Checking whether {TARGET_DATE} falls inside each station's "
      f"LOSO held-out fold date range...")

for s, fpath in FOLD_FILES.items():
    fold_df = pd.read_csv(fpath)
    date_col = None
    for cand in ['datetime', 'timestamp', 'date', 'datetime_local']:
        if cand in fold_df.columns:
            date_col = cand
            break
    if date_col is None:
        print(f"  {s}: no recognizable datetime column in {fpath.name} "
              f"(columns: {list(fold_df.columns)}) -- skipping date-range check")
        continue
    # utc=True handles files with mixed/inconsistent tz offsets (e.g. DST
    # transitions straddling the fold); strip tz after for a clean date compare
    fold_dates = pd.to_datetime(fold_df[date_col], utc=True, errors='coerce').dt.tz_localize(None)
    fold_dates = fold_dates.dropna()
    if fold_dates.empty:
        print(f"  {s}: could not parse any dates from column '{date_col}' -- skipping")
        continue
    in_range = (fold_dates.dt.date == date_obj).any()
    print(f"  {s}: fold covers {fold_dates.min().date()} -> {fold_dates.max().date()}  "
          f"| {TARGET_DATE} present in this fold: {in_range}")

# ──────────────────────────────────────────────────────────────
# STEP 5 — Local transfer-function steepness check (the "in-range but
# poorly-conditioned" hypothesis). Even with the raw input safely
# inside the fitted quantile range, interp1d draws a STRAIGHT LINE
# between adjacent fitted quantile points. If two adjacent model_q
# points are close together while their obs_q counterparts are far
# apart, that segment has a very steep slope -- a tiny shift in the
# raw input produces a large shift in the corrected output, with NO
# extrapolation involved.
# ──────────────────────────────────────────────────────────────
print(f"\n[5/5] Checking for locally steep transfer-function segments "
      f"near the observed input range (t_raw in [-1.45, -1.05])...")

# NOTE: after applying the symmetric slope-cap patch to
# spatial_quantile_mapping.py and regenerating transfer_functions_qm.npz,
# every station below should report max local slope <= MAX_SLOPE_CHECK.
MAX_SLOPE_CHECK = 2.0

REGION_LO, REGION_HI = -1.45, -1.05  # covers this window's t_raw_min/max with margin

all_passed = True
for s in STATION_NAMES:
    model_q = tf_npz[f"{s}_model_q"]
    obs_q = tf_npz[f"{s}_obs_q"]
    order = np.argsort(model_q)
    model_q, obs_q = model_q[order], obs_q[order]

    in_region = (model_q >= REGION_LO) & (model_q <= REGION_HI)
    idx = np.where(in_region)[0]
    if len(idx) < 2:
        # widen by one point on each side so we can see the segment
        # bracketing the region even if no quantile point falls inside it
        below = np.where(model_q < REGION_LO)[0]
        above = np.where(model_q > REGION_HI)[0]
        idx = np.array(([below[-1]] if len(below) else []) +
                        list(idx) +
                        ([above[0]] if len(above) else []))
    if len(idx) < 2:
        print(f"  {s}: not enough quantile points near this region to assess slope")
        continue

    seg_model = model_q[idx]
    seg_obs = obs_q[idx]
    slopes = np.diff(seg_obs) / np.diff(seg_model)
    max_slope = np.max(np.abs(slopes))
    steepest_at = seg_model[np.argmax(np.abs(slopes))]

    passed = max_slope <= MAX_SLOPE_CHECK + 1e-6
    all_passed &= passed
    status = "PASS" if passed else f"FAIL (still > {MAX_SLOPE_CHECK}x)"
    print(f"  {s}: max local slope near this region = {max_slope:.2f}x "
          f"(identity = 1.0x), steepest segment starts at model_q={steepest_at:.3f}  [{status}]")

print(f"\n  Overall slope-cap check: {'ALL PASS' if all_passed else 'ONE OR MORE STATIONS STILL EXCEED CAP'}")

# ──────────────────────────────────────────────────────────────
# Diagnostic plot — worst-offending PV's raw t vs station bounds
# ──────────────────────────────────────────────────────────────
if len(res_df) > 0 and res_df.iloc[0]['weighted_oor_frac'] > 0:
    worst_pv = res_df.iloc[0]['pv_name']
    raw_csi = day_raw[worst_pv].dropna()
    t_series = to_logit_space(raw_csi.values)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(raw_csi.index, t_series, color='black', lw=1.6, label=f'{worst_pv} raw (logit space)', zorder=5)

    colors = {'S1': '#e63946', 'S2': '#2a9d8f', 'S3': '#e76f51', 'P2': '#264653'}
    for s in STATION_NAMES:
        lo, hi = station_bounds[s]
        ax.axhspan(lo, hi, color=colors[s], alpha=0.06)
        ax.axhline(lo, color=colors[s], lw=1, ls=':', alpha=0.7)
        ax.axhline(hi, color=colors[s], lw=1, ls=':', alpha=0.7, label=f'{s} fitted range')

    ax.axvspan(pd.Timestamp(f"{TARGET_DATE} {WINDOW_START}"),
               pd.Timestamp(f"{TARGET_DATE} {WINDOW_END}"),
               color='red', alpha=0.08, label='Flagged window')

    ax.set_title(f'Raw prediction (logit space) vs fitted transfer-function bounds\n'
                 f'{worst_pv} — {TARGET_DATE} (worst offender in flagged window)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (PDT)')
    ax.set_ylabel('logit(scaled CSI)')
    ax.legend(fontsize=7.5, ncol=2, loc='upper left')
    ax.grid(alpha=0.25)

    out = QM_DIAG_DIR / f"fig_{TARGET_DATE}_extrapolation_diagnostic.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ {out.relative_to(FIG_DIR.parent)}")
else:
    print("\n  No out-of-range timesteps found -- extrapolation is NOT the cause. "
          "Skipping diagnostic plot; look at IDW weighting or fold-assignment logic instead.")

# ── Zoomed transfer-function plot around the flagged region ──────
fig, ax = plt.subplots(figsize=(9, 7))
colors = {'S1': '#e63946', 'S2': '#2a9d8f', 'S3': '#e76f51', 'P2': '#264653'}
pad = 0.3
zoom_lo, zoom_hi = REGION_LO - pad, REGION_HI + pad

for s in STATION_NAMES:
    model_q = tf_npz[f"{s}_model_q"]
    obs_q = tf_npz[f"{s}_obs_q"]
    order = np.argsort(model_q)
    model_q, obs_q = model_q[order], obs_q[order]
    mask = (model_q >= zoom_lo) & (model_q <= zoom_hi)
    ax.plot(model_q[mask], obs_q[mask], color=colors[s], lw=2, marker='o', ms=5,
            label=f'{s} transfer function')

ax.plot([zoom_lo, zoom_hi], [zoom_lo, zoom_hi], color='grey', lw=1, ls=':', label='Identity')
ax.axvspan(REGION_LO, REGION_HI, color='red', alpha=0.08,
           label=f"This window's raw input range")
ax.set_xlim(zoom_lo, zoom_hi)
ax.set_title(f'Zoomed transfer functions near the {TARGET_DATE} {WINDOW_START}-{WINDOW_END} '
             f'input region\n(look for near-vertical segments = locally steep correction)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Model (predicted) — logit(scaled CSI)')
ax.set_ylabel('Observed (measured) — logit(scaled CSI)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

out_zoom = QM_DIAG_DIR / f"fig_{TARGET_DATE}_transfer_function_zoom.png"
plt.tight_layout()
plt.savefig(out_zoom, dpi=160, bbox_inches='tight')
plt.close()
print(f"\n  ✓ {out_zoom.relative_to(FIG_DIR.parent)}")

print("\n✓ Diagnostic complete.")