"""
src/spatial_qm.py

Spatial quantile-mapping bias correction for DeepKriging's 178-PV field,
adapted from Bailey et al. 2024 ("Adapting Quantile Mapping to Bias
Correct Solar Radiation Data", arXiv:2405.20352).

Includes a symmetric slope-cap safeguard (see cap_transfer_function_slope)
found necessary after diagnose_qm_extrapolation.py traced a Dec 31, 2024
GHI overshoot to locally-steep-but-in-range transfer function segments.

Run:
    python src/correction/spatial_qm.py

Outputs (outputs/predictions/):
    ghi_pvs_qm.parquet           (T, 178)  spatially quantile-mapped GHI
    transfer_functions_qm.npz    the 4 fitted, slope-capped per-station T() functions

Outputs (outputs/figures/correction/):
    fig_qm_transfer_functions.png  diagnostic plot of the 4 T() curves
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    VAL_DIR, PRED_DIR, BG_DIR, FIG_DIR,
    STATIONS, KM_PER_LAT, KM_PER_LON,
)

# ── CONFIG ────────────────────────────────────────────────────
STATION_NAMES = ['S1', 'S2', 'S3', 'P2']
FOLD_FILES = {
    'S1': VAL_DIR / "fold_0_S1_predictions.csv",
    'S2': VAL_DIR / "fold_1_S2_predictions.csv",
    'S3': VAL_DIR / "fold_2_S3_predictions.csv",
    'P2': VAL_DIR / "fold_3_P2_predictions.csv",
}

CSI_CAP     = 1.3
EPS         = 1e-4
N_QUANTILES = 99
IDW_POWER   = 1.0

# ── NEW: slope-cap safeguard ──────────────────────────────────
# Caps how steeply the transfer function can amplify any single
# quantile-to-quantile segment. Found via diagnose_qm_extrapolation.py:
# the Dec 31 2024 10:00-10:45 overshoot traced to segments running
# 2.6x-4.5x steeper than identity in the partly-cloudy CSI band
# (t_raw ~ -1.45 to -1.05), shared across all 4 stations -- not an
# extrapolation issue, a locally-steep-in-range issue.
MAX_SLOPE = 2.0


def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * KM_PER_LAT
    dlon = (lon1 - lon2) * KM_PER_LON
    return np.sqrt(dlat ** 2 + dlon ** 2)


def idw_weights(target_lat, target_lon, src_lats, src_lons, power=IDW_POWER):
    d = dist_km(target_lat, target_lon, np.asarray(src_lats), np.asarray(src_lons))
    d = np.where(d < 1e-6, 1e-6, d)
    w = 1.0 / (d ** power)
    return w / w.sum()


def to_logit_space(csi):
    kc = np.clip(csi / CSI_CAP, EPS, 1 - EPS)
    return np.log(kc / (1 - kc))


def from_logit_space(t):
    kc = 1.0 / (1.0 + np.exp(-t))
    return kc * CSI_CAP


def fit_transfer_function(model_csi, obs_csi, n_q=N_QUANTILES):
    """
    Non-parametric quantile mapping in logit(clearsky-index) space,
    matching Bailey et al. Section 3: equally spaced quantiles, linear
    interpolation between them, linear extrapolation beyond the
    boundary quantiles.

    NEW: after computing the raw empirical quantile pairs, caps any
    segment steeper than MAX_SLOPE before building the interpolator.
    See cap_transfer_function_slope() for the mechanism.
    """
    t_model = to_logit_space(model_csi)
    t_obs   = to_logit_space(obs_csi)

    q = np.linspace(0.01, 0.99, n_q)
    model_q = np.quantile(t_model, q)
    obs_q   = np.quantile(t_obs, q)

    obs_q_capped, max_slope_before, max_slope_after, start_drift, end_drift = \
        cap_transfer_function_slope(model_q, obs_q, MAX_SLOPE)

    T = interp1d(model_q, obs_q_capped, kind='linear', fill_value='extrapolate',
                 assume_sorted=False)
    return T, q, model_q, obs_q_capped, max_slope_before, max_slope_after, start_drift, end_drift


def _cap_one_direction(model_q, obs_q, max_slope, forward):
    """One-directional cumulative slope cap, anchored at whichever end
    forward selects. Helper for the symmetric version below."""
    dx = np.diff(model_q)
    dy = np.diff(obs_q)
    slopes = np.clip(dy / dx, 0.0, max_slope)
    out = np.empty_like(obs_q)
    if forward:
        out[0] = obs_q[0]
        out[1:] = obs_q[0] + np.cumsum(slopes * dx)
    else:
        out[-1] = obs_q[-1]
        out[:-1] = obs_q[-1] - np.cumsum((slopes * dx)[::-1])[::-1]
    return out


def cap_transfer_function_slope(model_q, obs_q, max_slope):
    """
    Enforces a maximum local slope on a monotone empirical transfer
    function via SYMMETRIC (two-directional) cumulative slope-capping.

    A single-direction cumulative cap (anchor at the first point, walk
    forward) fixes steep segments but "runs out of budget" afterward --
    every quantile past the capped region inherits a permanent one-
    sided offset (found empirically: P2's whole upper-range correction
    shifted down after a forward-only cap, since P2 had the steepest
    original slope and paid the largest ongoing drift).

    Fix: run the cap forward (anchored at the first point) AND
    backward (anchored at the last point), then average the two
    reconstructions. Averaging two slope-capped monotone sequences:
      - stays monotone (average of non-decreasing functions is
        non-decreasing)
      - still respects max_slope (average of two slopes each <=
        max_slope is <= max_slope)
      - splits any unavoidable drift evenly across BOTH endpoints
        instead of dumping it all on one end

    Returns: (obs_q_capped, max_slope_before, max_slope_after,
              start_drift, end_drift)
    """
    order = np.argsort(model_q)
    model_q = np.asarray(model_q)[order]
    obs_q = np.asarray(obs_q)[order]

    dx = np.diff(model_q)
    dy = np.diff(obs_q)
    max_slope_before = np.max(np.abs(dy / dx))

    fwd = _cap_one_direction(model_q, obs_q, max_slope, forward=True)
    bwd = _cap_one_direction(model_q, obs_q, max_slope, forward=False)
    obs_q_capped = (fwd + bwd) / 2.0

    max_slope_after = np.max(np.abs(np.diff(obs_q_capped) / dx))
    start_drift = obs_q_capped[0] - obs_q[0]
    end_drift = obs_q_capped[-1] - obs_q[-1]

    return obs_q_capped, max_slope_before, max_slope_after, start_drift, end_drift


def apply_transfer_function(T, raw_csi):
    t_raw = to_logit_space(raw_csi)
    t_corrected = T(t_raw)
    return from_logit_space(t_corrected)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  spatial_qm.py — Bailey et al. adaptation")
    print("  (with slope-cap safeguard, MAX_SLOPE =", MAX_SLOPE, ")")
    print("=" * 60)

    print("\n[1/5] Loading LOSO held-out (model, observed) CSI pairs...")
    station_pairs = {}
    for s, fpath in FOLD_FILES.items():
        df = pd.read_csv(fpath)
        station_pairs[s] = (df['csi_pred_raw'].values, df['csi_true'].values)
        print(f"  {s}: {len(df)} held-out samples")

    print(f"\n[2/5] Fitting {len(STATION_NAMES)} per-station transfer functions...")
    transfer_fns = {}
    tf_data = {}
    for s in STATION_NAMES:
        model_csi, obs_csi = station_pairs[s]
        T, q, model_q, obs_q, slope_before, slope_after, start_drift, end_drift = fit_transfer_function(model_csi, obs_csi)
        transfer_fns[s] = T
        tf_data[s] = {'q': q, 'model_q': model_q, 'obs_q': obs_q}
        shift = np.median(obs_q - model_q)
        cap_note = (f"  [capped {slope_before:.2f}x -> {slope_after:.2f}x, "
                    f"start drift={start_drift:+.3f}, end drift={end_drift:+.3f}]") \
                   if slope_before > MAX_SLOPE else ""
        print(f"  {s}: median logit-space shift = {shift:+.3f}{cap_note}")

    np.savez(
        PRED_DIR / "transfer_functions_qm.npz",
        **{f"{s}_q": tf_data[s]['q'] for s in STATION_NAMES},
        **{f"{s}_model_q": tf_data[s]['model_q'] for s in STATION_NAMES},
        **{f"{s}_obs_q": tf_data[s]['obs_q'] for s in STATION_NAMES},
    )
    print(f"  ✓ transfer_functions_qm.npz saved")

    print("\n[3/5] Loading 178-PV raw predictions...")
    csi_raw_pvs = pd.read_parquet(PRED_DIR / "csi_pred_raw_pvs.parquet")
    bg_clearsky = pd.read_parquet(BG_DIR / "bg_clearsky_pvs.parquet")
    pv_file = Path(__file__).parent.parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df = pd.read_csv(pv_file)
    pv_names = list(csi_raw_pvs.columns)

    common = csi_raw_pvs.index.intersection(bg_clearsky.index)
    csi_raw_pvs = csi_raw_pvs.loc[common]
    bg_clearsky = bg_clearsky.loc[common]
    print(f"  {len(pv_names)} PVs, {len(common)} common timesteps")

    print("\n[4/5] Applying + IDW-blending per-PV corrections...")
    src_lats = [STATIONS[s]['lat'] for s in STATION_NAMES]
    src_lons = [STATIONS[s]['lon'] for s in STATION_NAMES]

    csi_corrected = pd.DataFrame(index=common, columns=pv_names, dtype=np.float32)

    for j, pv in enumerate(pv_names):
        row = pv_df[pv_df['pv_name'] == pv].iloc[0]
        w = idw_weights(row['pv_lat'], row['pv_lon'], src_lats, src_lons)

        raw = csi_raw_pvs[pv].values.astype(np.float64)
        day_mask = ~np.isnan(raw)

        blended = np.full(len(raw), np.nan)
        station_corrected = np.zeros((day_mask.sum(), len(STATION_NAMES)))
        for k, s in enumerate(STATION_NAMES):
            station_corrected[:, k] = apply_transfer_function(
                transfer_fns[s], raw[day_mask])

        blended[day_mask] = station_corrected @ w
        csi_corrected[pv] = np.clip(blended, 0.0, CSI_CAP)

        if (j + 1) % 30 == 0 or j == len(pv_names) - 1:
            print(f"  {j+1:>3}/{len(pv_names)} PVs done")

    print("\n[5/5] Converting to GHI and saving...")
    ghi_qm = csi_corrected * bg_clearsky[pv_names]
    ghi_qm.index.name = 'datetime_local'
    ghi_qm.to_parquet(PRED_DIR / "ghi_pvs_qm.parquet")
    print(f"  ✓ ghi_pvs_qm.parquet  {ghi_qm.shape}")

    ghi_baseline = pd.read_parquet(PRED_DIR / "ghi_pvs.parquet").loc[common, pv_names]

    ramp_qm  = ghi_qm.diff().abs().mean().mean()
    ramp_base = ghi_baseline.diff().abs().mean().mean()

    day = ghi_qm[ghi_qm.mean(axis=1) > 100]
    div_qm = day.round(0).nunique(axis=1).mean() if len(day) else float('nan')
    day_base = ghi_baseline[ghi_baseline.mean(axis=1) > 100]
    div_base = day_base.round(0).nunique(axis=1).mean() if len(day_base) else float('nan')

    means = ghi_qm.mean()
    spread_qm = means.max() - means.min()

    print(f"\n── Comparison vs. uncorrected baseline ─────────────────")
    print(f"  Ramp rate       : baseline={ramp_base:.2f}  qm-corrected={ramp_qm:.2f} W/m²")
    print(f"  Diversity       : baseline={div_base:.1f}/178  qm-corrected={div_qm:.1f}/178")
    print(f"  Per-PV spread   : {spread_qm:.2f} W/m²")

    FIG_DIR = FIG_DIR / "correction"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'S1': '#e63946', 'S2': '#2a9d8f', 'S3': '#e76f51', 'P2': '#264653'}
    lims = [-6, 6]
    ax.plot(lims, lims, color='grey', lw=1, ls=':', label='Identity (no correction)')
    for s in STATION_NAMES:
        ax.plot(tf_data[s]['model_q'], tf_data[s]['obs_q'],
                color=colors[s], lw=2, marker='o', ms=3, label=f'{s} transfer function')
    ax.set_xlabel('Model (predicted) — logit(scaled CSI)')
    ax.set_ylabel('Observed (measured) — logit(scaled CSI)')
    ax.set_title('Per-Station Quantile-Mapping Transfer Functions\n'
                 '(Bailey et al. 2024 adaptation, slope-capped)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    out = FIG_DIR / "fig_qm_transfer_functions.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ {out.name}")

    print(f"\n✓ spatial_qm.py complete")
    print(f"  KNOWN TRADEOFF (see src/correction/validate_qm_accuracy.py):")
    print(f"  this correction restores spatial diversity DeepKriging's mean-")
    print(f"  reversion had collapsed, but the honest leave-one-station-out")
    print(f"  test showed it does NOT improve, and can worsen, point-in-time")
    print(f"  RMSE at the 4 real stations (worst for S1, the geographic and")
    print(f"  bias outlier). Decide whether ghi_pvs_qm.parquet feeds Stage 2")
    print(f"  NNRF downscaling with this tradeoff in mind, not as a strict")
    print(f"  accuracy improvement over ghi_pvs.parquet.")