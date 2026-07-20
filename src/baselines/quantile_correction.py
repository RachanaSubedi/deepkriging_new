"""
src/spatial_quantile_mapping.py

Spatial quantile-mapping bias correction for DeepKriging's 178-PV field,
adapted from Bailey et al. 2024 ("Adapting Quantile Mapping to Bias
Correct Solar Radiation Data", arXiv:2405.20352).

WHY THIS EXISTS (vs. quantile_correction.py, already in this repo):
quantile_correction.py computes ONE scalar (month, hour) correction
factor from the median measured/predicted ratio and applies it
UNIFORMLY to all 178 PVs — it fixes systematic time-of-day bias but
adds no spatial diversity (every PV gets the same multiplier at a
given timestep). This script implements the actual method from the
paper: full-distribution quantile mapping (not just the median),
done separately per station, then spatially blended — so PVs near
different stations get genuinely different corrections.

ADAPTATION FROM THE PAPER (read this before trusting the output):
Bailey et al. have real observed data (NSRDB) at every model pixel,
so they fit a transfer function T() per pixel using that pixel + its
8 nearest neighbors (all real data). We only have real ground truth
at 4 point stations, not gridded coverage — so a literal per-pixel
fit is impossible here. Instead, following the OTHER validation design
in the same paper (fit T() on years/models with ground truth, apply
out-of-sample to years/models without it — their Section 3.1, second
design), we:

  1. Fit ONE transfer function per real station, using that station's
     own LOSO held-out (predicted, measured) CSI pairs — genuinely
     out-of-sample data already computed by train.py.
  2. For each of the 178 PVs (no ground truth), apply ALL 4 station
     transfer functions to that PV's own raw predicted CSI, then
     IDW-blend the 4 corrected results by geographic distance.

This introduces real spatial diversity because the 4 transfer
functions genuinely differ (S1's bias behaves differently from the
S2/S3/P2 cluster's — already established via LOSO), and PVs at
different locations get different blends of them. No randomness is
injected anywhere — every corrected value traces back to a real
station's observed-vs-predicted relationship.

Clearsky-index + logit transform (matching the paper exactly): CSI
is rescaled by CSI_CAP and logit-transformed before quantile mapping,
so all correction happens in unbounded space, then transformed back —
this replaces the old hand-tuned tiered CSI caps with a principled,
paper-grounded physical bound.

Run:
    python src/spatial_quantile_mapping.py

Outputs (outputs/predictions/):
    ghi_pvs_qm.parquet           (T, 178)  spatially quantile-mapped GHI
    transfer_functions_qm.npz    the 4 fitted per-station T() functions
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

sys.path.append(str(Path(__file__).parent.parent))
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

CSI_CAP     = 1.3        # matches the existing physical ceiling used elsewhere
EPS         = 1e-4       # keeps logit finite at the [0,1] boundary
N_QUANTILES = 99          # matches Bailey et al. exactly: 0.01, 0.02, ..., 0.99
IDW_POWER   = 1.0         # matches idw.py's convention


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
    """CSI -> rescaled-[0,1] -> logit. Matches Bailey et al.'s kc -> logit(kc)."""
    kc = np.clip(csi / CSI_CAP, EPS, 1 - EPS)
    return np.log(kc / (1 - kc))


def from_logit_space(t):
    """Inverse: logit -> rescaled-[0,1] -> CSI."""
    kc = 1.0 / (1.0 + np.exp(-t))
    return kc * CSI_CAP


def fit_transfer_function(model_csi, obs_csi, n_q=N_QUANTILES):
    """
    Non-parametric quantile mapping in logit(clearsky-index) space,
    matching Bailey et al. Section 3 exactly: equally spaced quantiles,
    linear interpolation between them, linear extrapolation beyond the
    boundary quantiles.
    """
    t_model = to_logit_space(model_csi)
    t_obs   = to_logit_space(obs_csi)

    q = np.linspace(0.01, 0.99, n_q)
    model_q = np.quantile(t_model, q)
    obs_q   = np.quantile(t_obs, q)

    # scipy's linear interp1d with fill_value='extrapolate' extends using
    # the slope from the two nearest boundary points — matches the paper's
    # "linearly extrapolate ... based on interpolating the two lowest and
    # two highest ... quantile values"
    T = interp1d(model_q, obs_q, kind='linear', fill_value='extrapolate',
                 assume_sorted=False)
    return T, q, model_q, obs_q


def apply_transfer_function(T, raw_csi):
    t_raw = to_logit_space(raw_csi)
    t_corrected = T(t_raw)
    return from_logit_space(t_corrected)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  spatial_quantile_mapping.py — Bailey et al. adaptation")
    print("=" * 60)

    print("\n[1/5] Loading LOSO held-out (model, observed) CSI pairs...")
    station_pairs = {}
    for s, fpath in FOLD_FILES.items():
        df = pd.read_csv(fpath)
        # csi_pred_raw = raw model output (unclipped), csi_true = measured
        # — exactly the genuinely out-of-sample pair we need
        station_pairs[s] = (df['csi_pred_raw'].values, df['csi_true'].values)
        print(f"  {s}: {len(df)} held-out samples")

    print(f"\n[2/5] Fitting {len(STATION_NAMES)} per-station transfer functions...")
    transfer_fns = {}
    tf_data = {}
    for s in STATION_NAMES:
        model_csi, obs_csi = station_pairs[s]
        T, q, model_q, obs_q = fit_transfer_function(model_csi, obs_csi)
        transfer_fns[s] = T
        tf_data[s] = {'q': q, 'model_q': model_q, 'obs_q': obs_q}
        shift = np.median(obs_q - model_q)
        print(f"  {s}: median logit-space shift = {shift:+.3f}")

    # Save the fitted transfer function data for reuse / inspection
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
    pv_file = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
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

    # ── Quick diagnostics: ramp rate + diversity, vs. baseline ────
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

    # ── Plot: the 4 fitted transfer functions ─────────────────────
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
                 '(Bailey et al. 2024 adaptation)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    out = FIG_DIR / "fig_qm_transfer_functions.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ {out.name}")

    print(f"\n✓ spatial_quantile_mapping.py complete")
    print(f"  Note: this is a SIDE EXPERIMENT, separate from ghi_pvs.parquet.")
    print(f"  Compare ghi_pvs_qm.parquet against the validated baseline before")
    print(f"  deciding whether to fold it into the main pipeline / re-run NNRF on it.")