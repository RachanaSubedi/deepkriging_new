"""
src/validate_qm_accuracy.py

Validates whether the spatial quantile-mapping correction actually
improves accuracy, using a SECOND layer of leave-one-out on top of
DeepKriging's own LOSO folds.

WHY A SECOND LOSO LAYER IS NECESSARY:
Each station's transfer function T_s is fit directly from that
station's own (raw model, measured) pairs. Applying T_S1 to S1's own
raw predictions and checking it against S1's own measurements is
circular -- it mostly just confirms the quantile-mapping fit did what
it was built to do, not whether it generalizes.

The real deployment scenario is the 178 PVs: none of them have their
own ground truth, so they get an IDW-blend of ALL 4 stations'
transfer functions. To honestly test whether that blend helps or
hurts, this script:

  For each station S:
    1. Pretends S has no ground truth (mirrors a real PV).
    2. Builds the correction for S using ONLY the OTHER 3 stations'
       transfer functions, IDW-blended by S's real geographic
       distance to each of them (same idw_weights() logic used for
       the 178 PVs, just re-normalized over 3 stations instead of 4).
    3. Applies that blended correction to S's own raw held-out
       predictions.
    4. Compares RMSE / MAE / bias against S's real measurements, for
       both the corrected and raw (uncorrected baseline) predictions.

This is the honest generalization test: does the blend actually help
at a real point when that point's own ground truth is excluded from
building its correction -- exactly the situation every PV is in.

Also reports the same metric using ALL 4 stations (including S's own
dedicated T_S) for reference/context -- but this is explicitly NOT a
fair generalization test, since S's own information leaks into T_S.

Reuses transfer_functions_qm.npz directly (the saved, POST-SLOPE-CAP
quantile arrays) rather than refitting, so this validates exactly
what's actually deployed to the 178 PVs -- not a re-derived
approximation of it.

Run:
    python src/validate_qm_accuracy.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import VAL_DIR, PRED_DIR, STATIONS, KM_PER_LAT, KM_PER_LON

STATION_NAMES = ['S1', 'S2', 'S3', 'P2']
FOLD_FILES = {
    'S1': VAL_DIR / "fold_0_S1_predictions.csv",
    'S2': VAL_DIR / "fold_1_S2_predictions.csv",
    'S3': VAL_DIR / "fold_2_S3_predictions.csv",
    'P2': VAL_DIR / "fold_3_P2_predictions.csv",
}

CSI_CAP = 1.3
EPS = 1e-4


def to_logit_space(csi):
    kc = np.clip(csi / CSI_CAP, EPS, 1 - EPS)
    return np.log(kc / (1 - kc))


def from_logit_space(t):
    kc = 1.0 / (1.0 + np.exp(-t))
    return kc * CSI_CAP


def dist_km(lat1, lon1, lat2, lon2):
    dlat = (lat1 - lat2) * KM_PER_LAT
    dlon = (lon1 - lon2) * KM_PER_LON
    return np.sqrt(dlat ** 2 + dlon ** 2)


def idw_weights(target_lat, target_lon, src_lats, src_lons, power=1.0):
    d = dist_km(target_lat, target_lon, np.asarray(src_lats), np.asarray(src_lons))
    d = np.where(d < 1e-6, 1e-6, d)
    w = 1.0 / (d ** power)
    return w / w.sum()


def metrics(pred, true):
    err = pred - true
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    return rmse, mae, bias


print("=" * 72)
print("  LOSO accuracy validation — does the QM blend help or hurt "
      "at real stations?")
print("=" * 72)

print("\nLoading saved (post-slope-cap) transfer function quantile arrays...")
tf_npz = np.load(PRED_DIR / "transfer_functions_qm.npz")

# rebuild each station's interpolator directly from the saved, already-
# capped arrays -- guarantees we're testing exactly what's deployed
transfer_fns = {}
for s in STATION_NAMES:
    model_q = tf_npz[f"{s}_model_q"]
    obs_q = tf_npz[f"{s}_obs_q"]
    transfer_fns[s] = interp1d(model_q, obs_q, kind='linear',
                                fill_value='extrapolate', assume_sorted=False)

print("Loading LOSO held-out (model, observed) CSI pairs per station...")
station_data = {}
for s, fpath in FOLD_FILES.items():
    df = pd.read_csv(fpath)
    station_data[s] = (df['csi_pred_raw'].values, df['csi_true'].values)
    print(f"  {s}: {len(df)} held-out samples")

results = []

for s in STATION_NAMES:
    raw_csi, true_csi = station_data[s]
    other_stations = [x for x in STATION_NAMES if x != s]

    # ── Honest test: blend using ONLY the other 3 stations ────────
    other_lats = [STATIONS[o]['lat'] for o in other_stations]
    other_lons = [STATIONS[o]['lon'] for o in other_stations]
    w_loso = idw_weights(STATIONS[s]['lat'], STATIONS[s]['lon'],
                          other_lats, other_lons)

    t_raw = to_logit_space(raw_csi)
    corrected_per_station = np.array([
        transfer_fns[o](t_raw) for o in other_stations
    ])  # shape (3, N)
    t_corrected_loso = w_loso @ corrected_per_station
    corrected_loso = np.clip(from_logit_space(t_corrected_loso), 0.0, CSI_CAP)

    rmse_raw, mae_raw, bias_raw = metrics(raw_csi, true_csi)
    rmse_loso, mae_loso, bias_loso = metrics(corrected_loso, true_csi)

    # ── Reference only: blend using ALL 4 (including S's own T_s) ──
    # NOT a fair generalization test -- S's own info leaks into T_s.
    all_lats = [STATIONS[o]['lat'] for o in STATION_NAMES]
    all_lons = [STATIONS[o]['lon'] for o in STATION_NAMES]
    w_all = idw_weights(STATIONS[s]['lat'], STATIONS[s]['lon'], all_lats, all_lons)
    corrected_per_station_all = np.array([
        transfer_fns[o](t_raw) for o in STATION_NAMES
    ])
    t_corrected_all = w_all @ corrected_per_station_all
    corrected_all = np.clip(from_logit_space(t_corrected_all), 0.0, CSI_CAP)
    rmse_all, mae_all, bias_all = metrics(corrected_all, true_csi)

    results.append({
        'station': s,
        'rmse_raw': rmse_raw, 'rmse_qm_loso': rmse_loso, 'rmse_qm_all4_ref': rmse_all,
        'mae_raw': mae_raw, 'mae_qm_loso': mae_loso,
        'bias_raw': bias_raw, 'bias_qm_loso': bias_loso,
        'rmse_pct_change': 100 * (rmse_loso - rmse_raw) / rmse_raw,
    })

res_df = pd.DataFrame(results)

print("\n" + "=" * 72)
print("RESULTS  (all metrics in CSI units, unitless)")
print("=" * 72)
print(res_df.round(4).to_string(index=False))

print("\n── Interpretation guide ─────────────────────────────────────")
print("  rmse_qm_loso    : honest generalization test (3-station blend,")
print("                    mirrors exactly what a real PV gets)")
print("  rmse_qm_all4_ref: reference only, NOT a fair test (includes S's")
print("                    own dedicated transfer function)")
print("  rmse_pct_change : negative = QM improved accuracy vs. raw baseline")
print("                    positive = QM made accuracy WORSE vs. raw baseline")

n_improved = (res_df['rmse_pct_change'] < 0).sum()
print(f"\n  {n_improved}/4 stations improved (RMSE decreased) under the honest "
      f"leave-one-station-out blend.")
mean_pct = res_df['rmse_pct_change'].mean()
print(f"  Mean RMSE change across all 4 stations: {mean_pct:+.2f}%")

print("\n✓ Validation complete.")
