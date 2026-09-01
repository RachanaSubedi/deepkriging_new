"""Generate 178-PV CSI/GHI fields with the final DK-6 production models.

Place at:
    src/predict/predict_dk6_pvs.py

Run after train_dk6_production.py:
    python src/predict/predict_dk6_pvs.py

The feature contract exactly matches training_matrix.py:
    [TPS K=16, coarse temporal K=16, existing 15 covariates]

Three raw-CSI predictions are retained. Their arithmetic mean is computed
before applying the common DK-0 low-sun blend/caps exactly once. Seed 42 is
also postprocessed separately, allowing later comparison of a controlled
single model against the three-seed production ensemble.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import (  # noqa: E402
    BG_DIR,
    C13_FEAT_DIR,
    DROPOUT,
    HIDDEN_SIZE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TRAIN_DIR,
)
from src.model import DeepKriging  # noqa: E402
from src.train.train_dk0_matched import (  # noqa: E402
    CLEARSKY_MIN,
    apply_dk0_postprocessing,
)
from src.train.train_dk2_spatial_basis import (  # noqa: E402
    TPS_DIR,
    load_temporal_artifact,
    select_temporal_levels,
)
from src.data_prep.temporal_basis import transform_temporal_basis  # noqa: E402


SEEDS = (42, 123, 2026)
N_SPATIAL_BASIS = 16
SELECTED_TEMPORAL_LEVELS = ("coarse_30d",)
N_COVARIATES = 15
INPUT_DIMENSION = 47
DEVICE = torch.device("cpu")

PRODUCTION_DIR = (
    Path(OUTPUT_DIR)
    / "experiments"
    / "DK6_production_direct_TPS_K016_three_seed"
)
MODEL_DIR = PRODUCTION_DIR / "models"
OUT_DIR = PRODUCTION_DIR / "predictions"

EXPECTED_COVARIATES = [
    "bg_csi",
    "bg_csi_lag30",
    "bg_csi_diff",
    "clearsky_frac",
    "cos_zenith",
    "bt_norm",
    "c02_norm",
    "temperature",
    "rh",
    "pressure",
    "elevation",
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
]

MET_NORM = {
    "temperature": (15.0, 20.0),
    "rh": (70.0, 30.0),
    "pressure": (990.0, 20.0),
    "cos_zenith": (0.0, 1.0),
    "elevation": (300.0, 100.0),
}
MET_KEYS = ("temperature", "rh", "pressure", "cos_zenith")


def norm(values: np.ndarray, reference: float, scale: float) -> np.ndarray:
    return (values - reference) / scale


def validate_training_feature_contract() -> None:
    feature_path = Path(TRAIN_DIR) / "feature_names.txt"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    names = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    if len(names) < N_COVARIATES:
        raise ValueError("feature_names.txt is shorter than the covariate block")
    observed = names[-N_COVARIATES:]
    if observed != EXPECTED_COVARIATES:
        raise ValueError(
            "PV inference covariate order disagrees with training_matrix.py.\n"
            f"Expected: {EXPECTED_COVARIATES}\nObserved: {observed}"
        )


def load_state_dict_safely(path: Path) -> dict:
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_models() -> list[DeepKriging]:
    models: list[DeepKriging] = []
    for seed in SEEDS:
        path = MODEL_DIR / f"production_seed{seed}_best.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = DeepKriging(INPUT_DIMENSION, HIDDEN_SIZE, DROPOUT).to(DEVICE)
        model.load_state_dict(load_state_dict_safely(path))
        model.eval()
        models.append(model)
    return models


def predict_numpy(model: DeepKriging, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32, copy=False)).to(DEVICE)
        return model(tensor).cpu().numpy().astype(np.float32, copy=False)


def build_pv_covariates(
    timestamps: pd.DatetimeIndex,
    bg_csi: np.ndarray,
    bg_clearsky: np.ndarray,
    c13_norm: np.ndarray,
    c02_norm: np.ndarray,
    met: dict[str, np.ndarray],
    elevation: float,
) -> np.ndarray:
    """Build the exact 15-column training covariate block for one PV."""
    bg = bg_csi.astype(np.float32, copy=False)
    bg_lag = np.concatenate(
        [np.array([np.nan], dtype=np.float32), bg[:-1]]
    )
    bg_diff = bg - bg_lag

    clear_series = pd.Series(bg_clearsky, index=timestamps)
    daily_max = clear_series.groupby(clear_series.index.date).transform("max").to_numpy()
    daily_max = np.where(daily_max < 1.0, 1.0, daily_max)
    clear_fraction = (bg_clearsky / daily_max).astype(np.float32)

    doy = timestamps.day_of_year.to_numpy()
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    doy_sin = np.sin(2.0 * np.pi * doy / 365.0).astype(np.float32)
    doy_cos = np.cos(2.0 * np.pi * doy / 365.0).astype(np.float32)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    elevation_array = np.full(len(timestamps), elevation, dtype=np.float32)

    return np.column_stack(
        [
            bg,
            bg_lag,
            bg_diff,
            clear_fraction,
            norm(met["cos_zenith"], *MET_NORM["cos_zenith"]),
            c13_norm.astype(np.float32, copy=False),
            c02_norm.astype(np.float32, copy=False),
            norm(met["temperature"], *MET_NORM["temperature"]),
            norm(met["rh"], *MET_NORM["rh"]),
            norm(met["pressure"], *MET_NORM["pressure"]),
            norm(elevation_array, *MET_NORM["elevation"]),
            doy_sin,
            doy_cos,
            hour_sin,
            hour_cos,
        ]
    ).astype(np.float32)


def save_frame(array: np.ndarray, index: pd.DatetimeIndex, columns: list[str], name: str) -> None:
    frame = pd.DataFrame(array, index=index, columns=columns)
    frame.index.name = "datetime_local"
    path = OUT_DIR / f"{name}.parquet"
    frame.to_parquet(path)
    print(f"  saved {name}.parquet {frame.shape}")


def main() -> None:
    print("=" * 78)
    print("DK-6 THREE-SEED PRODUCTION INFERENCE AT 178 PV LOCATIONS")
    print("=" * 78)
    validate_training_feature_contract()

    run_config_path = PRODUCTION_DIR / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(run_config_path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if run_config.get("n_spatial_basis") != N_SPATIAL_BASIS:
        raise ValueError("Production run_config does not specify TPS K=16")
    if run_config.get("selected_temporal_levels") != list(SELECTED_TEMPORAL_LEVELS):
        raise ValueError("Production run_config does not specify coarse_30d only")

    models = load_models()
    scaler_mean = np.load(MODEL_DIR / "production_scaler_mean.npy")
    scaler_std = np.load(MODEL_DIR / "production_scaler_std.npy")
    if scaler_mean.shape != (INPUT_DIMENSION,) or scaler_std.shape != (INPUT_DIMENSION,):
        raise ValueError("Production scaler does not have 47 columns")

    phi_path = Path(TPS_DIR) / "Phi_pvs.npy"
    if not phi_path.exists():
        raise FileNotFoundError(phi_path)
    phi_pvs_all = np.load(phi_path)
    phi_pvs = phi_pvs_all[:, :N_SPATIAL_BASIS].astype(np.float32, copy=False)

    pv_path = REPO_ROOT / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df = pd.read_csv(pv_path)
    if "pv_name" not in pv_df.columns:
        raise ValueError(f"{pv_path} lacks pv_name")
    pv_names = pv_df["pv_name"].astype(str).tolist()
    if phi_pvs.shape != (len(pv_names), N_SPATIAL_BASIS):
        raise ValueError(
            f"TPS PV matrix {phi_pvs.shape} does not match {len(pv_names)} PV names"
        )

    bg_csi = pd.read_parquet(Path(BG_DIR) / "bg_csi_pvs.parquet")
    bg_clear = pd.read_parquet(Path(BG_DIR) / "bg_clearsky_pvs.parquet")
    met_frames = {
        key: pd.read_parquet(Path(BG_DIR) / f"met_{key}_pvs.parquet")
        for key in MET_KEYS
    }
    elevation = np.load(Path(BG_DIR) / "elevation_pvs.npy")
    c13 = pd.read_parquet(Path(C13_FEAT_DIR) / "c13_feat_pvs.parquet")
    if len(elevation) != len(pv_names):
        raise ValueError("PV elevation vector and PV-name count differ")

    common = bg_csi.index
    for frame in [bg_clear, c13, *met_frames.values()]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    if not common.is_unique:
        raise ValueError("PV inference timestamps are not unique")
    if len(common) == 0:
        raise ValueError("PV input artifacts have no common timestamps")

    for label, frame in {
        "bg_csi": bg_csi,
        "bg_clear": bg_clear,
        **{f"met_{k}": v for k, v in met_frames.items()},
    }.items():
        missing = [name for name in pv_names if name not in frame.columns]
        if missing:
            raise ValueError(f"{label} lacks PV columns, first missing={missing[0]}")
    if not isinstance(c13.columns, pd.MultiIndex):
        raise ValueError("c13_feat_pvs.parquet must have MultiIndex columns")
    for feature in ("bt_norm", "c02_norm"):
        if feature not in c13.columns.get_level_values(1):
            raise ValueError(f"C13 PV artifact lacks feature {feature}")

    bg_csi = bg_csi.loc[common, pv_names]
    bg_clear = bg_clear.loc[common, pv_names]
    met_frames = {key: frame.loc[common, pv_names] for key, frame in met_frames.items()}
    c13 = c13.loc[common]
    c13_norm = c13.xs("bt_norm", axis=1, level=1)[pv_names]
    c02_norm = c13.xs("c02_norm", axis=1, level=1)[pv_names]

    psi_all, _, full_spec, _ = load_temporal_artifact()
    _, selected_temporal_spec = select_temporal_levels(
        psi_all, full_spec, SELECTED_TEMPORAL_LEVELS
    )
    del psi_all
    common_ns = common.asi8.astype(np.int64, copy=False)
    # Evaluate the unchanged saved knot formula directly. The stored training
    # matrix contains daytime station timestamps only, whereas PV inference
    # has a broader timestamp grid and must not require lookup membership.
    psi_common = transform_temporal_basis(
        common_ns, selected_temporal_spec
    ).astype(np.float32, copy=False)
    if psi_common.shape[1] != 16:
        raise ValueError("Selected temporal basis does not have 16 columns")

    T, M = len(common), len(pv_names)
    raw_by_seed = {
        seed: np.full((T, M), np.nan, dtype=np.float32) for seed in SEEDS
    }
    csi_ensemble = np.full((T, M), np.nan, dtype=np.float32)
    ghi_ensemble = np.full((T, M), np.nan, dtype=np.float32)
    csi_seed42 = np.full((T, M), np.nan, dtype=np.float32)
    ghi_seed42 = np.full((T, M), np.nan, dtype=np.float32)
    seed_std = np.full((T, M), np.nan, dtype=np.float32)
    imputed_feature_counts = np.zeros(M, dtype=np.int64)

    print(f"PV locations             : {M}")
    print(f"Aligned timestamps       : {T:,}")
    print(f"TPS shape                : {phi_pvs.shape}")
    print(f"Temporal basis shape     : {psi_common.shape}")
    print("Ensemble rule            : average raw CSI, then postprocess once")
    start = time.time()

    for j, pv_name in enumerate(pv_names):
        bg_j_all = bg_csi[pv_name].to_numpy(dtype=np.float32)
        clear_j_all = bg_clear[pv_name].to_numpy(dtype=np.float32)
        met_j = {
            key: met_frames[key][pv_name].to_numpy(dtype=np.float32)
            for key in MET_KEYS
        }
        cov_j_all = build_pv_covariates(
            common,
            bg_j_all,
            clear_j_all,
            c13_norm[pv_name].to_numpy(dtype=np.float32),
            c02_norm[pv_name].to_numpy(dtype=np.float32),
            met_j,
            float(elevation[j]),
        )
        day = clear_j_all >= CLEARSKY_MIN
        if not day.any():
            continue
        phi_j = np.broadcast_to(phi_pvs[j], (int(day.sum()), N_SPATIAL_BASIS))
        features = np.concatenate(
            [phi_j, psi_common[day], cov_j_all[day]], axis=1
        ).astype(np.float32, copy=False)
        features_scaled = (features - scaler_mean) / scaler_std
        imputed_feature_counts[j] = int((~np.isfinite(features_scaled)).sum())
        # Zero in standardized space is the production-training mean. This is
        # the same missing-covariate behavior as the repository's old predictor.
        features_scaled = np.nan_to_num(
            features_scaled, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)

        seed_predictions = []
        for seed, model in zip(SEEDS, models):
            prediction = predict_numpy(model, features_scaled)
            raw_by_seed[seed][day, j] = prediction
            seed_predictions.append(prediction)
        seed_stack = np.stack(seed_predictions, axis=0)
        raw_mean = seed_stack.mean(axis=0)
        raw_std = seed_stack.std(axis=0)

        processed_ensemble = apply_dk0_postprocessing(
            raw_mean, bg_j_all[day], clear_j_all[day]
        )
        processed_seed42 = apply_dk0_postprocessing(
            seed_stack[0], bg_j_all[day], clear_j_all[day]
        )
        csi_ensemble[day, j] = processed_ensemble
        ghi_ensemble[day, j] = processed_ensemble * clear_j_all[day]
        csi_seed42[day, j] = processed_seed42
        ghi_seed42[day, j] = processed_seed42 * clear_j_all[day]
        seed_std[day, j] = raw_std

        if (j + 1) % 25 == 0 or j == M - 1:
            print(
                f"  {j + 1:>3}/{M} PVs; elapsed={time.time() - start:.1f}s; "
                f"last={pv_name}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        save_frame(raw_by_seed[seed], common, pv_names, f"csi_raw_seed{seed}_pvs")
    raw_ensemble = np.mean(
        np.stack([raw_by_seed[seed] for seed in SEEDS], axis=0), axis=0
    )
    save_frame(raw_ensemble, common, pv_names, "csi_raw_ensemble_pvs")
    save_frame(seed_std, common, pv_names, "csi_raw_seed_std_pvs")
    save_frame(csi_seed42, common, pv_names, "csi_seed42_pvs")
    save_frame(ghi_seed42, common, pv_names, "ghi_seed42_pvs")
    save_frame(csi_ensemble, common, pv_names, "csi_pvs")
    save_frame(ghi_ensemble, common, pv_names, "ghi_pvs")

    missing_summary = pd.DataFrame(
        {"pv_name": pv_names, "standardized_values_imputed_to_zero": imputed_feature_counts}
    )
    missing_summary.to_csv(OUT_DIR / "missing_feature_summary.csv", index=False)

    finite_ghi = ghi_ensemble[np.isfinite(ghi_ensemble)]
    pv_mean = np.nanmean(ghi_ensemble, axis=0)
    summary = {
        "n_timestamps": T,
        "n_pvs": M,
        "seeds": list(SEEDS),
        "ensemble_rule": "mean_raw_CSI_then_single_common_postprocessing",
        "daytime_ghi_min": float(np.min(finite_ghi)),
        "daytime_ghi_max": float(np.max(finite_ghi)),
        "daytime_ghi_mean": float(np.mean(finite_ghi)),
        "per_pv_mean_ghi_min": float(np.min(pv_mean)),
        "per_pv_mean_ghi_max": float(np.max(pv_mean)),
        "mean_raw_seed_std": float(np.nanmean(seed_std)),
        "p95_raw_seed_std": float(np.nanquantile(seed_std, 0.95)),
        "total_missing_standardized_values_imputed_to_zero": int(
            imputed_feature_counts.sum()
        ),
    }
    (OUT_DIR / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("DK-6 PV INFERENCE COMPLETE")
    print("=" * 78)
    print(f"Daytime ensemble GHI range : {summary['daytime_ghi_min']:.1f}–{summary['daytime_ghi_max']:.1f} W/m²")
    print(f"Daytime ensemble GHI mean  : {summary['daytime_ghi_mean']:.1f} W/m²")
    print(f"Per-PV mean GHI range      : {summary['per_pv_mean_ghi_min']:.1f}–{summary['per_pv_mean_ghi_max']:.1f} W/m²")
    print(f"Mean/p95 raw seed std      : {summary['mean_raw_seed_std']:.4f} / {summary['p95_raw_seed_std']:.4f} CSI")
    print(f"Outputs                    : {OUT_DIR}")


if __name__ == "__main__":
    main()
