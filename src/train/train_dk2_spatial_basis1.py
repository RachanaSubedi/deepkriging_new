"""Train the controlled DK-2 spatial-basis comparison.

Place this file at:
    src/train/train_dk2_spatial_basis.py

Prerequisites:
    python src/train/shared_blocked_day_split.py --validation-fraction 0.20
    python src/data_prep/temporal_basis.py

    python src/data_prep/planar_eigen_tps_basis.py

Example (K=8 total planar eigen-TPS columns):
    python src/train/train_dk2_spatial_basis.py \
        --spatial-basis eigen_tps --n-spatial-basis 8 \
        --temporal-levels coarse_30d \
        --p2-provenance \
        "data/raw/stations/station_p2_full_year_GHI_zeroshot.csv"

Controlled comparison from the already-completed DK-1A:
    DK-1A = [411 Wendland, 16 coarse temporal, 15 covariates]
    DK-2  = [K planar eigen-TPS, 16 coarse temporal, 15 covariates]

Everything else is inherited from train_dk0_matched.py: direct-CSI target,
monolithic DNN, Huber loss, optimizer, enhancement oversampling, shared
blocked-day split, postprocessing, and genuine-only P2 supervision/testing.

The temporal basis is stored once for unique timestamps. This trainer maps
each training-matrix row to its unique timestamp and materializes temporal
features only for the current fold, avoiding a redundant full-year expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Repository imports and experiment paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import (  # noqa: E402
    BASIS_DIR,
    BATCH_SIZE,
    BG_DIR,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    STATIONS,
    TRAIN_DIR,
    WEIGHT_DECAY,
)
from src.model import DeepKriging, count_parameters  # noqa: E402
from src.train.shared_blocked_day_split import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    load_shared_split,
)

# Import the matched DK-0 implementation so all non-temporal logic remains
# exactly shared rather than being silently reimplemented differently.
from src.train.train_dk0_matched import (  # noqa: E402
    CLEARSKY_MIN,
    EARLY_STOP_PAT,
    FOLD_COLORS,
    HUBER_DELTA,
    N_COVARIATES_EXPECTED,
    SEED,
    apply_dk0_postprocessing,
    build_ground_truth_mask,
    predict,
    rmse,
    set_seed,
    standardise,
    train_one_fold,
)


TEMPORAL_DIR = Path(PROCESSED_DIR) / "temporal_basis"
TPS_DIR = Path(PROCESSED_DIR) / "basis_planar_eigen_tps"

# These are configured from --temporal-levels in main(). They remain module
# globals because the plotting/snapshot helpers use the experiment paths.
EXPERIMENT_NAME = "UNCONFIGURED_DK1"
EXPERIMENT_LABEL = "Unconfigured DK-1"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / EXPERIMENT_NAME
MODEL_DIR = EXPERIMENT_DIR / "models"
VAL_DIR = EXPERIMENT_DIR / "validation"
FIG_DIR = EXPERIMENT_DIR / "figures"
SPLIT_SNAPSHOT_DIR = EXPERIMENT_DIR / "split_snapshot"
TEMPORAL_SNAPSHOT_DIR = EXPERIMENT_DIR / "temporal_basis_snapshot"

ALLOWED_LEVEL_COMBINATIONS = {
    ("coarse_30d",): (
        "DK1A_temporal_30d_groundtruth",
        "DK-1A: 30-Day Temporal Basis",
    ),
    ("coarse_30d", "medium_7d"): (
        "DK1B_temporal_30d_7d_groundtruth",
        "DK-1B: 30-Day + 7-Day Temporal Basis",
    ),
    ("coarse_30d", "medium_7d", "fine_1d"): (
        "DK1_spatiotemporal_groundtruth",
        "DK-1C: 30-Day + 7-Day + 1-Day Temporal Basis",
    ),
}

STATION_NAMES = list(STATIONS.keys())
DEVICE = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train direct-CSI DK-1 with spatial and temporal basis functions "
            "and genuine-only P2 supervision."
        )
    )
    parser.add_argument(
        "--spatial-basis",
        choices=["wendland", "eigen_tps"],
        default="wendland",
        help="Spatial representation. Keep 'wendland' for DK-1; use 'eigen_tps' for DK-2.",
    )
    parser.add_argument(
        "--n-spatial-basis",
        type=int,
        default=None,
        help="Nested total K for eigen_tps (supported: 4, 8, 16, 32).",
    )
    parser.add_argument(
        "--temporal-levels",
        nargs="+",
        required=True,
        choices=["coarse_30d", "medium_7d", "fine_1d"],
        help=(
            "Nested temporal levels to include. Supported experiments are: "
            "coarse_30d (DK-1A); coarse_30d medium_7d (DK-1B); or all "
            "three levels (the already-completed full DK-1C)."
        ),
    )
    parser.add_argument(
        "--p2-provenance",
        type=Path,
        required=True,
        help=(
            "P2 CSV containing timestamp and source columns. Only "
            "source='measured' rows are accepted as P2 ground truth."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of an already-completed result folder.",
    )
    return parser.parse_args()


def configure_experiment(
    selected_levels: list[str], spatial_kind: str, spatial_k: int | None
) -> tuple[str, ...]:
    """Validate the nested ablation and configure isolated output paths."""
    global EXPERIMENT_NAME, EXPERIMENT_LABEL
    global EXPERIMENT_DIR, MODEL_DIR, VAL_DIR, FIG_DIR
    global SPLIT_SNAPSHOT_DIR, TEMPORAL_SNAPSHOT_DIR

    # Canonicalize command-line order according to the artifact hierarchy.
    canonical_order = ("coarse_30d", "medium_7d", "fine_1d")
    if len(selected_levels) != len(set(selected_levels)):
        raise ValueError("--temporal-levels contains a duplicate level")
    selected_set = set(selected_levels)
    canonical = tuple(name for name in canonical_order if name in selected_set)
    if canonical not in ALLOWED_LEVEL_COMBINATIONS:
        allowed = "; ".join(" ".join(levels) for levels in ALLOWED_LEVEL_COMBINATIONS)
        raise ValueError(
            "Unsupported temporal-level combination. Use one of: " + allowed
        )

    if spatial_kind == "wendland":
        if spatial_k is not None:
            raise ValueError("--n-spatial-basis is only valid with --spatial-basis eigen_tps")
        EXPERIMENT_NAME, EXPERIMENT_LABEL = ALLOWED_LEVEL_COMBINATIONS[canonical]
    else:
        if canonical != ("coarse_30d",):
            raise ValueError("DK-2 spatial comparison fixes temporal levels to coarse_30d")
        if spatial_k not in (4, 8, 16, 32):
            raise ValueError("eigen_tps requires --n-spatial-basis 4, 8, 16, or 32")
        EXPERIMENT_NAME = f"DK2_planar_eigen_TPS_K{spatial_k:03d}_groundtruth"
        EXPERIMENT_LABEL = f"DK-2 Planar Eigen-TPS K={spatial_k}"
    EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / EXPERIMENT_NAME
    MODEL_DIR = EXPERIMENT_DIR / "models"
    VAL_DIR = EXPERIMENT_DIR / "validation"
    FIG_DIR = EXPERIMENT_DIR / "figures"
    SPLIT_SNAPSHOT_DIR = EXPERIMENT_DIR / "split_snapshot"
    TEMPORAL_SNAPSHOT_DIR = EXPERIMENT_DIR / "temporal_basis_snapshot"
    return canonical


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_temporal_artifact() -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load and validate the artifact created by temporal_basis.py."""
    basis_path = TEMPORAL_DIR / "temporal_basis_unique.npy"
    timestamp_path = TEMPORAL_DIR / "temporal_basis_unique_timestamps.npy"
    spec_path = TEMPORAL_DIR / "temporal_basis_spec.json"
    summary_path = TEMPORAL_DIR / "temporal_basis_summary.json"

    required = [basis_path, timestamp_path, spec_path, summary_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing DK-1 temporal-basis artifact(s):\n  "
            + "\n  ".join(missing)
            + "\nRun: python src/data_prep/temporal_basis.py"
        )

    psi_unique = np.load(basis_path, allow_pickle=False)
    unique_timestamps = np.load(timestamp_path, allow_pickle=False).astype(
        np.int64, copy=False
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if psi_unique.ndim != 2:
        raise ValueError(f"Temporal basis must be 2-D; got {psi_unique.shape}")
    if unique_timestamps.ndim != 1:
        raise ValueError("Unique temporal timestamps must be one-dimensional")
    if psi_unique.shape[0] != unique_timestamps.size:
        raise ValueError(
            "Temporal basis rows and unique timestamps do not match: "
            f"{psi_unique.shape[0]:,} versus {unique_timestamps.size:,}"
        )
    if psi_unique.shape[1] != int(spec["n_basis"]):
        raise ValueError(
            "Temporal basis columns disagree with temporal_basis_spec.json"
        )
    if np.any(np.diff(unique_timestamps) <= 0):
        raise ValueError("Temporal artifact timestamps are not strictly increasing")
    if not np.isfinite(psi_unique).all():
        raise ValueError("Temporal basis contains non-finite values")
    if psi_unique.min() < 0.0 or psi_unique.max() > 1.000001:
        raise ValueError("Temporal Gaussian basis is outside [0, 1]")
    if not bool(summary.get("validation_support_passed", False)):
        raise ValueError("Temporal artifact did not pass validation-support checks")

    hashes = {
        "temporal_basis_unique.npy": sha256_file(basis_path),
        "temporal_basis_unique_timestamps.npy": sha256_file(timestamp_path),
        "temporal_basis_spec.json": sha256_file(spec_path),
        "temporal_basis_summary.json": sha256_file(summary_path),
    }
    return psi_unique.astype(np.float32, copy=False), unique_timestamps, spec, hashes


def select_temporal_levels(
    psi_all: np.ndarray,
    full_spec: dict,
    selected_levels: tuple[str, ...],
) -> tuple[np.ndarray, dict]:
    """Select complete level blocks while preserving artifact column order."""
    available_names = tuple(level["name"] for level in full_spec["levels"])
    missing = [name for name in selected_levels if name not in available_names]
    if missing:
        raise ValueError(
            f"Requested temporal levels are absent from the artifact: {missing}. "
            f"Available levels: {available_names}"
        )

    selected_columns: list[int] = []
    selected_level_specs: list[dict] = []
    all_feature_names: list[str] = []
    column_start = 0
    for level in full_spec["levels"]:
        n_level = int(level["n_basis"])
        column_end = column_start + n_level
        if level["name"] in selected_levels:
            selected_columns.extend(range(column_start, column_end))
            selected_level_specs.append(level)
            all_feature_names.extend(level["feature_names"])
        column_start = column_end

    if column_start != psi_all.shape[1]:
        raise ValueError(
            "Temporal specification does not span all temporal-basis columns"
        )
    if not selected_columns:
        raise ValueError("No temporal-basis columns were selected")

    selected_spec = dict(full_spec)
    selected_spec["version"] = (
        full_spec["version"] + "__" + "__".join(selected_levels)
    )
    selected_spec["levels"] = selected_level_specs
    selected_spec["n_basis"] = len(selected_columns)
    selected_spec["feature_names"] = all_feature_names
    selected_spec["selected_from_full_artifact"] = True
    selected_spec["selected_levels"] = list(selected_levels)

    return psi_all[:, selected_columns], selected_spec


def map_rows_to_unique_time(
    timestamps_ns: np.ndarray,
    unique_timestamps_ns: np.ndarray,
) -> np.ndarray:
    """Return the temporal-basis row corresponding to every matrix row."""
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    positions = np.searchsorted(unique_timestamps_ns, timestamps_ns)
    in_range = positions < unique_timestamps_ns.size
    matched = np.zeros(timestamps_ns.size, dtype=bool)
    matched[in_range] = (
        unique_timestamps_ns[positions[in_range]] == timestamps_ns[in_range]
    )
    if not np.all(matched):
        bad = int(np.flatnonzero(~matched)[0])
        missing_time = pd.Timestamp(
            int(timestamps_ns[bad]), unit="ns", tz="UTC"
        )
        raise KeyError(
            f"Training timestamp {missing_time} is absent from the temporal artifact"
        )
    return positions.astype(np.int64, copy=False)


def assemble_dk1_features(
    X_dk0: np.ndarray,
    row_indices: np.ndarray,
    original_wendland_columns: int,
    fold_ids: np.ndarray,
    spatial_basis_at_stations: np.ndarray,
    temporal_row_for_training_row: np.ndarray,
    psi_unique: np.ndarray,
) -> np.ndarray:
    """Assemble [Phi(s), Psi(t), covariates] for selected rows."""
    selected = X_dk0[row_indices]
    phi = spatial_basis_at_stations[fold_ids[row_indices]]
    covariates = selected[:, original_wendland_columns:]
    psi = psi_unique[temporal_row_for_training_row[row_indices]]
    output = np.concatenate([phi, psi, covariates], axis=1)
    return output.astype(np.float32, copy=False)


def snapshot_inputs() -> None:
    """Snapshot small split/temporal metadata; hash the large matrix."""
    SPLIT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "shared_train_mask.npy",
        "shared_val_mask.npy",
        "shared_validation_days.csv",
        "shared_split_summary.json",
    ]:
        source = DEFAULT_SPLIT_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Missing shared split file: {source}")
        shutil.copy2(source, SPLIT_SNAPSHOT_DIR / name)

    TEMPORAL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "temporal_basis_spec.json",
        "temporal_basis_summary.json",
        "temporal_basis_feature_names.txt",
        "temporal_basis_support.csv",
    ]:
        source = TEMPORAL_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"Missing temporal-basis metadata: {source}")
        shutil.copy2(source, TEMPORAL_SNAPSHOT_DIR / name)


def plot_loss_curves() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{EXPERIMENT_LABEL} - Shared Blocked-Day Split",
        fontsize=13,
        fontweight="bold",
    )

    for fold_k, station in enumerate(STATION_NAMES):
        history = pd.read_csv(VAL_DIR / f"fold_{fold_k}_history.csv")
        color = FOLD_COLORS[station]
        best_index = history["val_loss"].idxmin()
        best_epoch = int(history.loc[best_index, "epoch"])
        ax_train.plot(
            history["epoch"], history["train_loss"],
            color=color, lw=1.8, label=f"Train {station}",
        )
        ax_val.plot(
            history["epoch"], history["val_loss"],
            color=color, lw=1.8, label=f"Val {station}",
        )
        ax_val.scatter(
            [best_epoch], [history.loc[best_index, "val_loss"]],
            color=color, s=55, zorder=5,
        )

    for axis, title in [(ax_train, "Training Loss"), (ax_val, "Validation Loss")]:
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Huber Loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_loss_combined.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    selected_levels = configure_experiment(
        args.temporal_levels, args.spatial_basis, args.n_spatial_basis
    )
    completed_result = VAL_DIR / "loso_results.csv"
    if completed_result.exists() and not args.overwrite:
        raise FileExistsError(
            f"This experiment is already complete: {completed_result}\n"
            "Its outputs were left unchanged. Use --overwrite only if you "
            "intentionally want to rerun and replace it."
        )
    set_seed(SEED)
    print("=" * 72)
    print(f"{EXPERIMENT_LABEL.upper()} - LEAKAGE-SAFE P2 LOSO")
    print("=" * 72)

    X_dk0 = np.load(TRAIN_DIR / "X.npy")
    y = np.load(TRAIN_DIR / "y.npy")
    fold_ids = np.load(TRAIN_DIR / "fold_ids.npy")
    timestamps_ns = np.load(TRAIN_DIR / "timestamps.npy").astype(
        np.int64, copy=False
    )
    original_wendland = np.load(BASIS_DIR / "Phi_stations_scaled.npy")
    original_wendland_columns = int(original_wendland.shape[1])
    if args.spatial_basis == "wendland":
        spatial_basis = original_wendland.astype(np.float32, copy=False)
        spatial_spec = {"version": "existing_multiresolution_wendland"}
    else:
        tps_path = TPS_DIR / "Phi_stations.npy"
        tps_spec_path = TPS_DIR / "basis_spec.json"
        if not tps_path.exists() or not tps_spec_path.exists():
            raise FileNotFoundError(
                "Missing planar eigen-TPS artifact. Run: "
                "python src/data_prep/planar_eigen_tps_basis.py"
            )
        tps_all = np.load(tps_path)
        spatial_basis = tps_all[:, : args.n_spatial_basis].astype(
            np.float32, copy=False
        )
        spatial_spec = json.loads(tps_spec_path.read_text(encoding="utf-8"))
    n_spatial_basis = int(spatial_basis.shape[1])

    if not (len(X_dk0) == len(y) == len(fold_ids) == len(timestamps_ns)):
        raise ValueError("Training arrays do not have the same number of rows")
    if X_dk0.shape[1] != original_wendland_columns + N_COVARIATES_EXPECTED:
        raise ValueError(
            f"Expected {original_wendland_columns} original Wendland bases + "
            f"{N_COVARIATES_EXPECTED} covariates in X.npy, but found "
            f"{X_dk0.shape[1]} columns"
        )

    psi_all, temporal_unique_ns, full_temporal_spec, temporal_hashes = (
        load_temporal_artifact()
    )
    psi_unique, temporal_spec = select_temporal_levels(
        psi_all, full_temporal_spec, selected_levels
    )
    del psi_all
    n_temporal_basis = int(psi_unique.shape[1])
    n_total_basis = n_spatial_basis + n_temporal_basis
    input_dimension = n_total_basis + N_COVARIATES_EXPECTED
    temporal_row_for_training_row = map_rows_to_unique_time(
        timestamps_ns, temporal_unique_ns
    )

    shared_train_mask, shared_val_mask, split_summary = load_shared_split(
        timestamps_ns=timestamps_ns,
        fold_ids=fold_ids,
        split_dir=DEFAULT_SPLIT_DIR,
    )
    ground_truth_mask, provenance_summary = build_ground_truth_mask(
        timestamps_ns=timestamps_ns,
        fold_ids=fold_ids,
        p2_provenance_path=args.p2_provenance,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_inputs()
    (TEMPORAL_SNAPSHOT_DIR / "selected_temporal_basis_spec.json").write_text(
        json.dumps(temporal_spec, indent=2), encoding="utf-8"
    )
    np.save(EXPERIMENT_DIR / "ground_truth_mask.npy", ground_truth_mask)
    (EXPERIMENT_DIR / "p2_ground_truth_summary.json").write_text(
        json.dumps(provenance_summary, indent=2), encoding="utf-8"
    )

    bg_csi = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clear = pd.read_parquet(BG_DIR / "bg_clearsky_stations.parquet")

    temporary_model = DeepKriging(input_dimension, HIDDEN_SIZE, DROPOUT)
    print(f"Input dimension       : {input_dimension}")
    print(f"Spatial basis columns : {n_spatial_basis}")
    print(f"Temporal basis columns: {n_temporal_basis}")
    print(f"Selected levels       : {', '.join(selected_levels)}")
    for level in temporal_spec["levels"]:
        print(
            f"  {level['name']:<12}: {level['n_basis']:>4} bases "
            f"(spacing={level['spacing_days']:g} days)"
        )
    print(f"Covariates            : {N_COVARIATES_EXPECTED}")
    print(f"Parameters            : {count_parameters(temporary_model):,}")
    print(f"Target                : direct CSI")
    print(f"Loss                  : Huber(delta={HUBER_DELTA})")
    print(f"Validation days       : {split_summary['n_validation_days']}")
    print(
        "P2 genuine rows      : "
        f"{provenance_summary['p2_genuine_rows_retained']:,} / "
        f"{provenance_summary['p2_rows_in_training_matrix']:,}"
    )
    print(
        "P2 reconstructed excluded: "
        f"{provenance_summary['p2_reconstructed_rows_excluded']:,}"
    )
    print(f"Output directory      : {EXPERIMENT_DIR}")

    fold_results: list[dict[str, float | int | str]] = []

    for fold_k, test_station in enumerate(STATION_NAMES):
        print("\n" + "-" * 72)
        print(f"FOLD {fold_k} - held-out station: {test_station}")
        print("-" * 72)

        held_out_station_mask = fold_ids == fold_k
        test_mask = held_out_station_mask & ground_truth_mask
        train_mask = (
            (~held_out_station_mask) & shared_train_mask & ground_truth_mask
        )
        val_mask = (
            (~held_out_station_mask) & shared_val_mask & ground_truth_mask
        )

        if np.any(train_mask & val_mask):
            raise RuntimeError("Training and validation masks overlap")
        if np.any(held_out_station_mask & (train_mask | val_mask)):
            raise RuntimeError("Held-out station leaked into train/validation")

        train_indices = np.flatnonzero(train_mask)
        val_indices = np.flatnonzero(val_mask)
        test_indices = np.flatnonzero(test_mask)

        # Preserve DK-0 enhancement oversampling exactly: retain every original
        # row and append three more copies of rows with CSI > 0.85.
        enhancement_indices = train_indices[y[train_indices] > 0.85]
        n_enhancement_original = int(enhancement_indices.size)
        if n_enhancement_original:
            train_indices_expanded = np.concatenate(
                [train_indices, np.tile(enhancement_indices, 3)]
            )
        else:
            train_indices_expanded = train_indices

        X_train = assemble_dk1_features(
            X_dk0, train_indices_expanded, original_wendland_columns,
            fold_ids, spatial_basis,
            temporal_row_for_training_row, psi_unique,
        )
        X_val = assemble_dk1_features(
            X_dk0, val_indices, original_wendland_columns,
            fold_ids, spatial_basis,
            temporal_row_for_training_row, psi_unique,
        )
        X_test = assemble_dk1_features(
            X_dk0, test_indices, original_wendland_columns,
            fold_ids, spatial_basis,
            temporal_row_for_training_row, psi_unique,
        )
        y_train = y[train_indices_expanded]
        y_val = y[val_indices]
        y_test = y[test_indices]
        ts_test = timestamps_ns[test_indices]

        excluded_reconstructed_p2 = int(
            ((~ground_truth_mask) & (~held_out_station_mask)).sum()
        )
        print(f"Train samples : {len(y_train):,}")
        print(f"Val samples   : {len(y_val):,}")
        print(f"Test samples  : {len(y_test):,}")
        print(
            "Reconstructed P2 rows excluded from this fold's training pool: "
            f"{excluded_reconstructed_p2:,}"
        )
        print(
            f"Enhancement rows oversampled: {n_enhancement_original:,} "
            f"({3*n_enhancement_original:,} additional rows)"
        )

        # Spatial bases are response-independent and globally normalized by
        # construction; Psi is bounded. Only the 15 physical covariates are
        # standardized from the current fold's training rows.
        X_train_sc, X_val_sc, X_test_sc, scaler_mean, scaler_std = standardise(
            X_train, X_val, X_test, n_basis=n_total_basis
        )

        start = time.time()
        model, history = train_one_fold(X_train_sc, y_train, X_val_sc, y_val)
        elapsed_minutes = (time.time() - start) / 60.0
        history_df = pd.DataFrame(history)
        history_df.to_csv(VAL_DIR / f"fold_{fold_k}_history.csv", index=False)

        csi_pred_raw = predict(model, X_test_sc)
        rmse_csi_raw = rmse(y_test, csi_pred_raw)
        r2_csi_raw = float(r2_score(y_test, csi_pred_raw))

        ts_local = (
            pd.to_datetime(ts_test, unit="ns", utc=True)
            .tz_convert("America/Los_Angeles")
        )
        bg_clear_test = bg_clear[test_station].reindex(ts_local).to_numpy()
        bg_csi_test = bg_csi[test_station].reindex(ts_local).to_numpy()
        if np.isnan(bg_clear_test).any() or np.isnan(bg_csi_test).any():
            raise ValueError(
                f"Background data failed to align for held-out {test_station}"
            )

        csi_pred = apply_dk0_postprocessing(
            csi_pred_raw, bg_csi_test, bg_clear_test
        )
        csi_true = np.clip(y_test, 0.0, 1.3)
        ghi_pred = csi_pred * bg_clear_test
        ghi_true = csi_true * bg_clear_test
        daytime = bg_clear_test >= CLEARSKY_MIN

        rmse_csi = rmse(csi_true[daytime], csi_pred[daytime])
        r2_csi = float(r2_score(csi_true[daytime], csi_pred[daytime]))
        rmse_ghi = rmse(ghi_true[daytime], ghi_pred[daytime])
        r2_ghi = float(r2_score(ghi_true[daytime], ghi_pred[daytime]))

        print(f"Training time : {elapsed_minutes:.1f} min")
        print(
            "Best epoch    : "
            f"{int(history_df.loc[history_df.val_loss.idxmin(), 'epoch'])}"
        )
        print(f"CSI RMSE      : {rmse_csi:.4f}   R2={r2_csi:.4f}")
        print(f"GHI RMSE      : {rmse_ghi:.2f} W/m2   R2={r2_ghi:.4f}")

        torch.save(model.state_dict(), MODEL_DIR / f"fold_{fold_k}_best.pt")
        np.save(MODEL_DIR / f"fold_{fold_k}_scaler_mean.npy", scaler_mean)
        np.save(MODEL_DIR / f"fold_{fold_k}_scaler_std.npy", scaler_std)

        prediction_df = pd.DataFrame(
            {
                "datetime_local": ts_local,
                "station": test_station,
                "csi_true": csi_true,
                "csi_pred": csi_pred,
                "csi_pred_raw": csi_pred_raw,
                "bg_csi": bg_csi_test,
                "ghi_true": ghi_true,
                "ghi_pred": ghi_pred,
                "bg_clearsky": bg_clear_test,
            }
        )
        prediction_df.to_csv(
            VAL_DIR / f"fold_{fold_k}_{test_station}_predictions.csv",
            index=False,
        )

        fold_results.append(
            {
                "fold": fold_k,
                "test_station": test_station,
                "n_train_after_oversampling": len(y_train),
                "n_validation": len(y_val),
                "n_test": len(y_test),
                "rmse_csi_raw": rmse_csi_raw,
                "r2_csi_raw": r2_csi_raw,
                "rmse_csi_postprocessed": rmse_csi,
                "r2_csi_postprocessed": r2_csi,
                "rmse_ghi": rmse_ghi,
                "r2_ghi": r2_ghi,
                "epochs_run": len(history),
                "training_minutes": elapsed_minutes,
            }
        )

        # Release the largest fold-specific arrays before constructing the
        # next fold. This matters because DK-1 has 867 input columns.
        del X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc

    results = pd.DataFrame(fold_results)
    results.to_csv(VAL_DIR / "loso_results.csv", index=False)
    summary_columns = [
        "fold", "test_station", "rmse_csi_raw", "r2_csi_raw",
        "rmse_ghi", "r2_ghi",
    ]
    summary_text = results[summary_columns].round(4).to_string(index=False)
    (VAL_DIR / "loso_summary.txt").write_text(
        "DK-1 Spatial + Temporal Basis Ground-Truth LOSO Results\n"
        + "=" * 60
        + "\n"
        + summary_text
        + "\n\n"
        + f"Mean GHI RMSE: {results.rmse_ghi.mean():.4f}\n"
        + f"Mean GHI R2: {results.r2_ghi.mean():.4f}\n",
        encoding="utf-8",
    )

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "controlled_comparator": "DK0_matched_groundtruth",
        "single_intended_change": (
            "replace_wendland_with_planar_eigen_tps"
            if args.spatial_basis == "eigen_tps"
            else "append_multiresolution_temporal_basis"
        ),
        "target": "direct_csi",
        "input_order": [
            f"spatial_{args.spatial_basis}_basis",
            "temporal_gaussian_basis",
            "existing_15_covariates",
        ],
        "spatial_basis": args.spatial_basis,
        "n_spatial_basis": n_spatial_basis,
        "spatial_basis_spec": spatial_spec,
        "temporal_basis": temporal_spec["version"],
        "selected_temporal_levels": list(selected_levels),
        "n_temporal_basis": n_temporal_basis,
        "temporal_levels": [
            {
                "name": level["name"],
                "spacing_days": level["spacing_days"],
                "kappa_days": level["kappa_days"],
                "n_basis": level["n_basis"],
            }
            for level in temporal_spec["levels"]
        ],
        "temporal_basis_standardized": False,
        "n_covariates": N_COVARIATES_EXPECTED,
        "input_dimension": input_dimension,
        "architecture": "existing_monolithic_3_hidden_layer_mlp",
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "loss": "HuberLoss",
        "huber_delta": HUBER_DELTA,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOP_PAT,
        "seed": SEED,
        "validation_split": "shared_complete_local_days_stratified_by_month",
        "validation_days": split_summary["n_validation_days"],
        "p2_training_policy": "genuine_measured_only",
        "p2_testing_policy": "genuine_measured_only",
        "p2_genuine_rows": provenance_summary["p2_genuine_rows_retained"],
        "p2_reconstructed_rows_excluded": provenance_summary[
            "p2_reconstructed_rows_excluded"
        ],
        "temporal_artifact_sha256": temporal_hashes,
        "dk0_shared_implementation_sha256": sha256_file(
            REPO_ROOT / "src" / "train" / "train_dk0_matched.py"
        ),
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_loss_curves()

    print("\n" + "=" * 72)
    print(f"{EXPERIMENT_LABEL.upper()} TRAINING COMPLETE")
    print("=" * 72)
    print(summary_text)
    print(f"\nMean GHI RMSE: {results.rmse_ghi.mean():.4f}")
    print(f"Mean GHI R2  : {results.r2_ghi.mean():.4f}")
    print(f"Outputs      : {EXPERIMENT_DIR}")
    print("\nDK0_matched_groundtruth outputs were not modified.")


if __name__ == "__main__":
    main()
