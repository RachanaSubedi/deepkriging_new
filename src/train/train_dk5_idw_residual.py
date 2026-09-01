"""Train DK-5: fold-safe station-IDW residual DeepKriging.

Place this file at:
    src/train/train_dk5_idw_residual.py

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
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


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
        "--loss", choices=["huber"], default="huber",
        help="Fixed to Huber for the controlled residual experiment.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Training seed (default 42 for the first controlled comparison).",
    )
    parser.add_argument(
        "--spatial-basis",
        choices=["wendland", "eigen_tps"],
        default="eigen_tps",
        help="Fixed to eigen_tps for DK-4.",
    )
    parser.add_argument(
        "--n-spatial-basis",
        type=int,
        default=16,
        help="Fixed to K=16 for DK-4.",
    )
    parser.add_argument(
        "--temporal-levels",
        nargs="+",
        default=["coarse_30d"],
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
    selected_levels: list[str], spatial_kind: str, spatial_k: int | None,
    loss_name: str, seed: int,
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

    if spatial_kind != "eigen_tps" or spatial_k != 16:
        raise ValueError("DK-4 fixes the spatial representation to eigen_tps K=16")
    if canonical != ("coarse_30d",):
        raise ValueError("DK-4 fixes the temporal representation to coarse_30d")
    EXPERIMENT_NAME = f"DK5_IDW_residual_TPS_K016_seed{seed}_groundtruth"
    EXPERIMENT_LABEL = f"DK-5 Station-IDW Residual, TPS K=16, seed={seed}"
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


def station_distance_matrix_km() -> np.ndarray:
    """Pairwise station distances using the configured local-km projection."""
    names = list(STATIONS)
    lat = np.array([STATIONS[name]["lat"] for name in names], dtype=float)
    lon = np.array([STATIONS[name]["lon"] for name in names], dtype=float)
    lat0 = float(lat.mean())
    km_per_lon = 111.0 * np.cos(np.deg2rad(lat0))
    xy = np.column_stack([lon * km_per_lon, lat * 111.0])
    difference = xy[:, None, :] - xy[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    np.fill_diagonal(distance, np.inf)
    return distance


def build_genuine_station_table(
    y: np.ndarray,
    fold_ids: np.ndarray,
    ground_truth_mask: np.ndarray,
    temporal_row_for_training_row: np.ndarray,
    n_unique_times: int,
) -> np.ndarray:
    """CSI(time, station), retaining only permitted genuine supervision."""
    table = np.full((n_unique_times, len(STATION_NAMES)), np.nan, dtype=np.float32)
    valid_rows = np.flatnonzero(ground_truth_mask)
    table[
        temporal_row_for_training_row[valid_rows], fold_ids[valid_rows]
    ] = y[valid_rows]
    return table


def fold_safe_spatial_background(
    row_indices: np.ndarray,
    fold_ids: np.ndarray,
    temporal_row_for_training_row: np.ndarray,
    station_csi: np.ndarray,
    held_out_fold: int,
    distance_km: np.ndarray,
    method: str = "idw",
    power: float = 2.0,
) -> np.ndarray:
    """Background for rows without using target or held-out station values.

    For a training row belonging to station j in LOSO fold i, sources exclude
    both i and j. For a held-out test row, j=i, so all training stations are
    eligible. Missing P2 timestamps are omitted and remaining weights are
    renormalized.
    """
    row_indices = np.asarray(row_indices, dtype=np.int64)
    targets = fold_ids[row_indices]
    time_rows = temporal_row_for_training_row[row_indices]
    output = np.full(len(row_indices), np.nan, dtype=np.float32)
    all_station_ids = np.arange(len(STATION_NAMES))

    for target in np.unique(targets):
        positions = np.flatnonzero(targets == target)
        sources = all_station_ids[
            (all_station_ids != target) & (all_station_ids != held_out_fold)
        ]
        # For held-out test rows, target==held_out; exclude it only once.
        if target == held_out_fold:
            sources = all_station_ids[all_station_ids != held_out_fold]
        if not len(sources):
            raise RuntimeError("No source station remains for spatial background")

        values = station_csi[time_rows[positions]][:, sources]
        available = np.isfinite(values)
        if method == "idw":
            weights = 1.0 / np.maximum(distance_km[target, sources], 1e-6) ** power
            weighted = np.where(available, values, 0.0) * weights[None, :]
            denominator = available @ weights
            valid = denominator > 0
            local = np.full(len(positions), np.nan, dtype=np.float64)
            local[valid] = weighted[valid].sum(axis=1) / denominator[valid]
        elif method == "nearest":
            ranked = np.argsort(distance_km[target, sources])
            local = np.full(len(positions), np.nan, dtype=np.float64)
            for source_column in ranked:
                take = np.isnan(local) & available[:, source_column]
                local[take] = values[take, source_column]
        else:
            raise ValueError(f"Unknown background method: {method}")
        output[positions] = local.astype(np.float32)

    # NaN is intentional when every legally eligible source station is
    # unavailable at a timestamp.  The caller must remove those rows.  Do not
    # substitute the target, the held-out station, NSRDB, or a global mean here:
    # each of those choices would either leak information or change the DK-5
    # background definition.
    return output


def append_idw_feature(X: np.ndarray, idw_background: np.ndarray) -> np.ndarray:
    if len(X) != len(idw_background):
        raise ValueError("Feature rows and IDW background rows do not match")
    return np.column_stack([X, idw_background]).astype(np.float32, copy=False)


def standardise_residual_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_basis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Leave Phi/Psi fixed; standardize 15 covariates plus IDW background."""
    n_nonbasis = N_COVARIATES_EXPECTED + 1
    if X_train.shape[1] != n_basis + n_nonbasis:
        raise ValueError(
            f"Expected {n_basis} bases + {n_nonbasis} non-basis features; "
            f"got {X_train.shape[1]} columns"
        )
    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std = np.ones(X_train.shape[1], dtype=np.float32)
    mean[n_basis:] = X_train[:, n_basis:].mean(axis=0)
    std[n_basis:] = X_train[:, n_basis:].std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_loss_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_name: str,
) -> tuple[DeepKriging, list[dict[str, float]]]:
    """Train the unchanged monolithic model with only the criterion varied."""
    model = DeepKriging(X_train.shape[1], HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion: nn.Module
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=HUBER_DELTA)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    best_val_loss = float("inf")
    best_weights = None
    patience_count = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(y_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(yb)
        val_loss /= len(y_val)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    epoch {epoch:>3d}  train={train_loss:.5f}  "
                f"val={val_loss:.5f}  patience={patience_count}/{EARLY_STOP_PAT}"
            )
        if patience_count >= EARLY_STOP_PAT:
            print(
                f"    Early stop at epoch {epoch} "
                f"(best val={best_val_loss:.5f})"
            )
            break

    if best_weights is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_weights)
    return model, history


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
        axis.set_ylabel("Training criterion")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_loss_combined.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    selected_levels = configure_experiment(
        args.temporal_levels, args.spatial_basis, args.n_spatial_basis,
        args.loss, args.seed,
    )
    completed_result = VAL_DIR / "loso_results.csv"
    if completed_result.exists() and not args.overwrite:
        raise FileExistsError(
            f"This experiment is already complete: {completed_result}\n"
            "Its outputs were left unchanged. Use --overwrite only if you "
            "intentionally want to rerun and replace it."
        )
    set_seed(args.seed)
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
    # One additional standardized input is the fold-safe station-IDW level.
    input_dimension = n_total_basis + N_COVARIATES_EXPECTED + 1
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
    station_csi = build_genuine_station_table(
        y=y,
        fold_ids=fold_ids,
        ground_truth_mask=ground_truth_mask,
        temporal_row_for_training_row=temporal_row_for_training_row,
        n_unique_times=len(temporal_unique_ns),
    )
    distance_km = station_distance_matrix_km()

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
    print(f"Covariates            : {N_COVARIATES_EXPECTED} + fold-safe IDW level")
    print(f"Parameters            : {count_parameters(temporary_model):,}")
    print(f"Target                : CSI minus fold-safe station-IDW")
    loss_description = (
        "MSELoss" if args.loss == "mse" else f"Huber(delta={HUBER_DELTA})"
    )
    print(f"Loss                  : {loss_description}")
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

        # Construct backgrounds on the unexpanded row sets first. Occasionally
        # every legally eligible source is missing at the same timestamp. Such
        # a row cannot define an IDW-residual target and is excluded rather than
        # filled with a leaky or method-changing fallback.
        idw_train_base = fold_safe_spatial_background(
            train_indices, fold_ids, temporal_row_for_training_row,
            station_csi, fold_k, distance_km, method="idw",
        )
        idw_val = fold_safe_spatial_background(
            val_indices, fold_ids, temporal_row_for_training_row,
            station_csi, fold_k, distance_km, method="idw",
        )
        idw_test = fold_safe_spatial_background(
            test_indices, fold_ids, temporal_row_for_training_row,
            station_csi, fold_k, distance_km, method="idw",
        )

        train_bg_valid = np.isfinite(idw_train_base)
        val_bg_valid = np.isfinite(idw_val)
        test_bg_valid = np.isfinite(idw_test)
        dropped_train_no_source = int((~train_bg_valid).sum())
        dropped_val_no_source = int((~val_bg_valid).sum())
        dropped_test_no_source = int((~test_bg_valid).sum())

        train_indices = train_indices[train_bg_valid]
        idw_train_base = idw_train_base[train_bg_valid]
        val_indices = val_indices[val_bg_valid]
        idw_val = idw_val[val_bg_valid]
        test_indices = test_indices[test_bg_valid]
        idw_test = idw_test[test_bg_valid]

        if not len(train_indices) or not len(val_indices) or not len(test_indices):
            raise RuntimeError(
                f"Fold {fold_k} has an empty train/validation/test set after "
                "removing rows without a fold-safe IDW source."
            )

        # Preserve DK-0 enhancement oversampling exactly: retain every usable
        # original row and append three more copies of rows with CSI > 0.85.
        enhancement_mask = y[train_indices] > 0.85
        enhancement_indices = train_indices[enhancement_mask]
        n_enhancement_original = int(enhancement_indices.size)
        if n_enhancement_original:
            train_indices_expanded = np.concatenate(
                [train_indices, np.tile(enhancement_indices, 3)]
            )
            idw_train = np.concatenate(
                [idw_train_base, np.tile(idw_train_base[enhancement_mask], 3)]
            )
        else:
            train_indices_expanded = train_indices
            idw_train = idw_train_base

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
        nearest_test = fold_safe_spatial_background(
            test_indices, fold_ids, temporal_row_for_training_row,
            station_csi, fold_k, distance_km, method="nearest",
        )
        if not np.isfinite(nearest_test).all():
            raise RuntimeError(
                "Internal inconsistency: IDW was available but nearest-station "
                f"background was unavailable in fold {fold_k}."
            )
        X_train = append_idw_feature(X_train, idw_train)
        X_val = append_idw_feature(X_val, idw_val)
        X_test = append_idw_feature(X_test, idw_test)

        y_train_csi = y[train_indices_expanded]
        y_val_csi = y[val_indices]
        y_test = y[test_indices]
        y_train = y_train_csi - idw_train
        y_val = y_val_csi - idw_val
        y_test_residual = y_test - idw_test
        ts_test = timestamps_ns[test_indices]

        excluded_reconstructed_p2 = int(
            ((~ground_truth_mask) & (~held_out_station_mask)).sum()
        )
        print(f"Train samples : {len(y_train):,}")
        print(f"Val samples   : {len(y_val):,}")
        print(f"Test samples  : {len(y_test):,}")
        print(
            "No-source rows excluded: "
            f"train={dropped_train_no_source:,}, "
            f"val={dropped_val_no_source:,}, "
            f"test={dropped_test_no_source:,}"
        )
        print(
            "Reconstructed P2 rows excluded from this fold's training pool: "
            f"{excluded_reconstructed_p2:,}"
        )
        print(
            f"Enhancement rows oversampled: {n_enhancement_original:,} "
            f"({3*n_enhancement_original:,} additional rows)"
        )

        # Phi/Psi remain fixed. The 15 physical covariates and fold-safe IDW
        # level are standardized using only the current fold's training rows.
        X_train_sc, X_val_sc, X_test_sc, scaler_mean, scaler_std = standardise_residual_features(
            X_train, X_val, X_test, n_basis=n_total_basis
        )

        start = time.time()
        model, history = train_loss_fold(
            X_train_sc, y_train, X_val_sc, y_val, args.loss
        )
        elapsed_minutes = (time.time() - start) / 60.0
        history_df = pd.DataFrame(history)
        history_df.to_csv(VAL_DIR / f"fold_{fold_k}_history.csv", index=False)

        residual_pred = predict(model, X_test_sc)
        rmse_residual_raw = rmse(y_test_residual, residual_pred)
        r2_residual_raw = float(r2_score(y_test_residual, residual_pred))
        csi_pred_raw = idw_test + residual_pred

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
        csi_idw = np.clip(idw_test, 0.0, 1.3)
        csi_nearest = np.clip(nearest_test, 0.0, 1.3)
        csi_nsrdb = np.clip(bg_csi_test, 0.0, 1.3)
        ghi_pred = csi_pred * bg_clear_test
        ghi_idw = csi_idw * bg_clear_test
        ghi_nearest = csi_nearest * bg_clear_test
        ghi_nsrdb = csi_nsrdb * bg_clear_test
        ghi_true = csi_true * bg_clear_test
        daytime = bg_clear_test >= CLEARSKY_MIN

        rmse_csi = rmse(csi_true[daytime], csi_pred[daytime])
        r2_csi = float(r2_score(csi_true[daytime], csi_pred[daytime]))
        rmse_ghi = rmse(ghi_true[daytime], ghi_pred[daytime])
        r2_ghi = float(r2_score(ghi_true[daytime], ghi_pred[daytime]))
        rmse_ghi_idw = rmse(ghi_true[daytime], ghi_idw[daytime])

        print(f"Training time : {elapsed_minutes:.1f} min")
        print(
            "Best epoch    : "
            f"{int(history_df.loc[history_df.val_loss.idxmin(), 'epoch'])}"
        )
        print(f"CSI RMSE      : {rmse_csi:.4f}   R2={r2_csi:.4f}")
        print(f"GHI RMSE      : {rmse_ghi:.2f} W/m2   R2={r2_ghi:.4f}")
        print(f"IDW-only RMSE : {rmse_ghi_idw:.2f} W/m2")

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
                "residual_true": y_test_residual,
                "residual_pred": residual_pred,
                "csi_idw": csi_idw,
                "csi_nearest": csi_nearest,
                "csi_nsrdb": csi_nsrdb,
                "bg_csi": bg_csi_test,
                "ghi_true": ghi_true,
                "ghi_pred": ghi_pred,
                "ghi_idw": ghi_idw,
                "ghi_nearest": ghi_nearest,
                "ghi_nsrdb": ghi_nsrdb,
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
                "n_train_no_source_excluded": dropped_train_no_source,
                "n_validation_no_source_excluded": dropped_val_no_source,
                "n_test_no_source_excluded": dropped_test_no_source,
                "rmse_residual_raw": rmse_residual_raw,
                "r2_residual_raw": r2_residual_raw,
                "rmse_csi_postprocessed": rmse_csi,
                "r2_csi_postprocessed": r2_csi,
                "rmse_ghi": rmse_ghi,
                "r2_ghi": r2_ghi,
                "rmse_ghi_idw_only": rmse_ghi_idw,
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
        "fold", "test_station", "rmse_residual_raw", "r2_residual_raw",
        "rmse_ghi", "r2_ghi", "rmse_ghi_idw_only",
    ]
    summary_text = results[summary_columns].round(4).to_string(index=False)
    (VAL_DIR / "loso_summary.txt").write_text(
        "DK-5 Fold-Safe Station-IDW Residual LOSO Results\n"
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
        "controlled_comparator": "DK2_planar_eigen_TPS_K016_groundtruth",
        "single_intended_change": "direct_CSI_to_fold_safe_station_IDW_residual",
        "target": "csi_minus_fold_safe_station_idw",
        "input_order": [
            f"spatial_{args.spatial_basis}_basis",
            "temporal_gaussian_basis",
            "existing_15_covariates",
            "fold_safe_station_idw_background",
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
        "n_covariates": N_COVARIATES_EXPECTED + 1,
        "input_dimension": input_dimension,
        "architecture": "existing_monolithic_3_hidden_layer_mlp",
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "loss": "HuberLoss",
        "huber_delta": HUBER_DELTA,
        "idw_power": 2.0,
        "idw_training_policy": "exclude_target_station_and_held_out_station",
        "idw_testing_policy": "use_available_non_held_out_genuine_stations",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOP_PAT,
        "seed": args.seed,
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
    print("\nDirect-CSI comparator and earlier outputs were not modified.")


if __name__ == "__main__":
    main()
