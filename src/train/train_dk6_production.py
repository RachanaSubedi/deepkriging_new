"""Train final all-station direct-CSI DeepKriging production models.

Place at:
    src/train/train_dk6_production.py

Prerequisites:
    python src/train/shared_blocked_day_split.py --validation-fraction 0.20
    python src/data_prep/temporal_basis.py
    python src/data_prep/planar_eigen_tps_basis.py --max-basis 32

Run:
    python src/train/train_dk6_production.py --p2-provenance \
        "data/raw/stations/station_p2_full_year_GHI_zeroshot.csv"

Selected formulation (fixed before this production run):
    - direct CSI target
    - planar eigen-TPS spatial basis, K=16
    - coarse 30-day Gaussian temporal basis, K=16
    - existing 15 covariates and monolithic MLP
    - Huber loss, delta=0.1
    - seeds 42, 123, and 2026

For each seed, Stage A uses the shared blocked-day train/validation masks across
all stations to select the best epoch. Stage B reinitializes the network and
trains for exactly that many epochs on all genuine ground observations. Thus
the final checkpoint uses the former validation days without choosing its
training duration from production loss. Reconstructed P2 rows are never used.

This script does not perform LOSO evaluation and must not replace DK-2 LOSO
metrics. Its checkpoints are only for prediction at the 178 PV locations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import (  # noqa: E402
    BASIS_DIR,
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    OUTPUT_DIR,
    PROCESSED_DIR,
    TRAIN_DIR,
    WEIGHT_DECAY,
)
from src.model import DeepKriging, count_parameters  # noqa: E402
from src.train.shared_blocked_day_split import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    load_shared_split,
)
from src.train.train_dk0_matched import (  # noqa: E402
    HUBER_DELTA,
    N_COVARIATES_EXPECTED,
    build_ground_truth_mask,
    make_loader,
    set_seed,
    standardise,
    train_one_fold,
)
from src.train.train_dk2_spatial_basis import (  # noqa: E402
    TPS_DIR,
    assemble_dk1_features,
    load_temporal_artifact,
    map_rows_to_unique_time,
    select_temporal_levels,
    sha256_file,
)


SEEDS_DEFAULT = (42, 123, 2026)
N_SPATIAL_BASIS = 16
SELECTED_TEMPORAL_LEVELS = ("coarse_30d",)
EXPERIMENT_NAME = "DK6_production_direct_TPS_K016_three_seed"
EXPERIMENT_DIR = Path(OUTPUT_DIR) / "experiments" / EXPERIMENT_NAME
MODEL_DIR = EXPERIMENT_DIR / "models"
HISTORY_DIR = EXPERIMENT_DIR / "histories"
DEVICE = torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train three all-station production DeepKriging checkpoints."
    )
    parser.add_argument(
        "--p2-provenance",
        type=Path,
        required=True,
        help="P2 CSV with timestamps and source='measured' provenance.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS_DEFAULT),
        help="Production ensemble seeds (default: 42 123 2026).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of already-completed seed checkpoints.",
    )
    return parser.parse_args()


def expand_enhancement_rows(
    indices: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, int]:
    """Match the selected LOSO model's clear/enhancement oversampling."""
    enhancement = indices[y[indices] > 0.85]
    if len(enhancement):
        expanded = np.concatenate([indices, np.tile(enhancement, 3)])
    else:
        expanded = indices
    return expanded, int(len(enhancement))


def train_fixed_epochs(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
) -> tuple[DeepKriging, list[dict[str, float]]]:
    """Train a fresh production network for a preselected epoch count."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model = DeepKriging(X.shape[1], HIDDEN_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    loader = make_loader(X, y, BATCH_SIZE, shuffle=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        train_loss = total / len(y)
        history.append({"epoch": epoch, "train_loss": train_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"    production epoch {epoch:>3d}/{epochs:<3d} "
                f"train={train_loss:.5f}"
            )
    return model, history


def main() -> None:
    args = parse_args()
    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required")

    X_dk0 = np.load(Path(TRAIN_DIR) / "X.npy")
    y = np.load(Path(TRAIN_DIR) / "y.npy")
    fold_ids = np.load(Path(TRAIN_DIR) / "fold_ids.npy")
    timestamps_ns = np.load(Path(TRAIN_DIR) / "timestamps.npy").astype(
        np.int64, copy=False
    )
    original_wendland = np.load(Path(BASIS_DIR) / "Phi_stations_scaled.npy")
    original_wendland_columns = int(original_wendland.shape[1])

    if not (len(X_dk0) == len(y) == len(fold_ids) == len(timestamps_ns)):
        raise ValueError("Training arrays do not have equal row counts")
    if X_dk0.shape[1] != original_wendland_columns + N_COVARIATES_EXPECTED:
        raise ValueError(
            "X.npy does not have the expected Wendland-plus-covariate layout"
        )

    tps_path = Path(TPS_DIR) / "Phi_stations.npy"
    tps_spec_path = Path(TPS_DIR) / "basis_spec.json"
    if not tps_path.exists() or not tps_spec_path.exists():
        raise FileNotFoundError(
            "Missing planar eigen-TPS artifact. Run "
            "python src/data_prep/planar_eigen_tps_basis.py --max-basis 32"
        )
    tps_all = np.load(tps_path)
    if tps_all.shape[1] < N_SPATIAL_BASIS:
        raise ValueError(
            f"TPS artifact has {tps_all.shape[1]} columns; "
            f"{N_SPATIAL_BASIS} are required"
        )
    spatial_basis = tps_all[:, :N_SPATIAL_BASIS].astype(np.float32, copy=False)

    psi_all, temporal_unique_ns, full_temporal_spec, temporal_hashes = (
        load_temporal_artifact()
    )
    psi_unique, temporal_spec = select_temporal_levels(
        psi_all, full_temporal_spec, SELECTED_TEMPORAL_LEVELS
    )
    del psi_all
    n_temporal_basis = int(psi_unique.shape[1])
    n_total_basis = N_SPATIAL_BASIS + n_temporal_basis
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
    selection_train_indices = np.flatnonzero(
        shared_train_mask & ground_truth_mask
    )
    selection_val_indices = np.flatnonzero(shared_val_mask & ground_truth_mask)
    production_indices = np.flatnonzero(ground_truth_mask)
    if np.intersect1d(selection_train_indices, selection_val_indices).size:
        raise RuntimeError("Selection training and validation rows overlap")

    selection_train_expanded, n_select_enhancement = expand_enhancement_rows(
        selection_train_indices, y
    )
    production_expanded, n_production_enhancement = expand_enhancement_rows(
        production_indices, y
    )

    print("=" * 78)
    print("DK-6 FINAL ALL-STATION PRODUCTION TRAINING")
    print("=" * 78)
    print(f"Input dimension          : {input_dimension}")
    print(f"Spatial basis            : planar eigen-TPS K={N_SPATIAL_BASIS}")
    print(f"Temporal basis           : coarse 30-day K={n_temporal_basis}")
    print(f"Covariates               : {N_COVARIATES_EXPECTED}")
    print(f"Parameters               : {count_parameters(DeepKriging(input_dimension, HIDDEN_SIZE, DROPOUT)):,}")
    print(f"Target / loss            : direct CSI / Huber(delta={HUBER_DELTA})")
    print(f"Seeds                    : {list(seeds)}")
    print(f"Selection train rows     : {len(selection_train_indices):,}")
    print(f"Selection validation rows: {len(selection_val_indices):,}")
    print(f"Production genuine rows  : {len(production_indices):,}")
    print(
        "P2 genuine retained      : "
        f"{provenance_summary['p2_genuine_rows_retained']:,}"
    )
    print(
        "P2 reconstructed excluded: "
        f"{provenance_summary['p2_reconstructed_rows_excluded']:,}"
    )
    print(f"Output directory         : {EXPERIMENT_DIR}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EXPERIMENT_DIR / "ground_truth_mask.npy", ground_truth_mask)

    # Assemble fixed feature matrices once. The seed affects only network
    # initialization and shuffled mini-batch order.
    X_select_train = assemble_dk1_features(
        X_dk0, selection_train_expanded, original_wendland_columns,
        fold_ids, spatial_basis, temporal_row_for_training_row, psi_unique,
    )
    X_select_val = assemble_dk1_features(
        X_dk0, selection_val_indices, original_wendland_columns,
        fold_ids, spatial_basis, temporal_row_for_training_row, psi_unique,
    )
    y_select_train = y[selection_train_expanded]
    y_select_val = y[selection_val_indices]
    X_select_train_sc, X_select_val_sc, _, _, _ = standardise(
        X_select_train, X_select_val, X_select_val[:1], n_basis=n_total_basis
    )

    X_production = assemble_dk1_features(
        X_dk0, production_expanded, original_wendland_columns,
        fold_ids, spatial_basis, temporal_row_for_training_row, psi_unique,
    )
    y_production = y[production_expanded]
    X_production_sc, _, _, production_mean, production_std = standardise(
        X_production, X_production[:1], X_production[:1],
        n_basis=n_total_basis,
    )

    seed_rows: list[dict[str, float | int]] = []
    for seed in seeds:
        checkpoint = MODEL_DIR / f"production_seed{seed}_best.pt"
        if checkpoint.exists() and not args.overwrite:
            raise FileExistsError(
                f"Production checkpoint already exists: {checkpoint}. "
                "Use --overwrite only for an intentional replacement."
            )

        print("\n" + "-" * 78)
        print(f"SEED {seed}: STAGE A — SELECT EPOCH USING BLOCKED VALIDATION DAYS")
        print("-" * 78)
        set_seed(seed)
        start_a = time.time()
        _, selection_history = train_one_fold(
            X_select_train_sc, y_select_train,
            X_select_val_sc, y_select_val,
        )
        selection_history_df = pd.DataFrame(selection_history)
        best_row = selection_history_df.loc[
            selection_history_df["val_loss"].idxmin()
        ]
        best_epoch = int(best_row["epoch"])
        best_val_loss = float(best_row["val_loss"])
        selection_minutes = (time.time() - start_a) / 60.0
        selection_history_df.to_csv(
            HISTORY_DIR / f"seed{seed}_selection_history.csv", index=False
        )
        print(
            f"Selected epoch={best_epoch}, val_loss={best_val_loss:.6f}, "
            f"time={selection_minutes:.1f} min"
        )

        print("\n" + "-" * 78)
        print(f"SEED {seed}: STAGE B — RETRAIN ON ALL GENUINE OBSERVATIONS")
        print("-" * 78)
        set_seed(seed)
        start_b = time.time()
        production_model, production_history = train_fixed_epochs(
            X_production_sc, y_production, best_epoch
        )
        production_minutes = (time.time() - start_b) / 60.0
        pd.DataFrame(production_history).to_csv(
            HISTORY_DIR / f"seed{seed}_production_history.csv", index=False
        )
        torch.save(production_model.state_dict(), checkpoint)
        seed_rows.append({
            "seed": seed,
            "selected_epoch": best_epoch,
            "best_selection_val_loss": best_val_loss,
            "selection_minutes": selection_minutes,
            "production_minutes": production_minutes,
            "production_final_train_loss": production_history[-1]["train_loss"],
        })
        print(f"Saved: {checkpoint}")

    np.save(MODEL_DIR / "production_scaler_mean.npy", production_mean)
    np.save(MODEL_DIR / "production_scaler_std.npy", production_std)
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(EXPERIMENT_DIR / "production_seed_summary.csv", index=False)

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "purpose": "production_prediction_at_178_PV_locations",
        "validation_evidence": "DK2_planar_eigen_TPS_K016_groundtruth_LOSO",
        "not_for_accuracy_reporting": True,
        "target": "direct_csi",
        "spatial_basis": "planar_eigen_tps",
        "n_spatial_basis": N_SPATIAL_BASIS,
        "temporal_basis": temporal_spec["version"],
        "selected_temporal_levels": list(SELECTED_TEMPORAL_LEVELS),
        "n_temporal_basis": n_temporal_basis,
        "n_covariates": N_COVARIATES_EXPECTED,
        "input_order": [
            "planar_eigen_tps_K16", "coarse_30d_temporal_K16",
            "existing_15_covariates",
        ],
        "architecture": "existing_monolithic_3_hidden_layer_mlp",
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "loss": "HuberLoss",
        "huber_delta": HUBER_DELTA,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seeds": list(seeds),
        "seed_ensemble_rule": "arithmetic_mean_of_raw_CSI_then_common_postprocessing",
        "epoch_selection": "shared_blocked_days_then_retrain_all_genuine_rows",
        "validation_days": split_summary["n_validation_days"],
        "production_genuine_rows_before_oversampling": int(len(production_indices)),
        "production_rows_after_oversampling": int(len(production_expanded)),
        "selection_enhancement_rows": n_select_enhancement,
        "production_enhancement_rows": n_production_enhancement,
        "p2_policy": "genuine_measured_only",
        "p2_genuine_rows": provenance_summary["p2_genuine_rows_retained"],
        "p2_reconstructed_rows_excluded": provenance_summary[
            "p2_reconstructed_rows_excluded"
        ],
        "temporal_artifact_sha256": temporal_hashes,
        "tps_basis_sha256": sha256_file(tps_path),
        "training_matrix_sha256": {
            "X.npy": sha256_file(Path(TRAIN_DIR) / "X.npy"),
            "y.npy": sha256_file(Path(TRAIN_DIR) / "y.npy"),
            "fold_ids.npy": sha256_file(Path(TRAIN_DIR) / "fold_ids.npy"),
            "timestamps.npy": sha256_file(Path(TRAIN_DIR) / "timestamps.npy"),
        },
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (EXPERIMENT_DIR / "p2_ground_truth_summary.json").write_text(
        json.dumps(provenance_summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("DK-6 PRODUCTION TRAINING COMPLETE")
    print("=" * 78)
    print(seed_summary.round(6).to_string(index=False))
    print(f"\nModels: {MODEL_DIR}")
    print(
        "These checkpoints are for 178-PV inference only. Continue reporting "
        "DK-2 LOSO metrics as the validation results."
    )


if __name__ == "__main__":
    main()
