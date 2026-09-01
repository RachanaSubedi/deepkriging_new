"""
Train the spatial-only DK-0 model with the shared blocked-day split.

Place this file at:
    src/train/train_dk0_matched.py

Prerequisite:
    python src/train/shared_blocked_day_split.py --validation-fraction 0.20

Run:
    python src/train/train_dk0_matched.py --p2-provenance \
        "data/raw/stations/station_p2_full_year_GHI_zeroshot.csv"

This is a matched comparator for DK-1. Relative to the original DK-0, the
model, direct-CSI target, spatial bases, 15 covariates, Huber loss, optimizer,
oversampling, and post-processing are unchanged. It uses complete validation
days distributed across the year and shared by every station. Additionally,
only genuine measured P2 rows are eligible for LOSO training, validation, or
testing; reconstructed P2 rows are excluded to prevent leakage from the S2/S3
anchors used by the P2 Transformer.

All outputs are isolated under:
    outputs/experiments/DK0_matched_groundtruth/

The original outputs/models and outputs/validation folders are not modified.
"""

from __future__ import annotations

import json
import argparse
import hashlib
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
    STATIONS,
    TRAIN_DIR,
    WEIGHT_DECAY,
)
from src.model import DeepKriging, count_parameters  # noqa: E402
from src.train.shared_blocked_day_split import (  # noqa: E402
    DEFAULT_SPLIT_DIR,
    load_shared_split,
)


EXPERIMENT_NAME = "DK0_matched_groundtruth"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments" / EXPERIMENT_NAME
MODEL_DIR = EXPERIMENT_DIR / "models"
VAL_DIR = EXPERIMENT_DIR / "validation"
FIG_DIR = EXPERIMENT_DIR / "figures"
SPLIT_SNAPSHOT_DIR = EXPERIMENT_DIR / "split_snapshot"

SEED = 42
EARLY_STOP_PAT = 30
CLEARSKY_MIN = 10.0
HUBER_DELTA = 0.1
N_COVARIATES_EXPECTED = 15
STATION_NAMES = list(STATIONS.keys())
DEVICE = torch.device("cpu")

FOLD_COLORS = {
    "S1": "#e63946",
    "S2": "#2a9d8f",
    "S3": "#e76f51",
    "P2": "#264653",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the matched DK0 model while allowing only genuine P2 "
            "measurements in LOSO training, validation, and testing."
        )
    )
    parser.add_argument(
        "--p2-provenance",
        type=Path,
        required=True,
        help=(
            "P2 CSV containing a timestamp column and a source column. "
            "Only rows with source='measured' are treated as ground truth."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    by_lower = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"Could not find any of {candidates}. Available columns: "
        f"{list(frame.columns)}"
    )


def provenance_times_to_local_naive(values: pd.Series) -> pd.DatetimeIndex:
    """Convert provenance timestamps to local timezone-naive timestamps."""
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            f"P2 provenance contains {int(parsed.isna().sum())} invalid timestamps"
        )

    index = pd.DatetimeIndex(parsed)
    if index.tz is None:
        # The project CSV stores local wall-clock timestamps without a zone.
        local_naive = index
    else:
        local_naive = index.tz_convert("America/Los_Angeles").tz_localize(None)
    return local_naive


def build_ground_truth_mask(
    timestamps_ns: np.ndarray,
    fold_ids: np.ndarray,
    p2_provenance_path: Path,
) -> tuple[np.ndarray, dict]:
    """
    Mark S1/S2/S3 rows as observed and retain only measured P2 timestamps.

    Transformer-imputed, IDW-fallback, and NSRDB-fallback P2 rows are excluded
    from LOSO training, validation, and testing.
    """
    path = Path(p2_provenance_path)
    if not path.exists():
        raise FileNotFoundError(f"P2 provenance file not found: {path}")

    provenance = pd.read_csv(path)
    datetime_column = _find_column(
        provenance,
        ["datetime_local", "datetime", "timestamp", "time", "date_time"],
    )
    source_column = _find_column(provenance, ["source", "provenance"])

    sources = provenance[source_column].astype(str).str.strip().str.lower()
    measured_rows = sources.eq("measured")
    if not measured_rows.any():
        raise ValueError("P2 provenance has no rows with source='measured'")

    measured_times = provenance_times_to_local_naive(
        provenance.loc[measured_rows, datetime_column]
    ).unique()

    training_times_local = (
        pd.to_datetime(timestamps_ns, unit="ns", utc=True)
        .tz_convert("America/Los_Angeles")
        .tz_localize(None)
    )

    p2_index = STATION_NAMES.index("P2")
    p2_rows = fold_ids == p2_index
    p2_measured_rows = p2_rows & np.asarray(
        training_times_local.isin(measured_times), dtype=bool
    )

    ground_truth_mask = ~p2_rows
    ground_truth_mask[p2_measured_rows] = True

    n_p2_total = int(p2_rows.sum())
    n_p2_measured = int(p2_measured_rows.sum())
    if n_p2_measured < 2:
        raise ValueError(
            "Fewer than two measured P2 rows aligned with timestamps.npy. "
            "Check timestamp columns and timezone conventions."
        )

    summary = {
        "p2_provenance_path": str(path.resolve()),
        "p2_provenance_sha256": sha256_file(path),
        "datetime_column": str(datetime_column),
        "source_column": str(source_column),
        "accepted_source": "measured",
        "p2_rows_in_training_matrix": n_p2_total,
        "p2_genuine_rows_retained": n_p2_measured,
        "p2_reconstructed_rows_excluded": n_p2_total - n_p2_measured,
        "complete_station_policy": "S1/S2/S3 rows retained",
    }
    return ground_truth_mask, summary


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardise(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_basis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize only the 15 environmental/time covariates.

    The existing Wendland basis columns remain unchanged in [0, 1], matching
    DK-0. The scaler is fitted only to the current LOSO fold's training rows.
    """
    if X_train.shape[1] != n_basis + N_COVARIATES_EXPECTED:
        raise ValueError(
            f"Expected {n_basis} spatial bases + {N_COVARIATES_EXPECTED} "
            f"covariates, but X has {X_train.shape[1]} columns"
        )

    mean = np.zeros(X_train.shape[1], dtype=np.float32)
    std = np.ones(X_train.shape[1], dtype=np.float32)

    cov_mean = X_train[:, n_basis:].mean(axis=0)
    cov_std = X_train[:, n_basis:].std(axis=0)
    cov_std[cov_std < 1e-8] = 1.0
    mean[n_basis:] = cov_mean
    std[n_basis:] = cov_std

    return (
        (X_train - mean) / std,
        (X_val - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
    )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[DeepKriging, list[dict[str, float]]]:
    model = DeepKriging(
        X_train.shape[1], HIDDEN_SIZE, DROPOUT
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

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
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
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
                f"val={val_loss:.5f}  "
                f"patience={patience_count}/{EARLY_STOP_PAT}"
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


def predict(model: DeepKriging, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
    return output.cpu().numpy()


def apply_dk0_postprocessing(
    csi_pred_raw: np.ndarray,
    bg_csi: np.ndarray,
    bg_clearsky: np.ndarray,
) -> np.ndarray:
    """Apply the unchanged DK-0 low-sun blend and CSI caps."""
    csi_pred = np.clip(csi_pred_raw.copy(), 0.0, 1.3)

    low_sun = bg_clearsky < 200.0
    if low_sun.any():
        blend_weight = np.clip(bg_clearsky[low_sun] / 200.0, 0.0, 1.0)
        csi_pred[low_sun] = (
            blend_weight * csi_pred_raw[low_sun]
            + (1.0 - blend_weight) * np.clip(bg_csi[low_sun], 0.0, 1.0)
        )

    final_cap = np.where(
        bg_clearsky < 100.0,
        0.85,
        np.where(bg_clearsky < 200.0, 0.90,
                 np.where(bg_clearsky < 350.0, 0.95, 1.3)),
    )
    return np.clip(np.minimum(csi_pred, final_cap), 0.0, 1.3)


def plot_loss_curves() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "DK0 Matched Ground Truth - Shared Blocked-Day Split",
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


def snapshot_split() -> None:
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


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    print("=" * 72)
    print("DK0 MATCHED GROUND TRUTH - LEAKAGE-SAFE P2 LOSO")
    print("=" * 72)

    X = np.load(TRAIN_DIR / "X.npy")
    y = np.load(TRAIN_DIR / "y.npy")
    fold_ids = np.load(TRAIN_DIR / "fold_ids.npy")
    timestamps_ns = np.load(TRAIN_DIR / "timestamps.npy")
    spatial_basis = np.load(BASIS_DIR / "Phi_stations_scaled.npy")
    n_spatial_basis = int(spatial_basis.shape[1])

    if not (len(X) == len(y) == len(fold_ids) == len(timestamps_ns)):
        raise ValueError("Training arrays do not have the same number of rows")
    if X.shape[1] != n_spatial_basis + N_COVARIATES_EXPECTED:
        raise ValueError(
            f"DK0_matched expects {n_spatial_basis} spatial basis columns and "
            f"{N_COVARIATES_EXPECTED} covariates, but X has {X.shape[1]} columns"
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
    snapshot_split()
    np.save(EXPERIMENT_DIR / "ground_truth_mask.npy", ground_truth_mask)
    (EXPERIMENT_DIR / "p2_ground_truth_summary.json").write_text(
        json.dumps(provenance_summary, indent=2), encoding="utf-8"
    )

    bg_csi = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    bg_clear = pd.read_parquet(BG_DIR / "bg_clearsky_stations.parquet")

    temporary_model = DeepKriging(X.shape[1], HIDDEN_SIZE, DROPOUT)
    print(f"Input dimension       : {X.shape[1]}")
    print(f"Spatial basis columns : {n_spatial_basis}")
    print(f"Covariates            : {N_COVARIATES_EXPECTED}")
    print(f"Parameters            : {count_parameters(temporary_model):,}")
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

        excluded_reconstructed_p2 = int(
            ((~ground_truth_mask) & (~held_out_station_mask)).sum()
        )

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        ts_test = timestamps_ns[test_mask]

        # Preserve the original DK-0 enhancement-event oversampling.
        enhancement_mask = y_train > 0.85
        n_enhancement_original = int(enhancement_mask.sum())
        if n_enhancement_original > 0:
            X_extra = np.tile(X_train[enhancement_mask], (3, 1))
            y_extra = np.tile(y_train[enhancement_mask], 3)
            X_train = np.vstack([X_train, X_extra])
            y_train = np.concatenate([y_train, y_extra])

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

        X_train_sc, X_val_sc, X_test_sc, scaler_mean, scaler_std = standardise(
            X_train, X_val, X_test, n_basis=n_spatial_basis
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
        print(f"Best epoch    : {int(history_df.loc[history_df.val_loss.idxmin(), 'epoch'])}")
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

    results = pd.DataFrame(fold_results)
    results.to_csv(VAL_DIR / "loso_results.csv", index=False)
    summary_columns = [
        "fold", "test_station", "rmse_csi_raw", "r2_csi_raw",
        "rmse_ghi", "r2_ghi",
    ]
    summary_text = results[summary_columns].round(4).to_string(index=False)
    (VAL_DIR / "loso_summary.txt").write_text(
        "DK0 Matched Ground-Truth LOSO Results\n"
        + "=" * 50
        + "\n"
        + summary_text
        + "\n\n"
        + f"Mean GHI RMSE: {results.rmse_ghi.mean():.4f}\n"
        + f"Mean GHI R2: {results.r2_ghi.mean():.4f}\n",
        encoding="utf-8",
    )

    run_config = {
        "experiment": EXPERIMENT_NAME,
        "target": "direct_csi",
        "spatial_basis": "existing_wendland",
        "n_spatial_basis": n_spatial_basis,
        "temporal_basis": None,
        "n_covariates": N_COVARIATES_EXPECTED,
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
    }
    (EXPERIMENT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_loss_curves()

    print("\n" + "=" * 72)
    print("DK0 MATCHED GROUND-TRUTH TRAINING COMPLETE")
    print("=" * 72)
    print(summary_text)
    print(f"\nMean GHI RMSE: {results.rmse_ghi.mean():.4f}")
    print(f"Mean GHI R2  : {results.r2_ghi.mean():.4f}")
    print(f"Outputs      : {EXPERIMENT_DIR}")
    print("\nYour original DK0 output folders were not modified.")


if __name__ == "__main__":
    main()
