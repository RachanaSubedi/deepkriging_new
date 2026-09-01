"""Summarize the three-seed Wendland-411 vs eigen-TPS-K16 comparison.

Place at: src/validation/summarize_spatial_basis_seed_check.py

The existing reference runs are treated as seed 42:
  Wendland: DK1A_temporal_30d_groundtruth
  TPS K16 : DK2_planar_eigen_TPS_K016_groundtruth
Additional seed runs are produced by train_spatial_basis_seed_check.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import OUTPUT_DIR  # noqa: E402

EXPERIMENT_ROOT = Path(OUTPUT_DIR) / "experiments"
OUT_DIR = EXPERIMENT_ROOT / "DK2_spatial_basis_seed_summary"

RUNS = {
    ("wendland_K411", 42): "DK1A_temporal_30d_groundtruth",
    ("eigen_TPS_K16", 42): "DK2_planar_eigen_TPS_K016_groundtruth",
    ("wendland_K411", 123): "DK2_seedcheck_wendland_K411_seed123_groundtruth",
    ("eigen_TPS_K16", 123): "DK2_seedcheck_eigen_TPS_K016_seed123_groundtruth",
    ("wendland_K411", 2026): "DK2_seedcheck_wendland_K411_seed2026_groundtruth",
    ("eigen_TPS_K16", 2026): "DK2_seedcheck_eigen_TPS_K016_seed2026_groundtruth",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize DK-2 seed robustness")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pooled_rmse(frame: pd.DataFrame) -> float:
    n = frame["n_test"].to_numpy(dtype=float)
    e = frame["rmse_ghi"].to_numpy(dtype=float)
    return float(np.sqrt(np.sum(n * e * e) / np.sum(n)))


def load_all() -> pd.DataFrame:
    frames = []
    missing = []
    for (basis, seed), folder in RUNS.items():
        path = EXPERIMENT_ROOT / folder / "validation" / "loso_results.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        required = {"test_station", "n_test", "rmse_ghi", "r2_ghi"}
        absent = required - set(df.columns)
        if absent:
            raise ValueError(f"{path} lacks columns {sorted(absent)}")
        if set(df["test_station"]) != {"S1", "S2", "S3", "P2"}:
            raise ValueError(f"Unexpected station set in {path}")
        df = df.copy()
        df.insert(0, "basis", basis)
        df.insert(1, "seed", seed)
        df.insert(2, "experiment_folder", folder)
        frames.append(df)
    if missing:
        raise FileNotFoundError(
            "The following expected runs are missing:\n  " + "\n  ".join(missing)
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    marker = OUT_DIR / "seed_aggregate.csv"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"{marker} exists; use --overwrite intentionally")

    folds = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT_DIR / "all_seed_fold_results.csv", index=False)

    run_rows = []
    for (basis, seed), group in folds.groupby(["basis", "seed"], sort=True):
        complete = group[group["test_station"].isin(["S1", "S2", "S3"])]
        p2 = group[group["test_station"] == "P2"].iloc[0]
        run_rows.append({
            "basis": basis,
            "seed": int(seed),
            "macro_rmse_all4": float(group["rmse_ghi"].mean()),
            "pooled_rmse_all4": pooled_rmse(group),
            "macro_r2_all4": float(group["r2_ghi"].mean()),
            "macro_rmse_complete3": float(complete["rmse_ghi"].mean()),
            "pooled_rmse_complete3": pooled_rmse(complete),
            "worst_rmse_complete3": float(complete["rmse_ghi"].max()),
            "p2_rmse": float(p2["rmse_ghi"]),
            "p2_r2": float(p2["r2_ghi"]),
        })
    runs = pd.DataFrame(run_rows)
    runs.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)

    metric_cols = [c for c in runs.columns if c not in {"basis", "seed"}]
    aggregate_rows = []
    for basis, group in runs.groupby("basis", sort=True):
        row = {"basis": basis, "n_seeds": len(group)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        "pooled_rmse_complete3_mean"
    )
    aggregate.to_csv(marker, index=False)

    # Paired differences use identical seeds: negative means TPS is better.
    pivot = runs.pivot(index="seed", columns="basis", values=metric_cols)
    paired = pd.DataFrame({"seed": pivot.index})
    for col in metric_cols:
        paired[f"TPS_minus_Wendland__{col}"] = (
            pivot[(col, "eigen_TPS_K16")]
            - pivot[(col, "wendland_K411")]
        ).to_numpy()
    paired.to_csv(OUT_DIR / "paired_seed_differences.csv", index=False)

    station_seed = (
        folds.groupby(["basis", "test_station"], sort=True)
        .agg(
            rmse_mean=("rmse_ghi", "mean"),
            rmse_std=("rmse_ghi", "std"),
            rmse_min=("rmse_ghi", "min"),
            rmse_max=("rmse_ghi", "max"),
            r2_mean=("r2_ghi", "mean"),
            r2_std=("r2_ghi", "std"),
        )
        .reset_index()
    )
    station_seed.to_csv(OUT_DIR / "per_station_seed_stability.csv", index=False)

    winner = aggregate.iloc[0]["basis"]
    report = {
        "primary_selection_metric": "mean pooled GHI RMSE across S1/S2/S3",
        "primary_winner": winner,
        "important_secondary_checks": [
            "worst complete-station RMSE",
            "seed-to-seed standard deviation",
            "P2 genuine-winter RMSE as a secondary check",
            "178-PV spatial plausibility before final acceptance",
        ],
        "warning": (
            "This is a three-seed robustness screen, not uncertainty "
            "quantification and not evidence that four stations identify a "
            "high-dimensional spatial field."
        ),
    }
    (OUT_DIR / "selection_note.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("SPATIAL-BASIS SEED ROBUSTNESS SUMMARY")
    print("=" * 78)
    print("\nPer-seed results:")
    print(runs.round(4).to_string(index=False))
    print("\nThree-seed aggregate (primary columns):")
    show = [
        "basis", "pooled_rmse_complete3_mean", "pooled_rmse_complete3_std",
        "worst_rmse_complete3_mean", "p2_rmse_mean", "p2_rmse_std",
    ]
    print(aggregate[show].round(4).to_string(index=False))
    print(f"\nPrimary winner: {winner}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
