"""Compare seed-42 Huber and MSE under the fixed monolithic TPS-K16 model.

Place at: src/validation/compare_dk4_loss.py
Run after train_dk4_loss.py completes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import OUTPUT_DIR  # noqa: E402
from src.baselines.baseline_comparison import compute_metrics  # noqa: E402

ROOT = Path(OUTPUT_DIR) / "experiments"
RUNS = {
    "Huber": ROOT / "DK2_planar_eigen_TPS_K016_groundtruth" / "validation",
    "MSE": ROOT / "DK4_loss_MSE_TPS_K016_seed42_groundtruth" / "validation",
}
OUT_DIR = ROOT / "DK4_loss_comparison_seed42"
STATIONS = ["S1", "S2", "S3", "P2"]


def main() -> None:
    rows = []
    for loss_name, directory in RUNS.items():
        for fold, station in enumerate(STATIONS):
            path = directory / f"fold_{fold}_{station}_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path, parse_dates=["datetime_local"])
            df = df[df["bg_clearsky"] >= 10.0].copy()
            # Prediction files contain timezone-aware local timestamps. The
            # shared compute_metrics() helper calls pd.to_datetime() without
            # utc=True, which cannot consume an object array of aware Python
            # datetimes across PST/PDT. Convert once to a uniform UTC timeline
            # and remove timezone metadata. Elapsed 5-minute differences are
            # preserved, including across daylight-saving transitions.
            metric_timestamps = (
                pd.to_datetime(df["datetime_local"], utc=True)
                .dt.tz_localize(None)
                .to_numpy(dtype="datetime64[ns]")
            )
            metrics = compute_metrics(
                df["ghi_true"].to_numpy(),
                df["ghi_pred"].to_numpy(),
                ts=metric_timestamps,
            )
            rows.append(
                {"loss": loss_name, "fold": fold, "test_station": station,
                 "n": len(df), **metrics}
            )

    results = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "loss_fold_metrics.csv", index=False)

    complete = results[results["test_station"].isin(["S1", "S2", "S3"])]
    metric_names = [
        name for name in ["rmse", "mae", "r2", "bias", "peak_rmse", "ramp_rmse"]
        if name in results.columns
    ]
    aggregate = complete.groupby("loss")[metric_names].mean().reset_index()
    aggregate.to_csv(OUT_DIR / "loss_complete_station_macro.csv", index=False)

    pivot = results.pivot(index="test_station", columns="loss", values=metric_names)
    differences = pd.DataFrame(index=pivot.index)
    for metric in metric_names:
        differences[f"MSE_minus_Huber__{metric}"] = (
            pivot[(metric, "MSE")] - pivot[(metric, "Huber")]
        )
    differences.reset_index().to_csv(
        OUT_DIR / "loss_station_differences.csv", index=False
    )

    print("=" * 78)
    print("DK-4 HUBER VERSUS MSE — SEED 42")
    print("=" * 78)
    print("\nFold metrics:")
    print(results[["loss", "test_station", *metric_names]].round(4).to_string(index=False))
    print("\nMacro metrics over complete stations S1/S2/S3:")
    print(aggregate.round(4).to_string(index=False))
    print("\nMSE minus Huber by station (negative error differences favor MSE):")
    print(differences.reset_index().round(4).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
