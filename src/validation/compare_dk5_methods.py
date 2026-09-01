"""Compare DK-5 residual model with the four requested references.

Place at: src/validation/compare_dk5_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import OUTPUT_DIR  # noqa: E402
from src.baselines.baseline_comparison import compute_metrics  # noqa: E402

ROOT = Path(OUTPUT_DIR) / "experiments"
RESIDUAL_DIR = ROOT / "DK5_IDW_residual_TPS_K016_seed42_groundtruth" / "validation"
DIRECT_DIR = ROOT / "DK2_planar_eigen_TPS_K016_groundtruth" / "validation"
OUT_DIR = ROOT / "DK5_method_comparison_seed42"
STATIONS = ["S1", "S2", "S3", "P2"]


def metric_time(series: pd.Series) -> np.ndarray:
    return (
        pd.to_datetime(series, utc=True)
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )


def pooled_rmse(frame: pd.DataFrame) -> float:
    n = frame["n"].to_numpy(dtype=float)
    e = frame["rmse"].to_numpy(dtype=float)
    return float(np.sqrt(np.sum(n * e * e) / np.sum(n)))


def add_metrics(
    rows: list[dict], method: str, station: str,
    truth: np.ndarray, prediction: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    metrics = compute_metrics(truth, prediction, ts=timestamps)
    rows.append(
        {"method": method, "test_station": station, "n": len(truth), **metrics}
    )


def main() -> None:
    rows: list[dict] = []
    alignment_rows: list[dict] = []
    for fold, station in enumerate(STATIONS):
        residual_path = RESIDUAL_DIR / f"fold_{fold}_{station}_predictions.csv"
        direct_path = DIRECT_DIR / f"fold_{fold}_{station}_predictions.csv"
        if not residual_path.exists():
            raise FileNotFoundError(residual_path)
        if not direct_path.exists():
            raise FileNotFoundError(direct_path)

        residual = pd.read_csv(residual_path)
        direct = pd.read_csv(direct_path)
        residual = residual[residual["bg_clearsky"] >= 10.0].copy()
        direct = direct[direct["bg_clearsky"] >= 10.0].copy()

        required = {"ghi_true", "ghi_pred", "ghi_idw", "ghi_nearest", "ghi_nsrdb"}
        missing = required - set(residual.columns)
        if missing:
            raise ValueError(f"{residual_path} lacks columns {sorted(missing)}")

        # DK-5 legitimately excludes timestamps for which no fold-safe source
        # observation exists. Compare every method on the exact intersection
        # of timestamps so all reported errors use identical truth rows.
        residual["timestamp_key"] = pd.to_datetime(
            residual["datetime_local"], utc=True
        )
        direct["timestamp_key"] = pd.to_datetime(
            direct["datetime_local"], utc=True
        )
        if residual["timestamp_key"].duplicated().any():
            raise ValueError(f"Duplicate DK-5 timestamps for {station}")
        if direct["timestamp_key"].duplicated().any():
            raise ValueError(f"Duplicate direct-model timestamps for {station}")

        n_residual_before = len(residual)
        n_direct_before = len(direct)
        direct_for_merge = direct[
            ["timestamp_key", "ghi_true", "ghi_pred"]
        ].rename(
            columns={
                "ghi_true": "ghi_true_direct",
                "ghi_pred": "ghi_pred_direct",
            }
        )
        aligned = residual.merge(
            direct_for_merge,
            on="timestamp_key",
            how="inner",
            validate="one_to_one",
        ).sort_values("timestamp_key")
        if aligned.empty:
            raise ValueError(f"No common comparison timestamps for {station}")

        alignment_rows.append({
            "fold": fold,
            "test_station": station,
            "dk5_daytime_rows": n_residual_before,
            "direct_daytime_rows": n_direct_before,
            "common_daytime_rows": len(aligned),
            "dk5_rows_not_common": n_residual_before - len(aligned),
            "direct_rows_not_common": n_direct_before - len(aligned),
        })

        truth = aligned["ghi_true"].to_numpy()
        if not np.allclose(
            truth, aligned["ghi_true_direct"].to_numpy(), atol=1e-5,
            equal_nan=False,
        ):
            raise ValueError(f"Residual/direct truth differs for {station}")
        timestamps = (
            aligned["timestamp_key"].dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )

        add_metrics(rows, "station_IDW", station, truth,
                    aligned["ghi_idw"].to_numpy(), timestamps)
        add_metrics(rows, "direct_CSI_TPS", station, truth,
                    aligned["ghi_pred_direct"].to_numpy(), timestamps)
        add_metrics(rows, "nearest_station", station, truth,
                    aligned["ghi_nearest"].to_numpy(), timestamps)
        add_metrics(rows, "NSRDB", station, truth,
                    aligned["ghi_nsrdb"].to_numpy(), timestamps)
        add_metrics(rows, "IDW_residual_TPS", station, truth,
                    aligned["ghi_pred"].to_numpy(), timestamps)

    metrics = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT_DIR / "method_fold_metrics.csv", index=False)
    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(OUT_DIR / "timestamp_alignment.csv", index=False)

    complete = metrics[metrics["test_station"].isin(["S1", "S2", "S3"])]
    summary_rows = []
    for method, group in complete.groupby("method", sort=False):
        summary_rows.append({
            "method": method,
            "pooled_rmse_complete3": pooled_rmse(group),
            "macro_rmse_complete3": float(group["rmse"].mean()),
            "worst_rmse_complete3": float(group["rmse"].max()),
            "macro_mae_complete3": float(group["mae"].mean()),
            "macro_r2_complete3": float(group["r2"].mean()),
            "macro_abs_bias_complete3": float(group["bias"].abs().mean()),
            "macro_peak_rmse_complete3": float(group["peak_rmse"].mean()),
            "macro_ramp_rmse_complete3": float(group["ramp_rmse"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("pooled_rmse_complete3")
    summary.to_csv(OUT_DIR / "method_complete_station_summary.csv", index=False)

    p2 = metrics[metrics["test_station"] == "P2"].sort_values("rmse")
    p2.to_csv(OUT_DIR / "method_p2_summary.csv", index=False)

    print("=" * 88)
    print("DK-5 METHOD COMPARISON — SEED 42")
    print("=" * 88)
    print("\nTimestamp alignment (all methods evaluated on common rows):")
    print(alignment.to_string(index=False))
    print("\nStation-level metrics:")
    columns = [
        "method", "test_station", "rmse", "mae", "r2", "bias",
        "peak_rmse", "ramp_rmse",
    ]
    print(metrics[columns].round(4).to_string(index=False))
    print("\nComplete-station S1/S2/S3 summary:")
    print(summary.round(4).to_string(index=False))
    print("\nP2 genuine-winter summary:")
    print(p2[columns].round(4).to_string(index=False))
    print(f"\nOutputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
