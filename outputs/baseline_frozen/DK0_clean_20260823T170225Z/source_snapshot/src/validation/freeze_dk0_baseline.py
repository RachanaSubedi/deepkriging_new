"""
Freeze the clean direct-CSI DeepKriging baseline (DK0).

This script does NOT train or modify the model. Run it only after the
data-preparation, training, and PV-prediction steps listed in the handoff.

What it does
------------
1. Requires every LOSO model, scaler, history, prediction, and PV output.
2. Verifies that training_matrix/y.npy agrees with CSI recomputed from the
   current all_stations_GHI_5min_PST.csv. This catches stale training arrays.
3. Computes:
      - legacy metrics from the truth saved by train.py;
      - scientific metrics using the current measured GHI file directly;
      - genuine-P2-only primary metrics using the provenance source column;
      - gap-safe ramp RMSE using only consecutive timestamps;
      - clear/cloudy-smooth/broken-cloud GHI metrics.
4. Reports both macro (equal station weight) and pooled (equal row weight)
   summaries for the primary scientific evaluation.
5. Archives outputs, source code, git state, and SHA-256 input manifests.

Run from the repository root:

    python src/validation/freeze_dk0_baseline.py \
        --label DK0_clean \
        --p2-provenance "data/raw/stations/46.78, -119.22 2024.csv"

The script assumes that naive timestamps follow the same convention as the
current pipeline: pandas parses them with utc=True. If that convention is
changed in data preparation, change canonical_utc_index() here at the same time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs.config import (  # noqa: E402
    BASIS_DIR,
    BG_DIR,
    FIG_DIR,
    MODEL_DIR,
    PRED_DIR,
    RESID_DIR,
    STATIONS,
    STATION_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


STATION_NAMES = list(STATIONS.keys())
P2_NAME = "P2"
CLEARSKY_MIN = 10.0
DEFAULT_MASTER_GHI = STATION_DIR / "all_stations_GHI_5min_PST.csv"
DEFAULT_TRAIN_CLEARSKY = BG_DIR / "bg_clearsky_stations.parquet"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a clean DK0 run after validating its data lineage."
    )
    parser.add_argument("--label", default="DK0_clean")
    parser.add_argument(
        "--p2-provenance",
        type=Path,
        required=True,
        help="CSV containing datetime and source columns for P2.",
    )
    parser.add_argument(
        "--master-ghi",
        type=Path,
        default=DEFAULT_MASTER_GHI,
        help="Current four-station GHI CSV used to build csi_stations.parquet.",
    )
    parser.add_argument(
        "--expected-cadence-min",
        type=int,
        default=5,
        help="Expected sampling cadence in minutes (default: 5).",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=2e-5,
        help="Maximum allowed |y.npy - recomputed CSI| (default: 2e-5).",
    )
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    if not clean:
        raise ValueError("--label must contain at least one letter or number")
    return clean


def require_files(paths: Iterable[Path], heading: str) -> None:
    missing = [Path(p) for p in paths if not Path(p).is_file()]
    if missing:
        formatted = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(f"{heading}:\n  {formatted}")


def canonical_utc_index(values) -> pd.DatetimeIndex:
    """Convert timestamps to one comparison convention used by the pipeline."""
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return (result.stdout + result.stderr).strip()
    except (FileNotFoundError, OSError) as exc:
        return f"Unavailable: {exc}"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def infer_cadence(index: pd.DatetimeIndex) -> pd.Timedelta:
    idx = canonical_utc_index(index).sort_values().unique()
    if len(idx) < 2:
        raise ValueError("Cannot infer cadence from fewer than two timestamps")
    deltas = pd.Series(np.diff(idx))
    return pd.Timedelta(deltas.mode().iloc[0])


def compute_metrics(
    y_true,
    y_pred,
    timestamps,
    expected_cadence: pd.Timedelta,
) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    tt = canonical_utc_index(timestamps)

    valid = np.isfinite(yt) & np.isfinite(yp) & ~tt.isna()
    yt, yp, tt = yt[valid], yp[valid], tt[valid]
    if len(yt) < 2:
        return {
            "n": len(yt), "rmse": np.nan, "mae": np.nan,
            "r2": np.nan, "bias": np.nan, "peak_rmse": np.nan,
            "ramp_rmse": np.nan, "ramp_n_pairs": 0,
        }

    err = yp - yt
    sse = float(np.sum(err**2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    peak_threshold = float(np.quantile(yt, 0.90))
    peak = yt >= peak_threshold

    order = np.argsort(tt.asi8)
    yt_o, yp_o, tt_o = yt[order], yp[order], tt[order]
    consecutive = np.diff(tt_o.asi8) == expected_cadence.value
    if consecutive.any():
        ramp_error = np.diff(yp_o)[consecutive] - np.diff(yt_o)[consecutive]
        ramp_rmse = float(np.sqrt(np.mean(ramp_error**2)))
        ramp_n_pairs = int(consecutive.sum())
    else:
        ramp_rmse = np.nan
        ramp_n_pairs = 0

    return {
        "n": int(len(yt)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "bias": float(np.mean(err)),
        "peak_rmse": float(np.sqrt(np.mean(err[peak] ** 2))),
        "ramp_rmse": ramp_rmse,
        "ramp_n_pairs": ramp_n_pairs,
    }


def classify_days(csi: pd.Series) -> pd.Series:
    """Use the existing project thresholds, but on scientific CSI truth."""
    local = csi.copy()
    local.index = canonical_utc_index(local.index).tz_convert(
        "America/Los_Angeles"
    )
    daily_std = local.groupby(local.index.date).std()
    regime = pd.Series(index=daily_std.index, dtype="object")
    regime[daily_std < 0.10] = "clear"
    regime[(daily_std >= 0.10) & (daily_std < 0.20)] = "cloudy_smooth"
    regime[daily_std >= 0.20] = "broken_cloud"
    return regime


def append_metric(
    rows: list[dict],
    pooled: dict[str, list],
    *,
    evaluation: str,
    subset: str,
    target: str,
    fold: int,
    station: str,
    y_true,
    y_pred,
    timestamps,
    expected_cadence: pd.Timedelta,
    include_in_primary_pool: bool = False,
) -> None:
    metrics = compute_metrics(y_true, y_pred, timestamps, expected_cadence)
    metrics.update(
        evaluation=evaluation,
        subset=subset,
        target=target,
        fold=fold,
        test_station=station,
    )
    rows.append(metrics)
    if include_in_primary_pool:
        pooled["true"].append(np.asarray(y_true, dtype=float))
        pooled["pred"].append(np.asarray(y_pred, dtype=float))
        pooled["ts"].append(canonical_utc_index(timestamps))


# ---------------------------------------------------------------------------
# Input loading and lineage verification
# ---------------------------------------------------------------------------
def load_master_ghi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    if "datetime" not in df.columns:
        raise ValueError(f"{path} must contain a datetime column")
    df["datetime"] = canonical_utc_index(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={f"GHI_{s}": s for s in STATION_NAMES})
    missing_columns = [s for s in STATION_NAMES if s not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing station GHI columns: {missing_columns}")
    if df.index.has_duplicates:
        raise ValueError("Master GHI contains duplicate timestamps")
    return df[STATION_NAMES].astype(float)


def load_p2_genuine_timestamps(path: Path) -> pd.DatetimeIndex:
    prov = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {"datetime", "source"}
    if not required.issubset(prov.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    source = prov["source"].astype(str).str.strip().str.lower()
    measured = canonical_utc_index(prov.loc[source == "measured", "datetime"])
    measured = measured.drop_duplicates().sort_values()
    if len(measured) == 0:
        raise ValueError("P2 provenance file contains no source='measured' rows")
    return measured


def normalized_parquet_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = canonical_utc_index(out.index)
    return out.sort_index()


def verify_training_targets(
    master_ghi: pd.DataFrame,
    clear_sky_path: Path,
    tolerance: float,
) -> dict:
    """Verify y.npy was built from the current master GHI, not stale P2 data."""
    required = [
        TRAIN_DIR / "y.npy",
        TRAIN_DIR / "fold_ids.npy",
        TRAIN_DIR / "timestamps.npy",
        clear_sky_path,
    ]
    require_files(required, "Cannot verify training target lineage; files missing")

    y = np.load(TRAIN_DIR / "y.npy", allow_pickle=False)
    fold_ids = np.load(TRAIN_DIR / "fold_ids.npy", allow_pickle=False)
    ts_ns = np.load(TRAIN_DIR / "timestamps.npy", allow_pickle=False)
    if not (len(y) == len(fold_ids) == len(ts_ns)):
        raise ValueError("y.npy, fold_ids.npy, and timestamps.npy lengths differ")

    clear = normalized_parquet_index(pd.read_parquet(clear_sky_path))
    comparisons = []
    for fold, station in enumerate(STATION_NAMES):
        mask = fold_ids == fold
        ts = pd.to_datetime(ts_ns[mask], unit="ns", utc=True)
        if station not in clear.columns:
            raise ValueError(f"{station} missing from {clear_sky_path}")
        ghi = master_ghi[station].reindex(ts).to_numpy(dtype=float)
        cs = clear[station].reindex(ts).to_numpy(dtype=float)
        recomputed = np.zeros_like(ghi, dtype=float)
        day = cs >= CLEARSKY_MIN
        recomputed[day] = ghi[day] / cs[day]
        recomputed = np.clip(recomputed, 0.0, 2.0)
        diff = np.abs(y[mask].astype(float) - recomputed)
        finite = np.isfinite(diff)
        if not finite.all():
            raise ValueError(
                f"Target-lineage comparison for {station} contains missing values"
            )
        max_abs = float(diff.max()) if len(diff) else np.nan
        mean_abs = float(diff.mean()) if len(diff) else np.nan
        comparisons.append(
            {"station": station, "n": int(mask.sum()),
             "max_abs_difference": max_abs,
             "mean_abs_difference": mean_abs}
        )
        if max_abs > tolerance:
            raise RuntimeError(
                f"STALE OR INCONSISTENT TRAINING TARGET DETECTED for {station}: "
                f"max |y.npy - recomputed CSI| = {max_abs:.6g}, allowed "
                f"{tolerance:.6g}. Regenerate residuals.py and "
                "training_matrix.py before training/freezing DK0."
            )
    return {"tolerance": tolerance, "comparisons": comparisons}


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------
def manifest_row(path: Path) -> dict:
    stat = path.stat()
    row = {
        "file": str(path.relative_to(REPO_ROOT)),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False, mmap_mode="r")
        row.update(shape=str(arr.shape), dtype=str(arr.dtype))
    return row


def copy_preserving_relative_path(path: Path, destination_root: Path) -> None:
    destination = destination_root / path.relative_to(REPO_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def collect_source_files() -> list[Path]:
    files = list((REPO_ROOT / "src").rglob("*.py"))
    files += list((REPO_ROOT / "configs").rglob("*.py"))
    for name in ["README.md", "environment.yml", "requirements.txt", "pyproject.toml"]:
        candidate = REPO_ROOT / name
        if candidate.is_file():
            files.append(candidate)
    return sorted(set(files))


def write_git_and_environment_metadata(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "git_head.txt").write_text(
        run_text(["git", "rev-parse", "HEAD"]) + "\n", encoding="utf-8"
    )
    (destination / "git_status.txt").write_text(
        run_text(["git", "status", "--short"]) + "\n", encoding="utf-8"
    )
    (destination / "git_diff.patch").write_text(
        run_text(["git", "diff"]) + "\n", encoding="utf-8"
    )
    (destination / "git_diff_cached.patch").write_text(
        run_text(["git", "diff", "--cached"]) + "\n", encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (destination / "runtime.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    label = sanitize_label(args.label)
    master_path = args.master_ghi.resolve()
    provenance_path = args.p2_provenance.resolve()
    expected_cadence = pd.Timedelta(minutes=args.expected_cadence_min)

    required_outputs = []
    for fold, station in enumerate(STATION_NAMES):
        required_outputs += [
            VAL_DIR / f"fold_{fold}_{station}_predictions.csv",
            VAL_DIR / f"fold_{fold}_history.csv",
            MODEL_DIR / f"fold_{fold}_best.pt",
            MODEL_DIR / f"fold_{fold}_scaler_mean.npy",
            MODEL_DIR / f"fold_{fold}_scaler_std.npy",
        ]
    required_outputs += [
        VAL_DIR / "loso_summary.txt",
        VAL_DIR / "loso_results.csv",
        PRED_DIR / "ghi_pvs.parquet",
        PRED_DIR / "csi_pvs.parquet",
        master_path,
        provenance_path,
    ]
    require_files(required_outputs, "Required clean-run outputs are missing")

    master_ghi = load_master_ghi(master_path)
    observed_cadence = infer_cadence(master_ghi.index)
    if observed_cadence != expected_cadence:
        raise RuntimeError(
            f"Cadence mismatch: observed {observed_cadence}, expected "
            f"{expected_cadence}. Correct the data or pass the correct "
            "--expected-cadence-min value."
        )

    p2_genuine = load_p2_genuine_timestamps(provenance_path)
    p2_overlap = p2_genuine.intersection(master_ghi.index)
    overlap_fraction = len(p2_overlap) / len(p2_genuine)
    if overlap_fraction < 0.95:
        raise RuntimeError(
            f"Only {overlap_fraction:.1%} of genuine P2 timestamps overlap "
            "the master GHI file. Fix timezone/provenance alignment first."
        )

    lineage = verify_training_targets(
        master_ghi, DEFAULT_TRAIN_CLEARSKY, args.target_tolerance
    )

    metric_rows: list[dict] = []
    primary_pool = {"true": [], "pred": [], "ts": []}

    for fold, station in enumerate(STATION_NAMES):
        pred_path = VAL_DIR / f"fold_{fold}_{station}_predictions.csv"
        pred = pd.read_csv(pred_path)
        if "datetime_local" not in pred.columns:
            raise ValueError(f"{pred_path} lacks datetime_local")
        pred.index = canonical_utc_index(pred["datetime_local"])
        pred = pred.sort_index()
        required_cols = {
            "ghi_true", "ghi_pred", "csi_true", "csi_pred", "bg_clearsky"
        }
        missing_cols = sorted(required_cols - set(pred.columns))
        if missing_cols:
            raise ValueError(f"{pred_path} missing columns {missing_cols}")

        day = pred["bg_clearsky"].to_numpy(dtype=float) >= CLEARSKY_MIN
        pred_day = pred.loc[day].copy()

        # Historical/legacy metrics: exactly the truth train.py saved.
        append_metric(
            metric_rows, primary_pool,
            evaluation="legacy_saved", subset="overall", target="GHI",
            fold=fold, station=station,
            y_true=pred_day["ghi_true"], y_pred=pred_day["ghi_pred"],
            timestamps=pred_day.index, expected_cadence=expected_cadence,
        )
        append_metric(
            metric_rows, primary_pool,
            evaluation="legacy_saved", subset="overall", target="CSI",
            fold=fold, station=station,
            y_true=pred_day["csi_true"], y_pred=pred_day["csi_pred"],
            timestamps=pred_day.index, expected_cadence=expected_cadence,
        )

        # Scientific truth: use the current station GHI directly.
        ghi_true = master_ghi[station].reindex(pred_day.index)
        cs = pred_day["bg_clearsky"].astype(float)
        csi_true = ghi_true / cs
        scientific_valid = ghi_true.notna() & np.isfinite(csi_true)

        if station == P2_NAME:
            genuine_mask = pred_day.index.isin(p2_genuine)
            primary_valid = scientific_valid.to_numpy() & genuine_mask
            if primary_valid.sum() < 2:
                raise RuntimeError("Fewer than two genuine daytime P2 rows aligned")

            # Diagnostic only: reconstructed P2 values are not ground truth.
            append_metric(
                metric_rows, primary_pool,
                evaluation="scientific_reference", subset="P2_all_diagnostic",
                target="GHI", fold=fold, station=station,
                y_true=ghi_true[scientific_valid],
                y_pred=pred_day.loc[scientific_valid, "ghi_pred"],
                timestamps=pred_day.index[scientific_valid],
                expected_cadence=expected_cadence,
            )
        else:
            primary_valid = scientific_valid.to_numpy()

        # Primary scientific fold: genuine station observations only.
        primary_index = pred_day.index[primary_valid]
        primary_ghi_true = ghi_true.iloc[np.flatnonzero(primary_valid)]
        primary_ghi_pred = pred_day["ghi_pred"].iloc[np.flatnonzero(primary_valid)]
        primary_csi_true = csi_true.iloc[np.flatnonzero(primary_valid)]
        primary_csi_pred = pred_day["csi_pred"].iloc[np.flatnonzero(primary_valid)]

        append_metric(
            metric_rows, primary_pool,
            evaluation="scientific_primary", subset="overall", target="GHI",
            fold=fold, station=station,
            y_true=primary_ghi_true, y_pred=primary_ghi_pred,
            timestamps=primary_index, expected_cadence=expected_cadence,
            include_in_primary_pool=True,
        )
        append_metric(
            metric_rows, {"true": [], "pred": [], "ts": []},
            evaluation="scientific_primary", subset="overall", target="CSI",
            fold=fold, station=station,
            y_true=primary_csi_true, y_pred=primary_csi_pred,
            timestamps=primary_index, expected_cadence=expected_cadence,
        )

        # Regime-stratified scientific GHI metrics.
        csi_series = pd.Series(primary_csi_true.to_numpy(), index=primary_index)
        regimes = classify_days(csi_series)
        local_dates = primary_index.tz_convert("America/Los_Angeles").date
        row_regimes = pd.Series(local_dates).map(regimes).to_numpy()
        for regime_name in ["clear", "cloudy_smooth", "broken_cloud"]:
            regime_mask = row_regimes == regime_name
            if regime_mask.sum() < 2:
                continue
            append_metric(
                metric_rows, {"true": [], "pred": [], "ts": []},
                evaluation="scientific_primary", subset=regime_name,
                target="GHI", fold=fold, station=station,
                y_true=primary_ghi_true.to_numpy()[regime_mask],
                y_pred=primary_ghi_pred.to_numpy()[regime_mask],
                timestamps=primary_index[regime_mask],
                expected_cadence=expected_cadence,
            )

    metrics = pd.DataFrame(metric_rows)
    scientific = metrics[
        (metrics["evaluation"] == "scientific_primary")
        & (metrics["subset"] == "overall")
        & (metrics["target"] == "GHI")
    ].copy()
    if len(scientific) != len(STATION_NAMES):
        raise RuntimeError("Primary scientific summary does not contain four folds")

    macro_fields = ["rmse", "mae", "r2", "bias", "peak_rmse", "ramp_rmse"]
    macro = {field: float(scientific[field].mean()) for field in macro_fields}
    pooled_true = np.concatenate(primary_pool["true"])
    pooled_pred = np.concatenate(primary_pool["pred"])
    pooled_ts = pd.DatetimeIndex(
        np.concatenate([idx.asi8 for idx in primary_pool["ts"]]),
        tz="UTC",
    )
    pooled = compute_metrics(
        pooled_true, pooled_pred, pooled_ts, expected_cadence
    )
    # Do not calculate a pooled ramp by sorting observations from different
    # stations together: equal timestamps from four folds would interleave and
    # create cross-station differences. Pool the within-fold ramp squared
    # errors using their valid consecutive-pair counts instead.
    ramp_valid = scientific[
        scientific["ramp_rmse"].notna() & (scientific["ramp_n_pairs"] > 0)
    ]
    pooled_ramp_pairs = int(ramp_valid["ramp_n_pairs"].sum())
    if pooled_ramp_pairs:
        pooled["ramp_rmse"] = float(
            np.sqrt(
                np.sum(
                    ramp_valid["ramp_rmse"] ** 2
                    * ramp_valid["ramp_n_pairs"]
                )
                / pooled_ramp_pairs
            )
        )
        pooled["ramp_n_pairs"] = pooled_ramp_pairs
    else:
        pooled["ramp_rmse"] = np.nan
        pooled["ramp_n_pairs"] = 0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_archive = REPO_ROOT / "outputs" / "baseline_frozen" / f"{label}_{timestamp}"
    if final_archive.exists():
        raise FileExistsError(final_archive)
    work_archive = final_archive.with_name(final_archive.name + ".incomplete")
    if work_archive.exists():
        raise FileExistsError(work_archive)
    work_archive.mkdir(parents=True)

    try:
        metrics.to_csv(work_archive / "metrics_all.csv", index=False)
        scientific.to_csv(work_archive / "metrics_scientific_primary_folds.csv", index=False)
        (work_archive / "summary_scientific_primary.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "cadence": str(expected_cadence),
                    "p2_genuine_timestamps_total": len(p2_genuine),
                    "p2_genuine_overlap_fraction": overlap_fraction,
                    "macro_equal_station_weight": macro,
                    "pooled_equal_observation_weight": pooled,
                    "target_lineage_check": lineage,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

        # Hash every important input used by DK0.
        manifest_files = [
            master_path,
            provenance_path,
            DEFAULT_TRAIN_CLEARSKY,
            RESID_DIR / "csi_stations.parquet",
            TRAIN_DIR / "X.npy",
            TRAIN_DIR / "y.npy",
            TRAIN_DIR / "fold_ids.npy",
            TRAIN_DIR / "timestamps.npy",
            TRAIN_DIR / "feature_names.txt",
            BASIS_DIR / "Phi_stations_scaled.npy",
            BASIS_DIR / "Phi_pvs_scaled.npy",
            BASIS_DIR / "active_mask.npy",
        ]
        require_files(manifest_files, "Core baseline inputs missing")
        pd.DataFrame([manifest_row(p) for p in manifest_files]).to_csv(
            work_archive / "input_manifest.csv", index=False
        )

        output_files = list(required_outputs)
        output_files += list(PRED_DIR.glob("*.parquet"))
        output_files += list(FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
        output_root = work_archive / "run_outputs"
        for path in sorted(set(output_files)):
            copy_preserving_relative_path(path.resolve(), output_root)

        source_root = work_archive / "source_snapshot"
        for path in collect_source_files():
            copy_preserving_relative_path(path.resolve(), source_root)

        write_git_and_environment_metadata(work_archive / "metadata")
        (work_archive / "README.txt").write_text(
            f"{label} frozen at {timestamp}\n"
            "\n"
            "This is the clean direct-CSI DeepKriging comparator for DK1.\n"
            "The primary scientific evaluation uses original station GHI for\n"
            "S1/S2/S3 and genuine-measured P2 timestamps only. Reconstructed\n"
            "P2 all-timestamp metrics are diagnostic and are not ground-truth\n"
            "validation. Ramp metrics use only consecutive timestamps.\n"
            "\n"
            "No model or data files in the repository were modified by this\n"
            "freeze script.\n",
            encoding="utf-8",
        )

        work_archive.rename(final_archive)
    except Exception:
        print(f"Freeze failed; partial archive retained at {work_archive}")
        raise

    print("\nDK0 CLEAN BASELINE FROZEN SUCCESSFULLY")
    print(f"Archive: {final_archive}")
    print("\nPrimary scientific fold metrics:")
    print(
        scientific[
            ["test_station", "n", "rmse", "mae", "r2", "bias",
             "peak_rmse", "ramp_rmse"]
        ].round(4).to_string(index=False)
    )
    print("\nMacro GHI RMSE:", round(macro["rmse"], 4))
    print("Pooled GHI RMSE:", round(pooled["rmse"], 4))
    print("\nReview `git status` and commit/tag the code separately if desired.")


if __name__ == "__main__":
    main()
