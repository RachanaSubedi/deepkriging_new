"""
Create one deterministic, seasonally distributed blocked-day validation split.

Place this file at:
    src/train/shared_blocked_day_split.py

Why this exists
---------------
The original DK-0 training code uses the last 20% of each station as its
validation period. Absolute temporal basis functions added in DK-1 would then
have little or no training support near late-year temporal knots. This utility
instead selects COMPLETE LOCAL CALENDAR DAYS throughout the year and applies
the same validation dates to every station.

No five-minute rows from a selected validation day are allowed into training.
Days are selected independently of CSI/GHI values, so the split cannot favor
particular irradiance conditions.

Run once from the repository root after training_matrix.py:

    python src/train/shared_blocked_day_split.py --validation-fraction 0.20

Outputs in data/processed/training_matrix/shared_blocked_day_split/:
    shared_train_mask.npy
    shared_val_mask.npy
    shared_validation_days.csv
    shared_split_summary.json

The masks have the same length and row order as X.npy/y.npy/fold_ids.npy.
Both the matched spatial-only baseline and DK-1 must reuse these exact files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from configs.config import LOCAL_TZ, STATIONS, TRAIN_DIR  # noqa: E402


DEFAULT_SPLIT_DIR = TRAIN_DIR / "shared_blocked_day_split"


def sha256_array(arr: np.ndarray) -> str:
    """Hash array dtype, shape, and bytes so stale masks are rejected."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def timestamps_to_local_days(
    timestamps_ns: np.ndarray,
    local_timezone: str = LOCAL_TZ,
) -> pd.DatetimeIndex:
    """
    Convert saved nanosecond timestamps to normalized local calendar days.

    This matches the current train.py convention:

        pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert(LOCAL_TZ)

    If the timestamp serialization convention changes in training_matrix.py,
    update this function and train.py together.
    """
    ts_local = pd.to_datetime(timestamps_ns, unit="ns", utc=True).tz_convert(
        local_timezone
    )
    return ts_local.normalize()


def common_station_days(
    local_days: pd.DatetimeIndex,
    fold_ids: np.ndarray,
    station_names: Sequence[str],
) -> pd.DatetimeIndex:
    """Return calendar days having at least one usable row at every station."""
    day_sets: list[set[pd.Timestamp]] = []
    for station_idx, _ in enumerate(station_names):
        station_days = set(pd.DatetimeIndex(local_days[fold_ids == station_idx]))
        if not station_days:
            raise ValueError(f"Station index {station_idx} has no timestamps")
        day_sets.append(station_days)

    shared = set.intersection(*day_sets)
    if not shared:
        raise ValueError("No calendar day is represented at every station")

    return pd.DatetimeIndex(sorted(shared))


def _monthly_quotas(
    eligible_days: pd.DatetimeIndex,
    validation_fraction: float,
) -> dict[pd.Period, int]:
    """Allocate the exact target number of validation days across months."""
    months = eligible_days.tz_localize(None).to_period("M")
    counts = pd.Series(1, index=months).groupby(level=0).sum().sort_index()

    exact = counts.astype(float) * validation_fraction
    quotas = np.floor(exact).astype(int)
    target = int(round(len(eligible_days) * validation_fraction))

    # Give remaining days to months with the largest fractional remainders.
    remainder = target - int(quotas.sum())
    if remainder > 0:
        order = (exact - quotas).sort_values(ascending=False, kind="mergesort")
        for month in order.index[:remainder]:
            quotas.loc[month] += 1

    # Defensive bounds for very small or unusual datasets.
    quotas = np.minimum(quotas, counts)
    return {month: int(value) for month, value in quotas.items()}


def _evenly_spaced_indices(n_available: int, n_select: int) -> np.ndarray:
    """Choose deterministic, approximately evenly spaced positions."""
    if n_select < 0 or n_select > n_available:
        raise ValueError("n_select must satisfy 0 <= n_select <= n_available")
    if n_select == 0:
        return np.empty(0, dtype=int)
    if n_select == n_available:
        return np.arange(n_available, dtype=int)

    # Midpoints of n_select equal-width bins over the available positions.
    positions = (np.arange(n_select) + 0.5) * n_available / n_select - 0.5
    indices = np.rint(positions).astype(int)

    # Rounding should normally be unique; this guarantees it defensively.
    indices = np.unique(np.clip(indices, 0, n_available - 1))
    if len(indices) != n_select:
        remaining = np.setdiff1d(np.arange(n_available), indices)
        indices = np.sort(
            np.concatenate([indices, remaining[: n_select - len(indices)]])
        )
    return indices


def select_validation_days(
    eligible_days: pd.DatetimeIndex,
    validation_fraction: float = 0.20,
) -> pd.DatetimeIndex:
    """
    Select complete days, stratified by month and spread within each month.

    Selection depends only on dates, never on target values or weather regime.
    """
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if len(eligible_days) < 10:
        raise ValueError("At least 10 shared calendar days are required")

    quotas = _monthly_quotas(eligible_days, validation_fraction)
    month_periods = eligible_days.tz_localize(None).to_period("M")
    selected: list[pd.Timestamp] = []

    for month, quota in quotas.items():
        month_days = eligible_days[month_periods == month].sort_values()
        indices = _evenly_spaced_indices(len(month_days), quota)
        selected.extend(month_days[indices].tolist())

    selected_days = pd.DatetimeIndex(sorted(selected))
    target = int(round(len(eligible_days) * validation_fraction))
    if len(selected_days) != target:
        raise RuntimeError(
            f"Selected {len(selected_days)} validation days; expected {target}"
        )
    return selected_days


def build_shared_masks(
    timestamps_ns: np.ndarray,
    fold_ids: np.ndarray,
    station_names: Sequence[str],
    validation_fraction: float = 0.20,
    local_timezone: str = LOCAL_TZ,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Build row-level train/validation masks from shared calendar days."""
    timestamps_ns = np.asarray(timestamps_ns)
    fold_ids = np.asarray(fold_ids)

    if timestamps_ns.ndim != 1 or fold_ids.ndim != 1:
        raise ValueError("timestamps_ns and fold_ids must be one-dimensional")
    if len(timestamps_ns) != len(fold_ids):
        raise ValueError("timestamps_ns and fold_ids have different lengths")
    if len(timestamps_ns) == 0:
        raise ValueError("Training matrix contains no rows")

    observed_fold_ids = set(np.unique(fold_ids).tolist())
    expected_fold_ids = set(range(len(station_names)))
    if observed_fold_ids != expected_fold_ids:
        raise ValueError(
            f"fold_ids are {sorted(observed_fold_ids)}; expected "
            f"{sorted(expected_fold_ids)}"
        )

    local_days = timestamps_to_local_days(timestamps_ns, local_timezone)
    eligible_days = common_station_days(local_days, fold_ids, station_names)
    validation_days = select_validation_days(
        eligible_days, validation_fraction=validation_fraction
    )

    val_mask = np.asarray(local_days.isin(validation_days), dtype=bool)
    train_mask = ~val_mask

    if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
        raise RuntimeError("Train/validation masks are not a valid partition")

    # Every station must contain rows in both subsets.
    for station_idx, station in enumerate(station_names):
        station_mask = fold_ids == station_idx
        if not np.any(train_mask & station_mask):
            raise RuntimeError(f"{station} has no training rows")
        if not np.any(val_mask & station_mask):
            raise RuntimeError(f"{station} has no validation rows")

    return train_mask, val_mask, validation_days, eligible_days


def save_shared_split(
    output_dir: Path,
    timestamps_ns: np.ndarray,
    fold_ids: np.ndarray,
    station_names: Sequence[str],
    validation_fraction: float,
    local_timezone: str,
    overwrite: bool = False,
) -> None:
    """Create and save an immutable shared split plus integrity metadata."""
    output_dir = Path(output_dir)
    expected_outputs = [
        output_dir / "shared_train_mask.npy",
        output_dir / "shared_val_mask.npy",
        output_dir / "shared_validation_days.csv",
        output_dir / "shared_split_summary.json",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Split files already exist ({names}). Reuse them, or pass "
            "--overwrite only if you intentionally want a new split."
        )

    train_mask, val_mask, val_days, eligible_days = build_shared_masks(
        timestamps_ns=timestamps_ns,
        fold_ids=fold_ids,
        station_names=station_names,
        validation_fraction=validation_fraction,
        local_timezone=local_timezone,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "shared_train_mask.npy", train_mask)
    np.save(output_dir / "shared_val_mask.npy", val_mask)

    val_days_naive = val_days.tz_localize(None)
    day_table = pd.DataFrame(
        {
            "validation_date_local": val_days_naive.strftime("%Y-%m-%d"),
            "month": val_days_naive.strftime("%Y-%m"),
        }
    )
    day_table.to_csv(output_dir / "shared_validation_days.csv", index=False)

    station_rows = {}
    for station_idx, station in enumerate(station_names):
        station_mask = fold_ids == station_idx
        n_total = int(station_mask.sum())
        n_train = int((station_mask & train_mask).sum())
        n_val = int((station_mask & val_mask).sum())
        station_rows[station] = {
            "total_rows": n_total,
            "train_rows": n_train,
            "validation_rows": n_val,
            "validation_fraction_rows": n_val / n_total,
        }

    summary = {
        "method": "shared_complete_local_days_stratified_by_month",
        "selection_uses_target_values": False,
        "local_timezone": local_timezone,
        "requested_validation_fraction_days": validation_fraction,
        "n_rows": int(len(timestamps_ns)),
        "n_eligible_shared_days": int(len(eligible_days)),
        "n_validation_days": int(len(val_days)),
        "actual_validation_fraction_days": float(len(val_days) / len(eligible_days)),
        "validation_dates_local": day_table["validation_date_local"].tolist(),
        "station_rows": station_rows,
        "source_integrity": {
            "timestamps_sha256": sha256_array(timestamps_ns),
            "fold_ids_sha256": sha256_array(fold_ids),
        },
    }
    (output_dir / "shared_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("SHARED BLOCKED-DAY SPLIT CREATED")
    print("=" * 72)
    print(f"Output directory : {output_dir}")
    print(f"Eligible days    : {len(eligible_days)}")
    print(f"Validation days  : {len(val_days)} ({len(val_days)/len(eligible_days):.1%})")
    print(f"Training rows    : {int(train_mask.sum()):,}")
    print(f"Validation rows  : {int(val_mask.sum()):,}")
    print("\nPer-station rows:")
    for station, values in station_rows.items():
        print(
            f"  {station}: train={values['train_rows']:,}  "
            f"val={values['validation_rows']:,}  "
            f"val_fraction={values['validation_fraction_rows']:.1%}"
        )
    print("\nReuse these exact masks for DK0_matched and DK1.")


def load_shared_split(
    timestamps_ns: np.ndarray,
    fold_ids: np.ndarray,
    split_dir: Path = DEFAULT_SPLIT_DIR,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load saved masks and verify that they match the current training matrix.

    Import and call this function from train.py. It deliberately fails if the
    matrix has been rebuilt in a different row order.
    """
    split_dir = Path(split_dir)
    train_mask = np.load(split_dir / "shared_train_mask.npy")
    val_mask = np.load(split_dir / "shared_val_mask.npy")
    summary = json.loads(
        (split_dir / "shared_split_summary.json").read_text(encoding="utf-8")
    )

    if len(train_mask) != len(timestamps_ns) or len(val_mask) != len(timestamps_ns):
        raise ValueError("Saved split length does not match the training matrix")
    if train_mask.dtype != np.bool_ or val_mask.dtype != np.bool_:
        raise TypeError("Saved train/validation masks must be boolean")
    if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
        raise ValueError("Saved masks do not form a complete, disjoint partition")

    expected = summary["source_integrity"]
    if sha256_array(timestamps_ns) != expected["timestamps_sha256"]:
        raise ValueError(
            "timestamps.npy changed after the shared split was created. "
            "Regenerate the split deliberately."
        )
    if sha256_array(fold_ids) != expected["fold_ids_sha256"]:
        raise ValueError(
            "fold_ids.npy changed after the shared split was created. "
            "Regenerate the split deliberately."
        )
    return train_mask, val_mask, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a shared, seasonally distributed blocked-day split."
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of eligible complete calendar days used for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory in which the immutable split files are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace an existing saved split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamps_path = TRAIN_DIR / "timestamps.npy"
    fold_ids_path = TRAIN_DIR / "fold_ids.npy"
    if not timestamps_path.exists() or not fold_ids_path.exists():
        raise FileNotFoundError(
            "Run src/data_prep/training_matrix.py before creating the split."
        )

    timestamps_ns = np.load(timestamps_path)
    fold_ids = np.load(fold_ids_path)
    save_shared_split(
        output_dir=args.output_dir,
        timestamps_ns=timestamps_ns,
        fold_ids=fold_ids,
        station_names=list(STATIONS.keys()),
        validation_fraction=args.validation_fraction,
        local_timezone=LOCAL_TZ,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
