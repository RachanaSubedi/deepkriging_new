"""Build the DK-1 multiresolution temporal Gaussian basis.

DK-1 changes exactly one component relative to DK0_matched_groundtruth:
it appends explicit temporal basis functions to the existing spatial Wendland
basis and 15 covariates.  The response remains direct CSI, and the loss,
network, blocked-day split, LOSO folds, and genuine-only P2 policy remain
unchanged.

Temporal basis definition (following the additive space/time embedding used
in spatiotemporal DeepKriging):

    psi_j(t) = exp(-0.5 * ((t - v_j) / kappa_j)**2)

where v_j is a temporal knot and kappa_j equals the knot spacing.  We use
30-day, 7-day, and 1-day resolutions.  Spatial and temporal bases are
concatenated, not tensor-multiplied; the DNN learns their interactions.

Run from the repository root:

    python src/data_prep/temporal_basis.py

Add --overwrite only when intentionally rebuilding the same artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import PROCESSED_DIR, TRAIN_DIR  # noqa: E402


DAY_NS = 86_400_000_000_000
DEFAULT_LEVELS = (
    {"name": "coarse_30d", "spacing_days": 30.0},
    {"name": "medium_7d", "spacing_days": 7.0},
    {"name": "fine_1d", "spacing_days": 1.0},
)
DEFAULT_OUTPUT_DIR = Path(PROCESSED_DIR) / "temporal_basis"
DEFAULT_SPLIT_DIR = Path(TRAIN_DIR) / "shared_blocked_day_split"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamps_to_ns(values: np.ndarray) -> np.ndarray:
    """Return one-dimensional int64 nanosecond timestamps."""
    values = np.asarray(values).reshape(-1)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return parsed.astype("int64").to_numpy()


def _load_first_existing(directory: Path, names: Iterable[str]) -> np.ndarray:
    for name in names:
        path = directory / name
        if path.exists():
            return np.load(path, allow_pickle=False).astype(bool)
    expected = ", ".join(names)
    raise FileNotFoundError(f"None of these split masks exists in {directory}: {expected}")


def load_shared_masks(split_dir: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Load masks produced by shared_blocked_day_split.py."""
    train_mask = _load_first_existing(
        split_dir, ("train_mask.npy", "shared_train_mask.npy", "row_train_mask.npy")
    )
    val_mask = _load_first_existing(
        split_dir,
        ("val_mask.npy", "validation_mask.npy", "shared_val_mask.npy", "row_val_mask.npy"),
    )
    if train_mask.size != expected_rows or val_mask.size != expected_rows:
        raise ValueError(
            "Shared split masks do not match timestamps.npy: "
            f"timestamps={expected_rows:,}, train={train_mask.size:,}, val={val_mask.size:,}"
        )
    if np.any(train_mask & val_mask):
        raise ValueError("Shared train and validation masks overlap.")
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("Shared train and validation masks must both contain rows.")
    return train_mask, val_mask


def build_temporal_basis_spec(
    timestamps_ns: np.ndarray,
    levels: Iterable[dict[str, Any]] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Create a reproducible knot specification covering the timestamp domain."""
    ts_ns = _timestamps_to_ns(timestamps_ns)
    if ts_ns.size == 0:
        raise ValueError("Cannot build temporal bases from an empty timestamp array.")

    origin_ns = int(pd.Timestamp(int(ts_ns.min()), unit="ns", tz="UTC").normalize().value)
    t_days = (ts_ns.astype(np.float64) - origin_ns) / DAY_NS
    t_min = float(t_days.min())
    t_max = float(t_days.max())

    level_specs: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for level in levels:
        name = str(level["name"])
        spacing = float(level["spacing_days"])
        if spacing <= 0:
            raise ValueError(f"Temporal spacing must be positive; got {spacing} for {name}.")

        # One knot of padding on each side prevents weak support at year edges.
        first = np.floor(t_min / spacing) * spacing - spacing
        last = np.ceil(t_max / spacing) * spacing + spacing
        anchors = np.arange(first, last + 0.5 * spacing, spacing, dtype=np.float64)
        names = [f"psi_{name}_{i:04d}" for i in range(anchors.size)]
        feature_names.extend(names)
        level_specs.append(
            {
                "name": name,
                "spacing_days": spacing,
                "kappa_days": spacing,
                "anchor_days": anchors.tolist(),
                "n_basis": int(anchors.size),
                "feature_names": names,
            }
        )

    return {
        "version": "DK1_temporal_gaussian_v1",
        "formula": "exp(-0.5 * ((t_days - anchor_days) / kappa_days)^2)",
        "combination": "concatenate spatial basis, temporal basis, and covariates",
        "timezone_for_elapsed_time": "UTC",
        "origin_utc": pd.Timestamp(origin_ns, unit="ns", tz="UTC").isoformat(),
        "origin_ns": origin_ns,
        "domain_min_ns": int(ts_ns.min()),
        "domain_max_ns": int(ts_ns.max()),
        "levels": level_specs,
        "n_basis": len(feature_names),
        "feature_names": feature_names,
        "standardize": False,
    }


def transform_temporal_basis(
    timestamps_ns: np.ndarray,
    spec: dict[str, Any],
    chunk_size: int = 4096,
) -> np.ndarray:
    """Evaluate temporal bases, returning float32 values in [0, 1]."""
    ts_ns = _timestamps_to_ns(timestamps_ns)
    t_days = (ts_ns.astype(np.float64) - int(spec["origin_ns"])) / DAY_NS
    output = np.empty((ts_ns.size, int(spec["n_basis"])), dtype=np.float32)

    column_start = 0
    for level in spec["levels"]:
        anchors = np.asarray(level["anchor_days"], dtype=np.float64)
        kappa = float(level["kappa_days"])
        column_end = column_start + anchors.size
        for row_start in range(0, ts_ns.size, chunk_size):
            row_end = min(row_start + chunk_size, ts_ns.size)
            z = (t_days[row_start:row_end, None] - anchors[None, :]) / kappa
            output[row_start:row_end, column_start:column_end] = np.exp(-0.5 * z * z)
        column_start = column_end

    if not np.isfinite(output).all():
        raise ValueError("Temporal basis contains non-finite values.")
    if output.min() < 0.0 or output.max() > 1.000001:
        raise ValueError("Temporal Gaussian basis is outside its expected [0, 1] range.")
    return output


def lookup_temporal_basis(
    query_timestamps_ns: np.ndarray,
    unique_timestamps_ns: np.ndarray,
    temporal_basis_unique: np.ndarray,
) -> np.ndarray:
    """Map repeated training timestamps to the stored unique-time basis rows."""
    query = _timestamps_to_ns(query_timestamps_ns)
    unique = _timestamps_to_ns(unique_timestamps_ns)
    positions = np.searchsorted(unique, query)
    valid = positions < unique.size
    valid[valid] &= unique[positions[valid]] == query[valid]
    if not np.all(valid):
        example = pd.Timestamp(int(query[np.flatnonzero(~valid)[0]]), unit="ns", tz="UTC")
        raise KeyError(f"Timestamp {example} is absent from the stored temporal-basis grid.")
    return temporal_basis_unique[positions]


def _support_table(
    unique_ns: np.ndarray,
    psi_unique: np.ndarray,
    spec: dict[str, Any],
    train_unique_ns: np.ndarray,
    val_unique_ns: np.ndarray,
) -> pd.DataFrame:
    train_psi = lookup_temporal_basis(train_unique_ns, unique_ns, psi_unique)
    val_psi = lookup_temporal_basis(val_unique_ns, unique_ns, psi_unique)
    max_train = train_psi.max(axis=0)
    max_val = val_psi.max(axis=0)
    mean_train = train_psi.mean(axis=0)
    mean_val = val_psi.mean(axis=0)

    rows: list[dict[str, Any]] = []
    col = 0
    for level in spec["levels"]:
        for feature, anchor in zip(level["feature_names"], level["anchor_days"]):
            # A validation-active basis must have meaningful training activation.
            active_in_val = bool(max_val[col] >= 0.10)
            support_ok = bool((not active_in_val) or (max_train[col] >= 0.10))
            rows.append(
                {
                    "column": col,
                    "feature": feature,
                    "level": level["name"],
                    "spacing_days": level["spacing_days"],
                    "anchor_day": anchor,
                    "max_train": float(max_train[col]),
                    "max_validation": float(max_val[col]),
                    "mean_train": float(mean_train[col]),
                    "mean_validation": float(mean_val[col]),
                    "active_in_validation": active_in_val,
                    "support_ok": support_ok,
                }
            )
            col += 1
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamps", type=Path, default=Path(TRAIN_DIR) / "timestamps.npy")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if not args.timestamps.exists():
        raise FileNotFoundError(f"Missing {args.timestamps}; build the training matrix first.")
    if not args.split_dir.exists():
        raise FileNotFoundError(
            f"Missing {args.split_dir}; run src/train/shared_blocked_day_split.py first."
        )

    planned = (
        "temporal_basis_unique.npy",
        "temporal_basis_unique_timestamps.npy",
        "temporal_basis_spec.json",
        "temporal_basis_feature_names.txt",
        "temporal_basis_support.csv",
        "temporal_basis_summary.json",
    )
    existing = [output_dir / name for name in planned if (output_dir / name).exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"DK-1 temporal-basis outputs already exist ({names}). "
            "Use --overwrite only if rebuilding intentionally."
        )

    all_ts_ns = _timestamps_to_ns(np.load(args.timestamps, allow_pickle=False))
    train_mask, val_mask = load_shared_masks(args.split_dir, all_ts_ns.size)
    unique_ns = np.unique(all_ts_ns)
    train_unique_ns = np.unique(all_ts_ns[train_mask])
    val_unique_ns = np.unique(all_ts_ns[val_mask])

    spec = build_temporal_basis_spec(unique_ns)
    psi_unique = transform_temporal_basis(unique_ns, spec, args.chunk_size)
    support = _support_table(
        unique_ns, psi_unique, spec, train_unique_ns, val_unique_ns
    )
    failed = support.loc[~support["support_ok"]]
    if not failed.empty:
        examples = ", ".join(failed["feature"].head(10))
        raise RuntimeError(
            "Temporal bases active in validation lack training support. "
            f"First failures: {examples}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "temporal_basis_unique.npy", psi_unique)
    np.save(output_dir / "temporal_basis_unique_timestamps.npy", unique_ns)
    with (output_dir / "temporal_basis_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    (output_dir / "temporal_basis_feature_names.txt").write_text(
        "\n".join(spec["feature_names"]) + "\n", encoding="utf-8"
    )
    support.to_csv(output_dir / "temporal_basis_support.csv", index=False)

    summary = {
        "experiment": "DK1_direct_CSI_spatial_plus_temporal_basis",
        "timestamps_rows": int(all_ts_ns.size),
        "unique_timestamps": int(unique_ns.size),
        "train_unique_timestamps": int(train_unique_ns.size),
        "validation_unique_timestamps": int(val_unique_ns.size),
        "temporal_basis_columns": int(spec["n_basis"]),
        "levels": {level["name"]: level["n_basis"] for level in spec["levels"]},
        "matrix_shape": list(psi_unique.shape),
        "matrix_size_mib": float(psi_unique.nbytes / 1024**2),
        "validation_support_passed": True,
        "minimum_train_max_for_validation_active_basis": float(
            support.loc[support["active_in_validation"], "max_train"].min()
        ),
        "source_files": {
            str(args.timestamps): _sha256(args.timestamps),
        },
    }
    for mask_name in (
        "train_mask.npy",
        "shared_train_mask.npy",
        "row_train_mask.npy",
        "val_mask.npy",
        "validation_mask.npy",
        "shared_val_mask.npy",
        "row_val_mask.npy",
    ):
        path = args.split_dir / mask_name
        if path.exists():
            summary["source_files"][str(path)] = _sha256(path)
    with (output_dir / "temporal_basis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 72)
    print("DK-1 TEMPORAL BASIS CREATED")
    print("=" * 72)
    print(f"Output directory       : {output_dir}")
    print(f"Training-matrix rows   : {all_ts_ns.size:,}")
    print(f"Unique timestamps      : {unique_ns.size:,}")
    print(f"Temporal basis columns : {spec['n_basis']:,}")
    for level in spec["levels"]:
        print(
            f"  {level['name']:<12}: {level['n_basis']:>4} bases "
            f"(spacing=kappa={level['spacing_days']:g} days)"
        )
    print(f"Stored matrix          : {psi_unique.shape} float32 ({psi_unique.nbytes / 1024**2:.1f} MiB)")
    print("Shared split reused    : yes")
    print("Validation support     : PASSED")
    print("Standardize Psi        : no (bounded Gaussian basis in [0, 1])")
    print("\nNo model was trained. The next step is train_dk1.py.")


if __name__ == "__main__":
    main()
