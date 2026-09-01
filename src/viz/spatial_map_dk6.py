"""Plot DK-6 predictions for any requested day.

Place at:
    src/viz/spatial_map_dk6.py

Examples:
    python src/viz/spatial_map_dk6.py --date 2024-06-03
    python src/viz/spatial_map_dk6.py --date 2024-07-06 --field seed42
    python src/viz/spatial_map_dk6.py --date 2024-01-20 \
        --snapshot-times 09:00 11:00 13:00 15:00

Outputs:
    - all-178-PV daily time series with S1/S2/S3 and P2
    - four absolute-GHI spatial snapshots using one shared color scale
    - four spatial-anomaly snapshots (PV GHI minus contemporaneous PV median)

The old src/viz/spatial_map.py reads outputs/predictions/ghi_pvs.parquet and
therefore plots the previous LOSO/Wendland product. This script reads only the
isolated DK-6 production directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import BG_DIR, OUTPUT_DIR, RESID_DIR, STATIONS  # noqa: E402


LOCAL_TZ = "America/Los_Angeles"
SCRIPT_VERSION = "DK6-P2-v2-20260825"
PLOT_STATIONS = ["S1", "S2", "S3", "P2"]
PRODUCTION_DIR = (
    Path(OUTPUT_DIR)
    / "experiments"
    / "DK6_production_direct_TPS_K016_three_seed"
)
PRED_DIR = PRODUCTION_DIR / "predictions"
FIG_DIR = PRODUCTION_DIR / "figures" / "spatial_maps"
FIELD_FILES = {
    "ensemble": "ghi_pvs.parquet",
    "seed42": "ghi_seed42_pvs.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date", required=True,
        help="Local calendar date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--field", choices=sorted(FIELD_FILES), default="ensemble",
        help="DK-6 field to plot (default: three-seed ensemble).",
    )
    parser.add_argument(
        "--snapshot-times", nargs=4,
        default=["09:00", "11:00", "13:00", "15:00"],
        metavar=("T1", "T2", "T3", "T4"),
        help="Four requested local snapshot times (default: 09:00 11:00 13:00 15:00).",
    )
    parser.add_argument(
        "--p2-provenance",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "stations" / "station_p2_full_year_GHI_zeroshot.csv",
        help="P2 zero-shot CSV containing timestamp and source columns.",
    )
    return parser.parse_args()


def as_local_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    index = pd.to_datetime(output.index)
    if index.tz is None:
        # Processed project artifacts use local Pacific timestamps when naive.
        index = index.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        index = index.tz_convert(LOCAL_TZ)
    output.index = index
    if output.index.duplicated().any():
        raise ValueError("Duplicate timestamps in input artifact")
    return output.sort_index()


def nearest_day_timestamp(
    day_index: pd.DatetimeIndex, date: pd.Timestamp, clock: str
) -> pd.Timestamp:
    try:
        hour, minute = [int(part) for part in clock.split(":", 1)]
    except Exception as exc:
        raise ValueError(f"Invalid snapshot time {clock!r}; use HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid snapshot time {clock!r}; use HH:MM")
    target = pd.Timestamp(
        year=date.year, month=date.month, day=date.day,
        hour=hour, minute=minute, tz=LOCAL_TZ,
    )
    distances = np.abs(day_index.asi8 - target.value)
    return day_index[int(np.argmin(distances))]


def p2_day_label(path: Path, date_obj: object) -> str:
    """Describe whether the displayed P2 day is measured or reconstructed."""
    if not path.exists():
        return "P2 (provenance unavailable)"
    frame = pd.read_csv(path)
    column_lookup = {str(column).strip().lower(): column for column in frame.columns}
    source_column = column_lookup.get("source")
    if source_column is None:
        return "P2 (provenance unavailable)"
    candidates = [
        "datetime_local", "datetime", "timestamp", "time", "date_time"
    ]
    timestamp_column = next(
        (column_lookup[name] for name in candidates if name in column_lookup), None
    )
    if timestamp_column is None:
        return "P2 (provenance unavailable)"
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    day_sources = (
        frame.loc[parsed.dt.date == date_obj, source_column]
        .astype(str).str.strip().str.lower()
    )
    # Night labels do not determine the provenance of the plotted daytime GHI.
    daytime_sources = day_sources[~day_sources.isin(["night", "measured_night"])]
    if daytime_sources.empty:
        return "P2 (no daytime provenance)"
    measured = daytime_sources.eq("measured")
    if measured.all():
        return "P2 measured"
    if measured.any():
        return "P2 mixed measured/reconstructed"
    unique_sources = set(daytime_sources.unique())
    if unique_sources == {"transformer_zeroshot"}:
        return "P2 reconstructed (Transformer)"
    return "P2 reconstructed (zero-shot/fallback)"


def spatial_absolute_axis(
    axis: plt.Axes,
    values: pd.Series,
    station_values: pd.Series,
    pv_df: pd.DataFrame,
    vmin: float,
    vmax: float,
    timestamp: pd.Timestamp,
) -> object:
    pv_values = values.reindex(pv_df.index).to_numpy(dtype=float)
    scatter = axis.scatter(
        pv_df["pv_lon"], pv_df["pv_lat"], c=pv_values,
        cmap="viridis", vmin=vmin, vmax=vmax,
        s=58, edgecolors="0.35", linewidths=0.35, zorder=3,
    )
    for station in PLOT_STATIONS:
        info = STATIONS[station]
        measured = float(station_values.get(station, np.nan))
        axis.scatter(
            info["lon"], info["lat"], c=[measured], cmap="viridis",
            vmin=vmin, vmax=vmax, marker="*", s=280,
            edgecolors="black", linewidths=1.0, zorder=5,
        )
        axis.annotate(
            station, (info["lon"], info["lat"]), xytext=(4, 4),
            textcoords="offset points", fontsize=8, fontweight="bold",
        )
    axis.set_title(
        f"{timestamp.strftime('%H:%M')} local\n"
        f"PV mean={np.nanmean(pv_values):.0f}, "
        f"std={np.nanstd(pv_values):.1f} W/m²",
        fontsize=10,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.22, ls="--")
    return scatter


def spatial_anomaly_axis(
    axis: plt.Axes,
    values: pd.Series,
    pv_df: pd.DataFrame,
    limit: float,
    timestamp: pd.Timestamp,
) -> object:
    pv_values = values.reindex(pv_df.index).to_numpy(dtype=float)
    median = float(np.nanmedian(pv_values))
    anomalies = pv_values - median
    scatter = axis.scatter(
        pv_df["pv_lon"], pv_df["pv_lat"], c=anomalies,
        cmap="RdBu_r", vmin=-limit, vmax=limit,
        s=58, edgecolors="0.35", linewidths=0.35, zorder=3,
    )
    for station in PLOT_STATIONS:
        info = STATIONS[station]
        axis.scatter(
            info["lon"], info["lat"], marker="*", s=240,
            color="black", edgecolors="white", linewidths=0.7, zorder=5,
        )
        axis.annotate(
            station, (info["lon"], info["lat"]), xytext=(4, 4),
            textcoords="offset points", fontsize=8, fontweight="bold",
        )
    axis.set_title(
        f"{timestamp.strftime('%H:%M')} local\n"
        f"median={median:.0f}, anomaly std={np.nanstd(anomalies):.1f} W/m²",
        fontsize=10,
    )
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.22, ls="--")
    return scatter


def main() -> None:
    args = parse_args()
    target_date = pd.Timestamp(args.date)
    if target_date.strftime("%Y-%m-%d") != args.date:
        raise ValueError("--date must use exact YYYY-MM-DD format")

    prediction_path = PRED_DIR / FIELD_FILES[args.field]
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    ghi = as_local_index(pd.read_parquet(prediction_path))

    pv_path = REPO_ROOT / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df = pd.read_csv(pv_path)
    required_pv = {"pv_name", "pv_lat", "pv_lon"}
    if not required_pv.issubset(pv_df.columns):
        raise ValueError(f"{pv_path} lacks {sorted(required_pv - set(pv_df.columns))}")
    pv_df["pv_name"] = pv_df["pv_name"].astype(str)
    pv_df = pv_df.set_index("pv_name")
    pv_names = ghi.columns.astype(str).tolist()
    missing_pvs = [name for name in pv_names if name not in pv_df.index]
    if missing_pvs:
        raise ValueError(f"PV coordinate table lacks {missing_pvs[0]}")
    pv_df = pv_df.loc[pv_names]

    station_csi = as_local_index(
        pd.read_parquet(Path(RESID_DIR) / "csi_stations.parquet")
    )
    station_clear = as_local_index(
        pd.read_parquet(Path(BG_DIR) / "bg_clearsky_stations.parquet")
    )
    common_station = station_csi.index.intersection(station_clear.index)
    station_ghi = (
        station_csi.loc[common_station, PLOT_STATIONS].clip(0.0, 1.3)
        * station_clear.loc[common_station, PLOT_STATIONS]
    )

    date_obj = target_date.date()
    p2_label = p2_day_label(args.p2_provenance, date_obj)
    day = ghi.loc[ghi.index.date == date_obj, pv_names].dropna(how="all")
    if day.empty:
        raise ValueError(
            f"No DK-6 predictions for {args.date}; available range is "
            f"{ghi.index.min().date()} to {ghi.index.max().date()}"
        )
    day_station = station_ghi.loc[station_ghi.index.date == date_obj]
    snapshots = [
        nearest_day_timestamp(day.index, target_date, clock)
        for clock in args.snapshot_times
    ]
    if len(set(snapshots)) != 4:
        raise ValueError(
            "Requested snapshot times mapped to duplicate available timestamps; "
            "choose more widely separated times."
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.date}_{args.field}"
    print("=" * 76)
    print("DK-6 DAY-SPECIFIC SPATIAL MAP")
    print("=" * 76)
    print(f"Script version    : {SCRIPT_VERSION}")
    print(f"Date              : {args.date}")
    print(f"Field             : {args.field}")
    print(f"Daytime rows      : {len(day)}")
    print(f"Day GHI range     : {np.nanmin(day.to_numpy()):.1f}–{np.nanmax(day.to_numpy()):.1f} W/m²")
    print(f"P2 provenance     : {p2_label}")
    print("Snapshots         : " + ", ".join(ts.strftime("%H:%M") for ts in snapshots))

    # 1. Daily bundle and measured complete stations.
    fig, axis = plt.subplots(figsize=(14, 5.8))
    for pv in pv_names:
        axis.plot(day.index, day[pv], color="#4393c3", alpha=0.12, lw=0.65)
    pv_median = day.median(axis=1)
    pv_q10 = day.quantile(0.10, axis=1)
    pv_q90 = day.quantile(0.90, axis=1)
    axis.fill_between(day.index, pv_q10, pv_q90, color="#92c5de", alpha=0.30, label="PV 10–90% range")
    axis.plot(day.index, pv_median, color="#2166ac", lw=2.3, label=f"PV median prediction")
    station_colors = {
        "S1": "#d73027", "S2": "#1a9850",
        "S3": "#762a83", "P2": "#f46d43",
    }
    for station in PLOT_STATIONS:
        if station in day_station:
            station_label = p2_label if station == "P2" else f"{station} measured"
            axis.plot(
                day_station.index, day_station[station],
                color=station_colors[station], lw=1.8, ls="--",
                label=station_label,
            )
    axis.set_title(
        f"Predicted GHI at 178 PV Locations vs Station Measurements - {args.date}",
        fontsize=13, fontweight="bold",
    )
    axis.set_xlabel(f"Local time ({LOCAL_TZ})")
    axis.set_ylabel("GHI (W/m²)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=5, fontsize=8, loc="upper left")
    axis.set_xlim(day.index.min(), day.index.max())
    fig.tight_layout()
    time_path = FIG_DIR / f"fig_{tag}_timeseries.png"
    fig.savefig(time_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Values at snapshot timestamps and station values nearest within 3 min.
    station_at_snapshot: dict[pd.Timestamp, pd.Series] = {}
    for timestamp in snapshots:
        if day_station.empty:
            station_at_snapshot[timestamp] = pd.Series(dtype=float)
            continue
        position = int(np.argmin(np.abs(day_station.index.asi8 - timestamp.value)))
        nearest = day_station.index[position]
        if abs(nearest.value - timestamp.value) > pd.Timedelta(minutes=3).value:
            station_at_snapshot[timestamp] = pd.Series(dtype=float)
        else:
            station_at_snapshot[timestamp] = day_station.iloc[position]

    # 2. Absolute spatial maps with a genuinely shared color scale.
    snapshot_prediction_values = np.concatenate(
        [day.loc[timestamp].to_numpy(dtype=float) for timestamp in snapshots]
    )
    snapshot_station_values = np.concatenate(
        [series.to_numpy(dtype=float) for series in station_at_snapshot.values() if len(series)]
    ) if any(len(series) for series in station_at_snapshot.values()) else np.array([])
    scale_values = np.concatenate([snapshot_prediction_values, snapshot_station_values])
    scale_values = scale_values[np.isfinite(scale_values)]
    vmax = max(1.0, float(np.nanquantile(scale_values, 0.99)))

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.7))
    for axis, timestamp in zip(axes, snapshots):
        scatter = spatial_absolute_axis(
            axis, day.loc[timestamp], station_at_snapshot[timestamp],
            pv_df, 0.0, vmax, timestamp,
        )
    colorbar = fig.colorbar(scatter, ax=list(axes), shrink=0.72, pad=0.02)
    colorbar.set_label("Absolute GHI (W/m²)")
    fig.suptitle(
        f"DK-6 {args.field.title()} Absolute GHI Spatial Field — {args.date}\n"
        f"PV circles; S1–S3 measured; {p2_label}; shared color scale",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.80, wspace=0.32, right=0.91)
    absolute_path = FIG_DIR / f"fig_{tag}_spatial_absolute.png"
    fig.savefig(absolute_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Median-centered anomaly maps to reveal spatial structure hidden by
    # the much larger absolute irradiance scale.
    anomaly_arrays = []
    for timestamp in snapshots:
        values = day.loc[timestamp].to_numpy(dtype=float)
        anomaly_arrays.append(values - np.nanmedian(values))
    all_anomalies = np.concatenate(anomaly_arrays)
    finite_anomalies = np.abs(all_anomalies[np.isfinite(all_anomalies)])
    anomaly_limit = max(1.0, float(np.nanquantile(finite_anomalies, 0.98)))

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.7))
    for axis, timestamp in zip(axes, snapshots):
        scatter = spatial_anomaly_axis(
            axis, day.loc[timestamp], pv_df, anomaly_limit, timestamp
        )
    colorbar = fig.colorbar(scatter, ax=list(axes), shrink=0.72, pad=0.02)
    colorbar.set_label("PV GHI minus contemporaneous PV median (W/m²)")
    fig.suptitle(
        f"DK-6 {args.field.title()} Median-Centered Spatial Anomalies — {args.date}\n"
        "Use this figure to evaluate satellite-conditioned spatial diversity",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(top=0.80, wspace=0.32, right=0.91)
    anomaly_path = FIG_DIR / f"fig_{tag}_spatial_anomaly.png"
    fig.savefig(anomaly_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    snapshot_rows = []
    for timestamp in snapshots:
        values = day.loc[timestamp].to_numpy(dtype=float)
        snapshot_rows.append(
            {
                "datetime_local": timestamp,
                "pv_mean_ghi": float(np.nanmean(values)),
                "pv_median_ghi": float(np.nanmedian(values)),
                "pv_std_ghi": float(np.nanstd(values)),
                "pv_min_ghi": float(np.nanmin(values)),
                "pv_max_ghi": float(np.nanmax(values)),
                "n_unique_ghi_0p1": int(len(np.unique(np.round(values[np.isfinite(values)], 1)))),
            }
        )
    summary_path = FIG_DIR / f"fig_{tag}_snapshot_summary.csv"
    pd.DataFrame(snapshot_rows).to_csv(summary_path, index=False)

    print("Saved:")
    print(f"  {time_path}")
    print(f"  {absolute_path}")
    print(f"  {anomaly_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()