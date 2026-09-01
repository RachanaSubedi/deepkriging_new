"""Diagnose absolute level plausibility of DK-6 predictions at 178 PVs.

Place at:
    src/validation/diagnose_dk6_pv_levels.py

Run:
    python src/validation/diagnose_dk6_pv_levels.py

The diagnostic uses S1/S2/S3 only as the measured reference. P2 is excluded
because most of its annual record is reconstructed. It compares:
    - DK-6 ensemble PV median
    - DK-6 seed-42 PV median
    - measured S1/S2/S3 median and min/max envelope
    - fold-independent station-IDW field at all PVs, using S1/S2/S3
    - NSRDB PV-field median

These are plausibility diagnostics, not additional validation metrics: PV
truth is unavailable and a real PV need not always lie inside the station
envelope.
"""

from __future__ import annotations

import json
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


PRODUCTION_DIR = (
    Path(OUTPUT_DIR)
    / "experiments"
    / "DK6_production_direct_TPS_K016_three_seed"
)
PRED_DIR = PRODUCTION_DIR / "predictions"
OUT_DIR = PRODUCTION_DIR / "level_diagnostics"
COMPLETE_STATIONS = ["S1", "S2", "S3"]
IDW_POWER = 2.0
LOCAL_TZ = "America/Los_Angeles"
CLEARSKY_MIN = 10.0


def utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.to_datetime(result.index, utc=True)
    if result.index.duplicated().any():
        raise ValueError("Duplicate timestamps found")
    return result.sort_index()


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    radius = 6371.0088
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def station_idw_pv_median(
    station_csi: np.ndarray,
    pv_clearsky: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return median PV GHI of the S1/S2/S3 IDW field at each timestamp."""
    n_time = station_csi.shape[0]
    output = np.full(n_time, np.nan, dtype=np.float32)
    for start in range(0, n_time, chunk_size):
        end = min(start + chunk_size, n_time)
        values = station_csi[start:end]
        available = np.isfinite(values)
        numerator = np.nan_to_num(values, nan=0.0) @ weights.T
        denominator = available.astype(np.float64) @ weights.T
        with np.errstate(invalid="ignore", divide="ignore"):
            csi_field = numerator / denominator
        csi_field[denominator <= 0] = np.nan
        csi_field = np.clip(csi_field, 0.0, 1.3)
        ghi_field = csi_field * pv_clearsky[start:end]
        output[start:end] = np.nanmedian(ghi_field, axis=1)
    return output


def select_unique_day(
    ranking: pd.Series,
    already_selected: set,
    ascending: bool,
) -> object:
    for date in ranking.sort_values(ascending=ascending).index:
        if date not in already_selected:
            already_selected.add(date)
            return date
    raise RuntimeError("Could not select a unique representative day")


def main() -> None:
    required = {
        "ensemble_ghi": PRED_DIR / "ghi_pvs.parquet",
        "seed42_ghi": PRED_DIR / "ghi_seed42_pvs.parquet",
        "ensemble_csi": PRED_DIR / "csi_pvs.parquet",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)

    ensemble_ghi = utc_index(pd.read_parquet(required["ensemble_ghi"]))
    seed42_ghi = utc_index(pd.read_parquet(required["seed42_ghi"]))
    ensemble_csi = utc_index(pd.read_parquet(required["ensemble_csi"]))
    bg_csi_pv = utc_index(pd.read_parquet(Path(BG_DIR) / "bg_csi_pvs.parquet"))
    bg_clear_pv = utc_index(pd.read_parquet(Path(BG_DIR) / "bg_clearsky_pvs.parquet"))
    station_csi = utc_index(pd.read_parquet(Path(RESID_DIR) / "csi_stations.parquet"))
    station_clear = utc_index(pd.read_parquet(Path(BG_DIR) / "bg_clearsky_stations.parquet"))

    pv_names = ensemble_ghi.columns.astype(str).tolist()
    for label, frame in {
        "seed42": seed42_ghi,
        "ensemble_csi": ensemble_csi,
        "bg_csi_pv": bg_csi_pv,
        "bg_clear_pv": bg_clear_pv,
    }.items():
        missing = [pv for pv in pv_names if pv not in frame.columns]
        if missing:
            raise ValueError(f"{label} lacks PV {missing[0]}")
    for station in COMPLETE_STATIONS:
        if station not in station_csi or station not in station_clear:
            raise ValueError(f"Station artifact lacks {station}")

    common = ensemble_ghi.index
    for frame in [seed42_ghi, ensemble_csi, bg_csi_pv, bg_clear_pv, station_csi, station_clear]:
        common = common.intersection(frame.index)
    common = common.sort_values()
    if len(common) == 0:
        raise ValueError("No common timestamps across diagnostic inputs")

    ensemble_ghi = ensemble_ghi.loc[common, pv_names]
    seed42_ghi = seed42_ghi.loc[common, pv_names]
    ensemble_csi = ensemble_csi.loc[common, pv_names]
    bg_csi_pv = bg_csi_pv.loc[common, pv_names]
    bg_clear_pv = bg_clear_pv.loc[common, pv_names]
    station_csi = station_csi.loc[common, COMPLETE_STATIONS]
    station_clear = station_clear.loc[common, COMPLETE_STATIONS]

    pv_path = REPO_ROOT / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df = pd.read_csv(pv_path)
    if not {"pv_name", "pv_lat", "pv_lon"}.issubset(pv_df.columns):
        raise ValueError("pv_nn_assignments.csv lacks pv_name/pv_lat/pv_lon")
    pv_df = pv_df.set_index("pv_name").loc[pv_names]
    station_lat = np.array([STATIONS[s]["lat"] for s in COMPLETE_STATIONS])
    station_lon = np.array([STATIONS[s]["lon"] for s in COMPLETE_STATIONS])
    pv_lat = pv_df["pv_lat"].to_numpy(dtype=float)
    pv_lon = pv_df["pv_lon"].to_numpy(dtype=float)
    distances = haversine_km(
        pv_lat[:, None], pv_lon[:, None],
        station_lat[None, :], station_lon[None, :],
    )
    weights = 1.0 / np.maximum(distances, 1e-6) ** IDW_POWER

    station_ghi_values = (
        np.clip(station_csi.to_numpy(dtype=float), 0.0, 1.3)
        * station_clear.to_numpy(dtype=float)
    )
    station_median = np.nanmedian(station_ghi_values, axis=1)
    station_min = np.nanmin(station_ghi_values, axis=1)
    station_max = np.nanmax(station_ghi_values, axis=1)
    station_csi_median = np.nanmedian(station_csi.to_numpy(dtype=float), axis=1)

    ensemble_values = ensemble_ghi.to_numpy(dtype=float)
    seed42_values = seed42_ghi.to_numpy(dtype=float)
    pv_clear_values = bg_clear_pv.to_numpy(dtype=float)
    nsrdb_values = (
        np.clip(bg_csi_pv.to_numpy(dtype=float), 0.0, 1.3) * pv_clear_values
    )
    ensemble_median = np.nanmedian(ensemble_values, axis=1)
    seed42_median = np.nanmedian(seed42_values, axis=1)
    nsrdb_median = np.nanmedian(nsrdb_values, axis=1)
    idw_median = station_idw_pv_median(
        station_csi.to_numpy(dtype=float), pv_clear_values, weights
    )

    valid_station_envelope = np.isfinite(station_min) & np.isfinite(station_max)
    below_fraction = np.full(len(common), np.nan)
    above_fraction = np.full(len(common), np.nan)
    envelope_values = ensemble_values[valid_station_envelope]
    valid_pv = np.isfinite(envelope_values)
    denominator = valid_pv.sum(axis=1)
    usable = denominator > 0
    below_local = np.full(len(envelope_values), np.nan)
    above_local = np.full(len(envelope_values), np.nan)
    below_local[usable] = (
        (
            valid_pv
            & (envelope_values < station_min[valid_station_envelope, None])
        ).sum(axis=1)[usable]
        / denominator[usable]
    )
    above_local[usable] = (
        (
            valid_pv
            & (envelope_values > station_max[valid_station_envelope, None])
        ).sum(axis=1)[usable]
        / denominator[usable]
    )
    below_fraction[valid_station_envelope] = below_local
    above_fraction[valid_station_envelope] = above_local

    local_index = common.tz_convert(LOCAL_TZ)
    time_series = pd.DataFrame(
        {
            "dk6_ensemble_pv_median": ensemble_median,
            "dk6_seed42_pv_median": seed42_median,
            "station_median": station_median,
            "station_min": station_min,
            "station_max": station_max,
            "station_idw_pv_median": idw_median,
            "nsrdb_pv_median": nsrdb_median,
            "fraction_pvs_below_station_min": below_fraction,
            "fraction_pvs_above_station_max": above_fraction,
            "station_csi_median": station_csi_median,
        },
        index=local_index,
    )
    time_series.index.name = "datetime_local"
    daytime = np.nanmedian(pv_clear_values, axis=1) >= CLEARSKY_MIN
    time_series = time_series.loc[daytime]
    time_series["date"] = time_series.index.date
    time_series["dk6_minus_idw"] = (
        time_series["dk6_ensemble_pv_median"]
        - time_series["station_idw_pv_median"]
    )
    time_series["dk6_minus_station_median"] = (
        time_series["dk6_ensemble_pv_median"] - time_series["station_median"]
    )
    time_series["ensemble_minus_seed42"] = (
        time_series["dk6_ensemble_pv_median"]
        - time_series["dk6_seed42_pv_median"]
    )

    grouped = time_series.groupby("date")
    daily = grouped.agg(
        n_daytime=("dk6_ensemble_pv_median", "count"),
        dk6_mean=("dk6_ensemble_pv_median", "mean"),
        seed42_mean=("dk6_seed42_pv_median", "mean"),
        station_mean=("station_median", "mean"),
        idw_mean=("station_idw_pv_median", "mean"),
        nsrdb_mean=("nsrdb_pv_median", "mean"),
        mean_dk6_minus_idw=("dk6_minus_idw", "mean"),
        mean_dk6_minus_station=("dk6_minus_station_median", "mean"),
        mean_ensemble_minus_seed42=("ensemble_minus_seed42", "mean"),
        mean_fraction_below=("fraction_pvs_below_station_min", "mean"),
        mean_fraction_above=("fraction_pvs_above_station_max", "mean"),
        mean_station_csi=("station_csi_median", "mean"),
    )
    ramp_p95 = grouped["station_median"].apply(
        lambda x: float(np.nanquantile(np.abs(np.diff(x.to_numpy())), 0.95))
        if len(x) > 1 else np.nan
    )
    daily["station_median_ramp_p95"] = ramp_p95

    selected: set = set()
    representative = {
        "largest_low_bias": select_unique_day(
            daily["mean_dk6_minus_idw"], selected, ascending=True
        ),
        "largest_high_bias": select_unique_day(
            daily["mean_dk6_minus_idw"], selected, ascending=False
        ),
        "most_variable": select_unique_day(
            daily["station_median_ramp_p95"], selected, ascending=False
        ),
        "clearest": select_unique_day(
            daily["mean_station_csi"], selected, ascending=False
        ),
    }
    for label, date in representative.items():
        daily.loc[date, "representative_reason"] = label

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    time_series.drop(columns="date").to_parquet(OUT_DIR / "level_timeseries.parquet")
    daily.to_csv(OUT_DIR / "daily_level_diagnostics.csv")

    fig, axes = plt.subplots(4, 1, figsize=(14, 17), sharex=False)
    colors = {
        "ensemble": "#2166ac",
        "seed42": "#67a9cf",
        "station": "#111111",
        "idw": "#1b9e77",
        "nsrdb": "#d95f02",
    }
    for axis, (reason, date) in zip(axes, representative.items()):
        day = time_series[time_series["date"] == date]
        axis.fill_between(
            day.index,
            day["station_min"],
            day["station_max"],
            color="0.75", alpha=0.45, label="S1–S3 envelope",
        )
        axis.plot(day.index, day["station_median"], color=colors["station"], lw=1.8, label="S1–S3 median")
        axis.plot(day.index, day["station_idw_pv_median"], color=colors["idw"], lw=1.6, label="Station-IDW PV median")
        axis.plot(day.index, day["nsrdb_pv_median"], color=colors["nsrdb"], lw=1.3, label="NSRDB PV median")
        axis.plot(day.index, day["dk6_seed42_pv_median"], color=colors["seed42"], lw=1.2, ls="--", label="DK-6 seed 42")
        axis.plot(day.index, day["dk6_ensemble_pv_median"], color=colors["ensemble"], lw=2.0, label="DK-6 ensemble")
        axis.set_title(
            f"{reason.replace('_', ' ').title()} — {date} — "
            f"mean DK6−IDW={daily.loc[date, 'mean_dk6_minus_idw']:+.1f} W/m²"
        )
        axis.set_ylabel("GHI (W/m²)")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9, loc="upper left")
    axes[-1].set_xlabel(f"Local time ({LOCAL_TZ})")
    fig.suptitle("DK-6 Absolute-Level Plausibility Diagnostic", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_DIR / "representative_level_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "reference_stations": COMPLETE_STATIONS,
        "p2_excluded_from_reference": True,
        "n_common_timestamps": int(len(common)),
        "mean_dk6_minus_idw_wm2": float(time_series["dk6_minus_idw"].mean()),
        "median_dk6_minus_idw_wm2": float(time_series["dk6_minus_idw"].median()),
        "p05_dk6_minus_idw_wm2": float(time_series["dk6_minus_idw"].quantile(0.05)),
        "p95_dk6_minus_idw_wm2": float(time_series["dk6_minus_idw"].quantile(0.95)),
        "mean_abs_ensemble_minus_seed42_wm2": float(time_series["ensemble_minus_seed42"].abs().mean()),
        "mean_fraction_pvs_below_station_min": float(time_series["fraction_pvs_below_station_min"].mean()),
        "mean_fraction_pvs_above_station_max": float(time_series["fraction_pvs_above_station_max"].mean()),
        "representative_days": {key: str(value) for key, value in representative.items()},
        "interpretation_warning": (
            "Station-envelope comparisons are plausibility diagnostics only; "
            "PV-site truth is unavailable."
        ),
    }
    (OUT_DIR / "level_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 82)
    print("DK-6 LEVEL DIAGNOSTIC COMPLETE")
    print("=" * 82)
    print(f"Mean DK6 minus station-IDW : {summary['mean_dk6_minus_idw_wm2']:+.2f} W/m²")
    print(f"Median DK6 minus IDW       : {summary['median_dk6_minus_idw_wm2']:+.2f} W/m²")
    print(
        "5th–95th percentile gap   : "
        f"{summary['p05_dk6_minus_idw_wm2']:+.2f} to "
        f"{summary['p95_dk6_minus_idw_wm2']:+.2f} W/m²"
    )
    print(
        "Mean |ensemble−seed42|    : "
        f"{summary['mean_abs_ensemble_minus_seed42_wm2']:.2f} W/m²"
    )
    print(
        "Mean PV fraction below/above S1–S3 envelope: "
        f"{summary['mean_fraction_pvs_below_station_min']:.1%} / "
        f"{summary['mean_fraction_pvs_above_station_max']:.1%}"
    )
    print("Representative days:")
    for reason, date in representative.items():
        print(f"  {reason:<20}: {date}")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
