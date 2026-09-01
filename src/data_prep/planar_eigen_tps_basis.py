"""Build an ordered planar eigen-thin-plate-spline basis for DK-2.

Place at: src/data_prep/planar_eigen_tps_basis.py

The fixed NSRDB grid is used as the knot set.  This is response-independent,
covers the feeder domain, and avoids choosing knots from held-out stations.
The TPS kernel is projected off its affine null space [1, x, y], then
eigendecomposed.  Positive eigencomponents are ordered by decreasing
eigenvalue (broad/smooth components first).  New sites are evaluated with a
Nyström extension.  The first three columns are the affine null space.

Outputs are isolated from the existing Wendland artifact.
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

from configs.config import (  # noqa: E402
    KM_PER_LAT, KM_PER_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    NSRDB_RES, PROCESSED_DIR, STATIONS,
)

OUT_DIR = Path(PROCESSED_DIR) / "basis_planar_eigen_tps"
DEFAULT_MAX_K = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build planar eigen-TPS bases")
    p.add_argument("--max-basis", type=int, default=DEFAULT_MAX_K,
                   help="Maximum TOTAL columns saved, including 1/x/y (default 32)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def project_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Local equirectangular coordinates in km, centered on the domain."""
    lat0 = 0.5 * (LAT_MIN + LAT_MAX)
    lon0 = 0.5 * (LON_MIN + LON_MAX)
    x = (np.asarray(lon) - lon0) * KM_PER_LON
    y = (np.asarray(lat) - lat0) * KM_PER_LAT
    return np.column_stack([x, y]).astype(np.float64)


def tps_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Planar order-2 TPS kernel r^2 log(r), with phi(0)=0."""
    d = a[:, None, :] - b[None, :, :]
    r = np.sqrt(np.sum(d * d, axis=2))
    out = np.zeros_like(r)
    positive = r > 0
    out[positive] = r[positive] ** 2 * np.log(r[positive])
    return out


def affine_matrix(xy: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (xy - center) / scale
    return np.column_stack([np.ones(len(xy)), z[:, 0], z[:, 1]])


def fit_basis(knots: np.ndarray, max_total: int) -> dict:
    if max_total < 3:
        raise ValueError("--max-basis must be at least 3 (affine null space)")
    center = knots.mean(axis=0)
    scale = knots.std(axis=0, ddof=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate knot coordinates")

    T = affine_matrix(knots, center, scale)
    # Orthogonal projector onto the complement of [1, x, y].
    Q = np.eye(len(knots)) - T @ np.linalg.pinv(T)
    E = tps_kernel(knots, knots)
    A = 0.5 * (Q @ E @ Q + (Q @ E @ Q).T)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    keep = eigenvalues > max(1.0, abs(eigenvalues).max()) * 1e-10
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_eigen = min(max_total - 3, len(eigenvalues))
    if n_eigen < max_total - 3:
        raise ValueError(
            f"Only {len(eigenvalues)} positive projected TPS eigencomponents "
            f"are available; cannot build {max_total} total columns"
        )
    return {
        "center": center, "scale": scale, "T": T, "Q": Q,
        "eigenvalues": eigenvalues[:n_eigen],
        "eigenvectors": eigenvectors[:, :n_eigen],
    }


def evaluate_basis(xy: np.ndarray, knots: np.ndarray, fitted: dict) -> np.ndarray:
    """Evaluate affine terms plus Nyström-extended TPS eigenfunctions."""
    T_new = affine_matrix(xy, fitted["center"], fitted["scale"])
    K_new = tps_kernel(xy, knots)
    lam = fitted["eigenvalues"]
    vec = fitted["eigenvectors"]
    # Remove the affine component on both sides.  The second term is the
    # out-of-sample counterpart of left multiplication by Q.  At the knot
    # sites this becomes Q E Q, so the extension exactly equals
    # V*sqrt(lambda), up to numerical precision.
    E_knots = tps_kernel(knots, knots)
    cross_projected = (
        K_new @ fitted["Q"]
        - T_new @ np.linalg.pinv(fitted["T"]) @ E_knots @ fitted["Q"]
    )
    nonlinear = (cross_projected @ vec) / np.sqrt(lam)[None, :]
    phi = np.column_stack([T_new, nonlinear])
    return phi.astype(np.float32)


def load_locations() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    station_names = list(STATIONS)
    st_lat = np.array([STATIONS[n]["lat"] for n in station_names])
    st_lon = np.array([STATIONS[n]["lon"] for n in station_names])
    stations = project_km(st_lat, st_lon)

    lats = np.arange(LAT_MIN, LAT_MAX + NSRDB_RES / 2, NSRDB_RES)
    lons = np.arange(LON_MIN, LON_MAX + NSRDB_RES / 2, NSRDB_RES)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    nsrdb = project_km(lat_grid.ravel(), lon_grid.ravel())

    pv_path = REPO_ROOT / "data" / "raw" / "pv_nn_assignments.csv"
    pv = pd.read_csv(pv_path)
    required = {"pv_lat", "pv_lon"}
    if not required.issubset(pv.columns):
        raise ValueError(f"{pv_path} lacks columns {sorted(required)}")
    pvs = project_km(pv["pv_lat"].to_numpy(), pv["pv_lon"].to_numpy())
    pv_names = (pv["pv_name"].astype(str).tolist() if "pv_name" in pv
                else [f"PV_{i:03d}" for i in range(len(pv))])
    return stations, nsrdb, pvs, station_names + pv_names


def main() -> None:
    args = parse_args()
    marker = OUT_DIR / "basis_spec.json"
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"{marker} already exists; use --overwrite intentionally")

    stations, knots, pvs, names = load_locations()
    fitted = fit_basis(knots, args.max_basis)
    phi_st = evaluate_basis(stations, knots, fitted)
    phi_ns = evaluate_basis(knots, knots, fitted)
    phi_pv = evaluate_basis(pvs, knots, fitted)

    # Response-independent column normalization fitted on the fixed knot grid.
    # This prevents large TPS eigenvalues from dominating the 15 atmospheric
    # covariates merely because of numerical scale.
    column_rms = np.sqrt(np.mean(phi_ns.astype(np.float64) ** 2, axis=0))
    if np.any(column_rms < 1e-12):
        raise ValueError("A TPS basis column has effectively zero knot-grid RMS")
    phi_st = (phi_st / column_rms).astype(np.float32)
    phi_ns = (phi_ns / column_rms).astype(np.float32)
    phi_pv = (phi_pv / column_rms).astype(np.float32)

    # Validate the Nyström identity at knots and numerical quality.
    expected_nonlinear = fitted["eigenvectors"] * np.sqrt(
        fitted["eigenvalues"]
    )[None, :] / column_rms[3:][None, :]
    nystrom_error = float(np.max(np.abs(phi_ns[:, 3:] - expected_nonlinear)))
    if nystrom_error > 2e-3:
        raise RuntimeError(f"Nyström knot reconstruction failed: {nystrom_error:g}")
    if not all(np.isfinite(x).all() for x in (phi_st, phi_ns, phi_pv)):
        raise ValueError("Non-finite eigen-TPS basis values")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "Phi_stations.npy", phi_st)
    np.save(OUT_DIR / "Phi_nsrdb.npy", phi_ns)
    np.save(OUT_DIR / "Phi_pvs.npy", phi_pv)
    np.save(OUT_DIR / "knot_xy_km.npy", knots)
    np.save(OUT_DIR / "eigenvalues.npy", fitted["eigenvalues"])
    np.save(OUT_DIR / "projection_center_km.npy", fitted["center"])
    np.save(OUT_DIR / "projection_scale_km.npy", fitted["scale"])
    np.save(OUT_DIR / "basis_column_rms.npy", column_rms)

    ranks = {str(k): int(np.linalg.matrix_rank(phi_st[:, :k]))
             for k in (4, 8, 16, 32) if k <= phi_st.shape[1]}
    spec = {
        "version": "planar_projected_eigen_tps_v1",
        "kernel": "r^2 log(r)",
        "coordinate_system": "local_equirectangular_km",
        "knot_source": "fixed_nsrdb_grid",
        "n_knots": int(len(knots)),
        "n_total_basis": int(phi_st.shape[1]),
        "n_affine_null_space": 3,
        "n_positive_eigen_basis": int(phi_st.shape[1] - 3),
        "column_order": ["intercept", "x_standardized", "y_standardized"]
                        + [f"eigen_tps_{i+1:03d}" for i in range(phi_st.shape[1]-3)],
        "eigen_order": "decreasing_positive_eigenvalue",
        "column_normalization": "RMS on fixed NSRDB knot grid",
        "candidate_total_K": [k for k in (4, 8, 16, 32) if k <= phi_st.shape[1]],
        "station_matrix_rank_by_K": ranks,
        "nystrom_max_abs_error_at_knots": nystrom_error,
        "response_data_used": False,
        "note": "K counts affine and nonlinear columns; select K by strict LOSO.",
    }
    marker.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print("=" * 72)
    print("PLANAR EIGEN-TPS BASIS CREATED")
    print("=" * 72)
    print(f"Output directory : {OUT_DIR}")
    print(f"Knots (NSRDB)    : {len(knots)}")
    print(f"Saved columns    : {phi_st.shape[1]} (3 affine + {phi_st.shape[1]-3} eigen)")
    print(f"Station ranks    : {ranks}")
    print(f"Nyström error    : {nystrom_error:.3e}")
    print("No irradiance targets were used and no model was trained.")


if __name__ == "__main__":
    main()
