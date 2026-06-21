"""
src/basis_functions.py

Builds multi-resolution Wendland RBF basis matrix for all locations.

"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Allow running from project root or src/
sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    KM_PER_LAT, KM_PER_LON, BASIS_LEVELS,
    NSRDB_LAT_MIN, NSRDB_LAT_MAX,
    NSRDB_LON_MIN, NSRDB_LON_MAX, NSRDB_RES,
    STATIONS, BASIS_DIR,
)

# ── DOMAIN & LEVELS ───────────────────────────────────────────
domain = {
    'lat_min': LAT_MIN, 'lat_max': LAT_MAX,
    'lon_min': LON_MIN, 'lon_max': LON_MAX,
}
levels = BASIS_LEVELS


# ── STEP 1: BUILD KNOT GRID ───────────────────────────────────
def build_knot_grid(domain, levels, km_per_lat, km_per_lon):
    """
    Returns list of (knots_array, theta_km) per level.
    knots_array shape: (n_knots, 2) = [lat, lon]
    """
    result = []
    for lv in levels:
        step_lat = lv['spacing_km'] / km_per_lat
        step_lon = lv['spacing_km'] / km_per_lon

        lat_knots = np.arange(domain['lat_min'],
                              domain['lat_max'] + step_lat * 0.5,
                              step_lat)
        lon_knots = np.arange(domain['lon_min'],
                              domain['lon_max'] + step_lon * 0.5,
                              step_lon)

        lat_grid, lon_grid = np.meshgrid(lat_knots, lon_knots,
                                          indexing='ij')
        knots = np.column_stack([lat_grid.ravel(),
                                  lon_grid.ravel()])
        result.append((knots, lv['theta_km']))

        print(f"  Level {len(result)} "
              f"({lv['spacing_km']} km): "
              f"{len(lat_knots)} × {len(lon_knots)} "
              f"= {len(knots)} knots")
    return result


# ── STEP 2: WENDLAND BASIS FUNCTION ──────────────────────────
def wendland(d_norm):
    """
    Compactly supported Wendland basis function.
    d_norm = physical_distance_km / theta_km
    Returns 0 when d_norm >= 1.
    """
    out = np.zeros_like(d_norm, dtype=np.float32)
    mask = d_norm < 1.0
    d = d_norm[mask]
    out[mask] = ((1 - d) ** 6 * (35 * d**2 + 18 * d + 3)) / 3.0
    return out


# ── STEP 3: COMPUTE BASIS MATRIX ─────────────────────────────
def compute_basis_matrix(locations, knot_levels,
                          km_per_lat, km_per_lon):
    """
    locations  : np.array shape (N, 2) = [[lat, lon], ...]
    knot_levels: output of build_knot_grid()

    Returns Phi (N, K) and column metadata list.
    """
    cols = []
    meta = []

    for lv_idx, (knots, theta_km) in enumerate(knot_levels):
        for k_idx, knot in enumerate(knots):
            delta_lat_km = (locations[:, 0] - knot[0]) * km_per_lat
            delta_lon_km = (locations[:, 1] - knot[1]) * km_per_lon
            dist_km = np.sqrt(delta_lat_km**2 + delta_lon_km**2)
            phi_j = wendland(dist_km / theta_km)
            cols.append(phi_j)
            meta.append((lv_idx + 1, k_idx))

    Phi = np.column_stack(cols).astype(np.float32)
    return Phi, meta


# ── STEP 4: ZERO COLUMN REMOVAL ──────────────────────────────
def remove_zero_columns(Phi_all, meta,
                        n_stations, n_nsrdb, n_pvs,
                        threshold=1e-6):
    """
    Remove basis columns that are zero at ALL locations.
    CRITICAL: single mask applied to all splits.
    """
    col_max = np.abs(Phi_all).max(axis=0)
    active_mask = col_max > threshold

    print(f"\n── Zero Column Removal ──────────────────────────")
    print(f"Total columns before removal : {Phi_all.shape[1]}")
    print(f"Active columns after removal : {active_mask.sum()}")
    print(f"Removed (all-zero)           : {(~active_mask).sum()}")

    level_arr = np.array([m[0] for m in meta])
    for lv in sorted(set(level_arr)):
        lv_mask = level_arr == lv
        kept  = (active_mask & lv_mask).sum()
        total = lv_mask.sum()
        print(f"  Level {lv}: {kept} / {total} active")

    Phi_filtered = Phi_all[:, active_mask]
    active_meta  = [m for m, a in zip(meta, active_mask) if a]

    i1 = n_stations
    i2 = i1 + n_nsrdb
    i3 = i2 + n_pvs

    Phi_stations = Phi_filtered[:i1]
    Phi_nsrdb    = Phi_filtered[i1:i2]
    Phi_pvs      = Phi_filtered[i2:i3]

    print(f"\nOutput shapes:")
    print(f"  Phi_stations : {Phi_stations.shape}")
    print(f"  Phi_nsrdb    : {Phi_nsrdb.shape}")
    print(f"  Phi_pvs      : {Phi_pvs.shape}")

    return Phi_stations, Phi_nsrdb, Phi_pvs, active_mask, active_meta


# ── STEP 5: NORMALISE ────────────────────────────────────────
def normalise_basis(Phi_stations, Phi_nsrdb, Phi_pvs):
    """
    Raw Wendland values are already in [0, 1] by construction
    (=1 at knot center, →0 beyond support). No scaling needed.
    Per-column min-max fit on 4 stations divides PV values by tiny
    station-derived ranges, exploding them to huge numbers — and
    destroys the smooth RBF spatial decay. Pass through unchanged.
    """
    col_min   = np.zeros(Phi_stations.shape[1], dtype=np.float32)
    col_range = np.ones(Phi_stations.shape[1],  dtype=np.float32)
    return Phi_stations, Phi_nsrdb, Phi_pvs, col_min, col_range


# ── STEP 6: UNIQUENESS CHECK ─────────────────────────────────
def check_uniqueness(Phi_pvs, pv_names, threshold=1e-4):
    """
    Verify every PV location has a unique spatial fingerprint.
    Distance = 0 means identical coordinates in the PV file.
    """
    from scipy.spatial.distance import cdist

    D = cdist(Phi_pvs, Phi_pvs, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    min_dist = D.min(axis=1)

    n_duplicates = (min_dist < threshold).sum()
    print(f"\n── Uniqueness Check ─────────────────────────────")
    print(f"Min φ(s) distance between any two PVs : "
          f"{min_dist.min():.6f}")
    print(f"PVs with near-duplicate fingerprint   : {n_duplicates}")

    if n_duplicates > 0:
        idx = np.where(min_dist < threshold)[0]
        for i in idx:
            j = np.argmin(D[i])
            print(f"  {pv_names[i]} ↔ {pv_names[j]}  "
                  f"dist={D[i,j]:.6f}  "
                  f"(identical coordinates — expected)")
    else:
        print("  ✓ All PV locations have unique spatial fingerprints")


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load locations ────────────────────────────────────────
    pv_df = pd.read_csv(
        Path(__file__).parent.parent /
        "data" / "raw" / "pv_nn_assignments.csv"
    )
    pv_locations = pv_df[['pv_lat', 'pv_lon']].values
    pv_names     = pv_df['pv_name'].tolist()

    # NSRDB grid from config constants
    lat_vals = np.arange(NSRDB_LAT_MIN,
                          NSRDB_LAT_MAX + NSRDB_RES * 0.5,
                          NSRDB_RES)
    lon_vals = np.arange(NSRDB_LON_MIN,
                          NSRDB_LON_MAX + NSRDB_RES * 0.5,
                          NSRDB_RES)
    lat_g, lon_g = np.meshgrid(lat_vals, lon_vals, indexing='ij')
    nsrdb_locations = np.column_stack(
        [lat_g.ravel(), lon_g.ravel()])

    # Station locations from config
    station_locations = np.array([
        [v['lat'], v['lon']] for v in STATIONS.values()
    ])

    n_stations = len(station_locations)   # 4
    n_nsrdb    = len(nsrdb_locations)     # 182
    n_pvs      = len(pv_locations)        # 178

    all_locations = np.vstack([
        station_locations,
        nsrdb_locations,
        pv_locations,
    ])

    print(f"Locations: {n_stations} stations + "
          f"{n_nsrdb} NSRDB + {n_pvs} PVs = "
          f"{len(all_locations)} total")

    # ── Build knot grid ───────────────────────────────────────
    print("\nBuilding knot grid:")
    knot_levels = build_knot_grid(
        domain, levels, KM_PER_LAT, KM_PER_LON)
    total_knots = sum(len(k) for k, _ in knot_levels)
    print(f"Total knots: {total_knots}")

    # ── Compute basis matrix ──────────────────────────────────
    print("\nComputing basis matrix...")
    Phi_all, meta = compute_basis_matrix(
        all_locations, knot_levels, KM_PER_LAT, KM_PER_LON)
    print(f"Phi_all shape: {Phi_all.shape}")

    # ── Zero removal ──────────────────────────────────────────
    Phi_stations, Phi_nsrdb, Phi_pvs, active_mask, active_meta = \
        remove_zero_columns(
            Phi_all, meta, n_stations, n_nsrdb, n_pvs)

    # ── Normalise ─────────────────────────────────────────────
    Phi_st_sc, Phi_ns_sc, Phi_pv_sc, col_min, col_range = \
        normalise_basis(Phi_stations, Phi_nsrdb, Phi_pvs)

    # ── Uniqueness check ──────────────────────────────────────
    check_uniqueness(Phi_pv_sc, pv_names)

    # ── Save ──────────────────────────────────────────────────
    BASIS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(BASIS_DIR / "active_mask.npy",   active_mask)
    np.save(BASIS_DIR / "phi_col_min.npy",   col_min)
    np.save(BASIS_DIR / "phi_col_range.npy", col_range)
    np.save(BASIS_DIR / "Phi_stations_scaled.npy", Phi_st_sc)
    np.save(BASIS_DIR / "Phi_nsrdb_scaled.npy",    Phi_ns_sc)
    np.save(BASIS_DIR / "Phi_pvs_scaled.npy",      Phi_pv_sc)

    print(f"\n✓ Basis matrices saved to {BASIS_DIR}")
    print(f"  Final K          : {Phi_pv_sc.shape[1]}")
    print(f"  Final input dim  : "
          f"15 covariates + {Phi_pv_sc.shape[1]} basis "
          f"= {15 + Phi_pv_sc.shape[1]}")