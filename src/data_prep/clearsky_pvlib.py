"""
src/clearsky_pvlib.py

Computes deterministic clear-sky GHI at all locations using the
pvlib Ineichen model. This replaces the IDW-interpolated NSRDB
clearsky, which carried interpolation error and (more importantly)
made the background CSI = NSRDB_GHI/NSRDB_clearsky pin to 1.0 on
clear days, biasing the residual target.

Clear-sky GHI is deterministic (function of lat, lon, altitude,
time, and climatological Linke turbidity) so pvlib gives the exact
value at every PV location — no interpolation needed.

Run (once):
    python src/clearsky_pvlib.py

Outputs (data/processed/background_field/):
    clearsky_pvlib_stations.parquet   (T, 4)
    clearsky_pvlib_pvs.parquet        (T, 178)
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from pvlib.location import Location

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import STATIONS, BG_DIR

TZ = 'America/Los_Angeles'


def compute_clearsky(times, lat, lon, altitude):
    """Ineichen clear-sky GHI for one location over the full time index."""
    loc = Location(latitude=lat, longitude=lon, tz=TZ, altitude=altitude)
    cs  = loc.get_clearsky(times, model='ineichen')   # DataFrame ghi/dni/dhi
    return cs['ghi'].values.astype(np.float32)


if __name__ == "__main__":

    print("=" * 55)
    print("  clearsky_pvlib.py — Deterministic Clear-Sky (Ineichen)")
    print("=" * 55)

    # ── Match the existing time index exactly ─────────────────
    ref = pd.read_parquet(BG_DIR / "bg_clearsky_stations.parquet")
    times = ref.index
    if times.tz is None:
        times = times.tz_localize(TZ)
    print(f"\n  Time index : {len(times)} steps  "
          f"({times[0]} → {times[-1]})")

    # ── Stations ──────────────────────────────────────────────
    print("\n[1/2] Computing station clear-sky...")
    elev_st = np.load(BG_DIR / "elevation_stations.npy")   # (4,)
    station_names = list(STATIONS.keys())

    cs_st = {}
    for i, s in enumerate(station_names):
        cs_st[s] = compute_clearsky(
            times, STATIONS[s]['lat'], STATIONS[s]['lon'],
            float(elev_st[i]))
        print(f"  {s}: max={cs_st[s].max():.1f} W/m²  "
              f"(lat={STATIONS[s]['lat']}, elev={elev_st[i]:.0f} m)")

    cs_st_df = pd.DataFrame(cs_st, index=times)
    cs_st_df.index.name = 'datetime_local'
    cs_st_df.to_parquet(BG_DIR / "clearsky_pvlib_stations.parquet")
    print(f"  ✓ clearsky_pvlib_stations.parquet  {cs_st_df.shape}")

    # ── PVs ───────────────────────────────────────────────────
    print("\n[2/2] Computing PV clear-sky (178 locations)...")
    pv_df    = pd.read_csv(Path(__file__).parent.parent /
                           "data" / "raw" / "pv_nn_assignments.csv")
    pv_names = pv_df['pv_name'].tolist()
    pv_lat   = pv_df.set_index('pv_name')['pv_lat']
    pv_lon   = pv_df.set_index('pv_name')['pv_lon']
    elev_pv  = np.load(BG_DIR / "elevation_pvs.npy")   # (178,)

    cs_pv = {}
    for j, pv in enumerate(pv_names):
        cs_pv[pv] = compute_clearsky(
            times, float(pv_lat[pv]), float(pv_lon[pv]),
            float(elev_pv[j]))
        if (j + 1) % 50 == 0 or j == len(pv_names) - 1:
            print(f"  {j+1}/{len(pv_names)} done  "
                  f"(last {pv}: max={cs_pv[pv].max():.1f} W/m²)")

    cs_pv_df = pd.DataFrame(cs_pv, index=times)
    cs_pv_df.index.name = 'datetime_local'
    cs_pv_df.to_parquet(BG_DIR / "clearsky_pvlib_pvs.parquet")
    print(f"  ✓ clearsky_pvlib_pvs.parquet  {cs_pv_df.shape}")

    print(f"\n✓ Done. Clear-sky max across all PVs: "
          f"{cs_pv_df.max().max():.1f} W/m²")
    print(f"  Output dir: {BG_DIR}")