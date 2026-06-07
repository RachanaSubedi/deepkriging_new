"""
src/pixel_mapping.py

Snaps every PV location and station to its nearest GOES-18 pixel center.
Saves two lookup tables that everything downstream uses:

  data/processed/
    pv_pixel_map.csv      one row per PV  (178 rows)
    goes_pixel_list.csv   one row per unique pixel (expect ~33)

Run:
    python src/pixel_mapping.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    GOES_RES_LAT, GOES_RES_LON,
    STATIONS, PROCESSED_DIR,
)

# ── SNAP FUNCTION ─────────────────────────────────────────────
def snap_to_pixel(lat, lon):
    """
    Round lat/lon to nearest GOES-18 pixel center.
    GOES pixels sit on a regular grid at ~2km spacing.
    """
    plat = round(float(np.round(lat / GOES_RES_LAT) * GOES_RES_LAT), 6)
    plon = round(float(np.round(lon / GOES_RES_LON) * GOES_RES_LON), 6)
    return plat, plon


if __name__ == "__main__":

    print("=" * 55)
    print("  pixel_mapping.py — GOES Pixel Assignment")
    print("=" * 55)

    # ── Load PV locations ─────────────────────────────────────
    pv_path = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df   = pd.read_csv(pv_path)
    print(f"\nLoaded {len(pv_df)} PV locations")

    # ── Snap PVs to GOES grid ─────────────────────────────────
    snapped = pv_df[['pv_name', 'pv_lat', 'pv_lon']].copy()
    snapped[['pixel_lat', 'pixel_lon']] = snapped.apply(
        lambda r: pd.Series(snap_to_pixel(r['pv_lat'], r['pv_lon'])),
        axis=1
    )
    snapped['pixel_id'] = (
        'px_' +
        snapped['pixel_lat'].astype(str) + '_' +
        snapped['pixel_lon'].astype(str)
    )

    # ── Add station rows ──────────────────────────────────────
    station_rows = []
    for name, info in STATIONS.items():
        plat, plon = snap_to_pixel(info['lat'], info['lon'])
        station_rows.append({
            'pv_name'   : f'STATION_{name}',
            'pv_lat'    : info['lat'],
            'pv_lon'    : info['lon'],
            'pixel_lat' : plat,
            'pixel_lon' : plon,
            'pixel_id'  : f'px_{plat}_{plon}',
        })
    station_df = pd.DataFrame(station_rows)

    # ── Combine and save pv_pixel_map.csv ─────────────────────
    full_map = pd.concat([snapped, station_df], ignore_index=True)
    out_dir  = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    full_map.to_csv(out_dir / "pv_pixel_map.csv", index=False)
    print(f"\n✓ pv_pixel_map.csv saved  ({len(full_map)} rows)")

    # ── Build unique pixel list ───────────────────────────────
    unique_pixels = (
        full_map.groupby(['pixel_id', 'pixel_lat', 'pixel_lon'])
        .agg(n_locations=('pv_name', 'count'),
             location_names=('pv_name', lambda x: '|'.join(x[:5])))
        .reset_index()
        .sort_values(['pixel_lat', 'pixel_lon'])
        .reset_index(drop=True)
    )
    unique_pixels.index.name = 'pixel_index'
    unique_pixels.to_csv(out_dir / "goes_pixel_list.csv")
    print(f"✓ goes_pixel_list.csv saved  ({len(unique_pixels)} unique pixels)\n")

    # ── Print pixel table ────────────────────────────────────
    print("─" * 65)
    print(f"{'#':<4} {'pixel_id':<32} {'lat':>8} {'lon':>10} {'n_locs':>7}")
    print("─" * 65)
    for i, row in unique_pixels.iterrows():
        has_station = any(
            f"STATION_{s}" in row['location_names']
            for s in STATIONS
        )
        flag = " ← has station" if has_station else ""
        print(f"{i:<4} {row['pixel_id']:<32} "
              f"{row['pixel_lat']:>8.4f} {row['pixel_lon']:>10.4f} "
              f"{row['n_locations']:>7}{flag}")
    print("─" * 65)

    # ── Verify station C13 files vs pixel assignments ─────────
    print("\n── Station → Pixel mapping ─────────────────────")
    for _, sr in station_df.iterrows():
        sname = sr['pv_name'].replace('STATION_', '')
        print(f"  {sname:4s}  ({sr['pv_lat']:.4f}, {sr['pv_lon']:.4f})"
              f"  →  pixel ({sr['pixel_lat']:.4f}, {sr['pixel_lon']:.4f})"
              f"  [{sr['pixel_id']}]")

    print(f"\nTotal unique pixels to extract from GEE : {len(unique_pixels)}")
    print("Run gee/extract_c13_pixels.py next.")