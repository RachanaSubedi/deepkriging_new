"""
gee/extract_c13_c02_pixels.py

Extracts GOES-18 C13 brightness temperature AND C02 visible/near-IR
reflectance at all unique GOES pixels covering the domain, using a
Google Earth Engine service account.

Extends extract_c13_pixels.py to pull a second band (CMI_C02) in the
SAME export pass — same pixels, same collection, same timestamps —
rather than running a separate extraction.

C13 = 10.3 um clean longwave IR   -> cloud-top temperature signal
C02 = 0.64 um visible reflectance -> cloud brightness/reflectance,
      physically distinct from C13 (temperature vs. reflectance).
      Daytime-only signal (meaningless at night — fine for GHI work
      since irradiance is zero at night anyway).

NOTE on cadence: NOAA/GOES/.../MCMIPC is published at ~10-minute
cadence (confirmed empirically for this collection), not 5-minute.
If downstream work needs a 5-minute grid, this collection alone will
not provide it natively — merge_asof with a tolerance window against
the 5-min target grid is still required after download, same as the
original C13-only pipeline.

Prerequisites:
    pip install earthengine-api

Run:
    python gee/extract_c13_c02_pixels.py

Reads:
    data/processed/goes_pixel_list.csv   (from pixel_mapping.py)

Exports to Google Drive:
    Folder: goes18_c13_c02_pixels/
    One CSV per pixel: goes18_c13_c02_px_{lat}_{lon}.csv
    Columns: datetime_utc, bt_c13_raw, refl_c02_raw, pixel_id

After GEE exports finish (check Task Manager at code.earthengine.google.com):
    Download CSVs from Drive → data/raw/goes_c13_c02/extracted_pixels/
"""

import ee
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    GEE_PROJECT, GEE_DATASET,
    GOES_SCALE_C13, GOES_BUFFER_M,
    PROCESSED_DIR,
)

# ── SETTINGS ─────────────────────────────────────────────────
DRIVE_FOLDER  = "goes18_c13_c02_pixels"
START_DATE    = "2024-01-01"
END_DATE      = "2025-01-01"
SCALE_METERS  = 2000


# ── AUTHENTICATE ──────────────────────────────────────────────
def init_gee():
    """
    Personal OAuth — service accounts have no Drive quota.
    First run opens a browser for one-time consent.
    Credentials cached in ~/.config/earthengine/ after that.
    """
    ee.Authenticate(auth_mode='notebook')
    ee.Initialize(project=GEE_PROJECT)
    print("✓ GEE initialized with personal OAuth")


# ── BUILD EXPORT TASK FOR ONE PIXEL ──────────────────────────
def export_pixel(pixel_id, lat, lon, task_list):
    """
    Extracts a time series of C13 BT and C02 reflectance at
    (lat, lon) and submits a GEE export task to Google Drive.

    The export produces one CSV row per GOES-18 image (~10-min cadence).
    Columns: datetime_utc, bt_c13_raw, refl_c02_raw, pixel_id
    """
    point = ee.Geometry.Point([lon, lat])

    # Load and filter GOES-18 MCMIPC collection — both bands in one pass
    col = (
        ee.ImageCollection(GEE_DATASET)
        .filterDate(START_DATE, END_DATE)
        .select(['CMI_C13', 'CMI_C02'])
    )

    # Map over images: extract values at point, attach timestamp
    def extract_value(image):
        vals = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(GOES_BUFFER_M),
            scale=SCALE_METERS,
        )
        return ee.Feature(None, {
            'datetime_utc'  : image.date().format('YYYY-MM-dd HH:mm:ss'),
            'bt_c13_raw'    : vals.get('CMI_C13'),
            'refl_c02_raw'  : vals.get('CMI_C02'),
            'pixel_id'      : pixel_id,
        })

    features = col.map(extract_value)
    fc = ee.FeatureCollection(features)

    # Sanitised filename (no dots or minus signs that cause Drive issues)
    safe_name = (f"goes18_c13_c02_{pixel_id}"
                 .replace('.', 'p')
                 .replace('-', 'n'))

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"c13_c02_{pixel_id}"[:100],
        folder=DRIVE_FOLDER,
        fileNamePrefix=safe_name,
        fileFormat='CSV',
        selectors=['datetime_utc', 'bt_c13_raw', 'refl_c02_raw', 'pixel_id'],
    )
    task.start()
    task_list.append((pixel_id, task))
    return task


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  extract_c13_c02_pixels.py — GOES-18 C13+C02 GEE Extraction")
    print("=" * 60)

    # ── 1. Authenticate ───────────────────────────────────────
    init_gee()

    # ── 2. Load pixel list ────────────────────────────────────
    pixel_csv = PROCESSED_DIR / "goes_pixel_list.csv"
    if not pixel_csv.exists():
        raise FileNotFoundError(
            f"Run src/pixel_mapping.py first — {pixel_csv} not found"
        )
    pixels = pd.read_csv(pixel_csv)
    print(f"\nLoaded {len(pixels)} unique GOES pixels to extract")
    print(f"Date range : {START_DATE}  →  {END_DATE}")
    print(f"Drive folder: {DRIVE_FOLDER}/\n")

    # ── 3. Submit export tasks ────────────────────────────────
    tasks = []
    for _, row in pixels.iterrows():
        pid  = row['pixel_id']
        lat  = float(row['pixel_lat'])
        lon  = float(row['pixel_lon'])

        export_pixel(pid, lat, lon, tasks)
        print(f"  ✓ Submitted  {pid:40s}  ({lat:.4f}, {lon:.4f})")
        time.sleep(0.3)   # avoid GEE rate limit

    # ── 4. Summary ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {len(tasks)} export tasks submitted to GEE")
    print(f"{'─'*60}")
    print("""
NEXT STEPS:
  1. Go to  https://code.earthengine.google.com/tasks
     to monitor task progress  (each task ~5-15 min)

  2. When ALL tasks show 'COMPLETED':
     Open Google Drive → goes18_c13_c02_pixels/
     Download all CSV files

  3. Place downloaded CSVs in:
     data/raw/goes_c13_c02/extracted_pixels/
     Expected filenames:
       goes18_c13_c02_px_46p6141_n119p2086.csv
       goes18_c13_c02_px_46p6141_n119p1823.csv
       ... (one per pixel)

  4. Sanity check before trusting refl_c02_raw in any downstream
     feature: pick a day you already know well (e.g. the March 22
     cloud-enhancement day) and confirm C02 reflectance rises with
     known cloud cover and is null/near-zero at night.

  5. Run:  python src/c13_features.py
     (will need a small update to also load/normalize refl_c02_raw
      — not yet wired in; this script only extracts the raw data)
""")