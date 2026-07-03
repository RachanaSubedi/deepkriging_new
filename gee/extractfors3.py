"""
gee/extract_s3_pixel_standalone.py

One-off extraction of GOES-18 C13+C02 for S3's own coordinate,
separate from the shared S2/S3 pixel file currently in use.

Background: pixel_mapping.py's snap logic assigned S2 and S3 to the
identical GOES pixel (px_46.828829_-119.155673) because they sit
~977m apart, close to a pixel boundary, and the snap rounded both to
the same nearest pixel center. v3's older 30-min extraction resolved
them as distinct series (mean diff ~0.62K, correlation 0.9996) —
this script re-extracts S3 using its real coordinate directly
(no snapping) to recover that same distinction for v4's 5-min data.

S2's true coordinate : 46.823242, -119.163197  (845m from shared pixel — keep as-is)
S3's true coordinate : 46.821036, -119.150761  (944m from shared pixel — re-extract this one)

Run:
    python gee/extract_s3_pixel_standalone.py

Exports to Google Drive folder: goes18_c13_c02_pixels/
    goes18_c13_c02_px_s3_standalone.csv
    Columns: datetime_utc, bt_c13_raw, refl_c02_raw, pixel_id
"""

import ee
import time

# ── SETTINGS ─────────────────────────────────────────────────
DRIVE_FOLDER  = "goes18_c13_c02_pixels"
START_DATE    = "2024-01-01"
END_DATE      = "2025-01-01"
SCALE_METERS  = 2000
GOES_BUFFER_M = 1000          # match original script's point buffer
GEE_DATASET   = "NOAA/GOES/18/MCMIPC"   # same collection as original script
GEE_PROJECT   = None          # set to your GEE project id if required

S3_LAT, S3_LON = 46.821036, -119.150761
PIXEL_ID = "s3_standalone"


def init_gee():
    ee.Authenticate(auth_mode='notebook')
    if GEE_PROJECT:
        ee.Initialize(project=GEE_PROJECT)
    else:
        ee.Initialize()
    print("✓ GEE initialized")


def export_s3_pixel():
    point = ee.Geometry.Point([S3_LON, S3_LAT])

    col = (
        ee.ImageCollection(GEE_DATASET)
        .filterDate(START_DATE, END_DATE)
        .select(['CMI_C13', 'CMI_C02'])
    )

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
            'pixel_id'      : PIXEL_ID,
        })

    features = col.map(extract_value)
    fc = ee.FeatureCollection(features)

    safe_name = f"goes18_c13_c02_px_{PIXEL_ID}"

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"c13_c02_{PIXEL_ID}"[:100],
        folder=DRIVE_FOLDER,
        fileNamePrefix=safe_name,
        fileFormat='CSV',
        selectors=['datetime_utc', 'bt_c13_raw', 'refl_c02_raw', 'pixel_id'],
    )
    task.start()
    print(f"  ✓ Submitted task for S3 at ({S3_LAT}, {S3_LON})")
    print(f"    Output file: {safe_name}.csv")
    return task


if __name__ == "__main__":
    print("=" * 60)
    print("  extract_s3_pixel_standalone.py — S3-only GOES-18 C13+C02")
    print("=" * 60)

    init_gee()
    task = export_s3_pixel()

    print(f"\n{'─'*60}")
    print("NEXT STEPS:")
    print("  1. Monitor at https://code.earthengine.google.com/tasks")
    print("     (task usually completes in 5-15 min)")
    print("  2. Download goes18_c13_c02_px_s3_standalone.csv from Drive")
    print("  3. Replace S3's entry in cfg.RAW['c13c02_s3'] in config.py")
    print("     to point to this new file instead of the shared S2/S3 one")
    print(f"{'─'*60}")