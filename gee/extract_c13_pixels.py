"""
gee/extract_c13_pixels.py

Extracts GOES-18 C13 brightness temperature for:
  - 33 unique GOES pixels covering 178 PV locations
  - 4 station locations (if not already extracted)

Run locally with: python gee/extract_c13_pixels.py
Or on HPC with service account (see bottom of file)
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    GEE_PROJECT, GEE_DATASET, GOES_SCALE_C13,
    GOES_BUFFER_M, LOCAL_TZ, C13_PIXEL_DIR,
    C13_STAT_DIR, STATIONS
)

# ── AUTHENTICATE ──────────────────────────────────────────────
def init_gee(use_service_account=False,
             service_account_key=None):
    """
    Authenticate GEE.
    Local:  browser OAuth (default)
    HPC:    service account JSON key
    """
    if use_service_account:
        credentials = ee.ServiceAccountCredentials(
            email=None,
            key_file=service_account_key
        )
        ee.Initialize(credentials=credentials,
                      project=GEE_PROJECT)
        print("GEE initialized with service account")
    else:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print("GEE initialized with browser OAuth")

# ── BUILD EXTRACTION TARGETS ──────────────────────────────────
def build_targets(pv_csv_path):
    """
    Build complete list of (pixel_id, lat, lon) to extract.
    Combines unique GOES pixels from PVs + 4 station pixels.
    """
    goes_res_lat = 2.0 / 111.0
    goes_res_lon = 2.0 / 75.8

    def snap(lat, lon):
        plat = round(np.round(lat / goes_res_lat) * goes_res_lat, 6)
        plon = round(np.round(lon / goes_res_lon) * goes_res_lon, 6)
        return plat, plon

    # PV pixels
    pv_df = pd.read_csv(pv_csv_path)
    pv_df['goes_lat'] = pv_df['pv_lat'].apply(
        lambda x: snap(x, 0)[0])
    pv_df['goes_lon'] = pv_df['pv_lon'].apply(
        lambda x: snap(0, x)[1])
    pv_df['pixel_id'] = ('pv_pixel_' +
        pv_df['goes_lat'].astype(str) + '_' +
        pv_df['goes_lon'].astype(str))

    pv_pixels = (pv_df[['pixel_id', 'goes_lat', 'goes_lon']]
                 .drop_duplicates()
                 .rename(columns={'goes_lat': 'lat',
                                   'goes_lon': 'lon'}))

    # Station pixels
    station_rows = []
    for name, coords in STATIONS.items():
        plat, plon = snap(coords['lat'], coords['lon'])
        station_rows.append({
            'pixel_id': f'station_{name.lower()}',
            'lat': plat,
            'lon': plon
        })
    station_pixels = pd.DataFrame(station_rows)

    # Combine, deduplicate
    all_targets = (pd.concat([pv_pixels, station_pixels])
                   .drop_duplicates(subset=['lat', 'lon'])
                   .reset_index(drop=True))

    print(f"PV pixels:      {len(pv_pixels)}")
    print(f"Station pixels: {len(station_pixels)}")
    print(f"Total unique:   {len(all_targets)}")

    # Save pixel → PV mapping for later use
    pv_pixel_map = pv_df[['pv_name', 'pixel_id',
                            'goes_lat', 'goes_lon']]
    pv_pixel_map.to_csv(
        Path(C13_PIXEL_DIR).parent / 'pv_pixel_mapping.csv',
        index=False
    )
    print("Saved: pv_pixel_mapping.csv")

    return all_targets

# ── WEEKLY EXTRACTION PER PIXEL ───────────────────────────────
def extract_pixel(pixel_id, lat, lon,
                   start_date='2024-01-01',
                   end_date='2025-01-01',
                   output_dir=None):
    """
    Extract C13 time series for one pixel location.
    Saves to CSV in output_dir.
    Skips if file already exists.
    """
    output_dir = Path(output_dir or C13_PIXEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pixel_id}_c13_raw.csv"

    if out_path.exists():
        print(f"  SKIP {pixel_id} (already extracted)")
        return

    print(f"  Extracting {pixel_id} ({lat:.4f}, {lon:.4f})")

    # Build weekly date ranges
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end   = datetime.strptime(end_date,   '%Y-%m-%d')
    weeks = []
    cur = start
    while cur < end:
        wend = min(cur + timedelta(days=7), end)
        weeks.append((cur.strftime('%Y-%m-%d'),
                      wend.strftime('%Y-%m-%d')))
        cur = wend

    geom = ee.Geometry.Point([lon, lat]).buffer(GOES_BUFFER_M)

    def image_to_feature(img):
        c13  = img.select('CMI_C13')
        dqf  = img.select('DQF_C13')
        mask = dqf.lte(1)
        c13v = c13.updateMask(mask)
        stats = c13v.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=2000,
            maxPixels=1e8,
            bestEffort=True
        )
        has_val = ee.Dictionary(stats).contains('CMI_C13')
        val = ee.Algorithms.If(
            has_val, stats.get('CMI_C13'), -9999
        )
        return ee.Feature(None, {
            'datetime_utc': img.date().format(
                'YYYY-MM-dd HH:mm:ss'),
            'BT_C13_raw':   val,
        })

    all_chunks = []
    for i, (ws, we) in enumerate(weeks):
        retries = 3
        for attempt in range(retries):
            try:
                ic = (ee.ImageCollection(GEE_DATASET)
                      .filterDate(ws, we)
                      .select(['CMI_C13', 'DQF_C13']))
                fc  = ee.FeatureCollection(ic.map(image_to_feature))
                rows = [f['properties']
                        for f in fc.getInfo()['features']]
                df_week = pd.DataFrame(rows)
                all_chunks.append(df_week)
                break
            except Exception as ex:
                if attempt < retries - 1:
                    print(f"    Retry {attempt+1} for week {i+1}")
                    time.sleep(10)
                else:
                    print(f"    FAILED week {i+1}: {ex}")

    if not all_chunks:
        print(f"  WARNING: no data for {pixel_id}")
        return

    df = pd.concat(all_chunks, ignore_index=True)
    df['datetime_utc'] = pd.to_datetime(
        df['datetime_utc'], utc=True)
    df = (df.sort_values('datetime_utc')
            .drop_duplicates('datetime_utc')
            .reset_index(drop=True))
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path.name}")

# ── POST-PROCESS: SCALE + RESAMPLE TO 30-MIN ─────────────────
def postprocess_pixel(pixel_id, raw_dir=None, out_dir=None):
    """
    Apply scale factor, compute BT_anomaly, lag,
    and resample to 30-min intervals.
    """
    raw_dir = Path(raw_dir or C13_PIXEL_DIR)
    out_dir = Path(out_dir or C13_FEAT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{pixel_id}_c13_raw.csv"
    out_path = out_dir / f"{pixel_id}_c13_30min.csv"

    if not raw_path.exists():
        print(f"  Missing raw file: {raw_path.name}")
        return

    df = pd.read_csv(raw_path)
    df['datetime_utc'] = pd.to_datetime(
        df['datetime_utc'], utc=True)
    df['datetime_local'] = df['datetime_utc'].dt.tz_convert(
        LOCAL_TZ)

    # Apply scale factor
    df['BT_C13_raw'] = pd.to_numeric(
        df['BT_C13_raw'], errors='coerce')
    df['BT_C13_raw'] = df['BT_C13_raw'].replace(-9999, np.nan)
    df['BT_C13'] = df['BT_C13_raw'] * GOES_SCALE_C13

    # Filter to local 2024
    df = df[
        (df['datetime_local'] >=
         pd.Timestamp('2024-01-01', tz=LOCAL_TZ)) &
        (df['datetime_local'] <=
         pd.Timestamp('2024-12-31 23:59:59', tz=LOCAL_TZ))
    ].copy()

    # Resample to 30-min (mean of ~3 GOES images per window)
    df30 = (df.set_index('datetime_local')[['BT_C13']]
              .resample('30min').mean()
              .reset_index())

    # Monthly anomaly: BT_anomaly = BT_C13 - monthly mean
    df30['month'] = df30['datetime_local'].dt.month
    monthly_mean = df30.groupby('month')['BT_C13'].transform(
        'mean')
    monthly_std  = df30.groupby('month')['BT_C13'].transform(
        'std')
    df30['BT_anomaly'] = (df30['BT_C13'] - monthly_mean) / \
                          monthly_std.replace(0, 1)

    # Lag features
    df30['BT_C13_lag1'] = df30['BT_C13'].shift(1)   # 30-min lag

    df30['pixel_id'] = pixel_id
    df30.to_csv(out_path, index=False)
    print(f"  Postprocessed → {out_path.name}  ({len(df30)} rows)")

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Initialize GEE
    #    Local:  use_service_account=False
    #    HPC:    use_service_account=True, provide key path
    init_gee(use_service_account=False)

    # 2. Build extraction targets
    pv_csv = Path(__file__).parent.parent / \
             'data' / 'raw' / 'pv_nn_assignments.csv'
    targets = build_targets(pv_csv)
    print(targets)

    # 3. Extract all pixels (skips already-done ones)
    print("\n── Extracting C13 ───────────────────────────────")
    for _, row in targets.iterrows():
        extract_pixel(
            pixel_id=row['pixel_id'],
            lat=row['lat'],
            lon=row['lon'],
            output_dir=C13_PIXEL_DIR
        )

    # 4. Postprocess all pixels
    print("\n── Postprocessing ───────────────────────────────")
    for _, row in targets.iterrows():
        postprocess_pixel(
            pixel_id=row['pixel_id'],
            raw_dir=C13_PIXEL_DIR,
            out_dir=str(Path(C13_PIXEL_DIR).parent.parent.parent /
                        'processed' / 'c13_features')
        )

    print("\n✓ C13 extraction complete.")