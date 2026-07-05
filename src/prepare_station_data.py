"""
src/prepare_station_data_v4.py

Reads raw 5-min Ambient Weather CSVs for S1, S2, S3 and the
Transformer-imputed P2 file, snaps everything to a clean shared
5-min master grid (PST/America/Los_Angeles), applies the same
three-tier self-contained gap-fill used in the v3 pipeline for S3's
46-day outage, then writes:

    data/raw/stations/all_stations_GHI_5min_PST.csv

Output schema (same columns as the old 30-min file, just denser):
    datetime   GHI_S1   GHI_S2   GHI_S3   GHI_P2

Gap-fill tiers (applied to S1, S2, S3 — NOT P2, which is already full):
    Tier 1  : partial-day gap — some real readings exist that day
              → estimate day's CSI median from real readings, apply
                to missing hours via GHI_clear
    Tier 2a : full-day gap, short (within ~5 days of real data)
              → median CSI from nearby calendar days
    Tier 2b : full-day gap, long (e.g. S3's 46-day stretch)
              → seasonal/monthly CSI from station's own real history,
                expanding outward month by month until enough days found

S2 and S3 share a physical location but are separate sensors — gap-fill
uses each station's own data only, so S3's filled values reflect S3's
own seasonal pattern (slightly different from S2, as expected for two
distinct sensors at the same site).

Run:
    python src/prepare_station_data_v4.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import STATION_DIR, PROCESSED_DIR, LOCAL_TZ, STATIONS

# ── CONFIG ────────────────────────────────────────────────────
RAW_FILES = {
    'S1': STATION_DIR / '46.59, -119.15 2024.csv',
    'S2': STATION_DIR / '46.82, -119.16 2024.csv',
    'S3': STATION_DIR / '46.82, -119.15 2024.csv',
}

# P2 comes from the Transformer imputation pipeline (v4 output)
P2_FILE = STATION_DIR / 'station_46_78_full_year_GHI_v4.csv'

OUT_FILE = STATION_DIR / 'all_stations_GHI_5min_PST.csv'

SNAP_TOL   = '3min'    # tolerance for snapping irregular readings to 5-min grid
FFILL_LIMIT = 2        # max consecutive 5-min steps to forward-fill (10 min)
MIN_REAL_DAYS = 5      # min days needed for seasonal CSI estimate
MAX_MONTH_RADIUS = 4   # expand outward this many months when seeking real days


# ── CLEARSKY ──────────────────────────────────────────────────
def compute_clearsky(index_local, lat, lon, alt=120):
    """Compute clear-sky GHI (Ineichen) on a tz-aware local index."""
    loc = pvlib.location.Location(lat, lon, tz=LOCAL_TZ, altitude=alt)
    cs = loc.get_clearsky(index_local, model='ineichen')
    return cs['ghi']


# ── LOAD + SNAP RAW STATION FILE ─────────────────────────────
def load_raw_station(path, station_name):
    """
    Load Ambient Weather CSV, extract Solar Radiation column,
    snap irregular timestamps to clean 5-min grid.
    Returns Series indexed by datetime_local (5-min, tz-aware PST).
    """
    df = pd.read_csv(path, low_memory=False)

    # Parse timestamps — raw files have mixed UTC offset strings
    df['dt'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(LOCAL_TZ)
    df = df.sort_values('dt').drop_duplicates('dt')
    df = df.set_index('dt')[['Solar Radiation (W/m^2)']].rename(
        columns={'Solar Radiation (W/m^2)': 'GHI'})
    df['GHI'] = pd.to_numeric(df['GHI'], errors='coerce')


    # Build clean 5-min master grid for the full year
    master = pd.date_range(
        start='2024-01-01 00:00',
        end='2024-12-31 23:55',
        freq='5min',
        tz=LOCAL_TZ,
    )

    # Snap: reindex with nearest-match tolerance, then forward-fill tiny gaps
    ghi = df['GHI'].reindex(master, method='nearest', tolerance=SNAP_TOL)
    ghi = ghi.ffill(limit=FFILL_LIMIT)
    ghi.name = station_name

    n_null = ghi.isna().sum()
    print(f"  {station_name}: {ghi.notna().sum()} valid, {n_null} still-null after snap+ffill")
    return ghi


# ── THREE-TIER GAP-FILL ───────────────────────────────────────
def gap_fill(ghi_series, ghi_clear_series, station_name):
    """
    Fill remaining NaN in ghi_series using only that station's own data.

    ghi_series      : Series (5-min, tz-aware) of raw/snapped GHI
    ghi_clear_series: Series (5-min, tz-aware) of clear-sky GHI
    """
    ghi   = ghi_series.copy().astype(float)
    ghi_c = ghi_clear_series.copy().astype(float)

    daytime = ghi_c >= 10
    missing = ghi.isna() & daytime

    if not missing.any():
        print(f"  {station_name}: no daytime gaps — skipping gap-fill")
        return ghi

    # CSI where we have real data
    csi = (ghi / ghi_c.clip(lower=1)).where(daytime & ghi.notna())
    csi = csi.clip(0, 1.5)

    df = pd.DataFrame({'GHI': ghi, 'GHI_clear': ghi_c, 'CSI': csi})
    df['date']  = df.index.date
    df['month'] = df.index.month

    all_dates = sorted(df['date'].unique())

    # Per-day median CSI from real readings
    daily_csi = (
        df.loc[daytime & df['CSI'].notna()]
          .groupby('date')['CSI'].median()
    )
    days_with_data = set(daily_csi.index)

    def _nearby_day_csi(target_date, max_look=5):
        """Tier 2a: median from nearby calendar days with real data."""
        idx = all_dates.index(target_date)
        for offset in range(1, max_look + 1):
            cands = []
            for j in (idx - offset, idx + offset):
                if 0 <= j < len(all_dates):
                    d = all_dates[j]
                    if d in days_with_data:
                        cands.append(daily_csi[d])
            if cands:
                return float(np.median(cands))
        return np.nan

    def _seasonal_csi(target_date):
        """Tier 2b: seasonal CSI from same station's own real history."""
        target_month = target_date.month
        for radius in range(0, MAX_MONTH_RADIUS + 1):
            cands = []
            for delta in ([0] if radius == 0 else [-radius, radius]):
                m = ((target_month - 1 + delta) % 12) + 1
                mask = (df['month'] == m) & df['CSI'].notna() & daytime
                days_in_month = df.loc[mask, 'date'].unique()
                real_days = [d for d in days_in_month if d in days_with_data]
                if len(real_days) >= MIN_REAL_DAYS:
                    cands.extend(daily_csi[real_days].tolist())
            if len(cands) >= MIN_REAL_DAYS:
                return float(np.median(cands))
        return 0.3  # last-resort fallback: mild overcast

    # Apply tiers
    filled = 0
    for date, group in df[missing].groupby('date'):
        # Tier 1: partial day — use that day's own observed CSI
        if date in days_with_data:
            csi_est = daily_csi[date]
        else:
            # Tier 2a: nearby days
            csi_est = _nearby_day_csi(date)
            if np.isnan(csi_est):
                # Tier 2b: seasonal
                csi_est = _seasonal_csi(date)

        idx = group.index
        ghi.loc[idx] = (csi_est * ghi_c.loc[idx]).clip(lower=0)
        filled += len(idx)

    # Nighttime stays 0
    ghi.loc[~daytime] = ghi.loc[~daytime].fillna(0.0)

    print(f"  {station_name}: filled {filled} daytime steps via gap-fill tiers")
    return ghi


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  prepare_station_data_v4.py  —  5-min station alignment")
    print("=" * 60)

    master = pd.date_range(
        '2024-01-01 00:00', '2024-12-31 23:55',
        freq='5min', tz=LOCAL_TZ
    )

    # ── 1. Load + snap S1, S2, S3 ─────────────────────────────
    print("\n[1/4] Loading raw station CSVs...")
    raw = {}
    for sname, fpath in RAW_FILES.items():
        raw[sname] = load_raw_station(fpath, sname)

    # ── 2. Compute clear-sky per station ──────────────────────
    print("\n[2/4] Computing clear-sky GHI...")
    ghi_clear = {}
    for sname, info in STATIONS.items():
        if sname == 'P2':
            continue
        cs = compute_clearsky(master, info['lat'], info['lon'])
        ghi_clear[sname] = cs
        print(f"  {sname}: clear-sky ok, peak={cs.max():.1f} W/m²")

    # ── 3. Gap-fill S1, S2, S3 ────────────────────────────────
    print("\n[3/4] Applying three-tier gap-fill...")
    filled = {}
    for sname in ['S1', 'S2', 'S3']:
        filled[sname] = gap_fill(raw[sname], ghi_clear[sname], sname)

    # ── 4. Load P2 imputed file ───────────────────────────────
    print("\n[4/4] Loading P2 imputed file...")
    p2 = pd.read_csv(P2_FILE)
    p2['dt'] = pd.to_datetime(p2['datetime']).dt.tz_localize(LOCAL_TZ, nonexistent='shift_forward', ambiguous='NaT')
    p2 = p2.set_index('dt')['GHI_imputed']
    p2 = p2[~p2.index.duplicated(keep='first')]
    p2 = p2.reindex(master).ffill(limit=12).bfill(limit=12)
    p2.name = 'P2'
    print(f"  P2: {p2.notna().sum()} valid, {p2.isna().sum()} null")

    # ── 5. Combine and save ───────────────────────────────────
    filled['S2'] = filled['S2'].clip(upper=1200)

    out = pd.DataFrame({
        'GHI_S1': filled['S1'],
        'GHI_S2': filled['S2'],
        'GHI_S3': filled['S3'],
        'GHI_P2': p2,
    }, index=master)

    out.index.name = 'datetime'

    total_null = out.isna().sum()
    print(f"\nFinal null counts:\n{total_null}")

    out.to_csv(OUT_FILE)
    print(f"\n✓ Saved: {OUT_FILE}")
    print(f"  Shape : {out.shape}")
    print(f"  Range : {out.index[0]}  →  {out.index[-1]}")