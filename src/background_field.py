"""
src/background_field.py

Computes IDW-interpolated background CSI field AND meteorological
covariates from all 182 NSRDB grid points, at station and PV locations.

NSRDB filename pattern: {lat}_{lon}_{year}.csv

Run:
    python src/background_field.py

Outputs (data/processed/background_field/):
    bg_csi_stations.parquet        (T, 4)    background CSI at stations
    bg_csi_pvs.parquet             (T, 178)  background CSI at PVs
    bg_clearsky_stations.parquet   (T, 4)    IDW clearsky GHI at stations
    bg_clearsky_pvs.parquet        (T, 178)  IDW clearsky GHI at PVs
    met_<var>_stations.parquet     (T, 4)    per met variable at stations
    met_<var>_pvs.parquet          (T, 178)  per met variable at PVs
    elevation_stations.npy         (4,)      static elevation at stations
    elevation_pvs.npy              (178,)    static elevation at PVs
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from configs.config import (
    NSRDB_DIR, NSRDB_SKIPROWS, LOCAL_TZ,
    STATIONS, BG_DIR,
    KM_PER_LAT, KM_PER_LON,
)

# ── CONSTANTS ─────────────────────────────────────────────────
IDW_POWER           = 2.0
CLEARSKY_NIGHT_W_M2 = 10.0
CSI_CLIP_MAX        = 2.0

# Met variables to extract (continuous → IDW, categorical → nearest-neighbour)
MET_CONTINUOUS = ['Temperature', 'Relative Humidity', 'Pressure',
                  'Precipitable Water', 'Solar Zenith Angle']
MET_CATEGORICAL = ['Cloud Type']   # nearest-neighbour only
MET_KEYS = {
    'Temperature'       : 'temperature',
    'Relative Humidity' : 'rh',
    'Pressure'          : 'pressure',
    'Precipitable Water': 'pw',
    'Solar Zenith Angle': 'cos_zenith',   # stored as cos(SZA)
    'Cloud Type'        : 'cloud_type',
}


# ── STEP 1: PARSE LAT/LON FROM FILENAME ──────────────────────
def parse_lat_lon(filepath):
    parts = filepath.stem.split('_')
    return float(parts[0]), float(parts[1])


# ── STEP 2: READ ELEVATION FROM METADATA ROW ─────────────────
def read_elevation(filepath):
    """
    Row 1 of NSRDB file (0-indexed):
    NSRDB,ID,City,State,Country,Lat,Lon,TZ,Elevation,...
    Elevation is field index 8.
    """
    with open(filepath, 'r') as f:
        f.readline()           # row 0 — headers
        meta = f.readline()    # row 1 — values
    return float(meta.strip().split(',')[8])


# ── STEP 3: LOAD ALL NSRDB FILES ─────────────────────────────
def load_nsrdb(nsrdb_dir):
    """
    Single-pass load of all 182 NSRDB CSVs.

    Returns
    -------
    nsrdb_locs  : (N, 2)   [lat, lon]
    ghi         : (N, T)
    ghi_clear   : (N, T)
    met         : dict  var_key → (N, T)   met arrays
    elevations  : (N,)   elevation in metres
    timestamps  : DatetimeIndex  local time
    """
    files = sorted(nsrdb_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NSRDB CSVs found in {nsrdb_dir}")
    print(f"  Found {len(files)} NSRDB files")

    lats, lons, elevs = [], [], []
    ghi_list, ghic_list = [], []
    met_lists = {k: [] for k in MET_KEYS.values()}
    timestamps = None

    for f in files:
        lat, lon = parse_lat_lon(f)
        lats.append(lat);  lons.append(lon)
        elevs.append(read_elevation(f))

        df = pd.read_csv(f, skiprows=NSRDB_SKIPROWS, low_memory=False)

        # ── Timestamps (NSRDB is in Local Standard Time = PST = UTC-8) ──
        if timestamps is None:
            dt_local = pd.to_datetime({
                'year'  : df['Year'].astype(int),
                'month' : df['Month'].astype(int),
                'day'   : df['Day'].astype(int),
                'hour'  : df['Hour'].astype(int),
                'minute': df['Minute'].astype(int),
            })
            timestamps = pd.DatetimeIndex(
                dt_local.dt.tz_localize('Etc/GMT+8')
                         .dt.tz_convert(LOCAL_TZ)
            )

        ghi_list.append(df['GHI'].values.astype(np.float32))
        ghic_list.append(df['Clearsky GHI'].values.astype(np.float32))

        # ── Met variables ─────────────────────────────────────
        for col, key in MET_KEYS.items():
            if key == 'cos_zenith':
                # Convert Solar Zenith Angle → cos(SZA)
                sza_rad = np.deg2rad(df['Solar Zenith Angle'].values)
                met_lists[key].append(np.cos(sza_rad).astype(np.float32))
            else:
                met_lists[key].append(
                    df[col].values.astype(np.float32))

    nsrdb_locs = np.column_stack([lats, lons])
    ghi        = np.stack(ghi_list,  axis=0)
    ghi_clear  = np.stack(ghic_list, axis=0)
    met        = {k: np.stack(v, axis=0) for k, v in met_lists.items()}
    elevations = np.array(elevs, dtype=np.float32)

    return nsrdb_locs, ghi, ghi_clear, met, elevations, timestamps


# ── STEP 4: COMPUTE CSI ───────────────────────────────────────
def compute_csi(ghi, ghi_clear):
    csi = np.zeros_like(ghi, dtype=np.float32)
    # Use a higher floor than the nighttime threshold — CSI is numerically
    # unstable when clearsky is near zero (e.g. clearsky=1.7 W/m² right
    # after sunrise), producing erratic CSI values that destabilize the
    # model's lag features (bg_csi_diff jumps wildly at the day boundary).
    stable = ghi_clear >= 20.0
    csi[stable] = ghi[stable] / ghi_clear[stable]
    np.clip(csi, 0.0, CSI_CLIP_MAX, out=csi)
    return csi


# ── STEP 5: IDW WEIGHTS ───────────────────────────────────────
def idw_weights(nsrdb_locs, target_locs, power=IDW_POWER):
    M = len(target_locs);  N = len(nsrdb_locs)
    W = np.zeros((M, N), dtype=np.float64)
    for m, (tlat, tlon) in enumerate(target_locs):
        dlat = (nsrdb_locs[:, 0] - tlat) * KM_PER_LAT
        dlon = (nsrdb_locs[:, 1] - tlon) * KM_PER_LON
        d    = np.sqrt(dlat**2 + dlon**2)
        if d.min() < 0.001:
            W[m, d.argmin()] = 1.0
        else:
            w = 1.0 / d**power;  W[m] = w / w.sum()
    return W


# ── STEP 6: NEAREST-NEIGHBOUR INDEX ──────────────────────────
def nearest_idx(nsrdb_locs, target_locs):
    """Return index of closest NSRDB point for each target (for cloud type)."""
    idx = np.zeros(len(target_locs), dtype=int)
    for m, (tlat, tlon) in enumerate(target_locs):
        dlat = (nsrdb_locs[:, 0] - tlat) * KM_PER_LAT
        dlon = (nsrdb_locs[:, 1] - tlon) * KM_PER_LON
        idx[m] = np.sqrt(dlat**2 + dlon**2).argmin()
    return idx


# ── STEP 7: APPLY IDW ────────────────────────────────────────
def apply_idw(W, arr):
    return (W @ arr).astype(np.float32)   # (M, N) @ (N, T) = (M, T)


# ── STEP 8: SAVE HELPER ──────────────────────────────────────
def save_df(arr, timestamps, names, path):
    df = pd.DataFrame(arr.T, index=timestamps, columns=names)
    df.index.name = 'datetime_local'
    df.to_parquet(path)
    return df


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  background_field.py — IDW NSRDB Background + Met")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────
    print("\n[1/6] Loading NSRDB files...")
    nsrdb_locs, ghi, ghi_clear, met, elevations, timestamps = \
        load_nsrdb(NSRDB_DIR)
    print(f"      NSRDB grid points : {nsrdb_locs.shape[0]}")
    print(f"      Timesteps         : {ghi.shape[1]}")
    print(f"      Time range        : {timestamps[0]}  →  {timestamps[-1]}")
    print(f"      Elevation range   : {elevations.min():.0f} – "
          f"{elevations.max():.0f} m")

    # ── 2. Compute CSI ────────────────────────────────────────
    print("\n[2/6] Computing CSI...")
    csi = compute_csi(ghi, ghi_clear)
    print(f"      CSI range   : [{csi.min():.3f}, {csi.max():.3f}]")
    print(f"      Night frac  : {(ghi_clear < CLEARSKY_NIGHT_W_M2).mean():.1%}")

    # ── 3. Target locations ───────────────────────────────────
    print("\n[3/6] Loading target locations...")
    station_names = list(STATIONS.keys())
    station_locs  = np.array([[v['lat'], v['lon']]
                               for v in STATIONS.values()])

    pv_path = Path(__file__).parent.parent / "data" / "raw" / "pv_nn_assignments.csv"
    pv_df   = pd.read_csv(pv_path)
    pv_locs  = pv_df[['pv_lat', 'pv_lon']].values
    pv_names = pv_df['pv_name'].tolist()
    print(f"      Stations : {station_names}")
    print(f"      PVs      : {len(pv_names)}")

    # ── 4. IDW weights + nearest-neighbour index ──────────────
    print("\n[4/6] Computing IDW weights...")
    W_st  = idw_weights(nsrdb_locs, station_locs)
    W_pvs = idw_weights(nsrdb_locs, pv_locs)
    nn_st  = nearest_idx(nsrdb_locs, station_locs)
    nn_pvs = nearest_idx(nsrdb_locs, pv_locs)
    print(f"      W_stations row-sum : {W_st.sum(axis=1).min():.6f}")
    print(f"      W_pvs     row-sum  : {W_pvs.sum(axis=1).min():.6f}")

    # ── 5. Compute + save all outputs ─────────────────────────
    print("\n[5/6] Applying IDW and saving...")
    BG_DIR.mkdir(parents=True, exist_ok=True)

    # CSI
    save_df(apply_idw(W_st,  csi),       timestamps, station_names,
            BG_DIR / "bg_csi_stations.parquet")
    save_df(apply_idw(W_pvs, csi),       timestamps, pv_names,
            BG_DIR / "bg_csi_pvs.parquet")

    # Clearsky GHI
    save_df(apply_idw(W_st,  ghi_clear), timestamps, station_names,
            BG_DIR / "bg_clearsky_stations.parquet")
    save_df(apply_idw(W_pvs, ghi_clear), timestamps, pv_names,
            BG_DIR / "bg_clearsky_pvs.parquet")

    print("      ✓ CSI and clearsky saved")

    # Met variables
    for col, key in MET_KEYS.items():
        arr = met[key]   # (N, T)

        if col in MET_CONTINUOUS:
            # IDW interpolation for continuous variables
            st_arr  = apply_idw(W_st,  arr)
            pv_arr  = apply_idw(W_pvs, arr)
        else:
            # Nearest-neighbour for cloud type (categorical)
            st_arr  = arr[nn_st,  :]   # (4,   T)
            pv_arr  = arr[nn_pvs, :]   # (178, T)

        save_df(st_arr,  timestamps, station_names,
                BG_DIR / f"met_{key}_stations.parquet")
        save_df(pv_arr,  timestamps, pv_names,
                BG_DIR / f"met_{key}_pvs.parquet")
        print(f"      ✓ {key:20s}  stations {st_arr.shape}  "
              f"range=[{arr.min():.2f}, {arr.max():.2f}]")

    # Elevation (static — IDW of NSRDB elevations)
    elev_st  = (W_st  @ elevations).astype(np.float32)   # (4,)
    elev_pvs = (W_pvs @ elevations).astype(np.float32)   # (178,)
    np.save(BG_DIR / "elevation_stations.npy", elev_st)
    np.save(BG_DIR / "elevation_pvs.npy",      elev_pvs)
    print(f"      ✓ elevation  stations={elev_st.round(1)}  "
          f"pv range=[{elev_pvs.min():.0f}, {elev_pvs.max():.0f}] m")

    # ── 6. Sanity check ───────────────────────────────────────
    print("\n[6/6] Sanity check — background CSI at stations:")
    bg_csi_check = pd.read_parquet(BG_DIR / "bg_csi_stations.parquet")
    for s in station_names:
        day = bg_csi_check[s][bg_csi_check[s] > 0.01]
        print(f"  {s}: daytime mean={day.mean():.3f}  "
              f"max={day.max():.3f}  "
              f"daytime frac={len(day)/len(bg_csi_check):.1%}")

    print("\n✓ background_field.py complete")
    print(f"  Output dir: {BG_DIR}")