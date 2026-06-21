"""
Diagnostic: characterize the morning/evening CSI bump.
Run:  python src/diagnose.py
"""
import pandas as pd
import numpy as np

csi = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\outputs\predictions\csi_pvs.parquet")
ghi = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\outputs\predictions\ghi_pvs.parquet")
cs  = pd.read_parquet(r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar_Copy\data\processed\background_field\clearsky_pvlib_pvs.parquet")

for d in [csi, ghi, cs]:
    d.index = pd.to_datetime(d.index, utc=True).tz_convert('America/Los_Angeles')

# ── Align all three on common index AND common columns ────────
common_idx = csi.index.intersection(ghi.index).intersection(cs.index)
common_cols = csi.columns.intersection(cs.columns)
print(f"csi shape: {csi.shape}  ghi shape: {ghi.shape}  cs shape: {cs.shape}")
print(f"Common timesteps: {len(common_idx)}  Common columns: {len(common_cols)}")

csi = csi.loc[common_idx, common_cols]
ghi = ghi.loc[common_idx, common_cols]
cs  = cs.loc[common_idx, common_cols]
print(f"After align — all shapes: {csi.shape}")

print("\n" + "=" * 60)
print("Morning over-prediction check (full year, all PVs)")
print("=" * 60)

cs_flat  = cs.values.flatten()
csi_flat = csi.values.flatten()
mask = ~np.isnan(csi_flat) & (cs_flat > 5)
cs_v  = cs_flat[mask]
csi_v = csi_flat[mask]

print(f"\n{'clearsky bin':>16}  {'n':>8}  {'mean CSI':>9}  {'median':>9}  {'%ceiling':>10}")
for lo, hi in [(5,50),(50,100),(100,200),(200,400),(400,600),(600,1100)]:
    b = (cs_v >= lo) & (cs_v < hi)
    if b.sum() == 0: continue
    atc = (csi_v[b] >= 1.19).mean() * 100
    print(f"{f'{lo}-{hi}':>16}  {b.sum():>8}  {csi_v[b].mean():>9.3f}  "
          f"{np.median(csi_v[b]):>9.3f}  {atc:>9.1f}%")

print("\n" + "=" * 60)
print("Morning vs afternoon CSI at matched clearsky (150-250 W/m2)")
print("=" * 60)
hour_2d = np.tile(csi.index.hour.values[:, None], (1, csi.shape[1]))
band = (cs_arr >= 150) & (cs_arr < 250) & ~np.isnan(csi_arr)
mm = band & (hour_2d < 11)
am = band & (hour_2d >= 13)
print(f"  Morning (<11h)    mean CSI = {csi_arr[mm].mean():.3f}  n={mm.sum()}")
print(f"  Afternoon (>=13h) mean CSI = {csi_arr[am].mean():.3f}  n={am.sum()}")

print("\n" + "=" * 60)
print("Spatial pattern of morning over-prediction")
print("=" * 60)
pv_mc = {}
for pv in csi.columns:
    cspv = cs[pv].values; csipv = csi[pv].values; h = csi.index.hour
    m = (cspv >= 100) & (cspv < 200) & (h < 11) & ~np.isnan(csipv)
    if m.sum() > 0: pv_mc[pv] = csipv[m].mean()
s = pd.Series(pv_mc).sort_values(ascending=False)
pv_map = pd.read_csv(r"data\raw\pv_nn_assignments.csv").set_index('pv_name')
print("\nTop 8 PVs by morning low-sun CSI:")
for pv in s.head(8).index:
    print(f"  {pv}: CSI={s[pv]:.3f}  lat={pv_map.loc[pv,'pv_lat']:.4f} lon={pv_map.loc[pv,'pv_lon']:.4f}")
print("Bottom 4:")
for pv in s.tail(4).index:
    print(f"  {pv}: CSI={s[pv]:.3f}  lat={pv_map.loc[pv,'pv_lat']:.4f} lon={pv_map.loc[pv,'pv_lon']:.4f}")
