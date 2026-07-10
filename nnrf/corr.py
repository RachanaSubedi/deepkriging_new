import pandas as pd
dk = pd.read_parquet('outputs/predictions/ghi_pvs.parquet')
idw = pd.read_parquet('outputs/idw/ghi_pvs_idw.parquet')
nsrdb = pd.read_parquet('data/processed/nnrf_features/nsrdb_ghi_global_pvs.parquet')
common = dk.index.intersection(nsrdb.index)
print('DK vs NSRDB corr (mean across PVs):', dk.loc[common].corrwith(nsrdb.loc[common]).mean())
print('IDW vs NSRDB corr (mean across PVs):', idw.loc[common].corrwith(nsrdb.loc[common]).mean())