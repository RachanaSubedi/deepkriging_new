import pandas as pd
dk = pd.read_parquet('outputs/predictions/ghi_pvs.parquet')
idw = pd.read_parquet('outputs/idw/ghi_pvs_idw.parquet')
common = dk.index.intersection(idw.index)
dk_r = dk.loc[common].diff().abs().mean().mean()
idw_r = idw.loc[common].diff().abs().mean().mean()
print(f'DK ramp (quantile-sampled): {dk_r:.2f} W/m2')
print(f'IDW ramp (reference):       {idw_r:.2f} W/m2')