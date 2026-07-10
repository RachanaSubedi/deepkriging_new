import pandas as pd
dk  = pd.read_parquet('outputs/predictions/ghi_pvs.parquet')
idw = pd.read_parquet('outputs/idw/ghi_pvs_idw.parquet')
common = dk.index.intersection(idw.index)
dk, idw = dk.loc[common], idw.loc[common]

dk_ramp  = dk.diff().abs().mean().mean()
idw_ramp = idw.diff().abs().mean().mean()
print(f'Mean 5-min ramp (step-to-step change), averaged over 178 PVs:')
print(f'  DK:  {dk_ramp:.2f} W/m2')
print(f'  IDW: {idw_ramp:.2f} W/m2')