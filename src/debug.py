import pandas as pd
dk_new = pd.read_parquet('outputs/predictions/ghi_pvs.parquet')
idw = pd.read_parquet('outputs/idw/ghi_pvs_idw.parquet')
common = dk_new.index.intersection(idw.index)
dk_new, idw = dk_new.loc[common], idw.loc[common]

dk_ramp = dk_new.diff().abs().mean().mean()
idw_ramp = idw.diff().abs().mean().mean()
print(f'Mean 5-min ramp, single production model (no ensemble):')
print(f'  DK (new):  {dk_ramp:.2f} W/m2')
print(f'  IDW:       {idw_ramp:.2f} W/m2')

means = dk_new.mean()
print(f'\nPer-PV mean GHI spread: [{means.min():.1f}, {means.max():.1f}] (range {means.max()-means.min():.1f})')

day = dk_new[dk_new.mean(axis=1) > 100]
div = day.round(0).nunique(axis=1).mean()
print(f'Distinct GHI values per timestep: {div:.1f} / 178')