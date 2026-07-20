import pandas as pd
cs = pd.read_parquet('data/processed/background_field/bg_clearsky_pvs.parquet')
day = cs.loc['2024-12-31', 'pv_1141']
print(day.tail(20))